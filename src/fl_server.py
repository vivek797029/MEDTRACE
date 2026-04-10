"""
MedTrace Federated Learning — Aggregation Server
=================================================
Central coordinator that aggregates LoRA weight deltas from hospitals
using Federated Averaging (FedAvg). Never sees any patient data.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover — GPU path only
    torch = None  # type: ignore[assignment]

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover — GPU path only
    LoraConfig = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover — GPU path only
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

from fl_config import (
    AggregationError, FLConfig, Metrics, WeightDict, config as default_config,
)
from fl_tracker import ExperimentTracker, NoOpTracker

logger = logging.getLogger(__name__)


class FederatedServer:
    """Central aggregation server for federated MedTrace training."""

    def __init__(
        self,
        device: str = "cpu",
        cfg: FLConfig = default_config,
        tracker: Optional[ExperimentTracker] = None,
    ):
        self.device = device
        self.cfg = cfg
        self.global_weights: Optional[WeightDict] = None
        self.round_metrics: List[Metrics] = []
        self.hospital_contributions: Dict[str, Metrics] = {}
        self.tracker: ExperimentTracker = tracker if tracker is not None else NoOpTracker()

        logger.info("Aggregation server initialized | Strategy: %s | Hospitals: %d | Rounds: %d",
                     cfg.aggregation_strategy, cfg.num_hospitals, cfg.fl_rounds)
        if cfg.dp.enabled:
            logger.info("Differential Privacy: ON (eps=%.1f) | Secure Aggregation: %s",
                        cfg.dp.epsilon, "ON" if cfg.dp.secure_aggregation else "OFF")

    # ─── LoRA Helper (DRY — single source of truth) ───────────

    def _make_lora_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.cfg.lora.r,
            lora_alpha=self.cfg.lora.alpha,
            lora_dropout=self.cfg.lora.dropout,
            target_modules=list(self.cfg.lora.target_modules),
            bias=self.cfg.lora.bias,
            task_type=self.cfg.lora.task_type,
        )

    def _load_peft_model(self, weights: Optional[WeightDict] = None,
                         dtype=None):
        """Load base model with LoRA, optionally inject weights. Caller owns cleanup.

        Args:
            weights: LoRA state dict to inject (if any).
            dtype: torch dtype override. Defaults to float32 for training
                   precision. Use float16 for save/eval to halve RAM usage.
        """
        if dtype is None:
            dtype = torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = get_peft_model(base, self._make_lora_config())
        if weights is not None:
            model.load_state_dict(weights, strict=False)
        return model

    # ─── Initialization ───────────────────────────────────────

    def initialize_global_model(self) -> WeightDict:
        """Create initial global model with random LoRA weights."""
        logger.info("Initializing global model...")
        model = self._load_peft_model()

        self.global_weights = OrderedDict()
        for name, param in model.named_parameters():
            if "lora_" in name:
                self.global_weights[name] = param.detach().cpu().clone()

        param_count = sum(p.numel() for p in self.global_weights.values())
        size_kb = sum(p.element_size() * p.numel() for p in self.global_weights.values()) / 1024
        logger.info("Global LoRA params: %s | Size: %.1f KB", f"{param_count:,}", size_kb)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.global_weights

    # ─── Aggregation ──────────────────────────────────────────

    # Maximum number of individual hospital contribution lines to log.
    # Beyond this threshold a summary line is used instead.
    _CONTRIBUTION_LOG_LIMIT: int = 10

    def aggregate(
        self,
        client_updates: Dict[str, Tuple[WeightDict, Metrics]],
        round_num: int,
    ) -> WeightDict:
        """
        Federated Averaging (FedAvg) — McMahan et al., 2017
        w_global = Σ (n_k / n_total) · w_k

        Scalability design
        ------------------
        * **Streaming aggregation**: weights are accumulated one client at a time
          and their references released immediately, so peak memory is
          O(2 × model_size) regardless of the number of clients.
        * **Key-consistency guard**: all clients must share identical weight keys;
          mismatches are caught before any accumulation begins.
        * **Sampled divergence**: for deployments with >``_DIVERGENCE_SAMPLE``
          clients the divergence metric is estimated from a random sample.
        * **Log throttle**: individual contribution lines are printed only for
          fleets of ≤ ``_CONTRIBUTION_LOG_LIMIT`` hospitals; larger fleets get a
          compact summary.
        """
        logger.info("Aggregating Round %d (%d clients)...", round_num + 1, len(client_updates))
        t0 = time.time()

        if not client_updates:
            raise AggregationError("aggregate() called with no client updates")

        if self.cfg.dp.secure_aggregation:
            logger.info("Secure aggregation: simulating encrypted weight transfer")

        # ── Phase 1: validate keys and compute sample fractions ─────────────
        reference_keys = None
        total_samples: int = 0
        fracs: Dict[str, float] = {}

        for hospital_id, (weights, metrics) in client_updates.items():
            n = metrics["num_samples"]
            total_samples += n

            # Key consistency: every client must share the same LoRA weight names
            keys = set(weights.keys())
            if reference_keys is None:
                reference_keys = keys
            elif keys != reference_keys:
                missing = reference_keys - keys
                extra   = keys - reference_keys
                raise AggregationError(
                    f"Weight key mismatch for client {hospital_id!r}. "
                    f"Missing: {missing}. Extra: {extra}."
                )

        if total_samples == 0:
            raise AggregationError("All clients reported 0 training samples")

        for hospital_id, (_, metrics) in client_updates.items():
            fracs[hospital_id] = metrics["num_samples"] / total_samples

        # ── Phase 2: streaming weighted average ─────────────────────────────
        # Weights are accumulated one client at a time; no full list is kept in
        # memory.  Each client's weight dict is referenced only during its
        # accumulation step.
        aggregated: WeightDict = OrderedDict()
        hospital_names: Dict[str, str] = {}   # hospital_id → display name

        for hospital_id, (weights, metrics) in client_updates.items():
            frac = fracs[hospital_id]
            name = metrics["hospital"]
            hospital_names[hospital_id] = name
            self.hospital_contributions[name] = {
                "round": round_num,
                "weight": frac,
                "samples": metrics["num_samples"],
            }

            for key, param in weights.items():
                p = param.float()
                if key not in aggregated:
                    aggregated[key] = frac * p
                else:
                    aggregated[key] = aggregated[key] + frac * p

        self._validate_weights(aggregated)
        self.global_weights = aggregated
        elapsed = time.time() - t0

        # ── Phase 3: divergence (sampled for large fleets) ──────────────────
        divergence = self._compute_divergence_sampled(
            client_updates, aggregated, max_clients=self._DIVERGENCE_SAMPLE,
        )

        # ── Logging: throttle per-client lines for large fleets ─────────────
        n_clients = len(client_updates)
        if n_clients <= self._CONTRIBUTION_LOG_LIMIT:
            for hid, name in hospital_names.items():
                logger.info(
                    "  %s: weight=%.3f (%d samples)",
                    name, fracs[hid], client_updates[hid][1]["num_samples"],
                )
        else:
            min_frac = min(fracs.values())
            max_frac = max(fracs.values())
            logger.info(
                "  %d clients | sample weights [%.3f, %.3f] | total=%d",
                n_clients, min_frac, max_frac, total_samples,
            )

        # ── Metrics ─────────────────────────────────────────────────────────
        losses = [m["train_loss"] for _, m in client_updates.values()]
        sample_fracs_list = [fracs[hid] for hid in client_updates]
        avg_loss = sum(l * w for l, w in zip(losses, sample_fracs_list))

        round_metrics: Metrics = {
            "round": round_num,
            "num_hospitals": n_clients,
            "total_samples": total_samples,
            "aggregation_time": round(elapsed, 3),
            "weight_divergence": divergence,
            "avg_loss": avg_loss,
            "contribution_weights": {
                hospital_names[hid]: fracs[hid] for hid in client_updates
            },
        }
        self.round_metrics.append(round_metrics)

        self.tracker.log(
            {
                "round/avg_loss": avg_loss,
                "round/min_loss": min(losses),
                "round/max_loss": max(losses),
                "round/weight_divergence": divergence,
                "round/aggregation_time": elapsed,
                "round/total_samples": total_samples,
                "round/num_hospitals": n_clients,
            },
            step=round_num,
        )
        # Per-hospital contribution weights — emit individually for small fleets,
        # skip for large ones (too many metrics would bloat tracking back-ends)
        if n_clients <= self._CONTRIBUTION_LOG_LIMIT:
            contribution_log = {
                f"contribution/{hospital_names[hid]}": fracs[hid]
                for hid in client_updates
            }
            self.tracker.log(contribution_log, step=round_num)
        else:
            self.tracker.log(
                {
                    "contribution/min_weight": min_frac,
                    "contribution/max_weight": max_frac,
                },
                step=round_num,
            )

        logger.info(
            "Aggregation complete (%.3fs) | Clients: %d | Divergence: %.6f | Avg loss: %.4f",
            elapsed, n_clients, divergence, avg_loss,
        )
        return self.global_weights

    # ─── Validation ───────────────────────────────────────────

    # For large fleets, divergence is estimated from a random sample of clients.
    _DIVERGENCE_SAMPLE: int = 10

    @staticmethod
    def _validate_weights(weights: WeightDict) -> None:
        """Ensure aggregated weights contain no NaN or Inf."""
        for name, param in weights.items():
            if torch.isnan(param).any():
                raise ValueError(f"NaN detected in aggregated weight: {name}")
            if torch.isinf(param).any():
                raise ValueError(f"Inf detected in aggregated weight: {name}")

    @staticmethod
    def _compute_divergence_sampled(
        client_updates: Dict[str, Tuple[WeightDict, Metrics]],
        global_weights: WeightDict,
        max_clients: int = 10,
    ) -> float:
        """
        Mean squared L2-norm divergence between client weights and the global model.

        For large fleets (> ``max_clients``) a random sample is used so the
        computation stays O(max_clients × n_params) rather than O(n × n_params).
        This is an unbiased estimator when the fleet is large relative to the sample.
        """
        import random as _random

        all_ids = list(client_updates.keys())
        sample_ids = (
            all_ids
            if len(all_ids) <= max_clients
            else _random.sample(all_ids, max_clients)
        )

        total = 0.0
        count = 0
        for hid in sample_ids:
            cw = client_updates[hid][0]
            for key in global_weights:
                if key in cw:
                    total += torch.norm((cw[key].float() - global_weights[key].float())).item() ** 2
                    count += 1
        return total / max(count, 1)

    # ─── Save / Evaluate ─────────────────────────────────────

    def save_global_model(self, tokenizer: AutoTokenizer, round_num: int,
                          save_dir: Optional[str] = None) -> str:
        """Save the aggregated global model as a LoRA adapter.

        Every file is verified and fsync'd to survive Colab/Kaggle disconnects.
        """
        save_dir = save_dir or self.cfg.global_model_dir
        round_dir = os.path.join(save_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        # Free any lingering memory before loading the base model.
        # Use float16 (not float32) to halve RAM usage — the base model
        # is only needed to structure the LoRA save; precision doesn't
        # matter here.  float32 loading was the root cause of OOM crashes
        # on Colab free tier (12.7 GB RAM).
        import gc
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass  # GPU context may be corrupt after training — safe to ignore

        model = self._load_peft_model(self.global_weights, dtype=torch.float16)
        model.save_pretrained(round_dir)

        del model
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        tokenizer.save_pretrained(round_dir)

        metadata = {
            "round": round_num,
            "config": self.cfg.to_dict(),
            "hospital_contributions": self.hospital_contributions,
        }
        meta_path = os.path.join(round_dir, "fl_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # ── Verify & fsync ALL saved files ────────────────────────
        saved_files = [
            f for f in os.listdir(round_dir)
            if os.path.isfile(os.path.join(round_dir, f))
        ]
        if not saved_files:
            raise RuntimeError(
                f"save_global_model: no files written to {round_dir}"
            )

        total_bytes = 0
        for fname in saved_files:
            fpath = os.path.join(round_dir, fname)
            fsize = os.path.getsize(fpath)
            if fsize == 0 and fname != "__init__.py":
                logger.warning("Zero-byte file after save: %s", fpath)
            total_bytes += fsize
            # Force each file to disk (survives Colab/Kaggle disconnect)
            try:
                fd = os.open(fpath, os.O_RDONLY)
                os.fsync(fd)
                os.close(fd)
            except OSError:
                pass  # best-effort — some FS don't support fsync on read fd

        size_mb = total_bytes / (1024 * 1024)
        logger.info(
            "Global model saved & verified: %s (%d files, %.1f MB)",
            round_dir, len(saved_files), size_mb,
        )

        return round_dir

    def evaluate_global(self, tokenizer: AutoTokenizer,
                        eval_questions: List[str],
                        max_eval: int = 3) -> List[Dict[str, str]]:
        """
        Quick evaluation of the global model on test questions.

        Args:
            tokenizer: shared tokenizer instance.
            eval_questions: list of questions to evaluate.
            max_eval: maximum number of questions to run (default 3) to keep
                      evaluation fast during training. Pass len(eval_questions)
                      to evaluate all questions.
        """
        n = min(max_eval, len(eval_questions))
        logger.info("Evaluating global model on %d/%d questions...", n, len(eval_questions))

        import gc
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        model = self._load_peft_model(self.global_weights, dtype=torch.float16)
        model.to(self.device)
        model.eval()

        results = []
        for q in eval_questions[:n]:
            prompt = f"<|system|>\n{self.cfg.system_msg}</s>\n<|user|>\n{q}</s>\n<|assistant|>\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=200, temperature=0.7,
                    do_sample=True, pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append({"question": q, "response": response[:300]})
            logger.info("Q: %s...", q[:80])
            logger.info("A: %s...", response[:150])

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log evaluation results as a table and as individual text entries
        if results:
            final_round = self.cfg.fl_rounds - 1
            self.tracker.log_table("eval/responses", results, step=final_round)
            for i, r in enumerate(results):
                q_slug = self.tracker._slug(r["question"])
                self.tracker.log_text(
                    f"eval/q{i:02d}_{q_slug}",
                    f"Q: {r['question']}\n\nA: {r['response']}",
                    step=final_round,
                )

        return results

    # ─── Reporting ────────────────────────────────────────────

    def get_metrics(self) -> List[Metrics]:
        """Return a copy of round metrics to prevent external mutation."""
        return list(self.round_metrics)

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a full training report.

        Returns copies of internal metric collections so that callers cannot
        accidentally mutate server state by appending to the returned lists/dicts.
        """
        return {
            "config": self.cfg.to_dict(),
            "round_metrics": list(self.round_metrics),
            "hospital_contributions": dict(self.hospital_contributions),
        }
