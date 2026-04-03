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

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from fl_config import FLConfig, Metrics, WeightDict, config as default_config
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

    def _load_peft_model(self, weights: Optional[WeightDict] = None):
        """Load base model with LoRA, optionally inject weights. Caller owns cleanup."""
        base = AutoModelForCausalLM.from_pretrained(self.cfg.base_model, torch_dtype=torch.float32)
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

    def aggregate(
        self,
        client_updates: Dict[str, Tuple[WeightDict, Metrics]],
        round_num: int,
    ) -> WeightDict:
        """
        Federated Averaging (FedAvg) — McMahan et al., 2017
        w_global = sum( (n_k / n_total) * w_k )
        """
        logger.info("Aggregating Round %d...", round_num + 1)
        t0 = time.time()

        if self.cfg.dp.secure_aggregation:
            logger.info("Secure aggregation: simulating encrypted weight transfer")

        all_weights: List[WeightDict] = []
        sample_counts: List[int] = []
        hospital_names: List[str] = []

        for hospital_id, (weights, metrics) in client_updates.items():
            all_weights.append(weights)
            sample_counts.append(metrics["num_samples"])
            hospital_names.append(metrics["hospital"])

        total_samples = sum(sample_counts)
        fracs = [n / total_samples for n in sample_counts]

        for name, frac, n in zip(hospital_names, fracs, sample_counts):
            logger.info("  %s: weight=%.3f (%d samples)", name, frac, n)
            self.hospital_contributions[name] = {"round": round_num, "weight": frac, "samples": n}

        # Weighted average — use explicit zero init to avoid int+Tensor ambiguity
        aggregated = OrderedDict()
        for key in all_weights[0].keys():
            acc = fracs[0] * all_weights[0][key]
            for i in range(1, len(all_weights)):
                acc = acc + fracs[i] * all_weights[i][key]
            aggregated[key] = acc

        self._validate_weights(aggregated)
        self.global_weights = aggregated
        elapsed = time.time() - t0

        divergence = self._compute_divergence(all_weights, aggregated)
        metrics: Metrics = {
            "round": round_num,
            "num_hospitals": len(client_updates),
            "total_samples": total_samples,
            "aggregation_time": round(elapsed, 3),
            "weight_divergence": divergence,
            "contribution_weights": dict(zip(hospital_names, fracs)),
        }
        self.round_metrics.append(metrics)

        # ── Compute and log round-level aggregated loss ──────────────────────
        # Weighted average of per-hospital losses (FedAvg loss)
        losses = [m["train_loss"] for _, m in client_updates.values()]
        weights_losses = [
            m["num_samples"] / total_samples for _, m in client_updates.values()
        ]
        avg_loss = sum(l * w for l, w in zip(losses, weights_losses))

        self.tracker.log(
            {
                "round/avg_loss": avg_loss,
                "round/min_loss": min(losses),
                "round/max_loss": max(losses),
                "round/weight_divergence": divergence,
                "round/aggregation_time": elapsed,
                "round/total_samples": total_samples,
                "round/num_hospitals": len(client_updates),
            },
            step=round_num,
        )
        # Per-hospital contribution weights (useful for stacked bar chart)
        contribution_log = {
            f"contribution/{name}": frac
            for name, frac in zip(hospital_names, fracs)
        }
        self.tracker.log(contribution_log, step=round_num)

        logger.info("Aggregation complete (%.3fs) | Divergence: %.6f | Avg loss: %.4f",
                    elapsed, divergence, avg_loss)
        return self.global_weights

    # ─── Validation ───────────────────────────────────────────

    @staticmethod
    def _validate_weights(weights: WeightDict) -> None:
        """Ensure aggregated weights contain no NaN or Inf."""
        for name, param in weights.items():
            if torch.isnan(param).any():
                raise ValueError(f"NaN detected in aggregated weight: {name}")
            if torch.isinf(param).any():
                raise ValueError(f"Inf detected in aggregated weight: {name}")

    @staticmethod
    def _compute_divergence(client_weights: List[WeightDict], global_weights: WeightDict) -> float:
        """Measure how much hospital models diverge from the global model."""
        total = 0.0
        count = 0
        for key in global_weights.keys():
            g = global_weights[key]
            for cw in client_weights:
                total += torch.norm((cw[key] - g).float()).item() ** 2
                count += 1
        return total / max(count, 1)

    # ─── Save / Evaluate ─────────────────────────────────────

    def save_global_model(self, tokenizer: AutoTokenizer, round_num: int,
                          save_dir: Optional[str] = None) -> str:
        """Save the aggregated global model as a LoRA adapter."""
        save_dir = save_dir or self.cfg.global_model_dir
        round_dir = os.path.join(save_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        model = self._load_peft_model(self.global_weights)
        model.save_pretrained(round_dir)
        tokenizer.save_pretrained(round_dir)

        metadata = {
            "round": round_num,
            "config": self.cfg.to_dict(),
            "hospital_contributions": self.hospital_contributions,
        }
        with open(os.path.join(round_dir, "fl_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Global model saved: %s", round_dir)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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

        model = self._load_peft_model(self.global_weights)
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
