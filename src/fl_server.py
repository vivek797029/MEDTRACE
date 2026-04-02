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

from fl_config import FLConfig, config as default_config

logger = logging.getLogger(__name__)

# Type aliases
WeightDict = OrderedDict  # OrderedDict[str, torch.Tensor]
Metrics = Dict[str, Any]


class FederatedServer:
    """Central aggregation server for federated MedTrace training."""

    def __init__(self, device: str = "cpu", cfg: FLConfig = default_config):
        self.device = device
        self.cfg = cfg
        self.global_weights: Optional[WeightDict] = None
        self.round_metrics: List[Metrics] = []
        self.hospital_contributions: Dict[str, Metrics] = {}

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
        size_kb = sum(p.nbytes for p in self.global_weights.values()) / 1024
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

        # Weighted average
        aggregated = OrderedDict()
        for key in all_weights[0].keys():
            aggregated[key] = sum(fracs[i] * all_weights[i][key] for i in range(len(all_weights)))

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

        logger.info("Aggregation complete (%.3fs) | Divergence: %.6f", elapsed, divergence)
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
                        eval_questions: List[str]) -> List[Dict[str, str]]:
        """Quick evaluation of the global model on test questions."""
        logger.info("Evaluating global model on %d questions...", len(eval_questions))

        model = self._load_peft_model(self.global_weights)
        model.to(self.device)
        model.eval()

        results = []
        for q in eval_questions[:3]:
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
        return results

    # ─── Reporting ────────────────────────────────────────────

    def get_metrics(self) -> List[Metrics]:
        return self.round_metrics

    def generate_report(self) -> Dict[str, Any]:
        """Generate a full training report."""
        return {
            "config": self.cfg.to_dict(),
            "round_metrics": self.round_metrics,
            "hospital_contributions": self.hospital_contributions,
        }
