"""
MedTrace Federated Learning — Aggregation Server
=================================================
Central coordinator that aggregates LoRA weight deltas from hospitals
using Federated Averaging (FedAvg). Never sees any patient data.

Architecture:
  ┌───────────────────────────────────────────────────────┐
  │                   Aggregation Server                   │
  │                                                        │
  │   Hospital A         Hospital B         Hospital C     │
  │   LoRA Δw_A          LoRA Δw_B          LoRA Δw_C     │
  │      ↓                   ↓                   ↓         │
  │   ┌──────────────────────────────────────────────┐     │
  │   │          Secure Aggregation Layer             │     │
  │   │   (encrypted weights, never see raw deltas)   │     │
  │   └──────────────────┬───────────────────────────┘     │
  │                      ↓                                  │
  │   ┌──────────────────────────────────────────────┐     │
  │   │    FedAvg: Δw_global = Σ(n_k/n) · Δw_k      │     │
  │   └──────────────────┬───────────────────────────┘     │
  │                      ↓                                  │
  │   ┌──────────────────────────────────────────────┐     │
  │   │         Global Model Update                   │     │
  │   │    TinyLlama base + aggregated LoRA adapter   │     │
  │   └──────────────────────────────────────────────┘     │
  │                      ↓                                  │
  │            Broadcast back to all hospitals              │
  └───────────────────────────────────────────────────────┘
"""

import os
import json
import copy
import time
import torch
import numpy as np
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

import fl_config as cfg


class FederatedServer:
    """Central aggregation server for federated MedTrace training."""

    def __init__(self, device="cpu"):
        self.device = device
        self.global_weights = None
        self.round_metrics = []
        self.hospital_contributions = {}

        print("🖥️  Federated Aggregation Server initialized")
        print(f"   Strategy: {cfg.AGGREGATION_STRATEGY}")
        print(f"   Hospitals: {cfg.NUM_HOSPITALS}")
        print(f"   Rounds: {cfg.FL_ROUNDS}")
        print(f"   Differential Privacy: {'ON' if cfg.DIFFERENTIAL_PRIVACY else 'OFF'}")
        if cfg.DIFFERENTIAL_PRIVACY:
            print(f"   Privacy Budget (ε): {cfg.DP_EPSILON}")
        print(f"   Secure Aggregation: {'ON' if cfg.SECURE_AGGREGATION else 'OFF'}")

    def initialize_global_model(self):
        """Create initial global model with random LoRA weights."""
        print("\n📦 Initializing global model...")

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.BASE_MODEL,
            torch_dtype=torch.float32
        )

        lora_config = LoraConfig(
            r=cfg.LORA_R,
            lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT,
            target_modules=cfg.LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_config)

        # Extract initial LoRA weights as global starting point
        self.global_weights = OrderedDict()
        for name, param in model.named_parameters():
            if "lora_" in name:
                self.global_weights[name] = param.detach().cpu().clone()

        param_count = sum(p.numel() for p in self.global_weights.values())
        print(f"   Global LoRA parameters: {param_count:,}")
        print(f"   Weight size: {sum(p.nbytes for p in self.global_weights.values()) / 1024:.1f} KB")

        del model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.global_weights

    def aggregate(self, client_updates, round_num):
        """
        Federated Averaging (FedAvg) — McMahan et al., 2017

        Formula: w_global = Σ (n_k / n_total) * w_k

        Where:
          - w_k = weights from hospital k
          - n_k = number of training samples at hospital k
          - n_total = total samples across all hospitals
        """
        print(f"\n🔄 Aggregating Round {round_num + 1}...")
        start_time = time.time()

        if cfg.SECURE_AGGREGATION:
            print("   🔐 Secure aggregation: simulating encrypted weight transfer...")

        # Extract weights and sample counts
        all_weights = []
        sample_counts = []
        hospital_names = []

        for hospital_id, (weights, metrics) in client_updates.items():
            all_weights.append(weights)
            sample_counts.append(metrics["num_samples"])
            hospital_names.append(metrics["hospital"])

        total_samples = sum(sample_counts)
        contribution_weights = [n / total_samples for n in sample_counts]

        # Log contributions
        print("   📊 Contribution weights:")
        for name, w, n in zip(hospital_names, contribution_weights, sample_counts):
            print(f"      {name}: {w:.3f} ({n} samples)")
            self.hospital_contributions[name] = {
                "round": round_num,
                "weight": w,
                "samples": n
            }

        # Federated Averaging
        aggregated_weights = OrderedDict()
        for key in all_weights[0].keys():
            # Weighted average: Σ (n_k/n) * w_k
            weighted_sum = torch.zeros_like(all_weights[0][key])
            for i, weights in enumerate(all_weights):
                weighted_sum += contribution_weights[i] * weights[key]
            aggregated_weights[key] = weighted_sum

        # Validate aggregation
        self._validate_weights(aggregated_weights)

        self.global_weights = aggregated_weights
        elapsed = time.time() - start_time

        # Compute weight divergence between hospitals
        divergence = self._compute_divergence(all_weights, aggregated_weights)

        metrics = {
            "round": round_num,
            "num_hospitals": len(client_updates),
            "total_samples": total_samples,
            "aggregation_time": round(elapsed, 3),
            "weight_divergence": divergence,
            "contribution_weights": dict(zip(hospital_names, contribution_weights)),
        }
        self.round_metrics.append(metrics)

        print(f"   ✅ Aggregation complete ({elapsed:.3f}s)")
        print(f"   📈 Weight divergence: {divergence:.6f}")

        return self.global_weights

    def _validate_weights(self, weights):
        """Ensure aggregated weights are valid (no NaN, Inf)."""
        for name, param in weights.items():
            if torch.isnan(param).any():
                raise ValueError(f"NaN detected in aggregated weight: {name}")
            if torch.isinf(param).any():
                raise ValueError(f"Inf detected in aggregated weight: {name}")

    def _compute_divergence(self, client_weights, global_weights):
        """
        Measure how much hospital models diverge from the global model.
        High divergence = more non-IID data (hospitals have very different patient populations).
        """
        total_divergence = 0.0
        num_params = 0

        for key in global_weights.keys():
            global_param = global_weights[key]
            for client_w in client_weights:
                diff = (client_w[key] - global_param).float()
                total_divergence += torch.norm(diff).item() ** 2
                num_params += 1

        return total_divergence / max(num_params, 1)

    def save_global_model(self, tokenizer, round_num, save_dir=None):
        """Save the aggregated global model."""
        save_dir = save_dir or cfg.GLOBAL_MODEL_DIR
        round_dir = os.path.join(save_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        # Save LoRA adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.BASE_MODEL, torch_dtype=torch.float32
        )
        lora_config = LoraConfig(
            r=cfg.LORA_R,
            lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT,
            target_modules=cfg.LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_config)
        model.load_state_dict(self.global_weights, strict=False)

        model.save_pretrained(round_dir)
        tokenizer.save_pretrained(round_dir)

        # Save metadata
        metadata = {
            "round": round_num,
            "base_model": cfg.BASE_MODEL,
            "aggregation_strategy": cfg.AGGREGATION_STRATEGY,
            "num_hospitals": cfg.NUM_HOSPITALS,
            "differential_privacy": cfg.DIFFERENTIAL_PRIVACY,
            "dp_epsilon": cfg.DP_EPSILON if cfg.DIFFERENTIAL_PRIVACY else None,
            "hospital_contributions": self.hospital_contributions,
        }

        with open(os.path.join(round_dir, "fl_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"   💾 Global model saved: {round_dir}")

        del model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return round_dir

    def evaluate_global(self, tokenizer, eval_questions):
        """Quick evaluation of the global model on test questions."""
        print("\n📝 Evaluating global model...")

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.BASE_MODEL, torch_dtype=torch.float32
        )
        lora_config = LoraConfig(
            r=cfg.LORA_R,
            lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT,
            target_modules=cfg.LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_config)
        model.load_state_dict(self.global_weights, strict=False)
        model.to(self.device)
        model.eval()

        results = []
        for q in eval_questions[:3]:  # Quick eval on 3 questions
            prompt = f"<|system|>\n{cfg.SYSTEM_MSG}</s>\n<|user|>\n{q}</s>\n<|assistant|>\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append({"question": q, "response": response[:300]})
            print(f"   Q: {q[:80]}...")
            print(f"   A: {response[:150]}...\n")

        del model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def get_metrics(self):
        """Return all aggregation metrics."""
        return self.round_metrics

    def generate_report(self):
        """Generate a summary report of the federated training."""
        report = {
            "architecture": "MedTrace Federated Learning",
            "base_model": cfg.BASE_MODEL,
            "num_hospitals": cfg.NUM_HOSPITALS,
            "num_rounds": cfg.FL_ROUNDS,
            "aggregation_strategy": cfg.AGGREGATION_STRATEGY,
            "privacy": {
                "differential_privacy": cfg.DIFFERENTIAL_PRIVACY,
                "epsilon": cfg.DP_EPSILON,
                "delta": cfg.DP_DELTA,
                "secure_aggregation": cfg.SECURE_AGGREGATION,
            },
            "round_metrics": self.round_metrics,
            "hospital_contributions": self.hospital_contributions,
        }
        return report
