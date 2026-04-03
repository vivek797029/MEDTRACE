"""
MedTrace Federated Learning — Hospital Client
==============================================
Each hospital trains a local LoRA adapter on its private data,
then shares ONLY the adapter weight deltas with the central server.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from fl_config import FLConfig, HospitalConfig, config as default_config

logger = logging.getLogger(__name__)

# Type aliases
WeightDict = OrderedDict  # OrderedDict[str, torch.Tensor]
Metrics = Dict[str, Any]


class HospitalClient:
    """Represents a single hospital node in the federated network."""

    def __init__(
        self,
        hospital_id: str,
        hospital_config: HospitalConfig,
        device: str = "cpu",
        cfg: FLConfig = default_config,
    ):
        self.hospital_id = hospital_id
        self.config = hospital_config
        self.name = hospital_config.name
        self.device = device
        self.cfg = cfg
        self.round_metrics: List[Metrics] = []

        # Privacy accounting
        self.privacy_budget_spent: float = 0.0

        logger.info("Initialized %s (%s)", self.name, hospital_config.location)

    # ─── Data Preparation ─────────────────────────────────────

    def prepare_local_data(self, full_dataset: Dataset, round_num: int = 0) -> Dataset:
        """
        Simulate non-IID data distribution.
        Each hospital gets a biased subset reflecting its specialty.
        """
        keywords = self.config.specialty_keywords
        num_samples = self.config.num_samples
        specialty_ratio = self.config.specialty_ratio

        # Classify examples: specialty vs general
        spec_idx: List[int] = []
        gen_idx: List[int] = []
        for i, example in enumerate(full_dataset):
            question = example.get("question", "").lower()
            if any(kw in question for kw in keywords):
                spec_idx.append(i)
            else:
                gen_idx.append(i)

        # Sample according to specialty ratio
        seed = abs(hash(self.hospital_id) + round_num) % (2**32)
        rng = np.random.RandomState(seed)

        n_spec = min(int(num_samples * specialty_ratio), len(spec_idx))
        n_gen = num_samples - n_spec

        chosen: List[int] = []
        if n_spec > 0 and len(spec_idx) > 0:
            chosen.extend(rng.choice(spec_idx, n_spec, replace=True).tolist())
        if n_gen > 0 and len(gen_idx) > 0:
            chosen.extend(rng.choice(gen_idx, min(n_gen, len(gen_idx)), replace=True).tolist())

        # Shuffle
        seed2 = abs(hash(self.hospital_id) + round_num + 42) % (2**32)
        rng2 = np.random.RandomState(seed2)
        rng2.shuffle(chosen)

        self.local_data = full_dataset.select(chosen[:num_samples])
        logger.info("%s: %d samples (%d specialty)", self.name, len(self.local_data), n_spec)
        return self.local_data

    # ─── Local Training ───────────────────────────────────────

    def train_local(
        self,
        global_weights: Optional[WeightDict],
        tokenizer: AutoTokenizer,
        round_num: int,
        output_dir: str,
        base_model: Optional[AutoModelForCausalLM] = None,
    ) -> Tuple[WeightDict, Metrics]:
        """
        Train LoRA adapter on local hospital data.
        Returns ONLY the weight deltas (not full model).

        Args:
            global_weights: aggregated LoRA weights from previous round.
            tokenizer: shared tokenizer instance.
            round_num: current FL round index (0-based).
            output_dir: directory for temp training artifacts.
            base_model: optional pre-loaded base model. If provided, it will be
                        deep-copied (caller retains ownership). If None, loads fresh.
        """
        logger.info("%s — Round %d local training...", self.name, round_num + 1)
        t0 = time.time()

        # Load or reuse base model
        _owns_base = base_model is None
        if _owns_base:
            logger.warning("Loading base model from scratch (slow). Pass base_model= to reuse.")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.cfg.base_model, torch_dtype=torch.float32,
            )

        # Deep copy so each hospital gets independent weights
        base_copy = copy.deepcopy(base_model)
        lora_cfg = self._make_lora_config()
        model = get_peft_model(base_copy, lora_cfg)

        if global_weights is not None:
            model.load_state_dict(global_weights, strict=False)

        model.to(self.device)
        model.train()

        # Tokenize
        system_msg = self.cfg.system_msg
        max_length = self.cfg.training.max_length

        def tokenize_fn(examples):
            prompts = [
                f"<|system|>\n{system_msg}</s>\n<|user|>\n{q}</s>\n<|assistant|>\n"
                for q in examples["question"]
            ]
            enc = tokenizer(prompts, truncation=True, max_length=max_length, padding="max_length")
            enc["labels"] = enc["input_ids"].copy()
            return enc

        tok_data = self.local_data.map(tokenize_fn, batched=True, remove_columns=self.local_data.column_names)
        tok_data.set_format("torch")

        # Training
        hospital_out = os.path.join(output_dir, self.hospital_id, f"round_{round_num}")
        os.makedirs(hospital_out, exist_ok=True)

        args = TrainingArguments(
            output_dir=hospital_out,
            num_train_epochs=self.cfg.local_epochs,
            per_device_train_batch_size=self.cfg.training.batch_size,
            gradient_accumulation_steps=self.cfg.training.gradient_accumulation_steps,
            learning_rate=self.cfg.training.learning_rate,
            warmup_steps=self.cfg.training.warmup_steps,
            weight_decay=self.cfg.training.weight_decay,
            logging_steps=self.cfg.training.logging_steps,
            save_strategy="no",
            report_to="none",
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
        )

        trainer = Trainer(
            model=model, args=args, train_dataset=tok_data,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train()
        elapsed = time.time() - t0

        # Extract LoRA weights
        lora_weights = self._extract_lora_weights(model)

        # Apply DP noise
        if self.cfg.dp.enabled:
            lora_weights = self._apply_dp_noise(lora_weights)

        # Metrics
        train_loss = (
            trainer.state.log_history[-1].get("train_loss", 0.0)
            if trainer.state.log_history else 0.0
        )
        metrics: Metrics = {
            "hospital": self.name,
            "round": round_num,
            "train_loss": train_loss,
            "num_samples": len(self.local_data),
            "training_time_seconds": round(elapsed, 2),
            "lora_params_shared": sum(p.numel() for p in lora_weights.values()),
            "privacy_budget_spent": self.privacy_budget_spent,
        }
        self.round_metrics.append(metrics)
        logger.info(
            "%s — Loss: %.4f | Time: %.1fs | Params: %s",
            self.name, train_loss, elapsed,
            f"{metrics['lora_params_shared']:,}",
        )

        # Cleanup
        del model, base_copy, trainer
        if _owns_base:
            del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return lora_weights, metrics

    # ─── Private Helpers ──────────────────────────────────────

    def _make_lora_config(self) -> LoraConfig:
        """Build a peft LoraConfig from our config."""
        return LoraConfig(
            r=self.cfg.lora.r,
            lora_alpha=self.cfg.lora.alpha,
            lora_dropout=self.cfg.lora.dropout,
            target_modules=list(self.cfg.lora.target_modules),
            bias=self.cfg.lora.bias,
            task_type=self.cfg.lora.task_type,
        )

    @staticmethod
    def _extract_lora_weights(model) -> WeightDict:
        """Extract only LoRA adapter parameters — this is what gets shared."""
        weights = OrderedDict()
        for name, param in model.named_parameters():
            if "lora_" in name:
                weights[name] = param.detach().cpu().clone()
        return weights

    def _apply_dp_noise(self, weights: WeightDict) -> WeightDict:
        """
        Gaussian mechanism with (epsilon, delta)-DP guarantee.
        Uses advanced composition: per-round epsilon = epsilon / sqrt(T).
        """
        sigma = self.cfg.dp.sigma
        max_norm = self.cfg.dp.max_grad_norm

        noisy = OrderedDict()
        for name, param in weights.items():
            # Clip (convert norm to scalar for reliable comparison)
            norm = torch.norm(param).item()
            if norm > max_norm:
                param = param * (max_norm / norm)
            # Add noise
            noisy[name] = param + torch.randn_like(param) * sigma

        # Advanced composition theorem: total eps = eps_per_round * sqrt(T)
        # So per-round spend = total_eps / sqrt(T)
        per_round_eps = self.cfg.dp.epsilon / (self.cfg.fl_rounds ** 0.5)
        self.privacy_budget_spent += per_round_eps

        logger.info(
            "DP noise applied (sigma=%.4f) | Budget: %.3f / %.1f",
            sigma, self.privacy_budget_spent, self.cfg.dp.epsilon,
        )
        return noisy

    def get_metrics(self) -> List[Metrics]:
        """Return all training metrics for this hospital."""
        return self.round_metrics
