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
import re
import time
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover — GPU path only
    torch = None  # type: ignore[assignment]

try:
    from datasets import Dataset
except ImportError:  # pragma: no cover — GPU path only
    Dataset = None  # type: ignore[assignment]

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover — GPU path only
    LoraConfig = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
except ImportError:  # pragma: no cover — GPU path only
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    DataCollatorForLanguageModeling = None  # type: ignore[assignment]
    Trainer = None  # type: ignore[assignment]
    TrainingArguments = None  # type: ignore[assignment]

from fl_config import FLConfig, HospitalConfig, Metrics, WeightDict, config as default_config
from fl_tracker import ExperimentTracker, NoOpTracker

# Imported lazily inside methods to avoid requiring fl_adaptive_dp at import time
# when the feature is disabled.  The type hint below is for IDE support only.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fl_adaptive_dp import AdaptiveDPMechanism

logger = logging.getLogger(__name__)


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

        # Populated by prepare_local_data(); None until that method is called.
        # train_local() will raise a clear error if called before data preparation.
        self.local_data: Optional[Dataset] = None

        logger.info("Initialized %s (%s)", self.name, hospital_config.location)

    # ─── Data Preparation ─────────────────────────────────────

    def prepare_local_data(self, full_dataset: Dataset, round_num: int = 0) -> Dataset:
        """
        Simulate non-IID data distribution.
        Each hospital gets a biased subset reflecting its specialty.

        Scalability notes
        -----------------
        * Keywords are compiled into a single regex pattern once per call so
          classification is O(|dataset|) rather than O(|dataset| × |keywords|).
        * If the dataset is smaller than ``num_samples``, sampling uses
          replacement so training always sees the configured batch size.
        * If zero specialty examples are found (unlikely but possible with
          unusual datasets), the hospital falls back to random sampling with a
          warning rather than crashing.
        """
        keywords = self.config.specialty_keywords
        num_samples = min(self.config.num_samples, len(full_dataset))
        specialty_ratio = self.config.specialty_ratio

        # Compile keyword list into one regex for O(n) classification.
        # re.escape protects against keywords that contain regex meta-chars.
        pattern = re.compile(
            "|".join(re.escape(kw) for kw in keywords),
            re.IGNORECASE,
        )

        # Classify examples: specialty vs general
        spec_idx: List[int] = []
        gen_idx: List[int] = []
        for i, example in enumerate(full_dataset):
            question = example.get("question", "")
            if pattern.search(question):
                spec_idx.append(i)
            else:
                gen_idx.append(i)

        # Sample according to specialty ratio
        seed = abs(hash(self.hospital_id) + round_num) % (2**32)
        rng = np.random.RandomState(seed)

        n_spec = min(int(num_samples * specialty_ratio), len(spec_idx))
        n_gen  = num_samples - n_spec

        chosen: List[int] = []
        if n_spec > 0 and spec_idx:
            chosen.extend(rng.choice(spec_idx, n_spec, replace=True).tolist())
        if n_gen > 0 and gen_idx:
            chosen.extend(rng.choice(gen_idx, n_gen, replace=True).tolist())

        # Fallback: if both pools are empty (should not happen with real data),
        # sample randomly from the full dataset rather than crashing.
        if not chosen:
            logger.warning(
                "%s: no samples matched specialty/general split — "
                "falling back to random sampling from full dataset.",
                self.name,
            )
            all_idx = list(range(len(full_dataset)))
            chosen = rng.choice(all_idx, num_samples, replace=True).tolist()

        # Shuffle
        seed2 = abs(hash(self.hospital_id) + round_num + 42) % (2**32)
        rng2 = np.random.RandomState(seed2)
        rng2.shuffle(chosen)

        self.local_data = full_dataset.select(chosen[:num_samples])
        logger.info(
            "%s: %d samples (specialty=%d, general=%d)",
            self.name, len(self.local_data), n_spec, len(chosen) - n_spec,
        )
        return self.local_data

    # ─── Local Training ───────────────────────────────────────

    def train_local(
        self,
        global_weights: Optional[WeightDict],
        tokenizer: AutoTokenizer,
        round_num: int,
        output_dir: str,
        base_model: Optional[AutoModelForCausalLM] = None,
        tracker: Optional[ExperimentTracker] = None,
        adaptive_dp_mechanism: Optional["AdaptiveDPMechanism"] = None,
        round_epsilon: Optional[float] = None,
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
            tracker: experiment tracker for logging per-client metrics.
                     Defaults to NoOpTracker (no logging).
            adaptive_dp_mechanism: optional AdaptiveDPMechanism instance.  When
                provided (and cfg.adaptive_dp.enabled), replaces the fixed
                Gaussian mechanism with per-client adaptive noise.  The
                mechanism is owned by the simulation orchestrator and shared
                across all hospital clients.
            round_epsilon: pre-computed ε allocation for this client this round,
                as returned by AdaptiveDPMechanism.compute_epsilon_allocation().
                Required when adaptive_dp_mechanism is not None.
        """
        _tracker = tracker if tracker is not None else NoOpTracker()
        if self.local_data is None:
            raise RuntimeError(
                f"{self.name}: call prepare_local_data() before train_local()"
            )

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

        # Log per-step training loss for the loss-curve chart
        # trainer.state.log_history contains {"loss": ..., "step": ...} dicts
        for entry in trainer.state.log_history:
            if "loss" in entry and "step" in entry:
                global_step = round_num * 1000 + int(entry["step"])
                _tracker.log(
                    {f"{self.hospital_id}/step_loss": entry["loss"]},
                    step=global_step,
                )

        # Extract LoRA weights
        lora_weights = self._extract_lora_weights(model)

        # Apply DP noise — adaptive per-client or standard global mechanism
        use_adaptive = (
            self.cfg.dp.enabled
            and self.cfg.adaptive_dp.enabled
            and adaptive_dp_mechanism is not None
            and round_epsilon is not None
        )
        if use_adaptive:
            lora_weights = adaptive_dp_mechanism.apply_noise(
                self.hospital_id, lora_weights, round_num, round_epsilon,
            )
            # Sync privacy_budget_spent from the mechanism's authoritative accounting
            self.privacy_budget_spent = adaptive_dp_mechanism.get_budget_spent(
                self.hospital_id
            )
        elif self.cfg.dp.enabled:
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
            "dp_mode": "adaptive" if use_adaptive else ("standard" if self.cfg.dp.enabled else "none"),
        }
        self.round_metrics.append(metrics)
        logger.info(
            "%s — Loss: %.4f | Time: %.1fs | Params: %s",
            self.name, train_loss, elapsed,
            f"{metrics['lora_params_shared']:,}",
        )

        # Log per-client round metrics to the experiment tracker
        budget_pct = (
            (self.privacy_budget_spent / self.cfg.dp.epsilon * 100)
            if self.cfg.dp.enabled and self.cfg.dp.epsilon > 0 else 0.0
        )
        tracker_payload = {
            f"{self.hospital_id}/train_loss": train_loss,
            f"{self.hospital_id}/num_samples": len(self.local_data),
            f"{self.hospital_id}/training_time_seconds": elapsed,
            f"{self.hospital_id}/lora_params_shared": metrics["lora_params_shared"],
            f"{self.hospital_id}/privacy_budget_spent": self.privacy_budget_spent,
            f"{self.hospital_id}/privacy_budget_pct": budget_pct,
        }
        if use_adaptive and round_epsilon is not None:
            tracker_payload[f"{self.hospital_id}/adaptive_dp_round_epsilon"] = round_epsilon
        _tracker.log(tracker_payload, step=round_num)

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
        """Return a copy of all training metrics for this hospital."""
        return list(self.round_metrics)
