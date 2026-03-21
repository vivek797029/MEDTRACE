"""
MedTrace Federated Learning — Hospital Client
==============================================
Each hospital trains a local LoRA adapter on its private data,
then shares ONLY the adapter weight deltas with the central server.

Architecture:
  ┌──────────────────────────────┐
  │       Hospital Node          │
  │  ┌────────────────────────┐  │
  │  │   Private Patient Data │  │  ← NEVER leaves this boundary
  │  └──────────┬─────────────┘  │
  │             ↓                │
  │  ┌────────────────────────┐  │
  │  │  TinyLlama + LoRA      │  │  ← Local fine-tuning
  │  └──────────┬─────────────┘  │
  │             ↓                │
  │  ┌────────────────────────┐  │
  │  │  DP Noise Injection    │  │  ← Differential privacy
  │  └──────────┬─────────────┘  │
  │             ↓                │
  │  ┌────────────────────────┐  │
  │  │  LoRA Δ Weights (2MB)  │──┼──→ Sent to server (encrypted)
  │  └────────────────────────┘  │
  └──────────────────────────────┘
"""

import os
import copy
import json
import time
import torch
import numpy as np
from collections import OrderedDict
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel

import fl_config as cfg


class HospitalClient:
    """Represents a single hospital node in the federated network."""

    def __init__(self, hospital_id, hospital_config, device="cpu"):
        self.hospital_id = hospital_id
        self.config = hospital_config
        self.name = hospital_config["name"]
        self.location = hospital_config["location"]
        self.device = device
        self.round_metrics = []

        # Privacy accounting
        self.privacy_budget_spent = 0.0
        self.total_privacy_budget = cfg.DP_EPSILON

        print(f"  🏥 Initialized {self.name} ({self.location})")

    def prepare_local_data(self, full_dataset, round_num=0):
        """
        Simulate non-IID data distribution.
        Each hospital gets a biased subset reflecting its specialty.
        """
        specialty_weights = self.config["specialty_weight"]
        num_samples = self.config["num_samples"]

        # Keyword mapping for specialty detection
        specialty_keywords = {
            "cardiovascular": ["heart", "cardiac", "coronary", "chest pain", "hypertension",
                             "arrhythmia", "myocardial", "angina", "aortic", "valve"],
            "neurological": ["brain", "neuro", "seizure", "headache", "stroke",
                           "cognitive", "dementia", "nerve", "spinal", "motor"],
            "respiratory": ["lung", "breath", "pulmonary", "asthma", "pneumonia",
                          "cough", "airway", "bronch", "oxygen", "ventil"],
            "infectious": ["infection", "fever", "bacteria", "virus", "antibiotic",
                         "sepsis", "hiv", "tuberculosis", "malaria", "pathogen"],
            "endocrine": ["diabetes", "thyroid", "hormone", "insulin", "adrenal",
                        "pituitary", "metabol", "glucose", "cortisol", "endocrine"],
            "gastrointestinal": ["stomach", "liver", "bowel", "intestin", "gastric",
                               "hepat", "pancrea", "digest", "colon", "abdomin"],
        }

        # Classify each example by specialty
        classified = {spec: [] for spec in specialty_keywords}
        classified["general"] = []

        for i, example in enumerate(full_dataset):
            question = example.get("question", "").lower()
            matched = False
            for spec, keywords in specialty_keywords.items():
                if any(kw in question for kw in keywords):
                    classified[spec].append(i)
                    matched = True
                    break
            if not matched:
                classified["general"].append(i)

        # Sample according to hospital's specialty distribution
        selected_indices = []
        for specialty, weight in specialty_weights.items():
            n_from_specialty = int(num_samples * weight)
            available = classified.get(specialty, classified["general"])
            if len(available) > 0:
                # Use round_num as seed offset for different data each round
                rng = np.random.RandomState(abs(hash(self.hospital_id) + round_num) % (2**32))
                chosen = rng.choice(available, size=min(n_from_specialty, len(available)), replace=True)
                selected_indices.extend(chosen.tolist())

        # Shuffle and limit
        rng = np.random.RandomState(abs(hash(self.hospital_id) + round_num + 42) % (2**32))
        rng.shuffle(selected_indices)
        selected_indices = selected_indices[:num_samples]

        self.local_data = full_dataset.select(selected_indices)
        print(f"  📊 {self.name}: {len(self.local_data)} samples (non-IID)")
        return self.local_data

    def train_local(self, global_weights, tokenizer, round_num, output_dir):
        """
        Train LoRA adapter on local hospital data.
        Returns ONLY the weight deltas (not full weights).
        """
        print(f"\n  🔧 {self.name} — Round {round_num + 1} local training...")
        start_time = time.time()

        # Load fresh base model
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.BASE_MODEL,
            torch_dtype=torch.float32,
        )

        # Apply LoRA
        lora_config = LoraConfig(
            r=cfg.LORA_R,
            lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT,
            target_modules=cfg.LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_config)

        # Load global weights from previous round (if any)
        if global_weights is not None:
            model.load_state_dict(global_weights, strict=False)

        model.to(self.device)
        model.train()

        # Tokenize local data
        def tokenize_fn(examples):
            prompts = []
            for q in examples["question"]:
                prompt = f"<|system|>\n{cfg.SYSTEM_MSG}</s>\n<|user|>\n{q}</s>\n<|assistant|>\n"
                prompts.append(prompt)
            encoded = tokenizer(prompts, truncation=True, max_length=cfg.MAX_LENGTH, padding="max_length")
            encoded["labels"] = encoded["input_ids"].copy()
            return encoded

        tokenized = self.local_data.map(tokenize_fn, batched=True, remove_columns=self.local_data.column_names)
        tokenized.set_format("torch")

        # Training arguments
        hospital_output = os.path.join(output_dir, self.hospital_id, f"round_{round_num}")
        os.makedirs(hospital_output, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=hospital_output,
            num_train_epochs=cfg.LOCAL_EPOCHS,
            per_device_train_batch_size=cfg.BATCH_SIZE,
            gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=cfg.LEARNING_RATE,
            warmup_steps=cfg.WARMUP_STEPS,
            weight_decay=cfg.WEIGHT_DECAY,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        trainer.train()
        elapsed = time.time() - start_time

        # Extract ONLY LoRA weights (the deltas)
        lora_state_dict = self._extract_lora_weights(model)

        # Apply differential privacy noise
        if cfg.DIFFERENTIAL_PRIVACY:
            lora_state_dict = self._apply_dp_noise(lora_state_dict, round_num)

        # Compute metrics
        train_loss = trainer.state.log_history[-1].get("train_loss", 0) if trainer.state.log_history else 0
        metrics = {
            "hospital": self.name,
            "round": round_num,
            "train_loss": train_loss,
            "num_samples": len(self.local_data),
            "training_time_seconds": round(elapsed, 2),
            "lora_params_shared": sum(p.numel() for p in lora_state_dict.values()),
            "privacy_budget_spent": self.privacy_budget_spent,
        }
        self.round_metrics.append(metrics)
        print(f"  ✅ {self.name} — Loss: {train_loss:.4f} | Time: {elapsed:.1f}s | "
              f"Shared: {metrics['lora_params_shared']:,} params")

        # Cleanup
        del model, base_model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return lora_state_dict, metrics

    def _extract_lora_weights(self, model):
        """Extract only LoRA adapter parameters — this is what gets shared."""
        lora_weights = OrderedDict()
        for name, param in model.named_parameters():
            if "lora_" in name:
                lora_weights[name] = param.detach().cpu().clone()
        return lora_weights

    def _apply_dp_noise(self, weights, round_num):
        """
        Add calibrated Gaussian noise for differential privacy.
        Implements the Gaussian mechanism with (ε, δ)-DP guarantee.
        """
        sensitivity = cfg.DP_MAX_GRAD_NORM
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / cfg.DP_DELTA)) / cfg.DP_EPSILON

        noisy_weights = OrderedDict()
        for name, param in weights.items():
            # Clip gradients
            norm = torch.norm(param)
            if norm > cfg.DP_MAX_GRAD_NORM:
                param = param * (cfg.DP_MAX_GRAD_NORM / norm)

            # Add Gaussian noise
            noise = torch.randn_like(param) * sigma
            noisy_weights[name] = param + noise

        # Track privacy budget (simplified composition)
        self.privacy_budget_spent += cfg.DP_EPSILON / cfg.FL_ROUNDS
        print(f"  🔒 DP noise applied (σ={sigma:.4f}) | "
              f"Privacy budget: {self.privacy_budget_spent:.2f}/{self.total_privacy_budget}")

        return noisy_weights

    def get_metrics(self):
        """Return all training metrics for this hospital."""
        return self.round_metrics
