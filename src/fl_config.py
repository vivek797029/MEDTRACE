"""
MedTrace Federated Learning Configuration
==========================================
Immutable, validated configuration using dataclasses.
Override values via FLConfig.create() — never mutate after creation.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ─── Shared Type Aliases ─────────────────────────────────────────────────────
# Defined here so client, server, and simulation modules all import from a
# single source of truth instead of duplicating these aliases.
# torch is intentionally NOT imported here — fl_config has no runtime torch
# dependency and must remain importable without the ML stack installed.
WeightDict = OrderedDict  # OrderedDict[str, torch.Tensor]  (torch for type doc only)
Metrics = Dict[str, Any]


@dataclass(frozen=True)
class HospitalConfig:
    """Configuration for a single hospital node."""
    name: str
    location: str
    specialty_keywords: List[str]
    num_samples: int = 1500
    specialty_ratio: float = 0.6  # fraction of samples from specialty

    def __post_init__(self):
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        if not 0.0 <= self.specialty_ratio <= 1.0:
            raise ValueError(f"specialty_ratio must be 0-1, got {self.specialty_ratio}")


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple = ("q_proj", "v_proj")
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.r}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"LoRA dropout must be [0, 1), got {self.dropout}")


@dataclass(frozen=True)
class DPConfig:
    """Differential privacy configuration."""
    enabled: bool = True
    epsilon: float = 8.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    secure_aggregation: bool = True

    def __post_init__(self):
        if self.enabled:
            if self.epsilon <= 0:
                raise ValueError(f"DP epsilon must be positive, got {self.epsilon}")
            if not 0 < self.delta < 1:
                raise ValueError(f"DP delta must be (0, 1), got {self.delta}")
            if self.max_grad_norm <= 0:
                raise ValueError(f"max_grad_norm must be positive, got {self.max_grad_norm}")

    @property
    def sigma(self) -> float:
        """Gaussian noise std for the Gaussian mechanism."""
        if not self.enabled:
            return 0.0
        return self.max_grad_norm * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 2e-4
    batch_size: int = 4
    max_length: int = 512
    warmup_steps: int = 10
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    logging_steps: int = 5

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


@dataclass(frozen=True)
class FLConfig:
    """
    Top-level federated learning configuration. Immutable after creation.

    Usage:
        # Default config
        cfg = FLConfig()

        # Custom config — never mutate, create new
        cfg = FLConfig.create(fl_rounds=5, dp=DPConfig(epsilon=4.0))

        # Quick demo config
        cfg = FLConfig.quick_demo()
    """
    # Model
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Federated learning
    fl_rounds: int = 20
    local_epochs: int = 1
    aggregation_strategy: str = "fedavg"

    # Sub-configs
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Hospitals
    hospitals: Dict[str, HospitalConfig] = field(default_factory=lambda: {
        "hospital_A": HospitalConfig(
            name="Metro General (Cardiology)",
            location="New York, USA",
            specialty_keywords=["heart", "cardiac", "coronary", "chest pain",
                                "myocardial", "arrhythmia", "angina", "aortic"],
            num_samples=1500,
        ),
        "hospital_B": HospitalConfig(
            name="Royal London (Neurology)",
            location="London, UK",
            specialty_keywords=["brain", "neuro", "seizure", "headache",
                                "stroke", "cognitive", "dementia", "nerve"],
            num_samples=1200,
        ),
        "hospital_C": HospitalConfig(
            name="AIIMS Delhi (Infectious)",
            location="New Delhi, India",
            specialty_keywords=["infection", "fever", "bacteria", "virus",
                                "antibiotic", "sepsis", "tuberculosis", "malaria"],
            num_samples=1300,
        ),
    })

    # Paths
    output_dir: str = "outputs/federated"
    global_model_dir: str = "outputs/federated/global_model"
    hospital_models_dir: str = "outputs/federated/hospital_models"
    metrics_dir: str = "outputs/federated/metrics"

    # System message
    system_msg: str = (
        "You are MedTrace, a federated clinical reasoning system. "
        "You generate step-by-step, auditable diagnostic reasoning chains "
        "with typed steps (symptom, finding, mechanism, rule, inference, conclusion) "
        "and explicit dependency tracking. Your training preserves patient privacy "
        "through federated learning — no patient data ever leaves the hospital."
    )

    def __post_init__(self):
        if self.fl_rounds <= 0:
            raise ValueError(f"fl_rounds must be positive, got {self.fl_rounds}")
        if self.local_epochs <= 0:
            raise ValueError(f"local_epochs must be positive, got {self.local_epochs}")
        if self.aggregation_strategy not in ("fedavg",):
            raise ValueError(f"Unknown aggregation: {self.aggregation_strategy}")

    @property
    def num_hospitals(self) -> int:
        return len(self.hospitals)

    @classmethod
    def create(cls, **overrides) -> "FLConfig":
        """Create config with overrides. Sub-configs can be passed as objects."""
        return cls(**overrides)

    @classmethod
    def quick_demo(cls) -> "FLConfig":
        """Minimal config for fast testing — 2 rounds, 100 samples."""
        small_hospitals = {
            hid: HospitalConfig(
                name=h.name, location=h.location,
                specialty_keywords=h.specialty_keywords, num_samples=100,
            )
            for hid, h in cls().hospitals.items()
        }
        return cls(fl_rounds=2, hospitals=small_hospitals)

    def to_dict(self) -> dict:
        """Serialize config to dict for metadata/logging."""
        return {
            "base_model": self.base_model,
            "fl_rounds": self.fl_rounds,
            "local_epochs": self.local_epochs,
            "aggregation_strategy": self.aggregation_strategy,
            "num_hospitals": self.num_hospitals,
            "lora": {"r": self.lora.r, "alpha": self.lora.alpha,
                     "dropout": self.lora.dropout,
                     "target_modules": list(self.lora.target_modules)},
            "dp": {"enabled": self.dp.enabled, "epsilon": self.dp.epsilon,
                   "delta": self.dp.delta, "max_grad_norm": self.dp.max_grad_norm},
            "training": {"lr": self.training.learning_rate,
                         "batch_size": self.training.batch_size,
                         "max_length": self.training.max_length},
        }


# ─── Default instance for backward compatibility ─────────────────
# Import this in modules: `from fl_config import config`
# Never mutate it. Create a new FLConfig() if you need different values.
config = FLConfig()
