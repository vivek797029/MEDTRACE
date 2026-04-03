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
class EvalConfig:
    """
    Configuration for the evaluation system.

    Controls when evaluation runs, how many samples are used, and where
    results (JSON, CSV, plots) are written.  All randomness is seeded via
    ``eval_seed`` so every experiment is exactly reproducible.
    """
    # When to evaluate
    enabled: bool = True
    eval_every_n_rounds: int = 1        # 1 = every round, 2 = every other, etc.

    # Dataset split
    num_eval_samples: int = 200         # questions sampled from MedQA test set
    eval_seed: int = 42                 # fixed seed for reproducible splits

    # MCQ scoring
    max_eval_batch: int = 32            # max examples per accuracy pass (memory)

    # Output
    output_dir: str = "outputs/evaluation"
    save_plots: bool = True
    plot_format: str = "png"            # "png" | "pdf" | "svg"
    plot_dpi: int = 150

    def __post_init__(self):
        if self.eval_every_n_rounds < 1:
            raise ValueError(f"eval_every_n_rounds must be ≥ 1, got {self.eval_every_n_rounds}")
        if self.num_eval_samples < 1:
            raise ValueError(f"num_eval_samples must be ≥ 1, got {self.num_eval_samples}")
        if self.plot_format not in ("png", "pdf", "svg"):
            raise ValueError(f"plot_format must be png/pdf/svg, got {self.plot_format!r}")


@dataclass(frozen=True)
class AdaptiveDPConfig:
    """
    Configuration for adaptive per-client differential privacy.

    When ``enabled=True`` (and ``DPConfig.enabled=True``), each hospital
    receives a dynamically calibrated noise level every round instead of the
    fixed global sigma from ``DPConfig.sigma``.  The mechanism adapts two
    independent axes:

    1. **Clipping norm** — per-client EMA of observed gradient norms replaces
       the fixed ``DPConfig.max_grad_norm``.  Hospitals with tightly-clustered
       updates get tighter clipping and therefore less noise.

    2. **Epsilon allocation** — the global budget is redistributed each round
       in proportion to inverse training loss (softmax).  Hospitals that have
       already converged receive a larger ε slice (less noise, more useful
       updates); hospitals still learning rapidly receive a smaller slice.

    Attributes
    ----------
    enabled:
        Activate adaptive DP.  Requires ``DPConfig.enabled=True``.
    ema_alpha:
        EMA smoothing factor α ∈ (0, 1] for sensitivity estimation.
        Smaller values prioritise historical norms; larger values react
        quickly to current gradients.  Default 0.1 is a stable starting point.
    min_epsilon_fraction:
        Floor on each client's ε weight before softmax normalisation.
        A value of 0.1 ensures every client gets ≥ 10 % of the mean
        per-round budget, preventing starvation of high-loss hospitals.
    """
    enabled: bool = False
    ema_alpha: float = 0.1
    min_epsilon_fraction: float = 0.1

    def __post_init__(self):
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {self.ema_alpha}")
        if not 0.0 < self.min_epsilon_fraction <= 1.0:
            raise ValueError(
                f"min_epsilon_fraction must be in (0, 1], got {self.min_epsilon_fraction}"
            )


@dataclass(frozen=True)
class TrackerConfig:
    """
    Experiment tracking configuration.

    Set ``backend`` to select the tracking system:

    * ``"none"``    — no-op (default, no dependencies required)
    * ``"mlflow"``  — local or remote MLflow server (``pip install mlflow``)
    * ``"wandb"``   — Weights & Biases cloud/local (``pip install wandb``)

    MLflow runs a local SQLite-backed server by default (``mlflow_uri``
    is the directory for the MLflow store).  Point it at a remote URI
    (``http://...``) to send metrics to a shared server.

    W&B requires a free account and ``WANDB_API_KEY`` env var, or call
    ``wandb login`` once before training.
    """
    backend: str = "none"               # "none" | "mlflow" | "wandb"
    project: str = "medtrace-fl"        # W&B project / MLflow experiment name
    run_name: Optional[str] = None      # auto-generated if None
    mlflow_uri: str = "mlflow_runs"     # local dir or http://host:port
    wandb_entity: Optional[str] = None  # W&B team / user (None = default)
    tags: List[str] = field(default_factory=list)
    log_model_artifact: bool = False    # upload final model as artifact (slow)

    def __post_init__(self):
        if self.backend not in ("none", "mlflow", "wandb"):
            raise ValueError(
                f"TrackerConfig.backend must be 'none', 'mlflow', or 'wandb', "
                f"got {self.backend!r}"
            )


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
    adaptive_dp: AdaptiveDPConfig = field(default_factory=AdaptiveDPConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

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
            "adaptive_dp": {"enabled": self.adaptive_dp.enabled,
                            "ema_alpha": self.adaptive_dp.ema_alpha,
                            "min_epsilon_fraction": self.adaptive_dp.min_epsilon_fraction},
            "training": {"lr": self.training.learning_rate,
                         "batch_size": self.training.batch_size,
                         "max_length": self.training.max_length},
            "tracker": {"backend": self.tracker.backend,
                        "project": self.tracker.project},
            "eval": {"enabled": self.eval.enabled,
                     "num_eval_samples": self.eval.num_eval_samples,
                     "eval_seed": self.eval.eval_seed,
                     "eval_every_n_rounds": self.eval.eval_every_n_rounds},
        }


# ─── Default instance for backward compatibility ─────────────────
# Import this in modules: `from fl_config import config`
# Never mutate it. Create a new FLConfig() if you need different values.
config = FLConfig()
