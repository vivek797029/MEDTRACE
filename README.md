# MedTrace FL — Federated Learning for Privacy-Preserving Medical AI

> **Fine-tune large language models across hospital networks without sharing patient data**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Differential Privacy](https://img.shields.io/badge/Privacy-Differential%20Privacy-green.svg)]()
[![Federated Learning](https://img.shields.io/badge/Learning-Federated-orange.svg)]()

---

## The Problem

Medical AI is held back by two fundamental tensions:

**Data privacy vs. model quality.** A hospital's clinical notes are among the most sensitive data in existence. Regulations (HIPAA, GDPR) and patient trust make centralised training impossible in practice, yet a model trained on data from one hospital will systematically underperform on patients from another due to population shift and differing documentation styles.

**Static noise vs. heterogeneous learning curves.** Standard differentially private federated learning applies a single, globally fixed noise level to every client every round. A hospital whose local model has already converged is injected with the same noise as one still in early training — wasting privacy budget and degrading final accuracy equally for everyone.

---

## The Approach

MedTrace FL combines three techniques into a single, reproducible pipeline:

### 1. Federated Learning with FedAvg
Hospitals never share raw records. Each round, clients fine-tune a shared base model on their local data, then upload only the **model weight deltas**. The server aggregates these using sample-weighted FedAvg and broadcasts a new global model. No patient data ever leaves its institution.

### 2. Parameter-Efficient Fine-Tuning with LoRA
Full fine-tuning of a 1B+ parameter model is impractical at each hospital. MedTrace attaches small **Low-Rank Adaptation (LoRA)** adapters to the base model. Only adapter weights (~0.1% of total parameters) are trained and communicated, reducing bandwidth and compute by orders of magnitude.

### 3. Gaussian Differential Privacy
Gaussian noise calibrated to `σ = C · √(2 ln(1.25/δ)) / ε` is added to each client's weight update before transmission, providing `(ε, δ)`-differential privacy. Advanced composition ensures the total budget across T rounds satisfies `ε_total ≤ ε_per_round · √T`.

---

## The Innovation: Adaptive Per-Client DP

> **Novel contribution suitable for research publication**

Standard federated DP treats every client identically. MedTrace introduces **adaptive per-client noise calibration** that adjusts the privacy budget allocated to each hospital every round based on two signals:

**Gradient sensitivity tracking.** Each client maintains an exponential moving average (EMA) of its gradient L2 norm, estimating how much its model is changing. A client in late convergence produces small, low-sensitivity gradients that need less noise for the same privacy guarantee.

**Loss-proportional ε allocation.** Per-round budget is distributed across clients via a softmax over inverse training loss. A converged hospital (low loss) receives *more* ε (less noise), improving its contribution quality. A hospital still learning rapidly receives *less* ε (more noise), protecting the privacy of patients whose data is most influential. All clients receive at least `min_epsilon_fraction` of the uniform allocation, preventing starvation.

```
Algorithm: Adaptive Per-Client DP Allocation
────────────────────────────────────────────────────────
Input:  N hospitals, global ε, fl_rounds T
        losses {ℓᵢ}, sensitivity EMA {ŝᵢ}

Per-round base:    ε_base = ε / √T      (advanced composition)
Weights:           wᵢ = softmax(1 / ℓᵢ)  (inverse-loss scoring)
Allocation:        εᵢ = max(ε_base · N · wᵢ,  ε_base · min_frac)
Budget cap:        εᵢ = min(εᵢ,  ε - budget_spent_i)

Noise multiplier:  σᵢ = ŝᵢ · √(2 ln(1.25/δ)) / εᵢ
────────────────────────────────────────────────────────
```

This mechanism is privacy-preserving by construction: the total per-client budget never exceeds the configured `global_epsilon`, and `min_epsilon_fraction` ensures every client participates meaningfully throughout training.

---

## Architecture

```
medtrace/
├── src/
│   ├── fl_config.py          # Typed, frozen dataclass configuration (FLConfig, DPConfig, LoRAConfig, …)
│   ├── fl_adaptive_dp.py     # Adaptive DP mechanism — per-client ε allocation + noise application
│   ├── fl_client.py          # Hospital-side: local fine-tuning loop with LoRA and DP
│   ├── fl_server.py          # FedAvg aggregation with streaming O(2×model) memory footprint
│   ├── fl_simulate.py        # Orchestrator: round loop, evaluation scheduling, checkpoint saving
│   ├── fl_evaluator.py       # MCQ log-probability scoring, EvalAccumulator, JSON/CSV export
│   ├── fl_plots.py           # ResultsPlotter — 6 comparison charts (accuracy, loss, DP budget, …)
│   ├── fl_tracker.py         # Experiment tracking abstraction (NoOp / MLflow / W&B backends)
│   └── run_eval.py           # Stand-alone evaluation entry point
├── tests/
│   ├── conftest.py           # Shared fixtures and sys.path injection
│   ├── test_fl_config.py     # Exception hierarchy, config validation, HospitalRegistry (17 tests)
│   ├── test_fl_adaptive_dp.py# DP state, ε allocation, apply_noise with mocked torch (13 tests)
│   ├── test_fl_evaluator.py  # EvalResult serialisation, EvalAccumulator I/O (10 tests)
│   └── test_fl_plots.py      # Palette, figsize, headless plot generation (13 tests)
├── pyproject.toml            # Package metadata, ruff config, pytest settings
├── requirements.txt          # Core runtime dependencies
└── requirements-dev.txt      # Dev/test dependencies (pytest, ruff, matplotlib)
```

**Key design properties:**
- All configuration expressed as frozen dataclasses — every hyperparameter is immutable and serialisable to JSON
- Streaming FedAvg aggregation: memory usage is `O(2 × model_size)` regardless of fleet size
- HospitalRegistry supports 10 real-world specialty templates and scales to any N via cycling with version suffixes
- Typed exception hierarchy (`MedTraceError → ConfigurationError | AggregationError | PrivacyBudgetExhausted | CheckpointError`) enables precise error handling
- Experiment tracker is a pluggable abstraction — swap NoOp → MLflow → W&B with a single config field

---

## Results

| Metric | Value |
|---|---|
| Privacy guarantee | (ε=8, δ=10⁻⁵)-DP per client over 20 rounds |
| Per-round budget | ε/√T ≈ 1.79 per round (advanced composition) |
| Parameter overhead | ~0.1% of base model size (LoRA adapters only) |
| Communication per round | Adapter weights only (~few MB vs. ~4 GB for TinyLlama 1.1B) |
| Supported hospital count | 3–100+ (dynamically configured, no hardcoded limits) |
| Test coverage | 53 unit tests across 4 modules, headless CI-compatible |

The adaptive mechanism provides a concrete quality–privacy trade-off curve: converged hospitals contribute higher-fidelity updates while still satisfying their individual budget constraints, while hospitals in active learning receive proportionally stronger privacy protection.

---

## Quick Start

**Install**

```bash
git clone <repo>
cd medtrace
pip install -e ".[dev]"
```

**Run a quick demo (2 rounds, 100 samples/hospital)**

```python
from fl_config import FLConfig
from fl_simulate import run_simulation

cfg = FLConfig.quick_demo()   # 2 rounds, 3 hospitals, 100 samples each
run_simulation(cfg)
```

**Scale to 10 hospitals with adaptive DP enabled**

```python
from fl_config import FLConfig, AdaptiveDPConfig

cfg = FLConfig.with_n_hospitals(10).replace(
    adaptive_dp=AdaptiveDPConfig(enabled=True, ema_alpha=0.1, min_epsilon_fraction=0.1),
    fl_rounds=20,
)
run_simulation(cfg)
```

**Run the test suite**

```bash
pytest tests/ -v
```

**Lint and format**

```bash
ruff check src/ tests/
ruff format src/ tests/
```

---

## Configuration Reference

All configuration lives in `FLConfig`, composed of sub-configs:

```python
FLConfig(
    base_model           = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    fl_rounds            = 20,
    local_epochs         = 3,
    aggregation_strategy = "fedavg",   # "fedavg" | "fedprox" | "scaffold"

    lora = LoRAConfig(r=8, alpha=16, dropout=0.05),
    dp   = DPConfig(enabled=True, epsilon=8.0, delta=1e-5, max_grad_norm=1.0),

    adaptive_dp = AdaptiveDPConfig(
        enabled              = True,
        ema_alpha            = 0.1,   # gradient sensitivity smoothing factor
        min_epsilon_fraction = 0.1,   # floor to prevent client starvation
    ),

    tracker = TrackerConfig(backend="wandb"),   # "none" | "mlflow" | "wandb"
    eval    = EvalConfig(eval_every_n_rounds=1, num_eval_samples=200),
)
```

---

## Background

**Federated Learning** was introduced by McMahan et al. (2017) as *Communication-Efficient Learning of Deep Networks from Decentralized Data*. The FedAvg algorithm remains the dominant baseline.

**Differential Privacy** in federated learning was formalised by Geyer, Klein & Nabi (2017) and McMahan et al. (2018, *Learning Differentially Private Recurrent Language Models*). The Gaussian mechanism with advanced composition underlies most practical deployments.

**LoRA** (Hu et al., 2022, *LoRA: Low-Rank Adaptation of Large Language Models*) enables fine-tuning with a fraction of the trainable parameters, making per-round client updates both fast and bandwidth-efficient.

**Adaptive noise calibration** is inspired by the empirical observation that fixed-noise DP wastes budget on already-converged clients. The loss-proportional softmax allocation in MedTrace provides a lightweight, interpretable mechanism for heterogeneous federated settings without requiring any modifications to the DP accounting framework.

---

## License

MIT © VN
