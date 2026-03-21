"""
MedTrace Federated Learning Configuration
==========================================
Privacy-preserving medical AI training across distributed hospital nodes.
Only LoRA adapter deltas are shared — no patient data ever leaves the hospital.
"""

# ─── Base Model ───────────────────────────────────────────────
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ─── Federated Learning ──────────────────────────────────────
NUM_HOSPITALS = 3                  # Simulated hospital nodes
FL_ROUNDS = 10                     # 10 rounds × ~60 min/round ≈ 10 hours on T4
LOCAL_EPOCHS = 2                   # 2 epochs per hospital per round (deeper learning)
EXAMPLES_PER_HOSPITAL = 1500       # 1500 samples per hospital (4500 total per round)
AGGREGATION_STRATEGY = "fedavg"    # FedAvg (weighted by dataset size)

# ─── LoRA Configuration ──────────────────────────────────────
LORA_R = 8                         # LoRA rank (low = smaller updates to share)
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# ─── Training Hyperparameters ─────────────────────────────────
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
MAX_LENGTH = 512
WARMUP_STEPS = 10
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 2

# ─── Privacy & Security ──────────────────────────────────────
DIFFERENTIAL_PRIVACY = True         # Enable DP noise injection
DP_EPSILON = 8.0                    # Privacy budget (lower = more private)
DP_DELTA = 1e-5                     # Probability of privacy breach
DP_MAX_GRAD_NORM = 1.0              # Gradient clipping for DP
SECURE_AGGREGATION = True           # Simulate encrypted aggregation

# ─── Hospital Specializations (non-IID data simulation) ──────
# Each hospital sees a different distribution of medical cases
# This simulates real-world data heterogeneity
HOSPITAL_CONFIGS = {
    "hospital_A": {
        "name": "Metro General (Cardiology Focus)",
        "location": "New York, USA",
        "specialty_weight": {
            "cardiovascular": 0.5,
            "respiratory": 0.2,
            "general": 0.3
        },
        "num_samples": 1500
    },
    "hospital_B": {
        "name": "Royal London (Neurology Focus)",
        "location": "London, UK",
        "specialty_weight": {
            "neurological": 0.5,
            "endocrine": 0.2,
            "general": 0.3
        },
        "num_samples": 1200
    },
    "hospital_C": {
        "name": "AIIMS Delhi (Infectious Disease Focus)",
        "location": "New Delhi, India",
        "specialty_weight": {
            "infectious": 0.5,
            "gastrointestinal": 0.2,
            "general": 0.3
        },
        "num_samples": 1300
    }
}

# ─── Paths ────────────────────────────────────────────────────
OUTPUT_DIR = "outputs/federated"
GLOBAL_MODEL_DIR = "outputs/federated/global_model"
HOSPITAL_MODELS_DIR = "outputs/federated/hospital_models"
METRICS_DIR = "outputs/federated/metrics"

# ─── System Message ──────────────────────────────────────────
SYSTEM_MSG = (
    "You are MedTrace, a federated clinical reasoning system. "
    "You generate step-by-step, auditable diagnostic reasoning chains "
    "with typed steps (symptom, finding, mechanism, rule, inference, conclusion) "
    "and explicit dependency tracking. Your training preserves patient privacy "
    "through federated learning — no patient data ever leaves the hospital."
)
