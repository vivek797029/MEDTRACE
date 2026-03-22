"""
MedTrace Federated Learning — Full Simulation
==============================================
Simulates federated training across 3 hospital nodes with:
  - Non-IID data distribution (each hospital has specialty bias)
  - Differential privacy (Gaussian noise on weight deltas)
  - Secure aggregation (simulated encryption)
  - FedAvg aggregation with weighted contributions
  - Per-round evaluation and metrics logging

Usage:
  python fl_simulate.py                    # Full simulation
  python fl_simulate.py --rounds 3         # Custom rounds
  python fl_simulate.py --quick            # Quick demo (2 rounds, 100 samples)

Architecture Overview:
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │   Round 1    Round 2    Round 3    Round 4    Round 5        │
  │     ↓          ↓          ↓          ↓          ↓           │
  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐         │
  │  │ A,B,C│  │ A,B,C│  │ A,B,C│  │ A,B,C│  │ A,B,C│         │
  │  │train │  │train │  │train │  │train │  │train │         │
  │  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘         │
  │     ↓          ↓          ↓          ↓          ↓           │
  │  FedAvg     FedAvg     FedAvg     FedAvg     FedAvg         │
  │     ↓          ↓          ↓          ↓          ↓           │
  │  Global     Global     Global     Global     Global         │
  │  Model v1   Model v2   Model v3   Model v4   Model v5      │
  │                                                 ↓           │
  │                                          Final Model        │
  │                                          + Report           │
  └──────────────────────────────────────────────────────────────┘
"""

import os
import sys
import json
import time
import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fl_config as cfg
from fl_client import HospitalClient
from fl_server import FederatedServer


def load_medical_data():
    """Load MedQA USMLE dataset."""
    print("📚 Loading MedQA USMLE dataset...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    print(f"   Total examples: {len(ds)}")
    return ds


def build_reasoning_format(dataset):
    """Add reasoning trace format to questions."""

    def format_example(example):
        q = example["question"]
        options = example.get("options", {})
        if isinstance(options, dict):
            opts_str = "\n".join([f"  {k}. {v}" for k, v in options.items()])
        else:
            opts_str = str(options)

        example["question"] = f"Question: {q}\n\nOptions:\n{opts_str}"
        return example

    return dataset.map(format_example)


def run_simulation(args):
    """Execute the full federated learning simulation."""

    print("=" * 70)
    print("  🧬 MedTrace Federated Learning Simulation")
    print("  Privacy-Preserving Medical AI Training")
    print("=" * 70)

    # Override config for quick mode
    if args.quick:
        cfg.FL_ROUNDS = 2
        cfg.EXAMPLES_PER_HOSPITAL = 100
        for h in cfg.HOSPITAL_CONFIGS.values():
            h["num_samples"] = 100
        print("\n⚡ Quick mode: 2 rounds, 100 samples per hospital\n")

    if args.rounds:
        cfg.FL_ROUNDS = args.rounds

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Using Apple Silicon MPS")
    else:
        device = "cpu"
        print("💻 Using CPU")

    # Load data
    dataset = load_medical_data()
    dataset = build_reasoning_format(dataset)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ─── Checkpoint helpers ───────────────────────────────────
    # Works on both Kaggle (/kaggle/working) and Colab (/content)
    if os.path.exists("/kaggle/working"):
        CKPT_DIR = "/kaggle/working/fl_checkpoints"
    else:
        CKPT_DIR = "/content/fl_checkpoints"
    os.makedirs(CKPT_DIR, exist_ok=True)

    def save_ckpt(weights, round_num):
        path = os.path.join(CKPT_DIR, f"round_{round_num}.pt")
        torch.save(weights, path)
        with open(os.path.join(CKPT_DIR, "last_round.txt"), "w") as f:
            f.write(str(round_num))
        print(f"  💾 Checkpoint saved → {path}")

    def load_ckpt():
        marker = os.path.join(CKPT_DIR, "last_round.txt")
        if not os.path.exists(marker):
            return None, -1
        with open(marker) as f:
            last = int(f.read().strip())
        path = os.path.join(CKPT_DIR, f"round_{last}.pt")
        if os.path.exists(path):
            w = torch.load(path, map_location="cpu")
            print(f"  ✅ Resumed from checkpoint: round_{last}.pt")
            return w, last
        return None, -1

    # Initialize server
    server = FederatedServer(device=device)

    # Try to resume from checkpoint
    global_weights, last_completed = load_ckpt()
    start_round = last_completed + 1
    if global_weights is None:
        print("  No checkpoint found — starting fresh")
        global_weights = server.initialize_global_model()
        start_round = 0
    else:
        print(f"  Resuming from round {start_round + 1}/{cfg.FL_ROUNDS}")

    # Initialize hospital clients
    print("\n🏥 Initializing Hospital Nodes...")
    hospitals = {}
    for hospital_id, hospital_config in cfg.HOSPITAL_CONFIGS.items():
        hospitals[hospital_id] = HospitalClient(hospital_id, hospital_config, device=device)

    # Prepare output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.HOSPITAL_MODELS_DIR, exist_ok=True)
    os.makedirs(cfg.METRICS_DIR, exist_ok=True)

    # ─── Federated Training Loop ─────────────────────────────
    total_start = time.time()
    all_round_metrics = []

    for round_num in range(start_round, cfg.FL_ROUNDS):
        round_start = time.time()
        print(f"\n{'='*70}")
        print(f"  🔄 FEDERATED ROUND {round_num + 1}/{cfg.FL_ROUNDS}")
        print(f"{'='*70}")

        # Step 1: Each hospital prepares its local data (non-IID)
        print("\n📊 Distributing data to hospitals (non-IID)...")
        for hospital_id, client in hospitals.items():
            client.prepare_local_data(dataset, round_num)

        # Step 2: Each hospital trains locally
        print("\n🔧 Local training phase...")
        client_updates = {}
        for hospital_id, client in hospitals.items():
            weights, metrics = client.train_local(
                global_weights, tokenizer, round_num, cfg.HOSPITAL_MODELS_DIR
            )
            client_updates[hospital_id] = (weights, metrics)

        # Step 3: Server aggregates (FedAvg)
        global_weights = server.aggregate(client_updates, round_num)

        # Step 4: Save checkpoint after EVERY round
        save_ckpt(global_weights, round_num)
        if (round_num + 1) % 2 == 0 or round_num == cfg.FL_ROUNDS - 1:
            server.save_global_model(tokenizer, round_num)

        round_elapsed = time.time() - round_start
        print(f"\n  ⏱️  Round {round_num + 1} complete: {round_elapsed:.1f}s")

        # Collect metrics
        round_data = {
            "round": round_num + 1,
            "time_seconds": round(round_elapsed, 2),
            "hospital_metrics": {hid: m for hid, (_, m) in client_updates.items()},
        }
        all_round_metrics.append(round_data)

    total_elapsed = time.time() - total_start

    # ─── Final Evaluation ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("  📝 FINAL EVALUATION")
    print(f"{'='*70}")

    eval_questions = [
        "A 55-year-old woman presents with sudden onset of left-sided weakness and slurred speech. CT shows no hemorrhage. What is the next step?",
        "A 30-year-old man presents with high fever, neck stiffness, and photophobia for 2 days. What is the most likely diagnosis?",
        "A 65-year-old diabetic patient presents with crushing chest pain. Troponin is elevated. What is the most appropriate management?",
    ]

    server.save_global_model(tokenizer, cfg.FL_ROUNDS - 1, cfg.GLOBAL_MODEL_DIR)
    eval_results = server.evaluate_global(tokenizer, eval_questions)

    # ─── Generate Report ──────────────────────────────────────
    report = server.generate_report()
    report["total_training_time"] = round(total_elapsed, 2)
    report["eval_results"] = eval_results
    report["all_round_metrics"] = all_round_metrics

    # Save report
    report_path = os.path.join(cfg.METRICS_DIR, "fl_training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ─── Summary ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  🎉 FEDERATED TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"""
  📊 Summary:
     Hospitals:            {cfg.NUM_HOSPITALS}
     Rounds:               {cfg.FL_ROUNDS}
     Total time:           {total_elapsed:.1f}s
     Differential Privacy: {'ON (ε={})'.format(cfg.DP_EPSILON) if cfg.DIFFERENTIAL_PRIVACY else 'OFF'}
     Secure Aggregation:   {'ON' if cfg.SECURE_AGGREGATION else 'OFF'}
     Global model:         {cfg.GLOBAL_MODEL_DIR}
     Report:               {report_path}

  🔒 Privacy Guarantee:
     No patient data was transmitted between hospitals.
     Only LoRA adapter deltas ({sum(p.numel() for p in global_weights.values()):,} params)
     were shared, with differential privacy noise applied.

  🏥 Hospital Contributions:""")

    for name, info in report.get("hospital_contributions", {}).items():
        print(f"     {name}: weight={info['weight']:.3f}, samples={info['samples']}")

    print(f"\n  💡 Next: Upload to HuggingFace or deploy on HF Spaces")
    print(f"{'='*70}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedTrace Federated Learning Simulation")
    parser.add_argument("--rounds", type=int, default=None, help="Number of FL rounds")
    parser.add_argument("--quick", action="store_true", help="Quick demo mode")
    args = parser.parse_args()

    run_simulation(args)
