"""
MedTrace Federated Learning — Full Simulation
==============================================
Orchestrates federated training across hospital nodes with:
  - Non-IID data distribution per hospital specialty
  - Differential privacy (Gaussian mechanism)
  - Secure aggregation (simulated)
  - FedAvg aggregation with weighted contributions
  - Per-round checkpointing + auto-resume

Usage:
  python fl_simulate.py                    # Full simulation (20 rounds)
  python fl_simulate.py --rounds 3         # Custom round count
  python fl_simulate.py --quick            # Quick demo (2 rounds, 100 samples)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import List, Optional

try:
    import torch
except ImportError:  # pragma: no cover — GPU path only
    torch = None  # type: ignore[assignment]

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover — GPU path only
    load_dataset = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover — GPU path only
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

# Add src to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fl_client import HospitalClient
from fl_config import ConfigurationError, FLConfig
from fl_server import FederatedServer
from fl_tracker import ExperimentTracker, create_tracker

# Adaptive DP — optional; imported here so simulate.py works even if the
# feature is disabled (AdaptiveDPMechanism is only instantiated when needed).
from fl_adaptive_dp import AdaptiveDPMechanism

logger = logging.getLogger(__name__)


# ─── Logging Setup ────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging for the FL simulation."""
    fmt = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # Silence noisy third-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)


# ─── Checkpoint Helpers ───────────────────────────────────────

class CheckpointManager:
    """Manages round-level checkpoint save/load with verification."""

    def __init__(self, checkpoint_dir: str):
        self.dir = checkpoint_dir
        os.makedirs(self.dir, exist_ok=True)

    def save(self, weights, round_num: int) -> None:
        path = os.path.join(self.dir, f"round_{round_num}.pt")
        marker = os.path.join(self.dir, "last_round.txt")

        torch.save(weights, path)

        # Verify the file exists and is non-empty.
        if not os.path.exists(path):
            raise RuntimeError(f"Checkpoint save failed — file not created: {path}")
        size_bytes = os.path.getsize(path)
        if size_bytes == 0:
            raise RuntimeError(f"Checkpoint is zero bytes (write error): {path}")

        with open(marker, "w") as f:
            f.write(str(round_num))
            f.flush()
            os.fsync(f.fileno())

        # Force checkpoint .pt file to disk so it survives a crash/disconnect.
        # Without this, the OS may buffer the write and lose it on sudden exit.
        try:
            fd = os.open(path, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
        except OSError:
            pass  # best-effort — some filesystems don't support fsync on read fd

        size_mb = size_bytes / (1024 * 1024)
        logger.info("Checkpoint saved & verified: round_%d.pt (%.3fMB)", round_num, size_mb)

    def load(self):
        marker = os.path.join(self.dir, "last_round.txt")
        if not os.path.exists(marker):
            return None, -1

        with open(marker) as f:
            last = int(f.read().strip())

        path = os.path.join(self.dir, f"round_{last}.pt")
        if not os.path.exists(path):
            logger.warning("Marker says round %d but file missing", last)
            return None, -1

        weights = torch.load(path, map_location="cpu", weights_only=False)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info("Resumed from round_%d.pt (%.1fMB)", last, size_mb)
        return weights, last


# ─── Device Detection ─────────────────────────────────────────

def detect_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info("GPU detected: %s", name)
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple Silicon MPS detected")
        return "mps"
    logger.info("Using CPU")
    return "cpu"


# ─── Data Loading ─────────────────────────────────────────────

def load_medical_data():
    """Load MedQA USMLE dataset."""
    logger.info("Loading MedQA USMLE dataset...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    logger.info("Dataset loaded: %d examples", len(ds))
    return ds


def build_reasoning_format(dataset):
    """
    Append answer options to each question for multi-choice reasoning format.

    Note: HuggingFace Arrow-backed datasets are immutable — in-place key assignment
    on `example` is silently ignored. The map function must return a dict with the
    updated keys so the library can write a new column.
    """
    def format_example(example):
        q = example["question"]
        options = example.get("options", {})
        if isinstance(options, dict):
            opts_str = "\n".join([f"  {k}. {v}" for k, v in options.items()])
        else:
            opts_str = str(options)
        # Must return a dict with updated keys — HuggingFace Arrow datasets are
        # immutable; any direct key assignment on `example` is silently ignored.
        return {"question": f"Question: {q}\n\nOptions:\n{opts_str}"}

    return dataset.map(format_example)


# ─── Main Simulation ──────────────────────────────────────────

def run_simulation(
    cfg: FLConfig,
    checkpoint_dir: Optional[str] = None,
    tracker: Optional[ExperimentTracker] = None,
    on_round_end: Optional[callable] = None,
) -> dict:
    """
    Execute the full federated learning simulation.

    Args:
        cfg: immutable FL configuration.
        checkpoint_dir: directory for round-level checkpoints.  Auto-detected
            for Colab/Kaggle if None.
        tracker: experiment tracker instance.  If None, one is created from
            ``cfg.tracker.backend``.  Pass a pre-built tracker to share a run
            across multiple simulations, or pass ``NoOpTracker()`` to silence
            all tracking regardless of config.
        on_round_end: optional callback invoked after every round, signature::

                def on_round_end(round_num: int,
                                 global_weights: WeightDict,
                                 privacy_budget_spent: float) -> None

            Used by the evaluation runner to trigger per-round model evaluation
            without modifying this function's core logic.
    """
    # Build tracker from config if caller didn't supply one
    if tracker is None:
        tracker = create_tracker(cfg)

    logger.info("=" * 60)
    logger.info("MedTrace Federated Learning Simulation")
    logger.info("Rounds: %d | Hospitals: %d | DP: %s (eps=%.1f)",
                cfg.fl_rounds, cfg.num_hospitals,
                "ON" if cfg.dp.enabled else "OFF", cfg.dp.epsilon)
    logger.info("=" * 60)

    # Fail fast with a clear message instead of an obscure error later
    if not cfg.hospitals:
        raise ConfigurationError(
            "FLConfig.hospitals is empty — provide at least one hospital. "
            "Use HospitalRegistry.build(n) to create n hospital configs."
        )

    device = detect_device()

    # Data
    dataset = load_medical_data()
    dataset = build_reasoning_format(dataset)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Checkpointing
    if checkpoint_dir is None:
        if os.path.exists("/kaggle/working"):
            checkpoint_dir = "/kaggle/working/fl_checkpoints"
        elif os.path.exists("/content/drive/MyDrive"):
            checkpoint_dir = "/content/drive/MyDrive/MedTrace/fl_checkpoints"
        elif os.path.exists("/content"):
            checkpoint_dir = "/content/fl_checkpoints"
        else:
            checkpoint_dir = os.path.join(cfg.output_dir, "checkpoints")

    ckpt = CheckpointManager(checkpoint_dir)

    # Start experiment run (logs all hyperparameters as config)
    tracker.start_run(cfg)

    # Server + resume
    server = FederatedServer(device=device, cfg=cfg, tracker=tracker)
    global_weights, last_completed = ckpt.load()
    start_round = last_completed + 1

    if global_weights is None:
        logger.info("No checkpoint — starting fresh")
        global_weights = server.initialize_global_model()
        start_round = 0
    else:
        server.global_weights = global_weights
        logger.info("Resuming from round %d/%d", start_round + 1, cfg.fl_rounds)

    # Initialize hospitals
    hospitals = {
        hid: HospitalClient(hid, hcfg, device=device, cfg=cfg)
        for hid, hcfg in cfg.hospitals.items()
    }

    # Output dirs
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.hospital_models_dir, exist_ok=True)
    os.makedirs(cfg.metrics_dir, exist_ok=True)

    # ─── Adaptive DP Setup ────────────────────────────────────
    # Instantiate once before the training loop so the per-client EMA state
    # and budget accounting persist across all rounds.
    adaptive_dp: Optional[AdaptiveDPMechanism] = None
    if cfg.dp.enabled and cfg.adaptive_dp.enabled:
        adaptive_dp = AdaptiveDPMechanism(
            hospital_ids=list(hospitals.keys()),
            global_epsilon=cfg.dp.epsilon,
            delta=cfg.dp.delta,
            fl_rounds=cfg.fl_rounds,
            initial_sensitivity=cfg.dp.max_grad_norm,
            ema_alpha=cfg.adaptive_dp.ema_alpha,
            min_epsilon_fraction=cfg.adaptive_dp.min_epsilon_fraction,
        )
        logger.info(
            "Adaptive DP enabled — per-client noise calibration active "
            "(ε=%.1f, α=%.2f, floor=%.2f)",
            cfg.dp.epsilon, cfg.adaptive_dp.ema_alpha, cfg.adaptive_dp.min_epsilon_fraction,
        )

    # ─── Training Loop ────────────────────────────────────────
    total_start = time.time()
    all_round_metrics = []

    for round_num in range(start_round, cfg.fl_rounds):
        round_start = time.time()
        logger.info("=" * 60)
        logger.info("ROUND %d/%d", round_num + 1, cfg.fl_rounds)

        # Step 1: Distribute data (non-IID)
        for client in hospitals.values():
            client.prepare_local_data(dataset, round_num)

        # Step 2: Local training (shared base model loaded ONCE per round)
        logger.info("Loading base model once for round (shared across hospitals)...")
        shared_base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, torch_dtype=torch.float32,
        )
        shared_base.eval()

        # Compute per-client ε allocations BEFORE training so each client
        # knows its noise level.  On round 0 all clients start with the same
        # allocation (no loss history yet); from round 1 onwards the mechanism
        # uses each hospital's previous loss to redistribute the budget.
        round_allocations: dict = {}
        if adaptive_dp is not None:
            round_allocations = adaptive_dp.compute_epsilon_allocation(round_num)
            # Log adaptive DP allocation metrics to tracker
            tracker.log(
                adaptive_dp.allocation_log_dict(round_allocations, round_num),
                step=round_num,
            )
            logger.info(
                "Round %d adaptive ε allocation: %s",
                round_num + 1,
                {hid: f"{eps:.4f}" for hid, eps in round_allocations.items()},
            )

        client_updates = {}
        failed_clients: List[str] = []
        for hid, client in hospitals.items():
            try:
                w, m = client.train_local(
                    global_weights, tokenizer, round_num,
                    cfg.hospital_models_dir, base_model=shared_base,
                    tracker=tracker,
                    adaptive_dp_mechanism=adaptive_dp,
                    round_epsilon=round_allocations.get(hid),
                )
                client_updates[hid] = (w, m)
            except Exception as _client_exc:
                logger.warning(
                    "Client %s failed round %d (skipped from aggregation): %s",
                    hid, round_num + 1, _client_exc,
                )
                failed_clients.append(hid)

        if not client_updates:
            logger.error(
                "ALL %d clients failed round %d — skipping aggregation.",
                len(hospitals), round_num + 1,
            )
            del shared_base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if failed_clients:
            logger.warning(
                "Round %d: %d/%d clients failed; aggregating over %d successful clients.",
                round_num + 1, len(failed_clients), len(hospitals), len(client_updates),
            )

        # Record each client's training loss in the adaptive DP mechanism so
        # it can weight the ε allocation for the next round.
        if adaptive_dp is not None:
            for hid, (_, metrics_dict) in client_updates.items():
                adaptive_dp.record_loss(hid, metrics_dict.get("train_loss", float("inf")))

        del shared_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 3: Aggregate
        global_weights = server.aggregate(client_updates, round_num)

        # Step 4: Checkpoint
        ckpt.save(global_weights, round_num)

        # Step 5: Per-round evaluation callback (used by run_eval.py)
        if on_round_end is not None:
            max_budget_spent = max(
                h.privacy_budget_spent for h in hospitals.values()
            )
            try:
                on_round_end(round_num, global_weights, max_budget_spent)
            except Exception as _cb_exc:
                logger.warning("on_round_end callback raised (non-fatal): %s", _cb_exc)

        # Save LoRA adapter after EVERY round so it can be loaded for
        # inference even if training is interrupted.  Previous behaviour
        # (every 5th round) left a gap where the checkpoint .pt existed
        # but the model files the eval cell needs did not.
        server.save_global_model(tokenizer, round_num, cfg.global_model_dir)

        elapsed = time.time() - round_start
        rounds_done = round_num - start_round + 1
        # Use rolling average over all completed rounds for a stable ETA estimate.
        # Using only the current round's time produces wild swings early in training.
        avg = (time.time() - total_start) / rounds_done
        remaining = (cfg.fl_rounds - round_num - 1) * avg / 60
        logger.info("Round %d complete: %.1fs | ETA: %.0f min", round_num + 1, elapsed, remaining)

        # Privacy budget: use the maximum spent across all hospitals
        # (conservative accounting — each hospital independently noises its update)
        max_budget_spent = max(
            h.privacy_budget_spent for h in hospitals.values()
        )
        budget_remaining = max(0.0, cfg.dp.epsilon - max_budget_spent)
        budget_pct = min(100.0, max_budget_spent / cfg.dp.epsilon * 100) if cfg.dp.epsilon > 0 else 0.0

        tracker.log(
            {
                "round/time_seconds": elapsed,
                "round/eta_minutes": remaining,
                "privacy/budget_total_epsilon": cfg.dp.epsilon,
                "privacy/budget_spent": max_budget_spent,
                "privacy/budget_remaining": budget_remaining,
                "privacy/budget_pct_used": budget_pct,
            },
            step=round_num,
        )

        all_round_metrics.append({
            "round": round_num + 1,
            "time_seconds": round(elapsed, 2),
            "hospital_metrics": {hid: m for hid, (_, m) in client_updates.items()},
        })

    total_elapsed = time.time() - total_start

    # ─── Final Evaluation ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION")

    # Use caller-supplied eval questions from EvalConfig, or fall back to a
    # built-in set that covers the three most common specialties in the default
    # hospital registry.
    _DEFAULT_EVAL_QUESTIONS: List[str] = [
        "A 55-year-old woman presents with sudden onset of left-sided weakness and "
        "slurred speech. CT shows no hemorrhage. What is the next step?",
        "A 30-year-old man presents with high fever, neck stiffness, and photophobia "
        "for 2 days. What is the most likely diagnosis?",
        "A 65-year-old diabetic patient presents with crushing chest pain. Troponin is "
        "elevated. What is the most appropriate management?",
    ]
    eval_questions = cfg.eval.eval_questions or _DEFAULT_EVAL_QUESTIONS

    server.save_global_model(tokenizer, cfg.fl_rounds - 1, cfg.global_model_dir)
    eval_results = server.evaluate_global(tokenizer, eval_questions)

    # ─── Report ───────────────────────────────────────────────
    report = server.generate_report()
    report["total_training_time"] = round(total_elapsed, 2)
    report["eval_results"] = eval_results
    report["all_round_metrics"] = all_round_metrics
    if adaptive_dp is not None:
        report["adaptive_dp_summary"] = adaptive_dp.summary()

    report_path = os.path.join(cfg.metrics_dir, "fl_training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE | Total: %.1fs | Report: %s", total_elapsed, report_path)
    logger.info("Model: %s", cfg.global_model_dir)

    # ── Final summary + tracker close ────────────────────────────────────────
    # Compute final round's avg loss from the last server round_metrics entry
    server_metrics = server.get_metrics()
    final_round_metrics = server_metrics[-1] if server_metrics else {}
    final_avg_loss = final_round_metrics.get("avg_loss", 0.0)
    final_divergence = final_round_metrics.get("weight_divergence", 0.0)

    max_budget_final = max(h.privacy_budget_spent for h in hospitals.values())
    tracker.log_summary(
        {
            "total_training_time_minutes": total_elapsed / 60,
            "rounds_completed": cfg.fl_rounds - start_round,
            "final_weight_divergence": final_divergence,
            "final_privacy_budget_spent": max_budget_final,
            "final_privacy_budget_pct": min(100.0, max_budget_final / cfg.dp.epsilon * 100)
            if cfg.dp.epsilon > 0 else 0.0,
        }
    )

    # Upload the JSON report as a tracked artifact
    tracker.log_artifact(report_path)

    tracker.end_run()

    return report


# ─── CLI Entry Point ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedTrace Federated Learning Simulation")
    parser.add_argument("--rounds", type=int, default=None, help="Number of FL rounds")
    parser.add_argument("--quick", action="store_true", help="Quick demo mode (2 rounds, 100 samples)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    # Build config (immutable — no mutation of globals)
    if args.quick:
        cfg = FLConfig.quick_demo()
        logger.info("Quick mode: 2 rounds, 100 samples per hospital")
    elif args.rounds:
        cfg = FLConfig.create(fl_rounds=args.rounds)
    else:
        cfg = FLConfig()

    run_simulation(cfg, checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
