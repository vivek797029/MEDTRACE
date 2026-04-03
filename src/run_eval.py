"""
MedTrace Federated Learning — Comparison Evaluation Runner
===========================================================
Runs multiple federated training configurations side-by-side, evaluating
the global model at every round, then generates comparison plots and saves
structured results for offline analysis.

Comparison configurations
--------------------------
By default two systems are compared:

  1. **No-DP Baseline** — FedAvg with ``DPConfig(enabled=False)``.
     No gradient noise, maximum utility, acts as the upper-bound reference.

  2. **FedAvg + DP (ε=8.0)** — the current system with Gaussian-mechanism DP.
     Demonstrates the privacy–utility trade-off.

Additional epsilon values can be added via ``--epsilons`` to sweep the full
privacy–performance Pareto front.

Reproducibility
---------------
All experiments use a single ``--seed`` value (default 42) that controls:
  - dataset splitting (train / eval)
  - model weight initialisation
  - data sampling order per hospital

The output directory contains ``config.json`` with the exact seeds, software
versions, and config objects so every run can be recreated exactly.

Usage
-----
Quick demo (2 rounds, 100 samples, ~5 min on a T4):
    python run_eval.py --quick

Standard run (5 rounds, full data):
    python run_eval.py --rounds 5

Full paper run (20 rounds, multiple ε values):
    python run_eval.py --rounds 20 --epsilons 2.0 4.0 8.0 inf

Replot from saved results (no training):
    python run_eval.py --plot-only outputs/evaluation/run_YYYYMMDD/results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

# Allow running from src/ directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fl_config import DPConfig, EvalConfig, FLConfig, HospitalRegistry, TrackerConfig
from fl_evaluator import EvalAccumulator, RoundEvaluator, set_all_seeds
from fl_plots import ResultsPlotter
from fl_simulate import run_simulation, setup_logging

logger = logging.getLogger(__name__)


# ─── Experiment Configuration Builder ─────────────────────────────────────────

def build_experiment_configs(
    base_cfg: FLConfig,
    epsilons: Optional[List[float]] = None,
) -> Dict[str, FLConfig]:
    """
    Return an ordered dict of {display_label: FLConfig} for the comparison.

    The no-DP baseline always comes first so plots render it as the reference
    line.  Additional epsilon values are appended in ascending order (more
    private → right side of privacy-performance plot).

    Uses ``FLConfig.replace()`` (backed by ``dataclasses.replace``) to carry
    all fields from ``base_cfg`` automatically — new fields added to ``FLConfig``
    are propagated without any changes here.
    """
    if epsilons is None:
        epsilons = [base_cfg.dp.epsilon]

    configs: Dict[str, FLConfig] = {}

    # Baseline: inherit all fields from base_cfg, only override dp
    configs["No-DP Baseline"] = base_cfg.replace(dp=DPConfig(enabled=False))

    # DP variants — one per epsilon value
    for eps in sorted(epsilons):
        if eps == float("inf") or eps <= 0:
            continue
        label = f"FedAvg + DP (ε={eps:.1f})"
        configs[label] = base_cfg.replace(
            dp=DPConfig(
                enabled=True,
                epsilon=eps,
                delta=base_cfg.dp.delta,
                max_grad_norm=base_cfg.dp.max_grad_norm,
            )
        )

    return configs


# ─── Comparison Runner ────────────────────────────────────────────────────────

class ComparisonRunner:
    """
    Runs each experimental configuration through the full FL simulation,
    evaluating the global model at each round, then generates all plots.

    The runner is stateless between calls — results are written to ``output_dir``
    and returned as an ``EvalAccumulator`` for further programmatic use.
    """

    def __init__(
        self,
        output_dir: str,
        seed: int = 42,
        checkpoint_root: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.seed = seed
        self.checkpoint_root = checkpoint_root or os.path.join(output_dir, "checkpoints")
        os.makedirs(output_dir, exist_ok=True)

    def run(
        self,
        configs: Dict[str, FLConfig],
        accumulator: Optional[EvalAccumulator] = None,
    ) -> EvalAccumulator:
        """
        Execute all configurations and return a populated EvalAccumulator.

        If ``accumulator`` is supplied, results are appended to it (allows
        incremental runs without re-running completed configs).
        """
        if accumulator is None:
            accumulator = EvalAccumulator()

        # Write experiment metadata once
        meta = {
            "seed": self.seed,
            "output_dir": self.output_dir,
            "num_configs": len(configs),
            "config_labels": list(configs.keys()),
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            import torch, transformers, peft
            meta["software"] = {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
            }
        except ImportError:
            pass
        accumulator.set_metadata(**meta)

        for label, cfg in configs.items():
            logger.info("=" * 60)
            logger.info("Running: %s", label)
            logger.info("=" * 60)

            set_all_seeds(self.seed)

            ckpt_dir = os.path.join(
                self.checkpoint_root,
                label.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "p"),
            )

            label_results = self._run_single(label, cfg, ckpt_dir, accumulator)
            logger.info(
                "Finished %s | rounds evaluated: %d | best acc: %.3f",
                label,
                len(label_results),
                max((r.accuracy for r in label_results), default=0.0),
            )

        return accumulator

    def _run_single(
        self,
        label: str,
        cfg: FLConfig,
        ckpt_dir: str,
        accumulator: EvalAccumulator,
    ) -> list:
        """
        Run one configuration.  Injects a custom ``evaluator_hook`` into
        ``run_simulation`` via the ``on_round_end`` callback so the model
        is evaluated at every (or every Nth) round without re-loading weights
        separately.
        """
        from datasets import load_dataset
        from transformers import AutoTokenizer
        from fl_evaluator import RoundEvaluator
        from fl_tracker import create_tracker

        # Load eval data (same split for all configs, pinned by seed)
        set_all_seeds(self.seed)
        logger.info("Loading MedQA eval split…")
        full_ds   = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tracker   = create_tracker(cfg)
        evaluator = RoundEvaluator(full_ds, tokenizer, cfg, device="cpu", tracker=tracker)

        eval_results: list = []

        def on_round_end(round_num: int, global_weights, privacy_budget_spent: float):
            """Called by run_simulation at the end of each round."""
            ecfg = cfg.eval
            if not ecfg.enabled:
                return
            if (round_num + 1) % ecfg.eval_every_n_rounds != 0:
                return
            result = evaluator.evaluate_round(
                global_weights=global_weights,
                round_num=round_num,
                run_label=label,
                privacy_budget_spent=privacy_budget_spent,
            )
            eval_results.append(result)
            accumulator.add_eval_result(result)

        # Run full simulation; results are captured via the hook
        report = run_simulation(cfg, checkpoint_dir=ckpt_dir,
                                on_round_end=on_round_end)

        # Collect client + privacy snapshots from the report
        for round_entry in report.get("all_round_metrics", []):
            round_num = round_entry["round"] - 1  # 0-based
            for hid, hm in round_entry.get("hospital_metrics", {}).items():
                accumulator.add_client_metrics(hid, hm)

        if cfg.dp.enabled:
            for rn, result in enumerate(eval_results):
                accumulator.add_privacy_snapshot(
                    round_num=result.round_num,
                    budget_spent=result.privacy_budget_spent,
                    budget_remaining=max(0.0, cfg.dp.epsilon - result.privacy_budget_spent),
                    total_epsilon=cfg.dp.epsilon,
                )

        return eval_results

    def plot_and_save(self, accumulator: EvalAccumulator, fmt: str = "png", dpi: int = 150):
        """Generate all plots and save results to the output directory."""
        plots_dir = os.path.join(self.output_dir, "plots")
        plotter   = ResultsPlotter(output_dir=plots_dir, fmt=fmt, dpi=dpi)
        saved_plots = plotter.save_all(accumulator)

        # JSON + CSV
        json_path = os.path.join(self.output_dir, "results.json")
        csv_path  = os.path.join(self.output_dir, "results.csv")
        accumulator.save_json(json_path)
        accumulator.save_csv(csv_path)

        # Summary
        summary = accumulator.summary()
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("  Plots: %s", plots_dir)
        logger.info("  JSON:  %s", json_path)
        logger.info("  CSV:   %s", csv_path)
        logger.info("  Summary: %s", summary_path)
        logger.info("=" * 60)
        logger.info("Final results:")
        for variant, s in summary.items():
            logger.info(
                "  %-35s Acc: %.1f%%  Loss: %.4f  ε: %.3f",
                variant, s["final_accuracy"] * 100,
                s["final_loss"], s["final_privacy_budget"],
            )

        return {
            "plots": saved_plots,
            "json": json_path,
            "csv": csv_path,
            "summary": summary,
        }


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MedTrace Evaluation: Baseline vs FedAvg+DP comparison",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick demo: 2 rounds, 100 samples per hospital (fast, ~5 min on GPU)",
    )
    parser.add_argument(
        "--rounds", type=int, default=None,
        help="Number of FL rounds (overrides --quick and default of 20)",
    )
    parser.add_argument(
        "--epsilons", type=float, nargs="+", default=None,
        help="ε values to compare, e.g. --epsilons 2.0 4.0 8.0\n"
             "Default: use the value from FLConfig (8.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master random seed for full reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for plots and results\n"
             "(default: outputs/evaluation/run_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--eval-samples", type=int, default=200,
        help="Number of MedQA questions to evaluate on per round (default: 200)",
    )
    parser.add_argument(
        "--eval-every", type=int, default=1,
        help="Evaluate every N rounds (default: 1 = every round)",
    )
    parser.add_argument(
        "--plot-format", choices=["png", "pdf", "svg"], default="png",
        help="Plot output format (default: png)",
    )
    parser.add_argument(
        "--plot-only", type=str, default=None, metavar="RESULTS_JSON",
        help="Skip training — load existing results.json and regenerate plots",
    )
    parser.add_argument(
        "--no-dp-only", action="store_true",
        help="Run only the no-DP baseline (skip DP variants)",
    )
    parser.add_argument(
        "--num-hospitals", type=int, default=None, metavar="N",
        help="Number of federated hospital clients (default: 3).\n"
             "Uses HospitalRegistry to auto-generate N specialty hospitals.\n"
             f"Available specialties: {', '.join(HospitalRegistry.specialties())}",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    # ── Plot-only mode ──────────────────────────────────────────────────────
    if args.plot_only:
        if not os.path.exists(args.plot_only):
            logger.error("File not found: %s", args.plot_only)
            sys.exit(1)
        logger.info("Plot-only mode: loading %s", args.plot_only)
        acc = EvalAccumulator.load_json(args.plot_only)
        out_dir = args.output_dir or os.path.dirname(args.plot_only)
        runner  = ComparisonRunner(output_dir=out_dir, seed=args.seed)
        runner.plot_and_save(acc, fmt=args.plot_format)
        return

    # ── Build base config ───────────────────────────────────────────────────
    run_ts  = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join("outputs", "evaluation", f"run_{run_ts}")

    eval_cfg = EvalConfig(
        num_eval_samples=args.eval_samples,
        eval_seed=args.seed,
        eval_every_n_rounds=args.eval_every,
        output_dir=out_dir,
        plot_format=args.plot_format,
    )

    # Determine hospital fleet size
    n_hospitals = args.num_hospitals  # None → use FLConfig default (3)

    if args.quick:
        base_cfg = FLConfig.quick_demo()
        if n_hospitals is not None:
            base_cfg = base_cfg.replace(
                hospitals=HospitalRegistry.build(n_hospitals, num_samples=100)
            )
        # Override eval config on the quick demo
        base_cfg = base_cfg.replace(
            eval=EvalConfig(
                num_eval_samples=min(args.eval_samples, 50),
                eval_seed=args.seed,
                eval_every_n_rounds=1,
                output_dir=out_dir,
                plot_format=args.plot_format,
            ),
        )
        logger.info(
            "Quick mode: 2 rounds, 100 samples/hospital (%d hospitals), 50 eval questions",
            len(base_cfg.hospitals),
        )
    elif args.rounds:
        overrides = {"fl_rounds": args.rounds, "eval": eval_cfg}
        if n_hospitals is not None:
            overrides["hospitals"] = HospitalRegistry.build(n_hospitals)
        base_cfg = FLConfig.create(**overrides)
        logger.info("Custom: %d rounds, %d hospitals", args.rounds, len(base_cfg.hospitals))
    else:
        overrides = {"eval": eval_cfg}
        if n_hospitals is not None:
            overrides["hospitals"] = HospitalRegistry.build(n_hospitals)
        base_cfg = FLConfig.create(**overrides)
        logger.info(
            "Full run: %d rounds, %d hospitals",
            base_cfg.fl_rounds, len(base_cfg.hospitals),
        )

    # ── Build experiment matrix ─────────────────────────────────────────────
    all_configs = build_experiment_configs(base_cfg, epsilons=args.epsilons)
    if args.no_dp_only:
        all_configs = {"No-DP Baseline": all_configs["No-DP Baseline"]}

    logger.info("Experiment configurations:")
    for label, cfg in all_configs.items():
        dp_str = f"ε={cfg.dp.epsilon:.1f}" if cfg.dp.enabled else "disabled"
        logger.info("  %-35s rounds=%d  DP=%s", label, cfg.fl_rounds, dp_str)

    # ── Save run config ─────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    config_dump = {
        "seed": args.seed,
        "quick": args.quick,
        "configs": {k: v.to_dict() for k, v in all_configs.items()},
        "output_dir": out_dir,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_dump, f, indent=2)
    logger.info("Config saved: %s/config.json", out_dir)

    # ── Run ─────────────────────────────────────────────────────────────────
    set_all_seeds(args.seed)
    runner = ComparisonRunner(output_dir=out_dir, seed=args.seed)
    acc    = runner.run(all_configs)
    runner.plot_and_save(acc, fmt=args.plot_format)


if __name__ == "__main__":
    main()
