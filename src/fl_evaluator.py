"""
MedTrace Federated Learning — Evaluation System
================================================
Provides reproducible evaluation of the global federated model at each round.

Algorithm
---------
MCQ accuracy uses *log-probability scoring* (no text generation required):

  1. Format: ``<system> … <user> {question} {options} Answer with a single letter. <assistant>``
  2. Compute logit for tokens "A", "B", "C", "D" at the final position.
  3. Predicted answer = argmax over those four logits.
  4. Accuracy = fraction of questions where prediction == ground-truth letter.

This is faster and more reliable than free-form generation for multiple-choice
benchmarks, and is the approach used in evaluation harnesses like lm-evaluation-harness.

Reproducibility
---------------
Call ``set_all_seeds(seed)`` before any training or data splitting.  The eval
split is fixed at construction time using ``cfg.eval.eval_seed``, so the same
200 questions are always used regardless of training randomness.

Output Format
-------------
Results are stored as ``EvalResult`` dataclasses and aggregated in
``EvalAccumulator``.  The accumulator can be serialised to JSON (for git) and
CSV (for pandas), and loaded back for offline plotting.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

# torch is imported lazily inside functions that need it so that
# EvalResult, EvalAccumulator, and set_all_seeds remain importable
# in environments without the full ML stack (e.g. CI, plotting-only mode).
if TYPE_CHECKING:
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from fl_config import FLConfig
    from fl_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


# ─── Seed Control ─────────────────────────────────────────────────────────────

def set_all_seeds(seed: int) -> None:
    """
    Set every source of randomness for a fully reproducible run.

    Call this once at the very start of ``run_simulation`` / ``run_eval``
    before any dataset shuffles, model init, or training steps.

    torch is imported lazily so this function works even when torch is not
    installed (in that case only Python + numpy seeds are set).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN ops (slightly slower, but reproducible)
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    logger.debug("All seeds set to %d", seed)


# ─── Result Dataclass ─────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """
    Snapshot of model quality at one FL round.

    ``run_label`` distinguishes different experimental conditions
    (e.g. "FedAvg+DP" vs "baseline_no_dp") when overlaid on the same plot.
    """
    run_label: str           # e.g. "FedAvg+DP (ε=8)" or "No-DP Baseline"
    round_num: int           # 0-based FL round index
    accuracy: float          # MCQ accuracy on eval split (0–1)
    loss: float              # average cross-entropy loss
    perplexity: float        # exp(loss)
    num_eval_samples: int    # number of questions evaluated
    elapsed_seconds: float   # wall-clock time for this eval pass
    privacy_budget_spent: float = 0.0   # cumulative epsilon consumed
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalResult":
        # extra may be missing in older saves
        d.setdefault("extra", {})
        return cls(**d)


# ─── Core Evaluator ───────────────────────────────────────────────────────────

class RoundEvaluator:
    """
    Evaluates the global federated model after each aggregation round.

    Usage::

        evaluator = RoundEvaluator(eval_dataset, tokenizer, cfg, device, tracker)
        # inside training loop:
        result = evaluator.evaluate_round(global_weights, round_num,
                                          run_label="FedAvg+DP",
                                          privacy_budget_spent=budget)
    """

    def __init__(
        self,
        eval_dataset: "Dataset",
        tokenizer: "AutoTokenizer",
        cfg: "FLConfig",
        device: str = "cpu",
        tracker: Optional["ExperimentTracker"] = None,
    ):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.tracker = tracker

        # Fix the eval split once at construction — never changes across rounds.
        eval_cfg = cfg.eval
        n = min(eval_cfg.num_eval_samples, len(eval_dataset))
        rng = random.Random(eval_cfg.eval_seed)
        idx = list(range(len(eval_dataset)))
        rng.shuffle(idx)
        self.eval_data = eval_dataset.select(idx[:n])
        logger.info(
            "RoundEvaluator: %d eval samples fixed (seed=%d)",
            n, eval_cfg.eval_seed,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate_round(
        self,
        global_weights,                 # WeightDict | None
        round_num: int,
        run_label: str,
        privacy_budget_spent: float = 0.0,
    ) -> EvalResult:
        """
        Load the current global model, run MCQ accuracy + loss, clean up.

        Safe to call at every round: GPU memory is always freed on exit.
        """
        t0 = time.time()
        logger.info("[EVAL] Round %d — %s …", round_num + 1, run_label)

        import torch as _torch
        model = self._load_model(global_weights)
        try:
            accuracy = self._compute_accuracy(model)
            avg_loss, perplexity = self._compute_loss(model)
        finally:
            del model
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

        elapsed = time.time() - t0
        result = EvalResult(
            run_label=run_label,
            round_num=round_num,
            accuracy=accuracy,
            loss=avg_loss,
            perplexity=perplexity,
            num_eval_samples=len(self.eval_data),
            elapsed_seconds=round(elapsed, 2),
            privacy_budget_spent=privacy_budget_spent,
        )

        logger.info(
            "[EVAL] Round %d | Acc: %.3f | Loss: %.4f | PPL: %.2f | ε: %.3f | %.1fs",
            round_num + 1, accuracy, avg_loss, perplexity,
            privacy_budget_spent, elapsed,
        )

        # Push to experiment tracker
        if self.tracker is not None:
            safe_label = run_label.replace(" ", "_").replace("(", "").replace(")", "")
            self.tracker.log(
                {
                    f"eval/{safe_label}/accuracy": accuracy,
                    f"eval/{safe_label}/loss": avg_loss,
                    f"eval/{safe_label}/perplexity": perplexity,
                },
                step=round_num,
            )

        return result

    # ── Private: model loading ────────────────────────────────────────────────

    def _load_model(self, weights):
        """Instantiate base + LoRA adapter, inject weights, set to eval mode."""
        import torch as _torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model, torch_dtype=_torch.float32,
        )
        lora_cfg = LoraConfig(
            r=self.cfg.lora.r,
            lora_alpha=self.cfg.lora.alpha,
            lora_dropout=0.0,           # no dropout during eval
            target_modules=list(self.cfg.lora.target_modules),
            bias=self.cfg.lora.bias,
            task_type=self.cfg.lora.task_type,
        )
        model = get_peft_model(base, lora_cfg)
        if weights is not None:
            model.load_state_dict(weights, strict=False)
        model.to(self.device)
        model.eval()
        return model

    # ── Private: accuracy ─────────────────────────────────────────────────────

    def _compute_accuracy(self, model) -> float:
        """
        MCQ accuracy via log-probability scoring.

        For each question the model scores choice tokens {A, B, C, D} at the
        final prompt position.  Predicted = argmax; correct = ground-truth letter.
        """
        correct = 0
        total = 0

        for example in self.eval_data:
            question = example.get("question", "")
            options  = example.get("options", {})
            answer   = example.get("answer_idx", example.get("answer", ""))

            if not question or not options or not answer:
                continue

            opts_str = "\n".join(f"  {k}. {v}" for k, v in options.items())
            prompt = (
                f"<|system|>\n{self.cfg.system_msg}</s>\n"
                f"<|user|>\n{question}\n\n{opts_str}\n\n"
                f"Answer with a single letter.</s>\n"
                f"<|assistant|>\n"
            )

            scores = self._score_choices(model, prompt, list(options.keys()))
            if not scores:
                continue

            predicted = max(scores, key=scores.get)
            # Normalise: strip punctuation, take first char, uppercase
            gold = str(answer).strip().rstrip(".").upper()[:1]
            pred = str(predicted).strip().rstrip(".").upper()[:1]
            if pred == gold:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    def _score_choices(
        self, model, prompt: str, choices: List[str]
    ) -> Dict[str, float]:
        """
        Return {choice: logit} for each choice letter at the next token position.
        Uses a single forward pass — no autoregressive decoding.
        """
        import torch as _torch
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.training.max_length,
        ).to(self.device)

        with _torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]  # (vocab_size,)

        scores: Dict[str, float] = {}
        for choice in choices:
            ids = self.tokenizer.encode(choice, add_special_tokens=False)
            if ids:
                scores[choice] = logits[ids[0]].item()
        return scores

    # ── Private: loss ─────────────────────────────────────────────────────────

    def _compute_loss(self, model) -> Tuple[float, float]:
        """
        Average token-level cross-entropy loss and perplexity on the eval set.

        The full prompt + correct answer is tokenised; the model's own loss
        (teacher-forcing) is collected.
        """
        total_loss = 0.0
        count = 0

        for example in self.eval_data:
            question   = example.get("question", "")
            options    = example.get("options", {})
            answer_key = str(example.get("answer_idx", example.get("answer", ""))).strip()

            if not question or not answer_key:
                continue

            answer_text = options.get(answer_key, answer_key)
            opts_str    = "\n".join(f"  {k}. {v}" for k, v in options.items())
            full_text   = (
                f"<|system|>\n{self.cfg.system_msg}</s>\n"
                f"<|user|>\n{question}\n\n{opts_str}</s>\n"
                f"<|assistant|>\n{answer_key}. {answer_text}"
            )

            import torch as _torch
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.cfg.training.max_length,
            ).to(self.device)
            inputs["labels"] = inputs["input_ids"].clone()

            with _torch.no_grad():
                out = model(**inputs)
                if out.loss is not None:
                    total_loss += out.loss.item()
                    count += 1

        if count == 0:
            return 0.0, 1.0
        avg_loss   = total_loss / count
        perplexity = math.exp(min(avg_loss, 20.0))   # cap to avoid overflow
        return avg_loss, perplexity


# ─── Result Accumulator ───────────────────────────────────────────────────────

class EvalAccumulator:
    """
    Collects ``EvalResult`` objects and per-client / privacy metrics across
    all rounds and experiment conditions.

    Designed to be serialised to JSON after a run and loaded back for
    offline plotting, without re-running training.

    Example::

        acc = EvalAccumulator()
        # during training loop:
        acc.add_eval_result(evaluator.evaluate_round(...))
        acc.add_privacy_snapshot(round_num, budget_spent, budget_remaining)
        acc.add_client_metrics("hospital_A", round_metrics)
        # after training:
        acc.save_json("outputs/evaluation/results.json")
        acc.save_csv("outputs/evaluation/results.csv")
        # to reload and replot:
        acc2 = EvalAccumulator.load_json("outputs/evaluation/results.json")
    """

    def __init__(self):
        # eval_results[label] = [EvalResult, ...]
        self._eval: Dict[str, List[EvalResult]] = {}
        # client_metrics[hospital_id] = [dict, ...] one per round
        self._client: Dict[str, List[dict]] = {}
        # privacy_metrics = [{round, budget_spent, budget_remaining, ...}, ...]
        self._privacy: List[dict] = []
        # metadata written once
        self.metadata: dict = {}

    # ── Add methods ───────────────────────────────────────────────────────────

    def add_eval_result(self, result: EvalResult) -> None:
        self._eval.setdefault(result.run_label, []).append(result)

    def add_client_metrics(self, hospital_id: str, metrics: dict) -> None:
        self._client.setdefault(hospital_id, []).append(metrics)

    def add_privacy_snapshot(
        self,
        round_num: int,
        budget_spent: float,
        budget_remaining: float,
        total_epsilon: float,
    ) -> None:
        self._privacy.append({
            "round": round_num,
            "budget_spent": budget_spent,
            "budget_remaining": budget_remaining,
            "total_epsilon": total_epsilon,
            "budget_pct": (budget_spent / total_epsilon * 100) if total_epsilon > 0 else 0.0,
        })

    def set_metadata(self, **kwargs) -> None:
        self.metadata.update(kwargs)

    # ── Read methods ──────────────────────────────────────────────────────────

    def labels(self) -> List[str]:
        return list(self._eval.keys())

    def get_eval_results(self, label: str) -> List[EvalResult]:
        return list(self._eval.get(label, []))

    def get_all_eval(self) -> Dict[str, List[EvalResult]]:
        return {k: list(v) for k, v in self._eval.items()}

    def get_client_metrics(self) -> Dict[str, List[dict]]:
        return dict(self._client)

    def get_privacy_metrics(self) -> List[dict]:
        return list(self._privacy)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "eval_results": {
                label: [r.to_dict() for r in results]
                for label, results in self._eval.items()
            },
            "client_metrics": self._client,
            "privacy_metrics": self._privacy,
        }

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("Eval results saved: %s", path)

    def save_csv(self, path: str) -> None:
        """Export flattened EvalResults for pandas / Excel analysis."""
        rows: List[dict] = []
        for results in self._eval.values():
            for r in results:
                d = r.to_dict()
                d.pop("extra", None)    # drop nested dict from flat CSV
                rows.append(d)
        if not rows:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("CSV saved: %s", path)

    @classmethod
    def load_json(cls, path: str) -> "EvalAccumulator":
        with open(path) as f:
            data = json.load(f)
        acc = cls()
        acc.metadata = data.get("metadata", {})
        for label, results in data.get("eval_results", {}).items():
            for r in results:
                acc.add_eval_result(EvalResult.from_dict(r))
        acc._client  = data.get("client_metrics", {})
        acc._privacy = data.get("privacy_metrics", [])
        return acc

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """High-level summary dict — best accuracy and final loss per label."""
        out = {}
        for label, results in self._eval.items():
            if not results:
                continue
            best_acc   = max(r.accuracy for r in results)
            final      = results[-1]
            out[label] = {
                "best_accuracy":        round(best_acc, 4),
                "final_accuracy":       round(final.accuracy, 4),
                "final_loss":           round(final.loss, 4),
                "final_perplexity":     round(final.perplexity, 2),
                "final_privacy_budget": round(final.privacy_budget_spent, 4),
                "rounds_evaluated":     len(results),
            }
        return out
