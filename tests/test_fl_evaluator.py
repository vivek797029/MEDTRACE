"""
Unit tests for fl_evaluator.py

Covers the pure-Python components that do not require a GPU or model weights:
  - set_all_seeds (Python + numpy branch; torch branch skipped without torch)
  - EvalResult serialisation round-trip
  - EvalAccumulator add / query / serialise / load / summary

Model-dependent tests (RoundEvaluator) are marked @pytest.mark.requires_torch
and skipped when torch is not installed.
"""

import csv
import json
import os
import tempfile
import pytest

from fl_evaluator import EvalAccumulator, EvalResult, set_all_seeds


# ─── set_all_seeds ────────────────────────────────────────────────────────────

class TestSetAllSeeds:
    def test_runs_without_torch(self):
        """Should complete without raising even if torch is not installed."""
        set_all_seeds(42)   # no assertion — just must not raise

    def test_different_seeds_give_different_random_ints(self):
        import random
        set_all_seeds(0)
        a = random.randint(0, 10**9)
        set_all_seeds(1)
        b = random.randint(0, 10**9)
        assert a != b  # overwhelmingly likely with different seeds

    def test_same_seed_reproducible(self):
        import random
        set_all_seeds(99)
        a = [random.random() for _ in range(5)]
        set_all_seeds(99)
        b = [random.random() for _ in range(5)]
        assert a == b


# ─── EvalResult ───────────────────────────────────────────────────────────────

class TestEvalResult:
    def _make(self, **kwargs):
        defaults = dict(
            run_label="FedAvg+DP",
            round_num=0,
            accuracy=0.65,
            loss=1.23,
            perplexity=3.42,
            num_eval_samples=200,
            elapsed_seconds=12.5,
            privacy_budget_spent=0.89,
        )
        defaults.update(kwargs)
        return EvalResult(**defaults)

    def test_to_dict_contains_all_fields(self):
        r = self._make()
        d = r.to_dict()
        for field in ("run_label", "round_num", "accuracy", "loss",
                      "perplexity", "num_eval_samples", "elapsed_seconds",
                      "privacy_budget_spent", "extra"):
            assert field in d, f"to_dict() missing field: {field!r}"

    def test_from_dict_roundtrip(self):
        r = self._make(accuracy=0.72, round_num=5)
        d = r.to_dict()
        r2 = EvalResult.from_dict(d)
        assert r2.accuracy == r.accuracy
        assert r2.round_num == r.round_num
        assert r2.run_label == r.run_label

    def test_from_dict_handles_missing_extra(self):
        d = self._make().to_dict()
        del d["extra"]
        r = EvalResult.from_dict(d)
        assert r.extra == {}

    def test_extra_field_preserved(self):
        r = self._make(extra={"custom_key": "val"})
        d = r.to_dict()
        assert d["extra"]["custom_key"] == "val"
        r2 = EvalResult.from_dict(d)
        assert r2.extra["custom_key"] == "val"


# ─── EvalAccumulator ─────────────────────────────────────────────────────────

def _make_result(label="RunA", round_num=0, accuracy=0.5, privacy=0.0):
    return EvalResult(
        run_label=label,
        round_num=round_num,
        accuracy=accuracy,
        loss=1.0 - accuracy,
        perplexity=2.0,
        num_eval_samples=100,
        elapsed_seconds=5.0,
        privacy_budget_spent=privacy,
    )


class TestEvalAccumulator:
    def test_starts_empty(self):
        acc = EvalAccumulator()
        assert acc.labels() == []
        assert acc.get_all_eval() == {}

    def test_add_eval_result_creates_label(self):
        acc = EvalAccumulator()
        acc.add_eval_result(_make_result("MyRun"))
        assert "MyRun" in acc.labels()

    def test_add_multiple_results_same_label(self):
        acc = EvalAccumulator()
        for rn in range(5):
            acc.add_eval_result(_make_result("RunA", round_num=rn))
        results = acc.get_eval_results("RunA")
        assert len(results) == 5

    def test_add_results_different_labels(self):
        acc = EvalAccumulator()
        acc.add_eval_result(_make_result("Base"))
        acc.add_eval_result(_make_result("DP"))
        assert set(acc.labels()) == {"Base", "DP"}

    def test_get_eval_results_returns_copy(self):
        acc = EvalAccumulator()
        acc.add_eval_result(_make_result("A"))
        results = acc.get_eval_results("A")
        results.clear()   # mutating returned list must not affect accumulator
        assert len(acc.get_eval_results("A")) == 1

    def test_get_eval_results_unknown_label(self):
        acc = EvalAccumulator()
        assert acc.get_eval_results("nonexistent") == []

    def test_add_client_metrics(self):
        acc = EvalAccumulator()
        acc.add_client_metrics("hospital_00", {"train_loss": 0.5, "round": 0})
        m = acc.get_client_metrics()
        assert "hospital_00" in m
        assert m["hospital_00"][0]["train_loss"] == 0.5

    def test_add_privacy_snapshot(self):
        acc = EvalAccumulator()
        acc.add_privacy_snapshot(0, budget_spent=1.0, budget_remaining=7.0, total_epsilon=8.0)
        privacy = acc.get_privacy_metrics()
        assert len(privacy) == 1
        assert privacy[0]["budget_spent"] == 1.0
        assert privacy[0]["budget_pct"] == pytest.approx(12.5)

    def test_set_metadata(self):
        acc = EvalAccumulator()
        acc.set_metadata(seed=42, run_id="abc")
        assert acc.metadata["seed"] == 42
        assert acc.metadata["run_id"] == "abc"

    def test_summary_best_and_final_accuracy(self):
        acc = EvalAccumulator()
        for rn, acc_val in enumerate([0.5, 0.7, 0.6]):
            acc.add_eval_result(_make_result("Run", round_num=rn, accuracy=acc_val))
        s = acc.summary()
        assert s["Run"]["best_accuracy"] == pytest.approx(0.7, abs=1e-4)
        assert s["Run"]["final_accuracy"] == pytest.approx(0.6, abs=1e-4)

    def test_summary_rounds_evaluated(self):
        acc = EvalAccumulator()
        for rn in range(3):
            acc.add_eval_result(_make_result("R", round_num=rn))
        s = acc.summary()
        assert s["R"]["rounds_evaluated"] == 3

    # ── Serialisation ────────────────────────────────────────────────────────

    def test_save_and_load_json(self):
        acc = EvalAccumulator()
        acc.set_metadata(seed=42)
        for rn in range(3):
            acc.add_eval_result(_make_result("Run", round_num=rn, accuracy=0.5 + rn * 0.1))
        acc.add_client_metrics("h_00", {"train_loss": 0.8, "round": 0})
        acc.add_privacy_snapshot(0, 1.0, 7.0, 8.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.json")
            acc.save_json(path)
            assert os.path.exists(path)

            loaded = EvalAccumulator.load_json(path)

        assert loaded.metadata["seed"] == 42
        assert "Run" in loaded.labels()
        assert len(loaded.get_eval_results("Run")) == 3
        assert loaded.get_eval_results("Run")[2].accuracy == pytest.approx(0.7, abs=1e-4)
        assert len(loaded.get_privacy_metrics()) == 1
        assert "h_00" in loaded.get_client_metrics()

    def test_save_json_creates_parent_dirs(self):
        acc = EvalAccumulator()
        acc.add_eval_result(_make_result())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "deeply", "results.json")
            acc.save_json(path)
            assert os.path.exists(path)

    def test_save_csv(self):
        acc = EvalAccumulator()
        for rn in range(2):
            acc.add_eval_result(_make_result("A", round_num=rn))
        for rn in range(2):
            acc.add_eval_result(_make_result("B", round_num=rn))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            acc.save_csv(path)
            assert os.path.exists(path)

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        assert len(rows) == 4   # 2 rounds × 2 labels
        labels_in_csv = {r["run_label"] for r in rows}
        assert labels_in_csv == {"A", "B"}

    def test_save_csv_empty_is_noop(self):
        """save_csv on an empty accumulator should not create a file."""
        acc = EvalAccumulator()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            acc.save_csv(path)
            assert not os.path.exists(path)

    def test_json_roundtrip_with_extra_fields(self):
        acc = EvalAccumulator()
        r = EvalResult(
            run_label="X", round_num=0, accuracy=0.8, loss=0.5,
            perplexity=1.6, num_eval_samples=50, elapsed_seconds=3.0,
            privacy_budget_spent=0.5, extra={"custom": "data"},
        )
        acc.add_eval_result(r)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.json")
            acc.save_json(path)
            loaded = EvalAccumulator.load_json(path)
        assert loaded.get_eval_results("X")[0].extra["custom"] == "data"
