"""
Unit tests for fl_adaptive_dp.py

Tests cover:
  - ClientDPState mutation and serialisation
  - AdaptiveDPMechanism epsilon allocation (uniform at round 0, loss-proportional thereafter)
  - Budget exhaustion cap
  - apply_noise with mocked torch tensors
  - allocation_log_dict key structure
  - summary() format
"""

import math
import sys
from unittest.mock import MagicMock, patch
import pytest

from fl_adaptive_dp import AdaptiveDPMechanism, ClientDPState


# ─── ClientDPState ────────────────────────────────────────────────────────────

class TestClientDPState:
    def test_default_latest_loss_is_inf(self):
        s = ClientDPState(hospital_id="h1")
        assert s.latest_loss == float("inf")

    def test_record_loss_updates_latest(self):
        s = ClientDPState(hospital_id="h1")
        s.record_loss(2.5)
        assert s.latest_loss == 2.5
        s.record_loss(1.0)
        assert s.latest_loss == 1.0

    def test_update_sensitivity_ema(self):
        s = ClientDPState(hospital_id="h1", sensitivity_ema=1.0)
        s.update_sensitivity(0.5, alpha=0.5)
        assert abs(s.sensitivity_ema - 0.75) < 1e-9   # 0.5*1.0 + 0.5*0.5

    def test_update_sensitivity_alpha_1_replaces(self):
        s = ClientDPState(hospital_id="h1", sensitivity_ema=1.0)
        s.update_sensitivity(0.3, alpha=1.0)
        assert abs(s.sensitivity_ema - 0.3) < 1e-9

    def test_record_round_appends(self):
        s = ClientDPState(hospital_id="h1")
        s.record_round(0.5, 2.0, 1.0)
        s.record_round(0.6, 1.8, 0.9)
        assert s.round_epsilons == [0.5, 0.6]
        assert s.round_sigmas == [2.0, 1.8]
        assert s.round_sensitivities == [1.0, 0.9]

    def test_to_dict_contains_all_fields(self):
        s = ClientDPState(hospital_id="audit_h", sensitivity_ema=0.8, budget_spent=1.5)
        d = s.to_dict()
        assert d["hospital_id"] == "audit_h"
        assert "sensitivity_ema" in d
        assert "budget_spent" in d
        assert "round_epsilons" in d
        assert "round_sigmas" in d
        assert "round_sensitivities" in d
        assert "loss_history" in d

    def test_to_dict_rounds_floats(self):
        s = ClientDPState(hospital_id="h", sensitivity_ema=1.23456789)
        d = s.to_dict()
        # should be rounded to 6 decimal places
        assert len(str(d["sensitivity_ema"]).split(".")[-1]) <= 6


# ─── AdaptiveDPMechanism — epsilon allocation ─────────────────────────────────

class TestEpsilonAllocation:
    def _make_mechanism(self, hospital_ids, eps=8.0, rounds=20):
        return AdaptiveDPMechanism(
            hospital_ids=hospital_ids,
            global_epsilon=eps,
            delta=1e-5,
            fl_rounds=rounds,
            initial_sensitivity=1.0,
            ema_alpha=0.1,
            min_epsilon_fraction=0.1,
        )

    def test_round0_allocation_is_uniform(self):
        """All clients have inf loss initially → uniform allocation."""
        mech = self._make_mechanism(["A", "B", "C"])
        alloc = mech.compute_epsilon_allocation(0)
        # Advanced composition per-round base: ε / √T
        per_round_base = 8.0 / math.sqrt(20)
        for hid, eps in alloc.items():
            assert abs(eps - per_round_base) < 1e-6, (
                f"Expected uniform {per_round_base:.6f} but got {eps:.6f} for {hid}"
            )

    def test_low_loss_client_gets_more_epsilon(self):
        """Converged client (low loss) should receive higher ε (less noise)."""
        mech = self._make_mechanism(["A", "B"])
        mech.record_loss("A", 0.1)   # low loss → more ε
        mech.record_loss("B", 2.0)   # high loss → less ε
        alloc = mech.compute_epsilon_allocation(1)
        assert alloc["A"] > alloc["B"], (
            f"Low-loss client A should have more ε: A={alloc['A']:.4f} B={alloc['B']:.4f}"
        )

    def test_allocations_sum_to_approximately_n_times_per_round_base(self):
        """After softmax normalisation, ∑ weights = 1, ∑ allocs ≈ N × per_round_base."""
        n = 4
        mech = self._make_mechanism(["A", "B", "C", "D"], eps=8.0, rounds=20)
        alloc = mech.compute_epsilon_allocation(0)
        per_round_base = 8.0 / math.sqrt(20)
        expected_total = n * per_round_base
        actual_total = sum(alloc.values())
        assert abs(actual_total - expected_total) < 1e-6

    def test_budget_cap_returns_zero_when_exhausted(self):
        mech = self._make_mechanism(["X"])
        mech.states["X"].budget_spent = 8.0   # fully exhausted
        alloc = mech.compute_epsilon_allocation(5)
        assert alloc["X"] == 0.0

    def test_budget_cap_does_not_exceed_remaining(self):
        mech = self._make_mechanism(["X"])
        mech.states["X"].budget_spent = 7.9   # nearly exhausted
        alloc = mech.compute_epsilon_allocation(0)
        assert alloc["X"] <= 0.1 + 1e-9   # at most remaining budget

    def test_min_epsilon_fraction_prevents_starvation(self):
        """Even the worst-loss client should get ε > 0."""
        mech = self._make_mechanism(["A", "B", "C"])
        mech.record_loss("A", 0.001)    # excellent
        mech.record_loss("B", 0.001)    # excellent
        mech.record_loss("C", 100.0)    # terrible
        alloc = mech.compute_epsilon_allocation(1)
        assert alloc["C"] > 0.0, "min_epsilon_fraction floor should prevent zero allocation"

    def test_record_loss_affects_next_round(self):
        mech = self._make_mechanism(["A", "B"])
        alloc_r0 = mech.compute_epsilon_allocation(0)
        # After recording different losses, allocations should diverge
        mech.record_loss("A", 0.1)
        mech.record_loss("B", 5.0)
        alloc_r1 = mech.compute_epsilon_allocation(1)
        assert alloc_r1["A"] != alloc_r0["A"] or alloc_r1["B"] != alloc_r0["B"]

    def test_get_budget_spent(self):
        mech = self._make_mechanism(["H"])
        mech.states["H"].budget_spent = 3.14
        assert mech.get_budget_spent("H") == 3.14


# ─── AdaptiveDPMechanism — apply_noise (torch mocked) ────────────────────────

class TestApplyNoise:
    """
    Tests for apply_noise().  torch is mocked so these run without a GPU
    or even a PyTorch install.
    """

    def _mock_tensor(self, *shape, value=0.5):
        """Create a mock tensor with minimal torch-like API."""
        t = MagicMock()
        t.float.return_value = t
        t.reshape.return_value = t
        t.shape = shape
        # torch.norm(t).item() → a float
        import fl_adaptive_dp as fadp
        return t

    def test_apply_noise_updates_budget(self):
        mech = AdaptiveDPMechanism(
            hospital_ids=["H"],
            global_epsilon=8.0,
            delta=1e-5,
            fl_rounds=20,
        )
        # Patch torch inside the module
        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.reshape.return_value = mock_tensor
        mock_torch.norm.return_value.item.return_value = 0.5
        mock_torch.randn_like.return_value = mock_tensor
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        from collections import OrderedDict
        weights = OrderedDict({"lora_A": mock_tensor})

        with patch("fl_adaptive_dp.torch", mock_torch):
            mech.apply_noise("H", weights, round_num=0, round_epsilon=0.5)

        assert mech.states["H"].budget_spent == 0.5

    def test_apply_noise_zero_epsilon_skips(self):
        """When round_epsilon=0, weights should be returned unchanged."""
        mech = AdaptiveDPMechanism(
            hospital_ids=["H2"],
            global_epsilon=8.0,
            delta=1e-5,
            fl_rounds=20,
        )
        sentinel = MagicMock()
        from collections import OrderedDict
        weights = OrderedDict({"lora_A": sentinel})

        result = mech.apply_noise("H2", weights, round_num=0, round_epsilon=0.0)
        # Budget should not have changed
        assert mech.states["H2"].budget_spent == 0.0

    def test_apply_noise_records_round(self):
        mech = AdaptiveDPMechanism(
            hospital_ids=["H3"],
            global_epsilon=8.0,
            delta=1e-5,
            fl_rounds=20,
        )
        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.reshape.return_value = mock_tensor
        mock_torch.norm.return_value.item.return_value = 0.8
        mock_torch.randn_like.return_value = mock_tensor
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        from collections import OrderedDict
        weights = OrderedDict({"lora_A": mock_tensor})

        with patch("fl_adaptive_dp.torch", mock_torch):
            mech.apply_noise("H3", weights, round_num=2, round_epsilon=0.4)

        assert len(mech.states["H3"].round_epsilons) == 1
        assert mech.states["H3"].round_epsilons[0] == 0.4


# ─── allocation_log_dict ──────────────────────────────────────────────────────

class TestAllocationLogDict:
    def test_key_structure(self):
        mech = AdaptiveDPMechanism(
            hospital_ids=["H_A", "H_B"],
            global_epsilon=8.0,
            delta=1e-5,
            fl_rounds=10,
        )
        alloc = mech.compute_epsilon_allocation(0)
        log_dict = mech.allocation_log_dict(alloc, round_num=0)

        for hid in ["H_A", "H_B"]:
            assert f"adaptive_dp/{hid}/round_epsilon" in log_dict
            assert f"adaptive_dp/{hid}/sigma" in log_dict
            assert f"adaptive_dp/{hid}/sensitivity_ema" in log_dict
            assert f"adaptive_dp/{hid}/budget_spent" in log_dict

    def test_sigma_is_positive_for_positive_epsilon(self):
        mech = AdaptiveDPMechanism(
            hospital_ids=["H"],
            global_epsilon=8.0,
            delta=1e-5,
            fl_rounds=10,
        )
        alloc = mech.compute_epsilon_allocation(0)
        log_dict = mech.allocation_log_dict(alloc, round_num=0)
        assert log_dict["adaptive_dp/H/sigma"] > 0

    def test_zero_epsilon_gives_inf_sigma(self):
        mech = AdaptiveDPMechanism(
            hospital_ids=["H"],
            global_epsilon=8.0,
            delta=1e-5,
            fl_rounds=10,
        )
        log_dict = mech.allocation_log_dict({"H": 0.0}, round_num=0)
        assert log_dict["adaptive_dp/H/sigma"] == float("inf")


# ─── summary() ───────────────────────────────────────────────────────────────

class TestSummary:
    def test_summary_structure(self):
        mech = AdaptiveDPMechanism(
            hospital_ids=["X", "Y"],
            global_epsilon=4.0,
            delta=1e-5,
            fl_rounds=5,
            ema_alpha=0.2,
        )
        s = mech.summary()
        assert s["mechanism"] == "adaptive_dp"
        assert s["global_epsilon"] == 4.0
        assert s["delta"] == 1e-5
        assert s["fl_rounds"] == 5
        assert s["ema_alpha"] == 0.2
        assert "per_client" in s
        assert "X" in s["per_client"]
        assert "Y" in s["per_client"]

    def test_summary_per_client_has_budget_spent(self):
        mech = AdaptiveDPMechanism(
            hospital_ids=["H"],
            global_epsilon=8.0,
            delta=1e-5,
            fl_rounds=10,
        )
        mech.states["H"].budget_spent = 1.23
        s = mech.summary()
        assert s["per_client"]["H"]["budget_spent"] == 1.23
