"""
Unit tests for fl_config.py

Covers:
  - MedTraceError exception hierarchy
  - HospitalConfig validation
  - HospitalRegistry factory methods
  - DPConfig sigma formula
  - AdaptiveDPConfig validation and immutability
  - FLConfig defaults, factory methods, replace(), to_dict()
  - EvalConfig and TrackerConfig validation
"""

import dataclasses
import math
import pytest

from fl_config import (
    AdaptiveDPConfig,
    AggregationError,
    CheckpointError,
    ConfigurationError,
    DPConfig,
    EvalConfig,
    FLConfig,
    HospitalConfig,
    HospitalRegistry,
    LoRAConfig,
    MedTraceError,
    PrivacyBudgetExhausted,
    TrackerConfig,
    TrainingConfig,
)


# ─── Exception hierarchy ──────────────────────────────────────────────────────

class TestExceptionHierarchy:
    def test_medtrace_error_is_exception(self):
        assert issubclass(MedTraceError, Exception)

    def test_subclasses_inherit_from_base(self):
        for exc_cls in (
            ConfigurationError, AggregationError,
            PrivacyBudgetExhausted, CheckpointError,
        ):
            assert issubclass(exc_cls, MedTraceError), (
                f"{exc_cls.__name__} should inherit from MedTraceError"
            )

    def test_subclasses_are_catchable_as_base(self):
        with pytest.raises(MedTraceError):
            raise ConfigurationError("test")

    def test_exceptions_carry_message(self):
        msg = "something went wrong"
        err = AggregationError(msg)
        assert str(err) == msg


# ─── HospitalConfig ──────────────────────────────────────────────────────────

class TestHospitalConfig:
    def test_valid_config(self):
        h = HospitalConfig(
            name="Test Hospital",
            location="Testville",
            specialty_keywords=["test", "demo"],
        )
        assert h.name == "Test Hospital"
        assert h.num_samples == 1500      # default
        assert h.specialty_ratio == 0.6  # default

    def test_zero_samples_raises(self):
        with pytest.raises(ValueError, match="num_samples must be positive"):
            HospitalConfig(
                name="X", location="X",
                specialty_keywords=["x"], num_samples=0,
            )

    def test_negative_samples_raises(self):
        with pytest.raises(ValueError):
            HospitalConfig(name="X", location="X", specialty_keywords=["x"], num_samples=-1)

    def test_invalid_specialty_ratio_raises(self):
        with pytest.raises(ValueError, match="specialty_ratio"):
            HospitalConfig(
                name="X", location="X",
                specialty_keywords=["x"], specialty_ratio=1.5,
            )

    def test_boundary_specialty_ratio(self):
        # 0.0 and 1.0 are valid boundaries
        h0 = HospitalConfig(name="X", location="X", specialty_keywords=[], specialty_ratio=0.0)
        h1 = HospitalConfig(name="X", location="X", specialty_keywords=[], specialty_ratio=1.0)
        assert h0.specialty_ratio == 0.0
        assert h1.specialty_ratio == 1.0

    def test_is_frozen(self):
        h = HospitalConfig(name="X", location="X", specialty_keywords=[])
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            h.name = "Y"  # type: ignore[misc]


# ─── HospitalRegistry ────────────────────────────────────────────────────────

class TestHospitalRegistry:
    def test_build_3_returns_3(self):
        h = HospitalRegistry.build(3)
        assert len(h) == 3

    def test_build_all_keys_are_unique(self):
        h = HospitalRegistry.build(3)
        assert len(set(h.keys())) == 3

    def test_build_returns_hospital_config_instances(self):
        h = HospitalRegistry.build(5)
        assert all(isinstance(v, HospitalConfig) for v in h.values())

    def test_build_10_uses_all_templates(self):
        h = HospitalRegistry.build(10)
        assert len(h) == 10

    def test_build_12_wraps_catalogue(self):
        """Building beyond 10 cycles with version suffixes."""
        h = HospitalRegistry.build(12)
        assert len(h) == 12
        # All IDs must be unique
        assert len(set(h.keys())) == 12

    def test_build_respects_num_samples_override(self):
        h = HospitalRegistry.build(3, num_samples=50)
        assert all(v.num_samples == 50 for v in h.values())

    def test_build_respects_specialty_ratio_override(self):
        h = HospitalRegistry.build(3, specialty_ratio=0.8)
        assert all(v.specialty_ratio == 0.8 for v in h.values())

    def test_build_zero_raises(self):
        with pytest.raises(ValueError, match="n must be positive"):
            HospitalRegistry.build(0)

    def test_build_negative_raises(self):
        with pytest.raises(ValueError):
            HospitalRegistry.build(-1)

    def test_build_all_returns_10(self):
        h = HospitalRegistry.build_all()
        assert len(h) == 10

    def test_get_known_specialty(self):
        h = HospitalRegistry.get("cardiology")
        assert isinstance(h, HospitalConfig)
        assert "cardiology" in h.name.lower() or "cardiac" in " ".join(h.specialty_keywords)

    def test_get_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="unknown_xyz"):
            HospitalRegistry.get("unknown_xyz")

    def test_specialties_returns_list(self):
        s = HospitalRegistry.specialties()
        assert isinstance(s, list)
        assert len(s) == 10
        assert "cardiology" in s
        assert "neurology" in s

    def test_all_10_specialties_are_documented(self):
        expected = {
            "cardiology", "neurology", "infectious", "oncology", "pediatrics",
            "emergency", "pulmonology", "endocrinology", "gastroenterology", "nephrology",
        }
        assert set(HospitalRegistry.specialties()) == expected


# ─── DPConfig ────────────────────────────────────────────────────────────────

class TestDPConfig:
    def test_sigma_formula(self):
        """σ = C · √(2 ln(1.25/δ)) / ε — standard Gaussian mechanism."""
        dp = DPConfig(enabled=True, epsilon=8.0, delta=1e-5, max_grad_norm=1.0)
        expected = 1.0 * math.sqrt(2 * math.log(1.25 / 1e-5)) / 8.0
        assert abs(dp.sigma - expected) < 1e-12

    def test_sigma_zero_when_disabled(self):
        dp = DPConfig(enabled=False)
        assert dp.sigma == 0.0

    def test_sigma_scales_with_max_grad_norm(self):
        dp1 = DPConfig(max_grad_norm=1.0, epsilon=8.0)
        dp2 = DPConfig(max_grad_norm=2.0, epsilon=8.0)
        assert abs(dp2.sigma - 2 * dp1.sigma) < 1e-12

    def test_sigma_inversely_proportional_to_epsilon(self):
        dp_hi = DPConfig(epsilon=8.0)
        dp_lo = DPConfig(epsilon=4.0)
        assert dp_lo.sigma > dp_hi.sigma

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            DPConfig(enabled=True, epsilon=0.0)

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError, match="delta"):
            DPConfig(enabled=True, delta=1.5)

    def test_invalid_max_grad_norm_raises(self):
        with pytest.raises(ValueError, match="max_grad_norm"):
            DPConfig(enabled=True, max_grad_norm=0.0)

    def test_disabled_skips_validation(self):
        # When disabled, epsilon/delta/max_grad_norm can be invalid
        dp = DPConfig(enabled=False, epsilon=-1.0)
        assert not dp.enabled

    def test_is_frozen(self):
        dp = DPConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            dp.epsilon = 1.0  # type: ignore[misc]


# ─── AdaptiveDPConfig ────────────────────────────────────────────────────────

class TestAdaptiveDPConfig:
    def test_defaults(self):
        cfg = AdaptiveDPConfig()
        assert cfg.enabled is False
        assert cfg.ema_alpha == 0.1
        assert cfg.min_epsilon_fraction == 0.1

    def test_is_frozen(self):
        cfg = AdaptiveDPConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            cfg.enabled = True  # type: ignore[misc]

    def test_invalid_ema_alpha_zero(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            AdaptiveDPConfig(ema_alpha=0.0)

    def test_invalid_ema_alpha_above_one(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            AdaptiveDPConfig(ema_alpha=1.1)

    def test_valid_ema_alpha_boundary(self):
        # 1.0 is a valid upper boundary (single-step EMA = current value)
        cfg = AdaptiveDPConfig(ema_alpha=1.0)
        assert cfg.ema_alpha == 1.0

    def test_invalid_min_epsilon_fraction_zero(self):
        with pytest.raises(ValueError, match="min_epsilon_fraction"):
            AdaptiveDPConfig(min_epsilon_fraction=0.0)

    def test_invalid_min_epsilon_fraction_above_one(self):
        with pytest.raises(ValueError, match="min_epsilon_fraction"):
            AdaptiveDPConfig(min_epsilon_fraction=1.5)


# ─── FLConfig ────────────────────────────────────────────────────────────────

class TestFLConfig:
    def test_default_has_3_hospitals(self):
        cfg = FLConfig()
        assert len(cfg.hospitals) == 3

    def test_num_hospitals_property(self):
        cfg = FLConfig()
        assert cfg.num_hospitals == len(cfg.hospitals)

    def test_default_aggregation_fedavg(self):
        cfg = FLConfig()
        assert cfg.aggregation_strategy == "fedavg"

    def test_invalid_rounds_raises(self):
        with pytest.raises(ValueError, match="fl_rounds"):
            FLConfig(fl_rounds=0)

    def test_invalid_local_epochs_raises(self):
        with pytest.raises(ValueError, match="local_epochs"):
            FLConfig(local_epochs=0)

    def test_invalid_aggregation_raises(self):
        with pytest.raises(ValueError, match="aggregation"):
            FLConfig(aggregation_strategy="unknown_algo")

    def test_create_factory(self):
        cfg = FLConfig.create(fl_rounds=5)
        assert cfg.fl_rounds == 5

    def test_replace_single_field(self):
        base = FLConfig(fl_rounds=10)
        modified = base.replace(fl_rounds=5)
        assert modified.fl_rounds == 5
        assert modified.dp == base.dp            # unchanged
        assert modified.hospitals == base.hospitals

    def test_replace_sub_config(self):
        base = FLConfig()
        no_dp = base.replace(dp=DPConfig(enabled=False))
        assert not no_dp.dp.enabled
        assert no_dp.fl_rounds == base.fl_rounds  # preserved

    def test_replace_preserves_adaptive_dp(self):
        base = FLConfig(adaptive_dp=AdaptiveDPConfig(enabled=True))
        variant = base.replace(fl_rounds=10)
        assert variant.adaptive_dp.enabled

    def test_quick_demo_rounds_and_samples(self):
        cfg = FLConfig.quick_demo()
        assert cfg.fl_rounds == 2
        assert all(h.num_samples == 100 for h in cfg.hospitals.values())

    def test_with_n_hospitals(self):
        cfg = FLConfig.with_n_hospitals(7)
        assert len(cfg.hospitals) == 7

    def test_with_n_hospitals_passes_num_samples(self):
        cfg = FLConfig.with_n_hospitals(5, num_samples=200)
        assert all(h.num_samples == 200 for h in cfg.hospitals.values())

    def test_to_dict_contains_expected_keys(self):
        d = FLConfig().to_dict()
        for key in ("base_model", "fl_rounds", "lora", "dp", "adaptive_dp",
                    "training", "tracker", "eval", "hospitals"):
            assert key in d, f"to_dict() missing key: {key!r}"

    def test_to_dict_hospitals_is_name_map(self):
        d = FLConfig().to_dict()
        assert isinstance(d["hospitals"], dict)
        # Values should be hospital display names (strings), not objects
        assert all(isinstance(v, str) for v in d["hospitals"].values())

    def test_to_dict_adaptive_dp_fields(self):
        d = FLConfig().to_dict()
        adp = d["adaptive_dp"]
        assert "enabled" in adp
        assert "ema_alpha" in adp
        assert "min_epsilon_fraction" in adp

    def test_is_frozen(self):
        cfg = FLConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            cfg.fl_rounds = 99  # type: ignore[misc]


# ─── EvalConfig ──────────────────────────────────────────────────────────────

class TestEvalConfig:
    def test_defaults(self):
        ec = EvalConfig()
        assert ec.enabled is True
        assert ec.eval_every_n_rounds == 1
        assert ec.num_eval_samples == 200
        assert ec.eval_seed == 42
        assert ec.eval_questions is None

    def test_invalid_eval_every_n_rounds(self):
        with pytest.raises(ValueError, match="eval_every_n_rounds"):
            EvalConfig(eval_every_n_rounds=0)

    def test_invalid_num_eval_samples(self):
        with pytest.raises(ValueError, match="num_eval_samples"):
            EvalConfig(num_eval_samples=0)

    def test_invalid_plot_format(self):
        with pytest.raises(ValueError, match="plot_format"):
            EvalConfig(plot_format="bmp")

    def test_valid_plot_formats(self):
        for fmt in ("png", "pdf", "svg"):
            ec = EvalConfig(plot_format=fmt)
            assert ec.plot_format == fmt

    def test_eval_questions_accepts_list(self):
        questions = ["Q1?", "Q2?", "Q3?"]
        ec = EvalConfig(eval_questions=questions)
        assert ec.eval_questions == questions


# ─── TrackerConfig ────────────────────────────────────────────────────────────

class TestTrackerConfig:
    def test_default_backend_is_none(self):
        tc = TrackerConfig()
        assert tc.backend == "none"

    def test_valid_backends(self):
        for backend in ("none", "mlflow", "wandb"):
            tc = TrackerConfig(backend=backend)
            assert tc.backend == backend

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend"):
            TrackerConfig(backend="tensorboard")
