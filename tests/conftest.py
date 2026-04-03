"""
Pytest configuration and shared fixtures for the MedTrace test suite.

The source package lives in ``src/`` without a package ``__init__.py``,
so this conftest adds that directory to sys.path so every test module can
import fl_* modules directly.
"""

import sys
import os
import pytest

# Add src/ to import path so tests can ``import fl_config``, ``import fl_adaptive_dp``, etc.
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))


# ─── Shared fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def default_fl_config():
    """A default FLConfig with 3 hospitals (fast, no ML stack needed)."""
    from fl_config import FLConfig
    return FLConfig()


@pytest.fixture
def small_fl_config():
    """Minimal FLConfig for tests that simulate training logic."""
    from fl_config import FLConfig
    return FLConfig.quick_demo()


@pytest.fixture
def three_hospital_ids():
    return ["hospital_00", "hospital_01", "hospital_02"]


@pytest.fixture
def basic_adaptive_mechanism(three_hospital_ids):
    """AdaptiveDPMechanism pre-configured for 3 hospitals."""
    from fl_adaptive_dp import AdaptiveDPMechanism
    return AdaptiveDPMechanism(
        hospital_ids=three_hospital_ids,
        global_epsilon=8.0,
        delta=1e-5,
        fl_rounds=20,
        initial_sensitivity=1.0,
        ema_alpha=0.1,
        min_epsilon_fraction=0.1,
    )
