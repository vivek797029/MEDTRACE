"""
MedTrace FL — End-to-End Integration Test
==========================================
Runs the FULL training pipeline (fl_simulate.run_simulation + run_eval
ComparisonRunner) with thin mock stubs for torch / transformers / peft /
datasets.  The mock stubs use numpy arrays so all the real aggregation,
differential-privacy, evaluation, logging and plotting code is exercised.

Run with:
    python tests/integration_e2e.py
"""

from __future__ import annotations

import copy
import json
import math
import os
import pickle
import random
import sys
import tempfile
import traceback
import types
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("e2e")

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# MOCK LAYER  — injected into sys.modules BEFORE any project import
# ──────────────────────────────────────────────────────────────────────────────

# ── Numpy-backed Tensor ───────────────────────────────────────────────────────

class MockTensor:
    """Numpy-backed tensor mock that satisfies all torch ops used in MedTrace."""

    def __init__(self, data):
        if isinstance(data, MockTensor):
            self._data = data._data.copy()
        elif isinstance(data, np.ndarray):
            self._data = data.astype(np.float32)
        else:
            self._data = np.array(data, dtype=np.float32)
        self.data = self          # supports `param.data = ...` pattern

    # ── Tensor arithmetic ────────────────────────────────────────────────────
    def __mul__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data * other._data)
        return MockTensor(self._data * float(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data + other._data)
        return MockTensor(self._data + float(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data - other._data)
        return MockTensor(self._data - float(other))

    def __truediv__(self, other):
        return MockTensor(self._data / float(other))

    def __neg__(self):
        return MockTensor(-self._data)

    # ── Tensor ops ───────────────────────────────────────────────────────────
    def float(self):     return MockTensor(self._data.astype(np.float32))
    def cpu(self):       return self
    def detach(self):    return self
    def clone(self):     return MockTensor(self._data.copy())
    def reshape(self, *args): return MockTensor(self._data.reshape(*args))
    def numel(self):     return int(self._data.size)
    def element_size(self): return 4  # float32
    def item(self):      return float(self._data.item() if self._data.size == 1 else self._data.mean())
    def any(self):       return MockTensor(np.array(bool(np.any(self._data))))
    def numpy(self):     return self._data
    def __bool__(self):  return bool(np.any(self._data))
    def __len__(self):   return len(self._data)
    def __repr__(self):  return f"MockTensor({self._data.shape})"

    # Indexing and slicing — supports tensor[0], tensor[2:], tensor[1, :]
    def __getitem__(self, idx):
        result = self._data[idx]
        if isinstance(result, np.ndarray):
            return MockTensor(result)
        return float(result)

    # Shape property — returns numpy shape tuple
    @property
    def shape(self):
        return self._data.shape

    # Support `param.data = new_value` by re-assigning the underlying array
    def __setattr__(self, name, value):
        if name == "data" and hasattr(self, "_data") and isinstance(value, MockTensor):
            self._data = value._data.copy()
        else:
            object.__setattr__(self, name, value)

    def __deepcopy__(self, memo):
        return MockTensor(self._data.copy())


# ── Mock torch module ─────────────────────────────────────────────────────────

class _MockCuda:
    def is_available(self): return False
    def empty_cache(self):  pass
    def manual_seed_all(self, seed): pass

class _MockBackendsMPS:
    def is_available(self): return False

class _MockBackends:
    mps = _MockBackendsMPS()
    cudnn = types.SimpleNamespace(deterministic=False)

class _MockNoGrad:
    def __enter__(self): return self
    def __exit__(self, *a): pass

_MOCK_FLOAT32 = "float32"  # just a sentinel

def _mock_randn_like(tensor):
    return MockTensor(np.random.randn(*tensor._data.shape).astype(np.float32))

def _mock_norm(tensor):
    flat = tensor._data.flatten()
    return MockTensor(np.array(float(np.linalg.norm(flat)), dtype=np.float32))

def _mock_cat(tensors):
    arrs = [t._data.flatten() for t in tensors]
    return MockTensor(np.concatenate(arrs, axis=0))

def _mock_isnan(tensor):
    return MockTensor(np.array(bool(np.any(np.isnan(tensor._data)))))

def _mock_isinf(tensor):
    return MockTensor(np.array(bool(np.any(np.isinf(tensor._data)))))

def _mock_save(obj, path, **kw):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def _mock_load(path, **kw):
    with open(path, "rb") as f:
        return pickle.load(f)

def _mock_zeros(*shape, **kw):
    return MockTensor(np.zeros(shape, dtype=np.float32))

def _mock_ones(*shape, **kw):
    return MockTensor(np.ones(shape, dtype=np.float32))

def _mock_tensor(data, **kw):
    return MockTensor(np.array(data, dtype=np.float32))

torch_mod = types.ModuleType("torch")
torch_mod.Tensor = MockTensor
torch_mod.float32 = _MOCK_FLOAT32
torch_mod.cuda = _MockCuda()
torch_mod.backends = _MockBackends()
torch_mod.no_grad = _MockNoGrad
torch_mod.randn_like = _mock_randn_like
torch_mod.norm = _mock_norm
torch_mod.cat = _mock_cat
torch_mod.isnan = _mock_isnan
torch_mod.isinf = _mock_isinf
torch_mod.save = _mock_save
torch_mod.load = _mock_load
torch_mod.zeros = _mock_zeros
torch_mod.ones = _mock_ones
torch_mod.tensor = _mock_tensor
torch_mod.__version__ = "2.2.0+mock"

sys.modules["torch"] = torch_mod


# ── Mock Dataset ──────────────────────────────────────────────────────────────

_QUESTIONS = [
    {"question": "Cardiac arrest patient in ER. Best intervention?",
     "options": {"A": "CPR", "B": "Observation", "C": "Discharge", "D": "Vitamins"},
     "answer_idx": 0},
    {"question": "Patient with chest pain and elevated troponin. Diagnosis?",
     "options": {"A": "GERD", "B": "MI", "C": "Anxiety", "D": "Musculoskeletal"},
     "answer_idx": 1},
    {"question": "Child with fever, stiff neck, photophobia. Treatment?",
     "options": {"A": "Ibuprofen", "B": "Antibiotics", "C": "Rest", "D": "Antivirals"},
     "answer_idx": 1},
    {"question": "Diabetic foot ulcer with necrosis. Next step?",
     "options": {"A": "Insulin adjustment", "B": "Debridement", "C": "MRI", "D": "Amputation"},
     "answer_idx": 1},
    {"question": "Hypertensive patient with headache and blurred vision. Urgency?",
     "options": {"A": "Urgent IV therapy", "B": "Oral meds", "C": "Discharge", "D": "Watchful waiting"},
     "answer_idx": 0},
    {"question": "Stroke patient within 3 hours of onset. Best treatment?",
     "options": {"A": "Aspirin", "B": "tPA", "C": "Heparin", "D": "Surgery"},
     "answer_idx": 1},
    {"question": "Pulmonary embolism suspected. Best initial test?",
     "options": {"A": "CXR", "B": "CT pulmonary angiography", "C": "Echo", "D": "ABG"},
     "answer_idx": 1},
    {"question": "Appendicitis with perforation. Management?",
     "options": {"A": "Antibiotics only", "B": "Surgery", "C": "Observation", "D": "Colonoscopy"},
     "answer_idx": 1},
    {"question": "Septic patient with BP 80/50. First intervention?",
     "options": {"A": "Vasopressors", "B": "IV fluids", "C": "Blood cultures", "D": "Antibiotics"},
     "answer_idx": 1},
    {"question": "Asthma attack not responding to bronchodilators. Next?",
     "options": {"A": "IV steroids", "B": "Oxygen alone", "C": "Discharge", "D": "Antihistamines"},
     "answer_idx": 0},
]

class MockDataset:
    def __init__(self, items=None):
        self._items = items if items is not None else (_QUESTIONS * 10)  # 100 items

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return MockDataset(self._items[idx])
        return self._items[idx]

    def select(self, indices):
        return MockDataset([self._items[i % len(self._items)] for i in indices])

    def map(self, fn, batched=False, remove_columns=None, **kw):
        if batched:
            # fn receives dict of lists
            batch = {k: [item[k] for item in self._items] for k in self._items[0]}
            result = fn(batch)
            # Reconstruct per-item dicts
            keys = list(result.keys())
            if remove_columns:
                keys = [k for k in keys if k not in remove_columns]
            n = len(result[keys[0]])
            new_items = []
            for i in range(n):
                new_items.append({k: result[k][i] for k in keys})
            return MockDataset(new_items)
        else:
            new_items = []
            for item in self._items:
                out = fn(item)
                if remove_columns:
                    out = {k: v for k, v in out.items() if k not in remove_columns}
                new_items.append(out)
            return MockDataset(new_items)

    def set_format(self, fmt):
        pass  # noop — already returns list-of-dicts

    @property
    def column_names(self):
        if not self._items:
            return []
        return list(self._items[0].keys())


def _mock_load_dataset(name, split=None, **kw):
    return MockDataset()


datasets_mod = types.ModuleType("datasets")
datasets_mod.load_dataset = _mock_load_dataset
datasets_mod.Dataset = MockDataset
sys.modules["datasets"] = datasets_mod


# ── Mock Model ────────────────────────────────────────────────────────────────
# Has 3 pairs of LoRA adapter parameters (A, B per target module).
# Weight shapes are kept tiny (4×4) so aggregation / DP / serialisation is fast.

_LORA_KEYS = [
    "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
    "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
    "base_model.model.layers.0.self_attn.v_proj.lora_A.weight",
    "base_model.model.layers.0.self_attn.v_proj.lora_B.weight",
    "base_model.model.layers.1.self_attn.q_proj.lora_A.weight",
    "base_model.model.layers.1.self_attn.q_proj.lora_B.weight",
]

class MockParam:
    def __init__(self, shape):
        self._arr = np.random.randn(*shape).astype(np.float32) * 0.01
        self.data = MockTensor(self._arr)

    def detach(self):   return self.data
    def numel(self):    return int(np.prod(self.data._data.shape))


class MockPEFTModel:
    def __init__(self, weights=None):
        if weights is None:
            self._params = {k: MockParam((4, 4)) for k in _LORA_KEYS}
        else:
            # Load from state dict (used when global_weights injected)
            self._params = {k: MockParam((4, 4)) for k in _LORA_KEYS}
        self._device = "cpu"

    def named_parameters(self):
        for k, p in self._params.items():
            yield k, p

    def load_state_dict(self, state_dict, strict=False):
        for k, v in state_dict.items():
            if k in self._params:
                if isinstance(v, MockTensor):
                    self._params[k] = MockParam((4, 4))
                    self._params[k].data = v

    def to(self, device):
        self._device = device
        return self

    def train(self):    return self
    def eval(self):     return self

    def generate(self, input_ids=None, **kw):
        # Return 2D (batch=1, seq_len) tensor mimicking real model.generate().
        # input_ids is a MockTensor with shape (1, seq_len); we append 5 new tokens.
        if input_ids is not None and hasattr(input_ids, "shape"):
            seq_len = input_ids.shape[-1] if len(input_ids.shape) >= 1 else 5
        else:
            seq_len = 5
        total_len = int(seq_len) + 5  # original + 5 generated tokens
        fake = np.ones((1, total_len), dtype=np.float32)
        return MockTensor(fake)

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "adapter_model.safetensors"), "wb") as f:
            f.write(b"mock")

    def __deepcopy__(self, memo):
        new = MockPEFTModel()
        new._params = {k: MockParam((4, 4)) for k in _LORA_KEYS}
        return new

    def parameters(self):
        return (p.data for p in self._params.values())


class MockAutoModel:
    @staticmethod
    def from_pretrained(name, torch_dtype=None, **kw):
        return MockPEFTModel()


class MockBatch(dict):
    """Dict subclass with a .to() method, mimicking HuggingFace tokenizer output."""

    def to(self, device):
        return self  # tensors already on "cpu"


class MockTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    eos_token_id = 2

    @staticmethod
    def from_pretrained(name, **kw):
        return MockTokenizer()

    def __call__(self, text, truncation=False, max_length=None, padding=None,
                 return_tensors=None):
        # Always produce 2D (batch, seq_len) tensors to match HuggingFace behaviour.
        if isinstance(text, list):
            n = len(text)
        else:
            n = 1  # single prompt → batch of 1
        ids  = np.ones((n, 5), dtype=np.float32)
        mask = np.ones((n, 5), dtype=np.float32)

        if return_tensors == "pt":
            return MockBatch({
                "input_ids": MockTensor(ids),
                "attention_mask": MockTensor(mask),
            })
        # Non-tensor return: plain lists (used by tokenize_fn inside train_local)
        ids_list = ids.astype(int).tolist()
        return {
            "input_ids": ids_list,
            "attention_mask": mask.astype(int).tolist(),
            "labels": ids_list,
        }

    def decode(self, ids, skip_special_tokens=True):
        return "The correct answer is B. Based on clinical presentation..."

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)


_transformers_mod = types.ModuleType("transformers")
_transformers_mod.AutoModelForCausalLM = MockAutoModel
_transformers_mod.AutoTokenizer = MockTokenizer
_transformers_mod.__version__ = "4.38.0+mock"


class MockTrainerState:
    log_history = [
        {"loss": 0.85, "step": 1},
        {"loss": 0.75, "step": 2},
        {"train_loss": 0.72, "epoch": 1.0},
    ]


class MockTrainer:
    def __init__(self, model=None, args=None, train_dataset=None, data_collator=None, **kw):
        self.model = model
        self.state = MockTrainerState()

    def train(self):
        # Simulate training by slightly perturbing model params
        if hasattr(self.model, "_params"):
            for p in self.model._params.values():
                p.data._data += np.random.randn(*p.data._data.shape).astype(np.float32) * 0.001


class MockTrainingArguments:
    def __init__(self, output_dir=".", **kw):
        self.output_dir = output_dir
        for k, v in kw.items():
            setattr(self, k, v)


class MockDataCollator:
    def __init__(self, tokenizer, mlm=False, **kw): pass


_transformers_mod.Trainer = MockTrainer
_transformers_mod.TrainingArguments = MockTrainingArguments
_transformers_mod.DataCollatorForLanguageModeling = MockDataCollator
sys.modules["transformers"] = _transformers_mod


# ── Mock PEFT ─────────────────────────────────────────────────────────────────

class MockLoraConfig:
    def __init__(self, r=8, lora_alpha=16, lora_dropout=0.05,
                 target_modules=None, bias="none", task_type="CAUSAL_LM", **kw):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.bias = bias
        self.task_type = task_type


def _mock_get_peft_model(model, lora_cfg):
    """Wrap model (or return an independent MockPEFTModel with LoRA params)."""
    return MockPEFTModel()


_peft_mod = types.ModuleType("peft")
_peft_mod.LoraConfig = MockLoraConfig
_peft_mod.get_peft_model = _mock_get_peft_model
_peft_mod.__version__ = "0.9.0+mock"
sys.modules["peft"] = _peft_mod

# ── Mock accelerate (imported by some transformers internals) ─────────────────
sys.modules["accelerate"] = types.ModuleType("accelerate")


# ──────────────────────────────────────────────────────────────────────────────
# PROJECT IMPORTS  — after mocks are in place
# ──────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fl_config import (
    AdaptiveDPConfig, DPConfig, EvalConfig, FLConfig, HospitalRegistry, TrackerConfig,
)
from fl_evaluator import EvalAccumulator, EvalResult, set_all_seeds
from fl_plots import ResultsPlotter
from fl_simulate import run_simulation, setup_logging
from fl_tracker import create_tracker


# ──────────────────────────────────────────────────────────────────────────────
# TEST HARNESS
# ──────────────────────────────────────────────────────────────────────────────

PASSED = 0
FAILED = 0
RESULTS = []


def check(name, fn):
    global PASSED, FAILED
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"  PASS  [{elapsed:.2f}s]  {name}")
        PASSED += 1
        RESULTS.append({"name": name, "status": "PASS", "elapsed": elapsed})
    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        print(f"  FAIL  [{elapsed:.2f}s]  {name}")
        print(f"         {e}")
        for line in tb.splitlines()[1:]:
            print(f"         {line}")
        FAILED += 1
        RESULTS.append({"name": name, "status": "FAIL", "error": str(e), "elapsed": elapsed})


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1: UNIT CHECKS WITHOUT TORCH (already verified, smoke-test here)
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Phase 1: Config & Evaluator Smoke Tests ──────────────────────────────")

def p1_config_builds():
    cfg = FLConfig.quick_demo()
    assert cfg.fl_rounds == 2
    assert len(cfg.hospitals) == 3

check("FLConfig.quick_demo()", p1_config_builds)


def p1_n_hospitals():
    for n in (3, 5, 10, 12):
        cfg = FLConfig.with_n_hospitals(n)
        assert len(cfg.hospitals) == n, f"expected {n}, got {len(cfg.hospitals)}"

check("FLConfig.with_n_hospitals(3,5,10,12)", p1_n_hospitals)


def p1_eval_accumulator_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        acc = EvalAccumulator()
        acc.set_metadata(seed=42, test="e2e")
        for r in range(3):
            acc.add_eval_result(EvalResult(
                run_label="Test", round_num=r, accuracy=0.5+r*0.05,
                loss=0.8-r*0.05, perplexity=2.0, num_eval_samples=10,
                elapsed_seconds=0.1, privacy_budget_spent=r*0.5,
            ))
        path = os.path.join(tmp, "results.json")
        acc.save_json(path)
        loaded = EvalAccumulator.load_json(path)
        assert loaded.metadata["seed"] == 42
        assert len(loaded.get_eval_results("Test")) == 3

check("EvalAccumulator JSON roundtrip", p1_eval_accumulator_roundtrip)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2: MOCK TENSOR OPS (core aggregation math)
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Phase 2: Mock Tensor Operations ──────────────────────────────────────")

def p2_tensor_arithmetic():
    a = MockTensor(np.array([1.0, 2.0, 3.0]))
    b = MockTensor(np.array([4.0, 5.0, 6.0]))
    c = a + b
    assert np.allclose(c._data, [5., 7., 9.])
    d = 0.5 * a
    assert np.allclose(d._data, [0.5, 1.0, 1.5])

check("MockTensor: arithmetic", p2_tensor_arithmetic)


def p2_norm():
    t = MockTensor(np.array([3.0, 4.0]))
    n = torch_mod.norm(t).item()
    assert abs(n - 5.0) < 1e-5

check("MockTensor: norm (3,4)→5", p2_norm)


def p2_randn_like():
    t = MockTensor(np.zeros((4, 4)))
    noise = torch_mod.randn_like(t)
    assert noise._data.shape == (4, 4)
    # Should not be all zeros
    assert np.any(noise._data != 0.0)

check("MockTensor: randn_like", p2_randn_like)


def p2_cat():
    a = MockTensor(np.array([1.0, 2.0]))
    b = MockTensor(np.array([3.0, 4.0, 5.0]))
    c = torch_mod.cat([a, b])
    assert c._data.shape == (5,)
    assert np.allclose(c._data, [1., 2., 3., 4., 5.])

check("MockTensor: cat", p2_cat)


def p2_save_load():
    from collections import OrderedDict
    with tempfile.TemporaryDirectory() as tmp:
        weights = OrderedDict({
            "lora_A": MockTensor(np.eye(4, dtype=np.float32)),
            "lora_B": MockTensor(np.zeros((4, 4), dtype=np.float32)),
        })
        path = os.path.join(tmp, "weights.pt")
        torch_mod.save(weights, path)
        assert os.path.exists(path)
        loaded = torch_mod.load(path, map_location="cpu", weights_only=False)
        assert set(loaded.keys()) == set(weights.keys())
        assert np.allclose(loaded["lora_A"]._data, np.eye(4))

check("torch.save / torch.load", p2_save_load)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3: FEDERATED SERVER AGGREGATION (real code, mock tensors)
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Phase 3: FedAvg Aggregation ──────────────────────────────────────────")

from fl_server import FederatedServer
from fl_config import AggregationError
from collections import OrderedDict

def make_lora_weights(scale=1.0):
    return OrderedDict({
        k: MockTensor(np.random.randn(4, 4).astype(np.float32) * scale)
        for k in _LORA_KEYS
    })


def p3_aggregate_two_clients():
    cfg = FLConfig.quick_demo()
    server = FederatedServer(cfg=cfg)
    w1 = make_lora_weights(1.0)
    w2 = make_lora_weights(0.5)
    updates = {
        "hospital_00": (w1, {"hospital": "H1", "train_loss": 0.8, "num_samples": 100}),
        "hospital_01": (w2, {"hospital": "H2", "train_loss": 0.6, "num_samples": 200}),
    }
    global_w = server.aggregate(updates, round_num=0)
    assert set(global_w.keys()) == set(_LORA_KEYS)
    # Weighted average: H1 gets 100/300=1/3, H2 gets 200/300=2/3
    for k in _LORA_KEYS:
        expected = (1/3) * w1[k]._data + (2/3) * w2[k]._data
        assert np.allclose(global_w[k]._data, expected, atol=1e-5), f"Mismatch on {k}"

check("FedAvg: weighted average is correct", p3_aggregate_two_clients)


def p3_aggregate_empty_raises():
    cfg = FLConfig.quick_demo()
    server = FederatedServer(cfg=cfg)
    try:
        server.aggregate({}, round_num=0)
        raise AssertionError("should have raised AggregationError")
    except AggregationError:
        pass

check("FedAvg: empty updates raises AggregationError", p3_aggregate_empty_raises)


def p3_aggregate_zero_samples_raises():
    cfg = FLConfig.quick_demo()
    server = FederatedServer(cfg=cfg)
    updates = {
        "h1": (make_lora_weights(), {"hospital": "H1", "train_loss": 0.8, "num_samples": 0}),
    }
    try:
        server.aggregate(updates, round_num=0)
        raise AssertionError("should have raised AggregationError")
    except AggregationError:
        pass

check("FedAvg: zero samples raises AggregationError", p3_aggregate_zero_samples_raises)


def p3_aggregate_key_mismatch_raises():
    cfg = FLConfig.quick_demo()
    server = FederatedServer(cfg=cfg)
    w1 = make_lora_weights()
    w2 = OrderedDict({"different_key": MockTensor(np.zeros((4,4)))})
    updates = {
        "h1": (w1, {"hospital": "H1", "train_loss": 0.5, "num_samples": 100}),
        "h2": (w2, {"hospital": "H2", "train_loss": 0.5, "num_samples": 100}),
    }
    try:
        server.aggregate(updates, round_num=0)
        raise AssertionError("should have raised AggregationError")
    except AggregationError:
        pass

check("FedAvg: key mismatch raises AggregationError", p3_aggregate_key_mismatch_raises)


def p3_generate_report():
    cfg = FLConfig.quick_demo()
    server = FederatedServer(cfg=cfg)
    w = make_lora_weights()
    updates = {"h1": (w, {"hospital": "H1", "train_loss": 0.7, "num_samples": 50})}
    server.aggregate(updates, round_num=0)
    report = server.generate_report()
    assert "round_metrics" in report
    assert "hospital_contributions" in report
    assert "config" in report
    assert len(report["round_metrics"]) == 1
    rm = report["round_metrics"][0]
    assert rm["round"] == 0
    assert rm["num_hospitals"] == 1
    assert rm["total_samples"] == 50
    assert abs(rm["avg_loss"] - 0.7) < 1e-6

check("FedAvg: generate_report() structure correct", p3_generate_report)


def p3_divergence_metric():
    """Weight divergence should be 0 when client == global."""
    cfg = FLConfig.quick_demo()
    server = FederatedServer(cfg=cfg)
    w = make_lora_weights(scale=1.0)
    # Run aggregation once
    updates = {"h1": (w, {"hospital": "H1", "train_loss": 0.5, "num_samples": 100})}
    server.aggregate(updates, round_num=0)
    rm = server.round_metrics[0]
    # Divergence of a single client from its own aggregate is 0
    assert rm["weight_divergence"] < 1e-8, f"Expected ~0, got {rm['weight_divergence']}"

check("FedAvg: single-client divergence is 0", p3_divergence_metric)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 4: DIFFERENTIAL PRIVACY
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Phase 4: Differential Privacy ────────────────────────────────────────")

from fl_adaptive_dp import AdaptiveDPMechanism


def p4_fixed_dp_noise():
    """Standard DP noise in HospitalClient._apply_dp_noise."""
    from fl_client import HospitalClient
    hcfg = list(FLConfig.quick_demo().hospitals.values())[0]
    cfg = FLConfig.quick_demo()
    client = HospitalClient("h1", hcfg, device="cpu", cfg=cfg)
    client.privacy_budget_spent = 0.0
    w = make_lora_weights(scale=1.0)
    orig_vals = {k: v._data.copy() for k, v in w.items()}
    noisy = client._apply_dp_noise(w)
    # Noisy weights should differ from originals
    changed = any(
        not np.allclose(noisy[k]._data, orig_vals[k]) for k in noisy
    )
    assert changed, "DP noise did not change any weights"
    # Budget should have increased
    per_round_eps = cfg.dp.epsilon / math.sqrt(cfg.fl_rounds)
    assert abs(client.privacy_budget_spent - per_round_eps) < 1e-9

check("Fixed DP: noise applied and budget tracked", p4_fixed_dp_noise)


def p4_adaptive_dp_uniform_round0():
    mech = AdaptiveDPMechanism(
        hospital_ids=["A", "B", "C"],
        global_epsilon=8.0, delta=1e-5, fl_rounds=20,
    )
    alloc = mech.compute_epsilon_allocation(0)
    per_round = 8.0 / math.sqrt(20)
    for hid, eps in alloc.items():
        assert abs(eps - per_round) < 1e-6, f"{hid}: {eps} != {per_round}"

check("Adaptive DP: round-0 allocation uniform", p4_adaptive_dp_uniform_round0)


def p4_adaptive_dp_noise_changes_weights():
    mech = AdaptiveDPMechanism(
        hospital_ids=["H"], global_epsilon=8.0, delta=1e-5, fl_rounds=20,
    )
    alloc = mech.compute_epsilon_allocation(0)
    w = make_lora_weights(scale=0.1)
    orig = {k: v._data.copy() for k, v in w.items()}
    noisy = mech.apply_noise("H", w, round_num=0, round_epsilon=alloc["H"])
    changed = any(not np.allclose(noisy[k]._data, orig[k]) for k in noisy)
    assert changed
    assert abs(mech.states["H"].budget_spent - alloc["H"]) < 1e-9

check("Adaptive DP: noise applied, budget updated", p4_adaptive_dp_noise_changes_weights)


def p4_adaptive_dp_budget_cap():
    mech = AdaptiveDPMechanism(
        hospital_ids=["X"], global_epsilon=8.0, delta=1e-5, fl_rounds=20,
    )
    mech.states["X"].budget_spent = 8.0  # fully exhausted
    alloc = mech.compute_epsilon_allocation(5)
    assert alloc["X"] == 0.0

check("Adaptive DP: exhausted budget yields ε=0", p4_adaptive_dp_budget_cap)


def p4_adaptive_dp_low_loss_gets_more_eps():
    mech = AdaptiveDPMechanism(
        hospital_ids=["Good", "Bad"],
        global_epsilon=8.0, delta=1e-5, fl_rounds=20, min_epsilon_fraction=0.05,
    )
    mech.record_loss("Good", 0.01)
    mech.record_loss("Bad", 5.0)
    alloc = mech.compute_epsilon_allocation(1)
    assert alloc["Good"] > alloc["Bad"], (
        f"Low-loss 'Good' ({alloc['Good']:.4f}) should get more ε than 'Bad' ({alloc['Bad']:.4f})"
    )

check("Adaptive DP: low-loss client gets more ε", p4_adaptive_dp_low_loss_gets_more_eps)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 5: EXPERIMENT TRACKER
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Phase 5: Experiment Tracker ──────────────────────────────────────────")


def p5_noop_tracker():
    cfg = FLConfig.quick_demo().replace(
        tracker=TrackerConfig(backend="none")
    )
    tracker = create_tracker(cfg)
    tracker.start_run(cfg)
    tracker.log({"metric/a": 1.0, "metric/b": 0.5}, step=0)
    tracker.log_summary({"total_time": 1.23})
    tracker.end_run()

check("NoOp tracker: full lifecycle runs without error", p5_noop_tracker)


def p5_tracker_log_table_and_text():
    cfg = FLConfig.quick_demo().replace(tracker=TrackerConfig(backend="none"))
    tracker = create_tracker(cfg)
    tracker.start_run(cfg)
    rows = [{"q": "question?", "a": "answer."}]
    tracker.log_table("eval/responses", rows, step=0)
    tracker.log_text("eval/q00", "Q: ...\n\nA: ...", step=0)
    tracker.end_run()

check("NoOp tracker: log_table and log_text work", p5_tracker_log_table_and_text)


def p5_tracker_artifact():
    cfg = FLConfig.quick_demo().replace(tracker=TrackerConfig(backend="none"))
    tracker = create_tracker(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.json")
        with open(path, "w") as f:
            json.dump({"key": "value"}, f)
        tracker.start_run(cfg)
        tracker.log_artifact(path)
        tracker.end_run()

check("NoOp tracker: log_artifact works", p5_tracker_artifact)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 6: CHECKPOINT MANAGER
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Phase 6: Checkpoint Manager ──────────────────────────────────────────")

from fl_simulate import CheckpointManager


def p6_save_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = CheckpointManager(tmp)
        w = make_lora_weights(scale=0.5)
        ckpt.save(w, round_num=0)

        loaded, last = ckpt.load()
        assert last == 0, f"expected last=0, got {last}"
        assert loaded is not None
        assert set(loaded.keys()) == set(w.keys())
        for k in w:
            assert np.allclose(loaded[k]._data, w[k]._data), f"Mismatch on {k}"

check("CheckpointManager: save→load roundtrip", p6_save_and_load)


def p6_resume_multi_round():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = CheckpointManager(tmp)
        for rnd in range(3):
            ckpt.save(make_lora_weights(), round_num=rnd)
        _, last = ckpt.load()
        assert last == 2

check("CheckpointManager: last_round.txt updated correctly", p6_resume_multi_round)


def p6_load_missing():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = CheckpointManager(tmp)
        loaded, last = ckpt.load()
        assert loaded is None
        assert last == -1

check("CheckpointManager: no checkpoint returns (None, -1)", p6_load_missing)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 7: DATA DISTRIBUTION (NON-IID SPLIT)
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Phase 7: Non-IID Data Distribution ───────────────────────────────────")

from fl_client import HospitalClient


def p7_prepare_data_respects_num_samples():
    hcfg = list(FLConfig.quick_demo().hospitals.values())[0]
    cfg = FLConfig.quick_demo()
    client = HospitalClient("hospital_00", hcfg, device="cpu", cfg=cfg)
    dataset = MockDataset()  # 100 items
    local = client.prepare_local_data(dataset, round_num=0)
    assert len(local) == min(hcfg.num_samples, len(dataset))

check("Data split: num_samples respected", p7_prepare_data_respects_num_samples)


def p7_prepare_data_reproducible():
    hcfg = list(FLConfig.quick_demo().hospitals.values())[0]
    cfg = FLConfig.quick_demo()
    dataset = MockDataset()
    c1 = HospitalClient("hospital_00", hcfg, device="cpu", cfg=cfg)
    c2 = HospitalClient("hospital_00", hcfg, device="cpu", cfg=cfg)
    d1 = c1.prepare_local_data(dataset, round_num=5)
    d2 = c2.prepare_local_data(dataset, round_num=5)
    # Same client, same round → same data
    for i in range(len(d1)):
        assert d1[i]["question"] == d2[i]["question"], f"item {i} differs"

check("Data split: reproducible across two instances", p7_prepare_data_reproducible)


def p7_different_clients_get_different_data():
    hospitals = FLConfig.quick_demo().hospitals
    hcfgs = list(hospitals.values())
    hids = list(hospitals.keys())
    cfg = FLConfig.quick_demo()
    dataset = MockDataset(_QUESTIONS * 5)  # 50 items for variety

    c0 = HospitalClient(hids[0], hcfgs[0], cfg=cfg)
    c1 = HospitalClient(hids[1], hcfgs[1], cfg=cfg)

    d0 = [item["question"] for item in c0.prepare_local_data(dataset, 0)]
    d1 = [item["question"] for item in c1.prepare_local_data(dataset, 0)]
    # Different hospitals should not have identical question orderings
    # (they can share some questions due to random sampling, but not all in same order)
    assert d0 != d1, "Different hospitals got identical dataset splits"

check("Data split: different clients get different splits", p7_different_clients_get_different_data)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 8: FULL END-TO-END PIPELINE  (run_simulation with quick_demo config)
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Phase 8: Full run_simulation() Pipeline ──────────────────────────────")


def p8_run_simulation_no_dp():
    """No-DP baseline: 2 rounds, 3 hospitals, tiny data."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = FLConfig.quick_demo().replace(
            dp=DPConfig(enabled=False),
            adaptive_dp=AdaptiveDPConfig(enabled=False),
            output_dir=os.path.join(tmp, "output"),
        )

        report = run_simulation(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))

        # Structure checks
        assert "config" in report, "report missing 'config'"
        assert "round_metrics" in report
        assert "eval_results" in report
        assert "total_training_time" in report
        assert len(report["round_metrics"]) == cfg.fl_rounds

        # Round metric checks
        for rm in report["round_metrics"]:
            assert rm["num_hospitals"] == cfg.num_hospitals
            assert rm["total_samples"] > 0
            assert isinstance(rm["avg_loss"], float)
            assert rm["avg_loss"] >= 0

        # Checkpoint should exist
        last_ckpt = os.path.join(tmp, "ckpt", f"round_{cfg.fl_rounds-1}.pt")
        assert os.path.exists(last_ckpt), f"checkpoint missing: {last_ckpt}"

        # Report JSON should be saved
        report_json = os.path.join(cfg.metrics_dir, "fl_training_report.json")
        assert os.path.exists(report_json), f"report JSON missing: {report_json}"

        # Config in report should match
        assert report["config"]["fl_rounds"] == cfg.fl_rounds

check("run_simulation: No-DP baseline, 2 rounds, 3 hospitals", p8_run_simulation_no_dp)


def p8_run_simulation_with_fixed_dp():
    """Standard Gaussian DP: verify budget is tracked."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = FLConfig.quick_demo().replace(
            dp=DPConfig(enabled=True, epsilon=4.0, delta=1e-5, max_grad_norm=1.0),
            adaptive_dp=AdaptiveDPConfig(enabled=False),
            output_dir=os.path.join(tmp, "output"),
        )

        report = run_simulation(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))

        assert len(report["round_metrics"]) == 2

        # Report should have total_training_time
        assert report["total_training_time"] > 0

check("run_simulation: Fixed DP (ε=4), 2 rounds", p8_run_simulation_with_fixed_dp)


def p8_run_simulation_with_adaptive_dp():
    """Adaptive per-client DP: verify adaptive_dp_summary in report."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = FLConfig.quick_demo().replace(
            dp=DPConfig(enabled=True, epsilon=8.0, delta=1e-5, max_grad_norm=1.0),
            adaptive_dp=AdaptiveDPConfig(enabled=True, ema_alpha=0.1, min_epsilon_fraction=0.1),
            output_dir=os.path.join(tmp, "output"),
        )

        report = run_simulation(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))

        assert "adaptive_dp_summary" in report, "adaptive_dp_summary missing from report"
        summary = report["adaptive_dp_summary"]
        assert summary["mechanism"] == "adaptive_dp"
        assert summary["global_epsilon"] == 8.0
        # All hospital IDs present in summary
        for hid in cfg.hospitals:
            assert hid in summary["per_client"], f"{hid} missing from per_client summary"

check("run_simulation: Adaptive DP, 2 rounds", p8_run_simulation_with_adaptive_dp)


def p8_run_simulation_scalability():
    """10-hospital fleet: verify aggregation works at scale."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = FLConfig.with_n_hospitals(10).replace(
            fl_rounds=1,
            dp=DPConfig(enabled=False),
            adaptive_dp=AdaptiveDPConfig(enabled=False),
            output_dir=os.path.join(tmp, "output"),
        )

        report = run_simulation(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))

        assert len(report["round_metrics"]) == 1
        rm = report["round_metrics"][0]
        assert rm["num_hospitals"] == 10

check("run_simulation: 10-hospital fleet, 1 round", p8_run_simulation_scalability)


def p8_checkpoint_resume():
    """Simulate a resume: run 1 round, then resume from checkpoint for round 2."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = os.path.join(tmp, "ckpt")
        out_dir = os.path.join(tmp, "output")
        cfg2 = FLConfig.quick_demo().replace(
            dp=DPConfig(enabled=False),
            adaptive_dp=AdaptiveDPConfig(enabled=False),
            fl_rounds=2,
            output_dir=out_dir,
        )

        # Full run — produces checkpoint for round 0 and round 1
        report = run_simulation(cfg2, checkpoint_dir=ckpt_dir)
        assert os.path.exists(os.path.join(ckpt_dir, "round_1.pt"))

        # Simulate resume: delete round_1 checkpoint, keep round_0
        os.remove(os.path.join(ckpt_dir, "round_1.pt"))
        with open(os.path.join(ckpt_dir, "last_round.txt"), "w") as f:
            f.write("0")

        # Re-run — should only run round 1 (resume from round 0)
        report2 = run_simulation(cfg2, checkpoint_dir=ckpt_dir)
        assert len(report2["round_metrics"]) == 1

check("run_simulation: checkpoint resume skips completed rounds", p8_checkpoint_resume)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 9: LOGGING AND METRICS CORRECTNESS
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Phase 9: Logging and Metrics Correctness ─────────────────────────────")


def p9_tracker_receives_round_metrics():
    """Verify that the NoOp tracker receives the expected metric keys."""
    from fl_tracker import NoOpTracker

    logged_calls = []

    class SpyTracker(NoOpTracker):
        def log(self, metrics, step=None):
            logged_calls.append((step, dict(metrics)))

        def log_summary(self, metrics):
            logged_calls.append(("summary", dict(metrics)))

    with tempfile.TemporaryDirectory() as tmp:
        cfg = FLConfig.quick_demo().replace(
            dp=DPConfig(enabled=False),
            adaptive_dp=AdaptiveDPConfig(enabled=False),
            output_dir=os.path.join(tmp, "output"),
        )
        spy = SpyTracker()
        run_simulation(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"), tracker=spy)

    steps = [c[0] for c in logged_calls]
    all_keys = set()
    for _, metrics in logged_calls:
        all_keys |= metrics.keys()

    assert 0 in steps, "Round 0 metrics not logged"
    assert "round/avg_loss" in all_keys, f"Missing round/avg_loss. Keys: {sorted(all_keys)}"
    assert "round/weight_divergence" in all_keys
    assert "round/aggregation_time" in all_keys
    assert "round/total_samples" in all_keys
    assert "summary" in steps, "log_summary never called"

check("Tracker: receives round/avg_loss, divergence, samples", p9_tracker_receives_round_metrics)


def p9_per_client_metrics_logged():
    """Verify per-client training metrics are sent to the tracker."""
    from fl_tracker import NoOpTracker

    logged_keys = set()

    class SpyTracker(NoOpTracker):
        def log(self, metrics, step=None):
            logged_keys.update(metrics.keys())

    with tempfile.TemporaryDirectory() as tmp:
        cfg = FLConfig.quick_demo().replace(
            dp=DPConfig(enabled=False),
            adaptive_dp=AdaptiveDPConfig(enabled=False),
            output_dir=os.path.join(tmp, "output"),
        )
        spy = SpyTracker()
        run_simulation(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"), tracker=spy)

    for hid in cfg.hospitals:
        for suffix in ("train_loss", "num_samples", "training_time_seconds"):
            key = f"{hid}/{suffix}"
            assert key in logged_keys, f"Missing tracker key: {key}"

check("Tracker: per-client metrics logged for each hospital", p9_per_client_metrics_logged)


def p9_privacy_budget_logged():
    """Privacy budget keys must appear in tracker when DP is enabled."""
    from fl_tracker import NoOpTracker

    logged_keys = set()

    class SpyTracker(NoOpTracker):
        def log(self, metrics, step=None):
            logged_keys.update(metrics.keys())

    with tempfile.TemporaryDirectory() as tmp:
        cfg = FLConfig.quick_demo().replace(
            dp=DPConfig(enabled=True, epsilon=8.0, delta=1e-5, max_grad_norm=1.0),
            adaptive_dp=AdaptiveDPConfig(enabled=False),
            output_dir=os.path.join(tmp, "output"),
        )
        spy = SpyTracker()
        run_simulation(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"), tracker=spy)

    assert "privacy/budget_spent" in logged_keys, f"Keys: {sorted(logged_keys)}"
    assert "privacy/budget_remaining" in logged_keys
    assert "privacy/budget_pct_used" in logged_keys

check("Tracker: privacy budget keys logged when DP enabled", p9_privacy_budget_logged)


def p9_report_json_is_valid():
    """Report JSON must load back cleanly and contain required top-level keys."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = FLConfig.quick_demo().replace(
            dp=DPConfig(enabled=False),
            adaptive_dp=AdaptiveDPConfig(enabled=False),
            output_dir=os.path.join(tmp, "output"),
        )
        run_simulation(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))

        report_path = os.path.join(cfg.metrics_dir, "fl_training_report.json")
        with open(report_path) as f:
            loaded = json.load(f)

    required = {"config", "round_metrics", "hospital_contributions",
                "total_training_time", "eval_results", "all_round_metrics"}
    missing = required - loaded.keys()
    assert not missing, f"Report JSON missing keys: {missing}"

check("Report JSON: all required top-level keys present", p9_report_json_is_valid)


def p9_adaptive_dp_budget_consistency():
    """ε spent per hospital must match the sum of per-round allocations."""
    mech = AdaptiveDPMechanism(
        hospital_ids=["A", "B"], global_epsilon=8.0, delta=1e-5, fl_rounds=5,
        min_epsilon_fraction=0.1,
    )
    total_alloc = {"A": 0.0, "B": 0.0}
    for rnd in range(3):
        alloc = mech.compute_epsilon_allocation(rnd)
        for hid in ["A", "B"]:
            w = make_lora_weights(scale=0.1)
            mech.apply_noise(hid, w, rnd, alloc[hid])
            total_alloc[hid] += alloc[hid]

    for hid in ["A", "B"]:
        spent = mech.get_budget_spent(hid)
        assert abs(spent - total_alloc[hid]) < 1e-9, (
            f"{hid}: spent={spent:.6f} != total_alloc={total_alloc[hid]:.6f}"
        )

check("Adaptive DP: budget_spent = sum of per-round allocations", p9_adaptive_dp_budget_consistency)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 10: EVALUATION SYSTEM
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Phase 10: Evaluation System ──────────────────────────────────────────")


def p10_eval_accumulator_privacy_snapshots():
    acc = EvalAccumulator()
    for rnd in range(5):
        acc.add_privacy_snapshot(
            round_num=rnd,
            budget_spent=rnd * 1.0,
            budget_remaining=8.0 - rnd * 1.0,
            total_epsilon=8.0,
        )
    snaps = acc.get_privacy_metrics()
    assert len(snaps) == 5
    assert snaps[0]["budget_spent"] == 0.0
    assert abs(snaps[4]["budget_pct"] - 50.0) < 1e-6

check("EvalAccumulator: privacy snapshots tracked correctly", p10_eval_accumulator_privacy_snapshots)


def p10_eval_accumulator_client_metrics():
    acc = EvalAccumulator()
    for rnd in range(3):
        acc.add_client_metrics("hospital_00", {
            "hospital": "General Hospital",
            "train_loss": 0.8 - rnd * 0.1,
            "round": rnd,
        })
    df = acc.get_client_metrics()
    # get_client_metrics() returns Dict[hospital_id → List[dict]]
    assert isinstance(df, dict), f"Expected dict, got {type(df)}"
    assert "hospital_00" in df, f"hospital_00 missing from client metrics: {df.keys()}"
    rows = df["hospital_00"]
    assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"
    assert rows[0]["hospital"] == "General Hospital"

check("EvalAccumulator: client metrics tracked", p10_eval_accumulator_client_metrics)


def p10_eval_summary_best_vs_final():
    acc = EvalAccumulator()
    accs = [0.50, 0.65, 0.60, 0.70, 0.68]
    for rn, a in enumerate(accs):
        acc.add_eval_result(EvalResult(
            run_label="Run", round_num=rn, accuracy=a,
            loss=1.0-a, perplexity=2.0, num_eval_samples=50,
            elapsed_seconds=0.1, privacy_budget_spent=rn * 0.5,
        ))
    s = acc.summary()
    assert abs(s["Run"]["best_accuracy"] - 0.70) < 1e-9, f"best={s['Run']['best_accuracy']}"
    assert abs(s["Run"]["final_accuracy"] - 0.68) < 1e-9, f"final={s['Run']['final_accuracy']}"
    # best_round is the index of the highest-accuracy result (round with acc=0.70 is index 3)
    assert s["Run"]["best_round"] == 3, f"best_round={s['Run']['best_round']}"
    assert s["Run"]["rounds_evaluated"] == 5

check("EvalAccumulator: summary best vs final accuracy", p10_eval_summary_best_vs_final)


def p10_on_round_end_callback():
    """Verify the on_round_end hook fires with correct signature."""
    callback_calls = []

    def on_round_end(round_num, global_weights, privacy_budget_spent):
        callback_calls.append({
            "round": round_num,
            "num_weights": len(global_weights),
            "budget": privacy_budget_spent,
        })

    with tempfile.TemporaryDirectory() as tmp:
        cfg = FLConfig.quick_demo().replace(
            dp=DPConfig(enabled=False),
            adaptive_dp=AdaptiveDPConfig(enabled=False),
            output_dir=os.path.join(tmp, "output"),
        )
        run_simulation(
            cfg,
            checkpoint_dir=os.path.join(tmp, "ckpt"),
            on_round_end=on_round_end,
        )

    assert len(callback_calls) == cfg.fl_rounds, (
        f"Expected {cfg.fl_rounds} callback calls, got {len(callback_calls)}"
    )
    for i, call in enumerate(callback_calls):
        assert call["round"] == i
        assert call["num_weights"] == len(_LORA_KEYS)
        assert call["budget"] >= 0.0

check("on_round_end: fires once per round with correct args", p10_on_round_end_callback)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 11: PLOTTING SYSTEM
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Phase 11: Plotting System ─────────────────────────────────────────────")

import matplotlib
matplotlib.use("Agg")


def p11_plots_generated_from_pipeline():
    """Full pipeline → accumulate real metrics → save plots."""
    from fl_tracker import NoOpTracker

    with tempfile.TemporaryDirectory() as tmp:
        acc = EvalAccumulator()
        cfg = FLConfig.quick_demo().replace(
            dp=DPConfig(enabled=False),
            adaptive_dp=AdaptiveDPConfig(enabled=False),
            output_dir=os.path.join(tmp, "output"),
        )

        call_count = [0]

        def on_round_end(round_num, global_weights, budget):
            call_count[0] += 1
            acc.add_eval_result(EvalResult(
                run_label="No-DP",
                round_num=round_num,
                accuracy=0.55 + round_num * 0.05,
                loss=0.80 - round_num * 0.05,
                perplexity=2.0,
                num_eval_samples=10,
                elapsed_seconds=0.1,
                privacy_budget_spent=budget,
            ))
            acc.add_privacy_snapshot(round_num, budget, 0.0, 8.0)

        run_simulation(
            cfg, checkpoint_dir=os.path.join(tmp, "ckpt"), on_round_end=on_round_end
        )

        plot_dir = os.path.join(tmp, "plots")
        plotter = ResultsPlotter(output_dir=plot_dir, fmt="png", dpi=72)
        paths = plotter.save_all(acc)

        assert len(paths) > 0, "No plots generated"
        for p in paths:
            assert os.path.exists(p), f"Plot file missing: {p}"
            assert os.path.getsize(p) > 0, f"Plot file is empty: {p}"

check("Plotting: real pipeline metrics → PNG files generated", p11_plots_generated_from_pipeline)


def p11_multi_run_comparison_plots():
    """Simulate a 2-config comparison and verify plots contain both series."""
    with tempfile.TemporaryDirectory() as tmp:
        acc = EvalAccumulator()
        for label, base_acc in [("No-DP", 0.60), ("DP ε=8", 0.55)]:
            for rnd in range(3):
                acc.add_eval_result(EvalResult(
                    run_label=label, round_num=rnd,
                    accuracy=base_acc + rnd * 0.03,
                    loss=1.0 - (base_acc + rnd * 0.03),
                    perplexity=2.0, num_eval_samples=50,
                    elapsed_seconds=0.5, privacy_budget_spent=rnd * 0.5,
                ))

        plot_dir = os.path.join(tmp, "plots")
        plotter = ResultsPlotter(output_dir=plot_dir, fmt="png", dpi=72)
        paths = plotter.save_all(acc)
        assert len(paths) >= 1
        # Accuracy curve should be among the plots
        acc_plots = [p for p in paths if "accuracy" in os.path.basename(p)]
        assert len(acc_plots) > 0, f"No accuracy plot found. Plots: {paths}"

check("Plotting: 2-series comparison plot includes accuracy curve", p11_multi_run_comparison_plots)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 12: EDGE CASES & ERROR HANDLING
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Phase 12: Edge Cases & Error Handling ────────────────────────────────")


def p12_empty_hospitals_raises():
    from fl_config import ConfigurationError
    cfg = FLConfig(hospitals={}, fl_rounds=1)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run_simulation(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
        raise AssertionError("should have raised ConfigurationError")
    except ConfigurationError:
        pass

check("run_simulation: empty hospitals raises ConfigurationError", p12_empty_hospitals_raises)


def p12_single_hospital():
    """Single-hospital degenerate case: aggregation of one client."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = FLConfig.with_n_hospitals(1).replace(
            fl_rounds=1,
            dp=DPConfig(enabled=False),
            adaptive_dp=AdaptiveDPConfig(enabled=False),
            output_dir=os.path.join(tmp, "output"),
        )
        report = run_simulation(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
        rm = report["round_metrics"][0]
        assert rm["num_hospitals"] == 1

check("run_simulation: single-hospital works", p12_single_hospital)


def p12_failed_client_isolated():
    """Patch one client to raise — pipeline should continue with remaining clients."""
    from fl_client import HospitalClient

    original_train = HospitalClient.train_local
    call_count = [0]

    def flaky_train(self, *args, **kwargs):
        call_count[0] += 1
        if self.hospital_id == "hospital_00" and call_count[0] == 1:
            raise RuntimeError("Simulated client failure")
        return original_train(self, *args, **kwargs)

    HospitalClient.train_local = flaky_train

    try:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = FLConfig.quick_demo().replace(
                dp=DPConfig(enabled=False),
                adaptive_dp=AdaptiveDPConfig(enabled=False),
                output_dir=os.path.join(tmp, "output"),
            )
            report = run_simulation(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            # Round 0 should complete with 2/3 hospitals; round 1 with all 3
            assert len(report["round_metrics"]) == 2
    finally:
        HospitalClient.train_local = original_train

check("Error isolation: failed client skipped, training continues", p12_failed_client_isolated)


def p12_dp_sigma_formula_correctness():
    """σ = C · √(2 ln(1.25/δ)) / ε — verify against direct calculation."""
    import math
    from fl_config import DPConfig
    for eps, C, delta in [(1.0, 1.0, 1e-5), (8.0, 2.0, 1e-6), (0.5, 0.5, 1e-3)]:
        dp = DPConfig(enabled=True, epsilon=eps, delta=delta, max_grad_norm=C)
        expected = C * math.sqrt(2.0 * math.log(1.25 / delta)) / eps
        assert abs(dp.sigma - expected) < 1e-12, f"sigma mismatch for eps={eps}"

check("DPConfig.sigma: formula correct for multiple (ε, C, δ) combinations", p12_dp_sigma_formula_correctness)


def p12_fl_config_replace_immutability():
    """replace() must not mutate the original config."""
    base = FLConfig(fl_rounds=10)
    modified = base.replace(fl_rounds=5)
    assert base.fl_rounds == 10
    assert modified.fl_rounds == 5
    assert modified.dp == base.dp  # other fields propagated

check("FLConfig.replace(): original is unchanged, other fields propagated", p12_fl_config_replace_immutability)


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

total = PASSED + FAILED
print(f"\n{'='*60}")
print(f"RESULTS: {PASSED}/{total} passed, {FAILED} failed")
print(f"{'='*60}")

if RESULTS:
    # Save results to file for review
    summary_path = os.path.join(
        os.path.dirname(__file__), "..", "outputs", "integration_test_results.json"
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "total": total,
            "passed": PASSED,
            "failed": FAILED,
            "results": RESULTS,
        }, f, indent=2)
    print(f"Full results saved to: {summary_path}")

if FAILED > 0:
    print("\nFailed tests:")
    for r in RESULTS:
        if r["status"] == "FAIL":
            print(f"  - {r['name']}: {r.get('error', '')}")
    sys.exit(1)
else:
    print("\nAll integration tests passed ✓")
    sys.exit(0)
