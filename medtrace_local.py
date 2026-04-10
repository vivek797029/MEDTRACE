"""
MedTrace — Local inference script (Mac M2 / CPU)
================================================
Runs the trained federated LoRA adapter on your Mac M2.
Uses Apple MPS (Metal) when available — much faster than pure CPU.

SETUP (one-time):
    pip install torch torchvision torchaudio      # PyTorch with MPS support
    pip install transformers peft safetensors accelerate

USAGE:
    python medtrace_local.py
    python medtrace_local.py --adapter ~/Downloads/round_9
    python medtrace_local.py --adapter ~/Downloads/round_9 --max-tokens 600
"""

import argparse
import os
import sys
import time

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="MedTrace local inference (Mac M2 / CPU)")
parser.add_argument(
    "--adapter",
    default=os.path.expanduser("~/Downloads/round_9"),
    help=(
        "Path to adapter folder (round_9) OR merged model folder (medtrace_merged). "
        "If merged, set --base-model to the same path."
    ),
)
parser.add_argument(
    "--base-model",
    default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    help=(
        "HuggingFace model id, OR local path to merged model folder. "
        "Default: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (downloaded automatically)"
    ),
)
parser.add_argument(
    "--max-tokens", type=int, default=400,
    help="Max new tokens to generate per answer (default: 400)",
)
parser.add_argument(
    "--device", default="auto",
    choices=["auto", "mps", "cpu"],
    help="Device: auto = MPS if available, else CPU (default: auto)",
)
args = parser.parse_args()

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftConfig, get_peft_model
    import safetensors.torch as st
except ImportError as e:
    print(f"\n❌  Missing package: {e}")
    print("    Run:  pip install torch transformers peft safetensors accelerate\n")
    sys.exit(1)

# ── Pick device ───────────────────────────────────────────────────────────────
if args.device == "auto":
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
else:
    DEVICE = args.device

print(f"\n🖥  Device : {DEVICE.upper()}"
      + ("  (Apple Silicon GPU — fast!)" if DEVICE == "mps" else "  (CPU — ~1-2 min/question)"))

# ── Validate adapter path ─────────────────────────────────────────────────────
ADAPTER = os.path.expanduser(args.adapter)
if not os.path.isdir(ADAPTER):
    print(f"\n❌  Adapter folder not found: {ADAPTER}")
    print("    Download the round_9 folder from Google Drive:")
    print("    Drive → MedTrace/outputs/global_model/round_9  →  Download")
    print(f"    Then run:  python medtrace_local.py --adapter /path/to/round_9\n")
    sys.exit(1)

ADAPTER_FILE  = os.path.join(ADAPTER, "adapter_model.safetensors")
ADAPTER_BIN   = os.path.join(ADAPTER, "adapter_model.bin")
# A merged model has model.safetensors but no adapter_config.json LoRA files
IS_MERGED = (
    os.path.exists(os.path.join(ADAPTER, "model.safetensors")) and
    not os.path.exists(os.path.join(ADAPTER, "adapter_config.json"))
)
if not IS_MERGED and not os.path.exists(ADAPTER_FILE) and not os.path.exists(ADAPTER_BIN):
    print(f"\n❌  No adapter weights found in {ADAPTER}")
    print("    Expected:  adapter_model.safetensors  (or adapter_model.bin)")
    print("    Or a merged model with:  model.safetensors\n")
    sys.exit(1)

SYSTEM_MSG = (
    "You are a helpful clinical reasoning assistant trained across multiple "
    "hospital datasets using federated learning. Provide clear, accurate, "
    "evidence-based answers to clinical questions."
)

# ── Load tokenizer ────────────────────────────────────────────────────────────
print(f"\n📂  Model  : {ADAPTER}")
print(f"📦  Base   : {args.base_model if not IS_MERGED else '(self-contained merged model)'}")
print(f"🔀  Type   : {'Merged (no adapter needed)' if IS_MERGED else 'Base + LoRA adapter'}")
print("\nLoading tokenizer...")
# Fix bad tokenizer_class written by some PEFT/transformers versions
import json as _json
_tc_path = os.path.join(ADAPTER, "tokenizer_config.json")
if os.path.exists(_tc_path):
    with open(_tc_path) as _f:
        _tc = _json.load(_f)
    if _tc.get("tokenizer_class") not in (None, "LlamaTokenizer", "LlamaTokenizerFast", "PreTrainedTokenizerFast"):
        print(f"  Fixing tokenizer_class: {_tc['tokenizer_class']} → LlamaTokenizerFast")
        _tc["tokenizer_class"] = "LlamaTokenizerFast"
        with open(_tc_path, "w") as _f:
            _json.dump(_tc, _f, indent=2)

from transformers import LlamaTokenizerFast
try:
    tokenizer = LlamaTokenizerFast.from_pretrained(ADAPTER)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER, use_fast=False)

t0 = time.time()

if IS_MERGED:
    # ── Merged model: single load, no PEFT needed ─────────────────────────────
    print("Loading merged model (self-contained, no HuggingFace download)...")
    model = AutoModelForCausalLM.from_pretrained(
        ADAPTER,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
else:
    # ── Base + LoRA adapter ───────────────────────────────────────────────────
    print("Loading base model (downloads ~2 GB first time, cached after)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    print("Attaching LoRA adapter...")
    peft_config = PeftConfig.from_pretrained(ADAPTER)
    model = get_peft_model(base, peft_config)

    # Load adapter weights with device='cpu' — zero GPU/MPS calls during load
    if os.path.exists(ADAPTER_FILE):
        adapter_weights = st.load_file(ADAPTER_FILE, device="cpu")
    else:
        adapter_weights = torch.load(ADAPTER_BIN, map_location="cpu")

    missing, unexpected = model.load_state_dict(adapter_weights, strict=False)
    print(f"Adapter loaded  (missing={len(missing)}, unexpected={len(unexpected)})")

model.eval()

# ── Move to target device ─────────────────────────────────────────────────────
print(f"Moving model to {DEVICE.upper()}...")
try:
    model = model.to(DEVICE)
    print(f"✅  Model ready on {DEVICE.upper()}  ({time.time()-t0:.0f}s load time)\n")
except Exception as e:
    DEVICE = "cpu"
    print(f"⚠️  {DEVICE} move failed ({e}) — falling back to CPU\n")

# ── Inference helper ──────────────────────────────────────────────────────────
def ask(question: str, max_new_tokens: int = args.max_tokens) -> str:
    prompt = (
        f"<|system|>\n{SYSTEM_MSG}</s>\n"
        f"<|user|>\n{question}</s>\n"
        f"<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    t_start = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t_start
    answer = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return answer, elapsed

# ── Interactive chat loop ─────────────────────────────────────────────────────
print("=" * 70)
print("  MedTrace Clinical Reasoning — Local Chat")
print(f"  Model  : TinyLlama 1.1B + LoRA  |  Device : {DEVICE.upper()}")
print("  Type a clinical question, or type 'quit' to exit.")
print("=" * 70 + "\n")

while True:
    try:
        question = input("Question: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break

    if not question or question.lower() in ("quit", "exit", "q"):
        print("Goodbye.")
        break

    print("\nThinking...\n")
    answer, elapsed = ask(question)
    print(f"Answer:\n{answer}")
    print(f"\n[{elapsed:.1f}s]")
    print("-" * 70 + "\n")
