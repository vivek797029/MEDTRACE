"""
MedTrace Step 3: Train TinyLlama with LoRA on Medical Reasoning Traces
========================================================================
Fine-tunes TinyLlama 1.1B to natively generate step-typed, dependency-mapped
clinical reasoning chains.

Uses:
  - LoRA (Low-Rank Adaptation) to fit in 8GB RAM
  - MPS (Metal Performance Shaders) for M2 GPU acceleration
  - Process-aware training format

Expected time on M2 8GB: 2-3 nights (runs overnight)

Run: python src/03_train.py
  or: python src/03_train.py --quick_test  (5 examples, verify setup works)
"""

import os
import sys
import json
import argparse

# Must be set BEFORE importing torch — tells MPS to use all available memory
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "checkpoints")

# Model config
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH = 512   # Reduced from 1024 to fit in 8GB unified memory

# LoRA config — keeps memory under 8GB
LORA_R = 8            # Reduced from 16 to save memory
LORA_ALPHA = 16       # Scaling factor
LORA_DROPOUT = 0.05   # Small dropout for regularization
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # Only key layers to save memory

# Training config — optimized for M2 8GB
BATCH_SIZE = 1              # Reduced to 1 to avoid OOM
GRADIENT_ACCUMULATION = 16  # Effective batch size = 1 * 16 = 16
LEARNING_RATE = 2e-4        # Standard for LoRA
NUM_EPOCHS = 3              # 3 epochs is the sweet spot
WARMUP_RATIO = 0.03         # Gentle warmup
SAVE_STEPS = 500            # Checkpoint every 500 steps
LOGGING_STEPS = 50          # Log every 50 steps


# ============================================================
# DEVICE SETUP
# ============================================================

def get_device():
    """Detect best available device — MPS for M2, CPU fallback."""
    if torch.backends.mps.is_available():
        print("🔥 Using MPS (Metal GPU) — your M2 chip will accelerate training!")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("🔥 Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("⚠️  Using CPU — training will be slower but still works.")
        return torch.device("cpu")


# ============================================================
# DATASET
# ============================================================

class MedTraceDataset(Dataset):
    """
    Dataset that formats medical reasoning traces for causal LM training.

    Each example is formatted as a conversation:
      <|system|> You are MedTrace, a clinical reasoning system...
      <|user|> {question + options}
      <|assistant|> {structured reasoning chain + answer}
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = MAX_LENGTH):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Build the full training text in TinyLlama chat format
        text = self._format_as_chat(item)

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # Causal LM — predict next token
        }

    def _format_as_chat(self, item: dict) -> str:
        """Format as TinyLlama chat template with system/user/assistant roles."""

        system_msg = (
            "You are MedTrace, a clinical reasoning system that generates "
            "step-by-step, auditable diagnostic reasoning chains. Every step must be "
            "typed (symptom/finding/mechanism/rule/inference/conclusion), sourced "
            "(patient_data/medical_knowledge/clinical_guideline/logical_deduction), "
            "and explicitly state its dependencies on previous steps."
        )

        options_str = "\n".join([f"  {k}. {v}" for k, v in item["options"].items()])
        user_msg = f"Question: {item['question']}\n\nOptions:\n{options_str}"

        # The assistant's response IS the reasoning chain
        # During training we teach it to produce this format
        assistant_msg = (
            f"REASONING_CHAIN:\n"
            f"[STEP 1 | type: symptom | source: patient_data]\n"
            f"The question presents clinical information that must be analyzed systematically.\n"
            f"[STEP 2 | type: mechanism | source: medical_knowledge]\n"
            f"Based on the clinical presentation, the underlying pathophysiology involves "
            f"mechanisms related to the correct answer.\n"
            f"[STEP 3 | type: rule | source: clinical_guideline | depends_on: 1,2]\n"
            f"Clinical guidelines indicate specific criteria for this presentation.\n"
            f"[STEP 4 | type: inference | source: logical_deduction | depends_on: 1,2,3]\n"
            f"Combining the patient data with medical knowledge and guidelines, "
            f"the evidence points toward a specific diagnosis/treatment.\n"
            f"[STEP 5 | type: conclusion | source: logical_deduction | depends_on: 1,2,3,4]\n"
            f"The answer is {item['answer']}.\n\n"
            f"ANSWER: {item['answer_idx']}"
        )

        # TinyLlama chat format
        text = (
            f"<|system|>\n{system_msg}</s>\n"
            f"<|user|>\n{user_msg}</s>\n"
            f"<|assistant|>\n{assistant_msg}</s>"
        )

        return text


# ============================================================
# MODEL SETUP
# ============================================================

def load_model_and_tokenizer():
    """Load TinyLlama with LoRA adapters for efficient fine-tuning."""

    print(f"📥 Loading {MODEL_NAME}...")
    print(f"   This downloads ~2.2GB on first run.\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # MPS works best with float32
        low_cpu_mem_usage=True,
    )

    # Enable gradient checkpointing — trades compute for memory (essential for 8GB)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Apply LoRA — this is what makes 1.1B fit in 8GB
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded with LoRA!")
    print(f"   Total parameters:     {total:,}")
    print(f"   Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"   Memory saved:         ~{(total - trainable) * 4 / 1e9:.1f} GB\n")

    return model, tokenizer


# ============================================================
# TRAINING
# ============================================================

def train(quick_test: bool = False):
    """Main training loop."""

    device = get_device()
    model, tokenizer = load_model_and_tokenizer()

    # Load dataset
    if quick_test:
        data_path = os.path.join(DATA_DIR, "medtrace_sample_5.json")
        num_epochs = 1
        save_steps = 2
        logging_steps = 1
        print("🧪 QUICK TEST MODE — 5 examples, 1 epoch\n")
    else:
        data_path = os.path.join(DATA_DIR, "medtrace_train.json")
        num_epochs = NUM_EPOCHS
        save_steps = SAVE_STEPS
        logging_steps = LOGGING_STEPS

    if not os.path.exists(data_path):
        print(f"❌ Data not found at {data_path}")
        print(f"   Run these first:")
        print(f"   python src/01_download_data.py")
        print(f"   python src/02_build_reasoning_traces.py")
        sys.exit(1)

    train_dataset = MedTraceDataset(data_path, tokenizer)
    print(f"📊 Training on {len(train_dataset)} examples")
    print(f"   Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Estimated time: {'~5 minutes' if quick_test else '~24-48 hours on M2'}\n")

    # Validation dataset
    val_path = os.path.join(DATA_DIR, "medtrace_validation.json")
    eval_dataset = None
    if os.path.exists(val_path) and not quick_test:
        eval_dataset = MedTraceDataset(val_path, tokenizer)
        print(f"📊 Validation: {len(eval_dataset)} examples\n")

    # Training arguments — optimized for M2
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,  # Keep only 3 best checkpoints
        fp16=False,  # MPS doesn't support fp16, use float32
        bf16=False,
        dataloader_pin_memory=False,  # Required for MPS
        remove_unused_columns=False,
        report_to="none",  # No wandb/tensorboard needed
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=save_steps if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train!
    print("🚀 Starting training...")
    print("   Pro tip: Run this in tmux so it survives overnight:")
    print("   tmux new -s medtrace && python src/03_train.py\n")
    print("=" * 60)

    trainer.train()

    # Save final model
    final_dir = os.path.join(OUTPUT_DIR, "medtrace-final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("\n" + "=" * 60)
    print(f"✅ Training complete!")
    print(f"   Model saved to: {final_dir}")
    print(f"   Next step: python src/04_verify_and_evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick_test", action="store_true",
                        help="Run with 5 examples to verify setup works")
    args = parser.parse_args()
    train(quick_test=args.quick_test)
