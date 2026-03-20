"""
MedTrace Fast Training — Designed for M2 8GB Mac
==================================================
Trains in short 2-3 hour rounds using 1000 examples per round.
You can run multiple rounds — each round resumes from the last checkpoint.

Round 1: python src/03_train_fast.py --round 1
Round 2: python src/03_train_fast.py --round 2
Round 3: python src/03_train_fast.py --round 3
...and so on until you're satisfied with the results!

Each round:
  - Trains on 1000 examples (1 epoch)
  - Takes ~2-3 hours on M2 8GB
  - Saves a checkpoint you can use immediately
  - Next round loads from the previous checkpoint

You can test your model after ANY round:
  python src/05_inference.py --interactive
"""

import os
import sys
import json
import argparse
import random

# Must be set BEFORE importing torch
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "checkpoints")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH = 512
EXAMPLES_PER_ROUND = 1000   # 1000 examples per round = ~2-3 hours

# LoRA config
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Training config
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8   # Effective batch = 8
LEARNING_RATE = 2e-4
LOGGING_STEPS = 10


# ============================================================
# DATASET
# ============================================================

class MedTraceDataset(Dataset):
    def __init__(self, data: list, tokenizer, max_length: int = MAX_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = self._format_as_chat(item)
        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }

    def _format_as_chat(self, item: dict) -> str:
        system_msg = (
            "You are MedTrace, a clinical reasoning system that generates "
            "step-by-step, auditable diagnostic reasoning chains. Every step must be "
            "typed (symptom/finding/mechanism/rule/inference/conclusion), sourced "
            "(patient_data/medical_knowledge/clinical_guideline/logical_deduction), "
            "and explicitly state its dependencies on previous steps."
        )
        options_str = "\n".join([f"  {k}. {v}" for k, v in item["options"].items()])
        user_msg = f"Question: {item['question']}\n\nOptions:\n{options_str}"
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
        return (
            f"<|system|>\n{system_msg}</s>\n"
            f"<|user|>\n{user_msg}</s>\n"
            f"<|assistant|>\n{assistant_msg}</s>"
        )


# ============================================================
# TRAINING
# ============================================================

def train_round(round_num: int):
    """Train one round on a chunk of data."""

    print(f"\n{'='*60}")
    print(f"  MedTrace Training — Round {round_num}")
    print(f"{'='*60}\n")

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🔥 Using MPS (Metal GPU)")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU")

    # Load full training data
    data_path = os.path.join(DATA_DIR, "medtrace_train.json")
    if not os.path.exists(data_path):
        print("❌ Run 01_download_data.py and 02_build_reasoning_traces.py first!")
        sys.exit(1)

    with open(data_path) as f:
        all_data = json.load(f)

    # Shuffle and pick chunk for this round
    random.seed(round_num * 42)  # Different seed per round = different data
    random.shuffle(all_data)
    round_data = all_data[:EXAMPLES_PER_ROUND]

    print(f"📊 Round {round_num}: Training on {len(round_data)} examples (of {len(all_data)} total)")
    print(f"   Each round uses a different random subset of your data.\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Check if previous checkpoint exists
    prev_checkpoint = os.path.join(OUTPUT_DIR, f"medtrace-round-{round_num - 1}")
    final_checkpoint = os.path.join(OUTPUT_DIR, "medtrace-final")

    if round_num > 1 and os.path.exists(prev_checkpoint):
        print(f"📥 Loading from Round {round_num - 1} checkpoint...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        )
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
        model = PeftModel.from_pretrained(base_model, prev_checkpoint, is_trainable=True)
        print("✅ Resumed from previous round!\n")
    elif round_num > 1 and os.path.exists(final_checkpoint):
        print(f"📥 Loading from medtrace-final checkpoint...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        )
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
        model = PeftModel.from_pretrained(base_model, final_checkpoint, is_trainable=True)
        print("✅ Resumed from final checkpoint!\n")
    else:
        print(f"📥 Loading fresh TinyLlama + LoRA...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        )
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGET_MODULES, bias="none",
        )
        model = get_peft_model(base_model, lora_config)
        print("✅ Fresh model loaded!\n")

    # Dataset
    train_dataset = MedTraceDataset(round_data, tokenizer)

    # Calculate steps
    total_steps = len(round_data) // (BATCH_SIZE * GRADIENT_ACCUMULATION)
    est_hours = (total_steps * 120) / 3600  # ~120 seconds per step estimate

    print(f"   Steps this round:  {total_steps}")
    print(f"   Estimated time:    ~{est_hours:.1f} hours")
    print(f"   Logging every:     {LOGGING_STEPS} steps\n")

    # Training args
    round_output = os.path.join(OUTPUT_DIR, f"round-{round_num}-working")
    training_args = TrainingArguments(
        output_dir=round_output,
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=5,
        logging_steps=LOGGING_STEPS,
        save_steps=total_steps,  # Save only at end
        save_total_limit=2,
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, data_collator=data_collator,
    )

    # Train
    print("🚀 Training started!")
    print(f"   Watch the loss go down — lower = model is learning better.\n")
    print("=" * 60)

    trainer.train()

    # Save this round's checkpoint
    save_path = os.path.join(OUTPUT_DIR, f"medtrace-round-{round_num}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Also save as "final" so inference script can find it
    model.save_pretrained(final_checkpoint)
    tokenizer.save_pretrained(final_checkpoint)

    print("\n" + "=" * 60)
    print(f"✅ Round {round_num} complete!")
    print(f"   Checkpoint: {save_path}")
    print(f"   Final model: {final_checkpoint}")
    print(f"\n   🧪 Test it now:  python src/05_inference.py --interactive")
    print(f"   📋 Verify it:    python src/04_verify_and_evaluate.py")
    print(f"   🔄 Next round:   python src/03_train_fast.py --round {round_num + 1}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=1,
                        help="Training round number (1, 2, 3, ...)")
    args = parser.parse_args()
    train_round(round_num=args.round)
