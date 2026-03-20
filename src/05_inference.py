"""
MedTrace Step 5: Run Inference with Trained Model
===================================================
Load your fine-tuned MedTrace model and generate auditable
reasoning chains for any medical question.

Run: python src/05_inference.py
     python src/05_inference.py --interactive   (chat mode)
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "outputs", "checkpoints", "medtrace-final")
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_medtrace_model():
    """Load the fine-tuned MedTrace model."""
    print("📥 Loading MedTrace model...")

    if not os.path.exists(CHECKPOINT_DIR):
        print(f"❌ Trained model not found at {CHECKPOINT_DIR}")
        print(f"   Run training first: python src/03_train.py")
        sys.exit(1)

    device = get_device()
    print(f"   Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
    model = model.to(device)
    model.eval()

    print("✅ MedTrace loaded and ready!\n")
    return model, tokenizer, device


def generate_reasoning_chain(
    model, tokenizer, device,
    question: str,
    options: dict = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate a structured reasoning chain for a medical question."""

    system_msg = (
        "You are MedTrace, a clinical reasoning system that generates "
        "step-by-step, auditable diagnostic reasoning chains. Every step must be "
        "typed (symptom/finding/mechanism/rule/inference/conclusion), sourced "
        "(patient_data/medical_knowledge/clinical_guideline/logical_deduction), "
        "and explicitly state its dependencies on previous steps."
    )

    if options:
        options_str = "\n".join([f"  {k}. {v}" for k, v in options.items()])
        user_msg = f"Question: {question}\n\nOptions:\n{options_str}"
    else:
        user_msg = f"Question: {question}"

    prompt = (
        f"<|system|>\n{system_msg}</s>\n"
        f"<|user|>\n{user_msg}</s>\n"
        f"<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract only the assistant's response
    if "<|assistant|>" in full_output:
        response = full_output.split("<|assistant|>")[-1]
        response = response.replace("</s>", "").strip()
    else:
        response = full_output[len(prompt):]

    return response


def run_demo():
    """Run demo with sample medical questions."""

    model, tokenizer, device = load_medtrace_model()

    demo_questions = [
        {
            "question": (
                "A 45-year-old woman presents with fatigue, weight gain, "
                "constipation, and cold intolerance for 3 months. Physical exam "
                "shows dry skin and delayed deep tendon reflexes. TSH is 12 mIU/L "
                "(normal: 0.5-4.5). What is the most appropriate treatment?"
            ),
            "options": {
                "A": "Levothyroxine",
                "B": "Methimazole",
                "C": "Radioactive iodine",
                "D": "Propranolol",
            },
        },
        {
            "question": (
                "A 60-year-old male with a history of smoking presents with "
                "hemoptysis, weight loss, and a 3cm hilar mass on chest CT. "
                "What is the most likely diagnosis?"
            ),
            "options": {
                "A": "Tuberculosis",
                "B": "Lung adenocarcinoma",
                "C": "Squamous cell carcinoma of the lung",
                "D": "Small cell lung cancer",
            },
        },
    ]

    from src_verify import evaluate_model_output  # noqa

    for i, q in enumerate(demo_questions, 1):
        print(f"\n{'='*70}")
        print(f"  QUESTION {i}")
        print(f"{'='*70}")
        print(f"  {q['question'][:100]}...")

        response = generate_reasoning_chain(
            model, tokenizer, device,
            question=q["question"],
            options=q["options"],
        )

        print(f"\n  MODEL OUTPUT:")
        print(f"  {'-'*60}")
        print(response)

        print(f"\n  VERIFICATION:")
        print(f"  {'-'*60}")
        try:
            # Import verifier
            sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
            from importlib import import_module
            verify_mod = import_module("04_verify_and_evaluate")
            result = verify_mod.evaluate_model_output(response)
            status = "✅ VALID" if result["is_valid"] else "❌ INVALID"
            print(f"  {status} | Score: {result['score']:.3f}")
            for e in result.get("errors", []):
                print(f"    ❌ {e}")
            for w in result.get("warnings", []):
                print(f"    ⚠️  {w}")
        except Exception as e:
            print(f"  (Verification skipped: {e})")


def run_interactive():
    """Interactive chat mode — enter any medical question."""

    model, tokenizer, device = load_medtrace_model()

    print("=" * 70)
    print("  MedTrace Interactive Mode")
    print("  Enter any medical question. Type 'quit' to exit.")
    print("=" * 70)

    while True:
        print()
        question = input("🩺 Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
        if not question:
            continue

        print("\n🔬 Generating reasoning chain...\n")
        response = generate_reasoning_chain(
            model, tokenizer, device,
            question=question,
        )
        print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive chat mode")
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    else:
        run_demo()
