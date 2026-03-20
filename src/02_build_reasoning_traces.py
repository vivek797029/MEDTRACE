"""
MedTrace Step 2: Build Structured Reasoning Traces
====================================================
Converts raw MedQA questions into step-typed, dependency-mapped
reasoning chains. This is the CORE of MedTrace.

Each reasoning step has:
  - step_id: unique identifier
  - claim: what the step asserts
  - type: one of [symptom, finding, mechanism, rule, inference, conclusion]
  - source_type: one of [patient_data, medical_knowledge, clinical_guideline, logical_deduction]
  - depends_on: list of step_ids this step relies on

Run: python src/02_build_reasoning_traces.py
"""

import json
import os
import re
from typing import List, Dict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# ============================================================
# REASONING STEP TYPES
# ============================================================
# These are the categories that make your model AUDITABLE.
# A doctor can look at any step and verify it independently.

STEP_TYPES = {
    "symptom": "Observable patient symptom or complaint",
    "finding": "Lab result, imaging, or examination finding",
    "mechanism": "Pathophysiological mechanism or process",
    "rule": "Established medical rule, guideline, or criteria",
    "inference": "Logical deduction from previous steps",
    "conclusion": "Final diagnostic or treatment recommendation",
}

SOURCE_TYPES = {
    "patient_data": "Information from the patient case",
    "medical_knowledge": "General medical/scientific knowledge",
    "clinical_guideline": "Established clinical guidelines (ACC/AHA/WHO etc.)",
    "logical_deduction": "Derived from combining previous steps",
}


# ============================================================
# REASONING TRACE TEMPLATES
# ============================================================
# These templates teach the model HOW to structure clinical reasoning.
# The model learns to generate this format natively.

TRACE_PROMPT_TEMPLATE = """You are a clinical reasoning system. Given a medical question,
produce a structured reasoning chain where EVERY step is independently verifiable.

Question: {question}

Options:
{options}

Generate a reasoning chain in this EXACT format:

REASONING_CHAIN:
[STEP 1 | type: symptom/finding | source: patient_data]
<extract key clinical information from the question>
[STEP 2 | type: mechanism | source: medical_knowledge]
<explain the relevant pathophysiology>
[STEP 3 | type: rule | source: clinical_guideline | depends_on: 1,2]
<cite the relevant clinical rule or guideline>
[STEP 4 | type: inference | source: logical_deduction | depends_on: 1,2,3]
<combine previous steps to narrow down>
[STEP 5 | type: conclusion | source: logical_deduction | depends_on: 1,2,3,4]
<state final answer with justification>

ANSWER: <correct option letter>"""


def build_training_format(question: str, options: dict, answer: str, answer_idx: str) -> dict:
    """
    Build the training input/output pair in the structured reasoning format.

    This is what the model learns to GENERATE. Each example teaches it
    to produce step-typed, dependency-mapped reasoning.
    """

    # Format options
    options_str = "\n".join([f"  {k}. {v}" for k, v in options.items()])

    # Build the input prompt
    input_text = TRACE_PROMPT_TEMPLATE.format(
        question=question,
        options=options_str,
    )

    return {
        "input": input_text,
        "answer": answer,
        "answer_idx": answer_idx,
        "options": options,
        "question": question,
    }


def parse_reasoning_chain(raw_text: str) -> List[Dict]:
    """
    Parse a reasoning chain string into structured steps.
    Used during evaluation to verify model outputs.
    """
    steps = []
    pattern = r'\[STEP (\d+) \| type: (\w+)(?:/\w+)? \| source: (\w+)(?:\s*\|\s*depends_on:\s*([\d,\s]+))?\]\s*\n(.+?)(?=\[STEP|\nANSWER:|$)'

    for match in re.finditer(pattern, raw_text, re.DOTALL):
        step_id = int(match.group(1))
        step_type = match.group(2)
        source = match.group(3)
        depends_raw = match.group(4)
        claim = match.group(5).strip()

        depends_on = []
        if depends_raw:
            depends_on = [int(d.strip()) for d in depends_raw.split(",") if d.strip()]

        steps.append({
            "step_id": step_id,
            "type": step_type,
            "source_type": source,
            "depends_on": depends_on,
            "claim": claim,
        })

    return steps


def build_all_traces():
    """Convert entire MedQA dataset into structured reasoning format."""

    print("🔬 Building structured reasoning traces from MedQA...")
    print("   This converts raw Q&A into step-typed, auditable reasoning chains.\n")

    for split in ["train", "validation", "test"]:
        input_path = os.path.join(DATA_DIR, f"medqa_raw_{split}.json")

        if not os.path.exists(input_path):
            print(f"⚠️  {input_path} not found. Run 01_download_data.py first.")
            continue

        with open(input_path) as f:
            raw_data = json.load(f)

        traces = []
        for item in raw_data:
            trace = build_training_format(
                question=item["question"],
                options=item["options"],
                answer=item["answer"],
                answer_idx=item["answer_idx"],
            )
            traces.append(trace)

        output_path = os.path.join(DATA_DIR, f"medtrace_{split}.json")
        with open(output_path, "w") as f:
            json.dump(traces, f, indent=2)

        print(f"✅ {split}: {len(traces)} reasoning trace templates saved to {output_path}")

    # Also create a small sample for quick testing
    sample_path = os.path.join(DATA_DIR, "medtrace_sample_5.json")
    with open(os.path.join(DATA_DIR, "medtrace_train.json")) as f:
        full = json.load(f)
    with open(sample_path, "w") as f:
        json.dump(full[:5], f, indent=2)
    print(f"\n📋 Sample of 5 examples saved to {sample_path} (for quick testing)")

    # Print one example
    print("\n" + "=" * 70)
    print("EXAMPLE TRAINING INPUT:")
    print("=" * 70)
    print(full[0]["input"][:800])
    print("...")
    print("=" * 70)

    print("\n✅ Step 2 complete! Next: python src/03_train.py")


if __name__ == "__main__":
    build_all_traces()
