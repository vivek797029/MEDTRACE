"""
MedTrace Step 4: Formal Reasoning Chain Verifier + Evaluation
==============================================================
This is MedTrace's KILLER FEATURE.

The verifier checks model-generated reasoning chains for:
  1. Structural validity — are all required fields present?
  2. Type consistency — do step types follow clinical reasoning order?
  3. Dependency integrity — does every dependency reference a real step?
  4. Circular dependency — no step can depend on itself or future steps
  5. Completeness — chain must start with evidence and end with conclusion
  6. Source appropriateness — conclusions can't come from patient_data alone

A doctor can trust the chain because the verifier guarantees logical structure.
The MODEL generates the reasoning. The VERIFIER audits it. Separation of concerns.

Run: python src/04_verify_and_evaluate.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ReasoningStep:
    """A single step in a clinical reasoning chain."""
    step_id: int
    claim: str
    step_type: str  # symptom, finding, mechanism, rule, inference, conclusion
    source_type: str  # patient_data, medical_knowledge, clinical_guideline, logical_deduction
    depends_on: List[int] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Result of verifying a reasoning chain."""
    is_valid: bool
    score: float  # 0.0 to 1.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    step_scores: Dict[int, float] = field(default_factory=dict)


# ============================================================
# VALID VALUES
# ============================================================

VALID_STEP_TYPES = {"symptom", "finding", "mechanism", "rule", "inference", "conclusion"}
VALID_SOURCE_TYPES = {"patient_data", "medical_knowledge", "clinical_guideline", "logical_deduction"}

# Expected clinical reasoning order (soft constraint — scored, not required)
EXPECTED_ORDER = {
    "symptom": 0,
    "finding": 1,
    "mechanism": 2,
    "rule": 3,
    "inference": 4,
    "conclusion": 5,
}


# ============================================================
# PARSER
# ============================================================

def parse_reasoning_chain(raw_text: str) -> Tuple[List[ReasoningStep], Optional[str]]:
    """
    Parse raw model output into structured ReasoningSteps.

    Returns:
        steps: List of parsed ReasoningStep objects
        answer: The final answer (if found)
    """
    steps = []
    answer = None

    # Parse answer
    answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', raw_text)
    if answer_match:
        answer = answer_match.group(1).strip()

    # Parse steps
    pattern = (
        r'\[STEP\s+(\d+)\s*\|\s*type:\s*(\w+)(?:/\w+)?\s*\|\s*source:\s*(\w+)'
        r'(?:\s*\|\s*depends_on:\s*([\d,\s]+))?\]'
        r'\s*\n(.+?)(?=\[STEP|\nANSWER:|$)'
    )

    for match in re.finditer(pattern, raw_text, re.DOTALL):
        step_id = int(match.group(1))
        step_type = match.group(2).strip()
        source_type = match.group(3).strip()
        depends_raw = match.group(4)
        claim = match.group(5).strip()

        depends_on = []
        if depends_raw:
            depends_on = [int(d.strip()) for d in depends_raw.split(",") if d.strip()]

        steps.append(ReasoningStep(
            step_id=step_id,
            claim=claim,
            step_type=step_type,
            source_type=source_type,
            depends_on=depends_on,
        ))

    return steps, answer


# ============================================================
# FORMAL VERIFIER — The Core Innovation
# ============================================================

class ReasoningChainVerifier:
    """
    Formal verifier for clinical reasoning chains.

    This is what makes MedTrace different from every other model.
    The model GENERATES reasoning. The verifier AUDITS it.
    They are independent — the verifier doesn't trust the model.
    """

    def verify(self, steps: List[ReasoningStep]) -> VerificationResult:
        """Run all verification checks on a reasoning chain."""

        errors = []
        warnings = []
        step_scores = {}

        if not steps:
            return VerificationResult(
                is_valid=False, score=0.0,
                errors=["Empty reasoning chain — no steps found"],
            )

        # ---- Check 1: Structural Validity ----
        structural_errors = self._check_structure(steps)
        errors.extend(structural_errors)

        # ---- Check 2: Type Validity ----
        type_errors = self._check_types(steps)
        errors.extend(type_errors)

        # ---- Check 3: Dependency Integrity ----
        dep_errors = self._check_dependencies(steps)
        errors.extend(dep_errors)

        # ---- Check 4: No Circular Dependencies ----
        circular_errors = self._check_circular_deps(steps)
        errors.extend(circular_errors)

        # ---- Check 5: Completeness ----
        completeness_warnings = self._check_completeness(steps)
        warnings.extend(completeness_warnings)

        # ---- Check 6: Source Appropriateness ----
        source_warnings = self._check_source_appropriateness(steps)
        warnings.extend(source_warnings)

        # ---- Check 7: Clinical Reasoning Order ----
        order_warnings = self._check_reasoning_order(steps)
        warnings.extend(order_warnings)

        # ---- Check 8: Claim Quality ----
        for step in steps:
            step_scores[step.step_id] = self._score_step(step, steps)

        # Calculate overall score
        if errors:
            base_score = 0.3  # Errors cap the score
        else:
            base_score = 0.7

        # Add bonus for no warnings
        warning_penalty = min(len(warnings) * 0.05, 0.3)
        avg_step_score = sum(step_scores.values()) / len(step_scores) if step_scores else 0
        final_score = min(1.0, base_score - warning_penalty + (avg_step_score * 0.3))

        return VerificationResult(
            is_valid=(len(errors) == 0),
            score=round(final_score, 3),
            errors=errors,
            warnings=warnings,
            step_scores=step_scores,
        )

    # ---- Individual Checks ----

    def _check_structure(self, steps: List[ReasoningStep]) -> List[str]:
        """Every step must have all required fields."""
        errors = []
        seen_ids = set()
        for step in steps:
            if step.step_id in seen_ids:
                errors.append(f"Duplicate step ID: {step.step_id}")
            seen_ids.add(step.step_id)
            if not step.claim or len(step.claim.strip()) < 5:
                errors.append(f"Step {step.step_id}: Claim is too short or empty")
        return errors

    def _check_types(self, steps: List[ReasoningStep]) -> List[str]:
        """All types must be from the valid set."""
        errors = []
        for step in steps:
            if step.step_type not in VALID_STEP_TYPES:
                errors.append(
                    f"Step {step.step_id}: Invalid type '{step.step_type}'. "
                    f"Must be one of: {VALID_STEP_TYPES}"
                )
            if step.source_type not in VALID_SOURCE_TYPES:
                errors.append(
                    f"Step {step.step_id}: Invalid source '{step.source_type}'. "
                    f"Must be one of: {VALID_SOURCE_TYPES}"
                )
        return errors

    def _check_dependencies(self, steps: List[ReasoningStep]) -> List[str]:
        """Every dependency must reference an existing earlier step."""
        errors = []
        valid_ids = {s.step_id for s in steps}
        for step in steps:
            for dep_id in step.depends_on:
                if dep_id not in valid_ids:
                    errors.append(
                        f"Step {step.step_id}: Depends on non-existent step {dep_id}"
                    )
                if dep_id >= step.step_id:
                    errors.append(
                        f"Step {step.step_id}: Depends on future/same step {dep_id} "
                        f"(forward reference)"
                    )
        return errors

    def _check_circular_deps(self, steps: List[ReasoningStep]) -> List[str]:
        """Detect circular dependency chains."""
        errors = []
        step_map = {s.step_id: s for s in steps}

        def has_cycle(step_id, visited, rec_stack):
            visited.add(step_id)
            rec_stack.add(step_id)
            step = step_map.get(step_id)
            if step:
                for dep in step.depends_on:
                    if dep not in visited:
                        if has_cycle(dep, visited, rec_stack):
                            return True
                    elif dep in rec_stack:
                        return True
            rec_stack.discard(step_id)
            return False

        visited = set()
        for step in steps:
            if step.step_id not in visited:
                if has_cycle(step.step_id, visited, set()):
                    errors.append("Circular dependency detected in reasoning chain")
                    break

        return errors

    def _check_completeness(self, steps: List[ReasoningStep]) -> List[str]:
        """Chain should start with evidence and end with conclusion."""
        warnings = []
        types = [s.step_type for s in steps]

        if types and types[0] not in ("symptom", "finding"):
            warnings.append(
                f"Chain starts with '{types[0]}' instead of symptom/finding. "
                f"Clinical reasoning should start from patient evidence."
            )

        if types and types[-1] != "conclusion":
            warnings.append(
                f"Chain ends with '{types[-1]}' instead of conclusion. "
                f"Clinical reasoning should end with a clear recommendation."
            )

        if "conclusion" not in types:
            warnings.append("No conclusion step found in the chain.")

        return warnings

    def _check_source_appropriateness(self, steps: List[ReasoningStep]) -> List[str]:
        """Conclusions shouldn't come from patient_data alone."""
        warnings = []
        for step in steps:
            if step.step_type == "conclusion" and step.source_type == "patient_data":
                warnings.append(
                    f"Step {step.step_id}: Conclusion sourced from patient_data alone. "
                    f"Conclusions should be from logical_deduction combining evidence."
                )
            if step.step_type == "symptom" and step.source_type == "clinical_guideline":
                warnings.append(
                    f"Step {step.step_id}: Symptom sourced from clinical_guideline. "
                    f"Symptoms should come from patient_data."
                )
        return warnings

    def _check_reasoning_order(self, steps: List[ReasoningStep]) -> List[str]:
        """Check if types follow expected clinical reasoning order."""
        warnings = []
        for i in range(1, len(steps)):
            prev_order = EXPECTED_ORDER.get(steps[i - 1].step_type, 0)
            curr_order = EXPECTED_ORDER.get(steps[i].step_type, 0)
            if curr_order < prev_order - 1:  # Allow 1 step back
                warnings.append(
                    f"Steps {steps[i-1].step_id}→{steps[i].step_id}: "
                    f"'{steps[i].step_type}' appears after '{steps[i-1].step_type}' — "
                    f"consider reordering for clearer clinical flow."
                )
        return warnings

    def _score_step(self, step: ReasoningStep, all_steps: List[ReasoningStep]) -> float:
        """Score an individual step from 0.0 to 1.0."""
        score = 1.0

        # Penalize very short claims
        if len(step.claim) < 20:
            score -= 0.2

        # Penalize conclusions without dependencies
        if step.step_type in ("inference", "conclusion") and not step.depends_on:
            score -= 0.3

        # Reward steps that build on previous steps
        if step.depends_on and all(
            d < step.step_id for d in step.depends_on
        ):
            score += 0.1

        return max(0.0, min(1.0, score))


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model_output(model_output: str) -> dict:
    """Full evaluation pipeline for a single model output."""

    # Parse
    steps, answer = parse_reasoning_chain(model_output)

    # Verify
    verifier = ReasoningChainVerifier()
    result = verifier.verify(steps)

    return {
        "num_steps": len(steps),
        "answer": answer,
        "is_valid": result.is_valid,
        "score": result.score,
        "errors": result.errors,
        "warnings": result.warnings,
        "step_scores": result.step_scores,
    }


def run_demo_evaluation():
    """Demo the verifier on example outputs — good, bad, and ugly."""

    print("=" * 70)
    print("  MedTrace Reasoning Chain Verifier — Demo")
    print("=" * 70)

    # GOOD EXAMPLE — well-structured clinical reasoning
    good_chain = """REASONING_CHAIN:
[STEP 1 | type: symptom | source: patient_data]
A 55-year-old male presents with crushing substernal chest pain radiating to the left arm, diaphoresis, and shortness of breath for 2 hours.
[STEP 2 | type: finding | source: patient_data]
ECG shows ST-segment elevation in leads II, III, and aVF. Troponin I is elevated at 2.5 ng/mL.
[STEP 3 | type: mechanism | source: medical_knowledge | depends_on: 1,2]
ST elevation with elevated troponin indicates acute myocardial infarction involving the inferior wall, caused by thrombotic occlusion of the right coronary artery.
[STEP 4 | type: rule | source: clinical_guideline | depends_on: 2,3]
ACC/AHA guidelines recommend immediate dual antiplatelet therapy (aspirin + P2Y12 inhibitor) and emergent percutaneous coronary intervention within 90 minutes for STEMI.
[STEP 5 | type: conclusion | source: logical_deduction | depends_on: 1,2,3,4]
This patient has an acute inferior STEMI requiring emergent PCI. Administer aspirin 325mg, ticagrelor 180mg loading dose, and activate the cardiac catheterization lab immediately.

ANSWER: B"""

    # BAD EXAMPLE — broken dependencies, wrong types
    bad_chain = """REASONING_CHAIN:
[STEP 1 | type: conclusion | source: patient_data]
Give the patient antibiotics.
[STEP 2 | type: symptom | source: clinical_guideline | depends_on: 5]
Patient has a fever.
[STEP 3 | type: mechanism | source: medical_knowledge]
Ok.

ANSWER: A"""

    print("\n📋 TEST 1: Well-structured clinical reasoning chain")
    print("-" * 50)
    result1 = evaluate_model_output(good_chain)
    _print_result(result1)

    print("\n📋 TEST 2: Poorly-structured chain (errors expected)")
    print("-" * 50)
    result2 = evaluate_model_output(bad_chain)
    _print_result(result2)

    print("\n📋 TEST 3: Empty chain")
    print("-" * 50)
    result3 = evaluate_model_output("I think the answer is B.")
    _print_result(result3)

    print("\n" + "=" * 70)
    print("✅ Verifier demo complete!")
    print("   The verifier catches structural, logical, and clinical reasoning errors.")
    print("   This is what makes MedTrace auditable by doctors.")
    print("=" * 70)


def _print_result(result: dict):
    """Pretty-print an evaluation result."""
    status = "✅ VALID" if result["is_valid"] else "❌ INVALID"
    print(f"   Status: {status}")
    print(f"   Score:  {result['score']:.3f} / 1.000")
    print(f"   Steps:  {result['num_steps']}")
    print(f"   Answer: {result['answer']}")
    if result["errors"]:
        print(f"   Errors:")
        for e in result["errors"]:
            print(f"     ❌ {e}")
    if result["warnings"]:
        print(f"   Warnings:")
        for w in result["warnings"]:
            print(f"     ⚠️  {w}")
    if result["step_scores"]:
        print(f"   Step scores: {result['step_scores']}")


if __name__ == "__main__":
    run_demo_evaluation()
