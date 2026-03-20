# MedTrace 🩺

**A process-supervised medical reasoning model with auditable, step-verified diagnostic chains.**

> Most medical AI gives you an answer. MedTrace gives you a **proof.**

## What is this?

MedTrace is a fine-tuned language model that generates **structured, step-typed, dependency-mapped clinical reasoning chains**. Every diagnostic step can be independently verified by a doctor, auditor, or another AI system.

Instead of: *"The patient has hypothyroidism."*

MedTrace produces:
```
[STEP 1 | type: symptom | source: patient_data]
Patient presents with fatigue, weight gain, cold intolerance, constipation.

[STEP 2 | type: finding | source: patient_data]
TSH is 12 mIU/L (elevated above normal range 0.5-4.5).

[STEP 3 | type: mechanism | source: medical_knowledge | depends_on: 1,2]
Elevated TSH with these symptoms indicates primary hypothyroidism —
the thyroid gland is underproducing hormones, triggering pituitary TSH release.

[STEP 4 | type: rule | source: clinical_guideline | depends_on: 2,3]
ATA guidelines recommend levothyroxine replacement for overt hypothyroidism
with TSH > 10 mIU/L.

[STEP 5 | type: conclusion | source: logical_deduction | depends_on: 1,2,3,4]
Diagnosis: Primary hypothyroidism. Treatment: Levothyroxine.
```

A formal **verifier** independently checks every chain for structural validity, dependency integrity, circular references, type consistency, and clinical reasoning order.

## Why does this matter?

Medical AI is powerful but **untrustworthy** because it can't show its work. Doctors can't defend AI-assisted decisions in court. Regulators can't audit them. MedTrace solves this by making every reasoning step typed, sourced, and verifiable.

## Quick Start (Mac M2)

```bash
# 1. Setup
chmod +x setup.sh && ./setup.sh
source medtrace_env/bin/activate

# 2. Download medical data
python src/01_download_data.py

# 3. Build reasoning traces
python src/02_build_reasoning_traces.py

# 4. Quick test (5 min) — verify everything works
python src/03_train.py --quick_test

# 5. Full training (run overnight in tmux)
tmux new -s medtrace
python src/03_train.py

# 6. Test the verifier
python src/04_verify_and_evaluate.py

# 7. Run inference
python src/05_inference.py --interactive
```

## Architecture

```
Patient Data → TinyLlama 1.1B (LoRA) → Structured Reasoning Chain → Formal Verifier
                                              ↓                          ↓
                                    [typed steps with               [validity check,
                                     dependencies and                score, errors,
                                     source citations]               warnings]
```

- **Base model**: TinyLlama 1.1B Chat
- **Fine-tuning**: LoRA (r=16, alpha=32) — fits in 8GB unified memory
- **Training data**: MedQA (USMLE) — 12,000 medical exam questions
- **Hardware**: Apple Silicon M2 with MPS acceleration
- **Training time**: ~24-48 hours

## Step Types

| Type | Description | Example |
|------|-------------|---------|
| `symptom` | Patient-reported complaint | "Chest pain radiating to left arm" |
| `finding` | Lab/imaging/exam result | "Troponin I: 2.5 ng/mL (elevated)" |
| `mechanism` | Pathophysiology | "Thrombotic occlusion of coronary artery" |
| `rule` | Clinical guideline | "ACC/AHA: emergent PCI for STEMI" |
| `inference` | Logical deduction | "ST elevation + troponin = acute MI" |
| `conclusion` | Final recommendation | "Activate cath lab immediately" |

## Verification Checks

The formal verifier runs 7 independent checks:

1. **Structural validity** — all required fields present
2. **Type consistency** — valid step types and source types
3. **Dependency integrity** — all references point to real earlier steps
4. **Circular dependency detection** — no logical loops
5. **Completeness** — chain starts with evidence, ends with conclusion
6. **Source appropriateness** — conclusions require logical deduction
7. **Clinical reasoning order** — types follow expected diagnostic flow

## License

MIT — use this for research, learning, and building better medical AI.

## Author

Built by VN as a proof-of-concept for process-supervised medical reasoning.
