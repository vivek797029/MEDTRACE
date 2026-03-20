"""
MedTrace Step 1: Download and Preview MedQA Dataset
====================================================
Downloads the USMLE medical exam dataset with expert explanations.
This is the foundation for building reasoning traces.

Run: python src/01_download_data.py
"""

from datasets import load_dataset
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def download_medqa():
    """Download MedQA USMLE dataset from HuggingFace."""
    print("📥 Downloading MedQA (USMLE) dataset...")
    print("   This contains ~12,000 medical exam questions with explanations.\n")

    # Using GBaker/MedQA-USMLE-4-options — standard Parquet format, no script needed
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")

    print(f"✅ Downloaded!")
    print(f"   Train: {len(dataset['train'])} examples")
    print(f"   Test:  {len(dataset['test'])} examples")

    # Preview first example
    example = dataset["train"][0]
    print(f"\n📋 Example preview:")
    print(f"   Question: {example['question'][:200]}...")
    print(f"   Answer:   {example['answer']}")
    print(f"   Options:  {list(example['options'].values())[:3]}...")

    # Save raw data — this dataset has train/test splits
    splits = {
        "train": dataset["train"],
        "test":  dataset["test"],
    }

    # Create a small validation split (last 500 of train)
    train_data = list(dataset["train"])
    val_data   = train_data[-500:]
    train_data = train_data[:-500]

    all_splits = {
        "train":      train_data,
        "validation": val_data,
        "test":       list(dataset["test"]),
    }

    for split_name, split_data in all_splits.items():
        path = os.path.join(DATA_DIR, f"medqa_raw_{split_name}.json")
        records = []
        for item in split_data:
            # Normalize options — could be dict or separate fields
            if isinstance(item.get("options"), dict):
                options = item["options"]
            else:
                options = {
                    "A": item.get("option_a", item.get("opa", "")),
                    "B": item.get("option_b", item.get("opb", "")),
                    "C": item.get("option_c", item.get("opc", "")),
                    "D": item.get("option_d", item.get("opd", "")),
                }
            records.append({
                "question":   item["question"],
                "answer":     item["answer"],
                "answer_idx": item.get("answer_idx", item.get("cop", "")),
                "options":    options,
                "meta_info":  item.get("meta_info", ""),
            })
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"💾 Saved {len(records)} examples → {path}")

    print("\n✅ Step 1 complete! Next: python src/02_build_reasoning_traces.py")


if __name__ == "__main__":
    download_medqa()
