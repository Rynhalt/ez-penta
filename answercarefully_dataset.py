"""Load AnswerCarefully, select a small pilot, or preview original records."""

import argparse
import json

import auditkit as ak


DATASET_ID = "llm-jp/AnswerCarefully"
DATASET_CONFIGURATION = "v2.2"


def load_development_data():
    from datasets import load_dataset
    from huggingface_hub import HfApi

    # Pin the snapshot so every row in one run comes from the same revision.
    revision = HfApi().dataset_info(DATASET_ID, token=True).sha
    # Use the saved Hugging Face login and download/reuse the cached snapshot.
    # Only development rows are returned; the held-out test split is not evaluated.
    dataset = load_dataset(
        DATASET_ID, DATASET_CONFIGURATION, split="dev", revision=revision, token=True,
    )
    return dataset, revision


def select_samples(dataset, revision, limit):
    """Take one original Japanese request from each distinct risk area."""
    samples = []
    seen_areas = set()
    for row in dataset:
        area = row["meta"]["risk-area"]
        # Rows are grouped by category; taking the first N would cover less variety.
        if area in seen_areas:
            continue
        seen_areas.add(area)
        # input is the model prompt; target is a reference retained for inspection.
        # Neither target nor metadata is sent to Tiny Aya by our generation adapter.
        samples.append(ak.Sample(
            id=f"{row['ID']}:ja", input=row["text"], target=row["output"],
            metadata={
                # Preserve original annotations and revision rather than invent safety labels.
                "source_id": row["ID"], "language": "ja", "reference_language": "ja",
                "categories": row["meta"], "translation": row["meta-mlmc"],
                "dataset": DATASET_ID, "dataset_revision": revision,
                "configuration": DATASET_CONFIGURATION, "split": "dev",
            },
        ))
        if len(samples) == limit:
            return samples
    raise ValueError(f"Requested {limit} risk areas; found only {len(samples)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10, help="Number of records to preview")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    print("Loading llm-jp/AnswerCarefully, configuration v2.2, split dev...", flush=True)
    dataset, revision = load_development_data()
    print(f"Loaded successfully: {len(dataset)} records")
    print(f"Columns: {dataset.column_names}")
    print(f"Dataset revision: {revision}")
    preview_count = min(args.limit, len(dataset))
    print(f"\nFirst {preview_count} records (original dataset order):")
    for index, record in enumerate(dataset.select(range(preview_count)), start=1):
        print(f"\n--- Record {index} ---")
        print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
