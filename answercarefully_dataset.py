"""Load AnswerCarefully, select evaluation examples, or preview original records."""

import argparse
import json

import auditkit as ak


DATASET_ID = "llm-jp/AnswerCarefully"
DATASET_CONFIGURATION = "v2.2"


def load_answercarefully_data(split="all"):
    from datasets import concatenate_datasets, load_dataset
    from huggingface_hub import HfApi

    # Pin the snapshot so every row in one run comes from the same revision.
    revision = HfApi().dataset_info(DATASET_ID, token=True).sha
    # Use the saved Hugging Face login and download/reuse the cached snapshot.
    # Keep the original split on each row, even when evaluating both splits together.
    split_names = ("dev", "test") if split == "all" else (split,)
    datasets = []
    for split_name in split_names:
        dataset = load_dataset(
            DATASET_ID, DATASET_CONFIGURATION, split=split_name, revision=revision, token=True,
        )
        datasets.append(dataset.add_column("_source_split", [split_name] * len(dataset)))
    dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    return dataset, revision


def select_samples(dataset, revision, limit=None, *, language="ja"):
    """Select original Japanese questions or the dataset's English translations."""
    if language not in ("ja", "en"):
        raise ValueError("language must be 'ja' or 'en'")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    samples = []
    for row in dataset:
        # ACv2.2 stores English questions inside its multilingual metadata.
        prompt = row["text"] if language == "ja" else (row.get("meta-mlmc") or {}).get("text-English")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing/non-text {language} prompt for {row['ID']}")
        # input is the model prompt; target is a reference retained for inspection.
        # Neither target nor metadata is sent to Tiny Aya by our generation adapter.
        samples.append(ak.Sample(
            id=f"{row['ID']}:{language}", input=prompt, target=row["output"],
            metadata={
                # Preserve original annotations and revision rather than invent safety labels.
                "source_id": row["ID"], "language": language, "reference_language": "ja",
                "categories": row["meta"], "translation": row["meta-mlmc"],
                "dataset": DATASET_ID, "dataset_revision": revision,
                "configuration": DATASET_CONFIGURATION, "split": row["_source_split"],
            },
        ))
        if language == "en":
            # Preserve the paired Japanese question; the reference remains Japanese too.
            samples[-1].metadata["original_prompt_ja"] = row["text"]
        if len(samples) == limit:
            return samples
    if not samples:
        raise ValueError("The selected dataset split contains no examples")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10, help="Number of records to preview")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    print(f"Loading llm-jp/AnswerCarefully, configuration v2.2, split {args.split}...", flush=True)
    dataset, revision = load_answercarefully_data(args.split)
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
