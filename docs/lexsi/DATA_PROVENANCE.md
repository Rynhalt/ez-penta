# Dataset2 review provenance

## Sources

- Saved GGUF exports are in [combined results](https://drive.google.com/drive/folders/10jMgtMgBc0JMnQO62TjipE2W5zQkXnko) under the [experiment root](https://drive.google.com/drive/folders/186O2mPKq1yXLRVEWm4lUjucI3hWtmpaM). Use [part 1 SQLite](https://drive.google.com/file/d/19txJJE5pM2F6pjaeVh4thUha6Inq9Isy/view) and [part 2 SQLite](https://drive.google.com/file/d/1wSxvkDLbcQRSKYjGtYmHclMvM9f68mYy/view).
- Save the two files as `data_ready/codex_inputs/dataset2_2026-11-10/part1.sqlite3` and `part2.sqlite3`. The review command hashes both files and preserves their generation and guard settings in `outputs/lexsi/provenance.json`.
- The previously exported 30-pair HTML was used for the first blind inspection. It contains the original Japanese and English prompts and responses. Twenty-four judgments from that material are staged in `bilingual_provisional.jsonl` beside the SQLite files. The combined annotation input `provisional_reviews.jsonl` also includes the calibration, roleplay and sanity-control judgments. The guard verdicts were consulted only after recording the initial judgments. Roleplay screens were ordered by split, guard direction and saved answer length during review. All twenty-four are from the analysis split.

## Runtime and limitations

The saved Aya runs used CohereLabs Tiny Aya Water GGUF Q4_K_M via llama.cpp with its embedded tokenizer and chat template. The guard was Qwen3Guard-Gen-0.6B GGUF Q8_0. Aya decoding used a 1,024-token cap and the guard cap was 128. Exact source revisions, hashes, template configuration and runtime settings are available in the shard metadata. The tracked `eval/eval_dataset2.ipynb` uses Transformers. GGUF execution is documented in `eval/DATASET2_HANDOFF.md` and `eval/DATASET2_RUN.md`.

The original publication excluded responses when parsing the prompt verdict raised an error. The recovery script reads the two saved moderation records independently in the documented response-then-prompt order. Missing, ambiguous and Controversial response verdicts remain outside binary denominators. Full saved text is retained, including token-cap output where relevant to provenance. The CircuitKIT screen requires both answers to have completed. The dataset contains duplicate harmful/twin requests, potentially invalid twins and variants whose framing or translation changes the requested action.

The prepared CSV bundle was unavailable. The review matrices are reconstructed from the saved GGUF exports. Calibration is sampled deterministically within categories, with 40 Safe/Safe and 40 Unsafe/Unsafe analysis pairs. The original sample membership and the recomputed-comparisons CSV require the prepared bundle for a row-level comparison.

## Reproduce

With the repository's Python 3.11+ environment and AuditKIT installed, run from the repository root.

```sh
LEXSI_INPUT=data_ready/codex_inputs/dataset2_2026-11-10
.venv/bin/python -m scripts.lexsi.review_dataset2 \
  --part1 "$LEXSI_INPUT/part1.sqlite3" --part2 "$LEXSI_INPUT/part2.sqlite3" \
  --provisional "$LEXSI_INPUT/provisional_reviews.jsonl"
LEXSI_DATASET2_INPUT="$LEXSI_INPUT" .venv/bin/python -m unittest discover -s tests -p test_lexsi_dataset2_review.py -v
```

Omit `--provisional` to regenerate empty review fields. Add `--human-approvals PATH` only for adjudicated human sign-off records. See the [annotation guide](ANNOTATION_GUIDE.md).

The calibration judgments cover the ten shortest complete response pairs in the reconstructed cohort. This convenience subset does not estimate guard accuracy.
