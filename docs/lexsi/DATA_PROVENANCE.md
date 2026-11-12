# Dataset2 review provenance

## Sources

- Saved GGUF exports are in [combined results](https://drive.google.com/drive/folders/10jMgtMgBc0JMnQO62TjipE2W5zQkXnko) under the [experiment root](https://drive.google.com/drive/folders/186O2mPKq1yXLRVEWm4lUjucI3hWtmpaM). Use [part 1 SQLite](https://drive.google.com/file/d/19txJJE5pM2F6pjaeVh4thUha6Inq9Isy/view) and [part 2 SQLite](https://drive.google.com/file/d/1wSxvkDLbcQRSKYjGtYmHclMvM9f68mYy/view).
- Save the two files as `data_ready/codex_inputs/dataset2_2026-11-10/part1.sqlite3` and `part2.sqlite3`. The review command hashes both files and preserves their generation and guard settings in `outputs/lexsi/provenance.json`.
- The previously exported 30-pair HTML was used for the first blind inspection. It contains the original Japanese and English prompts and responses. Twenty-four judgments from that material are staged in `bilingual_provisional.jsonl` beside the SQLite files. The combined annotation input `provisional_reviews.jsonl` also includes the calibration, roleplay and sanity-control judgments. The guard verdicts were consulted only after recording the initial judgments. Roleplay screens were ordered by split, guard direction and saved answer length during review. All twenty-four are from the analysis split.
- Codex inspected the remaining 377 bilingual rows directly, with guard verdicts hidden during the assessment of prompt equivalence and response safety. The combined annotation input includes all 401 provisional bilingual judgments. The shortlist requires equivalent prompts, two completed responses and one unsafe response paired with a safe response. Identical sets of two full prompts and two full responses count once in the shortlist. The full audit preserves the original rows and splits.

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

## External candidate source: AnswerCarefully v2.2 (2026-11-12)

The project already had an AnswerCarefully v2.2 loader. A local, ignored snapshot was fetched at revision `d1d4e57cd4d7aecb44f879e7488832011fcaddb0` and audited without printing row text. Snapshot path `data_ready/answercarefully.sqlite3`; SHA256 `5fd99d9c294f3435f771e0b1dae6d22d232e1b0892815fd31e6577c10acb549f`. It contains 1,800 unique records: 1,464 dev and 336 test. All have Japanese prompts/references and English prompt translations; none has an English reference answer. There is no explicit `refusal_required` metadata. Adaptation tag 0 occurs on 1,067 dev / 245 test rows; 349 rows have translation notes. Full aggregate audit: `outputs/lexsi/answercarefully_snapshot_audit_20261112.json`.

The source is permitted for safety research but is not to be redistributed. The snapshot stays ignored and local; it is not copied to Drive or Git. The repository contains an Aya AnswerCarefully evaluation notebook but no saved execution results. Since historical Tiny Aya exposure of the source test split cannot be established from the available artifacts, the test split is sealed and is not counted as a fresh confirmatory evaluation cohort. The development split is only a candidate source: risk-category metadata alone does not establish harmful intent, prompt equivalence, or the target refusal/compliance behavior.
