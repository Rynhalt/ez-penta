---
tags:
- curatorkit
---

# Dataset Card: pipeline3_culture_us_helpful_pairs_v2

## Summary

- **Run timestamp**: 2026-11-12T06:19:37.265647Z
- **Wall clock time**: 324.1s
- **Pipeline config hash**: `18b6ef6455d28528`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONReader | 101 | 101 | — | 0 | 101 |
| SchemaGate | 101 | 101 | 0 | 0 | 101 |
| ExactDeduplicator | 101 | 99 | — | — | 99 |
| TextCleaner | 99 | 99 | — | — | 99 |
| PreferenceGenerationTask | 99 | 99 | — | 0 | 99 |
| MaxSamplesTruncator | 99 | 99 | — | — | 99 |

## Gate Pass Rates and Rejection Breakdown

- **JSONReader**: 101/0 passed (N/A), 0 rejected
- **SchemaGate**: 101/101 passed (100.0%), 0 rejected
- **PreferenceGenerationTask**: 99/99 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - exact_duplicates_removed: 2

## Known Limitations

- No MinHash deduplication applied.
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.24
- **python**: 3.12.16
