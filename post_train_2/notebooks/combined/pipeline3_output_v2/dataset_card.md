# Dataset Card: pipeline3_culture_us_safety_pairs_v2

## Summary

- **Run timestamp**: 2026-11-12T06:14:13.158035Z
- **Wall clock time**: 415.0s
- **Pipeline config hash**: `3d158ec14f4de71c`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONReader | 135 | 135 | — | 0 | 135 |
| SchemaGate | 135 | 135 | 0 | 0 | 135 |
| ExactDeduplicator | 135 | 133 | — | — | 133 |
| TextCleaner | 133 | 133 | — | — | 133 |
| PreferenceGenerationTask | 133 | 133 | — | 0 | 133 |
| MaxSamplesTruncator | 133 | 133 | — | — | 133 |

## Gate Pass Rates and Rejection Breakdown

- **JSONReader**: 135/0 passed (N/A), 0 rejected
- **SchemaGate**: 135/135 passed (100.0%), 0 rejected
- **PreferenceGenerationTask**: 133/133 passed (100.0%), 0 rejected


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
