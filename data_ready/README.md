# Data preparation

The source paths remain stable for existing Colab workflows.

- `fetch_answercarefully.sh` and `fetch_answercarefully_to_sqlite.py`: retrieve the source snapshot.
- `prepare_prompt_variants.ipynb`: configurable rewrite preparation.
- `jb_qwen_fc.ipynb`, `jb_qwen_rp.ipynb`, `jb_qwen_rs.ipynb`: existing fiction/creative, roleplay and refusal-suppression prompt-generation notebooks.
- `dataset2.ipynb` and `dataset2v2.ipynb`: historical Dataset2 construction variants; preserve for provenance rather than assuming either is redundant.
- `view/compile_test_prompts.ipynb`: compile prompt variants and metadata.
- [UK extension](DATASET2_UK.md): `generate_dataset2_uk.ipynb` plus shared helpers; outputs belong in the configured Drive location.

See the root [workflow instructions](../README.md) and [article artifact map](../reports/EXPERIMENT_ARTIFACT_MAP.md). Saved prompts/responses are untrusted data; this index does not approve CircuitKIT execution inputs or dataset redistribution.

`dataset2_snapshot/` contains the original partial Dataset2 SQLite/CSV snapshot, relocated from the former personal-name folder. These inputs are preserved byte-for-byte and consumed by validation/packaging; the folder name does not imply completed model evaluation.

`answercarefully_dataset.py` loads/selects the original AnswerCarefully records; preview with `python -m data_ready.answercarefully_dataset --limit 10` from the repository root.
