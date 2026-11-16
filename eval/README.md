# Evaluation and analysis

- `eval_jp_en_variants.ipynb`: AnswerCarefully prompt-variant evaluation.
- `eval_dataset2_uk.ipynb` and `uk_hf_evaluation.py`: UK/neutral HF baseline and saved-adapter evaluation; see [UK instructions](../data_ready/DATASET2_UK.md).
- `dataset2_data.py`, `dataset2_runtime.py`, `dataset2_analysis.py`: shared loading, inference/checkpoint handling and matched analysis.
- `track2_extension/`: CircuitKIT preparation/integration; follow [roadmap gates](../docs/lexsi/CIRCUITKIT_ROADMAP.md).

The root `guard_judge.py` supplies the AuditKit/Qwen3Guard wrapper. Other experiments can invoke Qwen3Guard directly; the [article](../reports/experiment_report.md) documents the distinction. Input and checkpoint paths are retained to keep the existing Colab workflow compatible.

`review_ui/` contains the HTML/CSS/JavaScript assets served by the root `eval/review_results.py` review application. The server path was updated when the UI moved here.

AnswerCarefully commands, from the repository root:

```bash
python -m eval.evaluate_answercarefully run --limit 3
python -m eval.inspect_evaluation
python -m eval.review_results
```

These modules generate/judge, inspect saved outcomes, and serve human review respectively. Dataset selection is `python -m data_ready.answercarefully_dataset --limit 10`.
