# Primary sources, implementation references, and research decisions

**Verified / accessed around 2026-11-11.** Source links below should be resolved by Codex from the actual pinned repository revision before implementation. The upstream project can change, so link presence does not replace source-code or real-runtime checks.

## User project: actual artifacts inspected

- `Rynhalt/ez-penta` repository: https://github.com/Rynhalt/ez-penta
- Dataset2 methodology: https://github.com/Rynhalt/ez-penta/blob/main/data_ready/dataset_2.md
- Tracked evaluator (HF Transformers; not the executed GGUF shards): https://github.com/Rynhalt/ez-penta/blob/main/eval/eval_dataset2.ipynb
- Executed five-pipeline Drive folder: https://drive.google.com/drive/folders/1Bu_eGYuGyMWhzHZipbHJHC8W2qcKXvuE
- Five-pipeline results `combined` folder: https://drive.google.com/drive/folders/1ZB8Rgb0JxyGFCJ6KEJx8VfR2CC_kxSDu
- Original 21K Dataset2 GGUF combined data: https://drive.google.com/drive/folders/10jMgtMgBc0JMnQO62TjipE2W5zQkXnko
- GGUF part1/part2: https://drive.google.com/drive/folders/186O2mPKq1yXLRVEWm4lUjucI3hWtmpaM

Actual results inspected (not inferred): Pipeline1 US vs neutral OR 1.9822024722961555 (reported CI 1.5831753321715596–2.481801327581078, p 2.2075405470463405e-09); Japan vs neutral OR 1.0974400938756737 (p .48746508677598244); Pipeline4 chosen-only 22 SFT rows, restricted layers 18+23, broad comparison trained; Pipeline5 reported NaN MH OR for both baseline and restricted, 40 prompts. Its notebook uses 192 max new tokens and max 1,024 input tokens; GGUF original used output cap 1,024 and context 8,192. `Pipeline4` saves merged checkpoints under `/content` and results JSON does not contain weight bytes. Case-specific assumptions must be rechecked after source retrieval.

## Model and safety judge

1. Tiny Aya Water model card, 3.35B, Cohere2, 8K context, CC BY-NC: https://huggingface.co/CohereLabs/tiny-aya-water
2. Qwen3Guard-Gen-0.6B model card, safe/controversial/unsafe and refusal judgments: https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B
3. AuditKIT repository/version (prior user implementation used 1.1.3): https://github.com/Lexsi-Labs/AuditKIT
4. PEFT quantized fine-tuning (NF4+LoRA): https://huggingface.co/docs/peft/main/developer_guides/quantization
5. PEFT checkpoint semantics and why adapter checkpoint is distinct from merged base: https://huggingface.co/docs/peft/main/developer_guides/checkpoint
6. PEFT loading adapters: https://huggingface.co/docs/transformers/peft

## CircuitKIT: MUST inspect these exact sources

- `CircuitKIT` repo: https://github.com/Lexsi-Labs/CircuitKIT
- **Tiny Aya / Cohere2 compatibility port + parity gate:** https://github.com/Lexsi-Labs/CircuitKIT/blob/main/docs/advanced/tiny-aya.md
- Architecture registry (Cohere experimental): https://github.com/Lexsi-Labs/CircuitKIT/blob/main/docs/advanced/architecture-registry.md
- Port source: https://github.com/Lexsi-Labs/CircuitKIT/tree/main/src/circuitkit/backends/_tl_compat
- `ENVIRONMENT.md` (TransformerLens 3.8.0, gate, HF_TOKEN): https://github.com/Lexsi-Labs/CircuitKIT/blob/main/ENVIRONMENT.md
- Explicit custom-data contrasts and schema key gotchas: https://github.com/Lexsi-Labs/CircuitKIT/blob/main/docs/user-guide/custom-data.md
- EAP vs EAP-IG gradients and defaults: https://github.com/Lexsi-Labs/CircuitKIT/blob/main/docs/algorithms/eap.md
- EAP-IG memory levers: https://github.com/Lexsi-Labs/CircuitKIT/blob/main/docs/guides/eap_ig_memory_optimization.md
- Circuit faithfulness evaluation: https://github.com/Lexsi-Labs/CircuitKIT/blob/main/docs/user-guide/evaluation.md
- Refusal-style custom data runnable example: https://github.com/Lexsi-Labs/CircuitKIT/blob/main/examples/case-studies/24-safety-refusal-custom-data.py
- Jailbreak-refusal example notebook: https://github.com/Lexsi-Labs/CircuitKIT/blob/main/examples/case-studies/23-jailbreak-refusal-multi-model.ipynb

**Important source-level finding:** `docs/advanced/tiny-aya.md` says five Tiny Aya repositories, including `CohereLabs/tiny-aya-water`, are supported by a CircuitKIT-added TransformerLens 3.8 compatibility port. It explicitly warns of experimental status, gated HF access, required HF-vs-TL parity test, and single-token answer assumptions for built-in logit-difference tasks. The package `ENVIRONMENT.md` mentions a specific NeMo H200 reproduction image, not a guarantee that Colab T4 will reproduce memory use; implement a Colab-specific tested environment, pin versions, record any deviation. **Do not blindly combine old CUDA/PyTorch sample pins**; inspect current pyproject and Python compatibility.

## Statistical and safety-evaluation references

- Statsmodels Mantel–Haenszel and stratified 2x2 documentation (underlying assumption structure): https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.StratifiedTable.html
- `StrongREJECT` warns that generic jailbreak-judge heuristics can overestimate real harmful compliance: https://arxiv.org/abs/2402.10260
- HarmBench standardized harmful-compliance evaluation: https://proceedings.mlr.press/v235/mazeika24a.html
- WildGuard moderation, refusal and harmfulness benchmarking: https://proceedings.neurips.cc/paper_files/paper/2024/hash/0f69b4b96a46f284b726fbd70f74fb3b-Abstract.html

These benchmarks provide methodological grounding and possible *external* evaluation subsets, not drop-in gold labels for this project's bilingual, culture-controlled prompts. Their classifier availability, licensing, language coverage, and model sizes must be checked before use. Qwen3Guard's 3-way safety classification is a useful automated proxy but cannot replace validated unsafe-compliance annotations.

## Operational / autonomous tooling references

- Official `/goal` overview, scope, stopping criteria: https://learn.chatgpt.com/docs/use-cases/follow-goals
- Official Codex Goals cookbook: https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex
- `/goal` command reference and controls: https://learn.chatgpt.com/docs/developer-commands
- Google Colab mount/runtime/Drive quota limitations: https://research.google.com/colaboratory/faq.html

`/goal` belongs to a Codex thread; markdown files and committed status logs preserve decisions beyond a single long context. There is no guarantee that the Codex account, Drive, Colab session, GPUs or external services will remain continuously accessible. Design resumability and hard budget limits accordingly.
