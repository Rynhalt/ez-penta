# Tiny Aya Safety: hackathon submission status

**LEXSI AI Safety & Alignment Hackathon 2026, Track 2.** Deadline: Nov 15, 2026, 11:59 PM AoE.
Team: Isaac Shane and Marcus Izumi.

## What was done

### 1. A clean public GitHub repo
**https://github.com/ishane0620/tiny-aya-safety**

The latest `post_train_2` experiment, moved out of the research repo `Rynhalt/ez-penta`, which was left unchanged.

| Folder or file | Contents |
|---|---|
| `notebooks/01–05` | The five pipeline notebooks, now runnable on Colab or locally. Hard-coded `/content` paths were replaced with a shared config. Notebook 05 loads the adapters from Hugging Face. |
| `notebooks/06_explore_results.ipynb` | **The entry point for judges.** It recomputes every metric (CPU, no token, about a minute), lets you browse all 684 answers by category, and lets you try your own prompt (GPU). |
| `notebooks/executed/` | The original executed notebooks from the Nov 12 run, unchanged (provenance). |
| `src/tiny_aya_safety/` | Config and paths, metrics, split checks, generation and judging, llama.cpp helpers. |
| `data/` | All prompts, training pairs and per-step results, with a SHA-256 manifest. |
| `results/` | Recomputed metrics (JSON), tables and two figures. |
| `docs/` | Methodology, dataset, experiments, results, limitations, reproducibility. |
| `SUBMISSION.md` | 425-word abstract, requirement checklist, verification record, all links. |
| `demo/`, `hf/` | Static demo Space; model and dataset cards; publish script. |
| `tests/` | 53 tests. |

The README has **Open in Colab** badges for all six notebooks, a pipeline diagram, a plain-language explanation of LoRA, the results, adapter loading code and limitations.

### 2. Public Hugging Face release (account `isaac369369`)
- **Model collection:** https://huggingface.co/collections/isaac369369/tiny-aya-water-safety-lora-adapters-6af671c237fdafbed3b7362a
  - Restricted adapter (layer 18, 209K parameters): https://huggingface.co/isaac369369/tiny-aya-water-safety-lora-layer18
  - Broad adapter (all layers, 30.2M parameters): https://huggingface.co/isaac369369/tiny-aya-water-safety-lora-broad
  - Demo Space (saved outputs, no live model): https://huggingface.co/spaces/isaac369369/tiny-aya-safety-explorer
- **Dataset collection:** https://huggingface.co/collections/isaac369369/tiny-aya-safety-datasets-6af671c33b325a1767adb0a5
  - Dataset2: 21,184 Japanese/English jailbreak prompts with the original model's judged answers.
  - Training pairs: 195 kept, 37 rejected.
  - Evaluation results: all 684 answers with verdicts, metrics, per-step logs and executed notebooks.

Everything is CC BY-NC 4.0, matching the base model, and the base model itself isn't re-uploaded. All repos and collections were confirmed public and un-gated from a client with no token.

### 3. Verification
| Check | Result |
|---|---|
| Headline metrics recomputed from the saved answers | Match the original run exactly: unsafe 12/115 → 1/117 and 1/118; needless refusals 8 → 16 and 29 out of 80 |
| Notebook 01 rerun from the published Dataset2 table | 16/16 statistics identical |
| Prompt builders rerun | All 4 prompt files byte-identical |
| Train/validation separation | 0 shared seeds, 0 shared prompt texts, 0 heldout rows used |
| Adapter structure | Matches Tiny Aya's architecture: 208,896 and 30,228,480 parameters |
| Seed-level robustness | Restricted adapter: −8.8 points unsafe (95% CI −14.8 to −3.8), +10 points over-refusal. Broad adapter: +26.3 points over-refusal |
| Notebook 06 on a fresh Colab VM | 0 errors, identical numbers |
| GPU smoke test of the **published** adapters (Colab T4) | Passed: both load, change layer 18's weights, and generate English and Japanese answers that Qwen3Guard judges |
| Static check that all notebooks can run | Passed: every cell parses, every input resolves, every toolkit import exists |
| Tests | 53 passed |

### 4. Corrections found along the way
- The original Dataset2 baseline used **greedy decoding** with a quantized Q4_K_M GGUF model, **not temperature 0.2** as stated earlier. The new docs are correct; `ez-penta/post_train_2/report.md` (assumption 10) still has the old statement.
- `metrics.json` was writing invalid JSON (`NaN`). Fixed.
- 29/80 rounds to **36.3%**, not 36.2%. Fixed everywhere.

## Requirement checklist
| Requirement | Status |
|---|---|
| Abstract under 500 words | Done (425) |
| Public, un-gated HF model collection | Done |
| Public HF dataset collection (data, logs, traces) | Done |
| Write-up / article | Done (README + `docs/`) |
| GitHub repo with reproducibility notes | Done |
| Track, problem statements, repos and models stated | Done |
| 2+ toolkit repos (3+ preferred) | Done: AuditKIT, SafeTune, CuratorKIT |
| More than one before/after metric | Done |
| Public deployment | Done: repo, Colab notebook, HF Space |
| ROI comparison (optional) | Brief: 145× fewer parameters, 0.8 MB vs 121 MB, under 10 GB GPU, about 15 min training |
| **Registered email and form submission** | **Not done: yours to do** |

## What's left

**Before the deadline (yours):**
1. **Submit the form** at https://hackathon.lexsi.ai/submission/ using the links and abstract in `SUBMISSION.md`. Make sure the Hugging Face username on the registration is `isaac369369`, because credits are matched to it.
2. **Revoke the Hugging Face token pasted in chat** (it's also in your zsh history) at https://huggingface.co/settings/tokens. The token this Mac is currently logged in with can stay.

**Not done, and stated in the docs:**
- **Full answer regeneration (notebook 05).** It didn't finish on the T4, and it was skipped by decision. It takes about 8 minutes on a G4.
- **Full re-runs of notebooks 02–04** after migration. Only the static check was done.
- **Human review** of the Qwen3Guard labels.
- **The heldout evaluation:** the 60 untouched seeds.
- **A layer-choice control:** restricted adapters on other layers, to show whether layer 18 itself matters or the adapter is simply smaller.

**Optional cleanup:**
- Fix assumption 10 in `ez-penta/post_train_2/report.md`.
- Remove the duplicate `post_train_2` folder in `MyDrive/ez-penta`.
- Add Marcus as a collaborator on the GitHub repo.
