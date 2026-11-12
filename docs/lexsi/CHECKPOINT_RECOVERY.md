# Track-2 checkpoint recovery

**Updated:** 2026-11-12  
**Status:** `INCOMPLETE_DRIVE_CANDIDATE`; no complete checkpoint has been recovered or independently reloaded.

## Historical checkpoint references

The supplied Pipeline 4 report describes two one-epoch SFT LoRA arms trained from the Tiny Aya Water HF model using 22 generated pairs:

| Arm | Reported configuration | Recorded location in `pipeline4_results.json` | Recovery status |
|---|---|---|---|
| Restricted | LoRA on `down_proj` in layers 18 and 23; 417,792 trainable parameters | `/content/pipeline4_restricted` | Not found in the searched Drive; path is Colab-local |
| Broad | LoRA over attention and MLP projections; about 30.2M trainable parameters | `/content/pipeline4_full` | Not found in the searched Drive; path is Colab-local |

The teammate report says these were saved after training, but the current Drive audit did not find checkpoint files. Inspection of the Pipeline 4 notebook confirms it calls `merge_and_unload()` on both models before `save_checkpoint`. The intended saved artifact is therefore a merged model, not an unmerged PEFT adapter; its actual type, quantized merge fidelity, and reloadability cannot be established without the weight files and configs.

## Search performed

On 2026-11-11, the official Colab CLI mounted the authorized Google Drive `MyDrive` in a CPU session. A recursive filename and extension inventory searched all mounted `MyDrive` for names associated with Pipeline 4, restricted/full adapters, checkpoints, Tiny Aya, and common model weight/archive extensions. A narrower follow-up examined project-related paths. The output lists paths and byte sizes only:

- [All-MyDrive search result](../../outputs/lexsi/checkpoint_search_all_mydrive.json)
- [Project-folder search result](../../outputs/lexsi/checkpoint_search_project_folders.json)
- [Colab execution and shutdown log](COLAB_DEBUG_LOG.md#track-2-drive-artifact-inventory-2026-11-11)

The all-MyDrive search returned six candidate files: an unrelated attachment archive, three llama-cuda cache archives, the historical Tiny Aya GGUF, and the Qwen3Guard GGUF. It found no `.safetensors`, PyTorch checkpoint, or file whose name identified either Pipeline 4 arm. The project folder contains `combined/pipeline4_results.json` and the five notebook files, but no adapter weights in the searched locations.

The prior 2026-11-11 recursive search did not find weights. A subsequent connected-Drive inspection on 2026-11-12 found newly created checkpoint folders in the project extension area:

- Restricted folder: contains `config.json` (2,500 bytes) and `model.safetensors` (25,231,360 bytes). The active L4 runtime's copy of the model file was 3,599,063,241 bytes, so the Drive weight is only about 0.7% of expected size and is not a complete merged model. The folder also lacks an `adapter_config.json` and complete shard index/weights.
- Broad folder: empty.
- Drive searches for `adapter_model.safetensors`, `model-00001-of-00002.safetensors`, `pipeline4_full model.safetensors`, and `adapter_config.json` found no additional files.

The Drive folder/file IDs and byte-level metadata are preserved privately in ignored `outputs/lexsi/checkpoint_drive_candidate_inventory_20261112.json`. I downloaded the 25,231,360-byte restricted candidate to ignored `outputs/lexsi/checkpoint_candidate_inspection_20261112/model.safetensors` (SHA256 `62eb698703051260588f1cc0c2e85a1bbbfd4117b740e64d2eca028241feadc1`). `safetensors.safe_open` failed with `SafetensorError: Error while deserializing header: incomplete metadata, file not fully covered`. This is direct evidence that the candidate is not a valid complete safetensors file, not merely evidence of a size mismatch. No attempt was made to instantiate it as a model. These findings still do not establish that a checkpoint is reloadable. A teammate-owned Drive, Shared Drive, other Colab runtime, or other storage may contain complete copies.

### Connected Drive recheck (2026-11-12)

Using the connected Google Drive file tools, searched `pipeline4`, `adapter_model.safetensors`, `pytorch_model.bin`, and `Tiny Aya checkpoint`, and listed the exact restricted and broad folders. The restricted folder ID `1zdKfJL4wqDbEyu5lm6n1AJmYcg2sok9D` still contains only `model.safetensors` (25,231,360 bytes) and `config.json` (2,500 bytes); broad folder ID `1VeM-YYRiQ_oorh2Vlhvb9_UQlYG8MpEG` remains empty. Searches found no adapter or full-model filename. This recheck found no new recoverable weight artifact. The partial file was then downloaded and its Safetensors header was tested; it failed deserialization as described above.

## Recovery gate and next action

Do not report an adapter evaluation or a before/after improvement until the exact weights are recovered and validated. If recovered, preserve original files, record Drive IDs/paths and SHA256, inspect `config.json` / `adapter_config.json` and weight tensor metadata, then reload each arm with the exact base revision in a fresh process and compare fixed-prompt outputs against saved reference outputs.

At the earlier L4 smoke run, the two complete checkpoints existed only in runtime scratch. They were not persisted, and no current session now contains them. The CLI currently reports 99.90 compute units and no session. Keep them classified as previously observed runtime-local, not currently verified or durable. The Drive mount to that runtime previously failed with HTTP 400; do not retry the same CLI mount configuration or claim its upload succeeded. If complete weights cannot be copied from that runtime or another authorized source, mark the historical arms unrecoverable. Any new training is a replacement experiment, not a reproduction by default. Before rerunning training, freeze the paired evaluation cohort and verify training data, leakage exclusions, answer-only loss masking, adapter target modules and checkpoint-save/reload behavior. Save PEFT adapters before any merge and copy them atomically to private Drive with checksums before releasing Colab.

The private search outputs are ignored by Git under `outputs/lexsi/.gitignore`. No weights or prompt/response contents were copied into this report.
# Historical training data recovered for seed audit (2026-11-12)

The project Drive `Pipeline1111/combined/` folder contains `pipeline3_safety_pairs.jsonl` (22 records, 26,660 bytes). Its fields are `instruction`, `chosen`, `rejected`, and an opaque `source_sample_id`; it has no Dataset2 `seed_id`. The companion `pipeline3_seed_prompts.json` contains 24 records and only language/framing/culture metadata, also without seed IDs. Accepted-pair counts by language/framing are EN control 6, EN roleplay 5, JA control 6, and JA roleplay 5. All 22 opaque IDs were exactly matched against `matched_observations.csv`'s prompt-ID columns: 22 unique prompt mappings, no ambiguous mappings, and 21 distinct source seeds (one repeated seed).

This verifies the Pipeline3 source seed exposure. It is not enough to complete the required registry because prior Pipeline2 probe seeds and any other tuning exposures remain to be inventoried. See the private aggregate `outputs/lexsi/prior_seed_inventory_partial_20261112.json` and incomplete `outputs/lexsi/circuitkit_prior_seed_registry.json`. The notebook 03 gate remains closed.


## Drive revision-history and alternate-copy recheck (2026-11-12)

The restricted `model.safetensors` Drive file has exactly one revision (25,231,360 bytes) and no `previousRevisionId`, so Drive revision history offers no earlier complete file. A fresh global Drive search for `pipeline4_restricted`, `pipeline4_full`, `adapter_model.safetensors`, and `tiny-aya-water adapter` found no alternate full checkpoint or adapter. The existing partial file remains invalid and no recoverable historical weights were found. Inventory: `outputs/lexsi/checkpoint_drive_candidate_inventory_20261112.json`.


### Latest connected-Drive checkpoint recheck (2026-11-12 11:02 UTC)

A fresh metadata search for `pipeline4_restricted`, `pipeline4_full`, `adapter_model.safetensors`, and Tiny Aya checkpoints returned only the known restricted/broad folders. Read-only folder listings confirm the restricted folder still contains only the 25,231,360-byte incomplete `model.safetensors` and its 2,500-byte config; the broad folder is still empty. No additional adapters or weights were found. The prior Safetensors parse failure remains authoritative for the restricted candidate; it is not a usable checkpoint.

### Latest broad Drive search and file integrity check (2026-11-12 15:56 UTC)

A new Drive search for `pipeline4_full`, `adapter_config`, and Tiny Aya LoRA/checkpoint names returned no checkpoint files beyond the known folders. Folder readback: restricted has only its 25,231,360-byte `model.safetensors` and 2,500-byte `config.json`; broad remains empty. Reopening the private restricted candidate with `safetensors.safe_open` again raises `SafetensorError: Error while deserializing header: incomplete metadata, file not fully covered`. The candidate is confirmed unusable. Reviewer CSV metadata is unchanged from the 2026-11-12 06:16 UTC upload. Colab has 99.90 CUs, no session, and no usage accruing. This recheck does not recover either training arm.
