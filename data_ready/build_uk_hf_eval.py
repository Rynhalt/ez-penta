"""Build the UK HF/PEFT evaluator using shared Dataset2 mechanics."""
def build(md, code, bundle, write):
    files = ['eval/__init__.py','eval/checkpoints.py','eval/dataset2_data.py',
             'eval/dataset2_runtime.py','eval/dataset2_analysis.py','eval/scoring.py',
             'eval/uk_hf_evaluation.py','guard_judge.py','chat.py']
    cells = [md('''# UK evaluation: original and trained Tiny Aya
Run in a fresh Colab GPU runtime with access to Tiny Aya on Hugging Face.
This notebook follows `eval_baseline.ipynb` and `eval_restricted.ipynb`: original HF
weights, and PEFT `merge_and_unload()` for the saved trained adapter.
Defaults to the full UK+neutral dataset and both baseline/restricted models.
Responses and judgments are new UK-cohort measurements; the friend's historical
228-prompt answers are not substituted for this different prompt inventory.

Both models use identical precision and greedy decoding, 512 answer tokens and 64
guard tokens. Inputs are preserved in full; a context overflow fails rather than
silently truncating at the reference notebooks' 1,024-token input cap. One model
runs at a time. Unknown verdicts and incomplete outputs stay explicit.
'''), code('''from pathlib import Path
COHORT = 'full'
ROOT = Path('/content/drive/MyDrive/ez-penta/dataset2_uk')
DRIVE_PROJECT_DIR = Path('/content/drive/MyDrive/ez-penta')
MODELS = ['baseline', 'restricted']  # Can also include 'broad', or select restricted only.
ADAPTER_OVERRIDES = {}  # Example: {'restricted': '/content/drive/MyDrive/.../pipeline4_restricted_adapter'}
MODEL_ID = 'CohereLabs/tiny-aya-water'
GUARD_ID = 'Qwen/Qwen3Guard-Gen-0.6B'
MODEL_REVISION = None  # Optional known training-base commit; otherwise first resolved SHA is saved/reused.
GUARD_REVISION = None
DTYPE = 'auto'  # bf16 on supported GPUs; fp16 on T4. The same dtype is used for both models.
MAX_NEW_TOKENS = 512
GUARD_MAX_NEW_TOKENS = 64
MAX_RETRIES = 1
RETRY_ERRORS = True
'''), code('''# Same HF/PEFT setup approach as the supplied reference notebooks.
# Run in a fresh runtime. Resolved versions are saved to environment.txt for reproduction.
%pip install -q -U transformers accelerate peft torchao
%pip install -q auditkit==1.1.3 'pandas>=2.2,<4'
'''), bundle(files,'/content/ez-penta-uk-hf-eval'),code('''from google.colab import drive
drive.mount('/content/drive')
import os, json, subprocess, sys, traceback
from datetime import datetime, timezone
import pandas as pd
import torch
from huggingface_hub import HfApi, login
from eval.checkpoints import file_sha256, config_id
from eval.dataset2_data import load_dataset, quality_flags
from eval.dataset2_runtime import init_run, atomic_json, stage, make_judge, checkpoint_summary
from eval.dataset2_analysis import observation_table, export_bundle
from eval.uk_hf_evaluation import AdapterHFRunner, find_adapter, adapter_identity, paired_changes
if not torch.cuda.is_available():
    raise RuntimeError('Select Runtime > Change runtime type > GPU.')
token = os.environ.get('HF_TOKEN')
if not token:
    try:
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
    except Exception:
        pass
if token:
    login(token=token, add_to_git_credential=False)
del token
if COHORT not in ('full','pilot') or not MODELS or len(set(MODELS)) != len(MODELS):
    raise ValueError('Choose full/pilot and a nonempty unique model list.')
if not set(MODELS) <= {'baseline','restricted','broad'}:
    raise ValueError('Models must be baseline, restricted, or broad.')
DATA_DIR = ROOT / 'pilot' if COHORT == 'pilot' else ROOT
DATASET_PATH = DATA_DIR / 'dataset2.sqlite3'
dataset_metadata = json.loads(DATASET_PATH.with_suffix('.manifest.json').read_text())
if file_sha256(DATASET_PATH) != dataset_metadata['dataset_sha256']:
    raise ValueError('Dataset hash differs from manifest.')
all_prompts, seeds = load_dataset(DATASET_PATH, expected_count=dataset_metadata['prompt_count'], cultures=('neutral','uk'))
jobs = quality_flags(all_prompts).assign(shard_id=1)
rows = jobs.astype(object).where(jobs.notna(), None).to_dict('records')
# Copy the prompt DB off Drive before SQL reads in downstream analysis when needed.
ROOT_RUN = DATA_DIR / 'evaluation_hf' / dataset_metadata['dataset_sha256'][:16]
ROOT_RUN.mkdir(parents=True, exist_ok=True)
revision_file = ROOT_RUN / 'model_revisions.json'
previous = json.loads(revision_file.read_text()) if revision_file.exists() else {}
aya_revision = MODEL_REVISION or previous.get(MODEL_ID) or HfApi().model_info(MODEL_ID).sha
guard_revision = GUARD_REVISION or previous.get(GUARD_ID) or HfApi().model_info(GUARD_ID).sha
atomic_json(revision_file, {MODEL_ID:aya_revision, GUARD_ID:guard_revision})
dtype = ('bfloat16' if torch.cuda.is_bf16_supported() else 'float16') if DTYPE == 'auto' else DTYPE
if dtype not in ('bfloat16','float16','float32'):
    raise ValueError('Unsupported DTYPE')
if dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
    raise RuntimeError('This GPU does not support bf16; choose auto or float16.')
adapters = {name: find_adapter(DRIVE_PROJECT_DIR, name, ADAPTER_OVERRIDES.get(name))
            for name in MODELS if name != 'baseline'}
settings = dict(models=MODELS, model_id=MODEL_ID, model_revision=aya_revision,
                guard_id=GUARD_ID, guard_revision=guard_revision, dtype=dtype,
                max_new_tokens=MAX_NEW_TOKENS, guard_max_new_tokens=GUARD_MAX_NEW_TOKENS,
                adapters={name:adapter_identity(path) for name,path in adapters.items()},
                input_policy='no_silent_truncation', batch_size=1,
                helper_sha256=file_sha256(REPO_DIR / 'eval/uk_hf_evaluation.py'),
                runtime_sha256=file_sha256(REPO_DIR / 'eval/dataset2_runtime.py'))
RUN_ROOT = ROOT_RUN / config_id(settings)[:16]
RUN_ROOT.mkdir(parents=True, exist_ok=True)
atomic_json(RUN_ROOT / 'settings.json', settings)
(RUN_ROOT / 'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
(RUN_ROOT / 'gpu.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
print('Models:', MODELS, '| GPU:', torch.cuda.get_device_name(0), '| dtype:',dtype)
print('Matched prompt inventory:',len(rows),'prompts across',len(seeds),'original seeds')
print('Adapter folders:',{k:str(v) for k,v in adapters.items()})
print('Outputs:',RUN_ROOT)
print('Base revision must match training for strict reproduction; inspect model_revisions.json and adapter_config.json.')
display(jobs.groupby(['culture','framing','prompt_language']).size().rename('prompts').to_frame())
'''),md('''## Generate, judge and export each model
Each answer is journaled to Drive. Compatible answers are reused on resume.
The base model is released before loading the guard. The guard uses the same
response parser as Dataset2 and additionally records prompt safety independently.
No prompt truncation is performed. Token-cap answers are retained as truncated
and excluded from safety scoring. Context errors remain errors.
'''),code('''tables = {}
for name in MODELS:
    run_dir = init_run(RUN_ROOT / name, 1, dataset_metadata, 'uk_hf_'+name)
    active = None
    phase = 'model_loading'
    started = datetime.now(timezone.utc).isoformat()
    resources = {}
    try:
        atomic_json(run_dir / 'execution_status.json',dict(status='running',phase=phase,started_at=started))
        torch.cuda.reset_peak_memory_stats()
        active = AdapterHFRunner(MODEL_ID, aya_revision, dtype=dtype,
                                 max_new_tokens=MAX_NEW_TOKENS, adapter=adapters.get(name))
        generation_config = dict(active.configuration, model_arm=name)
        phase = 'answers'
        checkpoint_summary(rows, run_dir/'answers.jsonl',1,generation_config,retry_errors=RETRY_ERRORS)
        answers = stage(rows,run_dir/'answers.jsonl',1,generation_config,active.answer,
                        max_retries=MAX_RETRIES,retry_errors=RETRY_ERRORS)
        resources['answer_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        active.close()
        active = None
        phase = 'guard_loading'
        torch.cuda.reset_peak_memory_stats()
        active = AdapterHFRunner(GUARD_ID, guard_revision, dtype=dtype,
                                 max_new_tokens=GUARD_MAX_NEW_TOKENS, judge=True)
        score_answer, judge_config = make_judge(active,max_tokens=GUARD_MAX_NEW_TOKENS)
        phase = 'scoring'
        checkpoint_summary(rows, run_dir/'scores.jsonl',1,judge_config,answers=answers,retry_errors=RETRY_ERRORS)
        scores = stage(rows,run_dir/'scores.jsonl',1,judge_config,score_answer,answers=answers,
                       max_retries=MAX_RETRIES,retry_errors=RETRY_ERRORS)
        resources['guard_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        active.close()
        active = None
        del score_answer
        phase = 'export'
        table = observation_table(rows,answers,scores)
        metadata = dict(shard_id=1,run_mode='uk_hf_'+name,dataset=dataset_metadata,
                        generation=generation_config,judge=judge_config,
                        expected_prompts_per_seed=dataset_metadata['expected_prompts_per_seed'],
                        prompt_ids=sorted(jobs.prompt_id), model_arm=name,
                        settings=settings, resources=resources,started_at=started,
                        analysis_policy='descriptive-seed-bootstrap-unvalidated-expectations')
        results,coverage = export_bundle(run_dir,table,answers,scores,metadata)
        # Alias fields expected by the friend's inspection notebooks; Dataset2 columns are preserved.
        view = table.assign(id=table.prompt_id,model=name,language=table.prompt_language,label=table.safety)
        view['pair_id'] = [config_id([r['seed_id'],r['kind'],r['framing'],r['prompt_language'],r['output_language']])[:24] for r in rows]
        view['pair_bucket'] = view['kind']+'|'+view['framing']+'|'+view['prompt_language']+'|'+view['output_language']
        view.to_json(RUN_ROOT/f'{name}_answers.jsonl',orient='records',lines=True,force_ascii=False)
        tables[name] = table
        atomic_json(run_dir/'execution_status.json',dict(status='exported',coverage=coverage,resources=resources))
        print(name,'exported:',run_dir)
        display(pd.DataFrame([coverage]))
        display(results['comparisons'])
    except Exception:
        (run_dir/'failure_traceback.txt').write_text(traceback.format_exc())
        atomic_json(run_dir/'execution_status.json',dict(status='failed',phase=phase,started_at=started))
        raise
    finally:
        if active is not None:
            active.close()
'''),md('''## Original versus trained model
Paired safety/refusal outcomes use identical prompt IDs. The intervals resample
source seeds, accounting for multiple variants of each seed. They are descriptive,
unadjusted intervals. A twin refusal is a measured refusal; over-refusal requires
validation that the request is benign. The UK cohort reuses existing seeds and
cannot be described as an untouched test set without an exposure audit.
'''),code('''comparison = paired_changes(tables)
comparison.to_csv(RUN_ROOT/'before_after.csv',index=False)
print('Before/after comparison:',RUN_ROOT/'before_after.csv')
if not comparison.empty:
    display(comparison)
else:
    print('Select baseline and a trained arm together to produce before/after comparisons.')
print('Per-model observations and run_metadata.json are under <model>/part1/.')
print('Compatibility views:',*[str(RUN_ROOT/f'{name}_answers.jsonl') for name in MODELS])
print('Run status, resources, raw guard records, settings and dependencies are saved on Drive.')
''')]
    write('eval/eval_dataset2_uk.ipynb',cells)
