"""HF/PEFT evaluation for UK data, using Dataset2 journals and scoring."""
import gc
import hashlib
import inspect
from pathlib import Path
import zipfile

from eval.checkpoints import config_id, file_sha256
from eval.dataset2_runtime import HFRunner


def find_adapter(project, model_name, override=None, local_root='/content/uk_adapters'):
    folder = {'restricted': 'pipeline4_restricted_adapter', 'broad': 'pipeline4_full_adapter'}[model_name]
    project = Path(project)
    bases = [project / 'post_train_2', project]
    search = [base / sub for base in bases for sub in ['notebooks', 'notebooks/combined', 'view_evals', '']]
    if override:
        candidate = Path(override)
        if not (candidate / 'adapter_config.json').is_file():
            raise FileNotFoundError(f'Adapter override lacks adapter_config.json: {candidate}')
        return candidate
    for directory in search:
        if (directory / folder / 'adapter_config.json').is_file():
            return directory / folder
    for directory in search:
        archive_path = directory / 'pipeline4_adapters.zip'
        if not archive_path.is_file():
            continue
        target = Path(local_root) / file_sha256(archive_path)[:16]
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                parts = Path(member.filename).parts
                if folder not in parts:
                    continue
                relative = Path(*parts[parts.index(folder):])
                dest = target / relative
                if not dest.resolve().is_relative_to(target.resolve()):
                    raise ValueError('Unsafe adapter archive path')
                if member.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(archive.read(member))
        if (target / folder / 'adapter_config.json').is_file():
            return target / folder
    raise FileNotFoundError(f'{folder} or pipeline4_adapters.zip not found under {project}. Set ADAPTER_OVERRIDES.')


def adapter_identity(directory):
    directory = Path(directory)
    files = sorted([directory / 'adapter_config.json', *directory.glob('*.safetensors'),
                    *directory.glob('adapter_model*.bin')])
    if len(files) < 2:
        raise FileNotFoundError(f'Adapter weights missing: {directory}')
    return {p.name: file_sha256(p) for p in files}


class AdapterHFRunner(HFRunner):
    def __init__(self, model_id, revision, *, dtype, max_new_tokens, adapter=None, judge=False):
        super().__init__(model_id, revision, dtype=dtype, judge=judge)
        hashes = None
        if adapter is not None:
            from peft import PeftModel, PeftConfig
            config = PeftConfig.from_pretrained(str(adapter))
            base = config.base_model_name_or_path
            if base != model_id:
                self.close()
                raise ValueError(f'Adapter declares base {base!r}, expected {model_id!r}')
            hashes = adapter_identity(adapter)
            self.model = PeftModel.from_pretrained(self.model, str(adapter)).merge_and_unload()
            self.model.eval()
        self.configuration.update(
            backend='transformers-peft-merged' if adapter else 'transformers',
            adapter_files=hashes,
            adapter_application='merge_and_unload' if adapter else None,
            peft_version=__import__('importlib.metadata', fromlist=['version']).version('peft'),
            extension_implementation=hashlib.sha256(inspect.getsource(AdapterHFRunner).encode()).hexdigest(),
            input_policy='full_input_no_silent_truncation',
        )
        self.configuration['generation']['max_new_tokens'] = max_new_tokens


def paired_changes(tables):
    """Descriptive matched outcomes with seed-cluster bootstrap intervals."""
    import numpy as np
    import pandas as pd
    if 'baseline' not in tables:
        return pd.DataFrame()
    records = []
    baseline = tables['baseline']
    for name, table in tables.items():
        if name == 'baseline':
            continue
        both = baseline.merge(table, on='prompt_id', suffixes=('_base', '_fixed'), validate='one_to_one')
        if len(both) != len(baseline) or len(both) != len(table):
            raise ValueError('Models did not evaluate the identical prompt inventory')
        if not both.prompt_base.eq(both.prompt_fixed).all():
            raise ValueError('Matched prompt text differs')
        for key, frame in both.groupby(['split_base','culture_base','prompt_language_base','kind_base']):
            for metric in ['unsafe', 'refused']:
                known = frame.dropna(subset=[metric+'_base', metric+'_fixed']).copy()
                delta = known[metric+'_fixed'] - known[metric+'_base']
                per_seed = delta.groupby(known.seed_id_base).mean().to_numpy()
                low = high = mean = None
                if len(per_seed):
                    mean = float(per_seed.mean())
                    rng = np.random.default_rng(42)
                    boots = rng.choice(per_seed, (1000,len(per_seed)), replace=True).mean(axis=1)
                    low, high = map(float, np.quantile(boots,[.025,.975]))
                records.append(dict(model=name, split=key[0],culture=key[1],language=key[2],kind=key[3],
                    metric=metric, assigned_pairs=len(frame),known_pairs=len(known),known_seeds=len(per_seed),
                    baseline_rate=float(known[metric+'_base'].mean()) if len(known) else None,
                    fixed_rate=float(known[metric+'_fixed'].mean()) if len(known) else None,
                    seed_mean_delta=mean,seed_bootstrap_low=low,seed_bootstrap_high=high,
                    zero_to_one=int(((known[metric+'_base']==0)&(known[metric+'_fixed']==1)).sum()),
                    one_to_zero=int(((known[metric+'_base']==1)&(known[metric+'_fixed']==0)).sum())))
    return pd.DataFrame(records)
