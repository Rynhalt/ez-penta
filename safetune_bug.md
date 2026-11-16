# SafeTune bug: `_fp32_master_weights` upcasts an orphaned, fully-unfrozen model copy instead of the real LoRA-wrapped one

**Package:** `safetune` 0.1.16 (installed locally at
`.venv`'s site-packages equivalent; same version used in all Colab notebooks this
project ran — `pip install safetune`).
**Source:** https://github.com/Lexsi-Labs/SafeTune (source-available under Lexsi Labs'
own license, not an OSI-approved open-source license, but the code itself is public and
readable/forkable for noncommercial use — so "where in the open source code" below means
this repo's public source tree).

This is the bug that caused pipeline 4's repeated CUDA OOM crashes before we bypassed
`PlainSFTTrainer` with a manual PyTorch training loop. Root-caused by reading the
installed package source directly, not by guessing from symptoms.

## Where

- `safetune/runner/harden/_base.py:342-350` — `_HardenBase._fp32_master_weights`
- `safetune/runner/harden/_base.py:307-325` — `_HardenBase._lora_base`
- `safetune/runner/harden/_base.py:352-358` — `_HardenBase._training_args` (the caller)
- `safetune/runner/harden/_base.py:285-294` — `_HardenBase.model` property
- `safetune/runner/harden/_gradient_surgery.py:17-34` — `PlainSFTTrainer.train` (the
  concrete trainer we actually ran; several other trainers in `_dpo_adversarial.py`,
  `_other.py`, `_representation.py`, `_regularization.py`, `_data_shaping.py`,
  `_tamper_resistant.py` share the exact same `_lora_base()` → `_training_args()` /
  `_configure_args()` call pattern and are therefore exposed to the same bug).

## The exact mechanism

`PlainSFTTrainer.train()`:
```python
def train(self, train_dataset, out_dir: str = None, **kwargs) -> str:
    out_dir = self._resolve_out_dir(out_dir)
    model = self._lora_base()                      # (1)
    ...
    tr = Trainer(model=model,
                 args=self._training_args(out_dir), # (2)
                 ...)
    tr.train()
```

**(1) `self._lora_base()` consumes and clears `self._model`:**
```python
def _lora_base(self):
    if self._model is not None:
        m = self._model
        self._model = None          # <-- cleared here, and NEVER restored
    else:
        ...
        m = load_model(base_name)
    ...
    return lora_wrap(m)              # the real, LoRA-wrapped model is returned
                                      # only as a local value — never cached back
                                      # onto self._model
```
After this line, `self._model` is `None` again. The real, correctly LoRA-wrapped model
(frozen base + small trainable adapter) only exists as the local variable `model` in
`train()`.

**(2) `self._training_args(out_dir)` calls `self._fp32_master_weights(args)`, which reads
the `self.model` *property*, not the local `model` variable:**
```python
@property
def model(self):
    if getattr(self, '_model', None) is None:
        from safetune.runner.utils.model_utils import load_model
        self._model = load_model(self.model_id)   # <-- fresh, FULLY-TRAINABLE load
    return self._model

def _fp32_master_weights(self, args):
    """fp16 mixed precision keeps fp32 master weights. ... cast those to fp32 ..."""
    if getattr(args, "fp16", False) and getattr(self, "model", None) is not None:
        for p in self.model.parameters():
            if p.requires_grad and p.dtype == torch.float16:
                p.data = p.data.float()
    return args
```

Because `_lora_base()` already set `self._model = None`, this property access triggers a
**second, completely separate `load_model()` call** — a brand-new copy of the base model,
loaded straight from `from_pretrained()` with its default `requires_grad=True` on every
parameter (nothing has been frozen or LoRA-wrapped on this second copy; `lora_wrap()` was
only ever applied to the first copy, which is sitting in the local `model` variable).

On a T4, `_precision()` (lines 56-82 of the same file) silently downgrades a requested
`bf16=True` to `fp16=True` (T4 is pre-Ampere and has no native bf16 tensor cores — this
part is correct, intentional behavior, not the bug). That sets `args.fp16 = True`, which
is exactly the condition that makes `_fp32_master_weights` loop over `self.model.parameters()` —
and on this orphaned second copy, **every parameter has `requires_grad=True`**, so every
one of them gets upcast from fp16 to fp32 in place.

## Why this is a bug (concrete failure scenarios)

1. **Silent extra multi-GB model copy, never used for anything.** For a ~3.35B-parameter
   model (this project's Tiny Aya Water), this second copy alone is ~6.7GB in fp16 before
   the upcast, growing to ~13.4GB once every parameter is promoted to fp32 — on top of
   the real model (`model`, correctly LoRA-restricted) that `Trainer` actually trains.
   On a 14-16GB T4 this alone is enough to OOM, even though the *intended* trainable set
   was a few hundred thousand to a few tens of millions of LoRA parameters (417,792 in
   our circuit-restricted arm; 30,228,480 in our broad-LoRA arm) — four to five orders of
   magnitude smaller than what actually got upcast.
2. **The fix defeats its own purpose.** The docstring's stated intent is to protect the
   *trainable* parameters' GradScaler compatibility ("a model loaded in fp16 ... has
   fp16 trainable parameters, which the GradScaler refuses"). Checking `p.requires_grad`
   was meant to scope this to just the LoRA adapter weights. Because of the
   `self.model` property's lazy reload, that scoping silently fails whenever
   `_lora_base()` has already cleared `self._model` — which is every single time
   `PlainSFTTrainer.train()` (and any sibling trainer using the same pattern) runs.
3. **Wasted compute, independent of whether it OOMs.** Even on hardware with enough
   headroom to absorb this, every `.train()` call now does one extra full
   `from_pretrained()` load and one extra full-parameter dtype cast for a model object
   that is discarded at the end of the function and never referenced again.
4. **A latent correctness trap, not just a performance one.** Any caller that inspects
   `trainer.model` *after* calling `.train()` gets this orphaned, randomly-initialized-
   and-partially-upcast second copy — not the model that was actually trained (which only
   ever existed as `train()`'s local `model` variable and was discarded along with the
   function's stack frame once `_save_merged` finished with it).

## How we actually found it

Not from this source read alone — we found it live, under real OOM pressure, in this
order: (a) ruled out the LoRA restriction itself by directly auditing
`model.named_parameters()` on the object actually passed to `Trainer` (clean, 417,792
trainable params, ~0.00GB delta); (b) ruled out `bitsandbytes` quantization being bypassed
(fixed a separate `device_map="cuda"` vs `"auto"` issue); (c) only then read
`_training_args` → `_fp32_master_weights` → `self.model` end-to-end and found the
`_lora_base()` / property interaction above. The eventual fix was to stop calling
`PlainSFTTrainer`/`transformers.Trainer` entirely and write a short manual PyTorch
training loop (forward → `loss.backward()` → `optimizer.step()`) directly on the model
object we already had in hand — see
[`notebooks/pipeline4_circuit_restricted_fix.ipynb`](notebooks/pipeline4_circuit_restricted_fix.ipynb),
cell 10 (`manual_lora_train`). That loop never touches `self.model`/`self._model` at all,
which is why it doesn't trigger this bug.

## What a real fix would look like

**Option A (minimal, targeted) — pass the real model explicitly instead of reading the
lazy property:**
```python
# _base.py
def _fp32_master_weights(self, args, model=None):
    model = model if model is not None else getattr(self, "model", None)
    if getattr(args, "fp16", False) and model is not None:
        for p in model.parameters():
            if p.requires_grad and p.dtype == torch.float16:
                p.data = p.data.float()
    return args

def _training_args(self, out_dir, model=None):
    return self._fp32_master_weights(_make_training_args(
        out_dir, self.epochs, self.batch_size, self.lr,
        self.bf16, self.optimizer, self.logging_steps,
        fp16=self.fp16, wandb=self.wandb,
    ), model=model)
```
```python
# _gradient_surgery.py, PlainSFTTrainer.train (and every other _HardenBase subclass
# with the same pattern)
model = self._lora_base()
tr = Trainer(model=model,
             args=self._training_args(out_dir, model=model),   # <-- pass it through
             ...)
```
This guarantees the upcast loop only ever touches the exact object being trained, never
a reloaded stand-in.

**Option B (root-cause, broader fix) — stop letting `_lora_base()` desync `self._model`
from reality:**
```python
def _lora_base(self):
    if self._model is not None:
        m = self._model
    else:
        ...
        m = load_model(base_name)
    ...
    wrapped = lora_wrap(m)
    self._model = wrapped   # keep the property honest instead of clearing it
    return wrapped
```
This fixes the bug for every caller that later touches `self.model` (not just
`_fp32_master_weights`), at the cost of needing to confirm nothing else relies on
`self._model` being `None` right after `_lora_base()` (a quick grep of `_base.py` and its
subclasses for `self._model` would need to accompany this change).

**Option C (defensive, cheap to add regardless of A/B) — fail loudly instead of silently
reloading:**
```python
@property
def model(self):
    if getattr(self, '_model', None) is None:
        if getattr(self, '_warned_reload', False) is False:
            logger.warning(
                "safetune: self.model triggered a fresh load_model() call — if this "
                "happens mid-training, a previous step likely cleared self._model "
                "without restoring it (see _lora_base)."
            )
            self._warned_reload = True
        self._model = load_model(self.model_id)
    return self._model
```
This wouldn't fix the bug by itself, but would have turned several hours of
source-diving into a five-minute log read.

**Recommendation:** Option A is the smallest, safest patch (touches only the two
functions actually responsible for the bad upcast) and is what we'd suggest upstreaming
first; Option B is more thorough and worth doing too, but touches more call sites and
deserves its own review; Option C is worth adding either way as a tripwire against this
whole class of "lazy property silently reloads a stand-in object" bug recurring
elsewhere in `_base.py`'s subclasses.
