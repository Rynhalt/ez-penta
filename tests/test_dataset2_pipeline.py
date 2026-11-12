"""CPU acceptance tests for the two Colab shards, journals, guard and merge."""

import copy
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from eval.checkpoints import config_id
from eval.dataset2_data import (
    load_dataset,
    make_manifest,
    verify_manifest,
    quality_flags,
    missing_variants,
)
from eval.dataset2_runtime import (
    Journal,
    stage,
    init_run,
    make_judge,
    response_identity,
    score_identity,
)
from eval.dataset2_analysis import (
    observation_table,
    summarize,
    matched_observations,
    comparison_summary,
    export_bundle,
)
from eval.merge_dataset2_results import merge

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "marcus/dataset2_partial.sqlite3"
MANIFEST = ROOT / "eval/manifests/dataset2_shards.csv"


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p, cls.s = load_dataset(DB)
        cls.m = pd.read_csv(MANIFEST)
        cls.meta = json.loads(MANIFEST.with_suffix(".json").read_text())
        cls.all = quality_flags(verify_manifest(cls.p, cls.s, cls.m, cls.meta, DB))

    def test_exact_partition_reproducible(self):
        pd.testing.assert_frame_equal(
            make_manifest(
                self.p.sample(frac=1, random_state=7),
                self.s.sample(frac=1, random_state=7),
            ),
            self.m,
        )
        self.assertEqual(self.all.groupby("shard_id").size().tolist(), [10592, 10592])
        self.assertEqual(self.m.groupby("shard_id").size().tolist(), [160, 160])
        self.assertEqual(self.all.groupby("seed_id").shard_id.nunique().max(), 1)
        self.assertEqual(len(set(self.all.prompt_id)), 21184)
        self.assertTrue(
            self.m.groupby(["category", "split", "shard_id"])
            .size()
            .unstack()
            .apply(lambda r: r[1] == r[2], axis=1)
            .all()
        )
        self.assertFalse(self.all.prompt_id.duplicated().any())

    def test_validation_and_quality(self):
        controls = self.all[self.all.framing == "control"]
        self.assertEqual(int(controls.identical_twin_flag.sum()), 996)
        self.assertEqual(controls[controls.identical_twin_flag].seed_id.nunique(), 119)
        self.assertEqual(len(missing_variants(self.p, self.s)), 5696)
        self.assertTrue(self.all.generation_eligible.all())
        self.assertEqual(self.all.seed_id.nunique(), 320)
        self.assertEqual(int(self.s.complete.sum()), 42)
        self.assertFalse(self.all.expectation_validated.any())
        pd.testing.assert_series_equal(
            self.p.prompt,
            self.all.set_index("prompt_id")
            .loc[self.p.prompt_id, "prompt"]
            .reset_index(drop=True),
            check_names=False,
        )
        altered = self.m.copy()
        altered.loc[0, "shard_id"] = 3
        with self.assertRaises(ValueError):
            verify_manifest(self.p, self.s, altered, self.meta, DB)
        with self.assertRaises(ValueError):
            load_dataset(DB, expected_count=1)

    def test_auxiliary_seed_relationships_and_wide_counts(self):
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "invalid.sqlite3"
            shutil.copyfile(DB, db)
            with sqlite3.connect(db) as con:
                con.execute(
                    "UPDATE seed_texts SET source_id='orphan' WHERE rowid=(SELECT MIN(rowid) FROM seed_texts)"
                )
            with self.assertRaisesRegex(
                ValueError, "Orphan seed relationship in seed_texts"
            ):
                load_dataset(db)
            shutil.copyfile(DB, db)
            with sqlite3.connect(db) as con:
                con.execute(
                    "UPDATE prompts_wide SET n_variants=n_variants+1 WHERE rowid=(SELECT MIN(rowid) FROM prompts_wide)"
                )
            with self.assertRaisesRegex(ValueError, "Wide-table variant counts"):
                load_dataset(db)

    def test_source_unchanged(self):
        from eval.checkpoints import file_sha256

        self.assertEqual(file_sha256(DB), self.meta["dataset_sha256"])


class JournalTests(unittest.TestCase):
    def test_interruption_preserves_answers_and_scores_without_repeating_work(self):
        from eval.dataset2_runtime import checkpoint_summary
        generated = []
        def generate(row):
            if len(generated) == 1:
                raise KeyboardInterrupt()
            generated.append(row["prompt_id"])
            return dict(status="completed", answer="saved answer")
        with self.assertRaises(KeyboardInterrupt):
            stage(self.rows, self.path, 1, {}, generate)
        self.assertEqual(checkpoint_summary(self.rows, self.path, 1, {})["reused"], 1)
        def finish(row):
            self.assertNotIn(row["prompt_id"], generated)
            generated.append(row["prompt_id"])
            return dict(status="completed", answer="saved answer")
        answers = stage(self.rows, self.path, 1, {}, finish)
        scores_path = self.path.with_name("scores.jsonl")
        judged = []
        def judge(row, answer):
            if len(judged) == 1:
                raise KeyboardInterrupt()
            judged.append(row["prompt_id"])
            return dict(status="completed", safety="Safe")
        with self.assertRaises(KeyboardInterrupt):
            stage(self.rows, scores_path, 1, {}, judge, answers=answers)
        self.assertEqual(checkpoint_summary(self.rows, scores_path, 1, {}, answers=answers)["reused"], 1)
        def finish_judge(row, answer):
            self.assertNotIn(row["prompt_id"], judged)
            judged.append(row["prompt_id"])
            return dict(status="completed", safety="Safe")
        stage(self.rows, scores_path, 1, {}, finish_judge, answers=answers)
        self.assertEqual(len(generated), len(self.rows))
        self.assertEqual(len(judged), len(self.rows))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "answers.jsonl"
        self.rows = [
            dict(prompt_id=str(i), seed_id="s", shard_id=1, prompt="hello " + str(i))
            for i in range(3)
        ]

    def test_resume_idempotence_invalidation(self):
        calls = []

        def worker(row):
            calls.append(row["prompt_id"])
            return {"answer": "hello", "status": "completed"}

        stage(self.rows[:1], self.path, 1, {"model": "a"}, worker)
        stage(self.rows, self.path, 1, {"model": "a"}, worker)
        self.assertEqual(len(calls), 3)
        stage(self.rows, self.path, 1, {"model": "a"}, worker)
        self.assertEqual(len(calls), 3)
        stage(self.rows, self.path, 1, {"model": "b"}, worker)
        self.assertEqual(len(calls), 6)
        changed = copy.deepcopy(self.rows)
        changed[0]["prompt"] = "changed"
        stage(changed, self.path, 1, {"model": "b"}, worker)
        self.assertEqual(len(calls), 7)

    def test_tail_corruption_and_interior_failure(self):
        j = Journal(self.path, 1)
        r = dict(prompt_id="a", identity="x", shard_id=1, status="completed")
        j.append(r)
        with self.path.open("ab") as f:
            f.write(b'{"broken":')
        recovered = Journal(self.path, 1)
        self.assertEqual(recovered.get("a", "x"), r)
        self.assertEqual(len(list(self.path.parent.glob("*.damaged-*"))), 1)
        recovered.append(dict(r, prompt_id="b"))
        self.assertEqual(len(Journal(self.path, 1).records), 2)
        self.path.write_text("{bad}\n" + json.dumps(r) + "\n")
        with self.assertRaises(ValueError):
            Journal(self.path, 1)

    def test_latest_success_preferred_foreign_shard_rejected(self):
        j = Journal(self.path, 1)
        r = dict(prompt_id="a", identity="x", shard_id=1, status="completed")
        j.append(r)
        j.append(dict(r, status="error"))
        self.assertEqual(j.get("a", "x")["status"], "completed")
        with self.assertRaises(ValueError):
            Journal(self.path, 2)
        with self.assertRaises(ValueError):
            j.append(dict(r, shard_id=2))
        self.assertNotEqual(
            init_run(self.path.parent, 1, {}), init_run(self.path.parent, 2, {})
        )
        with self.assertRaises(ValueError):
            init_run(self.path.parent, 1, {"changed": True})

    def test_errors_retry_rescore_without_regeneration(self):
        attempts = []

        def transient(row):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("temporary")
            return dict(answer="hello", status="completed")

        answers = stage(self.rows[:1], self.path, 1, {}, transient, max_retries=1)
        self.assertEqual(len(attempts), 2)
        scores_path = self.path.parent / "scores.jsonl"
        calls = []

        def score(row, a):
            calls.append(1)
            return dict(status="completed", scores={})

        stage(self.rows[:1], scores_path, 1, {"parser": "a"}, score, answers=answers)
        stage(self.rows[:1], scores_path, 1, {"parser": "b"}, score, answers=answers)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(attempts), 2)
        failed = stage(
            self.rows[1:2],
            self.path,
            1,
            {},
            lambda r: (_ for _ in ()).throw(ValueError("permanent")),
        )
        self.assertEqual(failed["1"]["status"], "error")
        self.assertTrue(failed["1"]["permanent_error"])


class IdentitySeparationTests(unittest.TestCase):
    def test_parser_revision_preserves_generation_identity(self):
        from eval.dataset2_runtime import implementation_id

        generation = implementation_id("generation")
        judge = implementation_id("judge")

        def revised_parser(text):
            return 0.0, {"label": "Unparsed", "unknown": True}

        with patch("guard_judge.parse_guard", revised_parser):
            self.assertEqual(implementation_id("generation"), generation)
            self.assertNotEqual(implementation_id("judge"), judge)

    def test_resumed_answers_refresh_derived_flags(self):
        row = {
            "prompt_id": "p",
            "seed_id": "s",
            "shard_id": 1,
            "prompt": "Hello",
            "semantic_review_required": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answers.jsonl"
            stage([row], path, 1, {}, lambda _: dict(status="completed", answer="Hi"))
            revised = {**row, "semantic_review_required": False}
            answers = stage(
                [revised],
                path,
                1,
                {},
                lambda _: self.fail("Unchanged prompt/config must resume"),
            )
            self.assertFalse(answers["p"]["semantic_review_required"])
            self.assertEqual(answers["p"]["answer"], "Hi")


class GuardTests(unittest.TestCase):
    def test_real_auditkit_precomputed_guard(self):
        class FakeRunner:
            configuration = {"model_id": "fixture", "revision": "0" * 40}

            def generate(self, messages, max_new_tokens):
                return dict(
                    answer="Safety: Safe\nCategories: None"
                    + ("\nRefusal: No" if len(messages) == 2 else ""),
                    status="completed",
                )

        worker, configuration = make_judge(FakeRunner())
        r = worker(
            {"prompt_id": "p", "prompt": "Hello"}, {"answer": "Hi", "identity": "a"}
        )
        self.assertEqual(r["scores"]["guard_response"]["metadata"]["label"], "Safe")
        self.assertEqual(r["scores"]["guard_prompt"]["metadata"]["label"], "Safe")
        self.assertEqual(len(r["raw_guard_outputs"]), 2)
        self.assertIn("parser", configuration)

    def test_guard_failure_preserves_raw_trace(self):
        class FailedRunner:
            configuration = {"model_id": "fixture"}

            def generate(self, messages, max_new_tokens):
                return {
                    "answer": "Safety: Saf",
                    "status": "truncated",
                    "output_ids": [1, 2],
                }

        worker, configuration = make_judge(FailedRunner())
        row = {"prompt_id": "p", "seed_id": "s", "shard_id": 1, "prompt": "Hello"}
        answer = {"status": "completed", "answer": "Hi", "identity": "answer"}
        with tempfile.TemporaryDirectory() as tmp:
            records = stage(
                [row],
                Path(tmp) / "scores.jsonl",
                1,
                configuration,
                worker,
                answers={"p": answer},
                max_retries=0,
            )
        self.assertEqual(records["p"]["status"], "error")
        self.assertTrue(records["p"]["raw_guard_outputs"])
        self.assertEqual(records["p"]["raw_guard_outputs"][0]["answer"], "Safety: Saf")

    def test_unknown_states(self):
        from guard_judge import parse_guard

        for text in ["bad", "Safety: Controversial", "Safety: Safe\nSafety: Unsafe"]:
            self.assertIsNone(parse_guard(text)[1]["unsafe"])


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p, s = load_dataset(DB)
        m = pd.read_csv(MANIFEST)
        cls.rows = (
            quality_flags(p.merge(m[["seed_id", "shard_id"]], on="seed_id"))
            .head(20)
            .astype(object)
            .where(lambda x: x.notna(), None)
            .to_dict("records")
        )

    def test_unknown_denominators(self):
        r = self.rows[0]
        a = {r["prompt_id"]: dict(status="completed", answer="Hi", identity="a")}
        t = observation_table(self.rows, a, {})
        summary = summarize(t)
        unsafe = summary[summary.metric == "unsafe_rate"].iloc[0]
        self.assertEqual(unsafe.eligible, 1)
        self.assertEqual(unsafe.known, 0)
        self.assertEqual(unsafe.unknown, 1)
        self.assertTrue(np.isnan(unsafe.rate))
        self.assertEqual(unsafe.missing, 19)
        self.assertTrue(t.expected_behavior_correct.isna().all())

    def test_matching_never_crosses_seeds(self):
        rows = []
        for seed in ["s1", "s2"]:
            for lang in ["ja", "en"]:
                rows.append(
                    dict(
                        self.rows[0],
                        seed_id=seed,
                        prompt_id=seed + lang,
                        kind="harmful",
                        culture="neutral",
                        framing="control",
                        prompt_language=lang,
                        output_language=lang,
                        split="analysis",
                        analysis_eligible=True,
                    )
                )
        t = observation_table(rows, {}, {})
        t["unsafe"] = [0.0, 1.0, 1.0, 0.0]
        t["refused"] = [1.0, 0.0, 0.0, 1.0]
        pairs = matched_observations(t)
        self.assertEqual(len(pairs), 4)
        self.assertTrue(
            all(r.prompt_id_a[:2] == r.prompt_id_b[:2] for r in pairs.itertuples())
        )
        summary = comparison_summary(pairs, 100)
        self.assertTrue(summary.seed_mean_delta.eq(0).all())
        self.assertTrue(summary.seed_count.eq(2).all())

    def test_merge_full_real_manifest_with_explicit_missing(self):
        p, s = load_dataset(DB)
        m = pd.read_csv(MANIFEST)
        dm = json.loads(MANIFEST.with_suffix(".json").read_text())
        full = quality_flags(verify_manifest(p, s, m, dm, DB))
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for shard in [1, 2]:
                subset = full[full.shard_id == shard]
                rows = (
                    subset.astype(object).where(subset.notna(), None).to_dict("records")
                )
                path = Path(tmp) / f"part{shard}"
                paths.append(path)
                meta = {
                    "shard_id": shard,
                    "dataset": dm,
                    "run_mode": "full_shard",
                    "generation": {},
                    "judge": {},
                    "analysis_policy": "test",
                }
                answers = {}
                scores = {}
                for number, row in enumerate(rows[: (1 if shard == 1 else 3)]):
                    identity = response_identity(row, meta["generation"])
                    answers[row["prompt_id"]] = {
                        **row,
                        "status": "completed",
                        "answer": "Hello",
                        "identity": identity,
                        "configuration": meta["generation"],
                    }
                    answer = answers[row["prompt_id"]]
                    scores[row["prompt_id"]] = {
                        "prompt_id": row["prompt_id"],
                        "seed_id": row["seed_id"],
                        "shard_id": shard,
                        "status": "completed",
                        "identity": score_identity(row, answer, meta["judge"]),
                        "configuration": meta["judge"],
                        "scores": {
                            "guard_response": {
                                "metadata": {
                                    "label": "Unsafe" if number == 0 else "Safe",
                                    "refusal": "No",
                                }
                            },
                            "guard_prompt": {"metadata": {"label": "Safe"}},
                        },
                    }
                export_bundle(
                    path,
                    observation_table(rows, answers, scores),
                    answers,
                    scores,
                    meta,
                )
            tables, coverage = merge(
                *paths, Path(tmp) / "combined", dataset=DB, manifest=MANIFEST
            )
            self.assertEqual(coverage["assigned"], 21184)
            self.assertEqual(coverage["duplicate_prompt_ids"], 0)
            self.assertEqual(coverage["completed_answers"], 4)
            self.assertFalse(coverage["successful_full_evaluation"])
            self.assertEqual(
                tables["headline"].query("metric == 'unsafe_rate'")["missing"].sum(),
                21180,
            )
            self.assertEqual(
                tables["headline"].query("metric == 'unsafe_rate'")["events"].sum(), 2
            )
            self.assertEqual(
                tables["headline"].query("metric == 'unsafe_rate'")["known"].sum(), 4
            )
            self.assertEqual(set(tables["observations"].prompt_id), set(full.prompt_id))
            meta = json.loads((paths[1] / "run_metadata.json").read_text())
            meta["judge"] = {"different": True}
            (paths[1] / "run_metadata.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "judge"):
                merge(*paths, Path(tmp) / "bad", dataset=DB, manifest=MANIFEST)

    def test_current_duplicate_rejected(self):
        from eval.merge_dataset2_results import read_current

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x"
            r = dict(prompt_id="a", shard_id=1)
            p.write_text((json.dumps(r) + "\n") * 2)
            with self.assertRaises(ValueError):
                read_current(p, {"a"}, 1)


class NotebookTests(unittest.TestCase):
    def test_notebooks_shared_logic_and_full_defaults(self):
        sources = []
        for shard in [1, 2]:
            n = json.loads((ROOT / f"eval/eval_dataset2_part{shard}.ipynb").read_text())
            self.assertEqual(n["nbformat"], 4)
            text = "\n".join(
                "".join(c["source"]) for c in n["cells"] if c["cell_type"] == "code"
            )
            compile(text, "notebook", "exec")
            self.assertIn(f"SHARD_ID = {shard}", text)
            self.assertIn('RUN_MODE = "full_shard"', text)
            self.assertIn("SMOKE_SEED_LIMIT = None", text)
            self.assertIn("finally:", text)
            self.assertIn("GGUFRunner", text)
            self.assertIn("export_bundle", text)
            self.assertNotIn("hf_" + "secret", text)
            sources.append(text.replace(f"SHARD_ID = {shard}", "SHARD_ID = X"))
        self.assertEqual(sources[0], sources[1])


class NotebookExecutionTests(unittest.TestCase):
    def test_each_notebook_executes_cpu_fixture_top_to_bottom(self):
        # Environment/install/login require Colab. Every subsequent code cell executes
        # unchanged with one complete seed and real AuditKIT, journal, analysis/export.
        class FixtureRunner:
            def __init__(self, role, model_path, build, log_dir, context_size=8192):
                self.judge = role == "guard"
                self.configuration = {
                    "backend": "gguf-fixture",
                    "model_id": role,
                    "fixture": True,
                }

            def answer(self, row):
                return dict(
                    answer="Hello.",
                    status="completed",
                    input_ids=[1],
                    output_ids=[2],
                    input_tokens=1,
                    output_tokens=1,
                    finish_reason="eos",
                    truncated=False,
                )

            def generate(self, messages, cap):
                return dict(
                    answer="Safety: Safe\nCategories: None"
                    + ("\nRefusal: No" if len(messages) == 2 else ""),
                    status="completed",
                )

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            for shard in [1, 2]:
                n = json.loads(
                    (ROOT / f"eval/eval_dataset2_part{shard}.ipynb").read_text()
                )
                codes = [
                    "".join(c["source"]) for c in n["cells"] if c["cell_type"] == "code"
                ]
                namespace = {"display": lambda *args: None}
                exec(codes[0], namespace)
                namespace.update(
                    REPO_DIR=ROOT,
                    DATASET_PATH=DB,
                    MANIFEST_PATH=MANIFEST,
                    OUTPUT_ROOT=Path(tmp),
                    USE_DRIVE=False,
                    RUN_MODE="smoke",
                    SMOKE_SEED_LIMIT=1,
                )
                with patch("eval.dataset2_gguf.GGUFRunner", FixtureRunner), patch(
                    "eval.dataset2_gguf.ensure_cuda_server",
                    return_value={"fixture": True},
                ), patch(
                    "eval.dataset2_gguf.prepare_model",
                    return_value=Path("fixture.gguf"),
                ):
                    for source in codes[3:]:
                        exec(compile(source, "notebook-cell", "exec"), namespace)
                self.assertTrue(namespace["coverage"]["successful_full_evaluation"])
                self.assertEqual(namespace["coverage"]["seeds"], 1)
                self.assertTrue((namespace["RUN_DIR"] / "results.sqlite3").exists())


class ModelMechanicsTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
        "Install eval/requirements-dataset2.txt for CPU model checks",
    )
    def test_teacher_forcing_matches_full_forward(self):
        import torch
        from transformers import Cohere2Config, Cohere2ForCausalLM
        from eval.dataset2_likelihood import token_sequence_logprobs

        torch.manual_seed(42)
        config = Cohere2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
            sliding_window=32,
        )
        model = Cohere2ForCausalLM(config).eval()
        context = [1, 3, 5]
        target = [7, 9, 2]
        got = token_sequence_logprobs(model, context, target, chunk_tokens=2)
        with torch.inference_mode():
            logits = model(torch.tensor([context + target[:-1]])).logits[0].float()
            expected = (
                torch.log_softmax(logits[len(context) - 1 :], -1)
                .gather(-1, torch.tensor(target)[:, None])
                .flatten()
            )
        np.testing.assert_allclose(got["token_logprobs"], expected.numpy(), atol=1e-5)
        self.assertEqual(got["token_count"], 3)
        self.assertAlmostEqual(got["sum_logprob"], float(expected.sum()), places=5)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "Install torch for tensor mechanics"
    )
    def test_generation_eos_cap_counts_and_attention(self):
        import torch
        from types import SimpleNamespace
        from eval.dataset2_runtime import HFRunner

        class Encoded(dict):
            def to(self, device):
                return self

        class Tokenizer:
            pad_token_id = 0

            def apply_chat_template(self, messages, **kwargs):
                return "rendered"

            def __call__(self, text, **kwargs):
                self.kwargs = kwargs
                return Encoded(
                    input_ids=torch.tensor([[1, 5, 6]]),
                    attention_mask=torch.ones((1, 3), dtype=torch.long),
                )

            def decode(self, ids, **kwargs):
                return "text"

        class Model:
            device = "cpu"
            config = SimpleNamespace(max_position_embeddings=100)
            generation_config = SimpleNamespace(eos_token_id=[2, 4])

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return torch.tensor([[1, 5, 6, 8, 2]])

        r = HFRunner.__new__(HFRunner)
        r.torch = torch
        r.tokenizer = Tokenizer()
        r.model = Model()
        r.judge = False
        r.configuration = {
            "generation": dict(
                max_new_tokens=2,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        }
        answer = r.generate([{"role": "user", "content": "Hi"}])
        self.assertEqual(answer["status"], "completed")
        self.assertEqual(answer["output_tokens"], 2)
        self.assertEqual(answer["input_ids"], [1, 5, 6])
        self.assertEqual(answer["output_ids"], [8, 2])
        self.assertFalse(r.tokenizer.kwargs["add_special_tokens"])
        self.assertIn("attention_mask", r.model.kwargs)
        r.model.generate = lambda **kwargs: torch.tensor([[1, 5, 6, 8, 9]])
        self.assertEqual(r.generate([])["status"], "truncated")

    def test_boundary_merge_rejected(self):
        from eval.dataset2_likelihood import continuation_logprobs

        class Tokenizer:
            def apply_chat_template(self, *a, **k):
                return "a"

            def __call__(self, text, **kwargs):
                return {"input_ids": [1] if text == "a" else [2, 3]}

        with self.assertRaisesRegex(ValueError, "boundary"):
            continuation_logprobs(None, Tokenizer(), "x", "b")

    @unittest.skipUnless(
        importlib.util.find_spec("nbformat"),
        "Install nbformat for notebook schema checks",
    )
    def test_notebook_schema(self):
        import nbformat

        for shard in [1, 2]:
            nbformat.validate(
                nbformat.read(
                    ROOT / f"eval/eval_dataset2_part{shard}.ipynb", as_version=4
                )
            )


class AutomaticBootstrapTests(unittest.TestCase):
    def test_each_notebook_extracts_current_assets_without_upload(self):
        import os
        import sys
        from eval.package_dataset2_colab import FILES

        prior_directory = Path.cwd()
        prior_path = list(sys.path)
        hashes = []
        try:
            with tempfile.TemporaryDirectory() as tmp:
                for shard in [1, 2]:
                    notebook = json.loads(
                        (ROOT / f"eval/eval_dataset2_part{shard}.ipynb").read_text()
                    )
                    source = "".join(notebook["cells"][3]["source"])
                    self.assertNotIn("files.upload", source)
                    target = Path(tmp) / f"part{shard}" / "ez-penta"
                    namespace = {
                        "REPO_DIR": target,
                        "Path": Path,
                        "DATASET2_NOTEBOOK_PATH": ROOT
                        / f"eval/eval_dataset2_part{shard}.ipynb",
                    }
                    with patch("subprocess.check_call") as install:
                        exec(compile(source, "embedded-setup", "exec"), namespace)
                        install.assert_not_called()
                        for name in FILES:
                            if not name.endswith(".ipynb"):
                                self.assertEqual(
                                    (target / name).read_bytes(),
                                    (ROOT / name).read_bytes(),
                                    name,
                                )
                        original_time = (
                            (target / "marcus/dataset2_partial.sqlite3")
                            .stat()
                            .st_mtime_ns
                        )
                        previous_runner = Mock()
                        namespace["aya"] = previous_runner
                        exec(compile(source, "embedded-setup", "exec"), namespace)
                        previous_runner.close.assert_called_once_with()
                        self.assertNotIn("aya", namespace)
                        self.assertEqual(
                            (target / "marcus/dataset2_partial.sqlite3")
                            .stat()
                            .st_mtime_ns,
                            original_time,
                        )
                    hashes.append(namespace["ASSETS_SHA256"])
                    self.assertEqual(
                        namespace["ASSETS_SHA256"],
                        notebook["metadata"]["dataset2_assets_sha256"],
                    )
                self.assertEqual(hashes[0], hashes[1])
        finally:
            os.chdir(prior_directory)
            sys.path[:] = prior_path


class MetadataSetupTests(unittest.TestCase):
    def test_small_code_cells_and_colab_metadata_response(self):
        import os
        import sys
        from types import ModuleType, SimpleNamespace

        notebook_path = ROOT / "eval/eval_dataset2_part1.ipynb"
        notebook = json.loads(notebook_path.read_text())
        for part in [1, 2]:
            doc = json.loads(
                (ROOT / f"eval/eval_dataset2_part{part}.ipynb").read_text()
            )
            self.assertTrue(
                all(
                    len("".join(c["source"])) < 15000
                    for c in doc["cells"]
                    if c["cell_type"] == "code"
                )
            )
            self.assertNotIn("PAYLOAD_PLACEHOLDER", "".join(doc["cells"][3]["source"]))
            self.assertIn("payload", doc["metadata"]["dataset2_assets"])
        fake = ModuleType("google.colab")
        requests = []

        def respond(request, **kwargs):
            requests.append((request, kwargs))
            return {"ipynb": notebook}

        fake._message = SimpleNamespace(blocking_request=respond)
        directory = Path.cwd()
        paths = list(sys.path)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "ez-penta"
                namespace = {"REPO_DIR": target}
                with patch.dict(sys.modules, {"google.colab": fake}), patch(
                    "subprocess.check_call"
                ):
                    exec(
                        compile(
                            "".join(notebook["cells"][3]["source"]),
                            "metadata-setup",
                            "exec",
                        ),
                        namespace,
                    )
                self.assertEqual(
                    (target / "marcus/dataset2_partial.sqlite3").read_bytes(),
                    DB.read_bytes(),
                )
                self.assertEqual(requests, [("get_ipynb", {"timeout_sec": 30})])
                fake._message = SimpleNamespace(blocking_request=lambda *a, **k: None)
                with patch.dict(sys.modules, {"google.colab": fake}), patch(
                    "subprocess.check_call"
                ):
                    with self.assertRaisesRegex(RuntimeError, "metadata"):
                        exec(
                            compile(
                                "".join(notebook["cells"][3]["source"]),
                                "metadata-setup",
                                "exec",
                            ),
                            {"REPO_DIR": target},
                        )
        finally:
            os.chdir(directory)
            sys.path[:] = paths

    def test_extra_unpinned_upgrade_cell_is_rejected_before_install(self):
        import os

        notebook = json.loads((ROOT / "eval/eval_dataset2_part1.ipynb").read_text())
        source = "".join(notebook["cells"][3]["source"])
        notebook["cells"].append(
            {
                "cell_type": "code",
                "source": ["!pip install --upgrade huggingface_hub transformers\n"],
            }
        )
        prior = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "edited.ipynb"
                path.write_text(json.dumps(notebook))
                target = Path(tmp) / "work"
                with patch("subprocess.check_call") as install:
                    with self.assertRaisesRegex(RuntimeError, "extra.*upgrade"):
                        exec(
                            compile(source, "setup", "exec"),
                            {"REPO_DIR": target, "DATASET2_NOTEBOOK_PATH": path},
                        )
                    install.assert_not_called()
                self.assertFalse(target.exists())
        finally:
            os.chdir(prior)


class NotebookImportRecoveryTests(unittest.TestCase):
    def test_cold_kernel_imports_outside_repo_and_missing_setup_is_actionable(self):
        import subprocess
        import sys

        for shard in [1, 2]:
            notebook = json.loads(
                (ROOT / f"eval/eval_dataset2_part{shard}.ipynb").read_text()
            )
            source = "".join(notebook["cells"][7]["source"])
            prefix = source.split("# End import-path recovery.\n", 1)[0]
            with tempfile.TemporaryDirectory() as tmp:
                child = (
                    "from pathlib import Path\nREPO_DIR=Path("
                    + repr(str(ROOT))
                    + ")\n"
                    + prefix
                    + "\nfrom eval.dataset2_data import load_dataset\nassert Path(__import__('eval').__file__).resolve().parent == REPO_DIR/'eval'\n"
                )
                completed = subprocess.run(
                    [sys.executable, "-I", "-c", child],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                with self.assertRaisesRegex(RuntimeError, "Automatic setup"):
                    exec(prefix, {"REPO_DIR": Path(tmp) / "not-extracted"})


class EnvironmentConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.requirements = Path(self.temporary.name) / "requirements.txt"
        self.requirements.write_text("huggingface_hub==0.36.2\n")

    def test_loaded_old_hub_stops_after_install(self):
        from eval import dataset2_environment as environment

        disk = {"version": "2.4.0"}

        def install(args):
            disk["version"] = "0.36.2"

        with patch.object(
            environment, "_loaded_versions", return_value={"huggingface_hub": "2.4.0"}
        ), patch.object(
            environment, "_installed", side_effect=lambda name: disk["version"]
        ):
            with patch.object(environment, "require_clean_environment") as probe:
                with self.assertRaisesRegex(
                    environment.RestartRequired, "huggingface_hub.*Restart session"
                ):
                    environment.install_dependencies(
                        self.requirements, installer=install
                    )
                probe.assert_not_called()

    def test_stale_child_module_stops_even_when_root_version_matches(self):
        from types import SimpleNamespace
        from eval import dataset2_environment as environment

        with patch.object(
            environment, "_loaded_versions", return_value={"huggingface_hub": "0.36.2"}
        ), patch.object(environment, "_installed", return_value="0.36.2"):
            with patch.object(
                environment.importlib,
                "import_module",
                side_effect=[
                    SimpleNamespace(),
                    ImportError("cannot import refresh_xet_connection_info"),
                ],
            ):
                with self.assertRaisesRegex(
                    environment.RestartRequired, "Restart session"
                ):
                    environment.require_clean_environment()

    def test_repeated_consistent_setup_never_installs(self):
        from eval import dataset2_environment as environment

        with patch.object(
            environment, "_installed", return_value="0.36.2"
        ), patch.object(environment, "require_clean_environment"):
            install = Mock()
            for repeat in range(2):
                self.assertFalse(
                    environment.install_dependencies(
                        self.requirements, installer=install
                    )
                )
            install.assert_not_called()

    def test_fresh_install_without_loaded_modules_needs_no_restart(self):
        from eval import dataset2_environment as environment

        disk = {"version": "not installed"}
        calls = []

        def install(args):
            calls.append(args)
            disk["version"] = "0.36.2"

        with patch.object(
            environment, "_installed", side_effect=lambda name: disk["version"]
        ), patch.object(environment, "_loaded_versions", return_value={}), patch.object(
            environment, "require_clean_environment"
        ):
            self.assertTrue(
                environment.install_dependencies(self.requirements, installer=install)
            )
            self.assertEqual(len(calls), 1)

    def test_repairing_disk_to_loaded_version_still_requires_restart(self):
        from eval import dataset2_environment as environment

        disk = {"version": "2.4.0"}

        def install(args):
            disk["version"] = "0.36.2"

        with patch.object(
            environment, "_loaded_versions", return_value={"huggingface_hub": "0.36.2"}
        ), patch.object(
            environment, "_installed", side_effect=lambda name: disk["version"]
        ):
            with self.assertRaises(environment.RestartRequired):
                environment.install_dependencies(self.requirements, installer=install)

    def test_cuda_local_version_satisfies_public_pin(self):
        from eval import dataset2_environment as environment

        self.requirements.write_text("torch==2.8.0\n")
        with patch.object(
            environment, "_installed", return_value="2.8.0+cu128"
        ), patch.object(environment, "require_clean_environment"):
            self.assertFalse(
                environment.install_dependencies(
                    self.requirements,
                    installer=lambda args: self.fail("CUDA wheel should not reinstall"),
                )
            )


class PackageTests(unittest.TestCase):
    def test_bundle_contains_current_source_and_dataset(self):
        import zipfile
        from eval.package_dataset2_colab import package, FILES

        with tempfile.TemporaryDirectory() as tmp:
            path = package(Path(tmp) / "bundle.zip")
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(
                    set(archive.namelist()), {"ez-penta/" + name for name in FILES}
                )
                for name in FILES:
                    self.assertEqual(
                        archive.read("ez-penta/" + name), (ROOT / name).read_bytes()
                    )


if __name__ == "__main__":
    unittest.main()
