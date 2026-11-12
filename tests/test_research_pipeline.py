"""Research notebook mechanics, exercised with original toy cases and no model weights."""

from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
from types import ModuleType
import os
import sqlite3
import sys
import unittest
from unittest.mock import patch

import auditkit as ak
import numpy as np
import pandas as pd

from eval.analysis import build_results
from eval.checkpoints import ANSWER_KEY, append_records, case_id, config_id, fingerprint, generation_status, load_jsonl
from eval.scoring import score_answers, score_stage
from data_ready.rewrite_helpers import run_rewrites


def toy_answer(source="one", language="ja", **changes):
    return {"source_id": source, "source_split": "dev", "variant": "original", "language": language,
            "prompt": f"Example question {source} {language}", "reference": "Example safe reference",
            "answer": "Example answer", "truncated": False, "error": None, **changes}


def toy_score(row, label="Safe", refusal="No"):
    return {"stage": "answer", "sample_id": case_id(row),
            "fingerprint": fingerprint(row["prompt"], row["answer"], row["reference"]),
            "scores": {"guard_response": {"value": float(label == "Unsafe"),
                       "metadata": {"label": label, "refusal": refusal, "categories": [],
                                    "unknown": label in ("Controversial", "Unparsed")}}}, "context": {}}


class ToyMetric:
    def __init__(self, name="guard_response", revision="one"):
        self.name, self.revision = name, revision

    def identity(self):
        return {"name": self.name, "revision": self.revision}


class ResearchCheckpointTests(unittest.TestCase):
    def test_scalar_booleans_missing_flags_and_errors_are_distinct(self):
        self.assertEqual(generation_status(toy_answer(truncated=np.bool_(False))), "completed")
        self.assertEqual(generation_status(toy_answer(truncated=np.bool_(True))), "truncated")
        for value in (None, pd.NA, np.nan):
            self.assertEqual(generation_status(toy_answer(truncated=value)), "missing")
        self.assertEqual(generation_status(toy_answer(answer=None, error="GPU failure")), "error")
        self.assertEqual(generation_status(toy_answer(answer="  ")), "missing")

    def test_checkpoint_append_preserves_unicode_and_last_line_separator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            path.write_text('{"id": "first", "answer": "こんにちは"}', encoding="utf-8")
            append_records(path, [{"id": "second", "answer": "回答"}])
            self.assertEqual(len(load_jsonl(path, ("id",))), 2)
            self.assertIn("こんにちは", path.read_text())
            path.write_text('{"id":', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint.jsonl:1"):
                load_jsonl(path, ("id",))


class DrivePathTests(unittest.TestCase):
    def setup_code(self, root):
        project = Path(__file__).resolve().parents[1]
        notebook = json.loads((project / "eval/eval_jp_en_variants.ipynb").read_text())
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if "if ON_COLAB and (FORCE_DRIVE_REMOUNT" in "".join(cell["source"]))
        source = source[source.index("if ON_COLAB and (FORCE_DRIVE_REMOUNT"):]
        return source.split('if MODE == "saved":', 1)[0].replace("/content/drive", str(root))

    def test_mounted_shortcut_target_by_id_rebases_all_input_paths(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            (root / "MyDrive").mkdir()
            target = root / ".shortcut-targets-by-id" / "shared-id" / "ez_penta"
            target.mkdir(parents=True)
            for name in ("answercarefully_eval.sqlite3", "eval_aya_answers.jsonl", "eval_scores.jsonl"):
                (target / name).touch()
            namespace = {"Path": Path, "DATA_DIR": root / "MyDrive" / "ez_penta", "ON_COLAB": True,
                         "FORCE_DRIVE_REMOUNT": False, "SHARED_FOLDER_ID": "shared-id", "DRIVE_FOLDER": "ez_penta"}
            exec(self.setup_code(root), namespace)
            self.assertEqual(namespace["DATA_DIR"], target)
            self.assertEqual(namespace["DB_SAVED"], target / "answercarefully_eval.sqlite3")

    def test_missing_path_reports_visible_names_without_accepting_an_empty_alias(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            (root / "MyDrive" / "ez-penta").mkdir(parents=True)
            desired = root / "MyDrive" / "ez_penta"
            namespace = {"Path": Path, "DATA_DIR": desired, "ON_COLAB": True,
                         "FORCE_DRIVE_REMOUNT": False, "SHARED_FOLDER_ID": "shared-id", "DRIVE_FOLDER": "ez_penta"}
            with self.assertRaisesRegex(FileNotFoundError, "ez-penta"):
                exec(self.setup_code(root), namespace)
            self.assertFalse(desired.exists())

    def test_hyphen_alias_with_expected_files_is_used_and_cache_folder_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            target = root / "MyDrive" / "ez-penta"
            target.mkdir(parents=True)
            (root / "MyDrive" / "ez-penta-cache").mkdir()
            for name in ("answercarefully_eval.sqlite3", "eval_aya_answers.jsonl", "eval_scores.jsonl"):
                (target / name).touch()
            namespace = {"Path": Path, "DATA_DIR": root / "MyDrive" / "ez_penta", "ON_COLAB": True,
                         "FORCE_DRIVE_REMOUNT": False, "SHARED_FOLDER_ID": "shared-id", "DRIVE_FOLDER": "ez_penta"}
            exec(self.setup_code(root), namespace)
            self.assertEqual(namespace["DATA_DIR"], target)
            self.assertEqual(namespace["SCORES_SAVED"], target / "eval_scores.jsonl")


class ResearchScoringTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

    def evaluate(self, samples, *, scorers, **kwargs):
        self.calls.append(samples)
        predictions = [SimpleNamespace(sample_id=sample.id, context={}, metadata={"scores": [
            {"name": scorer.name, "value": 0.0, "metadata": {"label": "Safe", "refusal": "No"}}
            for scorer in scorers]}) for sample in samples]
        return SimpleNamespace(predictions=predictions, errors=[])

    def test_only_two_complete_answers_enter_pair_scoring(self):
        rows = [toy_answer(source, language, truncated=np.bool_(source == "cut" and language == "ja"))
                for source in ("complete", "cut") for language in ("ja", "en")]
        rows.append(toy_answer("missing-language", "ja"))
        with tempfile.TemporaryDirectory() as directory:
            individual, pairs = score_answers(pd.DataFrame(rows), ["original"], [ToyMetric()],
                                              [ToyMetric("cosine_similarity")], Path(directory) / "scores.jsonl",
                                              {"guard_model": "toy"}, evaluate=self.evaluate)
        self.assertEqual(len(individual), 4)
        self.assertEqual(list(pairs), ["complete|dev|original"])
        pair_sample = self.calls[-1][0]
        self.assertEqual(pair_sample.actual_output, "Example answer")
        self.assertEqual(pair_sample.target, "Example answer")

    def test_resume_invalidates_model_metric_and_text_changes(self):
        sample = ak.Sample(id="one", input="Question", actual_output="Answer", target="Reference")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            def run(model="guard-one", metric_revision="one"):
                return score_stage("answer", [sample], [ToyMetric(revision=metric_revision)], path,
                                   {"judge_model": model}, evaluate=self.evaluate)
            run()
            run()
            self.assertEqual(len(self.calls), 1)
            run(model="guard-two")
            run(model="guard-two", metric_revision="two")
            sample.actual_output = "A different answer"
            run(model="guard-two", metric_revision="two")
            self.assertEqual(len(self.calls), 4)
            self.assertEqual(len(path.read_text().splitlines()), 4)

    def test_failed_scoring_is_not_a_safe_verdict_and_is_retried(self):
        sample = ak.Sample(id="one", input="Q", actual_output="A", target="R")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            failed = score_stage("answer", [sample], [ToyMetric()], path, {},
                                 evaluate=lambda *args, **kwargs: SimpleNamespace(predictions=[], errors=["failed"]))
            self.assertEqual(failed["one"]["scoring_status"], "error")
            self.assertNotIn("guard_response", failed["one"]["scores"])
            recovered = score_stage("answer", [sample], [ToyMetric()], path, {}, evaluate=self.evaluate)
            self.assertEqual(recovered["one"]["scoring_status"], "completed")

    def test_real_auditkit_precomputed_path_needs_no_model(self):
        sample = ak.Sample(id="one", input="Example", actual_output="same answer", target="same answer")
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"XDG_CACHE_HOME": directory}):
            result = score_stage("pair", [sample], [ak.ChrF()], Path(directory) / "scores.jsonl", {"protocol": "toy"})
            self.assertEqual(result["one"]["scoring_status"], "completed")
            self.assertAlmostEqual(result["one"]["scores"]["chrf"]["value"], 1.0)


class ResearchAnalysisTests(unittest.TestCase):
    def test_expected_grid_preserves_missing_languages_errors_and_unknown_labels(self):
        rows = [toy_answer("complete", language) for language in ("ja", "en")]
        rows += [toy_answer("cut", language, truncated=language == "ja") for language in ("ja", "en")]
        rows += [toy_answer("missing-language", "ja")]
        rows += [toy_answer("error", "ja", answer=None, error="GPU failure"), toy_answer("error", "en")]
        rows += [toy_answer("unknown", language) for language in ("ja", "en")]
        sources = pd.DataFrame([{"source_id": source, "source_split": "dev", "risk_area_en": "Example"}
                                for source in ("complete", "cut", "missing-language", "error", "unknown")])
        scores = {case_id(row): toy_score(row, "Controversial" if row["source_id"] == "unknown" and row["language"] == "ja" else "Safe")
                  for row in rows if generation_status(row) == "completed"}
        # Historical similarity for an incomplete pair must be ignored even if it exists.
        result = build_results(pd.DataFrame(rows), sources, ["original"], scores,
                               {"cut|dev|original": {"fingerprint": "invalid", "scores": {"cosine_similarity": {"value": 0.99}}}})
        pairs, table = result["pairs"], result["table"]
        self.assertEqual(len(pairs), 5)
        self.assertEqual(len(table), 10)
        self.assertEqual(int(pairs["answers_complete"].sum()), 2)
        self.assertEqual(int(pairs["safety_known"].sum()), 1)
        self.assertEqual(int(pairs["refusal_known"].sum()), 2)
        unknown = table[(table.source_id == "unknown") & (table.language == "ja")].iloc[0]
        self.assertEqual(unknown.safety, "Controversial")
        self.assertTrue(pd.isna(unknown.unsafe))
        self.assertEqual(set(table.generation_status), {"completed", "truncated", "missing", "error"})
        summary = result["by_variant_language"].loc[("original", "ja")]
        self.assertEqual(summary["unsafe_rate_n"], 2)
        self.assertEqual(summary["refusal_rate_n"], 3)
        self.assertTrue(pairs["similarity_ja_en"].isna().all())

    def test_empty_binary_cohort_has_no_measured_zero_or_spurious_p_value(self):
        rows = [toy_answer(language=language) for language in ("ja", "en")]
        sources = pd.DataFrame([{"source_id": "one", "source_split": "dev"}])
        scores = {case_id(row): toy_score(row, "Unparsed") for row in rows}
        result = build_results(pd.DataFrame(rows), sources, ["original"], scores, {})
        self.assertEqual(result["by_variant"].loc["original", "unsafe_pairs"], 0)
        self.assertTrue(pd.isna(result["by_variant"].loc["original", "unsafe_mcnemar_p_adjusted"]))
        self.assertTrue(result["by_variant_language"]["unsafe_rate"].isna().all())

    def test_stale_score_fingerprints_are_rejected(self):
        rows = [toy_answer(language=language) for language in ("ja", "en")]
        score = toy_score(rows[0])
        score["fingerprint"] = "different-answer"
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_results(pd.DataFrame(rows), pd.DataFrame([{"source_id": "one", "source_split": "dev"}]),
                          ["original"], {case_id(rows[0]): score}, {})

    def test_notebook_code_cells_compile_without_a_model(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("eval/eval_jp_en_variants.ipynb", "data_ready/prepare_prompt_variants.ipynb",
                     "data_ready/view/compile_test_prompts.ipynb"):
            notebook = json.loads((root / name).read_text())
            for i, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    source = "\n".join(line for line in "".join(cell["source"]).splitlines() if not line.startswith("%"))
                    compile(source, f"{name}:cell{i}", "exec")

    def test_saved_mode_notebook_runs_end_to_end_without_model_or_network_calls(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "eval/eval_jp_en_variants.ipynb").read_text())
        rows = [toy_answer(language=language) for language in ("ja", "en")]
        pair = {"source_id": "one", "source_split": "dev", "variant": "original",
                "risk_area_en": "Example", "harm_type_en": "Example"}
        score_records = [toy_score(row) for row in rows]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            database = directory / "answercarefully_eval.sqlite3"
            connection = sqlite3.connect(database)
            pd.DataFrame(rows).to_sql("answers", connection, index=False)
            pd.DataFrame([pair]).to_sql("pairs", connection, index=False)
            pd.DataFrame([{"aya_model": "toy", "rows": 1}]).to_sql("eval_info", connection, index=False)
            connection.close()
            answers_path, scores_path = directory / "eval_aya_answers.jsonl", directory / "eval_scores.jsonl"
            append_records(answers_path, rows)
            append_records(scores_path, score_records)
            before = {p.name: p.read_bytes() for p in (database, answers_path, scores_path)}
            display = ModuleType("IPython.display")
            display.display = lambda *args, **kwargs: None
            namespace = {}
            working_directory = Path.cwd()
            try:
                os.chdir(root)
                with patch.dict(sys.modules, {"IPython.display": display}), patch(
                    "subprocess.run", side_effect=AssertionError("Saved mode must not install or start inference")
                ), redirect_stdout(io.StringIO()):
                    for cell in notebook["cells"]:
                        if cell["cell_type"] != "code":
                            continue
                        source = "\n".join(line for line in "".join(cell["source"]).splitlines() if not line.startswith("%"))
                        exec(source, namespace)
                        if source.startswith("from pathlib import Path"):
                            namespace.update({"DATA_DIR": directory, "ON_COLAB": False,
                                              "DB_SAVED": database, "ANSWERS_SAVED": answers_path,
                                              "SCORES_SAVED": scores_path, "VARIANTS": {"original": ("text", "text_en")}})
                    self.assertEqual(len(namespace["results"]["pairs"]), 1)
                    self.assertTrue(namespace["DB_OUT"].is_file())
            finally:
                os.chdir(working_directory)
            self.assertEqual(before, {p.name: p.read_bytes() for p in (database, answers_path, scores_path)})


class RewritePreparationTests(unittest.TestCase):
    def test_resume_tracks_source_messages_and_configuration(self):
        rows = pd.DataFrame([{"source_id": "one", "source_split": "dev", "text": "日本語", "text_en": "English"}])
        languages = {"ja": ("text", "Japanese"), "en": ("text_en", "English")}
        calls = []

        def generate(messages):
            calls.append(messages)
            return ["Rewritten prompt"] * len(messages), [False] * len(messages)

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            checkpoint = Path(directory) / "rewrites.jsonl"
            def run(configuration):
                return run_rewrites(rows, languages, lambda prompt, language: [{"role": "user", "content": prompt}],
                                    checkpoint, configuration, generate=generate, count_tokens=lambda messages: 10)
            run({"model": "one"})
            run({"model": "one"})
            self.assertEqual(len(calls), 1)
            rows.loc[0, "text_en"] = "Changed English"
            run({"model": "one"})
            self.assertEqual(len(calls[-1]), 1)
            run({"model": "two"})
            self.assertEqual(len(calls[-1]), 2)

    def test_compilation_keeps_quality_flags_and_source_metadata(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "data_ready/view/compile_test_prompts.ipynb").read_text())
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root_dir = Path(directory)
            work = root_dir / "work"
            work.mkdir()
            base = {"source_id": "one", "source_split": "dev", "text": "Example Japanese", "text_en": "Example English", "output": "Reference"}
            simple = work / "simple.sqlite3"
            paths = {strategy: work / f"{strategy}.sqlite3" for strategy in ("fc", "rp", "rs")}
            for name, path in [("simple", simple), *paths.items()]:
                values = dict(base)
                if name == "simple":
                    values["flake_example"] = "Altered prompt"
                    columns = "flake_example"
                else:
                    columns = f"Old_{name}_ja, Old_{name}_en"
                    for language in ("ja", "en"):
                        values[f"Old_{name}_{language}"] = "Rewritten prompt"
                        values[f"Old_{name}_{language}_truncated"] = name == "fc" and language == "ja"
                connection = sqlite3.connect(path)
                pd.DataFrame([values]).to_sql("records", connection, index=False)
                pd.DataFrame([{"columns": columns, "model": "toy-model"}]).to_sql("generation_info", connection, index=False)
                pd.DataFrame([{"revision": "frozen-source-revision"}]).to_sql("snapshot_info", connection, index=False)
                connection.close()
            namespace = {"WORK_DIR": work, "SIMPLE_DB": simple, "QWEN_DBS": paths, "KEY": ["source_id", "source_split"],
                         "SHARED": ["text", "text_en", "output"], "DB_OUT": root_dir / "compiled_v2.sqlite3",
                         "shutil": __import__("shutil"), "display": lambda *args: None}
            exec("".join(notebook["cells"][4]["source"]), namespace)
            exec("".join(notebook["cells"][6]["source"]), namespace)
            exec("".join(notebook["cells"][9]["source"]), namespace)
            connection = sqlite3.connect(namespace["DB_OUT"])
            self.assertEqual(connection.execute("SELECT fc_ja_truncated,fc_en_truncated FROM records").fetchone(), (1, 0))
            metadata = connection.execute("SELECT generation_info_json,snapshot_info_json,sha256 FROM compile_info").fetchall()
            self.assertEqual(len(metadata), 4)
            self.assertTrue(all("toy-model" in row[0] and "frozen-source-revision" in row[1] and len(row[2]) == 64 for row in metadata))
            connection.close()
