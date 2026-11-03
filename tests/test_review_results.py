from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

from review_results import ReviewWorkspace, make_handler


def create_run(root, run_id="example-run"):
    directory = root / run_id
    directory.mkdir()
    records = []
    predictions = []
    for identifier, label in (("safe", "Safe"), ("unsafe", "Unsafe"), ("controversial", "Controversial"),
                              ("excluded", "Excluded"), ("error", "Error"), ("mismatch", "Safe"), ("pending", "Pending")):
        prompt = f"日本語の質問 {identifier} <script>alert('test')</script>"
        answer = f"保存した回答 {identifier}"
        sample = {"id": identifier, "input": prompt, "actual_output": answer, "target": "参考回答",
                  "metadata": {"split": "dev", "categories": {"risk-area": "テスト領域"},
                               "translation": {"text-English": "English translation"}}}
        records.append({"sample": sample, "generation": {"finish_reason": "length" if label == "Excluded" else "stop"}})
        if label not in ("Excluded", "Pending"):
            score = {"name": "guard_judge:test", "value": float(label == "Unsafe"),
                     "metadata": {"label": label, "unknown": label == "Controversial", "refusal": "Yes", "raw": f"Safety: {label}"}}
            predictions.append({"sample_id": identifier, "prompt": prompt,
                                "raw_output": answer if identifier != "mismatch" else "Different generation",
                                "metadata": {"scores": [] if label == "Error" else [score]}})
    (directory / "tiny-aya-responses.jsonl").write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding="utf-8")
    (directory / "auditkit-guard-results.json").write_text(json.dumps({"predictions": predictions}), encoding="utf-8")
    (directory / "summary.json").write_text("{}", encoding="utf-8")
    return directory


class ReviewWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = create_run(self.root)
        self.workspace = ReviewWorkspace(self.root)

    def review(self, reviewer="Alice", label="over_refusal"):
        return {"sample_id": "safe", "content_hash": self.workspace.records("example-run")["safe"]["content_hash"],
                "reviewer": reviewer, "label": label, "notes": "The answer unnecessarily refuses a benign request."}

    def test_all_responses_are_visible_and_unscored_are_not_safe(self):
        result = self.workspace.results("example-run", {})
        self.assertEqual(result["stats"]["total"], 7)
        self.assertEqual(result["stats"]["scored"], 2)
        self.assertEqual(result["stats"]["unsafe_rate"], 0.5)
        self.assertEqual(result["stats"]["guard_counts"]["Excluded"], 1)
        self.assertEqual(result["stats"]["guard_counts"]["Mismatched"], 1)
        self.assertEqual(len(result["rows"]), 7)
        self.assertEqual(self.workspace.record("example-run", "excluded")["answer"], "保存した回答 excluded")

    def test_search_filters_and_pagination(self):
        self.assertEqual(self.workspace.results("example-run", {"verdict": "Unsafe"})["total"], 1)
        self.assertEqual(self.workspace.results("example-run", {"query": "English translation"})["total"], 7)
        self.assertEqual(self.workspace.results("example-run", {"query": "日本語の質問"})["total"], 7)
        self.assertEqual(len(self.workspace.results("example-run", {"offset": "2", "limit": "2"})["rows"]), 2)
        self.assertEqual(self.workspace.results("example-run", {"refusal": "Yes"})["total"], 3)

    def test_reviews_are_separate_per_reviewer_and_persist_after_reopening(self):
        original = (self.run / "tiny-aya-responses.jsonl").read_bytes()
        self.workspace.save_review("example-run", self.review())
        self.workspace.save_review("example-run", self.review("Bob", "appropriate_refusal"))
        reopened = ReviewWorkspace(self.root)
        self.assertEqual(len(reopened.record("example-run", "safe")["reviews"]), 2)
        self.assertEqual(reopened.results("example-run", {"review": "over_refusal"})["total"], 1)
        self.assertEqual(reopened.results("example-run", {"review": "over_refusal", "reviewer": "Bob"})["total"], 0)
        self.assertEqual(reopened.results("example-run", {"reviewer": "Alice"})["stats"]["over_refusals"], 1)
        self.assertEqual((self.run / "tiny-aya-responses.jsonl").read_bytes(), original)

    def test_changed_answers_invalidate_old_reviews(self):
        review = self.review()
        self.workspace.save_review("example-run", review)
        path = self.run / "tiny-aya-responses.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["sample"]["actual_output"] = "A different regenerated response"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.assertEqual(self.workspace.record("example-run", "safe")["reviews"], [])
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.workspace.save_review("example-run", review)

    def test_exports_contain_annotations_not_original_text_and_imports_preserve_conflicts(self):
        self.workspace.save_review("example-run", self.review())
        exported = self.workspace.export_reviews("example-run")
        self.assertNotIn("日本語の質問", json.dumps(exported, ensure_ascii=False))
        self.assertNotIn("保存した回答", json.dumps(exported, ensure_ascii=False))
        imported = deepcopy(exported)
        imported["reviews"][0]["label"] = "appropriate_answer"
        imported["reviews"].append({**self.review("Bob"), "updated_at": "unused"})
        self.assertEqual(self.workspace.import_reviews("example-run", imported), {"imported": 1, "unchanged": 0, "conflicts": 1})
        alice = next(review for review in self.workspace.record("example-run", "safe")["reviews"] if review["reviewer"] == "Alice")
        self.assertEqual(alice["label"], "over_refusal")

    def test_invalid_import_is_atomic_and_paths_cannot_escape_runs(self):
        imported = {"schema": "ez-penta-reviews-v1", "reviews": [self.review(), {**self.review("Bob"), "content_hash": "wrong"}]}
        with self.assertRaises(ValueError):
            self.workspace.import_reviews("example-run", imported)
        self.assertEqual(self.workspace.record("example-run", "safe")["reviews"], [])
        for run_id in ("..", "../outside", str(self.root.parent)):
            with self.assertRaises(ValueError):
                self.workspace.records(run_id)


class ReviewHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        create_run(root)
        self.workspace = ReviewWorkspace(root)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.workspace))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()

    def test_static_page_utf8_api_and_write_round_trip(self):
        with urlopen(self.base_url) as response:
            self.assertIn(b"Response review", response.read())
        with urlopen(self.base_url + "/api/record?run=example-run&id=safe") as response:
            record = json.load(response)
            self.assertIn("日本語", record["prompt"])
        payload = {"run": "example-run", "sample_id": "safe", "content_hash": record["content_hash"],
                   "reviewer": "Teammate", "label": "over_refusal", "notes": "Review note"}
        request = Request(self.base_url + "/api/review", json.dumps(payload).encode(),
                          {"Content-Type": "application/json", "X-Review-App": "1"})
        with urlopen(request) as response:
            self.assertEqual(json.load(response)["review"]["label"], "over_refusal")

    def test_cross_origin_writes_and_arbitrary_file_reads_are_rejected(self):
        request = Request(self.base_url + "/api/review", b"{}", {
            "Content-Type": "application/json", "X-Review-App": "1", "Origin": "https://unrelated.example",
        })
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 403)
        with self.assertRaises(HTTPError) as error:
            urlopen(self.base_url + "/../requirements.txt")
        self.assertEqual(error.exception.code, 404)
