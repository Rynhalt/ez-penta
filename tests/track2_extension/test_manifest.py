import hashlib
import tempfile
import unittest
from pathlib import Path

from eval.track2_extension.data_io import iter_observations, write_jsonl_atomic
from eval.track2_extension.manifest import build_matched_candidates, validate_candidate_manifest


def row(prompt_id, culture, **overrides):
    value = {
        "prompt_id": prompt_id,
        "seed_id": "seed-1",
        "split": "heldout",
        "kind": "harmful",
        "category": "violence",
        "framing": "control",
        "prompt_language": "en",
        "mix_direction": "none",
        "output_language": "en",
        "culture": culture,
        "prompt": f"synthetic prompt {prompt_id}",
        "analysis_eligible": True,
        "generation_status": "completed",
        "scoring_status": "completed",
        "safety": "Safe",
        "identical_twin_flag": False,
        "duplicate_text_flag": False,
        "expected_behavior_conflict": False,
        "expected": "harmful",
    }
    value.update(overrides)
    return value


class CandidateManifestTests(unittest.TestCase):
    def test_pairs_exact_metadata_and_never_asserts_human_approval(self):
        candidates, counts = build_matched_candidates(
            [row("n1", "neutral", mix_direction=None), row("u1", "us", mix_direction=None)]
        )
        self.assertEqual(counts["candidate_prompt_rows"], 2)
        self.assertEqual(counts["candidate_pairs"], 1)
        self.assertEqual(counts["unique_candidate_seeds"], 1)
        self.assertTrue(all(r["leakage_flag"] for r in candidates))
        self.assertTrue(all(r["human_prompt_valid"] is None for r in candidates))
        self.assertTrue(all(r["human_unsafe_compliance"] is None for r in candidates))
        self.assertTrue(all(not r["confirmatory_eligible"] for r in candidates))
        self.assertEqual(
            candidates[0]["prompt_sha256"],
            hashlib.sha256(candidates[0]["prompt"].encode()).hexdigest(),
        )

    def test_excludes_unmatched_duplicate_and_invalid_source_rows(self):
        candidates, counts = build_matched_candidates(
            [
                row("n1", "neutral"),
                row("u1", "us"),
                row("n2", "neutral", seed_id="seed-2"),
                row("u2", "us", seed_id="seed-2", generation_status="truncated"),
                row("n3", "neutral", seed_id="seed-3", identical_twin_flag=True),
                row("u3", "us", seed_id="seed-3"),
            ]
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(counts["source_generation_not_completed"], 1)
        self.assertEqual(counts["identical_twin_excluded"], 1)
        self.assertEqual(counts["non_unique_or_unmatched_strata"], 2)

    def test_rejects_duplicate_ids_and_synthetic_human_approval(self):
        rows, _ = build_matched_candidates([row("n1", "neutral"), row("u1", "us")])
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_candidate_manifest(rows + [rows[0]])
        rows[0]["human_prompt_valid"] = True
        with self.assertRaisesRegex(ValueError, "human review"):
            validate_candidate_manifest(rows)

    def test_readonly_sqlite_iterator_and_atomic_jsonl(self):
        import sqlite3

        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "source.sqlite3"
            connection = sqlite3.connect(db)
            fields = {
                "prompt_id": "TEXT", "seed_id": "TEXT", "split": "TEXT", "kind": "TEXT",
                "category": "TEXT", "culture": "TEXT", "framing": "TEXT",
                "prompt_language": "TEXT", "mix_direction": "TEXT", "output_language": "TEXT",
                "prompt": "TEXT", "analysis_eligible": "INTEGER", "generation_status": "TEXT",
                "scoring_status": "TEXT", "safety": "TEXT", "identical_twin_flag": "INTEGER",
                "duplicate_text_flag": "INTEGER", "expected_behavior_conflict": "INTEGER",
                "expected": "TEXT",
            }
            connection.execute("CREATE TABLE observations (" + ",".join(f'"{k}" {v}' for k, v in fields.items()) + ")")
            connection.execute(
                "INSERT INTO observations VALUES (" + ",".join("?" for _ in fields) + ")",
                tuple(row("p1", "neutral")[key] for key in fields),
            )
            connection.commit()
            connection.close()
            rows = list(iter_observations(db))
            self.assertEqual(len(rows), 1)
            out = Path(temp) / "private" / "manifest.jsonl"
            self.assertEqual(write_jsonl_atomic(out, iter(rows)), 1)
            self.assertEqual(out.read_text().count("\n"), 1)


if __name__ == "__main__":
    unittest.main()
