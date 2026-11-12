import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.track2_extension.resume_integrity import (
    assert_local_arm_paths,
    assert_resume_compatible,
    file_tree_fingerprint,
    generation_key,
    judgment_key,
    validate_local_model_checkpoint,
)
from eval.track2_extension.colab_pilot_eval import (
    load_existing,
    load_model_for_arm,
    make_shared_generation_config,
    make_shared_model_load_config,
    merge_resume_files,
)
from transformers import GenerationConfig


class ResumeIntegrityTests(unittest.TestCase):
    def test_shared_generation_config_overrides_checkpoint_defaults(self):
        class Tokenizer:
            eos_token_id = 2
            bos_token_id = 1
            pad_token_id = None

        config = make_shared_generation_config(
            Tokenizer(),
            GenerationConfig(do_sample=True, num_beams=4, eos_token_id=9, pad_token_id=8),
        )
        self.assertFalse(config["do_sample"])
        self.assertEqual(config["num_beams"], 1)
        self.assertEqual(config["eos_token_id"], 2)
        self.assertEqual(config["bos_token_id"], 1)
        self.assertEqual(config["pad_token_id"], 2)
        self.assertEqual(config["min_new_tokens"], 0)

    def test_all_model_arms_get_one_dense_dtype_selected_by_device_support(self):
        for supported, expected_name, expected_dtype in (
            (True, "bfloat16", __import__("torch").bfloat16),
            (False, "float16", __import__("torch").float16),
        ):
            with self.subTest(bf16_supported=supported):
                provenance, load_dtype = make_shared_model_load_config(
                    bf16_supported=supported
                )
                self.assertEqual(load_dtype, expected_dtype)
                self.assertEqual(provenance["torch_dtype"], expected_name)
                self.assertEqual(provenance["weight_representation"], "dense")
                self.assertIsNone(provenance["quantization"])

    def test_baseline_and_local_checkpoints_use_identical_dense_loader(self):
        import torch

        with patch("eval.track2_extension.colab_pilot_eval.AutoModelForCausalLM.from_pretrained") as load:
            load_model_for_arm("CohereLabs/tiny-aya-water", compute_dtype=torch.float16)
            load_model_for_arm("/local/restricted", compute_dtype=torch.float16)
        calls = load.call_args_list
        self.assertEqual(calls[0].args, ("CohereLabs/tiny-aya-water",))
        self.assertEqual(calls[1].args, ("/local/restricted",))
        for call in calls:
            self.assertEqual(call.kwargs["torch_dtype"], torch.float16)
            self.assertEqual(call.kwargs["device_map"], "auto")
            self.assertNotIn("quantization_config", call.kwargs)
        self.assertEqual(calls[0].kwargs["revision"], "41a99216d12b3ef4881df762934186e5e4d2bbc1")
        self.assertNotIn("revision", calls[1].kwargs)

    def test_file_tree_fingerprint_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}", encoding="utf-8")
            first = file_tree_fingerprint(root)
            self.assertEqual(first, file_tree_fingerprint(root))
            (root / "config.json").write_text("{\"x\":1}", encoding="utf-8")
            self.assertNotEqual(first["sha256"], file_tree_fingerprint(root)["sha256"])

    def test_missing_local_arm_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            present = Path(tmp) / "adapter"
            present.mkdir()
            (present / "config.json").write_text("{}", encoding="utf-8")
            from safetensors.torch import save_file
            import torch

            save_file({"weight": torch.ones(1)}, str(present / "model.safetensors"))
            assert_local_arm_paths({"base": "org/model", "restricted": str(present)}, "org/model")
            with self.assertRaisesRegex(FileNotFoundError, "broad=.*missing"):
                assert_local_arm_paths({"base": "org/model", "broad": str(Path(tmp) / "missing")}, "org/model")

    def test_local_checkpoint_preflight_checks_config_and_safetensors_header(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
            save_file({"weight": torch.ones(2)}, str(root / "model.safetensors"))
            validate_local_model_checkpoint(root, arm="valid")
            (root / "model.safetensors").write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "invalid safetensors metadata"):
                validate_local_model_checkpoint(root, arm="truncated")

    def test_local_checkpoint_preflight_rejects_missing_config_or_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "lacks config.json"):
                validate_local_model_checkpoint(root)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                '{"weight_map":{"w":"model-00001-of-00002.safetensors"}}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "shard missing"):
                validate_local_model_checkpoint(root)

    def test_generation_key_changes_when_model_or_decoding_changes(self):
        args = dict(
            arm="baseline",
            model_fingerprint={"model": "m", "revision": "r1"},
            tokenizer_fingerprint={"template": "t"},
            prompt_id="p1",
            input_token_hash="tokens",
            generation_config={"max_new_tokens": 100},
        )
        key = generation_key(**args)
        self.assertEqual(key, generation_key(**args))
        self.assertNotEqual(key, generation_key(**{**args, "model_fingerprint": {"model": "m", "revision": "r2"}}))
        self.assertNotEqual(key, generation_key(**{**args, "generation_config": {"max_new_tokens": 200}}))

    def test_judgment_key_changes_when_answer_or_guard_revision_changes(self):
        args = dict(prompt_id="p1", answer_sha256="a1", guard_model="g", guard_revision="r1", guard_input_token_hash="t", guard_config={"max_new_tokens": 64})
        key = judgment_key(**args)
        self.assertNotEqual(key, judgment_key(**{**args, "answer_sha256": "a2"}))
        self.assertNotEqual(key, judgment_key(**{**args, "guard_revision": "r2"}))

    def test_resume_rejects_stale_duplicate_and_missing_provenance(self):
        expected = {"p1": "good"}
        self.assertEqual(assert_resume_compatible([{"prompt_id": "p1", "key": "good"}], expected, key_field="key", stage="generation"), {"p1"})
        for rows in (
            [{"prompt_id": "p1", "key": "old"}],
            [{"prompt_id": "p1", "key": "good"}, {"prompt_id": "p1", "key": "good"}],
            [{"prompt_id": "p2", "key": "good"}],
            [{"prompt_id": "p1"}],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                assert_resume_compatible(rows, expected, key_field="key", stage="generation")

    def test_resume_journal_merges_partial_and_final_without_losing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "run.jsonl"
            partial = Path(tmp) / "run.jsonl.partial"
            a = {"prompt_id": "p1", "generation_key": "k1", "answer": "a"}
            b = {"prompt_id": "p2", "generation_key": "k2", "answer": "b"}
            final.write_text(__import__("json").dumps(a) + "\n", encoding="utf-8")
            partial.write_text(__import__("json").dumps(b) + "\n", encoding="utf-8")
            merged = merge_resume_files(final, partial)
            self.assertEqual(set(merged), {"p1", "p2"})
            self.assertEqual(set(load_existing(partial)), {"p1", "p2"})

    def test_resume_journal_rejects_conflicting_duplicate_between_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "run.jsonl"
            partial = Path(tmp) / "run.jsonl.partial"
            final.write_text('{"prompt_id":"p1","answer":"a"}\n', encoding="utf-8")
            partial.write_text('{"prompt_id":"p1","answer":"different"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "conflicting final/partial"):
                merge_resume_files(final, partial)

    def test_resume_journal_recovers_only_an_unterminated_partial_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "run.jsonl"
            partial = Path(tmp) / "run.jsonl.partial"
            final.write_text('{"prompt_id":"p1","answer":"a"}\n', encoding="utf-8")
            partial.write_bytes(b'{"prompt_id":"p2","answer":"b"}\n{"prompt_id":"p3","ans')
            merged = merge_resume_files(final, partial)
            self.assertEqual(set(merged), {"p1", "p2"})
            self.assertEqual(set(load_existing(partial)), {"p1", "p2"})

    def test_resume_journal_removes_torn_fragment_when_it_is_the_only_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "run.jsonl"
            partial = Path(tmp) / "run.jsonl.partial"
            partial.write_bytes(b'{"prompt_id":"p1","answer":')
            self.assertEqual(merge_resume_files(final, partial), {})
            self.assertEqual(partial.read_bytes(), b"")

    def test_resume_journal_rejects_malformed_completed_or_interior_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            path.write_bytes(b'{"prompt_id":"p1"}\nnot-json\n')
            with self.assertRaisesRegex(ValueError, "line 2"):
                load_existing(path, tolerate_torn_tail=True)
            path.write_bytes(b'{"prompt_id":"p1"}\nnot-json')
            with self.assertRaisesRegex(ValueError, "invalid JSONL"):
                load_existing(path)


if __name__ == "__main__":
    unittest.main()
