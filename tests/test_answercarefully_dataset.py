import unittest

from answercarefully_dataset import select_samples


def original_row(identifier, risk_area):
    return {
        "ID": identifier, "text": f"日本語の質問 {identifier}", "output": "参考回答",
        "_source_split": "dev",
        "meta": {"risk-area": risk_area, "harm-type": "example", "specific-harm": "example"},
        "meta-mlmc": {"text-English": "Example question", "adaptation-tag": "0", "translation-notes": ""},
    }


class DatasetTests(unittest.TestCase):
    def test_selection_preserves_reference_and_metadata_across_risk_areas(self):
        rows = [original_row("first", "area-one"), original_row("duplicate-area", "area-one"),
                original_row("second", "area-two")]
        rows[2]["_source_split"] = "test"
        samples = select_samples(rows, "pinned-revision")
        self.assertEqual([sample.id for sample in samples], ["first:ja", "duplicate-area:ja", "second:ja"])
        self.assertEqual([sample.metadata["split"] for sample in samples], ["dev", "dev", "test"])
        self.assertEqual(samples[0].input, rows[0]["text"])
        self.assertEqual(samples[0].target, rows[0]["output"])
        self.assertEqual(samples[0].metadata["translation"], rows[0]["meta-mlmc"])
        self.assertEqual(samples[0].metadata["dataset_revision"], "pinned-revision")
        self.assertIsNone(samples[0].actual_output)

    def test_explicit_limit_selects_rows_not_risk_areas(self):
        rows = [original_row(str(index), "same-area") for index in range(8)]
        self.assertEqual(len(select_samples(rows, "revision", 6)), 6)
        self.assertEqual(len(select_samples(rows, "revision", 20)), 8)

    def test_english_uses_translation_and_keeps_paired_source_and_japanese_reference(self):
        rows = [original_row("first", "area"), original_row("second", "area")]
        samples = select_samples(rows, "revision", 1, language="en")
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample.id, "first:en")
        self.assertEqual(sample.input, rows[0]["meta-mlmc"]["text-English"])
        self.assertEqual(sample.target, rows[0]["output"])
        self.assertEqual(sample.metadata["language"], "en")
        self.assertEqual(sample.metadata["reference_language"], "ja")
        self.assertEqual(sample.metadata["source_id"], "first")
        self.assertEqual(sample.metadata["original_prompt_ja"], rows[0]["text"])
        self.assertEqual(sample.metadata["translation"], rows[0]["meta-mlmc"])
        self.assertEqual(rows[0]["text"], "日本語の質問 first")

    def test_missing_english_never_silently_falls_back_to_japanese(self):
        for value in (None, "", "   ", 123):
            row = original_row("missing", "area")
            row["meta-mlmc"]["text-English"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "en prompt for missing"):
                select_samples([row], "revision", language="en")
        row["meta-mlmc"] = None
        with self.assertRaisesRegex(ValueError, "en prompt for missing"):
            select_samples([row], "revision", language="en")
        with self.assertRaisesRegex(ValueError, "language"):
            select_samples([original_row("first", "area")], "revision", language="mixed")

    def test_empty_data_and_invalid_limits_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no examples"):
            select_samples([], "revision")
        for limit in (0, -1):
            with self.assertRaisesRegex(ValueError, "positive"):
                select_samples([original_row("one", "area")], "revision", limit)
