import unittest

from answercarefully_dataset import select_samples


def original_row(identifier, risk_area):
    return {
        "ID": identifier, "text": f"日本語の質問 {identifier}", "output": "参考回答",
        "meta": {"risk-area": risk_area, "harm-type": "example", "specific-harm": "example"},
        "meta-mlmc": {"text-English": "Example question", "adaptation-tag": "0", "translation-notes": ""},
    }


class DatasetTests(unittest.TestCase):
    def test_selection_preserves_reference_and_metadata_across_risk_areas(self):
        rows = [original_row("first", "area-one"), original_row("duplicate-area", "area-one"),
                original_row("second", "area-two")]
        samples = select_samples(rows, "pinned-revision", 2)
        self.assertEqual([sample.id for sample in samples], ["first:ja", "second:ja"])
        self.assertEqual(samples[0].input, rows[0]["text"])
        self.assertEqual(samples[0].target, rows[0]["output"])
        self.assertEqual(samples[0].metadata["translation"], rows[0]["meta-mlmc"])
        self.assertEqual(samples[0].metadata["dataset_revision"], "pinned-revision")
        self.assertIsNone(samples[0].actual_output)

    def test_insufficient_risk_areas_are_not_silently_repeated(self):
        with self.assertRaisesRegex(ValueError, "found only 1"):
            select_samples([original_row("first", "one")], "revision", 2)
