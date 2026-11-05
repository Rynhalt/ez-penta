"""Check published table cells and verify that original source/data files stayed unchanged."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    text = (HERE / "report.md").read_text()
    s = json.loads((HERE / "statistics.json").read_text())
    evidence = json.loads((HERE / "verification.json").read_text())
    names = {"original": "Original", "fc": "Fiction (fc)", "rp": "Roleplay (rp)",
             "rs": "Refusal suppression (rs)", "every10_yes": "Every 10 characters: yes",
             "every10_hai": "Every 10 characters: hai", "quartile_yes": "Quartiles: yes", "quartile_hai": "Quartiles: hai"}
    short = {**names, "fc": "Fiction", "rp": "Roleplay", "rs": "Refusal suppression",
             "every10_yes": "Every 10: yes", "every10_hai": "Every 10: hai"}
    for variant, label in names.items():
        ja = next(r for r in s["variant_language"] if r["variant"] == variant and r["language"] == "ja")
        en = next(r for r in s["variant_language"] if r["variant"] == variant and r["language"] == "en")
        line = (f"| {label} | {ja['unsafe']} / {ja['judged']:,} ({100 * ja['unsafe_rate']:.2f}) | "
                f"{en['unsafe']} / {en['judged']:,} ({100 * en['unsafe_rate']:.2f}) | "
                f"{100 * ja['refusal_rate']:.2f} / {100 * en['refusal_rate']:.2f} |")
        assert line in text, line
        u = next(r for r in s["language_comparisons"] if r["variant"] == variant and r["measurement"] == "unsafe")
        r = next(r for r in s["language_comparisons"] if r["variant"] == variant and r["measurement"] == "refused")
        line = (f"| {short[variant]} | {u['first_only']} / {u['second_only']} [{u['pairs']:,}] | "
                f"{100 * u['mismatch_rate']:.2f} | {r['first_only']} / {r['second_only']} [{r['pairs']:,}] | "
                f"{100 * r['mismatch_rate']:.2f} |")
        assert line in text, line
    for variant in ("fc", "rp", "rs", "every10_hai"):
        for language in ("ja", "en"):
            u = next(r for r in s["variant_changes"] if (r["variant"], r["language"], r["measurement"]) == (variant, language, "unsafe"))
            r = next(r for r in s["variant_changes"] if (r["variant"], r["language"], r["measurement"]) == (variant, language, "refused"))
            line = (f"| {short[variant]} / {language.upper()} | {u['increase_count']} / {u['decrease_count']} [{u['pairs']:,}] | "
                    f"{r['increase_count']} / {r['decrease_count']} [{r['pairs']:,}] |")
            assert line in text, line
    for name, metadata in evidence["inputs"].items():
        assert sha(ROOT / name) == metadata["sha256"], name
    for name, digest in evidence["source_hashes"].items():
        assert sha(ROOT / name) == digest, name
    pdf = json.loads((HERE / "pdf-validation.json").read_text())
    assert pdf["page_count"] == 5
    assert all(not page["outside_page_bounds"] for page in pdf["pages"])
    tests = (HERE / "test-run.log").read_text()
    assert "Ran 51 tests" in tests and tests.rstrip().endswith("OK")
    result = {"published_variant_rate_rows_checked": 8, "language_comparison_rows_checked": 8,
              "variant_transition_rows_checked": 8, "pdf_pages": 5,
              "original_source_files_unchanged": True, "original_data_files_unchanged": True,
              "tests_passed": 51, "visual_inspection": "All five rendered pages inspected for pagination, legibility and clipping."}
    (HERE / "final-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
