# Report reproduction

All outputs in this folder are local documentation artifacts. The analysis opens
the original SQLite file read-only. It does not run models or upload data.

From the repository root, use a separate temporary environment so project
dependencies remain unchanged:

```sh
python3 -m venv /tmp/ez-penta-report-env
/tmp/ez-penta-report-env/bin/python -m pip install -r reports/20261105T134711Z/rendering-requirements.txt
/tmp/ez-penta-report-env/bin/python -B reports/20261105T134711Z/collect_evidence.py
/tmp/ez-penta-report-env/bin/python -B reports/20261105T134711Z/render_report.py
/tmp/ez-penta-report-env/bin/python -B reports/20261105T134711Z/validate_report.py
```

The renderer used macOS Arial TrueType fonts. A DejaVu Sans fallback is supported
on Linux, although different font metrics can require pagination adjustments.
`previews/page-*.png` contains the inspected raster previews of all five PDF pages.

`verification.json` records the exact source and data hashes used for this report.
`library-evidence.json` records the inspected AuditKIT source and version.
`statistics.json` and five CSV files preserve unrounded results. The unit-test log
is from the existing project environment; it is not a GPU experiment log.

The four case studies are targeted examples re-read from the SQLite answer table;
they are not a representative human accuracy study. Source IDs and languages in
the report identify the original records. Actionable harmful passages were not
reproduced in the report.
