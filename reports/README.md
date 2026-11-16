# Research report package

- [Canonical technical article](experiment_report.md) and [submission abstract](submission_abstract.md).
- [Claim audit](CLAIM_AUDIT.md), [artifact map](EXPERIMENT_ARTIFACT_MAP.md), [source inventory](source_inventory.json), and [implementation review](../_archive/superseded_docs/IMPLEMENTATION_REVIEW.md).
- [Technical appendix](technical_appendix.md) and [reproduction guide](REPRODUCE.md).
- `figures/` contains PNG/SVG charts and their generator; `tables/` contains plotted data.

From the repository root:

```bash
python reports/figures/generate_figures.py
python reports/validate_report.py
```

The historical November 5 report is preserved under [archive/reports/](../_archive/reports/20261105T134711Z/). It uses an earlier population and is not the canonical article.
