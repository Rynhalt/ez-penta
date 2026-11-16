# Japanese-culture evaluation

This folder contains the paired US-culture/Japan-culture evaluation of the English- and Japanese-trained Tiny Aya adapters.

Project inventory: three execution pipelines produced six named completed experiments (steering, bilingual training, and E1–E4), plus one L4/G4 hardware-comparability check. The Japan-culture J2–J5 analysis is an additional, ongoing evaluation of the existing seed-0 adapters, not a new training experiment.

- [Interim findings report](INTERIM_REPORT.md): E1–E4-style narrative and interpretation of the current 4/6-chunk results.
- [Google Docs version with all charts embedded, including the individual J2–J5 charts](https://docs.google.com/document/d/1eyKsxkpaj9v5FuEAEk-kacc1J-CaMCgWu7KYn75vc0s/edit): formatted copy of the interim findings report.
- [JA-arm report](ja/report.md) and [JA results](ja/results.json): machine-generated J2/J3/J5 outputs.
- [Combined report](combined/report.md) and [combined results](combined/results.json): EN/JA comparisons, including J4.
- [Current status](status_ja.json): completed chunks, seed counts, base provenance and stop/final state.
- Interim paired-effects chart with 95% confidence intervals: [view in the repo](figures/interim_paired_effects.png) or [open from Google Drive](https://drive.google.com/file/d/1Pu8UXrmehaUav0AgYqUVvRgYgqsmvbIJ/view). Source notebook: [interim_paired_effects.ipynb](ja/notebooks/interim_paired_effects.ipynb).
- Individual J2–J5 charts with definitions of prompt language, cultural cue, %, and pp: [source notebook](ja/notebooks/j2_j5_visual_charts.ipynb); embedded in the [Google Docs interim report](https://docs.google.com/document/d/1eyKsxkpaj9v5FuEAEk-kacc1J-CaMCgWu7KYn75vc0s/edit).
- Current plots: [safety](combined/figures/safety.png), [fix rate](combined/figures/fix_rate.png), and [over-refusal](combined/figures/over_refusal.png).

The interim results are automated and require human review. Chunks c0009 and c0011 remain outstanding for the JA arm; do not present this analysis as final until those chunks are evaluated and the six-chunk analysis is refreshed.
