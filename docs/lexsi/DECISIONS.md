# Dataset2 decision log

- The supplied Drive folder contains the original GGUF exports. The prepared review CSVs were absent from the accessible folder and exact-name Drive search. Review inputs reconstructed from saved SQLite rows are tracked by hash.
- Keep heldout rows in the audit inventory. The plan records prior exposure of 91 heldout disagreements. Exclude those seeds from contrast selection and tuning, and use fresh untouched seeds for confirmatory generalization.
- Assign each contrast a stable ID from the ordered prompt IDs. Treat guard direction as source evidence only. Require reviewed behavior and explicit human adjudication before export.
- Review all 87 reconstructed roleplay screens. Four analysis seeds and one tune seed have a provisional refusal-to-compliance contrast, all in English. The tune example retains its split for later validation. Keep the approved export empty until adjudication.
- Screen both-refuse controls from analysis pairs with two Safe verdicts and two published guard refusal flags. Use the saved responses to verify refusals. Use reviewed roleplay screens for both-comply controls. Keep sanity controls disjoint from the shortlist by source seed.
