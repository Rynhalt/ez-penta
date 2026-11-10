# Dataset2 decision log

- The supplied Drive folder contains the original GGUF exports. The prepared review CSVs were absent from the accessible folder and exact-name Drive search. Review inputs reconstructed from saved SQLite rows are tracked by hash.
- Keep heldout rows in the audit inventory. The plan records prior exposure of 91 heldout disagreements. Exclude those seeds from contrast selection and tuning, and use fresh untouched seeds for confirmatory generalization.
- Assign each contrast a stable ID from the ordered prompt IDs. Treat guard direction as source evidence only. Require reviewed behavior and explicit human adjudication before export.
