# Eval corpus

Drop the **single, stable source document** the golden dataset is written against here
(e.g. `ai_engineering_report.txt`). Unlike `data/` (gitignored, user uploads), this corpus
is committed so evaluations are reproducible in CI.

Keep it small and fixed — if the corpus changes, re-derive `../golden_dataset.jsonl`.
See `docs/01_eval_gated_ci.md`, Phase 2.
