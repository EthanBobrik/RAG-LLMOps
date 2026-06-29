# Build Guide 1 — Eval-Gated CI (the LLMOps spine)

**Goal:** every pull request automatically runs an offline RAG evaluation against a
golden dataset, computes quality metrics (correctness + RAGAS faithfulness / answer
relevancy / context precision), and **fails the Jenkins build** if any metric drops
below a configured threshold. This is the rarest and most differentiating piece — and
it produces the real numbers you'll cite in every other resume bullet.

**End state resume bullet:**
> Built an eval-driven CI pipeline (LangSmith + RAGAS) that gates merges on RAG
> faithfulness, answer-relevancy, and context-precision thresholds — catching quality
> regressions on a 50-example golden dataset before they reach production.

---

## Prerequisites / what already exists

- `run_evaluations.py` — already wires a LangSmith `evaluate(...)` run with a custom
  `correctness_evaluator` (LLM-as-judge using Gemini).
- `multi_doc_chat/src/document_chat/retrieval.py` — `ConversationalRAG.invoke()`.
- `multi_doc_chat/src/document_ingestion/data_ingestion.py` — `ChatIngestor.build_retriever()`.
- `Jenkinsfile.test` — runs pytest today; we'll add an `Evaluate` stage.
- Env keys required: `GOOGLE_API_KEY` (judge + embeddings), `LANGSMITH_API_KEY` (optional
  if you want the LangSmith UI; we'll also support a fully-local mode).

---

## Phase 1 — Capture retrieval context (needed for RAGAS)

RAGAS metrics like faithfulness and context-precision need the **retrieved contexts**,
not just the final answer. Today the RAG target only returns `{"answer": ...}`.

1. **Add a context-returning entry point** in `retrieval.py`. Add a method
   `invoke_with_context(self, user_input, chat_history=None) -> dict` that returns
   `{"answer": str, "contexts": list[str]}`.
   - Reuse the existing chain, but also fetch the retrieved docs so you can return them.
   - Simplest approach: call `self.retriever.invoke(rewritten_question)` to get docs,
     `self._format_docs(docs)` for the prompt, then run the QA prompt + LLM yourself, and
     return both the answer and `[d.page_content for d in docs]`.
2. **Keep `invoke()` unchanged** so `main.py` and existing tests are unaffected.
3. **Unit test** it in `tests/unit/test_retrieval.py` with the `stub_model_loader`
   fixture: assert the dict has both keys and `contexts` is a non-empty list.

## Phase 2 — Create the golden dataset

1. **Pick a single, stable source doc** for evals so retrieval is deterministic. Put a
   text file under `eval/corpus/` (not `data/`, which is gitignored). Example:
   `eval/corpus/ai_engineering_report.txt`.
2. **Write 30–50 Q/A pairs** in `eval/golden_dataset.jsonl`, one JSON object per line:
   ```json
   {"question": "What is RAG?", "ground_truth": "Retrieval-Augmented Generation grounds an LLM's answer in retrieved documents..."}
   ```
   - Cover easy lookups, multi-sentence synthesis, and a few "not in the document"
     questions (the answer should be "I don't know") to test faithfulness.
3. **Decide storage mode.** Two supported paths:
   - **LangSmith mode:** upload the dataset once via the LangSmith SDK
     (`Client().create_dataset(...)` + `create_examples(...)`). Good for the UI/portfolio
     screenshots.
   - **Local mode:** read the JSONL directly in the eval runner. No external account
     needed; better for CI determinism. **Recommended to support both**, defaulting to
     local in CI.

## Phase 3 — Add RAGAS metrics

1. **Add the dependency:** `uv add ragas datasets`.
2. **Create `eval/metrics.py`** with a function `compute_ragas(samples) -> dict` where
   each sample is `{question, answer, contexts, ground_truth}`.
   - Use RAGAS metrics: `faithfulness`, `answer_relevancy`, `context_precision`,
     `context_recall`.
   - Configure RAGAS to use **your** models (Gemini LLM + Google embeddings) via the
     RAGAS LLM/embeddings wrappers, so you're not implicitly requiring OpenAI. Verify the
     exact wrapper names against the installed RAGAS version — this API moves between
     releases (check `ragas.__version__` and the docs before finalizing).
3. **Return a flat dict** of metric_name -> float (0..1), aggregated across samples.

## Phase 4 — Build the gating runner

1. **Create `eval/run_eval.py`** (separate from the existing LangSmith
   `run_evaluations.py`, or extend it) that:
   1. Ingests `eval/corpus/` once via `ChatIngestor.build_retriever()`.
   2. Loads `eval/golden_dataset.jsonl`.
   3. For each question, calls `ConversationalRAG.invoke_with_context()` to get
      `{answer, contexts}`.
   4. Runs `correctness_evaluator` (already exists) **and** `compute_ragas(...)`.
   5. Aggregates: mean correctness, mean of each RAGAS metric.
2. **Add thresholds** in a config block (or `eval/thresholds.yaml`):
   ```yaml
   correctness: 0.80
   faithfulness: 0.85
   answer_relevancy: 0.80
   context_precision: 0.70
   ```
3. **Exit-code contract:** print a table of `metric | score | threshold | PASS/FAIL`,
   write `eval/report.json`, and `sys.exit(1)` if **any** metric is below threshold,
   else `sys.exit(0)`. This exit code is what makes CI gate.
4. **Add `--report-only` flag** (always exit 0) so you can run it locally without failing.

## Phase 5 — Wire into Jenkins

1. **Edit `Jenkinsfile.test`** — add an `Evaluate` stage after `Run Tests`:
   - Inject `GOOGLE_API_KEY` (and `LANGSMITH_API_KEY` if used) via Jenkins credentials,
     not plaintext.
   - Run `python eval/run_eval.py` (no `--report-only`) so a regression fails the stage.
   - `archiveArtifacts 'eval/report.json'` and publish it.
2. **Make it branch-aware / cost-aware:** the eval calls real LLM APIs and costs money +
   time. Gate the stage so it only runs on PRs to `main` (or behind a `when { branch }`
   block), not on every poll. Document this so you're not silently skipping coverage.
3. **Add a threshold-bump policy note** in the PR template: thresholds ratchet up, never
   down, without explicit justification.

## Phase 6 — Prove it works (the demo that earns the bullet)

1. **Baseline run:** run the eval on `main`, commit `eval/report.json` as the baseline.
2. **Regression test:** deliberately break retrieval (e.g., set `k=1`, or a bad chunk
   size) on a branch and show the CI build going red with the failing metric named.
3. **Improvement test:** make a real improvement (see Guide 2's hybrid-retrieval, or tune
   chunking) and show the metric going up in the report. **This delta is your quantified
   bullet.**

## Definition of done

- [ ] `invoke_with_context()` returns answer + contexts, unit-tested.
- [ ] `eval/golden_dataset.jsonl` with 30–50 examples incl. "I don't know" cases.
- [ ] `eval/run_eval.py` computes correctness + 4 RAGAS metrics, exits non-zero on
      threshold breach, writes `eval/report.json`.
- [ ] Jenkins `Evaluate` stage fails the build on regression; artifacts archived.
- [ ] A captured before/after metric delta you can quote with a real number.
