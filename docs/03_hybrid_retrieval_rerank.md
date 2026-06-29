# Build Guide 3 — Hybrid Retrieval + Cross-Encoder Reranking

**Goal:** replace the single dense (FAISS/MMR) retriever with a **hybrid** retriever that
fuses sparse lexical search (BM25) and dense semantic search, then reranks the merged
candidates with a **cross-encoder**. This measurably improves retrieval precision —
especially for keyword-heavy or acronym queries that pure embeddings miss — and the gain
is provable with the eval harness from Guide 1.

**End state resume bullet:**
> Lifted retrieval precision@5 by N% by fusing BM25 + dense FAISS retrieval and reranking
> candidates with a cross-encoder, validated against a labeled eval set in CI.

---

## Why this works

- **Dense (FAISS)** captures meaning but can miss exact terms, IDs, rare acronyms.
- **Sparse (BM25)** nails exact-term matches but misses paraphrases.
- **Ensemble fusion** unions both candidate sets (reciprocal-rank fusion / weighted).
- **Cross-encoder reranking** re-scores each (query, chunk) pair jointly — far more
  accurate than the bi-encoder cosine used for first-stage retrieval — so you fetch many
  (e.g. 20) cheaply, then keep the best `k` (e.g. 5).

Pipeline: `query → [BM25 ∪ FAISS] → ~20 candidates → cross-encoder rerank → top 5 → LLM`.

---

## Phase 1 — Persist chunk text for BM25

BM25 is in-memory and needs the raw chunk texts; FAISS only persists vectors. Today
`FaissManager` saves the index but not a reloadable copy of the chunks.

1. **In `data_ingestion.py`**, when building the index, also write the chunks to disk:
   `faiss_index/<session_id>/chunks.jsonl` (one `{page_content, metadata}` per line).
2. Add a loader `load_chunks(index_dir) -> list[Document]` (put it in
   `hybrid_retrieval.py`) that reads that file back into `Document` objects.
3. Keep this backward-compatible: if `chunks.jsonl` is absent, fall back to dense-only.

## Phase 2 — Build the hybrid retriever

1. **`uv add rank-bm25`** (BM25Retriever backend).
2. **In `multi_doc_chat/src/document_chat/hybrid_retrieval.py`** implement
   `build_hybrid_retriever(faiss_vectorstore, chunks, *, k, dense_weight, sparse_weight)`:
   - `dense = faiss_vectorstore.as_retriever(search_type="mmr", search_kwargs={...})`
   - `sparse = BM25Retriever.from_documents(chunks); sparse.k = fetch_k`
   - `EnsembleRetriever(retrievers=[dense, sparse], weights=[dense_weight, sparse_weight])`
   - Return the ensemble. Default weights `[0.5, 0.5]` — tune later via eval.

## Phase 3 — Add the cross-encoder reranker

1. **Pick a reranker** (choose one, make it config-driven):
   - **Local (free):** `HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")` +
     `CrossEncoderReranker(top_n=k)` from `langchain_community`. Needs `sentence-transformers`
     (`uv add sentence-transformers`). No API cost; slower cold start.
   - **API (fast, paid):** Cohere Rerank via `CohereRerank` (needs `COHERE_API_KEY`).
2. **In `hybrid_retrieval.py`** implement
   `build_reranking_retriever(base_retriever, *, top_n, model_name) -> ContextualCompressionRetriever`:
   - Wrap the ensemble in `ContextualCompressionRetriever(base_compressor=reranker,
     base_retriever=base_retriever)` so reranking happens transparently on `.invoke()`.
3. **Fetch wide, keep narrow:** set the ensemble/first-stage to return ~20, reranker
   `top_n=5`.

## Phase 4 — Config

1. **In `multi_doc_chat/config/config.yaml`** add a `reranker` + `hybrid` block (already
   scaffolded as a placeholder):
   ```yaml
   hybrid:
     enabled: true
     dense_weight: 0.5
     sparse_weight: 0.5
     fetch_k: 20
   reranker:
     enabled: true
     provider: "huggingface"        # or "cohere"
     model_name: "BAAI/bge-reranker-base"
     top_n: 5
   ```
2. Read these in the retriever-construction path; both blocks must be independently
   toggleable so you can A/B (dense-only vs hybrid vs hybrid+rerank) in evals.

## Phase 5 — Wire into the RAG layer

1. **In `retrieval.py`**, `load_retriever_from_faiss()` (or a new
   `load_hybrid_retriever_from_faiss()`): after loading the FAISS vectorstore, if
   `hybrid.enabled`, build the ensemble via `build_hybrid_retriever(...)`; if
   `reranker.enabled`, wrap it via `build_reranking_retriever(...)`. Assign the result to
   `self.retriever` and proceed to `_build_lcel_chain()` unchanged — the LCEL chain is
   retriever-agnostic, so nothing downstream changes.
2. **Fallback:** if `chunks.jsonl` is missing or hybrid disabled, use today's dense path.

## Phase 6 — Prove the gain (uses Guide 1)

1. Run `eval/run_eval.py` in three configs: dense-only, hybrid, hybrid+rerank.
2. Compare `context_precision` / `context_recall` (and end-to-end correctness).
3. The best delta is your bullet's number. Commit the three `report.json`s as evidence.

## Definition of done

- [ ] Chunks persisted to `chunks.jsonl` at ingest; `load_chunks()` reloads them.
- [ ] `build_hybrid_retriever()` (BM25 + dense ensemble) implemented.
- [ ] `build_reranking_retriever()` (cross-encoder) implemented, config-toggleable.
- [ ] RAG layer builds hybrid+rerank when enabled, falls back gracefully.
- [ ] Eval comparison across 3 configs with a measured precision delta.
