# Build Guide 4 — Agentic Corrective-RAG (LangGraph)

**Goal:** wrap retrieval + generation in a small **stateful agent graph** (LangGraph) that
*self-corrects*: it grades whether retrieved documents are actually relevant, rewrites the
query and re-retrieves when they aren't, generates an answer, and finally checks the
answer for hallucination against the retrieved context before returning it. This is the
CRAG (Corrective RAG) pattern and is current SOTA — the flashiest single line on the
resume.

**End state resume bullet:**
> Designed an agentic corrective-RAG loop (LangGraph) with document-relevance grading,
> query rewriting, and answer-faithfulness self-checks — improving grounded-answer rate on
> out-of-distribution questions by N%.

---

## The graph

```
          ┌─────────┐
          │ retrieve│◄────────────────┐
          └────┬────┘                 │ (re-retrieve after rewrite)
               ▼                       │
        ┌─────────────┐         ┌──────┴───────┐
        │grade_documents│──poor──►│transform_query│
        └──────┬────────┘        └──────────────┘
               │ relevant
               ▼
          ┌─────────┐
          │ generate│
          └────┬────┘
               ▼
     ┌──────────────────────┐  hallucinated / doesn't answer
     │grade_generation      │──────────────► (retry generate or transform_query)
     └──────────┬───────────┘
                │ grounded & answers
                ▼
              END
```

**State** (a `TypedDict`): `question`, `original_question`, `documents`, `generation`,
`retry_count`, `chat_history`.

---

## Phase 1 — Dependencies & module

1. **`uv add langgraph`** (and `langchain-core` is already present).
2. **Create `multi_doc_chat/src/document_chat/agentic_rag.py`** (scaffolded as a
   placeholder). Keep all `langgraph` imports **inside** functions or guarded at module
   load, so importing the package never breaks the main app if langgraph isn't installed.

## Phase 2 — Define graph state and nodes

Each node is a function `(state: dict) -> dict` returning state updates.

1. **`retrieve(state)`** — run the (hybrid, if Guide 3 is in) retriever on
   `state["question"]`; set `state["documents"]`.
2. **`grade_documents(state)`** — for each doc, an LLM grader returns yes/no "is this doc
   relevant to the question?" Keep only relevant docs. Use a small structured-output
   prompt (`with_structured_output` or a strict JSON parse). Record whether any survived.
3. **`transform_query(state)`** — an LLM rewrites `state["question"]` into a better search
   query (more specific, expands acronyms). Increment `retry_count`.
4. **`generate(state)`** — run the existing QA prompt + LLM over the surviving documents;
   set `state["generation"]`. Reuse `ConversationalRAG`'s QA prompt for consistency.
5. **`grade_generation(state)`** — two checks via LLM:
   - **Faithfulness:** is the generation grounded in the documents (no hallucination)?
   - **Answer relevance:** does it actually answer the question?

## Phase 3 — Define edges (control flow)

1. **`decide_to_generate(state)`** (conditional edge after `grade_documents`): if relevant
   docs exist → `"generate"`; else if `retry_count < MAX_RETRIES` → `"transform_query"`;
   else → `"generate"` (answer with what we have / say "I don't know").
2. **`grade_generation_edge(state)`** (conditional edge after `grade_generation`):
   - grounded **and** answers → `END`
   - not grounded **and** retries left → `"generate"` (regenerate)
   - doesn't answer **and** retries left → `"transform_query"` (re-retrieve)
   - retries exhausted → `END`
3. **Always cap retries** (`MAX_RETRIES = 2`) so the graph can't loop forever — call this
   out explicitly; it's a common failure and a good thing to mention you guarded against.

## Phase 4 — Compile and expose

1. **`build_corrective_rag_graph(retriever, llm)`** — assemble `StateGraph`, add nodes,
   set entry point `retrieve`, wire conditional edges, `return graph.compile()`.
2. **`CorrectiveRAG`** class mirroring `ConversationalRAG`'s interface:
   - `__init__(session_id)`, `load_retriever_from_faiss(...)` (reuse/delegate),
     `invoke(question, chat_history)` → runs `graph.invoke(initial_state)` and returns
     `state["generation"]`.
   - Optional `astream` for token streaming of the final `generate` node (composes with
     Guide 2).

## Phase 5 — Integrate behind a flag

1. **Config toggle** in `config.yaml`:
   ```yaml
   rag:
     engine: "standard"   # "standard" | "corrective"
     max_retries: 2
   ```
2. **In `main.py`**, pick the engine in `/chat` based on config:
   `engine = CorrectiveRAG(...) if cfg == "corrective" else ConversationalRAG(...)`.
   Both expose the same `invoke()` signature, so the route code is otherwise unchanged.
3. Keep `"standard"` as the default so existing behavior/tests are untouched.

## Phase 6 — Prove it (uses Guide 1)

1. Build an eval slice of **adversarial / out-of-context questions** (answer not in the
   docs, or needs query rewriting). Standard RAG will hallucinate or miss; corrective RAG
   should say "I don't know" or recover via rewrite.
2. Compare faithfulness + correctness: `standard` vs `corrective`. That delta is your
   bullet.
3. **Trace it:** with LangSmith tracing on (Guide 1 / observability), capture a screenshot
   of the graph taking the `grade_documents → transform_query → retrieve` corrective path.
   That visual is portfolio gold.

## Definition of done

- [ ] `agentic_rag.py` with state, 5 nodes, 2 conditional edges, retry cap.
- [ ] `build_corrective_rag_graph()` compiles a runnable graph.
- [ ] `CorrectiveRAG` matches `ConversationalRAG`'s interface; selectable via config.
- [ ] Default engine stays `standard`; existing tests pass unchanged.
- [ ] Eval delta (standard vs corrective) on an adversarial slice, plus a trace
      screenshot of the corrective path firing.
