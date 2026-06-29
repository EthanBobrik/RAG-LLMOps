# Architecture Upgrade Roadmap

Four upgrades that turn this RAG demo into a portfolio-grade LLMOps project. Build in
order — Guide 1 is the measurement spine that lets you put real numbers on 2–4.

| # | Guide | Resume angle | Key new/changed files |
|---|-------|--------------|------------------------|
| 1 | [Eval-gated CI](01_eval_gated_ci.md) | Quality gating most people never build | `eval/run_eval.py`, `eval/metrics.py`, `eval/thresholds.yaml`, `eval/golden_dataset.jsonl`, `Jenkinsfile.test`, `retrieval.invoke_with_context()` |
| 2 | [Token streaming](02_token_streaming.md) | UX / perceived-latency win | `main.py:/chat/stream`, `retrieval.astream_answer()`, `templates/index.html:sendMessageStreaming()` |
| 3 | [Hybrid retrieval + rerank](03_hybrid_retrieval_rerank.md) | Measurable retrieval-quality gain | `multi_doc_chat/src/document_chat/hybrid_retrieval.py`, `config.yaml:hybrid/reranker` |
| 4 | [Agentic corrective-RAG](04_agentic_corrective_rag.md) | SOTA buzzword bullet | `multi_doc_chat/src/document_chat/agentic_rag.py`, `config.yaml:rag.engine` |

## Status

All integration points are **scaffolded as minimal placeholders**: each
function/method/endpoint exists with its signature, a one-line `TODO: see docs/...`
pointer, and a `raise NotImplementedError` body — nothing more. You write the
implementations.

Existing app behavior and the test suite are unaffected — placeholders are never on the
default code path (engine defaults to `standard`, hybrid/reranker `enabled: false`, and
`/chat/stream` is unused by the current frontend, which still calls `/chat`).

Find what's left to build:  `grep -rn "NotImplementedError\|TODO" main.py eval multi_doc_chat templates`
