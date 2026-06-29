#!/usr/bin/env python3
"""Eval-gated CI runner — the LLMOps spine.

Contract CI depends on:
  - exit 0 if every metric >= its threshold in eval/thresholds.yaml
  - exit 1 if any metric is below threshold (this gates merges)
  - --report-only always exits 0 (local runs); writes eval/report.json

Run:  python eval/run_eval.py [--report-only]
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import json
from statistics import mean
from types import SimpleNamespace

import yaml
from dotenv import load_dotenv

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT= EVAL_DIR.parent
for _p in (str(EVAL_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0,_p)

load_dotenv()

import os
EVAL_K = int(os.getenv("EVAL_K", "5"))
EVAL_CHUNK_SIZE = int(os.getenv("EVAL_CHUNK_SIZE", "1000"))
EVAL_CHUNK_OVERLAP = int(os.getenv("EVAL_CHUNK_OVERLAP", "200"))
_LIMIT = int(os.getenv("EVAL_LIMIT","0"))

THRESHOLDS_PATH = EVAL_DIR / "thresholds.yaml"
GOLDEN_PATH = EVAL_DIR / "golden_dataset.jsonl"
CORPUS_DIR = EVAL_DIR / "corpus"
REPORT_PATH = EVAL_DIR / "report.json"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

from metrics import compute_ragas

from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from langchain_core.prompts import ChatPromptTemplate

class _LocalFile:
    """Minimal uploaded-file adapter for ChatIngestor (needs .name + .getbuffer)."""
    def __init__(self, path:Path):
        self.path= path
        self.name = path.name
    
    def getbuffer(self) -> bytes:
        return self.path.read_bytes()


def load_thresholds() -> dict[str, float]:
    return yaml.safe_load(THRESHOLDS_PATH.read_text(encoding="utf-8")) or {}

def load_golden() -> list[dict]:
    return [json.loads(ln) for ln in GOLDEN_PATH.read_text(encoding='utf-8').splitlines() if ln.strip()]


EVAL_SESSION = "eval_corpus"  # fixed session dir, rebuilt each run (no accumulation)


def _load_rag():
    """Ingest the corpus into a fixed session dir and return the configured RAG engine."""
    import shutil
    from multi_doc_chat.utils.config_loader import load_config

    files = [_LocalFile(p) for p in sorted(CORPUS_DIR.glob("*"))
             if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        raise SystemExit(f"No corpus files found in {CORPUS_DIR}")

    # Wipe the fixed eval dirs so each run rebuilds cleanly (handles chunk-size changes
    # and avoids accumulating a new session dir per run).
    for base in ("data", "faiss_index"):
        shutil.rmtree(Path(base) / EVAL_SESSION, ignore_errors=True)

    ingestor = ChatIngestor(temp_base="data", faiss_base="faiss_index",
                            use_session_dirs=True, session_id=EVAL_SESSION)
    ingestor.build_retriever(files, chunk_size=EVAL_CHUNK_SIZE, chunk_overlap=EVAL_CHUNK_OVERLAP, k=EVAL_K)

    # Respect the configured engine ("standard" | "corrective").
    engine = (load_config().get("rag", {}) or {}).get("engine", "standard")
    if engine == "corrective":
        from multi_doc_chat.src.document_chat.agentic_rag import CorrectiveRAG
        rag = CorrectiveRAG(session_id=EVAL_SESSION)
    else:
        rag = ConversationalRAG(session_id=EVAL_SESSION)
    rag.load_retriever_from_faiss(index_path=f"faiss_index/{EVAL_SESSION}", k=EVAL_K)
    return rag

_CORRECTNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
    "You are an evaluator judging correctness. Correctness means how well the actual "
    "answer matches the reference answer in factual accuracy, coverage, and meaning. "
    "If it matches semantically (even if worded differently), it is CORRECT. If it misses "
    "key facts, contradicts the reference, or is factually wrong, it is INCORRECT. Do not "
    "penalize stylistic or formatting differences unless they change meaning."),
    ("Question:\n{question}\n\nReference answer:\n{reference}\n\nActual answer:\n{actual}\n\n"
    "Respond with exactly one word: CORRECT or INCORRECT.") 
])

_judge = None  # built once, reused across all questions


def _get_judge():
    global _judge
    if _judge is None:
        from multi_doc_chat.utils.model_loader import ModelLoader
        _judge = ModelLoader().load_llm()  # follows LLM_PROVIDER (groq) from config
    return _judge
    

def _score_correctness(question: str, answer: str, ground_truth: str) -> int:
    chain = _CORRECTNESS_PROMPT | _get_judge()
    try:
        verdict = chain.invoke(
            {"question": question, "reference": ground_truth, "actual": answer}
        ).content.upper()
        # "INCORRECT" contains "CORRECT", so check the negative first.
        return 0 if "INCORRECT" in verdict else (1 if "CORRECT" in verdict else 0)
    except Exception:
        return 0

def evaluate() -> dict[str, float]:
    """Run the eval, return {metric_name: mean_score}. See docs/01, Phase 4."""
    golden = load_golden()
    if _LIMIT:
        golden = golden[:_LIMIT]

    rag = _load_rag()

    samples: list[dict] = []
    correctness: list[int] = []
    for i, row in enumerate(golden,1):
        q, gt = row['question'], row['ground_truth']
        try:
            res = rag.invoke_with_context(q, chat_history=[])
            answer, contexts = res['answer'], res['contexts']
        except Exception as e:
            answer, contexts= f"ERROR: {e}",[]
        samples.append({"question":q,"answer":answer,"contexts":contexts, "ground_truth":gt})
        correctness.append(_score_correctness(q,answer,gt))
        print(f"[{i}/{len(golden)}] {q[:60]}")

    scores: dict[str, float] = {"correctness": mean(correctness) if correctness else 0.0}
    scores.update(compute_ragas(samples))  # faithfulness, answer_relevancy, context_precision/recall
    return scores


def gate(scores: dict[str, float], thresholds: dict[str, float]) -> bool:
    """Return True iff every metric meets its threshold. See docs/01, Phase 4."""
    passed=True
    print(f"\n{'metric':<20}{'score':>8}{'thresh':>8}  result")
    print("-" * 46)
    for name, thresh in thresholds.items():
        s = scores.get(name)
        ok = s is not None and s >= thresh
        passed = passed and ok
        shown = f"{s:.3f}" if s is not None else "  n/a"
        print(f"{name:<20}{shown:>8}{thresh:>8.2f}  {'PASS' if ok else 'FAIL'}")
    print("-" * 46)
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval-gated CI runner")
    parser.add_argument("--report-only", action="store_true", help="Always exit 0.")
    args = parser.parse_args()
    
    thresholds = load_thresholds()
    scores = evaluate()
    passed = gate(scores, thresholds)

    REPORT_PATH.write_text(
        json.dumps({"scores": scores, "thresholds": thresholds, "passed": passed}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {REPORT_PATH}")
    sys.exit(0 if (args.report_only or passed) else 1)


if __name__ == "__main__":
    main()
