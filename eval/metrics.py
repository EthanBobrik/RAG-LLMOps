"""Hand-rolled RAG metrics (LLM-as-judge) — no ragas dependency.

Computes faithfulness, answer_relevancy, context_precision, and context_recall using
the project's Gemini judge (from config via ModelLoader). Each metric is one judge call
per sample returning a 0.0-1.0 score. See docs/01_eval_gated_ci.md, Phase 3.
"""

from __future__ import annotations
import re
from statistics import mean
from typing import List, TypedDict
import json

from langchain_core.messages import SystemMessage, HumanMessage


class EvalSample(TypedDict):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str


_METRIC_KEYS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def _parse_metrics_json(text:str) ->dict:
    m = _JSON_RE.search(text or "")
    if not m:
        return {}
    try:
        data = json.loads(m.group())
    except Exception:
        return {}
    out ={}
    for k in _METRIC_KEYS:
        v= data.get(k)
        if isinstance(v, (int, float)):
            out[k] = max(0.0, min(1.0, float(v)))
    return out

_judge = None  # built once, reused across every metric call


def _get_judge():
    global _judge
    if _judge is None:
        from multi_doc_chat.utils.model_loader import ModelLoader
        _judge = ModelLoader().load_llm()  # reads model from config.yaml (single source of truth)
    return _judge

def _format_contexts(contexts: List[str]) -> str:
    if not contexts:
        return "(no context retrieved)"
    return "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))

def _score_sample(s: EvalSample) -> dict:
    """One judge call returning all four metrics for a sample as JSON."""
    system = (
        "You are a strict RAG evaluator. Score each metric from 0.0 to 1.0 and reply with "
        "ONLY a compact JSON object with keys faithfulness, answer_relevancy, "
        "context_precision, context_recall.\n"
        "- faithfulness: fraction of the ANSWER's claims supported by CONTEXT "
        "(1.0 if the answer says it doesn't know and CONTEXT indeed lacks the info).\n"
        "- answer_relevancy: how well the ANSWER addresses the QUESTION.\n"
        "- context_precision: fraction of CONTEXT chunks relevant to the QUESTION.\n"
        "- context_recall: fraction of the REFERENCE answer's facts supported by CONTEXT."
    )
    human = (
        f"QUESTION:\n{s['question']}\n\n"
        f"REFERENCE:\n{s['ground_truth']}\n\n"
        f"ANSWER:\n{s['answer']}\n\n"
        f"CONTEXT:\n{_format_contexts(s['contexts'])}"
    )
    try:
        resp = _get_judge().invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return _parse_metrics_json(getattr(resp, "content", str(resp)))
    except Exception:
        return {}


def compute_ragas(samples: List[EvalSample]) -> dict[str, float]:
    """Mean of the four LLM-judged metrics across samples (one judge call per sample)."""
    if not samples:
        return {}
    acc = {k: [] for k in _METRIC_KEYS}
    for s in samples:
        scored = _score_sample(s)
        empty_ctx = not s["contexts"]
        for k in _METRIC_KEYS:
            v = 0.0 if (empty_ctx and k in ("context_precision", "context_recall")) else scored.get(k, 0.0)
            acc[k].append(v)
    return {k: mean(v) for k, v in acc.items()}