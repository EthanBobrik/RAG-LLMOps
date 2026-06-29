"""Agentic Corrective-RAG (CRAG) using LangGraph.

  Graph: retrieve → grade_documents → (generate | transform_query→retrieve)
         generate → grade_generation → (END | regenerate | rewrite→retrieve)
  Token-lean: grading is one LLM call per step (not per document). See docs/04.
"""

from __future__ import annotations
import json
import re
from typing import List, Optional, TypedDict

from langchain_core.messages import SystemMessage, HumanMessage

MAX_RETRIES = 2  # hard cap so the corrective loop can never run forever


class GraphState(TypedDict, total=False):
    question:str
    original_question:str
    documents: list
    generation:str
    retry_count:int
    grounded:bool
    answers:bool

def _format_docs(docs) ->str:
    return "\n\n".join(f"[{i+1}] {getattr(d, 'page_content',str(d))}" for i, d in enumerate(docs))

def _ask(llm, system:str, human:str) -> str:
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    return getattr(resp, "content", str(resp))


def build_corrective_rag_graph(retriever, llm, qa_prompt, max_retries: int = MAX_RETRIES):
    from langgraph.graph import StateGraph, END

    # ---- nodes (closures over retriever + llm) ----
    def retrieve(state: GraphState) -> GraphState:
        return {"documents":retriever.invoke(state['question'])}
    
    def grade_documents(state: GraphState) -> GraphState:
        docs = state['documents']
        if not docs:
            return {"documents":[]}
        listing = "\n\n".join(f"[{i}] {getattr(d, "page_content", '')[:500]}" for i,d in enumerate(docs))
        raw = _ask(
            llm,
            "You grade retrieved documents for relevance to a question. Reply with ONLY a JSON "
              "array of the integer indices of the documents that are relevant.",
              f"QUESTION:\n{state['question']}\n\nDOCUMENTS:\n{listing}",
            )
        m = re.search(r"\[.*\]", raw , re.DOTALL)
        keep = docs
        if m:
            try:
                idxs = json.loads(m.group())
                keep = [docs[i] for i in idxs if isinstance(i, int) and 0 <= i < len(docs)]
            except Exception:
                pass
        return {"documents":keep}
    
    def transform_query(state: GraphState) -> GraphState:
        better = _ask(
            llm,
            "Rewrite the user's question to be more specific and retrieval-friendly "
              "(expand acronyms, add key terms). Reply with ONLY the rewritten question.",
              state.get("original_question", state["question"]),
        ).strip()
        return {"question": better or state['question'],
                "retry_count":state.get("retry_count",0)+1}
    
    def generate(state: GraphState) -> GraphState:
        messages = qa_prompt.format_messages(
            context = _format_docs(state['documents']), input = state['question'], chat_history=[]
        )
        resp = llm.invoke(messages)
        return {'generation': getattr(resp, "content",str(resp))}
    
    def grade_generation(state: GraphState) -> GraphState:
        raw = _ask(
            llm,
            "Check an ANSWER against CONTEXT and a QUESTION. Reply with ONLY a JSON object: "
              '{"grounded": true/false, "answers": true/false}. grounded = answer supported by '
              "context; answers = it actually addresses the question.",
              f"QUESTION:\n{state['question']}\n\nCONTEXT:\n{_format_docs(state['documents'])}\n\n"
              f"ANSWER:\n{state.get('generation', '')}",
        )
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        verdict= {"grounded":True, "answers": True}
        if m:
            try:
                verdict = json.loads(m.group())
            except Exception:
                pass
        return {"grounded": bool(verdict.get("grounded", True)),
                "answers": bool(verdict.get("answers", True))}

    # Conditional edges
    def decide_to_generate(state: GraphState) -> str:
        if state['documents']:
            return "generate"
        if state.get("retry_count",0) < max_retries:
            return "transform_query"
        return "generate" # out of retries 
    
    def grade_generation_edge(state: GraphState) -> str:
        if (state.get("grounded") and state.get("answers")) or state.get("retry_count",0) >= max_retries:
            return "useful"
        return "regenerate" if not state.get("grounded") else "rewrite"
    
    # assemble 
    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("transform_query", transform_query)
    g.add_node("generate", generate)
    g.add_node("grade_generation", grade_generation)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", decide_to_generate,
                             {"generate":"generate", "transform_query":"transform_query"})

    g.add_edge("transform_query", "retrieve")
    g.add_edge("generate", "grade_generation")
    g.add_conditional_edges("grade_generation", grade_generation_edge,
                            {"useful": END, "regenerate": "generate", "rewrite":"transform_query"})
    return g.compile()

class CorrectiveRAG:
    """Drop-in alternative to ConversationalRAG, selectable via config (rag.engine)."""

    def __init__(self, session_id: Optional[str], retriever=None):
        from multi_doc_chat.utils.model_loader import ModelLoader
        from multi_doc_chat.utils.config_loader import load_config
        from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY
        from multi_doc_chat.model.models import PromptType

        self.session_id = session_id
        self.llm = ModelLoader().load_llm()
        self.qa_prompt = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]
        self.max_retries = (load_config().get("rag",{}) or {}).get("max_retries", MAX_RETRIES)
        self.retriever = retriever
        self.graph = None
        if retriever is not None:
            self.graph = build_corrective_rag_graph(retriever, self.llm, self.qa_prompt, self.max_retries)


    def load_retriever_from_faiss(self, index_path: str, **kwargs):
        from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
        base = ConversationalRAG(session_id=self.session_id)
        base.load_retriever_from_faiss(index_path, **kwargs)
        self.retriever = base.retriever
        self.graph = build_corrective_rag_graph(self.retriever, self.llm, self.qa_prompt, self.max_retries)
    

    def invoke(self, user_input: str, chat_history: Optional[list] = None) -> str:
        if self.graph is None:
            raise RuntimeError("Call load_retriever_from_faiss() before invoke().")
        final = self.graph.invoke(
            {"question": user_input, "original_question":user_input, "retry_count":0}
        )
        return final.get("generation","")
    
    async def astream_answer(self, user_input:str, chat_history: Optional[list]=None):
        #CRAG runs a multi-step graph; emit the final answer as a single chunk
        import asyncio
        answer = await asyncio.to_thread(self.invoke, user_input, chat_history)
        yield answer
