import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY
from multi_doc_chat.model.models import PromptType, ChatAnswer
from pydantic import ValidationError


class ConversationalRAG:
    """
    LCEL-based Conversational RAG with lazy retriever initialization.

    Usage:
        rag = ConversationalRAG(session_id="abc")
        rag.load_retriever_from_faiss(index_path="faiss_index/abc", k=5, index_name="index")
        answer = rag.invoke("What is ...?", chat_history=[])
    """

    def __init__(self, session_id: Optional[str], retriever=None):
        try:
            self.session_id = session_id

            # one model loader, reused for the LLM and embeddings (avoids re-reading
            # .env/config separately for each)
            self._model_loader = ModelLoader()

            # load llm and prompts once
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            # lazy pieces
            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            log.info("ConversationalRAG initialized", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG",sys)
    
    def load_retriever_from_faiss(
            self,
            index_path: str,
            k: int =5,
            index_name: str = "index",
            search_type: str = "mmr",
            fetch_k: int =20,
            lambda_mult: float= 0.5,
            search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Load FAISS vectorstore from disk and build retriever + LCEL chain.
        
        Args:
            index_path: Path to FAISS index directory
            k: Number of documents to return
            index_name: Name of the index file
            search_type: Type of search ("similarity", "mmr", "similarity_score_threshold")
            fetch_k: Number of documents to fetch before MMR re-ranking (only for MMR)
            lambda_mult: Diversity parameter for MMR (0=max diversity, 1=max relevance)
            search_kwargs: Custom search kwargs (overrides other parameters if provided)
        """
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")
            
            embeddings = self._model_loader.load_embeddings()
            vectorstore = FAISS.load_local(
                index_path, embeddings, index_name=index_name, allow_dangerous_deserialization=True
            )

            from multi_doc_chat.utils.config_loader import load_config
            from multi_doc_chat.src.document_chat.hybrid_retrieval import (load_chunks, build_hybrid_retriever, build_reranking_retriever)
            cfg = load_config()
            hcfg = cfg.get("hybrid",{}) or {}
            rcfg = cfg.get("reranker", {}) or {}
            chunks = load_chunks(index_path) if hcfg.get("enabled") else []

            if chunks:
                rerank_on = bool(rcfg.get("enabled"))
                # fetch a wide pool when a reranker will narrow it; else return final k
                hybrid_k = hcfg.get("fetch_k",20) if rerank_on else k
                self.retriever = build_hybrid_retriever(
                    vectorstore, chunks,
                    k=hybrid_k,
                    fetch_k=hcfg.get("fetch_k", 20),
                    dense_weight=hcfg.get("dense_weight", 0.5),
                    sparse_weight=hcfg.get("sparse_weight", 0.5)
                )
                if rerank_on:
                    self.retriever = build_reranking_retriever(
                        self.retriever,
                        top_n = rcfg.get("top_n",k),
                        model_name=rcfg.get("model_name", "Xenova/ms-marco-MiniLM-L-6-v2")
                    )
                log.info("Using hybrid retrieval", reranker=rerank_on, hybrid_k=hybrid_k)
            else:
                # Dense-only fallback (hybrid off, or no chunks.jsonl for this session)
                if search_kwargs is None:
                    search_kwargs = {"k":k}
                    if search_type == 'mmr':
                        search_kwargs['fetch_k'] = fetch_k
                        search_kwargs['lambda_mult'] = lambda_mult
                self.retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
            
            self._build_lcel_chain()

            log.info(
                "FAISS retriever loaded successfully",
                index_path=index_path,
                index_name=index_name,
                search_type=search_type,
                k=k,
                fetch_k= fetch_k if search_type == 'mmr' else None,
                lambda_mult=lambda_mult if search_type == 'mmr' else None,
                session_id = self.session_id,
            )
            return self.retriever
        except Exception as e:
            log.error("Failed to load retriver from FAISS", error=str(e))
            raise DocumentPortalException("Loading error in ConversationalRAG",sys)
        
    def invoke(self, user_input:str, chat_history: Optional[List[BaseMessage]]= None) -> str:
        """Invoke the LCEL pipeline."""
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys
                )
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)
            if not answer:
                log.warning("No answer generated", user_input=user_input, session_id=self.session_id)
                return "no answer generated."
            
            # validate answer type adn length using pydantic model
            try:
                validated = ChatAnswer(answer=str(answer))
                answer= validated.answer
            except ValidationError as ve:
                log.error("Invalid chat answer", error=str(ve))
                raise DocumentPortalException("Invalid Chat answer",sys)
            log.info("Chain invoked successfully", session_id = self.session_id, user_input=user_input, answer_preview=str(answer)[:150])
            return answer
        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG",sys)

    def invoke_with_context(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> Dict[str, Any]:
        """Like invoke() but also return retrieved contexts: {"answer", "contexts"}.

        Used by the eval harness — the faithfulness / context-precision metrics need the
        retrieved chunks, which the normal chain collapses into a string and discards.
        """
        try:
            if self.retriever is None or self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke_with_context().", sys
                )
            chat_history = chat_history or []

            # rewrite into a standalone question
            rewriter = self.contextualize_prompt | self.llm | StrOutputParser()
            standalone_question = rewriter.invoke({"input":user_input, "chat_history":chat_history})

            # retrieve docs for that question
            docs  = self.retriever.invoke(standalone_question)

            # generate the answer from those same docs
            context = self._format_docs(docs)
            answer_chain = self.qa_prompt | self.llm | StrOutputParser()
            answer = answer_chain.invoke({
                "context":context,
                "input":user_input,
                "chat_history":chat_history
            })

            if not answer:
                log.warning("No answer generated", user_input=user_input, session_id=self.session_id)
                answer = "no answer generated."

            contexts = [getattr(d,"page_content", str(d)) for d in docs]
            log.info("invoke_with_context succeeded", session_id=self.session_id, num_contexts=len(contexts))
            return {"answer":answer, "contexts":contexts}
        except Exception as e:
            log.error("Failed to invoke_with_context", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG", sys)
        

    async def astream_answer(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None):
        """Async-generator of answer tokens for streaming (see docs/02, Phase 1).

            Streams the SAME LCEL chain invoke() runs — retrieval happens first (buffered),
            then the LLM's answer streams token-by-token. No per-token validation.
        """
        if self.chain is None:
            raise DocumentPortalException("RAG chain not initialized. Call load_retriever_from_faiss() before astream_answer()")
        chat_history = chat_history or []
        payload = {"input":user_input,"chat_history":chat_history}
        async for chunk in self.chain.astream(payload):
            if chunk:
                yield chunk

# INTERNALS

    def _load_llm(self):
        try:
            llm = self._model_loader.load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id = self.session_id)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", sys)

    @staticmethod
    def _format_docs(docs) ->str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain",sys)

            question_rewritter = (
                {"input": itemgetter('input'), "chat_history": itemgetter('chat_history')}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            retrieve_docs = question_rewritter | self.retriever | self._format_docs

            self.chain =(
                {
                    "context": retrieve_docs,
                    "input": itemgetter('input'),
                    "chat_history": itemgetter('chat_history'),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            log.info("LCEL graph built successfully", session_id = self.session_id)
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id) 
            raise DocumentPortalException("Failed to build LCEL chain", sys)