import pathlib
import pytest
from langchain_core.runnables import RunnableLambda
from langchain_classic.schema import Document

from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException


def test_conversationalrag_error_handling(tmp_dirs, stub_model_loader):
    rag = ConversationalRAG(session_id="s1")
    with pytest.raises(DocumentPortalException):
        rag.invoke("hello")
    with pytest.raises(DocumentPortalException):
        rag.load_retriever_from_faiss(index_path="faiss_index/does_not_exist")
    

def _wire_runnable_stubs(rag):
    # conftest's _StubLLM isn't a Runnable, so it can't be used in the LCEL
    # `prompt | llm | parser` pipeline. Swap llm + retriever for RunnableLambdas.
    rag.llm = RunnableLambda(lambda _:"stubbed answer")
    rag.retriever = RunnableLambda(
        lambda q: [
            Document(page_content="chunk A", metadata={"source":"a.txt"}),
            Document(page_content="chunk B", metadata={"source":"b.txt"})
        ]
    )
    rag.chain = object()

def test_invoke_with_context_returns_answer_and_contexts(tmp_dirs, stub_model_loader):
    rag = ConversationalRAG(session_id="s1")
    _wire_runnable_stubs(rag)

    result = rag.invoke_with_context("what is X?", chat_history=[])

    assert set(result.keys()) == {"answer", "contexts"}
    assert result['answer'] == "stubbed answer"
    assert result['contexts'] == ['chunk A','chunk B']
    assert all(isinstance(c,str) for c in result['contexts'])

def test_invoke_with_context_raises_without_retriever(tmp_dirs, stub_model_loader):
    rag=ConversationalRAG(session_id="s1")
    with pytest.raises(DocumentPortalException):
        rag.invoke_with_context("hello")