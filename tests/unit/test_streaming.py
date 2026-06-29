import asyncio
import pytest

from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException


def test_astream_answer_yields_tokens(stub_model_loader):
    rag = ConversationalRAG(session_id="s1")

    class _FakeChain:
        async def astream(self, payload):
            for t in ["Hello", ", ", "world"]:
                yield t

    rag.chain = _FakeChain()  # astream_answer only needs self.chain

    async def _collect():
        return [c async for c in rag.astream_answer("hi", [])]

    # asyncio.run drives the async generator from a sync test (no pytest-asyncio needed)
    assert asyncio.run(_collect()) == ["Hello", ", ", "world"]


def test_astream_answer_raises_without_chain(stub_model_loader):
    rag = ConversationalRAG(session_id="s1")  # no retriever loaded -> chain is None

    async def _collect():
        return [c async for c in rag.astream_answer("hi", [])]

    with pytest.raises(DocumentPortalException):
        asyncio.run(_collect())
