import main


class _FakeRAG:
    def __init__(self, session_id=None, retriever=None):
        self.session_id = session_id

    def load_retriever_from_faiss(self, index_path, **kwargs):
        return None


def test_rag_cache_evicts_least_recently_used(clear_sessions, monkeypatch):
    monkeypatch.setattr(main, "ConversationalRAG", _FakeRAG)
    monkeypatch.setattr(main, "RAG_CACHE_MAX", 2)

    main._get_rag("s1")
    main._get_rag("s2")
    main._get_rag("s1")   # touch s1 -> now most-recently-used
    main._get_rag("s3")   # over capacity -> evict the LRU entry (s2)

    assert set(main.RAG_CACHE.keys()) == {"s1", "s3"}


def test_rag_cache_reuses_same_instance(clear_sessions, monkeypatch):
    monkeypatch.setattr(main, "ConversationalRAG", _FakeRAG)
    first = main._get_rag("sess")
    second = main._get_rag("sess")
    assert first is second  # cached, not rebuilt
