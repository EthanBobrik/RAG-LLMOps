from multi_doc_chat.utils.session_store import InMemorySessionStore, get_session_store


def test_in_memory_store_lifecycle():
    s = InMemorySessionStore()
    assert not s.exists("a")

    s.create("a")
    assert s.exists("a")
    assert s.history("a") == []

    s.append("a", "user", "hi")
    s.append("a", "assistant", "yo")
    assert s.history("a") == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]

    # history() returns a copy — mutating it must not corrupt the store
    s.history("a").append({"role": "x", "content": "y"})
    assert len(s.history("a")) == 2

    s.clear()
    assert not s.exists("a")


def test_factory_defaults_to_in_memory(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert isinstance(get_session_store(), InMemorySessionStore)
