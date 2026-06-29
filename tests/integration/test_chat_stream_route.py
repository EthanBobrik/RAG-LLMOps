import json

import main


def _parse_sse(body: str):
    """Parse an SSE body into (tokens, done, error)."""
    tokens, done, error = [], False, None
    for frame in body.split("\n\n"):
        frame = frame.strip()
        if not frame.startswith("data: "):
            continue
        payload = json.loads(frame[len("data: "):])
        if "token" in payload:
            tokens.append(payload["token"])
        if payload.get("done"):
            done = True
        if "error" in payload:
            error = payload["error"]
    return tokens, done, error


class _FakeStreamRAG:
    def __init__(self, session_id=None, retriever=None):
        self.session_id = session_id

    def load_retriever_from_faiss(self, index_path, **kwargs):
        return None

    async def astream_answer(self, message, chat_history=None):
        for tok in ["Hello", ", ", "world"]:
            yield tok


def test_chat_stream_streams_tokens_and_persists_once(client, clear_sessions, monkeypatch):
    sid = "sess_stream"
    main.SESSIONS.create(sid)
    monkeypatch.setattr(main, "ConversationalRAG", _FakeStreamRAG)

    resp = client.post("/chat/stream", json={"session_id": sid, "message": "hi"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    tokens, done, error = _parse_sse(resp.text)
    assert error is None
    assert "".join(tokens) == "Hello, world"
    assert done is True

    # History persisted exactly once, only after the stream completed.
    assert main.SESSIONS.history(sid) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello, world"},
    ]


def test_chat_stream_invalid_session_returns_400(client, clear_sessions):
    resp = client.post("/chat/stream", json={"session_id": "does-not-exist", "message": "hi"})
    assert resp.status_code == 400


def test_chat_stream_empty_message_returns_400(client, clear_sessions):
    sid = "sess_empty"
    main.SESSIONS.create(sid)
    resp = client.post("/chat/stream", json={"session_id": sid, "message": "   "})
    assert resp.status_code == 400
