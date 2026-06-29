# Build Guide 2 — End-to-End Token Streaming (SSE)

**Goal:** stream the assistant's answer token-by-token from the LCEL chain through
FastAPI to the browser, rendering incrementally instead of waiting for the full
response. This collapses *perceived* latency (time-to-first-token) and demonstrates
async streaming competence.

**End state resume bullet:**
> Implemented end-to-end token streaming (FastAPI `StreamingResponse` + LangChain LCEL
> `astream`) with incremental browser rendering, cutting time-to-first-token from ~Ns to
> <500ms and eliminating the blocking "Thinking…" wait.

---

## How it works (architecture)

Today `/chat` calls `rag.invoke(...)`, which blocks until the entire answer is generated,
then returns one JSON blob. The frontend shows "Thinking…" the whole time.

New flow:
1. Browser POSTs to a new `/chat/stream` endpoint.
2. FastAPI returns a `StreamingResponse` backed by an **async generator**.
3. The generator consumes `chain.astream(payload)` — because the chain ends in
   `StrOutputParser()`, each yielded chunk is a plain string token.
4. Each token is emitted as a **Server-Sent Event** (`data: <token>\n\n`).
5. The browser reads the stream with `fetch()` + a `ReadableStream` reader and appends
   tokens to the assistant bubble as they arrive.

> Note: native `EventSource` only does GET. Because `/chat` is a POST with a JSON body,
> use `fetch()` streaming (below), not `EventSource`. We keep the SSE wire format (`data:`
> lines) because it's simple and standard.

---

## Phase 1 — Add a streaming method to the RAG layer

1. **In `retrieval.py`**, add an async method:
   `async def astream_answer(self, user_input, chat_history=None)` that:
   - Guards `self.chain is None` exactly like `invoke()` does.
   - Builds the same `payload = {"input": ..., "chat_history": ...}`.
   - `async for chunk in self.chain.astream(payload): yield chunk`.
2. **Skip per-token pydantic validation.** The `ChatAnswer` length check in `invoke()`
   can't run mid-stream; either drop it for streaming or validate the *accumulated*
   string after the loop and log (don't raise) if it's off.
3. **Keep `invoke()`** for the eval harness and non-streaming callers.
4. **Unit test:** with `stub_model_loader`, the stub LLM must support `astream`. Add an
   async stub that yields a couple of chunks; assert you can collect them into the full
   string. Mark the test `@pytest.mark.asyncio` (add `pytest-asyncio`: `uv add --dev
   pytest-asyncio`).

## Phase 2 — Add the streaming endpoint

1. **In `main.py`**, add `POST /chat/stream`:
   - Same validation as `/chat` (session exists, message non-empty).
   - Build `ConversationalRAG`, `load_retriever_from_faiss(...)`, convert history to
     `HumanMessage`/`AIMessage` (reuse the existing logic — consider extracting a helper
     to avoid duplication with `/chat`).
   - Define an async generator `event_stream()`:
     ```
     full = []
     try:
         async for token in rag.astream_answer(message, lc_history):
             full.append(token)
             yield f"data: {json.dumps({'token': token})}\n\n"
         # after generation: persist history and signal completion
         answer = "".join(full)
         SESSIONS[session_id].append({"role": "user", "content": message})
         SESSIONS[session_id].append({"role": "assistant", "content": answer})
         yield f"data: {json.dumps({'done': True})}\n\n"
     except Exception as e:
         yield f"data: {json.dumps({'error': str(e)})}\n\n"
     ```
   - Return `StreamingResponse(event_stream(), media_type="text/event-stream")` with
     headers `Cache-Control: no-cache` and `X-Accel-Buffering: no` (prevents proxy
     buffering that would defeat streaming).
2. **JSON-encode each token** (don't raw-concatenate) so newlines/spaces survive the SSE
   `data:` framing. Decode on the client.
3. **Persist history only after** the stream completes successfully, so a half-failed
   generation doesn't poison the session.

## Phase 3 — Frontend incremental rendering

1. **In `templates/index.html`**, change `sendMessage()` to call `/chat/stream` and read
   the body as a stream:
   ```js
   const res = await fetch('/chat/stream', {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({ session_id: sessionId, message: text })
   });
   const reader = res.body.getReader();
   const decoder = new TextDecoder();
   const bubble = appendMessage('assistant', '');   // create empty bubble, return its node
   let buffer = '';
   while (true) {
     const { value, done } = await reader.read();
     if (done) break;
     buffer += decoder.decode(value, { stream: true });
     // split on SSE event boundaries
     const parts = buffer.split('\n\n');
     buffer = parts.pop();                 // keep incomplete trailing chunk
     for (const part of parts) {
       if (!part.startsWith('data: ')) continue;
       const payload = JSON.parse(part.slice(6));
       if (payload.token) { bubble.textContent += payload.token; scrollToBottom(); }
       if (payload.error) { toast('Stream error'); }
       // payload.done -> finished
     }
   }
   ```
2. **Refactor `appendMessage`** to return the created bubble element so you can append
   tokens to it (today it returns nothing).
3. **Replace the binary "Thinking…" toggle** with: show the indicator until the **first**
   token arrives, then hide it — that first-token moment is the latency win you're
   demonstrating.
4. **Keep the old non-streaming path** as a fallback if `res.body` is unsupported.

## Phase 4 — Measure (this is the number on your resume)

1. **Instrument time-to-first-token (TTFT):** in the endpoint, record a monotonic
   timestamp when the request starts and when the first token is yielded; log the delta
   via the structlog logger. Also log total generation time.
2. **Compare** TTFT (streaming) vs. total latency of the old `/chat` blocking call on the
   same questions. The ratio is your bullet ("~Ns blocking → <500ms TTFT").
3. **Optional flourish:** expose these timings on the `/health` or a small `/metrics`
   endpoint — pairs naturally with the observability upgrade.

## Phase 5 — Tests & edge cases

- [ ] Async unit test for `astream_answer` collecting chunks.
- [ ] Integration test for `/chat/stream`: stub the RAG to yield tokens; assert the
      response is `text/event-stream` and the concatenated tokens equal the expected
      answer. Use `with client.stream("POST", ...)` in Starlette's TestClient.
- [ ] Invalid/expired `session_id` still returns a clean error event, not a 500 mid-stream.
- [ ] Empty message rejected before the stream opens.
- [ ] History is persisted exactly once, only on success.

## Definition of done

- [ ] `astream_answer()` async generator on `ConversationalRAG`, unit-tested.
- [ ] `POST /chat/stream` returns SSE; history persisted post-completion.
- [ ] Frontend renders tokens incrementally into a live bubble; indicator hides on first
      token.
- [ ] Logged TTFT vs. blocking-latency comparison with a real measured delta.
