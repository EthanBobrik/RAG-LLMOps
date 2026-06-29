# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This is an **early-stage scaffold** for "MultiDocChat" — a RAG (Retrieval-Augmented Generation) app that lets users upload documents, index them, and chat against them. Most of the structure is intentional placeholders; very little is wired together yet. Be aware of what exists vs. what is planned:

- **Implemented**: the browser frontend (`templates/index.html`, `static/styles.css`) and an in-progress RAG pipeline prototype in `notebook/RAG.ipynb`.
- **Stub only**: `main.py` is a `print("Hello from rag-llmops!")` placeholder, not the app entrypoint.
- **Empty placeholders**: the `multi_doc_chat/` package and all its subpackages (`config/`, `exceptions/`, `logger/`, `model/`, `prompts/`, `src/`, `utils/`) contain no files yet. Same for `data/` and `test/`.

When asked to "build the backend," the intended target is a web server (the frontend below tells you the contract).

## Environment & commands

- Python **3.12**, dependencies managed with **`uv`** (`uv.lock` is the source of truth; `pyproject.toml` lists deps).
- `requirements.txt` is a minimal/stale subset — prefer `pyproject.toml` + `uv` for the real dependency set.
- Install / sync deps: `uv sync`
- Run a one-off command in the project env: `uv run <cmd>` (e.g. `uv run python main.py`)
- Add a dependency: `uv add <package>` (updates `pyproject.toml` and `uv.lock`)
- Secrets live in `.env` (loaded via `python-dotenv`). Required: `OPENAI_API_KEY`.

There is no test runner, linter, or build step configured yet. If you add tests, put them in `test/`.

## Intended architecture

The frontend in `templates/index.html` is the de-facto API spec for the backend that needs to be built. It expects a server that:

- Serves `index.html` and mounts `static/` at `/static`.
- `POST /upload` — accepts `multipart/form-data` with one or more `files`; ingests + indexes them; returns JSON `{ session_id }`. The frontend persists `session_id` in `localStorage` (`mdc_session_id`) and scopes all chat to it.
- `POST /chat` — accepts JSON `{ session_id, message }`; runs retrieval + LLM over that session's index; returns JSON `{ answer }`. Errors should return JSON with a `detail` field (the frontend reads `.detail`).

The RAG core (document loading → chunking → embeddings → vector store → retrieval → LLM answer) is being prototyped in `notebook/RAG.ipynb` using **LangChain** + **OpenAI**. When promoting notebook code into the package, the empty `multi_doc_chat/` subpackages name the intended layering: `config/`, `logger/`, `exceptions/`, `prompts/`, `model/`, `utils/`, with orchestration in `src/`.

Key libraries (from `pyproject.toml`): `langchain` (v1.x) + `langchain-community`, `openai` (v2.x), `tiktoken`, and `rapidocr-onnxruntime` (OCR for image/scanned-document ingestion).
