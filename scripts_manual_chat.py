"""Manual end-to-end smoke script for ingestion + conversational RAG.

Not a pytest test — it starts an interactive chat loop. Run directly:

    python scripts_manual_chat.py [path/to/doc1 path/to/doc2 ...]

With no arguments it ingests every supported file in ./data.
Requires GOOGLE_API_KEY (and GROQ_API_KEY if LLM_PROVIDER=groq) in your .env.
"""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
DATA_DIR = Path(__file__).resolve().parent / "data"


def _resolve_input_files(argv: list[str]) -> list[Path]:
    if argv:
        return [Path(p) for p in argv]
    # Default: everything supported in ./data
    return [p for p in sorted(DATA_DIR.glob("*")) if p.suffix.lower() in SUPPORTED_EXTENSIONS]


def run_manual_chat(argv: list[str]) -> None:
    try:
        candidate_paths = _resolve_input_files(argv)
        uploaded_files = []
        for file_path in candidate_paths:
            if file_path.exists():
                uploaded_files.append(open(file_path, "rb"))
            else:
                print(f"File does not exist: {file_path}")

        if not uploaded_files:
            print("No valid files to ingest. Pass file paths or add documents to ./data.")
            sys.exit(1)

        # Build index using single-module ChatIngestor.
        ci = ChatIngestor(temp_base="data", faiss_base="faiss_index", use_session_dirs=True)

        # MMR (Maximal Marginal Relevance) for diverse results.
        # - fetch_k: documents fetched before MMR re-ranking
        # - lambda_mult: 0=max diversity, 1=max relevance, 0.5=balanced
        ci.build_retriever(
            uploaded_files,
            chunk_size=200,
            chunk_overlap=20,
            k=5,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5,
        )

        # Close file handles.
        for f in uploaded_files:
            try:
                f.close()
            except Exception:
                pass

        session_id = ci.session_id
        index_dir = os.path.join("faiss_index", session_id)

        # Load RAG with MMR search.
        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(
            index_path=index_dir,
            k=5,
            index_name=os.getenv("FAISS_INDEX_NAME", "index"),
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5,
        )

        # Interactive multi-turn chat loop.
        chat_history = []
        print("\nType 'exit' to quit the chat.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q", ":q"}:
                print("Goodbye!")
                break

            answer = rag.invoke(user_input, chat_history=chat_history)
            print("Assistant:", answer)

            # Maintain conversation history for context in subsequent turns.
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

    except Exception as e:
        print(f"Manual chat failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    run_manual_chat(sys.argv[1:])
