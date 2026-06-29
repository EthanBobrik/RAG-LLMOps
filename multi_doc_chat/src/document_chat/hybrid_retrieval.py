"""Hybrid (BM25 + dense) retrieval with cross-encoder reranking.

TODO: implement. See docs/03_hybrid_retrieval_rerank.md.
Do optional imports (rank_bm25, sentence_transformers, reranker classes) lazily INSIDE
the functions so importing this module never breaks the app when extras aren't installed.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, TYPE_CHECKING
from langchain_core.documents.compressor import BaseDocumentCompressor

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

CHUNKS_FILE = "chunks.jsonl"
_RERANKER_CACHE: dict = {} # model_name -> TextCrossEncoder (loaded once)

def _get_cross_encoder(model_name:str):
    enc = _RERANKER_CACHE.get(model_name)
    if enc is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        enc = TextCrossEncoder(model_name=model_name)
        _RERANKER_CACHE[model_name] = enc
    return enc

class FastEmbedReranker(BaseDocumentCompressor):
    """LangChain compressor backed by fastembed's onnx cross-encoder (no torch).

      Re-scores each retrieved doc against the query and keeps the top_n most relevant.
    """
    model_name: str = "Xenova/ms-macro-MiniLM-L-6-v2"
    top_n: int = 5

    def compress_documents(self, documents, query, callbacks = None):
        docs = list(documents)
        if not docs:
            return docs
        scores = list(_get_cross_encoder(self.model_name).rerank(
            query, [d.page_content for d in docs]
        ))
        ranked = sorted(zip(docs, scores), key=lambda ds: ds[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_n]]

def persist_chunks(chunks: List["Document"], index_dir: Path) -> Path:
    """Write chunks to <index_dir>/chunks.jsonl so BM25 can be
  rebuilt on load.

      FAISS persists only vectors; BM25 needs the raw chunk text + metadata.
      """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    out = index_dir / CHUNKS_FILE
    with out.open("w",encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(
                {"page_content": c.page_content, "metadata":c.metadata or {}}, ensure_ascii=False
            )+ "\n")
    return out

def load_chunks(index_dir: Path) -> List["Document"]:
    """Reload chunks from <index_dir>/chunks.jsonl into Documents ([] if absent)."""    
    from langchain_core.documents import Document
    path = Path(index_dir) / CHUNKS_FILE
    if not path.exists():
        return []
    docs: List["Document"] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        docs.append(Document(page_content=rec['page_content'], metadata=rec.get("metadata",{})))
    return docs

def build_hybrid_retriever(
    faiss_vectorstore,
    chunks: List["Document"],
    *,
    k: int = 5,
    fetch_k: int = 20,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> "BaseRetriever":
    """Fuse dense (FAISS/MMR) + sparse (BM25) via EnsembleRetriever. See docs/03, Phase 2."""
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever

    # Dense leg: semantic similarity with MMR diversification
    dense = faiss_vectorstore.as_retriever(
        search_type = "mmr",
        search_kwargs={"k":k, "fetch_k":fetch_k,"lambda_mult":0.5}
    )

    # Sparse leg: lexical BM25 over the same chunks
    sparse = BM25Retriever.from_documents(chunks)
    sparse.k = k

    # Reciprocal-rank fusion of the two legs; weights bias toward dense vs sparse
    return EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[dense_weight, sparse_weight]
    )


def build_reranking_retriever(
    base_retriever: "BaseRetriever",
    *,
    top_n: int = 5,
    model_name: str = "Xenova/ms-macro-MiniLM-L-6-v2",
) -> "BaseRetriever":
    """Wrap a retriever with a cross-encoder reranker. See docs/03, Phase 3."""
    from langchain_classic.retrievers import ContextualCompressionRetriever
    reranker = FastEmbedReranker(model_name=model_name, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )
