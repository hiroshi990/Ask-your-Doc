"""BM25 sparse retrieval index (single corpus)."""

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Optional

from rank_bm25 import BM25Okapi

from app.config import Settings, get_settings
from app.ingestion.chunker import ChunkRecord

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Single-corpus BM25 index with disk persistence."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._index_dir = self.settings.data_dir / "bm25"
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._index_dir / "corpus.pkl"
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: list[dict[str, Any]] = []

    def _load(self) -> tuple[Optional[BM25Okapi], list[dict[str, Any]]]:
        if self._bm25 is not None:
            return self._bm25, self._corpus

        if self._index_path.exists():
            with open(self._index_path, "rb") as f:
                data = pickle.load(f)
            self._bm25 = data["bm25"]
            self._corpus = data["corpus"]
            return self._bm25, self._corpus

        return None, []

    def _save(self) -> None:
        with open(self._index_path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "corpus": self._corpus}, f)

    def _rebuild(self, corpus: list[dict[str, Any]]) -> None:
        if not corpus:
            self._bm25 = None
            self._corpus = []
        else:
            tokenized = [entry["tokens"] for entry in corpus]
            self._bm25 = BM25Okapi(tokenized)
            self._corpus = corpus
        self._save()

    def add_chunks(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return

        _, corpus = self._load()

        new_entries = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "tokens": _tokenize(chunk.text),
            }
            for chunk in chunks
        ]

        corpus = corpus + new_entries
        self._rebuild(corpus)
        logger.info("BM25 index updated: %d total chunks", len(corpus))

    def remove_document(self, document_id: str) -> None:
        _, corpus = self._load()
        corpus = [e for e in corpus if e["document_id"] != document_id]
        self._rebuild(corpus)
        logger.info(
            "Removed document %s from BM25 index: %d chunks remain",
            document_id,
            len(corpus),
        )

    def search(
        self,
        query: str,
        top_k: int = 20,
        metadata_filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        bm25, corpus = self._load()
        if not corpus or bm25 is None:
            return []

        query_tokens = _tokenize(query)
        scores = bm25.get_scores(query_tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results: list[dict[str, Any]] = []
        for idx, score in ranked:
            if score <= 0:
                continue
            entry = corpus[idx]
            meta = entry["metadata"]
            if metadata_filters:
                if not all(meta.get(k) == v for k, v in metadata_filters.items()):
                    continue
            results.append({
                "chunk_id": entry["chunk_id"],
                "document_id": entry["document_id"],
                "text": entry["text"],
                "metadata": meta,
                "sparse_score": float(score),
            })
            if len(results) >= top_k:
                break

        return results
