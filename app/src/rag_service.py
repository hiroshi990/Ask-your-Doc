"""Orchestrates the complete RAG pipeline."""

import logging

from app.cache.redis_cache import RedisCache
from app.generation.answer import AnswerGenerator
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    RetrievedChunk,
)
from app.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.cache = RedisCache()
        self.generator = AnswerGenerator()

    def chat(self, request: ChatRequest) -> ChatResponse:
        if request.use_cache:
            cached = self.cache.get(request.query)
            if cached:
                citations = [
                    Citation(**c) if isinstance(c, dict) else c
                    for c in cached.get("citations", [])
                ]
                return ChatResponse(
                    query=request.query,
                    answer=cached["answer"],
                    citations=citations,
                    retrieved_chunks=[],
                    cache_hit=True,
                )

        chunks = self.retriever.retrieve(
            query=request.query,
        )

        answer, citations = self.generator.generate(request.query, chunks)

        retrieved = [
            RetrievedChunk(
                chunk_id=c["chunk_id"],
                document_id=c.get("document_id", ""),
                text=c["text"],
                metadata=c.get("metadata", {}),
                dense_score=c.get("dense_score"),
                sparse_score=c.get("sparse_score"),
                rrf_score=c.get("rrf_score"),
                rerank_score=c.get("rerank_score"),
            )
            for c in chunks
        ]

        if request.use_cache:
            self.cache.set(
                query=request.query,
                answer=answer,
                citations=[c.model_dump() for c in citations],
                chunk_ids=[c.chunk_id for c in retrieved],
            )

        return ChatResponse(
            query=request.query,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved,
            cache_hit=False,
        )
