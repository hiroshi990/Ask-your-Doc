"""Hybrid retrieval pipeline: dense + sparse -> RRF -> dedup -> rerank."""

import logging
from typing import Any, Optional

from app.config import Settings, get_settings
from app.ingestion.embedder import BGEEmbedder, get_embedder
from app.retrieval.reranker import CohereReranker
from app.retrieval.rrf import deduplicate_chunks, reciprocal_rank_fusion
from app.storage.bm25_index import BM25Index
from app.storage.qdrant_store import QdrantStore
from langsmith import traceable

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Retrieval pipeline:
    Dense (Qdrant) + Sparse (BM25) -> RRF Fusion -> Deduplication -> Reranking
    """

    def __init__(
        self,
        qdrant: Optional[QdrantStore] = None,
        bm25: Optional[BM25Index] = None,
        embedder: Optional[BGEEmbedder] = None,
        reranker: Optional[CohereReranker] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.qdrant = qdrant or QdrantStore()
        self.bm25 = bm25 or BM25Index()
        self.embedder = embedder or get_embedder()
        self.reranker = reranker or CohereReranker()

    @traceable(name='retriever')
    def retrieve(
        self,
        query: str,
        metadata_filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        query_vector = self.embedder.embed_query(query)

        dense_results = self.qdrant.dense_search(
            query_vector=query_vector,
            top_k=self.settings.dense_top_k,
            metadata_filters=metadata_filters,
        )
        logger.info("Dense retrieval: %d results", len(dense_results))

        sparse_results = self.bm25.search(
            query=query,
            top_k=self.settings.sparse_top_k,
            metadata_filters=metadata_filters,
        )
        logger.info("Sparse retrieval: %d results", len(sparse_results))

        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self.settings.rrf_k,
        )
        deduped = deduplicate_chunks(fused)
        logger.info("After RRF + dedup: %d results", len(deduped))

        reranked = self.reranker.rerank(
            query=query,
            chunks=deduped,
            top_k=self.settings.rerank_top_k,
        )
        logger.info("After reranking: %d results", len(reranked))

        return reranked
