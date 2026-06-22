import logging
import uuid
from typing import Any, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import Settings, get_settings
from app.ingestion.chunker import ChunkRecord

logger = logging.getLogger(__name__)


class QdrantStore:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._client = QdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
        )
        self._collection = self.settings.qdrant_collection_name

    def ensure_collection(self, dimension: int) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s", self._collection)

    def upsert_chunks(
        self,
        chunks: list[ChunkRecord],
        embeddings: np.ndarray,
    ) -> None:
        if len(chunks) == 0:
            return
        dimension = embeddings.shape[1]
        self.ensure_collection(dimension)

        points = []
        for chunk, vector in zip(chunks, embeddings):
            payload = {**chunk.metadata, "text": chunk.text}
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        self._client.upsert(collection_name=self._collection, points=points)
        logger.info("Upserted %d points to %s", len(points), self._collection)

    def dense_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        metadata_filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            return []

        must_conditions = []
        if metadata_filters:
            for key, value in metadata_filters.items():
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        results = self._client.query_points(
            collection_name=self._collection,
            query=query_vector.tolist(),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        
        return [
            {
                "chunk_id": hit.payload.get("chunk_id", str(hit.id)),  # pyright: ignore[reportOptionalMemberAccess]
                "document_id": hit.payload.get("document_id", ""),  # pyright: ignore[reportOptionalMemberAccess]
                "text": hit.payload.get("text", ""),  # pyright: ignore[reportOptionalMemberAccess]
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"},  # pyright: ignore[reportOptionalMemberAccess]
                "dense_score": hit.score
            }
            for hit in results.points
        ]
    def delete_everything(self) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            return 
            logger.info("Already empty")
        self._client.delete_collection(collection_name=self._collection)
        logger.info("Deleted entire collection")
    
    def delete_by_document_id(self, document_id: str) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            return

        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
        logger.info("Deleted chunks for document %s from Qdrant", document_id)
