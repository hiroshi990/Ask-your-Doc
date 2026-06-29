import logging
import uuid
from typing import Any, Optional

import numpy as np
from qdrant_client import AsyncQdrantClient
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
import asyncio
logger = logging.getLogger(__name__)


class QdrantStore:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._client = AsyncQdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
        )
        self._collection = self.settings.qdrant_collection_name

    async def ensure_collection(self, dimension: int) -> None:
        collections = [c.name for c in await self._client.get_collections().collections]
        if self._collection not in collections:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s", self._collection)

    async def upsert_chunks(
        self,
        chunks: list[ChunkRecord],
        embeddings: np.ndarray,
    ) -> None:
        if len(chunks) == 0:
            return
        dimension = embeddings.shape[1]
        await self.ensure_collection(dimension)

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

        await self._client.upsert(collection_name=self._collection, points=points)
        logger.info("Upserted %d points to %s", len(points), self._collection)

    async def dense_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20
    ) -> list[dict[str, Any]]:
        collections = [c.name for c in await self._client.get_collections().collections]
        if self._collection not in collections:
            return []

        results = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector.tolist(),
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
    async def delete_everything(self) -> None:
        collections = [c.name for c in await self._client.get_collections().collections]
        if self._collection not in collections:
            return 
        logger.info("Already empty")    
        await self._client.delete_collection(collection_name=self._collection)
        logger.info("Deleted entire collection")
    
    async def delete_by_document_id(self, document_id: str) -> None:
        collections = [c.name for c in await self._client.get_collections().collections]
        if self._collection not in collections:
            return

        await self._client.delete(
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
