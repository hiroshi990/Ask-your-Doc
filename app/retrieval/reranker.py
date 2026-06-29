import logging
from typing import Any, Optional

import cohere

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class CohereReranker:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._client: Optional[cohere.ClientV2] = None
        if self.settings.cohere_api_key:
            self._client = cohere.ClientV2(api_key=self.settings.cohere_api_key)

    async def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        if not chunks:
            return []

        top_k = top_k or self.settings.rerank_top_k

        if not self._client:
            logger.warning("Cohere API key not set; returning RRF order without reranking")
            return chunks[:top_k]

        documents = [c["text"] for c in chunks]
        try:
            response = self._client.rerank(
                model="rerank-v3.5",
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
            )
        except Exception:
            logger.exception("Cohere rerank failed; falling back to RRF order")
            return chunks[:top_k]

        reranked: list[dict[str, Any]] = []
        for result in response.results:
            chunk = {**chunks[result.index]}
            chunk["rerank_score"] = result.relevance_score
            reranked.append(chunk)

        return reranked
