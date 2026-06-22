import logging
from functools import lru_cache
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class BGEEmbedder:
    """Dense embeddings using BAAI/bge-m3."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        logger.info("Loading embedding model: %s", self.settings.embedding_model)
        self._model = SentenceTransformer(
            self.settings.embedding_model,
            device=self.settings.embedding_device,
        )
        self._dimension = self._model.get_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension or 0

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.array([])
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self._model.encode(
            query,
            normalize_embeddings=True,
        )
        return np.asarray(embedding, dtype=np.float32)


@lru_cache
def get_embedder() -> BGEEmbedder:
    return BGEEmbedder()
