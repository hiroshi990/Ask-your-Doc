"""End-to-end document ingestion pipeline."""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.cache.redis_cache import RedisCache
from app.ingestion.chunker import ChunkRecord, StructureAwareChunker
from app.ingestion.embedder import BGEEmbedder, get_embedder
from app.ingestion.parser import DocumentParser
from app.models.schemas import DocumentInfo
from app.storage.bm25_index import BM25Index
from app.storage.document_store import DocumentStore
from app.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        document_store: Optional[DocumentStore] = None,
        qdrant: Optional[QdrantStore] = None,
        bm25: Optional[BM25Index] = None,
        embedder: Optional[BGEEmbedder] = None,
        cache: Optional[RedisCache] = None,
    ) -> None:
        self.parser = DocumentParser()
        self.chunker = StructureAwareChunker()
        self.document_store = document_store or DocumentStore()
        self.qdrant = qdrant or QdrantStore()
        self.bm25 = bm25 or BM25Index()
        self.embedder = embedder or get_embedder()
        self.cache = cache or RedisCache()

    def ingest_file(
        self,
        file_path: Path,
        document_name: Optional[str] = None,
    ) -> tuple[DocumentInfo, list[ChunkRecord]]:
        document_id = str(uuid.uuid4())
        name = document_name or file_path.stem
        source_type = DocumentParser.detect_source_type(file_path)

        doc = self.parser.parse_file(file_path)
        chunks = self.chunker.chunk_document(
            doc=doc,
            document_id=document_id,
            document_name=name,
            source_type=source_type,
        )

        self._index_chunks(chunks)

        now = datetime.now(timezone.utc)
        doc_info = DocumentInfo(
            document_id=document_id,
            document_name=name,
            source_type=source_type,
            filename=file_path.name,
            chunk_count=len(chunks),
            uploaded_at=now,
        )
        self.document_store.register(doc_info)
        self.cache.flush_all()
        logger.info("Ingested %s: %d chunks", name, len(chunks))
        return doc_info, chunks

    def ingest_pasted_text(
        self,
        content: str,
        title: Optional[str] ,
    ) -> tuple[DocumentInfo, list[ChunkRecord]]:
        document_id = str(uuid.uuid4())
        source_type = "pasted_text"

        doc = self.parser.parse_text(content, title=title or "Pasted Text")
        chunks = self.chunker.chunk_document(
            doc=doc,
            document_id=document_id,
            document_name=title or "Pasted Text",
            source_type=source_type,
        )

        self._index_chunks(chunks)

        now = datetime.now(timezone.utc)
        doc_info = DocumentInfo(
            document_id=document_id,
            document_name=title or "Pasted Text",
            source_type=source_type,
            filename=None,
            chunk_count=len(chunks),
            uploaded_at=now,
        )
        self.document_store.register(doc_info)
        self.cache.flush_all()
        return doc_info, chunks

    def delete_document(self, document_id: str) -> bool:
        doc = self.document_store.get(document_id)
        if not doc:
            return False

        self.qdrant.delete_by_document_id(document_id)
        self.bm25.remove_document(document_id)
        self.document_store.delete(document_id)
        self.cache.flush_all()
        logger.info("Deleted document %s and all its chunks", document_id)
        return True
    
    def database_flush(self) -> bool:
        doc = self.document_store._load()
        if not doc :
            return False
        
        self.qdrant.delete_everything()
        self.bm25.remove_everything()
        self.document_store.delete_all()
        self.cache.flush_all()
        logger.info("Removed everything from database")
        return True


    def _index_chunks(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        self.qdrant.upsert_chunks(chunks=chunks, embeddings=embeddings)
        self.bm25.add_chunks(chunks)
