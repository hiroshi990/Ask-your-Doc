import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import Settings, get_settings
from app.models.schemas import DocumentInfo

logger = logging.getLogger(__name__)


class DocumentStore:
    """File-backed registry of uploaded documents (not used for retrieval)."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._store_path = self.settings.data_dir / "documents.json"
        self._store_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        if not self._store_path.exists():
            return {"documents": {}}
        with open(self._store_path, encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: dict) -> None:
        with open(self._store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def register(self, doc_info: DocumentInfo) -> None:
        data = self._load()
        data["documents"][doc_info.document_id] = {
            "document_id": doc_info.document_id,
            "document_name": doc_info.document_name,
            "source_type": doc_info.source_type,
            "filename": doc_info.filename,
            "chunk_count": doc_info.chunk_count,
            "uploaded_at": doc_info.uploaded_at.isoformat(),
        }
        self._save(data)
        logger.info("Registered document: %s", doc_info.document_name)

    def list_all(self) -> list[DocumentInfo]:
        data = self._load()
        return [
            DocumentInfo(
                document_id=doc["document_id"],
                document_name=doc["document_name"],
                source_type=doc["source_type"],
                filename=doc.get("filename"),
                chunk_count=doc["chunk_count"],
                uploaded_at=datetime.fromisoformat(doc["uploaded_at"]),
            )
            for doc in data.get("documents", {}).values()
        ]

    def get(self, document_id: str) -> Optional[DocumentInfo]:
        data = self._load()
        doc = data.get("documents", {}).get(document_id)
        if not doc:
            return None
        return DocumentInfo(
            document_id=doc["document_id"],
            document_name=doc["document_name"],
            source_type=doc["source_type"],
            filename=doc.get("filename"),
            chunk_count=doc["chunk_count"],
            uploaded_at=datetime.fromisoformat(doc["uploaded_at"]),
        )

    
    def delete_all(self) -> None:
        data = self._load()
        data['documents'] = {}
        self._save(data)
        logger.info("Removed all documents from document store")
    
    def delete(self, document_id: str) -> bool:
        data = self._load()
        if document_id not in data.get("documents", {}):
            return False
        del data["documents"][document_id]
        self._save(data)
        logger.info("Removed document from manifest: %s", document_id)
        return True
