import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import DoclingDocument
from transformers import AutoTokenizer

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class StructureAwareChunker:
    
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._tokenizer = AutoTokenizer.from_pretrained(self.settings.embedding_model)
        hf_tokenizer = HuggingFaceTokenizer(
            tokenizer=self._tokenizer,
            max_tokens=self.settings.chunk_max_tokens,
        )
        self._hybrid_chunker = HybridChunker(
            tokenizer=hf_tokenizer,
            merge_peers=True,
        )

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def _extract_page_number(self, chunk_meta: Any) -> Optional[int]:
        if not hasattr(chunk_meta, "doc_items") or not chunk_meta.doc_items:
            return None
        for item in chunk_meta.doc_items:
            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    if hasattr(prov, "page_no"):
                        return prov.page_no
        return None

    def _build_section_path(self, headings: list[str]) -> str:
        return " > ".join(headings) if headings else ""

    def _apply_overlap(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        """Add token overlap between consecutive chunks from the same section."""
        if not chunks or self.settings.chunk_overlap_tokens <= 0:
            return chunks

        overlap = self.settings.chunk_overlap_tokens
        result: list[ChunkRecord] = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
                continue

            prev = chunks[i - 1]
            prev_section = prev.metadata.get("section_path", "")
            curr_section = chunk.metadata.get("section_path", "")

            if prev_section != curr_section:
                result.append(chunk)
                continue

            prev_tokens = self._tokenizer.encode(
                prev.text, add_special_tokens=False
            )
            overlap_tokens = prev_tokens[-overlap:] if len(prev_tokens) > overlap else prev_tokens
            overlap_text = self._tokenizer.decode(overlap_tokens, skip_special_tokens=True)

            merged_text = f"{overlap_text}\n\n{chunk.text}".strip()
            merged = ChunkRecord(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=merged_text,
                metadata={**chunk.metadata, "has_overlap": True},
            )
            result.append(merged)

        return result

    def chunk_document(
        self,
        doc: DoclingDocument,
        document_id: str,
        document_name: str,
        source_type: str,
    ) -> list[ChunkRecord]:
        raw_chunks = list(self._hybrid_chunker.chunk(doc))
        logger.info(
            "HybridChunker produced %d raw chunks for document %s",
            len(raw_chunks),
            document_id,
        )

        records: list[ChunkRecord] = []
        for idx, chunk in enumerate(raw_chunks):
            text = self._hybrid_chunker.contextualize(chunk)
            if not text or not text.strip():
                continue

            meta = chunk.meta
            headings = list(meta.headings) if hasattr(meta, "headings") and meta.headings else []  # pyright: ignore[reportAttributeAccessIssue]
            section_title = headings[-1] if headings else None
            section_path = self._build_section_path(headings)
            page_number = self._extract_page_number(meta)
            token_count = self.count_tokens(text)

            chunk_id = f"{document_id}_chunk_{idx:04d}_{uuid.uuid4().hex[:8]}"
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=text.strip(),
                    metadata={
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "document_name": document_name,
                        "source_type": source_type,
                        "chunk_index": idx,
                        "page_number": page_number,
                        "section_title": section_title,
                        "section_path": section_path,
                        "headings": headings,
                        "token_count": token_count,
                    },
                )
            )

        records = self._apply_overlap(records)
        logger.info(
            "Final chunk count after overlap: %d for document %s",
            len(records),
            document_id,
        )
        return records
