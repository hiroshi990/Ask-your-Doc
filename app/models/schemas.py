from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    document_id: str
    document_name: str
    source_type: str
    filename: Optional[str] = None
    chunk_count: int
    uploaded_at: datetime


class DocumentUploadResponse(BaseModel):
    document: DocumentInfo
    chunks_created: int
    message: str


class PastedTextRequest(BaseModel):
    title: Optional[str] = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    section_path: Optional[str] = None
    source_type: str
    excerpt: str
    relevance_score: Optional[float] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    use_cache: bool = True


class ChatResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    cache_hit: bool

class EvaluatedResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    faithfulness_score: float
    answer_relevancy: float
    retrieved_chunks: list[RetrievedChunk]
