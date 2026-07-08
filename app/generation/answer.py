import logging
import re
from typing import Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from app.config import Settings, get_settings
from app.models.schemas import Citation

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Custom knowledgeable assistant that answers questions based ONLY on the provided context.

Rules:
1. Answer using only information from the context chunks below.
2. If the context does not contain enough information, say so clearly.
3. Include inline citation markers like [1], [2] referencing the chunk numbers provided.
4. Be precise, concise, and factual.
5. Do not invent information not present in the context.
6. Do not guess
"""


class AnswerGenerator:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._client = wrap_openai(OpenAI(api_key=self.settings.openai_api_key))

    def _build_context(self, chunks: list[dict[str, Any]]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})
            header = (
                f"[{i}] Document: {meta.get('document_name', 'Unknown')} | "
                f"Section: {meta.get('section_path') or meta.get('section_title', 'N/A')} | "
                f"Page: {meta.get('page_number', 'N/A')} | "
                f"Source: {meta.get('source_type', 'unknown')}"
            )
            parts.append(f"{header}\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def generate(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> tuple[str, list[Citation]]:
        if not chunks:
            return (
                "I could not find relevant information in the knowledge base to answer your question.",
                [],
            )

        context = self._build_context(chunks)
        user_prompt = f"""Context chunks:
{context}

Question: {query}

Provide a grounded answer with inline citations [1], [2], etc. referencing the chunk numbers above."""

        response = self._client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1
        )
        answer = response.choices[0].message.content or ""

        citations = self._extract_citations(answer, chunks)
        return answer, citations

    def _extract_citations(
        self,
        answer: str,
        chunks: list[dict[str, Any]],
    ) -> list[Citation]:
        cited_indices = set(int(m) for m in re.findall(r"\[(\d+)\]", answer))
        citations: list[Citation] = []

        for idx in sorted(cited_indices):
            if idx < 1 or idx > len(chunks):
                continue
            chunk = chunks[idx - 1]
            meta = chunk.get("metadata", {})
            excerpt = chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else "")
            citations.append(
                Citation(
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk.get("document_id", meta.get("document_id", "")),
                    document_name=meta.get("document_name", "Unknown"),
                    page_number=meta.get("page_number"),
                    section_title=meta.get("section_title"),
                    section_path=meta.get("section_path"),
                    source_type=meta.get("source_type", "unknown"),
                    excerpt=excerpt,
                    relevance_score=chunk.get("rerank_score") or chunk.get("rrf_score"),
                )
            )

        if not citations and chunks:
            chunk = chunks[0]
            meta = chunk.get("metadata", {})
            citations.append(
                Citation(
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk.get("document_id", ""),
                    document_name=meta.get("document_name", "Unknown"),
                    page_number=meta.get("page_number"),
                    section_title=meta.get("section_title"),
                    section_path=meta.get("section_path"),
                    source_type=meta.get("source_type", "unknown"),
                    excerpt=chunk["text"][:300],
                    relevance_score=chunk.get("rerank_score"),
                )
            )

        return citations
