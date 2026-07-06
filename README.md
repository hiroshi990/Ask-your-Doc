# Ask your Doc !

Custom Knowledge Base Retrieval-Augmented Generation (RAG) system: upload documents, build a searchable corpus, and chat with citation-grounded answers.

## Features

- **Multi-format ingestion** — PDF, DOCX, PPTX, XLSX, HTML, Markdown, TXT, images (via Docling)
- **Pasted text** — Treated exactly like uploaded documents
- **Structure-aware chunking** — Docling `HybridChunker` respects headers, sections, paragraphs (~500–700 tokens, 75-token overlap)
- **Rich chunk metadata** — `chunk_id`, `document_id`, `document_name`, `page_number`, `section_title`, `section_path`, `source_type`
- **Hybrid retrieval** — Dense (Qdrant + BGE-M3) + Sparse (BM25) → RRF fusion → deduplication → Cohere reranking
- **Redis caching** — Exact query cache, flushed on document changes
- **Citation-grounded answers** — GPT with inline `[1]`, `[2]` citations

## Architecture

Single-corpus design: all documents share one Qdrant collection and one BM25 index.

```
User Query
    ↓
Redis Exact Cache Check
    ↓
Dense Retrieval(Qdrant + BAAI/bge-m3) + Sparse Retrieval(BM25) [Asynchronous]
    ↓
RRF Fusion → Deduplication → Cohere Rerank
    ↓
GPT Answer Generation with Citations
```

## Quick Start

### 1. Start infrastructure

```bash
docker compose up -d
```

This starts **Qdrant** (port 6333) and **Redis Stack** (port 6379).

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and COHERE_API_KEY
```

### 3. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 4. Run the API

```bash
python run.py
```
<img width="1920" height="933" alt="Screenshot 2026-07-06 193150" src="https://github.com/user-attachments/assets/62df3f94-531b-45e0-b0e8-6fa4ad513c67" />






API docs: [http://localhost:8000/docs](http://localhost:8000/docs)


```

## Requirements

- Python 3.10+
- Docker (for Qdrant + Redis)
- OpenAI API key (answer generation and LLM as a judge)
- Cohere API key (reranking; optional — falls back to RRF order)
