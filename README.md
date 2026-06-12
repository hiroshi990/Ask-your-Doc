# Enterprise RAG Knowledge Assistant

Portfolio-grade Retrieval-Augmented Generation (RAG) system: upload documents, build a searchable corpus, and chat with citation-grounded answers.

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
Dense Retrieval (Qdrant + BAAI/bge-m3)
    +
Sparse Retrieval (BM25)
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

API docs: http://localhost:8000/docs

## API Usage

### Upload a document

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@document.pdf" \
  -F "document_name=My Report"
```

### Paste text

```bash
curl -X POST "http://localhost:8000/api/v1/documents/paste" \
  -H "Content-Type: application/json" \
  -d '{"title": "Meeting Notes", "content": "Your pasted text here..."}'
```

### List documents

```bash
curl "http://localhost:8000/api/v1/documents"
```

### Delete a document

```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/<DOCUMENT_ID>"
```

### Chat with citations

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key findings?"
  }'
```

### Metadata-filtered retrieval

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize section 2",
    "metadata_filters": {"source_type": "pdf"}
  }'
```

## Project Structure

```
app/
├── api/routes/
│   ├── documents.py      # upload, paste, list, delete
│   └── chat.py           # chat endpoint
├── cache/
│   └── redis_cache.py    # exact query cache
├── config.py             # settings from .env
├── generation/
│   └── answer.py         # GPT answer + citation extraction
├── ingestion/
│   ├── parser.py         # Docling document parsing
│   ├── chunker.py        # structure-aware chunking
│   ├── embedder.py       # BGE-M3 embeddings
│   └── pipeline.py       # ingest orchestration
├── models/
│   └── schemas.py        # Pydantic request/response models
├── retrieval/
│   ├── hybrid.py         # dense + sparse + RRF + rerank
│   ├── rrf.py            # reciprocal rank fusion
│   └── reranker.py       # Cohere cross-encoder
├── services/
│   └── rag_service.py    # chat orchestration
└── storage/
    ├── qdrant_store.py   # single Qdrant collection
    ├── bm25_index.py     # single BM25 corpus index
    └── document_store.py # lightweight document manifest
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_COLLECTION_NAME` | documents | Single Qdrant collection name |
| `CHUNK_MAX_TOKENS` | 700 | Max tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | 75 | Overlap between chunks |
| `DENSE_TOP_K` | 20 | Dense retrieval candidates |
| `SPARSE_TOP_K` | 20 | BM25 retrieval candidates |
| `RRF_K` | 60 | RRF constant |
| `RERANK_TOP_K` | 8 | Final chunks after reranking |

## Requirements

- Python 3.10+
- Docker (for Qdrant + Redis)
- OpenAI API key (answer generation)
- Cohere API key (reranking; optional — falls back to RRF order)

## Migration Note

If upgrading from the previous multi-KB architecture, delete the `data/` folder and any old Qdrant collections. The system will recreate a single `documents` collection on first ingest.
