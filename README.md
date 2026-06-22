# Ask your Doc !

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
<img width="1853" height="577" alt="Screenshot 2026-06-22 214942" src="https://github.com/user-attachments/assets/61ec592b-3bbe-4d09-a4db-a915e5b6e1e0" />

<img width="1383" height="96" alt="Screenshot 2026-06-22 215104" src="https://github.com/user-attachments/assets/6cd31973-d523-4d91-aa11-cde60a980fa8" />

<img width="1283" height="433" alt="Screenshot 2026-06-22 220749" src="https://github.com/user-attachments/assets/32791fa3-2a4a-4b45-8636-5d2063e6c75d" />

<img width="1458" height="340" alt="Screenshot 2026-06-22 222313" src="https://github.com/user-attachments/assets/5745366f-8de0-46e4-8c5b-279c8149bf52" />





API docs: [http://localhost:8000/docs](http://localhost:8000/docs)


```

## Requirements

- Python 3.10+
- Docker (for Qdrant + Redis)
- OpenAI API key (answer generation and LLM as a judge)
- Cohere API key (reranking; optional — falls back to RRF order)
