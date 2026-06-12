I want to simplify and refactor the current RAG project architecture.

The project is being built primarily as a portfolio and resume project to demonstrate strong RAG engineering skills, retrieval knowledge, vector databases, chunking strategies, reranking, caching, and LLM integration.

The current implementation has started to accumulate production-oriented concepts that add significant complexity without adding much value for a recruiter evaluating the project.

My goal is:

* Keep the project technically impressive.
* Keep modern RAG best practices.
* Demonstrate production awareness.
* Remove unnecessary architectural complexity.
* Make the codebase easier to understand and maintain.
* Focus on RAG concepts rather than SaaS multi-tenancy concepts.

Please review the existing architecture and refactor it according to the following principles.

---

# Core Principle

This project should demonstrate:

* Document ingestion
* Document parsing
* Structure-aware chunking
* Embedding generation
* Vector database storage
* Hybrid retrieval
* RRF fusion
* Reranking
* Redis caching
* GPT generation
* Citation grounding

It should NOT try to become a full enterprise multi-tenant SaaS platform.

---

# Remove Unnecessary Complexity

Remove or simplify components that primarily exist for multi-tenancy, large-scale production management, or SaaS concerns.

Examples include:

* Workspace abstraction
* Workspace IDs
* Workspace filtering
* Knowledge base version tracking
* Version hashes
* Cache invalidation based on KB versions
* File-based knowledge base metadata stores
* Multiple collection management per knowledge base
* Excessive ID generation layers
* Complex hierarchy management that is not required for retrieval quality

If a component exists mainly to support multiple organizations, multiple tenants, enterprise isolation, or advanced operational concerns, remove or simplify it.

---

# Simplified Data Model

The system should assume:

One RAG application.

Users upload documents.

All uploaded documents belong to a single searchable corpus.

Optionally documents can retain:

* document_id
* document_name
* source_type
* upload_timestamp

for filtering and citations.

No workspace layer is required.

No knowledge-base-per-user architecture is required.

No collection-per-knowledge-base architecture is required.

---

# Simplified Vector Database Design

Use a single Qdrant collection.

Example:

documents

Store:

* embeddings
* chunk text
* chunk metadata

Payload metadata can include:

* document_id
* document_name
* page_number
* section_title
* section_path
* source_type

Do not create a separate Qdrant collection per knowledge base.

Use metadata filtering when needed.

Keep retrieval simple and easy to understand.

---

# Simplified Metadata Layer

Keep only metadata that directly improves:

* retrieval quality
* reranking quality
* citations
* debugging

Remove metadata that only exists for SaaS lifecycle management.

Examples of useful metadata:

* document id
* document name
* chunk id
* chunk index
* page number
* section title
* section path
* token count

Examples of metadata that can be removed:

* workspace id
* knowledge base id
* version hashes
* cache versioning
* tenant identifiers

---

# Keep These Features

These features are important and should remain.

## Ingestion

* PDF
* DOCX
* TXT
* Markdown
* Pasted text

Use Docling wherever appropriate.

---

## Structure-Aware Chunking

Keep:

* document hierarchy awareness
* section-aware chunking
* heading preservation
* page tracking
* chunk overlap

This is a major differentiator of the project.

---

## Embeddings

Use:

BAAI/bge-m3

Keep embedding generation modular.

---

## Vector Store

Keep:

Qdrant

Store embeddings and chunk metadata.

---

## Sparse Retrieval

Keep:

BM25

Sparse retrieval should remain a first-class retrieval method.

---

## Dense Retrieval

Keep:

Qdrant vector search.

---

## Hybrid Retrieval

Keep:

Dense Retrieval
+
BM25 Retrieval
+
RRF Fusion

This should remain a core project feature.

---

## Reranking

Keep reranking.

Use the existing cohere encoder for reranking..

This is an important advanced RAG feature.

---

## Redis

Keep Redis.

But simplify.

Implement:

Exact Query Cache

Example:

query -> answer

Do not implement complex cache invalidation systems unless truly required.

The goal is to demonstrate caching awareness, not build a distributed cache architecture.

---

## GPT Generation

Keep:

OpenAI GPT

Generation should be grounded using retrieved context.

---

## Citations

Keep citations.

This is one of the most important features.

Every answer should reference:

* document name
* page number
* section title

when available.

Before making changes, identify which files/classes/modules can be removed, merged, or simplified.

Then perform the refactor and explain the architectural changes made and why they improve the project.
