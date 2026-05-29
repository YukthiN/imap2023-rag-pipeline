# IMAP-2023 Military Airworthiness RAG Pipeline

> A production-grade Retrieval-Augmented Generation pipeline built on the Indian Military Airworthiness Procedure 2023 (IMAP-2023) document. Implements semantic chunking, hybrid retrieval, graph-RAG, RAPTOR hierarchical indexing, and grounded answer generation with evaluation metrics.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Components](#pipeline-components)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Evaluation Results](#evaluation-results)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)

---

## Overview

This pipeline answers natural language queries about Indian Military Airworthiness procedures by combining vector search, keyword search, and graph traversal — then generating grounded, cited answers using a language model.

The system was built to address the limitations of a standard single-vector RAG pipeline:
- Incoherent chunks from character-limit splitting
- Sparse retrieval using IDF-only instead of true BM25
- No graph-based context expansion
- Single-level summarisation without hierarchy
- Hallucination-prone generation without strict grounding

All of these are addressed in this implementation.

---

## Architecture

```text
PDF Document (IMAP-2023)
        │
        ▼
┌─────────────────────┐
│   Cell 2: Extraction │  PyMuPDF + pdfplumber
│   text, tables,      │  metadata: page, section,
│   image placeholders │  chunk_id, doc_name
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Cell 3: PII Scan  │  Regex — email, phone,
│                     │  Aadhaar, PAN, IP
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Cell 5: Semantic  │  Sentence embedding +
│   Chunking          │  cosine similarity slope
│                     │  topic-shift detection
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Cell 6: Embeddings│  BGE-small dense vectors
│                     │  True BM25 TF×IDF sparse
└─────────┬───────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
┌────────┐    ┌──────────┐
│Cell 7  │  │ Cell 9     │
│Qdrant  │  │ Neo4j      │
│dense + │  │ graph-RAG  │
│sparse  │  │ chunknodes │
└───┬────┘    └────┬─────┘
    │             │
    ▼             │
┌─────────────────────┐
│   Cell 8: RAPTOR    │  Two-level hierarchical
│                     │  summarisation
│   L1: 5 summaries   │  Soft GMM clustering
│   L2: 3 meta-summ.  │  Phi-3.5-mini summaries
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Cell 10: Hybrid   │  Dense + Sparse + Graph
│   Retrieval         │  → RRF → MMR → Reranking
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Cell 11: HyDE +  │  Hypothetical document
│   Generation       │  embedding + grounded
│                    │  answer with citations
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Cell 12: Eval    │  Faithfulness, Relevancy,
│                    │  Context Precision,
│                    │  Groundedness
└─────────────────────┘
```

---

## Pipeline Components

### Cell 1 — Environment Setup
- Pins `transformers==4.44.0` for Phi-3.5-mini compatibility
- Installs all dependencies
- Defines all constants and checkpoint paths

### Cell 2 — PDF Extraction
- **PyMuPDF** extracts text blocks with page numbers
- **pdfplumber** extracts structured tables
- Section assignment via PDF TOC or heading regex fallback
- Every document carries: `chunk_id`, `page`, `section`, `content_type`, `doc_name`

### Cell 3 — PII Scan
- Regex patterns for email, Indian phone numbers, Aadhaar, PAN, IP addresses
- Document came back clean — no PII detected

### Cell 4 — Model Loading

| Model | Purpose | Size |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | Dense embeddings | ~130MB |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking | ~90MB |
| `cross-encoder/nli-deberta-v3-small` | NLI evaluation | ~570MB |
| `microsoft/Phi-3.5-mini-instruct` | Generation + summaries | ~3.8GB float16 |

### Cell 5 — Semantic Chunking
- Short documents merged before chunking to prevent tiny fragments
- Each document split into sentences using `nltk.sent_tokenize`
- Sentences embedded with BGE-small
- Cosine similarity computed between consecutive sentences
- Cut where similarity drops below threshold (0.4) — topic shift
- Guardrails: min 200 chars, max 800 chars
- Tables and images kept as single chunks

### Cell 6 — Embeddings
- **Dense**: BGE-small-en-v1.5, 384-dim normalized vectors
- **Sparse**: True BM25Okapi — full `TF × IDF` per token per chunk

### Cell 7 — Qdrant Persistent Indexing
- Local persistent storage — survives session within Kaggle
- Named vectors: `"dense"` (cosine) + `"sparse"` (dot product)

### Cell 8 — RAPTOR Hierarchical Indexing
- PCA + soft GMM clustering
- Phi-3.5-mini generates summaries
- Summary nodes added into Qdrant alongside leaf chunks

### Cell 9 — Neo4j Graph-RAG
- Chunk-node graph retrieval with typed semantic edges
- Graph traversal expands retrieval context dynamically

### Cell 10 — Hybrid Retrieval
- Dense retrieval
- Sparse BM25 retrieval
- Graph traversal retrieval
- Reciprocal Rank Fusion (RRF)
- MMR diversity selection
- Cross-encoder reranking

### Cell 11 — HyDE + Generation
- Hypothetical document generation
- Grounded answer generation with citations
- Strict anti-hallucination prompting

### Cell 12 — Evaluation
Metrics:
- Faithfulness
- Answer Relevancy
- Context Precision
- Groundedness

---

## Tech Stack

| Category | Technology |
|---|---|
| Environment | Kaggle Notebooks — T4 GPU 15.6GB |
| PDF Extraction | PyMuPDF, pdfplumber |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Sparse Retrieval | rank-bm25 (BM25Okapi) |
| Vector Store | Qdrant |
| Graph Database | Neo4j Aura |
| Generation | Phi-3.5-mini-instruct |
| Framework | Python, PyTorch, Transformers |

---

## Known Limitations

- No multimodal image understanding yet
- Faithfulness metrics sensitive to wording
- No embedded PDF TOC in source document
- Image placeholders not semantically indexed

---

## Future Work

- Add multimodal models
- Improve RAPTOR hierarchy depth
- Deploy scalable Qdrant server
- Improve evaluation via RAGAS
- Add temporal graph edges

---

## Project Structure

```text
rag_project/
├── extracted_docs.json
├── chunks.json
├── embeddings.npy
├── embedding_store.json
├── raptor_summaries.json
├── evaluation_results_v2.json
└── qdrant_storage/
```

---

*Built on Kaggle — Tesla T4 GPU | IMAP-2023 Indian Military Airworthiness Procedure*
