# 🏛️ HR Policy Assistant — Enterprise RAG System

A production-grade **Retrieval-Augmented Generation** system that answers employee HR policy questions strictly from company documents — zero hallucinations, full source citations.

---

## 🏗️ Architecture

```
Documents (PDF/DOCX/TXT)
        │
        ▼
┌─────────────────┐
│  ingestion.py   │  Unstructured → Chunking → SentenceTransformers → FAISS + BM25
└─────────────────┘
        │  (offline, one-time)
        ▼
  [FAISS Index] + [BM25 Index]
        │
        │  On every query:
        ▼
┌─────────────────┐
│  retrieval.py   │  Dense (FAISS) + Sparse (BM25) → RRF Fusion
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  reranker.py    │  Cross-Encoder (ms-marco-MiniLM-L-6-v2)
└─────────────────┘
        │
        ▼
┌──────────────────────┐
│  rag_pipeline.py     │  Scope check → Confidence gate → LLaMA 3 (Groq)
└──────────────────────┘
        │
        ▼
┌─────────────────┐
│    api.py       │  FastAPI REST API
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  frontend/      │  Vanilla HTML/CSS/JS (zero dependencies)
└─────────────────┘
```

---

## 📁 File Structure

```
hr_rag_system/
├── config.py              # Centralised settings (env-driven)
├── ingestion.py           # Document parsing, chunking, indexing
├── retrieval.py           # Hybrid retrieval (FAISS + BM25 + RRF)
├── reranker.py            # Cross-encoder re-ranking
├── rag_pipeline.py        # End-to-end pipeline + CLI
├── prompt_templates.py    # Engineered prompts & context formatter
├── api.py                 # FastAPI REST API
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── setup.sh               # One-shot setup script
├── documents/             # ← Place your HR policy files here
├── vector_store/          # Auto-created: FAISS index
├── bm25_index.pkl         # Auto-created: BM25 index
└── frontend/
    └── index.html         # Self-contained frontend (no build step)
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.10+
- [Groq API key](https://console.groq.com) (free — provides LLaMA 3)

### 2. Install
```bash
git clone <repo>
cd hr_rag_system
chmod +x setup.sh && ./setup.sh
```

Or manually:
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — add GROQ_API_KEY
```

### 3. Add Documents
Place your HR policy files in `./documents/`:
```
documents/
├── Anti-fraud-and-anti-corruption-policy.pdf
├── Code-of-ethics-policy.pdf
├── Company-cyber-security-policy.docx
├── Employee-Attendance-Policy.pdf
├── Employee-Benefits-and-Perks.pdf
├── Employee-bonus-policy.pdf
├── Employee-Code-of-Conduct.pdf
├── Employee-leave-of-absence-policy.docx
└── Employee-Resignation-and-Termination.pdf
```

Supported formats: `.pdf`, `.docx`, `.doc`, `.txt`, `.md`

### 4. Ingest Documents
```bash
python ingestion.py
# Force rebuild: python ingestion.py --force
```

This will:
- Parse all documents (preserving page/section metadata)
- Chunk into 800-token segments with 150-token overlap
- Generate embeddings with `all-MiniLM-L6-v2`
- Build FAISS index (cosine similarity)
- Build BM25 index (term frequency)

### 5. Start the API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs

### 6. Open the Frontend
Open `frontend/index.html` in your browser directly, or serve it:
```bash
python -m http.server 5173 --directory frontend
# Then visit http://localhost:5173
```

### 7. CLI Mode (no frontend)
```bash
python rag_pipeline.py
```

---

## 🔌 API Reference

### `POST /api/query`
```json
{
  "question": "How many sick leave days am I entitled to per year?"
}
```

Response:
```json
{
  "answer": "According to the Employee Leave of Absence Policy...",
  "sources": [
    {
      "document": "Employee-leave-of-absence-policy.pdf",
      "section": "Sick Leave Entitlement",
      "page": 3,
      "relevance_score": 0.8734,
      "excerpt": "Employees are entitled to 10 days of paid sick leave..."
    }
  ],
  "confidence": 0.8734,
  "query": "How many sick leave days am I entitled to per year?",
  "latency_ms": 1247.5,
  "fallback_triggered": false
}
```

### `GET /api/health`
Returns pipeline status, index stats, model info.

### `GET /api/documents`
Lists all policy documents in the documents directory.

### `POST /api/ingest?force=true`
Triggers re-ingestion in the background.

---

## ⚙️ Configuration

All settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | **Required.** Get at console.groq.com |
| `LLM_MODEL` | `llama3-70b-8192` | Groq model (70b=quality, 8b=speed) |
| `LLM_TEMPERATURE` | `0.1` | Low = deterministic, factual |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `INITIAL_RETRIEVAL_K` | `20` | Candidates per retriever |
| `RERANK_TOP_K` | `5` | Final chunks for LLM context |
| `CONFIDENCE_THRESHOLD` | `0.3` | Below → "I don't know" |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-ranking model |

---

## 🛡️ Guardrails

| Layer | Mechanism |
|-------|-----------|
| **Scope check** | Keyword + question-type heuristic filters clearly off-topic queries |
| **Confidence gate** | Cross-encoder score < 0.3 → explicit "I don't know" |
| **System prompt** | Hard instructions: cite sources, no general knowledge, no assumptions |
| **Low temperature** | 0.1 → minimal creative deviation from retrieved facts |
| **No hallucination** | If evidence isn't in top-k chunks, model is instructed to say so |

---

## 🔄 Retrieval Strategy

```
Query
  ├── Dense: FAISS (cosine similarity on L2-normalised embeddings)
  │          → Top 20 candidates + scores
  └── Sparse: BM25 (Okapi BM25 term-frequency)
              → Top 20 candidates + scores

RRF Fusion: score(d) = Σ 1/(60 + rank(d, list_i))
  → Merged ranked list

Cross-Encoder: jointly re-scores each (query, passage) pair
  → Top 5 final chunks with sigmoid-normalised confidence scores
```

---

## 🚀 Production Deployment

```bash
# Production API (no reload, multiple workers)
gunicorn api:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or with uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2
```

For Docker, NGINX, or cloud deployment — the API is a standard ASGI app.

---

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| Document parsing | `unstructured` |
| Chunking | `LangChain RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Dense index | `FAISS (IndexFlatIP)` |
| Sparse index | `BM25Okapi (rank-bm25)` |
| Rank fusion | `Reciprocal Rank Fusion (RRF)` |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `LLaMA 3 70B via Groq API` |
| Framework | `LangChain LCEL` |
| API | `FastAPI + Uvicorn` |
| Frontend | `Vanilla HTML/CSS/JS` |
