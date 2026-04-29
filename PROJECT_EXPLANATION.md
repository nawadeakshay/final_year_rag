# 🏛️ HR Policy Assistant — Complete Codebase Explanation

## Executive Summary

This is an **Enterprise-Grade Retrieval-Augmented Generation (RAG) System** designed to answer employee HR policy questions accurately and reliably. It combines multiple retrieval strategies, intelligent ranking, and a large language model to provide **grounded answers with full source citations** — eliminating hallucinations.

**Key Goal:** Answer "What is our company's maternity leave policy?" by retrieving relevant policy documents, ranking them by relevance, and generating an answer strictly from those documents.

---

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Technology Stack](#technology-stack)
3. [Detailed File-by-File Breakdown](#detailed-file-by-file-breakdown)
4. [Data Flow & Business Logic](#data-flow--business-logic)
5. [Key Design Decisions](#key-design-decisions)
6. [Interview Talking Points](#interview-talking-points)

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Employee HR Question                        │
│            "How many days of leave am I entitled to?"        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │  Scope Check (Optional) │  ← Filters off-topic queries
         └────────────┬────────────┘
                      │
                      ▼
      ┌──────────────────────────────────┐
      │  Hybrid Retrieval (Dense + BM25) │
      │ • FAISS: Dense semantic search   │
      │ • BM25: Sparse keyword search    │
      │ • Fusion: Reciprocal Rank (RRF)  │
      └──────────┬───────────────────────┘
                 │
         Returns ~20 candidates
                 │
                 ▼
    ┌────────────────────────────┐
    │  Cross-Encoder Re-ranking  │  ← Semantic relevance scoring
    │  (ms-marco-MiniLM-L-6-v2)  │
    └──────────┬─────────────────┘
               │
          Top 5 chunks
               │
               ▼
  ┌──────────────────────────────┐
  │  Confidence Threshold Check  │  ← Block if relevance too low
  └──────────┬───────────────────┘
             │
             ▼
    ┌────────────────────┐
    │  Format Context    │  ← Structured prompt with citations
    └──────────┬─────────┘
               │
               ▼
┌──────────────────────────────────┐
│  LLaMA 3 (via Groq API)         │  ← Fast, free inference
│  Generation with constraints     │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  Final Answer + Source Citations        │
│  "Based on the Leave Policy document,   │
│   you are entitled to 20 days annually. │
│  (Source: Leave-Policy.pdf, Page 3)"    │
└──────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Document Parsing** | `unstructured` + `pdfminer` + `python-docx` | Extract text from PDF/DOCX/TXT files |
| **Chunking** | `langchain.text_splitters` | Split documents into meaningful overlapping chunks |
| **Dense Embedding** | `sentence-transformers` (all-MiniLM-L6-v2) | Create 384-dim vector embeddings for semantic search |
| **Vector Store** | `FAISS` (IndexFlatIP) | Fast similarity search using inner-product (cosine) |
| **Sparse Retrieval** | `rank-bm25` | Traditional keyword-based BM25 scoring |
| **Re-ranking** | `sentence-transformers.CrossEncoder` (ms-marco) | Score (query, passage) pairs jointly |
| **LLM** | `Groq API` + `LLaMA 3.1-8B` | Fast, free inference with proper grounding |
| **API Framework** | `FastAPI` | REST API with auto-generated Swagger docs |
| **Frontend** | Vanilla HTML/CSS/JS | Zero-dependency, self-contained chat UI |
| **Logging** | `loguru` | Structured logging for debugging |

---

## Detailed File-by-File Breakdown

### 1. **config.py** — Centralized Configuration

**Purpose:** Single source of truth for all settings. Environment variables with sensible defaults.

**Key Functions:**

```python
class Settings:
    """Configuration singleton accessed globally"""
```

**What it manages:**
- **LLM Settings:** Groq API key, model (LLaMA 3.1-8B), temperature, max tokens
- **Embeddings:** Which model to use (all-MiniLM-L6-v2)
- **Chunking:** Chunk size (800 chars), overlap (150 chars)
- **Retrieval Tuning:** Hybrid weights (0.6 dense, 0.4 sparse), K values
- **Re-ranker:** Model selection (ms-marco-MiniLM-L-6-v2)
- **Storage Paths:** Where to store FAISS and BM25 indices
- **API:** Host/port, CORS origins

**Business Logic:** 
- Validates API key at startup (raises error if missing)
- Checks if indices already exist before re-indexing
- All settings are env-driven for easy deployment

**Key Method:**
```python
def validate(self) -> None:
    """Raises EnvironmentError if GROQ_API_KEY missing"""
    
def indices_exist(self) -> bool:
    """Returns True if FAISS + BM25 indices are built"""
```

---

### 2. **ingestion.py** — Document Processing Pipeline

**Purpose:** One-time offline process to convert raw documents into searchable indices.

**Flow:** Parse → Chunk → Embed → FAISS Index → BM25 Index

#### **Step 1: Document Parsing**

```python
def _parse_with_unstructured(file_path: Path) -> List[Document]:
    """
    Uses 'unstructured' library for smart document parsing.
    - Preserves structure (titles → sections)
    - Extracts metadata (page numbers, element types)
    - Cleans whitespace
    """
```

**What it does:**
- Attempts to parse PDF/DOCX/TXT with `unstructured` library
- Tracks document structure (sections, page numbers)
- Falls back to `pdfminer` (PDF) or `python-docx` (DOCX) if needed
- Returns LangChain `Document` objects with rich metadata

**Business Logic:**
- Preserves context: which policy document, which section, which page
- Skips very small fragments (< 30 chars)
- Creates a doc_id (hash of filename) for deduplication later

#### **Step 2: Chunking**

```python
def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits parsed documents into overlapping chunks for embedding.
    """
```

**Configuration:**
- Chunk size: 800 characters (fits LLM context)
- Overlap: 150 characters (preserves context across chunk boundaries)
- Separators: `["\n\n", "\n", ". ", " "]` (respect document structure)

**Why overlapping chunks?**
- A question's answer might span two chunks
- Overlap ensures no information is lost at boundaries

**Business Logic:**
- Adds chunk_id and preview to metadata
- Recursive splitter respects natural document structure

#### **Step 3: Embedding**

```python
def embed_chunks(chunks: List[Document], model: SentenceTransformer) -> np.ndarray:
    """
    Converts chunk text → 384-dimensional vectors (all-MiniLM-L6-v2).
    Uses L2 normalization for cosine similarity.
    """
```

**What's happening:**
- Takes ~100 chars text → 384-dim dense vector
- All embeddings are **L2-normalized** (all vectors have length 1)
- Normalized embeddings enable **cosine similarity via inner product**

**Why this model?**
- all-MiniLM-L6-v2: Fast, small (33M params), trained on semantic similarity
- Perfect balance for HR questions (not as large as BGE-large, but better than older models)

#### **Step 4: FAISS Index**

```python
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds FAISS IndexFlatIP (exact inner-product search).
    """
```

**What FAISS does:**
- Takes 384-dim vectors → builds an index for fast similarity search
- IndexFlatIP: "Flat" means no quantization, "IP" means inner-product
- **O(1) search per query** after O(n) index build

**Example:**
```
Query: "How much leave?" 
→ Embed to [0.1, 0.2, ..., -0.3] (384 dims)
→ FAISS searches: max(query · chunk_i) for all chunks
→ Returns top-k most similar chunks in milliseconds
```

#### **Step 5: BM25 Index**

```python
def build_bm25_index(chunks: List[Document]) -> BM25Okapi:
    """
    Builds BM25 index for sparse (keyword) retrieval.
    """
```

**What BM25 does:**
- Traditional IR scoring: rewards document frequency, penalizes term frequency
- Great for exact keyword matches (e.g., "annual leave" vs "leaves annually")
- Complements dense search (different signal)

**Example:**
```
Query: "annual leave days"
→ Tokenize: ["annual", "leave", "days"]
→ BM25 scores each chunk based on term overlap
→ Rewards chunks with all three terms
```

#### **Orchestrator: run_ingestion()**

```python
def run_ingestion(force_rebuild: bool = False) -> None:
    """
    Runs full pipeline end-to-end: parse → chunk → embed → index.
    If force_rebuild=False, skips if indices already exist.
    """
```

**Business Logic:**
1. Check if FAISS and BM25 indices exist
2. If yes and force_rebuild=False → skip (indices cached)
3. If no → run full pipeline (slow first time, then cached)
4. Save indices to disk for next startup

**Why cache?**
- Ingestion takes minutes (document parsing, embedding all chunks)
- Query is subsecond (loads cached indices from disk)

---

### 3. **retrieval.py** — Hybrid Retrieval & Fusion

**Purpose:** Given a query, retrieve the most relevant document chunks using both dense and sparse search, then fuse results.

#### **Class: HybridRetriever**

```python
class HybridRetriever:
    """
    Holds FAISS index, BM25 index, embedding model.
    Instantiated once at API startup; reused for all queries.
    """
```

#### **Dense Retrieval (_dense_search)**

```python
def _dense_search(self, query: str, k: int) -> List[Tuple[int, float]]:
    """
    1. Embed query using same model as chunks
    2. Search FAISS for top-k most similar chunks
    3. Return (chunk_index, similarity_score) pairs
    """
```

**Example Output:**
```
Query: "How much leave am I entitled to?"
Embedding: [0.15, -0.2, 0.3, ...]  (384 dims)

Dense Results:
  (chunk_idx=42, score=0.87)  ← Leave Policy chunk
  (chunk_idx=15, score=0.82)  ← Benefits Overview chunk
  (chunk_idx=200, score=0.71) ← HR Contact chunk
  ...
```

#### **Sparse Retrieval (_bm25_search)**

```python
def _bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
    """
    1. Tokenize query (simple whitespace split)
    2. BM25 scores all chunks
    3. Return top-k (chunk_index, bm25_score) pairs
    """
```

**Example Output:**
```
Query: "How much leave?"
Tokens: ["how", "much", "leave"]

BM25 Results:
  (chunk_idx=42, score=45.2)   ← High score (all terms present)
  (chunk_idx=198, score=12.1)  ← Medium score (2/3 terms)
  (chunk_idx=300, score=8.9)   ← Low score (1/3 terms)
  ...
```

#### **Reciprocal Rank Fusion (RRF)**

```python
@staticmethod
def _reciprocal_rank_fusion(ranked_lists: List[List[Tuple[int, float]]]) -> List[Tuple[int, float]]:
    """
    Combines dense and sparse rankings using RRF.
    
    Formula: RRF_score(doc) = Σ 1/(k + rank(doc, list_i))
    """
```

**Why RRF?**
- Dense scores (0-1) and BM25 scores (0-100+) have different scales
- RRF uses **ranks**, not raw scores → scale-invariant
- Document appearing in top-5 of both lists gets boosted

**Example:**
```
Dense results:           BM25 results:
1. Chunk 42 (0.87)      1. Chunk 42 (45.2)
2. Chunk 15 (0.82)      2. Chunk 198 (12.1)
3. Chunk 200 (0.71)     3. Chunk 50 (10.0)
4. Chunk 100 (0.65)     ...
...

RRF Fusion (k=60):
Chunk 42:  1/(60+1) + 1/(60+1) = 0.033 ✓ (appears in both!)
Chunk 15:  1/(60+2) = 0.016
Chunk 198: 1/(60+2) = 0.016
Chunk 200: 1/(60+3) = 0.016
...

Final rank: Chunk 42 > {15, 198, 200, 50, ...}
```

#### **Public API: retrieve()**

```python
def retrieve(self, query: str, initial_k: int = None) -> List[Tuple[Document, float]]:
    """
    Returns top-k (Document, rrf_score) pairs.
    
    Flow:
    1. Dense search → 20 candidates
    2. BM25 search → 20 candidates  
    3. RRF fusion → re-rank by RRF score
    4. Return top results
    """
```

**Business Logic:**
- Returns ~20 candidates (NOT the final answer yet)
- Next step: re-rank with cross-encoder for better precision

---

### 4. **reranker.py** — Cross-Encoder Re-ranking

**Purpose:** Take ~20 retrieval candidates, score each (query, passage) pair, keep top-5 most relevant.

#### **Class: CrossEncoderReranker**

```python
class CrossEncoderReranker:
    """
    Wraps cross-encoder/ms-marco-MiniLM-L-6-v2.
    Cross-encoder: jointly scores (query, passage) pairs (better than bi-encoder).
    """
```

#### **Key Difference: Bi-Encoder vs Cross-Encoder**

| Aspect | Bi-Encoder (FAISS) | Cross-Encoder (Reranker) |
|--------|---|---|
| Input | Query alone, passage alone | (Query, passage) **together** |
| Score | Cosine sim: query · passage | Joint relevance score |
| Accuracy | Good (fast) | Excellent (slower) |
| Use Case | Initial retrieval (20 candidates) | Re-ranking (pick best 5) |

**Why not use cross-encoder for initial retrieval?**
- Cross-encoder requires comparing query against every chunk (slow)
- Bi-encoder embeds query once, then searches in index (fast)
- **Hybrid approach:** Use fast bi-encoder for broad retrieval, cross-encoder for precision

#### **rerank() Method**

```python
def rerank(self, query: str, candidates: List[Tuple[Document, float]], top_k: int = None) -> List[Tuple[Document, float]]:
    """
    1. Create (query, passage) pairs from candidates
    2. Cross-encoder scores each pair → raw logits
    3. Sigmoid normalize to [0, 1]
    4. Return top-k by score
    """
```

**Example:**
```
Input candidates (from HybridRetriever):
  - (chunk about leave policy, rrf_score=0.033)
  - (chunk about benefits, rrf_score=0.016)
  - (chunk about appraisal, rrf_score=0.010)

Cross-encoder scoring:
  Pair("How much leave?", "Leave Policy: 20 days annual") → raw_score=2.15
  Pair("How much leave?", "Benefits include health insurance") → raw_score=-0.8
  Pair("How much leave?", "Appraisal: performance metrics...") → raw_score=-1.5

Sigmoid normalize: sigmoid(2.15)=0.896, sigmoid(-0.8)=0.31, sigmoid(-1.5)=0.18

Output (top-1):
  - (chunk about leave policy, cross_encoder_score=0.896)
```

#### **best_confidence() Method**

```python
def best_confidence(self, reranked: List[Tuple[Document, float]]) -> float:
    """Returns highest confidence score for downstream threshold check."""
```

---

### 5. **rag_pipeline.py** — End-to-End Orchestration

**Purpose:** Tie everything together: query → scope check → retrieval → re-rank → generate answer → return with citations.

#### **is_likely_in_scope(query) — Scope Check**

```python
def is_likely_in_scope(query: str) -> bool:
    """
    Lightweight heuristic to filter off-topic queries.
    
    Passes if:
    - Contains HR keywords (leave, salary, ethics, etc.)
    - Starts with question word (what, how, who, etc.)
    - Contains a '?'
    
    Blocks if:
    - Query < 5 chars (too short)
    - No keywords AND not a question AND no '?'
    """
```

**Examples:**
```
✓ In scope:
  - "How many days of leave can I take?"
  - "What is the dress code?"
  - "Tell me about salary increment"
  - "can i work from home?"
  - "Is remote work allowed?"

✗ Out of scope:
  - "hello" (too short, no keyword)
  - "tell me a joke" (not HR-related)
  - "how to cook pasta" (off-topic)
```

**Business Logic:**
- **Why scope check?** Reduces API load from irrelevant queries
- **Why lenient?** Avoid false negatives (blocking valid HR questions)
- **Fallback:** If confidence is low anyway, model will refuse to answer

#### **Class: RAGResult**

```python
class RAGResult:
    """Structured output from pipeline.query()"""
    answer: str          # Generated answer
    sources: List[dict]  # Citation metadata
    confidence: float    # Cross-encoder best score
    query: str          # Original question
    latency_ms: float   # E2E latency
    fallback_triggered: bool  # Did we hit a fallback?
```

#### **Class: HRRagPipeline**

**Main orchestrator class:**

```python
class HRRagPipeline:
    """
    Singleton-friendly orchestrator.
    Instantiated once at API startup; thread-safe for concurrent requests.
    """
    
    def __init__(self):
        # Validate config
        # Load LLM (Groq)
        # Load retriever (FAISS + BM25)
        # Load reranker (cross-encoder)
        # Set _ready flag (or false with reason if errors)
```

**Thread-Safety Note:**
- SentenceTransformer embedding model: stateless at inference
- FAISS index: read-only after loading
- Groq client: stateless
- Safe for concurrent requests without locks

#### **query() — Main Query Method**

```python
def query(self, user_query: str) -> RAGResult:
    """
    7-step RAG pipeline:
    
    Step 1: Scope check
      - If not in scope → return FALLBACK_RESPONSE (don't waste compute)
    
    Step 2: Hybrid retrieval
      - Dense + BM25 + RRF fusion → ~20 candidates
      - If 0 candidates → return FALLBACK_RESPONSE
    
    Step 3: Cross-encoder re-ranking
      - Score top-20 candidates → return top-5
    
    Step 4: Confidence threshold check
      - If best_score < threshold → return FALLBACK_RESPONSE
      - Threshold default: 0.3 (configurable)
    
    Step 5: Format context
      - Build structured prompt with citations
    
    Step 6: Generate answer
      - Send (context + question) to LLaMA 3 via Groq
      - LLM instructed to cite sources
    
    Step 7: Build source list
      - Extract unique documents from reranked chunks
      - Return with document name, section, page, relevance score
    """
```

**Example flow:**
```
Query: "How many days of annual leave am I entitled to?"

Step 1: Scope check → ✓ "leave" keyword present
Step 2: Hybrid retrieval → ~20 candidates from policies
Step 3: Re-rank → Top 5 most relevant
Step 4: Confidence 0.89 > 0.3 → ✓ Proceed
Step 5: Format context:
  [Excerpt 1]
  Source: Leave-Policy.pdf
  Section: Annual Leave Entitlements
  Page: 3
  Relevance Score: 0.89
  
  All full-time employees are entitled to 20 days of paid annual leave...
  
  [Excerpt 2]
  Source: Leave-Policy.pdf
  Section: Leave Calculation
  Page: 4
  Relevance Score: 0.84
  
  Leave is calculated pro-rata for employees starting mid-year...

Step 6: LLM call:
  Prompt: "Answer from context only. Cite sources."
  Context: [formatted context from step 5]
  Question: "How many days of annual leave?"
  
  Response: "Based on the Leave Policy document, all full-time employees 
  are entitled to 20 days of paid annual leave per year.
  (Source: Leave-Policy.pdf, Section: Annual Leave Entitlements, Page: 3)"

Step 7: Return to user with:
  - Answer + sources + confidence (0.89) + latency (245ms)
```

#### **_build_sources() — Citation Builder**

```python
def _build_sources(reranked: List[Tuple[Document, float]]) -> List[dict]:
    """
    Deduplicates sources (same doc+section+page only counted once).
    Returns list of:
    {
        "document": "Leave-Policy.pdf",
        "section": "Annual Leave Entitlements",
        "page": 3,
        "relevance_score": 0.89,
        "excerpt": "All full-time employees are entitled to..."
    }
    """
```

**Business Logic:**
- Deduplication: avoid showing "source from page 3" twice
- Excerpt: first 200 chars for user reference
- Relevance score: helps user understand confidence

---

### 6. **prompt_templates.py** — Engineered Prompts

**Purpose:** Carefully crafted system and user prompts to enforce grounding and prevent hallucinations.

#### **HR_SYSTEM_PROMPT**

```python
HR_SYSTEM_PROMPT = """\
You are an HR Policy Assistant...

STRICT RULES:
1. Answer ONLY from provided [CONTEXT]. Do NOT use general knowledge.
2. Always cite policy document name and section.
3. If answer not in context, say: "I don't have enough information..."
4. Never assume or extrapolate beyond what's stated.
5. Never fabricate policy details, dates, percentages.
6. Keep answers concise, professional, easy to understand.
7. If multiple policies relevant, address each separately with citation.
"""
```

**Why so strict?**
- HR policies are **legally binding** — hallucinated leave policies are costly errors
- Employees act on answers — wrong information could lead to disputes
- Citations let employees verify and question if needed

#### **format_context(reranked_chunks) — Context Formatter**

```python
def format_context(reranked_chunks: list) -> str:
    """
    Converts reranked (Document, score) pairs into structured block.
    
    Output:
    [Excerpt 1]
    Source: Leave-Policy.pdf
    Section: Annual Leave
    Page: 3
    Relevance Score: 0.89
    
    All full-time employees are entitled to 20 days...
    
    ---
    
    [Excerpt 2]
    ...
    """
```

**Why structured?**
- LLM can parse sections clearly
- Makes it easy to cite (reference "Excerpt 1 → Leave-Policy.pdf, Section: Annual Leave")

#### **FALLBACK_RESPONSE**

```python
FALLBACK_RESPONSE = """\
I'm sorry, I don't have enough information in the provided policy documents 
to answer this question accurately. Please contact your HR department directly.
"""
```

**When triggered:**
- Scope check failed (off-topic)
- No candidates retrieved
- Confidence too low (< 0.3)

---

### 7. **api.py** — FastAPI REST Service

**Purpose:** Expose RAG pipeline as production-grade REST API with Swagger docs.

#### **Lifespan Context Manager**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Initialize pipeline (best-effort, no crash if indices missing)
    Shutdown: Cleanup (none needed currently)
    """
```

**Startup Logic:**
- Load pipeline singleton
- Check if pipeline._ready (indices exist?)
- If not ready, log warning but DON'T crash
- API still serves, but returns 503 on /api/query until indices built

#### **Routes**

##### **POST /api/query — Query the RAG System**

```python
@app.post("/api/query", response_model=QueryResponse)
async def query_policy(request: QueryRequest):
    """
    Input: { "question": "How many days of leave?" }
    Output: {
        "answer": "20 days annually...",
        "sources": [...],
        "confidence": 0.89,
        "latency_ms": 245,
        "fallback_triggered": false
    }
    """
```

**Error Handling:**
- 503 if pipeline not ready (indices missing)
- 500 if pipeline error
- 400 if input validation fails

##### **GET /api/health — System Health**

```python
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Returns:
    {
        "status": "ok" or "not_ready",
        "retriever": { "total_chunks": 1250, ... },
        "llm_model": "llama-3.1-8b-instant",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "confidence_threshold": 0.3
    }
    """
```

**Use Case:** Operators check if system is ready before going live

##### **POST /api/ingest — Trigger Re-indexing**

```python
@app.post("/api/ingest", response_model=IngestResponse)
async def trigger_ingestion(background_tasks: BackgroundTasks, force: bool = False):
    """
    Trigger document ingestion in background.
    
    Query parameters:
    - force=false (default): skip if indices exist
    - force=true: rebuild from scratch
    
    Returns immediately with "accepted" status.
    Poll /api/health to check when done.
    """
```

**Business Logic:**
- Background task doesn't block API
- After ingestion, calls `pipeline.reload_retrieval()` to hot-reload indices
- API can serve queries immediately after reload (no restart needed)

##### **GET /api/documents — List Ingested Documents**

```python
@app.get("/api/documents")
async def list_documents():
    """
    Lists all documents in ./documents/ directory.
    Returns:
    [
        { "filename": "Leave-Policy.pdf", "size_bytes": 45000, "extension": ".pdf" },
        ...
    ]
    """
```

#### **CORS Middleware**

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # From .env or defaults
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Why CORS?**
- Frontend (http://localhost:3000) calls API (http://localhost:8000)
- Without CORS headers, browser blocks cross-origin request

#### **Frontend Static Serving**

```python
if _frontend_dir is not None:
    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend(full_path: str = ""):
        """Serves index.html for all non-API routes (SPA routing)."""
```

**Pattern:** Single Page App (SPA) routing — all non-API paths → index.html

---

### 8. **frontend/index.html** — Chat UI

**Purpose:** User-facing interface for asking HR questions.

**Architecture:** Vanilla HTML/CSS/JS (zero build steps, zero dependencies)

#### **HTML Structure**

```html
<header>HR Policy Assistant</header>
<div id="chat"></div>  <!-- Messages appear here -->
<div class="input-box">
  <input id="question" placeholder="Ask your HR question..." />
  <button id="sendBtn" onclick="ask()">Send</button>
</div>
```

#### **JavaScript Logic**

```javascript
async function ask() {
    const question = document.getElementById("question").value;
    
    // Show user message
    addMessage(question, "user");
    
    // Call API
    const response = await fetch("http://127.0.0.1:8000/api/query", {
        method: "POST",
        body: JSON.stringify({ question: question })
    });
    
    const data = response.json();
    
    // Display bot response with sources
    let output = data.answer;
    output += `Confidence: ${data.confidence}`;
    output += `Sources: ${data.sources.map(s => s.document).join(", ")}`;
    
    addMessage(output, "bot");
}

// Enter key support
document.getElementById("question").addEventListener("keypress", function(e) {
    if (e.key === "Enter") ask();
});
```

**Styling:** Dark theme, responsive chat layout

---

## Data Flow & Business Logic

### **Complete End-to-End Example**

**Query:** "Can I work from home 3 days a week?"

#### **Phase 1: Ingestion (One-time, offline)**

```
1. Admin adds ./documents/Remote-Work-Policy.pdf

2. Parsing:
   PDF → "Remote Work Policy v1.2"
       → "Section 1: Eligibility"
       → "Full-time employees may request..."
       → [extracts 12 pages of policy text]

3. Chunking:
   Splits into ~50 chunks of 800 chars each with 150-char overlap
   Chunk 1: "Remote Work Policy v1.2. Remote work is available..."
   Chunk 2: "...employees may request up to 3 days per week."
   Chunk 3: "Approval requires manager agreement..."

4. Embedding:
   Chunk 1 → [0.12, -0.15, 0.08, ...] (384 dims)
   Chunk 2 → [0.13, -0.14, 0.10, ...] (384 dims)
   ...

5. FAISS Index:
   Stores 50 vectors in flat index for O(1) search

6. BM25 Index:
   Tokenizes each chunk: ["remote", "work", "policy", ...]
   Builds BM25 scorer

   Both indices persisted to disk.
```

#### **Phase 2: Query (Subsecond)**

```
1. User asks: "Can I work from home 3 days a week?"

2. API receives POST /api/query

3. Scope check:
   - "work" in query? ✓ YES → HR keyword found
   - In scope ✓

4. Dense retrieval:
   - Embed query → [0.14, -0.13, 0.09, ...] (same model as chunks)
   - FAISS search: argmax(query · chunk_i) for all chunks
   - Returns top-20 most similar chunks by cosine similarity
   
   Result: Chunk 2 (score 0.91), Chunk 3 (score 0.88), Chunk 5 (score 0.75), ...

5. BM25 retrieval:
   - Tokenize: ["can", "i", "work", "from", "home", "3", "days", "a", "week"]
   - BM25 scores all 50 chunks
   - High score for chunks containing "work from home"
   
   Result: Chunk 2 (score 42.1), Chunk 1 (score 18.3), Chunk 4 (score 15.2), ...

6. RRF fusion:
   - Combine ranks from dense and BM25
   - Chunk 2 appears top in both → boosted
   - Final ranking: Chunk 2 (fused_score 0.033), Chunk 3 (0.021), ...

7. Re-ranking:
   - Take top-20 from fusion
   - Cross-encoder scores each (query, chunk_text) pair
   - Chunk 2: sigmoid(2.1) = 0.891 ✓ Highest
   - Chunk 3: sigmoid(1.8) = 0.858
   - Return top-5
   
   Selected chunk: "...employees may request up to 3 days per week."

8. Confidence check:
   - Best score 0.891 > threshold 0.3? ✓ YES
   - Proceed to generation

9. Format context:
   [Excerpt 1]
   Source: Remote-Work-Policy.pdf
   Section: Eligibility
   Page: 2
   Relevance Score: 0.891
   
   Employees may request up to 3 days of remote work per week.
   Remote work is subject to manager approval...

10. LLM generation (Groq):
    System: "Answer ONLY from [CONTEXT]. Cite sources."
    Context: [formatted above]
    Question: "Can I work from home 3 days a week?"
    
    LLM response:
    "Yes, based on the Remote Work Policy, you can request up to 3 days 
    of remote work per week. This is subject to manager approval and 
    your specific role requirements.
    (Source: Remote-Work-Policy.pdf, Section: Eligibility, Page: 2)"

11. Build sources:
    [
        {
            "document": "Remote-Work-Policy.pdf",
            "section": "Eligibility",
            "page": 2,
            "relevance_score": 0.891,
            "excerpt": "Employees may request up to 3 days..."
        }
    ]

12. Return to frontend:
    {
        "answer": "Yes, based on... [full answer]",
        "sources": [{ "document": "...", ... }],
        "confidence": 0.891,
        "query": "Can I work from home 3 days a week?",
        "latency_ms": 312,
        "fallback_triggered": false
    }

13. Frontend displays:
    User: "Can I work from home 3 days a week?"
    Bot: "Yes, based on the Remote Work Policy, you can request up to 3 days...
         Confidence: 0.891 | Latency: 312ms
         Sources: Remote-Work-Policy.pdf - Eligibility"
```

---

## Key Design Decisions

### 1. **Hybrid Retrieval (Dense + Sparse)**

**Why not just dense (FAISS)?**
- Dense embedding captures semantic meaning (great for paraphrases)
- But misses exact keyword matches (if you ask for "PTO", dense might miss "paid time off")

**Why not just BM25?**
- BM25 is exact match (great for keywords)
- But misses semantic paraphrases ("maternity leave" vs "mother's leave")

**Hybrid Solution:**
- Dense for semantic similarity (0.6 weight)
- BM25 for keyword matches (0.4 weight)
- RRF combines both without calibrating scales

### 2. **Cross-Encoder Re-ranking**

**Why not use top-20 from FAISS directly?**
- FAISS uses bi-encoder: scores query and passage independently
- Doesn't capture (query, passage) interaction well

**Why cross-encoder?**
- Scores query and passage **jointly**
- Can understand nuance: "Is remote work allowed?" is more relevant to "We allow remote work" than "Remote work is a privilege"

**Trade-off:**
- Slower than bi-encoder (no pre-computed embeddings)
- But only on top-20 (not all 1000s)

### 3. **Confidence Threshold**

**Why not use lowest confidence?**
- Cross-encoder outputs are well-calibrated (via sigmoid normalization)
- Default threshold 0.3 means: "Must be at least 30% confident this chunk is relevant"

**What happens below threshold?**
- Returns FALLBACK_RESPONSE instead of hallucinating
- "I don't have information about that. Contact HR."

### 4. **Overlapping Chunks**

**Why 150-char overlap?**
- Questions often span chunk boundaries
- If chunk boundary splits a sentence, next chunk repeats end of sentence
- Ensures complete context

**Why 800-char chunks?**
- Fits comfortably in LLM context (most LLMs handle 4096+ tokens)
- ~320 tokens per 800 chars
- Leaves room for question, prompt, other context

### 5. **L2-Normalized Embeddings**

**Why normalize?**
- Enables cosine similarity via inner product
- Cosine similarity is scale-invariant (better for comparing similarity)

**Formula:**
```
embedding_normalized = embedding / ||embedding||_2
cosine_sim(q, doc) = q_norm · doc_norm (inner product)
```

### 6. **Groq API for LLM**

**Why Groq?**
- Free tier available (up to 30k requests/month)
- LLaMA 3 (open-source, capable)
- Fast inference (optimized hardware)

**Alternative:** OpenAI GPT-4, but costs money per query

### 7. **Background Ingestion**

**Why POST /api/ingest runs in background?**
- Full ingestion takes minutes (parse + embed all docs)
- Blocks on synchronous call
- Background task: user gets "accepted" immediately, ingestion happens asynchronously
- After done, pipeline reloads indices without restart

---

## Interview Talking Points

### **Problem Statement**
"We built a system to answer HR policy questions from company documents, avoiding hallucinations and providing source citations."

### **Key Innovation**
"Hybrid retrieval combining dense semantic search (FAISS) and sparse keyword search (BM25), fused with reciprocal rank fusion, then re-ranked with a cross-encoder for maximum relevance."

### **Scalability**
- Ingestion: O(n) where n = chunk count (done once, offline)
- Query: O(k) where k = number of results retrieved (typically 20, constant time)
- Subsecond latency per query

### **Production-Ready Features**
- Grounded generation: model cites sources, avoids hallucination
- Scope detection: filters off-topic queries
- Confidence threshold: returns "I don't know" if uncertain
- Hot-reload: /api/ingest doesn't require API restart
- Health checks: /api/health shows system status
- CORS, error handling, logging (loguru)

### **Challenges & Solutions**

| Challenge | Solution |
|-----------|----------|
| Documents in different formats (PDF/DOCX/TXT) | Tiered parsing: unstructured → pdfminer → python-docx → plain text |
| Chunk boundaries split answers | Overlapping chunks (800 chars, 150-char overlap) |
| Dense & sparse scores have different scales | RRF: rank-based fusion, scale-invariant |
| LLM hallucinations on HR policies | Engineered system prompt, confidence threshold, citation requirement |
| Ingestion takes minutes | Background task: user gets instant confirmation, doesn't block API |
| First query slow (loads indices)? | Indices cached to disk, loaded once at startup |

### **What I'd Do Differently (If Time)**
1. **Semantic caching:** Cache embeddings of common questions
2. **Multi-hop retrieval:** "First retrieve leave policies, then benefits policies" for complex questions
3. **Feedback loop:** Track which answers users found helpful, retrain ranker
4. **Hybrid chunking:** Variable chunk sizes (summaries, full sections, details) depending on document type
5. **Query expansion:** Expand "leave" → "leave, holiday, vacation, time off, days off" before retrieval

---

## How to Use This Project

### **Setup**
```bash
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env, add GROQ_API_KEY (get from https://console.groq.com)
```

### **Ingestion**
```bash
# One-time: parse + chunk + embed + index documents
python ingestion.py

# Or rebuild from scratch
python ingestion.py --force
```

### **Query via CLI**
```bash
python rag_pipeline.py

# Interactive prompt appears, type questions
```

### **Query via API**
```bash
# Start API
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, test
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of leave do I get?"}'

# Or open browser: http://localhost:8000/
```

### **API Endpoints**
- `POST /api/query` — Ask question
- `GET /api/health` — Check system status
- `POST /api/ingest` — Re-index documents
- `GET /api/documents` — List ingested documents
- `GET /docs` — Swagger API documentation

---

## Conclusion

This project demonstrates:
✅ **Practical NLP:** Embeddings, dense/sparse retrieval, cross-encoders  
✅ **System Design:** Hybrid strategies, fusion techniques, confidence thresholds  
✅ **Production Engineering:** APIs, error handling, monitoring, hot-reload  
✅ **Problem Solving:** Hallucination avoidance, grounding, citations  
✅ **Scalability:** Subsecond queries on hundreds of chunks  

**Final note:** The system is designed to be **accurate first** (avoid hallucinations), **helpful second** (cite sources), **fast third** (subsecond latency). In HR, accuracy matters most.
