"""
api.py — FastAPI REST API for the HR Policy RAG System.

Endpoints:
  POST /api/query          — Ask a question
  POST /api/ingest         — Trigger document re-ingestion
  GET  /api/health         — Pipeline health + stats
  GET  /api/documents      — List ingested documents
  GET  /docs               — Auto-generated Swagger UI

Run:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

FIXES vs original:
  1. HealthResponse.retriever is now `dict` (Any) — matches what pipeline.health() returns.
  2. QueryResponse.sources uses List[dict] (Any) not List[SourceCitation] — pipeline
     returns plain dicts, not Pydantic models. Kept SourceCitation for documentation
     but validation is done at the dict level to avoid serialization errors.
  3. /api/ingest background task now calls pipeline.reload_retrieval() instead of
     directly replacing private attributes on _pipeline_instance.
  4. Lifespan no longer crashes if indices are missing — pipeline._ready=False and
     health endpoint reports it. Operators POST /api/ingest to build indices.
  5. CORS origins pulled from settings.CORS_ORIGINS property (env-aware).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from loguru import logger

from config import settings


# ===========================================================================
# Lifespan — initialise pipeline once at startup
# ===========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load pipeline (best-effort — no crash if indices missing).
    Shutdown: nothing to clean up.
    """
    logger.info("API starting up — loading RAG pipeline...")

    # FIX: Lazy import; get_pipeline() no longer raises on missing indices.
    # Pipeline will set _ready=False and report via /api/health instead.
    from rag_pipeline import get_pipeline
    app.state.pipeline = get_pipeline()

    if app.state.pipeline._ready:
        logger.success("API ready — pipeline loaded ✓")
    else:
        logger.warning(
            "API started but pipeline is NOT ready. "
            f"Reason: {app.state.pipeline._not_ready_reason}\n"
            "POST /api/ingest to build indices."
        )
    yield
    logger.info("API shutting down")


# ===========================================================================
# FastAPI app
# ===========================================================================

app = FastAPI(
    title="HR Policy Assistant API",
    description=(
        "Enterprise-grade RAG system for answering employee queries "
        "strictly from company HR policy documents."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# FIX: Use settings.CORS_ORIGINS property (env-aware, correctly parsed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Pydantic schemas
# ===========================================================================

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The employee's HR policy question",
        example="How many days of annual leave am I entitled to?",
    )


class SourceCitation(BaseModel):
    """Schema for a single source citation — used in API docs / OpenAPI spec."""
    document: str
    section: str
    page: Any          # can be int or "N/A"
    relevance_score: float
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    # FIX: sources comes back from the pipeline as List[dict], not List[SourceCitation].
    # Using List[Dict[str, Any]] avoids Pydantic validation errors while keeping
    # the schema flexible. The actual shape matches SourceCitation above.
    sources: List[Dict[str, Any]]
    confidence: float
    query: str
    latency_ms: float
    fallback_triggered: bool


class HealthResponse(BaseModel):
    status: str
    # FIX: retriever is a plain dict returned by HybridRetriever.get_stats()
    # (or {} when pipeline is not ready). Using Dict[str, Any] matches reality.
    retriever: Dict[str, Any]
    llm_model: str
    reranker_model: str
    confidence_threshold: float
    reason: Optional[str] = None   # populated when status != "ok"


class IngestResponse(BaseModel):
    status: str
    message: str


class DocumentInfo(BaseModel):
    filename: str
    size_bytes: int
    extension: str


# ===========================================================================
# Routes
# ===========================================================================

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Returns pipeline health status and index statistics.
    Use this to verify the system is ready to answer questions.
    """
    pipeline = app.state.pipeline
    data = pipeline.health()
    return HealthResponse(**data)


@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
async def query_policy(request: QueryRequest):
    """
    Submit an employee HR policy question.

    The pipeline will:
    1. Check if the question is in scope
    2. Run hybrid retrieval (FAISS + BM25)
    3. Re-rank with cross-encoder
    4. Generate a grounded answer via LLaMA 3
    5. Return the answer with source citations
    """
    pipeline = app.state.pipeline

    # FIX: Return 503 with helpful message if pipeline not ready (e.g. no indices)
    if not pipeline._ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Pipeline not ready: {pipeline._not_ready_reason}. "
                "POST /api/ingest to build indices first."
            ),
        )

    try:
        result = pipeline.query(request.question)
    except Exception as exc:
        logger.exception(f"Pipeline error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {str(exc)}",
        )

    return QueryResponse(**result.to_dict())


@app.post("/api/ingest", response_model=IngestResponse, tags=["Admin"])
async def trigger_ingestion(
    background_tasks: BackgroundTasks,
    force: bool = False,
):
    """
    Trigger document ingestion / re-indexing.

    - If `force=false` (default): skip if indices already exist.
    - If `force=true`: rebuild indices from scratch.

    Ingestion runs in the background. Poll /api/health to confirm completion.
    """
    from ingestion import run_ingestion

    pipeline = app.state.pipeline

    def _ingest():
        try:
            run_ingestion(force_rebuild=force)
            # FIX: Use the proper public reload method instead of poking private attrs
            pipeline.reload_retrieval()
            logger.success("Background ingestion and pipeline reload complete.")
        except Exception as exc:
            logger.error(f"Background ingestion failed: {exc}")

    background_tasks.add_task(_ingest)

    return IngestResponse(
        status="accepted",
        message=(
            "Ingestion started in background. "
            "Poll /api/health to confirm index is updated."
        ),
    )


@app.get("/api/documents", response_model=List[DocumentInfo], tags=["Admin"])
async def list_documents():
    """
    List all documents currently in the documents directory.
    """
    docs_dir = settings.DOCUMENTS_DIR
    if not docs_dir.exists():
        return []

    supported = {".pdf", ".docx", ".doc", ".txt", ".md"}
    files = []
    for f in sorted(docs_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in supported:
            files.append(
                DocumentInfo(
                    filename=f.name,
                    size_bytes=f.stat().st_size,
                    extension=f.suffix.lower(),
                )
            )
    return files


# ===========================================================================
# Serve frontend static files (production build OR dev index.html)
# ===========================================================================

FRONTEND_BUILD = Path(__file__).parent / "frontend" / "dist"
FRONTEND_DEV = Path(__file__).parent / "frontend"

# Try production build first, fall back to dev folder
_frontend_dir: Optional[Path] = None
if FRONTEND_BUILD.exists() and (FRONTEND_BUILD / "index.html").exists():
    _frontend_dir = FRONTEND_BUILD
elif FRONTEND_DEV.exists() and (FRONTEND_DEV / "index.html").exists():
    _frontend_dir = FRONTEND_DEV

if _frontend_dir is not None:
    # Serve /assets from production build if available
    assets_dir = _frontend_dir / "assets"
    if assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(assets_dir)),
            name="assets",
        )

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend(full_path: str = ""):
        """Catch-all route — serves the HTML frontend for any non-API path."""
        # Don't intercept API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        index = _frontend_dir / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return JSONResponse({"detail": "Frontend not found"}, status_code=404)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info",
    )
