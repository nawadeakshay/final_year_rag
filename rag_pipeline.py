"""
rag_pipeline.py — End-to-end RAG orchestrator using LCEL.

Flow:
  Query → [Scope Check] → Hybrid Retrieval → Cross-Encoder Re-ranking
        → [Confidence Check] → Context Formatting → LLaMA 3 (Groq) → Response

Key design decisions:
  - LCEL RunnableSequence for composable, testable pipeline steps
  - Groq API for fast LLaMA 3 inference (free tier available)
  - Explicit fallback at two points: scope check + confidence threshold
  - Full source citation metadata returned alongside answer

FIX (startup): Pipeline now catches FileNotFoundError during index load and
  sets self._ready = False instead of crashing the entire API process. The
  /api/health endpoint will report status="not_ready" so operators know
  ingestion is needed.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from loguru import logger

from config import settings
from prompt_templates import (
    build_rag_prompt,
    format_context,
    FALLBACK_RESPONSE,
)


# ===========================================================================
# Query Scope Detection
# ===========================================================================

_HR_KEYWORDS = {
    # Leave / time off
    "leave", "holiday", "vacation", "absence", "sick", "maternity", "paternity",
    "annual", "pto", "time off", "days off",
    # Compensation / benefits
    "salary", "bonus", "pay", "compensation", "benefit", "reimbursement",
    "expense", "allowance", "pension", "insurance",
    # Conduct / ethics
    "conduct", "ethics", "fraud", "corruption", "harassment", "discrimination",
    "diversity", "grievance", "complaint", "misconduct",
    # Security
    "cyber", "security", "password", "data", "confidential",
    # Work arrangements
    "remote", "work", "office", "hours", "attendance", "probation",
    "performance", "appraisal",
    # Termination / resignation
    "resign", "resignation", "terminate", "termination", "notice", "redundancy",
    "dismiss", "dismissal",
    # Policy / code
    "policy", "code", "rule", "procedure", "guideline", "regulation",
    "conflict", "interest", "dress", "code",
}

# Question-opening words — allow any question through for retrieval to decide
_QUESTION_STARTERS = {
    "what", "how", "when", "where", "who", "can", "do", "is", "am",
    "will", "should", "does", "are", "which", "why", "tell", "explain",
    "describe", "list", "give", "show", "define",
}


def is_likely_in_scope(query: str) -> bool:
    """
    Lightweight heuristic to filter obviously off-topic queries.

    FIX: Much less aggressive than the original — we now pass through any
    question-shaped input (starts with a question word OR contains a '?'),
    and only block very short or non-question, non-HR-keyword queries.
    This prevents valid HR questions from being blocked by missing a keyword.
    """
    stripped = query.strip()
    if len(stripped) < 5:
        return False

    query_lower = stripped.lower()

    # Pass if any known HR keyword present
    for kw in _HR_KEYWORDS:
        if kw in query_lower:
            return True

    # Pass if query starts with a question word
    first_word = query_lower.split()[0]
    if first_word in _QUESTION_STARTERS:
        return True

    # Pass if it contains a question mark
    if "?" in query:
        return True

    # Block anything else (greetings, random text, etc.)
    return False


# ===========================================================================
# Pipeline Result
# ===========================================================================

class RAGResult:
    """Structured result returned by the pipeline."""

    def __init__(
        self,
        answer: str,
        sources: List[dict],
        confidence: float,
        query: str,
        latency_ms: float,
        fallback_triggered: bool = False,
    ):
        self.answer = answer
        self.sources = sources
        self.confidence = confidence
        self.query = query
        self.latency_ms = latency_ms
        self.fallback_triggered = fallback_triggered

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": round(self.confidence, 4),
            "query": self.query,
            "latency_ms": round(self.latency_ms, 2),
            "fallback_triggered": self.fallback_triggered,
        }


# ===========================================================================
# HR RAG Pipeline
# ===========================================================================

class HRRagPipeline:
    """
    Main RAG pipeline. Instantiate once at app startup; call .query() per request.

    Thread-safety note:
      SentenceTransformer and CrossEncoder are stateless at inference time.
      FAISS is read-only after index load. Groq client is stateless.
      Safe for concurrent requests with a single instance.
    """

    def __init__(self) -> None:
        logger.info("Initialising HR RAG Pipeline...")
        self._ready = False
        self._not_ready_reason = ""

        # ------ LLM (Groq) ------
        try:
            settings.validate()
        except EnvironmentError as exc:
            self._not_ready_reason = str(exc)
            logger.error(f"Config validation failed: {exc}")
            return

        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

        # ------ Retrieval + Reranker ------
        # FIX: Catch FileNotFoundError so the API starts even without indices.
        # Operators can then POST /api/ingest to build them.
        try:
            from retrieval import HybridRetriever
            from reranker import CrossEncoderReranker

            self.retriever = HybridRetriever()
            self.reranker = CrossEncoderReranker()
        except FileNotFoundError as exc:
            self._not_ready_reason = (
                f"Indices not found — run ingestion first. Details: {exc}"
            )
            logger.warning(self._not_ready_reason)
            return
        except Exception as exc:
            self._not_ready_reason = f"Failed to load retrieval components: {exc}"
            logger.error(self._not_ready_reason)
            return

        # ------ Prompt + chain ------
        self.prompt = build_rag_prompt()
        self._chain = self.prompt | self.llm

        self._ready = True
        logger.success("HR RAG Pipeline ready ✓")

    def _ensure_ready(self) -> None:
        """Raise if pipeline is not ready to serve queries."""
        if not self._ready:
            raise RuntimeError(
                f"Pipeline not ready: {self._not_ready_reason}. "
                "POST /api/ingest to build indices."
            )

    def reload_retrieval(self) -> None:
        """
        Reload retrieval components after a background ingestion completes.
        FIX: Used by the /api/ingest endpoint to hot-reload without restart.
        """
        from retrieval import HybridRetriever
        from reranker import CrossEncoderReranker

        try:
            self.retriever = HybridRetriever()
            self.reranker = CrossEncoderReranker()
            self._ready = True
            self._not_ready_reason = ""
            logger.success("Retrieval components reloaded after ingestion.")
        except Exception as exc:
            self._not_ready_reason = str(exc)
            logger.error(f"Reload failed: {exc}")

    # ------------------------------------------------------------------
    # Core query method
    # ------------------------------------------------------------------

    def query(self, user_query: str) -> RAGResult:
        """
        Process a user query through the full RAG pipeline.
        """
        self._ensure_ready()

        start = time.perf_counter()
        user_query = user_query.strip()
        logger.info(f"Query received: '{user_query[:100]}'")

        # ── Step 1: Scope check ──────────────────────────────────────
        if not is_likely_in_scope(user_query):
            logger.info("Query out of scope — returning fallback")
            return RAGResult(
                answer=FALLBACK_RESPONSE,
                sources=[],
                confidence=0.0,
                query=user_query,
                latency_ms=_elapsed_ms(start),
                fallback_triggered=True,
            )

        # ── Step 2: Hybrid retrieval ─────────────────────────────────
        candidates = self.retriever.retrieve(user_query)

        if not candidates:
            logger.warning("No candidates retrieved — returning fallback")
            return RAGResult(
                answer=FALLBACK_RESPONSE,
                sources=[],
                confidence=0.0,
                query=user_query,
                latency_ms=_elapsed_ms(start),
                fallback_triggered=True,
            )

        # ── Step 3: Cross-encoder re-ranking ─────────────────────────
        reranked = self.reranker.rerank(user_query, candidates)

        # ── Step 4: Confidence threshold check ───────────────────────
        top_confidence = self.reranker.best_confidence(reranked)

        if top_confidence < settings.CONFIDENCE_THRESHOLD:
            logger.info(
                f"Confidence {top_confidence:.3f} < threshold "
                f"{settings.CONFIDENCE_THRESHOLD} — returning fallback"
            )
            return RAGResult(
                answer=FALLBACK_RESPONSE,
                sources=[],
                confidence=top_confidence,
                query=user_query,
                latency_ms=_elapsed_ms(start),
                fallback_triggered=True,
            )

        # ── Step 5: Format context ────────────────────────────────────
        context_str = format_context(reranked)

        # ── Step 6: Generate answer via LLaMA 3 ──────────────────────
        logger.debug("Sending context + query to LLM...")
        llm_response = self._chain.invoke(
            {"context": context_str, "question": user_query}
        )
        answer = _extract_text(llm_response)

        # ── Step 7: Build source citations ────────────────────────────
        sources = _build_sources(reranked)

        elapsed = _elapsed_ms(start)
        logger.info(
            f"Query answered | confidence={top_confidence:.3f} | "
            f"sources={len(sources)} | latency={elapsed:.0f}ms"
        )

        return RAGResult(
            answer=answer,
            sources=sources,
            confidence=top_confidence,
            query=user_query,
            latency_ms=elapsed,
            fallback_triggered=False,
        )

    def health(self) -> dict:
        """Return pipeline health/stats for the /health endpoint."""
        if not self._ready:
            return {
                "status": "not_ready",
                "reason": self._not_ready_reason,
                "retriever": {},
                "llm_model": settings.LLM_MODEL,
                "reranker_model": settings.RERANKER_MODEL,
                "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
            }
        return {
            "status": "ok",
            "retriever": self.retriever.get_stats(),
            "llm_model": settings.LLM_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
        }


# ===========================================================================
# Helpers
# ===========================================================================

def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _extract_text(llm_response) -> str:
    """Extract string content from LangChain LLM response object."""
    if hasattr(llm_response, "content"):
        return llm_response.content.strip()
    return str(llm_response).strip()


def _build_sources(reranked: List[Tuple[Document, float]]) -> List[dict]:
    """Build clean source citation dicts from reranked chunks."""
    seen = set()
    sources = []
    for doc, score in reranked:
        meta = doc.metadata
        key = (meta.get("source", ""), meta.get("section", ""), meta.get("page", ""))
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "document": meta.get("source", "Unknown"),
                    "section": meta.get("section", "General"),
                    "page": meta.get("page", "N/A"),
                    "relevance_score": round(score, 4),
                    "excerpt": doc.page_content[:200].replace("\n", " ") + "...",
                }
            )
    return sources


# ===========================================================================
# Singleton — lazily initialised on first import by the API
# ===========================================================================

_pipeline_instance: Optional[HRRagPipeline] = None


def get_pipeline() -> HRRagPipeline:
    """Return (or create) the global pipeline singleton."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = HRRagPipeline()
    return _pipeline_instance


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    pipeline = get_pipeline()

    print("\n" + "=" * 60)
    print("  HR Policy Assistant — Interactive CLI")
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        result = pipeline.query(user_input)

        print(f"\nAssistant: {result.answer}")
        if result.sources:
            print("\nSources:")
            for src in result.sources:
                print(
                    f"  • {src['document']} | "
                    f"Section: {src['section']} | "
                    f"Page: {src['page']} | "
                    f"Score: {src['relevance_score']}"
                )
        print(f"\n[Confidence: {result.confidence:.2f} | Latency: {result.latency_ms:.0f}ms]\n")
