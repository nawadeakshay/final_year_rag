"""
config.py — Centralised configuration for the HR Policy RAG System.

All settings are loaded from environment variables (with sensible defaults).
Never hard-code secrets — use .env file.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()


class Settings:
    """
    Central settings object. Every module imports from here — no scattered
    os.getenv() calls across the codebase.
    """

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    # FIX: Removed hardcoded API key — must come from environment / .env file
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    HYBRID_DENSE_WEIGHT: float = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.6"))
    HYBRID_SPARSE_WEIGHT: float = float(os.getenv("HYBRID_SPARSE_WEIGHT", "0.4"))
    INITIAL_RETRIEVAL_K: int = int(os.getenv("INITIAL_RETRIEVAL_K", "20"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))

    # ------------------------------------------------------------------
    # Re-ranker
    # ------------------------------------------------------------------
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # ------------------------------------------------------------------
    # Storage paths
    # ------------------------------------------------------------------
    DOCUMENTS_DIR: Path = Path(os.getenv("DOCUMENTS_DIR", "./documents"))
    VECTOR_STORE_PATH: Path = Path(os.getenv("VECTOR_STORE_PATH", "./vector_store"))
    BM25_INDEX_PATH: Path = Path(os.getenv("BM25_INDEX_PATH", "./bm25_index.pkl"))

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # FIX: Parse CORS_ORIGINS from env string (JSON array) with safe fallback
    @property
    def CORS_ORIGINS(self) -> list[str]:
        raw = os.getenv("CORS_ORIGINS", "")
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        # Default origins for local development
        return [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]

    def validate(self) -> None:
        """Raise early if critical settings are missing."""
        if not self.GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Copy .env.example → .env and add your Groq API key.\n"
                "Get a free key at https://console.groq.com"
            )
        self.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    def indices_exist(self) -> bool:
        """Check whether FAISS and BM25 indices are already built."""
        return (
            (self.VECTOR_STORE_PATH / "index.faiss").exists()
            and (self.VECTOR_STORE_PATH / "chunks.pkl").exists()
            and self.BM25_INDEX_PATH.exists()
        )


# Singleton — import this everywhere
settings = Settings()
