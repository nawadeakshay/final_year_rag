"""
retrieval.py — Hybrid retrieval combining Dense (FAISS) + Sparse (BM25) search.

Strategy:
  - Dense retrieval: cosine similarity via FAISS IndexFlatIP
  - Sparse retrieval: BM25Okapi term-frequency scoring
  - Fusion: Reciprocal Rank Fusion (RRF) — rank-based, robust to score scale differences

Why RRF?
  Weighted score fusion requires calibrated score ranges across retrievers.
  RRF only uses rank positions, making it robust and parameter-free beyond k.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from loguru import logger
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

from config import settings
from ingestion import (
    load_embedding_model,
    load_faiss_index,
    load_bm25_index,
)


# ---------------------------------------------------------------------------
# RRF constant — standard choice; higher k → less weight on top ranks
# ---------------------------------------------------------------------------
RRF_K = 60


class HybridRetriever:
    """
    Stateful retriever that holds FAISS index, BM25 index, and the embedding
    model. Designed to be instantiated once at app startup and reused.
    """

    def __init__(self) -> None:
        logger.info("Initialising HybridRetriever...")

        # FIX: load_faiss_index and load_bm25_index now raise FileNotFoundError
        # with actionable messages when indices are missing, rather than cryptic IOErrors.
        self.faiss_index, self.chunks = load_faiss_index(settings.VECTOR_STORE_PATH)
        self.bm25: BM25Okapi = load_bm25_index(settings.BM25_INDEX_PATH)

        # Embedding model (same as used at ingestion time)
        self.embed_model: SentenceTransformer = load_embedding_model()

        # Pre-tokenise chunk texts for BM25 lookup
        self._tokenised_chunks: List[List[str]] = [
            chunk.page_content.lower().split() for chunk in self.chunks
        ]

        logger.success(
            f"HybridRetriever ready — {len(self.chunks)} chunks in index"
        )

    # ------------------------------------------------------------------
    # Dense retrieval
    # ------------------------------------------------------------------

    def _dense_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Embed query and search FAISS.
        Returns: list of (chunk_index, score) sorted by descending score.
        """
        query_vec = self.embed_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self.faiss_index.search(query_vec, k)

        results = [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]
        return results

    # ------------------------------------------------------------------
    # Sparse retrieval (BM25)
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        BM25 search over tokenised chunks.
        Returns: list of (chunk_index, bm25_score) sorted by descending score.
        """
        tokenised_query = query.lower().split()
        scores = self.bm25.get_scores(tokenised_query)

        top_indices = np.argsort(scores)[::-1][:k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        return results

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: List[List[Tuple[int, float]]],
        k: int = RRF_K,
    ) -> List[Tuple[int, float]]:
        """
        Combine multiple ranked lists via RRF.
        RRF score for doc d = Σ 1 / (k + rank(d, list_i))
        Returns merged list sorted by descending RRF score.
        """
        rrf_scores: dict[int, float] = {}

        for ranked_list in ranked_lists:
            for rank, (doc_idx, _raw_score) in enumerate(ranked_list, start=1):
                rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank)

        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return merged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        initial_k: int | None = None,
    ) -> List[Tuple[Document, float]]:
        """
        Run hybrid retrieval and return ranked (Document, rrf_score) pairs.
        """
        k = initial_k or settings.INITIAL_RETRIEVAL_K

        logger.debug(f"Hybrid retrieval | query='{query[:80]}' | k={k}")

        dense_results = self._dense_search(query, k)
        bm25_results = self._bm25_search(query, k)

        logger.debug(
            f"  Dense hits: {len(dense_results)} | BM25 hits: {len(bm25_results)}"
        )

        fused = self._reciprocal_rank_fusion([dense_results, bm25_results])

        retrieved: List[Tuple[Document, float]] = []
        for chunk_idx, rrf_score in fused:
            if chunk_idx < len(self.chunks):
                doc = self.chunks[chunk_idx]
                retrieved.append((doc, rrf_score))

        logger.debug(f"  Fused results: {len(retrieved)}")
        return retrieved

    def get_stats(self) -> dict:
        """Return index statistics for health-check endpoint."""
        return {
            "total_chunks": len(self.chunks),
            "faiss_vectors": self.faiss_index.ntotal,
            "embedding_model": settings.EMBEDDING_MODEL,
            "retrieval_k": settings.INITIAL_RETRIEVAL_K,
        }
