"""
reranker.py — Cross-Encoder re-ranking layer.

After hybrid retrieval returns ~20 candidates, the cross-encoder scores each
(query, passage) pair jointly — capturing semantic relevance far better than
bi-encoder similarity alone. We then select the top-k for generation.

Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS-MARCO passage retrieval (query-passage relevance)
  - Outputs a raw logit; higher == more relevant
  - ~12M parameters — fast inference on CPU
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from loguru import logger
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import settings


class CrossEncoderReranker:
    """
    Singleton-friendly cross-encoder wrapper.
    Call rerank() after hybrid retrieval to get the best chunks.
    """

    def __init__(self) -> None:
        logger.info(f"Loading cross-encoder: {settings.RERANKER_MODEL}")
        self._model = CrossEncoder(settings.RERANKER_MODEL, max_length=512)
        logger.success("Cross-encoder loaded")

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_k: int | None = None,
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank candidate (Document, retrieval_score) pairs.

        Args:
            query:      The user's question.
            candidates: Output of HybridRetriever.retrieve()
            top_k:      How many chunks to return. Defaults to settings.RERANK_TOP_K.

        Returns:
            Top-k (Document, cross_encoder_score) pairs, best first.
            cross_encoder_score is sigmoid-normalised to [0, 1].
        """
        k = top_k or settings.RERANK_TOP_K

        if not candidates:
            logger.warning("Reranker received empty candidate list")
            return []

        pairs = [(query, doc.page_content) for doc, _ in candidates]

        raw_scores: np.ndarray = self._model.predict(
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        normalised_scores = self._sigmoid(raw_scores)

        scored: List[Tuple[Document, float]] = [
            (doc, float(score))
            for (doc, _), score in zip(candidates, normalised_scores)
        ]

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]

        logger.debug(
            f"Reranker: {len(candidates)} candidates → top {len(top)} selected "
            f"(scores: {[round(s, 3) for _, s in top]})"
        )

        return top

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Stable sigmoid: 1 / (1 + e^-x)"""
        return 1.0 / (1.0 + np.exp(-x))

    def best_confidence(
        self, reranked: List[Tuple[Document, float]]
    ) -> float:
        """Return the highest confidence score from a reranked list."""
        if not reranked:
            return 0.0
        return reranked[0][1]
