"""Distance-based novelty detection using CNN embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoveltyReference:
    """Reference distribution estimated from training embeddings."""

    mean: np.ndarray
    precision: np.ndarray
    training_distances: np.ndarray


class MahalanobisNoveltyDetector:
    """Explainable distance-based novelty detector.

    The reference distribution is fitted using training embeddings only.

    Higher novelty score means the observation is farther from the
    learned training distribution.
    """

    def __init__(self, regularization: float = 1e-5) -> None:
        if regularization <= 0:
            raise ValueError("regularization must be positive")

        self.regularization = float(regularization)
        self.reference: NoveltyReference | None = None

    def fit(self, embeddings: np.ndarray) -> "MahalanobisNoveltyDetector":
        """Fit the reference distribution using training embeddings only."""
        embeddings = self._validate_embeddings(embeddings)

        mean = np.mean(embeddings, axis=0)

        centered = embeddings - mean

        covariance = np.cov(
            centered,
            rowvar=False,
        )

        covariance = np.atleast_2d(covariance)

        covariance += (
            self.regularization
            * np.eye(embeddings.shape[1])
        )

        precision = np.linalg.pinv(covariance)

        training_distances = self._mahalanobis_distance(
            embeddings,
            mean,
            precision,
        )

        self.reference = NoveltyReference(
            mean=mean.astype(np.float32),
            precision=precision.astype(np.float32),
            training_distances=training_distances.astype(np.float32),
        )

        return self

    def distance(self, embeddings: np.ndarray) -> np.ndarray:
        """Return raw Mahalanobis distance from the reference distribution."""
        self._check_fitted()

        embeddings = self._validate_embeddings(embeddings)

        return self._mahalanobis_distance(
            embeddings,
            self.reference.mean,
            self.reference.precision,
        ).astype(np.float32)

    def novelty_score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return novelty scores in [0, 1].

        The score is the empirical percentile of the observation's
        distance relative to the training-distance distribution.

        Higher score = more unusual relative to training data.
        """
        distances = self.distance(embeddings)

        training_distances = np.sort(
            self.reference.training_distances
        )

        ranks = np.searchsorted(
            training_distances,
            distances,
            side="right",
        )

        scores = (
            ranks + 1.0
        ) / (
            len(training_distances) + 1.0
        )

        return np.clip(
            scores,
            0.0,
            1.0,
        ).astype(np.float32)

    @staticmethod
    def _mahalanobis_distance(
        embeddings: np.ndarray,
        mean: np.ndarray,
        precision: np.ndarray,
    ) -> np.ndarray:
        centered = embeddings - mean

        squared_distance = np.einsum(
            "ij,jk,ik->i",
            centered,
            precision,
            centered,
        )

        return np.sqrt(
            np.maximum(
                squared_distance,
                0.0,
            )
        )

    @staticmethod
    def _validate_embeddings(
        embeddings: np.ndarray,
    ) -> np.ndarray:
        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "embeddings must be a 2D array "
                "(n_samples, embedding_dim)"
            )

        if embeddings.shape[0] == 0:
            raise ValueError(
                "embeddings must contain at least one sample"
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "embeddings contain NaN or Inf values"
            )

        return embeddings

    def _check_fitted(self) -> None:
        if self.reference is None:
            raise RuntimeError(
                "Novelty detector must be fitted before use"
            )