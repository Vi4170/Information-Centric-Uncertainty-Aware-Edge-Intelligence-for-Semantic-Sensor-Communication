import numpy as np
import pytest

from src.novelty.distance_novelty import (
    MahalanobisNoveltyDetector,
)


def test_fit_accepts_valid_embeddings():
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(50, 64)).astype(np.float32)

    detector = MahalanobisNoveltyDetector()
    detector.fit(embeddings)

    assert detector.reference is not None
    assert detector.reference.mean.shape == (64,)
    assert detector.reference.precision.shape == (64, 64)


def test_novelty_scores_are_between_zero_and_one():
    rng = np.random.default_rng(42)

    train = rng.normal(size=(50, 64)).astype(np.float32)
    test = rng.normal(size=(10, 64)).astype(np.float32)

    detector = MahalanobisNoveltyDetector()
    detector.fit(train)

    scores = detector.novelty_score(test)

    assert scores.shape == (10,)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_farther_samples_have_higher_novelty():
    rng = np.random.default_rng(42)

    train = rng.normal(size=(100, 64)).astype(np.float32)

    near = np.zeros((5, 64), dtype=np.float32)
    far = np.full((5, 64), 5.0, dtype=np.float32)

    detector = MahalanobisNoveltyDetector()
    detector.fit(train)

    near_scores = detector.novelty_score(near)
    far_scores = detector.novelty_score(far)

    assert np.mean(far_scores) > np.mean(near_scores)


def test_distance_requires_fitted_detector():
    detector = MahalanobisNoveltyDetector()

    embeddings = np.zeros((2, 64), dtype=np.float32)

    with pytest.raises(RuntimeError):
        detector.distance(embeddings)


def test_invalid_embedding_dimension_is_rejected():
    detector = MahalanobisNoveltyDetector()

    bad_embeddings = np.zeros(
        (10,),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        detector.fit(bad_embeddings)


def test_nan_embeddings_are_rejected():
    detector = MahalanobisNoveltyDetector()

    embeddings = np.zeros(
        (10, 64),
        dtype=np.float32,
    )

    embeddings[0, 0] = np.nan

    with pytest.raises(ValueError):
        detector.fit(embeddings)