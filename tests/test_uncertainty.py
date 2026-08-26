import numpy as np
import pytest

from src.uncertainty.entropy_uncertainty import (
    normalized_entropy,
    validate_probabilities,
)


def test_probability_shape_must_be_n_by_four():
    probabilities = np.ones(
        (5, 3),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        validate_probabilities(probabilities)


def test_probability_values_must_be_finite_and_valid():
    probabilities = np.array(
        [[0.25, 0.25, 0.25, np.nan]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        validate_probabilities(probabilities)


def test_probability_rows_must_sum_to_one():
    probabilities = np.array(
        [[0.2, 0.2, 0.2, 0.2]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        validate_probabilities(probabilities)


def test_confident_prediction_has_low_uncertainty():
    probabilities = np.array(
        [[1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    score = normalized_entropy(
        probabilities
    )

    assert score.shape == (1,)
    assert 0.0 <= score[0] <= 1.0
    assert score[0] < 1e-5


def test_uniform_prediction_has_maximum_uncertainty():
    probabilities = np.array(
        [[0.25, 0.25, 0.25, 0.25]],
        dtype=np.float32,
    )

    score = normalized_entropy(
        probabilities
    )

    assert np.isclose(
        score[0],
        1.0,
        atol=1e-6,
    )


def test_more_even_distribution_is_more_uncertain():
    confident = np.array(
        [[0.90, 0.05, 0.03, 0.02]],
        dtype=np.float32,
    )

    less_confident = np.array(
        [[0.40, 0.30, 0.20, 0.10]],
        dtype=np.float32,
    )

    confident_score = normalized_entropy(
        confident
    )[0]

    less_confident_score = normalized_entropy(
        less_confident
    )[0]

    assert (
        less_confident_score
        > confident_score
    )


def test_uncertainty_scores_are_finite_and_bounded():
    probabilities = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.4, 0.3, 0.2, 0.1],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=np.float32,
    )

    scores = normalized_entropy(
        probabilities
    )

    assert np.isfinite(scores).all()
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)