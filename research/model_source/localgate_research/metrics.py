"""Pure probability metrics with explicit support for undefined statistics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

EPSILON = 1e-9
KAPPA_LEVELS = 6
SWEEP_THRESHOLDS = tuple(np.arange(0.05, 1.0, 0.05))


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")


def _inputs(correct: np.ndarray, k: int, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _validate_k(k)
    counts = np.asarray(correct, dtype=np.float64)
    probabilities = np.asarray(predicted, dtype=np.float64)
    if counts.ndim != 1 or not counts.size or probabilities.shape != counts.shape:
        raise ValueError("Counts and probabilities must be nonempty aligned vectors")
    if not np.all(np.isfinite(counts)) or not np.all(np.isfinite(probabilities)):
        raise ValueError("Counts and probabilities must be finite")
    if np.any((counts < 0) | (counts > k) | (counts != np.floor(counts))):
        raise ValueError("Counts must be integers in 0..k")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must lie in [0, 1]")
    return counts, probabilities


def binomial_nll(correct: np.ndarray, k: int, predicted: np.ndarray) -> float:
    """Return count-scaled NLL, excluding the combinatorial constant."""
    counts, probabilities = _inputs(correct, k, predicted)
    p = np.clip(probabilities, EPSILON, 1 - EPSILON)
    return float(np.mean(-(counts * np.log(p) + (k - counts) * np.log1p(-p))))


def binomial_deviance(correct: np.ndarray, k: int, predicted: np.ndarray) -> float:
    counts, probabilities = _inputs(correct, k, predicted)
    return 2 * (binomial_nll(counts, k, probabilities) - binomial_nll(counts, k, counts / k))


def label_noise_floor(k: int, p: float = 0.5) -> float:
    """Return the nominal independent-Binomial rate SE, not a model accuracy ceiling."""
    _validate_k(k)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must lie in [0, 1]")
    return math.sqrt(p * (1 - p) / k)


def mae(correct: np.ndarray, k: int, predicted: np.ndarray) -> float:
    counts, probabilities = _inputs(correct, k, predicted)
    return float(np.mean(np.abs(probabilities - counts / k)))


def rmse(correct: np.ndarray, k: int, predicted: np.ndarray) -> float:
    return math.sqrt(brier(correct, k, predicted))


def brier(correct: np.ndarray, k: int, predicted: np.ndarray) -> float:
    """Return MSE against observed success fractions, not a trial-level Brier score."""
    counts, probabilities = _inputs(correct, k, predicted)
    return float(np.mean((probabilities - counts / k) ** 2))


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    _, starts, sizes = np.unique(values[order], return_index=True, return_counts=True)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.repeat(starts + (sizes - 1) / 2, sizes)
    return ranks


def spearman(correct: np.ndarray, k: int, predicted: np.ndarray) -> float:
    counts, probabilities = _inputs(correct, k, predicted)
    observed = _ranks(counts)
    forecast = _ranks(probabilities)
    observed -= observed.mean()
    forecast -= forecast.mean()
    denominator = np.linalg.norm(observed) * np.linalg.norm(forecast)
    if not denominator:
        return math.nan
    return float(np.clip(np.dot(observed, forecast) / denominator, -1, 1))


def success_level(rate: np.ndarray, levels: int = KAPPA_LEVELS) -> np.ndarray:
    """Map [0, 1] to equal-width bands, including p=1 in the final band."""
    if isinstance(levels, bool) or not isinstance(levels, (int, np.integer)) or levels < 2:
        raise ValueError("levels must be an integer of at least two")
    values = np.asarray(rate, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Rates must be finite probabilities")
    return np.minimum(np.floor(values * levels).astype(np.int64), levels - 1)


def kappa_w(
    correct: np.ndarray,
    k: int,
    predicted: np.ndarray,
    levels: int = KAPPA_LEVELS,
) -> float:
    """Return linearly weighted kappa on a fixed complete band scale."""
    counts, probabilities = _inputs(correct, k, predicted)
    observed = success_level(counts / k, levels)
    forecast = success_level(probabilities, levels)
    observed_frequency = np.bincount(observed, minlength=levels) / counts.size
    forecast_frequency = np.bincount(forecast, minlength=levels) / counts.size
    distances = np.abs(np.arange(levels)[:, None] - np.arange(levels))
    expected = float(np.sum(distances * observed_frequency[:, None] * forecast_frequency))
    if not expected:
        return math.nan
    return float(1 - np.mean(np.abs(observed - forecast)) / expected)


def binary_routing(
    correct: np.ndarray,
    k: int,
    predicted: np.ndarray,
    label_threshold: float,
    route_threshold: float,
) -> dict:
    counts, probabilities = _inputs(correct, k, predicted)
    if any(not math.isfinite(t) or not 0 <= t <= 1 for t in (label_threshold, route_threshold)):
        raise ValueError("Thresholds must lie in [0, 1]")
    truth = counts / k >= label_threshold
    local = probabilities >= route_threshold
    tp, fp = int(np.sum(local & truth)), int(np.sum(local & ~truth))
    tn, fn = int(np.sum(~local & ~truth)), int(np.sum(~local & truth))
    positives, negatives = tp + fn, tn + fp
    macro_f1 = math.nan
    if positives and negatives:
        macro_f1 = (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) / 2
    return {
        "true_pos": tp,
        "false_pos": fp,
        "true_neg": tn,
        "false_neg": fn,
        "fpr": fp / negatives if negatives else math.nan,
        "fnr": fn / positives if positives else math.nan,
        "macro_f1": macro_f1,
        "local_coverage": float(np.mean(local)),
    }


def threshold_sweep(
    correct: np.ndarray,
    k: int,
    predicted: np.ndarray,
    label_threshold: float = 0.5,
    thresholds: tuple[float, ...] = SWEEP_THRESHOLDS,
) -> list[dict]:
    _inputs(correct, k, predicted)
    return [
        {
            "threshold": round(float(t), 3),
            **binary_routing(correct, k, predicted, label_threshold, float(t)),
        }
        for t in thresholds
    ]


def evaluate(
    correct: np.ndarray,
    k: int,
    predicted: np.ndarray,
    label_threshold: float = 0.5,
    route_threshold: float = 0.5,
) -> dict:
    """Return count-scaled loss, forecast errors and routing statistics."""
    counts, probabilities = _inputs(correct, k, predicted)
    return {
        "n": int(counts.size),
        "kappa_w": kappa_w(counts, k, probabilities),
        "binomial_nll": binomial_nll(counts, k, probabilities),
        "binomial_deviance": binomial_deviance(counts, k, probabilities),
        "mae": mae(counts, k, probabilities),
        "rmse": rmse(counts, k, probabilities),
        "brier": brier(counts, k, probabilities),
        "spearman": spearman(counts, k, probabilities),
        "label_noise_floor": label_noise_floor(k),
        "routing": binary_routing(counts, k, probabilities, label_threshold, route_threshold),
        "threshold_sweep": threshold_sweep(counts, k, probabilities, label_threshold),
    }


def evaluate_predictions(
    counts: np.ndarray,
    probabilities: np.ndarray,
    pmf: np.ndarray | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Score aligned predictions; undefined statistics carry null, a reason and support."""
    counts, probabilities = _inputs(counts, k, probabilities)
    positives = int(np.sum(counts / k >= 0.5))
    negatives = int(counts.size) - positives
    reasons: dict[str, str] = {}

    def defined(name: str, value: float, reason: str) -> float | None:
        if math.isfinite(value):
            return value
        reasons[name] = reason
        return None

    routing = binary_routing(counts, k, probabilities, 0.5, 0.5)
    for name, reason in (
        ("fpr", "No observed negative examples"),
        ("fnr", "No observed positive examples"),
        ("macro_f1", "Both observed classes are required"),
    ):
        routing[name] = defined(f"routing.{name}", routing[name], reason)
    result: dict[str, Any] = {
        "n": int(counts.size),
        "k": int(k),
        "mean_log_loss": binomial_nll(counts, k, probabilities) / k,
        "linear_kappa": defined(
            "linear_kappa",
            kappa_w(counts, k, probabilities),
            "Expected disagreement is zero; both band distributions are the same constant",
        ),
        "empirical_rate_mse": brier(counts, k, probabilities),
        "mae": mae(counts, k, probabilities),
        "rmse": rmse(counts, k, probabilities),
        "spearman": defined(
            "spearman",
            spearman(counts, k, probabilities),
            "At least two examples and nonconstant observed and predicted ranks are required",
        ),
        "routing": routing,
        "support": {"total": int(counts.size), "positive": positives, "negative": negatives},
        "undefined": reasons,
    }
    if pmf is not None:
        distribution = np.asarray(pmf, dtype=np.float64)
        if distribution.shape != (counts.size, k + 1):
            raise ValueError("Native PMF must have shape (n, k + 1)")
        if not np.all(np.isfinite(distribution)) or np.any((distribution < 0) | (distribution > 1)):
            raise ValueError("Native PMF entries must be finite probabilities")
        if not np.allclose(distribution.sum(axis=1), 1, rtol=0, atol=1e-7):
            raise ValueError("Native PMF rows must sum to one")
        if not np.allclose(distribution @ (np.arange(k + 1) / k), probabilities, rtol=0, atol=1e-7):
            raise ValueError("Native PMF mean disagrees with the authoritative probability")
        observed_probability = distribution[np.arange(counts.size), counts.astype(np.int64)]
        positive_mass = observed_probability > 0
        result["support"]["native_count_positive_mass"] = int(positive_mass.sum())
        if positive_mass.all():
            result["native_count_nll"] = float(-np.log(observed_probability).mean())
        else:
            result["native_count_nll"] = None
            reasons["native_count_nll"] = "An observed count has zero or underflowed native mass"
        observed_cdf = counts[:, None] <= np.arange(k)
        result["native_rps"] = float(
            np.mean((distribution.cumsum(axis=1)[:, :-1] - observed_cdf) ** 2)
        )
    return result
