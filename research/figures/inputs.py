"""Validated, per-question inputs for the reproducible paper figures."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: expected a readable JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _level(value: object, *, path: Path, field: str) -> int:
    if type(value) is not int or value not in range(6):
        raise ValueError(f"{path}: {field} must be an integer from 0 through 5")
    return value


def load_human_matrix(path: Path) -> tuple[list[list[int]], int]:
    """Recompute the six-band human confusion matrix from report question rows."""

    payload = _load_json(path)
    rows = payload.get("per_prompt")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: per_prompt must contain per-question report rows")
    matrix = [[0 for _ in range(6)] for _ in range(6)]
    prompt_ids: set[str] = set()
    generations: set[int] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: per_prompt[{index}] must be an object")
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in prompt_ids:
            raise ValueError(f"{path}: per_prompt must have unique non-empty prompt_id values")
        prompt_ids.add(prompt_id)
        truth = _level(row.get("truth_level"), path=path, field="truth_level")
        human = _level(row.get("human_median_level"), path=path, field="human_median_level")
        generations_value = row.get("k")
        if type(generations_value) is not int or generations_value <= 0:
            raise ValueError(f"{path}: per_prompt[{index}].k must be a positive integer")
        generations.add(generations_value)
        matrix[truth][human] += 1
    if len(generations) != 1:
        raise ValueError(f"{path}: human figure requires one generation count across prompt rows")
    return matrix, generations.pop()


def load_classifier_predictions(path: Path) -> tuple[list[dict[str, float | int]], int]:
    """Load fresh held-out replay predictions and their observed generation counts."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"{path}: could not read held-out prediction JSONL") from exc
    rows: list[dict[str, float | int]] = []
    question_ids: set[int] = set()
    generations: set[int] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        question_id = row.get("question_id")
        probability = row.get("probability")
        correct_count = row.get("correct_count")
        generations_value = row.get("k")
        if type(question_id) is not int or question_id in question_ids:
            raise ValueError(f"{path}:{line_number}: question_id must be a unique integer")
        if (
            type(probability) not in (int, float)
            or isinstance(probability, bool)
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise ValueError(f"{path}:{line_number}: probability must be finite and in [0, 1]")
        if type(correct_count) is not int or type(generations_value) is not int or not 0 <= correct_count <= generations_value or generations_value <= 0:
            raise ValueError(f"{path}:{line_number}: correct_count must lie in [0, k]")
        question_ids.add(question_id)
        generations.add(generations_value)
        rows.append(
            {
                "question_id": question_id,
                "probability": float(probability),
                "correct_count": correct_count,
                "k": generations_value,
            }
        )
    if not rows:
        raise ValueError(f"{path}: expected at least one held-out prediction row")
    if len(generations) != 1:
        raise ValueError(f"{path}: classifier figure requires one generation count across prediction rows")
    return rows, generations.pop()


def classifier_operating_curve(
    rows: list[dict[str, float | int]],
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Compute every operating point and the two paper callouts from raw predictions."""

    thresholds = sorted({0.0, 0.5, 0.7, 1.0, *(float(row["probability"]) for row in rows)})
    points: list[dict[str, float]] = []
    highlighted: list[dict[str, float]] = []
    for threshold in thresholds:
        negative = positive = false_positive = false_negative = selected = 0
        for row in rows:
            truth = 2 * int(row["correct_count"]) >= int(row["k"])
            forecast = float(row["probability"]) >= threshold
            if truth:
                positive += 1
                false_negative += not forecast
            else:
                negative += 1
                false_positive += forecast
            selected += forecast
        if not negative or not positive:
            raise ValueError("classifier figure requires both observed binary outcome classes")
        point = {
            "threshold": threshold,
            "fpr": false_positive / negative,
            "fnr": false_negative / positive,
            "local_coverage": selected / len(rows),
        }
        points.append(point)
        if threshold in {0.5, 0.7}:
            highlighted.append(point)
    return points, highlighted


def calibration_bins(
    rows: list[dict[str, float | int]],
) -> tuple[list[dict[str, float | int | None]], float]:
    """Compute the paper's equal-width reliability bins directly from prediction rows."""

    grouped: list[list[tuple[float, float]]] = [[] for _ in range(10)]
    for row in rows:
        probability = float(row["probability"])
        grouped[min(int(probability * 10), 9)].append(
            (probability, int(row["correct_count"]) / int(row["k"]))
        )
    bins: list[dict[str, float | int | None]] = []
    weighted_gap = 0.0
    for index, values in enumerate(grouped):
        if values:
            mean_probability = sum(item[0] for item in values) / len(values)
            mean_observed_rate = sum(item[1] for item in values) / len(values)
            gap = abs(mean_probability - mean_observed_rate)
        else:
            mean_probability = mean_observed_rate = gap = None
        bins.append(
            {
                "lower": index / 10,
                "upper": (index + 1) / 10,
                "count": len(values),
                "mean_probability": mean_probability,
                "mean_observed_rate": mean_observed_rate,
                "absolute_gap": gap,
            }
        )
        if gap is not None:
            weighted_gap += len(values) * gap
    return bins, weighted_gap / len(rows)
