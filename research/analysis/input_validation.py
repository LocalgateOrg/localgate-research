"""Validate classifier inputs and align prediction rows with their reference labels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from research.analysis.data_io import read_object, read_records


class ValidationError(ValueError):
    """Inputs do not describe the same questions and reference labels."""


@dataclass(frozen=True)
class InputGroups:
    study: tuple[dict, ...]
    remainder: tuple[dict, ...]
    combined: tuple[dict, ...]


def index_records(rows: list[dict], name: str) -> dict[int, dict]:
    indexed = {}
    for row in rows:
        question_id = row.get("question_id")
        if type(question_id) is not int or question_id < 0 or question_id in indexed:
            raise ValidationError(
                f"{name} must contain unique integer question_id values"
            )
        indexed[question_id] = row
    return indexed


def _row(text_row: dict, label: dict) -> dict:
    text = text_row.get("open_question")
    if not isinstance(text, str) or not text.strip():
        raise ValidationError("converted prompt is missing open_question")
    if type(label.get("correct_count")) is not int or type(label.get("k")) is not int:
        raise ValidationError("label count and k must be integers")
    if label["k"] <= 0 or not 0 <= label["correct_count"] <= label["k"]:
        raise ValidationError("label count lies outside k")
    if (
        not isinstance(text_row.get("category"), str)
        or not text_row["category"].strip()
        or type(text_row.get("list_style")) is not bool
    ):
        raise ValidationError("prompt requires category and list_style metadata")
    for key in ("category", "list_style"):
        if text_row.get(key) != label.get(key):
            raise ValidationError(
                f"label metadata differs for {text_row['question_id']}"
            )
    return {
        "question_id": text_row["question_id"],
        "text": text,
        "category": text_row["category"],
        "list_style": text_row["list_style"],
        "correct_count": label["correct_count"],
        "k": label["k"],
    }


def prepare_test_inputs(
    data_dir: Path, *, expected_test: int = 1655
) -> tuple[dict, ...]:
    data_dir = Path(data_dir)
    split = read_object(data_dir / "split.json")
    assignment = {int(key): value for key, value in split.get("assignment", {}).items()}
    converted = index_records(
        read_records(data_dir / "converted.jsonl"), "converted.jsonl"
    )
    labels = index_records(
        read_records(data_dir / "labels_open.jsonl"), "labels_open.jsonl"
    )
    test_ids = sorted(
        q
        for q, row in converted.items()
        if row.get("convertible") is True and assignment.get(q) == "test"
    )
    if len(test_ids) != expected_test:
        raise ValidationError(
            f"expected {expected_test} convertible test prompts, found {len(test_ids)}"
        )
    if not set(test_ids) <= set(labels):
        raise ValidationError("open labels do not cover every convertible test prompt")
    rows = tuple(_row(converted[q], labels[q]) for q in test_ids)
    if any(row["k"] != 5 for row in rows):
        raise ValidationError("test reference labels must have k=5")
    return rows


def prepare_inputs(
    data_dir: Path, *, expected_test: int = 1655, expected_study: int = 280
) -> InputGroups:
    data_dir = Path(data_dir)
    combined = prepare_test_inputs(data_dir, expected_test=expected_test)
    test = {row["question_id"]: row for row in combined}
    corpus = index_records(
        read_records(data_dir / "study_prompts_corpus.jsonl"), "study corpus"
    )
    labels = index_records(
        read_records(data_dir / "labels_study_k30.jsonl"), "study labels"
    )
    if set(corpus) != set(labels) or len(labels) != expected_study:
        raise ValidationError(
            "study corpus and k30 labels must have the same expected IDs"
        )
    if not set(labels) <= set(test):
        raise ValidationError(
            "study prompts must be a subset of convertible test prompts"
        )
    study = []
    for question_id in sorted(labels):
        source = corpus[question_id]
        if source.get("split") != "test" or source.get("convertible") is not True:
            raise ValidationError(
                f"study prompt is not a convertible test prompt: {question_id}"
            )
        row = _row(source, labels[question_id])
        if row["k"] != 30:
            raise ValidationError("study reference labels must have k=30")
        if any(
            row[key] != test[question_id][key]
            for key in ("text", "category", "list_style")
        ):
            raise ValidationError(
                f"study prompt differs from converted input for {question_id}"
            )
        study.append(row)
    remainder = tuple(row for row in combined if row["question_id"] not in labels)
    return InputGroups(tuple(study), remainder, combined)


def index_predictions(rows: list[dict], name: str) -> dict[int, dict]:
    indexed = index_records(rows, name)
    for question_id, row in indexed.items():
        probability = row.get("probability")
        if (
            type(probability) not in (int, float)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValidationError(f"{name} has invalid probability for {question_id}")
        count, k = row.get("correct_count"), row.get("k")
        if (
            type(count) is not int
            or type(k) is not int
            or k <= 0
            or not 0 <= count <= k
        ):
            raise ValidationError(f"{name} has invalid label for {question_id}")
        if row.get("predicted_level") != min(int(6 * probability), 5):
            raise ValidationError(
                f"{name} prediction level differs from its probability for {question_id}"
            )
    return indexed


def verify_probability(row: dict, decoder: str) -> None:
    logits = row.get("logits")
    if (
        not isinstance(logits, list)
        or len(logits) != 2
        or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in logits
        )
    ):
        raise ValidationError(f"invalid logits for {row['question_id']}")
    first, second = map(float, logits)
    if decoder == "mc":
        expected = (
            1 / (1 + math.exp(-first))
            if first >= 0
            else math.exp(first) / (1 + math.exp(first))
        )
    elif decoder == "open":

        def softplus(value: float) -> float:
            return math.log1p(math.exp(-abs(value))) + max(value, 0)

        alpha, beta = softplus(first) + 1e-4, softplus(second) + 1e-4
        expected = alpha / (alpha + beta)
    else:
        raise ValidationError(f"unknown probability decoder: {decoder}")
    if not math.isclose(row["probability"], expected, rel_tol=1e-7, abs_tol=1e-7):
        raise ValidationError(
            f"probability/logit decoder mismatch: {row['question_id']}"
        )


def load_predictions(
    path: Path, expected: tuple[dict, ...], *, decoder: str | None = None
) -> list[dict]:
    rows = read_records(path)
    indexed = index_predictions(rows, str(path))
    source = {row["question_id"]: row for row in expected}
    if set(indexed) != set(source):
        raise ValidationError(f"{path}: prediction IDs do not match reference IDs")
    for question_id, prediction in indexed.items():
        reference = source[question_id]
        for key in ("category", "list_style", "correct_count", "k"):
            if prediction.get(key) != reference[key]:
                raise ValidationError(
                    f"{path}: {key} differs from reference for {question_id}"
                )
        if "text" in prediction and prediction["text"] != reference["text"]:
            raise ValidationError(
                f"{path}: text differs from reference for {question_id}"
            )
        if decoder is not None:
            verify_probability(prediction, decoder)
    return rows
