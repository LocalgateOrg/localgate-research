"""Compare open-trained and multiple-choice-trained routers on both question formats."""

from __future__ import annotations

import argparse
from pathlib import Path

from research.analysis import classifier
from research.analysis import input_validation as inputs
from research.analysis.data_io import read_records, sha256

VIEWS = {
    "open_on_open": "open",
    "open_on_mc": "mc",
    "mc_on_open": "open",
    "mc_on_mc": "mc",
}


def load_inputs(
    data_dir: Path, mc_corpus: Path, predictions: dict[str, Path]
) -> tuple[dict, dict]:
    opened = inputs.prepare_test_inputs(data_dir)
    closed = read_records(mc_corpus)
    indexed = inputs.index_records(closed, "multiple-choice corpus")
    if set(indexed) != {row["question_id"] for row in opened}:
        raise inputs.ValidationError(
            "open and multiple-choice corpora must match on all test question IDs"
        )
    open_index = {row["question_id"]: row for row in opened}
    for row in closed:
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise inputs.ValidationError(
                "multiple-choice corpus requires nonempty text"
            )
        if (
            type(row.get("k")) is not int
            or row["k"] != 5
            or type(row.get("correct_count")) is not int
            or not 0 <= row["correct_count"] <= 5
        ):
            raise inputs.ValidationError(
                "multiple-choice corpus requires five-generation reference counts"
            )
        if (
            not isinstance(row.get("category"), str)
            or type(row.get("list_style")) is not bool
        ):
            raise inputs.ValidationError(
                "multiple-choice corpus requires category and list_style metadata"
            )
        if any(
            row[key] != open_index[row["question_id"]][key]
            for key in ("category", "list_style")
        ):
            raise inputs.ValidationError(
                "multiple-choice and open question metadata differ"
            )
    corpora = {"open": opened, "mc": tuple(closed)}
    views = {
        name: inputs.load_predictions(
            path,
            corpora[VIEWS[name]],
            decoder="open" if name.startswith("open_") else "mc",
        )
        for name, path in predictions.items()
    }
    hashes = {
        "multiple_choice_corpus": sha256(mc_corpus),
        **{name: sha256(path) for name, path in predictions.items()},
        **{
            name: sha256(data_dir / name)
            for name in ("converted.jsonl", "labels_open.jsonl", "split.json")
        },
    }
    return views, hashes


def metrics(rows: list[dict]) -> dict:
    calibration = classifier.calibration(rows, include_native_k5=False)
    return {
        "n": len(rows),
        "ordinal_fixed_six": classifier.fixed_six_metrics(rows),
        "binary_at_0_5": classifier.binary_metrics(rows, 0.5),
        "probability": {
            "per_generation_logloss": calibration[
                "per_generation_logloss_observed_rate"
            ],
            "rmse": calibration["rmse_observed_rate"],
            "brier": calibration["brier_observed_rate"],
            "spearman": calibration["spearman_probability_observed_rate"],
            "ece_10_equal_width_observed_rate": calibration[
                "ece_10_equal_width_observed_rate"
            ],
        },
    }


def comparison(left: dict, right: dict) -> dict:
    """Descriptive left-minus-right contrasts on the same corpus."""

    def delta(path: tuple[str, ...]) -> float | None:
        a, b = left, right
        for key in path:
            a, b = a[key], b[key]
        return None if a is None or b is None else a - b

    return {
        "ordinal_kappa_delta": delta(("ordinal_fixed_six", "linear_weighted_kappa")),
        "binary_accuracy_delta": delta(("binary_at_0_5", "accuracy")),
        "binary_fpr_delta": delta(("binary_at_0_5", "fpr")),
        "binary_fnr_delta": delta(("binary_at_0_5", "fnr")),
        "logloss_delta": delta(("probability", "per_generation_logloss")),
        "rmse_delta": delta(("probability", "rmse")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--mc-corpus", required=True, type=Path)
    for name in VIEWS:
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        views, hashes = load_inputs(
            args.data_dir, args.mc_corpus, {name: getattr(args, name) for name in VIEWS}
        )
        cells = {
            name: {"corpus": VIEWS[name], **metrics(rows)}
            for name, rows in views.items()
        }
        report = {
            "schema_version": 1,
            "analysis": "classifier question-format comparison",
            "limitations": [
                "The models differ in training data, schedules, head parameterization and checkpoint selection; this comparison does not isolate a causal format effect."
            ],
            "provenance": {
                "input_files_sha256": hashes,
            },
            "definitions": {
                "truth_local": "2*correct_count >= 5",
                "classifier_local": "probability >= 0.5",
                "ordinal": "fixed six levels min(floor(6*x),5)",
            },
            "cells": cells,
            "same_corpus_descriptive_contrasts": {
                "open_corpus_open_minus_mc": comparison(
                    cells["open_on_open"], cells["mc_on_open"]
                ),
                "mc_corpus_mc_minus_open": comparison(
                    cells["mc_on_mc"], cells["open_on_mc"]
                ),
            },
        }
        classifier.write_private(report, args.output_dir, "format_comparison.json")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(args.output_dir / "format_comparison.json")
    return 0
