"""Calculate classifier metrics and optional paired human comparisons."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

from research.analysis import human_descriptive as rh
from research.analysis import input_validation as inputs
from research.analysis.data_io import read_object as read_json
from research.analysis.data_io import read_records as read_jsonl
from research.analysis.data_io import sha256
from research.analysis.input_validation import ValidationError, index_predictions

N_RESAMPLES = 10_000
SEED = 20_260_921


def load_inputs(
    study_path: Path,
    remainder_path: Path,
    data_dir: Path,
) -> tuple[dict[str, list[dict]], dict[str, str]]:
    expected = inputs.prepare_inputs(data_dir)
    study = inputs.load_predictions(study_path, expected.study)
    remainder = inputs.load_predictions(remainder_path, expected.remainder)
    by_id = {row["question_id"]: row for row in study + remainder}
    combined = [
        dict(
            by_id[source["question_id"]],
            correct_count=source["correct_count"],
            k=source["k"],
        )
        for source in expected.combined
    ]
    hashes = {
        "study_predictions": sha256(study_path),
        "heldout_predictions": sha256(remainder_path),
        **{
            name: sha256(Path(data_dir) / name)
            for name in (
                "converted.jsonl",
                "labels_open.jsonl",
                "labels_study_k30.jsonl",
                "study_prompts_corpus.jsonl",
                "split.json",
            )
        },
    }
    return {"study": study, "remainder": remainder, "combined": combined}, hashes


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def checked_probabilities(
    rows: list[dict], probabilities: list[float] | None
) -> list[float]:
    if probabilities is None:
        probabilities = [row["probability"] for row in rows]
    if len(probabilities) != len(rows):
        raise ValidationError("probability vector must have exactly one value per row")
    if any(
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in probabilities
    ):
        raise ValidationError(
            "probability vector must contain finite non-boolean values in [0, 1]"
        )
    return [float(value) for value in probabilities]


def binary_metrics(rows: list[dict], threshold: float) -> dict:
    tn = fp = fn = tp = 0
    for row in rows:
        truth, forecast = (
            2 * row["correct_count"] >= row["k"],
            row["probability"] >= threshold,
        )
        if truth and forecast:
            tp += 1
        elif truth:
            fn += 1
        elif forecast:
            fp += 1
        else:
            tn += 1
    n = len(rows)
    return {
        "threshold": threshold,
        "n": n,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "fpr": _ratio(fp, fp + tn),
        "fnr": _ratio(fn, fn + tp),
        "local_coverage": _ratio(tp + fp, n),
        "precision": _ratio(tp, tp + fp),
        "accuracy": _ratio(tp + tn, n),
    }


def fixed_six_metrics(
    rows: list[dict], probabilities: list[float] | None = None
) -> dict:
    probabilities = checked_probabilities(rows, probabilities)
    truth = [min(6 * row["correct_count"] // row["k"], 5) for row in rows]
    forecast = [min(int(6 * value), 5) for value in probabilities]
    kappa = rh.linear_weighted_kappa(truth, forecast)
    return {
        "linear_weighted_kappa": None if kappa is None else float(kappa),
        "linear_weighted_kappa_exact": None
        if kappa is None
        else f"{kappa.numerator}/{kappa.denominator}",
        "bias_mean_predicted_minus_observed": sum(
            a - b for a, b in zip(forecast, truth)
        )
        / len(rows),
        "mae_levels": sum(abs(a - b) for a, b in zip(forecast, truth)) / len(rows),
    }


def calibration(
    rows: list[dict],
    probabilities: list[float] | None = None,
    *,
    include_native_k5: bool = True,
) -> dict:
    probabilities = checked_probabilities(rows, probabilities)
    bins = [
        {
            "lower": i / 10,
            "upper": (i + 1) / 10,
            "count": 0,
            "mean_probability": None,
            "mean_observed_rate": None,
            "absolute_gap": None,
        }
        for i in range(10)
    ]
    grouped = [[] for _ in range(10)]
    for row, probability in zip(rows, probabilities):
        grouped[min(int(probability * 10), 9)].append(
            (probability, row["correct_count"] / row["k"])
        )
    ece = 0.0
    for item, values in zip(bins, grouped):
        if values:
            mp, mo = (
                sum(x[0] for x in values) / len(values),
                sum(x[1] for x in values) / len(values),
            )
            item.update(
                count=len(values),
                mean_probability=mp,
                mean_observed_rate=mo,
                absolute_gap=abs(mp - mo),
            )
            ece += len(values) / len(rows) * abs(mp - mo)
    observed = [row["correct_count"] / row["k"] for row in rows]

    def cross_entropy(target: float, probability: float) -> float:
        left = (
            0.0
            if target == 0
            else (-target * math.log(probability) if probability > 0 else math.inf)
        )
        right = (
            0.0
            if target == 1
            else (
                -(1 - target) * math.log1p(-probability)
                if probability < 1
                else math.inf
            )
        )
        return left + right

    logloss = [
        cross_entropy(target, probability)
        for target, probability in zip(observed, probabilities)
    ]
    spearman = stats.spearmanr(probabilities, observed).statistic
    native_terms = []
    for row in rows:
        pmf = row.get("native_k5_pmf")
        if (
            include_native_k5
            and row["k"] == 5
            and isinstance(pmf, list)
            and len(pmf) == 6
            and all(type(value) in (int, float) and value >= 0 for value in pmf)
            and math.isclose(sum(pmf), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        ):
            native_terms.append(
                -math.log(pmf[row["correct_count"]])
                if pmf[row["correct_count"]] > 0
                else math.inf
            )
        elif include_native_k5 and row["k"] == 5:
            raise ValidationError("k=5 replay row has no valid native_k5_pmf")
    return {
        "brier_observed_rate": sum(
            (p - target) ** 2 for p, target in zip(probabilities, observed)
        )
        / len(rows),
        "rmse_observed_rate": math.sqrt(
            sum((p - target) ** 2 for p, target in zip(probabilities, observed))
            / len(rows)
        ),
        "per_generation_logloss_observed_rate": None
        if not all(math.isfinite(value) for value in logloss)
        else sum(logloss) / len(rows),
        "spearman_probability_observed_rate": None
        if not math.isfinite(float(spearman))
        else float(spearman),
        "native_k5_count_nll": None
        if len(native_terms) != len(rows)
        or not all(math.isfinite(value) for value in native_terms)
        else sum(native_terms) / len(rows),
        "native_k5_count_nll_note": "only reported when every row is k=5; k=30 study rows use a k=5 PMF and are excluded",
        "ece_10_equal_width_observed_rate": ece,
        "bins": bins,
        "note": "retrospective equal-width bins; observed target is correct_count/k",
    }


def threshold_sweep(rows: list[dict]) -> list[dict]:
    thresholds = sorted(
        {0.0, 0.5, 0.7, 1.0, *(float(row["probability"]) for row in rows)}
    )
    return [binary_metrics(rows, threshold) for threshold in thresholds]


def category_baseline(
    rows: list[dict], development_path: Path, development_manifest: Path
) -> tuple[list[float], dict]:
    manifest = read_json(development_manifest)
    development_path = Path(development_path)
    if manifest.get("counts", {}).get("development") != 6705 or manifest.get(
        "development", {}
    ).get("sha256") != sha256(development_path):
        raise ValidationError(
            "development baseline manifest does not pin the expected 6705-row development file"
        )
    development = read_jsonl(development_path)
    if len(development) != 6705:
        raise ValidationError(
            "category baseline requires exactly 6705 development rows"
        )
    role_ids = manifest.get("role_ids", {})
    development_ids = {row.get("question_id") for row in development}
    allowed_ids = set(role_ids.get("train", [])) | set(role_ids.get("validation", []))
    original_test_ids = set(manifest.get("original_role_ids", {}).get("test", []))
    if (
        len(development_ids) != 6705
        or development_ids != allowed_ids
        or development_ids & original_test_ids
    ):
        raise ValidationError(
            "development baseline IDs are not exactly the frozen train/validation roles"
        )
    if development_ids & {row["question_id"] for row in rows}:
        raise ValidationError("development baseline overlaps the evaluation questions")
    values: dict[str, list[float]] = {}
    for row in development:
        category, count, k = row.get("category"), row.get("correct_count"), row.get("k")
        if (
            not isinstance(category, str)
            or type(count) is not int
            or type(k) is not int
            or not 0 <= count <= k
            or k <= 0
        ):
            raise ValidationError("development row is invalid")
        values.setdefault(category, []).append(count / k)
    means = {category: sum(items) / len(items) for category, items in values.items()}
    try:
        probabilities = [means[row["category"]] for row in rows]
    except KeyError as exc:
        raise ValidationError(
            f"test category missing from development baseline: {exc.args[0]}"
        ) from exc
    return probabilities, {
        "development_n": len(development),
        "development_sha256": sha256(development_path),
        "manifest_sha256": sha256(development_manifest),
        "category_means": means,
        "leakage_guard": "development.jsonl only (frozen train plus validation roles)",
    }


def _difference_statistic(
    truth: np.ndarray, classifier: np.ndarray, human: np.ndarray
) -> float:
    first, second = (
        rh.linear_weighted_kappa(truth.tolist(), classifier.tolist()),
        rh.linear_weighted_kappa(truth.tolist(), human.tolist()),
    )
    return math.nan if first is None or second is None else float(first - second)


def paired_kappa_difference(rows: list[dict], report_path: Path) -> dict:
    pairs = rh.load_prompt_pairs(report_path)
    predicted = index_predictions(rows, "study")
    report_ids = {int(item) for item in pairs.prompt_ids}
    if set(predicted) != report_ids:
        raise ValidationError(
            "human report and classifier study predictions do not have identical prompt IDs"
        )
    for human_row in read_json(report_path)["per_prompt"]:
        question_id = int(human_row["prompt_id"])
        prediction = predicted[question_id]
        if (human_row["correct_count"], human_row["k"]) != (
            prediction["correct_count"],
            prediction["k"],
        ):
            raise ValidationError(
                f"human/classifier reference labels differ for {question_id}"
            )
    ordered = sorted(report_ids)
    truth = np.asarray(
        [predicted[item]["correct_count"] for item in ordered], dtype=np.int64
    )
    k = [predicted[item]["k"] for item in ordered]
    if any(item != 30 for item in k):
        raise ValidationError("gold comparison requires k=30 classifier labels")
    truth_level = np.asarray([min(6 * item // 30, 5) for item in truth], dtype=np.int64)
    human_by_id = dict(zip((int(item) for item in pairs.prompt_ids), pairs.human))
    human = np.asarray([human_by_id[item] for item in ordered], dtype=np.int64)
    classifier = np.asarray(
        [int(predicted[item]["predicted_level"]) for item in ordered], dtype=np.int64
    )
    point = _difference_statistic(truth_level, classifier, human)
    if not math.isfinite(point):
        return {
            "human_report_sha256": sha256(report_path),
            "n": len(ordered),
            "point_classifier_minus_human": None,
            "bca_95": None,
            "standard_error": None,
            "undefined_reason": "classifier or human kappa has zero expected disagreement",
            "configuration": {
                "method": "SciPy BCa",
                "resamples": N_RESAMPLES,
                "seed": SEED,
                "paired_prompt_resampling": True,
            },
        }
    common = {
        "statistic": _difference_statistic,
        "n_resamples": N_RESAMPLES,
        "paired": True,
        "vectorized": False,
        "confidence_level": 0.95,
        "method": "BCa",
    }
    result = stats.bootstrap(
        (truth_level, classifier, human),
        alternative="two-sided",
        rng=np.random.default_rng(SEED),
        **common,
    )
    interval = result.confidence_interval
    values = (
        point,
        float(interval.low),
        float(interval.high),
        float(result.standard_error),
    )
    return {
        "human_report_sha256": sha256(report_path),
        "n": len(ordered),
        "point_classifier_minus_human": None if not math.isfinite(point) else point,
        "bca_95": None
        if not all(math.isfinite(item) for item in values)
        else {"lower": float(interval.low), "upper": float(interval.high)},
        "standard_error": None
        if not math.isfinite(float(result.standard_error))
        else float(result.standard_error),
        "configuration": {
            "method": "SciPy BCa",
            "resamples": N_RESAMPLES,
            "seed": SEED,
            "paired_prompt_resampling": True,
        },
    }


def write_private(
    report: dict, output: Path, filename: str = "classifier_analysis.json"
) -> None:
    output = Path(output)
    if not output.is_absolute() or output.exists() or not output.parent.is_dir():
        raise ValidationError(
            "output must be a new absolute directory below an existing parent"
        )
    if os.stat(output.parent).st_mode & 0o077:
        raise ValidationError(
            "output parent directory must be private (mode 0700 or stricter)"
        )
    output.mkdir(mode=0o700, parents=True)
    destination = output / filename
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-predictions", required=True, type=Path,
        help="classifier predictions JSONL for the study split",
    )
    parser.add_argument(
        "--heldout-predictions", required=True, type=Path,
        help="classifier predictions JSONL for the held-out split",
    )
    parser.add_argument(
        "--data-dir", required=True, type=Path,
        help="prepared corpus, labels, and split artifact directory",
    )
    parser.add_argument(
        "--development", required=True, type=Path,
        help="pinned 6,705-row development JSONL for category baselines",
    )
    parser.add_argument(
        "--development-manifest", required=True, type=Path,
        help="manifest that verifies the development artifact",
    )
    parser.add_argument(
        "--primary-human-report", type=Path,
        help="optional primary-cohort human_descriptive.json for paired comparison",
    )
    parser.add_argument(
        "--latest-human-report", type=Path,
        help="optional human report using the later complete P29 submission",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="new absolute private directory for classifier metrics JSON",
    )
    args = parser.parse_args(argv)
    try:
        groups, input_hashes = load_inputs(
            args.study_predictions,
            args.heldout_predictions,
            args.data_dir,
        )
        _, category_provenance = category_baseline(
            groups["combined"], args.development, args.development_manifest
        )
        paired_comparisons = {
            name: paired_kappa_difference(groups["study"], path)
            for name, path in (
                ("primary", args.primary_human_report),
                ("latest_p29", args.latest_human_report),
            )
            if path is not None
        }
        analyses = {}
        for name, rows in groups.items():
            probs = [
                category_provenance["category_means"][row["category"]] for row in rows
            ]
            cloud_rows = [dict(row, probability=0.0) for row in rows]
            sweep = threshold_sweep(rows)
            analyses[name] = {
                "classifier": {
                    "binary_at_0_5": binary_metrics(rows, 0.5),
                    "fixed_six": fixed_six_metrics(rows),
                    "calibration": calibration(rows),
                    "threshold_sweep": sweep,
                    "operating_constraints": [
                        x
                        for x in sweep
                        if x["fpr"] is not None
                        and x["fnr"] is not None
                        and x["fpr"] <= 0.1
                        and x["fnr"] <= 0.2
                    ],
                },
                "always_cloud": {
                    "binary_at_0_5": binary_metrics(cloud_rows, 0.5),
                    "fixed_six": fixed_six_metrics(rows, [0.0] * len(rows)),
                    "calibration": calibration(
                        rows, [0.0] * len(rows), include_native_k5=False
                    ),
                },
                "category_mean_development_only": {
                    "binary_at_0_5": binary_metrics(
                        [dict(row, probability=p) for row, p in zip(rows, probs)], 0.5
                    ),
                    "fixed_six": fixed_six_metrics(rows, probs),
                    "calibration": calibration(rows, probs, include_native_k5=False),
                    "provenance": category_provenance,
                },
            }
        report = {
            "schema_version": 1,
            "analysis": "classifier prediction analysis",
            "limitations": [
                "ECE bins and operating constraints were selected during analysis."
            ],
            "provenance": {
                "input_files_sha256": input_hashes,
                "runtime": {
                    "python": sys.version.split()[0],
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                },
            },
            "definitions": {
                "truth_local": "2*correct_count >= k",
                "classifier_local": "probability >= threshold",
                "fixed_six": "min(floor(6*x), 5)",
            },
            "views": analyses,
            "gold_paired_kappa_difference": paired_comparisons,
        }
        write_private(report, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(args.output_dir / "classifier_analysis.json")
    return 0
