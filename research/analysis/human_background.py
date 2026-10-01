"""Private participant demographics and background regressions for validated study data."""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm

try:
    from research.analysis import human_descriptive as descriptive
    from research.analysis import human_robustness
except ModuleNotFoundError:  # Allow direct execution outside the repository root.
    import human_descriptive as descriptive
    import human_robustness


ValidationError = descriptive.ValidationError
BACKGROUND_FIELDS = ("gender", "age", "education", "occupation", "genai_freq_post")
REGISTERED_PREDICTORS = ("age", "education", "student", "usage")
RAW_FIELD = {
    "age": "age",
    "education": "education",
    "student": "occupation",
    "usage": "genai_freq_post",
    "gender": "gender",
}

EDUCATION_CODES = {
    "high school diploma, currently studying for a state examination degree": 0.0,
    "high school degree": 0.0,
    "bachelor's": 1.0,
    "bachelor degree": 1.0,
    "bachelor": 1.0,
    "bachelor´s": 1.0,
    "bachlors": 1.0,
    "master": 2.0,
    "master's": 2.0,
    "state examination (lehramt an gymnasien, bavaria), m.ed. equivalent": 2.0,
}
CURRENT_STUDY_ONLY = {"3rd year bachelor", "cs bachelor student", "doing my bachelor"}
AMBIGUOUS_SCHOOL = {"school", "secondary education"}
STUDENT_VALUES = {
    "student", "phd-student", "phd student", "student & employee",
    "student and employee", "working student",
}
EMPLOYEE_VALUES = {"employee", "employed"}
USAGE_CODES = {
    "less than weekly": 0.0,
    "1–2 days/week": 1.0,
    "3–4 days/week": 2.0,
    "5–6 days/week": 3.0,
    "daily": 4.0,
}
GENDER_CODES = {"female": 0.0, "male": 1.0}

DESCRIPTIVE_EDUCATION_CATEGORIES = {
    "bachelor's": "bachelor_degree_or_current_bachelor_study",
    "bachelor degree": "bachelor_degree_or_current_bachelor_study",
    "bachelor": "bachelor_degree_or_current_bachelor_study",
    "bachelor´s": "bachelor_degree_or_current_bachelor_study",
    "bachlors": "bachelor_degree_or_current_bachelor_study",
    "3rd year bachelor": "bachelor_degree_or_current_bachelor_study",
    "cs bachelor student": "bachelor_degree_or_current_bachelor_study",
    "doing my bachelor": "bachelor_degree_or_current_bachelor_study",
    "master": "master",
    "master's": "master",
    "high school degree": "high_school",
    "high school diploma, currently studying for a state examination degree":
        "current_state_examination_study",
    "state examination (lehramt an gymnasien, bavaria), m.ed. equivalent":
        "completed_master_equivalent_state_examination",
}
DESCRIPTIVE_EDUCATION_ORDER = (
    "bachelor_degree_or_current_bachelor_study",
    "master",
    "high_school",
    "current_state_examination_study",
    "completed_master_equivalent_state_examination",
)


def _coded(raw: str, value: float | None, status: str, reason: str) -> dict[str, object]:
    return {
        "raw_value": raw,
        "raw_present": bool(raw.strip()),
        "value": value,
        "status": status,
        "reason": reason,
    }


def code_field(field: str, raw_value: str | None) -> dict[str, object]:
    raw = "" if raw_value is None else str(raw_value)
    normalized = raw.strip().lower()
    if not normalized:
        return _coded(raw, None, "missing", "blank_response")
    if field == "age":
        try:
            value = float(normalized)
        except ValueError:
            return _coded(raw, None, "invalid", "age_not_numeric")
        if not math.isfinite(value) or not 0 < value <= 120:
            return _coded(raw, None, "invalid", "age_not_finite_or_out_of_range")
        return _coded(raw, value, "coded", "finite_age_in_years")
    if field == "education":
        if normalized in EDUCATION_CODES:
            return _coded(raw, EDUCATION_CODES[normalized], "coded", "explicit_self_reported_qualification")
        if normalized in CURRENT_STUDY_ONLY:
            return _coded(raw, None, "missing", "current_study_only_no_completed_qualification")
        if normalized in AMBIGUOUS_SCHOOL:
            return _coded(raw, None, "ambiguous", "ambiguous_school_label")
        return _coded(raw, None, "ambiguous", "education_value_not_in_frozen_dictionary")
    if field == "student":
        if normalized in STUDENT_VALUES:
            return _coded(raw, 1.0, "coded", "explicit_student_or_mixed_student_status")
        if normalized in EMPLOYEE_VALUES:
            return _coded(raw, 0.0, "coded", "explicit_employee_only_status")
        return _coded(raw, None, "ambiguous", "occupation_not_explicitly_student_or_employee_only")
    if field == "usage":
        if normalized in USAGE_CODES:
            return _coded(raw, USAGE_CODES[normalized], "coded", "frozen_ordinal_usage_category")
        return _coded(raw, None, "ambiguous", "usage_value_not_in_frozen_dictionary")
    if field == "gender":
        if normalized in GENDER_CODES:
            return _coded(raw, GENDER_CODES[normalized], "coded", "binary_univariate_sensitivity_code")
        return _coded(raw, None, "ambiguous", "gender_value_not_in_frozen_sensitivity_dictionary")
    raise ValidationError(f"unknown background field: {field}")


def _count_record(count: int, denominator: int, denominator_name: str) -> dict[str, object]:
    return {
        "n": count,
        "percent": round(100.0 * count / denominator, 1) if denominator else None,
        "denominator": denominator,
        "denominator_name": denominator_name,
    }


def _categorical_demographics(
    raw_values: list[str],
    categories: dict[str, str],
    category_order: tuple[str, ...],
) -> dict[str, object]:
    normalized = [value.strip().lower() for value in raw_values]
    counts = Counter(categories[value] for value in normalized if value in categories)
    valid_n = sum(counts.values())
    cohort_n = len(raw_values)
    not_reported_n = sum(not value for value in normalized)
    ambiguous_n = cohort_n - valid_n - not_reported_n
    return {
        "valid_n": valid_n,
        "categories": {
            category: _count_record(counts[category], valid_n, "valid responses")
            for category in category_order
        },
        "nonblank_ambiguous": _count_record(ambiguous_n, cohort_n, "cohort"),
        "not_reported": _count_record(not_reported_n, cohort_n, "cohort"),
    }


def participant_demographics(metadata: dict[str, dict[str, str]]) -> dict[str, object]:
    """Aggregate the private response metadata without exporting participant rows."""
    cohort_n = len(metadata)
    raw = {
        field: [participant_metadata[field] for participant_metadata in metadata.values()]
        for field in BACKGROUND_FIELDS
    }
    age_codes = [code_field("age", value) for value in raw["age"]]
    ages = np.asarray(
        [float(coded["value"]) for coded in age_codes if coded["value"] is not None],
        dtype=float,
    )
    invalid_age_n = sum(coded["raw_present"] and coded["value"] is None for coded in age_codes)
    missing_age_n = sum(not coded["raw_present"] for coded in age_codes)
    age_summary = {
        "valid_n": len(ages),
        "mean": round(float(np.mean(ages)), 1) if len(ages) else None,
        "sample_sd": round(float(np.std(ages, ddof=1)), 1) if len(ages) > 1 else None,
        "median": float(np.median(ages)) if len(ages) else None,
        "q1": float(np.quantile(ages, 0.25, method="linear")) if len(ages) else None,
        "q3": float(np.quantile(ages, 0.75, method="linear")) if len(ages) else None,
        "minimum": float(np.min(ages)) if len(ages) else None,
        "maximum": float(np.max(ages)) if len(ages) else None,
        "nonblank_invalid": _count_record(invalid_age_n, cohort_n, "cohort"),
        "not_reported": _count_record(missing_age_n, cohort_n, "cohort"),
    }

    binary_gender = {"female": "female", "male": "male"}
    occupation = {
        **{value: "student_including_employed" for value in STUDENT_VALUES},
        **{value: "employee_only" for value in EMPLOYEE_VALUES},
    }
    usage = {
        value: value.replace("–", "-").replace("/", "_per_").replace(" ", "_")
        for value in USAGE_CODES
    }
    return {
        "cohort_n": cohort_n,
        "rounding": "means, sample SDs, and percentages rounded to one decimal; quartiles use linear interpolation",
        "age_years": age_summary,
        "gender": _categorical_demographics(raw["gender"], binary_gender, ("female", "male")),
        "occupation_status": _categorical_demographics(
            raw["occupation"], occupation, ("student_including_employed", "employee_only")
        ),
        "education_completed_or_current": _categorical_demographics(
            raw["education"], DESCRIPTIVE_EDUCATION_CATEGORIES, DESCRIPTIVE_EDUCATION_ORDER
        ),
        "ai_use_frequency": _categorical_demographics(
            raw["genai_freq_post"], usage, tuple(usage[value] for value in USAGE_CODES)
        ),
    }


def fit_ols_hc3(y: np.ndarray, x: np.ndarray, names: list[str]) -> dict[str, object]:
    if y.ndim != 1 or x.ndim != 2 or len(y) != x.shape[0] or len(names) != x.shape[1]:
        raise ValidationError("regression arrays and coefficient names have incompatible shapes")
    n, parameters = len(y), x.shape[1]
    rank = int(np.linalg.matrix_rank(x)) if n else 0
    base = {"n": n, "parameters": parameters, "design_rank": rank, "df_resid": n - rank}
    if rank < parameters:
        return {**base, "status": "failure", "reason": "rank_deficient_design"}
    if n <= parameters:
        return {**base, "status": "failure", "reason": "insufficient_residual_degrees_of_freedom"}
    fitted = sm.OLS(y, x).fit(cov_type="HC3", use_t=True)
    intervals = fitted.conf_int(alpha=0.05)
    return {
        **base,
        "status": "defined",
        "covariance": "HC3",
        "use_t": bool(fitted.use_t),
        "confidence_distribution": "Student t with residual degrees of freedom",
        "confidence_level": 0.95,
        "r_squared": float(fitted.rsquared),
        "coefficients": {
            name: {
                "estimate": float(fitted.params[index]),
                "standard_error_hc3": float(fitted.bse[index]),
                "t": float(fitted.tvalues[index]),
                "p_two_sided": float(fitted.pvalues[index]),
                "ci95": [float(intervals[index, 0]), float(intervals[index, 1])],
            }
            for index, name in enumerate(names)
        },
    }


def regression_record(rows: list[dict[str, object]], outcome: str, predictors: list[str]) -> dict[str, object]:
    raw_presence = {
        predictor: sum(bool(row[predictor]["raw_present"]) for row in rows)
        for predictor in predictors
    }
    coded_n = {
        predictor: sum(row[predictor]["value"] is not None for row in rows)
        for predictor in predictors
    }
    outcome_n = sum(row.get(outcome) is not None for row in rows)
    complete = [
        row for row in rows
        if row.get(outcome) is not None
        and all(row[predictor]["value"] is not None for predictor in predictors)
    ]
    x = np.asarray(
        [[1.0, *[float(row[predictor]["value"]) for predictor in predictors]] for row in complete],
        dtype=float,
    )
    if not complete:
        x = np.empty((0, len(predictors) + 1), dtype=float)
    y = np.asarray([float(row[outcome]) for row in complete], dtype=float)
    return {
        "outcome": outcome,
        "predictors": predictors,
        "raw_field_presence": raw_presence,
        "coded_predictor_n": coded_n,
        "outcome_present_n": outcome_n,
        "complete_case_n": len(complete),
        "complete_case_participant_ids": [row["participant_id"] for row in complete],
        "fit": fit_ols_hc3(y, x, ["intercept", *predictors]),
    }


def read_stable_metadata(path: Path, expected_participant: str) -> dict[str, str]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not {"participant_id", *BACKGROUND_FIELDS} <= fields:
            raise ValidationError(f"{path.name}: missing required metadata fields")
        rows = list(reader)
    if not rows:
        raise ValidationError(f"{path.name}: empty participant CSV")
    if any(row["participant_id"] != expected_participant for row in rows):
        raise ValidationError(f"{path.name}: participant_id does not match expected ID")
    metadata: dict[str, str] = {}
    for field in BACKGROUND_FIELDS:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise ValidationError(f"{path.name}: {field} metadata changes across rows")
        metadata[field] = values.pop()
    return metadata


def require_exact_participant_join(expected: set[str], observed: set[str]) -> None:
    if expected != observed:
        raise ValidationError(
            f"participant ID join is not exact; missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )


def _load_inputs(
    report: dict[str, object],
    dataset: descriptive.Dataset,
    csv_dir: Path,
) -> tuple[dict[str, dict[str, str]], list[dict[str, object]]]:
    csv_dir = Path(csv_dir)
    study = human_robustness.study_from_report(report)
    expected_ids = set(study.participant_ids)
    metadata = {
        participant: read_stable_metadata(csv_dir / dataset.participant_files[participant], participant)
        for participant in study.participant_ids
    }
    require_exact_participant_join(expected_ids, set(metadata))
    recalculated, _ = human_robustness.per_rater_descriptors(study)
    recalculated_by_id = {row["participant_id"]: row["descriptors"] for row in recalculated}

    rows: list[dict[str, object]] = []
    for participant in study.participant_ids:
        descriptors = recalculated_by_id[participant]
        raw = metadata[participant]
        rows.append({
            "participant_id": participant,
            "kappa": descriptors["linear_weighted_kappa"]["value"],
            "bias": descriptors["bias_levels"]["value"],
            "age": code_field("age", raw["age"]),
            "education": code_field("education", raw["education"]),
            "student": code_field("student", raw["occupation"]),
            "usage": code_field("usage", raw["genai_freq_post"]),
            "gender": code_field("gender", raw["gender"]),
        })
    return metadata, rows


def build_report(
    dataset: descriptive.Dataset,
    descriptive_report: dict[str, object],
    csv_dir: Path,
) -> dict[str, object]:
    metadata, rows = _load_inputs(descriptive_report, dataset, csv_dir)
    predictors = [*REGISTERED_PREDICTORS]
    models = []
    for outcome in ("kappa", "bias"):
        models.extend(regression_record(rows, outcome, [predictor]) for predictor in predictors)
        models.append(regression_record(rows, outcome, predictors))
        models.append(regression_record(rows, outcome, ["gender"]))
    raw_presence = {
        field: sum(bool(metadata[participant][field].strip()) for participant in metadata)
        for field in BACKGROUND_FIELDS
    }
    coded_presence = {
        predictor: sum(row[predictor]["value"] is not None for row in rows)
        for predictor in (*REGISTERED_PREDICTORS, "gender")
    }
    return {
        "schema_version": 4,
        "analysis": "participant demographics and retrospective exploratory background regressions",
        "status": "exploratory analysis; observational and underpowered",
        "configuration": {
            "outcomes": ["per-rater fixed-six linear weighted kappa", "per-rater mean signed bias"],
            "registered_predictors": predictors,
            "models_per_outcome": "four univariate registered predictors, one four-predictor joint model, and gender univariate sensitivity",
            "estimator": "OLS with HC3 heteroskedasticity-robust covariance",
            "intervals": "two-sided 95% Student-t intervals using residual degrees of freedom",
            "missingness": "model-specific complete cases; no imputation",
            "method_status": "coding and executable HC3 specification are retrospective",
        },
        "coding_dictionary": {
            "age": "finite numeric years in (0,120]",
            "education": "explicit completed high-school=0, bachelor=1, master or explicit M.Ed-equivalent state examination=2; current-study-only missing; School/Secondary education ambiguous",
            "student": "explicit student, PhD, working-student or mixed student/employee=1; employee-only=0; unclear occupation missing",
            "usage": "Less than weekly=0, 1-2=1, 3-4=2, 5-6=3, Daily=4; linear trend assumed",
            "gender": "female=0, male=1; separate univariate sensitivity only",
            "qualification_scope": "self-reported qualification, not independently verified completed credentials",
        },
        "provenance": {
            "labels_sha256": dataset.label_hash,
            "response_files": [
                {"path": name, "sha256": dataset.response_hashes[name]}
                for name in sorted(dataset.response_hashes)
            ],
            "runtime": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "statsmodels": statsmodels.__version__,
            },
        },
        "sample": {"participants": len(rows)},
        "field_presence": {
            "raw_nonblank_by_csv_field": raw_presence,
            "coded_by_predictor": coded_presence,
        },
        "participant_demographics": participant_demographics(metadata),
        "regressions": models,
    }
