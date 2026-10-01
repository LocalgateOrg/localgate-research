"""Retrospective rater-agreement and removal robustness for the human study."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

try:
    from research.analysis import human_descriptive as descriptive
except ModuleNotFoundError:  # Allow direct execution outside the repository root.
    import human_descriptive as descriptive


KRIPPENDORFF_PAPER = "https://repository.upenn.edu/server/api/core/bitstreams/0421f871-f005-4322-b06a-a66bec328e3b/content"
REFERENCE_IMPLEMENTATION = "https://github.com/pln-fing-udelar/fast-krippendorff"
ROBUST_SD_FACTOR = Fraction(7413, 5000)  # 1.4826


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    truth_level: int
    truth_local: bool
    ratings: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Study:
    prompts: tuple[Prompt, ...]
    participant_ids: tuple[str, ...]


def _fraction(value: Fraction | None) -> dict[str, object]:
    if value is None:
        return {"value": None, "exact": None}
    return {"value": float(value), "exact": f"{value.numerator}/{value.denominator}"}


def ordinal_krippendorff_alpha(units: Iterable[Iterable[int]]) -> dict[str, object]:
    coincidence = [[Fraction(0) for _ in range(6)] for _ in range(6)]
    pairable = unpairable = 0
    for raw_unit in units:
        unit = list(raw_unit)
        if any(type(value) is not int or value not in range(6) for value in unit):
            raise descriptive.ValidationError("alpha ratings must be exact integers on the fixed six-level scale")
        if len(unit) < 2:
            unpairable += 1
            continue
        pairable += 1
        counts = Counter(unit)
        denominator = len(unit) - 1
        for left in range(6):
            for right in range(6):
                ordered_pairs = counts[left] * counts[right]
                if left == right:
                    ordered_pairs -= counts[left]
                coincidence[left][right] += Fraction(ordered_pairs, denominator)
    marginals = [sum(row) for row in coincidence]
    total = sum(marginals)
    base = {
        "pairable_units": pairable,
        "unpairable_units": unpairable,
        "coincidence_total": _fraction(total),
        "coincidence_marginals": [_fraction(value) for value in marginals],
        "coincidence_matrix": [[_fraction(value) for value in row] for row in coincidence],
    }
    if total <= 1:
        return {**base, "alpha": None, "alpha_exact": None, "undefined_reason": "fewer than two pairable values"}

    distance = [[Fraction(0) for _ in range(6)] for _ in range(6)]
    for left in range(6):
        for right in range(6):
            low, high = sorted((left, right))
            midpoint_distance = sum(marginals[low: high + 1]) - Fraction(marginals[low] + marginals[high], 2)
            distance[left][right] = midpoint_distance * midpoint_distance
    observed = sum(coincidence[a][b] * distance[a][b] for a in range(6) for b in range(6))
    expected = sum(
        (marginals[a] * marginals[b] - (marginals[a] if a == b else 0))
        / (total - 1)
        * distance[a][b]
        for a in range(6) for b in range(6)
    )
    if expected == 0:
        return {
            **base,
            "alpha": None,
            "alpha_exact": None,
            "observed_weighted_disagreement": _fraction(observed),
            "expected_weighted_disagreement": _fraction(expected),
            "undefined_reason": "zero expected ordinal disagreement",
        }
    alpha = 1 - observed / expected
    return {
        **base,
        "alpha": float(alpha),
        "alpha_exact": f"{alpha.numerator}/{alpha.denominator}",
        "observed_weighted_disagreement": _fraction(observed),
        "expected_weighted_disagreement": _fraction(expected),
    }


def linear_quantile(values: Iterable[Fraction], probability: Fraction) -> Fraction:
    ordered = sorted(values)
    if not ordered or not 0 <= probability <= 1:
        raise descriptive.ValidationError("quantile needs values and a probability in [0,1]")
    position = (len(ordered) - 1) * probability
    lower, remainder = divmod(position.numerator, position.denominator)
    if remainder == 0:
        return ordered[lower]
    fraction = Fraction(remainder, position.denominator)
    return ordered[lower] + fraction * (ordered[lower + 1] - ordered[lower])


def iqr_fences(values: dict[str, Fraction | None]) -> dict[str, object]:
    undefined = sorted(identifier for identifier, value in values.items() if value is None)
    if undefined or len(values) < 4:
        return {"status": "undefined_kappa_or_too_few_raters", "undefined_rater_ids": undefined, "outlier_ids": None}
    defined = {identifier: value for identifier, value in values.items() if value is not None}
    q1 = linear_quantile(defined.values(), Fraction(1, 4))
    q3 = linear_quantile(defined.values(), Fraction(3, 4))
    spread = q3 - q1
    lower, upper = q1 - Fraction(3, 2) * spread, q3 + Fraction(3, 2) * spread
    outliers = sorted(identifier for identifier, value in defined.items() if value < lower or value > upper)
    return {
        "status": "defined",
        "quantile_method": "linear; h=(n-1)p",
        "strict_outside": True,
        "q1": float(q1), "q1_exact": f"{q1.numerator}/{q1.denominator}",
        "q3": float(q3), "q3_exact": f"{q3.numerator}/{q3.denominator}",
        "iqr": float(spread), "iqr_exact": f"{spread.numerator}/{spread.denominator}",
        "lower_fence": float(lower), "upper_fence": float(upper),
        "outlier_ids": outliers,
    }


def mad_fences(values: dict[str, Fraction | None]) -> dict[str, object]:
    undefined = sorted(identifier for identifier, value in values.items() if value is None)
    if undefined or not values:
        return {"status": "undefined_kappa", "undefined_rater_ids": undefined, "outlier_ids": None}
    defined = {identifier: value for identifier, value in values.items() if value is not None}
    center = linear_quantile(defined.values(), Fraction(1, 2))
    mad = linear_quantile((abs(value - center) for value in defined.values()), Fraction(1, 2))
    if mad == 0:
        return {
            "status": "undefined_zero_mad",
            "median": float(center), "median_exact": f"{center.numerator}/{center.denominator}",
            "mad": 0.0, "mad_exact": "0/1", "outlier_ids": None,
        }
    robust_sd = ROBUST_SD_FACTOR * mad
    lower, upper = center - 2 * robust_sd, center + 2 * robust_sd
    outliers = sorted(identifier for identifier, value in defined.items() if value < lower or value > upper)
    return {
        "status": "defined",
        "strict_outside": True,
        "median": float(center), "median_exact": f"{center.numerator}/{center.denominator}",
        "mad": float(mad), "mad_exact": f"{mad.numerator}/{mad.denominator}",
        "robust_sd_factor": 1.4826,
        "robust_sd": float(robust_sd),
        "lower_fence": float(lower), "upper_fence": float(upper),
        "outlier_ids": outliers,
    }


def _empty_branch() -> dict[str, object]:
    return {"n": 0, "linear_weighted_kappa": {"value": None, "exact": None, "undefined_reason": "no retained prompts"}, "undefined_reason": "no retained prompts"}


def aggregate_after_removal(study: Study, removed_ids: set[str]) -> dict[str, object]:
    unknown = removed_ids - set(study.participant_ids)
    if unknown:
        raise descriptive.ValidationError(f"unknown removed rater IDs: {sorted(unknown)}")
    truth: list[int] = []
    truth_local: list[bool] = []
    lower: list[int] = []
    upper: list[int] = []
    empty_ids: list[str] = []
    available = Counter()
    for prompt in study.prompts:
        ratings = sorted(level for participant, level in prompt.ratings if participant not in removed_ids)
        available[len(ratings)] += 1
        if not ratings:
            empty_ids.append(prompt.prompt_id)
            continue
        truth.append(prompt.truth_level)
        truth_local.append(prompt.truth_local)
        lower.append(ratings[(len(ratings) - 1) // 2])
        upper.append(ratings[len(ratings) // 2])
    lower_result = descriptive.describe_pairs(truth, lower, truth_local, [value >= 3 for value in lower]) if truth else _empty_branch()
    upper_result = descriptive.describe_pairs(truth, upper, truth_local, [value >= 3 for value in upper]) if truth else _empty_branch()
    return {
        "removed_rater_ids": sorted(removed_ids),
        "total_prompts": len(study.prompts),
        "retained_prompts": len(truth),
        "coverage": len(truth) / len(study.prompts) if study.prompts else None,
        "excluded_empty_prompt_ids": sorted(empty_ids),
        "available_rating_counts": {str(count): available[count] for count in sorted(available)},
        "two_rating_policy": "both lower and upper medians reported",
        "lower_median": lower_result,
        "upper_median": upper_result,
    }


def study_from_report(report: object) -> Study:
    pairs = descriptive.prompt_pairs(report)
    if not isinstance(report, dict):
        raise descriptive.ValidationError("descriptive report must be an object")
    prompts: list[Prompt] = []
    participant_counts = Counter()
    participants: set[str] = set()
    for row, expected_id in zip(report["per_prompt"], pairs.prompt_ids):
        if row["prompt_id"] != expected_id:
            raise descriptive.ValidationError("prompt ordering changed during study validation")
        raw_ratings = row.get("ratings")
        if not isinstance(raw_ratings, list) or len(raw_ratings) != 3:
            raise descriptive.ValidationError(f"prompt {expected_id}: expected exactly three source ratings")
        ratings: list[tuple[str, int]] = []
        for raw in raw_ratings:
            participant, level = raw.get("participant_id"), raw.get("rating_level")
            if not isinstance(participant, str) or type(level) is not int or level not in range(6):
                raise descriptive.ValidationError(f"prompt {expected_id}: invalid source rating")
            ratings.append((participant, level))
            participants.add(participant)
            participant_counts[participant] += 1
        if len({participant for participant, _ in ratings}) != 3:
            raise descriptive.ValidationError(f"prompt {expected_id}: duplicate source rater")
        levels = sorted(level for _, level in ratings)
        if levels[1] != row["human_median_level"]:
            raise descriptive.ValidationError(f"prompt {expected_id}: source ratings do not reproduce median")
        prompts.append(Prompt(expected_id, row["truth_level"], row["truth_local"], tuple(sorted(ratings))))
    expected_participants = {f"P{number:02d}" for number in range(1, 31)}
    if participants != expected_participants or any(participant_counts[p] != 28 for p in expected_participants):
        raise descriptive.ValidationError("source ratings do not form the complete 30x28 rater assignment")
    return Study(tuple(prompts), tuple(sorted(participants)))


def per_rater_descriptors(study: Study) -> tuple[list[dict[str, object]], dict[str, Fraction | None]]:
    ratings: dict[str, list[tuple[int, bool, int]]] = defaultdict(list)
    for prompt in study.prompts:
        for participant, level in prompt.ratings:
            ratings[participant].append((prompt.truth_level, prompt.truth_local, level))
    output, kappas = [], {}
    for participant in study.participant_ids:
        rows = ratings[participant]
        result = descriptive.describe_pairs(
            [row[0] for row in rows], [row[2] for row in rows],
            [row[1] for row in rows], [row[2] >= 3 for row in rows],
        )
        kappa = descriptive.linear_weighted_kappa([row[0] for row in rows], [row[2] for row in rows])
        kappas[participant] = kappa
        output.append({"participant_id": participant, "descriptors": result})
    return output, kappas


def _branch_range(results: list[dict[str, object]], branch: str) -> dict[str, object]:
    descriptors = [result[branch] for result in results]
    kappas = [item["linear_weighted_kappa"]["value"] for item in descriptors]
    defined = [value for value in kappas if value is not None]
    binaries = [item["binary"] for item in descriptors]
    return {
        "defined_kappas": len(defined),
        "kappa_range": {"min": min(defined), "max": max(defined)} if defined else None,
        "false_positive_range": {"min": min(item["confusion"]["fp"] for item in binaries), "max": max(item["confusion"]["fp"] for item in binaries)},
        "false_negative_range": {"min": min(item["confusion"]["fn"] for item in binaries), "max": max(item["confusion"]["fn"] for item in binaries)},
        "net_local_count_range": {"min": min(item["net_local_count"] for item in binaries), "max": max(item["net_local_count"] for item in binaries)},
    }


def _trimmed(rule: dict[str, object], study: Study) -> dict[str, object] | None:
    outliers = rule.get("outlier_ids")
    return None if outliers is None else aggregate_after_removal(study, set(outliers))


def build_report(study: Study) -> dict[str, object]:
    rater_results, kappas = per_rater_descriptors(study)
    iqr, mad = iqr_fences(kappas), mad_fences(kappas)
    leave_one_out = [aggregate_after_removal(study, {participant}) for participant in study.participant_ids]
    return {
        "schema_version": 1,
        "analysis": "retrospective ordinal agreement and rater-removal robustness",
        "configuration": {
            "ordinal_alpha": "coincidence-weighted Krippendorff alpha; ordinal distances use cumulative pooled coincidence marginals",
            "alpha_reference": KRIPPENDORFF_PAPER,
            "independent_implementation_reference": REFERENCE_IMPLEMENTATION,
            "rater_metric": "fixed-six linearly weighted Cohen kappa against each rater's assigned prompt truth",
            "iqr_rule": "linear Q1/Q3; strict outside Q1-1.5*IQR or Q3+1.5*IQR",
            "mad_rule": "strict outside median +/- 2*(1.4826*MAD); zero MAD undefined",
            "even_ratings": "report both lower-median and upper-median branches",
            "empty_prompts": "exclude and report retained coverage",
            "method_status": "retrospective implementation choices; no alpha confidence interval",
        },
        "matrix": {
            "raters": len(study.participant_ids), "prompts": len(study.prompts),
            "observed_cells": sum(len(prompt.ratings) for prompt in study.prompts),
            "possible_cells": len(study.participant_ids) * len(study.prompts),
        },
        "ordinal_krippendorff_alpha": ordinal_krippendorff_alpha([[level for _, level in prompt.ratings] for prompt in study.prompts]),
        "per_rater": rater_results,
        "outlier_rules": {
            "iqr": {**iqr, "trimmed": _trimmed(iqr, study)},
            "mad": {**mad, "trimmed": _trimmed(mad, study)},
        },
        "leave_one_out": leave_one_out,
        "leave_one_out_summary": {
            "coverage_range": {"min": min(item["coverage"] for item in leave_one_out), "max": max(item["coverage"] for item in leave_one_out)},
            "lower_median": _branch_range(leave_one_out, "lower_median"),
            "upper_median": _branch_range(leave_one_out, "upper_median"),
        },
    }
