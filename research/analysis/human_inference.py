"""Estimate agreement uncertainty and directional routing errors."""

from __future__ import annotations

import math
import sys
from fractions import Fraction

import numpy as np
import scipy
from scipy import stats

try:
    from research.analysis import human_descriptive as descriptive
except ModuleNotFoundError:  # Allow direct execution outside the repository root.
    import human_descriptive as descriptive


N_RESAMPLES = 10_000
SEED = 20_260_921
CONFIDENCE_LEVEL = 0.95
ADEQUACY_FLOOR = 0.5


def _upper_binomial_tail(successes: int, trials: int) -> Fraction:
    if type(successes) is not int or type(trials) is not int or not 0 <= successes <= trials:
        raise descriptive.ValidationError("binomial successes/trials must be exact integers with 0 <= successes <= trials")
    return Fraction(sum(math.comb(trials, value) for value in range(successes, trials + 1)), 2**trials)


def _p_value(value: Fraction) -> dict[str, object]:
    return {"p_value": float(value), "p_value_exact": f"{value.numerator}/{value.denominator}"}


def directional_mcnemar(false_positive: int, false_negative: int) -> dict[str, object]:
    if type(false_positive) is not int or type(false_negative) is not int:
        raise descriptive.ValidationError("McNemar cell counts must be exact integers")
    if false_positive < 0 or false_negative < 0:
        raise descriptive.ValidationError("McNemar cell counts must be nonnegative")
    discordant = false_positive + false_negative
    return {
        "discordant_pairs": discordant,
        "human_local_truth_cloud": false_positive,
        "human_cloud_truth_local": false_negative,
        "h3_excess_local": {
            "alternative": "human_local_truth_cloud > human_cloud_truth_local",
            **_p_value(_upper_binomial_tail(false_positive, discordant)),
        },
        "h2_excess_cloud": {
            "alternative": "human_cloud_truth_local > human_local_truth_cloud",
            **_p_value(_upper_binomial_tail(false_negative, discordant)),
        },
    }


def kendall_tau_b(truth: list[int] | tuple[int, ...], human: list[int] | tuple[int, ...]) -> dict[str, object]:
    if not truth or len(truth) != len(human):
        raise descriptive.ValidationError("Kendall inputs must be non-empty and paired")
    if any(type(value) is not int or value not in range(6) for value in list(truth) + list(human)):
        raise descriptive.ValidationError("Kendall inputs must use exact integers on the fixed six-level scale")
    concordant = discordant = truth_ties = human_ties = joint_ties = 0
    for left in range(len(truth) - 1):
        for right in range(left + 1, len(truth)):
            truth_delta = truth[left] - truth[right]
            human_delta = human[left] - human[right]
            if truth_delta == 0 and human_delta == 0:
                joint_ties += 1
            elif truth_delta == 0:
                truth_ties += 1
            elif human_delta == 0:
                human_ties += 1
            elif truth_delta * human_delta > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + truth_ties)
        * (concordant + discordant + human_ties)
    )
    result: dict[str, object] = {
        "n": len(truth),
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "truth_only_ties": truth_ties,
        "human_only_ties": human_ties,
        "joint_ties": joint_ties,
        "value": None if denominator == 0 else (concordant - discordant) / denominator,
    }
    if denominator == 0:
        result["undefined_reason"] = "zero Kendall tau-b denominator"
    return result


def _kappa_statistic(truth: np.ndarray, human: np.ndarray) -> float:
    value = descriptive.linear_weighted_kappa(
        [int(item) for item in truth.tolist()],
        [int(item) for item in human.tolist()],
    )
    return math.nan if value is None else float(value)


def bootstrap_kappa(
    truth: list[int] | tuple[int, ...],
    human: list[int] | tuple[int, ...],
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    adequacy_floor: float = ADEQUACY_FLOOR,
) -> dict[str, object]:
    point = descriptive.linear_weighted_kappa(truth, human)
    if point is None:
        return {
            "point_estimate": None,
            "point_estimate_exact": None,
            "two_sided_95": None,
            "one_sided_95_upper": None,
            "standard_error": None,
            "adequacy_floor": adequacy_floor,
            "upper_bound_below_floor": None,
            "undefined_reason": "point kappa has zero expected disagreement",
        }
    arrays = (np.asarray(truth, dtype=np.int64), np.asarray(human, dtype=np.int64))
    common = {
        "statistic": _kappa_statistic,
        "n_resamples": n_resamples,
        "paired": True,
        "vectorized": False,
        "confidence_level": CONFIDENCE_LEVEL,
        "method": "BCa",
    }
    two_sided = stats.bootstrap(
        arrays, alternative="two-sided", rng=np.random.default_rng(seed), **common
    )
    upper = stats.bootstrap(
        arrays, alternative="less", rng=np.random.default_rng(seed), **common
    )
    lower_value = float(two_sided.confidence_interval.low)
    upper_value = float(two_sided.confidence_interval.high)
    one_sided_upper = float(upper.confidence_interval.high)
    standard_error = float(two_sided.standard_error)
    values = (lower_value, upper_value, one_sided_upper, standard_error)
    if not all(math.isfinite(value) for value in values):
        return {
            "point_estimate": float(point),
            "point_estimate_exact": f"{point.numerator}/{point.denominator}",
            "two_sided_95": None,
            "one_sided_95_upper": None,
            "standard_error": None,
            "adequacy_floor": adequacy_floor,
            "upper_bound_below_floor": None,
            "undefined_reason": "SciPy BCa returned a non-finite interval or standard error",
        }
    return {
        "point_estimate": float(point),
        "point_estimate_exact": f"{point.numerator}/{point.denominator}",
        "two_sided_95": {"lower": lower_value, "upper": upper_value},
        "one_sided_95_upper": one_sided_upper,
        "standard_error": standard_error,
        "adequacy_floor": adequacy_floor,
        "upper_bound_below_floor": one_sided_upper < adequacy_floor,
    }


def build_report(pairs: descriptive.PromptPairs) -> dict[str, object]:
    fp = sum(not truth and human for truth, human in zip(pairs.truth_local, pairs.human_local))
    fn = sum(truth and not human for truth, human in zip(pairs.truth_local, pairs.human_local))
    return {
        "schema_version": 1,
        "analysis": "retrospective bounded inference for validated human prompt aggregates",
        "configuration": {
            "unit": "prompt",
            "pairing": "truth and human values resampled with the same prompt indices",
            "kappa_categories": [0, 1, 2, 3, 4, 5],
            "bootstrap_method": "SciPy BCa",
            "bootstrap_resamples": N_RESAMPLES,
            "bootstrap_seed": SEED,
            "two_sided_confidence_level": CONFIDENCE_LEVEL,
            "one_sided_upper_confidence_level": CONFIDENCE_LEVEL,
            "adequacy_floor": ADEQUACY_FLOOR,
            "ci_choice_status": "retrospective; the registered threshold did not specify a CI algorithm",
            "bootstrap_tail_p_value": "not computed",
        },
        "provenance": {
            "runtime": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
        },
        "validation": {"prompts": len(pairs.prompt_ids), "unique_prompt_ids": len(set(pairs.prompt_ids))},
        "directional_mcnemar_exact": directional_mcnemar(fp, fn),
        "kendall_tau_b_descriptive": kendall_tau_b(pairs.truth, pairs.human),
        "human_linear_weighted_kappa_bca": bootstrap_kappa(pairs.truth, pairs.human),
    }
