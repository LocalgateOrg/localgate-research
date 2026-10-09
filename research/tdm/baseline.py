"""Score the released MMLU-Pro-Open pipeline against the existing human labels (task B1).

No model is called. The report gives the "baseline against gold" figures the
study compares every candidate with, and the discordance rates the power
analysis (task H4) needs. Audit samples were drawn at different rates per
stratum, so population figures weight each stratum by its size, and confidence
intervals come from a bootstrap that resamples within strata.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from research.analysis.data_io import write_json_new
from research.analysis.judge_calibration import confusion, scotts_pi

BOOTSTRAP_SEED = 20261007


def read_table(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def quantiles(values: list[float], probabilities=(0.025, 0.975)) -> list[float]:
    ordered = sorted(values)
    return [ordered[min(len(ordered) - 1, int(p * len(ordered)))] for p in probabilities]


def weighted_rate(cells: dict[str, list[int]], populations: dict[str, int]) -> float:
    """Population-weighted mean of 0/1 outcomes sampled per cell."""
    total = sum(populations[c] for c in cells)
    return sum(populations[c] * sum(v) / len(v) for c, v in cells.items()) / total


def wilson(successes: int, n: int, z: float = 1.959964) -> list[float]:
    """95% Wilson interval for a proportion (well behaved for small n and 0 or n)."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def pi_from_table(table: dict[tuple[int, int], float]) -> float:
    """Scott's pi from a (possibly weighted) 2x2 table {(rater_a, rater_b): weight}."""
    total = sum(table.values())
    if total == 0:
        return float("nan")
    observed = (table.get((0, 0), 0) + table.get((1, 1), 0)) / total
    p1 = (sum(w for (a, _), w in table.items() if a == 1) + sum(w for (_, b), w in table.items() if b == 1)) / (2 * total)
    expected = p1 * p1 + (1 - p1) * (1 - p1)
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def stratified_bootstrap(cells, populations, statistic, resamples, rng) -> list[float]:
    draws = []
    for _ in range(resamples):
        resampled = {c: [rng.choice(v) for _ in v] for c, v in cells.items()}
        draws.append(statistic(resampled, populations))
    return draws


# ── Filter ───────────────────────────────────────────────────────────────────


def filter_report(rows: list[dict], resamples: int, rng: random.Random) -> dict:
    """Converter accept/reject against the Stage-1 human majority.

    Cells are (stratum, converter decision): the borderline stratum mixes kept
    and dropped items, which have different populations.
    """
    populations = Counter((r["stratum"], r["baseline_convertible"]) for r in rows)
    audited = [r for r in rows if r["human_majority"] is not None]
    cells: dict[tuple, list[int]] = defaultdict(list)
    for r in audited:
        cells[(r["stratum"], r["baseline_convertible"])].append(int(r["human_majority"]))

    def kept_cells(c):
        return {k: v for k, v in c.items() if k[1]}

    def dropped_cells(c):
        return {k: v for k, v in c.items() if not k[1]}

    def summary(c, pops):
        kept = kept_cells(c)
        dropped = dropped_cells(c)
        kept_total = sum(pops[k] for k in kept)
        dropped_total = sum(pops[k] for k in dropped)
        yes_kept = weighted_rate(kept, pops) * kept_total
        yes_dropped = weighted_rate(dropped, pops) * dropped_total
        return {
            "precision": yes_kept / kept_total,
            "share_of_rejections_humans_accept": yes_dropped / dropped_total,
            "recall": yes_kept / (yes_kept + yes_dropped),
            "estimated_convertible_rejections": yes_dropped,
        }

    def weighted_pi(c, pops):
        """Scott's pi between human majority and converter, each cell weighted to its population."""
        table: dict[tuple[int, int], float] = defaultdict(float)
        for (stratum, kept), labels in c.items():
            weight = pops[(stratum, kept)] / len(labels)
            for human in labels:
                table[(human, int(kept))] += weight
        return pi_from_table(table)

    def full(c, pops):
        return {**summary(c, pops), "scotts_pi": weighted_pi(c, pops),
                "error_rate": 1 - (summary(c, pops)["precision"] * sum(pops[k] for k in kept_cells(c))
                                   + (1 - summary(c, pops)["share_of_rejections_humans_accept"]) * sum(pops[k] for k in dropped_cells(c)))
                / sum(pops[k] for k in c)}

    point = full(cells, populations)
    draws = stratified_bootstrap(cells, populations, full, resamples, rng)
    intervals = {k: quantiles([d[k] for d in draws]) for k in point}
    per_cell = {}
    for (stratum, kept), v in sorted(cells.items()):
        agree = sum(v) if kept else len(v) - sum(v)  # converter and human majority say the same
        per_cell[f"{stratum}|{'kept' if kept else 'dropped'}"] = {
            "population": populations[(stratum, kept)],
            "audited": len(v),
            "human_convertible_rate": sum(v) / len(v),
            "agreement_with_converter": agree / len(v),
            "agreement_wilson_95": wilson(agree, len(v)),
        }
    pairs = [(int(r["human_majority"]), int(r["baseline_convertible"])) for r in audited]
    return {
        "audited_with_majority": len(audited),
        "audited_without_majority": sum(r["human_votes"] is not None for r in rows) - len(audited),
        "population_weighted": point,
        "bootstrap_95": intervals,
        "cells": per_cell,
        "sample_unweighted": {"scotts_pi": scotts_pi(pairs), **confusion(pairs)},
        "note": "population_weighted (with bootstrap_95) is the estimate for the whole conversion; "
        "sample_unweighted over-represents rejected strata. error_rate = population share where converter and human "
        "majority disagree. Per stratum, the converter's decision is constant except in the borderline stratum, so "
        "per-cell agreement replaces per-stratum precision/recall.",
    }


# ── Rewrites ─────────────────────────────────────────────────────────────────


def majority_yes(votes: list[str], reverse: set[int]) -> bool | None:
    """Majority of yes/no votes; 'borderline' counts as abstention."""
    flipped = []
    for index, vote in enumerate(votes):
        if index in reverse and vote in {"yes", "no"}:
            vote = "no" if vote == "yes" else "yes"
        flipped.append(vote)
    yes, no = flipped.count("yes"), flipped.count("no")
    if yes == no:
        return None
    return yes > no


def rewrite_report(rows: list[dict], reverse: set[int], resamples: int, rng: random.Random) -> dict:
    populations = Counter(r["stratum"] for r in rows)
    audited = [r for r in rows if r["human_same_question"] is not None]
    report = {"audited": len(audited), "reversed_self_contained_raters": sorted(reverse)}
    for field, rev in (("human_same_question", set()), ("human_self_contained", reverse)):
        cells: dict[str, list[int]] = defaultdict(list)
        ties = 0
        for r in audited:
            verdict = majority_yes(r[field], rev)
            if verdict is None:
                ties += 1
                continue
            cells[r["stratum"]].append(int(not verdict))  # 1 = failure
        point = weighted_rate(cells, populations)
        draws = stratified_bootstrap(cells, populations, weighted_rate, resamples, rng)
        report[field.removeprefix("human_")] = {
            "weighted_failure_rate": point,
            "bootstrap_95": quantiles(draws),
            "failures_by_stratum": {s: [sum(v), len(v)] for s, v in sorted(cells.items())},
            "ties_excluded": ties,
        }
    return report


# ── Judging ──────────────────────────────────────────────────────────────────


def judging_report(rows: list[dict]) -> dict:
    """Panel against the 100 human grades, and panel disagreement at scale."""
    graded = [r for r in rows if r["human_match"] is not None]
    pairs = [(int(r["human_match"]), int(r["baseline_match"])) for r in graded]
    per_judge = {}
    for judge in sorted(graded[0]["judge_verdicts"]):
        judge_pairs = [(int(r["human_match"]), int(r["judge_verdicts"][judge] == "match")) for r in graded]
        per_judge[judge] = {"scotts_pi": scotts_pi(judge_pairs), **confusion(judge_pairs)}

    def split_rate(group):
        return {k: {"responses": len(v), "panel_split_rate": 1 - sum(v) / len(v)} for k, v in sorted(group.items())}

    by_type, by_category, by_length = defaultdict(list), defaultdict(list), defaultdict(list)
    lengths = sorted(r["judge_tokens_in_median"] for r in rows)
    cuts = [lengths[int(q * len(lengths))] for q in (0.25, 0.5, 0.75)]
    for r in rows:
        unanimous = int(r["panel_unanimous_binary"])
        by_type[r["reference_type"]].append(unanimous)
        by_category[r["category"]].append(unanimous)
        quartile = sum(r["judge_tokens_in_median"] > c for c in cuts) + 1
        by_length[f"Q{quartile}"].append(unanimous)
    graded_lengths = sorted(r["output_tokens"] for r in graded)
    graded_cuts = [graded_lengths[int(q * len(graded_lengths))] for q in (1 / 3, 2 / 3)]

    def vs_humans(key):
        groups = defaultdict(list)
        for r in graded:
            groups[key(r)].append((int(r["human_match"]), int(r["baseline_match"])))
        out = {}
        for name, g in sorted(groups.items()):
            agree = sum(h == p for h, p in g)
            both_classes = len({h for h, _ in g} | {p for _, p in g}) == 2
            out[name] = {"responses": len(g), "accuracy": agree / len(g), "accuracy_wilson_95": wilson(agree, len(g)),
                         "scotts_pi": scotts_pi(g) if both_classes else None, **confusion(g)}
        return out

    def length_band(r):
        band = sum(r["output_tokens"] > c for c in graded_cuts)
        return ["short", "medium", "long"][band]

    disagree = sum(h != p for h, p in pairs)
    return {
        "calibration_sample": {
            "responses": len(graded),
            "panel": {"scotts_pi": scotts_pi(pairs), **confusion(pairs)},
            "per_judge": per_judge,
            "discordant_share": disagree / len(pairs),
            "discordant_wilson_95": wilson(disagree, len(pairs)),
            "panel_vs_humans_by_reference_type": vs_humans(lambda r: r["reference_type"]),
            "panel_vs_humans_by_category": vs_humans(lambda r: r["category"]),
            "panel_vs_humans_by_response_length": vs_humans(length_band),
            "response_length_tercile_cuts_tokens": graded_cuts,
            "note": "Groups are small (100 responses in all); read them as where errors concentrate, not as estimates.",
        },
        "panel_disagreement": {
            "responses": len(rows),
            "overall_split_rate": 1 - sum(r["panel_unanimous_binary"] for r in rows) / len(rows),
            "by_reference_type": split_rate(by_type),
            "by_category": split_rate(by_category),
            "by_judge_input_tokens_quartile": split_rate(by_length),
            "quartile_cuts_tokens": cuts,
        },
    }


def power_inputs(report: dict) -> dict:
    """Baseline error rate against gold per stage: what H4's power analysis starts from.

    The paired non-inferiority test needs the share of items where candidate and
    baseline disagree. Before any candidate exists, the baseline's own error rate
    against gold bounds it: if the candidate is about as accurate, discordance
    is roughly 2e(1-e) with independent errors and at most 2e with no overlap.
    """
    f, r, j = report["filter"], report["rewrite"], report["judging"]["calibration_sample"]
    stages = {
        "filter": {"error_rate": f["population_weighted"]["error_rate"], "ci_95": f["bootstrap_95"]["error_rate"],
                   "basis": f"{f['audited_with_majority']} Stage-1 audits, population-weighted"},
        "rewrite_meaning": {"error_rate": r["same_question"]["weighted_failure_rate"],
                            "ci_95": r["same_question"]["bootstrap_95"], "basis": f"{r['audited']} Stage-2 audits"},
        "rewrite_self_contained": {"error_rate": r["self_contained"]["weighted_failure_rate"],
                                   "ci_95": r["self_contained"]["bootstrap_95"],
                                   "basis": f"{r['audited']} Stage-2 audits (criterion under review, D2)"},
        "grading": {"error_rate": j["discordant_share"], "ci_95": j["discordant_wilson_95"],
                    "basis": f"{j['responses']} human-graded responses"},
    }
    for s in stages.values():
        e = s["error_rate"]
        s["discordance_if_independent"] = 2 * e * (1 - e)
        s["discordance_upper"] = min(1.0, 2 * e)
    return stages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True, help="output directory of `localgate-tdm items`")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument(
        "--reverse-self-contained",
        type=int,
        action="append",
        default=[],
        metavar="RATER",
        help="1-based rater whose self-containedness yes/no is reversed (the paper corrects one rater)",
    )
    args = parser.parse_args(argv)
    rng = random.Random(BOOTSTRAP_SEED)
    report = {
        "filter": filter_report(read_table(args.items / "filter.jsonl"), args.resamples, rng),
        "rewrite": rewrite_report(
            read_table(args.items / "rewrite.jsonl"),
            {r - 1 for r in args.reverse_self_contained},
            args.resamples,
            rng,
        ),
        "judging": judging_report(read_table(args.items / "judging.jsonl")),
        "bootstrap": {"resamples": args.resamples, "seed": BOOTSTRAP_SEED},
    }
    report["power_inputs"] = power_inputs(report)
    write_json_new(args.out, report)
    f = report["filter"]["population_weighted"]
    r = report["rewrite"]
    print(
        f"filter precision {f['precision']:.3f}, rejections humans accept "
        f"{f['share_of_rejections_humans_accept']:.3f}, recall {f['recall']:.3f}\n"
        f"rewrite meaning change {r['same_question']['weighted_failure_rate']:.4f}, "
        f"not self-contained {r['self_contained']['weighted_failure_rate']:.4f}\n"
        f"panel pi {report['judging']['calibration_sample']['panel']['scotts_pi']:.4f}"
    )
    return 0
