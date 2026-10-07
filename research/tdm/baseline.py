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

    point = summary(cells, populations)
    draws = stratified_bootstrap(cells, populations, summary, resamples, rng)
    intervals = {k: quantiles([d[k] for d in draws]) for k in point}
    per_cell = {
        f"{stratum}|{'kept' if kept else 'dropped'}": {
            "population": populations[(stratum, kept)],
            "audited": len(v),
            "human_convertible_rate": sum(v) / len(v),
        }
        for (stratum, kept), v in sorted(cells.items())
    }
    pairs = [(int(r["human_majority"]), int(r["baseline_convertible"])) for r in audited]
    return {
        "audited_with_majority": len(audited),
        "audited_without_majority": sum(r["human_votes"] is not None for r in rows) - len(audited),
        "population_weighted": point,
        "bootstrap_95": intervals,
        "cells": per_cell,
        "sample_unweighted": {"scotts_pi": scotts_pi(pairs), **confusion(pairs)},
        "note": "Sample pi and confusion are unweighted and over-represent rejected strata.",
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
    return {
        "calibration_sample": {
            "responses": len(graded),
            "panel": {"scotts_pi": scotts_pi(pairs), **confusion(pairs)},
            "per_judge": per_judge,
            "discordant_share": sum(h != p for h, p in pairs) / len(pairs),
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
