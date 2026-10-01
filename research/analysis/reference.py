"""Estimate label-noise limits under a fitted binomial mixture.

Fit rates on a fixed grid and simulate an oracle that knows each sampled rate."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from research.analysis.data_io import InputError, read_records, write_json
from research.analysis.input_validation import index_records
from research.model_source.localgate_research import metrics as prob_metrics

GRID_SIZE = 201
EM_ITERATIONS = 500
EM_TOLERANCE = 1e-10
SIMULATION_DRAWS = 40
SEED = 20260728


def binomial_likelihood(k: int, grid: np.ndarray) -> np.ndarray:
    """P(c | k, p) for every count c and every rate on the grid, shape (k+1, len(grid))."""
    counts = np.arange(k + 1)[:, None]
    coefficient = np.array([math.comb(k, int(c)) for c in range(k + 1)])[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        likelihood = coefficient * grid**counts * (1 - grid) ** (k - counts)
    # 0**0 is 1 here: a rate of exactly 0 explains a count of 0 perfectly.
    return np.nan_to_num(likelihood, nan=0.0)


def fit_rate_distribution(counts: np.ndarray, k: int, grid_size: int = GRID_SIZE) -> tuple:
    """Fit binomial mixture weights over a fixed rate grid using expectation-maximization."""
    grid = np.linspace(0.0, 1.0, grid_size)
    likelihood = binomial_likelihood(k, grid)
    observed = np.bincount(counts.astype(int), minlength=k + 1).astype(float)

    weights = np.full(grid_size, 1.0 / grid_size)
    for _ in range(EM_ITERATIONS):
        joint = likelihood * weights  # (k+1, grid)
        marginal = joint.sum(axis=1, keepdims=True)
        marginal[marginal == 0] = 1.0
        responsibility = joint / marginal
        updated = (observed[:, None] * responsibility).sum(axis=0)
        updated /= updated.sum()
        if np.abs(updated - weights).max() < EM_TOLERANCE:
            weights = updated
            break
        weights = updated
    return grid, weights


def reliability(grid: np.ndarray, weights: np.ndarray, k: int) -> dict:
    """Estimate true-rate variance relative to observed-rate variance under the mixture."""
    mean_rate = float((weights * grid).sum())
    variance_true = float((weights * (grid - mean_rate) ** 2).sum())
    # Sampling variance of an observed rate, averaged over the fitted rate distribution.
    variance_noise = float((weights * grid * (1 - grid)).sum() / k)
    observed_variance = variance_true + variance_noise
    ratio = variance_true / observed_variance if observed_variance else 0.0
    return {
        "mean_true_rate": mean_rate,
        "variance_true": variance_true,
        "variance_sampling_noise": variance_noise,
        "reliability": ratio,
        "max_attainable_correlation": math.sqrt(max(ratio, 0.0)),
    }


def irreducible_nll(grid: np.ndarray, weights: np.ndarray, k: int) -> float:
    """Expected count-weighted loss when every true rate is known."""
    safe = np.clip(grid, 1e-12, 1 - 1e-12)
    entropy = -(safe * np.log(safe) + (1 - safe) * np.log(1 - safe))
    return float(k * (weights * entropy).sum())


def simulate_oracle(
    grid: np.ndarray,
    weights: np.ndarray,
    k: int,
    n_items: int,
    draws: int = SIMULATION_DRAWS,
    seed: int = SEED,
) -> dict:
    """Score known true rates against repeated draws of binomial counts."""
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {}
    for _ in range(draws):
        rates = rng.choice(grid, size=n_items, p=weights)
        counts = rng.binomial(k, rates).astype(float)
        scored = prob_metrics.evaluate(counts, k, rates)
        for key in ("binomial_nll", "mae", "rmse", "spearman"):
            collected.setdefault(key, []).append(scored[key])
        collected.setdefault("macro_f1", []).append(scored["routing"]["macro_f1"])
        collected.setdefault("fpr", []).append(scored["routing"]["fpr"])
        collected.setdefault("fnr", []).append(scored["routing"]["fnr"])
    return {
        key: {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for key, values in collected.items()
    }


def draw_balanced_rates(
    grid: np.ndarray, weights: np.ndarray, n_items: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample across six bins defined by a noisy k=5 screening draw."""
    pool = rng.choice(grid, size=max(n_items * 30, 2000), p=weights)
    screen = rng.binomial(5, pool)
    per_bin = n_items // 6
    chosen: list[np.ndarray] = []
    short = n_items - per_bin * 6
    for level in range(6):
        members = pool[screen == level]
        take = per_bin + (1 if level in (2, 3) and short > 0 else 0)
        if short > 0 and level == 3:
            short = 0
        if len(members) == 0:
            continue
        picked = rng.choice(members, size=min(take, len(members)), replace=False)
        chosen.append(picked)
    rates = np.concatenate(chosen)
    if len(rates) < n_items:
        # Fill the requested sample size when screening bins supply too few rates.
        extra = rng.choice(pool, size=n_items - len(rates), replace=False)
        rates = np.concatenate([rates, extra])
    return rates[:n_items]


def kappa_ceiling(
    grid: np.ndarray,
    weights: np.ndarray,
    k: int,
    n_items: int,
    draws: int = 6000,
    seed: int = SEED,
    design: str = "corpus",
) -> dict:
    """Estimate weighted kappa for true rates binned against fresh observed counts."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        if design == "balanced":
            rates = draw_balanced_rates(grid, weights, n_items, rng)
        else:
            rates = rng.choice(grid, size=n_items, p=weights)
        counts = rng.binomial(k, rates).astype(float)
        values.append(prob_metrics.kappa_w(counts, k, rates))
    return {
        "k": k,
        "n_items": n_items,
        "draws": draws,
        "design": design,
        "seed": seed,
        "kappa_w": {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p2.5": float(np.percentile(values, 2.5)),
        },
    }


def corpus_summary(closed: dict[int, dict], opened: dict[int, dict]) -> dict:
    if not opened or not set(opened) <= set(closed):
        raise InputError("open corpus must be a nonempty subset of closed corpus")
    def summarize(rows: list[dict]) -> dict:
        counts = []
        for row in rows:
            if type(row["k"]) is not int or row["k"] != 5:
                raise InputError("corpus screening k must equal 5")
            if type(row.get("correct_count")) is not int or not 0 <= row["correct_count"] <= row["k"]:
                raise InputError("corpus counts must be integers in 0..k")
            counts.append(row["correct_count"])
        n = len(counts)
        return {"n": n, "mean_correct_count": sum(counts) / n if n else None,
                "count_frequencies_0_to_5": [counts.count(i) for i in range(6)]}
    return {"closed_full": summarize(list(closed.values())),
            "closed_retained": summarize([closed[key] for key in opened]),
            "closed_dropped": summarize([closed[key] for key in closed if key not in opened]),
            "open_retained": summarize(list(opened.values())),
            "retained_fraction": len(opened) / len(closed)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--closed-labels", type=Path, help="also compare complete, retained and dropped multiple-choice labels with --labels")
    parser.add_argument(
        "--kappa-ceiling",
        action="store_true",
        help="also simulate weighted agreement between known success rates "
        "and sampled reference labels",
    )
    parser.add_argument(
        "--study-k", type=int, default=30, help="generations per study prompt (default: 30)"
    )
    parser.add_argument("--study-n", type=int, default=280)
    parser.add_argument(
        "--sims", type=int, default=6000, help="Monte Carlo replicates (default: 6000)"
    )
    args = parser.parse_args(argv)

    try:
        rows = read_records(args.labels)
    except (OSError, InputError) as exc:
        parser.error(str(exc))
    if not rows:
        parser.error("labels input is empty")
    ks = {row["k"] for row in rows}
    if len(ks) != 1:
        # This likelihood assumes the same number of trials for every item.
        raise SystemExit(
            f"labels file mixes k values {sorted(ks)} — every row must have the same repeat count"
        )
    k = rows[0]["k"]
    counts = np.array([row["correct_count"] for row in rows], dtype=float)

    grid, weights = fit_rate_distribution(counts, k)
    stats = reliability(grid, weights, k)
    floor = irreducible_nll(grid, weights, k)
    oracle = simulate_oracle(grid, weights, k, n_items=len(rows))

    print(f"  items: {len(rows)}, runs per item: {k}")
    print("\n  === fitted distribution of success rates ===")
    for low, high in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.0)):
        mass = float(weights[(grid >= low) & (grid <= high)].sum())
        print(f"    rate in [{low:.1f}, {high:.1f}]: {mass:6.1%}")
    print(f"    mean true rate      : {stats['mean_true_rate']:.4f}")

    print("\n  === how much signal the labels carry ===")
    print(f"    variance of true rates      : {stats['variance_true']:.4f}")
    print(f"    variance from sampling noise: {stats['variance_sampling_noise']:.4f}")
    print(f"    reliability                 : {stats['reliability']:.4f}")
    print(f"    max attainable correlation  : {stats['max_attainable_correlation']:.4f}")

    print("\n  === irreducible loss (a model that knows every true rate) ===")
    print(f"    binomial NLL floor  : {floor:.4f}")

    print("\n  === attainable metric values, simulated oracle ===")
    for key in ("binomial_nll", "mae", "rmse", "spearman", "macro_f1", "fpr", "fnr"):
        entry = oracle[key]
        print(f"    {key:14s}: {entry['mean']:.4f} ± {entry['std']:.4f}")

    kappa = None
    if args.kappa_ceiling:
        print(f"\n  === kappa_w ceiling (n={args.study_n}, {args.sims} sims) ===")
        kappa = {}
        for design in ("balanced", "corpus"):
            for study_k in sorted({5, 20, args.study_k, 50}):
                result = kappa_ceiling(
                    grid, weights, k=study_k, n_items=args.study_n, draws=args.sims, design=design
                )
                kappa[f"{design}_k{study_k}"] = result
                marker = (
                    " <- selected simulation setting"
                    if (design == "balanced" and study_k == args.study_k)
                    else ""
                )
                print(
                    f"    {design:9s} k={study_k:3d}: "
                    f"{result['kappa_w']['mean']:.4f} ± {result['kappa_w']['std']:.4f}"
                    f"  (p2.5 {result['kappa_w']['p2.5']:.4f}){marker}"
                )

    corpus = None
    if args.closed_labels is not None:
        corpus = corpus_summary(index_records(read_records(args.closed_labels), "closed labels"), index_records(rows, "open labels"))

    write_json(
        args.out,
        {
                "k": k,
                "n_items": len(rows),
                "rate_distribution": {"grid": grid.tolist(), "weights": weights.tolist()},
                "reliability": stats,
                "irreducible_binomial_nll": floor,
                "oracle": oracle,
                "kappa_ceiling": kappa,
                **({"corpus_summary": corpus} if corpus is not None else {}),
            },
    )
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
