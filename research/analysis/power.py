"""Simulate power for a one-sided weighted-kappa test using a binned latent normal.

Uniform level marginals are the default; --labels supplies an observed marginal."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal, norm

from research.analysis.data_io import InputError, read_records, write_json

SEED = 20260814
REPLICATES = 6000
N_PROMPTS = 280
LEVELS = 6
ALPHA = 0.05
TARGET_POWER = 0.80

WEIGHTS = 1 - np.abs(np.arange(LEVELS)[:, None] - np.arange(LEVELS)[None, :]) / (LEVELS - 1)


def cuts_from_marginal(shares: np.ndarray) -> np.ndarray:
    """Latent cut points reproducing a six-level marginal under a standard normal."""
    cum = np.cumsum(shares)[:-1]
    return norm.ppf(np.clip(cum, 1e-9, 1 - 1e-9))


def joint_probs(rho: float, gt_cuts: np.ndarray, h_cuts: np.ndarray) -> np.ndarray:
    """P(gt level i, human level j) under the correlated latent pair."""
    edges_g = np.concatenate(([-np.inf], gt_cuts, [np.inf]))
    edges_h = np.concatenate(([-np.inf], h_cuts, [np.inf]))
    dist = multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]])

    def box_cdf(x: float, y: float) -> float:
        if np.isinf(x) and x < 0 or np.isinf(y) and y < 0:
            return 0.0
        return float(dist.cdf([min(x, 8.0), min(y, 8.0)]))

    grid = np.array([[box_cdf(x, y) for y in edges_h] for x in edges_g])
    probs = grid[1:, 1:] - grid[:-1, 1:] - grid[1:, :-1] + grid[:-1, :-1]
    return np.clip(probs, 0.0, None)


def kappa_from_joint(probs: np.ndarray) -> float:
    """Population linearly-weighted kappa of a 6x6 joint distribution."""
    marg_g = probs.sum(axis=1)
    marg_h = probs.sum(axis=0)
    expected = np.outer(marg_g, marg_h)
    po = float((WEIGHTS * probs).sum())
    pe = float((WEIGHTS * expected).sum())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def sample_kappa(gt_levels: np.ndarray, h_levels: np.ndarray) -> float:
    counts = np.zeros((LEVELS, LEVELS))
    np.add.at(counts, (gt_levels, h_levels), 1)
    return kappa_from_joint(counts / counts.sum())


def tune_rho(target_kappa: float, gt_cuts: np.ndarray, h_cuts: np.ndarray) -> float:
    """Bisect the latent correlation until the population kappa_w hits target."""
    lo, hi = -0.2, 0.9999
    for _ in range(60):
        mid = (lo + hi) / 2
        if kappa_from_joint(joint_probs(mid, gt_cuts, h_cuts)) < target_kappa:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def simulate(
    rho: float,
    gt_cuts: np.ndarray,
    h_cuts: np.ndarray,
    rng: np.random.Generator,
    n: int = N_PROMPTS,
    reps: int = REPLICATES,
) -> np.ndarray:
    cov = np.array([[1.0, rho], [rho, 1.0]])
    chol = np.linalg.cholesky(cov)
    draws = rng.standard_normal((reps, n, 2)) @ chol.T
    gt = np.searchsorted(gt_cuts, draws[:, :, 0])
    hu = np.searchsorted(h_cuts, draws[:, :, 1])
    return np.array([sample_kappa(gt[r], hu[r]) for r in range(reps)])


def power_at(target_kappa: float, critical: float, gt_cuts, h_cuts, rng) -> tuple[float, float]:
    rho = tune_rho(target_kappa, gt_cuts, h_cuts)
    kappas = simulate(rho, gt_cuts, h_cuts, rng)
    return float((kappas < critical).mean()), float(kappas.std(ddof=1))


def min_detectable(floor: float, gt_cuts, h_cuts, seed: int = SEED) -> dict:
    """Largest true kappa_w rejected as below `floor` with >= 80% power."""
    rng = np.random.default_rng(seed)  # One independent stream per floor.
    rho0 = tune_rho(floor, gt_cuts, h_cuts)
    critical = float(np.quantile(simulate(rho0, gt_cuts, h_cuts, rng), ALPHA))

    lo, hi = -0.05, floor
    for _ in range(12):
        mid = (lo + hi) / 2
        power, _ = power_at(mid, critical, gt_cuts, h_cuts, rng)
        if power >= TARGET_POWER:
            lo = mid
        else:
            hi = mid
    power, sd = power_at(lo, critical, gt_cuts, h_cuts, rng)
    return {
        "floor": floor,
        "critical": round(critical, 4),
        "min_detectable": round(lo, 3),
        "power_there": round(power, 3),
        "se_at_detectable": round(sd, 3),
    }


def gt_marginal_from_labels(path: Path) -> np.ndarray:
    counts = np.zeros(LEVELS)
    for row in read_records(path):
        p = row["correct_count"] / row["k"]
        counts[min(int(p * LEVELS), LEVELS - 1)] += 1
    if counts.sum() == 0:
        raise InputError(f"labels input is empty: {path}")
    return counts / counts.sum()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=None,
        help="single floor; default sweeps the registered table",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="labels file for the observed ground-truth marginal "
        "(default: uniform, the balanced design)",
    )
    parser.add_argument("--out", type=Path, default=None, help="also write the rows as json")
    args = parser.parse_args(argv)

    if args.labels:
        gt_cuts = cuts_from_marginal(gt_marginal_from_labels(args.labels))
        print(f"  ground-truth marginal from {args.labels}")
    else:
        gt_cuts = cuts_from_marginal(np.full(LEVELS, 1 / LEVELS))
    h_cuts = cuts_from_marginal(np.full(LEVELS, 1 / LEVELS))

    floors = [args.floor] if args.floor else [x / 100 for x in range(10, 91, 5)]
    print(f"  n={N_PROMPTS}, {REPLICATES} replicates, seed {SEED}, one-sided alpha={ALPHA}")
    print(f"  {'floor':>6} {'min detectable':>15} {'SE there':>9} {'critical':>9}")
    rows = []
    for floor in floors:
        row = min_detectable(floor, gt_cuts, h_cuts)
        rows.append(row)
        print(
            f"  {row['floor']:>6.2f} {row['min_detectable']:>15.2f} "
            f"{row['se_at_detectable']:>9.3f} {row['critical']:>9.4f}"
        )
    if args.out:
        write_json(args.out, rows)
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
