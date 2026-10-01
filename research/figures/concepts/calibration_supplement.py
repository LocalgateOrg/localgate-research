"""Render a supplementary reliability plot from held-out prediction records."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib import font_manager  # noqa: E402
from matplotlib import pyplot as plt

from research.figures.style import prepare_output_directory, figure_arguments
from research.figures.inputs import calibration_bins, load_classifier_predictions
from research.figures.style import INK, MUTED, clean_axes, configure, export_figure

COLUMN_WIDTH_INCHES = 3.3367578524975787
STEM = "calibration_reliability"


def validate_calibration(bins: list[dict[str, float | int | None]], ece: float, n: int) -> None:
    """Check count coverage, bin geometry, and the ECE definition after recomputation."""

    if len(bins) != 10:
        raise ValueError("Expected ten equal-width calibration bins")
    total = 0
    weighted_gap = 0.0
    for index, row in enumerate(bins):
        expected_lower = index / 10
        expected_upper = (index + 1) / 10
        if row["lower"] != expected_lower or row["upper"] != expected_upper:
            raise ValueError("Calibration bins are not the expected equal-width partition")
        count = row["count"]
        if type(count) is not int or count < 0:
            raise ValueError("Calibration count must be a non-negative integer")
        if count == 0:
            if any(row[field] is not None for field in ("mean_probability", "mean_observed_rate", "absolute_gap")):
                raise ValueError("Empty calibration bins must not invent a mean observation")
            continue
        for field in ("mean_probability", "mean_observed_rate", "absolute_gap"):
            value = row[field]
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid calibration field {field}")
        gap = abs(row["mean_probability"] - row["mean_observed_rate"])
        if abs(gap - row["absolute_gap"]) > 1e-14:
            raise ValueError("Saved calibration gap does not match its mean values")
        total += count
        weighted_gap += count * gap
    if total != n:
        raise ValueError(f"Expected {n} held-out items, found {total}")
    recomputed_ece = weighted_gap / total
    if abs(recomputed_ece - ece) > 1e-14:
        raise ValueError("Weighted absolute calibration gap does not reproduce ECE")




def render(bins: list[dict[str, float | int | None]], ece: float, *, n: int, generations: int):
    """Return a single-column reliability/count figure without uncertainty intervals."""

    configure()
    font_manager.findfont("Liberation Serif", fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": "Liberation Serif",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
        }
    )
    figure = plt.figure(figsize=(COLUMN_WIDTH_INCHES, 3.62))
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(2.35, 0.82),
        left=0.19,
        right=0.91,
        top=0.93,
        bottom=0.14,
        hspace=0.14,
    )
    reliability = figure.add_subplot(grid[0])
    counts = figure.add_subplot(grid[1], sharex=reliability)

    populated = [row for row in bins if int(row["count"]) > 0]
    predicted = [float(row["mean_probability"]) for row in populated]
    observed = [float(row["mean_observed_rate"]) for row in populated]
    centres = [(float(row["lower"]) + float(row["upper"])) / 2 for row in bins]
    sample_counts = [int(row["count"]) for row in bins]

    reliability.plot(
        [0, 1],
        [0, 1],
        color=MUTED,
        linestyle=(0, (3, 2)),
        linewidth=0.85,
        label="Perfect calibration",
        zorder=1,
    )
    reliability.plot(predicted, observed, color="#0072B2", linewidth=0.8, zorder=2)
    reliability.scatter(
        predicted,
        observed,
        color="#0072B2",
        edgecolor="white",
        linewidth=0.55,
        s=22,
        zorder=3,
        label="Bin mean",
    )
    reliability.set(
        xlim=(0, 1),
        ylim=(-0.025, 1.025),
        yticks=(0, 0.25, 0.5, 0.75, 1),
        ylabel=f"Mean observed success (k={generations})",
        title=f"Calibration on {n:,} test questions",
    )
    reliability.tick_params(labelbottom=False, length=2.5, pad=2.5)
    reliability.legend(loc="upper left", fontsize=8, handlelength=1.5, borderaxespad=0.15)
    reliability.text(
        0.98, 0.07, f"ECE = {ece:.3f}", ha="right", va="bottom", fontsize=8, color=INK
    )
    clean_axes(reliability, grid_axis="y")

    counts.bar(
        centres, sample_counts, width=0.086, color="#9BC7DE", edgecolor="#3E6E85", linewidth=0.55
    )
    counts.set(
        xlim=(0, 1),
        ylim=(0, max(sample_counts) * 1.18),
        xticks=(0, 0.25, 0.5, 0.75, 1),
        xlabel="Predicted probability",
        ylabel="Items",
    )
    counts.tick_params(length=2.5, pad=2.5)
    clean_axes(counts, grid_axis="y")
    return figure




def main() -> None:
    args = figure_arguments(__doc__, "--heldout-predictions")
    predictions, generations = load_classifier_predictions(args.heldout_predictions)
    bins, ece = calibration_bins(predictions)
    validate_calibration(bins, ece, len(predictions))
    output = prepare_output_directory(args.output_dir)
    figure = render(bins, ece, n=len(predictions), generations=generations)
    export_figure(figure, output, STEM)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
