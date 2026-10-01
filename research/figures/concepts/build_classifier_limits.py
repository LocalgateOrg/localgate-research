"""Render Figure 3 from fresh held-out classifier predictions."""


import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from research.figures.style import prepare_output_directory, figure_arguments
from research.figures.inputs import classifier_operating_curve, load_classifier_predictions
from research.figures.style import THRESHOLD_STYLES, configure, export_figure

COLUMN_WIDTH_INCHES = 3.3367578524975787


def main() -> None:
    args = figure_arguments(__doc__, "--heldout-predictions")
    predictions, _ = load_classifier_predictions(args.heldout_predictions)
    rows, selected = classifier_operating_curve(predictions)
    classifier = {"n": len(predictions)}
    width = COLUMN_WIDTH_INCHES
    output = prepare_output_directory(args.output_dir)

    configure()
    font_manager.findfont("Liberation Serif", fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": "Liberation Serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    height = 3.95
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0.50 / width, 1.10 / height, 2.60 / width, 2.60 / height))
    ax.add_patch(
        Rectangle(
            (0, 0),
            10,
            20,
            facecolor="#E4EEF0",
            edgecolor="#54666C",
            linewidth=0.65,
            linestyle=(0, (3, 2)),
            zorder=1,
        )
    )
    fpr = np.array([r["fpr"] for r in rows]) * 100
    fnr = np.array([r["fnr"] for r in rows]) * 100
    ax.plot(fpr, fnr, color="#245C70", linewidth=0.7, zorder=2)
    ax.scatter(fpr, fnr, s=1.2, color="#245C70", alpha=0.45, linewidths=0, zorder=3)
    # No smoothing, interpolation or resampling: every saved point is drawn.
    styles = [(THRESHOLD_STYLES[0.50][0], "s", (39, 34)), ("#202124", "X", (22, 57))]
    handles, labels = [], []
    for row, (color, marker, position) in zip(selected, styles, strict=True):
        x, y = row["fpr"] * 100, row["fnr"] * 100
        ax.scatter(
            [x], [y], marker=marker, s=31, color=color, edgecolor="white", linewidth=0.65, zorder=4
        )
        ax.annotate(
            f"t = {row['threshold']:.2f}",
            xy=(x, y),
            xytext=position,
            fontsize=9,
            va="center",
            ha="left",
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.65},
        )
        handles.append(Line2D([0], [0], marker=marker, color=color, linestyle="none", markersize=5))
        labels.append(f"t = {row['threshold']:.2f}: {row['local_coverage'] * 100:.1f}%")
    ax.annotate(
        "Target",
        xy=(5, 10),
        xytext=(20, 10),
        ha="left",
        va="center",
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": "#54666C", "linewidth": 0.6},
    )
    ax.set(
        xlim=(0, 100),
        ylim=(0, 100),
        xticks=range(0, 101, 20),
        yticks=range(0, 101, 20),
        xlabel="False-positive rate (%)",
        ylabel="False-negative rate (%)",
    )
    ax.tick_params(length=2.5, pad=4)
    ax.xaxis.labelpad = 7
    ax.yaxis.labelpad = 7
    ax.grid(color="#E4E7E8", linewidth=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(3.10 / width, 3.80 / height, f"n = {classifier['n']:,}", ha="right", fontsize=8)
    fig.text(0.50 / width, 0.52 / height, "Error limits: FPR ≤10%, FNR ≤20%", fontsize=8)
    fig.text(1.80 / width, 0.30 / height, "Routed locally", ha="center", fontsize=8)
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(1.80 / width, 0.13 / height),
        ncol=2,
        fontsize=8,
        handlelength=1,
        columnspacing=1.5,
    )
    export_figure(fig, output, "03_error_limits")
    plt.close(fig)

    print(output)


if __name__ == "__main__":
    main()
