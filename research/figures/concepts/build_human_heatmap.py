"""Render Figure 2 from fresh per-question human-study records."""


import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from research.figures.style import prepare_output_directory, figure_arguments
from research.figures.inputs import load_human_matrix
from research.figures.style import configure, export_figure

COLUMN_WIDTH_INCHES = 3.3367578524975787


def main() -> None:
    args = figure_arguments(__doc__, "--human-report")
    matrix_rows, generations = load_human_matrix(args.human_report)
    matrix = np.asarray(matrix_rows)
    human = {"n": int(matrix.sum()), "generations_per_question": generations}
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
    # Positions are inches: fixed square matrix, then label, key, and scale.
    ax = fig.add_axes((0.50 / width, 1.10 / height, 2.60 / width, 2.60 / height))
    cmap = LinearSegmentedColormap.from_list("localgate_counts", ["#F4F7F8", "#245C70"])
    image = ax.imshow(matrix, origin="lower", cmap=cmap, vmin=0, vmax=30, interpolation="nearest")
    for row in range(6):
        for column in range(6):
            rgb = np.asarray(cmap(image.norm(matrix[row, column]))[:3])
            linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
            white = 1.05 / (luminance + 0.05)
            black = (luminance + 0.05) / 0.05
            color = "white" if white > black else "black"
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=10,
                color=color,
            )
    # Outline exact agreement on the diagonal.
    for band in range(6):
        ax.add_patch(
            Rectangle(
                (band - 0.45, band - 0.45),
                0.90,
                0.90,
                fill=False,
                edgecolor="#202124",
                linewidth=0.65,
            )
        )
    for cut in (ax.axvline, ax.axhline):
        cut(2.5, color="#202124", linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set(
        xticks=range(6),
        yticks=range(6),
        xlim=(-0.5, 5.5),
        ylim=(-0.5, 5.5),
        xlabel="Human forecast band",
        ylabel="Observed success band",
    )
    ax.tick_params(length=0, pad=4)
    ax.xaxis.labelpad = 7
    ax.yaxis.labelpad = 7
    for spine in ax.spines.values():
        spine.set_color("#9AA5A9")
        spine.set_linewidth(0.5)
    fig.text(3.10 / width, 3.80 / height, f"n = {human['n']}", ha="right", fontsize=8)
    fig.legend(
        [
            Rectangle((0, 0), 1, 1, fill=False, edgecolor="#202124", linewidth=0.65),
            Line2D([0], [0], color="#202124", linewidth=0.8, linestyle=(0, (3, 2))),
        ],
        ["Same band", "Routing cut"],
        loc="center",
        ncol=2,
        fontsize=8,
        bbox_to_anchor=(1.80 / width, 0.60 / height),
        handlelength=1.4,
        columnspacing=1.6,
    )
    cax = fig.add_axes((0.50 / width, 0.40 / height, 2.60 / width, 0.07 / height))
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[0, 10, 20, 30])
    colorbar.ax.tick_params(labelsize=8, length=2, pad=2)
    colorbar.outline.set_visible(False)
    fig.text(
        1.80 / width, 0.035 / height, "Questions per cell", ha="center", va="bottom", fontsize=8
    )
    export_figure(fig, output, "02_human_disagreement")
    plt.close(fig)

    print(output)


if __name__ == "__main__":
    main()
