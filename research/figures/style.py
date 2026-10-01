"""Shared plot styling and PDF, SVG and PNG export."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import matplotlib

# The renderer is also used in clean, headless reproduction environments.
matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt  # noqa: E402

INK: Final = "#202124"
MUTED: Final = "#666A70"
GRID: Final = "#D7D9DC"
PANEL: Final = "#F4F5F6"
TARGET: Final = "#DCE8DC"
BOUND: Final = "#777B80"

# Okabe--Ito derived colours.  Every threshold also has a unique marker.
THRESHOLD_STYLES: Final = {
    0.25: ("#0072B2", "o"),
    0.50: ("#D55E00", "s"),
    0.75: ("#009E73", "D"),
    0.90: ("#7A3E9D", "^"),
}


def configure() -> None:
    """Apply the fixed, compact style used by every exported figure."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.65,
            "axes.grid": False,
            "axes.axisbelow": True,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.45,
            "legend.columnspacing": 1.0,
            "lines.linewidth": 1.15,
            "lines.markersize": 4.2,
            "patch.linewidth": 0.7,
            "text.color": INK,
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.transparent": False,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "aisus-paper-figures-v1",
            "path.simplify": False,
        }
    )


def clean_axes(ax, *, grid_axis: str | None = None) -> None:
    """Keep axes quiet while retaining an explicit quantitative frame."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.45, alpha=0.8)


def export_figure(fig, output_dir: str | Path, stem: str) -> None:
    """Write PDF, SVG, and 300 dpi PNG versions of the figure."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        destination / f"{stem}.pdf",
        format="pdf",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "AISUS figure renderer"},
    )
    fig.savefig(
        destination / f"{stem}.svg",
        format="svg",
        metadata={"Date": None, "Creator": "AISUS figure renderer"},
    )
    fig.savefig(
        destination / f"{stem}.png",
        format="png",
        dpi=300,
        metadata={"Software": "AISUS figure renderer"},
    )


def prepare_output_directory(output: Path) -> Path:
    """Return a resolved output directory only when no existing files can be replaced."""

    resolved = output.resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError("Output path must be a directory")
        if any(resolved.iterdir()):
            raise ValueError("Output directory must be new or empty")
    else:
        resolved.mkdir(parents=True)
    return resolved


def figure_arguments(description: str, input_flag: str) -> argparse.Namespace:
    """Read the input and output paths used by a single-input figure."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(input_flag, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()
