"""Render appendix-only oracle routing scenarios from fresh CSV endpoints.

The renderer consumes deterministic scenario outputs. It never invokes a
model or interpolates cloud-energy estimates: each plotted interval uses the
provided ``lower`` and ``upper`` rows directly.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

try:  # Support package imports and direct execution from research/figures.
    from .style import INK, MUTED, clean_axes, configure, export_figure
except ImportError:  # pragma: no cover - direct-file reproduction only
    from style import INK, MUTED, clean_axes, configure, export_figure


_REQUIRED_COLUMNS = frozenset(
    {
        "category",
        "n",
        "policy",
        "scenario",
        "local_retries",
        "cloud_length_multiplier",
        "cloud_estimate",
        "classifier_wh_per_request",
        "expected_wh_per_question",
        "all_cloud_wh_per_question",
        "electricity_relative_to_cloud",
        "expected_final_success",
        "local_routing_fraction",
    }
)
_POLICIES = frozenset(("all_cloud", "classifier", "human_aggregate"))
_SCENARIOS = frozenset(("all_cloud", "commit_local", "fallback"))
_ESTIMATES = frozenset(("lower", "upper"))
_POLICY_STYLE = {
    "human_aggregate": ("#0072B2", "o"),
    "classifier": ("#D55E00", "s"),
    "all_cloud": ("#777B80", "D"),
}
_ROUTING_POLICIES = ("human_aggregate", "classifier")
_WIDTH_INCHES = 7.0
_TRADEOFF_HEIGHT = 3.42
_RETRIES_HEIGHT = 3.25
_PANEL_LABELS = ("Study questions", "Test remainder")
_CLOUD_MODEL = "gpt-5.5-pro-2026-04-23"
_EQUAL_OUTPUT_LENGTH_MULTIPLIER = 1.0


@dataclass(frozen=True)
class Row:
    """One finite endpoint in the oracle scenario grid."""

    category: str
    n: int
    policy: str
    scenario: str
    local_retries: int
    cloud_length_multiplier: float
    cloud_estimate: str
    classifier_wh_per_request: float
    expected_wh_per_question: float
    all_cloud_wh_per_question: float
    electricity_relative_to_cloud: float
    expected_final_success: float
    local_routing_fraction: float
    model: str


@dataclass(frozen=True)
class EndpointPair:
    """The actual lower and upper endpoint rows for one condition."""

    lower: Row
    upper: Row

    def __post_init__(self) -> None:
        if self.lower.expected_wh_per_question > self.upper.expected_wh_per_question:
            raise ValueError("expected electricity endpoints are reversed")
        if self.lower.all_cloud_wh_per_question > self.upper.all_cloud_wh_per_question:
            raise ValueError("all-cloud electricity endpoints are reversed")


def _number(value: str | None, *, field: str, path: Path, line: int, minimum: float | None = None) -> float:
    if value is None:
        raise ValueError(f"{path}:{line}: missing {field}")
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{path}:{line}: {field} must be numeric") from exc
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        qualifier = "a finite value" if minimum is None else f"a finite value >= {minimum:g}"
        raise ValueError(f"{path}:{line}: {field} must be {qualifier}")
    return number


def _whole_number(value: str | None, *, field: str, path: Path, line: int, minimum: int) -> int:
    number = _number(value, field=field, path=path, line=line, minimum=float(minimum))
    if not number.is_integer():
        raise ValueError(f"{path}:{line}: {field} must be an integer")
    return int(number)


def _choice(value: str | None, *, field: str, allowed: frozenset[str], path: Path, line: int) -> str:
    if value not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"{path}:{line}: {field} must be one of {expected}")
    return value


def _read_csv(path: Path) -> list[Row]:
    """Load one CSV and reject missing fields, malformed values, and empty input."""

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames
        if not headers:
            raise ValueError(f"{path}: CSV header is missing")
        duplicates = {name for name in headers if headers.count(name) > 1}
        if duplicates:
            raise ValueError(f"{path}: duplicate CSV column(s): {', '.join(sorted(duplicates))}")
        missing = _REQUIRED_COLUMNS.difference(headers)
        if missing:
            raise ValueError(f"{path}: missing required column(s): {', '.join(sorted(missing))}")
        if "model" not in headers and "cloud_model" not in headers:
            raise ValueError(f"{path}: missing required cloud model column (model or cloud_model)")

        rows: list[Row] = []
        for line, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError(f"{path}:{line}: row has more fields than its header")
            if not any(value.strip() for value in raw.values() if value is not None):
                continue
            category = (raw["category"] or "").strip()
            model = (raw.get("model") or raw.get("cloud_model") or "").strip()
            if not category or not model:
                raise ValueError(f"{path}:{line}: category and model must be non-empty")
            success = _number(raw["expected_final_success"], field="expected_final_success", path=path, line=line)
            fraction = _number(raw["local_routing_fraction"], field="local_routing_fraction", path=path, line=line)
            if not 0.0 <= success <= 1.0:
                raise ValueError(f"{path}:{line}: expected_final_success must be a fraction in [0, 1]")
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"{path}:{line}: local_routing_fraction must be a fraction in [0, 1]")
            retries_value = (raw["local_retries"] or "").strip()
            if not retries_value:
                if raw["policy"] != "all_cloud":
                    raise ValueError(f"{path}:{line}: local_retries is required for routed policies")
                local_retries = 0
            else:
                local_retries = _whole_number(
                    retries_value,
                    field="local_retries",
                    path=path,
                    line=line,
                    minimum=0,
                )
            rows.append(
                Row(
                    category=category,
                    n=_whole_number(raw["n"], field="n", path=path, line=line, minimum=1),
                    policy=_choice(raw["policy"], field="policy", allowed=_POLICIES, path=path, line=line),
                    scenario=_choice(raw["scenario"], field="scenario", allowed=_SCENARIOS, path=path, line=line),
                    local_retries=local_retries,
                    cloud_length_multiplier=_number(raw["cloud_length_multiplier"], field="cloud_length_multiplier", path=path, line=line, minimum=0.0),
                    cloud_estimate=_choice(raw["cloud_estimate"], field="cloud_estimate", allowed=_ESTIMATES, path=path, line=line),
                    classifier_wh_per_request=_number(raw["classifier_wh_per_request"], field="classifier_wh_per_request", path=path, line=line, minimum=0.0),
                    expected_wh_per_question=_number(raw["expected_wh_per_question"], field="expected_wh_per_question", path=path, line=line, minimum=0.0),
                    all_cloud_wh_per_question=_number(raw["all_cloud_wh_per_question"], field="all_cloud_wh_per_question", path=path, line=line, minimum=0.0),
                    electricity_relative_to_cloud=_number(raw["electricity_relative_to_cloud"], field="electricity_relative_to_cloud", path=path, line=line, minimum=0.0),
                    expected_final_success=success,
                    local_routing_fraction=fraction,
                    model=model,
                )
            )
    if not rows:
        raise ValueError(f"{path}: CSV contains no data rows")
    return rows


def _same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def _pairs(rows: Iterable[Row], *, description: str) -> list[EndpointPair]:
    """Pair lower/upper rows while rejecting partial or duplicate conditions."""

    grouped: dict[tuple[object, ...], list[Row]] = defaultdict(list)
    for row in rows:
        key = (
            row.category,
            row.n,
            row.policy,
            row.scenario,
            row.local_retries,
            row.cloud_length_multiplier,
            row.classifier_wh_per_request,
            row.model,
        )
        grouped[key].append(row)

    pairs: list[EndpointPair] = []
    for key, group in grouped.items():
        by_estimate = {row.cloud_estimate: row for row in group}
        if len(group) != 2 or set(by_estimate) != _ESTIMATES:
            raise ValueError(f"{description}: expected exactly one lower and one upper row for {key!r}")
        pairs.append(EndpointPair(by_estimate["lower"], by_estimate["upper"]))
    return pairs


def _one_pair(rows: Iterable[Row], *, description: str) -> EndpointPair:
    pairs = _pairs(rows, description=description)
    if len(pairs) != 1:
        raise ValueError(f"{description}: expected one condition, found {len(pairs)}")
    return pairs[0]


def _select_endpoint_rows(
    rows: Sequence[Row],
    *,
    category: str,
    cohort: str,
    n: int,
    classifier_wh: float,
    cloud_length_multiplier: float,
) -> list[Row]:
    selected = [
        row
        for row in rows
        if row.category == category
        and row.n == n
        and _same_number(row.cloud_length_multiplier, cloud_length_multiplier)
        and (row.policy != "classifier" or _same_number(row.classifier_wh_per_request, classifier_wh))
    ]
    if not selected:
        raise ValueError(
            f"{cohort} category={category!r}: no rows match classifier_wh_per_request={classifier_wh:g} "
            f"and cloud_length_multiplier={cloud_length_multiplier:g}"
        )
    return selected


def _baseline_pair(rows: Sequence[Row], *, description: str) -> EndpointPair:
    """Collapse replicated all-cloud rows only after proving they are identical."""

    candidates = [row for row in rows if row.policy == "all_cloud"]
    if not candidates:
        raise ValueError(f"{description}: no all_cloud rows")
    by_estimate: dict[str, set[tuple[float, float, float]]] = defaultdict(set)
    representative: dict[str, Row] = {}
    for row in candidates:
        if not _same_number(row.expected_wh_per_question, row.all_cloud_wh_per_question):
            raise ValueError(f"{description}: all_cloud expected electricity differs from its baseline")
        by_estimate[row.cloud_estimate].add(
            (row.expected_wh_per_question, row.all_cloud_wh_per_question, row.expected_final_success)
        )
        representative.setdefault(row.cloud_estimate, row)
    if set(by_estimate) != _ESTIMATES or any(len(values) != 1 for values in by_estimate.values()):
        raise ValueError(f"{description}: all-cloud baseline differs across duplicated scenario rows")
    pair = EndpointPair(representative["lower"], representative["upper"])
    if not _same_number(pair.lower.expected_final_success, 1.0) or not _same_number(pair.upper.expected_final_success, 1.0):
        raise ValueError(f"{description}: all-cloud final success must be 1")
    return pair


def _x_limits(values: Sequence[float]) -> tuple[float, float]:
    lower, upper = min(values), max(values)
    span = upper - lower
    padding = max(0.04 * max(abs(lower), abs(upper), 0.01), 0.16 * span, 0.0005)
    return max(0.0, lower - padding), upper + padding


def _y_limits(values: Sequence[float]) -> tuple[float, float]:
    lower, upper = min(values), max(values)
    span = upper - lower
    padding = max(0.08 * max(abs(lower), abs(upper), 0.01), 0.16 * span, 0.0005)
    return max(0.0, lower - padding), upper + padding


def _draw_horizontal_interval(ax, pair: EndpointPair, *, y: float, color: str, marker: str, filled: bool) -> None:
    """Draw an EcoLogits electricity span at one expected-success value."""

    lower = pair.lower.expected_wh_per_question
    upper = pair.upper.expected_wh_per_question
    ax.hlines(y, lower, upper, color=color, linewidth=1.1, zorder=2)
    ax.vlines((lower, upper), y - 0.53, y + 0.53, color=color, linewidth=0.8, zorder=2)
    face = color if filled else "white"
    ax.scatter((lower, upper), (y, y), marker=marker, s=27, facecolors=face, edgecolors=color, linewidths=0.9, zorder=3)


def _draw_vertical_interval(ax, pair: EndpointPair, *, x: float, color: str, marker: str) -> None:
    """Draw the EcoLogits electricity range at one retry count."""

    lower = pair.lower.expected_wh_per_question
    upper = pair.upper.expected_wh_per_question
    ax.vlines(x, lower, upper, color=color, linewidth=1.1, zorder=3)
    ax.hlines((lower, upper), x - 0.075, x + 0.075, color=color, linewidth=0.8, zorder=3)
    ax.scatter((x, x), (lower, upper), marker=marker, s=25, facecolors="white", edgecolors=color, linewidths=0.9, zorder=4)


def _tradeoff_legend() -> list[Line2D]:
    human, classifier, baseline = _POLICY_STYLE["human_aggregate"], _POLICY_STYLE["classifier"], _POLICY_STYLE["all_cloud"]
    return [
        Line2D([], [], color=human[0], marker=human[1], markerfacecolor=human[0], label="Selected answer · Human aggregate"),
        Line2D([], [], color=classifier[0], marker=classifier[1], markerfacecolor=classifier[0], label="Selected answer · Classifier"),
        Line2D([], [], color=human[0], marker=human[1], markerfacecolor="white", label="Failure fallback · Human aggregate"),
        Line2D([], [], color=classifier[0], marker=classifier[1], markerfacecolor="white", label="Failure fallback · Classifier"),
        Line2D([], [], color=baseline[0], marker=baseline[1], markerfacecolor=baseline[0], label="Cloud reference"),
    ]


def _retried_pairs(rows: Sequence[Row], *, policy: str, description: str) -> dict[int, EndpointPair]:
    candidates = [row for row in rows if row.policy == policy and row.scenario == "fallback"]
    pairs = _pairs(candidates, description=description)
    by_retry = {pair.lower.local_retries: pair for pair in pairs}
    if set(by_retry) != set(range(6)):
        found = ", ".join(str(value) for value in sorted(by_retry))
        raise ValueError(f"{description}: expected local_retries 0 through 5, found {found or 'none'}")
    if len(by_retry) != len(pairs):
        raise ValueError(f"{description}: duplicate fallback retry conditions")
    for retries, pair in by_retry.items():
        if not _same_number(pair.lower.expected_final_success, 1.0) or not _same_number(pair.upper.expected_final_success, 1.0):
            raise ValueError(f"{description}: fallback retry {retries} must have final success 1")
    return by_retry


def _panel_data(
    rows: Sequence[Row],
    *,
    cohort: str,
    category: str,
    n: int,
    classifier_wh: float,
    cloud_length_multiplier: float,
) -> tuple[EndpointPair, dict[str, EndpointPair], dict[str, dict[int, EndpointPair]]]:
    if not _same_number(cloud_length_multiplier, _EQUAL_OUTPUT_LENGTH_MULTIPLIER):
        raise ValueError(
            "oracle figures support only cloud_length_multiplier=1 for equal output lengths"
        )
    selected = _select_endpoint_rows(
        rows,
        category=category,
        cohort=cohort,
        n=n,
        classifier_wh=classifier_wh,
        cloud_length_multiplier=cloud_length_multiplier,
    )
    baseline = _baseline_pair(selected, description=f"{cohort} category={category!r} all-cloud baseline")
    committed: dict[str, EndpointPair] = {}
    for policy in _ROUTING_POLICIES:
        candidates = [
            row
            for row in selected
            if row.policy == policy and row.scenario == "commit_local" and row.local_retries == 0
        ]
        if policy == "classifier" or candidates:
            committed[policy] = _one_pair(
                candidates,
                description=f"{cohort} category={category!r} A {policy}",
            )
    for policy, pair in committed.items():
        if min(pair.lower.expected_final_success, pair.upper.expected_final_success) < 0.5:
            raise ValueError(f"{cohort} category={category!r} A {policy}: expected success falls below the fixed 50% plotting floor")
    fallbacks = {
        policy: _retried_pairs(selected, policy=policy, description=f"{cohort} category={category!r} B {policy}")
        for policy in committed
    }
    return baseline, committed, fallbacks


def _panel_specs(rows: Sequence[Row], *, category: str) -> tuple[tuple[int, str], ...]:
    """Name the two source cohorts while taking their sizes from fresh scenario rows."""

    sizes = sorted({row.n for row in rows if row.category == category})
    if len(sizes) != len(_PANEL_LABELS):
        raise ValueError(
            f"oracle figures require two source cohorts for category={category!r}, found {sizes!r}"
        )
    return tuple(zip(sizes, _PANEL_LABELS, strict=True))


def render_tradeoff(
    rows: Sequence[Row],
    output_dir: Path,
    *,
    category: str,
    classifier_wh: float,
    cloud_length_multiplier: float,
) -> None:
    """Render the A/B electricity-versus-final-success appendix figure."""

    configure()
    specs = _panel_specs(rows, category=category)
    panels = [
        _panel_data(
            rows,
            cohort=label,
            category=category,
            n=n,
            classifier_wh=classifier_wh,
            cloud_length_multiplier=cloud_length_multiplier,
        )
        for n, label in specs
    ]
    shared_x_limits = _x_limits(
        [
            value
            for baseline, committed, fallbacks in panels
            for pair in [baseline, *committed.values(), *(fallbacks[policy][0] for policy in fallbacks)]
            for value in (pair.lower.expected_wh_per_question, pair.upper.expected_wh_per_question)
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(_WIDTH_INCHES, _TRADEOFF_HEIGHT), sharey=True)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.20, top=0.64, wspace=0.24)

    for ax, (n, label), (baseline, committed, fallbacks) in zip(axes, specs, panels):
        baseline_style = _POLICY_STYLE["all_cloud"]
        _draw_horizontal_interval(ax, baseline, y=100.0, color=baseline_style[0], marker=baseline_style[1], filled=True)
        for policy in committed:
            color, marker = _POLICY_STYLE[policy]
            pair = committed[policy]
            _draw_horizontal_interval(
                ax,
                pair,
                y=100.0 * pair.lower.expected_final_success,
                color=color,
                marker=marker,
                filled=True,
            )
            if not _same_number(pair.lower.expected_final_success, pair.upper.expected_final_success):
                raise ValueError(f"{label} A {policy}: lower/upper rows disagree on expected final success")
            _draw_horizontal_interval(ax, fallbacks[policy][0], y=100.0, color=color, marker=marker, filled=False)

        ax.set_xlim(*shared_x_limits)
        ax.set_ylim(50.0, 102.0)
        ax.set_yticks((50, 60, 70, 80, 90, 100))
        ax.set_title(f"{label} (n={n})", pad=6)
        clean_axes(ax, grid_axis="y")

    axes[0].set_ylabel("Expected final-answer success (%)")
    fig.supxlabel("Expected electricity (Wh / question)", y=0.075)
    fig.text(0.095, 0.97, "Electricity and answer success", ha="left", va="top", fontsize=8.8, fontweight="bold", color=INK)
    fig.text(
        0.095,
        0.91,
        "GPT-5.5 Pro costs with equal local and cloud output lengths",
        ha="left",
        va="top",
        fontsize=7.0,
        color=MUTED,
    )
    fig.legend(handles=_tradeoff_legend(), loc="upper right", bbox_to_anchor=(0.985, 0.835), ncol=3)
    try:
        export_figure(fig, output_dir, "oracle_tradeoff")
    finally:
        plt.close(fig)


def render_retries(
    rows: Sequence[Row],
    output_dir: Path,
    *,
    category: str,
    classifier_wh: float,
    cloud_length_multiplier: float,
) -> None:
    """Render B retry electricity endpoints only; final success is deliberately omitted."""

    configure()
    specs = _panel_specs(rows, category=category)
    panels = [
        _panel_data(
            rows,
            cohort=label,
            category=category,
            n=n,
            classifier_wh=classifier_wh,
            cloud_length_multiplier=cloud_length_multiplier,
        )
        for n, label in specs
    ]
    shared_y_limits = _y_limits(
        [
            value
            for baseline, _, fallbacks in panels
            for pair in [baseline, *(pair for values in fallbacks.values() for pair in values.values())]
            for value in (pair.lower.expected_wh_per_question, pair.upper.expected_wh_per_question)
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(_WIDTH_INCHES, _RETRIES_HEIGHT), sharex=True)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.20, top=0.64, wspace=0.24)

    for ax, (n, label), (baseline, _, fallbacks) in zip(axes, specs, panels):
        lower = baseline.lower.expected_wh_per_question
        upper = baseline.upper.expected_wh_per_question
        ax.axhspan(lower, upper, color=_POLICY_STYLE["all_cloud"][0], alpha=0.06, zorder=0)
        ax.hlines((lower, upper), -0.25, 5.25, color=_POLICY_STYLE["all_cloud"][0], linewidth=0.85, zorder=1)
        offsets = {"human_aggregate": -0.11, "classifier": 0.11} if len(fallbacks) == 2 else {"classifier": 0.0}
        for policy, offset in offsets.items():
            color, marker = _POLICY_STYLE[policy]
            ordered = [fallbacks[policy][retries] for retries in range(6)]
            positions = [retries + offset for retries in range(6)]
            ax.plot(
                positions,
                [pair.lower.expected_wh_per_question for pair in ordered],
                color=color,
                linewidth=0.8,
                alpha=0.75,
                zorder=2,
            )
            ax.plot(
                positions,
                [pair.upper.expected_wh_per_question for pair in ordered],
                color=color,
                linewidth=0.8,
                alpha=0.75,
                zorder=2,
            )
            for retries, pair in fallbacks[policy].items():
                _draw_vertical_interval(ax, pair, x=retries + offset, color=color, marker=marker)

        ax.set_xlim(-0.32, 5.32)
        ax.set_ylim(*shared_y_limits)
        ax.set_xticks(range(6))
        ax.set_title(f"{label} (n={n})", pad=6)
        clean_axes(ax, grid_axis="y")

    axes[0].set_ylabel("Expected electricity (Wh / question)")
    fig.supxlabel("Additional local attempts before cloud fallback", y=0.075)
    fig.text(0.095, 0.97, "Electricity with local retries", ha="left", va="top", fontsize=8.8, fontweight="bold", color=INK)
    fig.text(
        0.095,
        0.91,
        "GPT-5.5 Pro costs with equal local and cloud output lengths",
        ha="left",
        va="top",
        fontsize=7.0,
        color=MUTED,
    )
    fig.legend(
        handles=[
            Line2D([], [], color=_POLICY_STYLE["human_aggregate"][0], marker="o", markerfacecolor="white", label="Human aggregate"),
            Line2D([], [], color=_POLICY_STYLE["classifier"][0], marker="s", markerfacecolor="white", label="Classifier"),
            Line2D([], [], color=_POLICY_STYLE["all_cloud"][0], marker="D", markerfacecolor=_POLICY_STYLE["all_cloud"][0], label="Cloud reference"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.835),
        ncol=3,
    )
    try:
        export_figure(fig, output_dir, "oracle_retries")
    finally:
        plt.close(fig)


def _validate_sources(
    study_rows: Sequence[Row], heldout_rows: Sequence[Row], *, category: str
) -> list[Row]:
    """Confirm the selected cohort rows and one cloud model before output."""

    combined = [*study_rows, *heldout_rows]
    for rows, cohort in zip((study_rows, heldout_rows), _PANEL_LABELS, strict=True):
        sizes = {row.n for row in rows if row.category == category}
        if len(sizes) != 1:
            raise ValueError(f"{cohort} category={category!r}: expected one cohort size, got {sorted(sizes)!r}")
    models = {row.model for row in combined}
    if models != {_CLOUD_MODEL}:
        raise ValueError(
            f"oracle figures require only {_CLOUD_MODEL!r}, found {sorted(models)!r}"
        )
    return combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-csv", type=Path, required=True, help="Fresh study oracle CSV")
    parser.add_argument("--heldout-csv", type=Path, required=True, help="Fresh held-out oracle CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for appendix figure assets")
    parser.add_argument("--category", default="overall", help="Saved category to render (default: overall)")
    parser.add_argument(
        "--classifier-wh",
        type=float,
        help=(
            "Classifier electricity (Wh/request) used to select fresh rows. If omitted, the CSVs must "
            "contain exactly one nonzero value."
        ),
    )
    parser.add_argument(
        "--cloud-length-multiplier",
        type=float,
        default=1.0,
        help="Saved cloud output-length multiplier to plot (default: 1)",
    )
    return parser


def _resolve_classifier_wh(rows: Sequence[Row], requested: float | None) -> float:
    if requested is not None:
        if not math.isfinite(requested) or requested < 0:
            raise ValueError("--classifier-wh must be finite and nonnegative")
        return requested
    candidates = sorted({row.classifier_wh_per_request for row in rows if row.classifier_wh_per_request > 0.0})
    if len(candidates) != 1:
        shown = ", ".join(f"{value:g}" for value in candidates) or "none"
        raise ValueError(
            "--classifier-wh is required when the CSVs do not contain exactly one nonzero "
            f"classifier_wh_per_request value; found {shown}"
        )
    return candidates[0]


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.cloud_length_multiplier) or not _same_number(
        args.cloud_length_multiplier, _EQUAL_OUTPUT_LENGTH_MULTIPLIER
    ):
        raise ValueError("--cloud-length-multiplier must be 1 for equal output lengths")
    rows = _validate_sources(
        _read_csv(args.study_csv), _read_csv(args.heldout_csv), category=args.category
    )
    classifier_wh = _resolve_classifier_wh(rows, args.classifier_wh)
    render_tradeoff(
        rows,
        args.output_dir,
        category=args.category,
        classifier_wh=classifier_wh,
        cloud_length_multiplier=args.cloud_length_multiplier,
    )
    render_retries(
        rows,
        args.output_dir,
        category=args.category,
        classifier_wh=classifier_wh,
        cloud_length_multiplier=args.cloud_length_multiplier,
    )


if __name__ == "__main__":
    main()
