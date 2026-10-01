"""Regenerate the six research figures from fresh analysis outputs and records."""

from __future__ import annotations

import argparse
import base64
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from fontTools.ttLib import TTFont
from research.figures.style import prepare_output_directory

FIGURES = (
    "01_study_map",
    "02_human_disagreement",
    "03_error_limits",
    "oracle_tradeoff",
    "oracle_retries",
    "calibration_reliability",
)


def _run(command: list[str], cwd: Path) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        env={**os.environ, "SOURCE_DATE_EPOCH": "0", "FORCE_SOURCE_DATE": "1"},
    )
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed:\n{result.stdout}\n{result.stderr}")




def _copy_exports(
    source: Path,
    source_stem: str,
    destination: Path,
    target_stem: str,
    *,
    extensions: tuple[str, ...] = ("pdf", "svg", "png"),
) -> None:
    for extension in extensions:
        path = source / f"{source_stem}.{extension}"
        if not path.is_file():
            raise RuntimeError(f"Renderer did not create {path}")
        shutil.copyfile(path, destination / f"{target_stem}.{extension}")


def _stable_font_timestamps(svg: str) -> str:
    """Normalize dvisvgm's embedded WOFF timestamps for repeatable SVG bytes."""

    def replace(match: re.Match[str]) -> str:
        with TTFont(io.BytesIO(base64.b64decode(match[1])), recalcTimestamp=False) as font:
            font["head"].created = font["head"].modified = 2082844800
            stream = io.BytesIO()
            font.save(stream)
        return "base64," + base64.b64encode(stream.getvalue()).decode("ascii")

    return re.sub(r"base64,([A-Za-z0-9+/=]+)", replace, svg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Outputs:
  Writes PDF, SVG, and PNG versions of all six figures: 01_study_map,
  02_human_disagreement, 03_error_limits, oracle_tradeoff, oracle_retries,
  and calibration_reliability.

Dependencies:
  --oracle-study-csv and --oracle-heldout-csv are the CSV inputs for both
  oracle figures; the human report and held-out predictions feed the other
  four figures.""",
    )
    parser.add_argument(
        "--human-report", type=Path, required=True,
        help="fresh human_descriptive.json analysis output",
    )
    parser.add_argument(
        "--heldout-predictions", type=Path, required=True,
        help="fresh heldout1375_predictions.jsonl analysis input",
    )
    parser.add_argument(
        "--oracle-study-csv", type=Path, required=True,
        help="fresh study280_oracle.csv required by the oracle figures",
    )
    parser.add_argument(
        "--oracle-heldout-csv", type=Path, required=True,
        help="fresh heldout1375_oracle.csv required by the oracle figures",
    )
    parser.add_argument(
        "--classifier-wh", type=float, required=True,
        help="classifier Wh/request recorded for oracle CSV rows",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="new or empty directory for all six PDF, SVG, and PNG figure exports",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = prepare_output_directory(args.output_dir)
    root = Path(__file__).resolve().parents[2]

    with tempfile.TemporaryDirectory(prefix="localgate-figures-") as temporary:
        scratch = Path(temporary)
        jobs = (
            (
                "research.figures.concepts.build_study_overview",
                "study",
                "study_overview",
                "01_study_map",
                ["--human-report", str(args.human_report), "--heldout-predictions", str(args.heldout_predictions)],
            ),
            (
                "research.figures.concepts.build_human_heatmap",
                "human",
                "02_human_disagreement",
                "02_human_disagreement",
                ["--human-report", str(args.human_report)],
            ),
            (
                "research.figures.concepts.build_classifier_limits",
                "classifier",
                "03_error_limits",
                "03_error_limits",
                ["--heldout-predictions", str(args.heldout_predictions)],
            ),
            (
                "research.figures.concepts.calibration_supplement",
                "calibration",
                "calibration_reliability",
                "calibration_reliability",
                ["--heldout-predictions", str(args.heldout_predictions)],
            ),
        )
        for module, folder, source_stem, target_stem, inputs in jobs:
            destination = scratch / folder
            _run(
                [sys.executable, "-m", module, *inputs, "--output-dir", str(destination)],
                root,
            )
            _copy_exports(
                destination,
                source_stem,
                output,
                target_stem,
                extensions=("pdf", "png") if target_stem == "01_study_map" else ("pdf", "svg", "png"),
            )

        oracle = scratch / "oracle"
        _run(
            [
                sys.executable,
                "-m",
                "research.figures.oracle_scenarios",
                "--study-csv",
                str(args.oracle_study_csv),
                "--heldout-csv",
                str(args.oracle_heldout_csv),
                "--output-dir",
                str(oracle),
                "--category",
                "overall",
                "--classifier-wh",
                str(args.classifier_wh),
            ],
            root,
        )
        for stem in ("oracle_tradeoff", "oracle_retries"):
            _copy_exports(oracle, stem, output, stem)

        study = scratch / "study"
        _run(["latex", "-interaction=nonstopmode", "-halt-on-error", "study_overview.tex"], study)
        _run(
            [
                "dvisvgm",
                "--font-format=woff",
                "--bbox=papersize",
                "--output=study_overview.svg",
                "study_overview.dvi",
            ],
            study,
        )
        (output / "01_study_map.svg").write_text(
            _stable_font_timestamps((study / "study_overview.svg").read_text(encoding="utf-8")),
            encoding="utf-8",
        )

    missing = [
        f"{stem}.{extension}"
        for stem in FIGURES
        for extension in ("pdf", "svg", "png")
        if not (output / f"{stem}.{extension}").is_file()
    ]
    if missing:
        raise RuntimeError("Figure reproduction is incomplete: " + ", ".join(missing))
    print(output)


if __name__ == "__main__":
    main()
