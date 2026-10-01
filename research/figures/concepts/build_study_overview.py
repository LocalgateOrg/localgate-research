"""Render the study map from the active human and classifier input contracts."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from research.figures.style import prepare_output_directory
from research.figures.inputs import load_classifier_predictions, load_human_matrix



def run(command: list[str], output: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=output,
        capture_output=True,
        text=True,
        env={**os.environ, "SOURCE_DATE_EPOCH": "0", "FORCE_SOURCE_DATE": "1"},
    )
    if result.returncode:
        raise RuntimeError(
            f"{command[0]} failed with exit code {result.returncode}:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-report", type=Path, required=True)
    parser.add_argument("--heldout-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    matrix, study_generations = load_human_matrix(args.human_report)
    heldout, screening_generations = load_classifier_predictions(args.heldout_predictions)
    values = {
        "PAIRED": sum(map(sum, matrix)),
        "REMAINDER": len(heldout),
        "STUDY_K": study_generations,
        "SCREENING_K": screening_generations,
    }
    if any(type(v) is not int or v <= 0 for v in values.values()):
        raise ValueError("All sample sizes and generation counts must be positive integers")
    output = prepare_output_directory(args.output_dir)
    for filename in ("study_icons.tex", "ICON_LICENSE.txt", "icon_sources.json"):
        shutil.copyfile(source / filename, output / filename)
    for filename in ("study_overview.tex",):
        text = (source / filename).read_text()
        for key, value in values.items():
            text = text.replace("@@" + key + "@@", f"{value:,}")
        if "@@" in text:
            raise ValueError(f"Unresolved template field in {filename}")
        (output / filename).write_text(text)
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "study_overview.tex"], output)
    run(["pdftoppm", "-singlefile", "-r", "300", "-png", "study_overview.pdf", "study_overview"], output)
    for suffix in (".aux", ".log"):
        (output / f"study_overview{suffix}").unlink(missing_ok=True)
    print(output)


if __name__ == "__main__":
    main()
