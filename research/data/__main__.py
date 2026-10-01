"""Prepare the question sets and reference data used in the LocalGate study."""

from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    "corpus": "corpus",
    "convert": "convert",
    "generate": "generate",
    "closed-labels": "closed_labels",
    "study-draw": "study_draw",
    "study-assign": "study_assign",
    "audit-sample": "audit_sample",
    "calibration-sample": "calibration_sample",
}


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Commands:
  corpus              Prepare or validate the pinned MMLU-Pro answer key.
  convert             Convert eligible corpus questions to the study format.
  generate            Generate model-response records for the selected corpus.
  closed-labels       Build the closed-corpus labels used by offline judging.
  study-draw          Draw the stratified human-study question set.
  study-assign        Create the participant assignment plan.
  audit-sample        Draw the manual audit sample.
  calibration-sample  Draw the judge-calibration sample.

Run `localgate-data COMMAND --help` for that command's inputs.""",
    )
    parser.add_argument("command", choices=COMMANDS)
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0 if arguments else 2
    command = arguments.pop(0)
    if command not in COMMANDS:
        parser.error(f"unknown command {command!r}")
    module = importlib.import_module(f"research.data.{COMMANDS[command]}")
    return int(module.main(arguments) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
