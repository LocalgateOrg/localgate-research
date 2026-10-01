"""Run the LocalGate research analyses."""

from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    "human": "research.analysis.human",
    "conversion": "research.analysis.conversion",
    "conversion-audit": "research.analysis.conversion_audit",
    "judge-calibration": "research.analysis.judge_calibration",
    "energy": "research.analysis.energy",
    "power": "research.analysis.power",
    "reference": "research.analysis.reference",
}


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Commands:
  human              Validate and analyse participant forecasts.
  conversion         Compare converter decisions with reference judgments.
  conversion-audit   Audit conversion decisions.
  judge-calibration  Analyse judge-calibration records.
  energy             Calculate routing electricity and answer-success scenarios.
  power              Run power analyses.
  reference          Analyse reference-label stability and simulate sampling variation.
  classifier         Analyse study and held-out classifier predictions.

Classifier forms:
  classifier format [OPTIONS]   Compare open- and multiple-choice-trained classifiers across formats.
  classifier metrics [OPTIONS]  Alias for `classifier [OPTIONS]`.
  classifier [OPTIONS]          Calculate classifier metrics directly.

Run `localgate-analysis COMMAND --help` for command-specific inputs.""",
    )
    parser.add_argument(
        "command",
        choices=(*COMMANDS, "classifier"),
    )
    if not arguments:
        parser.print_help()
        return 2
    if arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command = arguments.pop(0)
    if command not in COMMANDS and command != "classifier":
        parser.error(
            f"unknown command {command!r}; choose from "
            + ", ".join((*COMMANDS, "classifier"))
        )
    if command == "classifier":
        if arguments and arguments[0] == "format":
            from research.analysis import format_comparison

            return format_comparison.main(arguments[1:])
        if arguments and arguments[0] == "metrics":
            arguments = arguments[1:]
        from research.analysis import classifier

        return classifier.main(arguments)
    module = importlib.import_module(COMMANDS[command])
    return int(module.main(arguments) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
