"""Run the typed-decision-model study commands."""

from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    "items": "research.tdm.items",
    "baseline": "research.tdm.baseline",
    "numeric": "research.tdm.numeric",
    "decide": "research.tdm.client",
}


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Commands:
  items      Build per-stage item tables from the released datasets (task S5).
  baseline   Score the released pipeline against existing human labels (task B1).
  numeric    Check the numeric matcher against the calibration items (task I2).
  decide     Send an instrument to a TDM provider, or build requests with --dry-run (task S3).

Run `localgate-tdm COMMAND --help` for command-specific inputs.""",
    )
    parser.add_argument("command", choices=COMMANDS)
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0 if arguments else 2
    command = arguments.pop(0)
    if command not in COMMANDS:
        parser.error(f"unknown command {command!r}; choose from " + ", ".join(COMMANDS))
    module = importlib.import_module(COMMANDS[command])
    return int(module.main(arguments) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
