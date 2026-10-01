"""Prepare, train, predict, and measure the selected LocalGate classifier."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .localgate_research.config import TrainConfig, load_config


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="validate the six pinned source inputs")
    prepare.add_argument(
        "--data-dir", type=Path, required=True,
        help="directory containing the six pinned source data artifacts",
    )
    prepare.add_argument(
        "--output-dir", type=Path, required=True,
        help="directory for prepared artifacts; an existing matching preparation is reused",
    )

    train = commands.add_parser("train", help="fit the selected beta-binomial classifier")
    train.add_argument(
        "--manifest", type=Path, required=True,
        help="manifest.json produced by prepare; pins the training inputs",
    )
    train.add_argument(
        "--base-model", type=Path, required=True,
        help="local base-model artifact directory to verify and adapt",
    )
    train.add_argument(
        "--output-dir", type=Path, required=True,
        help="directory for the trained adapter and receipt artifacts",
    )
    train.add_argument(
        "--config", type=Path,
        help="optional JSON training recipe (default: selected TrainConfig settings)",
    )
    train.add_argument(
        "--resume", action="store_true",
        help="resume a compatible interrupted training artifact in --output-dir",
    )

    predict = commands.add_parser("predict", help="write study or held-out predictions")
    predict.add_argument(
        "--config", type=Path, required=True,
        help="published fit-record JSON or completed training receipt.json; not adapter_config.json",
    )
    predict.add_argument(
        "--base-model", type=Path, required=True,
        help="verified local base-model artifact directory",
    )
    predict.add_argument(
        "--adapter", type=Path, required=True,
        help="published adapter directory or selected checkpoint adapter directory",
    )
    predict.add_argument(
        "--data-dir", type=Path, required=True,
        help="evaluation directory containing the five pinned corpus, label and split files",
    )
    predict.add_argument(
        "--group", choices=("study", "heldout", "combined"), required=True,
        help="study: 280 questions; heldout: 1,375 other test questions; combined: all 1,655 test questions",
    )
    predict.add_argument(
        "--output", type=Path, required=True, help="new predictions JSONL artifact",
    )
    predict.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu",
        help="inference device (default: cpu)",
    )

    commands.add_parser("measure", help="measure CPU inference energy from explicit inputs")
    return result


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "measure":
        from .measure_energy import main as measure

        return measure(raw[1:])
    arguments = parser().parse_args(raw)
    if arguments.command == "prepare":
        from .localgate_research.data import prepare_data

        document = prepare_data(arguments.data_dir, arguments.output_dir)
        print(arguments.output_dir / "manifest.json")
        print(json.dumps({"dataset_id": document["dataset_id"], "counts": document["counts"]}))
        return 0
    if arguments.command == "train":
        from .localgate_research.prediction import verify_base_model
        from .localgate_research.training import train

        if not arguments.base_model.is_dir():
            raise SystemExit(f"base model directory does not exist: {arguments.base_model}")
        verify_base_model(arguments.base_model)
        config = load_config(arguments.config) if arguments.config else TrainConfig()
        config = replace(config, model_name=str(arguments.base_model))
        if config.head != "beta_binomial" or config.stage != "final":
            raise SystemExit("the train command requires head=beta_binomial and stage=final")
        receipt = train(
            config,
            arguments.manifest,
            arguments.output_dir,
            resume=arguments.resume,
        )
        print(arguments.output_dir / "receipt.json")
        print(json.dumps({"fit_id": receipt["fit_id"], "best_epoch": receipt["best_epoch"]}))
        return 0
    if arguments.command == "predict":
        from .localgate_research.prediction import run_prediction

        predictions = run_prediction(
            arguments.config,
            arguments.base_model,
            arguments.adapter,
            arguments.data_dir,
            arguments.group,
            arguments.output,
            device_name=arguments.device,
        )
        print(arguments.output)
        print(json.dumps({"group": arguments.group, "rows": len(predictions)}))
        return 0
    raise AssertionError(f"unsupported command: {arguments.command}")
