"""Summarise participant forecasts and run the human-study statistical analyses."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from statistics import median

from research.analysis import (
    human_background,
    human_descriptive,
    human_inference,
    human_robustness,
)


def _prepare_output(path: Path) -> Path:
    destination = Path(path)
    if not destination.is_absolute():
        raise ValueError("output directory must be explicit and absolute")
    if destination.exists() or not destination.parent.is_dir():
        raise ValueError(
            "output directory must be new and its parent must already exist"
        )
    if os.stat(destination.parent).st_mode & 0o077:
        raise ValueError(
            "output parent directory must be private (mode 0700 or stricter)"
        )
    destination.mkdir(mode=0o700)
    return destination


def _indexed_rows(path: Path) -> dict[str, dict[str, object]]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) or "question_id" not in row for row in rows):
        raise human_descriptive.ValidationError(
            f"{path.name}: every row must contain question_id"
        )
    indexed = {str(row["question_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise human_descriptive.ValidationError(f"duplicate question ID in {path.name}")
    return indexed


def _sample_paths(
    screening: Path | None,
    corpus: Path | None,
    allocation: Path | None,
) -> tuple[Path, Path, Path] | None:
    if screening is None and corpus is None and allocation is None:
        return None
    if screening is None or corpus is None or allocation is None:
        raise human_descriptive.ValidationError(
            "screening, corpus, and allocation must be supplied together"
        )
    return Path(screening), Path(corpus), Path(allocation)


def _timing_summary(
    dataset: human_descriptive.Dataset,
    responses_dir: Path,
) -> tuple[dict[str, object], dict[str, int]]:
    all_times: list[int] = []
    item_times: list[int] = []
    block_times: list[int] = []
    fields = ("gender", "age", "education", "occupation", "genai_freq_post")
    present = Counter({field: 0 for field in fields})
    for filename in sorted(dataset.response_hashes):
        with (responses_dir / filename).open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            rows = list(csv.DictReader(handle))
        substantive: list[int] = []
        for row in rows:
            elapsed = int(row["response_time_ms"])
            all_times.append(elapsed)
            if not row["prompt_id"].startswith("ATT-"):
                substantive.append(elapsed)
        item_times.extend(substantive)
        block_times.append(sum(substantive))
        for field in fields:
            values = {row[field].strip() for row in rows}
            if len(values) != 1:
                raise human_descriptive.ValidationError(
                    f"{filename}: inconsistent background field: {field}"
                )
            present[field] += bool(next(iter(values)))
    return (
        {
            "median_substantive_seconds": median(item_times) / 1000,
            "median_including_attention_seconds": median(all_times) / 1000,
            "median_sum_substantive_minutes": median(block_times) / 60000,
            "note": (
                "Sum of recorded item times, not independently observed wall-clock "
                "duration."
            ),
        },
        dict(present),
    )


def _sample_report(
    dataset: human_descriptive.Dataset,
    responses_dir: Path,
    labels: Path,
    screening: Path,
    corpus: Path,
    allocation: Path,
) -> dict[str, object]:
    screened = _indexed_rows(screening)
    drawn = _indexed_rows(corpus)
    if set(drawn) != set(dataset.labels) or not set(drawn) <= set(screened):
        raise human_descriptive.ValidationError(
            "study corpus, labels, and screening IDs do not join exactly"
        )

    transitions = [[0] * 6 for _ in range(6)]
    actual: Counter[tuple[str, str]] = Counter()
    for question_id, label in dataset.labels.items():
        screened_row = screened[question_id]
        screened_k = screened_row.get("k")
        if screened_k != 5:
            raise human_descriptive.ValidationError("screening k must equal 5")
        before = human_descriptive.truth_level(
            screened_row.get("correct_count"), screened_k
        )
        after = human_descriptive.truth_level(label.correct_count, label.k)
        transitions[before][after] += 1
        drawn_row = drawn[question_id]
        if drawn_row.get("bin") != before:
            raise human_descriptive.ValidationError(
                "draw bin differs from screening label"
            )
        category = drawn_row.get("category")
        if not isinstance(category, str):
            raise human_descriptive.ValidationError("draw category must be a string")
        actual[(category, str(before))] += 1

    with allocation.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    raw_allocation = plan.get("allocation") if isinstance(plan, dict) else None
    if not isinstance(raw_allocation, dict):
        raise human_descriptive.ValidationError(
            "allocation file is missing its allocation object"
        )
    cells: list[dict[str, object]] = []
    for category, bins in raw_allocation.items():
        if not isinstance(category, str) or not isinstance(bins, dict):
            raise human_descriptive.ValidationError(
                "allocation categories and bins are malformed"
            )
        for level, cell in bins.items():
            if not isinstance(level, str) or not isinstance(cell, dict):
                raise human_descriptive.ValidationError("allocation cell is malformed")
            if cell.get("realised") != actual.pop((category, level), 0):
                raise human_descriptive.ValidationError(
                    "allocation differs from observed draw"
                )
            if any(
                type(cell.get(field)) is not int
                for field in ("realised", "target", "pool")
            ):
                raise human_descriptive.ValidationError(
                    "allocation counts must be exact integers"
                )
            cells.append(cell)
    if actual or sum(int(cell["realised"]) for cell in cells) != len(drawn):
        raise human_descriptive.ValidationError("allocation coverage mismatch")

    timing, background_presence = _timing_summary(dataset, responses_dir)
    return {
        "schema_version": 1,
        "analysis": "human-study sample allocation and response timing",
        "provenance": {
            "sources": {
                role: {"file": path.name, "sha256": human_descriptive.sha256(path)}
                for role, path in (
                    ("study_labels", labels),
                    ("screening_labels", screening),
                    ("study_corpus", corpus),
                    ("allocation", allocation),
                )
            },
            "response_files": [
                {"path": name, "sha256": dataset.response_hashes[name]}
                for name in sorted(dataset.response_hashes)
            ],
        },
        "timing": timing,
        "background_field_present_n": background_presence,
        "screening_to_k30_level_matrix": transitions,
        "same_level_n": sum(transitions[level][level] for level in range(6)),
        "same_binary_label_n": sum(
            transitions[before][after]
            for before in range(6)
            for after in range(6)
            if (before >= 3) == (after >= 3)
        ),
        "allocation": {
            "cells": len(cells),
            "below_target": sum(
                int(cell["realised"]) < int(cell["target"]) for cell in cells
            ),
            "above_target": sum(
                int(cell["realised"]) > int(cell["target"]) for cell in cells
            ),
            "pool_below_target": sum(
                int(cell["pool"]) < int(cell["target"]) for cell in cells
            ),
        },
    }


def run(
    responses_dir: Path,
    labels: Path,
    output_dir: Path,
    *,
    screening: Path | None = None,
    corpus: Path | None = None,
    allocation: Path | None = None,
) -> list[Path]:
    responses_dir, labels = Path(responses_dir), Path(labels)
    sample_paths = _sample_paths(screening, corpus, allocation)
    dataset = human_descriptive.load_and_validate(responses_dir, labels)
    sample_report = (
        _sample_report(dataset, responses_dir, labels, *sample_paths)
        if sample_paths is not None
        else None
    )

    output = _prepare_output(output_dir)
    descriptive_path = output / "human_descriptive.json"
    inference_path = output / "human_inference.json"
    robustness_path = output / "human_robustness.json"
    background_path = output / "human_background.json"

    descriptive_report = human_descriptive.build_report(dataset)
    human_descriptive.write_report(descriptive_report, descriptive_path)
    pairs = human_descriptive.prompt_pairs(descriptive_report)
    human_descriptive.write_report(
        human_inference.build_report(pairs),
        inference_path,
    )
    study = human_robustness.study_from_report(descriptive_report)
    human_descriptive.write_report(
        human_robustness.build_report(study),
        robustness_path,
    )
    human_descriptive.write_report(
        human_background.build_report(
            dataset,
            descriptive_report,
            responses_dir,
        ),
        background_path,
    )
    outputs = [descriptive_path, inference_path, robustness_path, background_path]
    if sample_report is not None:
        sample_path = output / "human_sample.json"
        human_descriptive.write_report(sample_report, sample_path)
        outputs.append(sample_path)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses-dir", required=True, type=Path,
        help="directory of participant response CSV files",
    )
    parser.add_argument(
        "--labels", required=True, type=Path,
        help="study label JSONL matched to the participant responses",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="new absolute private directory for human analysis JSON reports",
    )
    parser.add_argument(
        "--screening", type=Path,
        help="screening label JSONL; requires --corpus and --allocation",
    )
    parser.add_argument(
        "--corpus", type=Path,
        help="drawn study corpus JSONL; requires --screening and --allocation",
    )
    parser.add_argument(
        "--allocation", type=Path,
        help="study-draw allocation report JSON; requires --screening and --corpus",
    )
    args = parser.parse_args(argv)
    try:
        outputs = run(
            args.responses_dir,
            args.labels,
            args.output_dir,
            screening=args.screening,
            corpus=args.corpus,
            allocation=args.allocation,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
