"""Compare a completed three-judge grading panel with three human annotators."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from research.analysis.data_io import InputError, read_records, sha256, write_json_new
from research.grading.panel import VERDICTS, build_panel, verdict_binary

PANEL_GATE = 0.80
JUDGE_GATE = 0.60
EXPECTED_ITEMS = 100
EXPECTED_QUESTIONS = 90


class ValidationError(ValueError):
    """Calibration inputs do not form a complete three-human, three-judge panel."""


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValidationError(f"--human values must be ANNOTATOR=PATH, got {value!r}")
        if name in paths:
            raise ValidationError(f"--human repeats annotator {name!r}")
        paths[name] = Path(raw_path)
    if len(paths) != 3:
        raise ValidationError("--human requires exactly three distinct annotators")
    if len({path.resolve() for path in paths.values()}) != 3:
        raise ValidationError("the gate requires three distinct human files")
    return paths


def load_items(
    path: Path, expected_items: int, expected_questions: int
) -> dict[str, dict]:
    """Load and validate the released calibration stimulus manifest."""
    if expected_items < 1 or expected_questions < 1:
        raise ValidationError("expected item and question counts must be positive")
    rows = read_records(path)
    if len(rows) != expected_items:
        raise ValidationError(
            f"calibration manifest has {len(rows)} items, expected {expected_items}"
        )
    items: dict[str, dict] = {}
    question_ids: set[int] = set()
    for number, row in enumerate(rows, 1):
        key = row.get("key")
        question_id = row.get("question_id")
        repeat = row.get("rep")
        if not isinstance(key, str) or not key:
            raise ValidationError(f"{path}:{number}: missing string key")
        if (
            type(question_id) is not int
            or question_id < 0
            or type(repeat) is not int
            or repeat < 0
        ):
            raise ValidationError(
                f"{path}:{number}: question_id and rep must be exact nonnegative integers"
            )
        if key != f"{question_id}:{repeat}":
            raise ValidationError(
                f"{path}:{number}: key {key!r} disagrees with question_id/rep"
            )
        if key in items:
            raise ValidationError(f"{path}: duplicate calibration key {key}")
        for field in ("question", "reference", "response"):
            if not isinstance(row.get(field), str):
                raise ValidationError(f"{path}:{number}: {field} must be a string")
        if not row["question"].strip() or not row["reference"].strip():
            raise ValidationError(
                f"{path}:{number}: question and reference must be non-empty"
            )
        if type(row.get("list_style")) is not bool:
            raise ValidationError(f"{path}:{number}: list_style must be boolean")
        if type(row.get("chars")) is not int or row["chars"] != len(row["response"]):
            raise ValidationError(
                f"{path}:{number}: chars does not match the response length"
            )
        items[key] = row
        question_ids.add(question_id)
    if len(question_ids) != expected_questions:
        raise ValidationError(
            f"calibration manifest spans {len(question_ids)} questions, "
            f"expected {expected_questions}"
        )
    return items


def load_human(path: Path) -> dict[str, str]:
    verdicts: dict[str, str] = {}
    for number, row in enumerate(read_records(path), 1):
        key = row.get("key")
        if not isinstance(key, str) or not key:
            raise ValidationError(f"{path}:{number}: missing string key")
        if key in verdicts:
            raise ValidationError(f"{path}: duplicate verdict for key {key}")
        verdict = row.get("verdict")
        if verdict not in VERDICTS:
            raise ValidationError(
                f"{path}: key {key} has verdict {verdict!r}, not one of {VERDICTS}"
            )
        verdicts[key] = verdict
    if not verdicts:
        raise ValidationError(f"{path} holds no human verdicts")
    return verdicts


def human_consensus(
    human_paths: dict[str, Path], items: dict[str, dict]
) -> tuple[dict[str, int], dict[str, str], int, dict[str, dict[str, str]]]:
    panel = {name: load_human(path) for name, path in sorted(human_paths.items())}
    expected_keys = set(items)
    missing = {
        name: {
            "missing": sorted(expected_keys - set(rows))[:5],
            "extra": sorted(set(rows) - expected_keys)[:5],
        }
        for name, rows in panel.items()
        if set(rows) != expected_keys
    }
    if missing:
        raise ValidationError(f"human calibration differs from the item manifest: {missing}")

    binary_consensus: dict[str, int] = {}
    three_way: dict[str, str] = {}
    splits = 0
    for key in sorted(expected_keys):
        cast = [rows[key] for rows in panel.values()]
        binary_consensus[key] = 1 if sum(verdict_binary(value) for value in cast) >= 2 else 0
        top, count = Counter(cast).most_common(1)[0]
        if count >= 2:
            three_way[key] = top
        else:
            splits += 1
    return binary_consensus, three_way, splits, panel


def scotts_pi(pairs: list[tuple[int, int]]) -> float:
    """Two-rater Scott's pi over binary labels with pooled marginals."""
    if not pairs:
        raise ValidationError("Scott's pi requires at least one paired item")
    observed = sum(left == right for left, right in pairs) / len(pairs)
    pooled = [label for pair in pairs for label in pair]
    positive = sum(pooled) / len(pooled)
    expected = positive * positive + (1 - positive) * (1 - positive)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def confusion(pairs: list[tuple[int, int]]) -> dict:
    true_positive = sum(human == 1 and predicted == 1 for human, predicted in pairs)
    true_negative = sum(human == 0 and predicted == 0 for human, predicted in pairs)
    false_positive = sum(human == 0 and predicted == 1 for human, predicted in pairs)
    false_negative = sum(human == 1 and predicted == 0 for human, predicted in pairs)
    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": (true_positive + true_negative) / len(pairs),
    }


def score(
    human_binary: dict[str, int],
    human_three_way: dict[str, str],
    judge_files: list[Path],
) -> dict:
    panel = build_panel(judge_files, keys=set(human_binary))
    if set(panel) != set(human_binary):
        missing = sorted(set(human_binary) - set(panel))[:5]
        extra = sorted(set(panel) - set(human_binary))[:5]
        raise ValidationError(
            f"judge panel keys do not match human calibration; missing={missing}, extra={extra}"
        )
    keys = sorted(human_binary)
    judge_names = list(next(iter(panel.values()))["verdicts"])
    judges = {}
    failing_judges = []
    for name in judge_names:
        pairs = [
            (human_binary[key], verdict_binary(panel[key]["verdicts"][name])) for key in keys
        ]
        pi = scotts_pi(pairs)
        counts = confusion(pairs)
        three_way_confusion = Counter(
            (human_three_way[key], panel[key]["verdicts"][name])
            for key in keys
            if key in human_three_way
        )
        passes = pi >= JUDGE_GATE
        if not passes:
            failing_judges.append(name)
        judges[name] = {
            "scotts_pi": pi,
            "passes": passes,
            "confusion": counts,
            "three_way_confusion": {
                f"{human}->{judge}": count
                for (human, judge), count in sorted(three_way_confusion.items())
            },
        }

    panel_pairs = [(human_binary[key], panel[key]["correct"]) for key in keys]
    panel_pi = scotts_pi(panel_pairs)
    panel_counts = confusion(panel_pairs)

    human_marginal = Counter(human_binary.values())
    n_match, n_no_match = human_marginal[1], human_marginal[0]
    expected_ratio = n_no_match / n_match if n_match else 1.0
    lenient = [
        name
        for name, result in judges.items()
        if result["confusion"]["false_positive"] >= 8
        and result["confusion"]["false_positive"]
        > 2 * expected_ratio * max(result["confusion"]["false_negative"], 1)
    ]
    reasons = []
    if failing_judges:
        reasons.append(f"judges below {JUDGE_GATE}: {failing_judges}")
    if lenient:
        reasons.append(f"judges breach the preregistered leniency bound: {lenient}")
    if panel_pi < PANEL_GATE:
        reasons.append(f"panel Scott's pi {panel_pi} is below {PANEL_GATE}")
    return {
        "items": len(keys),
        "human_binary_marginal": {
            "match": human_marginal[1],
            "no_match_or_no_answer": human_marginal[0],
        },
        "judges": judges,
        "panel": {
            "scotts_pi": panel_pi,
            "passes": panel_pi >= PANEL_GATE,
            "confusion": panel_counts,
        },
        "gate": {
            "passes": not reasons,
            "panel_threshold": PANEL_GATE,
            "per_judge_threshold": JUDGE_GATE,
            "reasons": reasons,
        },
    }


def build_report(
    items_path: Path,
    human_paths: dict[str, Path],
    judge_files: list[Path],
    expected_items: int,
    expected_questions: int,
) -> dict:
    items = load_items(items_path, expected_items, expected_questions)
    human_binary, human_three_way, splits, human_panel = human_consensus(
        human_paths, items
    )
    result = score(human_binary, human_three_way, judge_files)
    result["human_three_way_splits"] = splits
    return {
        "schema_version": 1,
        "analysis": "automated grading panel calibration against human majority",
        "inputs": {
            "items": {
                "path": str(items_path),
                "sha256": sha256(items_path),
                "rows": len(items),
                "questions": len({row["question_id"] for row in items.values()}),
            },
            "human": {
                name: {
                    "path": str(path),
                    "sha256": sha256(path),
                    "rows": len(human_panel[name]),
                }
                for name, path in sorted(human_paths.items())
            },
            "judges": [
                {"path": str(path), "sha256": sha256(path)} for path in judge_files
            ],
        },
        "result": result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--human", action="append", required=True, metavar="ANNOTATOR=PATH"
    )
    parser.add_argument("--items", required=True, type=Path)
    parser.add_argument("--judges", nargs=3, required=True, type=Path)
    parser.add_argument("--expect-items", type=int, default=EXPECTED_ITEMS)
    parser.add_argument("--expect-questions", type=int, default=EXPECTED_QUESTIONS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        human_paths = parse_named_paths(args.human)
        report = build_report(
            args.items,
            human_paths,
            args.judges,
            args.expect_items,
            args.expect_questions,
        )
        write_json_new(args.output, report)
    except (OSError, InputError, ValidationError, ValueError) as exc:
        parser.error(str(exc))
    print(args.output)
    return 0 if report["result"]["gate"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
