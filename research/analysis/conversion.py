"""Recompute the converter bakeoff from row-level decisions and blind labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from research.analysis.data_io import InputError, read_json, read_records, sha256, write_json


class ValidationError(ValueError):
    """The converter comparison inputs do not form the original blind design."""


def _blind_design(blind_order_path: Path, human_path: Path) -> tuple[dict[str, int], dict[str, str]]:
    order = read_json(blind_order_path)
    human = read_json(human_path)
    if not isinstance(order, dict) or not order:
        raise ValidationError("blind order must be a nonempty JSON object")
    if not isinstance(human, dict):
        raise ValidationError("human verdicts must be a JSON object")
    try:
        mapped = {str(blind): int(question_id) for blind, question_id in order.items()}
    except (TypeError, ValueError) as exc:
        raise ValidationError("blind order values must be integer question IDs") from exc
    if len(set(mapped.values())) != len(mapped):
        raise ValidationError("blind order maps more than one blind position to a question")
    missing = set(mapped) - set(human)
    if missing:
        raise ValidationError(f"human verdicts omit blind positions: {sorted(missing)[:5]}")
    verdicts: dict[str, str] = {}
    for blind in mapped:
        record = human[blind]
        verdict = record.get("verdict") if isinstance(record, dict) else None
        if verdict not in {"yes", "no", "borderline"}:
            raise ValidationError(f"invalid human verdict at blind position {blind}: {verdict!r}")
        verdicts[blind] = verdict
    return mapped, verdicts


def _run(path: Path, expected_ids: set[int]) -> tuple[dict[int, bool], dict]:
    rows = read_records(path)
    decisions: dict[int, bool] = {}
    models: set[str] = set()
    prompts: set[str] = set()
    for number, row in enumerate(rows, 1):
        question_id = row.get("question_id")
        convertible = row.get("convertible")
        if type(question_id) is not int or question_id in decisions:
            raise ValidationError(f"{path}:{number}: question_id must be a unique integer")
        if type(convertible) is not bool:
            raise ValidationError(f"{path}:{number}: convertible must be boolean")
        decisions[question_id] = convertible
        if isinstance(row.get("model"), str):
            models.add(row["model"])
        if isinstance(row.get("prompts"), str):
            prompts.add(row["prompts"])
    if set(decisions) != expected_ids:
        missing = sorted(expected_ids - set(decisions))[:5]
        extra = sorted(set(decisions) - expected_ids)[:5]
        raise ValidationError(f"{path}: decisions do not match blind design; missing={missing}, extra={extra}")
    if len(models) > 1 or len(prompts) > 1:
        raise ValidationError(f"{path}: rows mix model or prompt instruments")
    return decisions, {
        "file": path.name,
        "sha256": sha256(path),
        "rows": len(rows),
        "model": next(iter(models), None),
        "prompt_digest": next(iter(prompts), None),
    }


def _score(decisions: dict[int, bool], order: dict[str, int], human: dict[str, str]) -> dict:
    agreement = false_acceptance = false_rejection = borderline = 0
    for blind, question_id in order.items():
        verdict = human[blind]
        if verdict == "borderline":
            borderline += 1
            continue
        expected = verdict == "yes"
        actual = decisions[question_id]
        if actual == expected:
            agreement += 1
        elif actual:
            false_acceptance += 1
        else:
            false_rejection += 1
    scored = agreement + false_acceptance + false_rejection
    return {
        "scored": scored,
        "borderline_excluded": borderline,
        "agreement": agreement,
        "agreement_rate": agreement / scored,
        "false_acceptance": false_acceptance,
        "false_rejection": false_rejection,
        "kept_all_items": sum(decisions.values()),
        "total_items": len(decisions),
    }


def compare(
    blind_order_path: Path,
    human_path: Path,
    model_runs: list[tuple[str, Path, Path]],
) -> dict:
    order, human = _blind_design(blind_order_path, human_path)
    expected_ids = set(order.values())
    results = []
    prompt_digests: set[str] = set()
    for label, first_path, second_path in model_runs:
        first, first_meta = _run(first_path, expected_ids)
        second, second_meta = _run(second_path, expected_ids)
        for meta in (first_meta, second_meta):
            if meta["prompt_digest"]:
                prompt_digests.add(meta["prompt_digest"])
        flips = sum(first[item] != second[item] for item in expected_ids)
        results.append(
            {
                "label": label,
                "runs": [
                    {**first_meta, "score": _score(first, order, human)},
                    {**second_meta, "score": _score(second, order, human)},
                ],
                "between_run_flips": flips,
                "between_run_flip_rate": flips / len(expected_ids),
            }
        )
    if len(prompt_digests) > 1:
        raise ValidationError(f"converter runs span prompt instruments: {sorted(prompt_digests)}")
    nonborderline = sum(value != "borderline" for value in human.values())
    return {
        "schema_version": 1,
        "analysis": "blind converter agreement and repeat stability",
        "sample": {
            "items": len(order),
            "nonborderline_scored": nonborderline,
            "borderline_excluded": len(order) - nonborderline,
        },
        "instrument": {"prompt_digest": next(iter(prompt_digests), None)},
        "inputs": {
            "blind_order": {"file": blind_order_path.name, "sha256": sha256(blind_order_path)},
            "human_verdicts": {"file": human_path.name, "sha256": sha256(human_path)},
        },
        "models": results,
        "limitations": [
            "The preliminary comparison was AI-assisted and supervised by one author.",
            "Borderline reference judgments are excluded from agreement but retained in stability and keep totals.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-order", required=True, type=Path)
    parser.add_argument("--human-verdicts", required=True, type=Path)
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("LABEL", "RUN1", "RUN2"),
        required=True,
        help="model label and its two final decision JSON/JSONL files; repeat per model",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = compare(
            args.blind_order,
            args.human_verdicts,
            [(label, Path(first), Path(second)) for label, first, second in args.model],
        )
        write_json(args.output, report)
    except (OSError, InputError, ValidationError) as exc:
        parser.error(str(exc))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
