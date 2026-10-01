"""Draw study prompts across categories and observed success-count bands."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from research.analysis.data_io import InputError, read_object, read_records

PER_CATEGORY = 20
BINS = 6
MMLU_PRO_CATEGORIES = (
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
)


def bin_targets(per_category: int) -> list[int]:
    """Return the target count for each of the six success-count bands."""
    base, extra = divmod(per_category, BINS)
    targets = [base] * BINS
    boundary_first = (2, 3, 1, 4, 0, 5)
    for index in range(extra):
        targets[boundary_first[index]] += 1
    return targets


def _require_unique_ids(rows: list[dict], path: Path) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for number, row in enumerate(rows, 1):
        if "question_id" not in row:
            raise InputError(f"missing question_id: {path}:{number}")
        question_id = str(row["question_id"])
        if question_id in seen:
            duplicates.append(question_id)
        seen.add(question_id)
    if duplicates:
        raise InputError(f"duplicate question_id values in {path}: {sorted(set(duplicates))[:5]}")


def load_labels(path: Path) -> list[dict]:
    """Read panel labels and require the open-corpus five-repeat schema."""
    rows = read_records(path)
    _require_unique_ids(rows, path)
    for number, row in enumerate(rows, 1):
        if row.get("k") != 5:
            raise InputError(
                f"question_id {row.get('question_id')} at {path}:{number}: k={row.get('k')} "
                "does not provide five-repeat screening counts"
            )
        missing = {"category", "correct_count", "no_answer_count"} - row.keys()
        if missing:
            raise InputError(
                f"question_id {row.get('question_id')} at {path}:{number} is missing "
                f"{sorted(missing)}"
            )
        if row["correct_count"] not in range(BINS):
            raise InputError(
                f"question_id {row.get('question_id')} at {path}:{number} has "
                f"correct_count={row['correct_count']!r}, expected 0 through 5"
            )
    return rows


def restrict_to_partition(rows: list[dict], split_path: Path, partition: str) -> list[dict]:
    """Keep rows assigned to one split partition and record that partition."""
    split = read_object(split_path)
    assignment = split.get("assignment")
    if not isinstance(assignment, dict):
        raise InputError(f"split assignment object required: {split_path}")
    kept = []
    for row in rows:
        assigned = assignment.get(str(row["question_id"]))
        if assigned is None:
            raise InputError(
                f"question_id {row['question_id']} has no split assignment; labels and {split_path} disagree"
            )
        if assigned == partition:
            kept.append({**row, "split": assigned})
    if not kept:
        raise InputError(f"no labels fall in partition {partition!r}")
    return kept


def validate_study_categories(rows: list[dict]) -> None:
    """Require the complete fixed set of MMLU-Pro study categories."""
    categories = {row["category"] for row in rows}
    expected = set(MMLU_PRO_CATEGORIES)
    if categories != expected:
        missing = sorted(expected - categories)
        unexpected = sorted(categories - expected)
        raise InputError(
            "study categories must be the 14 MMLU-Pro categories; "
            f"missing={missing}, unexpected={unexpected}"
        )


def reject_input_output_collisions(outputs: tuple[Path, ...], inputs: tuple[Path, ...]) -> None:
    """Reject an output path that would replace a selected input file."""
    input_paths = {path.resolve() for path in inputs}
    collisions = [path for path in outputs if path.resolve() in input_paths]
    if collisions:
        raise InputError(f"output path overlaps an input: {', '.join(map(str, collisions))}")


def _nearest_donors(level: int) -> list[int]:
    """Order fallback bands by distance from the requested band."""
    return sorted(
        (band for band in range(BINS) if band != level),
        key=lambda band: (abs(band - level), abs(band - 2.5), band),
    )


def draw(
    rows: list[dict], per_category: int = PER_CATEGORY, seed: int = 20260814
) -> tuple[list[dict], dict]:
    """Draw prompts and report each category's realised band allocation."""
    if per_category < 1:
        raise InputError("per_category must be positive")
    rng = random.Random(seed)
    by_category: dict[str, dict[int, list[dict]]] = {}
    for row in rows:
        by_category.setdefault(row["category"], {}).setdefault(row["correct_count"], []).append(row)

    drawn: list[dict] = []
    allocation: dict[str, dict] = {}
    for category in sorted(by_category):
        pools = by_category[category]
        available = sum(len(members) for members in pools.values())
        if available < per_category:
            raise InputError(f"category {category} holds only {available} items; cannot draw {per_category}")
        targets = bin_targets(per_category)
        take = list(targets)
        for level in range(BINS):
            deficit = take[level] - len(pools.get(level, []))
            if deficit <= 0:
                continue
            take[level] -= deficit
            for donor in _nearest_donors(level):
                headroom = len(pools.get(donor, [])) - take[donor]
                if headroom <= 0:
                    continue
                moved = min(deficit, headroom)
                take[donor] += moved
                deficit -= moved
                if deficit == 0:
                    break
            if deficit:
                raise InputError(f"category {category} cannot satisfy its {per_category}-item allocation")
        allocation[category] = {
            str(level): {"target": targets[level], "realised": take[level], "pool": len(pools.get(level, []))}
            for level in range(BINS)
        }
        for level in range(BINS):
            members = sorted(pools.get(level, []), key=lambda row: row["question_id"])
            rng.shuffle(members)
            drawn.extend({**row, "bin": level} for row in members[: take[level]])

    return drawn, {
        "seed": seed,
        "per_category": per_category,
        "n_drawn": len(drawn),
        "bin_targets": bin_targets(per_category),
        "list_style_drawn": sum(bool(row.get("list_style")) for row in drawn),
        "allocation": allocation,
    }


def write_draw(drawn: list[dict], report: dict, out: Path, force: bool = False) -> None:
    """Write the prompt JSONL and its allocation report without accidental replacement."""
    report_path = out.with_name(out.stem + "_allocation.json")
    collisions = [path for path in (out, report_path) if path.exists()]
    if collisions and not force:
        raise InputError(f"output already exists: {', '.join(map(str, collisions))}; use --force to replace it")
    if out.exists() and out.is_dir():
        raise InputError(f"output path is a directory: {out}")
    if report_path.exists() and report_path.is_dir():
        raise InputError(f"allocation-report path is a directory: {report_path}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in drawn:
            handle.write(json.dumps(row) + "\n")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  wrote {out} ({len(drawn)} prompts) and {report_path}")


def main(argv: list[str] | None = None) -> int:
    """Run the seeded study-prompt draw."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output/study_prompts.jsonl"))
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--per-category", type=int, default=PER_CATEGORY)
    parser.add_argument("--partition", default="test", choices=("test", "validation", "train"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    report_path = args.out.with_name(args.out.stem + "_allocation.json")
    reject_input_output_collisions((args.out, report_path), (args.labels, args.split))
    rows = restrict_to_partition(load_labels(args.labels), args.split, args.partition)
    validate_study_categories(rows)
    drawn, report = draw(rows, args.per_category, args.seed)
    expected_drawn = len(MMLU_PRO_CATEGORIES) * args.per_category
    if len(drawn) != expected_drawn:
        raise InputError(f"study draw must contain {expected_drawn} prompts, got {len(drawn)}")
    if args.per_category == PER_CATEGORY and len(drawn) != 280:
        raise InputError(f"default study draw must contain 280 prompts, got {len(drawn)}")
    report["partition"] = args.partition
    write_draw(drawn, report, args.out, args.force)
    print(f"  seed {report['seed']}; bin targets {report['bin_targets']}")
    print(f"  list-style drawn (covariate): {report['list_style_drawn']}")
    for category, cells in report["allocation"].items():
        realised = [cells[str(level)]["realised"] for level in range(BINS)]
        note = "" if realised == report["bin_targets"] else "   <- redistributed"
        print(f"    {category:18s} {realised}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
