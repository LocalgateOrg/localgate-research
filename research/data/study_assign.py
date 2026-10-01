"""Assign study prompts to raters with balanced category coverage."""

from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import random
import re
from pathlib import Path

from research.analysis.data_io import InputError, read_records
from research.data.study_draw import MMLU_PRO_CATEGORIES

PARTICIPANTS = 30
RATERS_PER_PROMPT = 3
PER_CATEGORY_PER_PARTICIPANT = 2
COLUMNS = ("participant_id", "prompt_id", "category", "which_of_the_following", "prompt_text")

ATTENTION_CATEGORY = "attention check"
ATTENTION_TEXT = (
    "This is an attention check, not a question. To show you are reading, "
    'please select "Very likely" for this item.'
)
ATTENTION_EXPECTED = "Very likely"

DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)")
MATH_CHARS = re.compile(r"[\\^_=]")


def looks_like_math(content: str) -> bool:
    """Distinguish mathematical dollar delimiters from ordinary prices."""
    stripped = content.strip()
    if not stripped:
        return False
    return not stripped[0].isdigit() or bool(MATH_CHARS.search(stripped))


def normalise_math(text: str) -> str:
    """Use the study platform's delimiters for dollar-delimited mathematics."""
    text = DISPLAY_MATH.sub(lambda match: rf"\[{match.group(1)}\]", text)

    def inline(match: re.Match[str]) -> str:
        content = match.group(1)
        return rf"\({content}\)" if looks_like_math(content) else match.group(0)

    return INLINE_MATH.sub(inline, text)


def _unique_records(rows: list[dict], key_name: str, path: Path) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for number, row in enumerate(rows, 1):
        if key_name not in row:
            raise InputError(f"missing {key_name}: {path}:{number}")
        value = str(row[key_name])
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise InputError(f"duplicate {key_name} values in {path}: {sorted(set(duplicates))[:5]}")


def load_prompts(prompts_path: Path, corpus_path: Path) -> list[dict]:
    """Join the seeded prompt draw to its open-question text."""
    drawn = read_records(prompts_path)
    corpus = read_records(corpus_path)
    _unique_records(drawn, "question_id", prompts_path)
    _unique_records(corpus, "question_id", corpus_path)
    text = {str(record["question_id"]): record for record in corpus}
    rows = []
    for number, item in enumerate(drawn, 1):
        missing = {"question_id", "category"} - item.keys()
        if missing:
            raise InputError(f"prompt row {prompts_path}:{number} is missing {sorted(missing)}")
        question_id = str(item["question_id"])
        if question_id not in text:
            raise InputError(f"drawn question {question_id} is not in {corpus_path}")
        question = str(text[question_id].get("open_question") or "").strip()
        if not question:
            raise InputError(f"drawn question {question_id} has an empty open_question")
        rendered = normalise_math(question)
        rows.append(
            {
                "prompt_id": question_id,
                "category": item["category"],
                "which_of_the_following": str(item.get("list_style", False)),
                "prompt_text": rendered,
                "bin": item.get("bin"),
                "correct_count": item.get("correct_count"),
                "math_rewritten": rendered != question,
            }
        )
    return rows


def assign(
    prompts: list[dict],
    rng: random.Random,
    participants: int = PARTICIPANTS,
    raters: int = RATERS_PER_PROMPT,
    per_participant: int = PER_CATEGORY_PER_PARTICIPANT,
) -> dict[int, list[dict]]:
    """Allocate each prompt to the requested number of distinct raters."""
    if participants < 1 or raters < 1 or per_participant < 1:
        raise InputError("participants, raters, and per_participant must be positive")
    by_category: dict[str, list[dict]] = collections.defaultdict(list)
    for row in prompts:
        by_category[row["category"]].append(row)
    if not by_category:
        raise InputError("the prompt draw is empty")
    plan: dict[int, list[dict]] = {participant: [] for participant in range(participants)}
    for category in sorted(by_category):
        items = sorted(by_category[category], key=lambda row: row["prompt_id"])
        count = len(items)
        if count * raters != participants * per_participant:
            raise InputError(
                f"{category}: {count} prompts x {raters} raters != {participants} participants "
                f"x {per_participant}"
            )
        if per_participant > count or count % per_participant:
            raise InputError(f"{category}: {per_participant} prompts per participant does not divide {count}")
        rng.shuffle(items)
        order = list(range(participants))
        rng.shuffle(order)
        for slot in range(count * raters):
            plan[order[slot // per_participant]].append(items[slot % count])
    return plan


def check_design(plan: dict[int, list[dict]], raters: int, per_participant: int) -> dict:
    """Validate prompt multiplicity, unique assignments, and category balance."""
    seen_by: dict[str, list[int]] = collections.defaultdict(list)
    expected_categories: set[str] | None = None
    for participant, rows in plan.items():
        identifiers = [row["prompt_id"] for row in rows]
        if len(set(identifiers)) != len(identifiers):
            raise InputError(f"participant {participant} sees a prompt twice")
        per_category = collections.Counter(row["category"] for row in rows)
        if expected_categories is None:
            expected_categories = set(per_category)
        if set(per_category) != expected_categories or any(
            count != per_participant for count in per_category.values()
        ):
            raise InputError(f"participant {participant} category counts: {dict(per_category)}")
        for prompt_id in identifiers:
            seen_by[prompt_id].append(participant)
    bad = {
        prompt_id: participants
        for prompt_id, participants in seen_by.items()
        if len(participants) != raters or len(set(participants)) != raters
    }
    if bad:
        raise InputError(f"prompts without {raters} distinct raters: {dict(list(bad.items())[:5])}")
    pairs = collections.Counter()
    for participants in seen_by.values():
        pairs.update(itertools.combinations(sorted(participants), 2))
    return {
        "prompts": len(seen_by),
        "raters_per_prompt": raters,
        "max_shared_prompts_between_two_raters": max(pairs.values()),
        "rater_pairs_that_share_any_prompt": len(pairs),
    }


def participant_rows(
    participant_id: str, rows: list[dict], rng: random.Random, attention: bool
) -> list[dict]:
    """Randomise one participant's prompts and insert the attention check."""
    ordered = list(rows)
    rng.shuffle(ordered)
    output = [{"participant_id": participant_id, **{key: row[key] for key in COLUMNS[1:]}} for row in ordered]
    if attention:
        position = rng.randrange(1, len(output) + 1)
        output.insert(
            position,
            {
                "participant_id": participant_id,
                "prompt_id": f"ATT-{participant_id}",
                "category": ATTENTION_CATEGORY,
                "which_of_the_following": "False",
                "prompt_text": ATTENTION_TEXT,
            },
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write one study-platform import file."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def reject_input_output_collisions(out: Path, participants: int, inputs: tuple[Path, ...]) -> None:
    """Reject generated CSV or manifest paths that overlap selected input files."""
    output_paths = [out, out / "manifest.json"]
    output_paths.extend(out / f"P{index + 1:02d}.csv" for index in range(participants))
    input_paths = {path.resolve() for path in inputs}
    collisions = [path for path in output_paths if path.resolve() in input_paths]
    if collisions:
        raise InputError(f"output path overlaps an input: {', '.join(map(str, collisions))}")


def validate_default_design(prompts: list[dict], plan: dict[int, list[dict]], stats: dict) -> None:
    """Require the fixed 280-prompt, 14-category, 30-rater study design."""
    categories = {row["category"] for row in prompts}
    if categories != set(MMLU_PRO_CATEGORIES):
        raise InputError("default assignment requires the 14 MMLU-Pro study categories")
    if len(prompts) != 280:
        raise InputError(f"default assignment requires 280 unique prompts, got {len(prompts)}")
    if len(plan) != PARTICIPANTS or any(len(rows) != 28 for rows in plan.values()):
        raise InputError("default assignment requires 30 participants with 28 substantive prompts each")
    if stats["prompts"] != 280 or stats["raters_per_prompt"] != RATERS_PER_PROMPT:
        raise InputError("default assignment requires three ratings for each of 280 prompts")


def main(argv: list[str] | None = None) -> int:
    """Run the seeded participant assignment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output/study_assignments"))
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--participants", type=int, default=PARTICIPANTS)
    parser.add_argument("--no-attention-check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    prompts = load_prompts(args.prompts, args.corpus)
    rng = random.Random(args.seed)
    plan = assign(prompts, rng, participants=args.participants)
    stats = check_design(plan, RATERS_PER_PROMPT, PER_CATEGORY_PER_PARTICIPANT)
    if args.participants == PARTICIPANTS:
        validate_default_design(prompts, plan, stats)
    reject_input_output_collisions(args.out, args.participants, (args.prompts, args.corpus))
    if args.out.exists() and not args.out.is_dir():
        raise InputError(f"assignment output must be a directory: {args.out}")
    if args.out.exists():
        existing = {path.name for path in args.out.iterdir()}
        if existing and not args.force:
            raise InputError(f"{args.out} already holds an assignment; use --force to replace it")
        expected = {"manifest.json", *(f"P{index + 1:02d}.csv" for index in range(args.participants))}
        if existing - expected:
            raise InputError("assignment directory contains other files; choose a fresh output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "participants": args.participants,
        "attention_check": not args.no_attention_check,
        "attention_expected": ATTENTION_EXPECTED,
        "design": stats,
        "math_rewritten": sorted(row["prompt_id"] for row in prompts if row["math_rewritten"]),
        "assignments": {},
    }
    for index in range(args.participants):
        participant_id = f"P{index + 1:02d}"
        rows = participant_rows(participant_id, plan[index], rng, attention=not args.no_attention_check)
        write_csv(args.out / f"{participant_id}.csv", rows)
        manifest["assignments"][participant_id] = [
            {
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                "bin": row["bin"],
                "correct_count": row["correct_count"],
            }
            for row in plan[index]
        ]
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    per_file = len(plan[0]) + (0 if args.no_attention_check else 1)
    print(f"  wrote {args.participants} files to {args.out}/ ({per_file} rows each) + manifest.json")
    print(
        f"  {stats['prompts']} prompts x {stats['raters_per_prompt']} raters; two raters share at most "
        f"{stats['max_shared_prompts_between_two_raters']} prompts"
    )
    print(
        f"  attention check: {'embedded' if not args.no_attention_check else 'none'}; math rewritten in "
        f"{len(manifest['math_rewritten'])} prompts; seed {args.seed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
