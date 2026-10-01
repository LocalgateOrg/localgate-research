"""Draw a fixed judge-calibration sample across response and category strata."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from research.analysis.data_io import InputError, read_records
from research.grading.judge import load_generation_items

SEED = 20260815
TOTAL = 100
PER_CATEGORY = 5
SHORTEST = 15
LONGEST = 10
LIST_STYLE = 5
ANNOTATORS = ("samuel", "federico", "noah")


def item_key(item: dict) -> str:
    """Return the question-and-generation-repeat identity for one response."""
    return f"{item['question_id']}:{item['rep']}"


def load_corpus_rows(path: Path) -> dict[int, dict]:
    """Read unique convertible corpus rows keyed by question identifier."""
    rows: dict[int, dict] = {}
    for number, row in enumerate(read_records(path), 1):
        if "question_id" not in row:
            raise InputError(f"missing question_id: {path}:{number}")
        if not row.get("convertible"):
            continue
        question_id = row["question_id"]
        if question_id in rows:
            raise InputError(f"duplicate convertible question_id {question_id} in {path}")
        rows[question_id] = row
    if not rows:
        raise InputError(f"no convertible records in {path}")
    return rows


def validate_items(items: list[dict], corpus_rows: dict[int, dict]) -> None:
    """Require each generation item to be unique and joined to the corpus."""
    if not items:
        raise InputError("no generation items were found")
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in items:
        missing = {"question_id", "rep", "question", "reference", "response"} - item.keys()
        if missing:
            raise InputError(f"generation item is missing {sorted(missing)}")
        key = item_key(item)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
        if item["question_id"] not in corpus_rows:
            raise InputError(f"generation item {key} has no convertible corpus row")
    if duplicates:
        raise InputError(f"duplicate generation identities: {sorted(set(duplicates))[:5]}")
    if len(items) < TOTAL:
        raise InputError(f"only {len(items)} generation items are available; calibration needs {TOTAL}")


def draw_calibration(items: list[dict], corpus_rows: dict[int, dict], seed: int = SEED) -> list[dict]:
    """Draw the fixed category, length, and list-style calibration sample."""
    rng = random.Random(seed)
    for item in items:
        item["list_style"] = bool(corpus_rows.get(item["question_id"], {}).get("list_style"))
    chosen: dict[str, dict] = {}

    def take(pool: list[dict], count: int) -> None:
        fresh = [item for item in pool if item_key(item) not in chosen]
        for item in fresh[:count]:
            chosen[item_key(item)] = item

    take(sorted(items, key=lambda item: len(item["response"])), SHORTEST)
    take(sorted(items, key=lambda item: -len(item["response"])), LONGEST)
    list_pool = [item for item in items if item["list_style"]]
    rng.shuffle(list_pool)
    take(list_pool, LIST_STYLE)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_category[item.get("category") or corpus_rows[item["question_id"]].get("category", "?")].append(item)
    for category in sorted(by_category):
        members = sorted(by_category[category], key=item_key)
        rng.shuffle(members)
        take(members, PER_CATEGORY)
    remainder = sorted(items, key=item_key)
    rng.shuffle(remainder)
    take(remainder, max(0, TOTAL - len(chosen)))
    return list(chosen.values())[:TOTAL]


def write_package(drawn: list[dict], out_dir: Path, seed: int = SEED) -> None:
    """Write the calibration rows and one randomised package per annotator."""
    if out_dir.exists() and not out_dir.is_dir():
        raise InputError(f"calibration output must be a directory: {out_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise InputError(f"calibration output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "items.jsonl"
    if items_path.exists():
        raise InputError(f"{items_path} exists; choose an unused output directory")
    with items_path.open("w", encoding="utf-8") as stream:
        for item in drawn:
            stream.write(
                json.dumps(
                    {
                        "key": item_key(item), "question_id": item["question_id"], "rep": item["rep"],
                        "question": item["question"], "reference": item["reference"], "response": item["response"],
                        "list_style": item["list_style"], "chars": len(item["response"]),
                    }
                )
                + "\n"
            )
    for annotator in ANNOTATORS:
        order = list(range(len(drawn)))
        random.Random(f"{seed}:{annotator}").shuffle(order)
        rows = [
            {
                "key": item_key(drawn[position]), "question": drawn[position]["question"],
                "reference_answer": drawn[position]["reference"], "model_response": drawn[position]["response"],
                "verdict": "", "note": "",
            }
            for position in order
        ]
        (out_dir / f"annotate_{annotator}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "README.md").write_text(
        f"""# Judge-calibration set ({len(drawn)} gradings)

Grade each `model_response` against `reference_answer` with exactly one verdict:

- `match` — its final answer states the same fact as the reference (paraphrase,
  other languages, extra correct detail all fine; numeric within 1%).
- `no_match` — it commits to a different/incomplete answer, or hedges between
  several.
- `no_answer` — it never commits to a final answer (cut off, wrong question, empty).

Judge only the final answer against the reference — do not re-solve the question.
This is the same rule the LLM judges are given; the point is to measure whether they
apply it the way a careful human does. Seed {seed}; drawn by
`localgate-data calibration-sample`.
""",
        encoding="utf-8",
    )
    print(f"  wrote {items_path} + {len(ANNOTATORS)} annotator files + README")


def main(argv: list[str] | None = None) -> int:
    """Run the seeded judge-calibration draw."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output/judge_calibration"))
    args = parser.parse_args(argv)
    corpus_rows = load_corpus_rows(args.corpus)
    items = load_generation_items(args.generations, args.corpus)
    validate_items(items, corpus_rows)
    drawn = draw_calibration(items, corpus_rows)
    write_package(drawn, args.out)
    lengths = sorted(len(item["response"]) for item in drawn)
    print(f"  drawn: {len(drawn)}  chars p0/p50/p100: {lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}")
    print(f"  list-style: {sum(item['list_style'] for item in drawn)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
