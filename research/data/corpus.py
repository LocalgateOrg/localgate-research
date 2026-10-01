"""Prepare and validate the MMLU-Pro answer key used for conversion."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from research.analysis.data_io import sha256

DATASET = "TIGER-Lab/MMLU-Pro"
SPLIT = "test"
KEY_FORMAT = "localgate-answer-key-v1"
SOURCE_IDENTITY_FIELDS = ("question", "options", "category")


def fetch(revision: str) -> list[dict]:
    """The fields the conversion needs, one row per source item."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "corpus preparation requires the optional 'datasets' package"
        ) from exc

    rows = load_dataset(DATASET, split=SPLIT, revision=revision)
    return [
        {
            "question_id": row["question_id"],
            "question": row["question"],
            "options": row["options"],
            "category": row["category"],
            "answer": row["answer"],
            "answer_index": row["answer_index"],
            "answer_text": row["options"][row["answer_index"]],
            "src": row["src"],
        }
        for row in rows
    ]


def index_rows(rows: list[dict], label: str) -> dict[int, dict]:
    indexed: dict[int, dict] = {}
    duplicates = []
    for row in rows:
        question_id = row["question_id"]
        if question_id in indexed:
            duplicates.append(question_id)
        indexed[question_id] = row
    if duplicates:
        raise SystemExit(
            f"{label} has {len(duplicates):,} duplicate question_id values "
            f"(first few: {duplicates[:5]})"
        )
    return indexed


def load(path: Path) -> dict[int, dict]:
    """The cached key, indexed by question id."""
    if not path.is_file():
        raise SystemExit(f"answer-key file does not exist: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise SystemExit(f"answer key contains no rows: {path}")
    identity_presence = [
        all(field in row for field in SOURCE_IDENTITY_FIELDS) for row in rows
    ]
    if any(identity_presence) and not all(identity_presence):
        raise SystemExit(f"answer key mixes legacy and source-bound rows: {path}")
    source_bound = all(identity_presence)

    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if source_bound:
        if not manifest_path.is_file():
            raise SystemExit(f"source-bound answer key is missing its manifest: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid answer-key manifest: {manifest_path}") from exc
        expected = {
            "format": KEY_FORMAT,
            "dataset": DATASET,
            "split": SPLIT,
            "rows": len(rows),
            "sha256": sha256(path),
        }
        if not isinstance(manifest, dict) or any(
            manifest.get(field) != value for field, value in expected.items()
        ):
            raise SystemExit(f"answer-key manifest does not match {path}")
        if not isinstance(manifest.get("revision"), str) or not re.fullmatch(
            r"[0-9a-f]{40}", manifest["revision"]
        ):
            raise SystemExit(f"answer-key manifest has no immutable revision: {manifest_path}")
    elif manifest_path.exists():
        raise SystemExit(
            f"answer-key manifest exists but {path} lacks full source identity fields"
        )
    return index_rows(rows, "answer key")


def prepare(out_path: Path, revision: str) -> int:
    """Fetch a pinned dataset revision and write the fields conversion needs."""
    if out_path.exists():
        raise SystemExit(f"output already exists: {out_path}")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit("dataset revision must be a full immutable commit SHA")
    manifest = out_path.with_suffix(out_path.suffix + ".manifest.json")
    if manifest.exists():
        raise SystemExit(f"output already exists: {manifest}")
    rows = fetch(revision)
    index_rows(rows, "fetched answer key")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in rows))
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "format": KEY_FORMAT,
                "dataset": DATASET,
                "split": SPLIT,
                "revision": revision,
                "rows": len(rows),
                "sha256": sha256(out_path),
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    return len(rows)


def check(answer_key_path: Path, labels_path: Path) -> int:
    """Verify source identity for prepared keys, or the legacy selected-answer join."""
    if not labels_path.is_file():
        raise SystemExit(f"labels file does not exist: {labels_path}")

    key = load(answer_key_path)
    labelled = [json.loads(line) for line in labels_path.read_text().splitlines() if line]
    labels = index_rows(labelled, "labels")
    missing = sorted(set(labels) - set(key))

    if missing:
        raise SystemExit(
            f"{len(missing):,} labelled items have no answer key (first few: {missing[:5]})"
        )

    unlabelled = sorted(set(key) - set(labels))
    if unlabelled:
        raise SystemExit(
            f"{len(unlabelled):,} answer-key items have no labelled row "
            f"(first few: {unlabelled[:5]})"
        )

    source_bound = all(
        all(field in answer for field in SOURCE_IDENTITY_FIELDS)
        for answer in key.values()
    )
    mismatched = 0
    for question_id, row in labels.items():
        answer = key[question_id]
        try:
            answer_index = answer["answer_index"]
            matches = (
                type(answer_index) is int
                and 0 <= answer_index < len(row["options"])
                and row["options"][answer_index] == answer["answer_text"]
            )
            if source_bound:
                matches = matches and all(
                    row.get(field) == answer[field] for field in SOURCE_IDENTITY_FIELDS
                )
        except (IndexError, KeyError, TypeError):
            matches = False
        mismatched += not matches
    if mismatched:
        raise SystemExit(
            f"{mismatched:,} items disagree with the cached source identity or answer; "
            "the labels and the key do not describe the same corpus"
        )
    if not source_bound:
        print(
            "WARNING: legacy answer key has no immutable-revision manifest or full "
            "source identity; validated IDs and selected answer text only"
        )
    return len(labelled)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="fetch and write the answer key")
    prepare_parser.add_argument("--out", type=Path, required=True)
    prepare_parser.add_argument(
        "--revision", required=True, help="full 40-character dataset commit SHA"
    )

    check_parser = commands.add_parser("check", help="validate an answer-key/labels join")
    check_parser.add_argument("--answer-key", type=Path, required=True)
    check_parser.add_argument("--labels", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        count = prepare(args.out, args.revision)
        print(f"wrote {count:,} rows -> {args.out}")
    else:
        count = check(args.answer_key, args.labels)
        print(f"validated {count:,} labelled rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
