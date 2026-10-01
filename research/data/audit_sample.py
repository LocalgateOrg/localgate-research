"""Draw a stratified conversion audit and prepare annotation packages."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from research.analysis.data_io import InputError, read_records, sha256

SEED = 20260806
DECISION_BAND = range(4, 8)
BORDERLINE_QUOTA = 50
MIN_PER_STRATUM = 12
BARE_NUMBER = re.compile(r"^[-+]?[\d.,]+$")
SENTENCE_FRAGMENT = re.compile(r"^(?:and|or|but|which|that|because|greater|less|more)\b", re.IGNORECASE)
ANNOTATOR_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def answer_is_suspect(record: dict) -> str:
    """Return a descriptive flag when a reference answer may not stand alone."""
    answer = record["reference_answer"].strip()
    if BARE_NUMBER.match(answer):
        return "bare_number"
    if SENTENCE_FRAGMENT.match(answer):
        return "fragment"
    if len(answer.split()) <= 2 and len(answer) <= 12:
        return "very_short"
    return ""


def stratum_of(record: dict) -> str:
    """Place an item in its conversion-stage or decision-band stratum."""
    if record.get("score") in DECISION_BAND:
        return "borderline"
    if record["convertible"]:
        return "kept_judge" if record["stage"] == "judge" else "kept_rescore"
    return "drop_rewrite" if record["stage"] == "rewrite" else "drop_rescore"


def allocate(populations: dict[str, int], total: int) -> dict[str, int]:
    """Allocate fixed borderline and square-root-weighted stratum quotas."""
    fixed = {"borderline": min(BORDERLINE_QUOTA, populations["borderline"])} if "borderline" in populations else {}
    remaining = {name: count for name, count in populations.items() if name not in fixed}
    budget = max(0, total - sum(fixed.values()))
    weights = {name: count**0.5 for name, count in remaining.items()}
    scale = budget / sum(weights.values()) if weights else 0
    draw = {name: max(MIN_PER_STRATUM, round(weight * scale)) for name, weight in weights.items()}
    draw.update(fixed)
    return {name: min(count, populations[name]) for name, count in draw.items()}


def blind_id(index: int) -> str:
    """Return the stable blind identifier for an audit row."""
    return f"A{index + 1:03d}"


def _validate_records(records: list[dict], corpus: Path) -> None:
    required = {
        "question_id",
        "question",
        "reference_answer",
        "convertible",
        "stage",
        "reason",
        "failure",
        "open_question",
    }
    seen: set[object] = set()
    duplicates: list[object] = []
    for number, record in enumerate(records, 1):
        missing = required - record.keys()
        if missing:
            raise InputError(f"corpus row {corpus}:{number} is missing {sorted(missing)}")
        question_id = record["question_id"]
        if question_id in seen:
            duplicates.append(question_id)
        seen.add(question_id)
    if not records:
        raise InputError(f"no records in {corpus}")
    if duplicates:
        raise InputError(f"duplicate question_id values in {corpus}: {sorted(set(duplicates))[:5]}")


def _load_options(labels: Path) -> dict[int, list[str]]:
    options: dict[int, list[str]] = {}
    for number, row in enumerate(read_records(labels), 1):
        missing = {"question_id", "options"} - row.keys()
        if missing:
            raise InputError(f"label row {labels}:{number} is missing {sorted(missing)}")
        question_id = row["question_id"]
        if question_id in options:
            raise InputError(f"duplicate question_id {question_id} in {labels}")
        if not isinstance(row["options"], list):
            raise InputError(f"options must be a list: {labels}:{number}")
        options[question_id] = row["options"]
    return options


def draw_sample(records: list[dict], total: int, seed: int = SEED) -> tuple[list[dict], dict[str, int], dict[str, int]]:
    """Draw and blind-order the sample with the fixed stratum allocation."""
    if total < 1:
        raise InputError("n must be positive")
    by_stratum: dict[str, list[dict]] = {}
    for record in records:
        by_stratum.setdefault(stratum_of(record), []).append(record)
    populations = {name: len(items) for name, items in by_stratum.items()}
    quota = allocate(populations, total)
    rng = random.Random(seed)
    audit: list[dict] = []
    for name, items in sorted(by_stratum.items()):
        audit.extend(rng.sample(items, quota[name]))
    rng.shuffle(audit)
    return audit, populations, quota


def annotation_package(audit: list[dict], options: dict[int, list[str]], annotator: str, *, fidelity: bool = False) -> list[dict]:
    """Build one annotator's blind or rewrite-fidelity package."""
    if fidelity:
        task = [
            {
                "id": blind_id(index), "original": record["question"],
                "options": options[record["question_id"]], "reference_answer": record["reference_answer"],
                "rewritten": record["open_question"], "same_question": "", "self_contained": "", "why": "",
            }
            for index, record in enumerate(audit)
            if record["convertible"] and record["open_question"]
        ]
    else:
        task = [
            {
                "id": blind_id(index), "question": record["question"],
                "options": options[record["question_id"]], "reference_answer": record["reference_answer"],
                "converts": "", "answer_stands_alone": "", "why": "",
            }
            for index, record in enumerate(audit)
        ]
    random.Random(SEED + int(fidelity) + sum(map(ord, annotator))).shuffle(task)
    return task


def _package_files(annotator: str, task: list[dict], *, fidelity: bool) -> dict:
    suffix = "2" if fidelity else ""
    fields = ("same_question", "self_contained", "why") if fidelity else ("converts", "answer_stands_alone", "why")
    return {
        f"annotate{suffix}_{annotator}.json": task,
        f"verdicts{suffix}_{annotator}.json": {row["id"]: {field: "" for field in fields} for row in task},
    }


def write_verdict_template(path: Path, template: dict) -> None:
    """Write an empty verdict template when a completed one is absent."""
    if path.exists():
        print(f"  keeping existing {path.name}")
        return
    path.write_text(json.dumps(template, indent=1), encoding="utf-8")


def _write_draw_files(out: Path, files: dict) -> None:
    if out.exists() and not out.is_dir():
        raise InputError(f"audit output must be a directory: {out}")
    if any(out.glob("verdicts*.json")) and not (out / "KEY.json").exists():
        raise InputError(f"{out}: existing verdicts require their matching KEY.json")
    for name, payload in files.items():
        path = out / name
        if not path.exists():
            continue
        existing = json.loads(path.read_text(encoding="utf-8"))
        matches = isinstance(existing, dict) and set(existing) == set(payload) if name.startswith("verdicts") else existing == payload
        if not matches:
            raise InputError(f"{path} differs from this draw; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        path = out / name
        if name.startswith("verdicts"):
            write_verdict_template(path, payload)
        elif not path.exists():
            path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the seeded conversion-audit draw."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output/audit_sample"))
    parser.add_argument("--n", type=int, default=240, help="audit sample size")
    parser.add_argument("--annotators", nargs="+", default=["samuel", "noah", "federico"])
    args = parser.parse_args(argv)
    if not args.annotators or len(set(args.annotators)) != len(args.annotators):
        raise InputError("annotators must be a non-empty list of distinct identifiers")
    invalid_annotators = [name for name in args.annotators if not ANNOTATOR_IDENTIFIER.fullmatch(name)]
    if invalid_annotators:
        raise InputError(f"annotator identifiers may use only letters, digits, underscores, and hyphens: {invalid_annotators}")

    records = read_records(args.corpus)
    _validate_records(records, args.corpus)
    audit, populations, quota = draw_sample(records, args.n)
    options = _load_options(args.labels)
    missing = {record["question_id"] for record in audit} - options.keys()
    if missing:
        raise InputError(f"{len(missing)} audited items are missing from {args.labels}: {sorted(missing)[:5]}")
    kept = [record for record in audit if record["convertible"]]
    flagged = sum(bool(answer_is_suspect(record)) for record in kept)
    print(f"  {len(records):,} records -> {len(audit)} audited")
    print(f"  {'stratum':14s} {'population':>10s} {'drawn':>6s} {'weight':>8s}")
    for name in sorted(populations):
        print(f"  {name:14s} {populations[name]:10,} {quota[name]:6d} {populations[name] / len(records):8.1%}")
    print(f"\n  reference answers flagged as not standing alone: {flagged} of {len(kept)} kept items in the sample")
    files = {
        "KEY.json": {
            blind_id(index): {
                "question_id": record["question_id"], "stratum": stratum_of(record),
                "convertible": record["convertible"], "stage": record["stage"], "score": record.get("score"),
                "reason": record["reason"], "failure": record["failure"], "answer_flag": answer_is_suspect(record),
            }
            for index, record in enumerate(audit)
        },
        "strata.json": populations,
        "provenance.json": {
            "corpus": str(args.corpus), "corpus_sha256": sha256(args.corpus), "n_records": len(records),
            "labels_sha256": sha256(args.labels), "seed": SEED, "requested_n": args.n, "quotas": quota,
            "model": records[0].get("model"), "prompts": records[0].get("prompts"),
        },
    }
    for annotator in args.annotators:
        files.update(_package_files(annotator, annotation_package(audit, options, annotator), fidelity=False))
    retained = [record for record in audit if record["convertible"] and record["open_question"]]
    for annotator in args.annotators:
        files.update(_package_files(annotator, annotation_package(audit, options, annotator, fidelity=True), fidelity=True))
    _write_draw_files(args.out, files)
    print(
        f"  wrote {args.out}/ — KEY.json, strata.json, provenance.json, {len(args.annotators)} blind packages "
        f"({len(audit)} items), {len(args.annotators)} rewrite-fidelity packages ({len(retained)} items), "
        "and a verdicts template per package"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
