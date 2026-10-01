"""Recompute the two-stage conversion audit from the released row-level inputs."""

from __future__ import annotations

import argparse
import copy
import re
from collections import Counter
from pathlib import Path

from research.analysis.data_io import (
    InputError,
    read_object,
    read_records,
    sha256,
    write_json_new,
)
from research.grading.panel import fleiss_kappa

EXPECTED_STAGE1_ITEMS = 248
EXPECTED_STAGE2_ITEMS = 152
YES_NO = {"yes": True, "no": False}
VOTES = {"yes", "no", "borderline"}
KEPT_STRATA = ("kept_judge", "kept_rescore", "borderline_kept")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ValidationError(ValueError):
    """Audit inputs do not form the released three-annotator design."""


def corpus_stratum(row: dict) -> str:
    """Recover the audit stratum from one conversion-corpus row."""
    score = row.get("score")
    if type(score) is int and 4 <= score <= 7:
        return "borderline"
    if row["convertible"]:
        return "kept_judge" if row["stage"] == "judge" else "kept_rescore"
    return "drop_rewrite" if row["stage"] == "rewrite" else "drop_rescore"


def load_corpus(path: Path) -> tuple[dict[int, dict], dict[str, int], dict[str, int]]:
    """Index the supplied corpus and derive full and retained populations."""
    rows = read_records(path)
    indexed: dict[int, dict] = {}
    populations: Counter = Counter()
    retained: Counter = Counter()
    for number, row in enumerate(rows, 1):
        question_id = row.get("question_id")
        if type(question_id) is not int or question_id in indexed:
            raise ValidationError(f"{path}:{number}: question_id must be a unique integer")
        if type(row.get("convertible")) is not bool:
            raise ValidationError(f"{path}:{number}: convertible must be boolean")
        if row.get("stage") not in {"judge", "rescore", "rewrite"}:
            raise ValidationError(
                f"{path}:{number}: unknown conversion stage {row.get('stage')!r}"
            )
        stratum = corpus_stratum(row)
        indexed[question_id] = row
        populations[stratum] += 1
        if row["convertible"]:
            retained["borderline_kept" if stratum == "borderline" else stratum] += 1
    if not rows:
        raise ValidationError("conversion corpus is empty")
    if set(retained) != set(KEPT_STRATA):
        raise ValidationError(
            f"corpus retained strata are {sorted(retained)}, expected {KEPT_STRATA}"
        )
    return (
        indexed,
        dict(sorted(populations.items())),
        {name: retained[name] for name in KEPT_STRATA},
    )


def load_key(path: Path, corpus: dict[int, dict]) -> dict[str, dict]:
    key = read_object(path)
    if len(key) != EXPECTED_STAGE1_ITEMS:
        raise ValidationError(
            f"stage-1 key has {len(key)} items, expected {EXPECTED_STAGE1_ITEMS}"
        )
    seen_questions: set[int] = set()
    for blind_id, entry in key.items():
        if not isinstance(blind_id, str) or not isinstance(entry, dict):
            raise ValidationError("audit key must map string blind IDs to objects")
        question_id = entry.get("question_id")
        if type(question_id) is not int or question_id in seen_questions:
            raise ValidationError(f"{path}: duplicate or invalid question_id at {blind_id}")
        seen_questions.add(question_id)
        if question_id not in corpus:
            raise ValidationError(
                f"{path}: {blind_id} question_id {question_id} is absent from corpus"
            )
        source = corpus[question_id]
        expected = corpus_stratum(source)
        if (
            entry.get("stratum") != expected
            or entry.get("convertible") is not source["convertible"]
        ):
            raise ValidationError(
                f"{path}: {blind_id} does not match corpus identity "
                f"(stratum={expected}, convertible={source['convertible']})"
            )
    return key


def parse_named_paths(values: list[str], option: str) -> dict[str, Path]:
    named: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValidationError(f"{option} values must be ANNOTATOR=PATH, got {value!r}")
        if name in named:
            raise ValidationError(f"{option} repeats annotator {name!r}")
        named[name] = Path(raw_path)
    if len(named) != 3:
        raise ValidationError(f"{option} requires exactly three distinct annotators")
    if len({path.resolve() for path in named.values()}) != 3:
        raise ValidationError(f"{option} requires three distinct annotator files")
    return named


def _validate_vote(value: object, path: Path, blind_id: str, field: str) -> str:
    if value not in VOTES:
        raise ValidationError(
            f"{path}: {blind_id}.{field} is {value!r}, expected yes, no, or borderline"
        )
    return str(value)


def load_stage1(paths: dict[str, Path], expected_ids: set[str]) -> dict[str, dict[str, dict]]:
    panel = {}
    for annotator, path in sorted(paths.items()):
        rows = read_object(path)
        if set(rows) != expected_ids:
            missing = sorted(expected_ids - set(rows))[:5]
            extra = sorted(set(rows) - expected_ids)[:5]
            raise ValidationError(f"{path}: stage-1 keys differ; missing={missing}, extra={extra}")
        for blind_id, row in rows.items():
            if not isinstance(row, dict):
                raise ValidationError(f"{path}: {blind_id} must be an object")
            for field in ("converts", "answer_stands_alone"):
                _validate_vote(row.get(field), path, blind_id, field)
        panel[annotator] = rows
    return panel


def load_stage2(
    paths: dict[str, Path], expected_ids: set[str]
) -> dict[str, dict[str, dict]]:
    panel = {}
    for annotator, path in sorted(paths.items()):
        indexed: dict[str, dict] = {}
        for number, row in enumerate(read_records(path), 1):
            blind_id = str(row.get("key") or row.get("id") or "")
            if not blind_id:
                raise ValidationError(f"{path}:{number}: missing key")
            if blind_id in indexed:
                raise ValidationError(
                    f"{path}: duplicate key {blind_id}; raw event logs are not valid"
                )
            for field in ("same_question", "self_contained"):
                _validate_vote(row.get(field), path, blind_id, field)
            indexed[blind_id] = row
        if set(indexed) != expected_ids:
            missing = sorted(expected_ids - set(indexed))[:5]
            extra = sorted(set(indexed) - expected_ids)[:5]
            raise ValidationError(f"{path}: stage-2 keys differ; missing={missing}, extra={extra}")
        panel[annotator] = indexed
    return panel


def majority(votes: list[str]) -> bool | None:
    counts = Counter(vote for vote in votes if vote in YES_NO)
    for value, count in counts.items():
        if count >= 2:
            return YES_NO[value]
    return None


def analyse_stage1(
    key: dict[str, dict], populations: dict[str, int], panel: dict[str, dict[str, dict]]
) -> dict:
    per_stratum = {
        name: {"population": population, "scored": 0, "agree": 0}
        for name, population in populations.items()
    }
    ratings = []
    no_majority = 0
    for blind_id, entry in key.items():
        votes = [panel[name][blind_id]["converts"] for name in panel]
        ratings.append(Counter(votes))
        human = majority(votes)
        if human is None:
            no_majority += 1
            continue
        stats = per_stratum[entry["stratum"]]
        stats["scored"] += 1
        stats["agree"] += human == entry["convertible"]

    total_population = sum(populations.values())
    covered_population = sum(
        stats["population"] for stats in per_stratum.values() if stats["scored"]
    )
    for stats in per_stratum.values():
        stats["share"] = stats["population"] / total_population
        stats["agreement"] = stats["agree"] / stats["scored"] if stats["scored"] else None
    weighted_error = sum(
        (1 - stats["agreement"]) * stats["population"]
        for stats in per_stratum.values()
        if stats["scored"]
    ) / covered_population
    return {
        "items": len(key),
        "per_stratum": per_stratum,
        "no_yes_no_majority": no_majority,
        "population_weighted_error": weighted_error,
        "fleiss_kappa": fleiss_kappa(ratings),
        "complete_items": len(ratings),
    }


def retained_ids(key: dict[str, dict]) -> set[str]:
    return {
        blind_id
        for blind_id, entry in key.items()
        if entry["stratum"] in {"kept_judge", "kept_rescore"}
        or (entry["stratum"] == "borderline" and entry["convertible"])
    }


def apply_corrections(
    panel: dict[str, dict[str, dict]],
    paths: dict[str, Path],
    specifications: list[str],
) -> tuple[dict[str, dict[str, dict]], list[dict]]:
    corrected = copy.deepcopy(panel)
    records = []
    seen: set[tuple[str, str]] = set()
    for specification in specifications:
        parts = specification.split(":")
        if len(parts) != 3:
            raise ValidationError(
                "--reverse-column must be ANNOTATOR:FIELD:SOURCE_SHA256"
            )
        annotator, field, expected_digest = parts
        if annotator not in corrected:
            raise ValidationError(f"correction names unknown annotator {annotator!r}")
        if field not in {"same_question", "self_contained"}:
            raise ValidationError(f"correction names unknown stage-2 field {field!r}")
        if (annotator, field) in seen:
            raise ValidationError(f"duplicate correction for {annotator}:{field}")
        if not SHA256_RE.fullmatch(expected_digest):
            raise ValidationError("correction source SHA-256 must be 64 lowercase hex characters")
        actual_digest = sha256(paths[annotator])
        if actual_digest != expected_digest:
            raise ValidationError(
                f"correction source identity failed for {annotator}: "
                f"expected {expected_digest}, got {actual_digest}"
            )
        reversed_count = 0
        untouched_borderline = 0
        for row in corrected[annotator].values():
            if row[field] == "yes":
                row[field] = "no"
                reversed_count += 1
            elif row[field] == "no":
                row[field] = "yes"
                reversed_count += 1
            else:
                untouched_borderline += 1
        records.append(
            {
                "operation": "reverse_yes_no",
                "annotator": annotator,
                "field": field,
                "source_path": str(paths[annotator]),
                "source_sha256": actual_digest,
                "reversed_values": reversed_count,
                "borderline_values_unchanged": untouched_borderline,
                "source_modified": False,
            }
        )
        seen.add((annotator, field))
    return corrected, records


def _field_decision(error: float) -> dict:
    if error <= 0.05:
        return {"outcome": "pass", "threshold": 0.05}
    if error <= 0.10:
        return {"outcome": "repair", "threshold": 0.05}
    return {"outcome": "revise_filter", "threshold": 0.05}


def analyse_stage2(
    key: dict[str, dict], populations: dict[str, int], panel: dict[str, dict[str, dict]]
) -> dict:
    ids = retained_ids(key)
    fields = {}
    for field in ("same_question", "self_contained"):
        per_stratum = {
            name: {"population": populations[name], "scored": 0, "errors": 0}
            for name in KEPT_STRATA
        }
        ratings = []
        splits = 0
        failing = []
        for blind_id in sorted(ids):
            votes = [panel[name][blind_id][field] for name in panel]
            ratings.append(Counter(votes))
            human = majority(votes)
            if human is None:
                splits += 1
                continue
            name = (
                "borderline_kept"
                if key[blind_id]["stratum"] == "borderline"
                else key[blind_id]["stratum"]
            )
            stats = per_stratum[name]
            stats["scored"] += 1
            if not human:
                stats["errors"] += 1
                failing.append(
                    {"blind_id": blind_id, "question_id": key[blind_id]["question_id"]}
                )
        covered_population = sum(
            stats["population"] for stats in per_stratum.values() if stats["scored"]
        )
        weighted_error = sum(
            stats["errors"] / stats["scored"] * stats["population"]
            for stats in per_stratum.values()
            if stats["scored"]
        ) / covered_population
        fields[field] = {
            "per_stratum": per_stratum,
            "population_weighted_error": weighted_error,
            "fleiss_kappa": fleiss_kappa(ratings),
            "complete_items": len(ratings),
            "no_yes_no_majority": splits,
            "failing": failing,
            "decision": _field_decision(weighted_error),
        }
    return {"items": len(ids), "fields": fields}


def build_report(
    corpus_path: Path,
    key_path: Path,
    stage1_paths: dict[str, Path],
    stage2_paths: dict[str, Path],
    correction_specs: list[str],
) -> dict:
    corpus, populations, retained_populations = load_corpus(corpus_path)
    key = load_key(key_path, corpus)
    stage1 = load_stage1(stage1_paths, set(key))
    ids = retained_ids(key)
    if len(ids) != EXPECTED_STAGE2_ITEMS:
        raise ValidationError(
            f"stage-2 retained sample has {len(ids)} items, expected {EXPECTED_STAGE2_ITEMS}"
        )
    stage2 = load_stage2(stage2_paths, ids)
    corrected, corrections = apply_corrections(
        stage2, stage2_paths, correction_specs
    )

    def input_meta(path: Path, rows: int) -> dict:
        return {"path": str(path), "sha256": sha256(path), "rows": rows}

    return {
        "schema_version": 1,
        "analysis": "two-stage human audit of conversion and retained rewrites",
        "inputs": {
            "corpus": input_meta(corpus_path, len(corpus)),
            "key": input_meta(key_path, len(key)),
            "stage1": {
                name: input_meta(path, len(stage1[name]))
                for name, path in sorted(stage1_paths.items())
            },
            "stage2": {
                name: input_meta(path, len(stage2[name]))
                for name, path in sorted(stage2_paths.items())
            },
        },
        "populations": {
            "full_conversion_corpus": populations,
            "retained_conversion_corpus": retained_populations,
        },
        "corrections": corrections,
        "stage1": analyse_stage1(key, populations, stage1),
        "stage2": analyse_stage2(key, retained_populations, corrected),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument(
        "--stage1", action="append", required=True, metavar="ANNOTATOR=PATH"
    )
    parser.add_argument(
        "--stage2", action="append", required=True, metavar="ANNOTATOR=PATH"
    )
    parser.add_argument(
        "--reverse-column",
        action="append",
        default=[],
        metavar="ANNOTATOR:FIELD:SOURCE_SHA256",
        help="reverse yes/no in one in-memory stage-2 column after validating its source digest",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        stage1_paths = parse_named_paths(args.stage1, "--stage1")
        stage2_paths = parse_named_paths(args.stage2, "--stage2")
        if set(stage1_paths) != set(stage2_paths):
            raise ValidationError("stage-1 and stage-2 annotator names differ")
        report = build_report(
            args.corpus,
            args.key,
            stage1_paths,
            stage2_paths,
            args.reverse_column,
        )
        write_json_new(args.output, report)
    except (OSError, InputError, ValidationError, ValueError) as exc:
        parser.error(str(exc))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
