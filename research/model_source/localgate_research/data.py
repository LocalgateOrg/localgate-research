"""Prepare a fingerprinted development corpus without exposing protected outcomes."""

from __future__ import annotations

import csv
import json
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from .artifacts import atomic_json, identity, sha256_file, write_jsonl

SOURCE_HASHES = {
    "labels.jsonl": "596bb6b7782053e16fa9384f5e9d4d83bfce03507d614a387ee40cdc931656ca",
    "labels_open.jsonl": "7a8f03c18c1d657fdff12fc2480410122253c0726b2df627e5a7fc81600b58ec",
    "converted.jsonl": "be78cccedb7778e479f5daec83c75f56b8aeac42b4c4225c5b78a8827586d4fd",
    "study_prompts.jsonl": "91fff21aea39867e9b550f502ee2c6a102fa71003f3304f13dec4134739cba45",
    "split.json": "a0f57e29add58d82527713f66c139e35af319bafe795038a05925808e2434e38",
    "open_ended_raw_performance.csv": (
        "a674988f2318121ce6503d9ffea6da59abd76b7d2fc311093711c8fcee21a7e6"
    ),
}
TRAINING_EXCLUSIONS = {4413, 4418, 10298}
EXPECTED_COUNTS = {"train": 5848, "validation": 857, "development": 6705, "families": 6581}
ALLOWED_ROLES = {"train", "validation", "test", "excluded_stem_leak"}
ROW_FIELDS = {
    "question_id",
    "text",
    "correct_count",
    "k",
    "category",
    "list_style",
    "family_id",
    "split",
    "mc_correct_count",
}
NORMALIZATION = "nfkc_casefold_whitespace"


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _records(path: Path) -> Iterator[dict]:
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if line.startswith("version https://git-lfs.github.com/spec/"):
                raise ValueError(f"LFS pointer requires materialization: {path.name}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object in {path.name}:{line_number}")
            question_id = row.get("question_id")
            if type(question_id) is not int or question_id < 0:
                raise ValueError(f"Invalid question ID in {path.name}:{line_number}")
            if question_id in seen:
                raise ValueError(f"Duplicate question ID {question_id} in {path.name}")
            seen.add(question_id)
            yield row


def build_families(original_text: dict[int, str], converted_text: dict[int, str]) -> dict[int, str]:
    """Join both text relations transitively before removing any bridge questions."""
    if not set(converted_text) <= set(original_text):
        raise ValueError("Converted questions missing from original corpus")
    parent = {question_id: question_id for question_id in original_text}

    def find(question_id: int) -> int:
        while parent[question_id] != question_id:
            parent[question_id] = parent[parent[question_id]]
            question_id = parent[question_id]
        return question_id

    for texts in (original_text, converted_text):
        representative = {}
        for question_id in sorted(texts):
            text = normalize_text(texts[question_id])
            if not text:
                raise ValueError(f"Empty question text for {question_id}")
            previous = representative.setdefault(text, question_id)
            left, right = find(question_id), find(previous)
            parent[max(left, right)] = min(left, right)
    members = defaultdict(list)
    for question_id in sorted(parent):
        members[find(question_id)].append(question_id)
    return {question_id: identity(ids) for ids in members.values() for question_id in ids}


def _source_hashes(data_dir: Path) -> dict[str, str]:
    actual = {}
    for name, expected in SOURCE_HASHES.items():
        path = data_dir / name
        with path.open("rb") as stream:
            if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/"):
                raise ValueError(f"LFS pointer requires materialization: {name}")
        actual[name] = sha256_file(path)
        if actual[name] != expected:
            raise ValueError(f"Source file differs from the pinned dataset: {name}")
    return actual


def _count(row: dict, field: str, question_id: int) -> int:
    value = row.get(field)
    if type(value) is not int or not 0 <= value <= 5:
        raise ValueError(f"Invalid {field} for question {question_id}")
    return value


def _metadata(row: dict) -> dict:
    question_id = row["question_id"]
    if not isinstance(row.get("category"), str) or not row["category"].strip():
        raise ValueError(f"Missing category for question {question_id}")
    if type(row.get("list_style")) is not bool:
        raise ValueError(f"Invalid list_style for question {question_id}")
    return {"category": row["category"], "list_style": row["list_style"]}


def _validate_rows(rows: list[dict], manifest: dict) -> None:
    ids = [row["question_id"] for row in rows]
    if ids != sorted(set(ids)):
        raise ValueError("Development IDs must be unique and sorted")
    roles = manifest["role_ids"]
    if set(roles) != {"train", "validation"}:
        raise ValueError("Unexpected development roles")
    expected_ids = set(roles["train"]) | set(roles["validation"])
    if set(ids) != expected_ids or set(roles["train"]) & set(roles["validation"]):
        raise ValueError("Development role membership mismatch")
    original_roles = manifest["original_role_ids"]
    if set(original_roles) != ALLOWED_ROLES:
        raise ValueError("Unexpected original roles")
    original_ids = [question_id for role_ids in original_roles.values() for question_id in role_ids]
    if len(original_ids) != len(set(original_ids)):
        raise ValueError("Original roles overlap")
    for role in roles:
        if not set(roles[role]) <= set(original_roles[role]):
            raise ValueError("Development role changed from original assignment")
    excluded = set().union(*(set(values) for values in manifest["exclusions"].values()))
    if set(ids) & (set(original_roles["test"]) | set(manifest["study_ids"]) | excluded):
        raise ValueError("Protected or excluded question in development data")
    families = defaultdict(set)
    for row in rows:
        if set(row) != ROW_FIELDS:
            raise ValueError("Development row has missing or forbidden fields")
        question_id = row["question_id"]
        if (
            type(question_id) is not int
            or not isinstance(row["text"], str)
            or not row["text"].strip()
        ):
            raise ValueError("Invalid development question")
        if type(row["k"]) is not int or row["k"] != 5:
            raise ValueError(f"Expected five trials for question {question_id}")
        _count(row, "correct_count", question_id)
        _count(row, "mc_correct_count", question_id)
        _metadata(row)
        if row["split"] not in roles or question_id not in roles[row["split"]]:
            raise ValueError("Development row has incorrect split")
        family = row["family_id"]
        if not isinstance(family, str) or len(family) != 64:
            raise ValueError("Invalid family identity")
        if family in manifest["protected_family_ids"]:
            raise ValueError("Protected family in development data")
        families[row["split"]].add(family)
    if families["train"] & families["validation"]:
        raise ValueError("Question family crosses training and validation")
    counts = {
        "train": len(roles["train"]),
        "validation": len(roles["validation"]),
        "development": len(rows),
        "families": len(set().union(*families.values())),
    }
    if counts != manifest["counts"] or counts != EXPECTED_COUNTS:
        raise ValueError(f"Development counts differ from the expected split: {counts}")


def _verify_csv(path: Path, rows: list[dict], open_no_answers: dict[int, int]) -> None:
    eligible = {row["question_id"]: row for row in rows}
    seen = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for stored in csv.DictReader(stream):
            question_id = int(stored["question_id"])
            if question_id not in eligible:
                continue
            if question_id in seen:
                raise ValueError(f"Duplicate development CSV ID {question_id}")
            seen.add(question_id)
            row = eligible[question_id]
            checks = {
                "category": row["category"],
                "split": row["split"],
                "prompt_text": row["text"],
            }
            if any(stored[key] != value for key, value in checks.items()):
                raise ValueError(f"Development CSV metadata mismatch for {question_id}")
            if stored["list_style"].lower() != str(row["list_style"]).lower():
                raise ValueError(f"Development CSV list_style mismatch for {question_id}")
            if (
                int(stored["open_correct_count"]) != row["correct_count"]
                or int(stored["mc_correct_count"]) != row["mc_correct_count"]
                or int(stored["open_no_answer_count"]) != open_no_answers[question_id]
            ):
                raise ValueError(f"Development CSV label mismatch for {question_id}")
    if seen != set(eligible):
        raise ValueError("Development CSV join is incomplete")


def prepare_data(data_dir: Path, output: Path) -> dict:
    """Validate pinned inputs and write only permitted development records."""
    data_dir, output = Path(data_dir), Path(output)
    hashes = _source_hashes(data_dir)
    if (output / "manifest.json").exists():
        existing, _ = load_development(output / "manifest.json")
        if existing["source_hashes"] != hashes:
            raise ValueError("Existing prepared corpus uses different sources")
        return existing
    if (output / "development.jsonl").exists():
        raise ValueError("Incomplete prepared corpus; use a fresh output directory")
    split = json.loads((data_dir / "split.json").read_text(encoding="utf-8"))
    assignment = {int(key): role for key, role in split["assignment"].items()}
    if set(assignment.values()) != ALLOWED_ROLES:
        raise ValueError("Unexpected saved split roles")
    original, metadata = {}, {}
    for row in _records(data_dir / "labels.jsonl"):
        question_id = row["question_id"]
        original[question_id] = row["question"]
        metadata[question_id] = _metadata(row)
    if set(original) != set(assignment):
        raise ValueError("Original corpus and saved split IDs differ")
    converted = {}
    for row in _records(data_dir / "converted.jsonl"):
        if row.get("convertible") is True:
            question_id = row["question_id"]
            if question_id not in original or _metadata(row) != metadata[question_id]:
                raise ValueError(f"Conversion metadata mismatch for {question_id}")
            if row["question"].strip() != original[question_id].strip():
                raise ValueError(f"Original conversion text mismatch for {question_id}")
            converted[question_id] = row["open_question"].strip()
    families = build_families(original, converted)
    study_ids = {row["question_id"] for row in _records(data_dir / "study_prompts.jsonl")}
    if len(study_ids) != 280 or any(assignment.get(q) != "test" for q in study_ids):
        raise ValueError("Study reservation differs from the pinned test assignment")
    protected_families = {families[q] for q, role in assignment.items() if role == "test"}
    if any(assignment.get(q) != "train" for q in TRAINING_EXCLUSIONS):
        raise ValueError("Training exclusions contain questions outside the training split")
    candidate_ids = {q for q in converted if assignment[q] in {"train", "validation"}}
    family_exclusions = {q for q in candidate_ids if families[q] in protected_families}
    eligible_ids = candidate_ids - TRAINING_EXCLUSIONS - family_exclusions
    open_labels, no_answers = {}, {}
    open_ids = set()
    for row in _records(data_dir / "labels_open.jsonl"):
        question_id = row["question_id"]
        open_ids.add(question_id)
        if question_id not in eligible_ids:
            continue
        if (
            type(row.get("k")) is not int
            or row["k"] != 5
            or _metadata(row) != metadata[question_id]
        ):
            raise ValueError(f"Open label metadata mismatch for {question_id}")
        open_labels[question_id] = _count(row, "correct_count", question_id)
        no_answers[question_id] = _count(row, "no_answer_count", question_id)
        if open_labels[question_id] > 5 or no_answers[question_id] > 5:
            raise ValueError(f"Open label counts exceed trials for {question_id}")
    if open_ids != set(converted) or set(open_labels) != eligible_ids:
        raise ValueError("Open labels and conversion eligibility differ")
    mc_labels = {}
    for row in _records(data_dir / "labels.jsonl"):
        question_id = row["question_id"]
        if question_id in eligible_ids:
            if type(row.get("k")) is not int or row["k"] != 5:
                raise ValueError(f"MC trial count differs for {question_id}")
            mc_labels[question_id] = _count(row, "correct_count", question_id)
    rows = [
        {
            "question_id": q,
            "text": converted[q],
            "correct_count": open_labels[q],
            "k": 5,
            **metadata[q],
            "family_id": families[q],
            "split": assignment[q],
            "mc_correct_count": mc_labels[q],
        }
        for q in sorted(eligible_ids)
    ]
    roles = {
        role: [q for q in sorted(eligible_ids) if assignment[q] == role]
        for role in ("train", "validation")
    }
    manifest = {
        "schema_version": 1,
        "text_view": "converted_open_question",
        "target": "open_correct_count",
        "trials": 5,
        "normalization": NORMALIZATION,
        "source_hashes": hashes,
        "role_ids": roles,
        "original_role_ids": {
            role: sorted(q for q in assignment if assignment[q] == role)
            for role in sorted(ALLOWED_ROLES)
        },
        "study_ids": sorted(study_ids),
        "protected_family_ids": sorted(protected_families),
        "exclusions": {
            "training_duplicate_ids": sorted(TRAINING_EXCLUSIONS),
            "protected_family_ids": sorted(family_exclusions),
        },
        "counts": dict(EXPECTED_COUNTS),
        "distributions": {
            role: {
                "category": dict(
                    sorted(Counter(row["category"] for row in rows if row["split"] == role).items())
                ),
                "correct_count": dict(
                    sorted(
                        Counter(
                            str(row["correct_count"]) for row in rows if row["split"] == role
                        ).items()
                    )
                ),
            }
            for role in roles
        },
    }
    _validate_rows(rows, manifest)
    _verify_csv(data_dir / "open_ended_raw_performance.csv", rows, no_answers)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "development.jsonl", rows)
    manifest["development"] = {
        "path": "development.jsonl",
        "sha256": sha256_file(output / "development.jsonl"),
    }
    manifest["dataset_id"] = identity(manifest)
    atomic_json(output / "manifest.json", manifest)
    return manifest


def load_development(manifest_path: Path) -> tuple[dict, list[dict]]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "dataset_id"}
    if identity(payload) != manifest.get("dataset_id"):
        raise ValueError("Development manifest identity mismatch")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("source_hashes") != SOURCE_HASHES
        or manifest.get("normalization") != NORMALIZATION
        or manifest.get("text_view") != "converted_open_question"
        or manifest.get("target") != "open_correct_count"
        or manifest.get("trials") != 5
    ):
        raise ValueError("Unsupported development manifest contract")
    if manifest["development"]["path"] != "development.jsonl":
        raise ValueError("Unexpected prepared development path")
    path = manifest_path.parent / "development.jsonl"
    if sha256_file(path) != manifest["development"]["sha256"]:
        raise ValueError("Prepared development file checksum mismatch")
    rows = list(_records(path))
    _validate_rows(rows, manifest)
    return manifest, rows
