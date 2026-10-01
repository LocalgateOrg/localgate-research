"""Join saved repeated MMLU-Pro logs into per-question success counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from research.data import mc_generation

LIST_STYLE_PREFIX = "which of the following"


def _manifest(run_dir: Path, allow_limit: bool) -> tuple[str, dict[str, Any]]:
    path = run_dir / "manifest.json"
    if not path.exists():
        raise ValueError(f"{run_dir}: required manifest.json is missing")
    text = path.read_text(encoding="utf-8")
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(manifest, dict):
        raise TypeError(f"{path}: expected a JSON object")
    limit = manifest.get("limit")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError(f"{path}: limit must be null or a positive integer")
    if limit is not None and not allow_limit:
        raise ValueError(f"{run_dir} was generated with limit={limit}; pass --allow-limit")
    revision = manifest.get("revision")
    if revision is not None:
        mc_generation.validate_revision(revision, dry_run=True)
    trust_remote_code = manifest.get("trust_remote_code", True)
    if type(trust_remote_code) is not bool:
        raise ValueError(f"{path}: trust_remote_code must be boolean")
    expected = mc_generation.build_manifest(
        model=mc_generation.DEFAULT_MODEL,
        revision=revision,
        trust_remote_code=trust_remote_code,
        limit=limit,
        versions=mc_generation.EXPECTED_VERSIONS,
    )
    problems = mc_generation.manifest_problems(manifest, expected)
    if problems:
        raise ValueError(
            f"{path}: manifest differs from the label protocol in: {', '.join(problems)}"
        )
    return text, manifest


def repeat_dirs(run_dir: Path) -> list[Path]:
    repeats: list[tuple[int, Path]] = []
    for path in run_dir.glob("rep*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("rep")
        if not suffix.isdigit():
            raise ValueError(f"{path}: repeat directory must be named repN")
        repeats.append((int(suffix), path))
    repeats.sort()
    indices = [index for index, _ in repeats]
    if indices and indices != list(range(len(indices))):
        raise ValueError(f"{run_dir}: repeat directories are not contiguous from rep0")
    return [path for _, path in repeats]


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL record")
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(sample, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, sample


def _response_text(sample: dict[str, Any], path: Path, line_number: int) -> str:
    resps = sample.get("resps")
    if not isinstance(resps, list) or not resps:
        raise ValueError(f"{path}:{line_number}: resps must contain a response")
    first = resps[0]
    if isinstance(first, list):
        if not first:
            raise ValueError(f"{path}:{line_number}: first response list is empty")
        first = first[0]
    if not isinstance(first, str):
        raise TypeError(f"{path}:{line_number}: first response must be a string")
    return first


def _score(sample: dict[str, Any], path: Path, line_number: int) -> bool:
    value = sample.get("exact_match")
    if isinstance(value, bool):
        return value
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) in (0.0, 1.0)
    ):
        return bool(value)
    raise ValueError(f"{path}:{line_number}: exact_match must be 0 or 1")


def _invalid(sample: dict[str, Any], path: Path, line_number: int) -> bool:
    filtered = sample.get("filtered_resps")
    if not isinstance(filtered, list) or not filtered:
        raise ValueError(f"{path}:{line_number}: filtered_resps must contain a value")
    return str(filtered[0]).strip() == "[invalid]"


def _identity(doc: dict[str, Any], path: Path, line_number: int) -> dict[str, Any]:
    question = doc.get("question")
    options = doc.get("options")
    category = doc.get("category")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{path}:{line_number}: doc.question must be non-empty")
    if (
        not isinstance(options, list)
        or not options
        or any(not isinstance(option, str) for option in options)
    ):
        raise ValueError(f"{path}:{line_number}: doc.options must be a non-empty string list")
    if not isinstance(category, str) or not category.strip():
        raise ValueError(f"{path}:{line_number}: doc.category must be non-empty")
    return {
        "question": question,
        "options": options,
        "category": category,
        "answer": doc.get("answer"),
        "answer_index": doc.get("answer_index"),
    }


def read_repeat(repeat_dir: Path) -> dict[int, dict[str, Any]]:
    observations: dict[int, dict[str, Any]] = {}
    sample_files = sorted(repeat_dir.rglob("samples_*.jsonl"))
    if not sample_files:
        raise ValueError(f"{repeat_dir}: no samples_*.jsonl files")
    for path in sample_files:
        for line_number, sample in _read_jsonl(path):
            doc = sample.get("doc")
            if not isinstance(doc, dict):
                raise TypeError(f"{path}:{line_number}: doc must be an object")
            question_id = doc.get("question_id")
            if type(question_id) is not int or question_id < 0:
                raise ValueError(f"{path}:{line_number}: question_id must be non-negative")
            if question_id in observations:
                raise ValueError(f"{path}:{line_number}: duplicate question_id {question_id}")
            identity = _identity(doc, path, line_number)
            response = _response_text(sample, path, line_number)
            observations[question_id] = {
                **identity,
                "correct": _score(sample, path, line_number),
                "invalid": _invalid(sample, path, line_number),
                "response_hash": hashlib.sha256(response.encode()).hexdigest()[:16],
            }
    if not observations:
        raise ValueError(f"{repeat_dir}: no observations")
    return observations


def _validate_coverage(
    repeats: list[Path], per_repeat: list[dict[int, dict[str, Any]]]
) -> list[int]:
    expected = set(per_repeat[0])
    for path, observations in zip(repeats[1:], per_repeat[1:], strict=True):
        actual = set(observations)
        if actual != expected:
            missing = sorted(expected - actual)[:5]
            extra = sorted(actual - expected)[:5]
            raise ValueError(
                f"{path}: repeat coverage differs; missing={missing}, extra={extra}"
            )
    return sorted(expected)


def _validate_identities(
    repeats: list[Path], per_repeat: list[dict[int, dict[str, Any]]], question_ids: list[int]
) -> None:
    fields = ("question", "options", "category", "answer", "answer_index")
    for question_id in question_ids:
        expected = {field: per_repeat[0][question_id][field] for field in fields}
        for path, observations in zip(repeats[1:], per_repeat[1:], strict=True):
            actual = {field: observations[question_id][field] for field in fields}
            if actual != expected:
                raise ValueError(f"{path}: question_id {question_id} identity differs")


def check_repeats_differ(
    repeats: list[Path], per_repeat: list[dict[int, dict[str, Any]]], question_ids: list[int]
) -> None:
    fingerprints = [
        tuple(observations[question_id]["response_hash"] for question_id in question_ids)
        for observations in per_repeat
    ]
    for left, right in combinations(range(len(repeats)), 2):
        if fingerprints[left] == fingerprints[right]:
            raise ValueError(
                f"{repeats[left].name} and {repeats[right].name} contain identical response vectors"
            )


def build_labels(run_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    repeats = repeat_dirs(run_dir)
    expected_repeats = manifest["preset"]["repeats"]
    if len(repeats) != expected_repeats:
        raise ValueError(
            f"{run_dir}: found {len(repeats)} repeats, manifest requires {expected_repeats}"
        )
    per_repeat = [read_repeat(path) for path in repeats]
    for repeat, (path, observations) in enumerate(zip(repeats, per_repeat, strict=True)):
        mc_generation.validate_repeat_result(
            path,
            observations,
            model=manifest["model"],
            revision=manifest.get("revision"),
            trust_remote_code=manifest.get("trust_remote_code", True),
            seed=manifest["seeds"][repeat],
            limit=manifest["limit"],
        )
    question_ids = _validate_coverage(repeats, per_repeat)
    _validate_identities(repeats, per_repeat, question_ids)
    check_repeats_differ(repeats, per_repeat, question_ids)

    labels: list[dict[str, Any]] = []
    for question_id in question_ids:
        first = per_repeat[0][question_id]
        question = first["question"]
        labels.append(
            {
                "question_id": question_id,
                "question": question,
                "options": first["options"],
                "category": first["category"],
                "correct_count": sum(row[question_id]["correct"] for row in per_repeat),
                "invalid_count": sum(row[question_id]["invalid"] for row in per_repeat),
                "k": len(repeats),
                "list_style": question.strip().lower().startswith(LIST_STYLE_PREFIX),
            }
        )
    return labels


def spread(labels: list[dict[str, Any]]) -> Counter:
    return Counter(label["correct_count"] for label in labels)


def check_spread(labels: list[dict[str, Any]]) -> list[str]:
    if not labels:
        return ["no labels produced"]
    counts = spread(labels)
    k = labels[0]["k"]
    if not any(0 < value < k for value in counts):
        return ["every item is 0/k or k/k; repeated sampling did not vary"]
    return []


def write_labels(labels: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for label in labels:
                handle.write(json.dumps(label) + "\n")
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-limit", action="store_true")
    return parser


def _output_paths(
    run_dir: Path, destination: Path, force: bool
) -> Path:
    manifest_copy = destination.with_name(destination.stem + "_manifest.json")
    outputs = [destination, manifest_copy]
    sources = [run_dir / "manifest.json", *run_dir.glob("rep*/**/samples_*.jsonl")]
    source_paths = {path.resolve() for path in sources if path.exists()}
    for path in outputs:
        if path.resolve() in source_paths:
            raise ValueError(f"output path is a source input: {path}")
        if path.is_dir():
            raise ValueError(f"output path is a directory: {path}")
    collisions = [path for path in outputs if path.exists()]
    if collisions and not force:
        joined = ", ".join(str(path) for path in collisions)
        raise ValueError(f"output exists; pass --force to overwrite: {joined}")
    return manifest_copy


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if arguments == ["--help"] or arguments == ["-h"]:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    manifest_text, manifest = _manifest(args.run_dir, args.allow_limit)
    manifest_copy = _output_paths(args.run_dir, args.out, args.force)
    labels = build_labels(args.run_dir, manifest)
    flags = check_spread(labels)
    write_labels(labels, args.out)
    manifest_copy.write_text(manifest_text, encoding="utf-8")
    print(f"wrote {manifest_copy}")

    counts = spread(labels)
    k = labels[0]["k"]
    mean_rate = sum(label["correct_count"] for label in labels) / (len(labels) * k)
    print(f"items: {len(labels)}")
    print(f"runs per item: {k}")
    print(f"mean success: {mean_rate:.4f}")
    print(f"list-style: {sum(label['list_style'] for label in labels)}")
    for value in range(k + 1):
        print(f"{value}/{k}: {counts.get(value, 0)}")
    print(f"wrote {args.out}")
    if flags:
        for flag in flags:
            print(f"suspect: {flag}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
