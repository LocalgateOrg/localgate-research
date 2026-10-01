"""Generate the saved open-ended or multiple-choice study responses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "google/gemma-4-E2B-it"
MAX_TOKENS = 6144
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 64
MAX_MODEL_LEN = 8192
CODECARBON_VERSION = "3.3.1"
REVISION_PATTERN = re.compile(r"[0-9a-fA-F]{40}")


@dataclass(frozen=True)
class OpenProfile:
    repeats: int
    seed_base: int
    energy_telemetry: bool
    versions: dict[str, str]


OPEN_PROFILES = {
    "initial": OpenProfile(
        repeats=5,
        seed_base=5042,
        energy_telemetry=False,
        versions={"vllm": "0.25.1", "torch": "2.11.0", "transformers": "5.14.1"},
    ),
    "study": OpenProfile(
        repeats=30,
        seed_base=6042,
        energy_telemetry=True,
        versions={
            "vllm": "0.29.0",
            "torch": "2.13.0",
            "transformers": "5.17.0",
            "codecarbon": CODECARBON_VERSION,
        },
    ),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL record")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def load_open_corpus(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load convertible prompts in stable question-ID order."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line_number, row in enumerate(_read_jsonl(path), 1):
        question_id = row.get("question_id")
        if type(question_id) is not int or question_id < 0:
            raise ValueError(f"{path}:{line_number}: question_id must be a non-negative integer")
        if question_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate question_id {question_id}")
        seen.add(question_id)
        convertible = row.get("convertible")
        if type(convertible) is not bool:
            raise ValueError(f"{path}:{line_number}: convertible must be boolean")
        if not convertible:
            continue
        text = row.get("open_question")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{line_number}: question_id {question_id} has no open text")
        category = row.get("category", "unknown")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"{path}:{line_number}: question_id {question_id} has no category")
        list_style = row.get("list_style", False)
        if type(list_style) is not bool:
            raise ValueError(f"{path}:{line_number}: list_style must be boolean")
        rows.append(
            {
                "question_id": question_id,
                "category": category,
                "list_style": list_style,
                "open_question": text.strip(),
            }
        )
    rows.sort(key=lambda row: row["question_id"])
    if not rows:
        raise ValueError(f"{path}: no convertible prompts")
    return rows[:limit] if limit is not None else rows


def corpus_fingerprint(rows: list[dict[str, Any]]) -> str:
    payload = "\0".join(f"{row['question_id']}\t{row['open_question']}" for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def validate_revision(revision: str | None, *, dry_run: bool) -> str | None:
    """Require an immutable Hub commit for every live generation run."""
    if revision is None:
        if dry_run:
            return None
        raise ValueError("live generation requires a 40-character hexadecimal --revision")
    if REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("model revision must be a 40-character hexadecimal commit SHA")
    return revision.lower()


def installed_generation_versions(*, energy_telemetry: bool = False) -> dict[str, str]:
    versions: dict[str, str] = {}
    distributions = ["vllm", "torch", "transformers"]
    if energy_telemetry:
        distributions.append("codecarbon")
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution).split("+")[0]
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not installed"
    return versions


def build_open_manifest(
    rows: list[dict[str, Any]],
    *,
    model: str,
    revision: str | None,
    trust_remote_code: bool,
    limit: int | None,
    repeats: int,
    seed_base: int,
    versions: dict[str, str],
    energy_telemetry: bool | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model,
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "task": "openended-label",
        "limit": limit,
        "items": len(rows),
        "corpus_fingerprint": corpus_fingerprint(rows),
        "preset": {"name": "openlabel", "repeats": repeats},
        "seeds": [seed_base + offset for offset in range(repeats)],
        "gen_kwargs": {
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "stop": [],
        },
        "chat_template": "model default, no system message, zero-shot",
        "versions": versions,
    }
    if energy_telemetry is not None:
        manifest["energy_telemetry"] = energy_telemetry
    return manifest


def check_manifest_compatible(existing: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    keys = (
        "model",
        "revision",
        "trust_remote_code",
        "task",
        "limit",
        "items",
        "corpus_fingerprint",
        "preset",
        "seeds",
        "gen_kwargs",
        "chat_template",
        "energy_telemetry",
    )
    problems = []
    for key in keys:
        legacy_remote_code = (
            key == "trust_remote_code"
            and key not in existing
            and fresh.get("revision") is None
        )
        if not legacy_remote_code and existing.get(key) != fresh.get(key):
            problems.append(
                f"{key}: existing {existing.get(key)!r} != this run {fresh.get(key)!r}"
            )
    existing_versions = existing.get("versions")
    fresh_versions = fresh.get("versions")
    versions_match = existing_versions == fresh_versions
    if (
        not versions_match
        and isinstance(existing_versions, dict)
        and isinstance(fresh_versions, dict)
    ):
        # The preserved study manifest predates telemetry-version recording. Its
        # emissions.csv files carry the CodeCarbon version and are validated per repeat.
        without_telemetry = {
            key: value for key, value in fresh_versions.items() if key != "codecarbon"
        }
        versions_match = existing_versions == without_telemetry
    if not versions_match:
        problems.append(
            f"versions: existing {existing_versions!r} != this run {fresh_versions!r}"
        )
    return problems


def pending_open_ids(rep_file: Path, all_ids: list[int], *, seed: int) -> list[int]:
    """Return unfinished IDs after validating every saved response record."""
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("expected question IDs contain duplicates")
    if not rep_file.exists():
        return list(all_ids)
    expected = set(all_ids)
    done: set[int] = set()
    for line_number, row in enumerate(_read_jsonl(rep_file), 1):
        question_id = row.get("question_id")
        if type(question_id) is not int or question_id not in expected:
            raise ValueError(f"{rep_file}:{line_number}: unexpected question_id {question_id!r}")
        if question_id in done:
            raise ValueError(f"{rep_file}:{line_number}: duplicate question_id {question_id}")
        if row.get("seed") != seed:
            raise ValueError(
                f"{rep_file}:{line_number}: seed {row.get('seed')!r} differs from {seed}"
            )
        if not isinstance(row.get("text"), str):
            raise TypeError(f"{rep_file}:{line_number}: text must be a string")
        if not isinstance(row.get("finish_reason"), str):
            raise TypeError(f"{rep_file}:{line_number}: finish_reason must be a string")
        for field in ("prompt_tokens", "output_tokens"):
            value = row.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{rep_file}:{line_number}: {field} must be non-negative")
        done.add(question_id)
    return [question_id for question_id in all_ids if question_id not in done]


def _ensure_final_newline(path: Path) -> None:
    if path.exists() and path.stat().st_size:
        with path.open("rb+") as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                handle.seek(0, 2)
                handle.write(b"\n")


def _start_energy_tracker(rep_dir: Path):
    try:
        from codecarbon import EmissionsTracker
    except ImportError as exc:
        raise RuntimeError("codecarbon is required by the selected generation profile") from exc
    tracker = EmissionsTracker(
        project_name="localgate-openlabel", output_dir=str(rep_dir), log_level="error"
    )
    tracker.start()
    return tracker


def validate_energy_telemetry(
    rep_dir: Path, *, expected_version: str = CODECARBON_VERSION
) -> None:
    """Require complete, versioned CodeCarbon records for a completed repeat."""
    path = rep_dir / "emissions.csv"
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"completed repeat has no readable telemetry at {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"completed repeat has no telemetry records in {path}")
    for index, row in enumerate(rows, 2):
        if row.get("project_name") != "localgate-openlabel":
            raise ValueError(f"{path}:{index}: unexpected CodeCarbon project")
        if row.get("codecarbon_version") != expected_version:
            raise ValueError(
                f"{path}:{index}: CodeCarbon version {row.get('codecarbon_version')!r} "
                f"differs from {expected_version}"
            )
        for field in ("duration", "energy_consumed"):
            try:
                value = float(row.get(field, ""))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{index}: invalid {field}") from exc
            if not math.isfinite(value) or value < 0 or (field == "duration" and value == 0):
                raise ValueError(f"{path}:{index}: invalid {field} {value!r}")


def generate_open_repeat(
    rows: list[dict[str, Any]],
    *,
    model: str,
    revision: str | None,
    trust_remote_code: bool,
    seed: int,
    rep_file: Path,
    energy_telemetry: bool,
) -> int:
    """Generate and append one seeded repeat with a fresh vLLM engine."""
    todo = pending_open_ids(rep_file, [row["question_id"] for row in rows], seed=seed)
    if not todo:
        if energy_telemetry:
            validate_energy_telemetry(rep_file.parent)
        return 0
    revision = validate_revision(revision, dry_run=False)

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vllm is required for generation") from exc

    by_id = {row["question_id"]: row for row in rows}
    tracker = _start_energy_tracker(rep_file.parent) if energy_telemetry else None
    try:
        llm = LLM(
            model=model,
            revision=revision,
            tokenizer_revision=revision,
            seed=seed,
            dtype="auto",
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
        )
        params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_tokens=MAX_TOKENS,
        )
        conversations = [
            [{"role": "user", "content": by_id[question_id]["open_question"]}]
            for question_id in todo
        ]
        outputs = llm.chat(conversations, params, use_tqdm=True)
        if len(outputs) != len(todo):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(todo)} prompts")

        rep_file.parent.mkdir(parents=True, exist_ok=True)
        _ensure_final_newline(rep_file)
        with rep_file.open("a", encoding="utf-8") as handle:
            for question_id, output in zip(todo, outputs, strict=True):
                if len(output.outputs) != 1:
                    raise RuntimeError(f"question_id {question_id} returned multiple completions")
                completion = output.outputs[0]
                record = {
                    "question_id": question_id,
                    "seed": seed,
                    "text": completion.text,
                    "finish_reason": completion.finish_reason,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "output_tokens": len(completion.token_ids),
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
        del llm
    finally:
        if tracker is not None:
            tracker.stop()
    if energy_telemetry:
        validate_energy_telemetry(rep_file.parent)
    return len(todo)


def run_dir_for(base: Path, limit: int | None) -> Path:
    return base.with_name(base.name + f"_limit{limit}") if limit is not None else base


def _write_new_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid manifest {path}: {exc}") from exc
        if not isinstance(existing, dict):
            raise ValueError(f"invalid manifest {path}: expected a JSON object")
        problems = check_manifest_compatible(existing, manifest)
        if problems:
            raise ValueError(f"{path.parent} holds a different run:\n  " + "\n  ".join(problems))
        return
    if path.parent.exists() and any(path.parent.iterdir()):
        raise ValueError(
            f"refusing to claim non-empty run directory without a manifest: {path.parent}"
        )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")


def _open_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("open", help="generate open-ended responses")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(OPEN_PROFILES), default="initial")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")


def _mc_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("mc", help="generate the repeated MMLU-Pro labels")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="format")
    _open_parser(subparsers)
    _mc_parser(subparsers)
    return parser


def _run_open(args: argparse.Namespace) -> int:
    profile = OPEN_PROFILES[args.profile]
    revision = validate_revision(args.revision, dry_run=args.dry_run)
    rows = load_open_corpus(args.corpus, args.limit)
    versions = installed_generation_versions(energy_telemetry=profile.energy_telemetry)
    manifest = build_open_manifest(
        rows,
        model=args.model,
        revision=revision,
        trust_remote_code=args.trust_remote_code,
        limit=args.limit,
        repeats=profile.repeats,
        seed_base=profile.seed_base,
        versions=versions,
        energy_telemetry=True if profile.energy_telemetry else None,
    )
    target = run_dir_for(args.out, args.limit)
    if args.dry_run:
        print(json.dumps({"output": str(target), "manifest": manifest}, indent=2))
        return 0
    if versions != profile.versions:
        raise RuntimeError(
            f"{args.profile} profile requires {profile.versions}; installed versions are {versions}"
        )
    target.mkdir(parents=True, exist_ok=True)
    _write_new_manifest(target / "manifest.json", manifest)
    print(f"items: {len(rows)}  repeats: {profile.repeats}  out: {target}")
    for offset, seed in enumerate(manifest["seeds"]):
        rep_file = target / f"rep{offset}" / "generations.jsonl"
        generated = generate_open_repeat(
            rows,
            model=args.model,
            revision=revision,
            trust_remote_code=args.trust_remote_code,
            seed=seed,
            rep_file=rep_file,
            energy_telemetry=profile.energy_telemetry,
        )
        print(f"rep{offset} seed={seed}: {generated} generated, {len(rows) - generated} saved")
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not arguments:
        parser.print_help()
        return 2
    if arguments == ["--help"] or arguments == ["-h"]:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    try:
        args.revision = validate_revision(args.revision, dry_run=args.dry_run)
    except ValueError as exc:
        parser.error(str(exc))
    if args.format == "open":
        return _run_open(args)
    if args.format == "mc":
        from research.data import mc_generation

        return mc_generation.run(
            out=args.out,
            model=args.model,
            revision=validate_revision(args.revision, dry_run=args.dry_run),
            trust_remote_code=args.trust_remote_code,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
