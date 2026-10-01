"""Atomic, content-addressed records for local and remote experiments."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def identity(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_text_exclusive(path: Path, content: str) -> None:
    """Atomically create *path* without replacing an output from another process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    content = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    atomic_text(path, content)


def write_jsonl_exclusive(path: Path, rows: list[dict]) -> None:
    content = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    atomic_text_exclusive(path, content)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects: {path}")
    return rows


def contained_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()) or Path(relative).is_absolute():
        raise ValueError("Artifact paths must stay inside their bundle")
    return candidate


def verify_files(root: Path, files: dict[str, str]) -> None:
    if not files:
        raise ValueError("Artifact inventory must not be empty")
    for name, expected in files.items():
        path = contained_path(root, name)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Artifact is missing or its hash changed: {name}")
