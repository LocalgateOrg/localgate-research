"""Small validated readers shared by analysis entry points."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class InputError(ValueError):
    """An input file does not contain the expected record collection."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"invalid JSON: {path}") from exc


def read_records(path: Path) -> list[dict]:
    """Read a JSON array or JSONL file into an ordered list of objects."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        records: list[dict] = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"invalid JSONL {path}:{number}") from exc
            if not isinstance(value, dict):
                raise InputError(f"JSONL row must be an object: {path}:{number}")
            records.append(value)
        return records

    value = read_json(path)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise InputError(f"JSON input must be an array of objects: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_json_new(path: Path, value: Any) -> None:
    """Atomically create a JSON file without replacing an existing path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise InputError(f"refusing to overwrite existing output: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict:
    value = read_json(path)
    if not isinstance(value, dict):
        raise InputError(f"JSON object required: {path}")
    return value
