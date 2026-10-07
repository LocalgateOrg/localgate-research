"""Build one item table per decision stage from the released MMLU-Pro-Open data.

Every later comparison (baseline against gold, candidate against baseline) is a
join on these tables, so each row carries the item, the released pipeline's
decision and every human label that exists for it. Sources are fetched at the
revisions the dataset paper cites and recorded with their SHA-256.
"""

from __future__ import annotations

import argparse
import math
import re
import urllib.request
from pathlib import Path

from research.analysis.data_io import InputError, sha256, write_json_new

# Revisions cited in the dataset paper; changing one changes the instrument.
SOURCES = {
    "conversion_record": (
        "localgate/mmlu-pro-open",
        "6bc069a7b1b83d93ee352dadc9fb49b59cb4dc5d",
        "conversion_record/train-00000-of-00001.parquet",
    ),
    "judgements": (
        "localgate/judge-panel-verdicts",
        "7d5bb21170294a4895bb9e3408794d7aa3fd55a1",
        "judgements/train-00000-of-00001.parquet",
    ),
    "pass1": (
        "localgate/audit-results",
        "a4dcb995522f89cc7c0b695624731c94cfb3df1d",
        "pass1/train-00000-of-00001.parquet",
    ),
    "pass2": (
        "localgate/audit-results",
        "a4dcb995522f89cc7c0b695624731c94cfb3df1d",
        "pass2/train-00000-of-00001.parquet",
    ),
    "calibration_items": (
        "localgate/audit-results",
        "a4dcb995522f89cc7c0b695624731c94cfb3df1d",
        "calibration_items/train-00000-of-00001.parquet",
    ),
}
# The k=5 open-ended responses that the panel graded, one file per split.
GENERATIONS_REPO = "localgate/mmlu-pro-open-gemma-generations"
GENERATIONS_REVISION = "44d8c1052854badc1091a64b4f5d8d8697339ab5"
for _split in ("train", "validation", "test", "excluded_stem_leak"):
    SOURCES[f"generations_{_split}"] = (
        GENERATIONS_REPO,
        GENERATIONS_REVISION,
        f"generations/{_split}-00000-of-00001.parquet",
    )

EXPECTED_ROWS = {
    "conversion_record": 12_032,
    "judgements": 126_810,
    "pass1": 248,
    "pass2": 152,
    "calibration_items": 100,
}

HUMANS = ("human_1", "human_2", "human_3")

# A reference is numeric when, after an optional sign or currency mark, it is a
# number followed by at most a short unit ("93.64 Btu/hr-ft", "$60.00", "1.5e3 J").
NUMERIC_REFERENCE = re.compile(
    r"^\s*[-+−~≈$€£]?\s*\d[\d,]*(\.\d+)?(\s*[eE][-+]?\d+|\s*[x×]\s*10\^?[-+−]?\d+)?"
    r"\s*%?(\s*[^\s\d][^\s]{0,15}){0,3}\s*\.?\s*$"
)


def reference_type(reference: str) -> str:
    """numeric, short_text (1-2 words), text (3-10) or long_text (more than 10)."""
    if NUMERIC_REFERENCE.match(reference or ""):
        return "numeric"
    words = len((reference or "").split())
    if words <= 2:
        return "short_text"
    return "text" if words <= 10 else "long_text"


def conversion_stratum(row: dict) -> str:
    """Audit stratum of one conversion-record row, matching the released audit key.

    Released scores are floats (NaN when the item was never rescored), so the
    borderline test works on the rounded value rather than on the type.
    """
    score = row.get("score")
    if score is not None and not (isinstance(score, float) and math.isnan(score)):
        if 4 <= round(score) <= 7:
            return "borderline"
    if row["convertible"]:
        return "kept_judge" if row["stage"] == "judge" else "kept_rescore"
    return "drop_rewrite" if row["stage"] == "rewrite" else "drop_rescore"


def fetch(cache: Path) -> dict[str, Path]:
    """Download every source once into the cache and return local paths."""
    paths = {}
    for name, (repo, revision, file) in SOURCES.items():
        target = cache / repo.replace("/", "__") / revision / file
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{file}"
            temporary = target.with_suffix(".part")
            urllib.request.urlretrieve(url, temporary)
            temporary.rename(target)
        paths[name] = target
    return paths


def _clean(value):
    """Parquet nulls and numpy scalars to plain JSON values."""
    if value is None:
        return None
    if hasattr(value, "tolist"):  # numpy arrays and scalars alike
        value = value.tolist()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _records(frame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in frame.to_dict("records")]


def build(paths: dict[str, Path]) -> dict[str, list[dict]]:
    import pandas as pd

    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    frames["generations"] = pd.concat(
        [frames.pop(name) for name in list(frames) if name.startswith("generations_")],
        ignore_index=True,
    )
    for name, expected in {**EXPECTED_ROWS, "generations": 42_270}.items():
        if len(frames[name]) != expected:
            raise InputError(f"{name}: expected {expected} rows, found {len(frames[name])}")

    conversion = {row["question_id"]: row for row in _records(frames["conversion_record"])}
    for row in conversion.values():
        row["stratum"] = conversion_stratum(row)
        row["reference_type"] = reference_type(row["reference_answer"])

    # Filter: every MMLU-Pro question with the converter's decision; human votes
    # where the Stage-1 audit covered it.
    audit1 = {row["question_id"]: row for row in _records(frames["pass1"])}
    filter_rows = []
    for qid, row in sorted(conversion.items()):
        human = audit1.get(qid)
        filter_rows.append(
            {
                "question_id": qid,
                "category": row["category"],
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "reference_type": row["reference_type"],
                "list_style": row["list_style"],
                "stratum": row["stratum"],
                "baseline_judge_convertible": row["judge_convertible"],
                "baseline_rescore": row["score"],
                "baseline_convertible": row["convertible"],
                "stage": row["stage"],
                "options": human["options"] if human else None,
                "audit_blind_id": human["blind_id"] if human else None,
                "human_votes": [human[f"{h}_converts"] for h in HUMANS] if human else None,
                "human_unsure": [human[f"{h}_unsure"] for h in HUMANS] if human else None,
                "human_majority": human["majority_converts"] if human else None,
                "human_answer_stands_alone": human["majority_answer_stands_alone"] if human else None,
            }
        )

    # Rewrites: every retained question; human votes for the Stage-2 sample.
    audit2 = {row["question_id"]: row for row in _records(frames["pass2"])}
    rewrite_rows = []
    for qid, row in sorted(conversion.items()):
        if not row["convertible"]:
            continue
        human = audit2.get(qid)
        rewrite_rows.append(
            {
                "question_id": qid,
                "category": row["category"],
                "original_question": row["question"],
                "open_question": row["open_question"],
                "reference_answer": row["reference_answer"],
                "reference_type": row["reference_type"],
                "stratum": row["stratum"],
                "options": human["options"] if human else None,
                "audit_blind_id": human["blind_id"] if human else None,
                "human_same_question": [human[f"{h}_same_question"] for h in HUMANS] if human else None,
                "human_self_contained": [human[f"{h}_self_contained"] for h in HUMANS] if human else None,
                "human_unsure": [human[f"{h}_unsure"] for h in HUMANS] if human else None,
                "human_majority_same_question": human["majority_same_question"] if human else None,
                "human_majority_self_contained": human["majority_self_contained"] if human else None,
            }
        )

    # Judging: one row per graded response with the three judges side by side.
    judgements = frames["judgements"]
    wide = judgements.pivot_table(
        index=["key", "question_id", "rep"],
        columns="judge_model",
        values="verdict",
        aggfunc="first",
    ).reset_index()
    tokens = judgements.groupby("key")["tokens_in"].median()
    calibration = {row["key"]: row for row in _records(frames["calibration_items"])}
    generations = {
        f"{row['question_id']}:{row['attempt']}": row
        for row in _records(frames["generations"][
            ["question_id", "attempt", "response_text", "finish_reason", "output_tokens"]
        ])
    }
    judge_models = sorted(judgements["judge_model"].unique())
    judging_rows = []
    for row in _records(wide):
        verdicts = {model: row[model] for model in judge_models}
        matches = sum(v == "match" for v in verdicts.values())
        source = conversion[row["question_id"]]
        human = calibration.get(row["key"])
        generation = generations.get(row["key"])
        if generation is None:
            raise InputError(f"no released generation for graded response {row['key']}")
        judging_rows.append(
            {
                "key": row["key"],
                "question_id": row["question_id"],
                "rep": row["rep"],
                "category": source["category"],
                "question": source["open_question"],
                "reference_answer": source["reference_answer"],
                "reference_type": source["reference_type"],
                "judge_verdicts": verdicts,
                "baseline_match": matches >= 2,
                "panel_unanimous_binary": matches in (0, 3),
                "judge_tokens_in_median": int(tokens[row["key"]]),
                "response": generation["response_text"],
                "finish_reason": generation["finish_reason"],
                "output_tokens": generation["output_tokens"],
                "human_verdicts": [human[f"{h}_verdict"] for h in HUMANS] if human else None,
                "human_match": bool(human["human_correct"]) if human else None,
            }
        )
    return {"filter": filter_rows, "rewrite": rewrite_rows, "judging": judging_rows}


def write_tables(tables: dict[str, list[dict]], paths: dict[str, Path], out: Path) -> dict:
    import json

    out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        target = out / f"{name}.jsonl"
        if target.exists():
            raise InputError(f"refusing to overwrite existing output: {target}")
        with target.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    manifest = {
        "sources": {
            name: {"repo": repo, "revision": revision, "file": file, "sha256": sha256(paths[name])}
            for name, (repo, revision, file) in SOURCES.items()
        },
        "rows": {name: len(rows) for name, rows in tables.items()},
        "with_human_labels": {
            "filter": sum(r["human_votes"] is not None for r in tables["filter"]),
            "rewrite": sum(r["human_same_question"] is not None for r in tables["rewrite"]),
            "judging": sum(r["human_verdicts"] is not None for r in tables["judging"]),
        },
    }
    write_json_new(out / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("output/hf-cache"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = fetch(args.cache)
    manifest = write_tables(build(paths), paths, args.out)
    print(f"wrote {manifest['rows']} to {args.out}; human-labelled: {manifest['with_human_labels']}")
    return 0
