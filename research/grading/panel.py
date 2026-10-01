"""Pure reduction of three completed grading logs into panel labels."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

VERDICTS = ("match", "no_match", "no_answer")


def verdict_binary(verdict: str) -> int:
    """Map the preregistered three-way verdict to its binary score."""
    return 1 if verdict == "match" else 0


def fleiss_kappa(ratings: list[Counter]) -> float:
    """Compute Fleiss' kappa for items rated by the same number of annotators."""
    if not ratings:
        return float("nan")
    n_raters = sum(ratings[0].values())
    if n_raters < 2:
        return float("nan")
    categories = sorted({category for item in ratings for category in item})
    totals = Counter()
    agreement = []
    for item in ratings:
        if sum(item.values()) != n_raters:
            raise ValueError("every item must have the same number of raters")
        totals.update(item)
        agreement.append(
            (sum(count * count for count in item.values()) - n_raters)
            / (n_raters * (n_raters - 1))
        )
    p_bar = sum(agreement) / len(ratings)
    grand_total = len(ratings) * n_raters
    p_expected = sum((totals[category] / grand_total) ** 2 for category in categories)
    if p_expected == 1.0:
        return 1.0
    return (p_bar - p_expected) / (1 - p_expected)


def build_panel(
    judge_files: list[Path], *, keys: set[str] | None = None
) -> dict[str, dict]:
    """Return one strict 2-of-3 panel record for every requested item key.

    ``keys`` permits a calibration subset to be reduced from completed bulk logs.
    With no subset, all clean records are joined.
    """
    if len(judge_files) != 3:
        raise SystemExit(f"the panel takes exactly 3 judge files, got {len(judge_files)}")
    resolved = {Path(path).resolve() for path in judge_files}
    if len(resolved) != 3:
        raise SystemExit("the panel requires three distinct judge files")

    votes: dict[str, dict[str, dict]] = {}
    judges = []
    instruments = set()
    keys_by_file: list[set[str]] = []
    for path in map(Path, judge_files):
        judge_name = None
        file_keys: set[str] = set()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: invalid JSON") from exc
            if "error" in record:
                continue
            key = record.get("key")
            if not isinstance(key, str) or not key:
                raise SystemExit(f"{path}:{number}: clean verdict has no string key")
            if keys is not None and key not in keys:
                continue
            verdict = record.get("verdict")
            if verdict not in VERDICTS:
                raise SystemExit(
                    f"{path}: key {record.get('key')} has verdict {verdict!r}, "
                    f"not one of {VERDICTS}"
                )
            record_judge = record.get("judge")
            if judge_name is not None and record_judge != judge_name:
                raise SystemExit(
                    f"{path}: mixes judge names {judge_name!r} and {record_judge!r}"
                )
            judge_name = record_judge
            if key in file_keys:
                raise SystemExit(
                    f"{path}: duplicate verdict for {key}; each judge must have one verdict per response"
                )
            file_keys.add(key)
            instruments.add(
                (
                    record.get("prompts"),
                    record.get("template"),
                    record.get("output_mode", "prompted"),
                )
            )
            votes.setdefault(key, {})[record_judge] = record
        if judge_name is None:
            raise SystemExit(f"{path} holds no clean verdicts")
        judges.append(judge_name)
        keys_by_file.append(file_keys)

    if len(set(judges)) != 3:
        raise SystemExit(f"expected three distinct judges, got {judges}")
    if (
        len({(prompt, mode) for prompt, _, mode in instruments}) > 1
        or len({template for _, template, _ in instruments if template is not None}) > 1
    ):
        raise SystemExit(
            f"the three judge files span different instruments "
            f"{sorted(instruments, key=repr)} — "
            "a judge was re-run under a changed prompt/template/mode; do not join them"
        )

    all_keys = set(keys) if keys is not None else set.union(*keys_by_file)
    incomplete = sorted(
        key
        for key in all_keys
        if any(key not in file_keys for file_keys in keys_by_file)
    )
    if incomplete:
        raise SystemExit(
            f"{len(incomplete)} (item, repeat) pairs lack a verdict from every judge "
            f"(first: {incomplete[:5]}). Re-run the short judges with --retry-failed; "
            "the panel never grades on two votes."
        )

    panel = {}
    for key in sorted(all_keys):
        by_judge = votes[key]
        verdicts = {name: by_judge[name]["verdict"] for name in judges}
        binary = sum(verdict_binary(verdict) for verdict in verdicts.values())
        panel[key] = {
            "verdicts": verdicts,
            "correct": 1 if binary >= 2 else 0,
            "n_no_answer": sum(verdict == "no_answer" for verdict in verdicts.values()),
            "unanimous": len(set(verdicts.values())) == 1,
        }
    return panel


def panel_agreement(panel: dict[str, dict]) -> dict:
    """Return three-way and binary Fleiss' kappa, unanimity and panel size."""
    three_way = [Counter(entry["verdicts"].values()) for entry in panel.values()]
    binary = [
        Counter(verdict_binary(value) for value in entry["verdicts"].values())
        for entry in panel.values()
    ]
    return {
        "fleiss_kappa_three_way": round(fleiss_kappa(three_way), 4),
        "fleiss_kappa_binary": round(fleiss_kappa(binary), 4),
        "unanimous_share": round(
            sum(entry["unanimous"] for entry in panel.values()) / len(panel), 4
        ),
        "n": len(panel),
    }


def panel_labels(panel: dict[str, dict], covariates: dict[int, dict] | None = None) -> list[dict]:
    """Aggregate panel verdicts into per-question success and no-answer counts."""
    by_question: dict[int, list[dict]] = {}
    for key, entry in panel.items():
        try:
            question_id = int(key.split(":")[0])
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"panel key {key!r} does not start with an integer question ID"
            ) from exc
        by_question.setdefault(question_id, []).append(entry)
    labels = []
    for question_id in sorted(by_question):
        entries = by_question[question_id]
        row = {
            "question_id": question_id,
            "correct_count": sum(entry["correct"] for entry in entries),
            "no_answer_count": sum(1 for entry in entries if entry["n_no_answer"] > 0),
            "k": len(entries),
        }
        if covariates is not None:
            if question_id not in covariates:
                raise SystemExit(
                    f"question_id {question_id} has no closed-label row — "
                    "cannot attach category/list_style for the draw"
                )
            row["category"] = covariates[question_id]["category"]
            row["list_style"] = covariates[question_id]["list_style"]
        labels.append(row)
    return labels
