"""Calculate descriptive statistics for the human-routing study."""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from research.analysis.data_io import InputError, read_records, sha256

CSV_FIELDS = (
    "participant_id", "prompt_id", "category", "which_of_the_following",
    "prompt_text", "rating", "rating_level", "response_time_ms",
    "attention_check", "reflection", "gender", "age", "education",
    "occupation", "genai_freq_post",
)
RATING_NAMES = (
    "Extremely unlikely", "Very unlikely", "Unlikely",
    "Likely", "Very likely", "Extremely likely",
)
ATTENTION_TEXT = (
    'This is an attention check, not a question. To show you are reading, '
    'please select "Very likely" for this item.'
)
PARTICIPANT_FILE = re.compile(r"(?:larp_)?(P\d{2})\.csv\Z")


class ValidationError(ValueError):
    """The supplied study inputs fail a format or consistency check."""


@dataclass(frozen=True)
class Record:
    participant_id: str
    prompt_id: str
    category: str
    list_style: bool
    rating_level: int
    source_file: str
    source_row: int


@dataclass(frozen=True)
class Label:
    prompt_id: str
    correct_count: int
    k: int
    category: str | None
    list_style: bool | None


@dataclass(frozen=True)
class Dataset:
    records: tuple[Record, ...]
    labels: dict[str, Label]
    participant_files: dict[str, str]
    response_hashes: dict[str, str]
    label_hash: str
    attention_checks: int


def truth_level(correct_count: int, k: int) -> int:
    if type(correct_count) is not int or type(k) is not int or k <= 0:
        raise ValidationError("correct_count and k must be integers with k > 0")
    if not 0 <= correct_count <= k:
        raise ValidationError("correct_count must lie in [0, k]")
    return min(6 * correct_count // k, 5)


def linear_weighted_kappa(
    truth: Iterable[int], human: Iterable[int]
) -> Fraction | None:
    truth_values = list(truth)
    human_values = list(human)
    if not truth_values or len(truth_values) != len(human_values):
        raise ValidationError("kappa inputs must be non-empty and equal length")
    if any(type(value) is not int or value not in range(6) for value in truth_values + human_values):
        raise ValidationError("kappa values must use the fixed six-level scale")
    n = len(truth_values)
    truth_counts, human_counts = Counter(truth_values), Counter(human_values)
    observed_distance = sum(abs(a - b) for a, b in zip(truth_values, human_values))
    expected_distance = sum(
        abs(a - b) * truth_counts[a] * human_counts[b]
        for a in range(6) for b in range(6)
    )
    if expected_distance == 0:
        return None
    return Fraction(expected_distance - n * observed_distance, expected_distance)


def _number(value: Fraction | None) -> dict[str, object]:
    if value is None:
        return {"value": None, "exact": None, "undefined_reason": "zero expected disagreement"}
    return {"value": float(value), "exact": f"{value.numerator}/{value.denominator}"}


def _marginal(values: Iterable[int]) -> list[int]:
    counts = Counter(values)
    return [counts[level] for level in range(6)]


def describe_pairs(
    truth: Iterable[int],
    human: Iterable[int],
    truth_local: Iterable[bool] | None = None,
    human_local: Iterable[bool] | None = None,
) -> dict[str, object]:
    truth_values, human_values = list(truth), list(human)
    if not truth_values or len(truth_values) != len(human_values):
        raise ValidationError("descriptor inputs must be non-empty and equal length")
    if any(type(value) is not int or value not in range(6) for value in truth_values + human_values):
        raise ValidationError("descriptor values must use the fixed six-level scale")
    reference_local = list(truth_local) if truth_local is not None else [x >= 3 for x in truth_values]
    routed_local = list(human_local) if human_local is not None else [x >= 3 for x in human_values]
    if len(reference_local) != len(truth_values) or len(routed_local) != len(truth_values):
        raise ValidationError("binary descriptor inputs must match ordinal input length")
    n = len(truth_values)
    confusion = [[0 for _ in range(6)] for _ in range(6)]
    for reference, forecast in zip(truth_values, human_values):
        confusion[reference][forecast] += 1
    tn = fp = fn = tp = 0
    for reference, forecast in zip(reference_local, routed_local):
        if reference and forecast:
            tp += 1
        elif reference:
            fn += 1
        elif forecast:
            fp += 1
        else:
            tn += 1
    reference_count, human_count = sum(reference_local), sum(routed_local)
    return {
        "n": n,
        "linear_weighted_kappa": _number(linear_weighted_kappa(truth_values, human_values)),
        "bias_levels": _number(Fraction(sum(b - a for a, b in zip(truth_values, human_values)), n)),
        "mae_levels": _number(Fraction(sum(abs(b - a) for a, b in zip(truth_values, human_values)), n)),
        "confusion_truth_rows_human_columns": confusion,
        "truth_level_marginal": _marginal(truth_values),
        "human_level_marginal": _marginal(human_values),
        "binary": {
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "accuracy": _number(Fraction(tn + tp, n)),
            "truth_local_count": reference_count,
            "human_local_count": human_count,
            "truth_local_proportion": _number(Fraction(reference_count, n)),
            "human_local_proportion": _number(Fraction(human_count, n)),
            "net_local_count": human_count - reference_count,
            "net_local_percentage_points": float(Fraction(100 * (human_count - reference_count), n)),
        },
    }


def _parse_bool(value: str, field: str) -> bool:
    if value in {"True", "TRUE"}:
        return True
    if value in {"False", "FALSE"}:
        return False
    raise ValidationError(f"{field} must be exactly True/TRUE or False/FALSE")


def _parse_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValidationError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field} must be an integer") from exc
    if str(parsed) != str(value):
        raise ValidationError(f"{field} must be a canonical integer")
    return parsed


def load_and_validate(
    input_dir: Path,
    labels_path: Path,
    *,
    expected_participants: int = 30,
    prompts_per_participant: int = 28,
    expected_prompts: int = 280,
    ratings_per_prompt: int = 3,
) -> Dataset:
    input_dir, labels_path = Path(input_dir), Path(labels_path)
    all_csv = sorted(input_dir.glob("*.csv"), key=lambda path: path.name)
    files = sorted(
        (path for path in all_csv if PARTICIPANT_FILE.fullmatch(path.name)),
        key=lambda path: path.name,
    )
    if len(files) != expected_participants:
        raise ValidationError(f"expected {expected_participants} CSV files, found {len(files)}")

    records: list[Record] = []
    participant_files: dict[str, str] = {}
    response_hashes: dict[str, str] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for path in files:
        match = PARTICIPANT_FILE.fullmatch(path.name)
        if not match:
            raise ValidationError(f"unrecognized participant filename: {path.name}")
        participant = match.group(1)
        if participant in participant_files:
            raise ValidationError(f"duplicate participant file for {participant}")
        participant_files[participant] = path.name
        response_hashes[path.name] = sha256(path)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                raise ValidationError(f"{path.name}: CSV fields do not match the required schema")
            rows = list(reader)
        if len(rows) != prompts_per_participant + 1:
            raise ValidationError(f"{path.name}: expected {prompts_per_participant + 1} rows, found {len(rows)}")
        attention_rows = []
        for source_row, row in enumerate(rows, start=2):
            if row["participant_id"] != participant:
                raise ValidationError(f"{path.name}:{source_row}: participant_id does not match filename")
            level = _parse_int(row["rating_level"], "rating_level")
            if level not in range(6) or row["rating"] != RATING_NAMES[level]:
                raise ValidationError(f"{path.name}:{source_row}: rating_level/rating mismatch")
            response_time = _parse_int(row["response_time_ms"], "response_time_ms")
            if response_time < 0:
                raise ValidationError(f"{path.name}:{source_row}: response_time_ms must be nonnegative")
            if row["attention_check"] != "Likely":
                raise ValidationError(f"{path.name}:{source_row}: post-task attention check failed")
            expected_attention_id = f"ATT-{participant}"
            if row["prompt_id"].startswith("ATT-"):
                if row["prompt_id"] != expected_attention_id:
                    raise ValidationError(f"{path.name}:{source_row}: unmatched attention ID")
                attention_rows.append((row, source_row, level))
                continue
            pair = (participant, row["prompt_id"])
            if pair in seen_pairs:
                raise ValidationError(f"duplicate participant/prompt pair: {participant}/{row['prompt_id']}")
            seen_pairs.add(pair)
            records.append(Record(
                participant, row["prompt_id"], row["category"],
                _parse_bool(row["which_of_the_following"], "which_of_the_following"),
                level, path.name, source_row,
            ))
        if len(attention_rows) != 1:
            raise ValidationError(f"{path.name}: expected exactly one explicit attention row")
        attention, source_row, level = attention_rows[0]
        if attention["prompt_text"] != ATTENTION_TEXT or level != 4 or attention["rating"] != "Very likely":
            raise ValidationError(f"{path.name}:{source_row}: embedded attention check failed")

    expected_ids = {f"P{number:02d}" for number in range(1, expected_participants + 1)}
    if set(participant_files) != expected_ids:
        raise ValidationError("participant IDs do not equal the complete expected assignment set")
    participant_counts = Counter(record.participant_id for record in records)
    if any(participant_counts[participant] != prompts_per_participant for participant in expected_ids):
        raise ValidationError("participant substantive counts are incomplete")

    labels: dict[str, Label] = {}
    try:
        label_rows = read_records(labels_path)
    except InputError as exc:
        raise ValidationError(str(exc)) from exc
    for row_number, raw in enumerate(label_rows, start=1):
        if not {"question_id", "correct_count", "k"} <= raw.keys():
            raise ValidationError(f"labels row {row_number}: missing required fields")
        prompt_id = str(_parse_int(raw["question_id"], "question_id"))
        if prompt_id in labels:
            raise ValidationError(f"duplicate label ID: {prompt_id}")
        correct, k = _parse_int(raw["correct_count"], "correct_count"), _parse_int(raw["k"], "k")
        if k != 30 or not 0 <= correct <= k:
            raise ValidationError(f"label {prompt_id}: expected k=30 and 0 <= correct_count <= k")
        category = raw.get("category")
        list_style = raw.get("list_style")
        if category is not None and not isinstance(category, str):
            raise ValidationError(f"label {prompt_id}: category must be a string")
        if list_style is not None and not isinstance(list_style, bool):
            raise ValidationError(f"label {prompt_id}: list_style must be boolean")
        labels[prompt_id] = Label(prompt_id, correct, k, category, list_style)
    if len(labels) != expected_prompts:
        raise ValidationError(f"expected {expected_prompts} labels, found {len(labels)}")
    record_ids = {record.prompt_id for record in records}
    if record_ids != set(labels):
        raise ValidationError("substantive prompt IDs and label IDs do not join exactly")
    prompt_counts = Counter(record.prompt_id for record in records)
    if any(prompt_counts[prompt_id] != ratings_per_prompt for prompt_id in labels):
        raise ValidationError("each prompt must have exactly the required number of ratings")
    expected_total = expected_prompts * ratings_per_prompt
    if len(records) != expected_total:
        raise ValidationError(f"expected {expected_total} substantive ratings, found {len(records)}")

    stimulus: dict[str, tuple[str, bool]] = {}
    for record in records:
        current = (record.category, record.list_style)
        if record.prompt_id in stimulus and stimulus[record.prompt_id] != current:
            raise ValidationError(f"prompt {record.prompt_id}: inconsistent stimulus metadata")
        stimulus[record.prompt_id] = current
        label = labels[record.prompt_id]
        if label.category is not None and label.category != record.category:
            raise ValidationError(f"prompt {record.prompt_id}: category differs from label")
        if label.list_style is not None and label.list_style != record.list_style:
            raise ValidationError(f"prompt {record.prompt_id}: list_style differs from label")
    return Dataset(
        tuple(records), labels, participant_files, response_hashes,
        sha256(labels_path), expected_participants,
    )


def _prompt_sort(prompt_id: str) -> tuple[int, int | str]:
    return (0, int(prompt_id)) if prompt_id.isdigit() else (1, prompt_id)


def build_report(dataset: Dataset) -> dict[str, object]:
    by_prompt: dict[str, list[Record]] = defaultdict(list)
    by_rater: dict[str, list[Record]] = defaultdict(list)
    for record in dataset.records:
        by_prompt[record.prompt_id].append(record)
        by_rater[record.participant_id].append(record)

    prompt_rows = []
    truth_values, human_values, truth_binary, human_binary = [], [], [], []
    for prompt_id in sorted(by_prompt, key=_prompt_sort):
        ratings = sorted(by_prompt[prompt_id], key=lambda record: record.participant_id)
        levels = sorted(record.rating_level for record in ratings)
        if len(levels) != 3:
            raise ValidationError("primary human aggregation requires exactly three ratings")
        human_level = levels[1]
        majority_local = sum(record.rating_level >= 3 for record in ratings) >= 2
        if majority_local != (human_level >= 3):
            raise ValidationError("binary majority is not equivalent to median threshold")
        label = dataset.labels[prompt_id]
        reference_level = truth_level(label.correct_count, label.k)
        reference_local = 2 * label.correct_count >= label.k
        truth_values.append(reference_level)
        human_values.append(human_level)
        truth_binary.append(reference_local)
        human_binary.append(majority_local)
        prompt_rows.append({
            "prompt_id": prompt_id,
            "category": ratings[0].category,
            "which_of_the_following": ratings[0].list_style,
            "correct_count": label.correct_count,
            "k": label.k,
            "truth_level": reference_level,
            "truth_local": reference_local,
            "human_median_level": human_level,
            "human_majority_local": majority_local,
            "ratings": [
                {"participant_id": record.participant_id, "rating_level": record.rating_level,
                 "source_file": record.source_file, "source_row": record.source_row}
                for record in ratings
            ],
        })

    overall = describe_pairs(truth_values, human_values, truth_binary, human_binary)
    per_rater = []
    for participant in sorted(by_rater):
        ratings = sorted(by_rater[participant], key=lambda record: _prompt_sort(record.prompt_id))
        references = [truth_level(dataset.labels[r.prompt_id].correct_count, dataset.labels[r.prompt_id].k) for r in ratings]
        forecasts = [r.rating_level for r in ratings]
        references_local = [2 * dataset.labels[r.prompt_id].correct_count >= dataset.labels[r.prompt_id].k for r in ratings]
        forecasts_local = [r.rating_level >= 3 for r in ratings]
        per_rater.append({
            "participant_id": participant,
            "source_file": dataset.participant_files[participant],
            "descriptors": describe_pairs(references, forecasts, references_local, forecasts_local),
            "lowest_level_count": forecasts.count(0),
        })
    individual_levels = [record.rating_level for record in dataset.records]
    lowest_raters = sorted({record.participant_id for record in dataset.records if record.rating_level == 0})
    return {
        "schema_version": 1,
        "analysis": "descriptive human-study statistics",
        "configuration": {
            "scale_levels": 6,
            "truth_level": "min(floor(6*correct_count/k),5)",
            "human_aggregate": "median of exactly three ordinal ratings",
            "binary_truth": "2*correct_count >= k",
            "binary_human": "majority of individual rating_level >= 3",
            "kappa": "Cohen linear weighted, fixed categories 0..5",
        },
        "provenance": {
            "labels_sha256": dataset.label_hash,
            "response_files": [
                {"path": name, "sha256": dataset.response_hashes[name]}
                for name in sorted(dataset.response_hashes)
            ],
        },
        "validation": {
            "participants": len(dataset.participant_files),
            "prompts": len(dataset.labels),
            "substantive_ratings": len(dataset.records),
            "ratings_per_prompt": 3,
            "attention_checks_passed": dataset.attention_checks,
        },
        "overall": overall,
        "individual_ratings": {
            "marginal": _marginal(individual_levels),
            "lowest_level_count": individual_levels.count(0),
            "raters_using_lowest_level_count": len(lowest_raters),
            "raters_using_lowest_level": lowest_raters,
        },
        "per_prompt": prompt_rows,
        "per_rater": per_rater,
    }


def write_report(report: dict[str, object], output_path: Path) -> None:
    output_path = Path(output_path)
    if not output_path.is_absolute():
        raise ValidationError("output path must be explicit and absolute")
    if not output_path.parent.is_dir():
        raise ValidationError("output parent directory must already exist")
    if os.stat(output_path.parent).st_mode & 0o077:
        raise ValidationError("output parent directory must be private (mode 0700 or stricter)")
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValidationError(f"refusing to overwrite existing output: {output_path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


@dataclass(frozen=True)
class PromptPairs:
    prompt_ids: tuple[str, ...]
    truth: tuple[int, ...]
    human: tuple[int, ...]
    truth_local: tuple[bool, ...]
    human_local: tuple[bool, ...]


def _exact_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValidationError(f"{field} must be an exact integer")
    return value


def prompt_pairs(report: object, *, expected_prompts: int = 280) -> PromptPairs:
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValidationError("unsupported descriptive report schema")
    rows = report.get("per_prompt")
    if not isinstance(rows, list) or len(rows) != expected_prompts:
        raise ValidationError(f"descriptive report must contain {expected_prompts} prompt rows")

    prompt_ids: list[str] = []
    truth: list[int] = []
    human: list[int] = []
    truth_local: list[bool] = []
    human_local: list[bool] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValidationError(f"prompt row {index} must be an object")
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValidationError(f"prompt row {index} has an invalid prompt_id")
        reference = _exact_int(row.get("truth_level"), "truth_level")
        forecast = _exact_int(row.get("human_median_level"), "human_median_level")
        if reference not in range(6) or forecast not in range(6):
            raise ValidationError("prompt levels must use the fixed six-level scale")
        reference_local = row.get("truth_local")
        forecast_local = row.get("human_majority_local")
        if type(reference_local) is not bool or type(forecast_local) is not bool:
            raise ValidationError("prompt binary labels must be booleans")
        correct_count = _exact_int(row.get("correct_count"), "correct_count")
        k = _exact_int(row.get("k"), "k")
        if k != 30:
            raise ValidationError(f"prompt {prompt_id}: expected k=30")
        if truth_level(correct_count, k) != reference:
            raise ValidationError(f"prompt {prompt_id}: truth level is inconsistent")
        if (2 * correct_count >= k) != reference_local:
            raise ValidationError(f"prompt {prompt_id}: truth binary label is inconsistent")
        if (forecast >= 3) != forecast_local:
            raise ValidationError(f"prompt {prompt_id}: human binary label is inconsistent")
        prompt_ids.append(prompt_id)
        truth.append(reference)
        human.append(forecast)
        truth_local.append(reference_local)
        human_local.append(forecast_local)
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValidationError("descriptive report contains duplicate prompt IDs")

    recalculated = describe_pairs(truth, human, truth_local, human_local)
    overall = report.get("overall")
    if not isinstance(overall, dict):
        raise ValidationError("descriptive report is missing overall results")
    if overall.get("linear_weighted_kappa") != recalculated["linear_weighted_kappa"]:
        raise ValidationError("descriptive report kappa does not match prompt rows")
    if overall.get("binary") != recalculated["binary"]:
        raise ValidationError("descriptive report binary results do not match prompt rows")
    return PromptPairs(
        tuple(prompt_ids), tuple(truth), tuple(human),
        tuple(truth_local), tuple(human_local),
    )


def load_prompt_pairs(path: Path, *, expected_prompts: int = 280) -> PromptPairs:
    with Path(path).open(encoding="utf-8") as handle:
        return prompt_pairs(json.load(handle), expected_prompts=expected_prompts)
