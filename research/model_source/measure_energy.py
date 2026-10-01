"""Measure warm, sequential CPU inference energy for the selected classifier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

OUTPUT_PATH = Path("output/classifier_energy.csv")

MODEL_ID = "answerdotai/ModernBERT-base"
BASE_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
EXPECTED_COUNT = 1655
REFERENCE_TOLERANCE = 1e-5
EXPECTED_HASHES = {
    "converted.jsonl": "be78cccedb7778e479f5daec83c75f56b8aeac42b4c4225c5b78a8827586d4fd",
    "labels_open.jsonl": "7a8f03c18c1d657fdff12fc2480410122253c0726b2df627e5a7fc81600b58ec",
    "labels_study_k30.jsonl": "68a067682507c7153e175a474191fccd1a40a2f5f59fa4394b0f94c520424cb3",
    "split.json": "a0f57e29add58d82527713f66c139e35af319bafe795038a05925808e2434e38",
    "study_prompts_corpus.jsonl": "ab3eb42bbf8d9dc81bf688a0a6f14cfeebf742551e1b1750e2bb414dadc5f0b2",
}
EXPECTED_SNAPSHOT_HASHES = {
    "config.json": "1609d59e627c33eaed524b4f01e546d42e84190a079a5a5ded84b212c41c324f",
    "model.safetensors": "340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9",
    "special_tokens_map.json": "ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049",
    "tokenizer.json": "9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1",
    "tokenizer_config.json": "3cd2017ff46d0a527e5d39cae39272eccfa1f19bb9f89b05d166aab2e38354e2",
}
INPUT_NAMES = (
    "converted.jsonl",
    "labels_open.jsonl",
    "labels_study_k30.jsonl",
    "split.json",
    "study_prompts_corpus.jsonl",
)


class ValidationError(ValueError):
    """An input differs from the pinned measurement contract."""


@dataclass(frozen=True)
class RaplDomain:
    path: Path
    name: str
    max_energy_uj: int


class RaplSampler:
    """Poll cumulative RAPL counters and integrate across counter wraparound."""

    def __init__(self, domains: tuple[RaplDomain, ...], interval_seconds: float = 1.0):
        self.domains = domains
        self.interval_seconds = interval_seconds
        self.energy_uj = {domain.path: 0 for domain in domains}
        self.previous: dict[Path, int] = {}
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _read(domain: RaplDomain) -> int:
        return int((domain.path / "energy_uj").read_text(encoding="ascii").strip())

    def _sample(self) -> None:
        for domain in self.domains:
            current = self._read(domain)
            previous = self.previous.get(domain.path)
            if previous is not None:
                delta = current - previous
                if delta < 0:
                    delta += domain.max_energy_uj
                self.energy_uj[domain.path] += delta
            self.previous[domain.path] = current

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._sample()
            except (OSError, ValueError) as exc:  # surfaced by stop()
                self.error = exc
                return

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._run, name="rapl-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        if self.error is not None:
            raise RuntimeError("RAPL sampling failed") from self.error
        totals: dict[str, float] = {}
        for domain in self.domains:
            totals[domain.name] = totals.get(domain.name, 0.0) + self.energy_uj[domain.path] / 3.6e12
        return totals


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ValidationError(f"missing pinned file: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValidationError(f"hash mismatch for {path}: {actual}")


def load_workload(data_dir: Path) -> tuple[dict, ...]:
    for name in INPUT_NAMES:
        verify_file(data_dir / name, EXPECTED_HASHES[name])
    converted = {}
    for line in (data_dir / "converted.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        converted[row["question_id"]] = row
    split = json.loads((data_dir / "split.json").read_text(encoding="utf-8"))
    assignment = {int(key): value for key, value in split["assignment"].items()}
    rows = tuple(
        {"question_id": question_id, "text": converted[question_id]["open_question"]}
        for question_id in sorted(converted)
        if converted[question_id].get("convertible") is True
        and assignment.get(question_id) == "test"
    )
    if len(rows) != EXPECTED_COUNT:
        raise ValidationError(f"expected {EXPECTED_COUNT} test prompts, found {len(rows)}")
    return rows


def load_reference_predictions(path: Path) -> dict[int, float]:
    reference = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValidationError(f"reference row {line_number} is not an object")
        question_id = row.get("question_id")
        probability = row.get("probability")
        if type(question_id) is not int or question_id in reference:
            raise ValidationError(f"reference row {line_number} has an invalid or duplicate ID")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValidationError(f"reference row {line_number} has an invalid probability")
        reference[question_id] = float(probability)
    if len(reference) != EXPECTED_COUNT:
        raise ValidationError(
            f"expected {EXPECTED_COUNT} reference predictions, found {len(reference)}"
        )
    return reference


def load_model(config_path: Path, base_model: Path, adapter: Path):
    snapshot = base_model
    for name, expected in EXPECTED_SNAPSHOT_HASHES.items():
        verify_file(snapshot / name, expected)
    from transformers import AutoTokenizer

    from .localgate_research.prediction import load_recipe, verify_adapter
    from .localgate_research.training import build_model

    config = replace(
        load_recipe(config_path, snapshot),
        model_name=str(snapshot),
        bf16=False,
        attention_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), trust_remote_code=False, local_files_only=True
    )
    verify_adapter(config_path, adapter)
    model = build_model(config, adapter, trainable=False).to("cpu")
    model.eval()
    return model, tokenizer, config


def discover_rapl_domains() -> tuple[RaplDomain, ...]:
    """Select package and DRAM domains, excluding overlapping core/uncore counters."""
    domains = []
    root = Path("/sys/class/powercap")
    candidates = {*root.glob("intel-rapl:*"), *root.rglob("intel-rapl:*")}
    seen = set()
    for path in sorted(candidates):
        try:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            name = (path / "name").read_text(encoding="ascii").strip()
            if not (name.startswith("package-") or name == "dram"):
                continue
            energy_path = path / "energy_uj"
            int(energy_path.read_text(encoding="ascii").strip())
            max_energy = int((path / "max_energy_range_uj").read_text(encoding="ascii").strip())
        except (FileNotFoundError, PermissionError, ValueError):
            continue
        domains.append(RaplDomain(path, name, max_energy))
    return tuple(domains)


def infer(model, tokenizer, config, rows: tuple[dict, ...]) -> tuple[str, float, tuple[tuple[int, float], ...]]:
    import torch

    from .localgate_research.heads import decode

    checksum = hashlib.sha256()
    probability_sum = 0.0
    predictions = []
    with torch.inference_mode():
        for row in rows:
            encoded = tokenizer(
                row["text"],
                truncation=True,
                max_length=config.max_length,
                padding=True,
                return_tensors="pt",
            )
            logits = model(**encoded).logits.float()
            probabilities, _ = decode(logits, "beta_binomial")
            probability = float(probabilities[0])
            probability_sum += probability
            checksum.update(f"{row['question_id']}:{probability:.17g}\n".encode())
            predictions.append((row["question_id"], probability))
    return checksum.hexdigest(), probability_sum, tuple(predictions)


def compare_with_reference(
    predictions: tuple[tuple[int, float], ...], reference: dict[int, float]
) -> float:
    predicted = {}
    for question_id, probability in predictions:
        if type(question_id) is not int or question_id in predicted:
            raise ValidationError("measured predictions have an invalid or duplicate ID")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValidationError("measured predictions have an invalid probability")
        predicted[question_id] = float(probability)
    if set(predicted) != set(reference):
        raise ValidationError("measured prediction IDs differ from the pinned replay")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in reference.values()):
        raise ValidationError("reference predictions contain an invalid probability")
    max_abs_error = max(
        abs(probability - reference[question_id])
        for question_id, probability in predicted.items()
    )
    if not math.isfinite(max_abs_error) or max_abs_error > REFERENCE_TOLERANCE:
        raise ValidationError(
            f"measured predictions differ from pinned replay: max abs error {max_abs_error}"
        )
    return max_abs_error


def codecarbon_measurement(model, tokenizer, config, rows: tuple[dict, ...], reference):
    from codecarbon import OfflineEmissionsTracker

    tracker = OfflineEmissionsTracker(
        country_iso_code="DEU",
        save_to_file=False,
        log_level="error",
        allow_multiple_runs=True,
        measure_power_secs=1,
        tracking_mode="machine",
    )
    tracker.start()
    started = time.perf_counter()
    checksum, probability_sum, predictions = infer(model, tokenizer, config, rows)
    duration = time.perf_counter() - started
    tracker.stop()
    data = tracker.final_emissions_data
    if data is None:
        raise RuntimeError("CodeCarbon produced no final energy data")
    cpu_kwh = float(data.cpu_energy)
    ram_kwh = float(data.ram_energy)
    return {
        "measurement_mode": "codecarbon_model_estimate",
        "hardware_measured": "false",
        "measurement_scope": "software-modeled CPU plus RAM only; GPU excluded; no idle subtraction",
        "rapl_domains": "",
        "duration_seconds": duration,
        "idle_duration_seconds": "",
        "energy_kwh": cpu_kwh + ram_kwh,
        "gross_energy_kwh": cpu_kwh + ram_kwh,
        "idle_energy_kwh": "",
        "idle_raw_energy_kwh": "",
        "marginal_energy_kwh": "",
        "marginal_raw_energy_kwh": "",
        "marginal_clamped": "",
        "cpu_kwh": cpu_kwh,
        "ram_kwh": ram_kwh,
        "gpu_kwh": 0.0,
        "cpu_power_w": float(data.cpu_power),
        "ram_power_w": float(data.ram_power),
        "prediction_checksum": checksum,
        "probability_sum": probability_sum,
        "reference_max_abs_error": compare_with_reference(predictions, reference),
    }


def rapl_measurement(domains, model, tokenizer, config, rows, reference):
    sampler = RaplSampler(domains)
    sampler.start()
    started = time.perf_counter()
    checksum, probability_sum, predictions = infer(model, tokenizer, config, rows)
    duration = time.perf_counter() - started
    energy = sampler.stop()
    idle_sampler = RaplSampler(domains)
    idle_sampler.start()
    idle_started = time.perf_counter()
    time.sleep(duration)
    idle_duration = time.perf_counter() - idle_started
    idle_energy = idle_sampler.stop()
    package_kwh = sum(value for name, value in energy.items() if name.startswith("package-"))
    dram_kwh = energy.get("dram", 0.0)
    idle_package_kwh = sum(
        value for name, value in idle_energy.items() if name.startswith("package-")
    )
    idle_dram_kwh = idle_energy.get("dram", 0.0)
    gross_kwh = package_kwh + dram_kwh
    idle_raw_kwh = idle_package_kwh + idle_dram_kwh
    idle_kwh = idle_raw_kwh * duration / idle_duration
    marginal_raw_kwh = gross_kwh - idle_kwh
    if marginal_raw_kwh <= 0:
        raise RuntimeError(
            "nonpositive idle-subtracted RAPL energy; reject this pass instead of reporting "
            f"an overhead (gross={gross_kwh}, normalized_idle={idle_kwh})"
        )
    marginal_kwh = marginal_raw_kwh
    names = sorted(energy)
    return {
        "measurement_mode": "rapl_cumulative_counters",
        "hardware_measured": "true",
        "measurement_scope": (
            "RAPL " + " plus ".join(names) + "; core and uncore excluded; matched idle subtracted"
        ),
        "rapl_domains": ";".join(f"{domain.name}:{domain.path}" for domain in domains),
        "duration_seconds": duration,
        "idle_duration_seconds": idle_duration,
        "energy_kwh": marginal_kwh,
        "gross_energy_kwh": gross_kwh,
        "idle_energy_kwh": idle_kwh,
        "idle_raw_energy_kwh": idle_raw_kwh,
        "marginal_energy_kwh": marginal_kwh,
        "marginal_raw_energy_kwh": marginal_raw_kwh,
        "marginal_clamped": "false",
        "cpu_kwh": package_kwh,
        "ram_kwh": dram_kwh,
        "gpu_kwh": 0.0,
        "cpu_power_w": package_kwh * 3.6e6 / duration,
        "ram_power_w": dram_kwh * 3.6e6 / duration,
        "prediction_checksum": checksum,
        "probability_sum": probability_sum,
        "reference_max_abs_error": compare_with_reference(predictions, reference),
    }


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def power_condition(prefix: str) -> dict[str, object]:
    batteries = []
    external_online = False
    for supply in sorted(Path("/sys/class/power_supply").glob("*")):
        try:
            supply_type = (supply / "type").read_text(encoding="ascii").strip()
        except (FileNotFoundError, PermissionError):
            continue
        if supply_type == "Battery":
            status_path = supply / "status"
            capacity_path = supply / "capacity"
            batteries.append(
                {
                    "name": supply.name,
                    "status": status_path.read_text(encoding="ascii").strip(),
                    "capacity": int(capacity_path.read_text(encoding="ascii").strip()),
                }
            )
        elif supply_type in {"Mains", "USB", "USB_C"}:
            online_path = supply / "online"
            if online_path.is_file():
                external_online |= online_path.read_text(encoding="ascii").strip() == "1"
    profile_path = Path("/sys/firmware/acpi/platform_profile")
    profile = profile_path.read_text(encoding="ascii").strip() if profile_path.is_file() else ""
    return {
        f"battery_{prefix}_status": ";".join(
            f"{battery['name']}:{battery['status']}" for battery in batteries
        ),
        f"battery_{prefix}_capacity_percent": ";".join(
            f"{battery['name']}:{battery['capacity']}" for battery in batteries
        ),
        f"external_power_{prefix}_online": str(external_online).lower(),
        f"platform_profile_{prefix}": profile,
    }


def common_fields(args, measurement_mode: str) -> dict[str, object]:
    config_document = json.loads(args.config.read_text(encoding="utf-8"))
    adapter_weights = args.adapter / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        adapter_weights = args.adapter / "adapter_model.bin"
    return {
        "model_id": MODEL_ID,
        "fit_id": config_document.get("fit_id", sha256(args.config)),
        "base_revision": BASE_REVISION,
        "base_model_sha256": sha256(args.base_model / "model.safetensors"),
        "adapter_sha256": sha256(adapter_weights),
        "adapter_config_sha256": sha256(args.adapter / "adapter_config.json"),
        "fit_sha256": sha256(args.config),
        "converted_sha256": EXPECTED_HASHES["converted.jsonl"],
        "labels_open_sha256": EXPECTED_HASHES["labels_open.jsonl"],
        "labels_study_sha256": EXPECTED_HASHES["labels_study_k30.jsonl"],
        "split_sha256": EXPECTED_HASHES["split.json"],
        "study_corpus_sha256": EXPECTED_HASHES["study_prompts_corpus.jsonl"],
        "reference_predictions_sha256": sha256(args.reference_predictions),
        "reference_tolerance": REFERENCE_TOLERANCE,
        "script_sha256": sha256(Path(__file__)),
        "device": "cpu",
        "cpu_model": cpu_model(),
        "threads": args.threads,
        "batch_size": 1,
        "warmup_queries": args.warmup_queries,
        "attention_implementation": "eager",
        "dtype": "float32",
        "python_version": platform.python_version(),
        "torch_version": version("torch"),
        "transformers_version": version("transformers"),
        "peft_version": version("peft"),
        "codecarbon_version": version("codecarbon") if measurement_mode.startswith("codecarbon") else "",
        "os": platform.platform(),
    }


def write_csv(path: Path, rows: list[dict[str, object]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite to replace it")
    fields = list(rows[0])
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup-queries", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.passes < 1 or args.threads < 1 or args.warmup_queries < 1:
        parser.error("passes, threads, and warmup queries must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    rows = load_workload(args.data_dir)
    reference = load_reference_predictions(args.reference_predictions)
    model, tokenizer, config = load_model(args.config, args.base_model, args.adapter)
    infer(model, tokenizer, config, rows[: args.warmup_queries])

    domains = discover_rapl_domains()
    mode = "rapl_cumulative_counters" if domains else "codecarbon_model_estimate"
    print(f"measurement_mode={mode}", flush=True)
    print(f"queries_per_pass={len(rows)} passes={args.passes} batch_size=1 threads={args.threads}", flush=True)
    print(f"output={args.output}", flush=True)
    records = []
    for pass_number in range(1, args.passes + 1):
        print(f"pass={pass_number} status=started", flush=True)
        started_utc = datetime.now(UTC).isoformat()
        condition_start = power_condition("start")
        if domains:
            measured = rapl_measurement(domains, model, tokenizer, config, rows, reference)
        else:
            measured = codecarbon_measurement(model, tokenizer, config, rows, reference)
        condition_end = power_condition("end")
        energy_wh = measured["energy_kwh"] * 1000
        record = {
            "record_type": "pass",
            "pass": pass_number,
            "started_utc": started_utc,
            **condition_start,
            **condition_end,
            **measured,
            "query_count": len(rows),
            "energy_wh": energy_wh,
            "wh_per_query": energy_wh / len(rows),
            "gross_wh_per_query": measured["gross_energy_kwh"] * 1000 / len(rows),
            "idle_wh_per_query": (
                measured["idle_energy_kwh"] * 1000 / len(rows)
                if measured["idle_energy_kwh"] != ""
                else ""
            ),
            "marginal_wh_per_query": (
                measured["marginal_energy_kwh"] * 1000 / len(rows)
                if measured["marginal_energy_kwh"] != ""
                else ""
            ),
            "seconds_per_query": measured["duration_seconds"] / len(rows),
            "wh_per_query_sd": "",
            "passes": 1,
            **common_fields(args, mode),
        }
        records.append(record)
        print(
            f"pass={pass_number} duration_seconds={measured['duration_seconds']:.3f} "
            f"wh_per_query={record['wh_per_query']:.9f}",
            flush=True,
        )

    checksums = {record["prediction_checksum"] for record in records}
    if len(checksums) != 1:
        raise RuntimeError("prediction checksums differ across passes")
    total_energy_kwh = sum(record["energy_kwh"] for record in records)
    total_duration = sum(record["duration_seconds"] for record in records)
    total_queries = sum(record["query_count"] for record in records)
    per_query = [record["wh_per_query"] for record in records]
    aggregate = {
        **records[0],
        "record_type": "aggregate",
        "pass": "all",
        "started_utc": records[0]["started_utc"],
        "battery_end_status": records[-1]["battery_end_status"],
        "battery_end_capacity_percent": records[-1]["battery_end_capacity_percent"],
        "external_power_end_online": records[-1]["external_power_end_online"],
        "platform_profile_end": records[-1]["platform_profile_end"],
        "duration_seconds": total_duration,
        "energy_kwh": total_energy_kwh,
        "gross_energy_kwh": sum(record["gross_energy_kwh"] for record in records),
        "idle_energy_kwh": (
            sum(record["idle_energy_kwh"] for record in records)
            if mode == "rapl_cumulative_counters"
            else ""
        ),
        "marginal_energy_kwh": (
            sum(record["marginal_energy_kwh"] for record in records)
            if mode == "rapl_cumulative_counters"
            else ""
        ),
        "marginal_raw_energy_kwh": (
            sum(record["marginal_raw_energy_kwh"] for record in records)
            if mode == "rapl_cumulative_counters"
            else ""
        ),
        "marginal_clamped": "false" if mode == "rapl_cumulative_counters" else "",
        "idle_duration_seconds": (
            sum(record["idle_duration_seconds"] for record in records)
            if mode == "rapl_cumulative_counters"
            else ""
        ),
        "idle_raw_energy_kwh": (
            sum(record["idle_raw_energy_kwh"] for record in records)
            if mode == "rapl_cumulative_counters"
            else ""
        ),
        "cpu_kwh": sum(record["cpu_kwh"] for record in records),
        "ram_kwh": sum(record["ram_kwh"] for record in records),
        "gpu_kwh": sum(record["gpu_kwh"] for record in records),
        "query_count": total_queries,
        "energy_wh": total_energy_kwh * 1000,
        "wh_per_query": total_energy_kwh * 1000 / total_queries,
        "gross_wh_per_query": (
            sum(record["gross_energy_kwh"] for record in records) * 1000 / total_queries
        ),
        "idle_wh_per_query": (
            sum(record["idle_energy_kwh"] for record in records) * 1000 / total_queries
            if mode == "rapl_cumulative_counters"
            else ""
        ),
        "marginal_wh_per_query": (
            sum(record["marginal_energy_kwh"] for record in records) * 1000 / total_queries
            if mode == "rapl_cumulative_counters"
            else ""
        ),
        "seconds_per_query": total_duration / total_queries,
        "wh_per_query_sd": statistics.stdev(per_query) if len(per_query) > 1 else 0.0,
        "passes": len(records),
        "cpu_power_w": sum(record["cpu_kwh"] for record in records) * 3.6e6 / total_duration,
        "ram_power_w": sum(record["ram_kwh"] for record in records) * 3.6e6 / total_duration,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, [*records, aggregate], args.overwrite)
    print(f"aggregate_wh_per_query={aggregate['wh_per_query']:.9f}", flush=True)
    print(f"csv={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    main()
