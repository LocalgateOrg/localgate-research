"""Run the fixed five-repeat MMLU-Pro label generation protocol."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research.data.generate import DEFAULT_MODEL, MAX_MODEL_LEN, validate_revision

TASK = "mmlu_pro"
REPEATS = 5
SEEDS = tuple(range(42, 47))
MAX_GEN_TOKS = 6144
GEN_KWARGS = "max_gen_toks=6144,do_sample=True,temperature=1.0,top_p=0.95,top_k=64"
GEN_SETTINGS = {
    "max_gen_toks": MAX_GEN_TOKS,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
}
EXPECTED_VERSIONS = {
    "lm_eval": "0.4.12",
    "vllm": "0.25.1",
    "torch": "2.11.0",
    "transformers": "5.14.1",
}


def installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in EXPECTED_VERSIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution).split("+")[0]
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not installed"
    return versions


def command_for_repeat(
    output: Path,
    seed: int,
    *,
    model: str = DEFAULT_MODEL,
    revision: str | None = None,
    trust_remote_code: bool = False,
    limit: int | None = None,
) -> list[str]:
    revision = validate_revision(revision, dry_run=True)
    revision_args = (
        (f"revision={revision}", f"tokenizer_revision={revision}")
        if revision is not None
        else ()
    )
    model_args = ",".join(
        (
            f"pretrained={model}",
            *revision_args,
            f"seed={seed}",
            "dtype=auto",
            f"max_model_len={MAX_MODEL_LEN}",
            "gpu_memory_utilization=0.9",
            "tensor_parallel_size=1",
            "enable_prefix_caching=True",
            f"trust_remote_code={trust_remote_code}",
        )
    )
    command = [
        "lm_eval",
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        TASK,
        "--num_fewshot",
        "0",
        "--batch_size",
        "auto",
        "--apply_chat_template",
        "--gen_kwargs",
        GEN_KWARGS,
        "--seed",
        str(seed),
        "--log_samples",
        "--output_path",
        str(output),
        "--confirm_run_unsafe_code",
    ]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        command.extend(("--limit", str(limit)))
    return command


def build_manifest(
    *,
    model: str,
    limit: int | None,
    versions: dict[str, str],
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model,
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "task": TASK,
        "limit": limit,
        "preset": {
            "name": "label",
            "num_fewshot": 0,
            "max_gen_toks": MAX_GEN_TOKS,
            "thinking": False,
            "sampled": True,
            "repeats": REPEATS,
        },
        "gen_kwargs": GEN_KWARGS,
        "seeds": list(SEEDS),
        "detected": {
            "max_model_len": MAX_MODEL_LEN,
            "dtype": "auto",
            "tensor_parallel_size": 1,
        },
        "versions": versions,
    }


def manifest_problems(existing: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    keys = (
        "model",
        "revision",
        "trust_remote_code",
        "task",
        "limit",
        "preset",
        "gen_kwargs",
        "seeds",
        "versions",
    )
    problems = []
    for key in keys:
        legacy_remote_code = (
            key == "trust_remote_code"
            and key not in existing
            and fresh.get("revision") is None
        )
        if not legacy_remote_code and existing.get(key) != fresh.get(key):
            problems.append(key)
    existing_config = existing.get("detected")
    fresh_config = fresh["detected"]
    if not isinstance(existing_config, dict) or any(
        existing_config.get(key) != value for key, value in fresh_config.items()
    ):
        problems.append("detected")
    return problems


def validate_repeat_result(
    path: Path,
    observations: dict[int, dict[str, Any]],
    *,
    model: str,
    revision: str | None,
    trust_remote_code: bool,
    seed: int,
    limit: int | None,
) -> None:
    """Validate the lm-eval result summary against its samples and fixed protocol."""
    result_files = sorted(path.rglob("results_*.json"))
    if len(result_files) != 1:
        raise ValueError(f"{path} must contain exactly one results_*.json file")
    result_path = result_files[0]
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid lm-eval result {result_path}: {exc}") from exc
    if not isinstance(result, dict):
        raise TypeError(f"{result_path}: expected a JSON object")
    results = result.get("results")
    task_result = results.get(TASK) if isinstance(results, dict) else None
    sample_len = task_result.get("sample_len") if isinstance(task_result, dict) else None
    if type(sample_len) is not int or sample_len < 1:
        raise ValueError(f"{result_path}: results.{TASK}.sample_len must be positive")
    if sample_len != len(observations):
        raise ValueError(
            f"{result_path}: sample_len={sample_len} but samples contain "
            f"{len(observations)} observations"
        )

    config = result.get("config")
    if not isinstance(config, dict):
        raise TypeError(f"{result_path}: missing config object")
    model_args = config.get("model_args")
    if not isinstance(model_args, dict):
        raise TypeError(f"{result_path}: missing config.model_args object")
    expected_model_args = {
        "pretrained": model,
        "seed": seed,
        "dtype": "auto",
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": 0.9,
        "tensor_parallel_size": 1,
        "enable_prefix_caching": True,
        "trust_remote_code": trust_remote_code,
    }
    if revision is not None:
        expected_model_args["revision"] = revision
        expected_model_args["tokenizer_revision"] = revision
    seed_fields = ("random_seed", "numpy_seed", "torch_seed", "fewshot_seed")
    problems = []
    if result.get("model_name") != model:
        problems.append("model_name")
    if config.get("model") != "vllm":
        problems.append("config.model")
    if model_args != expected_model_args:
        problems.append("config.model_args")
    if config.get("limit") != limit:
        problems.append("config.limit")
    if config.get("gen_kwargs") != GEN_SETTINGS:
        problems.append("config.gen_kwargs")
    if any(config.get(field) != seed for field in seed_fields):
        problems.append("config seeds")
    if result.get("lm_eval_version") != EXPECTED_VERSIONS["lm_eval"]:
        problems.append("lm_eval_version")
    if result.get("transformers_version") != EXPECTED_VERSIONS["transformers"]:
        problems.append("transformers_version")
    if problems:
        raise ValueError(f"{result_path}: result protocol differs in {', '.join(problems)}")


def _repeat_complete(
    path: Path,
    *,
    model: str,
    revision: str | None,
    trust_remote_code: bool,
    seed: int,
    limit: int | None,
) -> bool:
    sample_files = sorted(path.rglob("samples_*.jsonl")) if path.exists() else []
    result_files = sorted(path.rglob("results_*.json")) if path.exists() else []
    if not sample_files and not result_files:
        return False
    if not sample_files or not result_files:
        raise ValueError(f"{path} contains a partial lm-eval repeat")
    from research.data.closed_labels import read_repeat

    observations = read_repeat(path)
    validate_repeat_result(
        path,
        observations,
        model=model,
        revision=revision,
        trust_remote_code=trust_remote_code,
        seed=seed,
        limit=limit,
    )
    return True


def _write_new_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid manifest {path}: {exc}") from exc
        if not isinstance(existing, dict):
            raise TypeError(f"invalid manifest {path}: expected a JSON object")
        problems = manifest_problems(existing, manifest)
        if problems:
            raise ValueError(f"{path.parent} manifest differs in: {', '.join(problems)}")
        return
    if path.parent.exists() and any(path.parent.iterdir()):
        raise ValueError(
            f"refusing to claim non-empty run directory without a manifest: {path.parent}"
        )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")


def run(
    *,
    out: Path,
    model: str,
    revision: str | None,
    trust_remote_code: bool,
    limit: int | None,
    dry_run: bool,
) -> int:
    revision = validate_revision(revision, dry_run=dry_run)
    versions = installed_versions()
    manifest = build_manifest(
        model=model,
        revision=revision,
        trust_remote_code=trust_remote_code,
        limit=limit,
        versions=versions,
    )
    commands = [
        command_for_repeat(
            out / f"rep{repeat}",
            seed,
            model=model,
            revision=revision,
            trust_remote_code=trust_remote_code,
            limit=limit,
        )
        for repeat, seed in enumerate(SEEDS)
    ]
    if dry_run:
        print(
            json.dumps(
                {"output": str(out), "manifest": manifest, "commands": commands}, indent=2
            )
        )
        return 0
    if versions != EXPECTED_VERSIONS:
        raise RuntimeError(
            "multiple-choice generation requires "
            f"{EXPECTED_VERSIONS}; installed versions are {versions}"
        )
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    _write_new_manifest(manifest_path, manifest)

    for repeat, (seed, command) in enumerate(zip(SEEDS, commands, strict=True)):
        target = out / f"rep{repeat}"
        if _repeat_complete(
            target,
            model=model,
            revision=revision,
            trust_remote_code=trust_remote_code,
            seed=seed,
            limit=limit,
        ):
            print(f"rep{repeat}: saved")
            continue
        target.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            raise RuntimeError(f"lm_eval failed for rep{repeat} with exit code {result.returncode}")
        if not _repeat_complete(
            target,
            model=model,
            revision=revision,
            trust_remote_code=trust_remote_code,
            seed=seed,
            limit=limit,
        ):
            raise RuntimeError(f"lm_eval returned success without complete logs for rep{repeat}")
        print(f"rep{repeat}: generated")
    return 0
