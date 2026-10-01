"""One training loop for fixed fits, sweeps, cross-validation and final refits."""

from __future__ import annotations

import fcntl
import gc
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import signal
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .artifacts import (
    atomic_json,
    contained_path,
    identity,
    read_json,
    read_jsonl,
    sha256_file,
    verify_files,
    write_jsonl,
)

if TYPE_CHECKING:
    from .config import TrainConfig


def scientific_config(config: TrainConfig) -> dict:
    return config.to_dict()


def build_model(
    config: TrainConfig,
    adapter_path: Path | None = None,
    *,
    trainable: bool = True,
) -> Any:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoConfig, ModernBertForSequenceClassification

    from .heads import output_width

    model_config = AutoConfig.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        trust_remote_code=False,
        local_files_only=True,
    )
    if model_config.model_type != "modernbert":
        raise ValueError("The shared recipe requires ModernBERT")
    model_config.num_labels = output_width(config.head)
    model_config.classifier_pooling = "mean"
    model_config.classifier_dropout = config.readout_dropout
    base = ModernBertForSequenceClassification.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        config=model_config,
        attn_implementation=config.attention_implementation,
        trust_remote_code=False,
        local_files_only=True,
    )
    if adapter_path is not None:
        return PeftModel.from_pretrained(
            base,
            adapter_path,
            is_trainable=trainable,
            local_files_only=True,
        )
    adapter = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["Wqkv"],
        modules_to_save=["head", "classifier"],
        bias="none",
    )
    model = get_peft_model(base, adapter)
    validate_trainable_parameters(model)
    return model


def validate_trainable_parameters(model: Any) -> None:
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    for name in names:
        adapter = ".Wqkv.lora_" in name
        readout = any(f".{part}.modules_to_save." in name for part in ("head", "classifier"))
        if not (adapter or readout):
            raise ValueError(f"Unexpected trainable parameter: {name}")
    for component in (
        ".Wqkv.lora_A.",
        ".Wqkv.lora_B.",
        ".head.modules_to_save.",
        ".classifier.modules_to_save.",
    ):
        if not any(component in name for name in names):
            raise ValueError(f"Missing trainable component: {component}")
    expected = model.get_base_model().config.num_hidden_layers
    if sum(".Wqkv.lora_A." in name for name in names) != expected:
        raise ValueError("LoRA must cover the fused attention projection in every encoder layer")


def optimizer_groups(model: Any, weight_decay: float) -> list[dict]:
    decayed, exempt = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            target = exempt if name.endswith(".bias") or parameter.ndim < 2 else decayed
            target.append(parameter)
    if not decayed:
        raise ValueError("No trainable matrix parameters")
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": exempt, "weight_decay": 0.0},
    ]


def _resolve_roles(
    rows: list[dict],
    config: TrainConfig,
    train_ids: list[int] | None,
    validation_ids: list[int] | None,
    assessment_ids: list[int] | None,
) -> tuple[list[list[int]], list[list[dict]]]:
    indexed = {row["question_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate development question IDs")
    train_ids = (
        train_ids
        if train_ids is not None
        else [
            row["question_id"] for row in rows if config.stage == "final" or row["split"] == "train"
        ]
    )
    validation_ids = (
        validation_ids
        if validation_ids is not None
        else [
            row["question_id"]
            for row in rows
            if config.stage != "final" and row["split"] in {"val", "validation"}
        ]
    )
    assessment_ids = assessment_ids if assessment_ids is not None else []
    roles = [list(train_ids), list(validation_ids), list(assessment_ids)]
    if not roles[0] or (config.stage != "final" and not roles[1]):
        raise ValueError("Fitting and checkpoint-selection roles must be nonempty")
    if config.stage == "final" and (roles[1] or roles[2]):
        raise ValueError("Final refits cannot select or assess checkpoints on held-out roles")
    if config.stage == "final" and set(roles[0]) != set(indexed):
        raise ValueError("Final refits must use the complete development pool")
    if config.stage in {"pilot", "hpo", "replication"}:
        expected_train = {row["question_id"] for row in rows if row["split"] == "train"}
        expected_validation = {
            row["question_id"] for row in rows if row["split"] in {"val", "validation"}
        }
        if set(roles[0]) != expected_train or set(roles[1]) != expected_validation or roles[2]:
            raise ValueError("Screening must retain the original development roles")
    if config.stage == "cv" and (config.fold is None or not roles[2]):
        raise ValueError("Cross-validation requires a declared fold and outer assessment role")
    families = []
    for ids in roles:
        if len(set(ids)) != len(ids) or any(item not in indexed for item in ids):
            raise ValueError("Role IDs must be unique members of the verified development manifest")
        families.append({indexed[item]["family_id"] for item in ids})
    for index in range(3):
        for other in range(index):
            if families[index] & families[other]:
                raise ValueError("Question families overlap across training or evaluation roles")
    return roles, [[indexed[item] for item in ids] for ids in roles]


def _tokenize(
    tokenizer: Any, rows: list[dict], config: TrainConfig, directory: Path
) -> dict[int, dict]:
    tokens = {}
    lengths = {}
    for row in rows:
        encoded = tokenizer(row["text"], truncation=False, return_attention_mask=True)
        length = len(encoded["input_ids"])
        lengths[str(row["question_id"])] = length
        tokens[row["question_id"]] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
    audit = {
        "tokenizer_revision": config.tokenizer_revision,
        "max_length": config.max_length,
        "question_lengths": lengths,
        "truncated_ids": [int(key) for key, value in lengths.items() if value > config.max_length],
    }
    atomic_json(directory / "token_lengths.json", audit)
    if audit["truncated_ids"]:
        raise ValueError(
            "Development inputs exceed the frozen token cap; "
            "review token_lengths.json before launch"
        )
    return tokens


def _batch(tokenizer: Any, rows: list[dict], tokens: dict, device: Any) -> dict:
    padded = tokenizer.pad(
        [tokens[row["question_id"]] for row in rows],
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in padded.items()}


def predict(
    model: Any,
    tokenizer: Any,
    rows: list[dict],
    tokens: dict,
    config: TrainConfig,
    device: Any,
) -> list[dict]:
    import torch

    from .heads import decode

    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), config.eval_batch_size):
            batch_rows = rows[start : start + config.eval_batch_size]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.bf16):
                logits = model(**_batch(tokenizer, batch_rows, tokens, device)).logits
            mean, pmf = decode(logits.float(), config.head)
            for index, row in enumerate(batch_rows):
                native_pmf = None if pmf is None else pmf[index].double().cpu().tolist()
                record = {
                    "question_id": row["question_id"],
                    "family_id": row["family_id"],
                    "category": row["category"],
                    "text": row["text"],
                    "correct_count": row["correct_count"],
                    "k": row["k"],
                    "list_style": row.get("list_style"),
                    "logits": logits[index].double().cpu().tolist(),
                    "probability": float(mean[index]),
                    "predicted_level": min(int(6 * float(mean[index])), 5),
                    "native_k5_pmf": native_pmf,
                }
                if config.head == "twin":
                    record["auxiliary_binomial_probability"] = float(
                        logits[index, 6].float().sigmoid()
                    )
                predictions.append(record)
    return predictions


def _metrics_reproduce(computed: dict, expected: dict) -> bool:
    """Compare recomputed metrics tolerantly.

    Float summation order differs between machines, so a receipt written on the
    training host differed from a local recomputation by one or two ULPs and the
    exact comparison rejected it. Corruption moves these numbers by far more than
    the tolerance already used for the selection-rule check above.
    """
    if computed.keys() != expected.keys():
        return False
    for key, value in computed.items():
        other = expected[key]
        if isinstance(value, float) and isinstance(other, float):
            if not math.isclose(value, other, abs_tol=1e-8, rel_tol=1e-8):
                return False
        elif value != other:
            return False
    return True


def prediction_metrics(predictions: list[dict]) -> dict:
    from .metrics import evaluate_predictions

    if not predictions:
        raise ValueError("Evaluation must contain at least one question")
    has_pmf = [row["native_k5_pmf"] is not None for row in predictions]
    if any(has_pmf) != all(has_pmf):
        raise ValueError("Mixed probability-distribution availability")
    return evaluate_predictions(
        [row["correct_count"] for row in predictions],
        [row["probability"] for row in predictions],
        pmf=[row["native_k5_pmf"] for row in predictions] if all(has_pmf) else None,
        k=5,
    )


def _rng_state() -> dict:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(
    directory: Path,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    state: dict,
    fit_id: str,
    *,
    best: bool,
) -> Path:
    import torch

    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    name = f"step-{state['global_step']}-{uuid.uuid4().hex[:8]}"
    temporary = checkpoints / f".{name}"
    destination = checkpoints / name
    temporary.mkdir()
    if best:
        state["best_checkpoint"] = name
    try:
        model.save_pretrained(temporary / "adapter", safe_serialization=True)
        tokenizer.save_pretrained(temporary / "tokenizer")
        snapshot = {
            **state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": _rng_state(),
            "logical_fit_id": fit_id,
        }
        torch.save(snapshot, temporary / "training_state.pt")
        files = {
            str(path.relative_to(temporary)): sha256_file(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        atomic_json(temporary / "checkpoint.json", {"logical_fit_id": fit_id, "files": files})
        for path in temporary.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        os.replace(temporary, destination)
        atomic_json(directory / "latest.json", {"checkpoint": name, "logical_fit_id": fit_id})
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    retained = {name, state["best_checkpoint"]}
    for path in checkpoints.iterdir():
        if path.is_dir() and not path.is_symlink() and path.name not in retained:
            shutil.rmtree(path)
    return destination


def _checkpoint(directory: Path, name: str, fit_id: str) -> Path:
    path = contained_path(directory / "checkpoints", name)
    record = read_json(path / "checkpoint.json")
    if record["logical_fit_id"] != fit_id:
        raise ValueError("Checkpoint belongs to a different logical fit")
    verify_files(path, record["files"])
    return path


def _verify_receipt(output_dir: Path, receipt: dict) -> None:
    from .config import TrainConfig

    if receipt.get("status") != "complete" or receipt.get("receipt_schema_version") != 1:
        raise ValueError("Fit has no valid completion receipt")
    config = TrainConfig.from_dict(receipt["config"])
    expected_id = identity(
        {
            "config": scientific_config(config),
            "manifest_hash": receipt["manifest_hash"],
            "roles": [receipt[f"{role}_ids"] for role in ("train", "validation", "assessment")],
        }
    )
    if receipt["logical_fit_id"] != expected_id or receipt["fit_id"] != expected_id:
        raise ValueError("Receipt scientific identity changed")
    if receipt["config_hash"] != identity(scientific_config(config)):
        raise ValueError("Receipt configuration hash changed")
    if receipt["head"] != config.head or receipt["seed"] != config.seed:
        raise ValueError("Receipt head or seed differs from the configuration")
    if not 1 <= receipt["best_epoch"] <= config.training_epochs:
        raise ValueError("Selected checkpoint is outside the declared horizon")
    verify_files(output_dir, receipt["files"])
    if "curves.json" not in receipt["files"]:
        raise ValueError("Completion requires the complete learning curve")
    curves = json.loads((output_dir / "curves.json").read_text())
    if [row["epoch"] for row in curves] != list(range(1, config.training_epochs + 1)):
        raise ValueError("Learning curve does not cover the complete training horizon")
    if config.stage == "final":
        if receipt["best_epoch"] != config.training_epochs:
            raise ValueError("Final fitting must export end weights")
    else:
        scores = [row["validation/mean_log_loss"] for row in curves]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("Learning curve has non-finite selection scores")
        selected = min(range(len(scores)), key=lambda index: (scores[index], index))
        if receipt["best_epoch"] != selected + 1 or not math.isclose(
            receipt["validation_metrics"]["mean_log_loss"],
            scores[selected],
            abs_tol=1e-8,
            rel_tol=1e-8,
        ):
            raise ValueError("Selected checkpoint does not follow the shared selection rule")
    for role in ("validation", "assessment"):
        relative = receipt[f"{role}_prediction_path"]
        if relative is None:
            if receipt[f"{role}_ids"]:
                raise ValueError(f"Missing {role} predictions")
            continue
        if relative not in receipt["files"]:
            raise ValueError("Prediction evidence is not covered by the receipt inventory")
        predictions = read_jsonl(contained_path(output_dir, relative))
        if [row["question_id"] for row in predictions] != receipt[f"{role}_ids"]:
            raise ValueError(f"{role} prediction membership/order differs from the fit")
        computed = prediction_metrics(predictions)
        if not _metrics_reproduce(computed, receipt[f"{role}_metrics"]):
            raise ValueError(f"{role} metrics do not reproduce from predictions")
    if receipt["metrics"] != receipt["validation_metrics"]:
        raise ValueError("Reported metrics do not match the declared evaluation role")


def verify_receipt(output_dir: Path) -> dict:
    receipt = read_json(output_dir / "receipt.json")
    _verify_receipt(output_dir, receipt)
    return {**receipt, "_receipt_path": str((output_dir / "receipt.json").resolve())}


def train(
    config: TrainConfig,
    manifest_path: Path,
    output_dir: Path,
    *,
    train_ids: list[int] | None = None,
    validation_ids: list[int] | None = None,
    assessment_ids: list[int] | None = None,
    resume: bool = False,
) -> dict:
    """Train one declared fit; commit a receipt only after its outputs reproduce."""
    from .data import load_development

    manifest, rows = load_development(manifest_path)
    roles, role_rows = _resolve_roles(rows, config, train_ids, validation_ids, assessment_ids)
    manifest_hash = sha256_file(manifest_path)
    fit_id = identity(
        {"config": scientific_config(config), "manifest_hash": manifest_hash, "roles": roles}
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".fit.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process owns this fit") from error
        if (output_dir / "receipt.json").exists():
            receipt = verify_receipt(output_dir)
            if receipt["logical_fit_id"] != fit_id:
                raise ValueError("Output directory contains a different completed fit")
            return receipt
        return _train_locked(
            config, manifest, output_dir, roles, role_rows, fit_id, manifest_hash, resume
        )


def _train_locked(
    config: TrainConfig,
    manifest: dict,
    directory: Path,
    roles: list[list[int]],
    role_rows: list[list[dict]],
    fit_id: str,
    manifest_hash: str,
    resume: bool,
) -> dict:
    import numpy as np
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from .heads import loss
    existing = directory / "fit.json"
    if existing.exists():
        if read_json(existing)["logical_fit_id"] != fit_id:
            raise ValueError("Resume would change the data, roles or scientific recipe")
        if not resume:
            raise ValueError("An incomplete fit exists; continuation requires --resume")
    elif resume:
        raise ValueError("There is no fit to resume")
    fit = {
        "fit_id": fit_id,
        "logical_fit_id": fit_id,
        "config": config.to_dict(),
        "config_hash": identity(scientific_config(config)),
        "manifest_hash": manifest_hash,
        "train_ids": roles[0],
        "validation_ids": roles[1],
        "assessment_ids": roles[2],
    }
    atomic_json(existing, fit)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.bf16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError(
            "BF16 requires a supported CUDA device; CPU training requires bf16=false"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.tokenizer_revision,
        trust_remote_code=False,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("The pinned tokenizer has no padding token")
    tokens = _tokenize(tokenizer, [row for group in role_rows for row in group], config, directory)
    state = {
        "epoch": 0,
        "batch_offset": 0,
        "global_step": 0,
        "epoch_loss_sum": 0.0,
        "epoch_weight": 0,
        "curves": [],
        "best_score": None,
        "best_epoch": None,
        "best_checkpoint": None,
    }
    loaded = None
    resume_rng = None
    adapter_path = None
    if resume and (directory / "latest.json").exists():
        latest = read_json(directory / "latest.json")
        checkpoint = _checkpoint(directory, latest["checkpoint"], fit_id)
        # Only locally produced, checksum-verified recovery files may use pickle loading.
        loaded = torch.load(
            checkpoint / "training_state.pt", map_location="cpu", weights_only=False
        )
        if loaded["logical_fit_id"] != fit_id:
            raise ValueError("Recovery state has a different fit identity")
        state = {key: loaded[key] for key in state}
        adapter_path = checkpoint / "adapter"
    model = build_model(config, adapter_path).to(device)
    validate_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, config.weight_decay), lr=config.learning_rate
    )
    steps_per_epoch = math.ceil(len(role_rows[0]) / config.batch_size)
    schedule_steps = steps_per_epoch * config.schedule_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(schedule_steps * config.warmup_ratio),
        num_training_steps=schedule_steps,
    )
    if loaded is not None:
        optimizer.load_state_dict(loaded["optimizer"])
        scheduler.load_state_dict(loaded["scheduler"])
        resume_rng = loaded["rng"]
        del loaded
    environment = {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft")
        },
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "cuda": torch.version.cuda,
        "trainable_parameters": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        "total_parameters": sum(value.numel() for value in model.parameters()),
        "schedule_steps": schedule_steps,
    }
    atomic_json(directory / "environment.json", environment)
    atomic_json(
        directory / "input_manifest.json",
        {
            "manifest_hash": manifest_hash,
            "logical_fit_id": fit_id,
            "roles": fit,
            "source_hashes": manifest.get("source_hashes", manifest.get("sources", {})),
        },
    )
    stop_requested = []
    stop_file = os.environ.get("LOCALGATE_STOP_FILE")

    def stopping():
        return bool(stop_requested) or bool(stop_file and Path(stop_file).exists())

    def request_stop(signum, frame):
        stop_requested.append(signum)

    original_handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        if resume_rng is not None:
            _restore_rng(resume_rng)
        while state["epoch"] < config.training_epochs:
            model.train()
            generator = torch.Generator().manual_seed(config.seed + state["epoch"])
            order = torch.randperm(len(role_rows[0]), generator=generator).tolist()
            for offset in range(state["batch_offset"], len(order), config.batch_size):
                if stopping():
                    _save_checkpoint(
                        directory, model, tokenizer, optimizer, scheduler, state, fit_id, best=False
                    )
                    raise InterruptedError("Checkpoint saved after a shutdown request")
                batch_rows = [
                    role_rows[0][index] for index in order[offset : offset + config.batch_size]
                ]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.bf16):
                    logits = model(**_batch(tokenizer, batch_rows, tokens, device)).logits
                counts = torch.tensor([row["correct_count"] for row in batch_rows], device=device)
                objective = loss(logits.float(), counts, config.head)
                if not torch.isfinite(objective):
                    raise FloatingPointError("Non-finite training objective")
                objective.backward()
                torch.nn.utils.clip_grad_norm_(
                    [value for value in model.parameters() if value.requires_grad],
                    config.max_grad_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                scheduler.step()
                state["global_step"] += 1
                state["batch_offset"] = offset + len(batch_rows)
                weight = (
                    sum(min(row["correct_count"] + 1, 5) for row in batch_rows)
                    if config.head == "corn"
                    else len(batch_rows)
                )
                state["epoch_loss_sum"] += float(objective.detach()) * weight
                state["epoch_weight"] += weight
                if stopping():
                    _save_checkpoint(
                        directory, model, tokenizer, optimizer, scheduler, state, fit_id, best=False
                    )
                    raise InterruptedError("Checkpoint saved after a shutdown request")
            completed_epoch = state["epoch"] + 1
            values = {
                "epoch": completed_epoch,
                "training/objective": state["epoch_loss_sum"] / state["epoch_weight"],
                "training/learning_rate": scheduler.get_last_lr()[0],
                "training/step": state["global_step"],
            }
            improved = config.stage == "final"
            if role_rows[1]:
                evaluated = prediction_metrics(
                    predict(model, tokenizer, role_rows[1], tokens, config, device)
                )
                score = evaluated["mean_log_loss"]
                if not math.isfinite(score):
                    raise FloatingPointError("Non-finite checkpoint-selection metric")
                values.update(
                    {
                        f"validation/{key}": value
                        for key, value in evaluated.items()
                        if isinstance(value, (float, int))
                    }
                )
                improved = state["best_score"] is None or score < state["best_score"]
                if improved:
                    state["best_score"] = score
            if improved:
                state["best_epoch"] = completed_epoch
            state["curves"].append(values)
            state.update(epoch=completed_epoch, batch_offset=0, epoch_loss_sum=0.0, epoch_weight=0)
            _save_checkpoint(
                directory, model, tokenizer, optimizer, scheduler, state, fit_id, best=improved
            )
            atomic_json(directory / "curves.json", state["curves"])
            if stopping():
                raise InterruptedError("Checkpoint saved after a shutdown request")
        selected = _checkpoint(directory, state["best_checkpoint"], fit_id)
        del model, optimizer, scheduler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model = build_model(config, selected / "adapter").to(device)
        role_metrics = {"validation": {}, "assessment": {}}
        prediction_paths = {"validation": None, "assessment": None}
        for role, selected_rows in zip(("validation", "assessment"), role_rows[1:], strict=True):
            if selected_rows:
                predictions = predict(model, tokenizer, selected_rows, tokens, config, device)
                role_metrics[role] = prediction_metrics(predictions)
                prediction_paths[role] = f"{role}_predictions.jsonl"
                write_jsonl(directory / prediction_paths[role], predictions)
        if role_rows[1] and not math.isclose(
            role_metrics["validation"]["mean_log_loss"],
            state["best_score"],
            abs_tol=1e-8,
            rel_tol=1e-8,
        ):
            raise ValueError("Reloaded selected model does not reproduce its selection score")
        atomic_json(directory / "curves.json", state["curves"])
        evidence = {
            name: directory / name
            for name in (
                "fit.json",
                "input_manifest.json",
                "environment.json",
                "token_lengths.json",
                "curves.json",
            )
        }
        evidence.update({value: directory / value for value in prediction_paths.values() if value})
        selected_role = "assessment" if roles[2] else "validation"
        files = {name: sha256_file(path) for name, path in evidence.items()}
        for path in selected.rglob("*"):
            if path.is_file():
                files[str(path.relative_to(directory))] = sha256_file(path)
        receipt = {
            **fit,
            "receipt_schema_version": 1,
            "status": "complete",
            "head": config.head,
            "seed": config.seed,
            "best_epoch": state["best_epoch"],
            "selected_checkpoint": str(selected.relative_to(directory)),
            "metrics": role_metrics["validation"],
            "validation_metrics": role_metrics["validation"],
            "assessment_metrics": role_metrics["assessment"],
            "prediction_path": prediction_paths[selected_role],
            "validation_prediction_path": prediction_paths["validation"],
            "assessment_prediction_path": prediction_paths["assessment"],
            "files": files,
        }
        _verify_receipt(directory, receipt)
        atomic_json(directory / "receipt.json", receipt)
        return verify_receipt(directory)
    finally:
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)
