"""Load the selected adapter and emit validated classifier predictions."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

from .artifacts import read_json, sha256_file, write_jsonl_exclusive
from .config import TrainConfig
from .heads import decode
from .training import build_model

PREDICTION_SOURCE_HASHES = {
    "converted.jsonl": "be78cccedb7778e479f5daec83c75f56b8aeac42b4c4225c5b78a8827586d4fd",
    "labels_open.jsonl": "7a8f03c18c1d657fdff12fc2480410122253c0726b2df627e5a7fc81600b58ec",
    "labels_study_k30.jsonl": "68a067682507c7153e175a474191fccd1a40a2f5f59fa4394b0f94c520424cb3",
    "split.json": "a0f57e29add58d82527713f66c139e35af319bafe795038a05925808e2434e38",
    "study_prompts_corpus.jsonl": "ab3eb42bbf8d9dc81bf688a0a6f14cfeebf742551e1b1750e2bb414dadc5f0b2",
}
EXPECTED_GROUP_COUNTS = {"study": 280, "heldout": 1375, "combined": 1655}
BASE_MODEL_HASHES = {
    "config.json": "1609d59e627c33eaed524b4f01e546d42e84190a079a5a5ded84b212c41c324f",
    "model.safetensors": "340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9",
    "special_tokens_map.json": "ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049",
    "tokenizer.json": "9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1",
    "tokenizer_config.json": "3cd2017ff46d0a527e5d39cae39272eccfa1f19bb9f89b05d166aab2e38354e2",
}
PUBLISHED_FIT_ID = "389943547ae17873fe7255f49a382fa71b6f3f7a6181f21c7023ff6dd4091ce8"
PUBLISHED_ADAPTER_HASHES = {
    "adapter_model.safetensors": "a255fbb2c427a3252e028db3e9ae89b5d6e4f2fadf3b2ee6b6989c954adf2a83",
    "adapter_config.json": "331136e88b415ebb9ec0bdc816b58ba4fcd71b5e3c5dc0f0cd5bc921d60f2f68",
}


def verify_base_model(path: Path) -> None:
    for name, expected in BASE_MODEL_HASHES.items():
        candidate = path / name
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ValueError(f"base model file differs from the pinned snapshot: {name}")


def verify_adapter(config_path: Path, adapter: Path) -> None:
    document = read_json(config_path)
    if document.get("fit_id") == PUBLISHED_FIT_ID and document.get("status") is None:
        expected = PUBLISHED_ADAPTER_HASHES
    else:
        if document.get("status") != "complete" or document.get("receipt_schema_version") != 1:
            raise ValueError("adapter config must be the published fit record or a completion receipt")
        selected = document.get("selected_checkpoint")
        files = document.get("files")
        if not isinstance(selected, str) or not isinstance(files, dict):
            raise ValueError("completion receipt lacks selected adapter hashes")
        expected_path = (config_path.parent / selected / "adapter").resolve()
        if adapter.resolve() != expected_path:
            raise ValueError("adapter path does not match the completion receipt")
        prefix = f"{selected}/adapter/"
        expected = {
            name.removeprefix(prefix): digest
            for name, digest in files.items()
            if name.startswith(prefix) and "/" not in name.removeprefix(prefix)
        }
        required = {"adapter_model.safetensors", "adapter_config.json"}
        if not required <= set(expected):
            missing = sorted(required - set(expected))
            raise ValueError(
                f"completion receipt lacks required selected adapter files: {missing}"
            )
    for name, digest in expected.items():
        path = adapter / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"adapter file differs from its recorded identity: {name}")


def load_recipe(path: Path, base_model: Path) -> TrainConfig:
    document = read_json(path)
    values = document.get("config", document)
    if not isinstance(values, dict):
        raise TypeError("config must be a recipe object or a fit/receipt containing config")
    config = TrainConfig.from_dict(values)
    if config.head != "beta_binomial":
        raise ValueError("the selected classifier requires the beta_binomial head")
    if not base_model.is_dir():
        raise ValueError(f"base model directory does not exist: {base_model}")
    return replace(config, model_name=str(base_model))


def _jsonl(path: Path) -> list[dict]:
    rows = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        question_id = row.get("question_id") if isinstance(row, dict) else None
        if type(question_id) is not int or question_id in seen:
            raise ValueError(f"invalid or duplicate question ID in {path}:{line_number}")
        rows.append(row)
        seen.add(question_id)
    return rows


def load_prediction_rows(data_dir: Path, group: str) -> list[dict]:
    if group not in EXPECTED_GROUP_COUNTS:
        raise ValueError(f"unknown prediction group: {group}")
    for name, expected in PREDICTION_SOURCE_HASHES.items():
        path = data_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"prediction source changed or is missing: {name}")

    converted = {
        row["question_id"]: row
        for row in _jsonl(data_dir / "converted.jsonl")
        if row.get("convertible") is True
    }
    open_labels = {row["question_id"]: row for row in _jsonl(data_dir / "labels_open.jsonl")}
    study_corpus = {
        row["question_id"]: row for row in _jsonl(data_dir / "study_prompts_corpus.jsonl")
    }
    study_labels = {
        row["question_id"]: row for row in _jsonl(data_dir / "labels_study_k30.jsonl")
    }
    split = read_json(data_dir / "split.json")
    assignment = {int(key): value for key, value in split["assignment"].items()}
    combined_ids = {
        question_id
        for question_id in converted
        if assignment.get(question_id) == "test"
    }
    study_ids = set(study_corpus)
    if set(study_labels) != study_ids or not study_ids <= combined_ids:
        raise ValueError("study corpus, labels, and test split do not align")
    ids = {
        "study": study_ids,
        "heldout": combined_ids - study_ids,
        "combined": combined_ids,
    }[group]
    if len(ids) != EXPECTED_GROUP_COUNTS[group]:
        raise ValueError(f"expected {EXPECTED_GROUP_COUNTS[group]} {group} rows, found {len(ids)}")

    rows = []
    for question_id in sorted(ids):
        source = study_corpus[question_id] if group == "study" else converted[question_id]
        label = study_labels[question_id] if group == "study" else open_labels[question_id]
        text = source.get("open_question")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"missing open question for {question_id}")
        if any(source.get(key) != label.get(key) for key in ("category", "list_style")):
            raise ValueError(f"metadata mismatch for {question_id}")
        k = label.get("k")
        correct = label.get("correct_count")
        if type(k) is not int or type(correct) is not int or not 0 <= correct <= k:
            raise ValueError(f"invalid label for {question_id}")
        rows.append(
            {
                "question_id": question_id,
                "text": text.strip(),
                "correct_count": correct,
                "k": k,
                "category": label["category"],
                "list_style": label["list_style"],
            }
        )
    return rows


def run_prediction(
    config_path: Path,
    base_model: Path,
    adapter: Path,
    data_dir: Path,
    group: str,
    output: Path,
    *,
    device_name: str = "cpu",
) -> list[dict]:
    import torch
    from transformers import AutoTokenizer

    if output.exists():
        raise ValueError(f"refusing to overwrite prediction output: {output}")
    if not adapter.is_dir():
        raise ValueError(f"adapter directory does not exist: {adapter}")
    verify_base_model(base_model)
    verify_adapter(config_path, adapter)
    config = load_recipe(config_path, base_model)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    use_bf16 = False
    config = replace(config, bf16=False, attention_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model),
        revision=config.tokenizer_revision,
        trust_remote_code=False,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("the pinned tokenizer has no padding token")
    model = build_model(config, adapter, trainable=False).to(device).eval()
    rows = load_prediction_rows(data_dir, group)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), config.eval_batch_size):
            batch_rows = rows[start : start + config.eval_batch_size]
            encoded = tokenizer(
                [row["text"] for row in batch_rows],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            if encoded["input_ids"].shape[1] > config.max_length:
                raise ValueError("prediction input exceeds the frozen token cap")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**encoded).logits.float()
            probabilities, pmf = decode(logits, config.head)
            if pmf is None:
                raise ValueError("the selected beta-binomial head must emit a count PMF")
            for index, row in enumerate(batch_rows):
                probability = float(probabilities[index])
                native_pmf = pmf[index].double().cpu().tolist()
                if not math.isclose(sum(native_pmf), 1.0, abs_tol=1e-8, rel_tol=1e-8):
                    raise ValueError("decoded count PMF does not sum to one")
                predictions.append(
                    {
                        **row,
                        "logits": logits[index].double().cpu().tolist(),
                        "probability": probability,
                        "predicted_level": min(int(6 * probability), 5),
                        "native_k5_pmf": native_pmf,
                    }
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_exclusive(output, predictions)
    return predictions
