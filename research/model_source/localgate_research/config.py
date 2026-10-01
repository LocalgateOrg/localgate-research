"""Validated recipe for the selected LocalGate classifier."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

MODEL_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
HEAD_NAMES = (
    "softmax",
    "corn",
    "regression",
    "binomial",
    "beta_binomial",
    "twin",
    "softmax_rps",
)
@dataclass(frozen=True)
class TrainConfig:
    head: str = "beta_binomial"
    seed: int = 17
    stage: str = "final"
    fold: int | None = None
    model_name: str = "answerdotai/ModernBERT-base"
    model_revision: str = MODEL_REVISION
    tokenizer_revision: str = MODEL_REVISION
    learning_rate: float = 9.468154783470064e-05
    weight_decay: float = 0.1
    lora_dropout: float = 0.1
    lora_rank: int = 8
    lora_alpha: int = 16
    readout_dropout: float = 0.1
    batch_size: int = 32
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_length: int = 1024
    training_epochs: int = 8
    schedule_epochs: int = 20
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    bf16: bool = True
    attention_implementation: str = "sdpa"
    def __post_init__(self) -> None:
        if self.head not in HEAD_NAMES:
            raise ValueError(f"Unknown prediction head: {self.head}")
        if self.stage not in {"pilot", "hpo", "replication", "cv", "final", "smoke"}:
            raise ValueError(f"Unknown experiment stage: {self.stage}")
        integers = (
            "seed",
            "lora_rank",
            "lora_alpha",
            "batch_size",
            "eval_batch_size",
            "gradient_accumulation_steps",
            "max_length",
            "training_epochs",
            "schedule_epochs",
        )
        for name in integers:
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be a valid positive integer (seed may be zero)")
        if self.fold is not None and (type(self.fold) is not int or not 0 <= self.fold < 5):
            raise ValueError("fold must be an integer in 0..4 or null")
        for name in ("learning_rate", "weight_decay", "max_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
                raise ValueError(f"Invalid {name}")
        for name in ("lora_dropout", "readout_dropout", "warmup_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ValueError(f"{name} must lie in [0, 1)")
        if self.gradient_accumulation_steps != 1:
            raise ValueError("The training recipe requires gradient_accumulation_steps=1")
        if self.training_epochs > self.schedule_epochs:
            raise ValueError("Training cannot exceed the frozen schedule horizon")
        if self.max_length > 8192:
            raise ValueError("The ModernBERT token cap cannot exceed 8192")
        if type(self.bf16) is not bool:
            raise ValueError("bf16 must be a boolean")
        if self.attention_implementation not in {"sdpa", "eager"}:
            raise ValueError("Only sdpa and eager attention are supported")
        for name in ("model_revision", "tokenizer_revision"):
            if not re.fullmatch(r"[0-9a-f]{40}", getattr(self, name)):
                raise ValueError(f"{name} must pin a full Hub commit")
        for name in ("model_name",):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, document: dict) -> TrainConfig:
        if not isinstance(document, dict):
            raise TypeError("A recipe must be a JSON object")
        document = dict(document)
        for legacy in ("source_revision", "wandb_mode", "wandb_entity", "wandb_project"):
            document.pop(legacy, None)
        unknown = set(document) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown recipe fields: {sorted(unknown)}")
        return cls(**document)


def load_config(path: Path) -> TrainConfig:
    return TrainConfig.from_dict(json.loads(path.read_text()))
