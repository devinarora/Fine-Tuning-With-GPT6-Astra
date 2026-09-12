"""Validated experiment settings. Revisions are immutable Hugging Face commits."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import hashlib
import json
import re

import yaml


@dataclass
class Assets:
    model: str = "Helsinki-NLP/opus-mt-fr-en"
    model_revision: str = "c4aed37b318c763fd177aa449b44e3b783cc6c02"
    books: str = "Helsinki-NLP/opus_books"
    books_revision: str = "1f9f6191d0e91a3c539c2595e2fe48fc1420de9b"
    opus: str = "Helsinki-NLP/opus-100"
    opus_revision: str = "805090dc28bf78897da9641cdf08b61287580df9"
    scorer: str = "FacebookAI/roberta-base"
    scorer_revision: str = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"


@dataclass
class Runtime:
    run_root: str = "~/fr-en-ft/runs"
    cache_root: str = "~/fr-en-ft/cache"
    budget_seconds: int = 7200
    checkpoint_seconds: int = 240
    evaluation_reserve_seconds: int = 1800
    gpu_memory_gib: float = 4.8
    free_gpu_reserve_gib: float = 1.5
    cpu_threads: int = 4
    require_cuda: bool = True


@dataclass
class Data:
    train_per_source: int = 10000
    validation_per_source: int = 250
    test_per_source: int = 500
    max_source_tokens: int = 256
    max_target_tokens: int = 256
    max_character_ratio: float = 8.0


@dataclass
class Training:
    epochs: int = 2
    micro_batch_size: int = 2
    effective_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    validation_seconds: int = 600
    early_stopping_patience: int = 3
    calibration_warmup_updates: int = 2
    calibration_updates: int = 8
    max_steps: int | None = None


@dataclass
class Evaluation:
    beams: int = 4
    batch_size: int = 4
    bertscore: bool = True
    bertscore_layers: int = 10
    bootstrap_samples: int = 1000


@dataclass
class Config:
    seed: int = 42
    assets: Assets = field(default_factory=Assets)
    runtime: Runtime = field(default_factory=Runtime)
    data: Data = field(default_factory=Data)
    training: Training = field(default_factory=Training)
    evaluation: Evaluation = field(default_factory=Evaluation)

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return digest(self.to_dict())

    def validate(self):
        for name, value in asdict(self.assets).items():
            if name.endswith("_revision") and not re.fullmatch(r"[a-f0-9]{40}", value):
                raise ValueError(f"assets.{name} must be a full commit SHA")
        positive = {
            **{f"data.{k}": v for k, v in asdict(self.data).items()},
            **{f"runtime.{k}": getattr(self.runtime, k) for k in (
                "budget_seconds", "checkpoint_seconds", "evaluation_reserve_seconds",
                "gpu_memory_gib", "free_gpu_reserve_gib", "cpu_threads")},
            **{f"training.{k}": getattr(self.training, k) for k in (
                "epochs", "micro_batch_size", "effective_batch_size", "learning_rate",
                "validation_seconds", "early_stopping_patience", "calibration_updates")},
            "evaluation.beams": self.evaluation.beams,
            "evaluation.batch_size": self.evaluation.batch_size,
            "evaluation.bootstrap_samples": self.evaluation.bootstrap_samples,
        }
        for key, value in positive.items():
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{key} must be positive")
        if self.training.effective_batch_size % self.training.micro_batch_size:
            raise ValueError("effective_batch_size must be divisible by micro_batch_size")
        if not 0 <= self.training.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.training.max_steps is not None and self.training.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if max(self.data.max_source_tokens, self.data.max_target_tokens) > 512:
            raise ValueError("This Marian experiment supports at most 512 tokens")
        if self.runtime.evaluation_reserve_seconds >= self.runtime.budget_seconds:
            raise ValueError("Evaluation reserve must be smaller than the total budget")
        return self


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def from_dict(raw):
    raw = dict(raw)
    valid = {f.name for f in fields(Config)}
    if unknown := raw.keys() - valid:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    for key, cls in {"assets": Assets, "runtime": Runtime, "data": Data,
                     "training": Training, "evaluation": Evaluation}.items():
        values = raw.get(key, {})
        if not isinstance(values, dict):
            raise ValueError(f"{key} must be a mapping")
        unknown = values.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {key} keys: {sorted(unknown)}")
        raw[key] = cls(**values)
    return Config(**raw).validate()


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        return from_dict(yaml.safe_load(handle) or {})
