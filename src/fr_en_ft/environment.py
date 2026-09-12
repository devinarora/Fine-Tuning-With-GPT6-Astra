"""Runtime inspection without downloading assets or changing dependencies."""

import importlib.metadata
import os
import platform
import shutil
import sys

import psutil


PACKAGES = ["torch", "torchvision", "transformers", "datasets", "accelerate",
            "huggingface-hub", "safetensors", "sentencepiece", "sacremoses", "sacrebleu",
            "bert-score", "numpy", "PyYAML", "pandas", "matplotlib", "psutil", "ipykernel", "jupyterlab", "pytest", "packaging"]


def dependency_lock():
    """Installed transitive dependency closure, evaluated for this OS and Python."""
    from packaging.requirements import Requirement
    pending, seen, lines = [(name, frozenset()) for name in PACKAGES], set(), {}
    while pending:
        name, extras = pending.pop()
        normalized = name.lower().replace("_", "-")
        if (normalized, extras) in seen:
            continue
        seen.add((normalized, extras))
        distribution = importlib.metadata.distribution(name)
        lines[normalized] = f"{distribution.metadata['Name']}=={distribution.version}"
        for value in distribution.requires or []:
            requirement = Requirement(value)
            if requirement.marker is None or any(requirement.marker.evaluate({"extra": extra}) for extra in {"", *extras}):
                pending.append((requirement.name, frozenset(requirement.extras)))
    return "\n".join(sorted(lines.values(), key=str.lower)) + "\n"


def inspect_environment():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    import torch
    cuda = torch.cuda.is_available()
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "torch_disable_native_jit": os.environ.get("TORCH_DISABLE_NATIVE_JIT"), "attention": "sdpa",
            "packages": packages, "cuda": cuda, "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "bf16_supported": torch.cuda.is_bf16_supported() if cuda else False,
            "available_ram_gib": psutil.virtual_memory().available / 1024**3,
            "total_ram_gib": psutil.virtual_memory().total / 1024**3,
            "free_gpu_gib": torch.cuda.mem_get_info()[0] / 1024**3 if cuda else None}


def setup(config):
    import torch
    from transformers import set_seed
    from transformers.utils.import_utils import check_torch_load_is_safe
    torch.set_num_threads(config.runtime.cpu_threads)
    set_seed(config.seed)
    if config.runtime.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Use WSL's Python interpreter.")
    check_torch_load_is_safe()
    return "cuda" if torch.cuda.is_available() and config.runtime.require_cuda else "cpu"


def check_disk(path, required_gib=5):
    if shutil.disk_usage(path).free < required_gib * 1024**3:
        raise RuntimeError(f"Need at least {required_gib} GiB free for checkpoint rotation at {path}")
