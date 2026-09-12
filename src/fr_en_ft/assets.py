"""Explicit, revision-pinned downloads. Network time is outside the run budget."""

from pathlib import Path
import time

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

from .state import atomic_json, read_json


def retry_download(operation):
    for attempt in range(4):
        try:
            return operation()
        except HfHubHTTPError as exc:
            if exc.response.status_code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            retry_after = exc.response.headers.get("Retry-After", "")
            delay = max(15 * 2**attempt, int(retry_after) if retry_after.isdigit() else 0)
            print(f"Hugging Face returned {exc.response.status_code}; retrying in {delay} seconds", flush=True)
            time.sleep(delay)


def download(run):
    config = run.config
    start = time.monotonic()
    cache = Path(config.runtime.cache_root).expanduser() / "hub"
    model_patterns = ["*.json", "*.safetensors", "*.spm", "*.model", "merges.txt"]
    paths = {}
    for name in ("model", "books", "opus", "scorer"):
        if name == "scorer" and not config.evaluation.bertscore:
            continue
        repo = getattr(config.assets, name)
        revision = getattr(config.assets, f"{name}_revision")
        is_data = name in ("books", "opus")
        if is_data:
            # These exact files are part of the pinned revisions; no multilingual tree listing is needed.
            splits = ("train",) if name == "books" else ("train", "validation", "test")
            for split in splits:
                file = retry_download(lambda: hf_hub_download(repo, f"en-fr/{split}-00000-of-00001.parquet",
                                      repo_type="dataset", revision=revision, cache_dir=cache))
                paths[name] = str(Path(file).parent.parent)
        else:
            paths[name] = retry_download(lambda: snapshot_download(repo, revision=revision, cache_dir=cache,
                                        allow_patterns=model_patterns, max_workers=2))
    atomic_json(run.path / "assets.json", {"paths": paths, "revisions": vars(config.assets),
                "download_seconds": time.monotonic() - start})
    run.complete("download")
    return paths


def asset_paths(run):
    return read_json(run.path / "assets.json")["paths"]
