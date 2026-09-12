"""Source-grouped splits, cross-corpus leakage checks, and auditable exclusions."""

from collections import Counter
from pathlib import Path
import hashlib
import os
import shutil
import unicodedata
import uuid

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from .assets import asset_paths
from .config import digest
from .model import load_tokenizer
from .state import atomic_json, read_json


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_pair(raw, source, origin, index, max_ratio=8):
    translation = raw.get("translation")
    if not isinstance(translation, dict) or not all(isinstance(translation.get(k), str) for k in ("fr", "en")):
        return None, "invalid_schema"
    french, english = normalize(translation["fr"]), normalize(translation["en"])
    if not french or not english:
        return None, "empty"
    if max(len(french), len(english)) / min(len(french), len(english)) > max_ratio:
        return None, "character_ratio"
    pair_hash = digest([french, english])
    return {"id": f"{source}:{origin}:{index}:{pair_hash[:16]}", "source": french, "target": english,
            "corpus": source, "origin_split": origin, "origin_index": index,
            "source_hash": text_hash(french), "pair_hash": pair_hash}, None


def books_split(source_hash, seed):
    # Identical French stays in one split, independent of row order or translation variants.
    bucket = int(text_hash(f"{seed}:{source_hash}")[:8], 16) % 100
    return "test" if bucket < 5 else "validation" if bucket < 10 else "train"


def assert_disjoint(splits):
    seen = {}
    for name, rows in splits.items():
        for row in rows:
            key = row["source_hash"]
            if key in seen and seen[key] != name:
                raise ValueError(f"French source overlap between {seen[key]} and {name}")
            seen[key] = name


def prepare(run):
    config = run.config
    destination = run.path / "prepared"
    if destination.exists():
        manifest = read_json(destination / "manifest.json")
        if manifest["config_fingerprint"] != config.fingerprint:
            raise ValueError("Existing prepared data belongs to another configuration")
        atomic_json(run.path / "data_manifest.json", manifest)
        run.complete("prepare")
        return
    paths = asset_paths(run)
    tokenizer = load_tokenizer(paths["model"])
    stats = Counter()
    pools = {split: {corpus: [] for corpus in ("books", "opus")} for split in ("train", "validation", "test")}
    # Holdout precedence applies to all available rows, including unselected held-out rows.
    owners = {}
    for corpus in ("opus", "books"):
        files = {}
        for split in ("train", "validation", "test"):
            matches = sorted(str(p) for p in (Path(paths[corpus]) / "en-fr").glob(f"{split}-*.parquet"))
            if matches:
                files[split] = matches
        if "train" not in files:
            raise ValueError(f"No en-fr Parquet training files for {corpus}")
        raw = load_dataset("parquet", data_files=files, cache_dir=str(Path(config.runtime.cache_root).expanduser() / "arrow"))
        for origin, ds in raw.items():
            for index, example in enumerate(ds):
                if index % 100000 == 0:
                    if run.remaining <= 0:
                        raise RuntimeError("Budget exhausted during data preparation")
                    run.heartbeat()
                stats[f"{corpus}.raw.{origin}"] += 1
                row, reason = clean_pair(example, corpus, origin, index, config.data.max_character_ratio)
                if reason:
                    stats[f"{corpus}.excluded.{reason}"] += 1
                    continue
                split = books_split(row["source_hash"], config.seed) if corpus == "books" else origin
                rank = {"train": 0, "validation": 1, "test": 2}[split]
                owners[row["source_hash"]] = max(rank, owners.get(row["source_hash"], 0))
                pools[split][corpus].append(row)
    chosen = {split: [] for split in pools}
    for split in ("test", "validation", "train"):
        rank = {"train": 0, "validation": 1, "test": 2}[split]
        for corpus, rows in pools[split].items():
            # Sorting by seeded hashes samples the entire source, rather than its first N rows.
            rows.sort(key=lambda row: text_hash(f"{config.seed}:{row['id']}"))
            desired = getattr(config.data, f"{split}_per_source")
            seen = set()
            cross_corpus_pairs = {r["pair_hash"] for r in chosen[split]}
            selected = []
            eligible = []
            for row in rows:
                if owners[row["source_hash"]] != rank:
                    stats[f"{corpus}.{split}.excluded.cross_split_source"] += 1
                elif row["pair_hash"] in seen or row["pair_hash"] in cross_corpus_pairs:
                    stats[f"{corpus}.{split}.excluded.duplicate_pair"] += 1
                else:
                    seen.add(row["pair_hash"])
                    eligible.append(row)
            # Tokenize candidates in batches until the requested sample is full.
            for start in range(0, len(eligible), 256):
                batch = eligible[start:start + 256]
                encoded = tokenizer([r["source"] for r in batch], text_target=[r["target"] for r in batch], truncation=False)
                for index, row in enumerate(batch):
                    stats[f"{corpus}.{split}.tokenized_candidates"] += 1
                    if len(encoded["input_ids"][index]) > config.data.max_source_tokens:
                        stats[f"{corpus}.{split}.excluded.source_tokens"] += 1
                    elif len(encoded["labels"][index]) > config.data.max_target_tokens:
                        stats[f"{corpus}.{split}.excluded.target_tokens"] += 1
                    else:
                        selected.append({**row, **{k: encoded[k][index] for k in ("input_ids", "attention_mask", "labels")}})
                    if len(selected) == desired:
                        break
                if len(selected) == desired:
                    break
            if len(selected) != desired:
                raise ValueError(f"Only {len(selected)} eligible {corpus}/{split} pairs; requested {desired}")
            chosen[split].extend(selected)
    # Validate the quotas and deduplication before publishing any prepared data.
    for split, rows in chosen.items():
        if len({r["pair_hash"] for r in rows}) != len(rows):
            raise ValueError(f"Cross-corpus duplicate selected in {split}; change seed or data policy")
    assert_disjoint(chosen)
    for rows in chosen.values():
        rows.sort(key=lambda row: text_hash(f"order:{config.seed}:{row['id']}"))
    manifest = {"config_fingerprint": config.fingerprint, "stats": dict(stats),
                "splits": {name: [{k: r[k] for k in ("id", "source_hash", "pair_hash", "corpus")} for r in rows]
                           for name, rows in chosen.items()},
                "example_ids": [next(r["id"] for r in chosen["test"] if r["corpus"] == corpus) for corpus in ("books", "opus")],
                "scope": "Token-filtered sentence/short-paragraph holdouts; book/document independence is not established"}
    manifest["data_fingerprint"] = digest(manifest["splits"])
    temporary = run.path / f".prepared-{uuid.uuid4().hex}"
    try:
        DatasetDict({name: Dataset.from_list(rows) for name, rows in chosen.items()}).save_to_disk(str(temporary))
        atomic_json(temporary / "manifest.json", manifest)
        if destination.exists():
            raise FileExistsError(f"Prepared data already exists: {destination}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    atomic_json(run.path / "data_manifest.json", manifest)
    run.complete("prepare")
    run.log("data_prepared", counts={k: len(v) for k, v in chosen.items()})


def prepared(run):
    manifest = read_json(run.path / "data_manifest.json")
    if manifest["config_fingerprint"] != run.config.fingerprint:
        raise ValueError("Prepared data configuration does not match this run")
    return load_from_disk(str(run.path / "prepared"))
