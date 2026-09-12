"""Complete single-process Trainer checkpoints, published by atomic directory rename."""

from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import time
import uuid

from transformers import Seq2SeqTrainer

from .state import atomic_json, file_hash, read_json


def inspect_checkpoint(path, contract=None, verify_hashes=True):
    path = Path(path)
    manifest = read_json(path / "complete.json")
    if contract is not None and manifest["contract"] != contract:
        raise ValueError(f"Checkpoint configuration/data/software mismatch: {path}")
    required = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "pipeline_state.json"}
    if not required.issubset(manifest["files"]):
        raise ValueError(f"Checkpoint lacks resume state: {path}")
    if not any(name.endswith(".safetensors") for name in manifest["files"]):
        raise ValueError(f"Checkpoint lacks safetensors model weights: {path}")
    for name, expected in manifest["files"].items():
        file = path / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not file.is_file():
            raise ValueError(f"Invalid checkpoint file: {name}")
        if file.stat().st_size != expected["bytes"]:
            raise ValueError(f"Checkpoint size mismatch: {file}")
        if verify_hashes and file_hash(file) != expected["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {file}")
    return manifest


def complete_checkpoints(root, contract=None, verify_hashes=False):
    found = []
    for path in Path(root).glob("checkpoint-*"):
        try:
            manifest = inspect_checkpoint(path, contract, verify_hashes)
        except (OSError, ValueError, KeyError):
            continue
        found.append((manifest["global_step"], manifest["created_utc"], path))
    return [entry[2] for entry in sorted(found)]


def latest_checkpoint(root, contract):
    # Try newest first, falling back after a partial or corrupted write.
    for path in reversed(complete_checkpoints(root, contract)):
        try:
            inspect_checkpoint(path, contract, verify_hashes=True)
            return path
        except (OSError, ValueError):
            continue
    raise FileNotFoundError(f"No complete, compatible checkpoint in {root}")


class AtomicTrainer(Seq2SeqTrainer):
    """The checkpoint override is intentionally limited to one local process/GPU."""

    def __init__(self, *args, checkpoint_contract, progress=None, run_context=None, **kwargs):
        self.checkpoint_contract = checkpoint_contract
        self.progress = progress
        self.run_context = run_context
        super().__init__(*args, **kwargs)
        if self.args.world_size != 1:
            raise ValueError("AtomicTrainer supports one process only")
        # Marian's **kwargs signature makes Trainer assume it consumes num_items_in_batch,
        # but Marian computes a microbatch mean. Consume that denominator explicitly here.
        self.model_accepts_loss_kwargs = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        count = inputs["labels"].ne(-100).sum()
        denominator = count if num_items_in_batch is None else num_items_in_batch
        loss = outputs.loss * count / denominator
        return (loss, outputs) if return_outputs else loss

    def _save_checkpoint(self, model, trial):
        if trial is not None:
            raise ValueError("Hyperparameter search is outside this checkpoint implementation")
        start = time.monotonic()
        root = Path(self.args.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        suffix = uuid.uuid4().hex[:10]
        destination = root / f"checkpoint-{self.state.global_step:08d}-{suffix}"
        temporary = root / f".incomplete-{suffix}"
        temporary.mkdir()
        old_best = self.state.best_model_checkpoint
        try:
            if self.state.best_global_step == self.state.global_step:
                self.state.best_model_checkpoint = str(destination)
            self.store_flos()
            self.save_model(str(temporary), _internal_call=True)
            self._save_optimizer_and_scheduler(str(temporary))
            self._save_scaler(str(temporary))
            self._save_rng_state(str(temporary))
            self.state.save_to_json(str(temporary / "trainer_state.json"))
            progress = self.progress.snapshot() if self.progress else {}
            if self.run_context:
                self.run_context.heartbeat()
                progress["spent_seconds"] = self.run_context.elapsed
            atomic_json(temporary / "pipeline_state.json", progress)
            inventory = {}
            for file in sorted(temporary.rglob("*")):
                if file.is_file():
                    # Flush model/optimizer files before declaring the checkpoint complete.
                    with file.open("rb") as handle:
                        os.fsync(handle.fileno())
                    inventory[str(file.relative_to(temporary))] = {"bytes": file.stat().st_size, "sha256": file_hash(file)}
            atomic_json(temporary / "complete.json", {"contract": self.checkpoint_contract,
                        "global_step": self.state.global_step, "created_utc": datetime.now(timezone.utc).isoformat(),
                        "best_checkpoint": self.state.best_model_checkpoint, "files": inventory})
            os.replace(temporary, destination)
            if os.name == "posix":
                directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            # Keep two latest complete bundles and the best validation bundle.
            checkpoints = complete_checkpoints(root, self.checkpoint_contract)
            keep = set(checkpoints[-2:])
            if self.state.best_model_checkpoint:
                keep.add(Path(self.state.best_model_checkpoint))
            for path in checkpoints:
                if path not in keep:
                    shutil.rmtree(path)
            if self.run_context:
                self.run_context.log("checkpoint_saved", path=str(destination), step=self.state.global_step,
                                     seconds=time.monotonic() - start)
                self.run_context.heartbeat()
        except BaseException:
            self.state.best_model_checkpoint = old_best
            raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
