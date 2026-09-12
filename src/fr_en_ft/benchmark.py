"""Calibrate optimizer updates, worst-length memory, generation, and checkpoint I/O."""

from pathlib import Path
import gc
import math
import tempfile
import time

import numpy as np
import torch

from .assets import asset_paths
from .data import prepared
from .environment import check_disk, setup
from .evaluate import autocast, predict_batch, synchronize
from .model import collator, load_model, load_tokenizer, token_features
from .state import atomic_json


def plan_steps(config, count, benchmark, remaining, final_reserve):
    per_epoch = math.ceil(math.ceil(count / benchmark["micro_batch_size"]) / benchmark["gradient_accumulation_steps"])
    maximum = per_epoch * config.training.epochs
    if config.training.max_steps is not None:
        maximum = min(maximum, config.training.max_steps)
    update_seconds = benchmark["update_seconds_p75"]
    eval_seconds = benchmark["validation_seconds_estimate"]
    evaluation_steps = max(1, math.floor(config.training.validation_seconds / max(update_seconds, 1e-6)))
    checkpoint_overhead = benchmark["checkpoint_seconds"] / config.runtime.checkpoint_seconds
    # Reserve an end-of-training validation and save even when no periodic evaluation fits.
    available = remaining - final_reserve - eval_seconds - benchmark["checkpoint_seconds"] - 30
    cost_per_update = update_seconds * (1 + checkpoint_overhead) + eval_seconds / evaluation_steps
    steps = min(maximum, math.floor(max(0, available) / max(cost_per_update, 1e-6)))
    if steps < 1:
        raise RuntimeError("No training time remains after reserving final evaluation. Resume with an explicit budget extension.")
    return {"max_steps": steps, "max_epoch_steps": maximum, "eval_steps": min(evaluation_steps, steps),
            "micro_batch_size": benchmark["micro_batch_size"],
            "gradient_accumulation_steps": benchmark["gradient_accumulation_steps"],
            "gradient_checkpointing": benchmark["gradient_checkpointing"],
            "precision": benchmark["precision"], "final_reserve_seconds": final_reserve,
            "predicted_training_seconds": steps * cost_per_update + eval_seconds + benchmark["checkpoint_seconds"],
            "predicted_end_to_end_remaining_seconds": steps * cost_per_update + eval_seconds + benchmark["checkpoint_seconds"] + final_reserve}


def benchmark(run):
    config = run.config
    device = setup(config)
    check_disk(run.path)
    tokenizer = load_tokenizer(asset_paths(run)["model"])
    train = prepared(run)["train"]
    # Include both independent source/target maxima and ordinary randomly ordered training examples.
    ordered = sorted(list(train), key=lambda r: len(r["input_ids"]) + len(r["labels"]), reverse=True)
    candidates = [(config.training.micro_batch_size, False), (1, False), (1, True)]
    measurements = None
    for micro, checkpointing in dict.fromkeys(candidates):
        if config.training.effective_batch_size % micro:
            continue
        accumulation = config.training.effective_batch_size // micro
        model = optimizer = None
        try:
            model = load_model(asset_paths(run)["model"], device)
            model.config.use_cache = False
            if checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
            scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and not torch.cuda.is_bf16_supported())
            pad = collator(tokenizer, model)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            times = []
            updates = config.training.calibration_warmup_updates + config.training.calibration_updates
            for update in range(updates):
                if run.remaining <= config.runtime.evaluation_reserve_seconds:
                    raise RuntimeError("Calibration exhausted the training budget")
                synchronize(device)
                start = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                for micro_index in range(accumulation):
                    if update == 0:
                        # A worst-length synthetic batch tests the configured limits, not only the sample maximum.
                        batch = {"input_ids": torch.full((micro, config.data.max_source_tokens), 10, dtype=torch.long),
                                 "attention_mask": torch.ones((micro, config.data.max_source_tokens), dtype=torch.long),
                                 "labels": torch.full((micro, config.data.max_target_tokens), 10, dtype=torch.long)}
                    else:
                        offset = ((update * accumulation + micro_index) * micro) % len(train)
                        rows = [train[(offset + i) % len(train)] for i in range(micro)]
                        batch = pad([token_features(r) for r in rows])
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with autocast(device):
                        loss = model(**batch, use_cache=False).loss / accumulation
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite calibration loss")
                    scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError("Non-finite calibration gradients")
                scaler.step(optimizer)
                scaler.update()
                synchronize(device)
                if update >= config.training.calibration_warmup_updates:
                    times.append(time.monotonic() - start)
                run.heartbeat()
            peak = torch.cuda.max_memory_reserved() / 1024**3 if device == "cuda" else 0
            free = torch.cuda.mem_get_info()[0] / 1024**3 if device == "cuda" else None
            if device == "cuda" and (peak > config.runtime.gpu_memory_gib or free < config.runtime.free_gpu_reserve_gib):
                run.log("calibration_memory_retry", micro_batch=micro, checkpointing=checkpointing, peak_gib=peak, free_gib=free)
                continue
            with tempfile.TemporaryDirectory(prefix=".checkpoint-benchmark-", dir=run.path) as directory:
                start = time.monotonic()
                model.save_pretrained(directory, safe_serialization=True)
                torch.save(optimizer.state_dict(), Path(directory) / "optimizer.pt")
                # Include integrity hashing in the measured write cost.
                from .state import file_hash
                for path in Path(directory).iterdir():
                    file_hash(path)
                checkpoint_seconds = time.monotonic() - start
            model.eval()
            optimizer.zero_grad(set_to_none=True)
            sample = ordered[:min(16, len(ordered))]
            synchronize(device)
            start = time.monotonic()
            for offset in range(0, len(sample), config.evaluation.batch_size):
                predict_batch(model, tokenizer, sample[offset:offset + config.evaluation.batch_size], config, device)
            synchronize(device)
            seconds_per_example = (time.monotonic() - start) / len(sample)
            measurements = {"micro_batch_size": micro, "gradient_accumulation_steps": accumulation,
                            "gradient_checkpointing": checkpointing,
                            "precision": "bf16" if device == "cuda" and torch.cuda.is_bf16_supported() else "fp16" if device == "cuda" else "fp32",
                            "update_seconds_median": float(np.median(times)), "update_seconds_p75": float(np.percentile(times, 75)),
                            "update_samples_seconds": times, "peak_reserved_gpu_gib": peak,
                            "free_gpu_gib_after_training": free, "checkpoint_seconds": checkpoint_seconds,
                            "generation_and_nll_seconds_per_example": seconds_per_example,
                            "validation_seconds_estimate": seconds_per_example * config.data.validation_per_source * 2,
                            "test_seconds_estimate": seconds_per_example * config.data.test_per_source * 2,
                            "calibration_updates_discarded": updates}
            break
        except torch.cuda.OutOfMemoryError:
            run.log("calibration_oom", micro_batch=micro, checkpointing=checkpointing)
        finally:
            del model, optimizer
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    if measurements is None:
        raise RuntimeError("No batch configuration fits the configured GPU headroom")
    atomic_json(run.path / "benchmark.json", measurements)
    run.complete("benchmark")
    run.log("benchmark_complete", **measurements)
    return measurements
