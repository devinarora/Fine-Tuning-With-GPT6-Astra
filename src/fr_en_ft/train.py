"""Budget-aware Seq2SeqTrainer integration and resumable training control."""

import gc
import importlib.metadata
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch
from transformers import Seq2SeqTrainingArguments, TrainerCallback
from transformers.trainer_callback import TrainerState

from .assets import asset_paths
from .benchmark import plan_steps
from .checkpoints import AtomicTrainer, inspect_checkpoint, latest_checkpoint
from .config import digest
from .data import prepared
from .environment import check_disk, setup
from .evaluate import lexical_scores
from .model import collator, generation_options, load_model, load_tokenizer
from .state import atomic_json, read_json, source_manifest


class Progress(TrainerCallback):
    def __init__(self, run, plan, benchmark, saved=None):
        self.run = run
        self.plan = plan
        self.benchmark = benchmark
        saved = saved or {}
        self.best = saved.get("best")
        self.bad_evaluations = saved.get("bad_evaluations", 0)
        self.stop_reason = None
        self.interrupt = False
        self.last_save = time.monotonic()
        self.last_update = None
        self.recent_updates = []

    def snapshot(self):
        return {"best": self.best, "bad_evaluations": self.bad_evaluations, "stop_reason": self.stop_reason}

    def on_step_begin(self, args, state, control, **kwargs):
        self.last_update = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if self.last_update:
            self.recent_updates = (self.recent_updates + [time.monotonic() - self.last_update])[-50:]
        if time.monotonic() - self.last_save >= self.run.config.runtime.checkpoint_seconds:
            control.should_save = True
        reserve = self.plan["final_reserve_seconds"] + self.benchmark["validation_seconds_estimate"] + self.benchmark["checkpoint_seconds"] + 30
        if self.run.remaining <= reserve:
            self.stop_reason = "time_budget"
            control.should_training_stop = True
            control.should_save = True
            # A validation pass has been included in the reserved time.
            control.should_evaluate = True
        if state.global_step >= state.max_steps:
            control.should_save = True
            control.should_evaluate = True
        if self.interrupt:
            self.stop_reason = "interrupted"
            control.should_training_stop = True
            control.should_save = True
            control.should_evaluate = False
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        score = metrics["eval_chrf"]
        if not math.isfinite(score) or not math.isfinite(metrics["eval_loss"]):
            raise RuntimeError("Non-finite validation result")
        if self.best is None or score > self.best:
            self.best, self.bad_evaluations = score, 0
        else:
            self.bad_evaluations += 1
        control.should_save = True
        if self.bad_evaluations >= self.run.config.training.early_stopping_patience:
            self.stop_reason = "early_stopping"
            control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):
        self.last_save = time.monotonic()

    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ("loss", "grad_norm", "eval_loss"):
            if key in (logs or {}) and not math.isfinite(float(logs[key])):
                raise RuntimeError(f"Non-finite {key}; recover from the previous complete checkpoint")
        update = float(np.median(self.recent_updates)) if self.recent_updates else self.benchmark["update_seconds_p75"]
        self.run.log("trainer", step=state.global_step, training_eta_seconds=max(0, state.max_steps - state.global_step) * update,
                     remaining_budget_seconds=self.run.remaining, **(logs or {}))
        self.run.heartbeat()


def train(run, resume=False):
    config = run.config
    device = setup(config)
    check_disk(run.path)
    data = prepared(run)
    benchmark = read_json(run.path / "benchmark.json")
    plan_path = run.path / "training_plan.json"
    if plan_path.exists():
        plan = read_json(plan_path)
    else:
        baseline = read_json(run.path / "baseline_metrics.json")["performance"]
        reserve = max(config.runtime.evaluation_reserve_seconds,
                      1.4 * (baseline["generation_and_nll_seconds"] + baseline["semantic_seconds"] + baseline["model_load_seconds"]) + 60)
        plan = plan_steps(config, len(data["train"]), benchmark, run.remaining, reserve)
        atomic_json(plan_path, plan)
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate", "numpy", "datasets")}
    contract = digest({"config": config.fingerprint, "data": read_json(run.path / "data_manifest.json")["data_fingerprint"],
                       "plan": plan, "versions": versions, "source": source_manifest()["fingerprint"]})
    checkpoint_root = run.path / "checkpoints"
    checkpoint = latest_checkpoint(checkpoint_root, contract) if resume else None
    if not resume and checkpoint_root.exists() and any(checkpoint_root.glob("checkpoint-*")):
        raise ValueError("Checkpoints exist. Use --resume to continue this run.")
    saved = read_json(checkpoint / "pipeline_state.json") if checkpoint else None
    if saved:
        run.state["spent_seconds"] = max(run.state["spent_seconds"], saved.get("spent_seconds", 0))
    progress = Progress(run, plan, benchmark, saved)
    tokenizer = load_tokenizer(asset_paths(run)["model"])
    model = load_model(str(checkpoint) if checkpoint else asset_paths(run)["model"], device)
    model.config.use_cache = False
    model.generation_config.update(**generation_options(config))
    arguments = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_root), num_train_epochs=config.training.epochs, max_steps=plan["max_steps"],
        per_device_train_batch_size=plan["micro_batch_size"], gradient_accumulation_steps=plan["gradient_accumulation_steps"],
        per_device_eval_batch_size=config.evaluation.batch_size, learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay, warmup_steps=math.ceil(plan["max_steps"] * config.training.warmup_fraction),
        optim="adamw_torch", max_grad_norm=1.0, bf16=plan["precision"] == "bf16", fp16=plan["precision"] == "fp16",
        gradient_checkpointing=plan["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=config.seed, data_seed=config.seed, use_cpu=device == "cpu", dataloader_num_workers=0,
        eval_strategy="steps", eval_steps=plan["eval_steps"], save_strategy="steps", save_steps=plan["eval_steps"],
        save_total_limit=None, metric_for_best_model="chrf", greater_is_better=True, load_best_model_at_end=False,
        predict_with_generate=True, generation_config=model.generation_config,
        logging_steps=min(10, plan["max_steps"]), logging_first_step=True, logging_nan_inf_filter=False,
        report_to="none", disable_tqdm=True, push_to_hub=False,
    )

    def metrics(prediction):
        labels = np.where(prediction.label_ids == -100, tokenizer.pad_token_id, prediction.label_ids)
        predictions = np.where(prediction.predictions == -100, tokenizer.pad_token_id, prediction.predictions)
        refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
        hypotheses = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        scores, _ = lexical_scores(hypotheses, refs)
        return {"bleu": scores["bleu"], "chrf": scores["chrf++"], "ter": scores["ter"]}

    trainer = AtomicTrainer(model=model, args=arguments, train_dataset=data["train"], eval_dataset=data["validation"],
                            processing_class=tokenizer, data_collator=collator(tokenizer, model), compute_metrics=metrics,
                            callbacks=[progress], checkpoint_contract=contract, progress=progress, run_context=run)
    handlers = {}

    def request_stop(signum, frame):
        progress.interrupt = True
        print("Stop requested. Saving at the next complete optimizer update.", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.signal(sig, request_stop)
    run.log("training_plan", **plan)
    try:
        saved_trainer_state = TrainerState.load_from_json(str(checkpoint / "trainer_state.json")) if checkpoint else None
        finished = saved_trainer_state is not None and (
            saved_trainer_state.global_step >= plan["max_steps"] or saved.get("stop_reason") in ("time_budget", "early_stopping"))
        if finished:
            # A crash after the last checkpoint must not cause an extra optimizer update.
            trainer.state = saved_trainer_state
            progress.stop_reason = saved.get("stop_reason")
            if progress.stop_reason == "interrupted":
                progress.stop_reason = None
            if trainer.state.best_model_checkpoint is None:
                result = trainer.evaluate()
                trainer.state.best_metric = result["eval_chrf"]
                trainer.state.best_model_checkpoint = str(checkpoint)
        else:
            trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        if progress.stop_reason == "interrupted":
            raise InterruptedError("Training paused at a complete checkpoint. Use run --resume PATH.")
        if trainer.state.best_model_checkpoint is None:
            metrics_result = trainer.evaluate()
            trainer._determine_best_metric(metrics_result, trial=None)
            trainer._save_checkpoint(trainer.model, trial=None)
        best = Path(trainer.state.best_model_checkpoint)
        inspect_checkpoint(best, contract)
        atomic_json(run.path / "training_result.json", {"best_checkpoint": str(best), "contract": contract,
                    "best_validation_chrf": trainer.state.best_metric, "global_step": trainer.state.global_step,
                    "stop_reason": progress.stop_reason or "planned_steps", "planned_steps": plan["max_steps"],
                    "versions": versions})
        run.complete("train")
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        del trainer, model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
