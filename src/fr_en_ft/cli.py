"""Stage commands and orchestration. Run `python -m fr_en_ft --help`."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .config import load_config
from .state import Run, atomic_json, locked_run, read_json


STAGES = ("download", "prepare", "benchmark", "baseline", "train", "final", "report")


def run_stage(run, stage, resume_training=False):
    from .assets import asset_paths, download
    from .benchmark import benchmark
    from .data import prepare
    from .environment import inspect_environment, setup
    from .evaluate import evaluate
    from .report import report
    from .train import train
    with locked_run(run):
        run.state = read_json(run.path / "state.json")
        if stage in run.state["completed"]:
            print(f"Already complete: {stage}")
            return
        previous = STAGES[STAGES.index(stage) - 1] if stage != "download" else None
        if previous and previous not in run.state["completed"]:
            raise ValueError(f"Run the {previous} stage before {stage}")
        setup(run.config)
        if not (run.path / "environment.json").exists():
            atomic_json(run.path / "environment.json", inspect_environment())
        if stage != "download":
            run.start_clock()
        try:
            if stage == "download":
                download(run)
            elif stage == "prepare":
                prepare(run)
            elif stage == "benchmark":
                benchmark(run)
            elif stage == "baseline":
                evaluate(run, "baseline", asset_paths(run)["model"])
            elif stage == "train":
                train(run, resume=resume_training)
            elif stage == "final":
                from .checkpoints import inspect_checkpoint
                result = read_json(run.path / "training_result.json")
                inspect_checkpoint(result["best_checkpoint"], result["contract"])
                evaluate(run, "final", result["best_checkpoint"])
            elif stage == "report":
                report(run)
        finally:
            run.heartbeat()


def orchestrate(run):
    # A separate process per stage releases GPU allocations, including notebook kernel references.
    print(f"Run directory: {run.path}", flush=True)
    for stage in STAGES:
        run = Run(run.path)
        if stage in run.state["completed"]:
            continue
        command = [sys.executable, "-m", "fr_en_ft", stage, "--run-dir", str(run.path)]
        if stage == "train" and (run.path / "checkpoints").exists():
            command.append("--resume-training")
        result = subprocess.run(command)
        if result.returncode:
            print(f"Stage {stage} stopped with exit code {result.returncode}. Resume with: python -m fr_en_ft run --resume {run.path}", file=sys.stderr)
            return result.returncode
    return 0


def translate(run, text):
    import torch
    from .assets import asset_paths
    from .data import normalize
    from .environment import setup
    from .evaluate import autocast
    from .model import generation_options, load_model, load_tokenizer
    device = setup(run.config)
    tokenizer = load_tokenizer(asset_paths(run)["model"])
    inputs = tokenizer(normalize(text), return_tensors="pt", truncation=False)
    if inputs.input_ids.shape[1] > run.config.data.max_source_tokens:
        raise ValueError(f"Input exceeds {run.config.data.max_source_tokens} model tokens; split it into shorter paragraphs")
    if not text.strip():
        raise ValueError("Input must not be empty")
    model = load_model(read_json(run.path / "training_result.json")["best_checkpoint"], device).eval()
    with torch.inference_mode(), autocast(device):
        output = model.generate(**inputs.to(device), **generation_options(run.config))
    print(tokenizer.decode(output[0], skip_special_tokens=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect the current interpreter and dependencies")
    doctor.add_argument("--freeze", action="store_true", help="Print the installed project dependency closure")
    run_parser = commands.add_parser("run", help="Run all stages, or continue an existing run")
    run_parser.add_argument("--config", default="configs/local.yaml")
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--resume", type=Path)
    run_parser.add_argument("--extra-seconds", type=int, default=0, help="Explicitly extend an existing run's active-time budget")
    init_parser = commands.add_parser("init", help="Create a run directory without executing stages")
    init_parser.add_argument("--config", default="configs/local.yaml")
    init_parser.add_argument("--run-id")
    for stage in STAGES:
        stage_parser = commands.add_parser(stage)
        stage_parser.add_argument("--run-dir", type=Path, required=True)
        if stage == "train":
            stage_parser.add_argument("--resume-training", action="store_true")
    translation = commands.add_parser("translate")
    translation.add_argument("--run-dir", type=Path, required=True)
    translation.add_argument("--text", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            from .environment import dependency_lock, inspect_environment
            print(dependency_lock() if args.freeze else json.dumps(inspect_environment(), indent=2))
            return
        if args.command in ("run", "init"):
            resume = getattr(args, "resume", None)
            run = Run(resume) if resume else Run.create(load_config(args.config), args.run_id)
            extra = getattr(args, "extra_seconds", 0)
            if extra:
                if not resume or extra < 0:
                    raise ValueError("--extra-seconds must be positive and used with --resume")
                with locked_run(run):
                    run.state["extra_seconds"] = run.state.get("extra_seconds", 0) + extra
                    run.heartbeat()
                    run.log("explicit_budget_extension", seconds=extra)
            if args.command == "init":
                print(run.path)
            else:
                raise SystemExit(orchestrate(run))
        elif args.command == "translate":
            translate(Run(args.run_dir), args.text)
        else:
            run_stage(Run(args.run_dir), args.command, getattr(args, "resume_training", False))
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, InterruptedError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
