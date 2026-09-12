# Troubleshooting

## Wrong interpreter

Run `python -m fr_en_ft doctor` in WSL. Check that the executable belongs to the intended environment and that CUDA is available. Use `python -m pip` for package commands so pip uses that same interpreter.

Run tests with `python -m pytest -q` inside WSL. The test configuration rejects Windows execution.

## Dependency conflicts

Run `python -m pip check`. The selected Transformers stack needs a compatible CUDA-enabled PyTorch build, and checkpoint loading requires at least PyTorch 2.6. Use the WSL dependency lock to reproduce the package versions. See [environment setup](environments.md) for installation commands.

## C compiler or Triton error

Start a fresh CLI process with `TORCH_DISABLE_NATIVE_JIT=1`, the project default. PyTorch caches some settings at import time, so changing the variable after an import may not help. The notebook starts stages in separate processes. The supported path uses SDPA attention with generation compilation disabled.

## Hugging Face throttling or interrupted downloads

The download stage retries transient HTTP failures and honors numeric `Retry-After` delays. If a rate limit persists, wait and rerun the stage. Hugging Face's cache preserves completed files. The selected assets are ungated; an optional token can increase request limits. Keep credentials out of project files.

If a pinned revision becomes unavailable, choose and document a replacement in a new configuration and run. Replacing a SHA with `main` would make the source version change over time.

## GPU memory pressure

Calibration tries smaller microbatches and gradient checkpointing while preserving the effective batch size. If none fits, close GPU-heavy applications or start a new configuration with a lower batch or length limit.

Another workload can cause an out-of-memory error after calibration. Close it and resume from the last complete checkpoint. Device-wide free memory includes other applications; PyTorch reserved-memory figures cover only PyTorch allocations.

## Interrupted run

Use `python -m fr_en_ft run --resume PATH`, replacing `PATH` with the run directory. Completed stages are skipped. Training restores a compatible checkpoint, and evaluation resumes from saved prediction batches. Ctrl+C requests a save after the current optimizer update. A hard termination can lose work since the last complete checkpoint.

Directories named `.incomplete-*` contain unfinished checkpoint writes and are not loaded. Once no process is using the run, abandoned temporary directories can be removed. Keep complete recovery checkpoints and the validation-best checkpoint referenced by `training_result.json`.

Changes to source code, package versions, configuration, or selected data can invalidate checkpoint compatibility. Restore the recorded setup to resume, or start a new run. If the validation-best checkpoint is corrupt, restore it from another copy.

## Time budget

The training plan reserves time for final evaluation before choosing its maximum update count. Other workloads can change throughput. Completed evaluation batches are saved if the budget expires.

Use `python -m fr_en_ft run --resume PATH --extra-seconds 600` to add ten active minutes. The extension is logged without rewriting the training schedule. A small overrun can occur while finishing a batch, checkpoint, or scoring call.

## Interpreting translation scores

A fine-tune can improve reference likelihood while making generated translations worse. Compare quality metrics, paired intervals, and source-specific results. The original model already translates in this direction and may have seen OPUS material during pretraining. A smoke run is too small to support a quality claim.

## Sharing reports or notebooks

Local reports and notebook outputs can contain interpreter paths, environment details, hardware information, and timings. Prepare a separate publication copy using the [sharing guidance](workflow.md#sharing-results). Redact exported text rather than changing checkpoint state or its checksums.
