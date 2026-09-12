# Environment setup

Use Windows for editing and package inspection. Run training, evaluation, notebooks, and tests in WSL with a CUDA-enabled PyTorch installation.

The examples use `fr-en-ft` as a generic Conda environment name. Substitute the name of your existing environment when needed. Run commands from the repository root in a WSL terminal.

## Use an existing environment

Activate an environment that contains the required dependencies, then install the project:

```bash
conda activate fr-en-ft
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -m fr_en_ft doctor
```

The editable installation makes the CLI available and uses the source in this repository. `--no-deps` skips dependency installation. Confirm that `doctor` reports the intended interpreter and available CUDA support.

## Create an environment

The `environments/` directory contains Python environment definitions and platform-specific dependency locks. The `--name` argument below overrides the name stored in the YAML file.

```bash
conda env create --name fr-en-ft --file environments/wsl.yml
conda activate fr-en-ft
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r environments/requirements-wsl.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -m fr_en_ft doctor
```

Use the lock for the target operating system. The WSL lock requires a compatible CUDA wheel build and driver. The Windows lock is a reference for editing and inspection tools.

`python -m fr_en_ft doctor --freeze` lists the installed project dependencies, including transitive dependencies. Exact versions support reproducibility, though results can still vary across hardware and drivers.

## Runtime requirements

The pipeline measures available resources during calibration. Storage locations, memory limits, and the time budget are configured in `configs/local.yaml`. Choose limits that leave room for other applications and checkpoint writes. For WSL storage, check free space on both the Linux filesystem and its backing drive.

The project sets `TORCH_DISABLE_NATIVE_JIT=1` before importing PyTorch and disables generation compilation. This allows the supported path to use prebuilt CUDA kernels and SDPA attention without a C compiler. Start stages in fresh processes so these settings apply before PyTorch loads.

## Open the notebook

Select the project environment as the kernel in a WSL-aware editor, or start JupyterLab from the activated environment:

```bash
python -m jupyterlab notebooks/workflow.ipynb
```

The notebook runs CLI stages in separate processes and displays their saved artifacts. Set `RUN` to inspect an existing run, or enable stage execution to create one.

Environment inspection and notebook output can include absolute paths and local system details. Clear saved notebook outputs before publication and follow the [sharing guidance](workflow.md#sharing-results).
