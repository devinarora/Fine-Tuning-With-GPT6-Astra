"""Reproducible local French-to-English fine-tuning."""

import os

# PyTorch 2.14 can JIT-compile even eager tensor operations. Use shipped CUDA kernels
# so the pipeline does not require a Linux C compiler or spend its budget compiling.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

__version__ = "0.1.0"
