"""The project runs its tests exclusively inside WSL."""

import platform
import os

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")


def pytest_sessionstart(session):
    if platform.system() != "Linux" or "microsoft" not in platform.release().lower():
        raise RuntimeError("Run tests exclusively in WSL environment")
