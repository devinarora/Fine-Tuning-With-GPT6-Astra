from pathlib import Path

import nbformat
from nbclient import NotebookClient


def test_notebook_executes_in_observation_mode():
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks/workflow.ipynb", as_version=4)
    nbformat.validate(notebook)
    result = NotebookClient(notebook, timeout=60, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
    assert all(output.output_type != "error" for cell in result.cells if cell.cell_type == "code" for output in cell.outputs)
