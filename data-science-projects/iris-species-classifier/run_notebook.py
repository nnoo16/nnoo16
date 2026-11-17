"""Re-execute the notebook from a clean Python kernel; fail on any cell error."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
notebook_path = root / "analysis.ipynb"
notebook = nbformat.read(notebook_path, as_version=4)
for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
nbformat.validate(notebook)
assert all(output.output_type != "error" for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs)
nbformat.write(notebook, notebook_path)
print("PASS: all notebook cells executed without errors.")
