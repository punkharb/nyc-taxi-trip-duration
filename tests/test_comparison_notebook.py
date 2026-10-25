import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "02_model_comparison.ipynb"


def test_model_comparison_is_a_separate_runnable_notebook():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        compile("".join(cell["source"]), str(NOTEBOOK), "exec")
