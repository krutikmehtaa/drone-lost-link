"""Copy the team's workflow definitions from the notebook into an importable module.

The notebook remains the editable source. This exporter skips its teaching/demo
cells and preserves the workflow functions and schemas for the web app.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


WORKFLOW_CELLS = (24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48)
KEEP_MIXED_CASE = {"FlightPhase"}
SKIP_NAMES = {"RUN_LIVE"}

HEADER = '''"""Workflow exported from notebooks/drone_lost_link.ipynb.

Run scripts/export_workflow.py after changing workflow code in the notebook.
The web app uses the same schemas and functions without running notebook demos.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("DRONE_PROJECT_ROOT", Path(__file__).resolve().parent))
OUTPUT_ROOT = PROJECT_ROOT / "processed"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
_report = json.loads((OUTPUT_ROOT / "corpus_validation_report.json").read_text(encoding="utf-8"))
EMBEDDINGS_PATH = PROJECT_ROOT / _report["embedding_path"]
RUN_LIVE = False
llm_client = None

'''

FOOTER = '''
TRACE_DIR.mkdir(parents=True, exist_ok=True)


def configure_live_client(api_key: str) -> None:
    """Use the course gateway without storing the key in the repository."""
    global RUN_LIVE, llm_client
    if not api_key or not api_key.strip():
        raise ValueError("A course API key is required for a live run.")
    llm_client = OpenAI(api_key=api_key.strip(), base_url="https://api.portkey.ai/v1", timeout=60)
    RUN_LIVE = True
'''


def _source(cell: dict) -> str:
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def _keep(node: ast.stmt) -> bool:
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return True
    if isinstance(node, ast.Assign):
        return all(isinstance(target, ast.Name)
                   and (target.id.isupper() or target.id in KEEP_MIXED_CASE)
                   and target.id not in SKIP_NAMES for target in node.targets)
    return False


def export(notebook_path: Path, output_path: Path) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    parts = [HEADER]
    for index in WORKFLOW_CELLS:
        source = _source(notebook["cells"][index])
        lines = source.splitlines(keepends=True)
        tree = ast.parse(source)
        selected = [node for node in tree.body if _keep(node)]
        if not selected:
            raise ValueError(f"No workflow definitions found in cell {index}")
        parts.append(f"\n# Notebook cell {index}\n")
        for node in selected:
            start = min((d.lineno for d in node.decorator_list), default=node.lineno) if isinstance(
                node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) else node.lineno
            parts.append("".join(lines[start - 1:node.end_lineno]).rstrip() + "\n\n")
    parts.append(FOOTER)
    output_path.write_text("".join(parts), encoding="utf-8")
    ast.parse(output_path.read_text(encoding="utf-8"))
    print(f"Exported workflow to {output_path}")


if __name__ == "__main__":
    workspace = Path(__file__).resolve().parents[1]
    project = Path(__import__("os").environ.get("DRONE_PROJECT_ROOT", workspace))
    export(project / "notebooks" / "drone_lost_link.ipynb", workspace / "workflow_runtime.py")
