"""Convert notebooks/src/*.py (jupytext percent format) into executed notebooks/*.ipynb.

The .py sources are what gets reviewed and diffed; the .ipynb files are what GitHub renders.
Run from the repo root:  python scripts/build_notebooks.py [name-filter]
"""
import subprocess
import sys
from pathlib import Path

import jupytext
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "src"
OUT = ROOT / "notebooks"


def build(path: Path) -> None:
    nb = jupytext.read(path)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    ExecutePreprocessor(timeout=1200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT)}})
    nbformat.write(nb, OUT / f"{path.stem}.ipynb")
    print(f"built {path.stem}.ipynb")


if __name__ == "__main__":
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    subprocess.run([sys.executable, "-c", "import nhsae"], check=True)
    for p in sorted(SRC.glob("*.py")):
        if flt in p.stem:
            build(p)
