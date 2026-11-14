"""Execute every ```python block in learn/*.md, in order, one namespace per lesson (from the repo root)."""
import os
import re
from pathlib import Path

import matplotlib
import pytest

from nhsae.config import PANEL, ROOT

matplotlib.use("Agg")
LESSONS = sorted((ROOT / "learn").glob("*.md"))
BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def blocks(path: Path) -> list[str]:
    return BLOCK.findall(path.read_text())


@pytest.mark.skipif(not PANEL.exists(), reason="processed data absent")
@pytest.mark.parametrize("lesson", [p for p in LESSONS if blocks(p)], ids=lambda p: p.stem)
def test_lesson_code_runs(lesson, monkeypatch):
    monkeypatch.chdir(ROOT)
    ns: dict = {"__name__": "__learn__"}
    for i, code in enumerate(blocks(lesson)):
        try:
            exec(compile(code, f"{lesson.name}[block {i}]", "exec"), ns)
        except Exception as exc:  # pragma: no cover - surfaced as a test failure
            pytest.fail(f"{lesson.name} block {i} failed: {exc!r}\n{code}")
    assert os.getcwd() == str(ROOT)
