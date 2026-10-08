from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from astrolabe.dev_wrapper import resolve_default_gh_python, write_gh_python_hint


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_python(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_resolver_uses_existing_gh_hint_contract(tmp_path: Path) -> None:
    python_bin = _make_python(tmp_path / "python")
    write_gh_python_hint(tmp_path, str(python_bin))

    assert resolve_default_gh_python(tmp_path) == str(python_bin)


def test_checkout_launchers_report_astrolabe_version() -> None:
    env = dict(os.environ, GH_PY=sys.executable)
    for launcher in ("astrolabe", "gh_dev"):
        result = subprocess.run(
            [str(REPO_ROOT / "scripts" / launcher), "--version"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        assert result.stdout.strip() == "Astrolabe 0.1.0"
