from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_repo_python(snippet: str) -> subprocess.CompletedProcess[str]:
    body = textwrap.dedent(snippet).strip()
    code = (
        "import json\n"
        + "import sys\n"
        + "from pathlib import Path\n"
        + f"sys.path.insert(0, {str((REPO_ROOT / 'vendor')).__repr__()})\n"
        + f"sys.path.insert(0, {str(REPO_ROOT).__repr__()})\n"
        + body
        + "\n"
    )
    return subprocess.run(
        [sys.executable, '-c', code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_importing_cli_does_not_pull_torch_or_losses_package() -> None:
    proc = _run_repo_python(
        """
import astrolabe.cli
print(json.dumps({
    'torch': 'torch' in sys.modules,
    'losses_pkg': 'astrolabe.losses' in sys.modules,
    'stage_transition': 'astrolabe.stage_transition' in sys.modules,
}))
"""
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ''
    payload = json.loads(proc.stdout.strip())
    assert payload == {
        'torch': False,
        'losses_pkg': False,
        'stage_transition': False,
    }


def test_cli_help_has_no_torch_warning() -> None:
    proc = _run_repo_python(
        """
from astrolabe.cli import main
try:
    main(['-h'])
except SystemExit as exc:
    print(json.dumps({'exit_code': int(exc.code)}))
"""
    )

    assert proc.returncode == 0
    assert 'FutureWarning' not in proc.stderr
    assert 'pynvml' not in proc.stderr
    assert 'usage: astrolabe' in proc.stdout
    assert '{"exit_code": 0}' in proc.stdout


def test_completion_value_resolution_for_out_root_has_no_torch_warning() -> None:
    proc = _run_repo_python(
        """
from astrolabe.shell_completion import complete_words
items = complete_words(['posthoc', 'collect', '--out-root', ''], cword=3, cwd=Path.cwd())
print(json.dumps({
    'count': len(items),
    'contains_config': any(item.value == 'config/' for item in items),
    'contains_tests': any(item.value == 'tests/' for item in items),
    'torch': 'torch' in sys.modules,
}))
"""
    )

    assert proc.returncode == 0, proc.stderr
    assert 'FutureWarning' not in proc.stderr
    assert 'pynvml' not in proc.stderr
    payload = json.loads(proc.stdout.strip())
    assert payload['count'] > 0
    assert payload['contains_config'] is True
    assert payload['contains_tests'] is True
    assert payload['torch'] is False
