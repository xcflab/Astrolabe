import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

# Keep in sync with the `version` in setup.py; `astrolabe --version` and the installed
# distribution metadata must agree.
__version__ = "0.1.0"

# Installed layout: the repo root of a checkout is also the install root of a
# wheel, i.e. `run.py`, `config/` and `resources/` are installed beside the
# `astrolabe` package.
_INSTALL_ROOT = Path(__file__).resolve().parents[1]


def install_root() -> Path:
    """Directory holding `run.py`, `config/` and `resources/`."""
    return _INSTALL_ROOT


def config_dir() -> Path:
    """Directory holding the Hydra config resources (`config/`)."""
    return _INSTALL_ROOT / "config"


def run_script_path() -> Path:
    """Path of the pipeline runner script (`run.py`)."""
    return _INSTALL_ROOT / "run.py"


from .app import Astrolabe

__all__ = ["Astrolabe", "__version__", "install_root", "config_dir", "run_script_path"]
