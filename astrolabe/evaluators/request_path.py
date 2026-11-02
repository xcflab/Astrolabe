"""Validate an explicitly supplied private per-call directory."""
import os
from pathlib import Path
import stat


def request_directory(value):
    if not isinstance(value, str): raise ValueError('request directory must be a string')
    path = Path(value)
    if not path.is_absolute() or not path.name.startswith('request-'):
        raise ValueError('invalid request directory')
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('request directory must be private and owned by worker user')
    shared_root = os.environ.get('ASTROLABE_EXECUTION_ARTIFACT_ROOT')
    if shared_root and path.parent != Path(shared_root).resolve(strict=True):
        raise ValueError('request is outside the broker artifact root')
    if path.resolve(strict=True) != path:
        raise ValueError('request directory must be canonical and not a symlink')
    return path
