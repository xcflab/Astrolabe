"""Content identities independently recomputed in each worker process."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import sys
import json
import subprocess


def _digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest_path(path):
    path = Path(path).expanduser().resolve(strict=True)
    if path.is_file(): return _digest_file(path)
    if not path.is_dir(): raise ValueError(f'not a file or directory: {path}')
    h = hashlib.sha256()
    for file in sorted(p for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc')):
        rel = file.relative_to(path).as_posix().encode()
        h.update(len(rel).to_bytes(8, 'big')); h.update(rel)
        h.update(bytes.fromhex(_digest_file(file)))
    return h.hexdigest()


def identity(python, code_paths, weight_paths, device):
    probe = subprocess.run([str(python), '-c', 'import json,sys; print(json.dumps([sys.prefix,sys.base_prefix,sys.version]))'],
                           check=True, capture_output=True, text=True)
    prefix, base_prefix, version = json.loads(probe.stdout)
    return {'environment': {'python': str(Path(python).resolve(strict=True)), 'sha256': digest_path(python),
                            'prefix': prefix, 'base_prefix': base_prefix, 'version': version},
            'code': {str(Path(p).resolve(strict=True)): digest_path(p) for p in code_paths},
            'weights': {str(Path(p).resolve(strict=True)): digest_path(p) for p in weight_paths},
            'device': device}
