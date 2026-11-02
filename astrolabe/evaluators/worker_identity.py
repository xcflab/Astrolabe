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


def identity(python, code_paths, weight_paths, device, model_config=None):
    script = '''import json,sys,importlib.metadata as m
names = ('numpy', 'jax', 'jaxlib', 'dm-haiku', 'torch', 'transformers', 'omegaconf')
versions = {}
for name in names:
 try: versions[name] = m.version(name)
 except m.PackageNotFoundError: versions[name] = None
print(json.dumps([sys.prefix,sys.base_prefix,sys.version,versions]))'''
    probe = subprocess.run([str(python), '-c', script], check=True, capture_output=True, text=True)
    prefix, base_prefix, version, packages = json.loads(probe.stdout)
    return {'environment': {'python': str(Path(python).resolve(strict=True)), 'sha256': digest_path(python),
                            'prefix': prefix, 'base_prefix': base_prefix, 'version': version, 'packages': packages},
            'code': [digest_path(p) for p in code_paths],
            'weights': [digest_path(p) for p in weight_paths],
            'device': device,
            'model_config': model_config if model_config is not None else {}}


def probe_device(python, backend, requested=None, *, env=None):
    args = [str(python), '-m', 'astrolabe.evaluators.device_probe', backend]
    if requested is not None:
        args.append(requested)
    try:
        from astrolabe.runtime.admission import run_probe
    except ImportError:
        if os.environ.get('ASTROLABE_EXECUTION_ENDPOINT') or os.environ.get('ASTROLABE_EXECUTION_TOKEN'):
            raise RuntimeError('broker device probing requires runtime.admission')
        child_env = os.environ.copy()
        if env:
            child_env.update(env)
        proc = subprocess.run(args, check=True, capture_output=True, text=True, env=child_env)
        return json.loads(proc.stdout)
    return run_probe(args, env=env)
