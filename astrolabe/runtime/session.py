"""Conservative process isolation for the existing repeat lifecycle."""
import fcntl
import os
import pathlib
import subprocess
import stat
import sys
import tempfile
from omegaconf import OmegaConf


def run_trajectory(cfg):
    # A single per-user lock covers the whole trajectory, including unknown GPU
    # users and their child processes. It intentionally gives up GPU overlap.
    lock_dir = pathlib.Path('/tmp') / f'astrolabe-runtime-{os.getuid()}'
    lock_dir.mkdir(mode=0o700, exist_ok=True)
    info = lock_dir.lstat()
    if info.st_uid != os.getuid() or not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
        raise RuntimeError(f'unsafe runtime lock directory: {lock_dir}')
    lock_fd = os.open(lock_dir / 'exclusive.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(prefix='astrolabe-trajectory-', dir=lock_dir) as tmp:
            config_path = pathlib.Path(tmp) / 'config.yaml'
            OmegaConf.save(cfg, config_path, resolve=False)
            env = os.environ.copy()
            # CLI standalone mode can change cwd after importing this checkout.
            # Preserve the same import roots for the fresh interpreter.
            roots = [str(pathlib.Path(p).resolve()) for p in sys.path if p]
            env["PYTHONPATH"] = os.pathsep.join(roots + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
            subprocess.run([sys.executable, '-m', 'astrolabe.runtime.trajectory', str(config_path)], check=True, env=env, pass_fds=(lock_fd,))
    finally:
        os.close(lock_fd)
