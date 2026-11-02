"""CPU subprocess work with explicit JSON byte and in-flight limits."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import pathlib
import subprocess
import sys


def _invoke(name, encoded, limit):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['NVIDIA_VISIBLE_DEVICES'] = 'void'
    roots = [str(pathlib.Path(p).resolve()) for p in sys.path if p]
    env['PYTHONPATH'] = os.pathsep.join(roots + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    result = subprocess.run([sys.executable, '-m', 'astrolabe.runtime.cpu_worker', name, str(limit)],
                            input=encoded, stdout=subprocess.PIPE, check=True, env=env)
    if len(result.stdout) > limit:
        raise ValueError('CPU result exceeds max_pending_bytes')
    return result.stdout


class CPUExecutor:
    def __init__(self, workers, *, max_pending_bytes=1048576):
        if type(workers) is not int or workers < 1 or type(max_pending_bytes) is not int or max_pending_bytes < 1:
            raise ValueError('workers and max_pending_bytes must be positive integers')
        self.max_pending_bytes = max_pending_bytes
        self.max_pending_items = workers * 2
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def map(self, handler, items):
        """Return results in input order, bounded by in-flight input bytes and count."""
        pending = []
        size = 0
        for item in items:
            encoded = json.dumps(item, allow_nan=False, separators=(',', ':')).encode()
            n = len(encoded)
            if n > self.max_pending_bytes:
                raise ValueError('CPU work item exceeds max_pending_bytes')
            while pending and (size + n > self.max_pending_bytes or len(pending) >= self.max_pending_items):
                f, old = pending.pop(0)
                size -= old
                yield json.loads(f.result())
            pending.append((self.pool.submit(_invoke, handler, encoded, self.max_pending_bytes), n))
            size += n
        for f, _ in pending:
            yield json.loads(f.result())

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
