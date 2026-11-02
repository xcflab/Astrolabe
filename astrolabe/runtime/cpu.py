"""Session-lifetime CPU subprocess pool, bounded by input/output bytes."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import pathlib
import queue
import sys
from .client import DaemonClient


class CPUExecutor:
    def __init__(self, workers, *, max_pending_bytes=1048576, task_timeout=None):
        if type(workers) is not int or workers < 1 or type(max_pending_bytes) is not int or max_pending_bytes < 1:
            raise ValueError('workers and max_pending_bytes must be positive integers')
        self.max_pending_bytes = max_pending_bytes
        if task_timeout is not None and (not isinstance(task_timeout, (int, float)) or task_timeout <= 0):
            raise ValueError('task_timeout must be positive')
        self.task_timeout = task_timeout
        self.max_pending_items = workers * 2
        self.max_result_bytes = max_pending_bytes // self.max_pending_items
        if self.max_result_bytes < 1:
            raise ValueError('max_pending_bytes too small for worker count')
        self._clients = []
        self._available = queue.Queue()
        env = {'CUDA_VISIBLE_DEVICES': '', 'NVIDIA_VISIBLE_DEVICES': 'void',
               'HIP_VISIBLE_DEVICES': '', 'ROCR_VISIBLE_DEVICES': '',
               'PYTHONPATH': os.pathsep.join(str(pathlib.Path(p).resolve()) for p in sys.path if p)}
        try:
            for _ in range(workers):
                c = DaemonClient([sys.executable, '-m', 'astrolabe.runtime.cpu_worker', str(max_pending_bytes)],
                                 {'worker': 'cpu', 'protocol': 1}, env=env,
                                 max_message_bytes=max_pending_bytes + 65536, _direct=True)
                self._clients.append(c)
                self._available.put(c)
            self.pool = ThreadPoolExecutor(max_workers=workers)
        except BaseException:
            for c in self._clients:
                c.close()
            raise

    def _invoke(self, handler, encoded):
        c = self._available.get()
        try:
            return c.call({'handler': handler, 'item': json.loads(encoded),
                           'result_limit': self.max_result_bytes}, timeout=self.task_timeout)['value']
        finally:
            self._available.put(c)

    def map(self, handler, items):
        pending = []
        size = 0
        try:
            for item in items:
                encoded = json.dumps(item, allow_nan=False, separators=(',', ':')).encode()
                n = len(encoded)
                if n > self.max_pending_bytes:
                    raise ValueError('CPU work item exceeds max_pending_bytes')
                while pending and (size + n > self.max_pending_bytes or len(pending) >= self.max_pending_items):
                    f, old = pending.pop(0)
                    size -= old
                    yield f.result()
                pending.append((self.pool.submit(self._invoke, handler, encoded), n))
                size += n
            while pending:
                f, _ = pending.pop(0)
                yield f.result()
        finally:
            for f, _ in pending:
                f.cancel()

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        for c in self._clients:
            c.close()
        self._clients.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
