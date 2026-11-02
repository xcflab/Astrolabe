"""Bounded, synchronous JSONL transport for a resident worker."""
from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import threading
import time
import uuid


class WorkerError(RuntimeError):
    pass


class OutcomeUnknown(WorkerError):
    """The worker may have executed the request; never automatically replay it."""


def _encode(value, limit):
    data = (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if len(data) > limit:
        raise ValueError('JSONL message exceeds max_message_bytes')
    return data


class DaemonClient:
    def __init__(self, command, identity, *, env=None, max_message_bytes=1048576, startup_timeout=60):
        if max_message_bytes < 256:
            raise ValueError('max_message_bytes must be at least 256')
        self.limit = max_message_bytes
        self.epoch = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._used = set()
        self._buffer = b''
        child_env = os.environ.copy()
        if env:
            child_env.update(env)
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        env=child_env, bufsize=0, start_new_session=True)
        os.set_blocking(self.process.stdin.fileno(), False)
        try:
            hello = self._read(startup_timeout)
            if (hello.get('type') != 'hello' or hello.get('protocol') != 1 or
                hello.get('identity') != identity or hello.get('pid') != self.process.pid or
                type(hello.get('load_count')) is not int or hello['load_count'] < 0):
                raise WorkerError('worker identity or handshake mismatch')
            self.hello = hello
        except BaseException:
            self.close()
            raise

    def _read(self, timeout):
        deadline = None if timeout is None else time.monotonic() + timeout
        while b'\n' not in self._buffer:
            remaining = None if deadline is None else max(0, deadline - time.monotonic())
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                raise TimeoutError('worker response timed out')
            chunk = os.read(self.process.stdout.fileno(), min(65536, self.limit + 1 - len(self._buffer)))
            if not chunk:
                raise EOFError('worker exited')
            self._buffer += chunk
            if len(self._buffer) > self.limit:
                raise WorkerError('worker message exceeds max_message_bytes')
        line, self._buffer = self._buffer.split(b'\n', 1)
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise WorkerError('worker message must be an object')
        return obj

    def call(self, payload, *, timeout=None, request_id=None):
        if not isinstance(payload, dict):
            raise TypeError('payload must be a JSON object')
        if timeout is not None and timeout <= 0:
            raise ValueError('timeout must be positive')
        with self._lock:
            rid = request_id or uuid.uuid4().hex
            if not isinstance(rid, str) or not rid or rid in self._used:
                raise ValueError('request_id must be a new nonempty string')
            data = _encode(dict(type='call', request_id=rid, epoch=self.epoch, payload=payload), self.limit)
            self._used.add(rid)
            try:
                deadline = None if timeout is None else time.monotonic() + timeout
                view = memoryview(data)
                while view:
                    remaining = None if deadline is None else max(0, deadline - time.monotonic())
                    _, ready, _ = select.select([], [self.process.stdin], [], remaining)
                    if not ready:
                        raise TimeoutError('worker request timed out')
                    view = view[os.write(self.process.stdin.fileno(), view):]
                remaining = None if deadline is None else max(0, deadline - time.monotonic())
                result = self._read(remaining)
                if result.get('request_id') != rid or result.get('epoch') != self.epoch:
                    raise WorkerError('response identity mismatch')
                if result.get('type') == 'error':
                    raise WorkerError('worker error: ' + str(result.get('error')))
                if result.get('type') != 'result' or not isinstance(result.get('result'), dict):
                    raise WorkerError('invalid worker result')
                return result['result']
            except WorkerError as exc:
                self.close()
                if str(exc).startswith('worker error:'):
                    raise
                raise OutcomeUnknown('invalid response; request was not replayed') from exc
            except (OSError, EOFError, TimeoutError, ValueError) as exc:
                self.close()
                raise OutcomeUnknown('worker outcome unknown; request was not replayed') from exc

    def close(self):
        p = self.process
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
        for stream in (p.stdin, p.stdout):
            if stream:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
