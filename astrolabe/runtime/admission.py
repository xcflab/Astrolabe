"""Admit small device-discovery subprocesses to the session compute slot."""
import json
import os
import select
import signal
import subprocess
import time
import uuid


def _execute_probe(command, env, timeout, limit=65536):
    if timeout <= 0:
        raise TimeoutError('probe deadline expired')
    p = subprocess.Popen(command, stdout=subprocess.PIPE, env=env, start_new_session=True, bufsize=0)
    deadline = time.monotonic() + timeout
    data = bytearray()
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('device probe timed out')
            ready, _, _ = select.select([p.stdout], [], [], remaining)
            if not ready:
                raise TimeoutError('device probe timed out')
            chunk = os.read(p.stdout.fileno(), min(65536, limit + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > limit:
                raise ValueError('device probe output exceeds byte limit')
        remaining = max(0, deadline - time.monotonic())
        if p.wait(timeout=remaining) != 0:
            raise RuntimeError('device probe failed')
        result = json.loads(data)
        if not isinstance(result, dict):
            raise ValueError('device probe must return a JSON object')
        return result
    except BaseException:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
        raise
    finally:
        p.stdout.close()


def run_probe(command, *, env=None, timeout=60):
    """Return a small JSON device descriptor; never replay uncertain work."""
    endpoint = os.environ.get('ASTROLABE_EXECUTION_ENDPOINT')
    token = os.environ.get('ASTROLABE_EXECUTION_TOKEN')
    if bool(endpoint) != bool(token):
        raise ValueError('incomplete execution broker credentials')
    if not endpoint:
        child_env = os.environ.copy()
        if env:
            child_env.update(env)
        return _execute_probe(command, child_env, timeout)
    from .broker_client import BrokerClient
    proxy = object.__new__(BrokerClient)
    proxy.endpoint, proxy.token, proxy.limit = endpoint, token, 1048576
    overrides = dict(env or {})
    overrides.pop('ASTROLABE_EXECUTION_ENDPOINT', None)
    overrides.pop('ASTROLABE_EXECUTION_TOKEN', None)
    return proxy._rpc({'op': 'probe', 'command': command, 'env': overrides, 'timeout': timeout,
                       'epoch': uuid.uuid4().hex, 'request_id': uuid.uuid4().hex},
                      timeout + 5, submitted=True)['result']
