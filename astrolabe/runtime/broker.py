"""Private session broker; model residency and per-device compute are separate."""
import hashlib
import json
import os
import signal
import shutil
import socketserver
import sys
import threading
import time
import uuid
from .client import DaemonClient, OutcomeUnknown, WorkerError
from .admission import _execute_probe

MAX_CONTROL = 4 * 1024 * 1024


class State:
    def __init__(self, token, max_resident_models):
        self.token = token
        self.max_resident_models = max_resident_models
        self.lock = threading.Lock()
        self.start_lock = threading.Lock()
        self.workers = {}
        self.devices = {}
        self.used = set()
        self.poisoned = set()
        self.stopping = False

    def register(self, req):
        identity = req['identity']
        if not isinstance(identity, dict):
            raise ValueError('identity must be an object')
        if type(req.get('limit')) is not int or not 256 <= req['limit'] <= MAX_CONTROL - 65536:
            raise ValueError('invalid worker message limit')
        if not isinstance(req.get('env'), dict) or any(
            key in req['env'] for key in ('ASTROLABE_EXECUTION_ENDPOINT', 'ASTROLABE_EXECUTION_TOKEN')
        ):
            raise ValueError('invalid worker environment')
        if ('ASTROLABE_EXECUTION_ARTIFACT_ROOT' in req['env'] and
            req['env']['ASTROLABE_EXECUTION_ARTIFACT_ROOT'] != os.environ.get('ASTROLABE_EXECUTION_ARTIFACT_ROOT')):
            raise ValueError('worker environment changes artifact root')
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        spec = (req['command'], req['env'], req['limit'])
        timeout = req.get('startup_timeout')
        if not isinstance(timeout, (int, float)) or not 0 < timeout <= 3600:
            raise ValueError('invalid worker startup timeout')
        deadline = time.monotonic() + timeout
        device = identity.get('device', 'cpu')
        if isinstance(device, dict):
            visible = device.get('visible')
            platform = device.get('platform', device.get('name', ''))
            if platform == 'cpu':
                if visible != '':
                    raise ValueError('CPU workers must declare empty CUDA visibility')
            elif visible != os.environ.get('CUDA_VISIBLE_DEVICES'):
                raise ValueError('worker device visibility differs from session allocation')
            if platform != 'cpu' and (not visible or len(visible.split(',')) != 1):
                raise ValueError('GPU broker requires exactly one allocated visible device')
            if platform not in ('cpu', 'gpu', 'cuda:0', 'mock') or (
                platform == 'gpu' and device.get('id', 0) != 0
            ):
                raise ValueError('unsupported device selection')
            device_key = 'cpu' if platform == 'cpu' else 'gpu:' + visible
        elif device == 'cpu':
            device_key = 'cpu'
        else:
            raise ValueError('device must declare an allocated device')
        launch_env = dict(req['env'])
        if device_key == 'cpu':
            launch_env.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void',
                              HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')
        else:
            for env_key in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
                            'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
                if env_key in launch_env and launch_env[env_key] != os.environ.get(env_key):
                    raise ValueError('worker environment changes allocated device visibility')
        if not self.start_lock.acquire(timeout=timeout):
            raise TimeoutError('worker startup admission timed out')
        try:
            with self.lock:
                if self.stopping:
                    raise OutcomeUnknown('broker session is stopping')
                record = self.workers.get(key)
                if record is not None:
                    if record['spec'] != spec:
                        raise ValueError('identity reused with different launch command or environment')
                    return {'handle': key, 'hello': record['client'].hello}
                if len(self.workers) >= self.max_resident_models:
                    raise ValueError('resident model limit reached')
                device_lock = self.devices.setdefault(device_key, threading.Lock())
            # Model load may initialize the device. Keep the compute lease
            # during startup without holding the registry mutex.
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not device_lock.acquire(timeout=remaining):
                raise TimeoutError('device startup admission timed out')
            try:
                with self.lock:
                    if self.stopping or device_key in self.poisoned:
                        raise OutcomeUnknown('device fenced or session stopping')
                try:
                    client = DaemonClient(req['command'], identity, env=launch_env,
                                          max_message_bytes=req['limit'],
                                          startup_timeout=max(.001, deadline - time.monotonic()), _direct=True)
                except BaseException:
                    with self.lock:
                        self.poisoned.add(device_key)
                    raise
                record = {'client': client, 'spec': spec, 'device': device_key}
                with self.lock:
                    if self.stopping:
                        client.close()
                        raise OutcomeUnknown('broker session stopped during model startup')
                    self.workers[key] = record
            finally:
                device_lock.release()
            return {'handle': key, 'hello': record['client'].hello}
        finally:
            self.start_lock.release()

    def call(self, req):
        ident = (req['epoch'], req['request_id'])
        if not all(isinstance(x, str) and x for x in ident):
            raise ValueError('invalid request identity')
        with self.lock:
            if self.stopping:
                raise OutcomeUnknown('broker session is stopping')
            if ident in self.used:
                raise ValueError('duplicate request')
            if len(self.used) >= 100000:
                raise ValueError('session request limit reached')
            self.used.add(ident)
            record = self.workers[req['handle']]
            device_lock = self.devices[record['device']]
            if record['device'] in self.poisoned:
                raise OutcomeUnknown('device fenced after uncertain worker outcome')
        timeout = req.get('timeout')
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            raise ValueError('invalid timeout')
        start = time.monotonic()
        acquired = device_lock.acquire() if timeout is None else device_lock.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError('compute admission timed out before submission')
        try:
            with self.lock:
                if record['device'] in self.poisoned or self.stopping:
                    raise OutcomeUnknown('device fenced or session stopping')
            remaining = None if timeout is None else timeout - (time.monotonic() - start)
            if remaining is not None and remaining <= 0:
                raise TimeoutError('compute admission timed out before submission')
            try:
                return record['client'].call(req['payload'], timeout=remaining, request_id=uuid.uuid4().hex)
            except WorkerError:
                with self.lock:
                    self.poisoned.add(record['device'])
                raise
        finally:
            device_lock.release()

    def probe(self, req):
        timeout = req.get('timeout')
        if not isinstance(timeout, (int, float)) or not 0 < timeout <= 3600:
            raise ValueError('invalid probe timeout')
        overrides = req.get('env')
        if not isinstance(overrides, dict):
            raise ValueError('invalid probe environment')
        for env_key in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
                        'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
                        'ASTROLABE_EXECUTION_ENDPOINT', 'ASTROLABE_EXECUTION_TOKEN',
                        'ASTROLABE_EXECUTION_ARTIFACT_ROOT'):
            if env_key in overrides and overrides[env_key] != os.environ.get(env_key):
                raise ValueError('probe environment changes allocated resources')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible and len(visible.split(',')) != 1:
            raise ValueError('device probe requires exactly one allocated GPU')
        device_key = 'gpu:' + visible if visible else 'cpu'
        env = os.environ.copy()
        env.update(overrides)
        if not visible:
            env.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void',
                       HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')
        with self.lock:
            if self.stopping or device_key in self.poisoned:
                raise OutcomeUnknown('device fenced or session stopping')
            ident = (req.get('epoch'), req.get('request_id'))
            if not all(isinstance(x, str) and x for x in ident) or ident in self.used:
                raise ValueError('invalid or duplicate probe identity')
            if len(self.used) >= 100000:
                raise ValueError('session request limit reached')
            self.used.add(ident)
            device_lock = self.devices.setdefault(device_key, threading.Lock())
        start = time.monotonic()
        if not device_lock.acquire(timeout=timeout):
            raise TimeoutError('device probe admission timed out')
        try:
            with self.lock:
                if self.stopping or device_key in self.poisoned:
                    raise OutcomeUnknown('device fenced or session stopping')
            try:
                return _execute_probe(req['command'], env, timeout - (time.monotonic() - start))
            except Exception as exc:
                with self.lock:
                    self.poisoned.add(device_key)
                raise OutcomeUnknown('device probe failed; device fenced') from exc
        finally:
            device_lock.release()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            self.request.settimeout(30)
            raw = self.rfile.readline(MAX_CONTROL + 1)
            if len(raw) > MAX_CONTROL or not raw.endswith(b'\n'):
                raise ValueError('control message too large or incomplete')
            req = json.loads(raw)
            if req.get('token') != self.server.state.token:
                raise ValueError('unauthorized')
            if req.get('op') == 'register':
                result = self.server.state.register(req)
            elif req.get('op') == 'call':
                result = {'result': self.server.state.call(req)}
            elif req.get('op') == 'probe':
                result = {'result': self.server.state.probe(req)}
            else:
                raise ValueError('unknown broker operation')
            result['ok'] = True
        except Exception as exc:
            result = {'ok': False, 'unknown': isinstance(exc, OutcomeUnknown),
                      'error': f'{type(exc).__name__}: {exc}'}
        try:
            self.wfile.write((json.dumps(result, allow_nan=False, separators=(',', ':')) + '\n').encode())
        except OSError:
            pass


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = False
    block_on_close = True
    _slots = threading.BoundedSemaphore(32)

    def process_request(self, request, client_address):
        self._slots.acquire()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def main():
    endpoint, token_fd, parent_fd, artifact_root = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    max_resident_models = int(sys.argv[5])
    with os.fdopen(token_fd, 'rb') as token_pipe:
        token = token_pipe.read(128).decode()
    if not token:
        raise ValueError('missing broker token')
    address = '\0' + endpoint[1:] if endpoint.startswith('@') else endpoint
    server = Server(address, Handler)
    if not endpoint.startswith('@'):
        os.chmod(endpoint, 0o600)
    server.state = State(token, max_resident_models)
    def watch():
        # EOF means the owning session exited. In-flight calls retain their
        # leases until complete or the underlying worker has been terminated.
        with os.fdopen(parent_fd, 'rb') as parent:
            while parent.read(1):
                pass
        with server.state.lock:
            server.state.stopping = True
            # Interrupt active work, then let the handler wait for worker exit
            # while it still owns the compute lock. No request is replayed.
            for record in server.state.workers.values():
                process = record['client'].process
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        server.shutdown()
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        for record in server.state.workers.values():
            record['client'].close()
        if not endpoint.startswith('@'):
            try:
                os.unlink(endpoint)
            except FileNotFoundError:
                pass
        shutil.rmtree(artifact_root, ignore_errors=True)


if __name__ == '__main__':
    main()
