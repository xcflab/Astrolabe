"""Same-UID persistent model owner with bounded pickle5 request transport."""

from __future__ import annotations

import argparse
import atexit
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
from copy import deepcopy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import queue
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any
import uuid


PROTOCOL = "astrolabe-model-daemon/1"
MAX_FRAME_BYTES = 8 * 1024 * 1024
DEFAULT_DRAIN_TIMEOUT = 30.0
_REPO_ROOT = Path(__file__).resolve().parents[1]


class ModelDaemonError(RuntimeError):
    pass


class DaemonStartupError(ModelDaemonError):
    pass


class DaemonChildExited(ModelDaemonError):
    pass


class DaemonDisconnected(ModelDaemonError):
    pass


class DaemonDrainFailed(ModelDaemonError):
    """The child stopped, but at least one admitted reply could not be sent."""

    def __init__(self, result):
        super().__init__(f"Daemon stopped with incomplete reply drain: {result}")
        self.result = result


class IdentityMismatch(ModelDaemonError):
    pass


class DaemonRequestFailed(ModelDaemonError):
    def __init__(self, message: str, *, remote_type: str | None = None,
                 code: str | None = None) -> None:
        super().__init__(message)
        self.remote_type = remote_type
        self.code = code


class RequestOutcomeUnknown(ModelDaemonError):
    """A send was attempted, but no trustworthy completion was received.

    The request may have run. Callers must not automatically replay it.
    """
    outcome_unknown = True

    def __init__(
        self,
        message: str,
        *,
        request_id: str | None = None,
        generation: str | None = None,
    ) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.generation = generation


class FrameTooLarge(ModelDaemonError):
    pass


class ProtocolError(ModelDaemonError):
    pass


@dataclass(slots=True)
class DaemonSpec:
    backend: str
    factory: str
    settings: dict[str, Any]
    python_bin: str = sys.executable
    cwd: str | None = None
    env: dict[str, str] | None = None
    provider: dict[str, Any] | None = None
    max_frame_bytes: int = MAX_FRAME_BYTES


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"Daemon settings must be JSON-like, got {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _module_file(module_name: str, spec: DaemonSpec) -> Path | None:
    candidates = [_REPO_ROOT]
    if spec.cwd:
        candidates.append(Path(spec.cwd).expanduser())
    pythonpath = (spec.env or {}).get("PYTHONPATH", "")
    candidates.extend(Path(item) for item in pythonpath.split(os.pathsep) if item)
    parts = module_name.split(".")
    for root in candidates:
        root = root.expanduser().resolve()
        module_file = root.joinpath(*parts).with_suffix(".py")
        package_file = root.joinpath(*parts, "__init__.py")
        if module_file.is_file():
            return module_file
        if package_file.is_file():
            return package_file
    return None


def _adapter_hash(spec: DaemonSpec) -> str:
    module_name, separator, _ = spec.factory.partition(":")
    if not separator:
        module_name, _, _ = spec.factory.rpartition(".")
    module_file = _module_file(module_name, spec)
    digest = hashlib.sha256()
    if module_file is None:
        digest.update(spec.factory.encode("utf-8"))
    else:
        with module_file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def model_identity(spec: DaemonSpec, *, max_pending: int = 4) -> str:
    identity_settings = dict(spec.settings)
    identity_settings.pop("_initial_random_state", None)
    payload = {
        "protocol": PROTOCOL,
        "backend": spec.backend,
        "factory": spec.factory,
        "settings": identity_settings,
        "python": str(spec.python_bin),
        "cwd": str(Path(spec.cwd).expanduser().resolve()) if spec.cwd else None,
        "env": dict(sorted((spec.env or {}).items())),
        "provider": spec.provider,
        "adapter_sha256": _adapter_hash(spec),
        "max_pending": int(max_pending),
        "max_frame_bytes": int(spec.max_frame_bytes),
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _provider_selected(spec: DaemonSpec) -> bool:
    environment = os.environ.copy()
    environment.update(spec.env or {})
    return spec.provider is not None or bool(environment.get("ASTROLABE_MODEL_PROVIDERS"))


def _prepare_provider_spec(spec: DaemonSpec) -> DaemonSpec:
    if _provider_selected(spec):
        raise ModelDaemonError(
            "Provider configuration is not supported by the model execution foundation"
        )
    return spec


def _activate_child_provider(spec: DaemonSpec) -> None:
    # Validate again at the child boundary, before importing the factory.
    _prepare_provider_spec(spec)


def _provider_environment_for_spec(spec: DaemonSpec) -> dict[str, str]:
    value = (spec.env or {}).get("ASTROLABE_MODEL_PROVIDERS")
    return {"ASTROLABE_MODEL_PROVIDERS": str(value)} if value is not None else {}


def _load_factory(path: str) -> Callable[..., Any]:
    if ":" in path:
        module_name, attr_path = path.split(":", 1)
    else:
        module_name, _, attr_path = path.rpartition(".")
    if not module_name or not attr_path:
        raise ValueError("factory must be 'module:callable'")
    value: Any = importlib.import_module(module_name)
    for part in attr_path.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"factory is not callable: {path}")
    return value


def _handler_from(value: Any) -> Callable[[dict[str, Any]], Any]:
    if callable(value):
        return value
    infer = getattr(value, "infer", None)
    if callable(infer):
        return infer
    raise TypeError("factory must return a callable or an object with infer(payload)")


def _frame(payload: Mapping[str, Any], max_frame_bytes: int = MAX_FRAME_BYTES) -> bytes:
    body = pickle.dumps(dict(payload), protocol=5)
    if not body or len(body) > max_frame_bytes:
        raise FrameTooLarge(f"frame size {len(body)} exceeds bound {max_frame_bytes}")
    return len(body).to_bytes(8, "big") + body


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise DaemonDisconnected("connection closed before frame completed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(
    connection: socket.socket,
    max_frame_bytes: int = MAX_FRAME_BYTES,
) -> dict[str, Any]:
    header = _receive_exact(connection, 8)
    size = int.from_bytes(header, "big")
    if size <= 0 or size > max_frame_bytes:
        raise FrameTooLarge(f"frame size {size} exceeds bound {max_frame_bytes}")
    body = _receive_exact(connection, size)
    try:
        value = pickle.loads(body)
    except Exception as error:
        # Reconstruction can raise arbitrary ordinary exceptions from reducers,
        # not only PickleError. Keep transport reads separate from decoding.
        raise ProtocolError("invalid pickle frame") from error
    if not isinstance(value, dict):
        raise ProtocolError("frame must contain a JSON object")
    return value


def _send_frame(
    connection: socket.socket,
    payload: Mapping[str, Any],
    max_frame_bytes: int = MAX_FRAME_BYTES,
) -> None:
    connection.sendall(_frame(payload, max_frame_bytes))


def control_request(
    socket_path: str | Path,
    identity: str,
    command: str = "status",
    timeout: float = 10,
    max_frame_bytes: int = MAX_FRAME_BYTES,
    drain_timeout: float | None = None,
) -> dict[str, Any]:
    """Send one management request without loading or owning a model child."""
    if command not in {"status", "stop"}:
        raise ValueError("control command must be 'status' or 'stop'")
    if timeout <= 0:
        raise ValueError("control timeout must be positive")
    if max_frame_bytes <= 0:
        raise ValueError("max_frame_bytes must be positive")
    if drain_timeout is not None and (not math.isfinite(drain_timeout) or drain_timeout <= 0):
        raise ValueError("drain_timeout must be finite and positive")
    endpoint = Path(socket_path).expanduser().resolve()
    request_id = uuid.uuid4().hex
    message = {
        "protocol": PROTOCOL,
        "identity": identity,
        "request_id": request_id,
        "generation": "",
        "type": command,
    }
    if drain_timeout is not None:
        message["drain_timeout"] = drain_timeout
    frame = _frame(message, max_frame_bytes)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sent = False
    try:
        _validate_endpoint(endpoint)
        connection.settimeout(timeout)
        connection.connect(str(endpoint))
        if _peer_uid(connection) != os.getuid():
            raise PermissionError("daemon endpoint requires same UID")
        sent = True  # sendall may fail after transmitting part of the frame
        connection.sendall(frame)
        response = _read_frame(connection, max_frame_bytes)
    except (socket.timeout, TimeoutError) as error:
        if sent:
            raise RequestOutcomeUnknown(
                f"control request timed out after send; outcome unknown; command={command}",
                request_id=request_id,
            ) from error
        raise ModelDaemonError("control connection timed out") from error
    except (OSError, DaemonDisconnected, ProtocolError, FrameTooLarge) as error:
        if sent:
            raise RequestOutcomeUnknown(
                f"control request lost its response; outcome unknown; not replayed; command={command}",
                request_id=request_id,
            ) from error
        if isinstance(error, (ConnectionError, FileNotFoundError, DaemonDisconnected)):
            raise DaemonDisconnected("daemon disconnected before control request send") from error
        raise
    finally:
        connection.close()
    if response.get("protocol") != PROTOCOL:
        raise ProtocolError("daemon protocol mismatch")
    if response.get("request_id") != request_id:
        raise ProtocolError("daemon response request id mismatch")
    if response.get("identity") != identity:
        raise IdentityMismatch("daemon identity handshake failed")
    if response.get("type") == "error":
        if response.get("code") == "identity_mismatch":
            raise IdentityMismatch("daemon identity handshake failed")
        raise DaemonRequestFailed(
            str(response.get("message") or response.get("code") or "control request failed"),
            remote_type=str(response.get("error_type") or response.get("code") or "") or None,
            code=response.get("code"),
        )
    return response


def _peer_uid(connection: socket.socket) -> int:
    if not hasattr(socket, "SO_PEERCRED"):
        return os.getuid()
    credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    _, uid, _ = __import__("struct").unpack("3i", credentials)
    return int(uid)


def _validate_endpoint(path: Path) -> None:
    parent = path.parent.stat()
    endpoint = path.stat()
    if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) != 0o700:
        raise PermissionError(f"daemon directory must be owner-only 0700: {path.parent}")
    if endpoint.st_uid != os.getuid() or stat.S_IMODE(endpoint.st_mode) != 0o600:
        raise PermissionError(f"daemon socket must be owner-only 0600: {path}")
    if not stat.S_ISSOCK(endpoint.st_mode):
        raise ProtocolError(f"daemon endpoint is not a socket: {path}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _event(event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "protocol": PROTOCOL,
        "pid": os.getpid(),
        "time": time.time(),
        **fields,
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    sys.stdout.flush()


class _Server:
    def __init__(
        self,
        handler: Callable[[dict[str, Any]], Any],
        *,
        socket_path: Path,
        identity: str,
        backend: str,
        max_pending: int,
        max_frame_bytes: int,
        parent_pid: int | None,
    ) -> None:
        self.handler = handler
        self.socket_path = socket_path
        self.identity = identity
        self.backend = backend
        self.max_pending = max_pending
        self.max_frame_bytes = max_frame_bytes
        self.parent_pid = parent_pid
        self.started_at = time.time()
        self.requests = 0
        self.completed = 0
        self.failed = 0
        self.active_request: str | None = None
        self.tasks: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_pending)
        self.admission = threading.Lock()
        self.reply_condition = threading.Condition(self.admission)
        self.reply_connections: set[socket.socket] = set()
        self.reply_sent_count = 0
        self.reply_failed_count = 0
        self.reply_timeout_count = 0
        self.drain_result: dict[str, Any] | None = None
        self.stopping = False
        self.listener: socket.socket | None = None
        self.shutdown_complete = threading.Event()
        self.worker = threading.Thread(target=self._worker, name="model-daemon-inference")

    def _status(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "type": "status",
            "backend": self.backend,
            "identity": self.identity,
            "pid": os.getpid(),
            "parent_pid": self.parent_pid,
            "state": "draining" if self.stopping else "running",
            "queue_depth": self.tasks.qsize(),
            "active_request": self.active_request,
            "max_pending": self.max_pending,
            "request_count": self.requests,
            "completed_count": self.completed,
            "failed_count": self.failed,
            "model_load_count": 1,
            "pending_reply_count": len(self.reply_connections),
            "reply_sent_count": self.reply_sent_count,
            "reply_failed_count": self.reply_failed_count,
            "started_at": self.started_at,
            "socket": str(self.socket_path),
        }

    def _worker(self) -> None:
        while True:
            item = self.tasks.get()
            try:
                if item.get("stop"):
                    return
                request_id = item["request_id"]
                generation = item["generation"]
                self.active_request = request_id
                _event("inference_start", backend=self.backend, request_id=request_id)
                try:
                    result = self.handler(dict(item["payload"]))
                    response = {"type": "result", "result": result}
                    self.completed += 1
                except BaseException as error:
                    self.failed += 1
                    response = {
                        "type": "error",
                        "code": "inference_failed",
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                    print(traceback.format_exc(), file=sys.stderr, flush=True)
                response = {
                    "protocol": PROTOCOL,
                    "request_id": request_id,
                    "identity": self.identity,
                    "generation": generation,
                    "backend": self.backend,
                    "pid": os.getpid(),
                    **response,
                }
                try:
                    _frame(response, self.max_frame_bytes)
                except Exception as error:
                    response.pop("result", None)
                    response.update({
                        "type": "error",
                        "code": "response_not_materializable",
                        "error_type": type(error).__name__,
                        "message": str(error)[:512],
                    })
                item["response"] = response
                _event(
                    "inference_end",
                    backend=self.backend,
                    request_id=request_id,
                    ok=response["type"] == "result",
                )
            finally:
                self.active_request = None
                item["done"].set()
                self.tasks.task_done()

    def _handle(self, connection: socket.socket) -> None:
        request_id: str | None = None
        generation: str | None = None
        admitted = False
        reply_sent = False
        try:
            _check_child_parent(self.parent_pid)
            if _peer_uid(connection) != os.getuid():
                raise PermissionError("daemon requires same UID")
            request = _read_frame(connection, self.max_frame_bytes)
            request_id = str(request.get("request_id") or uuid.uuid4().hex)
            generation = str(request.get("generation") or "")
            if request.get("protocol") != PROTOCOL:
                raise ProtocolError("daemon protocol mismatch")
            if request.get("identity") != self.identity:
                _send_frame(
                    connection,
                    {
                        "protocol": PROTOCOL,
                        "type": "error",
                        "code": "identity_mismatch",
                        "request_id": request_id,
                        "generation": generation,
                        "identity": self.identity,
                        "backend": self.backend,
                        "pid": os.getpid(),
                    },
                    self.max_frame_bytes,
                )
                return
            request_type = request.get("type")
            if request_type == "status":
                _send_frame(
                    connection,
                    {
                        **self._status(),
                        "request_id": request_id,
                        "generation": generation,
                    },
                    self.max_frame_bytes,
                )
                return
            if request_type in {"stop", "shutdown"}:
                budget = float(request.get("drain_timeout", DEFAULT_DRAIN_TIMEOUT))
                if not math.isfinite(budget) or budget <= 0:
                    raise ValueError("drain_timeout must be finite and positive")
                self._drain(connection, request_id, generation, budget)
                return
            if request_type != "infer":
                raise ProtocolError(f"unknown request type: {request_type}")

            payload = request.get("payload")
            if not isinstance(payload, dict):
                raise ProtocolError("infer payload must be a JSON object")
            item = {
                "request_id": request_id,
                "generation": generation,
                "payload": payload,
                "done": threading.Event(),
                "response": None,
            }
            with self.admission:
                if self.stopping:
                    raise ProtocolError("daemon is stopping")
                self.requests += 1
                try:
                    self.tasks.put_nowait(item)
                except queue.Full as error:
                    raise ModelDaemonError("daemon pending queue is full") from error
                # Register while admission is locked so stop cannot miss a reply
                # whose computation has already completed in the worker.
                self.reply_connections.add(connection)
                admitted = True
            item["done"].wait()
            _send_frame(connection, item["response"], self.max_frame_bytes)
            reply_sent = True
        except (BrokenPipeError, ConnectionResetError, DaemonDisconnected):
            return
        except BaseException as error:
            if admitted:
                # A partial response must never be followed by a second frame.
                _event("reply_send_failed", request_id=request_id,
                       generation=generation, error_type=type(error).__name__)
                return
            try:
                _send_frame(
                    connection,
                    {
                        "protocol": PROTOCOL,
                        "type": "error",
                        "code": "protocol_error",
                        "request_id": request_id,
                        "generation": generation,
                        "identity": self.identity,
                        "backend": self.backend,
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "pid": os.getpid(),
                    },
                    self.max_frame_bytes,
                )
            except OSError:
                pass
        finally:
            try:
                connection.close()
            finally:
                if admitted:
                    with self.reply_condition:
                        self.reply_connections.remove(connection)
                        if reply_sent:
                            self.reply_sent_count += 1
                        else:
                            self.reply_failed_count += 1
                        self.reply_condition.notify_all()

    def _drain(self, connection: socket.socket, request_id: str, generation: str,
               drain_timeout: float = DEFAULT_DRAIN_TIMEOUT) -> None:
        deadline = time.monotonic() + drain_timeout
        with self.admission:
            first_stop = not self.stopping
            self.stopping = True
        _event("shutdown_received", backend=self.backend, request_id=request_id)
        try:
            if first_stop:
                if self.listener is not None:
                    try:
                        # close() in another thread need not interrupt accept().
                        self.listener.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    self.listener.close()
                self.tasks.join()
                self.tasks.put({"stop": True, "done": threading.Event()})
                self.worker.join()
                with self.reply_condition:
                    while self.reply_connections:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self.reply_condition.wait(remaining)
                    expired = tuple(self.reply_connections)
                    self.reply_timeout_count += len(expired)
                for pending in expired:
                    try:
                        pending.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                with self.reply_condition:
                    # shutdown interrupts blocked sends; wait until their handlers
                    # have closed sockets and recorded terminal send failures.
                    while self.reply_connections:
                        self.reply_condition.wait()
                    self.drain_result = {
                        "drained": self.reply_failed_count == 0 and self.reply_timeout_count == 0,
                        "compute_drained": True,
                        "replies_settled": True,
                        "completed_count": self.completed,
                        "failed_count": self.failed,
                        "reply_sent_count": self.reply_sent_count,
                        "reply_failed_count": self.reply_failed_count,
                        "reply_timeout_count": self.reply_timeout_count,
                    }
                _event("drain_complete", backend=self.backend, **self.drain_result)
            else:
                # A second accepted stop must not enqueue another worker sentinel.
                self.shutdown_complete.wait()
            _send_frame(
                connection,
                {
                    "protocol": PROTOCOL,
                    "type": "result",
                    "request_id": request_id,
                    "generation": generation,
                    "identity": self.identity,
                    "backend": self.backend,
                    "pid": os.getpid(),
                    "result": self.drain_result,
                },
                self.max_frame_bytes,
            )
        finally:
            if first_stop:
                # Completion belongs to the server, even if the requester left.
                self.shutdown_complete.set()

    def run(self) -> None:
        parent = self.socket_path.parent
        parent.mkdir(mode=0o700, parents=False, exist_ok=True)
        parent.chmod(0o700)
        if parent.stat().st_uid != os.getuid():
            raise PermissionError(f"daemon directory has wrong owner: {parent}")
        if self.socket_path.exists():
            raise FileExistsError(f"daemon socket already exists: {self.socket_path}")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener = listener
        listener.bind(str(self.socket_path))
        self.socket_path.chmod(0o600)
        listener.listen(max(16, self.max_pending * 4))
        self.worker.start()
        _event(
            "service_ready",
            backend=self.backend,
            identity=self.identity,
            socket=str(self.socket_path),
            max_pending=self.max_pending,
            model_load_count=1,
        )
        try:
            while not self.stopping:
                try:
                    connection, _ = listener.accept()
                except OSError:
                    if self.stopping:
                        break
                    raise
                threading.Thread(
                    target=self._handle,
                    args=(connection,),
                    name="model-daemon-request",
                    daemon=True,
                ).start()
        finally:
            try:
                listener.close()
            except OSError:
                pass
            if self.stopping:
                self.shutdown_complete.wait()
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass
        _event("service_stopped", backend=self.backend, identity=self.identity)


def _set_parent_death_signal(parent_pid: int | None) -> None:
    if parent_pid is None or not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    except Exception as error:
        raise RuntimeError("Linux parent-death protection is unavailable") from error
    if os.getppid() != parent_pid:
        raise RuntimeError("daemon parent ended during startup")


def _check_child_parent(parent_pid: int | None) -> None:
    if parent_pid is not None and os.getppid() != parent_pid:
        raise RuntimeError("daemon parent is no longer alive")


def serve(
    factory: Callable[[dict[str, Any]], Any],
    settings: dict[str, Any],
    *,
    socket_path: str | Path,
    identity: str = "local",
    backend: str = "local",
    max_pending: int = 4,
    max_frame_bytes: int = MAX_FRAME_BYTES,
    parent_pid: int | None = None,
) -> None:
    if max_pending < 1:
        raise ValueError("max_pending must be positive")
    if max_frame_bytes <= 0:
        raise ValueError("max_frame_bytes must be positive")
    _event("model_load_start", backend=backend)
    handler = _handler_from(factory(settings))
    _event("model_load_end", backend=backend, model_load_count=1)
    _Server(
        handler,
        socket_path=Path(socket_path),
        identity=identity,
        backend=backend,
        max_pending=max_pending,
        max_frame_bytes=max_frame_bytes,
        parent_pid=parent_pid,
    ).run()


class OwnedModelDaemon:
    """Lazy same-process owner, or identity-checked attachment to an existing owner.

    Closing an attachment releases only that client. An explicit request_stop()
    stops the service. Inherited clients are unusable after fork; a child may
    construct its own attachment. Provider execution is intentionally unsupported.
    """
    def __init__(
        self,
        spec: DaemonSpec,
        *,
        startup_timeout: float = 300,
        request_timeout: float | None = None,
        shutdown_timeout: float = 30,
        max_pending: int = 4,
        max_frame_bytes: int | None = None,
        socket_path: str | Path | None = None,
    ) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        for name, value in (
            ("startup_timeout", startup_timeout),
            ("shutdown_timeout", shutdown_timeout),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if request_timeout is not None and request_timeout <= 0:
            raise ValueError("request_timeout must be positive when set")
        if max_frame_bytes is not None and max_frame_bytes <= 0:
            raise ValueError("max_frame_bytes must be positive")
        self.spec = _prepare_provider_spec(spec)
        self.spec = replace(self.spec, settings=deepcopy(self.spec.settings),
                            env=dict(self.spec.env) if self.spec.env is not None else None)
        if max_frame_bytes is not None:
            self.spec = replace(self.spec, max_frame_bytes=int(max_frame_bytes))
        if self.spec.max_frame_bytes <= 0:
            raise ValueError("max_frame_bytes must be positive")
        self._creator_pid = os.getpid()
        self.startup_timeout = float(startup_timeout)
        self.request_timeout = request_timeout
        self.shutdown_timeout = float(shutdown_timeout)
        self.max_pending = int(max_pending)
        self.max_frame_bytes = int(
            self.spec.max_frame_bytes if max_frame_bytes is None else max_frame_bytes
        )
        self.identity = model_identity(self.spec, max_pending=self.max_pending)
        self._attached = socket_path is not None
        self._socket_path = Path(socket_path).expanduser().resolve() if socket_path else None
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_handle = None
        self._start_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._remote_pid: int | None = None
        self._last_drain_result: dict[str, Any] | None = None
        self.generation = uuid.uuid4().hex
        self._closed = False
        self.temp_dir: Path | None = None
        self.log_path: Path | None = None
        if not self._attached:
            atexit.register(self._atexit)
            from .model_execution import register_model_owner

            register_model_owner(self)

    @property
    def pid(self) -> int | None:
        if self._proc is not None:
            return self._proc.pid
        return self._remote_pid

    @property
    def socket_path(self) -> Path | None:
        return self._socket_path

    @property
    def attached(self) -> bool:
        return self._attached

    def _child_env(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update({str(key): str(value) for key, value in (self.spec.env or {}).items()})
        inherited = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(_REPO_ROOT) if not inherited else f"{_REPO_ROOT}{os.pathsep}{inherited}"
        )
        environment["PYTHONUNBUFFERED"] = "1"
        environment["ASTROLABE_MODEL_PARENT_PID"] = str(os.getpid())
        return environment

    def _check_process(self) -> None:
        if os.getpid() != self._creator_pid:
            raise ModelDaemonError(
                "daemon client cannot be used after fork; create a new client in this process"
            )

    def _start(self) -> None:
        self._check_process()
        with self._start_lock:
            if self._closed:
                raise ModelDaemonError("daemon owner is closed")
            if self._attached:
                return
            if self._proc is not None and self._proc.poll() is None:
                return
            if self._proc is not None:
                raise DaemonChildExited(
                    f"{self.spec.backend} child exited with code {self._proc.returncode}; "
                    f"log: {self.log_path}"
                )
            self.temp_dir = Path(tempfile.mkdtemp(prefix="astrolabe-model-daemon-")).resolve()
            self.temp_dir.chmod(0o700)
            self._socket_path = self.temp_dir / "model.sock"
            self.log_path = self.temp_dir / "owner.jsonl"
            spec_path = self.temp_dir / "spec.json"
            _atomic_json(
                spec_path,
                {
                    "protocol": PROTOCOL,
                    "backend": self.spec.backend,
                    "factory": self.spec.factory,
                    "settings": self.spec.settings,
                    "python_bin": self.spec.python_bin,
                    "cwd": self.spec.cwd,
                    "env": _provider_environment_for_spec(self.spec),
                    "provider": self.spec.provider,
                    "identity": self.identity,
                    "generation": self.generation,
                    "max_pending": self.max_pending,
                    "max_frame_bytes": self.spec.max_frame_bytes,
                },
            )
            spec_path.chmod(0o600)
            self._log_handle = self.log_path.open("ab", buffering=0)
            command = [
                self.spec.python_bin,
                "-u",
                "-m",
                "astrolabe.model_daemon",
                "child",
                "--spec",
                str(spec_path),
                "--socket",
                str(self._socket_path),
            ]
            launched = threading.Event()
            launch_error: list[BaseException] = []
            environment = self._child_env()

            def launch_child() -> None:
                proc: subprocess.Popen[bytes] | None = None
                try:
                    proc = subprocess.Popen(
                        command,
                        cwd=self.spec.cwd or str(_REPO_ROOT),
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=self._log_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        close_fds=True,
                    )
                    self._proc = proc
                except BaseException as error:
                    launch_error.append(error)
                finally:
                    launched.set()
                if proc is not None:
                    proc.wait()

            self._launch_thread = threading.Thread(
                target=launch_child,
                name=f"model-daemon-owner-{self.spec.backend}",
                daemon=True,
            )
            self._launch_thread.start()
            if not launched.wait(self.startup_timeout):
                self._startup_failed("child process launch timed out")
            if launch_error:
                self._startup_failed(f"child process launch failed: {launch_error[0]}")
            assert self._proc is not None
            _atomic_json(
                self.temp_dir / "lifecycle.json",
                {
                    "status": "starting",
                    "pid": self._proc.pid,
                    "parent_pid": os.getpid(),
                    "socket": str(self._socket_path),
                    "identity": self.identity,
                    "command": command,
                    "started_at": time.time(),
                },
            )
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    self._startup_failed("child exited during startup")
                if self._socket_path.exists():
                    try:
                        status = self._exchange({"type": "status"}, timeout=0.25)
                    except (OSError, ModelDaemonError):
                        status = None
                    if status and status.get("state") == "running":
                        self._remote_pid = int(status["pid"])
                        _atomic_json(
                            self.temp_dir / "lifecycle.json",
                            {
                                "status": "ready",
                                "pid": self._proc.pid,
                                "parent_pid": os.getpid(),
                                "socket": str(self._socket_path),
                                "identity": self.identity,
                                "generation": self.generation,
                                "command": command,
                                "started_at": time.time(),
                            },
                        )
                        return
                time.sleep(0.05)
            self._startup_failed("child did not become ready before timeout")

    def _startup_failed(self, message: str) -> None:
        self._stop_process(graceful=False)
        raise DaemonStartupError(f"{message}; log: {self.log_path}")

    def _exchange(
        self,
        request: Mapping[str, Any],
        *,
        timeout: float | None,
    ) -> dict[str, Any]:
        if self._socket_path is None:
            raise DaemonStartupError("daemon socket is not available")
        request_id = str(request.get("request_id") or uuid.uuid4().hex)
        message = {
            "protocol": PROTOCOL,
            "identity": self.identity,
            "request_id": request_id,
            "generation": self.generation,
            **request,
        }
        frame = _frame(message, self.max_frame_bytes)
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sent = False
        try:
            _validate_endpoint(self._socket_path)
            connection.settimeout(timeout if timeout is not None else min(self.startup_timeout, 10.0))
            connection.connect(str(self._socket_path))
            if _peer_uid(connection) != os.getuid():
                raise PermissionError("daemon endpoint requires same UID")
            sent = True  # a partial write cannot be safely replayed
            connection.sendall(frame)
            connection.settimeout(timeout)
            response = _read_frame(connection, self.max_frame_bytes)
        except (socket.timeout, TimeoutError) as error:
            if sent:
                raise RequestOutcomeUnknown(
                    f"{self.spec.backend} request timed out after send; outcome unknown; not replayed",
                    request_id=request_id,
                    generation=self.generation,
                ) from error
            raise ModelDaemonError(f"{self.spec.backend} connection timed out") from error
        except (OSError, DaemonDisconnected, ProtocolError, FrameTooLarge) as error:
            if not sent and not isinstance(error, (ConnectionError, FileNotFoundError, DaemonDisconnected)):
                raise
            proc = self._proc
            if proc is not None:
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
            cause = error
            if proc is not None and proc.poll() is not None:
                cause = DaemonChildExited(
                    f"{self.spec.backend} child exited with code {proc.returncode}; log: {self.log_path}"
                )
            if not sent:
                if isinstance(cause, DaemonChildExited):
                    raise cause from error
                raise DaemonDisconnected(f"{self.spec.backend} daemon disconnected before send") from error
            raise RequestOutcomeUnknown(
                f"{self.spec.backend} response lost after send; outcome unknown; not replayed",
                request_id=request_id,
                generation=self.generation,
            ) from cause
        finally:
            connection.close()
        try:
            if response.get("protocol") != PROTOCOL:
                raise ProtocolError("daemon protocol mismatch")
            if response.get("request_id") != request_id:
                raise ProtocolError("daemon response request id mismatch")
            if response.get("generation") != self.generation:
                raise ProtocolError("daemon response generation mismatch")
            expected_type = "status" if request.get("type") == "status" else "result"
            if response.get("type") not in {expected_type, "error"}:
                raise ProtocolError("unexpected daemon response type")
        except ProtocolError as error:
            raise RequestOutcomeUnknown(
                f"{self.spec.backend} response cannot confirm request; outcome unknown; not replayed",
                request_id=request_id,
                generation=self.generation,
            ) from error
        if response.get("identity") != self.identity:
            raise IdentityMismatch("daemon identity handshake failed")
        if response.get("type") == "error":
            code = response.get("code")
            if code == "identity_mismatch":
                raise IdentityMismatch("daemon identity handshake failed")
            raise DaemonRequestFailed(
                str(response.get("message") or code or "daemon request failed"),
                remote_type=str(response.get("error_type") or code or "") or None,
                code=code,
            )
        if response.get("pid") is not None:
            self._remote_pid = int(response["pid"])
        return response

    def request(self, payload: Mapping[str, Any], *, request_id: str | None = None) -> Any:
        if not isinstance(payload, Mapping):
            raise TypeError("request payload must be a mapping")
        self._start()
        message = {"type": "infer", "payload": dict(payload)}
        if request_id is not None:
            message["request_id"] = request_id
        response = self._exchange(message, timeout=self.request_timeout)
        return response.get("result")

    def status(self) -> dict[str, Any]:
        self._start()
        return self._exchange(
            {"type": "status"},
            timeout=self.request_timeout if self.request_timeout is not None else min(self.startup_timeout, 10.0),
        )

    def _stop_process(self, *, graceful: bool) -> None:
        proc = self._proc
        if proc is None:
            return
        drain_error = None
        if graceful and proc.poll() is None:
            try:
                response = self._exchange({"type": "stop", "drain_timeout": self.shutdown_timeout},
                                          timeout=self.shutdown_timeout)
                self._last_drain_result = response.get("result")
            except DaemonChildExited:
                # The process exited before this stop could be sent (for example,
                # after an independent controller stopped it). Close idempotently;
                # an absent receipt remains None, never an inferred drain success.
                pass
            except (ModelDaemonError, OSError) as error:
                drain_error = error
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if self.temp_dir is not None:
            _atomic_json(
                self.temp_dir / "lifecycle.json",
                {
                    "status": "stopped",
                    "pid": proc.pid,
                    "parent_pid": os.getpid(),
                    "socket": str(self._socket_path) if self._socket_path else None,
                    "identity": self.identity,
                    "exit_code": proc.returncode,
                    "drain_result": self._last_drain_result,
                    "drain_error_type": type(drain_error).__name__ if drain_error else None,
                    "drain_request_id": getattr(drain_error, "request_id", None),
                    "drain_generation": getattr(drain_error, "generation", None),
                    "stopped_at": time.time(),
                },
            )
        self._proc = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        if drain_error is not None:
            # Reclaiming our process does not prove that the drain was successful.
            raise drain_error
        if graceful and self._last_drain_result is not None and not self._last_drain_result.get("drained"):
            raise DaemonDrainFailed(self._last_drain_result)

    def shutdown(self) -> None:
        self._check_process()
        with self._shutdown_lock:
            with self._start_lock:
                if self._closed:
                    return
                self._closed = True
                try:
                    if not self._attached:
                        self._stop_process(graceful=True)
                finally:
                    if not self._attached:
                        atexit.unregister(self._atexit)

    def request_stop(self) -> dict[str, Any]:
        self._start()
        response = self._exchange({"type": "stop", "drain_timeout": self.shutdown_timeout},
                                  timeout=self.shutdown_timeout)
        self._last_drain_result = response.get("result")
        if not self._attached:
            with self._shutdown_lock:
                self._stop_process(graceful=False)
                self._closed = True
                atexit.unregister(self._atexit)
        return response

    def _atexit(self) -> None:
        try:
            self.shutdown()
        except BaseException:
            pass

    def __enter__(self) -> "OwnedModelDaemon":
        return self

    def __exit__(self, exc_type, error, traceback) -> None:
        self.shutdown()


def _child_main(spec_path: Path, socket_path: Path) -> int:
    parent_pid_value = os.environ.get("ASTROLABE_MODEL_PARENT_PID")
    parent_pid = int(parent_pid_value) if parent_pid_value else None
    lifecycle_path = spec_path.with_name("lifecycle.json")
    try:
        _set_parent_death_signal(parent_pid)
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        if payload.get("protocol") != PROTOCOL:
            raise ValueError("spec protocol mismatch")
        spec = DaemonSpec(
            backend=str(payload["backend"]),
            factory=str(payload["factory"]),
            settings=dict(payload.get("settings") or {}),
            python_bin=str(payload.get("python_bin") or sys.executable),
            cwd=payload.get("cwd"),
            env=payload.get("env"),
            provider=payload.get("provider"),
            max_frame_bytes=int(payload.get("max_frame_bytes", MAX_FRAME_BYTES)),
        )
        _activate_child_provider(spec)
        factory = _load_factory(str(payload["factory"]))
        _atomic_json(
            lifecycle_path,
            {
                "status": "loading",
                "pid": os.getpid(),
                "parent_pid": parent_pid,
                "socket": str(socket_path),
                "identity": payload["identity"],
                "provider": spec.provider,
            },
        )
        serve(
            factory,
            dict(payload.get("settings") or {}),
            socket_path=socket_path,
            identity=str(payload["identity"]),
            backend=str(payload["backend"]),
            max_pending=int(payload.get("max_pending", 4)),
            max_frame_bytes=int(payload.get("max_frame_bytes", MAX_FRAME_BYTES)),
            parent_pid=parent_pid,
        )
        _atomic_json(
            lifecycle_path,
            {
                "status": "exited",
                "pid": os.getpid(),
                "parent_pid": parent_pid,
                "socket": str(socket_path),
                "identity": payload["identity"],
                "exit_code": 0,
            },
        )
        return 0
    except BaseException as error:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        try:
            _atomic_json(
                lifecycle_path,
                {
                    "status": "startup_failed" if not socket_path.exists() else "failed",
                    "pid": os.getpid(),
                    "parent_pid": parent_pid,
                    "socket": str(socket_path),
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
        except BaseException:
            pass
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Astrolabe persistent model daemon")
    commands = parser.add_subparsers(dest="command", required=True)
    child = commands.add_parser("child")
    child.add_argument("--spec", required=True, type=Path)
    child.add_argument("--socket", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "child":
        return _child_main(arguments.spec, arguments.socket)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
