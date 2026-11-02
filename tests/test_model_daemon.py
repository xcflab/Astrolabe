from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import queue
import threading
import time
import sys

import numpy as np
import pytest

from test_model_execution_fixture import cpu_only, daemon as daemon_module

from _astrolabe_foundation_tests.model_daemon import (
    control_request,
    DaemonChildExited,
    DaemonRequestFailed,
    DaemonSpec,
    DaemonStartupError,
    IdentityMismatch,
    ModelDaemonError,
    OwnedModelDaemon,
    RequestOutcomeUnknown,
    model_identity,
)
from _astrolabe_foundation_tests.model_execution import model_run_scope


ADAPTER_SOURCE = r'''
from pathlib import Path
import numpy as np
import os
import time

LOAD_COUNT = 0


def _increment(path):
    path = Path(path)
    value = int(path.read_text() or "0") if path.exists() else 0
    path.write_text(str(value + 1))


class DecodeFailure:
    def __reduce__(self):
        return int, ([],)


class Handler:
    def __init__(self, settings):
        global LOAD_COUNT
        LOAD_COUNT += 1
        self.settings = settings

    def infer(self, payload):
        action = payload.get("action")
        if action == "provider":
            return {
                "provider_request": self.settings.get("_provider_request"),
                "pid": os.getpid(),
            }
        if action == "snapshot":
            return {
                "load_count": LOAD_COUNT,
                "pid": os.getpid(),
                "value": payload.get("value"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        if action == "arrays":
            array = np.asarray(payload["array"])
            return {
                "array": array[:, ::-1].copy(),
                "dtype": str(array.dtype),
                "shape": array.shape,
            }
        if action == "fail":
            raise ValueError("adapter failure")
        if action == "wait":
            Path(payload["started"]).write_text("started")
            release = Path(payload["release"])
            deadline = time.monotonic() + 10
            while not release.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            if payload.get("large_bytes"):
                return b"x" * payload["large_bytes"]
            return {"released": release.exists()}
        if action == "slow_count":
            _increment(payload["counter"])
            time.sleep(0.5)
            return {"count": int(Path(payload["counter"]).read_text())}
        if action == "decode_failure":
            _increment(payload["counter"])
            return DecodeFailure()
        if action == "unpicklable":
            return lambda: None
        if action == "large":
            return "x" * payload["size"]
        if action == "large_bytes":
            return b"x" * payload["size"]
        if action == "exit":
            os._exit(17)
        raise ValueError(f"unknown action: {action}")


def make(settings):
    if settings.get("wait_for_blocked_accept"):
        import __main__, sys, threading
        original = __main__._Server._drain
        def observed_drain(server, *args):
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                frame = sys._current_frames()[threading.main_thread().ident]
                while frame is not None and frame.f_code.co_name != "accept":
                    frame = frame.f_back
                if frame is not None:
                    Path(settings["wait_for_blocked_accept"]).write_text("blocked in accept")
                    return original(server, *args)
                time.sleep(.001)
            raise RuntimeError("test could not force blocked accept")
        __main__._Server._drain = observed_drain
    return Handler(settings)


def broken(settings):
    raise RuntimeError("factory failed")
'''


def _spec(tmp_path: Path, **updates) -> DaemonSpec:
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir(exist_ok=True)
    (adapter_dir / "adapter.py").write_text(ADAPTER_SOURCE, encoding="utf-8")
    values = {
        "backend": "cpu-test",
        "factory": "adapter:make",
        "settings": {"variant": "a"},
        "python_bin": sys.executable,
        "env": {
            "PYTHONPATH": str(adapter_dir),
            "CUDA_VISIBLE_DEVICES": "",
        },
    }
    values.update(updates)
    return DaemonSpec(**values)


def _wait_for(path: Path, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def test_subprocess_loads_once_recovers_failure_and_shuts_down(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec.env["UNRELATED_CREDENTIAL"] = "do-not-persist"
    with OwnedModelDaemon(spec, startup_timeout=20) as daemon:
        first = daemon.request({"action": "snapshot", "value": 1})
        second = daemon.request({"action": "snapshot", "value": 2})
        assert first["load_count"] == second["load_count"] == 1
        assert first["pid"] == second["pid"] == daemon.pid
        assert first["cuda_visible_devices"] == ""
        array = np.arange(12, dtype=np.float32).reshape(3, 4)
        array_result = daemon.request({"action": "arrays", "array": array})
        assert array_result["array"].dtype == np.float32
        assert array_result["array"].shape == (3, 4)
        assert np.array_equal(array_result["array"], array[:, ::-1])

        with pytest.raises(DaemonRequestFailed, match="adapter failure"):
            daemon.request({"action": "fail"})
        recovered = daemon.request({"action": "snapshot", "value": 3})
        assert recovered["value"] == 3
        status = daemon.status()
        assert status["state"] == "running"
        assert status["model_load_count"] == 1
        assert status["request_count"] == 5

    assert daemon.log_path is not None and daemon.log_path.exists()
    lifecycle = json.loads((daemon.temp_dir / "lifecycle.json").read_text())
    assert lifecycle["status"] == "stopped"
    persisted = "\n".join(
        path.read_text(errors="replace")
        for path in (daemon.temp_dir / "spec.json", daemon.log_path)
    )
    assert "do-not-persist" not in persisted


def test_attach_is_identity_checked_and_never_stops_owner(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    owner = OwnedModelDaemon(spec, startup_timeout=20)
    attached = None
    try:
        owner.request({"action": "snapshot"})
        socket_path = owner.status()["socket"]
        attached = OwnedModelDaemon(spec, socket_path=socket_path, startup_timeout=20)
        assert attached.request({"action": "snapshot"})["pid"] == owner.pid
        attached.shutdown()
        assert owner.status()["state"] == "running"

        mismatched = OwnedModelDaemon(
            replace(spec, settings={"variant": "b"}),
            socket_path=socket_path,
            startup_timeout=20,
        )
        with pytest.raises(IdentityMismatch):
            mismatched.request({"action": "snapshot"})
        assert owner.status()["state"] == "running"
        lexical_python = replace(spec, python_bin="python3")
        assert model_identity(lexical_python, max_pending=4) != model_identity(spec, max_pending=4)
    finally:
        if attached is not None:
            attached.shutdown()
        owner.shutdown()


def test_attached_client_can_explicitly_stop_owner(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    owner = OwnedModelDaemon(spec, startup_timeout=20)
    try:
        owner.request({"action": "snapshot"})
        attached = OwnedModelDaemon(spec, socket_path=owner.socket_path)
        assert control_request(owner.socket_path, owner.identity)["state"] == "running"
        response = attached.request_stop()
        assert response["result"]["drained"] is True
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                owner.status()
            except DaemonChildExited:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("owner child remained alive after stop acknowledgment")
        attached.shutdown()
    finally:
        owner.shutdown()


def test_control_request_checks_identity_and_can_stop(tmp_path: Path) -> None:
    owner = OwnedModelDaemon(_spec(tmp_path), startup_timeout=20)
    try:
        owner.request({"action": "snapshot"})
        with pytest.raises(IdentityMismatch):
            control_request(owner.socket_path, "not-the-owner")
        response = control_request(owner.socket_path, owner.identity, "stop")
        assert response["result"]["drained"] is True
    finally:
        owner.shutdown()


def test_status_works_while_busy_and_queue_is_bounded(tmp_path: Path) -> None:
    started = tmp_path / "started"
    release = tmp_path / "release"
    daemon = OwnedModelDaemon(_spec(tmp_path), startup_timeout=20, max_pending=1)
    outcomes: queue.Queue = queue.Queue()

    def wait_request() -> None:
        try:
            value = daemon.request(
                {"action": "wait", "started": str(started), "release": str(release)}
            )
            outcomes.put(("result", value))
        except BaseException as error:
            outcomes.put(("error", repr(error)))

    first = threading.Thread(target=wait_request)
    second = threading.Thread(target=wait_request)
    try:
        first.start()
        _wait_for(started)
        second.start()
        time.sleep(0.1)
        with pytest.raises(DaemonRequestFailed, match="pending queue is full"):
            daemon.request(
                {"action": "wait", "started": str(started), "release": str(release)}
            )
        assert daemon.status()["state"] == "running"
        release.write_text("go")
        first.join(timeout=10)
        second.join(timeout=10)
        assert not first.is_alive() and not second.is_alive()
        observed = [outcomes.get_nowait() for _ in range(outcomes.qsize())]
        log = daemon.log_path.read_text(errors="replace") if daemon.log_path else ""
        lifecycle = (
            json.loads((daemon.temp_dir / "lifecycle.json").read_text())
            if daemon.temp_dir is not None
            else {}
        )
        assert observed == [("result", {"released": True}), ("result", {"released": True})], (
            observed,
            log,
            {"test_pid": os.getpid(), "daemon_pid": daemon.pid, "lifecycle": lifecycle},
        )
        assert daemon.request({"action": "snapshot"})["load_count"] == 1
    finally:
        release.write_text("go")
        daemon.shutdown()


def test_timeout_is_outcome_unknown_without_replay(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    daemon = OwnedModelDaemon(
        _spec(tmp_path),
        startup_timeout=20,
        request_timeout=0.15,
    )
    try:
        with pytest.raises(RequestOutcomeUnknown, match="outcome unknown") as caught:
            daemon.request({"action": "slow_count", "counter": str(counter)})
        assert caught.value.request_id
        assert caught.value.generation == daemon.generation
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not counter.exists():
            time.sleep(0.01)
        time.sleep(0.6)
        assert counter.read_text() == "1"
        daemon.request_timeout = 2
        snapshot = daemon.request({"action": "snapshot"})
        assert snapshot["load_count"] == 1
        assert counter.read_text() == "1"
    finally:
        with pytest.raises(daemon_module.DaemonDrainFailed) as closed:
            daemon.shutdown()
        assert closed.value.result["reply_failed_count"] == 1
        lifecycle = json.loads((daemon.temp_dir / "lifecycle.json").read_text())
        assert lifecycle["drain_result"]["drained"] is False


def test_shutdown_stops_admission_and_drains_active_request(tmp_path: Path) -> None:
    started = tmp_path / "drain-started"
    release = tmp_path / "drain-release"
    daemon = OwnedModelDaemon(_spec(tmp_path), startup_timeout=20)
    outcome: queue.Queue = queue.Queue()

    def request() -> None:
        try:
            outcome.put(daemon.request({"action": "wait", "started": str(started), "release": str(release)}))
        except BaseException as error:
            outcome.put(error)

    request_thread = threading.Thread(target=request)
    request_thread.start()
    _wait_for(started)
    shutdown_thread = threading.Thread(target=daemon.shutdown)
    shutdown_thread.start()
    time.sleep(0.15)
    assert shutdown_thread.is_alive()
    release.write_text("go")
    request_thread.join(timeout=10)
    shutdown_thread.join(timeout=10)
    assert not request_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert outcome.get_nowait() == {"released": True}


def test_startup_failure_is_explicit(tmp_path: Path) -> None:
    spec = _spec(tmp_path, factory="adapter:broken")
    daemon = OwnedModelDaemon(spec, startup_timeout=5)
    with pytest.raises(DaemonStartupError, match="log:"):
        daemon.request({"action": "snapshot"})
    assert daemon.log_path is not None and daemon.log_path.exists()
    daemon.shutdown()


def test_child_death_is_explicit(tmp_path: Path) -> None:
    daemon = OwnedModelDaemon(_spec(tmp_path), startup_timeout=20)
    try:
        with pytest.raises(RequestOutcomeUnknown) as caught:
            daemon.request({"action": "exit"})
        assert isinstance(caught.value.__cause__, DaemonChildExited)
        with pytest.raises(DaemonChildExited):
            daemon.request({"action": "snapshot"})
    finally:
        daemon.shutdown()


def test_run_scope_stops_owned_children_and_detaches_clients(tmp_path: Path) -> None:
    outer = OwnedModelDaemon(_spec(tmp_path), startup_timeout=20)
    try:
        outer.request({"action": "snapshot"})
        with model_run_scope():
            inner = OwnedModelDaemon(_spec(tmp_path), startup_timeout=20)
            inner.request({"action": "snapshot"})
            attached = OwnedModelDaemon(
                outer.spec,
                socket_path=outer.socket_path,
                startup_timeout=20,
            )
            attached.request({"action": "snapshot"})
        with pytest.raises(ModelDaemonError, match="closed"):
            inner.status()
        assert outer.status()["state"] == "running"
        attached.shutdown()
        assert outer.status()["state"] == "running"
    finally:
        outer.shutdown()



@pytest.mark.parametrize("selection", ["spec", "empty_spec", "env", "inherited_env"])
def test_provider_fails_before_side_effects(tmp_path, monkeypatch, selection):
    spec = _spec(tmp_path)
    if selection == "spec":
        spec.provider = {"provider": "native"}
    elif selection == "empty_spec":
        spec.provider = {}
    elif selection == "env":
        spec.env["ASTROLABE_MODEL_PROVIDERS"] = '{"cpu-test": "native"}'
    else:
        monkeypatch.setenv("ASTROLABE_MODEL_PROVIDERS", "native")
    def forbidden(*args, **kwargs):
        pytest.fail("provider rejection must precede identity, spawn and sockets")
    monkeypatch.setattr(daemon_module, "model_identity", forbidden)
    monkeypatch.setattr(daemon_module.subprocess, "Popen", forbidden)
    with pytest.raises(ModelDaemonError, match="[Pp]rovider.*not supported"):
        OwnedModelDaemon(spec)


def test_attached_close_is_terminal_and_preserves_owner(tmp_path):
    with OwnedModelDaemon(_spec(tmp_path), startup_timeout=20) as owner:
        owner.status()
        attached = OwnedModelDaemon(owner.spec, socket_path=owner.socket_path)
        attached.status()
        attached.shutdown()
        attached.shutdown()
        for action in (attached.status, attached.request_stop,
                       lambda: attached.request({"action": "snapshot"})):
            with pytest.raises(ModelDaemonError, match="closed"):
                action()
        assert owner.status()["state"] == "running"


def test_frame_override_is_part_of_identity(tmp_path):
    spec = _spec(tmp_path)
    small = OwnedModelDaemon(spec, max_frame_bytes=4096)
    large = OwnedModelDaemon(spec, max_frame_bytes=8192)
    try:
        assert small.identity != large.identity
        assert small.spec.max_frame_bytes == 4096
        assert large.spec.max_frame_bytes == 8192
    finally:
        small.shutdown()
        large.shutdown()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_fork_cannot_use_or_stop_inherited_owner(tmp_path):
    # No child imports, models or GPU; each operation must reject before a lock/socket.
    with OwnedModelDaemon(_spec(tmp_path), startup_timeout=20) as owner:
        owner.status()
        attached = OwnedModelDaemon(owner.spec, socket_path=owner.socket_path)
        attached.status()
        locks = [lock for client in (owner, attached)
                 for lock in (client._start_lock, client._shutdown_lock)]
        for lock in locks:
            lock.acquire()
        read_fd, write_fd = os.pipe()
        child = os.fork()
        if child == 0:
            os.close(read_fd)
            try:
                outcomes = []
                for client in (owner, attached):
                    for action in (client.status, client.shutdown, client.request_stop,
                                   lambda client=client: client.request({"action": "snapshot"})):
                        try:
                            action()
                        except ModelDaemonError as error:
                            outcomes.append("fork" in str(error).lower())
                        else:
                            outcomes.append(False)
                with OwnedModelDaemon(owner.spec, socket_path=owner.socket_path) as fresh:
                    outcomes.append(fresh.status()["state"] == "running")
                os.write(write_fd, json.dumps(outcomes).encode())
            finally:
                os._exit(0)
        for lock in locks:
            lock.release()
        os.close(write_fd)
        try:
            import select
            assert select.select([read_fd], [], [], 10)[0], "fork guard blocked"
            result = json.loads(os.read(read_fd, 4096))
            assert result == [True] * 9
        finally:
            os.close(read_fd)
            # This is exclusively the child just created by this test.
            finished, _ = os.waitpid(child, os.WNOHANG)
            if not finished:
                import signal
                os.kill(child, signal.SIGKILL)
                os.waitpid(child, 0)
            attached.shutdown()
        assert owner.status()["state"] == "running"


@pytest.mark.parametrize("failure", ["eof", "partial_write", "timeout", "bad_frame", "oversize"])
def test_disconnect_after_send_is_unknown_without_replay(tmp_path, monkeypatch, failure):
    owner = OwnedModelDaemon(_spec(tmp_path), socket_path=tmp_path / "unused.sock")
    sent = []
    class Connection:
        def settimeout(self, value):
            pass
        def connect(self, path):
            pass
        def sendall(self, frame):
            sent.append(frame)
            if failure == "partial_write":
                raise BrokenPipeError("partial send")
        def recv(self, size):
            if failure == "timeout":
                raise TimeoutError("response timeout")
            if failure == "bad_frame":
                return (1).to_bytes(8, "big") if size == 8 else b"!"
            if failure == "oversize":
                return (owner.max_frame_bytes + 1).to_bytes(8, "big")
            return b""
        def close(self):
            pass
    monkeypatch.setattr(daemon_module, "_validate_endpoint", lambda path: None)
    monkeypatch.setattr(daemon_module, "_peer_uid", lambda connection: os.getuid())
    monkeypatch.setattr(daemon_module.socket, "socket", lambda *args: Connection())
    with pytest.raises(RequestOutcomeUnknown) as caught:
        owner.request({"action": "snapshot"})
    assert len(sent) == 1
    assert caught.value.request_id
    assert caught.value.generation == owner.generation
    owner.shutdown()


@pytest.mark.parametrize("action", ["unpicklable", "large"])
def test_response_serialization_failure_keeps_worker_alive(tmp_path, action):
    with OwnedModelDaemon(_spec(tmp_path), max_frame_bytes=4096,
                          request_timeout=3, startup_timeout=20) as owner:
        with pytest.raises(DaemonRequestFailed) as caught:
            owner.request({"action": action, "size": 4000})
        assert caught.value.remote_type in {"AttributeError", "FrameTooLarge"}
        assert owner.request({"action": "snapshot"})["load_count"] == 1


def test_identity_snapshots_settings_and_environment(tmp_path):
    spec = _spec(tmp_path, settings={"nested": {"value": 1}})
    with OwnedModelDaemon(spec) as owner:
        identity = owner.identity
        spec.settings["nested"]["value"] = 2
        spec.env["PYTHONPATH"] = "/unused"
        assert owner.spec.settings == {"nested": {"value": 1}}
        assert model_identity(owner.spec) == identity


@pytest.mark.parametrize("field,value", [
    ("backend", "other"), ("factory", "adapter:broken"),
    ("settings", {"variant": "b"}), ("env", {"EXAMPLE": "1"}),
    ("python_bin", "/different/python"), ("cwd", "/tmp"),
    ("max_frame_bytes", 8192),
])
def test_identity_changes_with_execution_inputs(tmp_path, field, value):
    spec = _spec(tmp_path)
    assert model_identity(replace(spec, **{field: value})) != model_identity(spec)


def test_identity_excludes_initial_rng_but_includes_source_and_queue(tmp_path):
    spec = _spec(tmp_path)
    original = model_identity(spec)
    seeded = replace(spec, settings={**spec.settings, "_initial_random_state": {"python": [1]}})
    assert model_identity(seeded) == original
    assert model_identity(spec, max_pending=1) != original
    adapter = Path(spec.env["PYTHONPATH"]) / "adapter.py"
    adapter.write_text(adapter.read_text() + "\n# distinct adapter revision\n")
    assert model_identity(spec) != original


def test_close_before_start_never_launches(tmp_path, monkeypatch):
    owner = OwnedModelDaemon(_spec(tmp_path))
    owner.shutdown()
    def forbidden(*args, **kwargs):
        pytest.fail("closed owner launched a child")
    monkeypatch.setattr(daemon_module.subprocess, "Popen", forbidden)
    for action in (owner.status, owner.request_stop,
                   lambda: owner.request({"action": "snapshot"})):
        with pytest.raises(ModelDaemonError, match="closed"):
            action()
    assert owner.temp_dir is None


@pytest.mark.parametrize("case", ["directory", "socket", "file"])
def test_endpoint_permissions_and_type_are_checked_in_temporary_directory(tmp_path, case):
    import socket
    secure = Path(daemon_module.tempfile.tempdir) / "secure"
    secure.mkdir(mode=0o700)
    endpoint = secure / "model.sock"
    if case == "file":
        endpoint.write_text("")
        endpoint.chmod(0o600)
        with pytest.raises(daemon_module.ProtocolError, match="not a socket"):
            daemon_module._validate_endpoint(endpoint)
        return
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.bind(str(endpoint))
        endpoint.chmod(0o600)
        daemon_module._validate_endpoint(endpoint)
        if case == "directory":
            secure.chmod(0o755)
        else:
            endpoint.chmod(0o644)
        with pytest.raises(PermissionError, match="owner-only"):
            daemon_module._validate_endpoint(endpoint)


@pytest.mark.parametrize("wire,error", [
    (b"\0" * 8, daemon_module.FrameTooLarge),
    ((4097).to_bytes(8, "big"), daemon_module.FrameTooLarge),
    ((1).to_bytes(8, "big") + b"!", daemon_module.ProtocolError),
    ((2).to_bytes(8, "big") + b"x", daemon_module.DaemonDisconnected),
])
def test_transport_rejects_invalid_or_incomplete_frames(wire, error):
    import socket
    left, right = socket.socketpair()
    with left, right:
        left.sendall(wire)
        left.shutdown(socket.SHUT_WR)
        with pytest.raises(error):
            daemon_module._read_frame(right, 4096)


def test_oversized_request_is_rejected_before_socket_or_send(tmp_path, monkeypatch):
    owner = OwnedModelDaemon(_spec(tmp_path), socket_path=tmp_path / "unused.sock",
                             max_frame_bytes=4096)
    def forbidden(*args):
        pytest.fail("oversized request reached socket creation")
    monkeypatch.setattr(daemon_module.socket, "socket", forbidden)
    with pytest.raises(daemon_module.FrameTooLarge) as caught:
        owner.request({"value": "x" * 8192})
    assert not getattr(caught.value, "outcome_unknown", False)


def test_pre_send_disconnect_is_known_and_not_replayed(tmp_path, monkeypatch):
    owner = OwnedModelDaemon(_spec(tmp_path), socket_path=tmp_path / "missing.sock")
    with pytest.raises(daemon_module.DaemonDisconnected) as caught:
        owner.request({"action": "snapshot"})
    assert not getattr(caught.value, "outcome_unknown", False)


@pytest.mark.parametrize("field", ["protocol", "request_id", "generation", "type"])
def test_untrustworthy_inference_response_is_unknown(tmp_path, monkeypatch, field):
    import pickle
    owner = OwnedModelDaemon(_spec(tmp_path), socket_path=tmp_path / "unused.sock")
    messages = []
    class Connection:
        def settimeout(self, value):
            pass
        def connect(self, path):
            pass
        def sendall(self, frame):
            messages.append(pickle.loads(frame[8:]))
        def close(self):
            pass
    def response(*args):
        request = messages[0]
        reply = {name: request[name] for name in ("protocol", "request_id", "generation", "identity")}
        reply.update(type="result", result="should not be trusted")
        reply[field] = "wrong"
        return reply
    monkeypatch.setattr(daemon_module, "_validate_endpoint", lambda path: None)
    monkeypatch.setattr(daemon_module, "_peer_uid", lambda connection: os.getuid())
    monkeypatch.setattr(daemon_module.socket, "socket", lambda *args: Connection())
    monkeypatch.setattr(daemon_module, "_read_frame", response)
    with pytest.raises(RequestOutcomeUnknown) as caught:
        owner.request({"action": "snapshot"})
    assert caught.value.request_id == messages[0]["request_id"]
    assert caught.value.generation == owner.generation
    assert len(messages) == 1


def test_idle_stop_wakes_blocked_accept_without_owner_cleanup(tmp_path):
    marker = tmp_path / "accept-blocked"
    spec = _spec(tmp_path, settings={"wait_for_blocked_accept": str(marker)})
    owner = OwnedModelDaemon(spec, startup_timeout=10)
    owner.status()
    process = owner._proc
    response = control_request(owner.socket_path, owner.identity, "stop", timeout=3)
    assert marker.read_text() == "blocked in accept"
    assert response["result"]["drained"] is True
    # No extra status connection, signal, owner.shutdown or terminate before this check.
    assert process.wait(timeout=2) == 0
    assert not owner.socket_path.exists()
    assert '"event": "service_stopped"' in owner.log_path.read_text()


def test_stop_requester_disconnect_does_not_skip_drain_completion(tmp_path):
    started, release = tmp_path / "started", tmp_path / "release"
    marker = tmp_path / "accept-blocked"
    owner = OwnedModelDaemon(_spec(tmp_path, settings={"wait_for_blocked_accept": str(marker)}),
                             startup_timeout=10, request_timeout=3)
    outcome = []
    def request():
        try:
            outcome.append(owner.request({"action": "wait", "started": str(started), "release": str(release)}))
        except BaseException as error:
            outcome.append(error)
    thread = threading.Thread(target=request)
    try:
        thread.start()
        _wait_for(started)
        process = owner._proc
        with pytest.raises(RequestOutcomeUnknown) as caught:
            control_request(owner.socket_path, owner.identity, "stop", timeout=.15)
        assert caught.value.request_id
        assert marker.exists()
    finally:
        release.write_text("go")
        thread.join(5)
    assert outcome == [{"released": True}]
    assert process.wait(timeout=2) == 0
    assert not owner.socket_path.exists()
    assert '"event": "service_stopped"' in owner.log_path.read_text()


def test_response_decode_exception_is_unknown_with_request_identity_and_no_replay(tmp_path):
    counter = tmp_path / "executions"
    with OwnedModelDaemon(_spec(tmp_path), startup_timeout=10, request_timeout=3) as owner:
        with pytest.raises(RequestOutcomeUnknown) as caught:
            owner.request({"action": "decode_failure", "counter": str(counter)})
        assert caught.value.outcome_unknown
        assert caught.value.request_id
        assert caught.value.generation == owner.generation
        assert isinstance(caught.value.__cause__, daemon_module.ProtocolError)
        assert isinstance(caught.value.__cause__.__cause__, TypeError)
        assert counter.read_text() == "1"
        assert owner.status()["completed_count"] == 1
        assert counter.read_text() == "1"


@pytest.mark.parametrize("command", ["status", "stop"])
def test_control_response_decode_exception_is_unknown_with_id(tmp_path, monkeypatch, command):
    import pickle
    bad = type("Bad", (), {"__reduce__": lambda self: (int, ([],))})
    wire = daemon_module._frame({"result": bad()})
    sent = []
    class Connection:
        def __init__(self):
            self.remaining = wire
        def settimeout(self, timeout):
            pass
        def connect(self, path):
            pass
        def sendall(self, frame):
            sent.append(pickle.loads(frame[8:]))
        def recv(self, size):
            result, self.remaining = self.remaining[:size], self.remaining[size:]
            return result
        def close(self):
            pass
    monkeypatch.setattr(daemon_module.socket, "socket", lambda *args: Connection())
    monkeypatch.setattr(daemon_module, "_validate_endpoint", lambda path: None)
    monkeypatch.setattr(daemon_module, "_peer_uid", lambda connection: os.getuid())
    with pytest.raises(RequestOutcomeUnknown) as caught:
        control_request(tmp_path / "unused.sock", "identity", command)
    assert caught.value.request_id == sent[0]["request_id"]
    assert len(sent) == 1
    assert isinstance(caught.value.__cause__, daemon_module.ProtocolError)


def _raw_inference(owner, payload, request_id="slow-consumer"):
    import socket
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    connection.settimeout(3)
    connection.connect(str(owner.socket_path))
    daemon_module._send_frame(connection, {
        "protocol": daemon_module.PROTOCOL, "identity": owner.identity,
        "generation": "reply-test", "request_id": request_id,
        "type": "infer", "payload": payload,
    })
    return connection


def _wait_computation(owner):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if '"event": "inference_end"' in owner.log_path.read_text():
            return
        time.sleep(.005)
    raise AssertionError("test inference did not finish computing")


def test_drain_waits_for_full_reply_to_a_bounded_slow_receiver(tmp_path):
    owner = OwnedModelDaemon(_spec(tmp_path), startup_timeout=10)
    owner.status()
    connection = _raw_inference(owner, {"action": "large_bytes", "size": 4 * 1024 * 1024})
    results, errors = [], []
    def receive():
        time.sleep(.25)
        try:
            results.append(daemon_module._read_frame(connection))
        except BaseException as error:
            errors.append(error)
    receiver = threading.Thread(target=receive)
    try:
        _wait_computation(owner)
        receiver.start()
        response = control_request(owner.socket_path, owner.identity, "stop", timeout=3)
        assert owner._proc.wait(timeout=2) == 0
        receiver.join(3)
        assert not receiver.is_alive()
        assert not errors, errors
        assert results[0]["result"] == b"x" * (4 * 1024 * 1024)
        assert response["result"]["drained"] is True
        assert response["result"]["reply_sent_count"] == 1
        assert response["result"]["reply_failed_count"] == 0
    finally:
        connection.close()
        if receiver.ident is not None:
            receiver.join(3)


def test_drain_send_budget_reports_failure_for_a_nonreading_receiver(tmp_path):
    import socket
    owner = OwnedModelDaemon(_spec(tmp_path), startup_timeout=10)
    owner.status()
    connection = _raw_inference(owner, {"action": "large_bytes", "size": 4 * 1024 * 1024})
    try:
        _wait_computation(owner)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as control:
            control.settimeout(3)
            control.connect(str(owner.socket_path))
            daemon_module._send_frame(control, {
                "protocol": daemon_module.PROTOCOL, "identity": owner.identity,
                "generation": "", "request_id": "short-drain", "type": "stop",
                "drain_timeout": .1,
            })
            response = daemon_module._read_frame(control)
        assert owner._proc.wait(timeout=2) == 0
        result = response["result"]
        assert result["drained"] is False
        assert result["reply_failed_count"] == 1
        assert result["reply_timeout_count"] == 1
        assert result["reply_sent_count"] == 0
    finally:
        connection.close()


def test_drain_does_not_report_success_after_inference_peer_disconnects(tmp_path):
    owner = OwnedModelDaemon(_spec(tmp_path), startup_timeout=10)
    owner.status()
    started, release = tmp_path / "reply-started", tmp_path / "reply-release"
    connection = _raw_inference(owner, {"action": "wait", "started": str(started),
                                       "release": str(release), "large_bytes": 4 * 1024 * 1024})
    try:
        _wait_for(started)
        connection.close()
    finally:
        release.write_text("go")
    response = owner.request_stop()
    assert response["result"]["drained"] is False
    assert response["result"]["reply_failed_count"] == 1
    assert response["result"]["reply_sent_count"] == 0


def test_disconnected_stop_client_still_finishes_a_large_inference_reply(tmp_path):
    import socket
    owner = OwnedModelDaemon(_spec(tmp_path), startup_timeout=10)
    owner.status()
    connection = _raw_inference(owner, {"action": "large_bytes", "size": 4 * 1024 * 1024})
    results, errors = [], []
    def receive():
        time.sleep(.25)
        try:
            results.append(daemon_module._read_frame(connection))
        except BaseException as error:
            errors.append(error)
    receiver = threading.Thread(target=receive)
    try:
        _wait_computation(owner)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stop:
            stop.connect(str(owner.socket_path))
            daemon_module._send_frame(stop, {
                "protocol": daemon_module.PROTOCOL, "identity": owner.identity,
                "generation": "", "request_id": "abandoned-stop", "type": "stop",
                "drain_timeout": 2,
            })
        receiver.start()
        assert owner._proc.wait(timeout=3) == 0
        receiver.join(3)
        assert not receiver.is_alive()
        assert not errors
        assert results[0]["result"] == b"x" * (4 * 1024 * 1024)
        log = owner.log_path.read_text()
        assert '"event": "drain_complete"' in log
        assert '"event": "service_stopped"' in log
    finally:
        connection.close()
        if receiver.ident is not None:
            receiver.join(3)


def test_owner_cleanup_preserves_an_unknown_stop_ack_instead_of_reporting_success(tmp_path, monkeypatch):
    owner = OwnedModelDaemon(_spec(tmp_path), startup_timeout=10)
    owner.status()
    process = owner._proc
    original_read = daemon_module._read_frame
    def lose_stop_ack(connection, max_frame_bytes=daemon_module.MAX_FRAME_BYTES):
        response = original_read(connection, max_frame_bytes)
        if response.get("type") == "result" and isinstance(response.get("result"), dict) and "drained" in response["result"]:
            raise daemon_module.DaemonDisconnected("simulated lost stop acknowledgment")
        return response
    monkeypatch.setattr(daemon_module, "_read_frame", lose_stop_ack)
    with pytest.raises(RequestOutcomeUnknown) as caught:
        owner.shutdown()
    assert caught.value.request_id
    assert caught.value.generation == owner.generation
    assert process.poll() == 0
    assert owner._closed
    lifecycle = json.loads((owner.temp_dir / "lifecycle.json").read_text())
    assert lifecycle["drain_result"] is None
    owner.shutdown()  # Local close remains idempotent and never replays the stop.
