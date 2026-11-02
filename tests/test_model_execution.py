import random
import sys
from dataclasses import dataclass
from collections import namedtuple

import numpy as np
import pytest

from test_model_execution_fixture import cpu_only, execution, fake_torch, daemon as daemon_module

from _astrolabe_foundation_tests.model_execution import (
    capture_random_state, host_result, kernel_handler, restore_random_state,
    model_run_scope, register_model_owner,
    reset_remote_random_state,
)


def test_kernel_preserves_constructor_and_interleaved_request_random_streams():
    random.seed(77)
    np.random.seed(77)
    initial = capture_random_state()
    loads = []

    def loader(settings):
        loads.append((random.random(), np.random.random()))
        def infer(payload):
            result = (random.random(), float(np.random.random()))
            if payload.get("fail"):
                raise ValueError("request failed after random draws")
            return result
        return infer, {"model": "fixture"}

    random.seed(77)
    np.random.seed(77)
    reference_constructor = (random.random(), np.random.random())
    expected_first = (random.random(), float(np.random.random()))
    expected_second = (random.random(), float(np.random.random()))
    handler = kernel_handler({"_initial_random_state": initial}, loader)
    initialized = handler({"operation": "describe"})["random_state"]
    assert loads == [reference_constructor]
    first = handler({"operation": "infer", "payload": {}, "random_state": initialized})
    random.seed(999)
    np.random.seed(999)
    other = handler({"operation": "infer", "payload": {}, "random_state": capture_random_state()})
    assert other["result"] != first["result"]
    second = handler({"operation": "infer", "payload": {}, "random_state": first["random_state"]})
    assert first["result"] == expected_first
    assert second["result"] == expected_second
    assert len(loads) == 1


def test_failure_returns_advanced_random_state_without_repeating_the_kernel():
    np.random.seed(13)
    state = capture_random_state()
    calls = []
    def infer(payload):
        calls.append(float(np.random.random()))
        raise ValueError("bad input")
    expected = np.random.RandomState(13)
    handler = kernel_handler({}, lambda settings: (infer, {}))
    failed = handler({"operation": "infer", "payload": {}, "random_state": state})
    assert failed["ok"] is False and failed["error_type"] == "ValueError"
    assert calls == [float(expected.random_sample())]
    restore_random_state(failed["random_state"])
    assert np.random.random() == expected.random_sample()


def test_host_materialization_preserves_array_dtype_and_shape():
    values = np.arange(6, dtype=np.int32).reshape(2, 3)
    result = host_result({"features": values})
    np.testing.assert_array_equal(result["features"], values)
    assert result["features"].dtype == np.int32


def test_kernel_rejects_unknown_operations():
    handler = kernel_handler({}, lambda settings: (lambda payload: payload, {}))
    with pytest.raises(ValueError, match="Unknown kernel operation"):
        handler({"operation": "arbitrary_method"})


def test_run_scope_releases_partial_startup_and_keeps_outer_owners_alive():
    stopped = []
    class Owner:
        def __init__(self, name, attached=False):
            self.name = name
            self.attached = attached
        def shutdown(self):
            stopped.append(self.name)

    with model_run_scope():
        outer = Owner("outer")
        register_model_owner(outer)
        with pytest.raises(ValueError, match="later component failed"):
            with model_run_scope():
                inner = Owner("inner")
                register_model_owner(inner)
                register_model_owner(inner)
                register_model_owner(Owner("external", attached=True))
                raise ValueError("later component failed")
        assert stopped == ["inner"]
    assert stopped == ["inner", "outer"]


def test_remote_cuda_state_stays_opaque_and_does_not_queue_device_callbacks(monkeypatch, fake_torch):
    torch = fake_torch

    reset_remote_random_state()
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    def forbidden(*args):
        pytest.fail("The CPU coordinator touched a CUDA generator")
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    state = capture_random_state()
    state["cuda"] = [[1, 2, 3]]
    try:
        restore_random_state(state)
        assert capture_random_state()["cuda"] == [[1, 2, 3]]
        state["cuda"] = [[4, 5, 6]]
        restore_random_state(state)
        assert capture_random_state()["cuda"] == [[4, 5, 6]]
        assert torch.initial_seed() == state["torch_seed"]
    finally:
        reset_remote_random_state()


def test_same_seed_restarts_the_remote_cuda_stream(monkeypatch, fake_torch):
    torch = fake_torch

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    torch.manual_seed(17)
    reset_remote_random_state()
    state = capture_random_state()
    state["cuda"] = [[1, 2, 3]]
    restore_random_state(state)
    assert "cuda" in capture_random_state()
    torch.manual_seed(17)
    reset_remote_random_state()
    assert "cuda" not in capture_random_state()


def test_remote_cuda_continues_after_constructor_interleaving_and_failure(monkeypatch, fake_torch):
    torch = fake_torch

    reset_remote_random_state()
    active = {"process": "parent"}
    device_generators = {name: torch.Generator() for name in ("one", "two")}
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: active["process"] != "parent")
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [
        device_generators[active["process"]].get_state()
    ])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states:
        device_generators[active["process"]].set_state(states[0]))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed:
        device_generators[active["process"]].manual_seed(seed))
    torch.set_rng_state(torch.Generator().manual_seed(17).get_state())
    reference = torch.Generator().manual_seed(17)
    def draw():
        return torch.rand(1, generator=device_generators[active["process"]]).item()
    def loader(settings):
        assert draw() == torch.rand(1, generator=reference).item()
        def infer(payload):
            value = draw()
            if payload.get("fail"):
                raise ValueError("after CUDA draw")
            return value
        return infer, {}
    handlers = {}
    try:
        for name in ("one", "two"):
            state = capture_random_state()
            active["process"] = name
            handlers[name] = kernel_handler({"_initial_random_state": state}, loader)
            description = handlers[name]({"operation": "describe"})
            active["process"] = "parent"
            restore_random_state(description["random_state"])
        for name, fail in (("one", False), ("two", True), ("one", False)):
            state = capture_random_state()
            active["process"] = name
            response = handlers[name]({"operation": "infer", "random_state": state,
                                       "payload": {"fail": fail}})
            expected = torch.rand(1, generator=reference).item()
            assert response["ok"] is (not fail)
            if not fail:
                assert response["result"] == expected
            active["process"] = "parent"
            restore_random_state(response["random_state"])
    finally:
        reset_remote_random_state()


def test_cpu_owner_preserves_incoming_cuda_and_observes_same_seed_constructor_reset(monkeypatch, fake_torch):
    torch = fake_torch

    reset_remote_random_state()
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    seed_calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", seed_calls.append)
    incoming = capture_random_state()
    incoming["cuda"] = [[1, 2, 3]]
    try:
        handler = kernel_handler({"_initial_random_state": incoming},
                                 lambda settings: (lambda payload: "cpu", {}))
        description = handler({"operation": "describe"})
        assert description["random_state"]["cuda"] == incoming["cuda"]
        response = handler({"operation": "infer", "payload": {},
                            "random_state": description["random_state"]})
        assert response["random_state"]["cuda"] == incoming["cuda"]
        assert seed_calls == []
        original_seed_function = torch.cuda.manual_seed_all
        def reseeded_loader(settings):
            torch.manual_seed(incoming["torch_seed"])
            def fail(payload):
                torch.manual_seed(incoming["torch_seed"])
                raise ValueError("after same-seed reset")
            return fail, {}
        reset_handler = kernel_handler({"_initial_random_state": incoming}, reseeded_loader)
        description = reset_handler({"operation": "describe"})
        assert "cuda" not in description["random_state"]
        assert description["random_state"]["cuda_seed"] == incoming["torch_seed"]
        failed = reset_handler({"operation": "infer", "payload": {}, "random_state": incoming})
        assert failed["ok"] is False
        assert "cuda" not in failed["random_state"]
        assert failed["random_state"]["cuda_seed"] == incoming["torch_seed"]
        assert torch.cuda.manual_seed_all is original_seed_function
    finally:
        reset_remote_random_state()


def test_cuda_child_initializes_before_restoring_advanced_state(monkeypatch, fake_torch):
    torch = fake_torch

    reset_remote_random_state()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    initialized = {"value": False}
    device_generator = torch.Generator().manual_seed(17)
    torch.rand(3, generator=device_generator)
    advanced = device_generator.get_state()
    expected = torch.rand(1, generator=device_generator).item()
    incoming = capture_random_state()
    incoming["cuda"] = [advanced.tolist()]
    incoming["cuda_visible_devices"] = None
    events = []
    def initialize():
        events.append("init")
        device_generator.manual_seed(999)
        initialized["value"] = True
    def restore(states):
        events.append("restore")
        assert initialized["value"]
        device_generator.set_state(states[0])
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized["value"])
    monkeypatch.setattr(torch.cuda, "init", initialize)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restore)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [device_generator.get_state()])
    def loader(settings):
        assert torch.rand(1, generator=device_generator).item() == expected
        return lambda payload: None, {}
    try:
        kernel_handler({"_initial_random_state": incoming}, loader, torch_device="cuda")
        assert events == ["init", "restore"]
    finally:
        reset_remote_random_state()


def test_native_client_subprocess_continues_constructor_failure_and_interleaved_rng(tmp_path):
    adapter = tmp_path / "rng_adapter.py"
    adapter.write_text('''
import random
from pathlib import Path
import numpy as np
from astrolabe.model_execution import kernel_handler

def make(settings):
    def loader(settings):
        constructor = (random.random(), float(np.random.random()))
        def infer(payload):
            counter = Path(settings["counter"])
            value = int(counter.read_text()) if counter.exists() else 0
            counter.write_text(str(value + 1))
            draw = (random.random(), float(np.random.random()))
            if payload.get("fail"):
                raise ValueError("after random draws")
            return draw
        return infer, {"constructor": constructor}
    return kernel_handler(settings, loader)
''')
    random.seed(77)
    np.random.seed(77)
    reference_python = random.Random(77)
    reference_numpy = np.random.RandomState(77)
    def expected():
        return reference_python.random(), float(reference_numpy.random_sample())
    counter = tmp_path / "counter"
    clients = []
    with model_run_scope():
        for _ in range(2):
            client = execution.NativeModelClient(
                "cpu-rng", "rng_adapter:make", {"counter": str(counter)},
                python_bin=sys.executable, env={"PYTHONPATH": str(tmp_path),
                                                 "CUDA_VISIBLE_DEVICES": ""},
                startup_timeout=20, request_timeout=3,
            )
            clients.append(client)
            assert client.metadata["constructor"] == expected()
        assert clients[0].request({}) == expected()
        with pytest.raises(RuntimeError, match="ValueError: after random draws"):
            clients[1].request({"fail": True})
        expected()  # The failed request consumed one draw from each stream.
        assert (random.random(), float(np.random.random())) == expected()
        assert clients[0].request({}) == expected()
        processes = [client.owner._proc for client in clients]
    assert counter.read_text() == "3"
    assert all(process.poll() is not None for process in processes)


def test_host_result_recurses_and_rejects_coordinator_objects():
    @dataclass
    class Result:
        values: object
    Pair = namedtuple("Pair", "left right")
    events = []
    class Tensor:
        __module__ = "torch.fake"
        def detach(self):
            events.append("detach")
            return self
        def cpu(self):
            events.append("cpu")
            return "host"
    result = host_result(Result(Pair([Tensor()], {"x": Tensor()})))
    assert result == Result(Pair(["host"], {"x": "host"}))
    assert events == ["detach", "cpu", "detach", "cpu"]
    Pose = type("Pose", (), {"__module__": "pyrosetta.rosetta"})
    with pytest.raises(TypeError, match="scientific coordinator"):
        host_result({"pose": Pose()})


def test_run_scope_cleanup_failure_preserves_primary_error():
    calls = []
    class Owner:
        def shutdown(self):
            calls.append("stop")
            raise ValueError("shutdown failed")
    with pytest.raises(KeyError, match="primary"):
        with model_run_scope():
            register_model_owner(Owner())
            raise KeyError("primary")
    with pytest.raises(RuntimeError, match="failed to stop"):
        with model_run_scope():
            register_model_owner(Owner())
    assert calls == ["stop", "stop"]


def test_execution_policy_keeps_source_default_and_validates_explicit_values(monkeypatch):
    monkeypatch.delenv("ASTROLABE_MODEL_EXECUTION", raising=False)
    assert execution.execution_mode() is execution.ExecutionMode.DAEMON
    monkeypatch.setenv("ASTROLABE_MODEL_EXECUTION", "in_process")
    assert execution.execution_mode() is execution.ExecutionMode.IN_PROCESS
    assert execution.execution_mode("daemon") is execution.ExecutionMode.DAEMON
    with pytest.raises(ValueError, match="'daemon' or 'in_process'"):
        execution.execution_mode("unsupported")


RNG_CONCURRENCY_ADAPTER = '''
import random, time
from pathlib import Path
import numpy as np
from astrolabe.model_execution import kernel_handler

class DecodeFailure:
    def __reduce__(self):
        return int, ([],)

def make(settings):
    def loader(settings):
        constructor = (random.random(), float(np.random.random()))
        def infer(payload):
            if payload.get("started"):
                Path(payload["started"]).write_text("started")
                deadline = time.monotonic() + 5
                while not Path(payload["release"]).exists():
                    if time.monotonic() > deadline:
                        raise RuntimeError("test release timeout")
                    time.sleep(.01)
            draw = (random.random(), float(np.random.random()))
            if payload.get("fail"):
                raise ValueError("after draws")
            if payload.get("decode_failure"):
                return DecodeFailure()
            if payload.get("unserializable"):
                return lambda: None
            return draw
        return infer, {"constructor": constructor}
    return kernel_handler(settings, loader)
'''


def _rng_client(tmp_path, socket_path=None):
    (tmp_path / "rng_concurrency_adapter.py").write_text(RNG_CONCURRENCY_ADAPTER)
    return execution.NativeModelClient(
        "cpu-rng-concurrency", "rng_concurrency_adapter:make", {},
        python_bin=sys.executable,
        env={"PYTHONPATH": str(tmp_path), "CUDA_VISIBLE_DEVICES": ""},
        startup_timeout=10, request_timeout=3, socket_path=socket_path,
    )


def _rng_reference():
    python = random.Random(0)
    python.setstate(random.getstate())
    numpy = np.random.RandomState(0)
    numpy.set_state(np.random.get_state())
    return lambda: (python.random(), float(numpy.random_sample()))


def _wait_file(path):
    import time
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert path.exists(), "CPU fixture did not start"


def test_attach_reads_metadata_without_rewinding_coordinator_rng(tmp_path):
    random.seed(77)
    np.random.seed(77)
    client = _rng_client(tmp_path)
    expected = _rng_reference()
    assert client.request({}) == expected()
    before = capture_random_state()
    attached = _rng_client(tmp_path, client.owner.socket_path)
    assert attached.metadata == client.metadata
    assert capture_random_state() == before
    assert attached.request({}) == expected()
    assert (random.random(), float(np.random.random())) == expected()


@pytest.mark.parametrize("other_client", ["same", "attached", "independent"])
def test_shared_rng_transaction_serializes_concurrent_clients(tmp_path, monkeypatch, other_client):
    import threading
    random.seed(77)
    np.random.seed(77)
    first = _rng_client(tmp_path)
    if other_client == "same":
        second = first
    elif other_client == "attached":
        second = _rng_client(tmp_path, first.owner.socket_path)
    else:
        second = _rng_client(tmp_path)
    expected = _rng_reference()
    expected_results = [expected(), expected()]
    started, release = tmp_path / "started", tmp_path / "release"
    attempted, captured = threading.Event(), threading.Event()
    original_capture = execution.capture_random_state
    def observed_capture():
        result = original_capture()
        if threading.current_thread().name == "second-rng-client":
            captured.set()
        return result
    monkeypatch.setattr(execution, "capture_random_state", observed_capture)
    results, errors = {}, []
    def call(index, client, payload):
        if index == 1:
            attempted.set()
        try:
            results[index] = client.request(payload)
        except BaseException as error:
            errors.append(error)
    one = threading.Thread(target=call, args=(0, first, {"started": str(started), "release": str(release)}))
    two = threading.Thread(target=call, name="second-rng-client", args=(1, second, {}))
    try:
        one.start()
        _wait_file(started)
        two.start()
        assert attempted.wait(3)
        captured_while_first_inflight = captured.wait(.2)
    finally:
        release.write_text("go")
        one.join(5)
        if two.ident is not None:
            two.join(5)
    assert not one.is_alive() and not two.is_alive()
    assert not errors
    assert not captured_while_first_inflight, "second client captured an unfinished shared stream"
    assert [results[0], results[1]] == expected_results
    assert (random.random(), float(np.random.random())) == expected()


def test_local_kernel_handlers_cannot_interleave_global_rng(tmp_path):
    import threading
    first_started, release, second_entered = (threading.Event() for _ in range(3))
    def make_loader(name):
        def loader(settings):
            def infer(payload):
                if name == "second":
                    second_entered.set()
                before = random.random(), float(np.random.random())
                if name == "first":
                    first_started.set()
                    assert release.wait(3)
                return before, (random.random(), float(np.random.random()))
            return infer, {}
        return loader
    states, expected, expected_next = [], [], []
    for seed in (17, 31):
        random.seed(seed)
        np.random.seed(seed)
        states.append(capture_random_state())
        reference = _rng_reference()
        expected.append((reference(), reference()))
        expected_next.append(reference())
    handlers = [kernel_handler({}, make_loader(name)) for name in ("first", "second")]
    responses = {}
    def call(index):
        responses[index] = handlers[index]({"operation": "infer", "payload": {}, "random_state": states[index]})
    one = threading.Thread(target=call, args=(0,))
    two = threading.Thread(target=call, args=(1,))
    try:
        one.start()
        assert first_started.wait(3)
        two.start()
        second_overlapped = second_entered.wait(.2)
    finally:
        release.set()
        one.join(5)
        if two.ident is not None:
            two.join(5)
    assert not second_overlapped
    for index in range(2):
        assert responses[index]["ok"]
        assert responses[index]["result"] == expected[index]
        restore_random_state(responses[index]["random_state"])
        assert (random.random(), float(np.random.random())) == expected_next[index]


def test_new_client_constructor_cannot_capture_an_inflight_rng_transaction(tmp_path, monkeypatch):
    import threading
    first = _rng_client(tmp_path)
    reference = _rng_reference()
    expected_result, expected_constructor = reference(), reference()
    started, release = tmp_path / "started", tmp_path / "release"
    attempted, captured = threading.Event(), threading.Event()
    original = execution.capture_random_state
    def capture():
        result = original()
        if threading.current_thread().name == "rng-constructor":
            captured.set()
        return result
    monkeypatch.setattr(execution, "capture_random_state", capture)
    results, errors = {}, []
    def request():
        try:
            results["infer"] = first.request({"started": str(started), "release": str(release)})
        except BaseException as error:
            errors.append(error)
    def construct():
        attempted.set()
        try:
            results["client"] = _rng_client(tmp_path)
        except BaseException as error:
            errors.append(error)
    one = threading.Thread(target=request)
    two = threading.Thread(target=construct, name="rng-constructor")
    try:
        one.start()
        _wait_file(started)
        two.start()
        assert attempted.wait(3)
        captured_during_request = captured.wait(.2)
    finally:
        release.write_text("go")
        one.join(5)
        if two.ident is not None:
            two.join(10)
    assert not one.is_alive() and not two.is_alive()
    assert not errors
    assert not captured_during_request
    assert results["infer"] == expected_result
    assert results["client"].metadata["constructor"] == expected_constructor
    assert (random.random(), float(np.random.random())) == reference()


@pytest.mark.parametrize("failure", ["kernel", "decode_unknown"])
def test_rng_transaction_releases_after_failure_without_automatic_replay(tmp_path, failure):
    import threading
    client = _rng_client(tmp_path)
    if failure == "kernel":
        error_type, payload = RuntimeError, {"fail": True}
    else:
        error_type, payload = daemon_module.RequestOutcomeUnknown, {"decode_failure": True}
    with pytest.raises(error_type):
        client.request(payload)
    entered = []
    blocked = []
    def acquire():
        try:
            with execution.model_random_transaction():
                entered.append(capture_random_state())
        except execution.RandomStateUncertain as error:
            blocked.append(error)
    thread = threading.Thread(target=acquire)
    thread.start()
    thread.join(3)
    assert not thread.is_alive()
    assert len(entered) == (1 if failure == "kernel" else 0)
    assert len(blocked) == (1 if failure == "decode_unknown" else 0)
    # Only the explicit failed inference (plus initial describe) was sent.
    assert client.owner.status()["completed_count"] == 2


def test_external_coordinator_draws_can_join_the_same_rng_transaction(tmp_path):
    import threading
    client = _rng_client(tmp_path)
    reference = _rng_reference()
    expected = [reference(), reference()]
    started, release = tmp_path / "started", tmp_path / "release"
    attempted, entered = threading.Event(), threading.Event()
    results, errors = {}, []
    def infer():
        try:
            results[0] = client.request({"started": str(started), "release": str(release)})
        except BaseException as error:
            errors.append(error)
    def draw():
        attempted.set()
        with execution.model_random_transaction():
            entered.set()
            results[1] = random.random(), float(np.random.random())
    one = threading.Thread(target=infer)
    two = threading.Thread(target=draw)
    try:
        one.start()
        _wait_file(started)
        two.start()
        assert attempted.wait(3)
        entered_early = entered.wait(.2)
    finally:
        release.write_text("go")
        one.join(5)
        if two.ident is not None:
            two.join(5)
    assert not one.is_alive() and not two.is_alive()
    assert not errors
    assert not entered_early
    assert [results[0], results[1]] == expected


def test_fork_resets_rng_mutex_and_rejects_inherited_client_before_locking(tmp_path):
    import os
    import select
    import signal
    import threading
    if not hasattr(os, "fork"):
        pytest.skip("requires fork")
    client = _rng_client(tmp_path)
    held, release = threading.Event(), threading.Event()
    def holder():
        with execution.model_random_transaction():
            held.set()
            assert release.wait(10)
    thread = threading.Thread(target=holder)
    thread.start()
    assert held.wait(3)
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        try:
            try:
                client.request({})
            except daemon_module.ModelDaemonError as error:
                assert "fork" in str(error)
            else:
                raise AssertionError("inherited client was usable")
            capture_random_state()  # The vanished holder must not leave a locked mutex.
            fresh = _rng_client(tmp_path, client.owner.socket_path)
            assert fresh.metadata == client.metadata
            fresh.shutdown()
            os.write(write_fd, b"ok")
        finally:
            os._exit(0)
    os.close(write_fd)
    try:
        assert select.select([read_fd], [], [], 5)[0], "fork inherited a locked RNG mutex"
        assert os.read(read_fd, 64) == b"ok"
    finally:
        release.set()
        thread.join(3)
        os.close(read_fd)
        finished, _ = os.waitpid(child, os.WNOHANG)
        if not finished:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0)
    assert client.owner.status()["state"] == "running"


@pytest.mark.parametrize("layout", ["same", "attached", "independent"])
def test_unknown_blocks_waiting_and_future_shared_rng_transactions(tmp_path, layout):
    import threading
    first = _rng_client(tmp_path)
    second = first if layout == "same" else _rng_client(
        tmp_path, first.owner.socket_path if layout == "attached" else None)
    second_initial_count = second.owner.status()["request_count"]
    first.owner.request_timeout = .25
    started, release = tmp_path / "uncertain-started", tmp_path / "uncertain-release"
    attempted, results, errors = threading.Event(), {}, {}
    def call(index, client, payload):
        if index == 1:
            attempted.set()
        try:
            results[index] = client.request(payload)
        except BaseException as error:
            errors[index] = error
    one = threading.Thread(target=call, args=(0, first, {"started": str(started), "release": str(release)}))
    two = threading.Thread(target=call, args=(1, second, {}))
    try:
        one.start()
        _wait_file(started)
        two.start()
        assert attempted.wait(3)
        one.join(2)
        assert not one.is_alive()
    finally:
        release.write_text("go")
        if two.ident is not None:
            two.join(3)
        one.join(3)
    assert isinstance(errors[0], daemon_module.RequestOutcomeUnknown)
    assert 1 not in results, "waiting client continued an uncertain shared stream"
    blocked = errors[1]
    assert isinstance(blocked, execution.RandomStateUncertain)
    assert blocked.origin_request_id == errors[0].request_id
    assert blocked.origin_generation == errors[0].generation
    assert second.owner.status()["request_count"] == (second_initial_count + (1 if second.owner is first.owner or layout == "attached" else 0))
    for action in (lambda: second.request({}), capture_random_state,
                   lambda: restore_random_state({}), reset_remote_random_state,
                   lambda: _rng_client(tmp_path)):
        with pytest.raises(execution.RandomStateUncertain):
            action()
    with pytest.raises(execution.RandomStateUncertain):
        with execution.model_random_transaction():
            pytest.fail("external shared transaction ran after unknown")
    assert second.owner.status()["request_count"] == (second_initial_count + (1 if second.owner is first.owner or layout == "attached" else 0))


def test_unknown_fence_survives_caught_errors_and_fork(tmp_path):
    import os
    import select
    import signal
    original = daemon_module.RequestOutcomeUnknown("unresolved", request_id="original-id", generation="original-generation")
    with pytest.raises(execution.RandomStateUncertain):
        with execution.model_random_transaction():
            with pytest.raises(daemon_module.RequestOutcomeUnknown):
                with execution.model_random_transaction():
                    raise original
            with pytest.raises(RuntimeError, match="uncertain"):
                capture_random_state()
    if not hasattr(os, "fork"):
        return
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        try:
            try:
                capture_random_state()
            except execution.RandomStateUncertain as error:
                assert error.origin_request_id == "original-id"
                os.write(write_fd, b"blocked")
        finally:
            os._exit(0)
    os.close(write_fd)
    try:
        assert select.select([read_fd], [], [], 3)[0]
        assert os.read(read_fd, 64) == b"blocked"
    finally:
        os.close(read_fd)
        finished, _ = os.waitpid(child, os.WNOHANG)
        if not finished:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0)


@pytest.mark.parametrize("failure", ["unserializable", "missing_state", "invalid_state"])
def test_missing_or_unusable_rng_receipt_also_fences_the_stream(tmp_path, monkeypatch, failure):
    client = _rng_client(tmp_path)
    second = _rng_client(tmp_path)
    if failure != "unserializable":
        original_request = client.owner.request
        def malformed(payload, **kwargs):
            response = original_request(payload, **kwargs)
            if failure == "missing_state":
                response.pop("random_state")
            else:
                response["random_state"] = {"python": [1]}
            return response
        monkeypatch.setattr(client.owner, "request", malformed)
    with pytest.raises(daemon_module.RequestOutcomeUnknown) as caught:
        client.request({"unserializable": failure == "unserializable"})
    assert caught.value.request_id
    assert caught.value.generation == client.owner.generation
    with pytest.raises(execution.RandomStateUncertain) as blocked:
        second.request({})
    assert blocked.value.origin_request_id == caught.value.request_id
    assert second.owner.status()["request_count"] == 1
