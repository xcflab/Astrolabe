"""Load only the delivered modules; children get a model-free temporary package."""
import importlib.util
from pathlib import Path
import random
import pickle
import shutil
import sys
import tempfile
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_astrolabe_foundation_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "astrolabe")]
sys.modules[PACKAGE] = package


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE}.{name}", ROOT / "astrolabe" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


execution = _load("model_execution")
daemon = _load("model_daemon")


@pytest.fixture(autouse=True)
def cpu_only(tmp_path, monkeypatch):
    """Keep sockets, package copies, RNG changes and child processes test-local."""
    saved = (random.getstate(), np.random.get_state())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("ASTROLABE_MODEL_PROVIDERS", raising=False)
    # Each test models a fresh coordinator. This is test-only lifetime isolation,
    # not a supported recovery mechanism for an uncertain production stream.
    monkeypatch.setattr(execution, "_RANDOM_STATE_FAILURE", None)

    site = tmp_path / "site"
    local_package = site / "astrolabe"
    local_package.mkdir(parents=True)
    (local_package / "__init__.py").write_text("")
    for name in ("model_execution.py", "model_daemon.py"):
        shutil.copyfile(ROOT / "astrolabe" / name, local_package / name)
    monkeypatch.setattr(daemon, "_REPO_ROOT", site)
    runtime = tempfile.TemporaryDirectory(prefix="d-")
    monkeypatch.setattr(daemon.tempfile, "tempdir", runtime.name)
    clients = []
    original_init = daemon.OwnedModelDaemon.__init__

    def track(client, *args, **kwargs):
        original_init(client, *args, **kwargs)
        clients.append(client)

    monkeypatch.setattr(daemon.OwnedModelDaemon, "__init__", track)
    execution.reset_remote_random_state()
    try:
        yield
    finally:
        for client in reversed(clients):
            try:
                client.shutdown()
            except daemon.DaemonDrainFailed:
                # Teardown only: a test may intentionally abandon its own reply.
                pass
        execution._RANDOM_STATE_FAILURE = None
        execution.reset_remote_random_state()
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        runtime.cleanup()


@pytest.fixture
def fake_torch(monkeypatch):
    """A CPU-only state-routing double, never an actual Torch/CUDA implementation."""
    class Generator:
        def __init__(self):
            self.manual_seed(0)

        def manual_seed(self, seed):
            self.seed = seed
            self.rng = np.random.RandomState(seed)
            return self

        def get_state(self):
            return np.frombuffer(pickle.dumps((self.seed, self.rng.get_state())),
                                 dtype=np.uint8).copy()

        def set_state(self, state):
            self.seed, value = pickle.loads(np.asarray(state, dtype=np.uint8).tobytes())
            self.rng.set_state(value)

    def forbidden(*args, **kwargs):
        pytest.fail("an unmocked CUDA operation was attempted")

    generator = Generator()
    cuda = types.SimpleNamespace(
        is_initialized=lambda: False, is_available=forbidden, init=forbidden,
        manual_seed_all=lambda seed: None, get_rng_state_all=forbidden,
        set_rng_state_all=forbidden,
    )
    torch = types.ModuleType("torch")
    torch.cuda = cuda
    torch.Generator = Generator
    torch.get_rng_state = generator.get_state
    torch.set_rng_state = generator.set_state
    torch.initial_seed = lambda: generator.seed
    torch.tensor = lambda value, dtype=None: np.asarray(value, dtype=dtype)
    torch.uint8 = np.uint8
    torch.rand = lambda size, generator=generator: np.asarray(generator.rng.random_sample(size))

    def manual_seed(seed):
        generator.manual_seed(seed)
        cuda.manual_seed_all(seed)

    torch.manual_seed = manual_seed
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch
