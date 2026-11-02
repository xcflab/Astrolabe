"""Owner-side AF3 resident forward client."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

from astrolabe.evaluators.tree_codec import dump_tree, load_tree
from astrolabe.evaluators.worker_identity import identity, probe_device
from astrolabe.evaluators.worker_session import WorkerArtifacts

BROKER_SHARED_ROOT = True
BROKER_ADMITTED_PROBE = True


class AF3Resident:
    def __init__(self, config):
        import alphafold3
        from omegaconf import OmegaConf
        from astrolabe.runtime.client import DaemonClient
        self._artifacts = WorkerArtifacts()
        root = self._artifacts.root
        model_config = OmegaConf.to_container(config, resolve=True)
        base = Path(__file__).resolve().parent
        code = [str(base / name) for name in ('alphafold3.py', 'tree_codec.py', 'request_path.py', 'worker_session.py', 'device_probe.py', 'worker_identity.py', 'wrappers/alphafold3.py', 'wrappers/af3_daemon.py')]
        code.append(str(Path(alphafold3.__file__).resolve().parent))
        weights = [str(Path(model_config['model_dir']).expanduser().resolve(strict=True))]
        env = {'PYTHONPATH': os.pathsep.join([str(Path(p).resolve()) for p in sys.path if p])}
        device = probe_device(sys.executable, 'af3', env=env)
        fields = ('flash_attention_implementation', 'num_samples', 'num_recycles', 'save_embeddings', 'jax_compilation_cache_dir', 'refstructure_workers', 'refstructure_cache', 'refstructure_cache_max_entries', 'msa_cache', 'msa_cache_max_entries')
        semantics = {key: model_config.get(key) for key in fields}
        fingerprint = identity(sys.executable, code, weights, device, semantics)
        worker_config = dict(semantics, model_dir=model_config['model_dir'])
        spec = {'model_config': worker_config, 'code_paths': code, 'weight_paths': weights, 'device': device}
        path = self._artifacts.specification(fingerprint, spec)
        self._client = DaemonClient([sys.executable, '-m', 'astrolabe.evaluators.wrappers.af3_daemon', str(path)], fingerprint, env=env, startup_timeout=float(model_config.get('daemon_startup_timeout', 600)))
        self.timeout = model_config.get('daemon_timeout')

    def forward(self, seeds, examples, cyclic, patches):
        from alphafold3.model.components.utils import remove_invalidly_typed_feats
        examples = [remove_invalidly_typed_feats(example) for example in examples]
        root = self._artifacts.root
        with tempfile.TemporaryDirectory(prefix='request-', dir=root) as tmp:
            path = Path(tmp)
            dump_tree(path / 'input', {'seeds': list(seeds), 'examples': examples, 'cyclic': cyclic, 'patches': patches})
            answer = self._client.call({'request_dir': str(path)}, timeout=self.timeout, request_id=uuid.uuid4().hex)
            if answer != {'request': path.name}: raise ValueError('AF3 worker returned wrong request')
            return load_tree(path / 'output')

    def close(self):
        if getattr(self, '_client', None) is not None:
            self._client.close()
            self._client = None
        self._artifacts.close()

    def __del__(self):
        try: self.close()
        except Exception: pass
