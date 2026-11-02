"""Owner-side AF3 resident forward client."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

from astrolabe.evaluators.tree_codec import dump_tree, load_tree
from astrolabe.evaluators.worker_identity import identity


class AF3Resident:
    def __init__(self, config):
        import alphafold3
        from omegaconf import OmegaConf
        from astrolabe.runtime.client import DaemonClient
        self._session = tempfile.TemporaryDirectory(prefix='astrolabe-af3-')
        root = Path(self._session.name).resolve()
        model_config = OmegaConf.to_container(config, resolve=True)
        code = [str(Path(__file__).resolve().parents[1]), str(Path(alphafold3.__file__).resolve().parent)]
        weights = [str(Path(model_config['model_dir']).expanduser().resolve(strict=True))]
        descriptor = model_config.get('daemon_device_identity')
        if not isinstance(descriptor, dict) or set(descriptor) != {'platform', 'id', 'device_kind'}:
            raise ValueError('AF3 daemon_device_identity must specify platform, id and device_kind')
        device = {**descriptor, 'visible': os.environ.get('CUDA_VISIBLE_DEVICES')}
        fingerprint = identity(sys.executable, code, weights, device)
        spec = {'session': str(root), 'model_config': model_config, 'code_paths': code, 'weight_paths': weights, 'device': device}
        path = root / 'worker.json'
        path.write_text(json.dumps(spec), encoding='utf-8')
        env = {'PYTHONPATH': os.pathsep.join([str(Path(p).resolve()) for p in sys.path if p])}
        self._client = DaemonClient([sys.executable, '-m', 'astrolabe.evaluators.wrappers.af3_daemon', str(path)], fingerprint, env=env, startup_timeout=float(model_config.get('daemon_startup_timeout', 600)))
        self.timeout = model_config.get('daemon_timeout')

    def forward(self, seeds, examples, cyclic, patches):
        from alphafold3.model.components.utils import remove_invalidly_typed_feats
        examples = [remove_invalidly_typed_feats(example) for example in examples]
        root = Path(self._session.name)
        with tempfile.TemporaryDirectory(prefix='request-', dir=root) as tmp:
            path = Path(tmp)
            dump_tree(path / 'input', {'seeds': list(seeds), 'examples': examples, 'cyclic': cyclic, 'patches': patches})
            answer = self._client.call({'request': path.name}, timeout=self.timeout, request_id=uuid.uuid4().hex)
            if answer != {'request': path.name}: raise ValueError('AF3 worker returned wrong request')
            return load_tree(path / 'output')

    def close(self):
        if getattr(self, '_client', None) is not None:
            self._client.close()
            self._client = None
        self._session.cleanup()

    def __del__(self):
        try: self.close()
        except Exception: pass
