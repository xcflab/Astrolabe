"""Cross-interpreter ESMFold2 session using the shared runtime transport."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid
from astrolabe.evaluators.worker_identity import identity


class ESMFold2Resident:
    def __init__(self, runner, payload):
        from astrolabe.runtime.client import DaemonClient
        python = shutil.which(runner.python_bin or sys.executable)
        if python is None: raise FileNotFoundError('ESMFold2 python executable not found')
        # Both models and both source checkouts must be locally pinned for a
        # content-verified resident identity; no implicit downloads are allowed.
        model = os.environ.get('GH_ESMFOLD2_MODEL_SNAPSHOT')
        esmc = os.environ.get('GH_ESMFOLD2_ESMC_SNAPSHOT')
        if not model or not esmc or not runner.esm_repo_path or not runner.transformers_repo_path:
            raise ValueError('resident ESMFold2 requires pinned model and ESMC snapshots and explicit esm/transformers source paths')
        if runner.model_name != "biohub/ESMFold2":
            raise ValueError('resident ESMFold2 requires the pinned biohub/ESMFold2 model')
        if not runner.local_files_only: raise ValueError('resident ESMFold2 requires local_files_only=true')
        if runner.device is None: raise ValueError('resident ESMFold2 requires explicit device')
        if not runner.daemon_device_kind: raise ValueError('resident ESMFold2 requires daemon_device_kind')
        if runner.seed is None: raise ValueError('resident ESMFold2 requires an explicit seed')
        esm_root = Path(runner.esm_repo_path).resolve(strict=True)
        transformer_root = Path(runner.transformers_repo_path).resolve(strict=True)
        esm_code = esm_root / 'esm'
        transformer_code = transformer_root / 'src' / 'transformers'
        if not esm_code.is_dir() or not transformer_code.is_dir():
            raise ValueError('ESMFold2 requires source checkouts containing esm/ and src/transformers/')
        code = [str(Path(__file__).resolve().parents[1]), str(esm_code), str(transformer_code)]
        weights = [str(Path(model).resolve(strict=True)), str(Path(esmc).resolve(strict=True))]
        device = {'name': runner.device, 'device_kind': runner.daemon_device_kind, 'visible': os.environ.get('CUDA_VISIBLE_DEVICES')}
        expected = identity(python, code, weights, device)
        self._session = tempfile.TemporaryDirectory(prefix='astrolabe-esmfold2-')
        root = Path(self._session.name).resolve()
        fixed = {k: v for k, v in payload.items() if k not in ('jobs', 'output_dir')}
        fixed['_verify_sources'] = True
        spec = {'session': str(root), 'payload': fixed, 'code_paths': code, 'weight_paths': weights, 'device': device}
        spec_path = root / 'worker.json'
        spec_path.write_text(json.dumps(spec))
        env = {'PYTHONPATH': os.pathsep.join([str(Path(p).resolve()) for p in sys.path if p])}
        self._client = DaemonClient([python, '-m', 'astrolabe.evaluators.wrappers.esmfold2_daemon', str(spec_path)], expected, env=env, startup_timeout=600)
        self.timeout = runner.timeout_seconds

    def run(self, payload, output_dir):
        root = Path(self._session.name)
        # Output must outlive prepare_output and must not alias prior requests.
        target = Path(output_dir).resolve() / ('request-' + uuid.uuid4().hex)
        target.mkdir(parents=True)
        with tempfile.TemporaryDirectory(prefix='request-', dir=root) as tmp:
            request = Path(tmp)
            (request / 'jobs.json').write_text(json.dumps(payload['jobs']))
            answer = self._client.call({'request': request.name}, timeout=self.timeout)
            if answer != {'request': request.name}: raise ValueError('ESMFold2 response request mismatch')
            result = json.loads((request / 'result.json').read_text())
            for job in result['jobs']:
                source = Path(job['structure_path']).resolve(strict=True)
                if source.parent != (request / 'structures').resolve(): raise ValueError('unsafe ESMFold2 output path')
                dest = target / source.name
                shutil.copyfile(source, dest)
                job['structure_path'] = str(dest)
            return result

    def close(self):
        if getattr(self, '_client', None) is not None:
            self._client.close(); self._client = None
        self._session.cleanup()

    def __del__(self):
        try: self.close()
        except Exception: pass
