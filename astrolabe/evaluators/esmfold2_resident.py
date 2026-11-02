"""Cross-interpreter ESMFold2 session using the shared runtime transport."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid
from astrolabe.evaluators.worker_identity import identity, probe_device
from astrolabe.evaluators.worker_session import WorkerArtifacts

BROKER_SHARED_ROOT = True
BROKER_ADMITTED_PROBE = True


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
        if runner.seed is None: raise ValueError('resident ESMFold2 requires an explicit seed')
        esm_root = Path(runner.esm_repo_path).resolve(strict=True)
        transformer_root = Path(runner.transformers_repo_path).resolve(strict=True)
        esm_code = esm_root / 'esm'
        transformer_code = transformer_root / 'src' / 'transformers'
        if not esm_code.is_dir() or not transformer_code.is_dir():
            raise ValueError('ESMFold2 requires source checkouts containing esm/ and src/transformers/')
        base = Path(__file__).resolve().parent
        code = [str(base / name) for name in ('tree_codec.py', 'request_path.py', 'worker_session.py', 'device_probe.py', 'worker_identity.py', 'wrappers/esmfold2.py', 'wrappers/esmfold2_daemon.py')] + [str(esm_code), str(transformer_code)]
        weights = [str(Path(model).resolve(strict=True)), str(Path(esmc).resolve(strict=True))]
        env = {'PYTHONPATH': os.pathsep.join([str(Path(p).resolve()) for p in sys.path if p])}
        for key in ('GH_ESMFOLD2_MODEL_SNAPSHOT', 'GH_ESMFOLD2_ESMC_SNAPSHOT', 'ESMCFOLD_CCD_PATH'):
            if key in os.environ: env[key] = os.environ[key]
        device = probe_device(python, 'esmfold2', runner.device, env=env)
        self._artifacts = WorkerArtifacts()
        root = self._artifacts.root
        fixed = {k: v for k, v in payload.items() if k not in ('jobs', 'output_dir')}
        fixed['device'] = device['name']
        fixed['_verify_sources'] = True
        semantics = {k: v for k, v in fixed.items() if k not in ('esm_repo_path', 'transformers_repo_path', 'cache_dir', '_verify_sources')}
        expected = identity(python, code, weights, device, semantics)
        spec = {'payload': fixed, 'model_identity': semantics, 'code_paths': code, 'weight_paths': weights, 'device': device}
        spec_path = self._artifacts.specification(expected, spec)
        self._client = DaemonClient([python, '-m', 'astrolabe.evaluators.wrappers.esmfold2_daemon', str(spec_path)], expected, env=env, startup_timeout=600)
        self.timeout = runner.timeout_seconds

    def run(self, payload, output_dir):
        root = self._artifacts.root
        # Output must outlive prepare_output and must not alias prior requests.
        target = Path(output_dir).resolve() / ('request-' + uuid.uuid4().hex)
        target.mkdir(parents=True)
        with tempfile.TemporaryDirectory(prefix='request-', dir=root) as tmp:
            request = Path(tmp)
            (request / 'jobs.json').write_text(json.dumps(payload['jobs']))
            answer = self._client.call({'request_dir': str(request)}, timeout=self.timeout)
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
        self._artifacts.close()

    def __del__(self):
        try: self.close()
        except Exception: pass
