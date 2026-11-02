"""Resident ESMFold2 worker using the existing inference pipeline."""
import contextlib
import json
from pathlib import Path
import sys


def main():
    import os
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = os.fdopen(protocol_fd, 'w', buffering=1)
    spec = json.loads(Path(sys.argv[1]).read_text())
    from astrolabe.evaluators.worker_identity import identity
    semantics = {k: v for k, v in spec['payload'].items() if k not in ('esm_repo_path', 'transformers_repo_path', 'cache_dir', '_verify_sources')}
    actual = identity(sys.executable, spec['code_paths'], spec['weight_paths'], spec['device'], semantics)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['device']['visible']:
        raise RuntimeError('visible device environment mismatch')
    with contextlib.redirect_stdout(sys.stderr):
        from astrolabe.evaluators.wrappers.esmfold2 import _worker_session, _worker_run
        if spec['payload'].get('_verify_sources'):
            from astrolabe.evaluators.wrappers.esmfold2 import _resolve_model_source, _resolve_esmc_snapshot_path
            model_source, pinned = _resolve_model_source(spec['payload']['model_name'])
            esmc_source = _resolve_esmc_snapshot_path()
            if not pinned or [str(Path(model_source).resolve()), str(esmc_source.resolve())] != spec['weight_paths']:
                raise RuntimeError('ESMFold2 loaded weight source identity mismatch')
        session = _worker_session(spec['payload'])
        actual_device = str(session[3])
        if actual_device != spec['device']['name']:
            raise RuntimeError(f'ESMFold2 device mismatch: {actual_device}')
        if spec['payload'].get('_verify_sources'):
            import torch
            actual_kind = torch.cuda.get_device_name(torch.device(actual_device)) if actual_device.startswith('cuda') else 'cpu'
            if actual_kind != spec['device']['device_kind']:
                raise RuntimeError(f'ESMFold2 device kind mismatch: {actual_kind}')
    from astrolabe.runtime.server import serve
    from astrolabe.evaluators.request_path import request_directory
    def handle(payload):
        request = request_directory(payload['request_dir'])
        jobs = json.loads((request / 'jobs.json').read_text())
        for job in jobs:
            name = job['name']
            if not isinstance(name, str) or not name or name in ('.', '..') or Path(name).name != name:
                raise ValueError('invalid job name')
        params = dict(spec['payload'], jobs=jobs, output_dir=str(request / 'structures'))
        with contextlib.redirect_stdout(sys.stderr):
            result = _worker_run(params, session=session)
        (request / 'result.json').write_text(json.dumps(result, allow_nan=False))
        return {'request': request.name}
    serve(handle, actual)


if __name__ == '__main__': main()
