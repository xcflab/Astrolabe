"""AF3 forward-only worker for the shared JSONL runtime."""
from __future__ import annotations
import contextlib
import json
from pathlib import Path
import sys


def main():
    import os
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = os.fdopen(protocol_fd, 'w', buffering=1)
    config_path = Path(sys.argv[1]).resolve(strict=True)
    config = json.loads(config_path.read_text())
    from astrolabe.evaluators.worker_identity import identity
    fields = ('flash_attention_implementation', 'num_samples', 'num_recycles', 'save_embeddings', 'jax_compilation_cache_dir', 'refstructure_workers', 'refstructure_cache', 'refstructure_cache_max_entries', 'msa_cache', 'msa_cache_max_entries')
    semantics = {key: config['model_config'].get(key) for key in fields}
    actual = identity(sys.executable, config['code_paths'], config['weight_paths'], config['device'], semantics)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != config['device']['visible']:
        raise RuntimeError('visible device environment mismatch')
    with contextlib.redirect_stdout(sys.stderr):
        from omegaconf import OmegaConf
        from astrolabe.evaluators.wrappers.alphafold3 import alphafold3_setup
        from astrolabe.evaluators.alphafold3 import AlphaFold3Evaluator
        import alphafold3
        if str(Path(alphafold3.__file__).resolve().parent) not in config['code_paths']:
            raise RuntimeError('AF3 imported an unexpected source checkout')
        if str(Path(config['model_config']['model_dir']).expanduser().resolve()) not in config['weight_paths']:
            raise RuntimeError('AF3 model directory identity mismatch')
        runner = alphafold3_setup(OmegaConf.create(config['model_config']))
        actual_device = {'platform': str(runner._device.platform), 'id': int(runner._device.id),
                         'device_kind': str(runner._device.device_kind),
                         'visible': os.environ.get('CUDA_VISIBLE_DEVICES')}
        if actual_device != config['device']:
            raise RuntimeError(f'AF3 device mismatch: {actual_device}')
        _ = runner.model_params
    from astrolabe.evaluators.tree_codec import load_tree, dump_tree
    from astrolabe.runtime.server import serve
    from astrolabe.evaluators.request_path import request_directory
    def handle(payload):
        request = request_directory(payload['request_dir'])
        batch = load_tree(request / 'input')
        with contextlib.redirect_stdout(sys.stderr):
            result = AlphaFold3Evaluator.forward_features(runner, batch['seeds'], batch['examples'], batch['cyclic'], batch['patches'])
        dump_tree(request / 'output', result)
        return {'request': request.name}
    serve(handle, actual)


if __name__ == '__main__': main()
