"""Real AF3 featurisation and lossless forward-bundle serialisation (CPU)."""
import json
import numpy as np


def test_real_af3_forward_feature_codec(tmp_path):
    from alphafold3.common import folding_input
    from alphafold3.model.components.utils import remove_invalidly_typed_feats
    from astrolabe.evaluators.alphafold3 import AlphaFold3Evaluator
    from astrolabe.evaluators.tree_codec import dump_tree, load_tree
    fold = folding_input.Input.from_json(json.dumps({
        'name': 'codec_canary', 'modelSeeds': [1729],
        'sequences': [{'protein': {'id': 'A', 'sequence': 'ACDEFGHIKLMNPQRSTVWY',
                                   'unpairedMsa': '', 'pairedMsa': '', 'templates': []}}],
        'dialect': 'alphafold3', 'version': 1,
    }))
    examples, _, _ = AlphaFold3Evaluator.prepare_features(fold, (32,))
    source = [remove_invalidly_typed_feats(example) for example in examples]
    assert source and source[0]
    dump_tree(tmp_path, source)
    actual = load_tree(tmp_path)
    assert len(actual) == len(source)
    for expected, restored in zip(source, actual):
        assert expected.keys() == restored.keys()
        for key, before in expected.items():
            after = restored[key]
            assert before.shape == after.shape, key
            assert before.dtype == after.dtype, key
            np.testing.assert_array_equal(before, after, err_msg=key)


def test_real_af3_features_jsonl_worker(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    import alphafold3
    import jax
    from alphafold3.common import folding_input
    from alphafold3.model.components.utils import remove_invalidly_typed_feats
    from astrolabe.evaluators.alphafold3 import AlphaFold3Evaluator
    from astrolabe.evaluators.tree_codec import dump_tree, load_tree
    from astrolabe.evaluators.worker_identity import identity
    source = Path(alphafold3.__file__).resolve().parent
    weights = tmp_path / 'weights'; weights.write_text('mock-model')
    d = jax.local_devices(backend='cpu')[0]
    device = {'platform': str(d.platform), 'id': int(d.id), 'device_kind': str(d.device_kind), 'visible': os.environ.get('CUDA_VISIBLE_DEVICES')}
    config = {'model_dir': str(weights), 'flash_attention_implementation': 'mock', 'num_samples': 1, 'num_recycles': 2, 'save_embeddings': False}
    semantics = {k: config.get(k) for k in ('flash_attention_implementation', 'num_samples', 'num_recycles', 'save_embeddings', 'jax_compilation_cache_dir', 'refstructure_workers', 'refstructure_cache', 'refstructure_cache_max_entries', 'msa_cache', 'msa_cache_max_entries')}
    spec = {'code_paths': [str(source)], 'weight_paths': [str(weights)], 'device': device, 'model_config': config}
    (tmp_path / 'worker.json').write_text(json.dumps(spec))
    (tmp_path / 'sitecustomize.py').write_text('''\
from types import SimpleNamespace
import astrolabe.evaluators.wrappers.alphafold3 as wrapper
class Runner:
    _device = SimpleNamespace(platform=%r, id=%r, device_kind=%r)
    model_params = {'loaded': True}
    def set_cyclic_asym_ids(self, value): self.cyclic = value
    def set_cyclic_relative_patches(self, value): self.patches = value
    def run_inference(self, example, key): return {'tokens': example['aatype'], 'cyclic': self.cyclic}
wrapper.alphafold3_setup = lambda config: Runner()
''' % (device['platform'], device['id'], device['device_kind']))
    fold = folding_input.Input.from_json(json.dumps({'name': 'jsonl', 'modelSeeds': [1729], 'sequences': [{'protein': {'id': 'A', 'sequence': 'ACDEFGHIKLMNPQRSTVWY', 'unpairedMsa': '', 'pairedMsa': '', 'templates': []}}], 'dialect': 'alphafold3', 'version': 1}))
    examples, _, _ = AlphaFold3Evaluator.prepare_features(fold, (32,))
    numeric = [remove_invalidly_typed_feats(x) for x in examples]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), str(Path(__file__).resolve().parents[1]), os.environ.get('PYTHONPATH', '')]))
    worker = subprocess.Popen([sys.executable, '-m', 'astrolabe.evaluators.wrappers.af3_daemon', str(tmp_path / 'worker.json')], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    def receive():
        line = worker.stdout.readline()
        assert line, worker.stderr.read()
        return json.loads(line)
    try:
        hello = receive()
        assert hello['load_count'] == 1 and hello['pid'] == worker.pid
        assert hello['identity'] == identity(sys.executable, [str(source)], [str(weights)], device, semantics)
        for index in range(2):
            trajectory = tmp_path / ('trajectory-' + str(index)); trajectory.mkdir()
            request = trajectory / ('request-' + str(index)); request.mkdir(mode=0o700)
            dump_tree(request / 'input', {'seeds': [1729], 'examples': numeric, 'cyclic': (index,), 'patches': ()})
            worker.stdin.write(json.dumps({'type': 'call', 'request_id': str(index), 'epoch': 'test', 'payload': {'request_dir': str(request)}}) + '\n'); worker.stdin.flush()
            response = receive()
            assert response['type'] == 'result', response
            result = load_tree(request / 'output')
            np.testing.assert_array_equal(result[0]['tokens'], numeric[0]['aatype'])
            assert result[0]['cyclic'] == (index,)
        worker.stdin.write('{"type":"shutdown"}\n'); worker.stdin.flush()
        assert worker.wait(timeout=10) == 0
    finally:
        if worker.poll() is None: worker.kill(); worker.wait()
        worker.stdin.close(); worker.stdout.close(); worker.stderr.close()
