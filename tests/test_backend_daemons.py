"""CPU transport tests of production handlers with mock model sessions."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from astrolabe.evaluators.tree_codec import dump_tree, load_tree
from astrolabe.evaluators.worker_identity import identity


def test_codec_roundtrip_and_rejection(tmp_path):
    source = {'a': np.arange(6, dtype=np.int16).reshape(2, 3),
              'o': np.array(['x', b'y', None], dtype=object),
              't': (np.float32(1.5), b'\x00', ['hello'])}
    dump_tree(tmp_path, source)
    got = load_tree(tmp_path)
    np.testing.assert_array_equal(got['a'], source['a'])
    np.testing.assert_array_equal(got['o'], source['o'])
    assert type(got['t']) is tuple and type(got['t'][0]) is np.float32
    assert got['t'][1:] == source['t'][1:]
    with pytest.raises(TypeError): dump_tree(tmp_path, {'bad': object()})
    dump_tree(tmp_path, source)
    meta = json.loads((tmp_path / 'tree.json').read_text())
    meta['tree']['v'][0][1]['shape'] = [999]
    (tmp_path / 'tree.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError): load_tree(tmp_path)


@pytest.mark.parametrize('backend', ['af3', 'esmfold2'])
def test_resident_jsonl_lifecycle(tmp_path, backend):
    repo = Path(__file__).resolve().parents[1]
    runtime = repo / 'astrolabe' / 'runtime'
    if not runtime.exists():
        runtime = repo.parent / 'runtime' / 'astrolabe' / 'runtime'
    if not runtime.exists(): pytest.skip('A transport not yet available')
    hook = tmp_path / 'sitecustomize.py'
    hook.write_text('''\
import sys, types
from pathlib import Path
import astrolabe
astrolabe.__path__.append(%r)
if %r == 'af3':
    af = types.ModuleType('alphafold3'); af.__file__ = %r; sys.modules['alphafold3'] = af
    m = types.ModuleType('astrolabe.evaluators.wrappers.alphafold3')
    class Runner:
        _device = types.SimpleNamespace(platform='cpu', id=0, device_kind='mock')
        model_params = {'loaded': True}
    m.alphafold3_setup = lambda config: Runner()
    sys.modules[m.__name__] = m
    m = types.ModuleType('astrolabe.evaluators.alphafold3')
    class Evaluator:
        @staticmethod
        def forward_features(runner, seeds, examples, cyclic, patches):
            return [{'input': x, 'seed': s, 'cyclic': cyclic} for x,s in zip(examples,seeds)]
    m.AlphaFold3Evaluator = Evaluator
    sys.modules[m.__name__] = m
else:
    import astrolabe.evaluators.wrappers.esmfold2 as m
    m._worker_session = lambda payload: (None, None, None, 'cpu', None, None)
    def fake(payload, session):
        out = Path(payload['output_dir']); out.mkdir()
        jobs = []
        for job in payload['jobs']:
            p = out / (job['name'] + '_model.cif'); p.write_text(job['chains'][0]['sequence'])
            jobs.append({'name': job['name'], 'structure_path': str(p), 'plddt': [1]})
        return {'jobs': jobs}
    m._worker_run = fake
''' % (str(runtime.parent), backend, str(tmp_path / 'fake-af3' / '__init__.py')))
    code = tmp_path / 'fake-af3'; code.mkdir(); (code / '__init__.py').write_text('code')
    weights = tmp_path / 'weights'; weights.write_text('weights')
    device = {'platform': 'cpu', 'id': 0, 'device_kind': 'mock', 'visible': ''} if backend == 'af3' else {'name': 'cpu', 'visible': ''}
    model_conf = {'model_dir': str(weights), 'flash_attention_implementation': 'mock', 'num_samples': 1, 'num_recycles': 2, 'save_embeddings': False}
    semantics = {k: model_conf.get(k) for k in ('flash_attention_implementation', 'num_samples', 'num_recycles', 'save_embeddings', 'jax_compilation_cache_dir', 'refstructure_workers', 'refstructure_cache', 'refstructure_cache_max_entries', 'msa_cache', 'msa_cache_max_entries')} if backend == 'af3' else {}
    spec = {'code_paths': [str(code)], 'weight_paths': [str(weights)], 'device': device,
            'model_config': model_conf, 'payload': {}}
    (tmp_path / 'worker.json').write_text(json.dumps(spec))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), str(repo)]),
               CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    worker = subprocess.Popen([sys.executable, '-m', 'astrolabe.evaluators.wrappers.' + backend + '_daemon', str(tmp_path / 'worker.json')],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    def receive():
        line = worker.stdout.readline()
        assert line, worker.stderr.read()
        return json.loads(line)
    def call(rid, name):
        worker.stdin.write(json.dumps({'type': 'call', 'epoch': 'e', 'request_id': rid, 'payload': {'request_dir': name}}) + '\n'); worker.stdin.flush()
        return receive()
    try:
        hello = receive()
        assert hello['pid'] == worker.pid and hello['load_count'] == 1
        assert hello['identity'] == identity(sys.executable, [str(code)], [str(weights)], device, semantics)
        assert call('bad', str(tmp_path / 'request-missing'))['type'] == 'error'
        for idx, seq in enumerate(['AAA', 'CCC']):
            parent = tmp_path / ('trajectory-' + str(idx)); parent.mkdir()
            request = parent / ('request-' + str(idx)); request.mkdir(mode=0o700)
            if backend == 'af3':
                dump_tree(request / 'input', {'seeds': [idx], 'examples': [seq], 'cyclic': (idx,), 'patches': ()})
            else:
                (request / 'jobs.json').write_text(json.dumps([{'name': 'A', 'chains': [{'id': 'A', 'sequence': seq}]}]))
            assert call(str(idx), str(request))['type'] == 'result'
            if backend == 'af3': assert load_tree(request / 'output')[0] == {'input': seq, 'seed': idx, 'cyclic': (idx,)}
            else: assert (request / 'structures' / 'A_model.cif').read_text() == seq
        worker.stdin.write('{"type":"shutdown"}\n'); worker.stdin.flush()
        assert worker.wait(timeout=10) == 0
    finally:
        if worker.poll() is None: worker.kill(); worker.wait()
        worker.stdin.close(); worker.stdout.close(); worker.stderr.close()


def test_af3_cache_and_constraints_are_on_execution_path(monkeypatch):
    from types import SimpleNamespace
    from astrolabe.evaluators.alphafold3 import AlphaFold3Evaluator
    import astrolabe.evaluators.worker_identity as ids
    from astrolabe.evaluators.wrappers.alphafold3 import ModelRunner
    import alphafold3
    monkeypatch.setattr(ids, 'digest_path', lambda path: 'source-v1')
    count = []
    def prepare(feat, buckets, **kwargs):
        count.append(feat.sequence)
        return ([{'sequence': feat.sequence}], (), ())
    monkeypatch.setattr(AlphaFold3Evaluator, 'prepare_features', staticmethod(prepare))
    monkeypatch.setattr(AlphaFold3Evaluator, 'finalize_features', staticmethod(lambda runner, feat, examples, result: result))
    monkeypatch.setattr(ModelRunner, 'extract_structures', lambda *a, **kw: [])
    evaluator = AlphaFold3Evaluator(None, Path('.'), {'feature_cache': True})
    evaluator.resident = SimpleNamespace(forward=lambda seeds, examples, cyclic, patches: examples)
    evaluator.buckets = {'A': None}
    def feat(seq):
        return SimpleNamespace(sequence=seq, rng_seeds=[1], chains=[SimpleNamespace(id='A')],
                               to_json=lambda: json.dumps({'name': 'candidate', 'sequences': [seq]}))
    evaluator.run_function({'A': feat('AAA')})
    evaluator.run_function({'A': feat('AAA')})
    evaluator.run_function({'A': feat('AAC')})
    assert count == ['AAA', 'AAC']
    evaluator.af3_conf = {'daemon_constraints': [{'kind': 'hard', 'mechanism': 'contact'}]}
    with pytest.raises(ValueError, match='Unsupported constraint'):
        evaluator.run_function({'A': feat('AAC')})


def test_identity_depends_on_model_configuration_not_worktree_path(tmp_path):
    left = tmp_path / 'checkout-a'; right = tmp_path / 'checkout-b'
    left.mkdir(); right.mkdir()
    for directory in (left, right):
        (directory / 'runner.py').write_text('same code')
    weight = tmp_path / 'weight'; weight.write_text('checkpoint')
    cfg = {'num_recycles': 10, 'num_samples': 5, 'precision': 'bf16'}
    first = identity(sys.executable, [left / 'runner.py'], [weight], {'id': 0}, cfg)
    second = identity(sys.executable, [right / 'runner.py'], [weight], {'id': 0}, cfg)
    assert first == second
    for field in cfg:
        changed = dict(cfg, **{field: 'changed'})
        assert first != identity(sys.executable, [right / 'runner.py'], [weight], {'id': 0}, changed)
    weight.write_text('different checkpoint')
    assert first != identity(sys.executable, [right / 'runner.py'], [weight], {'id': 0}, cfg)


def test_device_probe_obeys_cpu_visibility(monkeypatch):
    from astrolabe.evaluators.worker_identity import probe_device
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
    assert probe_device(sys.executable, 'af3', env=env)['platform'] == 'cpu'
    assert probe_device(sys.executable, 'esmfold2', env=env)['name'] == 'cpu'


def test_broker_artifact_spec_is_stable_and_owned(monkeypatch, tmp_path):
    from astrolabe.evaluators.worker_session import WorkerArtifacts
    from astrolabe.evaluators.request_path import request_directory
    root = tmp_path / 'session'; root.mkdir(mode=0o700)
    monkeypatch.setenv('ASTROLABE_EXECUTION_ENDPOINT', 'broker')
    monkeypatch.setenv('ASTROLABE_EXECUTION_TOKEN', 'token')
    monkeypatch.setenv('ASTROLABE_EXECUTION_ARTIFACT_ROOT', str(root))
    first, second = WorkerArtifacts(), WorkerArtifacts()
    identity = {'model': 'x', 'config': {'recycles': 10}}
    spec = {'config': 1}
    assert first.specification(identity, spec) == second.specification(identity, spec)
    with pytest.raises(ValueError, match='incompatible'):
        second.specification(identity, {'config': 2})
    request = root / 'request-unique'; request.mkdir(mode=0o700)
    assert request_directory(str(request)) == request
    elsewhere = tmp_path / 'request-outside'; elsewhere.mkdir(mode=0o700)
    with pytest.raises(ValueError, match='outside'):
        request_directory(str(elsewhere))
    first.close(); second.close()
    assert root.exists()


def test_device_probe_broker_requires_admission(monkeypatch):
    from astrolabe.evaluators.worker_identity import probe_device
    import importlib.util
    if importlib.util.find_spec('astrolabe.runtime.admission'):
        pytest.skip('covered by runtime broker admission tests after integration')
    monkeypatch.setenv('ASTROLABE_EXECUTION_ENDPOINT', 'broker')
    monkeypatch.setenv('ASTROLABE_EXECUTION_TOKEN', 'token')
    with pytest.raises(RuntimeError, match='runtime.admission'):
        probe_device(sys.executable, 'af3')
