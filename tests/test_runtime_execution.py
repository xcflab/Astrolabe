import os
import pathlib
import subprocess
import sys
import unittest

from astrolabe.runtime.client import DaemonClient, OutcomeUnknown, WorkerError
from astrolabe.runtime.cpu import CPUExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
IDENTITY = {'environment': 'test', 'code': 'test', 'weights': 'none', 'device': 'cpu'}
COMMAND = [sys.executable, str(ROOT / 'tests/runtime_fixture.py')]


class RuntimeTest(unittest.TestCase):
    def test_resident_process_and_request_ids(self):
        with DaemonClient(COMMAND, IDENTITY) as client:
            a = client.call({'value': 1}, request_id='one', timeout=3)
            b = client.call({'value': 2}, request_id='two', timeout=3)
            self.assertEqual(a['pid'], b['pid'])
            self.assertEqual(a['pid'], client.hello['pid'])
            self.assertEqual(client.hello['load_count'], 1)
            self.assertEqual([a['value'], b['value']], [1, 2])
            with self.assertRaises(ValueError):
                client.call({}, request_id='one')

    def test_bad_identity_and_uncertain_timeout(self):
        with self.assertRaises(WorkerError):
            DaemonClient(COMMAND, {**IDENTITY, 'weights': 'different'})
        with DaemonClient(COMMAND, IDENTITY) as client:
            with self.assertRaises(OutcomeUnknown):
                client.call({'sleep': 2}, timeout=0.1)
            self.assertIsNotNone(client.process.poll())

    def test_cpu_parallel_real_processes_and_bounded_inputs(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = 'sentinel'
        with CPUExecutor(2, max_pending_bytes=1024) as executor:
            a, b = list(executor.map('runtime_fixture:cpu_work', [
                {'rounds': 150000, 'delay': .25}, {'rounds': 150000, 'delay': .25}]))
            self.assertNotEqual(a['pid'], b['pid'])
            self.assertLess(max(a['start'], b['start']), min(a['end'], b['end']))
            self.assertEqual(a['digest'], b['digest'])
            self.assertEqual([a['device'], b['device']], ['', ''])
            c, d = list(executor.map('runtime_fixture:cpu_work', [
                {'rounds': 150000, 'delay': .1}, {'rounds': 150000, 'delay': .1}]))
            self.assertEqual({a['pid'], b['pid']}, {c['pid'], d['pid']})
            with self.assertRaises(ValueError):
                list(executor.map('runtime_fixture:cpu_work', [{'x': 'a' * 1025}]))


if __name__ == '__main__':
    unittest.main()


def test_session_broker_shares_resident_worker(monkeypatch, tmp_path):
    from astrolabe.runtime.session import BrokerSession
    with BrokerSession() as session:
        monkeypatch.setenv('ASTROLABE_EXECUTION_ENDPOINT', session.endpoint)
        monkeypatch.setenv('ASTROLABE_EXECUTION_TOKEN', session.token)
        a = DaemonClient(COMMAND, IDENTITY)
        b = DaemonClient(COMMAND, IDENTITY)
        try:
            assert a.hello['pid'] == b.hello['pid']
            assert a.hello['load_count'] == b.hello['load_count'] == 1
            first = a.call({'value': 1}, timeout=3)
            assert first['pid'] == b.call({'value': 2}, timeout=3)['pid']
            assert first['artifact_root'] == session.directory.name
            with __import__('pytest').raises(WorkerError, match='different launch'):
                DaemonClient(COMMAND + ['different'], IDENTITY)
            with __import__('pytest').raises(WorkerError, match='resident model limit'):
                DaemonClient(COMMAND, {**IDENTITY, 'weights': 'other'})
        finally:
            a.close()
            b.close()


def test_cpu_output_bound_and_cleanup():
    import pytest
    executor = CPUExecutor(1, max_pending_bytes=1024)
    processes = [c.process for c in executor._clients]
    try:
        with pytest.raises(WorkerError, match='result exceeds'):
            list(executor.map('runtime_fixture:cpu_big_result', [{'size': 1024}]))
    finally:
        executor.close()
    assert all(p.poll() is not None for p in processes)


def test_cpu_timeout_stops_worker():
    import pytest
    executor = CPUExecutor(1, max_pending_bytes=1024, task_timeout=.1)
    process = executor._clients[0].process
    try:
        with pytest.raises(OutcomeUnknown):
            list(executor.map('runtime_fixture:cpu_work', [{'rounds': 1, 'delay': 2}]))
    finally:
        executor.close()
    assert process.poll() is not None


def test_broker_timeout_fences_device(monkeypatch):
    import pytest
    from astrolabe.runtime.session import BrokerSession
    with BrokerSession() as session:
        monkeypatch.setenv('ASTROLABE_EXECUTION_ENDPOINT', session.endpoint)
        monkeypatch.setenv('ASTROLABE_EXECUTION_TOKEN', session.token)
        client = DaemonClient(COMMAND, IDENTITY)
        with pytest.raises(OutcomeUnknown):
            client.call({'sleep': 2}, timeout=.1)
        with pytest.raises(OutcomeUnknown, match='fenced'):
            client.call({'value': 1}, timeout=2)


def test_unsupported_parallel_component_is_rejected():
    import pytest
    from omegaconf import OmegaConf
    from astrolabe.runtime.session import validate_parallel
    cfg = OmegaConf.create({'stages': {'default': {'mutators': ['bad'], 'evaluator': 'dummy'}},
                            'mutator': {'mutator_terms': {'bad': {'_type': 'LigandMPNN'}}},
                            'evaluator': {'evaluator_terms': {'dummy': {'_type': 'dummy'}}}})
    with pytest.raises(ValueError, match='does not support mutator'):
        validate_parallel(cfg)


def test_broker_serializes_distinct_workers_on_same_device(monkeypatch, tmp_path):
    import json
    from concurrent.futures import ThreadPoolExecutor
    from astrolabe.runtime.session import BrokerSession
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'mock-device')
    with BrokerSession(max_resident_models=2) as session:
        monkeypatch.setenv('ASTROLABE_EXECUTION_ENDPOINT', session.endpoint)
        monkeypatch.setenv('ASTROLABE_EXECUTION_TOKEN', session.token)
        identity1 = {**IDENTITY, 'device': {'platform': 'mock', 'visible': 'mock-device'}, 'weights': 'one'}
        identity2 = {**IDENTITY, 'device': {'platform': 'mock', 'visible': 'mock-device'}, 'weights': 'two'}
        a = DaemonClient(COMMAND, identity1, env={'RUNTIME_TEST_IDENTITY': json.dumps(identity1)})
        b = DaemonClient(COMMAND, identity2, env={'RUNTIME_TEST_IDENTITY': json.dumps(identity2)})
        assert a.hello['pid'] != b.hello['pid']
        log = tmp_path / 'forward.jsonl'
        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(a.call, {'sleep': .2, 'event_log': str(log), 'owner': 1}, timeout=3)
            f2 = pool.submit(b.call, {'sleep': .2, 'event_log': str(log), 'owner': 2}, timeout=3)
            f1.result()
            f2.result()
        events = [json.loads(s) for s in log.read_text().splitlines()]
        assert [x['kind'] for x in events] == ['forward_start', 'forward_end', 'forward_start', 'forward_end']


def test_device_probe_uses_same_admission(monkeypatch, tmp_path):
    import json
    from concurrent.futures import ThreadPoolExecutor
    from astrolabe.runtime.admission import run_probe
    from astrolabe.runtime.session import BrokerSession
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'mock-device')
    with BrokerSession() as session:
        monkeypatch.setenv('ASTROLABE_EXECUTION_ENDPOINT', session.endpoint)
        monkeypatch.setenv('ASTROLABE_EXECUTION_TOKEN', session.token)
        identity = {**IDENTITY, 'device': {'platform': 'mock', 'visible': 'mock-device'}}
        client = DaemonClient(COMMAND, identity, env={'RUNTIME_TEST_IDENTITY': json.dumps(identity)})
        log = tmp_path / 'device-events.jsonl'
        probe_env = dict(os.environ, RUNTIME_PROBE_LOG=str(log))
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(client.call, {'sleep': .2, 'event_log': str(log)}, timeout=5)
            b = pool.submit(run_probe, [sys.executable, str(ROOT / 'tests/runtime_probe_fixture.py')],
                            env=probe_env, timeout=5)
            a.result()
            assert b.result() == {'visible': 'mock-device'}
        events = [json.loads(s)['kind'] for s in log.read_text().splitlines()]
        assert events in (['probe_start', 'probe_end', 'forward_start', 'forward_end'],
                          ['forward_start', 'forward_end', 'probe_start', 'probe_end'])


def test_parent_exit_interrupts_active_worker_without_replay(tmp_path):
    import json
    import time
    events, outcome = tmp_path / 'events', tmp_path / 'outcome'
    proc = subprocess.run([sys.executable, str(ROOT / 'tests/runtime_parent_fixture.py'),
                           str(events), str(outcome), str(ROOT / 'tests/runtime_fixture.py')],
                          env=os.environ.copy(), stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, timeout=15)
    assert proc.returncode == 0
    deadline = time.monotonic() + 10
    while not outcome.exists() and time.monotonic() < deadline:
        time.sleep(.05)
    assert outcome.read_text() == 'OutcomeUnknown'
    records = [json.loads(line) for line in events.read_text().splitlines()]
    assert [r['kind'] for r in records] == ['forward_start']
    pid = records[0]['pid']
    while time.monotonic() < deadline:
        stat = pathlib.Path(f'/proc/{pid}/stat')
        if not stat.exists() or stat.read_text().split()[2] == 'Z':
            break
        time.sleep(.05)
    else:
        raise AssertionError('worker survived owner exit')


def test_cli_isolated_repeat_matches_legacy(tmp_path):
    """Exercise the public CLI and compare real trajectory score/history rows."""
    config = tmp_path / 'run.yaml'
    config.write_text('''
task_name: runtime_smoke
out_dir: OUTPUT
raw_model_results_dir: RAW
repeats: 1
seed: 17
steps: 2
ananas_path: null
seqs: {A: AAAA}
mutator:
  MCMC: {mutation_method: random, select_positions: random}
  mutator_terms: {MCMC: {_type: MCMC}}
evaluator:
  evaluator_terms: {Dummy: {_type: dummy}}
loss:
  loss_terms:
    plddt: {_type: sequence_residue_fraction, included_protomers: A, residues: A}
stages:
  default:
    oligo: [A]
    evaluator: Dummy
    mutation_num: 1
    mutators: [MCMC]
    mutator_weights: [1.0]
    weighted_score_expression: plddt
    early_stopping: null
'''.replace('OUTPUT', str(tmp_path / 'out')).replace('RAW', str(tmp_path / 'raw')))
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    scores = []
    for enabled in ('false', 'true'):
        out = tmp_path / enabled
        cmd = [sys.executable, '-m', 'astrolabe', 'run', '--standalone', str(config),
               '--overrides', f'execution.enabled={enabled} out_dir={out}']
        subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=90)
        scores.append((out / 'runtime_smoke/001_run/scores.csv').read_bytes())
    assert scores[0] == scores[1]


def test_real_cli_parallel_repeat_broker_and_scientific_rows(tmp_path):
    import json
    config = tmp_path / 'parallel.yaml'
    config.write_text(f"""
task_name: parallel_smoke
out_dir: {tmp_path / 'unused'}
raw_model_results_dir: {tmp_path / 'raw'}
repeats: 2
seed: 17
steps: 2
ananas_path: null
seqs: {{A: AAAA}}
mutator:
  MCMC: {{mutation_method: random, select_positions: random}}
  mutator_terms: {{MCMC: {{_type: MCMC}}}}
evaluator:
  evaluator_terms: {{Dummy: {{_type: dummy}}}}
loss:
  loss_terms:
    plddt: {{_type: sequence_residue_fraction, included_protomers: A, residues: A}}
stages:
  default:
    oligo: [A]
    evaluator: Dummy
    mutation_num: 1
    mutators: [MCMC]
    mutator_weights: [1.0]
    weighted_score_expression: plddt
    early_stopping: null
""")
    root_env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    log = tmp_path / 'events.jsonl'
    rows = []
    for enabled in (False, True):
        out = tmp_path / ('runtime' if enabled else 'legacy')
        env = root_env.copy()
        if enabled:
            env.update(ASTROLABE_TEST_BROKER='1', ASTROLABE_TEST_EVENTS=str(log),
                       ASTROLABE_TEST_FIXTURE=str(ROOT / 'tests/runtime_fixture.py'),
                       PYTHONPATH=str(ROOT / 'tests/runtime_hook') + os.pathsep + root_env['PYTHONPATH'])
        overrides = f'execution.enabled={str(enabled).lower()} execution.max_active={2 if enabled else 1} out_dir={out}'
        subprocess.run([sys.executable, '-m', 'astrolabe', 'run', '--standalone', str(config),
                        '--overrides', overrides], cwd=ROOT, env=env, check=True,
                       capture_output=True, text=True, timeout=120)
        by_seed = {}
        for provenance in out.glob('parallel_smoke/*_run/provenance.json'):
            record = json.loads(provenance.read_text())
            assert record['state'] == 'completed'
            by_seed[record['run']['seed']] = (provenance.parent / 'scores.csv').read_bytes()
        rows.append(by_seed)
    assert set(rows[0]) == {17, 18}
    assert rows[0] == rows[1]
    events = [json.loads(line) for line in log.read_text().splitlines()]
    residents = [e for e in events if e['kind'] == 'resident']
    assert len({e['owner'] for e in residents}) == 2
    assert len({e['worker'] for e in residents}) == 1
    assert {e['loads'] for e in residents} == {1}
    prepares = {}
    forwards = {}
    overlap = False
    for event in events:
        owner, kind = event.get('owner'), event['kind']
        if kind == 'prepare_start':
            prepares[owner] = event['time']
        elif kind == 'prepare_end':
            start = prepares.pop(owner)
            for other, (begin, end) in forwards.items():
                if owner != other and max(start, begin) < min(event['time'], end):
                    overlap = True
        elif kind == 'forward_start':
            forwards[owner] = (event['time'], float('inf'))
        elif kind == 'forward_end':
            forwards[owner] = (forwards[owner][0], event['time'])
    # Also compare completed prepare intervals against completed forward intervals,
    # irrespective of the order in which their end records were observed.
    prepare_intervals, forward_intervals, starts = [], [], {}
    for e in events:
        kind, owner = e['kind'], e.get('owner')
        if kind in ('prepare_start', 'forward_start'):
            starts[(kind.split('_')[0], owner)] = e['time']
        elif kind in ('prepare_end', 'forward_end'):
            phase = kind.split('_')[0]
            (prepare_intervals if phase == 'prepare' else forward_intervals).append(
                (owner, starts.pop((phase, owner)), e['time']))
    assert overlap or any(a != b and max(s, u) < min(t, v)
                          for a, s, t in prepare_intervals
                          for b, u, v in forward_intervals)
