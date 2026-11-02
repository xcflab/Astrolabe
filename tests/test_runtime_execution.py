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
            with self.assertRaises(ValueError):
                list(executor.map('runtime_fixture:cpu_work', [{'x': 'a' * 1025}]))


if __name__ == '__main__':
    unittest.main()


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
