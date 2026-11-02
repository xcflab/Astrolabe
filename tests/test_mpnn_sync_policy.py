from pathlib import Path
import numpy as np
import pytest
from omegaconf import OmegaConf
from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.base import build_mutators
from astrolabe.utils.constant import aa_order
from test_multistate_proteinmpnn import _write_two_chain_pdb


class FakeRunner:
    supported_distribution_semantics = {"shared_reference_teacher_forcing"}
    instances = []
    def __init__(self, config):
        self.calls = []
        self.instances.append(self)
    def design_sequences(self, **features):
        self.calls.append(features)
        combined = 'symmetry_residues' in features and ',' in features['symmetry_residues']
        length = 12 if combined else (5 if str(features['pdb_path']).endswith('AB.pdb') else 7)
        probs = np.zeros((1, length, 20))
        chosen = 'C' if not combined and str(features['pdb_path']).endswith('AB.pdb') else 'D'
        probs[:, :, aa_order.index(chosen)] = 1.0
        if combined:
            # Joint sequence intentionally differs from every per-position distribution.
            generated = np.full((1, length), aa_order.index('C'), dtype=int)
            return {'sampling_probs':probs, 'generated_sequences':generated,
                    'chain_letters':tuple('AAABBCCCDDDD')}
        sequence = 'ACDEF' if length == 5 else 'ACDGHIK'
        reference = np.asarray([[aa_order.index(aa) for aa in sequence]])
        mask = np.asarray([[1, 0, 1] + [0] * (length - 3)])
        ranks = np.asarray(features['reference_decoding_ranks'])
        order = np.argsort((mask + 0.0001) * ranks)
        return {'sampling_probs':probs, 'conditioning_reference':reference, 'chain_mask':mask,
                'decoding_order':order, 'alphabet':tuple(aa_order[:20]),
                'chain_letters':tuple('AAABB' if length == 5 else 'AAABBBB'),
                'distribution_semantics':'shared_reference_teacher_forcing'}


def setup_case(tmp_path, monkeypatch, coupling):
    monkeypatch.setattr('astrolabe.mutators.wrappers.ligandmpnn_run.LigandMPNN', FakeRunner)
    global_config.reset()
    for key, value in [('segments', {}), ('named_files', {}), ('predefined_statement', {}),
                       ('exclude_AA', []), ('segment_manager', SegmentManager()),
                       ('out_dir', tmp_path), ('raw_model_results_dir', tmp_path / 'raw'),
                       ('enable_pyrosetta', False), ('skip_complex_poses', True)]:
        global_config.init_const(key, value)
    protomers = {'A': Protomer('A', 'A +C D', 'protein'),
                 'B': Protomer('B', '+EF', 'protein'), 'C': Protomer('C', '+GHIK', 'protein')}
    global_config.init_const('protomers', protomers)
    global_config.init_const('mutable_protomers', protomers)
    global_config.set_variable('mutation_num', 1)
    global_config.set_variable('step', 1)
    global_config.set_variable('ignored_complexes', [])
    global_config.set_variable('negative_design_complexes', [])
    ab, ac = tmp_path / 'AB.pdb', tmp_path / 'AC.pdb'
    _write_two_chain_pdb(ab, 'ACD', 'EF')
    _write_two_chain_pdb(ac, 'ACD', 'GHIK')
    scores = StandardScores(names=['AB','AC'], plddt={'AB':[.8], 'AC':[.8]}, complex_paths={'AB':ab,'AC':ac})
    term = {'_type':'ProteinMPNN', 'context_layout':'far_apart' if coupling == 'native_tie' else 'separate', 'coupling':coupling}
    if coupling == 'probability_mean': term['context_weights'] = {'AB':1, 'AC':2}
    cfg = OmegaConf.create({'enabled_aliases':['design'], 'mutator_terms':{'design':term}, 'LigandMPNN':{'model_type':'protein_mpnn'}})
    mutator = build_mutators(cfg, tmp_path)['design']
    mutator.mutation_weight = 1
    return protomers, scores, mutator


@pytest.mark.parametrize('coupling', ['native_tie', 'probability_mean'])
def test_builder_public_mutate_shared_fixed_and_rollback(tmp_path, monkeypatch, coupling):
    protomers, scores, mutator = setup_case(tmp_path, monkeypatch, coupling)
    manager = global_config.segment_manager
    segment = manager.segments['A_1']
    calls = []
    original_sample = segment.sample_mutations
    def counted_sample():
        calls.append(1)
        return original_sample()
    monkeypatch.setattr(segment, 'sample_mutations', counted_sample)
    np.random.seed(7)
    manager.begin_pre_update_step()
    mutator.mutate(protomers, scores, {'oligo':['AB','AC'], 'ignored_complexes':[], 'negative_design_complexes':[]})
    pending = [(name, mutation) for name, seg in manager.segments.items() for mutation in (seg.mutations or [])]
    assert len(pending) == 1
    assert len(calls) == 1
    if coupling == 'native_tie':
        assert pending[0][1][2] == 'C'
    else:
        assert pending[0][1][2] in {'C', 'D'}
        distribution = segment.probs[0][0]
        assert distribution[list(segment.aa_order).index('C')] == pytest.approx(1/3)
        assert distribution[list(segment.aa_order).index('D')] == pytest.approx(2/3)
    assert protomers['A'].sequence == 'ACD'
    manager.preapply_pending_mutations()
    assert protomers['A'].sequence[1] == 'C'
    assert protomers['A'].sequence != 'ACD'
    manager.finish_pre_update_step(accepted=False)
    assert protomers['A'].sequence == 'ACD'
    assert len(mutator.runner.calls) == (1 if coupling == 'native_tie' else 2)
    if coupling == 'native_tie':
        assert mutator.runner.calls[0]['symmetry_residues']
    global_config.reset()


def test_builder_rejects_ar_and_negative_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(FakeRunner, 'supported_distribution_semantics', set())
    with pytest.raises(ValueError, match='teacher-forcing'):
        setup_case(tmp_path, monkeypatch, 'probability_mean')
    from astrolabe.mutators.synchronization import SynchronizationPolicy
    with pytest.raises(ValueError, match='nonnegative'):
        SynchronizationPolicy(context_weights={'AB':1,'AC':-1})


@pytest.mark.parametrize('bad', ['reference', 'mask', 'order', 'alphabet', 'samples_only'])
def test_public_mutate_rejects_mismatched_conditioning(tmp_path, monkeypatch, bad):
    protomers, scores, mutator = setup_case(tmp_path, monkeypatch, 'probability_mean')
    original = mutator.runner.design_sequences
    def corrupted(**features):
        output = original(**features)
        if str(features['pdb_path']).endswith('AC.pdb'):
            if bad == 'reference': output['conditioning_reference'][0, 0] = aa_order.index('W')
            if bad == 'mask': output['chain_mask'][0, 0] = 0
            if bad == 'order':
                order = output['decoding_order'][0]
                a, b = np.where(order == 0)[0][0], np.where(order == 2)[0][0]
                order[a], order[b] = order[b], order[a]
            if bad == 'alphabet': output['alphabet'] = tuple(reversed(aa_order[:20]))
            if bad == 'samples_only': output.pop('conditioning_reference')
        return output
    monkeypatch.setattr(mutator.runner, 'design_sequences', corrupted)
    with pytest.raises(ValueError):
        mutator.mutate(protomers, scores, {'oligo':['AB','AC'], 'ignored_complexes':[], 'negative_design_complexes':[]})
    global_config.reset()
