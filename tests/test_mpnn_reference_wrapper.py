import numpy as np
import pytest
import torch
from test_ligandmpnn_device_binding import _args
from test_multistate_proteinmpnn import _write_two_chain_pdb
from astrolabe.mutators.wrappers.ligandmpnn_run import LigandMPNN


class ReferenceModel:
    def load_state_dict(self, _state): pass
    def to(self, _device): return self
    def eval(self): return self
    def score(self, features, use_sequence):
        assert use_sequence is True
        self.features = features
        seq = features['S']
        logits = torch.zeros((*seq.shape, 21), dtype=torch.float32)
        logits[..., 0] = 1
        order = torch.argsort((features['chain_mask'] + .0001) * features['randn'])
        return {'S':seq, 'logits':logits, 'log_probs':torch.log_softmax(logits, -1), 'decoding_order':order}


def test_wrapper_teacher_forcing_uses_reference_bias_mask_and_ranks(monkeypatch, tmp_path):
    model = ReferenceModel()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr('astrolabe.mutators.wrappers.ligandmpnn_run.torch.load', lambda *_a, **_k: {'num_edges':30, 'model_state_dict':{}})
    monkeypatch.setattr('astrolabe.mutators.wrappers.ligandmpnn_run.ProteinMPNN', lambda **_k: model)
    args = _args(tmp_path)
    args.model_type = 'protein_mpnn'
    args.temperature = 2.0
    args.bias_AA = 'C:2.0'
    runner = LigandMPNN(args)
    path = tmp_path/'AB.pdb'
    _write_two_chain_pdb(path,'ACD','EF')
    result = runner.design_sequences(pdb_path=str(path), fixed_residues='A2 B1 B2', redesigned_residues='',
        distribution_semantics='shared_reference_teacher_forcing', reference_decoding_ranks=[1,2,3,4,5])
    assert result['distribution_semantics'] == 'shared_reference_teacher_forcing'
    assert result['conditioning_reference'].shape == (1,5)
    assert result['chain_letters'] == ('A','A','A','B','B')
    assert result['decoding_order'].tolist() == [[1,3,4,0,2]]
    probs = result['sampling_probs'].numpy()[0]
    assert probs[0,1] / probs[0,0] == pytest.approx(np.exp((2-1)/2), rel=1e-5)
    assert probs[1,1] == 1.0 and probs[1].sum() == 1.0
    assert torch.equal(model.features['S'], result['conditioning_reference'])


def test_real_wrapper_teacher_forcing_through_public_mutator(monkeypatch, tmp_path):
    from test_mpnn_sync_policy import setup_case
    model = ReferenceModel()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr('astrolabe.mutators.wrappers.ligandmpnn_run.torch.load', lambda *_a, **_k: {'num_edges':30, 'model_state_dict':{}})
    monkeypatch.setattr('astrolabe.mutators.wrappers.ligandmpnn_run.ProteinMPNN', lambda **_k: model)
    args = _args(tmp_path)
    args.model_type = 'protein_mpnn'
    wrapper = LigandMPNN(args)
    protomers, scores, mutator = setup_case(tmp_path, monkeypatch, 'probability_mean')
    mutator.synchronization_policy.validate_backend(wrapper)
    mutator.runner = wrapper
    original = mutator.compute_mutation
    captured = {}
    def capture(features):
        captured.update(original(features))
        return captured
    monkeypatch.setattr(mutator, 'compute_mutation', capture)
    np.random.seed(25)
    mutator.mutate(protomers, scores, {'oligo':['AB','AC'], 'ignored_complexes':[], 'negative_design_complexes':[]})
    assert captured['AB']['distribution_semantics'] == captured['AC']['distribution_semantics'] == 'shared_reference_teacher_forcing'
    assert captured['AB']['conditioning_fingerprint'] == captured['AC']['conditioning_fingerprint']
    assert captured['AB']['conditioning_reference'][0,:3].tolist() == captured['AC']['conditioning_reference'][0,:3].tolist()
    assert captured['AB']['chain_mask'][0,:3].tolist() == [1,0,1]


def test_teacher_forcing_executes_actual_mpnn_model_cpu(monkeypatch, tmp_path):
    from astrolabe.mutators.wrappers.ligandmpnn_utils import ProteinMPNN
    torch.manual_seed(123)
    random_model = ProteinMPNN(k_neighbors=8, model_type='protein_mpnn')
    state = random_model.state_dict()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr('astrolabe.mutators.wrappers.ligandmpnn_run.torch.load', lambda *_a, **_k: {'num_edges':8, 'model_state_dict':state})
    args = _args(tmp_path)
    args.model_type = 'protein_mpnn'
    runner = LigandMPNN(args)
    path = tmp_path/'AB.pdb'
    _write_two_chain_pdb(path,'ACD','EF')
    result = runner.design_sequences(pdb_path=str(path), fixed_residues='A2 B1 B2', redesigned_residues='',
        distribution_semantics='shared_reference_teacher_forcing', reference_decoding_ranks=[1,2,3,4,5])
    values = result['sampling_probs'].numpy()
    assert values.shape == (1,5,20)
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values.sum(axis=-1), 1, atol=1e-6)
    assert result['decoding_order'].tolist() == [[1,3,4,0,2]]


def test_two_context_public_mutation_with_actual_mpnn_cpu(monkeypatch, tmp_path):
    from test_mpnn_sync_policy import setup_case
    from astrolabe.core import global_config
    from astrolabe.mutators.wrappers.ligandmpnn_utils import ProteinMPNN
    torch.manual_seed(123)
    state = ProteinMPNN(k_neighbors=8, model_type='protein_mpnn').state_dict()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr('astrolabe.mutators.wrappers.ligandmpnn_run.torch.load', lambda *_a, **_k: {'num_edges':8, 'model_state_dict':state})
    args = _args(tmp_path)
    args.model_type = 'protein_mpnn'
    args.temperature = 1.0
    runner = LigandMPNN(args)
    protomers, scores, mutator = setup_case(tmp_path, monkeypatch, 'probability_mean')
    mutator.runner = runner
    runtime = {'oligo':['AB','AC'], 'ignored_complexes':[], 'negative_design_complexes':[]}
    np.random.seed(17)
    global_config.segment_manager.begin_pre_update_step()
    mutator.mutate(protomers, scores, runtime)
    provenance = runtime['mutator_synchronization_provenance'][0]
    assert provenance['distribution_semantics'] == 'shared_reference_teacher_forcing'
    assert provenance['conditioning_fingerprints']['AB'] == provenance['conditioning_fingerprints']['AC']
    global_config.segment_manager.preapply_pending_mutations()
    assert protomers['A'].sequence[1] == 'C'
    global_config.segment_manager.finish_pre_update_step(accepted=False)
    assert protomers['A'].sequence == 'ACD'
    global_config.reset()


def test_reference_mapping_shared_segment_across_different_protomers(monkeypatch, tmp_path):
    from test_mpnn_sync_policy import setup_case
    from astrolabe.core import global_config
    from astrolabe.core.protomer_config_parser import Protomer
    from astrolabe.evaluators.utils import StandardScores
    from astrolabe.mutators.synchronization import SynchronizationPolicy
    model = ReferenceModel()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr('astrolabe.mutators.wrappers.ligandmpnn_run.torch.load', lambda *_a, **_k: {'num_edges':30, 'model_state_dict':{}})
    monkeypatch.setattr('astrolabe.mutators.wrappers.ligandmpnn_run.ProteinMPNN', lambda **_k: model)
    args = _args(tmp_path)
    args.model_type = 'protein_mpnn'
    wrapper = LigandMPNN(args)
    protomers, scores, mutator = setup_case(tmp_path, monkeypatch, 'probability_mean')
    segment = global_config.segments['A_1']
    global_config.segments['shared'] = segment
    protomers['X'] = Protomer('X', 'shared', 'protein')
    xc = tmp_path/'XC.pdb'
    _write_two_chain_pdb(xc,'A','GHIK')
    scores = StandardScores(names=['AB','XC'], plddt={'AB':[.8], 'XC':[.8]}, complex_paths={'AB':scores.complex_paths['AB'],'XC':xc})
    mutator.synchronization_policy = SynchronizationPolicy('separate','probability_mean',{'AB':1,'XC':2})
    mutator.runner = wrapper
    calls = []
    original = segment.sample_mutations
    def count():
        calls.append(1)
        return original()
    monkeypatch.setattr(segment, 'sample_mutations', count)
    runtime = {'oligo':['AB','XC'], 'ignored_complexes':[], 'negative_design_complexes':[]}
    np.random.seed(3)
    mutator.mutate(protomers, scores, runtime)
    assert len(calls) == 1
    assert runtime['mutator_synchronization_provenance'][0]['conditioning_fingerprints']['AB'] == runtime['mutator_synchronization_provenance'][0]['conditioning_fingerprints']['XC']
    global_config.reset()


def test_two_context_native_joint_with_actual_mpnn_cpu(monkeypatch, tmp_path):
    from test_mpnn_sync_policy import setup_case
    from astrolabe.core import global_config
    from astrolabe.mutators.wrappers.ligandmpnn_utils import ProteinMPNN
    torch.manual_seed(342)
    state = ProteinMPNN(k_neighbors=8, model_type='protein_mpnn').state_dict()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr('astrolabe.mutators.wrappers.ligandmpnn_run.torch.load', lambda *_a, **_k: {'num_edges':8, 'model_state_dict':state})
    args = _args(tmp_path)
    args.model_type = 'protein_mpnn'
    args.temperature = 1.0
    runner = LigandMPNN(args)
    protomers, scores, mutator = setup_case(tmp_path, monkeypatch, 'native_tie')
    mutator.runner = runner
    runtime = {'oligo':['AB','AC'], 'ignored_complexes':[], 'negative_design_complexes':[]}
    np.random.seed(13)
    global_config.segment_manager.begin_pre_update_step()
    mutator.mutate(protomers, scores, runtime)
    assert runtime['mutator_synchronization_provenance'][0]['distribution_semantics'] == 'native_joint_sequence'
    global_config.segment_manager.preapply_pending_mutations()
    assert protomers['A'].sequence[1] == 'C'
    global_config.segment_manager.finish_pre_update_step(accepted=False)
    assert protomers['A'].sequence == 'ACD'
    global_config.reset()
