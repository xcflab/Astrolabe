"""Exercise the real Protenix evaluator stages with a CPU mock runner."""
from types import SimpleNamespace
from unittest import mock

import torch


def test_forward_updates_length_and_finalize_retains_owner_atom_array():
    from astrolabe.evaluators.protenix import ProtenixEvaluator
    runner = SimpleNamespace(configs=SimpleNamespace(seed=11), dumper=SimpleNamespace(dump=mock.Mock()))
    runner.update_model_configs = mock.Mock()
    runner.predict = mock.Mock(return_value={'prediction': torch.tensor([42])})
    data = {'N_token': torch.tensor(9), 'entity_poly_type': {'1': 'protein'},
            'input_feature_dict': {'atom_to_token_idx': torch.tensor([0])}}
    atom_array = object()
    with mock.patch('astrolabe.evaluators.protenix.update_inference_configs', return_value={'length': 9}) as update:
        result = ProtenixEvaluator.forward_prepared(runner, data)
        update.assert_called_once_with(runner.configs, 9)
        runner.update_model_configs.assert_called_once_with({'length': 9})
        runner.predict.assert_called_once_with(data)
    out = ProtenixEvaluator.finalize_prediction(runner, 'step_001', 'A', data, atom_array, result)
    assert out is result
    assert out['atom_to_residue'] is data['input_feature_dict']['atom_to_token_idx']
    assert runner.dumper.dump.call_args.kwargs['atom_array'] is atom_array
    assert runner.dumper.dump.call_args.kwargs['seed'] == 11
