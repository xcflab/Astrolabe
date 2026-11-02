import numpy as np
import pytest
import torch
from astrolabe.evaluators.tree_codec import dump_tree, load_tree


@pytest.mark.parametrize('dtype', [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize('shape', [(), (0, 3), (2, 3)])
def test_torch_roundtrip_exact_bits(tmp_path, dtype, shape):
    n = int(np.prod(shape)) if shape else 1
    values = torch.arange(n).reshape(shape).to(dtype)
    dump_tree(tmp_path, {'tensor': values})
    result = load_tree(tmp_path)['tensor']
    assert result.dtype == values.dtype and result.shape == values.shape
    assert torch.equal(result.reshape(-1).view(torch.uint8), values.reshape(-1).view(torch.uint8))


def test_bfloat16_noncontiguous_and_unsupported_layout(tmp_path):
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.bfloat16).T
    dump_tree(tmp_path, x)
    y = load_tree(tmp_path)
    assert y.dtype == x.dtype and torch.equal(x, y)
    with pytest.raises(TypeError):
        dump_tree(tmp_path, torch.sparse_coo_tensor([[0]], [1.], (1,)))
