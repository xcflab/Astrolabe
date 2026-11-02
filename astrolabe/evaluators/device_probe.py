"""Discover only devices visible to the allocated worker process."""
import contextlib
import json
import os
import sys


def discover(backend, requested=None):
    with contextlib.redirect_stdout(sys.stderr):
        if backend == 'af3':
            import jax
            try:
                devices = jax.local_devices(backend='gpu')
            except Exception:
                devices = jax.local_devices(backend='cpu')
            d = devices[0]  # exactly the selection used by alphafold3_setup
            return {'platform': str(d.platform), 'id': int(d.id),
                    'device_kind': str(d.device_kind), 'visible': os.environ.get('CUDA_VISIBLE_DEVICES')}
        if backend == 'esmfold2':
            import torch
            name = requested or ('cuda' if torch.cuda.is_available() else 'cpu')
            kind = torch.cuda.get_device_name(torch.device(name)) if name.startswith('cuda') else 'cpu'
            return {'name': name, 'device_kind': kind, 'visible': os.environ.get('CUDA_VISIBLE_DEVICES')}
    raise ValueError('unsupported backend')


if __name__ == '__main__':
    print(json.dumps(discover(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)))
