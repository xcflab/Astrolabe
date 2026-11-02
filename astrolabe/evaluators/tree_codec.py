"""Versioned, non-pickle feature tree codec for trusted private request dirs."""
from __future__ import annotations
import base64
import json
from pathlib import Path
import numpy as np


def dump_tree(root, value):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    arrays = {}
    def encode(x):
        if x is None or type(x) in (str, bool, int, float):
            return {'t': 'scalar', 'v': x}
        if type(x) is bytes:
            return {'t': 'bytes', 'v': base64.b64encode(x).decode('ascii')}
        if isinstance(x, np.generic):
            return {'t': 'numpy_scalar', 'v': encode(np.asarray(x))}
        if isinstance(x, np.ndarray):
            if x.dtype.hasobject:
                return {'t': 'objects', 'shape': list(x.shape),
                        'v': [encode(v) for v in x.flat]}
            if x.dtype.fields is not None:
                raise TypeError('structured dtypes are unsupported')
            key = 'a' + str(len(arrays))
            arrays[key] = x
            return {'t': 'array', 'key': key, 'shape': list(x.shape), 'dtype': x.dtype.str,
                    'nbytes': x.nbytes}
        if type(x) in (tuple, list):
            return {'t': 'tuple' if type(x) is tuple else 'list', 'v': [encode(v) for v in x]}
        if type(x) is dict and all(type(k) is str for k in x):
            return {'t': 'dict', 'v': [[k, encode(v)] for k, v in x.items()]}
        raise TypeError(f'unsupported feature type: {type(x)!r}')
    metadata = {'version': 1, 'tree': encode(value)}
    np.savez(root / 'arrays.npz', **arrays)
    (root / 'tree.json').write_text(json.dumps(metadata, allow_nan=False), encoding='utf-8')


def load_tree(root):
    root = Path(root)
    metadata = json.loads((root / 'tree.json').read_text(encoding='utf-8'))
    if metadata.get('version') != 1:
        raise ValueError('unsupported feature codec version')
    with np.load(root / 'arrays.npz', allow_pickle=False) as arrays:
        used = set()
        def decode(node):
            t = node['t']
            if t == 'scalar':
                v = node['v']
                if v is not None and type(v) not in (str, bool, int, float): raise ValueError('invalid scalar')
                return v
            if t == 'bytes': return base64.b64decode(node['v'], validate=True)
            if t == 'numpy_scalar':
                v = decode(node['v'])
                if not isinstance(v, np.ndarray) or v.shape != (): raise ValueError('invalid numpy scalar')
                return v[()]
            if t == 'array':
                key = node['key']
                if key in used or key not in arrays: raise ValueError('invalid array key')
                used.add(key)
                a = arrays[key]
                if a.dtype.hasobject or a.dtype.fields is not None or list(a.shape) != node['shape'] or a.dtype.str != node['dtype'] or a.nbytes != node['nbytes']:
                    raise ValueError('invalid array metadata')
                return a
            if t == 'objects':
                values = [decode(v) for v in node['v']]
                if any(v is not None and type(v) not in (str, bytes, bool, int, float) for v in values): raise ValueError('invalid object array value')
                return np.array(values, dtype=object).reshape(node['shape'])
            if t in ('list', 'tuple'):
                v = [decode(x) for x in node['v']]
                return tuple(v) if t == 'tuple' else v
            if t == 'dict':
                pairs = node['v']
                if any(type(k) is not str for k, _ in pairs) or len({k for k, _ in pairs}) != len(pairs): raise ValueError('invalid dict')
                return {k: decode(v) for k, v in pairs}
            raise ValueError('invalid feature node')
        value = decode(metadata['tree'])
        if used != set(arrays.files): raise ValueError('unexpected array')
        return value
