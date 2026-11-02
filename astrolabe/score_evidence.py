"""Strict, opt-in score evidence and named-axis queries.

Legacy score expressions remain independent of this interface.
"""
from dataclasses import dataclass
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    source: str
    axes: tuple[str, ...] = ()
    entity: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'axes', tuple(self.axes))
        if not self.name or not self.source or len(set(self.axes)) != len(self.axes):
            raise ValueError('ScoreSpec requires a name, source, and unique axes')


@dataclass(frozen=True, init=False)
class ScoreRecord:
    spec: ScoreSpec
    snapshot: str
    coordinates: tuple[tuple[object, ...], ...]
    _data: bytes
    shape: tuple[int, ...]
    execution: str
    validity: str
    disposition: str

    def __init__(self, spec, snapshot, value, coordinates=(), *, execution='completed', validity='valid', disposition='undecided'):
        arr = np.asarray(value, dtype=np.float64)
        coords = tuple(tuple(x) for x in coordinates)
        if not snapshot or arr.ndim != len(spec.axes) or len(coords) != arr.ndim:
            raise ValueError('snapshot and coordinates matching the score axes are required')
        if any(len(c) != n or len(set(c)) != n for c, n in zip(coords, arr.shape)):
            raise ValueError('coordinate labels must be unique and match their axis')
        if execution not in {'completed', 'failed', 'pending'} or validity not in {'valid', 'invalid', 'missing'} or disposition not in {'undecided', 'accepted', 'rejected'}:
            raise ValueError('invalid evidence status')
        for name, val in [('spec', spec), ('snapshot', str(snapshot)), ('coordinates', coords), ('_data', arr.tobytes()), ('shape', arr.shape), ('execution', execution), ('validity', validity), ('disposition', disposition)]:
            object.__setattr__(self, name, val)

    @property
    def value(self):
        return np.frombuffer(self._data, dtype=np.float64).reshape(self.shape)


@dataclass(frozen=True)
class ScoreQuery:
    name: str
    source: str
    snapshot: str = 'last_accepted'
    axes: tuple[str, ...] = ()
    reduction: str = 'mean'
    select: Mapping | None = None
    weights: Mapping | None = None
    entity: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'axes', tuple(self.axes))
        if not self.name or not self.source or not self.snapshot:
            raise ValueError('query requires name, source, and snapshot')
        if len(set(self.axes)) != len(self.axes) or self.reduction not in {'mean', 'sum', 'min', 'max', 'identity'}:
            raise ValueError('invalid query axes or reduction')


def query_score(records, query, *, snapshots=None, generating_selector=False):
    snapshots = snapshots or {}
    if query.snapshot in {'last_accepted', 'current_proposal', 'proposal'} and query.snapshot not in snapshots:
        raise ValueError(f'missing concrete identity for {query.snapshot}')
    target = snapshots.get(query.snapshot, query.snapshot)
    if generating_selector and (query.snapshot in {'current_proposal', 'proposal'} or
                                target != snapshots.get('last_accepted')):
        raise ValueError('a selector must use the last accepted snapshot')
    matches = [r for r in records if r.spec.name == query.name and r.spec.source == query.source and r.snapshot == target and (query.entity is None or r.spec.entity == query.entity)]
    if len(matches) != 1:
        raise ValueError(f'expected one current score record, found {len(matches)}')
    rec = matches[0]
    if rec.execution != 'completed' or rec.validity != 'valid':
        raise ValueError('score evidence is not valid and completed')
    if not set(query.axes) <= set(rec.spec.axes):
        raise ValueError('unknown retained score axis')
    select = query.select or {}
    weights = query.weights or {}
    if not set(select) <= set(rec.spec.axes) or not set(weights) <= set(rec.spec.axes):
        raise ValueError('unknown selected or weighted score axis')
    data = rec.value
    coords = list(rec.coordinates)
    for i, axis in enumerate(rec.spec.axes):
        if axis in select:
            labels = tuple(select[axis])
            if not labels or len(set(labels)) != len(labels):
                raise ValueError('selected labels must be nonempty and unique')
            try:
                indices = [coords[i].index(label) for label in labels]
            except ValueError as exc:
                raise ValueError(f'missing coordinate on {axis}') from exc
            data = np.take(data, indices, axis=i)
            coords[i] = labels
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError('empty or nonfinite score evidence')
    reduce_axes = tuple(i for i, axis in enumerate(rec.spec.axes) if axis not in query.axes)
    if query.reduction == 'identity' and reduce_axes:
        raise ValueError('identity cannot discard axes')
    for i in reduce_axes:
        axis = rec.spec.axes[i]
        if axis in {'context', 'chain', 'physical_chain'} and len(coords[i]) > 1 and axis not in weights:
            raise ValueError(f'explicit weights required for {axis} reduction')
        if axis in {'position', 'target_position'} and axis not in weights:
            labels = [str(label).split(':') for label in coords[i]]
            chains = {(parts[0], parts[1]) for parts in labels if len(parts) == 4}
            if len(chains) > 1:
                raise ValueError(f'explicit weights required for cross-chain {axis} reduction')
    if weights and query.reduction != 'mean':
        raise ValueError('weights require mean reduction')
    weight_array = np.ones(data.shape)
    for axis, mapping in weights.items():
        if axis in query.axes:
            raise ValueError('weights must apply to a reduced axis')
        i = rec.spec.axes.index(axis)
        if set(mapping) != set(coords[i]):
            raise ValueError('weights must explicitly cover selected coordinates')
        w = np.asarray([mapping[x] for x in coords[i]], dtype=float)
        if not np.isfinite(w).all() or (w < 0).any():
            raise ValueError('weights must be finite and nonnegative')
        shape = [1] * data.ndim
        shape[i] = len(w)
        weight_array *= w.reshape(shape)
    if reduce_axes:
        if query.reduction == 'mean':
            denominator = weight_array.sum(axis=reduce_axes)
            if np.any(denominator <= 0):
                raise ValueError('score weights sum to zero')
            data = (data * weight_array).sum(axis=reduce_axes) / denominator
        else:
            data = getattr(np, query.reduction)(data, axis=reduce_axes)
    current_axes = tuple(a for a in rec.spec.axes if a in query.axes)
    if query.axes != current_axes:
        data = np.transpose(data, [current_axes.index(a) for a in query.axes])
    return data.item() if not query.axes else data


def records_from_contributions(contributions, catalog, snapshot):
    """Capture configured existing producers without flattening their values.

    Catalog entries specify source, metric name, axes and coordinate labels.
    Unlisted contributions remain available to the legacy expression path only.
    """
    from astrolabe.score_contributions import normalize_score_contributions
    specs = {}
    for config in catalog:
        config = dict(config)
        coordinates = config.pop('coordinates', ())
        spec = ScoreSpec(**config)
        key = (spec.source, spec.name, spec.entity)
        if key in specs:
            raise ValueError(f'duplicate score specification {key!r}')
        specs[key] = (spec, coordinates)
    records = []
    for contribution in normalize_score_contributions(contributions):
        key = (contribution.source, contribution.metric_name, contribution.entity)
        if contribution.enabled and key in specs:
            spec, coordinates = specs[key]
            records.append(ScoreRecord(spec, snapshot, contribution.value, coordinates))
    return records


def records_from_standard_scores(scores, snapshot, *, protomers=None):
    """Capture evaluator values with sample, context, and directed token axes."""
    from astrolabe.core import global_config
    from astrolabe.utils.mmcif_ids import int_id_to_str_id
    if not snapshot:
        raise ValueError('an explicit score snapshot identity is required')
    if protomers is None:
        try:
            protomers = global_config.protomers
        except RuntimeError:
            protomers = {}
    result = []
    for context in getattr(scores, 'names', ()) or ():
        context = str(context)
        for metric in ('plddt', 'ptm', 'iptm', 'pae', 'pde'):
            samples = (getattr(scores, metric, None) or {}).get(context)
            if samples is None:
                continue
            samples = samples if isinstance(samples, list) else [samples]
            if not samples:
                continue
            arrays = [np.asarray(sample, dtype=float) for sample in samples]
            if any(arr.shape != arrays[0].shape for arr in arrays):
                raise ValueError(f'inconsistent {metric} sample shapes in {context}')
            arr = np.stack(arrays)
            axes = ['context', 'sample']
            coords = [(context,), tuple(range(len(samples)))]
            shape = arrays[0].shape
            if metric in ('pae', 'pde') and len(shape) == 2:
                if shape[0] != shape[1]:
                    raise ValueError(f'{metric} requires a square directed token matrix')
                labels = _physical_labels(context, shape[0], protomers, int_id_to_str_id)
                axes.extend(['position', 'target_position'])
                coords.extend([labels, labels])
            elif len(shape) == 1:
                labels = _physical_labels(context, shape[0], protomers, int_id_to_str_id)
                axes.append('position')
                coords.append(labels)
            elif shape:
                raise ValueError(f'unsupported {metric} shape {shape}')
            result.append(ScoreRecord(ScoreSpec(metric, 'evaluator', tuple(axes), context), snapshot,
                                      arr[np.newaxis, ...], tuple(coords)))
    return result


def _physical_labels(context, length, protomers, int_id_to_str_id):
    labels = []
    if all(p in protomers for p in context):
        for i, proto in enumerate(context):
            chain = int_id_to_str_id(i + 1)
            labels.extend(f'{context}:{chain}:{pos + 1}:{proto}' for pos in range(int(protomers[proto].length)))
    if len(labels) != length:
        labels = [f'{context}:token:{i + 1}' for i in range(length)]
    return tuple(labels)
