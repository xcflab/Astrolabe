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

    def __post_init__(self):
        object.__setattr__(self, 'axes', tuple(self.axes))
        if not self.name or not self.source or not self.snapshot:
            raise ValueError('query requires name, source, and snapshot')
        if len(set(self.axes)) != len(self.axes) or self.reduction not in {'mean', 'sum', 'min', 'max', 'identity'}:
            raise ValueError('invalid query axes or reduction')


def query_score(records, query, *, snapshots=None, generating_selector=False):
    snapshots = snapshots or {}
    target = snapshots.get(query.snapshot, query.snapshot)
    if generating_selector and query.snapshot in {'current_proposal', 'proposal'}:
        raise ValueError('a proposal cannot select positions from its own score')
    matches = [r for r in records if r.spec.name == query.name and r.spec.source == query.source and r.snapshot == target]
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
        key = (spec.source, spec.name)
        if key in specs:
            raise ValueError(f'duplicate score specification {key!r}')
        specs[key] = (spec, coordinates)
    records = []
    for contribution in normalize_score_contributions(contributions):
        key = (contribution.source, contribution.metric_name)
        if contribution.enabled and key in specs:
            spec, coordinates = specs[key]
            records.append(ScoreRecord(spec, snapshot, contribution.value, coordinates))
    return records
