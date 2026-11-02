"""Position selector for explicitly mapped immutable score evidence."""
import numpy as np
from astrolabe.score_evidence import ScoreQuery, query_score
from astrolabe.selectors.base import ExplicitPositionSelector


def score_position_selector(evaluation_result, protomers, name, *, runtime_config=None):
    runtime_config = runtime_config or {}
    configs = runtime_config.get('score_selector_queries') or getattr(evaluation_result, 'score_selector_queries', None) or {}
    if name not in configs:
        raise ValueError(f'unknown score selector query {name!r}')
    config = dict(configs[name])
    query = ScoreQuery(**dict(config['query']))
    if len(query.axes) != 1:
        raise ValueError('score selector must retain exactly one physical position axis')
    records = list(getattr(evaluation_result, 'score_records', None) or [])
    records.extend(runtime_config.get('score_records', []) or [])
    capture = runtime_config.get('score_capture_snapshot')
    if capture:
        from astrolabe.score_evidence import records_from_standard_scores
        records.extend(records_from_standard_scores(evaluation_result, capture, protomers=protomers))
    catalog = runtime_config.get('score_catalog', []) or []
    if catalog:
        from astrolabe.score_evidence import records_from_contributions
        from astrolabe.score_contributions import score_contributions_from_runtime_config
        capture = runtime_config.get('score_capture_snapshot')
        if not capture:
            raise ValueError('score_capture_snapshot is required for catalog capture')
        contributions = list(getattr(evaluation_result, 'score_contributions', None) or [])
        contributions.extend(score_contributions_from_runtime_config(runtime_config))
        records.extend(records_from_contributions(contributions, catalog, capture))
    snapshots = runtime_config.get('score_snapshots') or getattr(evaluation_result, 'score_snapshots', None) or {}
    values = query_score(records, query, snapshots=snapshots, generating_selector=True)
    target = snapshots.get(query.snapshot, query.snapshot)
    rec = next(r for r in records if r.spec.name == query.name and r.spec.source == query.source and r.snapshot == target and (query.entity is None or r.spec.entity == query.entity))
    axis = rec.spec.axes.index(query.axes[0])
    labels = tuple((query.select or {}).get(query.axes[0], rec.coordinates[axis]))
    mapping = config['mapping']
    if mapping == 'auto':
        mapping = {}
        for label in labels:
            parts = str(label).split(':')
            if len(parts) != 4 or not parts[2].isdigit():
                raise ValueError('automatic mapping requires unambiguous physical residue labels')
            mapping[label] = (parts[3], int(parts[2]) - 1)
    if set(mapping) != set(labels):
        raise ValueError('position mapping must cover every selected physical coordinate')
    masks = {key: np.zeros(proto.length, dtype=bool) for key, proto in protomers.items()}
    sums = {}
    weight_sums = {}
    weights = config.get('position_weights', {})
    destinations = [tuple(mapping[label]) for label in labels]
    if len(set(destinations)) != len(destinations) and not weights:
        raise ValueError('repeated physical positions require explicit weights')
    if weights and set(weights) != set(labels):
        raise ValueError('position weights must cover every coordinate')
    for label, value in zip(labels, values):
        proto, index = mapping[label]
        if proto not in masks or not 0 <= int(index) < len(masks[proto]):
            raise ValueError('physical coordinate maps outside design positions')
        weight = float(weights[label]) if weights else 1.0
        if not np.isfinite(weight) or weight < 0:
            raise ValueError('position weights must be finite and nonnegative')
        key = (proto, int(index))
        sums[key] = sums.get(key, 0.0) + weight * value
        weight_sums[key] = weight_sums.get(key, 0.0) + weight
    threshold = float(config['threshold'])
    if not np.isfinite(threshold) or any(w <= 0 for w in weight_sums.values()):
        raise ValueError('invalid threshold or position weights')
    operation = config.get('comparison', 'ge')
    if operation not in {'ge', 'gt', 'le', 'lt'}:
        raise ValueError('unknown score comparison')
    for (proto, index), total in sums.items():
        masks[proto][index] = getattr(np, {'ge':'greater_equal', 'gt':'greater', 'le':'less_equal', 'lt':'less'}[operation])(total / weight_sums[(proto,index)], threshold)
    return ExplicitPositionSelector(masks)
