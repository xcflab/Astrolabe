"""Opt-in synchronization contracts shared by MPNN adapters."""
from dataclasses import dataclass
import math
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class SynchronizationPolicy:
    context_layout: str = 'separate'
    coupling: str = 'legacy_order_conditioned_marginals'
    context_weights: dict | None = None

    def __post_init__(self):
        if self.context_layout not in {'separate', 'far_apart'}:
            raise ValueError('unknown context layout')
        if self.coupling not in {'native_tie', 'probability_mean', 'legacy_order_conditioned_marginals'}:
            raise ValueError('unknown synchronization coupling')
        if self.coupling == 'native_tie' and self.context_layout != 'far_apart':
            raise ValueError('native_tie requires far_apart layout')
        if self.coupling == 'probability_mean' and self.context_layout != 'separate':
            raise ValueError('probability_mean requires separate context inference')
        if self.context_weights is not None:
            if not self.context_weights or any(not math.isfinite(float(w)) or float(w) < 0 for w in self.context_weights.values()):
                raise ValueError('probability_mean context weights must be finite and nonnegative')
            if not any(float(w) > 0 for w in self.context_weights.values()):
                raise ValueError('context weights cannot all be zero')
        if self.coupling != 'probability_mean' and self.context_weights is not None:
            raise ValueError('native_tie uses signed state_betas, not probability weights')

    def aggregate(self, scores, outputs, protomers, supported_types, legacy_collector):
        if self.coupling == 'probability_mean':
            return mean_context_probabilities(scores, outputs, protomers, self, supported_types)
        # Preserve the historical order-conditioned aggregation and RNG path.
        return legacy_collector()

    def validate_backend(self, runner):
        if self.coupling == 'probability_mean' and 'shared_reference_teacher_forcing' not in getattr(runner, 'supported_distribution_semantics', ()):
            raise ValueError('probability_mean requires shared-reference teacher-forcing capability')
        if self.coupling == 'native_tie' and not callable(getattr(runner, 'design_sequences', None)):
            raise ValueError('native_tie requires native joint decoding')


def _context_positions(contexts, protomers):
    """Map physical positions to canonical segment positions."""
    from astrolabe.core import global_config
    try:
        segments, manager = global_config.segments, global_config.segment_manager
    except RuntimeError:
        segments, manager = {}, None
    mapped = {}
    for context in contexts:
        positions = []
        for name in context:
            proto = protomers[name]
            components = getattr(proto, 'components', None)
            if components and manager is not None:
                for component in components:
                    segment = segments[component]
                    canonical_name = manager.primary_names[id(segment)]
                    positions.extend((canonical_name, i) for i in range(len(segment.sequence)))
            else:
                positions.extend((name, i) for i in range(int(proto.length)))
        mapped[context] = positions
    shared = {key for positions in mapped.values() for key in positions
              if sum(key in other for other in mapped.values()) > 1}
    if not shared:
        raise ValueError('probability_mean requires shared canonical positions')
    for positions in mapped.values():
        if any(positions.count(key) != 1 for key in shared if key in positions):
            raise ValueError('shared-reference mode requires one physical copy per shared position in each context')
    return mapped, shared


def mean_context_probabilities(scores, outputs, protomers, policy, supported_types):
    """Combine matching reference-conditioned distributions in probability space."""
    from astrolabe.utils.constant import aa_order
    from astrolabe.utils.mmcif_ids import int_id_to_str_id

    names = list(getattr(scores, 'names', []) or [])
    weights = policy.context_weights or {name: 1.0 for name in names}
    if set(weights) != set(names):
        raise ValueError('context weights must cover exactly the evaluated contexts')
    canonical, shared = _context_positions(names, protomers)
    reference_by_position, mask_by_position, previous_orders = {}, {}, []
    sums, totals = {}, {}
    position_sums, position_totals, proto_positions = {}, {}, {}
    for context in names:
        output = outputs.get(context)
        if not isinstance(output, dict) or 'sampling_probs' not in output:
            raise ValueError('probability_mean requires reference-conditioned probabilities for every context')
        if output.get('distribution_semantics') != 'shared_reference_teacher_forcing':
            raise ValueError('incompatible probability conditioning semantics')
        if tuple(output.get('alphabet', ())) != tuple(aa_order[:20]):
            raise ValueError('incompatible probability alphabet')
        def as_array(value):
            return np.asarray(value.detach().cpu().numpy() if hasattr(value, 'detach') else value)
        reference = as_array(output.get('conditioning_reference'))
        mask = as_array(output.get('chain_mask'))
        order = as_array(output.get('decoding_order'))
        if reference.ndim != 2 or reference.shape[0] != 1 or mask.shape != reference.shape or order.shape != reference.shape:
            raise ValueError('reference sequence, mask and order must describe one sample')
        if sorted(order[0].tolist()) != list(range(reference.shape[1])):
            raise ValueError('reference decoding order must be a complete permutation')
        expected_chains = tuple(int_id_to_str_id(i + 1) for i, p in enumerate(context)
                                for _ in range(int(protomers[p].length)))
        if tuple(output.get('chain_letters', ())) != expected_chains:
            raise ValueError('physical chain ordering does not match canonical mapping')
        arr = np.asarray(as_array(output['sampling_probs']), dtype=float)
        if arr.ndim != 3 or arr.shape[:2] != reference.shape or arr.shape[-1] != 20 or not np.isfinite(arr).all() or (arr < 0).any():
            raise ValueError('invalid reference-conditioned position probabilities')
        totals_by_row = arr.sum(axis=-1)
        if not np.allclose(totals_by_row, 1.0, atol=1e-5):
            raise ValueError('probability rows must be normalized')
        positions = canonical[context]
        if len(positions) != reference.shape[1]:
            raise ValueError('physical positions do not match canonical mapping')
        for i, key in enumerate(positions):
            if key not in shared:
                continue
            value, mutable = int(reference[0, i]), int(mask[0, i])
            if key in reference_by_position and (reference_by_position[key] != value or mask_by_position[key] != mutable):
                raise ValueError('incompatible shared reference or design mask')
            reference_by_position[key], mask_by_position[key] = value, mutable
        this_order = [positions[int(i)] for i in order[0] if positions[int(i)] in shared]
        for previous in previous_orders:
            common = set(previous).intersection(this_order)
            if [key for key in previous if key in common] != [key for key in this_order if key in common]:
                raise ValueError('incompatible shared decoding order')
        previous_orders.append(this_order)
        offset = 0
        for proto_name in context:
            proto = protomers[proto_name]
            length = int(proto.length)
            block = arr[:, offset:offset + length, :]
            observed = tuple(int(x) for x in reference[0, offset:offset + length])
            offset += length
            if proto.type not in supported_types:
                continue
            expected = tuple(aa_order.index(aa) for aa in proto.sequence)
            if observed != expected:
                raise ValueError(f'context {context} does not condition on the accepted {proto_name} sequence')
            weight = float(weights[context])
            proto_keys = positions[offset-length:offset]
            if proto_name in proto_positions and proto_positions[proto_name] != proto_keys:
                raise ValueError('protomer canonical mapping differs across contexts')
            proto_positions[proto_name] = proto_keys
            for local_index, key in enumerate(proto_keys):
                position_sums[key] = position_sums.get(key, np.zeros_like(block[:, local_index, :])) + weight * block[:, local_index, :]
                position_totals[key] = position_totals.get(key, 0.0) + weight
            if proto_name in sums and sums[proto_name].shape != block.shape:
                raise ValueError('context probability shapes do not match')
            sums[proto_name] = sums.get(proto_name, np.zeros_like(block)) + weight * block
            totals[proto_name] = totals.get(proto_name, 0.0) + weight
        if offset != arr.shape[1]:
            raise ValueError('context probability length does not match physical mapping')
    if any(total <= 0 for total in totals.values()):
        raise ValueError('no positive weight for a protomer')
    identity = [(key, reference_by_position[key], mask_by_position[key]) for key in sorted(shared)]
    fingerprint = hashlib.sha256(json.dumps({'positions': identity, 'orders': dict(zip(names, previous_orders))}, sort_keys=True).encode()).hexdigest()
    for context in names:
        outputs[context]['conditioning_fingerprint'] = fingerprint
    if any(total <= 0 for total in position_totals.values()):
        raise ValueError('no positive weight for a canonical position')
    return {name: np.stack([position_sums[key] / position_totals[key] for key in proto_positions[name]], axis=1)
            for name in sums}


def reference_decoding_ranks(contexts, protomers):
    """Place shared segment positions in canonical order before other mutable sites."""
    contexts = list(contexts)
    mapped, shared = _context_positions(contexts, protomers)
    rank_by_position = {key: i + 1 for i, key in enumerate(sorted(shared))}
    result = {}
    for context in contexts:
        ranks, extra = [], len(shared) + 1
        for key in mapped[context]:
            if key in shared:
                ranks.append(rank_by_position[key])
            else:
                ranks.append(extra)
                extra += 1
        if extra >= 9999:
            raise ValueError('structure too long for unambiguous reference ordering')
        result[context] = ranks
    return result
