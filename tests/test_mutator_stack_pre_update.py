import numpy as np

from astrolabe.core.protomer_config_parser import Segment, SegmentManager


def test_pre_update_reject_rolls_back_to_snapshot() -> None:
    manager = SegmentManager()
    segment = Segment("AAA", ptm_dict={})
    manager.add_segment("A", segment)

    manager.begin_pre_update_step()
    segment.add_mutations((0, "A", "G", 0.1, 0.0))
    manager.preapply_pending_mutations()
    assert segment.sequence == "GAA"

    segment.add_mutations((1, "A", "V", 0.1, 0.0))
    manager.finish_pre_update_step(accepted=False)

    assert segment.sequence == "AAA"
    assert segment.mutations == []


def test_pre_update_accept_keeps_staged_sequence_and_clears_mutations() -> None:
    manager = SegmentManager()
    segment = Segment("AAA", ptm_dict={})
    manager.add_segment("A", segment)

    manager.begin_pre_update_step()
    segment.add_mutations((0, "A", "G", 0.1, 0.0))
    manager.preapply_pending_mutations()
    segment.add_mutations((1, "A", "V", 0.1, 0.0))
    manager.finish_pre_update_step(accepted=True)

    assert segment.sequence == "GAA"
    assert segment.mutations == []


def test_pre_update_accept_applies_later_mutations_against_staged_sequence() -> None:
    manager = SegmentManager()
    segment = Segment("AAA", ptm_dict={})
    manager.add_segment("A", segment)

    manager.begin_pre_update_step()
    segment.add_mutations((0, "A", "G", 0.1, 0.0))
    manager.preapply_pending_mutations()
    segment.add_mutations((1, "A", "V", 0.1, 0.0))
    segment.apply_mutations()
    manager.finish_pre_update_step(accepted=True)

    assert segment.sequence == "GVA"
    assert segment.mutations == []


def test_pre_update_entropy_threshold_only_exposes_confident_mutations() -> None:
    manager = SegmentManager()
    segment = Segment("AAA", ptm_dict={})
    manager.add_segment("A", segment)

    manager.begin_pre_update_step()
    segment.add_mutations((0, "A", "G", 0.1, 0.0))
    segment.add_mutations((1, "A", "V", 2.0, 0.0))
    manager.preapply_pending_mutations(max_entropy=0.5)

    assert segment.sequence == "GAA"
    assert segment.mutations == [(1, "A", "V", 2.0, 0.0)]

    manager.finish_pre_update_step(accepted=False)
    assert segment.sequence == "AAA"


def test_sample_mutations_keeps_zero_entropy_deterministic_change() -> None:
    segment = Segment("L", ptm_dict={})
    probs = np.zeros((1, len(segment.aa_order)), dtype=float)
    probs[0, np.where(segment.aa_order == "I")[0][0]] = 1.0
    segment.add_probs(probs)

    assert segment.sample_mutations() == [(0, "L", "I", 0.0, 0.0)]


def test_sample_mutations_skips_zero_entropy_noop() -> None:
    segment = Segment("L", ptm_dict={})
    probs = np.zeros((1, len(segment.aa_order)), dtype=float)
    probs[0, np.where(segment.aa_order == "L")[0][0]] = 1.0
    segment.add_probs(probs)

    assert segment.sample_mutations() == []
