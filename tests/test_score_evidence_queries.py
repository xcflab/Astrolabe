from types import SimpleNamespace
import numpy as np
import pytest
from astrolabe.score_evidence import ScoreSpec, ScoreRecord, ScoreQuery, query_score
from astrolabe.score_contributions import ScoreContribution
from astrolabe.losses.base import LossEngine
from astrolabe.selectors.utils import construct_position_selector


def test_directed_pair_selector_and_shared_chain_mapping():
    # Direction matters: position -> target. Two physical copies map to A:0.
    values = np.array([[1., 3.], [7., 9.], [4., 6.]])
    record = ScoreRecord(ScoreSpec('pae', 'evaluator.af3', ('position', 'target')), 's1', values,
                         (('A0', 'B0', 'A1'), ('X0', 'X1')))
    values[:] = 99
    cfg = {'query': {'name': 'pae', 'source': 'evaluator.af3', 'axes': ['position'],
                     'select': {'target': ['X1']}},
           'mapping': {'A0': ('A', 0), 'B0': ('A', 0), 'A1': ('A', 1)},
           'position_weights': {'A0': 3, 'B0': 1, 'A1': 1},
           'threshold': 5, 'comparison': 'lt'}
    result = SimpleNamespace(score_records=[record], score_snapshots={'last_accepted': 's1'},
                             score_selector_queries={'directed': cfg})
    masks = construct_position_selector('score("directed")', result, {'A': SimpleNamespace(length=2)})
    assert masks['A'].tolist() == [True, False]
    assert record.value[0, 1] == 3
    with pytest.raises(ValueError):
        record.value[0, 0] = 8


def test_stale_duplicate_invalid_and_weighted_context():
    spec = ScoreSpec('x', 'mutator.m', ('context', 'residue'))
    rec = ScoreRecord(spec, 'old', [[1, 2], [5, 6]], (('c1', 'c2'), (0, 1)))
    q = ScoreQuery('x', 'mutator.m', axes=('residue',), weights={'context': {'c1': 3, 'c2': 1}})
    np.testing.assert_allclose(query_score([rec], q, snapshots={'last_accepted':'old'}), [2, 3])
    with pytest.raises(ValueError): query_score([rec], q, snapshots={'last_accepted':'new'})
    with pytest.raises(ValueError): query_score([rec, rec], q, snapshots={'last_accepted':'old'})
    with pytest.raises(ValueError): query_score([rec], ScoreQuery('x','mutator.m',axes=('residue',)), snapshots={'last_accepted':'old'})
    bad = ScoreRecord(spec, 'old', [[1, np.nan], [2, 3]], (('c1', 'c2'), (0, 1)))
    with pytest.raises(ValueError): query_score([bad], q, snapshots={'last_accepted':'old'})
    with pytest.raises(ValueError): query_score([rec], ScoreQuery('x','mutator.m',snapshot='current_proposal'), generating_selector=True)


def test_real_loss_engine_multi_producer_catalog():
    engine = LossEngine({'loss_terms': {}, 'score_catalog': [
        {'name':'q', 'source':'evaluator.a'}, {'name':'q', 'source':'predocker.b'}],
        'score_queries': {'a': {'name':'q','source':'evaluator.a'}, 'b': {'name':'q','source':'predocker.b'}}}, validate_config=False)
    engine.setup('a + b', None, None, validate_config=False)
    engine._extract_scores_with_registry = lambda _: ({}, {})
    engine._default_expression_symbols = lambda: {}
    scores = SimpleNamespace(score_contributions=[ScoreContribution(source='evaluator.a',metric_name='q',value=2)])
    runtime = {'score_contributions':[ScoreContribution(source='predocker.b',metric_name='q',value=3,flatten_as='legacy_b')],
               'score_capture_snapshot':'s1', 'score_snapshots':{'last_accepted':'s1'}}
    score, details = engine.compute_loss(scores, runtime)
    assert score == 5 and details['a'] == 2 and details['b'] == 3
