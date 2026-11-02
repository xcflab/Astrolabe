"""CPU-only stage and invalidation tests; these are not GPU certification."""
import os
import subprocess
import sys
import unittest
from unittest import mock
from astrolabe.backend_capabilities import resolve_protenix_identity

from astrolabe.feature_cache import FEATURE_DEPENDENCIES, feature_key
from astrolabe.constraints import Constraint, ConstraintKind, validate_constraints


class BackendContracts(unittest.TestCase):
    def test_content_invalidation(self):
        deps = {key: None for key in FEATURE_DEPENDENCIES}
        baseline = feature_key(deps)
        for key in FEATURE_DEPENDENCIES:
            changed = dict(deps, **{key: 'changed'})
            self.assertNotEqual(baseline, feature_key(changed), key)
        with self.assertRaises(ValueError):
            feature_key(dict(deps, candidate_id='unrelated'))
        with self.assertRaises(ValueError):
            feature_key({})

    def test_protenix_identity(self):
        self.assertEqual(resolve_protenix_identity('protenix-v2', True),
                         ('protenix-v2', 'protenix_base_constraint_v0.5.0'))
        with self.assertRaises(ValueError):
            resolve_protenix_identity('protenix-v2', True, 'strict')
        self.assertEqual(resolve_protenix_identity('protenix-v2', False, 'strict'),
                         ('protenix-v2', 'protenix-v2'))

    def test_constraints_fail_closed(self):
        soft = Constraint(ConstraintKind.SOFT, {}, 'contact')
        supported = {'protenix-v0.5-constraint': {(ConstraintKind.SOFT, 'contact')}}
        validate_constraints('protenix-v0.5-constraint', [soft], supported)
        with self.assertRaises(ValueError):
            validate_constraints('protenix-v2', [soft], supported)
        with self.assertRaises(ValueError):
            validate_constraints('unknown', [soft], supported)

    def test_af3_stages_in_subprocess(self):
        env = dict(os.environ, JAX_PLATFORMS='cpu')
        script = r'''
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock
from astrolabe.evaluators.alphafold3 import AlphaFold3Evaluator as E
import astrolabe.evaluators.alphafold3 as mod
class Runner:
    loads = 0
    def __init__(self):
        Runner.loads += 1
    def set_cyclic_asym_ids(self, ids): self.ids = ids
    def set_cyclic_relative_patches(self, patches): self.patches = patches
    def run_inference(self, example, key): return (example, self.ids, self.patches)
    def extract_structures(self, batch, result, target_name):
        return [(batch, result, target_name)]
r = Runner()
with mock.patch.object(mod.chemical_components, 'Ccd', return_value=object()), \\
     mock.patch.object(mod, 'allow_all_x_protein_chains_during_featurisation', return_value=nullcontext()), \\
     mock.patch.object(mod.featurisation, 'featurise_input', side_effect=lambda fold_input, **kw: [fold_input.name]), \\
     mock.patch.object(mod.jax.random, 'PRNGKey', side_effect=lambda seed: seed):
    for name, ids in [('first', (1,)), ('second', ())]:
        fold = SimpleNamespace(name=name, rng_seeds=[7], user_ccd=None)
        examples, cyclic, patches = E.prepare_features(fold, None, cyclic_asym_ids=ids)
        result = E.forward_features(r, fold.rng_seeds, examples, cyclic, patches)
        final = E.finalize_features(r, fold, examples, result)
        assert final[0].inference_results == [(name, (name, ids, ()), name)]
    assert Runner.loads == 1
'''.replace('\\\\\n', '\\\n')
        completed = subprocess.run([sys.executable, '-c', script], env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()
