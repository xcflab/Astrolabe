from __future__ import annotations

import os
import unittest
from unittest import mock


class TestAf3RefStructureCache(unittest.TestCase):
    def _make_fold_input(self, *, name: str, seq: str):
        from alphafold3.common.folding_input import Input, ProteinChain
        from alphafold3.constants import mmcif_names
        from alphafold3.data import msa as msa_module

        unpaired = msa_module.Msa.from_empty(
            query_sequence=seq,
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
        ).to_a3m()
        paired = msa_module.Msa.from_empty(
            query_sequence=seq,
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
        ).to_a3m()
        return Input(
            name=name,
            chains=[
                ProteinChain(
                    id="A",
                    sequence=seq,
                    ptms=[],
                    unpaired_msa=unpaired,
                    paired_msa=paired,
                    templates=[],
                )
            ],
            rng_seeds=[0],
            bonded_atom_pairs=None,
            user_ccd=None,
        )

    def _clear_refstructure_cache(self) -> None:
        # Cache is an implementation detail; clear it if present to avoid
        # cross-test interference.
        from alphafold3.model import features as af3_features

        cache = getattr(af3_features, "_REFSTRUCTURE_REFERENCE_CACHE", None)
        if cache is not None:
            cache.clear()

    def test_refstructure_cache_reuses_same_input(self) -> None:
        from alphafold3.constants import chemical_components
        from alphafold3.data import featurisation
        from alphafold3.data.tools import rdkit_utils

        self._clear_refstructure_cache()

        fold_input = self._make_fold_input(name="test_refstructure_cache_same", seq="A" * 10)
        ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)

        calls = 0
        original = rdkit_utils.get_random_conformer

        def wrapped(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        with mock.patch.dict(
            os.environ,
            {
                "ALPHAFOLD3_REFSTRUCTURE_CACHE": "1",
                "ALPHAFOLD3_REFSTRUCTURE_WORKERS": "1",
            },
            clear=False,
        ):
            with mock.patch.object(
                rdkit_utils,
                "get_random_conformer",
                side_effect=wrapped,
            ):
                featurisation.featurise_input(
                    fold_input=fold_input,
                    ccd=ccd,
                    buckets=None,
                    verbose=False,
                )
                first_calls = calls
                calls = 0
                featurisation.featurise_input(
                    fold_input=fold_input,
                    ccd=ccd,
                    buckets=None,
                    verbose=False,
                )
                second_calls = calls

        self.assertGreater(first_calls, 0)
        self.assertEqual(second_calls, 0)

    def test_refstructure_cache_only_recomputes_mutations(self) -> None:
        from alphafold3.constants import chemical_components
        from alphafold3.data import featurisation
        from alphafold3.data.tools import rdkit_utils

        self._clear_refstructure_cache()

        base = self._make_fold_input(name="test_refstructure_cache_base", seq="A" * 10)
        mutant = self._make_fold_input(
            name="test_refstructure_cache_mutant",
            seq="A" * 5 + "C" + "A" * 4,
        )
        ccd_base = chemical_components.Ccd(user_ccd=base.user_ccd)
        ccd_mutant = chemical_components.Ccd(user_ccd=mutant.user_ccd)

        calls = 0
        original = rdkit_utils.get_random_conformer

        def wrapped(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        with mock.patch.dict(
            os.environ,
            {
                "ALPHAFOLD3_REFSTRUCTURE_CACHE": "1",
                "ALPHAFOLD3_REFSTRUCTURE_WORKERS": "1",
            },
            clear=False,
        ):
            with mock.patch.object(
                rdkit_utils,
                "get_random_conformer",
                side_effect=wrapped,
            ):
                featurisation.featurise_input(
                    fold_input=base,
                    ccd=ccd_base,
                    buckets=None,
                    verbose=False,
                )
                calls = 0
                featurisation.featurise_input(
                    fold_input=mutant,
                    ccd=ccd_mutant,
                    buckets=None,
                    verbose=False,
                )
                mutant_calls = calls

        self.assertEqual(mutant_calls, 1)

