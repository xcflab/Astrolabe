from __future__ import annotations

import unittest
from unittest import mock


class TestAf3DeterministicFramesRefStructure(unittest.TestCase):
    def test_deterministic_frames_does_not_double_ref_structure_for_protein_only(self) -> None:
        from alphafold3.common.folding_input import Input, ProteinChain
        from alphafold3.constants import chemical_components, mmcif_names
        from alphafold3.data import featurisation, msa as msa_module
        from alphafold3.model import features as af3_features

        seq = "A" * 10
        unpaired = msa_module.Msa.from_empty(
            query_sequence=seq,
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
        ).to_a3m()
        paired = msa_module.Msa.from_empty(
            query_sequence=seq,
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
        ).to_a3m()
        fold_input = Input(
            name="test_af3_deterministic_frames",
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
        ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)

        calls = 0
        original = af3_features.RefStructure.compute_features

        def wrapped(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        with mock.patch.object(
            af3_features.RefStructure,
            "compute_features",
            side_effect=wrapped,
        ):
            featurisation.featurise_input(
                fold_input=fold_input,
                ccd=ccd,
                buckets=None,
                verbose=False,
            )

        self.assertEqual(calls, 1)

