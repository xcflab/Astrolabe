from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from astrolabe.core.folding_input import DnaChain, Input, ProteinChain, RnaChain

from astrolabe.core import global_config
from astrolabe.evaluators.wrappers.boltz import from_af3_to_boltz


class TestBoltzWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Keep this test class independent from whichever tests initialized
        # global_config earlier in the same pytest process.
        global_config.reset()
        global_config.init_const("binding_site", None)

    def test_from_af3_to_boltz_accepts_input_chains(self) -> None:
        inp = Input(
            chains=[
                ProteinChain(
                    id="A",
                    sequence="ACDEFGHIK",
                    ptms=[],
                    unpaired_msa="",
                    paired_msa="",
                    templates=[],
                ),
            ],
            name="A",
            rng_seeds=[0],
            bonded_atom_pairs=None,
            user_ccd=None,
        )
        result, chain2input = from_af3_to_boltz(inp)
        self.assertEqual(chain2input, {"A": "A"})
        self.assertEqual(result["version"], 1)
        self.assertEqual(result["constraints"], [])
        self.assertEqual(result["sequences"][0]["protein"]["id"], "A")
        self.assertEqual(result["sequences"][0]["protein"]["sequence"], "ACDEFGHIK")

    def test_from_af3_to_boltz_accepts_x_residue(self) -> None:
        inp = Input(
            chains=[
                ProteinChain(
                    id="A",
                    sequence="ACDXEFGHIK",
                    ptms=[],
                    unpaired_msa="",
                    paired_msa="",
                    templates=[],
                ),
            ],
            name="A",
            rng_seeds=[0],
            bonded_atom_pairs=None,
            user_ccd=None,
        )
        result, _chain2input = from_af3_to_boltz(inp)
        self.assertEqual(result["sequences"][0]["protein"]["sequence"], "ACDXEFGHIK")

    def test_from_af3_to_boltz_accepts_all_x_sequence(self) -> None:
        inp = Input(
            chains=[
                ProteinChain(
                    id="A",
                    sequence="XXXXX",
                    ptms=[],
                    unpaired_msa="",
                    paired_msa="",
                    templates=[],
                ),
            ],
            name="A",
            rng_seeds=[0],
            bonded_atom_pairs=None,
            user_ccd=None,
        )
        result, _chain2input = from_af3_to_boltz(inp)
        self.assertEqual(result["sequences"][0]["protein"]["sequence"], "XXXXX")

    def test_from_af3_to_boltz_rna_and_dna_ids(self) -> None:
        inp = Input(
            chains=[
                RnaChain(id="A", sequence="ACGU", modifications=[], unpaired_msa=""),
                DnaChain(id="B", sequence="ACGT", modifications=[]),
            ],
            name="AB",
            rng_seeds=[0],
            bonded_atom_pairs=None,
            user_ccd=None,
        )
        result, chain2input = from_af3_to_boltz(inp)
        # Input chains are already A/B, and we preserve that mapping.
        self.assertEqual(chain2input, {"A": "A", "B": "B"})
        self.assertEqual([list(x.keys())[0] for x in result["sequences"]], ["rna", "dna"])
        self.assertEqual(result["sequences"][0]["rna"]["id"], "A")
        self.assertEqual(result["sequences"][1]["dna"]["id"], "B")

    def test_from_af3_to_boltz_includes_bond_constraints(self) -> None:
        inp = Input(
            chains=[
                ProteinChain(
                    id="A",
                    sequence="ACDEFGHIK",
                    ptms=[],
                    unpaired_msa="",
                    paired_msa="",
                    templates=[],
                ),
            ],
            name="A",
            rng_seeds=[0],
            bonded_atom_pairs=[(("A", 1, "N"), ("A", 9, "C"))],
            user_ccd=None,
        )
        result, chain2input = from_af3_to_boltz(inp)
        self.assertEqual(chain2input, {"A": "A"})
        self.assertEqual(
            result["constraints"],
            [
                {"bond": {"atom1": ["A", 1, "N"], "atom2": ["A", 9, "C"]}},
            ],
        )

    def test_from_af3_to_boltz_marks_head_to_tail_chain_as_cyclic(self) -> None:
        inp = Input(
            chains=[
                ProteinChain(
                    id="A",
                    sequence="ACDEFGHIK",
                    ptms=[],
                    unpaired_msa="",
                    paired_msa="",
                    templates=[],
                ),
            ],
            name="A",
            rng_seeds=[0],
            bonded_atom_pairs=[(("A", 1, "N"), ("A", 9, "C"))],
            bonded_atom_pair_meta=[
                {"order": "single", "target_distance": 1.33, "is_head_to_tail": True}
            ],
            user_ccd=None,
        )
        result, _ = from_af3_to_boltz(inp)
        self.assertTrue(result["sequences"][0]["protein"].get("cyclic", False))

    def test_from_af3_to_boltz_preserves_explicit_cyclic_chain_ids_after_remap(self) -> None:
        inp = Input(
            chains=[
                ProteinChain(
                    id="X",
                    sequence="ACDE",
                    ptms=[],
                    unpaired_msa="",
                    paired_msa="",
                    templates=[],
                ),
            ],
            name="X",
            rng_seeds=[0],
            bonded_atom_pairs=None,
            cyclic_chain_ids=("X",),
            user_ccd=None,
        )
        result, chain2input = from_af3_to_boltz(inp)
        self.assertEqual(chain2input, {"A": "X"})
        self.assertTrue(result["sequences"][0]["protein"].get("cyclic", False))

    def test_from_af3_to_boltz_marks_inferred_cyclic_chain_with_noncanonical_id(self) -> None:
        inp = Input(
            chains=[
                ProteinChain(
                    id="X",
                    sequence="ACDE",
                    ptms=[],
                    unpaired_msa="",
                    paired_msa="",
                    templates=[],
                ),
            ],
            name="X",
            rng_seeds=[0],
            bonded_atom_pairs=[(("X", 1, "N"), ("X", 4, "C"))],
            bonded_atom_pair_meta=[{"is_head_to_tail": True}],
            user_ccd=None,
        )
        result, chain2input = from_af3_to_boltz(inp)
        self.assertEqual(chain2input, {"A": "X"})
        self.assertTrue(result["sequences"][0]["protein"].get("cyclic", False))

    def test_from_af3_to_boltz_does_not_mark_non_head_to_tail_as_cyclic(self) -> None:
        inp = Input(
            chains=[
                ProteinChain(
                    id="A",
                    sequence="ACDEFGHIK",
                    ptms=[],
                    unpaired_msa="",
                    paired_msa="",
                    templates=[],
                ),
            ],
            name="A",
            rng_seeds=[0],
            bonded_atom_pairs=[(("A", 2, "CA"), ("A", 7, "CB"))],
            bonded_atom_pair_meta=[
                {"order": "single", "target_distance": 1.54, "is_head_to_tail": False}
            ],
            user_ccd=None,
        )
        result, _ = from_af3_to_boltz(inp)
        self.assertFalse(result["sequences"][0]["protein"].get("cyclic", False))

    def test_from_af3_to_boltz_writes_unpaired_msa_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            global_config.reset()
            global_config.init_const("out_dir", Path(tmpdir))
            global_config.init_const("binding_site", None)

            unpaired = ">query\nACDEFGHIK\n>hit\nACDEYGHIK\n"
            inp = Input(
                chains=[
                    ProteinChain(
                        id="A",
                        sequence="ACDEFGHIK",
                        ptms=[],
                        unpaired_msa=unpaired,
                        paired_msa="",
                        templates=[],
                    ),
                ],
                name="A",
                rng_seeds=[0],
                bonded_atom_pairs=None,
                user_ccd=None,
            )
            result, _ = from_af3_to_boltz(inp)
            msa_path = Path(result["sequences"][0]["protein"]["msa"])
            self.assertTrue(msa_path.is_file())
            self.assertEqual(msa_path.read_text(encoding="utf-8"), unpaired)

    def test_from_af3_to_boltz_uses_paired_msa_when_unpaired_empty(self) -> None:
        with TemporaryDirectory() as tmpdir:
            global_config.reset()
            global_config.init_const("out_dir", Path(tmpdir))
            global_config.init_const("binding_site", None)

            paired = ">query\nACDEFGHIK\n>pair_hit\nACDEFGHIK\n"
            inp = Input(
                chains=[
                    ProteinChain(
                        id="A",
                        sequence="ACDEFGHIK",
                        ptms=[],
                        unpaired_msa="",
                        paired_msa=paired,
                        templates=[],
                    ),
                ],
                name="A",
                rng_seeds=[0],
                bonded_atom_pairs=None,
                user_ccd=None,
            )
            result, _ = from_af3_to_boltz(inp)
            msa_path = Path(result["sequences"][0]["protein"]["msa"])
            self.assertTrue(msa_path.is_file())
            self.assertEqual(msa_path.read_text(encoding="utf-8"), paired)
