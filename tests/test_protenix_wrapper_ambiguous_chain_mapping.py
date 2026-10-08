from __future__ import annotations

from pathlib import Path

import pytest

from astrolabe.core import global_config
from astrolabe.evaluators.wrappers.protenix import from_af3_to_protenix


def test_protenix_bond_constraint_rejects_ambiguous_chain_entity_mapping(
    tmp_path: Path,
) -> None:
    # NOTE: This test uses the AF3 folding_input classes provided by the GH test env.
    from alphafold3.common.folding_input import Input, Ligand, ProteinChain

    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("protenix_constraint", None)
    global_config.init_const("binding_site", None)
    global_config.set_variable("step", 0)

    inp = Input(
        name="test",
        chains=[
            ProteinChain(id="A", sequence="ACDE", ptms=[]),
            # A single ligand chain may expand into multiple Protenix entities
            # when multiple CCD IDs are provided.
            Ligand(id="L", ccd_ids=["AAA", "BBB"]),
        ],
        rng_seeds=[0],
        bonded_atom_pairs=[(("L", 1, "C1"), ("A", 1, "CA"))],
    )

    with pytest.raises(ValueError, match=r"exactly one entity|got 2 entities"):
        _ = from_af3_to_protenix(inp, "AL")

