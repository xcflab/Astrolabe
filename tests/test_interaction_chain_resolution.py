from __future__ import annotations

from types import SimpleNamespace

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.losses.interaction import _resolve_ligand_chains


def test_resolve_ligand_chains_keeps_all_repeated_ligand_copies() -> None:
    global_config.reset()
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(type=ProtomerTypes.CCD),
            "B": SimpleNamespace(type=ProtomerTypes.PROTEIN),
        },
    )

    ligands, _prot2chains = _resolve_ligand_chains("AAB")
    assert ligands == [("A", "A"), ("A", "B")]


def test_resolve_ligand_chains_uses_protomer_first_and_chain_override() -> None:
    global_config.reset()
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(type=ProtomerTypes.CCD),
            "B": SimpleNamespace(type=ProtomerTypes.PROTEIN),
        },
    )

    ligands_proto, _ = _resolve_ligand_chains("AAB", ligand_targets="B")
    assert ligands_proto == [("B", "C")]

    ligands_chain, _ = _resolve_ligand_chains("AAB", ligand_targets="chain:B")
    assert ligands_chain == [("A", "B")]
