from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from astrolabe.core import global_config
from astrolabe.core.folding_input import Input, Ligand, ProteinChain
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.wrappers.boltz import from_af3_to_boltz
from astrolabe.evaluators.wrappers.protenix import from_af3_to_protenix
from astrolabe.utils.base import construct_standard_input


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "nonstandard_residue_cases.yaml"


@dataclass
class _ProtomerStub:
    sequence: str
    mutated_sequence: str
    type: ProtomerTypes = ProtomerTypes.PROTEIN
    ptm_dict: dict[int, str] = field(default_factory=dict)
    homology: dict[str, object] = field(default_factory=dict)


def _fixture_cases() -> dict[str, object]:
    payload = OmegaConf.to_container(OmegaConf.load(FIXTURE_PATH), resolve=True)
    assert isinstance(payload, dict)
    cases = payload.get("cases")
    assert isinstance(cases, dict)
    return cases


def _reset_for_construct(
    *,
    cyclization: dict | None,
    protomers: dict[str, _ProtomerStub],
) -> None:
    global_config.reset()
    global_config.init_const("protomers", protomers)
    global_config.init_const("user_ccd", "data_TEST")
    global_config.init_const("cyclization", cyclization)


def _tuple_bonds(raw_bonds):
    return [
        ((str(a[0]), int(a[1]), str(a[2])), (str(b[0]), int(b[1]), str(b[2])))
        for a, b in raw_bonds
    ]


@pytest.mark.parametrize(
    "case_name",
    ["cyclic_peptide", "stapled_peptide", "lactone", "thiolactone"],
)
def test_representative_cyclization_fixtures_construct_standard_input(
    case_name: str,
) -> None:
    case = _fixture_cases()[case_name]
    assert isinstance(case, dict)
    sequence = str(case["sequence"])
    cyclization = case["cyclization"]
    expected = case["expected"]
    _reset_for_construct(
        cyclization=cyclization,
        protomers={"A": _ProtomerStub(sequence=sequence, mutated_sequence=sequence)},
    )

    result = construct_standard_input(["A"], seed=1)["A"]

    assert result.user_ccd == "data_TEST"
    assert result.bonded_atom_pairs == _tuple_bonds(expected["bonded_atom_pairs"])
    assert tuple(result.cyclic_chain_ids or ()) == tuple(expected["cyclic_chain_ids"])
    has_patches = bool(result.cyclic_relative_patches)
    assert has_patches is bool(expected["relative_patches"])
    assert result.bonded_atom_pair_meta is not None
    assert all(
        "order" in row and "target_distance" in row
        for row in result.bonded_atom_pair_meta
    )


def test_direct_ccd_fixture_constructs_ligand_chain() -> None:
    case = _fixture_cases()["csd_direct_ccd_ligand"]
    expected = case["expected"]
    ccd_ids = ",".join(case["ccd_ids"])
    _reset_for_construct(
        cyclization=None,
        protomers={
            "B": _ProtomerStub(
                sequence=ccd_ids,
                mutated_sequence=ccd_ids,
                type=ProtomerTypes.CCD,
            )
        },
    )

    result = construct_standard_input(["B"], seed=3)["B"]

    assert len(result.chains) == 1
    assert isinstance(result.chains[0], Ligand)
    assert list(result.chains[0].ccd_ids or []) == expected["ligand_ccd_ids"]
    assert result.user_ccd == "data_TEST"


def test_backend_wrappers_preserve_ptm_ccd_ligand_and_covalent_contracts(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("binding_site", None)
    global_config.init_const("protenix_constraint", None)

    inp = Input(
        name="AL",
        chains=[
            ProteinChain(
                id="A",
                sequence="ASAA",
                ptms=[("AS5", 2)],
                unpaired_msa="",
                paired_msa="",
                templates=[],
            ),
            Ligand(id="L", ccd_ids=["MLU"]),
        ],
        rng_seeds=[0],
        bonded_atom_pairs=[(("A", 2, "OG"), ("L", 1, "C1"))],
        user_ccd="data_AS5",
    )

    protenix = from_af3_to_protenix(inp, "AL", use_constraints=False)
    protein_entry = protenix["sequences"][0]["proteinChain"]
    ligand_entry = protenix["sequences"][1]["ligand"]
    assert protein_entry["modifications"] == [
        {"ptmType": "CCD_AS5", "ptmPosition": 2}
    ]
    assert ligand_entry["ligand"] == "CCD_MLU"
    assert protenix["covalent_bonds"][0]["atom1"] == "OG"
    assert protenix["covalent_bonds"][0]["atom2"] == "C1"

    boltz, _chain_map = from_af3_to_boltz(inp)
    assert boltz["sequences"][0]["protein"]["modifications"] == [
        {"position": 2, "ccd": "AS5"}
    ]
    assert boltz["sequences"][1]["ligand"]["ccd"] == "MLU"
    assert boltz["constraints"] == [
        {"bond": {"atom1": ["A", 2, "OG"], "atom2": ["B", 1, "C1"]}}
    ]
