from __future__ import annotations

from pathlib import Path

from astrolabe.core import global_config
from astrolabe.core.folding_input import Input, Ligand, ProteinChain
from astrolabe.evaluators.chai1 import Chai1Evaluator, _CCD_SMILES_CACHE


def test_chai_prepare_input_writes_ligand_from_ccd_smiles(
    tmp_path: Path, monkeypatch
) -> None:
    ccd_file = tmp_path / "components.cif"
    ccd_file.write_text(
        """
data_FAD
#
loop_
_pdbx_chem_comp_descriptor.comp_id
_pdbx_chem_comp_descriptor.type
_pdbx_chem_comp_descriptor.program
_pdbx_chem_comp_descriptor.program_version
_pdbx_chem_comp_descriptor.descriptor
FAD SMILES_CANONICAL CACTVS 1.0 "CCO"
#
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("GH_PROTENIX_CCD", str(ccd_file))
    _CCD_SMILES_CACHE.clear()

    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("binding_site", None)
    global_config.set_variable("step", 0)

    evaluator = Chai1Evaluator(
        model_config={"use_msa_server": False},
        output_dir=tmp_path / "chai1",
    )
    inp = Input(
        chains=[
            ProteinChain(
                id="A",
                sequence="ACDXEF",
                ptms=[],
                unpaired_msa="",
                paired_msa="",
                templates=[],
            ),
            Ligand(id="B", ccd_ids=["FAD"]),
        ],
        name="AB",
        rng_seeds=[0],
        bonded_atom_pairs=None,
        user_ccd=None,
    )

    prepared = evaluator.prepare_input({"AB": inp}, runtime_config={})["AB"]

    assert prepared["fasta_file"].read_text(encoding="utf-8").splitlines() == [
        ">protein|A",
        "ACDXEF",
        ">ligand|B",
        "CCO",
    ]
