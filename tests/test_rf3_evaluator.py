from __future__ import annotations

import json
import io
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import MMCIFIO, MMCIFParser, PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.core.folding_input import Input, Ligand, ProteinChain, Template
from astrolabe.evaluators.rf3 import (
    RoseTTAFold3Evaluator,
    _chemical_component_atom_names_from_cif,
)
from astrolabe.evaluators.utils import construct_evaluators
from astrolabe.evaluators.wrappers.rf3 import (
    RF3RunnerRequest,
    RF3RunnerResponse,
    RF3RunnerService,
    rf3_setup,
)


def _protein_chain(
    *,
    chain_id: str = "A",
    sequence: str = "ACDE",
    unpaired_msa: str = "",
    templates: list[Template] | None = None,
) -> ProteinChain:
    return ProteinChain(
        id=chain_id,
        sequence=sequence,
        ptms=[],
        unpaired_msa=unpaired_msa,
        paired_msa="",
        templates=templates or [],
    )


def _write_protein_pdb(path: Path, *, chain_id: str, resnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    serial = 1
    x = 0.0
    atom_specs = [
        ("N", "N"),
        ("CA", "C"),
        ("C", "C"),
        ("O", "O"),
        ("CB", "C"),
    ]
    for resseq, resname in enumerate(resnames, start=1):
        for atom_name, element in atom_specs:
            if resname == "GLY" and atom_name == "CB":
                continue
            x += 1.0
            lines.append(
                f"ATOM  {serial:5d} {atom_name:<4} {resname:>3} {chain_id}{resseq:4d}    "
                f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00          {element:>2}"
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _single_chain_mmcif_text(path: Path) -> str:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("template", str(path))
    out = io.StringIO()
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(out)
    return out.getvalue()


def _write_ligand_pdb(
    path: Path,
    *,
    chain_id: str,
    resname: str,
    resseq: int,
    atoms: list[tuple[str, float, float, float, str]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for serial, (atom_name, x, y, z, element) in enumerate(atoms, start=1):
        lines.append(
            f"HETATM{serial:5d} {atom_name:<4} {resname:>3} {chain_id}{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2}"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_ligand_cif(
    path: Path,
    *,
    chain_id: str,
    resname: str,
    resseq: int,
    atoms: list[tuple[str, float, float, float, str]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    structure = Structure("ligand")
    model = Model(0)
    chain = Chain(chain_id)
    residue = Residue((f"H_{resname}", resseq, " "), resname, "")
    for serial, (atom_name, x, y, z, element) in enumerate(atoms, start=1):
        residue.add(
            Atom(
                atom_name,
                np.array([x, y, z], dtype=float),
                20.0,
                1.0,
                " ",
                atom_name,
                serial,
                element=element,
            )
        )
    chain.add(residue)
    model.add(chain)
    structure.add(model)
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(path))
    return path


def _write_component_cif(path: Path, *, component_id: str = "R13SB") -> Path:
    path.write_text(
        f"""data_{component_id}
#
_chem_comp.id {component_id}
_chem_comp.name '13-cis-retinal covalent-ready'
_chem_comp.type 'NON-POLYMER'
_chem_comp.formula 'C20H28'
_chem_comp.mon_nstd_parent_comp_id ?
_chem_comp.pdbx_synonyms ?
_chem_comp.formula_weight 268.444
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
{component_id} C1 C 0 0.0 0.0 0.0
{component_id} C15 C 0 1.0 0.0 0.0
{component_id} H1 H 0 0.0 1.0 0.0
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
{component_id} C1 C15 SING N
{component_id} C1 H1 SING N
#
""",
        encoding="utf-8",
    )
    return path


def _ligand_atom_names(path: Path, *, chain_id: str, resname: str) -> list[str]:
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() == ".cif" else PDBParser(QUIET=True)
    structure = parser.get_structure("ligand", str(path))
    atom_names: list[str] = []
    for model in structure:
        for chain in model:
            if str(chain.id).strip() != chain_id:
                continue
            for residue in chain:
                if str(residue.get_resname()).strip() != resname:
                    continue
                atom_names.extend(str(atom.get_name()).strip() for atom in residue.get_atoms())
    return atom_names


def test_construct_evaluators_registers_rf3(monkeypatch, tmp_path: Path) -> None:
    class _EvalConf:
        evaluator_terms = {
            "RF3": {"_type": "rf3"},
        }
        rf3 = OmegaConf.create(
            {
                "repo_path": "/tmp/foundry",
                "ckpt_path": "/tmp/rf3.ckpt",
                "num_steps": 40,
            }
        )

    sentinel_runner = object()

    monkeypatch.setattr(
        "astrolabe.evaluators.wrappers.rf3.rf3_setup",
        lambda conf: sentinel_runner,
    )

    evaluators = construct_evaluators(
        _EvalConf(),
        raw_model_results_dir=tmp_path / "raw_model_results",
        oligo_names=["AL"],
    )

    evaluator = evaluators["RF3"]
    assert isinstance(evaluator, RoseTTAFold3Evaluator)
    assert evaluator.model_runner is sentinel_runner
    assert evaluator.output_dir == tmp_path / "raw_model_results" / "rf3"


def test_benchmark_channel_config_exposes_top_level_rf3_defaults() -> None:
    base_cfg = OmegaConf.load("config/base.yaml")
    benchmark_cfg = OmegaConf.load("config/benchmark_channel_v1_refine.yaml")
    merged_cfg = OmegaConf.merge(base_cfg, benchmark_cfg)

    assert "rf3" in merged_cfg.evaluator
    assert "repo_path" in merged_cfg.evaluator.rf3
    assert "ckpt_path" in merged_cfg.evaluator.rf3
    assert "python_bin" in merged_cfg.evaluator.rf3


def test_rf3_prepare_input_materializes_fixed_ligand_payload(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("channel", None)
    global_config.set_variable("step", 3)

    evaluator = RoseTTAFold3Evaluator(
        model_runner=object(),
        output_dir=tmp_path / "raw_model_results" / "rf3",
        rf3_conf=OmegaConf.create({}),
    )
    inputs = {
        "AL": Input(
            name="AL",
            chains=[
                _protein_chain(
                    sequence="ACDE",
                    unpaired_msa=">query\nACDE\n>hit\nACDF\n",
                ),
                Ligand(id="L", ccd_ids=["ATP"]),
            ],
            rng_seeds=[7],
            bonded_atom_pairs=[(("A", 2, "CA"), ("L", 1, "O1"))],
        )
    }

    prepared = evaluator.prepare_input(inputs, runtime_config={})
    request = prepared["AL"]

    assert isinstance(request, RF3RunnerRequest)
    assert request.request_id == "step_003_AL"
    assert request.output_dir == tmp_path / "raw_model_results" / "rf3" / "step_003_AL"
    assert request.template_selection == ["L/ATP/1/*"]
    assert request.ground_truth_conformer_selection == ["L/ATP/1/*"]
    assert request.rf3_overrides == {"seed": 7}

    payload = json.loads(request.prepared_input_path.read_text(encoding="utf-8"))
    assert payload["name"] == "step_003_AL"
    assert payload["components"] == [
        {
            "seq": "ACDE",
            "chain_id": "A",
            "msa_path": payload["components"][0]["msa_path"],
        },
        {
            "ccd_code": "ATP",
            "chain_id": "L",
        },
    ]
    assert Path(payload["components"][0]["msa_path"]).is_file()
    assert payload["bonds"] == [["A/CYS/2/CA", "L/ATP/1/O1"]]
    assert payload["template_selection"] == ["L/ATP/1/*"]
    assert payload["ground_truth_conformer_selection"] == ["L/ATP/1/*"]


def test_rf3_prepare_input_uses_fixed_ligand_config_path(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("channel", None)
    global_config.set_variable("step", 0)

    (tmp_path / "channel_input.pdb").write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N",
                "HETATM    2  C1  R13 B   1      14.500  11.200  10.000  1.00 20.00           C",
                "HETATM    3  C2  R13 B   1      15.500  11.200  10.000  1.00 20.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    evaluator = RoseTTAFold3Evaluator(
        model_runner=object(),
        output_dir=tmp_path / "raw_model_results" / "rf3",
        rf3_conf=OmegaConf.create(
            {
                "fixed_ligands": {
                    "B": {
                        "source": "structure_residue",
                        "structure_path": str(tmp_path / "channel_input.pdb"),
                        "residue": {
                            "chain_id": "B",
                            "resname": "R13",
                            "resseq": 1,
                            "icode": "",
                        },
                    }
                }
            }
        ),
    )
    inputs = {
        "AB": Input(
            name="AB",
            chains=[_protein_chain(sequence="ACDE"), Ligand(id="B", ccd_ids=["R13"])],
            rng_seeds=[7],
            bonded_atom_pairs=[(("A", 2, "CA"), ("B", 1, "C1"))],
        )
    }

    prepared = evaluator.prepare_input(inputs, runtime_config={"step": 0})
    request = prepared["AB"]
    payload = json.loads(request.prepared_input_path.read_text(encoding="utf-8"))

    assert payload["components"][1]["path"].endswith("step_000_AB__B_conformer.cif")
    assert Path(payload["components"][1]["path"]).exists()
    cif_text = Path(payload["components"][1]["path"]).read_text(encoding="utf-8")
    assert " R13 B " in cif_text
    assert " 1 B 1" in cif_text
    assert payload["template_selection"] == ["B/R13/1/*"]
    assert payload["ground_truth_conformer_selection"] == ["B/R13/1/*"]


def test_rf3_prepare_input_rewrites_fixed_ligand_source_residue_to_input_identity(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("channel", None)
    global_config.set_variable("step", 0)

    source_path = tmp_path / "channel_input.pdb"
    source_path.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N",
                "HETATM    2  C1  R13 L 101      14.500  11.200  10.000  1.00 20.00           C",
                "HETATM    3  C2  R13 L 101      15.500  11.200  10.000  1.00 20.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    evaluator = RoseTTAFold3Evaluator(
        model_runner=object(),
        output_dir=tmp_path / "raw_model_results" / "rf3",
        rf3_conf=OmegaConf.create(
            {
                "fixed_ligands": {
                    "B": {
                        "source": "structure_residue",
                        "structure_path": str(source_path),
                        "residue": {
                            "chain_id": "L",
                            "resname": "R13",
                            "resseq": 101,
                            "icode": "",
                        },
                    }
                },
                "template_selection": ["B/R13/1/*"],
                "ground_truth_conformer_selection": ["B/R13/1/*"],
            }
        ),
    )
    inputs = {
        "AB": Input(
            name="AB",
            chains=[_protein_chain(sequence="ACDE"), Ligand(id="B", ccd_ids=["R13"])],
            rng_seeds=[7],
            bonded_atom_pairs=[(("A", 1, "N"), ("B", 1, "C1"))],
        )
    }

    prepared = evaluator.prepare_input(inputs, runtime_config={"step": 0})
    request = prepared["AB"]
    payload = json.loads(request.prepared_input_path.read_text(encoding="utf-8"))
    cif_text = Path(payload["components"][1]["path"]).read_text(encoding="utf-8")

    assert payload["components"][1]["path"].endswith("step_000_AB__B_conformer.cif")
    assert " R13 B " in cif_text
    assert " 1 B 1" in cif_text
    assert "L 101" not in cif_text
    assert payload["bonds"] == [["A/ALA/1/N", "B/R13/1/C1"]]
    assert payload["template_selection"] == ["B/R13/1/*"]
    assert payload["ground_truth_conformer_selection"] == ["B/R13/1/*"]


def test_rf3_prepare_input_materializes_templated_protein_as_structure_start(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("channel", None)
    global_config.set_variable("step", 4)

    protein_template_pdb = _write_protein_pdb(
        tmp_path / "protein_template.pdb",
        chain_id="A",
        resnames=["ALA", "ALA", "ALA", "ALA"],
    )
    source_path = _write_ligand_pdb(
        tmp_path / "channel_input.pdb",
        chain_id="L",
        resname="R13",
        resseq=101,
        atoms=[
            ("C1", 14.5, 11.2, 10.0, "C"),
            ("C2", 15.5, 11.2, 10.0, "C"),
        ],
    )
    protein_template = Template(
        mmcif=_single_chain_mmcif_text(protein_template_pdb),
        query_to_template_map={0: 0, 1: 1, 2: 2, 3: 3},
    )

    evaluator = RoseTTAFold3Evaluator(
        model_runner=object(),
        output_dir=tmp_path / "raw_model_results" / "rf3",
        rf3_conf=OmegaConf.create(
            {
                "fixed_ligands": {
                    "B": {
                        "source": "structure_residue",
                        "structure_path": str(source_path),
                        "residue": {
                            "chain_id": "L",
                            "resname": "R13",
                            "resseq": 101,
                            "icode": "",
                        },
                    }
                },
                "template_selection": ["B/R13/1/*"],
                "ground_truth_conformer_selection": ["B/R13/1/*"],
            }
        ),
    )
    inputs = {
        "AB": Input(
            name="AB",
            chains=[
                _protein_chain(
                    sequence="GYSF",
                    unpaired_msa=">query\nGYSF\n",
                    templates=[protein_template],
                ),
                Ligand(id="B", ccd_ids=["R13"]),
            ],
            rng_seeds=[7],
            bonded_atom_pairs=[(("A", 2, "CA"), ("B", 1, "C1"))],
        )
    }

    prepared = evaluator.prepare_input(inputs, runtime_config={"step": 4})
    request = prepared["AB"]
    payload = json.loads(request.prepared_input_path.read_text(encoding="utf-8"))
    protein_component = payload["components"][0]

    assert set(protein_component) == {"path", "msa_paths"}
    assert protein_component["path"].endswith("step_004_AB__A_protein_template.pdb")
    assert protein_component["msa_paths"] == {"A": protein_component["msa_paths"]["A"]}
    assert Path(protein_component["msa_paths"]["A"]).is_file()
    protein_template_text = Path(protein_component["path"]).read_text(encoding="utf-8")
    assert " GLY A   1" in protein_template_text
    assert " TYR A   2" in protein_template_text
    assert " SER A   3" in protein_template_text
    assert " PHE A   4" in protein_template_text
    assert " ALA A" not in protein_template_text
    assert payload["components"][1]["path"].endswith("step_004_AB__B_conformer.cif")
    assert payload["bonds"] == [["A/TYR/2/CA", "B/R13/1/C1"]]
    assert payload["template_selection"] == ["B/R13/1/*", "A"]
    assert payload["ground_truth_conformer_selection"] == ["B/R13/1/*"]
    assert request.template_selection == ["B/R13/1/*", "A"]
    assert request.ground_truth_conformer_selection == ["B/R13/1/*"]


def test_rf3_prepare_input_uses_structure_residue_even_when_generated_standalone_exists(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("channel", None)
    global_config.set_variable("step", 0)

    source_path = _write_ligand_pdb(
        tmp_path / "channel_input.pdb",
        chain_id="B",
        resname="R13",
        resseq=1,
        atoms=[
            ("C1", 1.0, 2.0, 3.0, "C"),
            ("C2", 2.0, 2.0, 3.0, "C"),
        ],
    )
    standalone_path = _write_ligand_pdb(
        tmp_path / "raw_model_results" / "R13_0001.pdb",
        chain_id="X",
        resname="R13",
        resseq=1,
        atoms=[
            ("C1", 1.0, 2.0, 3.0, "C"),
            ("C2", 2.0, 2.0, 3.0, "C"),
            ("O1", 3.0, 2.0, 3.0, "O"),
        ],
    )
    assert standalone_path.exists()

    evaluator = RoseTTAFold3Evaluator(
        model_runner=object(),
        output_dir=tmp_path / "raw_model_results" / "rf3",
        rf3_conf=OmegaConf.create(
            {
                "fixed_ligands": {
                    "B": {
                        "source": "structure_residue",
                        "structure_path": str(source_path),
                        "residue": {
                            "chain_id": "B",
                            "resname": "R13",
                            "resseq": 1,
                            "icode": "",
                        },
                    }
                }
            }
        ),
    )
    inputs = {
        "AB": Input(
            name="AB",
            chains=[_protein_chain(sequence="ACDE"), Ligand(id="B", ccd_ids=["R13"])],
            rng_seeds=[7],
            bonded_atom_pairs=[(("A", 1, "N"), ("B", 1, "C1"))],
        )
    }

    prepared = evaluator.prepare_input(inputs, runtime_config={"step": 0})
    request = prepared["AB"]
    payload = json.loads(request.prepared_input_path.read_text(encoding="utf-8"))
    conformer_path = Path(payload["components"][1]["path"])

    assert conformer_path.exists()
    assert standalone_path.exists()
    assert _ligand_atom_names(conformer_path, chain_id="B", resname="R13") == ["C1", "C2"]
    assert request.rf3_overrides["add_missing_atoms"] is False


def test_rf3_prepare_input_uses_override_component_cif_name_for_bonds_and_selection(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("channel", None)
    global_config.set_variable("step", 0)

    component_path = _write_component_cif(tmp_path / "R13SB.cif")
    evaluator = RoseTTAFold3Evaluator(
        model_runner=object(),
        output_dir=tmp_path / "raw_model_results" / "rf3",
        rf3_conf=OmegaConf.create(
            {
                "fixed_ligands": {
                    "B": {
                        "source": "override",
                        "path": str(component_path),
                        "ccd_id": "R13SB",
                    }
                },
                "template_selection": ["B/R13SB/1/*"],
                "ground_truth_conformer_selection": ["B/R13SB/1/*"],
            }
        ),
    )
    inputs = {
        "AB": Input(
            name="AB",
            chains=[_protein_chain(sequence="ACDE"), Ligand(id="B", ccd_ids=["R13"])],
            rng_seeds=[7],
            bonded_atom_pairs=[(("A", 1, "N"), ("B", 1, "C15"))],
        )
    }

    prepared = evaluator.prepare_input(inputs, runtime_config={"step": 0})
    request = prepared["AB"]
    payload = json.loads(request.prepared_input_path.read_text(encoding="utf-8"))

    assert payload["components"][1] == {"path": str(component_path.resolve())}
    assert payload["bonds"] == [["A/ALA/1/N", "B/R13SB/1/C15"]]
    assert payload["template_selection"] == ["B/R13SB/1/*"]
    assert payload["ground_truth_conformer_selection"] == ["B/R13SB/1/*"]
    assert request.rf3_overrides["add_missing_atoms"] is False


def test_rf3_component_cif_atom_names_rejects_component_mismatch(tmp_path: Path) -> None:
    component_path = _write_component_cif(tmp_path / "R13SB.cif", component_id="R13SB")

    with pytest.raises(ValueError, match="_chem_comp.id='R13SB'.*expects 'R13'"):
        _chemical_component_atom_names_from_cif(
            component_path,
            component_id="R13",
        )


def test_rf3_setup_reuses_engine_for_identical_effective_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    init_calls: list[dict[str, object]] = []
    run_calls: list[dict[str, object]] = []

    class _FakeEngine:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)

        def run(self, **kwargs):
            run_calls.append(kwargs)
            out_dir = Path(kwargs["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            example_id = json.loads(Path(kwargs["inputs"]).read_text(encoding="utf-8"))["name"]
            (out_dir / f"{example_id}_model.cif").write_text("data_model\n", encoding="utf-8")
            (out_dir / f"{example_id}_summary_confidences.json").write_text(
                json.dumps({"overall_plddt": 91.2, "ptm": 0.61, "iptm": 0.72}),
                encoding="utf-8",
            )
            (out_dir / f"{example_id}_confidences.json").write_text("{}", encoding="utf-8")
            (out_dir / f"{example_id}_ranking_scores.csv").write_text(
                "seed,sample,ranking_score\n7,0,0.8\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        "astrolabe.evaluators.wrappers.rf3._load_rf3_inference_engine",
        lambda repo_path: _FakeEngine,
    )
    monkeypatch.setattr(
        "astrolabe.evaluators.wrappers.rf3._RUNNER_CACHE",
        {},
    )

    runner = rf3_setup(
        OmegaConf.create(
            {
                "repo_path": "/tmp/foundry",
                "ckpt_path": "/tmp/rf3.ckpt",
                "num_steps": 50,
            }
        )
    )
    input_path = tmp_path / "step_000_AL__input.json"
    input_path.write_text(json.dumps({"name": "step_000_AL"}), encoding="utf-8")
    request = RF3RunnerRequest(
        request_id="step_000_AL",
        prepared_input_path=input_path,
        output_dir=tmp_path / "rf3" / "step_000_AL",
        template_selection=["L/ATP/1/*"],
        ground_truth_conformer_selection=["L/ATP/1/*"],
        rf3_overrides={
            "fallback_conformer_to_input_coords": True,
            "early_stopping_plddt_threshold": None,
            "add_missing_atoms": False,
        },
    )

    response1 = runner.submit(request)
    response2 = runner.submit(request)

    assert response1.status == "success"
    assert response2.status == "success"
    assert len(init_calls) == 1
    assert len(run_calls) == 2
    assert init_calls[0]["fallback_conformer_to_input_coords"] is True
    assert init_calls[0]["early_stopping_plddt_threshold"] is None
    assert run_calls[0]["annotate_b_factor_with_plddt"] is True
    assert run_calls[0]["template_selection"] == ["L/ATP/1/*"]
    assert run_calls[0]["ground_truth_conformer_selection"] == ["L/ATP/1/*"]
    assert run_calls[0]["add_missing_atoms"] is False


def test_rf3_setup_skips_add_missing_atoms_when_engine_run_does_not_accept_it(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_calls: list[dict[str, object]] = []

    class _FakeEngine:
        def __init__(self, **kwargs):
            pass

        def initialize(self):
            return None

        def run(
            self,
            *,
            inputs,
            out_dir,
            annotate_b_factor_with_plddt,
            template_selection,
            ground_truth_conformer_selection,
        ):
            run_calls.append(
                {
                    "inputs": inputs,
                    "out_dir": out_dir,
                    "annotate_b_factor_with_plddt": annotate_b_factor_with_plddt,
                    "template_selection": template_selection,
                    "ground_truth_conformer_selection": ground_truth_conformer_selection,
                }
            )
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            example_id = json.loads(Path(inputs).read_text(encoding="utf-8"))["name"]
            (out_dir / f"{example_id}_model.cif").write_text("data_model\n", encoding="utf-8")
            (out_dir / f"{example_id}_summary_confidences.json").write_text(
                json.dumps({"overall_plddt": 91.2, "ptm": 0.61, "iptm": 0.72}),
                encoding="utf-8",
            )
            (out_dir / f"{example_id}_confidences.json").write_text("{}", encoding="utf-8")
            (out_dir / f"{example_id}_ranking_scores.csv").write_text(
                "seed,sample,ranking_score\n7,0,0.8\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        "astrolabe.evaluators.wrappers.rf3._load_rf3_inference_engine",
        lambda repo_path: _FakeEngine,
    )
    monkeypatch.setattr(
        "astrolabe.evaluators.wrappers.rf3._RUNNER_CACHE",
        {},
    )

    runner = rf3_setup(
        OmegaConf.create(
            {
                "repo_path": "/tmp/foundry",
                "ckpt_path": "/tmp/rf3.ckpt",
                "num_steps": 50,
            }
        )
    )
    input_path = tmp_path / "step_000_AL__input.json"
    input_path.write_text(json.dumps({"name": "step_000_AL"}), encoding="utf-8")
    request = RF3RunnerRequest(
        request_id="step_000_AL",
        prepared_input_path=input_path,
        output_dir=tmp_path / "rf3" / "step_000_AL",
        template_selection=["L/ATP/1/*"],
        ground_truth_conformer_selection=["L/ATP/1/*"],
        rf3_overrides={"add_missing_atoms": False},
    )

    response = runner.submit(request)

    assert response.status == "success"
    assert len(run_calls) == 1
    assert "add_missing_atoms" not in run_calls[0]


def test_rf3_prepare_output_maps_summary_confidence_to_standard_scores(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    evaluator = RoseTTAFold3Evaluator(
        model_runner=object(),
        output_dir=tmp_path / "raw_model_results" / "rf3",
        rf3_conf=OmegaConf.create({}),
    )
    inputs = {
        "AL": Input(
            name="AL",
            chains=[_protein_chain(sequence="ACDE"), Ligand(id="L", ccd_ids=["ATP"])],
            rng_seeds=[7],
        )
    }
    raw_output_dir = tmp_path / "raw_model_results" / "rf3" / "step_000_AL"
    raw_output_dir.mkdir(parents=True)
    raw_structure_path = raw_output_dir / "step_000_AL_model.cif"
    raw_structure_path.write_text("data_model\n#\n", encoding="utf-8")
    response = RF3RunnerResponse(
        status="success",
        request_id="step_000_AL",
        output_dir=raw_output_dir,
        predicted_structure_path=raw_structure_path,
        summary_confidence_path=raw_output_dir / "step_000_AL_summary_confidences.json",
        summary_confidences={
            "overall_plddt": 91.2,
            "ptm": 0.61,
            "iptm": 0.72,
            "chain_ptm": [87.0],
            "chain_pair_pae": [[None]],
            "chain_pair_pde": [[None]],
            "chain_pair_pae_min": [[None]],
            "chain_pair_pde_min": [[None]],
        },
        confidences_path=raw_output_dir / "step_000_AL_confidences.json",
        ranking_scores_path=raw_output_dir / "step_000_AL_ranking_scores.csv",
        error=None,
    )

    monkeypatch.setattr(
        "astrolabe.evaluators.rf3._extract_per_residue_plddt_from_structure",
        lambda structure_path: {
            "overall": np.array([0.91, 0.87, 0.83, 0.8], dtype=float),
            "by_chain": {
                "A": np.array([0.91, 0.87, 0.83, 0.8], dtype=float),
            },
        },
    )

    scores = evaluator.prepare_output({"AL": response}, inputs, runtime_config={})

    assert scores.names == ["AL"]
    assert np.allclose(scores.plddt["AL"][0], np.array([0.91, 0.87, 0.83, 0.8]))
    assert scores.ptm["AL"] == [0.61]
    assert scores.iptm["AL"] == [0.72]
    assert np.allclose(
        scores.per_proto_scores["A"][0],
        np.array([0.91, 0.87, 0.83, 0.8]),
    )
    assert scores.complex_paths["AL"] == tmp_path / "structures" / "step_000_AL_model.cif"


def test_rf3_prepare_output_materializes_canonical_structure_in_structures_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)

    evaluator = RoseTTAFold3Evaluator(
        model_runner=object(),
        output_dir=tmp_path / "raw_model_results" / "rf3",
        rf3_conf=OmegaConf.create({}),
    )
    inputs = {
        "AL": Input(
            name="AL",
            chains=[_protein_chain(sequence="ACDE"), Ligand(id="L", ccd_ids=["ATP"])],
            rng_seeds=[7],
        )
    }
    raw_output_dir = tmp_path / "raw_model_results" / "rf3" / "step_000_AL"
    raw_output_dir.mkdir(parents=True)
    raw_structure_path = raw_output_dir / "step_000_AL_model.cif"
    raw_structure_path.write_text("data_model\n#\n", encoding="utf-8")
    summary_path = raw_output_dir / "step_000_AL_summary_confidences.json"
    summary_path.write_text(
        json.dumps({"overall_plddt": 91.2, "ptm": 0.61, "iptm": 0.72}),
        encoding="utf-8",
    )

    response = RF3RunnerResponse(
        status="success",
        request_id="step_000_AL",
        output_dir=raw_output_dir,
        predicted_structure_path=raw_structure_path,
        summary_confidence_path=summary_path,
        summary_confidences={"overall_plddt": 91.2, "ptm": 0.61, "iptm": 0.72},
        confidences_path=raw_output_dir / "step_000_AL_confidences.json",
        ranking_scores_path=raw_output_dir / "step_000_AL_ranking_scores.csv",
        error=None,
    )

    monkeypatch.setattr(
        "astrolabe.evaluators.rf3._extract_per_residue_plddt_from_structure",
        lambda structure_path: {
            "overall": np.array([0.91, 0.87, 0.83, 0.8], dtype=float),
            "by_chain": {
                "A": np.array([0.91, 0.87, 0.83, 0.8], dtype=float),
            },
        },
    )

    scores = evaluator.prepare_output({"AL": response}, inputs, runtime_config={})

    canonical_path = tmp_path / "structures" / "step_000_AL_model.cif"
    assert canonical_path.exists()
    assert canonical_path.read_text(encoding="utf-8") == raw_structure_path.read_text(encoding="utf-8")
    assert scores.complex_paths["AL"] == canonical_path


def test_rf3_prepare_output_strips_fixed_ligand_model_to_reference_atom_set(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)

    reference_path = _write_ligand_pdb(
        tmp_path / "reference.pdb",
        chain_id="B",
        resname="R13",
        resseq=1,
        atoms=[
            ("C1", 1.0, 2.0, 3.0, "C"),
            ("C2", 2.0, 2.0, 3.0, "C"),
        ],
    )

    evaluator = RoseTTAFold3Evaluator(
        model_runner=object(),
        output_dir=tmp_path / "raw_model_results" / "rf3",
        rf3_conf=OmegaConf.create(
            {
                "fixed_ligands": {
                    "B": {
                        "source": "structure_residue",
                        "structure_path": str(reference_path),
                        "residue": {
                            "chain_id": "B",
                            "resname": "R13",
                            "resseq": 1,
                            "icode": "",
                        },
                    }
                }
            }
        ),
    )
    inputs = {
        "AB": Input(
            name="AB",
            chains=[_protein_chain(sequence="ACDE"), Ligand(id="B", ccd_ids=["R13"])],
            rng_seeds=[7],
        )
    }
    raw_output_dir = tmp_path / "raw_model_results" / "rf3" / "step_000_AB"
    raw_output_dir.mkdir(parents=True)
    raw_structure_path = _write_ligand_pdb(
        raw_output_dir / "step_000_AB_model.pdb",
        chain_id="B",
        resname="R13",
        resseq=1,
        atoms=[
            ("C1", 1.0, 2.0, 3.0, "C"),
            ("C2", 2.0, 2.0, 3.0, "C"),
            ("O1", 3.0, 2.0, 3.0, "O"),
        ],
    )
    summary_path = raw_output_dir / "step_000_AB_summary_confidences.json"
    summary_path.write_text(
        json.dumps({"overall_plddt": 91.2, "ptm": 0.61, "iptm": 0.72}),
        encoding="utf-8",
    )

    response = RF3RunnerResponse(
        status="success",
        request_id="step_000_AB",
        output_dir=raw_output_dir,
        predicted_structure_path=raw_structure_path,
        summary_confidence_path=summary_path,
        summary_confidences={"overall_plddt": 91.2, "ptm": 0.61, "iptm": 0.72},
        confidences_path=raw_output_dir / "step_000_AB_confidences.json",
        ranking_scores_path=raw_output_dir / "step_000_AB_ranking_scores.csv",
        error=None,
    )

    monkeypatch.setattr(
        "astrolabe.evaluators.rf3._extract_per_residue_plddt_from_structure",
        lambda structure_path: {
            "overall": np.array([0.91, 0.87, 0.83, 0.8], dtype=float),
            "by_chain": {
                "A": np.array([0.91, 0.87, 0.83, 0.8], dtype=float),
            },
        },
    )

    scores = evaluator.prepare_output({"AB": response}, inputs, runtime_config={})

    canonical_path = tmp_path / "structures" / "step_000_AB_model.pdb"
    assert _ligand_atom_names(raw_structure_path, chain_id="B", resname="R13") == ["C1", "C2"]
    assert _ligand_atom_names(canonical_path, chain_id="B", resname="R13") == ["C1", "C2"]
    assert scores.complex_paths["AB"] == canonical_path


def test_rf3_prepare_output_uses_structure_residue_atom_set_for_structure_residue_fixed_ligand(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)

    reference_path = _write_ligand_pdb(
        tmp_path / "reference.pdb",
        chain_id="B",
        resname="R13",
        resseq=1,
        atoms=[
            ("C1", 1.0, 2.0, 3.0, "C"),
            ("C2", 2.0, 2.0, 3.0, "C"),
        ],
    )
    standalone_path = _write_ligand_pdb(
        tmp_path / "raw_model_results" / "R13_0001.pdb",
        chain_id="X",
        resname="R13",
        resseq=1,
        atoms=[
            ("C1", 1.0, 2.0, 3.0, "C"),
            ("C2", 2.0, 2.0, 3.0, "C"),
            ("O1", 3.0, 2.0, 3.0, "O"),
        ],
    )
    assert standalone_path.exists()

    evaluator = RoseTTAFold3Evaluator(
        model_runner=object(),
        output_dir=tmp_path / "raw_model_results" / "rf3",
        rf3_conf=OmegaConf.create(
            {
                "fixed_ligands": {
                    "B": {
                        "source": "structure_residue",
                        "structure_path": str(reference_path),
                        "residue": {
                            "chain_id": "B",
                            "resname": "R13",
                            "resseq": 1,
                            "icode": "",
                        },
                    }
                }
            }
        ),
    )
    inputs = {
        "AB": Input(
            name="AB",
            chains=[_protein_chain(sequence="ACDE"), Ligand(id="B", ccd_ids=["R13"])],
            rng_seeds=[7],
        )
    }
    raw_output_dir = tmp_path / "raw_model_results" / "rf3" / "step_000_AB"
    raw_output_dir.mkdir(parents=True)
    raw_structure_path = _write_ligand_pdb(
        raw_output_dir / "step_000_AB_model.pdb",
        chain_id="B",
        resname="R13",
        resseq=1,
        atoms=[
            ("C1", 1.0, 2.0, 3.0, "C"),
            ("C2", 2.0, 2.0, 3.0, "C"),
            ("O1", 3.0, 2.0, 3.0, "O"),
        ],
    )
    summary_path = raw_output_dir / "step_000_AB_summary_confidences.json"
    summary_path.write_text(
        json.dumps({"overall_plddt": 91.2, "ptm": 0.61, "iptm": 0.72}),
        encoding="utf-8",
    )

    response = RF3RunnerResponse(
        status="success",
        request_id="step_000_AB",
        output_dir=raw_output_dir,
        predicted_structure_path=raw_structure_path,
        summary_confidence_path=summary_path,
        summary_confidences={"overall_plddt": 91.2, "ptm": 0.61, "iptm": 0.72},
        confidences_path=raw_output_dir / "step_000_AB_confidences.json",
        ranking_scores_path=raw_output_dir / "step_000_AB_ranking_scores.csv",
        error=None,
    )

    monkeypatch.setattr(
        "astrolabe.evaluators.rf3._extract_per_residue_plddt_from_structure",
        lambda structure_path: {
            "overall": np.array([0.91, 0.87, 0.83, 0.8], dtype=float),
            "by_chain": {
                "A": np.array([0.91, 0.87, 0.83, 0.8], dtype=float),
            },
        },
    )

    scores = evaluator.prepare_output({"AB": response}, inputs, runtime_config={})

    canonical_path = tmp_path / "structures" / "step_000_AB_model.pdb"
    assert _ligand_atom_names(raw_structure_path, chain_id="B", resname="R13") == ["C1", "C2"]
    assert _ligand_atom_names(canonical_path, chain_id="B", resname="R13") == ["C1", "C2"]
    assert scores.complex_paths["AB"] == canonical_path


def test_rf3_prepare_output_trims_component_cif_and_renames_for_rosetta(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)

    component_path = _write_component_cif(tmp_path / "R13SB.cif")
    evaluator = RoseTTAFold3Evaluator(
        model_runner=object(),
        output_dir=tmp_path / "raw_model_results" / "rf3",
        rf3_conf=OmegaConf.create(
            {
                "fixed_ligands": {
                    "B": {
                        "source": "override",
                        "path": str(component_path),
                        "ccd_id": "R13SB",
                        "output_resname": "R13",
                    }
                }
            }
        ),
    )
    inputs = {
        "AB": Input(
            name="AB",
            chains=[_protein_chain(sequence="ACDE"), Ligand(id="B", ccd_ids=["R13"])],
            rng_seeds=[7],
        )
    }
    raw_output_dir = tmp_path / "raw_model_results" / "rf3" / "step_000_AB"
    raw_output_dir.mkdir(parents=True)
    raw_structure_path = _write_ligand_cif(
        raw_output_dir / "step_000_AB_model.cif",
        chain_id="B",
        resname="R13SB",
        resseq=1,
        atoms=[
            ("C1", 1.0, 2.0, 3.0, "C"),
            ("C15", 2.0, 2.0, 3.0, "C"),
            ("O1", 3.0, 2.0, 3.0, "O"),
        ],
    )
    summary_path = raw_output_dir / "step_000_AB_summary_confidences.json"
    summary_path.write_text(
        json.dumps({"overall_plddt": 91.2, "ptm": 0.61, "iptm": 0.72}),
        encoding="utf-8",
    )

    response = RF3RunnerResponse(
        status="success",
        request_id="step_000_AB",
        output_dir=raw_output_dir,
        predicted_structure_path=raw_structure_path,
        summary_confidence_path=summary_path,
        summary_confidences={"overall_plddt": 91.2, "ptm": 0.61, "iptm": 0.72},
        confidences_path=raw_output_dir / "step_000_AB_confidences.json",
        ranking_scores_path=raw_output_dir / "step_000_AB_ranking_scores.csv",
        error=None,
    )

    monkeypatch.setattr(
        "astrolabe.evaluators.rf3._extract_per_residue_plddt_from_structure",
        lambda structure_path: {
            "overall": np.array([0.91, 0.87, 0.83, 0.8], dtype=float),
            "by_chain": {
                "A": np.array([0.91, 0.87, 0.83, 0.8], dtype=float),
            },
        },
    )

    scores = evaluator.prepare_output({"AB": response}, inputs, runtime_config={})

    canonical_path = tmp_path / "structures" / "step_000_AB_model.cif"
    assert _ligand_atom_names(raw_structure_path, chain_id="B", resname="R13") == ["C1", "C15"]
    assert _ligand_atom_names(raw_structure_path, chain_id="B", resname="R13SB") == []
    assert _ligand_atom_names(canonical_path, chain_id="B", resname="R13") == ["C1", "C15"]
    assert _ligand_atom_names(canonical_path, chain_id="B", resname="R13SB") == []
    assert scores.complex_paths["AB"] == canonical_path


def test_rf3_runner_response_discovers_nested_foundry_output_layout(tmp_path: Path) -> None:
    input_path = tmp_path / "step_000_AL__input.json"
    input_path.write_text(json.dumps({"name": "step_000_AL"}), encoding="utf-8")
    nested = tmp_path / "step_000_AL"
    nested.mkdir(parents=True)
    (nested / "step_000_AL_model.cif").write_text("data_model\n", encoding="utf-8")
    (nested / "step_000_AL_summary_confidences.json").write_text(
        json.dumps({"overall_plddt": 91.2, "ptm": 0.61, "iptm": 0.72}),
        encoding="utf-8",
    )
    (nested / "step_000_AL_confidences.json").write_text("{}", encoding="utf-8")
    (nested / "step_000_AL_ranking_scores.csv").write_text(
        "seed,sample,ranking_score\n7,0,0.8\n",
        encoding="utf-8",
    )

    response = RF3RunnerService._build_response(
        RF3RunnerRequest(
            request_id="step_000_AL",
            prepared_input_path=input_path,
            output_dir=tmp_path,
            template_selection=["L/ATP/1/*"],
            ground_truth_conformer_selection=["L/ATP/1/*"],
            rf3_overrides={},
        )
    )

    assert response.status == "success"
    assert response.predicted_structure_path == nested / "step_000_AL_model.cif"
    assert response.summary_confidence_path == nested / "step_000_AL_summary_confidences.json"
    assert response.confidences_path == nested / "step_000_AL_confidences.json"
    assert response.ranking_scores_path == nested / "step_000_AL_ranking_scores.csv"
    assert response.summary_confidences["iptm"] == 0.72


def test_rf3_runner_uses_external_python_daemon_when_python_bin_is_set(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakeInput:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def write(self, text: str) -> None:
            self.lines.append(text)

        def flush(self) -> None:
            return None

    class _FakeOutput:
        def __init__(self, lines: list[str]) -> None:
            self._lines = list(lines)

        def readline(self) -> str:
            if self._lines:
                return self._lines.pop(0)
            return ""

    class _FakeProc:
        def __init__(self, lines: list[str]) -> None:
            self.stdin = _FakeInput()
            self.stdout = _FakeOutput(lines)
            self._terminated = False

        def poll(self):
            return None if not self._terminated else 0

        def terminate(self) -> None:
            self._terminated = True

    popen_calls: list[dict[str, object]] = []

    def _fake_popen(*, args, cwd, stdin, stdout, stderr, text, bufsize):
        proc = _FakeProc(
            [json.dumps({"type": "result", "id": "step_000_AL", "status": "success"}) + "\n"]
        )
        popen_calls.append({"args": args, "cwd": cwd, "proc": proc})
        return proc

    monkeypatch.setattr(
        "astrolabe.evaluators.wrappers.rf3.subprocess.Popen",
        _fake_popen,
    )
    monkeypatch.setattr(
        "astrolabe.evaluators.wrappers.rf3._RUNNER_CACHE",
        {},
    )

    monkeypatch.setattr(
        RF3RunnerService,
        "_build_response",
        staticmethod(
            lambda request: RF3RunnerResponse(
                status="success",
                request_id=request.request_id,
                output_dir=request.output_dir,
                predicted_structure_path=None,
                summary_confidence_path=None,
                summary_confidences=None,
                confidences_path=None,
                ranking_scores_path=None,
                error=None,
            )
        ),
    )

    python_bin = tmp_path / "foundry_python"
    python_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    python_bin.chmod(0o755)

    runner = rf3_setup(
        OmegaConf.create(
            {
                "repo_path": "/tmp/foundry",
                "ckpt_path": "/tmp/rf3.ckpt",
                "python_bin": str(python_bin),
                "num_steps": 50,
            }
        )
    )
    input_path = tmp_path / "step_000_AL__input.json"
    input_path.write_text(json.dumps({"name": "step_000_AL"}), encoding="utf-8")
    request = RF3RunnerRequest(
        request_id="step_000_AL",
        prepared_input_path=input_path,
        output_dir=tmp_path / "rf3" / "step_000_AL",
        template_selection=["L/ATP/1/*"],
        ground_truth_conformer_selection=["L/ATP/1/*"],
        rf3_overrides={
            "fallback_conformer_to_input_coords": True,
            "early_stopping_plddt_threshold": None,
            "seed": 123,
        },
    )

    response = runner.submit(request)

    assert response.status == "success"
    assert len(popen_calls) == 1
    assert popen_calls[0]["args"][0] == str(python_bin)
    payload = json.loads(popen_calls[0]["proc"].stdin.lines[0])
    assert payload["config"]["repo_path"] == "/tmp/foundry"
    assert payload["config"]["ckpt_path"] == "/tmp/rf3.ckpt"
    assert payload["config"]["fallback_conformer_to_input_coords"] is True
    assert payload["config"]["early_stopping_plddt_threshold"] is None
    assert payload["config"]["seed"] == 123


def test_rf3_setup_strips_request_only_meta_keys_from_engine_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    init_calls: list[dict[str, object]] = []

    class _FakeEngine:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)

        def initialize(self):
            return None

        def run(self, **kwargs):
            out_dir = Path(kwargs["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            example_id = json.loads(Path(kwargs["inputs"]).read_text(encoding="utf-8"))["name"]
            (out_dir / f"{example_id}_model.cif").write_text("data_model\n", encoding="utf-8")
            (out_dir / f"{example_id}_summary_confidences.json").write_text(
                json.dumps({"overall_plddt": 91.2, "ptm": 0.61, "iptm": 0.72}),
                encoding="utf-8",
            )
            (out_dir / f"{example_id}_confidences.json").write_text("{}", encoding="utf-8")
            (out_dir / f"{example_id}_ranking_scores.csv").write_text(
                "seed,sample,ranking_score\n7,0,0.8\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        "astrolabe.evaluators.wrappers.rf3._load_rf3_inference_engine",
        lambda repo_path: _FakeEngine,
    )
    monkeypatch.setattr(
        "astrolabe.evaluators.wrappers.rf3._RUNNER_CACHE",
        {},
    )

    runner = rf3_setup(
        OmegaConf.create(
            {
                "repo_path": "/tmp/foundry",
                "ckpt_path": "/tmp/rf3.ckpt",
                "num_steps": 50,
                "template_selection": ["B/R13/1/*"],
                "ground_truth_conformer_selection": ["B/R13/1/*"],
                "fixed_ligands": {
                    "B": {
                        "source": "structure_residue",
                        "structure_path": "/tmp/example.pdb",
                        "residue": {
                            "chain_id": "L",
                            "resname": "R13",
                            "resseq": 101,
                            "icode": "",
                        },
                    }
                },
            }
        )
    )
    input_path = tmp_path / "step_000_AB__input.json"
    input_path.write_text(json.dumps({"name": "step_000_AB"}), encoding="utf-8")
    request = RF3RunnerRequest(
        request_id="step_000_AB",
        prepared_input_path=input_path,
        output_dir=tmp_path / "rf3" / "step_000_AB",
        template_selection=["B/R13/1/*"],
        ground_truth_conformer_selection=["B/R13/1/*"],
        rf3_overrides={"seed": 321},
    )

    response = runner.submit(request)

    assert response.status == "success"
    assert len(init_calls) == 1
    assert "template_selection" not in init_calls[0]
    assert "ground_truth_conformer_selection" not in init_calls[0]
    assert "fixed_ligands" not in init_calls[0]
    assert init_calls[0]["seed"] == 321


def test_rf3_setup_strips_null_fallback_conformer_from_engine_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    init_calls: list[dict[str, object]] = []

    class _FakeEngine:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)

        def initialize(self):
            return None

        def run(self, **kwargs):
            out_dir = Path(kwargs["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            example_id = json.loads(Path(kwargs["inputs"]).read_text(encoding="utf-8"))["name"]
            (out_dir / f"{example_id}_model.cif").write_text("data_model\n", encoding="utf-8")
            (out_dir / f"{example_id}_summary_confidences.json").write_text(
                json.dumps({"overall_plddt": 91.2, "ptm": 0.61, "iptm": 0.72}),
                encoding="utf-8",
            )
            (out_dir / f"{example_id}_confidences.json").write_text("{}", encoding="utf-8")
            (out_dir / f"{example_id}_ranking_scores.csv").write_text(
                "seed,sample,ranking_score\n7,0,0.8\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        "astrolabe.evaluators.wrappers.rf3._load_rf3_inference_engine",
        lambda repo_path: _FakeEngine,
    )
    monkeypatch.setattr(
        "astrolabe.evaluators.wrappers.rf3._RUNNER_CACHE",
        {},
    )

    runner = rf3_setup(
        OmegaConf.create(
            {
                "repo_path": "/tmp/foundry",
                "ckpt_path": "/tmp/rf3.ckpt",
                "fallback_conformer_to_input_coords": None,
            }
        )
    )
    input_path = tmp_path / "step_000_AB__input.json"
    input_path.write_text(json.dumps({"name": "step_000_AB"}), encoding="utf-8")
    request = RF3RunnerRequest(
        request_id="step_000_AB",
        prepared_input_path=input_path,
        output_dir=tmp_path / "rf3" / "step_000_AB",
        template_selection=["B/R13/1/*"],
        ground_truth_conformer_selection=["B/R13/1/*"],
        rf3_overrides={},
    )

    response = runner.submit(request)

    assert response.status == "success"
    assert len(init_calls) == 1
    assert "fallback_conformer_to_input_coords" not in init_calls[0]


def test_rf3_setup_restores_post_init_rng_state_per_request(monkeypatch, tmp_path: Path) -> None:
    init_calls: list[dict[str, object]] = []
    run_calls: list[dict[str, object]] = []
    restore_calls: list[int] = []
    capture_calls: list[int] = []

    class _FakeEngine:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)
            self.seed = kwargs.get("seed")

        def initialize(self):
            return None

        def run(self, **kwargs):
            run_calls.append(kwargs)
            out_dir = Path(kwargs["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            example_id = json.loads(Path(kwargs["inputs"]).read_text(encoding="utf-8"))["name"]
            (out_dir / f"{example_id}_model.cif").write_text("data_model\n", encoding="utf-8")
            (out_dir / f"{example_id}_summary_confidences.json").write_text(
                json.dumps({"overall_plddt": 91.2, "ptm": 0.61, "iptm": 0.72}),
                encoding="utf-8",
            )
            (out_dir / f"{example_id}_confidences.json").write_text("{}", encoding="utf-8")
            (out_dir / f"{example_id}_ranking_scores.csv").write_text(
                "seed,sample,ranking_score\n7,0,0.8\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        "astrolabe.evaluators.wrappers.rf3._load_rf3_inference_engine",
        lambda repo_path: _FakeEngine,
    )
    monkeypatch.setattr(
        "astrolabe.evaluators.wrappers.rf3._RUNNER_CACHE",
        {},
    )
    monkeypatch.setattr(
        "astrolabe.evaluators.wrappers.rf3._capture_rng_state",
        lambda: capture_calls.append(len(init_calls)) or f"snapshot-{len(init_calls)}",
    )
    monkeypatch.setattr(
        "astrolabe.evaluators.wrappers.rf3._restore_rng_state",
        lambda snapshot: restore_calls.append(int(str(snapshot).split('-')[-1])),
    )

    runner = rf3_setup(
        OmegaConf.create(
            {
                "repo_path": "/tmp/foundry",
                "ckpt_path": "/tmp/rf3.ckpt",
                "num_steps": 50,
            }
        )
    )
    input_path = tmp_path / "step_000_AB__input.json"
    input_path.write_text(json.dumps({"name": "step_000_AB"}), encoding="utf-8")
    request = RF3RunnerRequest(
        request_id="step_000_AB",
        prepared_input_path=input_path,
        output_dir=tmp_path / "rf3" / "step_000_AB",
        template_selection=["B/R13/1/*"],
        ground_truth_conformer_selection=["B/R13/1/*"],
        rf3_overrides={"seed": 321},
    )
    request_repeat = RF3RunnerRequest(
        request_id="step_001_AB",
        prepared_input_path=input_path,
        output_dir=tmp_path / "rf3" / "step_001_AB",
        template_selection=["B/R13/1/*"],
        ground_truth_conformer_selection=["B/R13/1/*"],
        rf3_overrides={"seed": 321},
    )
    request_new_seed = RF3RunnerRequest(
        request_id="step_002_AB",
        prepared_input_path=input_path,
        output_dir=tmp_path / "rf3" / "step_002_AB",
        template_selection=["B/R13/1/*"],
        ground_truth_conformer_selection=["B/R13/1/*"],
        rf3_overrides={"seed": 123},
    )

    response = runner.submit(request)
    response_repeat = runner.submit(request_repeat)
    response_new_seed = runner.submit(request_new_seed)

    assert response.status == "success"
    assert response_repeat.status == "success"
    assert response_new_seed.status == "success"
    assert len(init_calls) == 2
    assert init_calls[0]["seed"] == 321
    assert init_calls[1]["seed"] == 123
    assert len(run_calls) == 3
    assert capture_calls == [1, 2]
    assert restore_calls == [1, 1, 2]
