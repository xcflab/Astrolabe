from __future__ import annotations

import sys
import shutil
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import astrolabe.cli as cli


CONFIG_DIR = Path(cli.__file__).resolve().parents[1] / "config"


@pytest.fixture(autouse=True)
def _patch_default_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, config_dir, ignore=shutil.ignore_patterns("__pycache__"))
    (config_dir / "env").mkdir(exist_ok=True)
    (config_dir / "env" / "env_local.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_default_config_dir", lambda: config_dir)


def _run(argv: list[str]) -> int:
    parser = cli._build_parser()
    ns = parser.parse_args(argv)
    handler = getattr(ns, "_handler", None)
    assert handler is not None
    return int(handler(ns))


def _write_channel_input_pdb(
    path: Path,
    *,
    bond_record: str = "LINK",
    ligand_resname: str = "LIG",
    ligand_atom_name: str = "C1",
) -> Path:
    lines = [
        "ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N",
        "ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00 20.00           C",
        "ATOM      3  N   CYS A   2      12.000  10.000  10.000  1.00 20.00           N",
        "ATOM      4  CA  CYS A   2      13.000  10.000  10.000  1.00 20.00           C",
        "ATOM      5  SG  CYS A   2      14.000  10.000  10.000  1.00 20.00           S",
        "ATOM      6  N   GLY A   3      15.000  10.000  10.000  1.00 20.00           N",
        "ATOM      7  CA  GLY A   3      16.000  10.000  10.000  1.00 20.00           C",
        "ATOM      8  N   SER A   4      17.000  10.000  10.000  1.00 20.00           N",
        "ATOM      9  CA  SER A   4      18.000  10.000  10.000  1.00 20.00           C",
        f"HETATM   10  {ligand_atom_name:<4}{ligand_resname:>3} L 101      14.500  11.200  10.000  1.00 20.00           C",
        f"HETATM   11  O1  {ligand_resname:>3} L 101      15.500  11.200  10.000  1.00 20.00           O",
    ]
    if bond_record == "LINK":
        lines.append(
            f"LINK         SG  CYS A   2                 {ligand_atom_name:<4}{ligand_resname:>3} L 101     1555   1555  1.39"
        )
    elif bond_record == "CONECT":
        lines.extend(
            [
                "CONECT    5   10",
                "CONECT   10    5   11",
            ]
        )
    else:
        raise ValueError(f"Unsupported bond_record: {bond_record!r}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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


def test_cli_refine_channel_dry_run_extracts_structure_inputs(
    tmp_path: Path,
    capsys,
) -> None:
    structure = _write_channel_input_pdb(
        tmp_path / "channel_input.pdb",
        bond_record="LINK",
    )
    mutable_positions = tmp_path / "mutable_positions.txt"
    mutable_positions.write_text("2-3\n", encoding="utf-8")
    spheres = tmp_path / "spheres.pdb"
    spheres.write_text("REMARK spheres\nEND\n", encoding="utf-8")
    msa = tmp_path / "protein.a3m"
    msa.write_text(">query\nACGS\n", encoding="utf-8")

    rc = _run(
        [
            "refine",
            "channel",
            "--task-name",
            "channel_case",
            "--out-dir",
            str(tmp_path / "outputs"),
            "--structure",
            str(structure),
            "--protein-chain",
            "A",
            "--ligand-residue",
            "L:LIG:101",
            "--mutable-positions",
            str(mutable_positions),
            "--spheres",
            str(spheres),
            "--msa",
            str(msa),
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.seqs.A == "+A CG +S"
    assert cfg.seqs.B == "LIG"
    assert cfg.types.B == "c"
    assert str(cfg.homologies.A) == str(msa.resolve())
    assert cfg.evaluator.evaluator_terms.RF3._type == "rf3"
    assert cfg.mutator.LigandMPNN.temperature == 0.3
    assert cfg.mutator.LigandMPNN.ligand_mpnn_use_side_chain_context == 1
    assert cfg.mutator.mutator_terms.LMPNN_channel.temperature == 0.3
    assert "channel" not in cfg
    assert cfg.covalent_bonds[0].atom1.chain_id == "A"
    assert cfg.covalent_bonds[0].atom1.resseq == 2
    assert cfg.covalent_bonds[0].atom1.atom_name == "SG"
    assert cfg.covalent_bonds[0].atom2.chain_id == "B"
    assert cfg.covalent_bonds[0].atom2.resseq == 1
    assert cfg.covalent_bonds[0].atom2.atom_name == "C1"
    assert cfg.evaluator.rf3.fixed_ligands.B.source == "structure_residue"
    assert cfg.evaluator.rf3.fixed_ligands.B.structure_path == str(structure.resolve())
    assert cfg.evaluator.rf3.fixed_ligands.B.residue.chain_id == "L"
    assert cfg.evaluator.rf3.fixed_ligands.B.residue.resseq == 101
    assert cfg.evaluator.rf3.fixed_ligands.B.residue.resname == "LIG"
    assert cfg.evaluator.rf3.template_selection == ["B/LIG/1/*"]
    assert cfg.evaluator.rf3.ground_truth_conformer_selection == ["B/LIG/1/*"]
    assert cfg.loss.loss_terms.channel_hole.reference_structure_path == str(structure.resolve())
    assert cfg.loss.loss_terms.channel_hole.alignment_chains == "A"
    assert cfg.loss.loss_terms.channel_hole.reference_spheres_path == str(spheres.resolve())
    assert cfg.loss.expression_terms.channel_confidence_gate == (
        "channel_iptm_gate * channel_ptm_gate * channel_plddt_gate"
    )
    assert cfg.loss.expression_terms.channel_radius_bonus == (
        "sigmoid((best_channel_min_radius - 1.2) / 0.2)"
    )
    assert cfg.stages.default.weighted_score_expression == (
        "channel_hole + 0.5 * channel_radius_bonus + 0.05 * channel_confidence_gate"
    )
    assert "hole_surface(" in str(cfg.mutator.mutator_terms.LMPNN_channel.position_selector)
    assert "not ligand_contact(distance_max=5.0)" in str(
        cfg.mutator.mutator_terms.LMPNN_channel.position_selector
    )


def test_cli_refine_channel_accepts_chain_prefixed_mutable_positions(
    tmp_path: Path,
    capsys,
) -> None:
    structure = _write_channel_input_pdb(tmp_path / "channel_input_chain_prefix.pdb")
    mutable_positions = tmp_path / "mutable_positions.txt"
    mutable_positions.write_text("A2,A3\n", encoding="utf-8")
    spheres = tmp_path / "spheres.pdb"
    spheres.write_text("REMARK spheres\nEND\n", encoding="utf-8")

    rc = _run(
        [
            "refine",
            "channel",
            "--task-name",
            "channel_case_chain_prefix",
            "--out-dir",
            str(tmp_path / "outputs"),
            "--structure",
            str(structure),
            "--protein-chain",
            "A",
            "--ligand-residue",
            "L:LIG:101",
            "--mutable-positions",
            str(mutable_positions),
            "--spheres",
            str(spheres),
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.seqs.A == "+A CG +S"


def test_cli_refine_channel_uses_ligand_conformer_override_and_dump_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    structure = _write_channel_input_pdb(
        tmp_path / "channel_input_override.pdb",
        bond_record="CONECT",
    )
    mutable_positions = tmp_path / "mutable_positions.txt"
    mutable_positions.write_text("2-4\n", encoding="utf-8")
    spheres = tmp_path / "spheres.pdb"
    spheres.write_text("REMARK spheres\nEND\n", encoding="utf-8")
    override = tmp_path / "ligand_override.sdf"
    override.write_text("override\n", encoding="utf-8")
    dump_path = tmp_path / "composed.yaml"

    captured: dict[str, object] = {}

    def _fake_run_from_config(cfg, *, autoselect_gpu: bool) -> None:
        captured["cfg"] = OmegaConf.to_container(cfg, resolve=False)
        captured["autoselect_gpu"] = autoselect_gpu

    monkeypatch.setattr(cli, "_run_from_config", _fake_run_from_config)

    rc = _run(
        [
            "refine",
            "channel",
            "--task-name",
            "channel_override",
            "--out-dir",
            str(tmp_path / "outputs"),
            "--structure",
            str(structure),
            "--protein-chain",
            "A",
            "--ligand-residue",
            "L:LIG:101",
            "--mutable-positions",
            str(mutable_positions),
            "--spheres",
            str(spheres),
            "--ligand-conformer",
            str(override),
            "--dump-config",
            str(dump_path),
        ]
    )

    assert rc == 0
    assert dump_path.exists()
    dumped = OmegaConf.load(dump_path)
    assert dumped.evaluator.rf3.fixed_ligands.B.source == "override"
    assert dumped.evaluator.rf3.fixed_ligands.B.path == str(override.resolve())
    assert dumped.evaluator.rf3.fixed_ligands.B.residue.resname == "LIG"
    assert captured["autoselect_gpu"] is False
    captured_cfg = OmegaConf.create(captured["cfg"])
    assert captured_cfg.evaluator.rf3.fixed_ligands.B.source == "override"
    assert captured_cfg.evaluator.rf3.fixed_ligands.B.path == str(override.resolve())


def test_cli_refine_channel_uses_chemical_component_cif_for_rf3(
    tmp_path: Path,
    capsys,
) -> None:
    structure = _write_channel_input_pdb(
        tmp_path / "channel_input_r13.pdb",
        ligand_resname="R13",
        ligand_atom_name="C15",
    )
    mutable_positions = tmp_path / "mutable_positions.txt"
    mutable_positions.write_text("2-4\n", encoding="utf-8")
    spheres = tmp_path / "spheres.pdb"
    spheres.write_text("REMARK spheres\nEND\n", encoding="utf-8")
    component = _write_component_cif(tmp_path / "R13SB.cif")

    rc = _run(
        [
            "refine",
            "channel",
            "--task-name",
            "channel_r13sb",
            "--out-dir",
            str(tmp_path / "outputs"),
            "--structure",
            str(structure),
            "--protein-chain",
            "A",
            "--ligand-residue",
            "L:R13:101",
            "--mutable-positions",
            str(mutable_positions),
            "--spheres",
            str(spheres),
            "--ligand-conformer",
            str(component),
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.seqs.B == "R13SB"
    assert cfg.ligand_resnames.B == "R13"
    assert "data_R13SB" in str(cfg.user_ccd)
    assert cfg.covalent_bonds[0].atom2.atom_name == "C15"
    assert cfg.evaluator.rf3.fixed_ligands.B.source == "override"
    assert cfg.evaluator.rf3.fixed_ligands.B.path == str(component.resolve())
    assert cfg.evaluator.rf3.fixed_ligands.B.ccd_id == "R13SB"
    assert cfg.evaluator.rf3.fixed_ligands.B.output_resname == "R13"
    assert cfg.evaluator.rf3.fixed_ligands.B.residue.resname == "R13"
    assert cfg.evaluator.rf3.template_selection == ["B/R13SB/1/*"]
    assert cfg.evaluator.rf3.ground_truth_conformer_selection == ["B/R13SB/1/*"]


def test_cli_posthoc_channel_screen_wires_arguments(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    structure = _write_channel_input_pdb(
        tmp_path / "channel_screen_input.pdb",
        bond_record="LINK",
    )
    mutable_positions = tmp_path / "mutable_positions.txt"
    mutable_positions.write_text("2-4\n", encoding="utf-8")
    spheres = tmp_path / "spheres.pdb"
    spheres.write_text("REMARK spheres\nEND\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run_pocket_fixed_screen(**kwargs):
        captured.update(kwargs)
        out_root = Path(kwargs["output_root"])
        return {
            "structure_path": str(kwargs["structure_path"]),
            "output_root": str(out_root),
            "contact_distance": float(kwargs["contact_distance"]),
            "mutable_positions_count": 3,
            "protected_positions_count": 2,
            "design_positions_count": 1,
            "protected_residue_ids": ["A2", "A3"],
            "design_residue_ids": ["A4"],
            "target_aas": ["A", "T", "S", "G"],
            "best_variant": "pocket_fixed_all_ser",
            "best_variant_min_radius": 1.42,
            "plan_json": str(out_root / "01_prepare" / "pocket_fixed_screen_plan.json"),
            "summary_json": str(out_root / "03_collect" / "pocket_fixed_screen_summary.json"),
            "summary_csv": str(out_root / "03_collect" / "pocket_fixed_screen_summary.csv"),
            "rows": [],
        }

    fake_module = types.ModuleType("astrolabe.posthoc.rosetta_direct_channel")
    fake_module.run_pocket_fixed_screen = _fake_run_pocket_fixed_screen
    monkeypatch.setitem(sys.modules, "astrolabe.posthoc.rosetta_direct_channel", fake_module)

    rc = _run(
        [
            "posthoc",
            "channel-screen",
            "--structure",
            str(structure),
            "--protein-chain",
            "A",
            "--ligand-residue",
            "L:LIG:101",
            "--mutable-positions",
            str(mutable_positions),
            "--spheres",
            str(spheres),
            "--target-aas",
            "ala|thr|ser|gly",
            "--contact-distance",
            "4.5",
            "--format",
            "json",
        ]
    )

    assert rc == 0
    assert captured["structure_path"] == structure.resolve()
    assert captured["mutable_positions_path"] == mutable_positions.resolve()
    assert captured["reference_spheres_path"] == spheres.resolve()
    assert captured["protein_chain_id"] == "A"
    assert captured["ligand_chain_id"] == "L"
    assert captured["ligand_resname"] == "LIG"
    assert captured["ligand_resseq"] == 101
    assert captured["contact_distance"] == 4.5
    assert captured["target_aas"] == "ala|thr|ser|gly"
    assert str(captured["output_root"]).endswith("channel_screen_input__channel_screen")
    out = OmegaConf.create(capsys.readouterr().out)
    assert out.best_variant == "pocket_fixed_all_ser"
