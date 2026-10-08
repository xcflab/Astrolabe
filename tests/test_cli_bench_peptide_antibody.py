from __future__ import annotations

import pathlib
import shutil

import pytest
from omegaconf import OmegaConf

import astrolabe.cli as cli
from astrolabe.losses.base import LossEngine


CONFIG_DIR = pathlib.Path(cli.__file__).resolve().parents[1] / "config"


@pytest.fixture(autouse=True)
def _patch_default_config_dir(tmp_path: pathlib.Path, monkeypatch) -> None:
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


def _write_ala_pdb(path: pathlib.Path, chain_lengths: dict[str, int]) -> pathlib.Path:
    lines: list[str] = []
    serial = 1
    x = 0.0
    for chain_id, length in chain_lengths.items():
        for res_i in range(1, int(length) + 1):
            x += 1.0
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA {chain_id}{res_i:4d}    "
                f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
            )
            serial += 1
        lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _chain_a_homology_from_dry_run(out: str) -> dict:
    cfg = OmegaConf.to_container(OmegaConf.create(out), resolve=False)
    homology = cfg["homologies"]["A"]
    assert isinstance(homology, dict)
    return homology


def test_cli_design_binder_dry_run_with_target_structure(tmp_path, capsys):
    target_pdb = _write_ala_pdb(tmp_path / "target.pdb", {"B": 5})
    rc = _run(
        [
            "design",
            "binder",
            "--task-name",
            "t_bind_struct",
            "--out-dir",
            "outputs",
            "--binder-len",
            "80-80",
            "--target-structure",
            str(target_pdb),
            "--target-chain",
            "B",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "B: +AAAAA" in out
    assert "templates:" in out
    assert "mmcif:" in out


def test_cli_refine_binder_dry_run_with_reference_structure_start(
    tmp_path, capsys, monkeypatch
):
    complex_pdb = _write_ala_pdb(tmp_path / "complex.pdb", {"A": 4, "B": 5})
    monkeypatch.setattr(
        cli,
        "_compose_hydra_config",
        lambda **kwargs: OmegaConf.create(
            {
                "task_name": "",
                "out_dir": "",
                "seqs": {"A": "1-1", "B": "+A"},
                "homologies": {"B": None},
                "stages": {"default": {"oligo": ["AB"], "evaluator": "Alphafold3"}},
                "named_files": None,
                "structure_start": None,
            }
        ),
    )
    rc = _run(
        [
            "refine",
            "binder",
            "--task-name",
            "t_refine_reference",
            "--out-dir",
            "outputs",
            "--structure",
            str(complex_pdb),
            "--target-chain",
            "B",
            "--binder-chains",
            "A",
            "--reference",
            str(complex_pdb),
            "--reference-tmscore-threshold",
            "0.9",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "structure_start:" in out
    assert "mode: repredict_gate" in out
    assert "structure_alias: structure_start_ref" in out
    assert "threshold: 0.9" in out
    assert "A: A" in out
    assert "B: B" in out


def test_single_chain_mmcif_string_includes_revision_date(tmp_path):
    target_pdb = _write_ala_pdb(tmp_path / "target_template.pdb", {"B": 5})
    mmcif_str = cli._single_chain_mmcif_string(target_pdb, "B")
    assert "_pdbx_audit_revision_history.revision_date" in mmcif_str


def test_cli_design_peptide_binder_dry_run(capsys):
    rc = _run(
        [
            "design",
            "peptide-binder",
            "--task-name",
            "t_pep",
            "--out-dir",
            "outputs",
            "--peptide-len",
            "8-8",
            "--target-seq",
            "ACDEFGHIK",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    cfg = OmegaConf.create(out)
    assert "binding_site" not in cfg


def test_cli_design_peptide_binder_accepts_initial_binder_seq(capsys):
    rc = _run(
        [
            "design",
            "peptide-binder",
            "--task-name",
            "t_pep_seq",
            "--out-dir",
            "outputs",
            "--initial-binder-seq",
            "acdef",
            "--target-seq",
            "ACDEFGHIK",
            "--dry-run",
        ]
    )
    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.seqs.A == "ACDEF"
    assert cfg.seqs.B == "+ACDEFGHIK"


def test_cli_design_peptide_binder_accepts_initial_binder_expr(capsys):
    rc = _run(
        [
            "design",
            "peptide-binder",
            "--task-name",
            "t_pep_expr",
            "--out-dir",
            "outputs",
            "--initial-binder-expr",
            "+ACD 8-12",
            "--target-seq",
            "ACDEFGHIK",
            "--dry-run",
        ]
    )
    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.seqs.A == "+ACD 8-12"


def test_cli_design_aptamer_dry_run(capsys):
    rc = _run(
        [
            "design",
            "aptamer",
            "--task-name",
            "t_apt",
            "--out-dir",
            "outputs",
            "--aptamer-len",
            "28-36",
            "--target-seq",
            "ACDEFGHIK",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    cfg = OmegaConf.create(out)
    assert "A: 28-36" in out
    assert "types:" in out
    assert "A: r" in out
    assert "B: +ACDEFGHIK" in out
    assert "evaluator: Alphafold3" in out
    assert cfg.stages.default.mutators == ["NAMPNN"]
    assert cfg.mutator.mutator_terms.NAMPNN._type == "NAMPNN"
    assert cfg.mutator.NAMPNN.temperature == 0.3
    assert cfg.mutator.NAMPNN.design_na_only == 1
    assert cfg.loss.loss_terms.aptamer_ptm._type == "binder_ptm"
    assert cfg.loss.loss_terms.aptamer_ptm.binder_chain == "A"
    assert cfg.loss.loss_terms.aptamer_iptm._type == "binder_iptm"
    assert cfg.loss.loss_terms.aptamer_iptm.binder_chain == "A"
    assert cfg.loss.loss_terms.aptamer_plddt._type == "bindcraft_binder_plddt_norm"
    assert cfg.loss.loss_terms.aptamer_plddt.binder_chain == "A"
    assert (
        cfg.stages.default.weighted_score_expression
        == "aptamer_iptm + aptamer_plddt + 0.4 * aptamer_ptm"
    )


@pytest.mark.parametrize(
    "argv_prefix",
    [
        ["design", "aptamer"],
        ["bench-aptamer-v1"],
    ],
)
def test_cli_aptamer_one_shot_accepts_shared_run_options(argv_prefix, capsys):
    rc = _run(
        [
            *argv_prefix,
            "-t",
            "t_apt_shared",
            "-o",
            "outputs",
            "--aptamer-len",
            "31",
            "--target-seq",
            "ACDEFGHIK",
            "--num-runs",
            "2",
            "--seed",
            "123",
            "--steps",
            "4",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.task_name == "t_apt_shared"
    assert cfg.out_dir == "outputs"
    assert cfg.repeats == 2
    assert cfg.seed == 123
    assert cfg.steps == 4


def test_cli_design_aptamer_accepts_initial_binder_seq(capsys):
    rc = _run(
        [
            "design",
            "aptamer",
            "--task-name",
            "t_apt_seq",
            "--out-dir",
            "outputs",
            "--initial-binder-seq",
            "augc",
            "--target-seq",
            "ACDEFGHIK",
            "--dry-run",
        ]
    )
    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.seqs.A == "AUGC"
    assert cfg.types.A == "r"


def test_cli_design_aptamer_rejects_protein_initial_binder_seq():
    with pytest.raises(ValueError, match="AUGC"):
        _run(
            [
                "design",
                "aptamer",
                "--task-name",
                "t_apt_bad",
                "--out-dir",
                "outputs",
                "--initial-binder-seq",
                "ACDEF",
                "--target-seq",
                "ACDEFGHIK",
                "--dry-run",
            ]
        )


def test_cli_design_peptide_binder_dry_run_with_target_structure(tmp_path, capsys):
    target_pdb = _write_ala_pdb(tmp_path / "target_pep.pdb", {"B": 5})
    rc = _run(
        [
            "design",
            "peptide-binder",
            "--task-name",
            "t_pep_struct",
            "--out-dir",
            "outputs",
            "--peptide-len",
            "8-8",
            "--target-structure",
            str(target_pdb),
            "--target-chain",
            "B",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "B: +AAAAA" in out
    assert "templates:" in out
    assert "mmcif:" in out


def test_cli_design_peptide_binder_dry_run_with_binding_site(capsys):
    rc = _run(
        [
            "design",
            "peptide-binder",
            "--task-name",
            "t_pep_bs",
            "--out-dir",
            "outputs",
            "--peptide-len",
            "8-8",
            "--target-seq",
            "ACDEFGHIK",
            "--binding-site-target",
            "B:2-3",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "binding_site:" in out
    assert "binding_site_allowed_min_dist" in out


def test_cli_refine_ligand_pocket_dry_run_with_aligned_reference_pocket(
    tmp_path, capsys
):
    structure = _write_ala_pdb(tmp_path / "complex.pdb", {"A": 8})
    reference = _write_ala_pdb(tmp_path / "reference.pdb", {"A": 8})
    homology_json = tmp_path / "prot_homology.json"
    homology_json.write_text(
        (
            '{"name":"protein_target","sequences":[{"protein":{"id":"A","sequence":"AAAAAAAA"}}],'
            '"modelSeeds":[1],"dialect":"alphafold3","version":1}'
        ),
        encoding="utf-8",
    )

    rc = _run(
        [
            "refine",
            "ligand-pocket",
            "--task-name",
            "t_lig_pocket",
            "--out-dir",
            "outputs",
            "--structure",
            str(structure),
            "--protein-chain",
            "A",
            "--ligand",
            "PLP",
            "--ligand-type",
            "ccd",
            "--protein-homology",
            f"{homology_json}:A",
            "--protein-template-from-structure",
            "--reference-structure",
            str(reference),
            "--reference-chain",
            "A",
            "--reference-pocket",
            "1,3-4",
            "--mutation-num",
            "6",
            "--initial-temperature",
            "0.02",
            "--steps",
            "10",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    predocker = cfg.predocker.predocker_terms.autodock_vina_box
    assert cfg.initial_temperature == 0.02
    assert cfg.mutation_num == 6
    assert cfg.stages.default.mutation_num == 6
    assert cfg.stages.default.weighted_score_expression.endswith(" - autodock_penalty")
    assert cfg.homologies.A.templates
    assert str(cfg.homologies.A.unpairedMsa) == f"{homology_json}:A"
    assert str(cfg.homologies.A.pairedMsa) == f"{homology_json}:A"
    assert cfg.mutator.mutator_terms.LMPNN_allala_vina.predocker == [
        "alanine_relax",
        "autodock_vina_box",
    ]
    assert (
        cfg.mutator.mutator_terms.LMPNN_allala_vina.position_selector
        == "ligand_contact(distance_max=5.0)"
    )
    assert predocker.binding_site_strategy == "aligned_reference_pocket"
    assert predocker.reference_structure_path == str(reference.resolve())
    assert predocker.reference_structure_chain == "A"
    assert predocker.current_structure_chain == "A"
    assert list(predocker.reference_pocket_positions) == [1, 3, 4]


def test_cli_refine_ligand_pocket_requires_complete_reference_triplet(tmp_path):
    structure = _write_ala_pdb(tmp_path / "complex.pdb", {"A": 5})

    with pytest.raises(
        ValueError,
        match="--reference-structure, --reference-chain, and --reference-pocket must be provided together",
    ):
        _run(
            [
                "refine",
                "ligand-pocket",
                "--task-name",
                "t_lig_pocket_bad",
                "--out-dir",
                "outputs",
                "--structure",
                str(structure),
                "--protein-chain",
                "A",
                "--ligand",
                "PLP",
                "--reference-structure",
                str(structure),
                "--dry-run",
            ]
        )


def test_cli_refine_ligand_pocket_retarget_scoring_mode_adds_pocket_score(
    tmp_path,
    capsys,
) -> None:
    structure = _write_ala_pdb(tmp_path / "complex.pdb", {"A": 8})
    reference = _write_ala_pdb(tmp_path / "reference.pdb", {"A": 8})

    rc = _run(
        [
            "refine",
            "ligand-pocket",
            "--task-name",
            "t_lig_pocket_retarget",
            "--out-dir",
            "outputs",
            "--structure",
            str(structure),
            "--protein-chain",
            "A",
            "--ligand",
            "PLP",
            "--ligand-type",
            "ccd",
            "--scoring-mode",
            "retarget",
            "--reference-structure",
            str(reference),
            "--reference-chain",
            "A",
            "--reference-pocket",
            "1,3-4",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.ligand_retarget_scoring.mode == "retarget"
    assert cfg.ligand_retarget_scoring.reference_structure_path == str(reference.resolve())
    assert cfg.ligand_retarget_scoring.reference_structure_chain == "A"
    assert cfg.ligand_retarget_scoring.current_structure_chain == "A"
    assert list(cfg.ligand_retarget_scoring.reference_pocket_positions) == [1, 3, 4]
    assert cfg.ligand_retarget_scoring.ligand_targets == "B"
    assert (
        cfg.loss.loss_terms.ligand_pocket_occupancy._type
        == "ligand_aligned_reference_pocket_box_occupancy"
    )
    assert cfg.loss.loss_terms.ligand_pocket_occupancy.reference_structure_path == str(
        reference.resolve()
    )
    assert cfg.loss.loss_terms.ligand_pocket_occupancy.reference_structure_chain == "A"
    assert cfg.loss.loss_terms.ligand_pocket_occupancy.current_structure_chain == "A"
    assert list(cfg.loss.loss_terms.ligand_pocket_occupancy.reference_pocket_positions) == [
        1,
        3,
        4,
    ]
    assert cfg.loss.loss_terms.ligand_pocket_occupancy.ligand_targets == "B"
    assert cfg.stages.default.weighted_score_expression.endswith(
        " - autodock_penalty + ligand_pocket_occupancy"
    )
    loss_engine = LossEngine(OmegaConf.to_container(cfg.loss, resolve=True))
    loss_engine.setup(
        cfg.stages.default.weighted_score_expression,
        cfg.stages.default.get("early_stopping"),
        cfg.stages.default.get("acceptance_threshold"),
    )
    assert "ligand_pocket_occupancy" in loss_engine.loss_specs


def test_cli_refine_ligand_pocket_retarget_scoring_requires_reference_pocket(tmp_path):
    structure = _write_ala_pdb(tmp_path / "complex.pdb", {"A": 5})

    with pytest.raises(ValueError, match="--scoring-mode retarget requires"):
        _run(
            [
                "refine",
                "ligand-pocket",
                "--task-name",
                "t_lig_pocket_retarget_bad",
                "--out-dir",
                "outputs",
                "--structure",
                str(structure),
                "--protein-chain",
                "A",
                "--ligand",
                "PLP",
                "--scoring-mode",
                "retarget",
                "--dry-run",
            ]
        )


def test_cli_refine_ligand_pocket_dry_run_without_reference_keeps_default_box_and_smiles(
    tmp_path, capsys
):
    structure = _write_ala_pdb(tmp_path / "complex.pdb", {"A": 6})

    rc = _run(
        [
            "refine",
            "ligand-pocket",
            "--task-name",
            "t_lig_pocket_smiles",
            "--out-dir",
            "outputs",
            "--structure",
            str(structure),
            "--protein-chain",
            "A",
            "--ligand",
            "NCCc1ccccc1",
            "--ligand-type",
            "smiles",
            "--contact-distance",
            "6.0",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    predocker = cfg.predocker.predocker_terms.autodock_vina_box
    assert cfg.types.B == "s"
    assert not str(cfg.seqs.A).startswith("+")
    assert cfg.homologies.A is None
    assert predocker.binding_site_strategy == "residue"
    assert "reference_structure_path" not in predocker
    assert (
        cfg.mutator.mutator_terms.LMPNN_allala_vina.position_selector
        == "ligand_contact(distance_max=6)"
    )
    assert cfg.stages.default.weighted_score_expression.endswith(" - autodock_penalty")


def test_cli_refine_ligand_pocket_uses_input_chain_for_current_structure_chain(
    tmp_path, capsys
):
    structure = _write_ala_pdb(tmp_path / "complex_r.pdb", {"R": 6})
    reference = _write_ala_pdb(tmp_path / "reference_x.pdb", {"X": 6})

    rc = _run(
        [
            "refine",
            "ligand-pocket",
            "--task-name",
            "t_lig_pocket_chain_r",
            "--out-dir",
            "outputs",
            "--structure",
            str(structure),
            "--protein-chain",
            "R",
            "--ligand",
            "PLP",
            "--reference-structure",
            str(reference),
            "--reference-chain",
            "X",
            "--reference-pocket",
            "2-3",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    predocker = cfg.predocker.predocker_terms.autodock_vina_box
    assert predocker.reference_structure_chain == "X"
    assert predocker.current_structure_chain == "R"


def test_cli_design_peptide_binder_dry_run_with_cyclize_head_to_tail(capsys):
    rc = _run(
        [
            "design",
            "peptide-binder",
            "--task-name",
            "t_pep_cyc",
            "--out-dir",
            "outputs",
            "--peptide-len",
            "8-8",
            "--target-seq",
            "ACDEFGHIK",
            "--cyclize-head-to-tail",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "cyclization:" in out
    assert "head_to_tail: true" in out


def test_cli_design_peptide_binder_dry_run_with_cyclize_disulfide(capsys):
    rc = _run(
        [
            "design",
            "peptide-binder",
            "--task-name",
            "t_pep_ss",
            "--out-dir",
            "outputs",
            "--peptide-len",
            "8-8",
            "--target-seq",
            "ACDEFGHIK",
            "--cyclize-disulfide",
            "2-7",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "cyclization:" in out
    assert "atom: SG" in out
    assert "aa: C" in out


def test_cli_bench_peptide_binder_v1_dry_run_with_cyclize_bond_typed(capsys):
    rc = _run(
        [
            "bench-peptide-binder-v1",
            "--task-name",
            "t_pep_bond",
            "--out-dir",
            "outputs",
            "--peptide-len",
            "8-8",
            "--target-seq",
            "ACDEFGHIK",
            "--cyclize-bond-typed",
            "A:1@N,A:-1@C,single",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "cyclization:" in out
    assert "bonds:" in out
    assert "order: single" in out
    assert "cyclic_closure_norm:" in out
    assert "aggregate: min" in out
    assert "sigma: 0.2" in out
    assert "cyclic_closure_norm >= 0.9" in out
    assert "0.3 * cyclic_closure_norm" in " ".join(out.split())


def test_cli_bench_peptide_binder_v1_accepts_legacy_cyclize_bond(capsys):
    rc = _run(
        [
            "bench-peptide-binder-v1",
            "--task-name",
            "t_pep_bond_legacy",
            "--out-dir",
            "outputs",
            "--peptide-len",
            "8-8",
            "--target-seq",
            "ACDEFGHIK",
            "--cyclize-bond",
            "A:1@N",
            "A:-1@C",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "cyclization:" in out
    assert "bonds:" in out
    assert "atom: 'N'" in out
    assert "atom: C" in out


def test_cli_bench_peptide_binder_v1_dry_run_with_cyclic_encoding_all_bonds(capsys):
    rc = _run(
        [
            "bench-peptide-binder-v1",
            "--task-name",
            "t_pep_encoding",
            "--out-dir",
            "outputs",
            "--peptide-len",
            "8-8",
            "--target-seq",
            "ACDEFGHIK",
            "--cyclize-head-to-tail",
            "--cyclic-encoding",
            "all-bonds",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "encoding:" in out
    assert "scope: all_bonds" in out


def test_cli_bench_peptide_binder_v1_dry_run_with_cyclic_encoding_all_alias(capsys):
    rc = _run(
        [
            "bench-peptide-binder-v1",
            "--task-name",
            "t_pep_encoding_alias",
            "--out-dir",
            "outputs",
            "--peptide-len",
            "8-8",
            "--target-seq",
            "ACDEFGHIK",
            "--cyclize-head-to-tail",
            "--cyclic-encoding",
            "all",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "encoding:" in out
    assert "scope: all_bonds" in out


def test_cli_bench_peptide_binder_v1_dry_run_with_cyclic_encoding_highfold_cycpoem(capsys):
    rc = _run(
        [
            "bench-peptide-binder-v1",
            "--task-name",
            "t_pep_highfold_cycpoem",
            "--out-dir",
            "outputs",
            "--peptide-len",
            "8-8",
            "--target-seq",
            "ACDEFGHIK",
            "--cyclize-head-to-tail",
            "--cyclic-encoding",
            "highfold-cycpoem",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "encoding:" in out
    assert "mode: highfold_cycpoem" in out
    assert "scope: all_bonds" in out


def test_cli_bench_peptide_binder_v1_dry_run_with_cyclic_encoding_rfpeptide_like(capsys):
    rc = _run(
        [
            "bench-peptide-binder-v1",
            "--task-name",
            "t_pep_rfpeptide_like",
            "--out-dir",
            "outputs",
            "--peptide-len",
            "8-8",
            "--target-seq",
            "ACDEFGHIK",
            "--cyclize-disulfide",
            "2-7",
            "--cyclic-encoding",
            "rfpeptide-like",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "encoding:" in out
    assert "mode: global_equivalent" in out
    assert "scope: all_bonds" in out


def test_cli_bench_peptide_binder_v1_dry_run_with_nonstd_map(tmp_path, capsys):
    nonstd_map = tmp_path / "nonstd.yaml"
    nonstd_map.write_text(
        "\n".join(
            [
                "aliases:",
                "  S5:",
                "    ccd_id: S5A",
                "    parent_aa: S",
                "    source:",
                "      format: smiles",
                "      value: C=CC",
                "    atom_names: [C1, C2, C3]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    rc = _run(
        [
            "bench-peptide-binder-v1",
            "--task-name",
            "t_pep_nonstd",
            "--out-dir",
            "outputs",
            "--peptide-len",
            "8-8",
            "--target-seq",
            "ACDEFGHIK",
            "--nonstd-map",
            str(nonstd_map),
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "nonstandard_residues:" in out
    assert "S5:" in out


def test_cli_design_antibody_dry_run(capsys):
    rc = _run(
        [
            "design",
            "antibody",
            "--task-name",
            "t_ab",
            "--out-dir",
            "outputs",
            "--binder-seq",
            "+AAAA 3-3 +BBBB",
            "--target-seq",
            "ACDEFGHIK",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    cfg = OmegaConf.create(out)
    assert "binding_site" not in cfg


def test_cli_bench_antibody_v1_dry_run_uses_single_unselected_ligandmpnn_and_ipsae_score(capsys):
    rc = _run(
        [
            "bench-antibody-v1",
            "--task-name",
            "t_ab_bench",
            "--out-dir",
            "outputs",
            "--binder-seq",
            "+AAAA 3-3 +BBBB",
            "--target-seq",
            "ACDEFGHIK",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("_type: LigandMPNN") == 1
    assert "temperature: 0.3" in out
    assert "model_type: soluble_mpnn" not in out
    assert "position_selector:" not in out
    assert "LigandMPNN_itf" not in out
    assert "LigandMPNN_noitf" not in out
    assert "weighted_score_expression: iptm + interface_sc + ipsae" in out
    assert "ipsae:" in out
    assert "_type: ipsae" in out
    assert "chain_a: A" in out
    assert "chain_b: B" in out


def test_cli_design_antibody_dry_run_with_target_structure(tmp_path, capsys):
    target_pdb = _write_ala_pdb(tmp_path / "target_ab.pdb", {"B": 5})
    rc = _run(
        [
            "design",
            "antibody",
            "--task-name",
            "t_ab_struct",
            "--out-dir",
            "outputs",
            "--binder-seq",
            "+AAAA 3-3 +BBBB",
            "--target-structure",
            str(target_pdb),
            "--target-chain",
            "B",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "B: +AAAAA" in out
    assert "templates:" in out
    assert "mmcif:" in out


def test_cli_design_antibody_dry_run_with_binding_site(capsys):
    rc = _run(
        [
            "design",
            "antibody",
            "--task-name",
            "t_ab_bs",
            "--out-dir",
            "outputs",
            "--binder-seq",
            "+AAAA 3-3 +BBBB",
            "--target-seq",
            "ACDEFGHIK",
            "--binding-site-target",
            "B:2-3",
            "--binding-site-binder",
            "A:2-3",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "binding_site:" in out
    assert "binding_site_allowed_min_dist" in out
    assert "binding_site_framework_min_dist" in out


def test_cli_design_antibody_dry_run_with_epitope_only_binding_site(capsys):
    rc = _run(
        [
            "design",
            "antibody",
            "--task-name",
            "t_ab_bs_epitope_only",
            "--out-dir",
            "outputs",
            "--binder-seq",
            "+AAAA 3-3 +BBBB",
            "--target-seq",
            "ACDEFGHIK",
            "--binding-site-target",
            "B:2-3",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "binding_site:" in out
    assert "binder: A" in out
    assert "binding_site_allowed_min_dist" in out
    assert "binding_site_framework_min_dist" not in out


def test_cli_design_antibody_list_frameworks(capsys):
    rc = _run(["design", "antibody", "--list-frameworks"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "scfv_vh_linker_vl__rfantibody_4d5" in out
    assert "vhh__boltzgen_7eow" in out
    assert "scfv4d5_vhvl" in out


def test_cli_design_antibody_framework_dry_run(capsys):
    rc = _run(
        [
            "design",
            "antibody",
            "--task-name",
            "t_ab_fw",
            "--out-dir",
            "outputs",
            "--framework",
            "vhh__boltzgen_7eow",
            "--target-seq",
            "ACDEFGHIK",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "EVQLVESGGGLVQPGGSLRLSCAAS" in out
    assert "_type: antibody_framework_homology" in out
    assert "framework_name: vhh__boltzgen_7eow" in out


def test_cli_design_antibody_framework_alias_and_cdr_lengths(capsys):
    rc = _run(
        [
            "design",
            "antibody",
            "--task-name",
            "t_ab_fw_alias",
            "--out-dir",
            "outputs",
            "--framework",
            "vhh_7eow",
            "--cdr-lengths",
            "H1:31-31",
            "--target-seq",
            "ACDEFGHIK",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "31-31" in out
    assert "homologies:" in out
    assert "A:" in out


def test_cli_design_antibody_framework_pdb_uses_standard_templates_when_homology_omitted(
    tmp_path, capsys
):
    framework_pdb = _write_ala_pdb(tmp_path / "framework_only_h.pdb", {"H": 110})
    rc = _run(
        [
            "design",
            "antibody",
            "--task-name",
            "t_ab_fw_pdb_structure_only",
            "--out-dir",
            "outputs",
            "--framework-pdb",
            str(framework_pdb),
            "--framework-kind",
            "vhh",
            "--framework-heavy-chain",
            "H",
            "--target-seq",
            "ACDEFGHIK",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    homology_a = _chain_a_homology_from_dry_run(out)
    assert "templates" in homology_a
    assert "source" not in homology_a
    assert "_type" not in homology_a
    assert any("mmcif" in template for template in homology_a["templates"])


def test_cli_design_antibody_framework_pdb_with_framework_homology_uses_standard_unpaired_msa(
    tmp_path, capsys
):
    framework_pdb = _write_ala_pdb(tmp_path / "framework_only_h.pdb", {"H": 110})
    from astrolabe.antibody_framework_presets import build_framework_layout_from_structure

    _, framework_layout = build_framework_layout_from_structure(
        structure=str(framework_pdb),
        kind="vhh",
        heavy_chain="H",
    )
    source_query_len = sum(len(seg) for seg in framework_layout["fixed_segments"]) + sum(
        int(v) for v in framework_layout["source_loop_lengths"].values()
    )
    homology_a3m = tmp_path / "framework_only_h.a3m"
    homology_a3m.write_text(
        ">query\n" + ("A" * source_query_len) + "\n>hit1\n" + ("A" * source_query_len) + "\n",
        encoding="utf-8",
    )
    rc = _run(
        [
            "design",
            "antibody",
            "--task-name",
            "t_ab_fw_pdb_with_homology",
            "--out-dir",
            "outputs",
            "--framework-pdb",
            str(framework_pdb),
            "--framework-kind",
            "vhh",
            "--framework-heavy-chain",
            "H",
            "--framework-homology",
            str(homology_a3m),
            "--target-seq",
            "ACDEFGHIK",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    homology_a = _chain_a_homology_from_dry_run(out)
    assert "unpairedMsa" in homology_a
    assert ">query" in homology_a["unpairedMsa"]
    assert "mmcif" not in homology_a
    assert "source" not in homology_a
    assert "_type" not in homology_a


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--framework", "vhh__boltzgen_7eow"],
        ["--framework-pdb", "dummy.pdb"],
    ],
)
def test_cli_design_antibody_framework_mutual_exclusive(extra_args):
    with pytest.raises(
        ValueError, match="Use only one of: --binder-seq, --framework, --framework-pdb"
    ):
        _run(
            [
                "design",
                "antibody",
                "--task-name",
                "t_ab_fw_bad",
                "--out-dir",
                "outputs",
                "--binder-seq",
                "+AAAA 3-3 +BBBB",
                *extra_args,
                "--target-seq",
                "ACDEFGHIK",
                "--dry-run",
            ]
        )


def test_cli_design_antibody_cdr_lengths_requires_framework():
    with pytest.raises(ValueError, match="--cdr-lengths requires --framework or --framework-pdb"):
        _run(
            [
                "design",
                "antibody",
                "--task-name",
                "t_ab_cdr_bad",
                "--out-dir",
                "outputs",
                "--cdr-lengths",
                "H1:7-7",
                "--target-seq",
                "ACDEFGHIK",
                "--dry-run",
            ]
        )


def test_cli_design_antibody_cdr_lengths_not_allowed_with_binder_seq():
    with pytest.raises(ValueError, match="--cdr-lengths cannot be used with --binder-seq"):
        _run(
            [
                "design",
                "antibody",
                "--task-name",
                "t_ab_cdr_with_binder_seq",
                "--out-dir",
                "outputs",
                "--binder-seq",
                "+AAAA 3-3 +BBBB",
                "--cdr-lengths",
                "H1:7-7",
                "--target-seq",
                "ACDEFGHIK",
                "--dry-run",
            ]
        )


def test_cli_design_antibody_framework_pdb_accepts_ranged_cdr_lengths(tmp_path, capsys):
    framework_pdb = _write_ala_pdb(tmp_path / "framework_only_h.pdb", {"H": 110})
    rc = _run(
        [
            "design",
            "antibody",
            "--task-name",
            "t_ab_fw_pdb_range_lengths",
            "--out-dir",
            "outputs",
            "--framework-pdb",
            str(framework_pdb),
            "--framework-kind",
            "vhh",
            "--framework-heavy-chain",
            "H",
            "--cdr-lengths",
            "H1:7-11,H3:9-14",
            "--target-seq",
            "ACDEFGHIK",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "7-11" in out
    assert "9-14" in out
