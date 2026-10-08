from __future__ import annotations

import shutil
from pathlib import Path

import astrolabe.cli as cli
import numpy as np
import pytest
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.structure_processors.symmetry import SymmetryRescueProcessor


CONFIG_DIR = Path(cli.__file__).resolve().parents[1] / "config"


@pytest.fixture(autouse=True)
def _patch_default_config_dir(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "default_config"
    shutil.copytree(CONFIG_DIR, config_dir, ignore=shutil.ignore_patterns("__pycache__"))
    (config_dir / "env").mkdir(exist_ok=True)
    (config_dir / "env" / "env_local.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_default_config_dir", lambda: config_dir)


def _minimal_predocker_cfg():
    return OmegaConf.create(
        {
            "predocker": {"predocker_terms": {"symmetry": None}},
            "mutator": {
                "mutator_terms": {
                    "ligand_mpnn_refine": {"_type": "LigandMPNN"},
                }
            },
        }
    )


def _run(argv: list[str]) -> int:
    parser = cli._build_parser()
    ns = parser.parse_args(argv)
    handler = getattr(ns, "_handler", None)
    assert handler is not None
    return int(handler(ns))


def _config_dir_with_env(tmp_path: Path, *config_names: str) -> Path:
    dst_root = tmp_path / "config"
    (dst_root / "env").mkdir(parents=True)
    (dst_root / "env" / "env_local.yaml").write_text("{}\n", encoding="utf-8")
    shutil.copy2(CONFIG_DIR / "base.yaml", dst_root / "base.yaml")
    shutil.copy2(CONFIG_DIR / "env.yaml", dst_root / "env.yaml")
    for name in config_names:
        shutil.copy2(CONFIG_DIR / f"{name}.yaml", dst_root / f"{name}.yaml")
    return dst_root


def _write_ca_pdb(path: Path, chain_coords: dict[str, list[tuple[float, float, float]]]) -> Path:
    lines: list[str] = []
    serial = 1
    for chain_id, coords in chain_coords.items():
        for res_i, (x, y, z) in enumerate(coords, start=1):
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA {chain_id}{res_i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
            )
            serial += 1
        lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_cli_bench_pseudocycle_v1_dry_run(tmp_path: Path, capsys) -> None:
    config_dir = _config_dir_with_env(tmp_path, "benchmark_pseudocycle_v1")
    rc = _run(
        [
            "--config-dir",
            str(config_dir),
            "bench-pseudocycle-v1",
            "--task-name",
            "pseudo",
            "--out-dir",
            "outputs",
            "--unit-len",
            "12-12",
            "--copies",
            "4",
            "--symmetry-predocker",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "repeat_unit_rmsd_norm" in out
    assert "pseudocycle_closure_norm" in out
    assert "symmetry_canonicalizer" in out
    assert "mode: pseudocycle" in out
    assert "Alphafold2:" in out
    assert "Alphafold3:" in out


def test_cli_bench_pseudocycle_v1_dry_run_accepts_backend_override(
    tmp_path: Path,
    capsys,
) -> None:
    config_dir = _config_dir_with_env(tmp_path, "benchmark_pseudocycle_v1")
    rc = _run(
        [
            "--config-dir",
            str(config_dir),
            "bench-pseudocycle-v1",
            "--task-name",
            "pseudo",
            "--out-dir",
            "outputs",
            "--unit-len",
            "12-12",
            "--copies",
            "4",
            "--symmetry-predocker",
            "--symmetry-predocker-backend",
            "rfdiffusion",
            "--symmetry-order",
            "4",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "backend: rfdiffusion" in out
    assert "order: 4" in out


def test_enable_symmetry_predocker_sets_auto_interface_selection() -> None:
    cfg = _minimal_predocker_cfg()

    cli._enable_symmetry_predocker(
        cfg,
        mode="pseudocycle",
        oligo="A",
        backend="rpxdock",
        order=4,
        interface_selection="auto",
    )

    assert cfg.predocker.predocker_terms.symmetry.interface_selection == "auto"


def test_symmetry_predocker_runtime_consumes_interface_selection(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("is_debug", False)
    global_config.init_const("out_dir", tmp_path)
    input_file = tmp_path / "input.pdb"
    input_file.write_text("END\n", encoding="utf-8")
    predocker = SymmetryRescueProcessor(
        mode="pseudocycle",
        oligo="A",
        backend="rpxdock",
        order=4,
        interface_selection="manual",
        interface="1-2,5",
    )

    manual_key = predocker._generate_cache_key(input_file)
    auto_key = SymmetryRescueProcessor(
        mode="pseudocycle",
        oligo="A",
        backend="rpxdock",
        order=4,
        interface_selection="auto",
    )._generate_cache_key(input_file)

    assert predocker.interface_selection == "manual"
    assert predocker.interface == "1-2,5"
    assert manual_key != auto_key


def test_symmetry_predocker_manual_interface_filters_repeat_fit_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global_config.reset()
    global_config.init_const("is_debug", False)
    global_config.init_const("out_dir", tmp_path)
    monkeypatch.setattr(
        "astrolabe.structure_processors.symmetry.build_internal_repeat_residue_groups",
        lambda _proto: [
            np.asarray([0, 1, 2], dtype=int),
            np.asarray([3, 4, 5], dtype=int),
        ],
    )
    predocker = SymmetryRescueProcessor(
        mode="pseudocycle",
        oligo="A",
        backend="rpxdock",
        order=2,
        interface_selection="manual",
        interface="2,5",
    )

    groups = predocker._internal_repeat_fit_groups("A", residue_count=6)

    assert [group.tolist() for group in groups] == [[1], [4]]


def test_symmetry_predocker_requires_manual_interface_at_runtime(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("is_debug", False)
    global_config.init_const("out_dir", tmp_path)
    with pytest.raises(ValueError, match="requires interface"):
        SymmetryRescueProcessor(
            mode="pseudocycle",
            oligo="A",
            backend="rpxdock",
            order=4,
            interface_selection="manual",
        )


def test_enable_symmetry_predocker_requires_interface_for_manual_selection() -> None:
    cfg = _minimal_predocker_cfg()

    with pytest.raises(ValueError, match="manual requires --symmetry-interface"):
        cli._enable_symmetry_predocker(
            cfg,
            mode="pseudocycle",
            oligo="A",
            backend="rpxdock",
            order=4,
            interface_selection="manual",
        )


def test_cli_bench_tandem_repeat_v1_dry_run_translation_mode(capsys) -> None:
    rc = _run(
        [
            "bench-tandem-repeat-v1",
            "--task-name",
            "tandem",
            "--out-dir",
            "outputs",
            "--unit-len",
            "10-10",
            "--copies",
            "3",
            "--symmetry-predocker",
            "--repeat-canonicalizer-mode",
            "translation",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "repeat_unit_rmsd_norm" in out
    assert "repeat_canonicalizer_mode: translation" in out
    assert "mode: tandem_repeat" in out


def test_cli_refine_heterooligomer_v1_dry_run(capsys) -> None:
    rc = _run(
        [
            "refine-heterooligomer-v1",
            "--task-name",
            "hetero",
            "--out-dir",
            "outputs",
            "--seq",
            "ACDEFGHIK",
            "--copies",
            "3",
            "--interface",
            "3-5",
            "--symmetry-predocker",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "protomer_rmsd_norm" in out
    assert "mode: heterooligomer_shared_core" in out
    assert "shared_1" in out
    assert "itf_B_1" in out
    assert "Oligomer: ABC" in out


def test_cli_refine_oligomer_v3_can_attach_cyclic_symmetry_predocker(capsys) -> None:
    rc = _run(
        [
            "refine-oligomer-v3",
            "--task-name",
            "oligo",
            "--out-dir",
            "outputs",
            "--seq",
            "ACDEFGHIK",
            "--copies",
            "4",
            "--symmetry-predocker",
            "--symmetry-predocker-backend",
            "rfdiffusion",
            "--symmetry-order",
            "4",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "mode: cyclic_oligomer" in out
    assert "oligo: AAAA" in out
    assert "backend: rfdiffusion" in out


def test_cli_refine_oligomer_v3_defaults_cyclic_symmetry_predocker_to_supported_backend(
    capsys,
) -> None:
    rc = _run(
        [
            "refine-oligomer-v3",
            "--task-name",
            "oligo",
            "--out-dir",
            "outputs",
            "--seq",
            "ACDEFGHIK",
            "--copies",
            "4",
            "--symmetry-predocker",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "mode: cyclic_oligomer" in out
    assert "backend: rfdiffusion" in out


def test_cli_refine_oligomer_v3_rejects_legacy_cyclic_symmetry_backend() -> None:
    with pytest.raises(ValueError, match="backend='legacy' is not supported for mode='cyclic_oligomer'"):
        _run(
            [
                "refine-oligomer-v3",
                "--task-name",
                "oligo",
                "--out-dir",
                "outputs",
                "--seq",
                "ACDEFGHIK",
                "--copies",
                "4",
                "--symmetry-predocker",
                "--symmetry-predocker-backend",
                "legacy",
                "--dry-run",
            ]
        )


def test_enable_symmetry_predocker_accepts_dihedral_rpxdock_backend() -> None:
    cfg = _minimal_predocker_cfg()

    cli._enable_symmetry_predocker(
        cfg,
        mode="dihedral_oligomer",
        oligo="AAAAAA",
        backend="rpxdock",
        order=3,
    )

    symmetry_term = cfg.predocker.predocker_terms.symmetry
    assert symmetry_term.mode == "dihedral_oligomer"
    assert symmetry_term.backend == "rpxdock"
    assert symmetry_term.oligo == "AAAAAA"
    assert symmetry_term.order == 3


def test_enable_symmetry_predocker_rejects_dihedral_legacy_backend() -> None:
    cfg = _minimal_predocker_cfg()

    with pytest.raises(ValueError, match="backend='legacy' is not supported for mode='dihedral_oligomer'"):
        cli._enable_symmetry_predocker(
            cfg,
            mode="dihedral_oligomer",
            oligo="AAAAAA",
            backend="legacy",
            order=3,
        )


def test_cli_design_oligomer_c3_dry_run_uses_cyclic_rpxdock(tmp_path: Path, capsys) -> None:
    config_dir = _config_dir_with_env(tmp_path, "benchmark_oligomer_morphology_cyclic_v1")
    rc = _run(
        [
            "--config-dir",
            str(config_dir),
            "design",
            "oligomer",
            "--task-name",
            "c3",
            "--out-dir",
            "outputs",
            "--shape",
            "C3",
            "--protein-len",
            "72-72",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Shape: C3" in out
    assert "Template: benchmark_oligomer_morphology_cyclic_v1" in out
    assert "mode: cyclic_oligomer" in out
    assert "backend: rpxdock" in out
    assert "order: 3" in out
    assert "oligo: AAA" in out
    assert "A_longest_helix_run_frac" in out


def test_cli_design_oligomer_c8_dry_run_is_allowed(tmp_path: Path, capsys) -> None:
    config_dir = _config_dir_with_env(tmp_path, "benchmark_oligomer_morphology_cyclic_v1")
    rc = _run(
        [
            "--config-dir",
            str(config_dir),
            "design",
            "oligomer",
            "--task-name",
            "c8",
            "--out-dir",
            "outputs",
            "--shape",
            "C8",
            "--protein-len",
            "72-72",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Shape: C8" in out
    assert "order: 8" in out
    assert "oligo: AAAAAAAA" in out


def test_cli_design_oligomer_d4_dry_run_uses_dihedral_rpxdock_and_ananas(tmp_path: Path, capsys) -> None:
    config_dir = _config_dir_with_env(tmp_path, "benchmark_dihedral_d4_dplm_protenix_af2_rpxdock_v1")
    rc = _run(
        [
            "--config-dir",
            str(config_dir),
            "design",
            "oligomer",
            "--task-name",
            "d4",
            "--out-dir",
            "outputs",
            "--shape",
            "D4",
            "--protein-len",
            "72-72",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Shape: D4" in out
    assert "Template: benchmark_dihedral_d4_dplm_protenix_af2_rpxdock_v1" in out
    assert "mode: dihedral_oligomer" in out
    assert "backend: rpxdock" in out
    assert "order: 4" in out
    assert "oligo: AAAAAAAA" in out
    assert "d4_ananas" in out
    assert "dihedral_pair_contact_count_norm" in out


@pytest.mark.parametrize("shape", ["C0", "C1", "D5", "Q3"])
def test_cli_design_oligomer_rejects_invalid_shapes(shape: str) -> None:
    with pytest.raises(ValueError, match="Unsupported oligomer shape"):
        _run(
            [
                "design",
                "oligomer",
                "--task-name",
                "bad",
                "--out-dir",
                "outputs",
                "--shape",
                shape,
                "--protein-len",
                "72-72",
                "--dry-run",
            ]
        )


def test_cli_bench_tandem_repeat_v1_rejects_exact_backend_override() -> None:
    with pytest.raises(ValueError, match="backend='rfdiffusion' is not supported for mode='tandem_repeat'"):
        _run(
            [
                "bench-tandem-repeat-v1",
                "--task-name",
                "tandem",
                "--out-dir",
                "outputs",
                "--unit-len",
                "10-10",
                "--copies",
                "3",
                "--symmetry-predocker",
                "--symmetry-predocker-backend",
                "rfdiffusion",
                "--dry-run",
            ]
        )


def test_cli_bench_pseudocycle_v1_rejects_order_with_legacy_backend() -> None:
    cfg = _minimal_predocker_cfg()

    with pytest.raises(ValueError, match="Symmetry order is not supported for backend='legacy'"):
        cli._enable_symmetry_predocker(
            cfg,
            mode="pseudocycle",
            oligo="A",
            backend="legacy",
            order=5,
        )


def test_single_chain_mmcif_string_includes_release_date(tmp_path: Path) -> None:
    pdb_path = _write_ca_pdb(tmp_path / "template.pdb", {"A": [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0)]})

    mmcif_str = cli._single_chain_mmcif_string(pdb_path, "A")

    assert "_pdbx_audit_revision_history.revision_date" in mmcif_str


def test_rpxdock_symmetry_predocker_inherits_split_env_python_from_config(
    tmp_path: Path,
) -> None:
    config_dir = _config_dir_with_env(tmp_path, "benchmark_pseudocycle_v1")
    cfg = cli._compose_hydra_config(
        config_dir=config_dir,
        config_name="benchmark_pseudocycle_v1",
        overrides=[
            "subprocess_envs.rpxdock.python_bin=/envs/rpxdock/bin/python",
            "subprocess_envs.rpxdock.repo_path=/vendor/rpxdock",
        ],
    )
    cli._enable_symmetry_predocker(
        cfg,
        mode="pseudocycle",
        oligo="A",
        included_protomer="A",
        backend="rpxdock",
    )

    symmetry_term = OmegaConf.to_container(
        cfg.predocker.predocker_terms.symmetry,
        resolve=True,
    )
    symmetry_defaults = OmegaConf.to_container(
        cfg.predocker.symmetry_canonicalizer,
        resolve=True,
    )

    assert symmetry_defaults["rpxdock_python_bin"] == "/envs/rpxdock/bin/python"
    merged = {**symmetry_defaults, **symmetry_term}
    assert merged["rpxdock_python_bin"] == "/envs/rpxdock/bin/python"


def test_task_requirements_include_ananas_path_for_ananas_loss() -> None:
    req = cli._task_requirements(
        {
            "loss": {
                "loss_terms": {
                    "sym": {
                        "_type": "ananas_rmsd_norm",
                        "point_group": "D3",
                    }
                }
            }
        }
    )

    assert "ananas_path" in req["required_env_keys"]
    assert req["env_key_reasons"]["ananas_path"].startswith("required by AnAnaS")
