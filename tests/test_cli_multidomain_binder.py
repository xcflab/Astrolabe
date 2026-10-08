from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import astrolabe.cli as cli


CONFIG_DIR = Path(cli.__file__).resolve().parents[1] / "config"


@pytest.fixture(autouse=True)
def _patch_default_config_dir(tmp_path: Path, monkeypatch) -> None:
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


def _write_af3_homology_json(
    path: Path,
    *,
    sequence: str,
    chain_id: str = "A",
    unpaired_msa: str | None = None,
    paired_msa: str | None = None,
    templates: list[dict] | None = None,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "name": path.stem,
                "dialect": "alphafold3",
                "version": 1,
                "modelSeeds": [1],
                "sequences": [
                    {
                        "protein": {
                            "id": chain_id,
                            "sequence": sequence,
                            "unpairedMsa": unpaired_msa or "",
                            "pairedMsa": paired_msa or "",
                            "templates": templates or [],
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_design_multidomain_binder_af3_dry_run(tmp_path: Path, capsys) -> None:
    target_seq = "ACDEFGHIKL"
    full_homology = _write_af3_homology_json(
        tmp_path / "full_homology.json",
        sequence=target_seq,
        unpaired_msa=">query\nACDEFGHIKL\n>hit1\nACDEYGHIKL\n",
        paired_msa=">query\nACDEFGHIKL\n>pair1\nACDEFGHIKL\n",
        templates=[
            {
                "mmcif": "data_domain1",
                "queryIndices": [1, 2],
                "templateIndices": [101, 102],
            },
            {
                "mmcif": "data_domain2",
                "queryIndices": [6, 7],
                "templateIndices": [206, 207],
            },
        ],
    )

    rc = _run(
        [
            "design",
            "multidomain-binder",
            "--task-name",
            "t_multidomain_af3",
            "--out-dir",
            "outputs",
            "--variant",
            "af3",
            "--target-name",
            "cre_like",
            "--binder-len",
            "80-120",
            "--target-homology",
            f"{full_homology}:A",
            "--domain-indices",
            "2,3",
            "--domain-indices",
            "7-8",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.task_name == "t_multidomain_af3"
    assert cfg.multidomain_target.target_name == "cre_like"
    assert cfg.multidomain_target.domains[0].indices == [2, 3]
    assert cfg.multidomain_target.domains[1].indices == [7, 8]
    assert cfg.multidomain_target.domain_split_mode == "full"
    assert cfg.multidomain_target.domains[0].hotspots == [2, 3]
    assert cfg.multidomain_target.domains[1].hotspots == [7, 8]
    assert cfg.seqs.A == "80-120"
    assert cfg.seqs.B == f"+{target_seq}"
    assert cfg.binding_site.target == "B:2,3,7,8"
    assert cfg.loss.loss_terms.hotspot_target_global.select_expression == "B_1:2,3,7,8"
    assert cfg.homologies.B.unpairedMsa.startswith(">query")
    assert cfg.homologies.B.pairedMsa.startswith(">query")
    assert len(cfg.homologies.B.templates) == 2
    assert cfg.homologies.B.templates[0].mmcif == "data_domain1"
    assert cfg.homologies.B.templates[1].mmcif == "data_domain2"
    assert cfg.stages.target_complex.evaluator == "Alphafold3"


def test_cli_design_multidomain_binder_template_split_mode_keeps_full_homology(
    tmp_path: Path,
    capsys,
) -> None:
    target_seq = "ACDEFGHIKL"
    full_homology = _write_af3_homology_json(
        tmp_path / "full_homology.json",
        sequence=target_seq,
        unpaired_msa=">query\nACDEFGHIKL\n>hit1\nACDEYGHIKL\n",
        templates=[
            {
                "mmcif": "data_full",
                "queryIndices": [0, 1, 2, 3],
                "templateIndices": [100, 101, 102, 103],
            },
        ],
    )

    rc = _run(
        [
            "design",
            "multidomain-binder",
            "--task-name",
            "t_multidomain_template",
            "--out-dir",
            "outputs",
            "--variant",
            "af3",
            "--target-name",
            "cre_like",
            "--binder-len",
            "80-120",
            "--target-homology",
            f"{full_homology}:A",
            "--domain-split-mode",
            "template",
            "--domain-indices",
            "2,3",
            "--domain-indices",
            "7-8",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.multidomain_target.domain_split_mode == "template"
    assert cfg.homologies.B.templates[0].mmcif == "data_full"
    assert list(cfg.homologies.B.templates[0].queryIndices) == [0, 1, 2, 3]


def test_cli_design_multidomain_binder_protenix_dry_run_accepts_num_runs_and_seed(
    tmp_path: Path,
    capsys,
) -> None:
    target_seq = "ACDEFGHIKL"
    full_homology = _write_af3_homology_json(
        tmp_path / "full_homology.json",
        sequence=target_seq,
        unpaired_msa=">query\nACDEFGHIKL\n>hit1\nACDEFGHIKL\n",
        templates=[
            {
                "mmcif": "data_domain1",
                "queryIndices": [0, 1],
                "templateIndices": [10, 11],
            },
            {
                "mmcif": "data_domain2",
                "queryIndices": [7, 8],
                "templateIndices": [20, 21],
            },
        ],
    )

    rc = _run(
        [
            "design",
            "multidomain-binder",
            "-t",
            "t_multidomain_protenix",
            "-o",
            "outputs",
            "--variant",
            "protenix",
            "--target-name",
            "cre_like",
            "--binder-len",
            "80",
            "--target-homology",
            f"{full_homology}:A",
            "--domain-indices",
            "1-2",
            "--domain-indices",
            "8,9",
            "--num-runs",
            "2",
            "--seed",
            "17",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.repeats == 2
    assert cfg.seed == 17
    assert cfg.binding_site.target == "B:1,2,8,9"
    assert cfg.stages.target_complex.evaluator == "Protenix"


def test_cli_design_multidomain_binder_accepts_initial_binder_sources(capsys) -> None:
    rc = _run(
        [
            "design",
            "multidomain-binder",
            "-t",
            "t_multidomain_seq",
            "-o",
            "outputs",
            "--variant",
            "af3",
            "--target-name",
            "cre_like",
            "--initial-binder-seq",
            "acdefx",
            "--target-seq",
            "ACDEFGHIKL",
            "--domain-indices",
            "1-2",
            "--domain-indices",
            "8-9",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.seqs.A == "ACDEFX"

    rc = _run(
        [
            "design",
            "multidomain-binder",
            "-t",
            "t_multidomain_expr",
            "-o",
            "outputs",
            "--variant",
            "af3",
            "--target-name",
            "cre_like",
            "--initial-binder-expr",
            "+ACD 20-30",
            "--target-seq",
            "ACDEFGHIKL",
            "--domain-indices",
            "1-2",
            "--domain-indices",
            "8-9",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.seqs.A == "+ACD 20-30"


def test_cli_design_multidomain_binder_domain_hotspot_override_values(
    tmp_path: Path,
    capsys,
) -> None:
    target_seq = "ACDEFGHIKL"
    full_homology = _write_af3_homology_json(
        tmp_path / "full_homology.json",
        sequence=target_seq,
        unpaired_msa=">query\nACDEFGHIKL\n>hit1\nACDEFGHIKL\n",
        templates=[
            {
                "mmcif": "data_domain1",
                "queryIndices": [0, 1, 2, 3],
                "templateIndices": [10, 11, 12, 13],
            },
            {
                "mmcif": "data_domain2",
                "queryIndices": [6, 7, 8, 9],
                "templateIndices": [20, 21, 22, 23],
            },
        ],
    )

    rc = _run(
        [
            "design",
            "multidomain-binder",
            "--task-name",
            "t_multidomain_hotspot_override",
            "--out-dir",
            "outputs",
            "--variant",
            "af3",
            "--target-name",
            "cre_like",
            "--binder-len",
            "80",
            "--target-homology",
            f"{full_homology}:A",
            "--domain-indices",
            "1-4",
            "--domain-indices",
            "7-10",
            "--domain-hotspots",
            "2",
            "--domain-hotspots",
            "8-9",
            "--dry-run",
        ]
    )
    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.multidomain_target.domains[0].indices == [1, 2, 3, 4]
    assert cfg.multidomain_target.domains[0].hotspots == [2]
    assert cfg.multidomain_target.domains[1].indices == [7, 8, 9, 10]
    assert cfg.multidomain_target.domains[1].hotspots == [8, 9]
    assert cfg.binding_site.target == "B:2,8,9"


def test_cli_design_multidomain_binder_requires_multiple_domain_index_groups() -> None:
    with pytest.raises(ValueError, match="Provide at least two --domain-indices groups"):
        _run(
            [
                "design",
                "multidomain-binder",
                "--task-name",
                "t_multidomain_domains",
                "--out-dir",
                "outputs",
                "--variant",
                "af3",
                "--target-name",
                "cre_like",
                "--binder-len",
                "80",
                "--target-seq",
                "ACDEFGHIKL",
                "--domain-indices",
                "1-2",
                "--dry-run",
            ]
        )


def test_cli_design_multidomain_binder_rejects_mismatched_domain_hotspot_group_count() -> None:
    with pytest.raises(ValueError, match="Domain hotspot groups must match domain index groups"):
        _run(
            [
                "design",
                "multidomain-binder",
                "--task-name",
                "t_multidomain_bad_hotspots",
                "--out-dir",
                "outputs",
                "--variant",
                "af3",
                "--target-name",
                "cre_like",
                "--binder-len",
                "80",
                "--target-seq",
                "ACDEFGHIKL",
                "--domain-indices",
                "1-2",
                "--domain-indices",
                "8-9",
                "--domain-hotspots",
                "1",
                "--dry-run",
            ]
        )


def test_cli_design_multidomain_binder_rejects_hotspots_outside_domain() -> None:
    with pytest.raises(ValueError, match="must be a subset"):
        _run(
            [
                "design",
                "multidomain-binder",
                "--task-name",
                "t_multidomain_bad_hotspot_value",
                "--out-dir",
                "outputs",
                "--variant",
                "af3",
                "--target-name",
                "cre_like",
                "--binder-len",
                "80",
                "--target-seq",
                "ACDEFGHIKL",
                "--domain-indices",
                "1-2",
                "--domain-indices",
                "8-9",
                "--domain-hotspots",
                "3",
                "--domain-hotspots",
                "8",
                "--dry-run",
            ]
        )
