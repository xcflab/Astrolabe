from __future__ import annotations

import contextlib
import csv
import io
import json
import random
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from astrolabe import cli, paper
from astrolabe.shell_completion import complete_words


def invoke(arguments):
    ns = cli._build_parser().parse_args(["paper", *arguments])
    with contextlib.redirect_stdout(io.StringIO()):
        return ns._handler(ns)


def test_every_paper_protocol_has_a_composable_configuration():
    for name in paper.PROTOCOLS:
        ns = cli._build_parser().parse_args(["paper", "show", name, "--config-env", "env"])
        cfg = paper._compose(ns)
        assert cfg.paper_protocol
        assert cfg.steps >= 1


def test_paper_preview_and_last_override_need_no_models(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Preview must not enter model execution")

    monkeypatch.setattr(cli, "_run_from_config", forbidden)
    inputs = tmp_path / "input.yaml"
    inputs.write_text("seqs:\n  B: FAD\ntypes:\n  B: c\npaper_protocol:\n  supplied_input: local\n")
    dump = tmp_path / "filled.yaml"
    assert invoke(["run", "ligand-binder-af3", "--inputs", str(inputs),
                   "--seed", "17", "--overrides", "steps=7", "--dump-config", str(dump),
                   "--config-env", "env", "--dry-run"]) == 0
    cfg = OmegaConf.load(dump)
    assert cfg.seed == 17 and cfg.steps == 7
    assert cfg.paper_protocol.supplied_input == "local"
    assert cfg.seqs.B == "FAD"


def test_paper_run_reaches_existing_engine_and_resolves_input_paths(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    inputs = source / "fixed.yaml"
    inputs.write_text("seqs:\n  A: ACDEFG\n  B: +AGST\n")
    out = tmp_path / "results"
    seen = []
    previous = Path.cwd()

    def execute(cfg, *, autoselect_gpu):
        seen.append((Path.cwd(), cfg.seqs.A, cfg.seed, cfg.repeats, cfg.out_dir))

    monkeypatch.setattr(cli, "_run_from_config", execute)
    assert invoke(["run", "evaluate-af3", "--inputs", str(inputs), "--seed", "7",
                   "--num-runs", "5", "--out-dir", str(out), "--config-env", "env"]) == 0
    assert seen == [(source, "ACDEFG", 7, 5, str(out))]
    assert Path.cwd() == previous


def test_fixed_evaluation_rejects_random_sequence_generation(tmp_path, monkeypatch):
    inputs = tmp_path / "fixed.yaml"
    inputs.write_text("seqs:\n  A: 80-120\n  B: +AGST\n")
    monkeypatch.setattr(cli, "_run_from_config", lambda *a, **k: pytest.fail("random evaluation ran"))
    with pytest.raises(ValueError, match="literal sequences"):
        invoke(["run", "evaluate-af3", "--inputs", str(inputs), "--config-env", "env"])


def test_bivalent_archived_starts_use_their_original_seeds(tmp_path, monkeypatch):
    import astrolabe.rna_beam as beam
    seen = []

    def execute(cfg, **kwargs):
        seen.append((cfg.beam_search.archived_start, kwargs["seed"], kwargs["max_evaluations"]))
        return {"budget_complete": True}

    monkeypatch.setattr(beam, "run_bivalent_rna_beam", execute)
    assert invoke(["run", "bivalent-rna-archived", "--num-runs", "3", "--out-dir", str(tmp_path),
                   "--config-env", "env"]) == 0
    assert seen == [(0, 67101, 200), (1, 67102, 200), (2, 67103, 200)]


def test_nucleic_target_initial_sequences_match_original_seeded_generator(tmp_path, monkeypatch):
    inputs = tmp_path / "target.yaml"
    inputs.write_text("seqs:\n  B: +ACGU\nhomologies:\n  B: homology.json\n")
    seen = []
    monkeypatch.setenv("GH_WEIGHT_PROTENIX_V1", "/unused/model.pt")
    monkeypatch.setattr(cli, "_run_from_config", lambda cfg, **kw: seen.append((cfg.seed, cfg.seqs.A)))
    invoke(["run", "protein-binder-rna-archived", "--inputs", str(inputs), "--seed", "1729",
            "--num-runs", "2", "--out-dir", str(tmp_path / "out"), "--config-env", "env"])
    for (seed, sequence), expected_seed in zip(seen, (1729, 1730)):
        rng = random.Random(expected_seed)
        expected = "".join(rng.choices("ADEFGHIKLMNPQRSTVWY", k=rng.randint(60, 80)))
        assert seed == expected_seed and sequence == expected


def test_input_bundles_extract_through_public_command(tmp_path):
    root = Path(paper.__file__).resolve().parents[1] / "resources"
    assert invoke(["prepare-inputs", "--bundle", "cyclic", "--out-dir", str(tmp_path)]) == 0
    manifest = json.loads((root / "paper_inputs_cyclic.json").read_text())
    assert all((tmp_path / item["path"]).is_file() for item in manifest["files"])


def test_paper_fixed_completion_does_not_read_configuration_files(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Static paper completion must not scan the filesystem")

    monkeypatch.setattr(Path, "glob", forbidden)
    values = {item.value for item in complete_words(["paper", "run", "rna-"], cword=2)}
    assert values == {"rna-aptamer", "rna-oligomer-4", "rna-oligomer-5"}
    values = {item.value for item in complete_words(["paper", "run", "bivalent-rna", "--"], cword=3)}
    assert {"--data-dir", "--inputs", "--seed", "--dry-run"} <= values


def test_parent_redesign_fixes_only_strict_heavy_atom_contacts(tmp_path):
    parent = tmp_path / "parent.pdb"
    atoms = [("ALA", "A", 1, 0.0), ("SER", "A", 2, 10.0), ("GLY", "B", 1, 3.0)]
    lines = [f"ATOM  {i:5d}  CA  {res} {chain}{pos:4d}    {x:8.3f}{0.:8.3f}{0.:8.3f}  1.00 90.00           C\n"
             for i, (res, chain, pos, x) in enumerate(atoms, 1)]
    parent.write_text("".join(lines) + "END\n")
    ns = cli._build_parser().parse_args(["paper", "show", "solublempnn-control", "--config-env", "env"])
    cfg = paper._compose(ns)
    paper._prepare_parent_redesign(cfg, parent)
    assert cfg.seqs.A == "+A S"
    assert cfg.seqs.B == "+G"
    assert cfg.paper_protocol.fixed_parent_interface_positions == [1]
    assert cfg.initializer.structure.source == str(parent)


def test_explicit_framework_switch_changes_homology_and_database_subset():
    ns = cli._build_parser().parse_args(["paper", "run", "antibody-framework",
                                        "--framework", "scfv4d5_vhvl", "--config-env", "env"])
    cfg = paper._compose(ns)
    paper._framework(cfg, ns)
    assert cfg.pre_eval_hooks.af3_antibody_msa_cache.af3_turboab_db_subset == "IgG"
    assert cfg.homologies.A.framework_name == "scfv_vh_linker_vl__rfantibody_4d5"


def test_baseline_cli_preview_exposes_native_assets_without_execution(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("preview executed"))
    native = tmp_path / "launcher.yaml"
    native.write_text("native_config: target.yaml\nnum_designs: 1000\nbudget: 80\n")
    output = tmp_path / "baseline"
    assert invoke(["baseline", "boltzgen", "--inputs", str(native), "--out-dir", str(output), "--dry-run"]) == 0
    assert not output.exists()
    choices = {item.value for item in complete_words(["paper", "baseline", "bind"], cword=2)}
    assert choices == {"bindcraft", "bindcraft-dual"}


def test_statistics_commands_write_results_with_explicit_columns_and_seed(tmp_path):
    rows = [dict(unit_id="one", method="Astrolabe-AF3", group=ligand,
                 confidence=.9, ptm=.8, plddt=.9, ddg=-45, dsasa=500,
                 ligand_hbonds=3, sc=.7, rg_ratio=1.1,
                 condition="dual", molecule_type="minibinder", ddg_ab=-45, ddg_ac=-40)
            for ligand in ("FAD", "IAI", "OQO", "SAM")]
    records = tmp_path / "records.csv"
    with records.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    common = ["--records", str(records), "--column", "iptm=confidence"]
    ranking = tmp_path / "rank"
    assert invoke(["analyze", "ligand-rank", *common, "--out-dir", str(ranking)]) == 0
    summary = json.loads((ranking / "ligand_ranking.json").read_text())
    assert summary["fixed_design_units"] == 4
    assert summary["columns"]["iptm"] == "confidence"
    sensitivity = tmp_path / "sensitivity"
    assert invoke(["analyze", "sensitivity", "--endpoint", "ligand-binder", *common,
                   "--planned", str(records), "--out-dir", str(sensitivity),
                   "--bootstrap-samples", "20", "--seed", "0"]) == 0
    ddg = tmp_path / "ddg"
    assert invoke(["analyze", "ddg", "--records", str(records), "--out-dir", str(ddg),
                   "--bootstrap-samples", "20", "--seed", "0"]) == 0
    summary = json.loads((ddg / "ddg_bootstrap.json").read_text())
    assert summary["parameters"]["seed"] == 0
    assert all(row["median_signed_delta"] == 5 for row in summary["groups"])
