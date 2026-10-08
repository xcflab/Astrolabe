from __future__ import annotations

import shutil
import subprocess
import sys
import os
from pathlib import Path

import astrolabe.cli as cli
import astrolabe.shell_completion as shell_completion
from astrolabe.posthoc.db import open_db, upsert_records
from astrolabe.shell_completion import complete_words
from astrolabe import __version__


def test_cli_program_version_and_checkout_launchers_are_astrolabe() -> None:
    parser = cli._build_parser()
    assert parser.prog == "astrolabe"
    version_action = next(action for action in parser._actions if getattr(action, "version", None))
    assert version_action.version == f"Astrolabe {__version__}"

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["ASTROLABE_PY"] = sys.executable
    env.pop("GH_PY", None)
    for launcher in ("astrolabe", "gh_dev"):
        result = subprocess.run(
            [str(repo_root / "scripts" / launcher), "--version"],
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        assert result.stdout.strip() == f"Astrolabe {__version__}"


def test_completion_cache_uses_astrolabe_namespace(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ASTROLABE_COMPLETION_CACHE_DIR", raising=False)
    monkeypatch.delenv("GH_COMPLETION_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    assert shell_completion._metric_name_cache_dir() == tmp_path / "astrolabe" / "completion"


def test_release_options_are_discoverable_in_completion() -> None:
    values = {item.value for item in complete_words(["--"], cword=0, cwd=Path.cwd())}
    assert {"--version", "--debug"} <= values


def test_paper_protocol_completion_includes_methods_and_archived_variants(monkeypatch) -> None:
    from astrolabe.paper import PROTOCOLS

    def no_scan(*args, **kwargs):
        raise AssertionError("Paper protocol choices come from the parser")

    monkeypatch.setattr(Path, "glob", no_scan)
    values = {item.value for item in complete_words(["paper", "run", ""], cword=2)}
    assert set(PROTOCOLS) <= values
    assert {"two-state-af3", "two-state-af3-archived", "bivalent-rna-archived",
            "protein-binder-rna-archived", "tandem-repeat-3-archived",
            "cyclic-peptide-bound-archived", "protein-binder-af3-archived",
            "cross-reactive-vhh-archived", "single-target-vhh-control-archived"} <= values


def _run(argv: list[str]) -> int:
    parser = cli._build_parser()
    ns = parser.parse_args(argv)
    handler = getattr(ns, "_handler", None)
    assert handler is not None
    return int(handler(ns))


def _seed_db(db_path: Path, records: list[dict]) -> None:
    conn = open_db(db_path)
    try:
        upsert_records(conn, records)
    finally:
        conn.close()


def _write_completion_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "scores.sqlite"
    structure = tmp_path / "outputs" / "task_a" / "001_run" / "structures" / "step_000_AB_model.cif"
    structure.parent.mkdir(parents=True, exist_ok=True)
    structure.write_text("data\n", encoding="utf-8")
    _seed_db(
        db_path,
        [
            {
                "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure,
                "metrics": {
                    "weighted_score": 1.2,
                    "iptm": 0.81,
                    "AB.iptm": 0.81,
                    "A.iptm": 0.76,
                    "binder_ptm": 0.72,
                    "A.hotspot_target_global": 0.5,
                    "AB.binding_site_allowed_min_dist": 1.0,
                },
                "scores_mtime_ns": 1,
                "scores_size": 10,
            }
        ],
    )
    return db_path


def _capture_compose_type_choices(monkeypatch, section) -> list[str]:
    commands = iter(["add", ":back"])
    choices: list[str] = []
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))

    def capture(_prompt, values, *, allow_empty):
        assert allow_empty is True
        choices.extend(values)
        return ""

    monkeypatch.setattr(cli, "_prompt_choice_from_list", capture)
    section({}, Path(__file__).resolve().parents[1] / "config")
    return choices


def test_config_compose_completes_surface_mutator_types(monkeypatch) -> None:
    choices = _capture_compose_type_choices(monkeypatch, cli._compose_section_mutator)

    assert {"SurfDesign", "SurfPro"} <= set(choices)


def test_config_compose_completes_opendde_evaluator_type(monkeypatch) -> None:
    choices = _capture_compose_type_choices(monkeypatch, cli._compose_section_evaluator)

    assert "OpenDDE" in choices


def test_cli_completion_bash_output_registers_astrolabe(capsys) -> None:
    rc = _run(["completion", "bash"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "_astrolabe_completion" in out
    assert "complete -o nosort -F _astrolabe_completion astrolabe" in out
    assert "completion complete --shell bash" in out


def test_cli_completion_bash_init_output_enables_progcomp(capsys) -> None:
    rc = _run(["completion", "bash", "--init"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "shopt -s progcomp" in out


def test_cli_completion_zsh_output_registers_astrolabe(capsys) -> None:
    rc = _run(["completion", "zsh"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "#compdef astrolabe" in out
    assert "completion complete --shell zsh" in out
    assert "compdef -p _astrolabe_completion '*/astrolabe'" in out


def test_cli_completion_zsh_init_output_runs_compinit_when_needed(capsys) -> None:
    rc = _run(["completion", "zsh", "--init"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "autoload -Uz compinit" in out
    assert "compinit -i" in out


def test_complete_words_suggests_option_names_for_design_ligand_binder() -> None:
    items = complete_words(["design", "ligand-binder", "-"], cword=2)
    values = {item.value for item in items}

    assert "-t" in values
    assert "--task-name" not in values
    assert "-n" in values
    assert "--num-runs" not in values
    assert "--ligand-type" in values
    task_item = next(item for item in items if item.value == "-t")
    assert task_item.description is not None
    assert task_item.description.startswith("-t --task-name: ")
    runs_item = next(item for item in items if item.value == "-n")
    assert runs_item.description is not None
    assert runs_item.description.startswith("-n --num-runs --repeats: ")


def test_complete_words_suggests_initial_binder_options_for_design_binder() -> None:
    items = complete_words(["design", "binder", "-"], cword=2)
    values = {item.value for item in items}

    assert "--binder-len" in values
    assert "--initial-binder-seq" in values
    assert "--initial-binder-expr" in values


def test_complete_words_suggests_initial_binder_options_for_design_idr_binder() -> None:
    items = complete_words(["design", "idr-binder", "-"], cword=2)
    values = {item.value for item in items}

    assert "--binder-len" in values
    assert "--initial-binder-seq" in values
    assert "--initial-binder-expr" in values
    assert "--target-seq" in values


def test_complete_words_suggests_options_for_design_cid_generate() -> None:
    items = complete_words(["design", "cid-generate", "-"], cword=2)
    values = {item.value for item in items}

    assert "-t" in values
    assert "-o" in values
    assert "--protein-a-len" in values
    assert "--protein-a-seq" in values
    assert "--protein-a-expr" in values
    assert "--protein-b-len" in values
    assert "--protein-b-seq" in values
    assert "--protein-b-expr" in values
    assert "--ligand" in values
    assert "--ligand-type" in values


def test_complete_words_suggests_initial_binder_options_for_design_peptide() -> None:
    items = complete_words(["design", "peptide-binder", "-"], cword=2)
    values = {item.value for item in items}

    assert "--peptide-len" in values
    assert "--initial-binder-seq" in values
    assert "--initial-binder-expr" in values


def test_complete_words_suggests_initial_binder_options_for_design_aptamer() -> None:
    items = complete_words(["design", "aptamer", "-"], cword=2)
    values = {item.value for item in items}

    assert "-t" in values
    assert "-o" in values
    assert "-n" in values
    assert "--aptamer-len" in values
    assert "--aptamer-type" in values
    assert "--initial-binder-seq" in values
    assert "--initial-binder-expr" in values


def test_complete_words_suggests_initial_binder_options_for_design_dna_aptamer() -> None:
    items = complete_words(["design", "dna-aptamer", "-"], cword=2)
    values = {item.value for item in items}

    assert "--aptamer-len" in values
    assert "--initial-binder-seq" in values
    assert "--initial-binder-expr" in values


def test_complete_words_suggests_initial_binder_options_for_design_ligand() -> None:
    items = complete_words(["design", "ligand-binder", "-"], cword=2)
    values = {item.value for item in items}

    assert "--protein-len" in values
    assert "--initial-binder-seq" in values
    assert "--initial-binder-expr" in values


def test_complete_words_suggests_initial_binder_options_for_design_oligomer() -> None:
    items = complete_words(["design", "oligomer", "-"], cword=2)
    values = {item.value for item in items}

    assert "--shape" in values
    assert "--protein-len" in values
    assert "--initial-binder-seq" in values
    assert "--initial-binder-expr" in values


def test_complete_words_suggests_choice_values_for_oligomer_shape() -> None:
    items = complete_words(
        ["design", "oligomer", "--shape", "D"],
        cword=3,
    )
    values = {item.value for item in items}

    assert {"D2", "D3", "D4"} <= values
    c_items = complete_words(["design", "oligomer", "--shape", "C"], cword=3)
    c_values = {item.value for item in c_items}
    assert {"C2", "C8", "C12"} <= c_values
    assert "C1" not in c_values


def test_complete_words_suggests_choice_values_for_aptamer_type() -> None:
    items = complete_words(
        ["design", "aptamer", "--aptamer-type", "d"],
        cword=3,
    )
    values = {item.value for item in items}

    assert values == {"dna"}


def test_complete_words_suggests_choice_values_for_domain_split_mode() -> None:
    items = complete_words(
        ["design", "multidomain-binder", "--domain-split-mode", ""],
        cword=3,
    )
    values = {item.value for item in items}

    assert values == {"full", "template"}


def test_complete_words_suggests_option_names_for_refine_ligand_pocket() -> None:
    items = complete_words(["refine", "ligand-pocket", "-"], cword=2)
    values = {item.value for item in items}

    assert "--reference-structure" in values
    assert "--reference-pocket" in values
    assert "--mutation-num" in values
    assert "--initial-temperature" in values
    assert "--contact-distance" in values
    assert "--scoring-mode" in values


def test_complete_words_suggests_choice_values_for_ligand_pocket_scoring_mode() -> None:
    items = complete_words(
        ["refine", "ligand-pocket", "--scoring-mode", "r"],
        cword=3,
    )
    values = {item.value for item in items}

    assert values == {"retarget"}


def test_complete_words_suggests_choice_values_for_symmetry_interface_selection() -> None:
    items = complete_words(
        ["bench-pseudocycle-v1", "--symmetry-interface-selection", "a"],
        cword=2,
    )
    values = {item.value for item in items}

    assert values == {"auto"}


def test_complete_words_suggests_reference_options_for_refine_binder() -> None:
    items = complete_words(["refine", "binder", "-"], cword=2)
    values = {item.value for item in items}

    assert "--reference" in values
    assert "--reference-mode" in values
    assert "--reference-tmscore-threshold" in values


def test_complete_words_suggests_refine_channel_surface() -> None:
    items = complete_words(["refine", "channel", "-"], cword=2)
    values = {item.value for item in items}

    assert "--ligand-residue" in values
    assert "--ligand-conformer" in values
    assert "--mutable-positions" in values
    assert "--spheres" in values
    assert "--msa" in values


def test_complete_words_suggests_experimental_candidate_batch_options() -> None:
    items = complete_words(["experimental", "candidate-batch", "-"], cword=2)
    values = {item.value for item in items}

    assert "--config-yaml" in values
    assert "--out-root" in values
    assert "--num-candidates" in values
    assert "--candidate-timeout-s" in values


def test_complete_words_suggests_experimental_candidate_batch_format_choices() -> None:
    items = complete_words(
        ["experimental", "candidate-batch", "--format", "j"],
        cword=3,
    )
    values = {item.value for item in items}

    assert values == {"json"}


def test_complete_words_suggests_choice_values_for_run_selection() -> None:
    items = complete_words(
        ["posthoc", "filter", "--run-selection", "r"],
        cword=3,
    )
    values = {item.value for item in items}

    assert "representative-per-run" in values


def test_complete_words_suggests_choice_values_for_ligand_type() -> None:
    items = complete_words(
        ["design", "ligand-binder", "--ligand-type", "c"],
        cword=3,
    )
    values = {item.value for item in items}

    assert "ccd" in values


def test_complete_words_uses_action_metadata_for_all_new_fixed_values() -> None:
    cases = [
        (["refine", "ligand", "--ligand-type", ""], 3, ("ccd", "smiles")),
        (["refine", "ligand-pocket", "--ligand-type", ""], 3, ("ccd", "smiles")),
        (["design", "ligand-binder", "--ligand-type", ""], 3, ("ccd", "smiles")),
        (["design", "cid-generate", "--ligand-type", ""], 3, ("ccd", "smiles")),
        (["bench-antibody-v1", "--framework-kind", ""], 2, ("auto", "vhh", "scfv")),
        (["design", "antibody", "--framework-kind", ""], 3, ("auto", "vhh", "scfv")),
        (["bench-antibody-v1", "--scfv-order", ""], 2, ("vh_linker_vl", "vl_linker_vh")),
        (["design", "antibody", "--scfv-order", ""], 3, ("vh_linker_vl", "vl_linker_vh")),
        (["bench-peptide-binder-v1", "--closure-profile", ""], 2, ("strict", "min", "mean")),
        (["design", "peptide", "--closure-profile", ""], 3, ("strict", "min", "mean")),
    ]

    for words, cword, expected in cases:
        assert tuple(item.value for item in complete_words(words, cword=cword)) == expected


def test_static_completion_spec_uses_action_metadata_and_canonical_alias_contexts() -> None:
    spec = shell_completion._collect_static_completion_spec()
    static_values = spec["static_values"]
    expected = {
        "refine/ligand:--ligand-type": "ccd\tccd\nsmiles\tsmiles",
        "refine/ligand-pocket:--ligand-type": "ccd\tccd\nsmiles\tsmiles",
        "design/ligand-binder:--ligand-type": "ccd\tccd\nsmiles\tsmiles",
        "design/cid-generate:--ligand-type": "ccd\tccd\nsmiles\tsmiles",
        "bench-antibody-v1:--framework-kind": "auto\tauto\nvhh\tvhh\nscfv\tscfv",
        "design/antibody:--framework-kind": "auto\tauto\nvhh\tvhh\nscfv\tscfv",
        "bench-antibody-v1:--scfv-order": "vh_linker_vl\tvh_linker_vl\nvl_linker_vh\tvl_linker_vh",
        "design/antibody:--scfv-order": "vh_linker_vl\tvh_linker_vl\nvl_linker_vh\tvl_linker_vh",
        "bench-peptide-binder-v1:--closure-profile": "strict\tstrict\nmin\tmin\nmean\tmean",
        "design/peptide:--closure-profile": "strict\tstrict\nmin\tmin\nmean\tmean",
    }
    assert {key: static_values[key] for key in expected} == expected

    next_context = spec["next_context"]
    assert next_context["design:binder"] == "design/protein-binder"
    assert next_context["design:peptide-binder"] == "design/peptide"
    assert next_context["design:ligand"] == "design/ligand-binder"
    assert next_context["design:cid"] == "design/cid-generate"


def test_complete_words_suggests_choice_values_for_cyclic_encoding() -> None:
    highfold_items = complete_words(
        ["design", "peptide-binder", "--cyclic-encoding", "h"],
        cword=3,
    )
    highfold_values = {item.value for item in highfold_items}

    rfpeptide_items = complete_words(
        ["design", "peptide-binder", "--cyclic-encoding", "r"],
        cword=3,
    )
    rfpeptide_values = {item.value for item in rfpeptide_items}

    assert "highfold-cycpoem" in highfold_values
    assert "rfpeptide-like" in rfpeptide_values


def test_complete_words_suggests_task_templates_from_config_dir(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "alpha.yaml").write_text("x: 1\n", encoding="utf-8")
    (config_dir / "beta.yaml").write_text("x: 1\n", encoding="utf-8")

    items = complete_words(
        ["--config-dir", str(config_dir), "env", "wizard", "--task", "a"],
        cword=5,
    )
    values = {item.value for item in items}

    assert "alpha" in values
    assert "beta" not in values


def test_complete_words_suggests_filter_metric_names_from_db_expression(
    tmp_path: Path,
) -> None:
    db_path = _write_completion_db(tmp_path)

    items = complete_words(
        [
            "posthoc",
            "filter",
            "--db",
            str(db_path),
            "--expr",
            "weighted_score > AB.i",
        ],
        cword=5,
    )
    values = {item.value for item in items}

    assert "weighted_score > AB.iptm" in values


def test_complete_words_suggests_fixed_db_columns_for_filter_expression(
    tmp_path: Path,
) -> None:
    db_path = _write_completion_db(tmp_path)

    items = complete_words(
        [
            "posthoc",
            "filter",
            "--db",
            str(db_path),
            "--expr",
            "st",
        ],
        cword=5,
    )
    values = {item.value for item in items}

    assert "structure_path" in values
    assert "step" in values


def test_complete_words_suggests_filter_metric_names_when_db_is_after_cursor(
    tmp_path: Path,
) -> None:
    db_path = _write_completion_db(tmp_path)

    items = complete_words(
        [
            "posthoc",
            "filter",
            "--filter-expr",
            "(max(iptm) - ",
            "--db",
            str(db_path),
        ],
        cword=3,
    )
    values = {item.value for item in items}

    assert "(max(iptm) - AB.iptm" in values
    assert "(max(iptm) - max(" in values


def test_complete_words_suggests_filter_metric_names_after_open_paren(
    tmp_path: Path,
) -> None:
    db_path = _write_completion_db(tmp_path)

    items = complete_words(
        [
            "posthoc",
            "filter",
            "--filter-expr",
            "(",
            "--db",
            str(db_path),
        ],
        cword=3,
    )
    values = {item.value for item in items}

    assert "(A.hotspot_target_global" in values
    assert "(AB.binding_site_allowed_min_dist" in values
    assert "(iptm" in values
    assert "(max(" in values


def test_complete_words_suggests_operator_and_keyword_tokens_for_empty_filter_expr_slot(
    tmp_path: Path,
) -> None:
    db_path = _write_completion_db(tmp_path)

    items = complete_words(
        [
            "posthoc",
            "filter",
            "--filter-expr",
            "(",
            "--db",
            str(db_path),
        ],
        cword=3,
    )
    values = {item.value for item in items}

    assert "(+" in values
    assert "(>=" in values
    assert "(and" in values


def test_complete_words_suggests_unprefixed_metric_name_from_filter_expr_prefix(
    tmp_path: Path,
) -> None:
    db_path = _write_completion_db(tmp_path)

    items = complete_words(
        [
            "posthoc",
            "filter",
            "--filter-expr",
            "ip",
            "--db",
            str(db_path),
        ],
        cword=3,
    )
    values = {item.value for item in items}

    assert "iptm" in values


def test_complete_words_rewrites_leading_missing_operand_before_operator(
    tmp_path: Path,
) -> None:
    db_path = _write_completion_db(tmp_path)

    items = complete_words(
        [
            "posthoc",
            "filter",
            "--filter-expr",
            "  > 0",
            "--db",
            str(db_path),
        ],
        cword=3,
    )
    values = {item.value for item in items}

    assert "  iptm > 0" in values
    assert "  A.iptm > 0" in values


def test_complete_words_filter_expr_candidates_show_metric_name_descriptions(
    tmp_path: Path,
) -> None:
    db_path = _write_completion_db(tmp_path)

    items = complete_words(
        [
            "posthoc",
            "filter",
            "--filter-expr",
            "(  +A.hotspot_target_global + AB.binding_site_allowed_min_dist+",
            "--db",
            str(db_path),
        ],
        cword=3,
    )
    mapping = {item.value: item.description for item in items}

    assert (
        mapping[
            "(  +A.hotspot_target_global + AB.binding_site_allowed_min_dist+AB.iptm"
        ]
        == "AB.iptm"
    )
    assert (
        mapping[
            "(  +A.hotspot_target_global + AB.binding_site_allowed_min_dist+max("
        ]
        == "max("
    )


def test_load_metric_names_from_db_reuses_persistent_cache_without_rescan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = _write_completion_db(tmp_path)
    cache_dir = tmp_path / "completion-cache"
    monkeypatch.setenv("ASTROLABE_COMPLETION_CACHE_DIR", str(cache_dir))

    calls = {"count": 0}
    original = shell_completion._scan_metric_names_from_db

    def _counting_scan(path: str) -> list[str]:
        calls["count"] += 1
        return original(path)

    monkeypatch.setattr(shell_completion, "_scan_metric_names_from_db", _counting_scan)

    first = shell_completion._load_metric_names_from_db(str(db_path))
    shell_completion._METRIC_NAME_CACHE.clear()
    second = shell_completion._load_metric_names_from_db(str(db_path))

    assert calls["count"] == 1
    assert first == second
    assert (cache_dir).exists()


def test_load_metric_names_from_db_reuses_persistent_cache_for_same_inode_alias_within_ttl(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = _write_completion_db(tmp_path)
    alias_path = tmp_path / "scores-alias.sqlite"
    os.link(db_path, alias_path)
    cache_dir = tmp_path / "completion-cache"
    monkeypatch.setenv("ASTROLABE_COMPLETION_CACHE_DIR", str(cache_dir))

    calls = {"count": 0}
    original = shell_completion._scan_metric_names_from_db

    def _counting_scan(path: str) -> list[str]:
        calls["count"] += 1
        return original(path)

    monkeypatch.setattr(shell_completion, "_scan_metric_names_from_db", _counting_scan)
    shell_completion._METRIC_NAME_CACHE.clear()

    first = shell_completion._load_metric_names_from_db(str(db_path))
    shell_completion._METRIC_NAME_CACHE.clear()
    second = shell_completion._load_metric_names_from_db(str(alias_path))

    assert calls["count"] == 1
    assert first == second


def test_complete_words_db_argument_prefers_database_files_within_directory(
    tmp_path: Path,
) -> None:
    collected = tmp_path / "collected"
    collected.mkdir()
    (collected / "scores.sqlite").write_text("", encoding="utf-8")
    (collected / "other.db").write_text("", encoding="utf-8")
    (collected / "nested").mkdir()

    items = complete_words(
        ["posthoc", "filter", "--db", "collected/"],
        cword=3,
        cwd=tmp_path,
    )
    values = {item.value for item in items}

    assert "collected/scores.sqlite" in values
    assert "collected/other.db" in values
    assert "collected/" not in values
    assert "collected/nested/" not in values


def test_complete_words_groups_option_aliases_into_single_completion_item() -> None:
    items = complete_words(["posthoc", "filter", "--"], cword=2)
    values = {item.value for item in items}

    assert "--db" in values
    assert "--sqlite" not in values

    db_item = next(item for item in items if item.value == "--db")
    assert db_item.description is not None
    assert db_item.description.startswith("--db --sqlite: ")


def test_complete_words_subcommands_include_command_name_in_display() -> None:
    items = complete_words([""], cword=0)
    values = {item.value for item in items}

    assert "run" in values
    run_item = next(item for item in items if item.value == "run")
    assert run_item.description is not None
    assert run_item.description.startswith("run: Run generation/optimization.")


def test_complete_words_posthoc_subcommands_use_summary_help_not_long_description() -> None:
    items = complete_words(["posthoc", ""], cword=1)
    mapping = {item.value: item.description for item in items}

    assert mapping["channel-screen"] == "channel-screen: Directly mutate mutable residues to fixed amino acids and rescore HOLE."
    assert mapping["collect"] == "collect: Incrementally collect run scores into SQLite (optional CSV export)."
    assert mapping["filter"] == "filter: Filter structures using a DSL expression or a preset and hardlink renamed outputs."


def test_complete_words_suggests_posthoc_channel_screen_options() -> None:
    items = complete_words(["posthoc", "channel-screen", "--"], cword=2)
    values = {item.value for item in items}

    assert "--structure" in values
    assert "--ligand-residue" in values
    assert "--mutable-positions" in values
    assert "--target-aas" in values
    assert "--contact-distance" in values


def test_complete_words_design_subcommands_show_summary_text() -> None:
    items = complete_words(["design", ""], cword=1)
    mapping = {item.value: item.description for item in items}

    assert mapping["protein-binder"] == "protein-binder: Protein binder benchmark preset (v1.4)."
    assert mapping["binder"] == "binder (alias: protein-binder): Protein binder benchmark preset (v1.4)."
    assert mapping["idr-binder"] == "idr-binder: IDR-target protein binder benchmark preset (v1)."
    assert (
        mapping["cid-generate"]
        == "cid-generate: Chemical-induced dimer sensor generation preset (proteins A/B plus ligand C)."
    )
    assert (
        mapping["cid"]
        == "cid (alias: cid-generate): Chemical-induced dimer sensor generation preset (proteins A/B plus ligand C)."
    )
    assert mapping["dna-aptamer"] == "dna-aptamer: DNA aptamer benchmark preset (v1)."
    assert (
        mapping["multidomain-binder"]
        == "multidomain-binder: Flexible multi-domain target binder preset (single target chain B)."
    )
    assert mapping["oligomer"] == "oligomer: De novo homo-oligomer morphology preset (C2-N or D2-D4)."


def test_complete_words_suggests_option_names_for_design_multidomain_binder() -> None:
    items = complete_words(["design", "multidomain-binder", "-"], cword=2)
    values = {item.value for item in items}

    assert "-t" in values
    assert "--binder-len" in values
    assert "--initial-binder-seq" in values
    assert "--target-name" in values
    assert "--domain-indices" in values
    assert "--domain-hotspots" in values
    assert "--domain-split-mode" in values
    assert "--target-homology" in values


def test_complete_words_selects_matching_alias_for_long_option_prefix() -> None:
    sqlite_items = complete_words(["posthoc", "filter", "--sq"], cword=2)
    expr_items = complete_words(["posthoc", "filter", "--ex"], cword=2)
    out_db_items = complete_words(["posthoc", "filter", "--out"], cword=2)

    assert [item.value for item in sqlite_items] == ["--sqlite"]
    assert [item.value for item in expr_items] == ["--expr"]
    assert [item.value for item in out_db_items] == ["--out-db"]


def test_bash_completion_script_round_trip_for_filter_out_db_option() -> None:
    bash = shutil.which("bash")
    assert bash is not None
    repo_root = Path(__file__).resolve().parents[1]
    program = f"""
set -euo pipefail
astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
source <(astrolabe completion bash)
COMP_WORDS=(astrolabe posthoc filter --out)
COMP_CWORD=3
_astrolabe_completion
printf '%s\\n' "${{COMPREPLY[@]}}"
"""
    res = subprocess.run([bash, "-lc", program], check=True, text=True, capture_output=True)
    assert "--out-db" in res.stdout


def test_zsh_completion_script_round_trip_for_filter_out_db_option() -> None:
    zsh = shutil.which("zsh")
    assert zsh is not None
    repo_root = Path(__file__).resolve().parents[1]
    program = f"""
set -euo pipefail
function astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
function compdef() {{ :; }}
function compadd() {{
  local emit=0
  local arg
  for arg in "$@"; do
    if [[ "$emit" == 1 ]]; then
      print -- "$arg"
      continue
    fi
    if [[ "$arg" == -- ]]; then
      emit=1
    fi
  done
}}
source <(astrolabe completion zsh)
words=(astrolabe posthoc filter --out)
CURRENT=4
PREFIX='--out'
SUFFIX=''
_astrolabe_completion
"""
    res = subprocess.run([zsh, "-lc", program], check=True, text=True, capture_output=True)
    assert "--out-db" in res.stdout


def test_bash_completion_script_round_trip_for_new_fixed_choices() -> None:
    bash = shutil.which("bash")
    assert bash is not None
    repo_root = Path(__file__).resolve().parents[1]
    program = f"""
set -euo pipefail
astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
source <(astrolabe completion bash)
complete_case() {{
  COMP_WORDS=($1)
  COMP_CWORD=$2
  _astrolabe_completion
  printf '%s\\n' "${{COMPREPLY[@]}}"
  printf '%s\\n' '---'
}}
complete_case 'astrolabe design oligomer --shape C' 4
complete_case 'astrolabe design aptamer --aptamer-type d' 4
  complete_case 'astrolabe design multidomain-binder --domain-split-mode t' 4
  complete_case 'astrolabe refine ligand-pocket --scoring-mode r' 4
  complete_case 'astrolabe bench-pseudocycle-v1 --symmetry-interface-selection m' 3
  complete_case 'astrolabe design ligand-binder --ligand-type c' 4
  complete_case 'astrolabe design antibody --framework-kind v' 4
  complete_case 'astrolabe design antibody --scfv-order vl' 4
  complete_case 'astrolabe design peptide --closure-profile m' 4
  complete_case 'astrolabe paper run rna-' 3
  complete_case 'astrolabe paper analyze scores --endpoint cyclic-' 5
  complete_case 'astrolabe paper analyze a' 3
  complete_case 'astrolabe paper analyze sensitivity --denominator a' 5
  complete_case 'astrolabe paper baseline rfdiffusion' 3
"""
    res = subprocess.run([bash, "-lc", program], check=True, text=True, capture_output=True)
    sections = res.stdout.split("---\n")
    c_values = set(sections[0].splitlines())
    assert {"C2", "C12"} <= c_values
    assert "C1" not in c_values
    assert "dna" in set(sections[1].splitlines())
    assert "template" in set(sections[2].splitlines())
    assert "retarget" in set(sections[3].splitlines())
    assert "manual" in set(sections[4].splitlines())
    assert set(sections[5].splitlines()) == {"ccd"}
    assert set(sections[6].splitlines()) == {"vhh"}
    assert set(sections[7].splitlines()) == {"vl_linker_vh"}
    assert set(sections[8].splitlines()) == {"min", "mean"}
    assert set(sections[9].splitlines()) == {"rna-aptamer", "rna-oligomer-4", "rna-oligomer-5"}
    assert set(sections[10].splitlines()) == {"cyclic-bound", "cyclic-multiring"}
    assert set(sections[11].splitlines()) == {"agreement"}
    assert set(sections[12].splitlines()) == {"available"}
    assert set(sections[13].splitlines()) == {"rfdiffusion-proteinmpnn", "rfdiffusion2", "rfdiffusion3"}


def test_zsh_completion_script_round_trip_for_new_fixed_choices(tmp_path: Path) -> None:
    zsh = shutil.which("zsh")
    assert zsh is not None
    repo_root = Path(__file__).resolve().parents[1]
    marker = tmp_path / "python_complete_called.txt"
    program = f"""
set -euo pipefail
function astrolabe() {{
  if [[ "$1" == completion && "$2" == complete ]]; then
    print CALLED >> "{marker}"
  fi
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
function compdef() {{ :; }}
function compadd() {{
  local emit=0
  local arg
  for arg in "$@"; do
    if [[ "$emit" == 1 ]]; then
      print -- "$arg"
      continue
    fi
    if [[ "$arg" == -- ]]; then
      emit=1
    fi
  done
}}
source <(astrolabe completion zsh)
complete_case() {{
  words=(${{=1}})
  CURRENT=$2
  PREFIX=$3
  SUFFIX=''
  _astrolabe_completion
  print -- '---'
}}
complete_case 'astrolabe design oligomer --shape C' 5 C
complete_case 'astrolabe design aptamer --aptamer-type d' 5 d
  complete_case 'astrolabe design multidomain-binder --domain-split-mode t' 5 t
  complete_case 'astrolabe refine ligand-pocket --scoring-mode r' 5 r
  complete_case 'astrolabe bench-pseudocycle-v1 --symmetry-interface-selection m' 4 m
  complete_case 'astrolabe design ligand --ligand-type c' 5 c
  complete_case 'astrolabe design antibody --framework-kind v' 5 v
  complete_case 'astrolabe design antibody --scfv-order vl' 5 vl
  complete_case 'astrolabe design peptide-binder --closure-profile m' 5 m
  complete_case 'astrolabe paper run rna-' 4 rna-
  complete_case 'astrolabe paper analyze scores --endpoint cyclic-' 6 cyclic-
  complete_case 'astrolabe paper analyze a' 4 a
  complete_case 'astrolabe paper analyze sensitivity --denominator a' 6 a
  complete_case 'astrolabe paper baseline rfdiffusion' 4 rfdiffusion
"""
    res = subprocess.run([zsh, "-lc", program], check=True, text=True, capture_output=True)
    sections = res.stdout.split("---\n")
    c_values = set(sections[0].splitlines())
    assert {"C2", "C12"} <= c_values
    assert "C1" not in c_values
    assert "dna" in set(sections[1].splitlines())
    assert "template" in set(sections[2].splitlines())
    assert "retarget" in set(sections[3].splitlines())
    assert "manual" in set(sections[4].splitlines())
    assert set(sections[5].splitlines()) == {"ccd"}
    assert set(sections[6].splitlines()) == {"vhh"}
    assert set(sections[7].splitlines()) == {"vl_linker_vh"}
    assert set(sections[8].splitlines()) == {"min", "mean"}
    assert set(sections[9].splitlines()) == {"rna-aptamer", "rna-oligomer-4", "rna-oligomer-5"}
    assert set(sections[10].splitlines()) == {"cyclic-bound", "cyclic-multiring"}
    assert set(sections[11].splitlines()) == {"agreement"}
    assert set(sections[12].splitlines()) == {"available"}
    assert set(sections[13].splitlines()) == {"rfdiffusion-proteinmpnn", "rfdiffusion2", "rfdiffusion3"}
    # Protocol and comparator are positional choices, completed through the shared parser.
    assert marker.read_text().splitlines() == ["CALLED", "CALLED"]


def test_bash_completion_script_round_trip_for_experimental_candidate_batch_option() -> None:
    bash = shutil.which("bash")
    assert bash is not None
    repo_root = Path(__file__).resolve().parents[1]
    program = f"""
set -euo pipefail
astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
source <(astrolabe completion bash)
COMP_WORDS=(astrolabe experimental candidate-batch --candidate)
COMP_CWORD=3
_astrolabe_completion
printf '%s\\n' "${{COMPREPLY[@]}}"
"""
    res = subprocess.run([bash, "-lc", program], check=True, text=True, capture_output=True)
    assert "--candidate-steps" in res.stdout
    assert "--candidate-timeout-s" in res.stdout


def test_zsh_completion_script_round_trip_for_experimental_candidate_batch_option() -> None:
    zsh = shutil.which("zsh")
    assert zsh is not None
    repo_root = Path(__file__).resolve().parents[1]
    program = f"""
set -euo pipefail
function astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
function compdef() {{ :; }}
function compadd() {{
  local emit=0
  local arg
  for arg in "$@"; do
    if [[ "$emit" == 1 ]]; then
      print -- "$arg"
      continue
    fi
    if [[ "$arg" == -- ]]; then
      emit=1
    fi
  done
}}
source <(astrolabe completion zsh)
words=(astrolabe experimental candidate-batch --candidate)
CURRENT=4
PREFIX='--candidate'
SUFFIX=''
_astrolabe_completion
"""
    res = subprocess.run([zsh, "-lc", program], check=True, text=True, capture_output=True)
    assert "--candidate-steps" in res.stdout
    assert "--candidate-timeout-s" in res.stdout


def test_bash_completion_script_round_trip_for_filter_expr(tmp_path: Path) -> None:
    bash = shutil.which("bash")
    assert bash is not None
    db_path = _write_completion_db(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    program = f"""
set -euo pipefail
astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
source <(astrolabe completion bash)
COMP_WORDS=(astrolabe posthoc filter --db "{db_path}" --expr 'weighted_score > AB.i')
COMP_CWORD=6
_astrolabe_completion
printf '%s\\n' "${{COMPREPLY[@]}}"
"""
    res = subprocess.run([bash, "-lc", program], check=True, text=True, capture_output=True)
    assert "weighted_score > AB.iptm" in res.stdout


def test_bash_completion_script_round_trip_for_refine_channel_option() -> None:
    bash = shutil.which("bash")
    assert bash is not None
    repo_root = Path(__file__).resolve().parents[1]
    program = f"""
set -euo pipefail
astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
source <(astrolabe completion bash)
COMP_WORDS=(astrolabe refine channel --lig)
COMP_CWORD=3
_astrolabe_completion
printf '%s\\n' "${{COMPREPLY[@]}}"
"""
    res = subprocess.run([bash, "-lc", program], check=True, text=True, capture_output=True)
    assert "--ligand-residue" in res.stdout
    assert "--ligand-conformer" in res.stdout


def test_bash_completion_script_round_trip_for_posthoc_channel_screen_option() -> None:
    bash = shutil.which("bash")
    assert bash is not None
    repo_root = Path(__file__).resolve().parents[1]
    program = f"""
set -euo pipefail
astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
source <(astrolabe completion bash)
COMP_WORDS=(astrolabe posthoc channel-screen --ta)
COMP_CWORD=3
_astrolabe_completion
printf '%s\\n' "${{COMPREPLY[@]}}"
"""
    res = subprocess.run([bash, "-lc", program], check=True, text=True, capture_output=True)
    assert "--target-aas" in res.stdout


def test_zsh_completion_script_round_trip_for_filter_expr(tmp_path: Path) -> None:
    zsh = shutil.which("zsh")
    assert zsh is not None
    db_path = _write_completion_db(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    program = f"""
set -euo pipefail
function astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
function compdef() {{ :; }}
function compadd() {{
  local emit=0
  local arg
  for arg in "$@"; do
    if [[ "$emit" == 1 ]]; then
      print -- "${{IPREFIX-}}${{arg}}"
      continue
    fi
    if [[ "$arg" == -- ]]; then
      emit=1
    fi
  done
}}
    source <(astrolabe completion zsh)
    words=(astrolabe posthoc filter --db "{db_path}" --expr 'weighted_score > AB.i')
    CURRENT=7
    PREFIX='weighted_score > AB.i'
    SUFFIX=''
_astrolabe_completion
"""
    res = subprocess.run([zsh, "-lc", program], check=True, text=True, capture_output=True)
    assert "weighted_score > AB.iptm" in res.stdout


def test_zsh_completion_script_round_trip_for_refine_channel_subcommand() -> None:
    zsh = shutil.which("zsh")
    assert zsh is not None
    repo_root = Path(__file__).resolve().parents[1]
    program = f"""
set -euo pipefail
function astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
function compdef() {{ :; }}
function compadd() {{
  local emit=0
  local arg
  for arg in "$@"; do
    if [[ "$emit" == 1 ]]; then
      print -- "${{IPREFIX-}}${{arg}}"
      continue
    fi
    if [[ "$arg" == -- ]]; then
      emit=1
    fi
  done
}}
source <(astrolabe completion zsh)
words=(astrolabe refine ch)
CURRENT=3
PREFIX='ch'
SUFFIX=''
_astrolabe_completion
"""
    res = subprocess.run([zsh, "-lc", program], check=True, text=True, capture_output=True)
    assert "channel" in res.stdout


def test_zsh_completion_script_round_trip_for_posthoc_channel_screen_option() -> None:
    zsh = shutil.which("zsh")
    assert zsh is not None
    repo_root = Path(__file__).resolve().parents[1]
    program = f"""
set -euo pipefail
function astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
function compdef() {{ :; }}
function compadd() {{
  local emit=0
  local arg
  for arg in "$@"; do
    if [[ "$emit" == 1 ]]; then
      print -- "${{IPREFIX-}}${{arg}}"
      continue
    fi
    if [[ "$arg" == -- ]]; then
      emit=1
    fi
  done
}}
source <(astrolabe completion zsh)
words=(astrolabe posthoc channel-screen --ta)
CURRENT=4
PREFIX='--ta'
SUFFIX=''
_astrolabe_completion
"""
    res = subprocess.run([zsh, "-lc", program], check=True, text=True, capture_output=True)
    assert "--target-aas" in res.stdout


def test_zsh_completion_init_script_registers_without_manual_compinit() -> None:
    zsh = shutil.which("zsh")
    assert zsh is not None
    repo_root = Path(__file__).resolve().parents[1]
    program = f"""
set -euo pipefail
function astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
source <(astrolabe completion zsh --init)
whence -w compdef
print SEP
(( $+functions[_astrolabe_completion] )) && print HAS_FUNC || print NO_FUNC
"""
    res = subprocess.run([zsh, "-lc", program], check=True, text=True, capture_output=True)
    assert "compdef: function" in res.stdout
    assert "HAS_FUNC" in res.stdout


def test_zsh_completion_static_option_context_does_not_call_python_complete(tmp_path: Path) -> None:
    zsh = shutil.which("zsh")
    assert zsh is not None
    repo_root = Path(__file__).resolve().parents[1]
    marker = tmp_path / "python_complete_called.txt"
    program = f"""
set -euo pipefail
function astrolabe() {{
  if [[ "$1" == completion && "$2" == zsh ]]; then
    "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
    return
  fi
  if [[ "$1" == completion && "$2" == complete ]]; then
    print CALLED >> "{marker}"
    return 0
  fi
  return 0
}}
function compdef() {{ :; }}
function compadd() {{ :; }}
source <(astrolabe completion zsh)
words=(astrolabe posthoc filter --)
CURRENT=3
PREFIX='--'
SUFFIX=''
_astrolabe_completion
"""
    subprocess.run([zsh, "-lc", program], check=True, text=True, capture_output=True)
    assert not marker.exists()


def test_zsh_completion_dynamic_expr_context_still_calls_python_complete(tmp_path: Path) -> None:
    zsh = shutil.which("zsh")
    assert zsh is not None
    repo_root = Path(__file__).resolve().parents[1]
    marker = tmp_path / "python_complete_called.txt"
    db_path = _write_completion_db(tmp_path)
    program = f"""
set -euo pipefail
function astrolabe() {{
  if [[ "$1" == completion && "$2" == zsh ]]; then
    "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
    return
  fi
  if [[ "$1" == completion && "$2" == complete ]]; then
    print CALLED >> "{marker}"
    "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
    return
  fi
  return 0
}}
function compdef() {{ :; }}
function compadd() {{ :; }}
source <(astrolabe completion zsh)
words=(astrolabe posthoc filter --db "{db_path}" --expr 'weighted_score > AB.i')
CURRENT=7
PREFIX='weighted_score > AB.i'
SUFFIX=''
_astrolabe_completion
"""
    subprocess.run([zsh, "-lc", program], check=True, text=True, capture_output=True)
    assert marker.exists()


def test_zsh_completion_script_uses_design_child_subcommands_not_root_commands(
    tmp_path: Path,
) -> None:
    zsh = shutil.which("zsh")
    assert zsh is not None
    repo_root = Path(__file__).resolve().parents[1]
    marker = tmp_path / "python_complete_called.txt"
    program = f"""
set -euo pipefail
function astrolabe() {{
  if [[ "$1" == completion && "$2" == zsh ]]; then
    "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
    return
  fi
  if [[ "$1" == completion && "$2" == complete ]]; then
    print CALLED >> "{marker}"
    "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
    return
  fi
  return 0
}}
function compdef() {{ :; }}
function compadd() {{
  local emit=0
  local arg
  for arg in "$@"; do
    if [[ "$emit" == 1 ]]; then
      print -- "${{IPREFIX-}}${{arg}}"
      continue
    fi
    if [[ "$arg" == -- ]]; then
      emit=1
    fi
  done
}}
source <(astrolabe completion zsh)
words=(astrolabe design '')
CURRENT=3
PREFIX=''
SUFFIX=''
_astrolabe_completion
"""
    res = subprocess.run([zsh, "-lc", program], check=True, text=True, capture_output=True)
    assert not marker.exists()
    assert "protein-binder" in res.stdout
    assert "run" not in res.stdout


def test_zsh_completion_script_uses_prefix_for_midword_filter_expr(tmp_path: Path) -> None:
    zsh = shutil.which("zsh")
    assert zsh is not None
    db_path = _write_completion_db(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    program = f"""
set -euo pipefail
function astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
function compdef() {{ :; }}
function compadd() {{
  local emit=0
  local arg
  for arg in "$@"; do
    if [[ "$emit" == 1 ]]; then
      print -- "${{IPREFIX-}}${{arg}}"
      continue
    fi
    if [[ "$arg" == -- ]]; then
      emit=1
    fi
  done
}}
source <(astrolabe completion zsh)
words=(astrolabe posthoc filter --db "{db_path}" --expr '(max(iptm) - iptm  ) < 0.2')
CURRENT=7
PREFIX='(max(iptm) - iptm  '
SUFFIX=') < 0.2'
_astrolabe_completion
"""
    res = subprocess.run([zsh, "-lc", program], check=True, text=True, capture_output=True)
    assert "(max(iptm) - iptm  AB.iptm" in res.stdout


def test_zsh_completion_script_handles_full_expr_word_without_complete_in_word(
    tmp_path: Path,
) -> None:
    zsh = shutil.which("zsh")
    assert zsh is not None
    db_path = _write_completion_db(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    program = f"""
set -euo pipefail
function astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
function compdef() {{ :; }}
function compadd() {{
  local emit=0
  local arg
  for arg in "$@"; do
    if [[ "$emit" == 1 ]]; then
      print -- "${{IPREFIX-}}${{arg}}"
      continue
    fi
    if [[ "$arg" == -- ]]; then
      emit=1
    fi
  done
}}
source <(astrolabe completion zsh)
words=(astrolabe posthoc filter --filter-expr '  > 0' --db "{db_path}")
CURRENT=5
PREFIX='  > 0'
IPREFIX=''
SUFFIX=''
ISUFFIX=''
_astrolabe_completion
"""
    res = subprocess.run([zsh, "-lc", program], check=True, text=True, capture_output=True)
    assert "  iptm > 0" in res.stdout


def test_zsh_completion_script_uses_iprefix_for_filter_expr_after_open_paren(
    tmp_path: Path,
) -> None:
    zsh = shutil.which("zsh")
    assert zsh is not None
    db_path = _write_completion_db(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    program = f"""
set -euo pipefail
function astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
function compdef() {{ :; }}
function compadd() {{
  local emit=0
  local arg
  for arg in "$@"; do
    if [[ "$emit" == 1 ]]; then
      print -- "${{IPREFIX-}}${{arg}}"
      continue
    fi
    if [[ "$arg" == -- ]]; then
      emit=1
    fi
  done
}}
source <(astrolabe completion zsh)
words=(astrolabe posthoc filter --filter-expr '' --db "{db_path}")
CURRENT=5
PREFIX=''
IPREFIX='('
SUFFIX=' +A.hotspot_target_global + AB.binding_site_allowed_min_dist ) > 0'
ISUFFIX=''
_astrolabe_completion
"""
    res = subprocess.run([zsh, "-lc", program], check=True, text=True, capture_output=True)
    assert "(A.hotspot_target_global" in res.stdout


def test_zsh_completion_script_visible_candidates_do_not_duplicate_iprefix_after_open_paren(
    tmp_path: Path,
) -> None:
    zsh = shutil.which("zsh")
    assert zsh is not None
    db_path = _write_completion_db(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    program = f"""
set -euo pipefail
function astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
function compdef() {{ :; }}
function compadd() {{
  local emit=0
  local arg
  for arg in "$@"; do
    if [[ "$emit" == 1 ]]; then
      print -- "${{IPREFIX-}}${{arg}}"
      continue
    fi
    if [[ "$arg" == -- ]]; then
      emit=1
    fi
  done
}}
source <(astrolabe completion zsh)
words=(astrolabe posthoc filter --filter-expr '' --db "{db_path}")
CURRENT=5
PREFIX=''
IPREFIX='('
SUFFIX=' +A.hotspot_target_global + AB.binding_site_allowed_min_dist ) > 0'
ISUFFIX=''
_astrolabe_completion
"""
    res = subprocess.run([zsh, "-lc", program], check=True, text=True, capture_output=True)
    assert "(A.hotspot_target_global" in res.stdout
    assert "((A.hotspot_target_global" not in res.stdout


def test_zsh_completion_script_uses_iprefix_for_filter_expr_after_plus(
    tmp_path: Path,
) -> None:
    zsh = shutil.which("zsh")
    assert zsh is not None
    db_path = _write_completion_db(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    program = f"""
set -euo pipefail
function astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
function compdef() {{ :; }}
function compadd() {{
  local emit=0
  local arg
  for arg in "$@"; do
    if [[ "$emit" == 1 ]]; then
      print -- "${{IPREFIX-}}${{arg}}"
      continue
    fi
    if [[ "$arg" == -- ]]; then
      emit=1
    fi
  done
}}
source <(astrolabe completion zsh)
words=(astrolabe posthoc filter --filter-expr '' --db "{db_path}")
CURRENT=5
PREFIX=''
IPREFIX='( +A.hotspot_target_global + AB.binding_site_allowed_min_dist+'
SUFFIX=' ) > 0'
ISUFFIX=''
_astrolabe_completion
"""
    res = subprocess.run([zsh, "-lc", program], check=True, text=True, capture_output=True)
    assert "( +A.hotspot_target_global + AB.binding_site_allowed_min_dist+AB.iptm" in res.stdout


def test_zsh_completion_script_visible_candidates_do_not_duplicate_iprefix_after_plus(
    tmp_path: Path,
) -> None:
    zsh = shutil.which("zsh")
    assert zsh is not None
    db_path = _write_completion_db(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    program = f"""
set -euo pipefail
function astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
function compdef() {{ :; }}
function compadd() {{
  local emit=0
  local arg
  for arg in "$@"; do
    if [[ "$emit" == 1 ]]; then
      print -- "${{IPREFIX-}}${{arg}}"
      continue
    fi
    if [[ "$arg" == -- ]]; then
      emit=1
    fi
  done
}}
source <(astrolabe completion zsh)
words=(astrolabe posthoc filter --filter-expr '' --db "{db_path}")
CURRENT=5
PREFIX=''
IPREFIX='( +A.hotspot_target_global + AB.binding_site_allowed_min_dist+'
SUFFIX=' ) > 0'
    ISUFFIX=''
    _astrolabe_completion
"""
    res = subprocess.run([zsh, "-lc", program], check=True, text=True, capture_output=True)
    needle = "( +A.hotspot_target_global + AB.binding_site_allowed_min_dist+AB.iptm"
    assert needle in res.stdout
    assert (
        "( +A.hotspot_target_global + AB.binding_site_allowed_min_dist+"
        "( +A.hotspot_target_global + AB.binding_site_allowed_min_dist+AB.iptm"
    ) not in res.stdout

def test_zsh_completion_script_completes_config_dir_value_after_subcommand(
    tmp_path: Path,
) -> None:
    zsh = shutil.which("zsh")
    assert zsh is not None
    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    program = f"""
set -euo pipefail
cd "{tmp_path}"
function astrolabe() {{
  "{sys.executable}" -c "import sys; sys.path.insert(0, '{repo_root}/vendor'); sys.path.insert(0, '{repo_root}'); from astrolabe.cli import main; main(sys.argv[1:])" "$@"
}}
function compdef() {{ :; }}
function compadd() {{
  local emit=0
  local arg
  for arg in "$@"; do
    if [[ "$emit" == 1 ]]; then
      print -- "$arg"
      continue
    fi
    if [[ "$arg" == -- ]]; then
      emit=1
    fi
  done
}}
source <(astrolabe completion zsh)
words=(astrolabe design protein-binder --config-dir conf)
CURRENT=5
PREFIX='conf'
SUFFIX=''
_astrolabe_completion
"""
    res = subprocess.run([zsh, "-lc", program], check=True, text=True, capture_output=True)
    assert "configs/" in res.stdout


def test_complete_words_suggests_yaml_presets(tmp_path: Path) -> None:
    (tmp_path / "demo_preset.yaml").write_text("name: demo\nrules: []\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore\n", encoding="utf-8")

    items = complete_words(
        ["posthoc", "filter", "--preset", "de"],
        cword=3,
        cwd=tmp_path,
    )
    values = {item.value for item in items}

    assert "demo_preset.yaml" in values
    assert "notes.txt" not in values


def test_complete_words_preview_preset_suggests_only_preset_candidates(
    tmp_path: Path,
) -> None:
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    (presets_dir / "demo_preset.yaml").write_text(
        "name: demo_preset\nrules:\n  - key: weighted_score\n    op: gt\n    value: 0\n",
        encoding="utf-8",
    )
    (tmp_path / "not_a_preset.yaml").write_text("task: demo\n", encoding="utf-8")
    (tmp_path / "random.txt").write_text("ignore\n", encoding="utf-8")

    items = complete_words(
        ["posthoc", "filter", "--preview-preset", "--preset", ""],
        cword=4,
        cwd=tmp_path,
    )
    values = {item.value for item in items}

    assert "presets/demo_preset.yaml" in values
    assert "not_a_preset.yaml" not in values
    assert "presets/" not in values


def test_complete_words_non_preview_preset_keeps_path_completion(
    tmp_path: Path,
) -> None:
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    (tmp_path / "not_a_preset.yaml").write_text("task: demo\n", encoding="utf-8")

    items = complete_words(
        ["posthoc", "filter", "--preset", ""],
        cword=3,
        cwd=tmp_path,
    )
    values = {item.value for item in items}

    assert "presets/" in values
    assert "not_a_preset.yaml" in values


def test_complete_words_suggests_yaml_presets_next_to_db(tmp_path: Path) -> None:
    db_dir = tmp_path / "analysis"
    db_dir.mkdir()
    (db_dir / "demo_preset.yaml").write_text("name: demo\nrules: []\n", encoding="utf-8")
    db_path = db_dir / "scores.sqlite"
    db_path.write_text("", encoding="utf-8")

    items = complete_words(
        ["posthoc", "filter", "--db", str(db_path), "--preset", "de"],
        cword=5,
        cwd=tmp_path / "elsewhere",
    )
    values = {item.value for item in items}

    assert "demo_preset.yaml" in values
