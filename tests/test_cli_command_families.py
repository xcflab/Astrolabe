from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import astrolabe.cli as cli
from astrolabe import antibody_framework_presets


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(autouse=True)
def _patch_default_config_dir(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "default_config"
    shutil.copytree(CONFIG_DIR, config_dir, ignore=shutil.ignore_patterns("__pycache__"))
    (config_dir / "env").mkdir(exist_ok=True)
    (config_dir / "env" / "env_local.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_default_config_dir", lambda: config_dir)


def _run(argv: list[str]) -> int:
    parser = cli._build_parser()
    ns = parser.parse_args(cli._rewrite_legacy_posthoc_refold_argv(argv, parser))
    handler = getattr(ns, "_handler", None)
    assert handler is not None
    return int(handler(ns))


def _yaml_from_output(text: str):
    marker = "task_name:"
    idx = text.find(marker)
    assert idx >= 0, text
    return OmegaConf.create(text[idx:])


def _config_dir_with_env(tmp_path: Path, *config_names: str) -> Path:
    dst_root = tmp_path / "config"
    (dst_root / "env").mkdir(parents=True)
    (dst_root / "env" / "env_local.yaml").write_text("{}\n", encoding="utf-8")
    shutil.copy2(CONFIG_DIR / "base.yaml", dst_root / "base.yaml")
    shutil.copy2(CONFIG_DIR / "env.yaml", dst_root / "env.yaml")
    for name in config_names:
        shutil.copy2(CONFIG_DIR / f"{name}.yaml", dst_root / f"{name}.yaml")
    return dst_root


def _install_fake_module(monkeypatch, name: str, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _subparser_action(parser: argparse.ArgumentParser):
    return next(
        (
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ),
        None,
    )


def _parser_at(
    parser: argparse.ArgumentParser,
    path: tuple[str, ...],
) -> argparse.ArgumentParser:
    current = parser
    for name in path:
        current = _subparser_action(current).choices[name]
    return current


def _action_signature(action: argparse.Action) -> tuple[object, ...]:
    return (
        tuple(action.option_strings),
        action.dest,
        type(action),
        action.required,
        action.default,
        action.choices,
        action.nargs,
        action.type,
        action.metavar,
        action.help,
        getattr(action, "completion_choices", None),
    )


def test_bench_and_design_argument_surfaces_and_namespaces_remain_equal() -> None:
    pairs = [
        (("bench-binder-v1-4",), ("design", "protein-binder"), ["--task-name", "t", "--out-dir", "out", "--binder-len", "10", "--target-seq", "AC"], 17),
        (("bench-idr-binder-v1",), ("design", "idr-binder"), ["--task-name", "t", "--out-dir", "out", "--binder-len", "10", "--target-seq", "AC"], 17),
        (("bench-peptide-binder-v1",), ("design", "peptide"), ["--task-name", "t", "--out-dir", "out", "--peptide-len", "8", "--target-seq", "AC"], 27),
        (("bench-aptamer-v1",), ("design", "aptamer"), ["--task-name", "t", "--out-dir", "out", "--aptamer-len", "8", "--target-seq", "AC"], 18),
        (("bench-antibody-v1",), ("design", "antibody"), ["--task-name", "t", "--out-dir", "out", "--binder-seq", "+AAAA 3-3 +BBBB", "--target-seq", "AC"], 27),
    ]
    parser = cli._build_parser()

    for compatibility_path, canonical_path, argv, action_count in pairs:
        compatibility_actions = [
            _action_signature(action)
            for action in _parser_at(parser, compatibility_path)._actions
        ]
        canonical_actions = [
            _action_signature(action)
            for action in _parser_at(parser, canonical_path)._actions
        ]
        assert len(compatibility_actions) == action_count
        assert compatibility_actions == canonical_actions

        compatibility_ns = vars(parser.parse_args([*compatibility_path, *argv]))
        canonical_ns = vars(parser.parse_args([*canonical_path, *argv]))
        assert compatibility_ns["_handler"] is canonical_ns["_handler"]
        for namespace in (compatibility_ns, canonical_ns):
            namespace.pop("cmd", None)
            namespace.pop("design_cmd", None)
        assert compatibility_ns == canonical_ns


def test_design_native_alias_namespaces_remain_equal() -> None:
    cases = [
        (
            ("design", "binder"),
            ("design", "protein-binder"),
            ["--task-name", "t", "--out-dir", "out", "--binder-len", "10", "--target-seq", "AC"],
        ),
        (
            ("design", "peptide-binder"),
            ("design", "peptide"),
            ["--task-name", "t", "--out-dir", "out", "--peptide-len", "8", "--target-seq", "AC"],
        ),
    ]
    parser = cli._build_parser()

    for alias_path, canonical_path, argv in cases:
        alias_ns = vars(parser.parse_args([*alias_path, *argv]))
        canonical_ns = vars(parser.parse_args([*canonical_path, *argv]))
        assert alias_ns["_handler"] is canonical_ns["_handler"]
        for namespace in (alias_ns, canonical_ns):
            namespace.pop("cmd", None)
            namespace.pop("design_cmd", None)
        assert alias_ns == canonical_ns


def test_completion_only_action_metadata_does_not_add_runtime_choices() -> None:
    expected = [
        (("refine", "ligand"), "ligand_type", ("ccd", "smiles")),
        (("refine", "ligand-pocket"), "ligand_type", ("ccd", "smiles")),
        (("design", "ligand-binder"), "ligand_type", ("ccd", "smiles")),
        (("design", "cid-generate"), "ligand_type", ("ccd", "smiles")),
        (("bench-antibody-v1",), "framework_kind", ("auto", "vhh", "scfv")),
        (("design", "antibody"), "framework_kind", ("auto", "vhh", "scfv")),
        (("bench-antibody-v1",), "scfv_order", ("vh_linker_vl", "vl_linker_vh")),
        (("design", "antibody"), "scfv_order", ("vh_linker_vl", "vl_linker_vh")),
        (("bench-peptide-binder-v1",), "closure_profile", ("strict", "min", "mean")),
        (("design", "peptide"), "closure_profile", ("strict", "min", "mean")),
    ]
    parser = cli._build_parser()

    for path, dest, completion_choices in expected:
        action = next(
            action
            for action in _parser_at(parser, path)._actions
            if action.dest == dest
        )
        assert action.choices is None
        assert action.completion_choices == completion_choices


def test_completion_metadata_preserves_normalized_runtime_inputs() -> None:
    parser = cli._build_parser()
    for raw_ligand_type, normalized in [(" C ", "c"), (" s ", "s")]:
        ns = parser.parse_args(
            [
                "design",
                "ligand-binder",
                "--task-name",
                "t",
                "--out-dir",
                "out",
                "--protein-len",
                "10",
                "--ligand",
                "FAD",
                "--ligand-type",
                raw_ligand_type,
            ]
        )
        assert cli._ligand_type_to_alias(ns.ligand_type) == normalized

    antibody_ns = parser.parse_args(
        [
            "design",
            "antibody",
            "--framework-kind",
            " VHH ",
            "--scfv-order",
            "VH-Linker-VL",
        ]
    )
    assert antibody_framework_presets._normalize_kind(antibody_ns.framework_kind) == "vhh"
    assert (
        antibody_framework_presets._normalize_scfv_order(antibody_ns.scfv_order)
        == "vh_linker_vl"
    )

    peptide_ns = parser.parse_args(
        [
            "design",
            "peptide",
            "--task-name",
            "t",
            "--out-dir",
            "out",
            "--peptide-len",
            "8",
            "--target-seq",
            "AC",
            "--closure-profile",
            " MeAn ",
        ]
    )
    assert cli._parse_closure_profile(peptide_ns.closure_profile) == {
        "sigma": 0.20,
        "aggregate": "mean",
    }


def test_all_canonical_leaf_handlers_use_shared_registration() -> None:
    leaves = cli._canonical_leaf_parsers(cli._build_parser())

    assert len(leaves) == 57
    assert all(callable(leaf._defaults.get("_handler")) for leaf in leaves)
    assert all(
        leaf._defaults.get("_handler_registered") is leaf._defaults.get("_handler")
        for leaf in leaves
    )


def test_handler_registration_rejects_direct_handler_overwrite() -> None:
    parser = cli._build_parser()
    leaf = cli._canonical_leaf_parsers(parser)[0]
    leaf.set_defaults(_handler=lambda _ns: 0)

    with pytest.raises(RuntimeError, match="registered via cli_defs.set_handler"):
        cli._validate_handler_registrations(parser)


def test_set_handler_fixed_defaults_cannot_be_overridden() -> None:
    parser = argparse.ArgumentParser()
    handler = lambda _ns: 0

    cli.cli_defs.set_handler(
        parser,
        handler,
        _handler=lambda _ns: 1,
        _handler_registered=False,
    )

    assert parser.get_default("_handler") is handler
    assert parser.get_default("_handler_registered") is handler


def test_cli_interactive_imports_without_missing_core_helpers() -> None:
    sys.modules.pop("astrolabe.cli_interactive", None)
    mod = importlib.import_module("astrolabe.cli_interactive")
    assert hasattr(mod, "_cmd_compose")


def test_cli_run_dry_run_prints_config_and_skips_pipeline(tmp_path: Path, monkeypatch, capsys):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "demo.yaml").write_text("dummy: true\n", encoding="utf-8")

    called = {"run": False}

    def _fake_compose_hydra_config(*, config_dir, config_name, overrides):
        assert config_name == "demo"
        assert isinstance(config_dir, Path)
        assert isinstance(overrides, list)
        return OmegaConf.create({"task": "demo", "dry": True})

    def _fake_run_from_config(_cfg):
        called["run"] = True

    monkeypatch.setattr(cli, "_compose_hydra_config", _fake_compose_hydra_config)
    monkeypatch.setattr(cli, "_run_from_config", _fake_run_from_config)

    rc = _run(["--config-dir", str(config_dir), "run", "--task", "demo", "--dry-run"])
    assert rc == 0
    assert called["run"] is False
    assert "task: demo" in capsys.readouterr().out


def test_cli_run_without_config_selection_prints_help_and_returns_2(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_run_from_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    rc = _run(["run"])

    assert rc == 2
    out = capsys.readouterr()
    assert "usage: astrolabe run" in out.out
    assert "--task" in out.out
    assert "--standalone" in out.out


def test_cli_posthoc_collect_wires_arguments(monkeypatch, capsys):
    captured = {}

    class MissingRequiredMetricsError(RuntimeError):
        pass

    def _collect_scores(**kwargs):
        captured.update(kwargs)
        return {
            "out_root": str(kwargs["out_root"]),
            "db_path": str(kwargs["db_path"]),
            "runs_found": 7,
            "runs_skipped": 2,
            "runs_processed": 5,
            "export_csv": str(kwargs["export_csv"]) if kwargs["export_csv"] else None,
        }

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.collect",
        MissingRequiredMetricsError=MissingRequiredMetricsError,
        collect_scores=_collect_scores,
    )

    rc = _run(
        [
            "posthoc",
            "collect",
            "--out-root",
            "outputs",
            "--db",
            "scores.sqlite",
            "--workers",
            "3",
            "--task-regex",
            "task_.*",
            "--run-regex",
            "001_.*",
            "--export-csv",
            "scores.csv",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    assert captured["workers"] == 3
    assert captured["task_regex"] == "task_.*"
    assert captured["run_regex"] == "001_.*"
    assert str(captured["export_csv"]).endswith("scores.csv")
    assert '"runs_found": 7' in capsys.readouterr().out


def test_cli_posthoc_collect_accepts_repeated_out_roots(monkeypatch, capsys):
    captured = {}

    class MissingRequiredMetricsError(RuntimeError):
        pass

    def _collect_scores(**kwargs):
        captured.update(kwargs)
        return {
            "out_root": "",
            "out_roots": [str(p) for p in kwargs["out_roots"]],
            "db_path": str(kwargs["db_path"]),
            "runs_found": 2,
            "runs_skipped": 0,
            "runs_processed": 2,
            "export_csv": None,
        }

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.collect",
        MissingRequiredMetricsError=MissingRequiredMetricsError,
        collect_scores=_collect_scores,
    )

    rc = _run(
        [
            "posthoc",
            "collect",
            "--out-root",
            "outputs_a",
            "--out-root",
            "outputs_b",
            "--db",
            "scores.sqlite",
            "--format",
            "json",
        ]
    )

    assert rc == 0
    assert [str(p) for p in captured["out_roots"]] == ["outputs_a", "outputs_b"]
    out = capsys.readouterr().out
    assert '"runs_found": 2' in out
    assert '"out_roots": [' in out


def test_cli_posthoc_collect_wires_archive_arguments(monkeypatch, capsys):
    captured = {}

    class MissingRequiredMetricsError(RuntimeError):
        pass

    def _collect_scores(**kwargs):
        captured.update(kwargs)
        return {
            "out_root": str(kwargs["out_root"]),
            "db_path": str(kwargs["db_path"]),
            "runs_found": 1,
            "runs_skipped": 0,
            "runs_processed": 1,
            "archive_dir": str(kwargs["archive_dir"]),
            "archived_created": 1,
            "archived_skipped": 0,
            "archive_failed": 0,
        }

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.collect",
        MissingRequiredMetricsError=MissingRequiredMetricsError,
        collect_scores=_collect_scores,
    )

    rc = _run(
        [
            "posthoc",
            "collect",
            "--out-root",
            "outputs",
            "--db",
            "scores.sqlite",
            "--archive-dir",
            "archives",
            "--archive-workers",
            "2",
            "--archive-overwrite",
        ]
    )

    assert rc == 0
    assert str(captured["archive_dir"]).endswith("archives")
    assert captured["archive_workers"] == 2
    assert captured["archive_overwrite"] is True
    out = capsys.readouterr().out
    assert "archive_dir:" in out
    assert "archive: created=1 skipped=0 failed=0" in out


def test_cli_posthoc_filter_wires_arguments(monkeypatch, capsys):
    captured = {}

    class MissingRequiredMetricsError(RuntimeError):
        pass

    def _filter_structures(**kwargs):
        captured.update(kwargs)
        return {
            "db_path": str(kwargs["db_path"]),
            "preset": str(kwargs["preset_path"]),
            "prefer_oligo": kwargs["prefer_oligo"],
            "dst_dir": str(kwargs["dst_dir"]),
            "selected": 2,
            "records_considered": 3,
            "report_csv": str(kwargs["report_csv"]) if kwargs["report_csv"] else "",
            "out_db": str(kwargs["out_db"]) if kwargs["out_db"] else "",
        }

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.filtering",
        MissingRequiredMetricsError=MissingRequiredMetricsError,
        filter_structures=_filter_structures,
    )

    rc = _run(
        [
            "posthoc",
            "filter",
            "--db",
            "scores.sqlite",
            "--preset",
            "preset.yaml",
            "--dst",
            "filtered",
            "--copy",
            "--out-db",
            "filtered.sqlite",
            "--allow-missing-required",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    assert captured["hardlink"] is False
    assert str(captured["out_db"]).endswith("filtered.sqlite")
    assert captured["fail_on_missing_required"] is False
    assert captured["run_selection"] == "all-passing"
    assert '"out_db": "filtered.sqlite"' in capsys.readouterr().out


def test_cli_posthoc_prune_wires_arguments(monkeypatch, capsys):
    captured = {}

    class MissingRequiredMetricsError(RuntimeError):
        pass

    def _prune_outputs(**kwargs):
        captured.update(kwargs)
        return {
            "db_path": str(kwargs["db_path"]),
            "selector_kind": "dsl",
            "filter_expr": kwargs["filter_expr"],
            "prefer_oligo": kwargs["prefer_oligo"],
            "run_selection": kwargs["run_selection"],
            "delete_scope": kwargs["delete_scope"],
            "selected": 2,
            "deleted_paths": 2,
            "missing_paths": 0,
            "delete_errors": 0,
            "dry_run": kwargs["dry_run"],
            "report_csv": str(kwargs["report_csv"]) if kwargs["report_csv"] else "",
        }

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.filtering",
        MissingRequiredMetricsError=MissingRequiredMetricsError,
        prune_outputs=_prune_outputs,
    )

    rc = _run(
        [
            "posthoc",
            "prune",
            "--db",
            "scores.sqlite",
            "--filter-expr",
            "binder_ptm >= 0.8",
            "--delete-scope",
            "run_dir",
            "--run-selection",
            "representative-per-run",
            "--yes",
            "--report-csv",
            "prune.csv",
            "--format",
            "json",
        ]
    )

    assert rc == 0
    assert captured["delete_scope"] == "run_dir"
    assert captured["run_selection"] == "representative-per-run"
    assert captured["dry_run"] is False
    assert str(captured["report_csv"]).endswith("prune.csv")
    assert '"deleted_paths": 2' in capsys.readouterr().out


def test_cli_posthoc_report_wires_arguments(monkeypatch, capsys):
    captured = {}

    def _generate_posthoc_report(**kwargs):
        captured.update(kwargs)
        return {
            "db_path": str(kwargs["db_path"]),
            "selector_kind": "dsl",
            "filter_expr": kwargs["filter_expr"],
            "prefer_oligo": kwargs["prefer_oligo"],
            "dst_dir": str(kwargs["dst_dir"]),
            "trend_metrics": list(kwargs["trend_metrics"]),
            "summary_csv": str(kwargs["dst_dir"] / "posthoc_summary.csv"),
            "per_step_csv": str(kwargs["dst_dir"] / "posthoc_per_step.csv"),
            "report_md": str(kwargs["dst_dir"] / "posthoc_report.md"),
            "step_trends_png": str(kwargs["dst_dir"] / "step_trends.png"),
            "success_rate_png": str(kwargs["dst_dir"] / "success_rate.png"),
        }

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.report",
        generate_posthoc_report=_generate_posthoc_report,
    )

    rc = _run(
        [
            "posthoc",
            "report",
            "--db",
            "scores.sqlite",
            "--filter-expr",
            "binder_ptm >= 0.8",
            "--dst",
            "report_dir",
            "--trend-metric",
            "binder_ptm",
            "--trend-metric",
            "ipsae",
            "--format",
            "json",
        ]
    )

    assert rc == 0
    assert captured["filter_expr"] == "binder_ptm >= 0.8"
    assert captured["trend_metrics"] == ["binder_ptm", "ipsae"]
    assert "posthoc_summary.csv" in capsys.readouterr().out


def test_cli_posthoc_collect_accepts_simplified_aliases(monkeypatch, capsys):
    captured = {}

    class MissingRequiredMetricsError(RuntimeError):
        pass

    def _collect_scores(**kwargs):
        captured.update(kwargs)
        return {
            "out_root": str(kwargs["out_root"]),
            "db_path": str(kwargs["db_path"]),
            "runs_found": 1,
            "runs_skipped": 0,
            "runs_processed": 1,
            "export_csv": str(kwargs["export_csv"]) if kwargs["export_csv"] else None,
        }

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.collect",
        MissingRequiredMetricsError=MissingRequiredMetricsError,
        collect_scores=_collect_scores,
    )

    rc = _run(
        [
            "posthoc",
            "collect",
            "--outputs",
            "outputs",
            "--sqlite",
            "scores.sqlite",
            "--csv",
            "scores.csv",
        ]
    )

    assert rc == 0
    assert str(captured["out_root"]).endswith("outputs")
    assert str(captured["db_path"]).endswith("scores.sqlite")
    assert str(captured["export_csv"]).endswith("scores.csv")
    assert "runs: found=1" in capsys.readouterr().out


def test_cli_posthoc_collect_exposes_required_metric_options(monkeypatch, capsys):
    captured = {}

    class MissingRequiredMetricsError(RuntimeError):
        pass

    def _collect_scores(**kwargs):
        captured.update(kwargs)
        return {
            "out_root": str(kwargs["out_root"]),
            "db_path": str(kwargs["db_path"]),
            "runs_found": 1,
            "runs_skipped": 0,
            "runs_processed": 1,
            "required_metrics": list(kwargs["required_metrics"]),
            "required_failed_records_total": 0,
            "export_csv": None,
        }

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.collect",
        MissingRequiredMetricsError=MissingRequiredMetricsError,
        collect_scores=_collect_scores,
    )

    rc = _run(
        [
            "posthoc",
            "collect",
            "--out-root",
            "outputs",
            "--db",
            "scores.sqlite",
            "--required-metric",
            "binder_ptm",
            "--required-metric",
            "AB.iptm",
            "--required-metrics-preset",
            "preset.yaml",
            "--allow-missing-required",
        ]
    )

    assert rc == 0
    assert captured["required_metrics"] == ["binder_ptm", "AB.iptm"]
    assert str(captured["required_metrics_preset"]).endswith("preset.yaml")
    assert captured["fail_on_missing_required"] is False
    assert "required_metrics: binder_ptm,AB.iptm" in capsys.readouterr().out


def test_cli_posthoc_filter_accepts_simplified_aliases_and_run_selection(
    monkeypatch, capsys
):
    captured = {}

    class MissingRequiredMetricsError(RuntimeError):
        pass

    def _filter_structures(**kwargs):
        captured.update(kwargs)
        return {
            "db_path": str(kwargs["db_path"]),
            "selector_kind": "dsl",
            "filter_expr": kwargs["filter_expr"],
            "prefer_oligo": kwargs["prefer_oligo"],
            "dst_dir": str(kwargs["dst_dir"]),
            "selected": 1,
            "records_considered": 1,
            "report_csv": str(kwargs["report_csv"]) if kwargs["report_csv"] else "",
        }

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.filtering",
        MissingRequiredMetricsError=MissingRequiredMetricsError,
        filter_structures=_filter_structures,
    )

    rc = _run(
        [
            "posthoc",
            "filter",
            "--sqlite",
            "scores.sqlite",
            "--expr",
            "weighted_score > 0",
            "--dest",
            "filtered",
            "--oligo",
            "AB",
            "--run-selection",
            "representative-per-run",
        ]
    )

    assert rc == 0
    assert str(captured["db_path"]).endswith("scores.sqlite")
    assert captured["filter_expr"] == "weighted_score > 0"
    assert captured["prefer_oligo"] == "AB"
    assert captured["run_selection"] == "representative-per-run"
    assert "selected: 1 / 1" in capsys.readouterr().out


def test_cli_posthoc_filter_supports_dry_run_short_flag(monkeypatch, capsys):
    captured = {}

    class MissingRequiredMetricsError(RuntimeError):
        pass

    def _filter_structures(**kwargs):
        captured.update(kwargs)
        return {
            "db_path": str(kwargs["db_path"]),
            "selector_kind": "dsl",
            "filter_expr": kwargs["filter_expr"],
            "prefer_oligo": kwargs["prefer_oligo"],
            "dst_dir": str(kwargs["dst_dir"]),
            "dry_run": bool(kwargs["dry_run"]),
            "selected": 4,
            "records_considered": 10,
            "linkable": 3,
            "structure_missing": 1,
            "report_csv": None,
        }

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.filtering",
        MissingRequiredMetricsError=MissingRequiredMetricsError,
        filter_structures=_filter_structures,
    )

    rc = _run(
        [
            "posthoc",
            "filter",
            "--db",
            "scores.sqlite",
            "--expr",
            "weighted_score > 0",
            "-n",
            "--format",
            "json",
        ]
    )

    assert rc == 0
    assert captured["dry_run"] is True
    out = capsys.readouterr().out
    assert '"dry_run": true' in out
    assert '"linkable": 3' in out


def test_cli_posthoc_filter_prints_db_out_root_when_available(monkeypatch, capsys):
    class MissingRequiredMetricsError(RuntimeError):
        pass

    def _filter_structures(**kwargs):
        return {
            "db_path": str(kwargs["db_path"]),
            "db_out_root": "/tmp/source_outputs",
            "selector_kind": "dsl",
            "filter_expr": kwargs["filter_expr"],
            "prefer_oligo": kwargs["prefer_oligo"],
            "dst_dir": str(kwargs["dst_dir"]),
            "selected": 0,
            "records_considered": 1,
            "report_csv": str(kwargs["report_csv"]) if kwargs["report_csv"] else "",
        }

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.filtering",
        MissingRequiredMetricsError=MissingRequiredMetricsError,
        filter_structures=_filter_structures,
    )

    rc = _run(
        [
            "posthoc",
            "filter",
            "--db",
            "scores.sqlite",
            "--expr",
            'stage == "af3_final"',
            "--dst",
            "filtered",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "db_out_root: /tmp/source_outputs" in out


def test_cli_posthoc_filter_missing_required_returns_2(monkeypatch, capsys):
    class MissingRequiredMetricsError(RuntimeError):
        pass

    def _filter_structures(**_kwargs):
        raise MissingRequiredMetricsError("missing required metrics in db")

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.filtering",
        MissingRequiredMetricsError=MissingRequiredMetricsError,
        filter_structures=_filter_structures,
    )

    rc = _run(
        [
            "posthoc",
            "filter",
            "--db",
            "scores.sqlite",
            "--preset",
            "preset.yaml",
            "--dst",
            "filtered",
        ]
    )
    assert rc == 2
    assert "missing required metrics in db" in capsys.readouterr().err


def test_cli_posthoc_filter_runtime_error_returns_2(monkeypatch, capsys):
    class MissingRequiredMetricsError(RuntimeError):
        pass

    class FilterExpressionRuntimeError(RuntimeError):
        pass

    def _filter_structures(**_kwargs):
        raise FilterExpressionRuntimeError("filter expression runtime error")

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.filtering",
        MissingRequiredMetricsError=MissingRequiredMetricsError,
        FilterExpressionRuntimeError=FilterExpressionRuntimeError,
        filter_structures=_filter_structures,
    )

    rc = _run(
        [
            "posthoc",
            "filter",
            "--db",
            "scores.sqlite",
            "--filter-expr",
            "missing(weighted_score, weighted_score)",
            "--dst",
            "filtered",
        ]
    )

    assert rc == 2
    assert "filter expression runtime error" in capsys.readouterr().err


def test_cli_posthoc_refold_export_wires_arguments(monkeypatch, capsys):
    captured = {}

    def _export_refold_manifest(**kwargs):
        captured.update(kwargs)
        return {
            "db_path": str(kwargs["db_path"]),
            "preset": str(kwargs["preset_path"]),
            "prefer_oligo": kwargs["prefer_oligo"],
            "passed_only": kwargs["passed_only"],
            "out_manifest": str(kwargs["out_manifest"]),
            "report_csv": str(kwargs["report_csv"]) if kwargs["report_csv"] else None,
            "records_considered": 5,
            "records_passed": 4,
            "written": 4,
            "failed": 0,
        }

    def _run_refold(**_kwargs):
        raise AssertionError("run_refold should not be called in export test")

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.refold",
        export_refold_manifest=_export_refold_manifest,
        run_refold=_run_refold,
    )

    rc = _run(
        [
            "posthoc",
            "refold-export",
            "--db",
            "scores.sqlite",
            "--preset",
            "preset.yaml",
            "--out-manifest",
            "manifest.jsonl",
            "--no-passed-only",
            "--path-map",
            "/old=/new",
            "--limit",
            "5",
        ]
    )
    assert rc == 0
    assert captured["passed_only"] is False
    assert captured["path_map"] == ["/old=/new"]
    assert captured["limit"] == 5
    assert "records: considered=5 passed=4 written=4 failed=0" in capsys.readouterr().out


def test_cli_posthoc_refold_wires_selector_driven_workflow(monkeypatch, capsys):
    captured = {}

    def _refold_from_db(**kwargs):
        captured.update(kwargs)
        return {
            "db_path": str(kwargs["db_path"]),
            "selector_kind": "dsl",
            "filter_expr": kwargs["filter_expr"],
            "prefer_oligo": kwargs["prefer_oligo"],
            "run_selection": kwargs["run_selection"],
            "dst_dir": str(kwargs["dst_dir"]),
            "manifest_jsonl": str(kwargs["dst_dir"] / "refold_manifest.jsonl"),
            "refold_out_root": str(kwargs["dst_dir"] / "refold_runs"),
            "evaluators": kwargs["evaluators"],
            "collect_db_path": str(kwargs["db_path"]),
            "exported": 2,
            "processed": 2,
            "skipped": 0,
            "failed": 0,
            "collected_runs": 2,
            "dry_run": kwargs["dry_run"],
            "ran_af3": 1,
            "ran_protenix": 1,
            "reused_af3": 0,
            "reused_protenix": 0,
            "groups": 2,
            "resource_groups": 1,
            "evaluator_cache_hits": 1,
            "evaluator_cache_misses": 1,
            "workers_requested": kwargs["workers"],
            "workers": 1 if kwargs["dry_run"] else kwargs["workers"],
        }

    def _export_refold_manifest(**_kwargs):
        raise AssertionError("legacy export path should not be used in workflow test")

    def _run_refold(**_kwargs):
        raise AssertionError("legacy run path should not be used in workflow test")

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.refold",
        refold_from_db=_refold_from_db,
        export_refold_manifest=_export_refold_manifest,
        run_refold=_run_refold,
    )

    rc = _run(
        [
            "posthoc",
            "refold",
            "--db",
            "scores.sqlite",
            "--filter-expr",
            "binder_ptm >= 0.7",
            "--dst",
            "refold_work",
            "--prefer-oligo",
            "AB",
            "--run-selection",
            "representative-per-run",
            "--evaluators",
            "af3, protenix",
            "--collect-workers",
            "2",
            "--workers",
            "3",
            "--config-yaml",
            "refold_override.yaml",
            "--dry-run",
            "--format",
            "json",
        ]
    )

    assert rc == 0
    assert captured["filter_expr"] == "binder_ptm >= 0.7"
    assert captured["run_selection"] == "representative-per-run"
    assert captured["evaluators"] == ["af3", "protenix"]
    assert captured["collect_workers"] == 2
    assert captured["workers"] == 3
    assert str(captured["config_yaml"]).endswith("refold_override.yaml")
    assert captured["dry_run"] is True
    out = capsys.readouterr().out
    assert '"selector_kind": "dsl"' in out
    assert '"collect_db_path": "scores.sqlite"' in out
    assert '"resource_groups": 1' in out


def test_cli_posthoc_refold_run_wires_arguments(monkeypatch, capsys):
    captured = {}

    def _export_refold_manifest(**_kwargs):
        raise AssertionError("export_refold_manifest should not be called in run test")

    def _run_refold(**kwargs):
        captured.update(kwargs)
        return {
            "manifest_jsonl": str(kwargs["manifest_jsonl"]),
            "out_root": str(kwargs["out_root"]),
            "evaluators": kwargs["evaluators"],
            "processed": 1,
            "skipped": 2,
            "failed": 0,
            "dry_run": kwargs["dry_run"],
            "ran_af3": False,
            "ran_protenix": False,
            "groups": 3,
            "resource_groups": 2,
            "evaluator_cache_hits": 4,
            "evaluator_cache_misses": 2,
        }

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.refold",
        export_refold_manifest=_export_refold_manifest,
        run_refold=_run_refold,
    )

    rc = _run(
        [
            "posthoc",
            "refold-run",
            "--manifest",
            "manifest.jsonl",
            "--out-root",
            "refold_out",
            "--evaluators",
            "af3, protenix",
            "--path-map",
            "/src=/dst",
            "--workers",
            "4",
            "--config-yaml",
            "refold_override.yaml",
            "--dry-run",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    assert captured["evaluators"] == ["af3", "protenix"]
    assert captured["dry_run"] is True
    assert captured["workers"] == 4
    assert str(captured["config_yaml"]).endswith("refold_override.yaml")
    assert captured["path_map"] == ["/src=/dst"]
    assert '"dry_run": true' in capsys.readouterr().out


def test_cli_posthoc_refold_nested_low_level_argv_rewrites_to_canonical() -> None:
    parser = cli._build_parser()
    assert cli._rewrite_legacy_posthoc_refold_argv(
        ["posthoc", "refold", "export", "--db", "scores.sqlite"], parser
    ) == ["posthoc", "refold-export", "--db", "scores.sqlite"]
    assert cli._rewrite_legacy_posthoc_refold_argv(
        ["posthoc", "refold", "run", "--manifest", "manifest.jsonl"], parser
    ) == ["posthoc", "refold-run", "--manifest", "manifest.jsonl"]


def test_cli_posthoc_refold_run_text_prints_resource_group_summary(monkeypatch, capsys):
    def _export_refold_manifest(**_kwargs):
        raise AssertionError("export_refold_manifest should not be called in run test")

    def _run_refold(**kwargs):
        return {
            "manifest_jsonl": str(kwargs["manifest_jsonl"]),
            "out_root": str(kwargs["out_root"]),
            "evaluators": kwargs["evaluators"],
            "processed": 1,
            "skipped": 0,
            "failed": 0,
            "dry_run": kwargs["dry_run"],
            "ran_af3": 1,
            "ran_protenix": 0,
                "groups": 2,
                "resource_groups": 1,
                "evaluator_cache_hits": 1,
                "evaluator_cache_misses": 1,
                "workers_requested": kwargs["workers"],
                "workers": 1 if kwargs["dry_run"] else kwargs["workers"],
            }

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.refold",
        export_refold_manifest=_export_refold_manifest,
        run_refold=_run_refold,
    )

    rc = _run(
        [
            "posthoc",
            "refold",
            "run",
            "--manifest",
            "manifest.jsonl",
            "--out-root",
            "refold_out",
            "--dry-run",
            "--workers",
            "3",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "groups: output=2 resource=1 evaluator_cache=1/1" in out
    assert "workers: effective=1 requested=3" in out


def test_cli_posthoc_refold_rejects_unknown_evaluator(capsys):
    rc = _run(
        [
            "posthoc",
            "refold",
            "--db",
            "scores.sqlite",
            "--filter-expr",
            "binder_ptm >= 0.7",
            "--evaluators",
            "esmfold",
            "--dry-run",
        ]
    )

    assert rc == 2
    assert "Unsupported refold backend" in capsys.readouterr().err


def test_cli_posthoc_refold_run_rejects_unknown_evaluator(capsys):
    rc = _run(
        [
            "posthoc",
            "refold",
            "run",
            "--manifest",
            "manifest.jsonl",
            "--out-root",
            "refold_out",
            "--evaluators",
            "esmfold",
            "--dry-run",
        ]
    )

    assert rc == 2
    assert "Unsupported refold backend" in capsys.readouterr().err


def test_cli_env_doctor_soft_fail_reports_missing_paths(tmp_path: Path, capsys):
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir(parents=True)
    missing_path = tmp_path / "missing"
    env_path = tmp_path / "env.yaml"
    env_path.write_text(
        f"paths:\n  existing: {existing_dir}\n  missing: {missing_path}\n",
        encoding="utf-8",
    )

    rc = _run(["env", "doctor", "--env", str(env_path), "--soft-fail"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MISSING" in out
    assert "summary:" in out


def test_cli_env_doctor_missing_paths_fail_by_default(tmp_path: Path):
    env_path = tmp_path / "env.yaml"
    env_path.write_text(f"paths:\n  missing: {tmp_path / 'missing'}\n", encoding="utf-8")
    rc = _run(["env", "doctor", "--env", str(env_path)])
    assert rc == 2


def test_cli_env_doctor_uses_config_dir_default_env_local(tmp_path: Path, capsys):
    config_dir = tmp_path / "mycfg"
    env_dir = config_dir / "env"
    env_dir.mkdir(parents=True)
    env_path = env_dir / "env_local.yaml"
    env_path.write_text("paths: {}\n", encoding="utf-8")

    rc = _run(["--config-dir", str(config_dir), "env", "doctor"])

    assert rc == 0
    assert f"env: {env_path}" in capsys.readouterr().out


def test_cli_env_wizard_uses_config_dir_default_env_local(tmp_path: Path, monkeypatch):
    import astrolabe.cli_interactive as cli_interactive

    config_dir = tmp_path / "mycfg"
    env_dir = config_dir / "env"
    env_dir.mkdir(parents=True)
    env_path = env_dir / "env_local.yaml"
    env_path.write_text("# @package _global_\n\nsubprocess_envs: {}\n", encoding="utf-8")
    (config_dir / "demo.yaml").write_text("task: demo\n", encoding="utf-8")

    captured = {}

    def _fake_doctor(ns):
        captured["env"] = ns.env
        captured["config_dir"] = ns.config_dir
        return 0

    monkeypatch.setattr(cli_interactive, "_require_interactive_tty", lambda _cmd: True)
    monkeypatch.setattr(cli_interactive, "_cmd_doctor", _fake_doctor)
    monkeypatch.setattr(
        cli_interactive,
        "_write_astrolabe_python_hint_for_config_dir",
        lambda *_args, **_kwargs: None,
    )

    rc = cli_interactive._cmd_env_wizard(
        types.SimpleNamespace(
            config_dir=str(config_dir),
            task="demo",
            env=None,
            out=None,
            force=False,
            show_ok=False,
            soft_fail=False,
        )
    )

    assert rc == 0
    assert captured["config_dir"] == str(config_dir)
    assert captured["env"] == str(env_path)


def test_cli_posthoc_filter_wires_filter_expression(monkeypatch, capsys):
    captured = {}

    class MissingRequiredMetricsError(RuntimeError):
        pass

    def _filter_structures(**kwargs):
        captured.update(kwargs)
        return {
            "db_path": str(kwargs["db_path"]),
            "selector_kind": "dsl",
            "selector": kwargs["filter_expr"],
            "preset": "",
            "filter_expr": kwargs["filter_expr"],
            "prefer_oligo": kwargs["prefer_oligo"],
            "dst_dir": str(kwargs["dst_dir"]),
            "selected": 1,
            "records_considered": 2,
            "report_csv": str(kwargs["report_csv"]) if kwargs["report_csv"] else "",
        }

    def _preview_preset(_path: Path):
        raise AssertionError("preview_preset should not be called in filter-expression test")

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.filtering",
        MissingRequiredMetricsError=MissingRequiredMetricsError,
        filter_structures=_filter_structures,
        preview_preset=_preview_preset,
    )

    rc = _run(
        [
            "posthoc",
            "filter",
            "--db",
            "scores.sqlite",
            "--filter-expr",
            "weighted_score > 0 and AB.iptm >= 0.7",
            "--dst",
            "filtered",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    assert captured["filter_expr"] == "weighted_score > 0 and AB.iptm >= 0.7"
    assert captured["preset_path"] is None
    assert captured["run_selection"] == "all-passing"
    assert '"selector_kind": "dsl"' in capsys.readouterr().out


def test_cli_posthoc_filter_preview_preset_without_db_or_dst(monkeypatch, capsys):
    captured = {}

    class MissingRequiredMetricsError(RuntimeError):
        pass

    def _filter_structures(**_kwargs):
        raise AssertionError("filter_structures should not be called in preset preview test")

    def _preview_preset(path: Path):
        captured["preset_path"] = path
        return {
            "selector_kind": "preset",
            "selector": "demo_preset",
            "preset": "demo_preset",
            "rule_count": 2,
            "required_metrics": ["weighted_score"],
            "preview_filter_expr": "weighted_score > 0 and (missing(AB.iptm) or AB.iptm >= 0.7)",
            "rules": [
                {"key": "weighted_score", "op": "gt", "value": 0.0, "allow_missing": False},
                {"key": "AB.iptm", "op": "ge", "value": 0.7, "allow_missing": True},
            ],
        }

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.filtering",
        MissingRequiredMetricsError=MissingRequiredMetricsError,
        filter_structures=_filter_structures,
        preview_preset=_preview_preset,
    )

    rc = _run(
        [
            "posthoc",
            "filter",
            "--preset",
            "preset.yaml",
            "--preview-preset",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    assert str(captured["preset_path"]).endswith("preset.yaml")
    assert '"preview_filter_expr": "weighted_score > 0 and (missing(AB.iptm) or AB.iptm >= 0.7)"' in capsys.readouterr().out


def test_cli_posthoc_filter_preview_preset_without_preset_returns_2(capsys):
    rc = _run(
        [
            "posthoc",
            "filter",
            "--preview-preset",
        ]
    )

    assert rc == 2
    assert "--preview-preset requires --preset" in capsys.readouterr().err


def test_cli_posthoc_filter_auto_dst_when_missing(monkeypatch, capsys):
    captured = {}

    class MissingRequiredMetricsError(RuntimeError):
        pass

    def _filter_structures(**kwargs):
        captured.update(kwargs)
        return {
            "db_path": str(kwargs["db_path"]),
            "selector_kind": "dsl",
            "filter_expr": kwargs["filter_expr"],
            "prefer_oligo": kwargs["prefer_oligo"],
            "dst_dir": str(kwargs["dst_dir"]),
            "selected": 1,
            "records_considered": 1,
            "report_csv": str(kwargs["report_csv"]) if kwargs["report_csv"] else "",
        }

    _install_fake_module(
        monkeypatch,
        "astrolabe.posthoc.filtering",
        MissingRequiredMetricsError=MissingRequiredMetricsError,
        filter_structures=_filter_structures,
    )

    rc = _run(
        [
            "posthoc",
            "filter",
            "--db",
            "scores.sqlite",
            "--expr",
            "iptm > 0.8",
        ]
    )

    assert rc == 0
    assert str(captured["dst_dir"]).endswith("filter__iptm_gt_0_8")
    assert "dst: " in capsys.readouterr().out


def test_cli_design_ligand_binder_accepts_short_flags_num_runs_and_seed(capsys):
    rc = _run(
        [
            "design",
            "ligand-binder",
            "-t",
            "t_lig",
            "-o",
            "outputs",
            "--protein-len",
            "50",
            "--ligand",
            "PLP",
            "--num-runs",
            "3",
            "--seed",
            "17",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.task_name == "t_lig"
    assert cfg.out_dir == "outputs"
    assert cfg.repeats == 3
    assert cfg.seed == 17


def test_cli_design_idr_binder_dry_run_uses_idr_template(tmp_path: Path, capsys):
    config_dir = _config_dir_with_env(
        tmp_path,
        "benchmark_binder_v1_4",
        "benchmark_binder_idr_v1",
    )
    rc = _run(
        [
            "--config-dir",
            str(config_dir),
            "design",
            "idr-binder",
            "-t",
            "t_idr",
            "-o",
            "outputs",
            "--binder-len",
            "50",
            "--target-seq",
            "MSEQUENCE",
            "--num-runs",
            "2",
            "--seed",
            "11",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.task_name == "t_idr"
    assert cfg.out_dir == "outputs"
    assert cfg.repeats == 2
    assert cfg.seed == 11
    assert cfg.seqs.A == "50-50"
    assert cfg.seqs.B == "+MSEQUENCE"
    assert "target_non_loop_frac" in cfg.loss.expression_terms


def test_cli_bench_idr_binder_alias_uses_same_template(tmp_path: Path, capsys):
    config_dir = _config_dir_with_env(
        tmp_path,
        "benchmark_binder_v1_4",
        "benchmark_binder_idr_v1",
    )
    rc = _run(
        [
            "--config-dir",
            str(config_dir),
            "bench-idr-binder-v1",
            "-t",
            "t_idr_bench",
            "-o",
            "outputs",
            "--initial-binder-seq",
            "ACDEF",
            "--target-seq",
            "MSEQUENCE",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.task_name == "t_idr_bench"
    assert cfg.seqs.A == "ACDEF"
    assert cfg.seqs.B == "+MSEQUENCE"
    assert cfg.loss.loss_terms.target_loop_frac._type == "secondary_structure_fraction"


def test_cli_design_cid_generate_dry_run_fills_three_protomers(tmp_path: Path, capsys):
    config_dir = _config_dir_with_env(tmp_path, "benchmark_cid_generate_v1")
    rc = _run(
        [
            "--config-dir",
            str(config_dir),
            "design",
            "cid-generate",
            "-t",
            "t_cid",
            "-o",
            "outputs",
            "--protein-a-len",
            "80",
            "--protein-b-seq",
            "acdef",
            "--ligand",
            "ATP",
            "--ligand-type",
            "ccd",
            "--ligand-resname",
            "ATP",
            "--num-runs",
            "2",
            "--seed",
            "19",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = _yaml_from_output(capsys.readouterr().out)
    assert cfg.task_name == "t_cid"
    assert cfg.out_dir == "outputs"
    assert cfg.repeats == 2
    assert cfg.seed == 19
    assert cfg.seqs.A == "80-80"
    assert cfg.seqs.B == "ACDEF"
    assert cfg.seqs.C == "ATP"
    assert cfg.types.C == "c"
    assert cfg.ligand_resnames.C == "ATP"
    assert cfg.predocker.predocker_terms.dimer_interface._type == "dimer_interface_placer"
    assert cfg.predocker.predocker_terms.autodock_vina_box._type == "autodock_vina"
    assert cfg.mutator.mutator_terms.LMPNN_dimer_vina.predocker == [
        "dimer_interface",
        "autodock_vina_box",
    ]


def test_cli_design_cid_alias_accepts_protomer_expressions(tmp_path: Path, capsys):
    config_dir = _config_dir_with_env(tmp_path, "benchmark_cid_generate_v1")
    rc = _run(
        [
            "--config-dir",
            str(config_dir),
            "design",
            "cid",
            "-t",
            "t_cid_expr",
            "-o",
            "outputs",
            "--protein-a-expr",
            "+ACD 40-60",
            "--protein-b-expr",
            "+EFG 40-60",
            "--ligand",
            "C1CCOCC1",
            "--ligand-type",
            "smiles",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = _yaml_from_output(capsys.readouterr().out)
    assert cfg.seqs.A == "+ACD 40-60"
    assert cfg.seqs.B == "+EFG 40-60"
    assert cfg.types.C == "s"


def test_cli_design_cid_rejects_multiple_protein_a_sources():
    parser = cli._build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "design",
                "cid",
                "-t",
                "t_cid",
                "-o",
                "outputs",
                "--protein-a-len",
                "80",
                "--protein-a-seq",
                "ACDEF",
                "--protein-b-len",
                "80",
                "--ligand",
                "ATP",
                "--dry-run",
            ]
        )

    assert excinfo.value.code == 2


def test_cli_design_aptamer_type_dna_accepts_dna_sequence(tmp_path: Path, capsys):
    config_dir = _config_dir_with_env(tmp_path, "benchmark_aptamer_v1")
    rc = _run(
        [
            "--config-dir",
            str(config_dir),
            "design",
            "aptamer",
            "-t",
            "t_dna_apt",
            "-o",
            "outputs",
            "--aptamer-type",
            "dna",
            "--initial-binder-seq",
            "atgc",
            "--target-seq",
            "ACDEFGHIK",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.types.A == "d"
    assert cfg.seqs.A == "ATGC"


def test_cli_design_dna_aptamer_alias_sets_dna_type(tmp_path: Path, capsys):
    config_dir = _config_dir_with_env(tmp_path, "benchmark_aptamer_v1")
    rc = _run(
        [
            "--config-dir",
            str(config_dir),
            "design",
            "dna-aptamer",
            "-t",
            "t_dna_apt",
            "-o",
            "outputs",
            "--aptamer-len",
            "31",
            "--target-seq",
            "ACDEFGHIK",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.types.A == "d"
    assert cfg.seqs.A == "31-31"


def test_cli_design_multidomain_binder_accepts_short_flags_num_runs_and_seed(
    tmp_path: Path,
    capsys,
):
    json_path = tmp_path / "domain_templates.json"
    json_path.write_text(
        """
{
  "name": "domain_templates",
  "dialect": "alphafold3",
  "version": 1,
  "modelSeeds": [1],
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "ACDEFGHIKL",
        "templates": [
          {
            "mmcif": "data_domain",
            "queryIndices": [0, 1],
            "templateIndices": [10, 11]
          }
        ]
      }
    }
  ]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rc = _run(
        [
            "design",
            "multidomain-binder",
            "-t",
            "t_md",
            "-o",
            "outputs",
            "--variant",
            "af3",
            "--target-name",
            "cre_like",
            "--binder-len",
            "50",
            "--target-homology",
            f"{json_path}:A",
            "--domain-indices",
            "1-2",
            "--domain-indices",
            "8-9",
            "--num-runs",
            "3",
            "--seed",
            "17",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.task_name == "t_md"
    assert cfg.out_dir == "outputs"
    assert cfg.repeats == 3
    assert cfg.seed == 17


def test_cli_design_ligand_binder_accepts_initial_binder_seq(capsys):
    rc = _run(
        [
            "design",
            "ligand-binder",
            "-t",
            "t_lig_seq",
            "-o",
            "outputs",
            "--initial-binder-seq",
            "acdef",
            "--ligand",
            "PLP",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.seqs.A == "ACDEF"
    assert cfg.seqs.B == "PLP"


def test_cli_design_ligand_binder_accepts_initial_binder_expr(capsys):
    rc = _run(
        [
            "design",
            "ligand-binder",
            "-t",
            "t_lig_expr",
            "-o",
            "outputs",
            "--initial-binder-expr",
            "+ACD 140-160",
            "--ligand",
            "PLP",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.seqs.A == "+ACD 140-160"


def test_cli_design_binder_accepts_initial_binder_seq(capsys):
    rc = _run(
        [
            "design",
            "binder",
            "-t",
            "t_binder",
            "-o",
            "outputs",
            "--initial-binder-seq",
            "acdefx",
            "--target-seq",
            "MSEQUENCE",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.seqs.A == "ACDEFX"
    assert cfg.seqs.B == "+MSEQUENCE"


def test_cli_design_binder_accepts_initial_binder_expr(capsys):
    rc = _run(
        [
            "design",
            "protein-binder",
            "-t",
            "t_binder",
            "-o",
            "outputs",
            "--initial-binder-expr",
            "+ACD 20-30",
            "--target-seq",
            "MSEQUENCE",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.seqs.A == "+ACD 20-30"


def test_cli_design_binder_rejects_multiple_initial_binder_sources():
    parser = cli._build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "design",
                "binder",
                "-t",
                "t_binder",
                "-o",
                "outputs",
                "--binder-len",
                "80-120",
                "--initial-binder-seq",
                "ACDEF",
                "--target-seq",
                "MSEQUENCE",
                "--dry-run",
            ]
        )

    assert excinfo.value.code == 2


def test_cli_refine_ligand_pocket_accepts_short_flags_num_runs_seed_and_steps(
    tmp_path: Path, capsys
):
    structure = tmp_path / "complex.pdb"
    structure.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  ALA A   2       2.000   0.000   0.000  1.00 20.00           C",
                "ATOM      3  CA  ALA A   3       3.000   0.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = _run(
        [
            "refine",
            "ligand-pocket",
            "-t",
            "t_lig_pocket",
            "-o",
            "outputs",
            "--structure",
            str(structure),
            "--protein-chain",
            "A",
            "--ligand",
            "PLP",
            "--num-runs",
            "2",
            "--seed",
            "23",
            "--steps",
            "10",
            "--mutation-num",
            "4",
            "--contact-distance",
            "4.5",
            "--dry-run",
        ]
    )

    assert rc == 0
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.task_name == "t_lig_pocket"
    assert cfg.out_dir == "outputs"
    assert cfg.repeats == 2
    assert cfg.seed == 23
    assert cfg.steps == 10
    assert cfg.mutation_num == 4
    assert cfg.stages.default.mutation_num == 4
    assert (
        cfg.mutator.mutator_terms.LMPNN_allala_vina.position_selector
        == "ligand_contact(distance_max=4.5)"
    )


def test_parse_reference_pocket_positions_normalizes_ranges_and_deduplicates() -> None:
    assert cli._parse_reference_pocket_positions("7,3-5,4 10 7") == [7, 3, 4, 5, 10]


@pytest.mark.parametrize("raw", ["", "0", "5-3", "2,a"])
def test_parse_reference_pocket_positions_rejects_invalid_specs(raw: str) -> None:
    with pytest.raises(ValueError):
        cli._parse_reference_pocket_positions(raw)


def test_cli_config_compose_without_tty_returns_2(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    rc = _run(["config", "compose"])

    assert rc == 2
    assert "astrolabe config compose requires an interactive TTY." in capsys.readouterr().err


def test_cli_config_compose_help_mentions_write_modes(capsys):
    parser = cli._build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["config", "compose", "--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert ":write" in out
    assert ":write-all" in out
    assert ":write-overlay" in out


def test_cli_env_doctor_oc_env_guidance_mentions_env_var_and_dotenv(
    tmp_path: Path, capsys
):
    env_path = tmp_path / "env.yaml"
    env_path.write_text(
        "rhofold:\n  binary_dpath: ${oc.env:GH_RHOFOLD_BIN,???}\n",
        encoding="utf-8",
    )

    rc = _run(["env", "doctor", "--env", str(env_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "GH_RHOFOLD_BIN" in out
    assert ".env" in out
