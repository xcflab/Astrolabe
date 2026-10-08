from __future__ import annotations

import argparse
import ctypes
from types import SimpleNamespace

import pytest

from astrolabe import cli


def _run_main(argv: list[str] | None) -> int:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    return int(excinfo.value.code)


def test_successful_gpu_path_parses_then_initializes_then_enters_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_parse_args = argparse.ArgumentParser.parse_args

    def _parse_args(parser, *args, **kwargs):
        ns = original_parse_args(parser, *args, **kwargs)
        if parser.prog == "astrolabe":
            events.append("parse")
        return ns

    def _cu_init(flags: int) -> int:
        assert flags == 0
        events.append("cuInit")
        return 0

    def _handler(_ns: argparse.Namespace) -> int:
        events.append("handler/model import")
        return 0

    monkeypatch.setenv("GH_CUDA_PREINITIALIZE", "1")
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _parse_args)
    monkeypatch.setattr(
        ctypes,
        "CDLL",
        lambda _name: SimpleNamespace(cuInit=_cu_init),
    )
    monkeypatch.setattr(cli, "_cmd_run", _handler)

    assert _run_main(["run", "--task", "demo"]) == 0
    assert events == ["parse", "cuInit", "handler/model import"]


@pytest.mark.parametrize(
    ("handler_name", "argv"),
    [
        ("_cmd_completion_bash", ["completion", "bash"]),
        ("_cmd_compose", ["config", "compose"]),
        ("_cmd_doctor", ["env", "doctor"]),
        (
            "_cmd_experimental_candidate_batch",
            ["experimental", "candidate-batch", "--config-yaml", "c", "--out-root", "o"],
        ),
        (
            "_cmd_collect_scores",
            ["posthoc", "collect", "--out-root", "o", "--db", "x"],
        ),
    ],
)
def test_successful_cpu_groups_do_not_initialize_cuda(
    monkeypatch: pytest.MonkeyPatch,
    handler_name: str,
    argv: list[str],
) -> None:
    monkeypatch.setenv("GH_CUDA_PREINITIALIZE", "1")
    monkeypatch.setattr(
        ctypes,
        "CDLL",
        lambda _name: pytest.fail("CPU command initialized CUDA"),
    )
    monkeypatch.setattr(cli, handler_name, lambda _ns: 0)

    assert _run_main(argv) == 0


@pytest.mark.parametrize(
    ("argv", "exit_code"),
    [
        (["run", "--help"], 0),
        (["not-a-command"], 2),
        (["posthoc"], 2),
        (["posthoc", "refold"], 2),
    ],
)
def test_help_invalid_and_missing_arguments_exit_before_cuda(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    exit_code: int,
) -> None:
    monkeypatch.setenv("GH_CUDA_PREINITIALIZE", "1")
    monkeypatch.setattr(
        ctypes,
        "CDLL",
        lambda _name: pytest.fail("parse exit initialized CUDA"),
    )

    assert _run_main(argv) == exit_code
    capsys.readouterr()


def test_parsed_dry_run_suppresses_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GH_CUDA_PREINITIALIZE", "1")
    monkeypatch.setattr(
        ctypes,
        "CDLL",
        lambda _name: pytest.fail("dry-run initialized CUDA"),
    )
    monkeypatch.setattr(cli, "_cmd_run", lambda _ns: 0)

    assert _run_main(["run", "--task", "demo", "--dry-run"]) == 0


@pytest.mark.parametrize("env_value", [None, "0"])
def test_cuda_preinitialize_requires_explicit_runner_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
) -> None:
    if env_value is None:
        monkeypatch.delenv("GH_CUDA_PREINITIALIZE", raising=False)
    else:
        monkeypatch.setenv("GH_CUDA_PREINITIALIZE", env_value)
    monkeypatch.setattr(
        ctypes,
        "CDLL",
        lambda _name: pytest.fail("CUDA initialized without opt-in"),
    )
    monkeypatch.setattr(cli, "_cmd_run", lambda _ns: 0)

    assert _run_main(["run", "--task", "demo"]) == 0


_ONE_SHOT = ["--task-name", "t", "--out-dir", "o"]
_REFOLD_EXPORT = ["--db", "x", "--filter-expr", "x", "--out-manifest", "m"]
_REFOLD_RUN = ["--manifest", "m", "--out-root", "o"]


@pytest.mark.parametrize(
    ("canonical", "alias", "args"),
    [
        (("design", "protein-binder"), ("design", "binder"), [*_ONE_SHOT, "--binder-len", "10"]),
        (("design", "peptide"), ("design", "peptide-binder"), [*_ONE_SHOT, "--peptide-len", "10"]),
        (("design", "ligand-binder"), ("design", "ligand"), [*_ONE_SHOT, "--protein-len", "10", "--ligand", "FAD"]),
        (("design", "cid-generate"), ("design", "cid"), [*_ONE_SHOT, "--protein-a-len", "10", "--protein-b-len", "10", "--ligand", "FAD"]),
        (("posthoc", "refold-export"), ("posthoc", "refold-prepare"), _REFOLD_EXPORT),
        (("posthoc", "refold-run"), ("posthoc", "refold-predict"), _REFOLD_RUN),
    ],
)
def test_native_aliases_inherit_canonical_cuda_semantics(
    canonical: tuple[str, ...],
    alias: tuple[str, ...],
    args: list[str],
) -> None:
    parser = cli._build_parser()
    canonical_ns = parser.parse_args([*canonical, *args])
    alias_ns = parser.parse_args([*alias, *args])

    assert alias_ns._cuda_preinitialize is canonical_ns._cuda_preinitialize
    assert alias_ns._handler is canonical_ns._handler


@pytest.mark.parametrize(
    ("canonical", "legacy", "args"),
    [
        ("refold-export", "export", _REFOLD_EXPORT),
        ("refold-export", "prepare", _REFOLD_EXPORT),
        ("refold-run", "run", _REFOLD_RUN),
        ("refold-run", "predict", _REFOLD_RUN),
    ],
)
def test_legacy_refold_rewrites_inherit_canonical_cuda_semantics(
    canonical: str,
    legacy: str,
    args: list[str],
) -> None:
    parser = cli._build_parser()
    canonical_ns = parser.parse_args(["posthoc", canonical, *args])
    legacy_argv = cli._rewrite_legacy_posthoc_refold_argv(
        ["posthoc", "refold", legacy, *args], parser
    )
    legacy_ns = parser.parse_args(legacy_argv)

    assert legacy_ns._cuda_preinitialize is canonical_ns._cuda_preinitialize
    assert legacy_ns._handler is canonical_ns._handler


@pytest.mark.parametrize(
    "prefix",
    [
        ["--config-dir", "posthoc"],
        ["--config-dir=posthoc"],
        ["--autoselect-gpu"],
        ["--autoselect-gpu", "--config-dir", "posthoc"],
        ["--config-dir=posthoc", "--autoselect-gpu"],
    ],
)
@pytest.mark.parametrize("use_sys_argv", [False, True])
@pytest.mark.parametrize(
    ("canonical", "legacy", "args", "handler_name"),
    [
        ("refold-export", "export", _REFOLD_EXPORT, "_cmd_refold_export"),
        ("refold-export", "prepare", _REFOLD_EXPORT, "_cmd_refold_export"),
        ("refold-run", "run", _REFOLD_RUN, "_cmd_refold_run"),
        ("refold-run", "predict", _REFOLD_RUN, "_cmd_refold_run"),
    ],
)
def test_legacy_refold_rewrites_after_root_options_match_canonical_main(
    monkeypatch: pytest.MonkeyPatch,
    prefix: list[str],
    use_sys_argv: bool,
    canonical: str,
    legacy: str,
    args: list[str],
    handler_name: str,
) -> None:
    captured: list[argparse.Namespace] = []
    monkeypatch.setenv("GH_CUDA_PREINITIALIZE", "0")
    monkeypatch.setattr(cli, handler_name, lambda ns: captured.append(ns) or 0)

    for route in (["posthoc", canonical], ["posthoc", "refold", legacy]):
        argv = [*prefix, *route, *args]
        if use_sys_argv:
            monkeypatch.setattr(cli.sys, "argv", ["astrolabe", *argv])
            assert _run_main(None) == 0
        else:
            assert _run_main(argv) == 0

    assert vars(captured[0]) == vars(captured[1])


def test_legacy_rewrite_does_not_move_root_option_after_leaf() -> None:
    parser = cli._build_parser()
    argv = [
        "posthoc",
        "refold",
        "export",
        *_REFOLD_EXPORT,
        "--autoselect-gpu",
    ]

    with pytest.raises(SystemExit):
        parser.parse_args(cli._rewrite_legacy_posthoc_refold_argv(argv, parser))


@pytest.mark.parametrize(
    "prefix",
    [
        ["--unknown-root-option"],
        ["--autoselect-gpu=true"],
    ],
)
def test_legacy_rewrite_leaves_unknown_or_unsupported_root_options(
    prefix: list[str],
) -> None:
    parser = cli._build_parser()
    argv = [*prefix, "posthoc", "refold", "export", *_REFOLD_EXPORT]

    assert cli._rewrite_legacy_posthoc_refold_argv(argv, parser) == argv
