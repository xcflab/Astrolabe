#!/usr/bin/env python3
"""Exercise the complete introspected ``gh`` command surface inside a SIF."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


CONTAINER_REPO_ROOT = Path("/opt/Astrolabe")


def _detect_engine() -> str | None:
    for name in ("apptainer", "singularity"):
        executable = shutil.which(name)
        if executable:
            return executable
    return None


def _subparser_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _canonical_names(
    action: argparse._SubParsersAction,
) -> tuple[dict[int, str], list[tuple[str, argparse.ArgumentParser]]]:
    by_id: dict[int, str] = {}
    ordered: list[tuple[str, argparse.ArgumentParser]] = []
    for choice_action in getattr(action, "_choices_actions", ()):
        name = str(getattr(choice_action, "dest", "") or "")
        child = action.choices.get(name)
        if not name or child is None or id(child) in by_id:
            continue
        by_id[id(child)] = name
        ordered.append((name, child))
    for name, child in action.choices.items():
        if id(child) in by_id:
            continue
        by_id[id(child)] = str(name)
        ordered.append((str(name), child))
    return by_id, ordered


def _option_metadata(parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for action in parser._actions:
        option_strings = list(getattr(action, "option_strings", ()) or ())
        if not option_strings:
            continue
        choices = getattr(action, "completion_choices", None)
        if choices is None:
            choices = getattr(action, "choices", None)
        options.append(
            {
                "dest": str(action.dest),
                "options": option_strings,
                "choices": [str(value) for value in choices] if choices else [],
            }
        )
    return options


def _command_manifest() -> dict[str, Any]:
    from astrolabe import cli

    root = cli._build_parser()
    nodes: list[dict[str, Any]] = []
    aliases: list[dict[str, list[str]]] = []

    def visit(parser: argparse.ArgumentParser, path: tuple[str, ...]) -> None:
        subparsers = _subparser_action(parser)
        is_leaf = subparsers is None
        nodes.append(
            {
                "path": list(path),
                "leaf": is_leaf,
                "handler": callable(parser.get_default("_handler")),
                "options": _option_metadata(parser),
            }
        )
        if subparsers is None:
            return
        canonical_by_id, canonical_children = _canonical_names(subparsers)
        for name, child in subparsers.choices.items():
            canonical = canonical_by_id[id(child)]
            if str(name) != canonical:
                aliases.append(
                    {
                        "path": [*path, str(name)],
                        "canonical": [*path, canonical],
                    }
                )
        for canonical, child in canonical_children:
            visit(child, (*path, canonical))

    visit(root, ())
    legacy = [
        {"path": list(source), "canonical": list(target)}
        for source, target in cli.LEGACY_COMMAND_REWRITES
    ]
    return {"nodes": nodes, "aliases": aliases, "legacy_aliases": legacy}


def _completion_errors(manifest: dict[str, Any]) -> list[str]:
    from astrolabe.shell_completion import _collect_static_completion_spec

    spec = _collect_static_completion_spec()
    errors: list[str] = []
    expected_contexts = {
        "/".join(node["path"]) if node["path"] else "__root__"
        for node in manifest["nodes"]
    }
    actual_contexts = set(spec["subcommands"])
    if expected_contexts != actual_contexts:
        missing = sorted(expected_contexts - actual_contexts)
        extra = sorted(actual_contexts - expected_contexts)
        errors.append(f"completion contexts differ: missing={missing}, extra={extra}")

    for node in manifest["nodes"]:
        context = "/".join(node["path"]) if node["path"] else "__root__"
        for option in node["options"]:
            if not option["choices"]:
                continue
            for option_string in option["options"]:
                key = f"{context}:{option_string}"
                values = {
                    line.split("\t", 1)[0]
                    for line in str(spec["static_values"].get(key, "")).splitlines()
                }
                missing_values = sorted(set(option["choices"]) - values)
                if missing_values:
                    errors.append(f"completion choices missing for {key}: {missing_values}")
    return errors


def _invoke(command: list[str], *, retain_stdout: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    result = {
        "command": command,
        "returncode": proc.returncode,
        "seconds": round(time.monotonic() - started, 3),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }
    if retain_stdout:
        result["stdout"] = proc.stdout
    return result


def _inside() -> tuple[dict[str, Any], int]:
    manifest = _command_manifest()
    failures = _completion_errors(manifest)
    gh = shutil.which("astrolabe")
    if gh is None:
        raise RuntimeError("gh is not installed in the container PATH")

    help_paths = [node["path"] for node in manifest["nodes"]]
    help_paths.extend(alias["path"] for alias in manifest["aliases"])
    help_paths.extend(alias["path"] for alias in manifest["legacy_aliases"])
    results = [_invoke([gh, *path, "--help"]) for path in help_paths]
    for result in results:
        if result["returncode"] != 0:
            failures.append(
                f"help failed ({result['returncode']}): {' '.join(result['command'])}"
            )

    leaf_nodes = [node for node in manifest["nodes"] if node["leaf"]]
    handlerless = [" ".join(node["path"]) for node in leaf_nodes if not node["handler"]]
    if handlerless:
        failures.append(f"leaf commands without handlers: {handlerless}")

    completion_results = [
        _invoke([gh, "completion", shell, "--init"], retain_stdout=True)
        for shell in ("bash", "zsh")
    ]
    completion_results.extend(
        [
            _invoke(
                [
                    gh,
                    "completion",
                    "complete",
                    "--shell",
                    shell,
                    "--cword",
                    "1",
                    "astrolabe",
                    "",
                ]
            )
            for shell in ("bash", "zsh")
        ]
    )
    for result in completion_results:
        if result["returncode"] != 0 or not result["stdout_tail"].strip():
            failures.append(f"completion execution failed: {' '.join(result['command'])}")

    bash_script = completion_results[0].pop("stdout")
    completion_results[1].pop("stdout")
    with tempfile.NamedTemporaryFile("w", suffix=".bash", delete=False) as handle:
        handle.write(bash_script)
        bash_path = handle.name
    try:
        bash_check = _invoke(["bash", "-n", bash_path])
    finally:
        Path(bash_path).unlink()
    if bash_check["returncode"] != 0:
        failures.append("generated Bash completion script failed bash -n")

    report = {
        "schema_version": 1,
        "canonical_nodes": len(manifest["nodes"]),
        "canonical_leaves": len(leaf_nodes),
        "parser_aliases": len(manifest["aliases"]),
        "legacy_aliases": len(manifest["legacy_aliases"]),
        "fixed_choice_values": sum(
            len(option["choices"])
            for node in manifest["nodes"]
            for option in node["options"]
        ),
        "manifest": manifest,
        "help_results": results,
        "completion_results": completion_results,
        "bash_syntax_result": bash_check,
        "failures": failures,
    }
    return report, int(bool(failures))


def _outer(args: argparse.Namespace) -> int:
    engine = args.engine or _detect_engine()
    if engine is None:
        raise SystemExit("Could not find `apptainer` or `singularity` in PATH.")
    engine = str(Path(engine).resolve())
    sif = Path(os.path.expandvars(os.path.expanduser(args.sif))).resolve()
    if not sif.is_file():
        raise SystemExit(f"SIF not found: {sif}")
    repo_root = Path(__file__).resolve().parents[2]
    command = [
        engine,
        "exec",
        "-B",
        f"{repo_root}:{CONTAINER_REPO_ROOT}",
        str(sif),
        "python",
        str(CONTAINER_REPO_ROOT / "scripts" / "validation" / Path(__file__).name),
        "--inside",
    ]
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(Path(engine).parent), env.get("PATH", "")])
    proc = subprocess.run(command, text=True, capture_output=True, env=env, check=False)
    if not proc.stdout.strip():
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"Container command smoke produced no report (exit {proc.returncode}).")
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit("Container command smoke produced invalid JSON.")
    report["container_command"] = command
    report["container_stderr"] = proc.stderr
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "commands: "
        f"leaves={report['canonical_leaves']} "
        f"parser_aliases={report['parser_aliases']} "
        f"legacy_aliases={report['legacy_aliases']}"
    )
    print(f"report: {report_path}")
    for failure in report["failures"]:
        print(f"FAIL: {failure}", file=sys.stderr)
    return int(proc.returncode != 0 or bool(report["failures"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sif",
        default=str(Path.home() / ".gh_envs" / "images" / "Astrolabe_af3.sif"),
    )
    parser.add_argument("--engine", default=None)
    parser.add_argument("--report", default="logs/gh_sif_cli_smoketest.json")
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.inside:
        report, exit_code = _inside()
        print(json.dumps(report, sort_keys=True))
        return exit_code
    return _outer(args)


if __name__ == "__main__":
    raise SystemExit(main())
