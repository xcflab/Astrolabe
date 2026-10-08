from __future__ import annotations

import argparse
import pathlib
import shlex
import sys
from typing import Any, List

from omegaconf import OmegaConf

import astrolabe.cli as _core
from astrolabe.runtime_requirements import task_requirements as _task_requirements

_default_config_dir = _core._default_config_dir
_ensure_sys_path_has_repo_root = _core._ensure_sys_path_has_repo_root
_split_overrides_string = _core._split_overrides_string
_cmd_run = _core._cmd_run
_list_available_tasks = _core._list_available_tasks
_prompt_choice_from_list = _core._prompt_choice_from_list
_sanitize_name = _core._sanitize_name
_compose_write_yaml = _core._compose_write_yaml
_compose_get_section_status = _core._compose_get_section_status
_prompt_words = _core._prompt_words
_compose_validate_basic = _core._compose_validate_basic
_compose_section_basic = _core._compose_section_basic
_compose_section_evaluator = _core._compose_section_evaluator
_compose_section_mutator = _core._compose_section_mutator
_compose_section_loss = _core._compose_section_loss
_compose_section_predocker = _core._compose_section_predocker
_compose_section_stages = _core._compose_section_stages
_scan_missing_keys_for_scaffold = _core._scan_missing_keys_for_scaffold
_get_by_dotted_key = _core._get_by_dotted_key
_set_by_dotted_key = _core._set_by_dotted_key
_interactive_scaffold_prompt = _core._interactive_scaffold_prompt
_strip_defaults_key = _core._strip_defaults_key
_validate_path_value = _core._validate_path_value
_extract_oc_env_name = _core._extract_oc_env_name
_interactive_env_prompt = _core._interactive_env_prompt
_cmd_doctor = _core._cmd_doctor
_write_astrolabe_python_hint = _core.write_astrolabe_python_hint
_write_astrolabe_python_hint_for_config_dir = _core.write_astrolabe_python_hint_for_config_dir
_default_local_env_path_for_config_dir = _core.default_local_env_path_for_config_dir


def _require_interactive_tty(command_name: str) -> bool:
    if sys.stdin.isatty():
        return True
    print(
        f"{command_name} requires an interactive TTY.",
        file=sys.stderr,
    )
    return False


def _cmd_wizard(ns: argparse.Namespace) -> int:
    if not _require_interactive_tty("astrolabe config wizard"):
        return 2

    config_dir = pathlib.Path(ns.config_dir).resolve() if ns.config_dir else _default_config_dir()
    _ensure_sys_path_has_repo_root(config_dir)

    config_file_raw = ns.standalone
    if not config_file_raw:
        config_file_raw = input("config file path: ").strip()
    if not config_file_raw:
        raise ValueError("Missing config file path.")

    config_file_path = pathlib.Path(config_file_raw).expanduser().resolve()
    if not config_file_path.exists():
        raise FileNotFoundError(f"config file not found: {config_file_path}")

    external_cfg = OmegaConf.load(str(config_file_path))
    external_raw = OmegaConf.to_container(external_cfg, resolve=False)
    missing_keys: List[str] = []

    def _scan_raw(node: Any, prefix: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                _scan_raw(value, next_prefix)
            return
        if isinstance(node, list):
            for idx, value in enumerate(node):
                _scan_raw(value, f"{prefix}[{idx}]")
            return
        if isinstance(node, str) and node.strip() == "???" and prefix:
            missing_keys.append(prefix)

    _scan_raw(external_raw)
    missing_keys = sorted(set([k for k in missing_keys if k]))

    overrides: List[str] = []
    if missing_keys:
        print("Missing keys:")
        for key in missing_keys:
            print(f"- {key}")
        for key in missing_keys:
            value = input(f"{key} = ").strip()
            if value:
                overrides.append(f"{key}={value}")

    if ns.overrides:
        overrides.extend(_split_overrides_string(ns.overrides))

    cmd = [
        "astrolabe",
        *(["--config-dir", str(config_dir)] if ns.config_dir else []),
        "run",
        "--standalone",
        str(config_file_path),
    ]
    if ns.task:
        cmd.extend(["--task", ns.task])
    if overrides:
        cmd.extend(["--overrides", " ".join(overrides)])
    if ns.print_config:
        cmd.append("--print-config")
    if ns.dry_run:
        cmd.append("--dry-run")

    print("Command:")
    print(" ".join(shlex.quote(c) for c in cmd))

    if not ns.execute:
        return 0

    run_ns = argparse.Namespace(
        cmd="run",
        config_dir=str(config_dir) if ns.config_dir else None,
        task=ns.task,
        standalone=str(config_file_path),
        overlay=None,
        chdir=True,
        overrides=" ".join(overrides) if overrides else None,
        overrides_file=None,
        print_config=ns.print_config,
        dry_run=ns.dry_run,
        hydra_overrides=[],
        autoselect_gpu=bool(getattr(ns, "autoselect_gpu", False)),
    )
    return _cmd_run(run_ns)


def _cmd_compose(ns: argparse.Namespace) -> int:
    if not _require_interactive_tty("astrolabe config compose"):
        return 2

    config_dir = pathlib.Path(ns.config_dir).resolve() if ns.config_dir else _default_config_dir()
    _ensure_sys_path_has_repo_root(config_dir)

    base_kind = (ns.base or "").strip().lower() if getattr(ns, "base", None) else ""
    base_task = getattr(ns, "task", None)

    if not base_kind:
        raw = input("base [base/template] (default: base): ").strip().lower()
        base_kind = raw or "base"
    if base_kind not in {"base", "template"}:
        print("ERR: base must be 'base' or 'template'")
        return 2
    if base_kind == "template" and not base_task:
        tasks = _list_available_tasks(config_dir)
        if not tasks:
            raise RuntimeError(f"No templates found under: {config_dir}")
        print("Available templates:")
        for idx, task_name in enumerate(tasks, start=1):
            print(f"{idx:02d} {task_name}")
        base_task = _prompt_choice_from_list("template: ", tasks)

    name = (ns.name or "").strip() if getattr(ns, "name", None) else ""
    if not name:
        name = input("name (used for output filenames): ").strip()
    name = _sanitize_name(name or "compose")

    overlay_out = (
        pathlib.Path(ns.overlay_out).expanduser().resolve()
        if getattr(ns, "overlay_out", None)
        else pathlib.Path(f"d.{name}.overlay.yaml").resolve()
    )
    standalone_out = (
        pathlib.Path(ns.standalone_out).expanduser().resolve()
        if getattr(ns, "standalone_out", None)
        else pathlib.Path(f"{name}.yaml").resolve()
    )

    overlay_obj: dict = {}

    def _write(*, write_standalone: bool, write_overlay: bool) -> None:
        base_ref = base_task if base_kind == "template" else "base"
        if write_standalone:
            standalone_obj = {"defaults": [base_ref, "_self_"]}
            standalone_obj.update(OmegaConf.to_container(OmegaConf.create(overlay_obj), resolve=False) or {})
            _compose_write_yaml(standalone_out, standalone_obj, with_header=True)
            print(f"Wrote: {standalone_out}")
        if write_overlay:
            _compose_write_yaml(overlay_out, overlay_obj, with_header=True)
            print(f"Wrote: {overlay_out}")
        print("Run examples:")
        if write_standalone:
            print(f"  astrolabe run --standalone {shlex.quote(str(standalone_out))}")
        if write_overlay:
            if base_kind == "template" and base_task:
                print(f"  astrolabe run --task {shlex.quote(str(base_task))} --overlay {shlex.quote(str(overlay_out))}")
            else:
                print(f"  astrolabe run --task base --overlay {shlex.quote(str(overlay_out))}")

    while True:
        section_status = _compose_get_section_status(overlay_obj)
        print("== astrolabe config compose ==")
        print(f"base: {base_kind}{'/' + str(base_task) if base_kind == 'template' else ''}")
        print(f"overlay: {overlay_out.name}  standalone: {standalone_out.name}")
        print("sections: basic evaluator mutator loss predocker stages")
        print("status: " + " ".join([f"{key}={'ok' if value else 'empty'}" for key, value in section_status.items()]))
        raw = _prompt_words(
            "compose> ",
            [
                "basic",
                "evaluator",
                "mutator",
                "loss",
                "predocker",
                "stages",
                ":show",
                ":validate",
                ":write",
                ":write-all",
                ":write-overlay",
                ":quit",
                ":help",
            ],
        ).strip()
        if raw in {":q", ":quit", ":exit"}:
            return 0
        if raw in {":w", ":write"}:
            _write(write_standalone=True, write_overlay=False)
            continue
        if raw in {":write-all"}:
            _write(write_standalone=True, write_overlay=True)
            continue
        if raw in {":write-overlay"}:
            _write(write_standalone=False, write_overlay=True)
            continue
        if raw in {":v", ":validate"}:
            errs = _compose_validate_basic(overlay_obj)
            if errs:
                print("ERRS:")
                for err in errs:
                    print("- " + err)
            else:
                print("OK")
            continue
        if raw in {":show"}:
            print(OmegaConf.to_yaml(OmegaConf.create(overlay_obj), resolve=False))
            continue
        if raw in {":?", ":help"} or not raw:
            print("Commands: basic|evaluator|mutator|loss|predocker|stages | :show :validate :write :write-all :write-overlay :quit")
            continue

        cmd = raw.lower()
        if cmd == "basic":
            _compose_section_basic(overlay_obj)
            continue
        if cmd == "evaluator":
            _compose_section_evaluator(overlay_obj, config_dir)
            continue
        if cmd == "mutator":
            _compose_section_mutator(overlay_obj, config_dir)
            continue
        if cmd == "loss":
            _compose_section_loss(overlay_obj)
            continue
        if cmd == "predocker":
            _compose_section_predocker(overlay_obj)
            continue
        if cmd == "stages":
            _compose_section_stages(overlay_obj)
            continue
        print("ERR: unknown command (try :help)")


def _cmd_scaffold(ns: argparse.Namespace) -> int:
    if not _require_interactive_tty("astrolabe config scaffold"):
        return 2

    config_dir = pathlib.Path(ns.config_dir).resolve() if ns.config_dir else _default_config_dir()
    task = ns.task
    if not task:
        raise ValueError("--task is required for scaffold.")
    task_path = config_dir / f"{task}.yaml"
    if not task_path.exists():
        tasks = _list_available_tasks(config_dir)
        hint = f" Available tasks: {', '.join(tasks)}" if tasks else ""
        raise FileNotFoundError(f"Task template not found: {task_path}.{hint}")

    template_cfg = OmegaConf.load(str(task_path))
    template_raw = OmegaConf.to_container(template_cfg, resolve=False)
    missing_keys = _scan_missing_keys_for_scaffold(template_raw)
    missing_keys = list(dict.fromkeys(missing_keys))
    if not missing_keys:
        print("No '???' keys found in template; nothing to scaffold.")
        return 0

    out_path = pathlib.Path(ns.out).expanduser().resolve() if ns.out else pathlib.Path(f"d.{task}.overlay.yaml").resolve()
    overlay_obj: dict = {}

    index = 0
    while index < len(missing_keys):
        key = missing_keys[index]
        current = _get_by_dotted_key(overlay_obj, key)
        value = _interactive_scaffold_prompt(key, current)
        if value == ":back":
            index = max(0, index - 1)
            continue
        _set_by_dotted_key(overlay_obj, key, value)
        index += 1

    _compose_write_yaml(out_path, overlay_obj, with_header=True)
    print(f"Wrote: {out_path}")
    print("Next:")
    print(" ".join([shlex.quote("astrolabe"), "run", "--task", shlex.quote(task), "--overlay", shlex.quote(str(out_path))]))
    return 0


def _cmd_env_wizard(ns: argparse.Namespace) -> int:
    if not _require_interactive_tty("astrolabe env wizard"):
        return 2

    config_dir = pathlib.Path(ns.config_dir).resolve() if ns.config_dir else _default_config_dir()
    task_path = config_dir / f"{ns.task}.yaml"
    if not task_path.exists():
        tasks = _list_available_tasks(config_dir)
        hint = f" Available tasks: {', '.join(tasks)}" if tasks else ""
        raise FileNotFoundError(f"Task template not found: {task_path}.{hint}")

    task_cfg = OmegaConf.load(str(task_path))
    task_cfg = _strip_defaults_key(task_cfg)
    req = _task_requirements(OmegaConf.to_container(task_cfg, resolve=False))

    env_in = pathlib.Path(ns.env).expanduser().resolve() if ns.env else _default_local_env_path_for_config_dir(config_dir)
    env_out = pathlib.Path(ns.out).expanduser().resolve() if ns.out else env_in
    if env_in.exists():
        base_cfg = OmegaConf.load(str(env_in))
        base_obj = OmegaConf.to_container(base_cfg, resolve=False)
        if not isinstance(base_obj, dict):
            base_obj = {}
    else:
        base_obj = {}

    required_keys = list(req.get("required_env_keys") or [])
    reason_map = dict(req.get("env_key_reasons") or {})
    if not required_keys:
        print("No required env keys inferred for task.")

    index = 0
    while index < len(required_keys):
        key = required_keys[index]
        current_value = _get_by_dotted_key(base_obj, key)
        current = None if current_value is None else str(current_value)
        ok = False
        why = reason_map.get(key)
        if why:
            print(f"WHY: {key} -> {why}")
        env_var = _extract_oc_env_name(current or "")
        if env_var:
            print(
                f"NOTE: current value is an oc.env placeholder; set {env_var} in your shell/.env or replace it here."
            )
        if current:
            ok, reason = _validate_path_value(key, current, env_path=env_in)
            if reason == "interpolated":
                ok = False
        if ok and not ns.force:
            index += 1
            continue
        new_value = _interactive_env_prompt(key, current, env_path=env_in)
        if new_value == ":back":
            index = max(0, index - 1)
            continue
        _set_by_dotted_key(base_obj, key, new_value)
        index += 1

    _compose_write_yaml(env_out, base_obj, with_header=True)
    _write_astrolabe_python_hint_for_config_dir(
        config_dir,
        _get_by_dotted_key(base_obj, "subprocess_envs.gh.python_bin"),
    )
    print(f"Wrote: {env_out}")

    doctor_ns = argparse.Namespace(
        config_dir=str(config_dir),
        env=str(env_out),
        task=str(ns.task),
        resolve=True,
        strict=True,
        python_deps=True,
        format="text",
        show_ok=bool(ns.show_ok),
        soft_fail=bool(ns.soft_fail),
    )
    return _cmd_doctor(doctor_ns)
