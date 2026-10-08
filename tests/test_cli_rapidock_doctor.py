from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import astrolabe.cli as cli
import astrolabe.cli_interactive as cli_interactive


def _run(argv: list[str]) -> int:
    parser = cli._build_parser()
    ns = parser.parse_args(argv)
    handler = getattr(ns, "_handler", None)
    assert handler is not None
    return int(handler(ns))


def test_cli_env_doctor_runs_rapidock_probes_for_task(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)

    rapidock_weights = tmp_path / "weights"
    rapidock_weights.mkdir()
    (rapidock_weights / "model_parameters.yml").write_text("dummy: true\n", encoding="utf-8")
    (rapidock_weights / "rapidock_global.pt").write_text("stub", encoding="utf-8")

    python_bin = tmp_path / "python"
    python_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_bin.chmod(0o755)

    repo_path = tmp_path / "rapidock_repo"
    (repo_path / "RAPiDock").mkdir(parents=True)
    (repo_path / "RAPiDock" / "API.py").write_text("class RAPiDockInference:\n    pass\n", encoding="utf-8")
    (repo_path / "RAPiDock" / "daemon.py").write_text("print('unused')\n", encoding="utf-8")

    (config_dir / "env.yaml").write_text(
        "\n".join(
            [
                "weight_path:",
                f"  rapidock: {rapidock_weights}",
                "subprocess_envs:",
                "  rapidock:",
                f"    python_bin: {python_bin}",
                f"    repo_path: {repo_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "rapidock_task.yaml").write_text(
        "\n".join(
            [
                "predocker:",
                "  predocker_terms:",
                "    rapidock:",
                "      _type: RAPiDock",
                "  RAPiDock:",
                "    ckpt: rapidock_global.pt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    observed = {"imports": [], "daemon": []}

    monkeypatch.setattr(
        cli,
        "_probe_rapidock_imports",
        lambda *, python_bin, repo_path: observed["imports"].append((python_bin, repo_path)),
    )
    monkeypatch.setattr(
        cli,
        "_probe_rapidock_daemon",
        lambda *, python_bin, repo_path, model_dir, ckpt: observed["daemon"].append(
            (python_bin, repo_path, model_dir, ckpt)
        ),
    )

    rc = _run(
        [
            "--config-dir",
            str(config_dir),
            "env",
            "doctor",
            "--task",
            "rapidock_task",
            "--strict",
        ]
    )

    assert rc == 0
    assert observed["imports"] == [(str(python_bin), str(repo_path))]
    assert observed["daemon"] == [
        (str(python_bin), str(repo_path), str(rapidock_weights), "rapidock_global.pt")
    ]
    out = capsys.readouterr().out
    assert "summary:" in out


def test_probe_rapidock_daemon_uses_local_wrapper_without_upstream_daemon(tmp_path: Path) -> None:
    repo_path = tmp_path / "rapidock_repo"
    (repo_path / "RAPiDock").mkdir(parents=True)
    (repo_path / "RAPiDock" / "__init__.py").write_text("", encoding="utf-8")
    (repo_path / "inference.py").write_text("import argparse\nargparse.ArgumentParser()\n", encoding="utf-8")

    model_dir = tmp_path / "weights"
    model_dir.mkdir()
    (model_dir / "rapidock_global.pt").write_text("stub", encoding="utf-8")

    cli._probe_rapidock_daemon(
        python_bin=sys.executable,
        repo_path=str(repo_path),
        model_dir=str(model_dir),
        ckpt="rapidock_global.pt",
    )


def test_cli_env_doctor_task_ignores_unused_unresolved_evaluator_defaults(
    tmp_path: Path, capsys
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)

    (config_dir / "env.yaml").write_text("weight_path: {}\n", encoding="utf-8")
    (config_dir / "dummy_task.yaml").write_text(
        "\n".join(
            [
                "evaluator:",
                "  alphafold3:",
                "    model_dir: ${weight_path.alphafold3}",
                "    datapipeline:",
                "      jackhmmer_binary_path: ${hmmer.jackhmmer}",
                "  rhofold:",
                "    checkpoint: ${rhofold.checkpoint}",
                "  evaluator_terms:",
                "    Dummy:",
                "      _type: dummy",
                "stages:",
                "  default:",
                "    evaluator: Dummy",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = _run(
        [
            "--config-dir",
            str(config_dir),
            "env",
            "doctor",
            "--task",
            "dummy_task",
            "--soft-fail",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "summary:" in out
    assert "weight_path.alphafold3" not in out
    assert "hmmer.jackhmmer" not in out


def test_cli_env_doctor_reports_meeko_subprocess_import_failure(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)

    python_bin = tmp_path / "python"
    python_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_bin.chmod(0o755)
    obabel = tmp_path / "obabel"
    obabel.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    obabel.chmod(0o755)
    vina = tmp_path / "vina"
    vina.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    vina.chmod(0o755)
    mk_prepare_ligand = tmp_path / "mk_prepare_ligand.py"
    mk_prepare_ligand.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    mk_prepare_ligand.chmod(0o755)

    (config_dir / "env.yaml").write_text(
        "\n".join(
            [
                f"obabel_path: {obabel}",
                "autodock:",
                f"  vina_path: {vina}",
                f"  mk_prepare_ligand: {mk_prepare_ligand}",
                "subprocess_envs:",
                "  meeko:",
                f"    python_bin: {python_bin}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "vina_task.yaml").write_text(
        "predocker:\n"
        "  predocker_terms:\n"
        "    vina:\n"
        "      _type: autodock_vina\n",
        encoding="utf-8",
    )

    def fail_meeko_imports(*, python_bin: str, pythonpath: str | None = None) -> None:
        raise RuntimeError(f"failed: {python_bin}")

    monkeypatch.setattr(cli, "_probe_meeko_imports", fail_meeko_imports)

    rc = _run(
        [
            "--config-dir",
            str(config_dir),
            "env",
            "doctor",
            "--task",
            "vina_task",
            "--strict",
            "--soft-fail",
            "--format",
            "json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert {
        "type": "meeko_subprocess",
        "name": "imports",
        "error": f"failed: {python_bin}",
    } in payload["python_deps"]["missing"]


def test_cli_env_doctor_passes_meeko_pythonpath_to_probe(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)

    python_bin = tmp_path / "python"
    python_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_bin.chmod(0o755)
    pythonpath = tmp_path / "site-packages"
    pythonpath.mkdir()
    obabel = tmp_path / "obabel"
    obabel.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    obabel.chmod(0o755)
    vina = tmp_path / "vina"
    vina.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    vina.chmod(0o755)
    mk_prepare_ligand = tmp_path / "mk_prepare_ligand.py"
    mk_prepare_ligand.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    mk_prepare_ligand.chmod(0o755)

    (config_dir / "env.yaml").write_text(
        "\n".join(
            [
                f"obabel_path: {obabel}",
                "autodock:",
                f"  vina_path: {vina}",
                f"  mk_prepare_ligand: {mk_prepare_ligand}",
                "subprocess_envs:",
                "  meeko:",
                f"    python_bin: {python_bin}",
                f"    pythonpath: {pythonpath}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "vina_task.yaml").write_text(
        "predocker:\n"
        "  predocker_terms:\n"
        "    vina:\n"
        "      _type: autodock_vina\n",
        encoding="utf-8",
    )

    observed = []

    def record_meeko_imports(*, python_bin: str, pythonpath: str | None = None) -> None:
        observed.append((python_bin, pythonpath))

    monkeypatch.setattr(cli, "_probe_meeko_imports", record_meeko_imports)

    rc = _run(
        [
            "--config-dir",
            str(config_dir),
            "env",
            "doctor",
            "--task",
            "vina_task",
            "--strict",
            "--format",
            "json",
        ]
    )

    assert rc == 0
    json.loads(capsys.readouterr().out)
    assert observed == [(str(python_bin), str(pythonpath))]


def test_cli_env_wizard_task_ignores_unused_unresolved_evaluator_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    env_path = config_dir / "env.yaml"
    env_path.write_text("weight_path: {}\n", encoding="utf-8")
    (config_dir / "dummy_task.yaml").write_text(
        "\n".join(
            [
                "evaluator:",
                "  alphafold3:",
                "    model_dir: ${weight_path.alphafold3}",
                "    datapipeline:",
                "      jackhmmer_binary_path: ${hmmer.jackhmmer}",
                "  rhofold:",
                "    checkpoint: ${rhofold.checkpoint}",
                "  evaluator_terms:",
                "    Dummy:",
                "      _type: dummy",
                "stages:",
                "  default:",
                "    evaluator: Dummy",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    doctor_calls: list[argparse.Namespace] = []
    monkeypatch.setattr(cli_interactive, "_require_interactive_tty", lambda _cmd: True)
    monkeypatch.setattr(
        cli_interactive,
        "_interactive_env_prompt",
        lambda key, current, *, env_path: f"/tmp/{key.replace('.', '_')}",
    )
    monkeypatch.setattr(cli_interactive, "_cmd_doctor", lambda ns: doctor_calls.append(ns) or 0)
    monkeypatch.setattr(cli_interactive, "_write_astrolabe_python_hint", lambda *_a, **_k: None)

    rc = cli_interactive._cmd_env_wizard(
        argparse.Namespace(
            config_dir=str(config_dir),
            task="dummy_task",
            env=str(env_path),
            out=str(env_path),
            force=False,
            show_ok=False,
            soft_fail=True,
        )
    )

    assert rc == 0
    assert len(doctor_calls) == 1
    assert "alphafold3" not in env_path.read_text(encoding="utf-8")
    assert "hmmer" not in env_path.read_text(encoding="utf-8")


def test_cli_env_doctor_json_serializes_namedtuple_items(tmp_path: Path, capsys) -> None:
    env_path = tmp_path / "env.yaml"
    env_path.write_text(
        "rhofold:\n  binary_dpath: ${oc.env:GH_RHOFOLD_BIN,???}\n",
        encoding="utf-8",
    )

    rc = _run(["env", "doctor", "--env", str(env_path), "--format", "json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["interpolated"] == [
        {
            "exists": None,
            "key": "rhofold.binary_dpath",
            "kind": "unknown",
            "value": "${oc.env:GH_RHOFOLD_BIN,???}",
        }
    ]


def test_cli_env_doctor_reports_missing_tensorflow_for_af2_task(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()

    (config_dir / "env.yaml").write_text(
        f"weight_path:\n  alphafold: {weights_dir}\n",
        encoding="utf-8",
    )
    (config_dir / "af2_task.yaml").write_text(
        "\n".join(
            [
                "evaluator:",
                "  evaluator_terms:",
                "    AF2:",
                "      _type: alphafold2",
                "stages:",
                "  default:",
                "    evaluator: AF2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    real_import_module = __import__("importlib").import_module

    def _fake_import_module(name: str, package=None):
        if name == "tensorflow":
            raise ModuleNotFoundError("No module named 'tensorflow'")
        return real_import_module(name, package)

    monkeypatch.setattr("importlib.import_module", _fake_import_module)

    rc = _run(
        [
            "--config-dir",
            str(config_dir),
            "env",
            "doctor",
            "--task",
            "af2_task",
            "--python-deps",
        ]
    )

    assert rc == 2
    out = capsys.readouterr().out
    assert "PY MISS  import tensorflow" in out
