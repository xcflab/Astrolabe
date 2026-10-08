from __future__ import annotations

from pathlib import Path

import astrolabe.cli as cli


def test_task_requirements_for_rapidock_include_subprocess_env_keys() -> None:
    req = cli._task_requirements(
        {
            "predocker": {
                "predocker_terms": {
                    "rapidock": {
                        "_type": "RAPiDock",
                    }
                }
            }
        }
    )

    assert "weight_path.rapidock" in req["required_env_keys"]
    assert "subprocess_envs.rapidock.python_bin" in req["required_env_keys"]
    assert "subprocess_envs.rapidock.repo_path" in req["required_env_keys"]


def test_base_and_env_configs_expose_rapidock_runner_fields() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base_text = (repo_root / "config" / "base.yaml").read_text(encoding="utf-8")
    env_text = (repo_root / "config" / "env.yaml").read_text(encoding="utf-8")
    env_container_text = (repo_root / "config" / "env_container.yaml").read_text(
        encoding="utf-8"
    )
    third_party_text = (
        repo_root / "config" / "third_party_envs.yaml"
    ).read_text(encoding="utf-8")

    assert "python_bin: ${subprocess_envs.rapidock.python_bin}" in base_text
    assert "repo_path: ${subprocess_envs.rapidock.repo_path}" in base_text
    assert "num_samples_bank:" in base_text
    assert "response_cache_dir:" in base_text

    assert "rapidock:" in env_text
    assert "GH_RAPIDOCK_PYTHON" in env_text
    assert "GH_RAPIDOCK_REPO" in env_text

    assert "rapidock:" in env_container_text
    assert "GH_RAPIDOCK_PYTHON" in env_container_text
    assert "GH_RAPIDOCK_REPO" in env_container_text

    assert "- name: rapidock" in third_party_text
    assert "-e {{CLONE_ROOT}}/RAPiDock" not in third_party_text
    assert "subprocess_envs.rapidock.python_bin" in third_party_text
    assert "subprocess_envs.rapidock.repo_path" in third_party_text
