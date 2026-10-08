"""Public failure and fresh-configuration contracts; no model installation needed."""
from pathlib import Path

import pytest

from astrolabe import cli


@pytest.mark.parametrize("arguments, message", [
    (["run", "--standalone", "absent.yaml"], "--standalone not found"),
    (["run", "--task", "no_such_template", "--dry-run"], "Task template not found"),
    (["posthoc", "collect", "--out-root", "absent", "--db", "scores.sqlite"], "Output root is not a directory"),
    (["posthoc", "filter", "--db", "absent.sqlite", "--expr", "iptm > 0.8"], "Scores database not found"),
])
def test_bad_paths_are_actionable_and_do_not_create_outputs(arguments, message, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert message in stderr
    assert "Traceback" not in stderr
    assert list(tmp_path.iterdir()) == []


def test_fresh_config_uses_portable_environment_and_respects_explicit_choice(tmp_path):
    (tmp_path / "env").mkdir()
    (tmp_path / "base.yaml").write_text("defaults:\n  - env: env_local\n  - _self_\nseed: 17\n")
    (tmp_path / "env" / "env.yaml").write_text("location: portable\n")
    (tmp_path / "env" / "other.yaml").write_text("location: selected\n")
    cfg = cli._compose_hydra_config(config_dir=tmp_path, config_name="base", overrides=[])
    assert cfg.env.location == "portable"
    assert cfg.seed == 17
    cfg = cli._compose_hydra_config(config_dir=tmp_path, config_name="base", overrides=["env=other"])
    assert cfg.env.location == "selected"
    with pytest.raises(cli.MissingConfigException):
        cli._compose_hydra_config(config_dir=tmp_path, config_name="base", overrides=["env=env_local"])


def test_malformed_yaml_has_no_traceback(tmp_path, capsys):
    config = tmp_path / "bad.yaml"
    config.write_text("[unclosed\n")
    with pytest.raises(SystemExit) as error:
        cli.main(["run", "--overlay", str(config), "--dry-run", "env=env"])
    assert error.value.code == 2
    assert "Traceback" not in capsys.readouterr().err


def test_debug_retains_input_traceback(tmp_path):
    with pytest.raises(FileNotFoundError):
        cli.main(["--debug", "run", "--standalone", str(tmp_path / "absent.yaml")])
