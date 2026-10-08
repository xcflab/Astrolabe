from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "installation" / "env_setup.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("_test_gh_env_setup", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_esm2_inventory_variant(path: Path, **updates) -> None:
    data = yaml.safe_load(
        (REPO_ROOT / "config" / "third_party_sources.yaml").read_text(
            encoding="utf-8"
        )
    )
    data["components"]["esm2"].update(updates)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_env_setup_writes_canonical_runner_hint(tmp_path: Path) -> None:
    from astrolabe.dev_wrapper import resolve_default_astrolabe_python

    module = _load_script_module()
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    hint = module._write_gh_python_hint(tmp_path, str(python))
    assert hint == tmp_path / "config/env/.astrolabe_python_path"
    assert resolve_default_astrolabe_python(tmp_path) == str(python)


def test_select_env_specs_supports_exclude_after_default_selection() -> None:
    module = _load_script_module()
    envs = [
        {"name": "gh"},
        {"name": "rapidock"},
        {"name": "rapidock_assets"},
        {"name": "boltz"},
    ]

    selected = module._select_env_specs(
        envs, exclude=" rapidock, rapidock_assets "
    )

    assert [env["name"] for env in selected] == ["gh", "boltz"]


def test_select_env_specs_applies_exclude_after_only_selection() -> None:
    module = _load_script_module()
    envs = [
        {"name": "gh"},
        {"name": "rapidock"},
        {"name": "boltz"},
    ]

    selected = module._select_env_specs(
        envs, only="gh,rapidock", exclude="rapidock"
    )

    assert [env["name"] for env in selected] == ["gh"]


def test_runtime_profile_excludes_optional_heavy_backends() -> None:
    module = _load_script_module()

    assert module.ENV_PROFILES["runtime"] == (
        "gh",
        "dock",
        "boltz_cache",
        "alphafold3_assets",
        "alphafold2_assets",
        "protenix_assets",
        "usalign",
    )
    assert "rapidock" not in module.ENV_PROFILES["runtime"]
    assert "rapidock_assets" not in module.ENV_PROFILES["runtime"]
    assert "dplm" not in module.ENV_PROFILES["runtime"]
    assert "rpxdock" not in module.ENV_PROFILES["runtime"]


def test_needed_softwares_profile_adds_required_subprocess_backends() -> None:
    module = _load_script_module()

    assert module.ENV_PROFILES["needed_softwares"] == (
        "gh",
        "dock",
        "dplm",
        "rpxdock",
        "boltz_cache",
        "alphafold3_assets",
        "alphafold2_assets",
        "protenix_assets",
        "usalign",
    )
    assert "dplm" in module.ENV_PROFILES["needed_softwares"]
    assert "rpxdock" in module.ENV_PROFILES["needed_softwares"]
    assert "rapidock" not in module.ENV_PROFILES["needed_softwares"]
    assert "rapidock_assets" not in module.ENV_PROFILES["needed_softwares"]


def test_select_env_specs_rejects_invalid_entries() -> None:
    module = _load_script_module()

    try:
        module._select_env_specs([{"manager": "micromamba"}])
    except ValueError as exc:
        assert "Each env spec must be a mapping with a 'name'" in str(exc)
    else:
        raise AssertionError("invalid env spec should fail")


def test_validate_local_pip_specs_fails_for_empty_vendor_dir(tmp_path: Path) -> None:
    module = _load_script_module()
    vendor_dir = tmp_path / "vendor" / "protenix"
    vendor_dir.mkdir(parents=True)

    try:
        module._validate_local_pip_specs(
            "gh", [f"-e {vendor_dir} --no-deps"]
        )
    except RuntimeError as exc:
        text = str(exc)
        assert "Local pip source validation failed for env 'gh'" in text
        assert "git submodule update --init --recursive --depth 1 --jobs 4" in text
        assert "submodule URLs recorded in `.gitmodules`" in text
    else:
        raise AssertionError("empty vendored pip path should fail preflight")


def test_validate_local_pip_specs_accepts_local_python_project(tmp_path: Path) -> None:
    module = _load_script_module()
    project_dir = tmp_path / "vendor" / "protenix"
    project_dir.mkdir(parents=True)
    (project_dir / "setup.py").write_text("from setuptools import setup\n", encoding="utf-8")

    module._validate_local_pip_specs("gh", [f"-e {project_dir} --no-deps"])


def test_validate_local_pip_specs_accepts_editable_project_with_extras(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    project_dir = tmp_path / "vendor" / "opendde"
    project_dir.mkdir(parents=True)
    (project_dir / "pyproject.toml").write_text(
        "[project]\nname = 'opendde'\n",
        encoding="utf-8",
    )

    module._validate_local_pip_specs("opendde", [f"-e {project_dir}[gpu]"])


def test_repo_specs_cannot_target_vendored_submodule_paths() -> None:
    module = _load_script_module()
    repo_root = SCRIPT_PATH.parents[2]

    try:
        module._validate_repo_dest_not_submodule(
            {"dest": "{{REPO_ROOT}}/vendor/rpxdock"},
            {"REPO_ROOT": str(repo_root)},
        )
    except RuntimeError as exc:
        text = str(exc)
        assert "inside vendored submodule path vendor/rpxdock" in text
        assert "git submodule update --init --recursive vendor/rpxdock" in text
    else:
        raise AssertionError("repo spec targeting a submodule path should fail")


def test_repo_specs_cannot_target_vendored_submodule_subpaths() -> None:
    module = _load_script_module()
    repo_root = SCRIPT_PATH.parents[2]

    try:
        module._validate_repo_dest_not_submodule(
            {"dest": "{{REPO_ROOT}}/vendor/rpxdock/nested"},
            {"REPO_ROOT": str(repo_root)},
        )
    except RuntimeError as exc:
        text = str(exc)
        assert "inside vendored submodule path vendor/rpxdock" in text
        assert "vendor/rpxdock/nested" in text
    else:
        raise AssertionError("repo spec targeting a submodule child path should fail")


def test_third_party_env_repos_do_not_target_git_submodules() -> None:
    import yaml

    module = _load_script_module()
    config = yaml.safe_load((REPO_ROOT / "config" / "third_party_envs.yaml").read_text(encoding="utf-8"))
    submodule_paths = module._repo_submodule_paths(REPO_ROOT)

    collisions = []
    for env in config["envs"]:
        for repo in env.get("repos") or []:
            dest = repo.get("dest")
            if not dest:
                continue
            dest_path = Path(
                module._apply_env_substitutions(
                    dest,
                    {
                        "REPO_ROOT": str(REPO_ROOT),
                        "ROOT_PREFIX": "/tmp/gh_envs",
                        "CLONE_ROOT": "/tmp/gh_envs/repos",
                        "PREFIX": "/tmp/gh_envs/envs/example",
                    },
                )
            ).expanduser().resolve()
            if any(
                dest_path == submodule_path or submodule_path in dest_path.parents
                for submodule_path in submodule_paths
            ):
                collisions.append(str(dest_path.relative_to(REPO_ROOT)))

    assert collisions == []


def test_third_party_source_inventory_validates_current_gitmodules() -> None:
    module = _load_script_module()

    module._validate_third_party_source_inventory(REPO_ROOT)


def test_source_inventory_rejects_missing_gitmodule_entry(monkeypatch) -> None:
    module = _load_script_module()
    urls = dict(module._gitmodules_urls(REPO_ROOT))
    urls["vendor/new_backend"] = "https://example.invalid/new_backend.git"
    monkeypatch.setattr(module, "_gitmodules_urls", lambda _repo_root: urls)

    try:
        module._validate_third_party_source_inventory(REPO_ROOT)
    except RuntimeError as exc:
        assert "Git submodules missing from source inventory" in str(exc)
        assert "vendor/new_backend" in str(exc)
    else:
        raise AssertionError("unregistered git submodule should fail")


def test_source_inventory_rejects_url_mismatch(monkeypatch) -> None:
    module = _load_script_module()
    urls = dict(module._gitmodules_urls(REPO_ROOT))
    urls["vendor/protenix"] = "https://example.invalid/protenix.git"
    monkeypatch.setattr(module, "_gitmodules_urls", lambda _repo_root: urls)

    try:
        module._validate_third_party_source_inventory(REPO_ROOT)
    except RuntimeError as exc:
        text = str(exc)
        assert ".gitmodules url for vendor/protenix" in text
        assert "https://example.invalid/protenix.git" in text
    else:
        raise AssertionError("source URL mismatch should fail")


def test_source_inventory_rejects_missing_patch_group(monkeypatch) -> None:
    module = _load_script_module()
    monkeypatch.setattr(
        module,
        "_vendor_patch_script_groups",
        lambda _repo_root: {"alphafold2", "missing_patch_group"},
    )

    try:
        module._validate_third_party_source_inventory(REPO_ROOT)
    except RuntimeError as exc:
        text = str(exc)
        assert "Patch groups missing from source inventory" in text
        assert "missing_patch_group" in text
    else:
        raise AssertionError("missing patch group should fail")


def test_source_inventory_rejects_unknown_patch_group(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    inventory_path = tmp_path / "third_party_sources.yaml"
    text = (REPO_ROOT / "config" / "third_party_sources.yaml").read_text(encoding="utf-8")
    inventory_path.write_text(
        text.replace("patch_group: alphafold2", "patch_group: unknown_patch_group", 1),
        encoding="utf-8",
    )
    monkeypatch.setenv("GH_THIRD_PARTY_SOURCES", str(inventory_path))

    try:
        module._validate_third_party_source_inventory(REPO_ROOT)
    except RuntimeError as exc:
        text = str(exc)
        assert "Unknown patch groups in source inventory" in text
        assert "unknown_patch_group" in text
    else:
        raise AssertionError("unknown patch group should fail")


def test_source_inventory_rejects_fork_pending_without_fork_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module()
    inventory_path = tmp_path / "third_party_sources.yaml"
    _write_esm2_inventory_variant(
        inventory_path,
        fork_candidate=False,
        migration_status="fork_pending",
    )
    monkeypatch.setenv("GH_THIRD_PARTY_SOURCES", str(inventory_path))

    try:
        module._validate_third_party_source_inventory(REPO_ROOT)
    except RuntimeError as exc:
        text = str(exc)
        assert "fork_pending entries must set fork_candidate=true" in text
        assert "esm2" in text
    else:
        raise AssertionError("fork_pending source without fork_candidate should fail")


def test_source_inventory_rejects_fork_pending_without_official_patch_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module()
    inventory_path = tmp_path / "third_party_sources.yaml"
    _write_esm2_inventory_variant(
        inventory_path,
        fork_candidate=True,
        migration_status="fork_pending",
    )
    monkeypatch.setenv("GH_THIRD_PARTY_SOURCES", str(inventory_path))

    try:
        module._validate_third_party_source_inventory(REPO_ROOT)
    except RuntimeError as exc:
        text = str(exc)
        assert "fork_pending entries must use source_mode=official_submodule_plus_patch" in text
        assert "esm2" in text
    else:
        raise AssertionError("fork_pending source without official patch source should fail")


def test_source_inventory_rejects_branch_mismatch(monkeypatch) -> None:
    module = _load_script_module()
    branches = dict(module._gitmodules_branches(REPO_ROOT))
    branches["vendor/esm3"] = "main"
    monkeypatch.setattr(module, "_gitmodules_branches", lambda _repo_root: branches)

    try:
        module._validate_third_party_source_inventory(REPO_ROOT)
    except RuntimeError as exc:
        text = str(exc)
        assert ".gitmodules branch for vendor/esm3" in text
        assert "expected 'gh-package-rename'" in text
    else:
        raise AssertionError("source branch mismatch should fail")


def test_source_inventory_rejects_gitlink_mismatch(monkeypatch) -> None:
    module = _load_script_module()
    commits = dict(module._gitlink_commits(REPO_ROOT))
    commits["vendor/esm3"] = "0" * 40
    monkeypatch.setattr(module, "_gitlink_commits", lambda _repo_root: commits)

    with pytest.raises(RuntimeError, match="esm3: gitlink.*expected"):
        module._validate_third_party_source_inventory(REPO_ROOT)


def test_gitlink_inventory_reads_release_head_not_staged_index(tmp_path: Path) -> None:
    module = _load_script_module()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    head_commit = "1" * 40
    staged_commit = "2" * 40
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "update-index",
            "--add",
            "--cacheinfo",
            "160000",
            head_commit,
            "vendor/example",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "pin gitlink"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "update-index",
            "--cacheinfo",
            "160000",
            staged_commit,
            "vendor/example",
        ],
        check=True,
    )

    assert module._gitlink_commits(tmp_path) == {"vendor/example": head_commit}


def test_source_inventory_rejects_uninventoried_branch(monkeypatch) -> None:
    module = _load_script_module()
    branches = dict(module._gitmodules_branches(REPO_ROOT))
    branches["vendor/protenix"] = "main"
    monkeypatch.setattr(module, "_gitmodules_branches", lambda _repo_root: branches)

    try:
        module._validate_third_party_source_inventory(REPO_ROOT)
    except RuntimeError as exc:
        text = str(exc)
        assert ".gitmodules branch for vendor/protenix" in text
        assert "has no expected_branch" in text
    else:
        raise AssertionError("unregistered source branch should fail")


def test_source_inventory_rejects_malformed_inventory(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    inventory_path = tmp_path / "third_party_sources.yaml"
    inventory_path.write_text("components: []\n", encoding="utf-8")
    monkeypatch.setenv("GH_THIRD_PARTY_SOURCES", str(inventory_path))

    try:
        module._validate_third_party_source_inventory(REPO_ROOT)
    except ValueError as exc:
        assert "Invalid source inventory" in str(exc)
    else:
        raise AssertionError("malformed source inventory should fail")


def test_existing_git_repo_fetch_failure_is_visible(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    repo = tmp_path / "repos" / "example"
    (repo / ".git").mkdir(parents=True)

    def _fail_fetch(cmd, cwd=None):
        assert cmd == ["git", "fetch", "--all", "--tags"]
        assert cwd == str(repo)
        raise RuntimeError("fetch failed")

    monkeypatch.setattr(module, "_run", _fail_fetch)

    try:
        module._ensure_git_repo(
            {
                "url": "https://example.invalid/repo.git",
                "dest": str(repo),
            },
            {},
            non_interactive=True,
        )
    except RuntimeError as exc:
        assert "fetch failed" in str(exc)
    else:
        raise AssertionError("existing repo fetch failure should be visible")


def test_vendor_patch_groups_selected_for_patchable_editable_vendor_paths() -> None:
    module = _load_script_module()
    repo_root = SCRIPT_PATH.parents[2]

    groups = module._vendor_patch_groups_for_pip_specs(
        [
            f"-e {repo_root / 'vendor' / 'alphafold2'} --no-deps",
            f"-e {repo_root / 'vendor' / 'chai_lab'} --no-deps",
            f"-e {repo_root / 'vendor' / 'protenix'} --no-deps",
        ]
    )

    assert groups == ["alphafold2", "chai_lab"]


def test_apply_repo_vendor_patches_for_pip_specs_invokes_existing_patch_helper(monkeypatch) -> None:
    module = _load_script_module()
    repo_root = SCRIPT_PATH.parents[2]
    calls = []

    monkeypatch.setattr(module, "_run", lambda cmd: calls.append(cmd))

    module._apply_repo_vendor_patches_for_pip_specs(
        [
            f"-e {repo_root / 'vendor' / 'alphafold2'} --no-deps",
            f"-e {repo_root / 'vendor' / 'chai_lab'} --no-deps",
            "requests==2.32.3",
        ]
    )

    assert len(calls) == 1
    assert calls[0][1].endswith("scripts/maintenance/apply_vendor_patches.py")
    assert calls[0][2:] == ["--only", "alphafold2", "--only", "chai_lab"]


def test_apply_repo_vendor_patches_for_pip_specs_skips_unpatchable_paths(monkeypatch) -> None:
    module = _load_script_module()
    repo_root = SCRIPT_PATH.parents[2]
    calls = []

    monkeypatch.setattr(module, "_run", lambda cmd: calls.append(cmd))

    module._apply_repo_vendor_patches_for_pip_specs(
        [
            f"-e {repo_root / 'vendor' / 'protenix'} --no-deps",
            f"-e {repo_root / 'vendor' / 'alphafold3'} --no-deps",
            f"-e {repo_root / 'vendor' / 'esm2'} --no-deps",
            f"-e {repo_root / 'vendor' / 'esm3'} --no-deps",
            "requests==2.32.3",
        ]
    )

    assert calls == []


def test_setup_one_env_applies_vendor_patches_before_pip_install(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    repo_root = SCRIPT_PATH.parents[2]
    events = []

    monkeypatch.setattr(module, "_env_exists", lambda _prefix: False)
    monkeypatch.setattr(
        module,
        "_validate_local_pip_specs",
        lambda _env_name, _pip_specs: None,
    )
    monkeypatch.setattr(
        module,
        "_apply_repo_vendor_patches_for_pip_specs",
        lambda _pip_specs: events.append("patch"),
    )

    def _fake_run(cmd, env=None, cwd=None, check=True):
        if "pip" in cmd and "install" in cmd:
            events.append("pip")
        else:
            events.append("other")

    monkeypatch.setattr(module, "_run", _fake_run)

    ctx = {
        "root_prefix": tmp_path / "root",
        "micromamba": "/usr/bin/micromamba",
        "state_root": tmp_path / "state",
        "non_interactive": True,
        "force": False,
        "decisions": {},
        "substitutions": {
            "ROOT_PREFIX": str(tmp_path / "root"),
            "REPO_ROOT": str(repo_root),
        },
        "env": {},
        "conda_exe": None,
        "skip_commands": True,
        "rerun_steps": [],
        "clone_root": tmp_path / "repos",
    }

    module._setup_one_env(
        {
            "name": "gh",
            "manager": "micromamba",
            "python": "3.11",
            "channels": ["conda-forge"],
            "pip": [f"-e {repo_root / 'vendor' / 'alphafold2'} --no-deps"],
        },
        ctx,
    )

    assert events.index("patch") < events.index("pip")


def test_nampnn_repo_export_is_optional_for_runtime_profile() -> None:
    text = (REPO_ROOT / "config" / "third_party_envs.yaml").read_text(encoding="utf-8")

    assert "subprocess_envs.nampnn.repo_path:" in text
    assert 'value: "{{REPO_ROOT}}/vendor/na_mpnn"' in text
    assert "optional: true" in text


def test_vendor_patch_command_runs_before_editable_pip_installs() -> None:
    import yaml

    config = yaml.safe_load((REPO_ROOT / "config" / "third_party_envs.yaml").read_text(encoding="utf-8"))
    gh_env = next(env for env in config["envs"] if env["name"] == "gh")

    assert any("apply_vendor_patches.py" in cmd for cmd in gh_env["pre_pip_commands"])
    assert not any("apply_vendor_patches.py" in cmd for cmd in gh_env["commands"])


def test_external_env_missing_prefix_fails_visibly(tmp_path: Path) -> None:
    module = _load_script_module()
    missing_prefix = tmp_path / "missing_env"
    ctx = {
        "root_prefix": tmp_path / "root",
        "micromamba": "/usr/bin/micromamba",
        "state_root": tmp_path / "state",
        "non_interactive": True,
        "force": False,
        "decisions": {},
        "substitutions": {
            "ROOT_PREFIX": str(tmp_path / "root"),
            "REPO_ROOT": str(REPO_ROOT),
        },
        "env": {},
        "conda_exe": None,
        "skip_commands": True,
        "rerun_steps": [],
        "clone_root": tmp_path / "repos",
    }

    try:
        module._setup_one_env(
            {
                "name": "existing_gh",
                "manager": "external",
                "prefix": str(missing_prefix),
            },
            ctx,
        )
    except RuntimeError as exc:
        text = str(exc)
        assert "external env 'existing_gh' prefix is missing" in text
        assert str(missing_prefix / "bin" / "python") in text
        assert "manager=micromamba" in text
    else:
        raise AssertionError("missing external env prefix should fail before setup")


def test_external_env_missing_python_bin_fails_visibly(tmp_path: Path) -> None:
    module = _load_script_module()
    missing_python = tmp_path / "existing_env" / "bin" / "python"

    try:
        module._resolve_external_env_prefix(
            "existing_gh",
            {"name": "existing_gh", "manager": "external", "python_bin": str(missing_python)},
            {},
        )
    except RuntimeError as exc:
        text = str(exc)
        assert "external env 'existing_gh' python_bin does not exist" in text
        assert str(missing_python) in text
    else:
        raise AssertionError("missing external python_bin should fail")


def test_external_env_non_executable_python_bin_fails_visibly(tmp_path: Path) -> None:
    module = _load_script_module()
    python_bin = tmp_path / "existing_env" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("#!/bin/sh\n", encoding="utf-8")

    try:
        module._resolve_external_env_prefix(
            "existing_gh",
            {"name": "existing_gh", "manager": "external", "python_bin": str(python_bin)},
            {},
        )
    except RuntimeError as exc:
        text = str(exc)
        assert "external env 'existing_gh' python_bin is not executable" in text
        assert str(python_bin) in text
    else:
        raise AssertionError("non-executable external python_bin should fail")


def test_export_entry_prefers_existing_host_env_var() -> None:
    module = _load_script_module()

    value, meta = module._resolve_export_entry(
        "weight_path.alphafold3",
        {
            "value": "{{PREFIX}}/models",
            "env_var": "GH_WEIGHT_AF3",
            "optional": True,
        },
        {"PREFIX": "/new/root/alphafold3"},
        {"GH_WEIGHT_AF3": "/existing/af3/models"},
    )

    assert value == "/existing/af3/models"
    assert meta == {
        "optional": True,
        "env_var": "GH_WEIGHT_AF3",
        "used_env_var": True,
    }


def test_export_entry_uses_default_when_host_env_var_is_empty() -> None:
    module = _load_script_module()

    value, meta = module._resolve_export_entry(
        "weight_path.alphafold3",
        {
            "value": "{{PREFIX}}/models",
            "env_var": "GH_WEIGHT_AF3",
            "optional": True,
        },
        {"PREFIX": "/new/root/alphafold3"},
        {"GH_WEIGHT_AF3": ""},
    )

    assert value == "/new/root/alphafold3/models"
    assert meta == {
        "optional": True,
        "env_var": "GH_WEIGHT_AF3",
        "used_env_var": False,
    }


def test_asset_exports_declare_host_env_var_reuse() -> None:
    import yaml

    config = yaml.safe_load((REPO_ROOT / "config" / "third_party_envs.yaml").read_text(encoding="utf-8"))
    envs = {env["name"]: env for env in config["envs"]}

    expected = {
        ("boltz_cache", "weight_path.boltz"): "GH_WEIGHT_BOLTZ",
        ("alphafold3_assets", "weight_path.alphafold3"): "GH_WEIGHT_AF3",
        ("alphafold2_assets", "weight_path.alphafold"): "GH_WEIGHT_AF2",
        ("rapidock_assets", "weight_path.rapidock"): "GH_WEIGHT_RAPIDOCK",
        ("opendde_assets", "weight_path.opendde"): "OPENDDE_ROOT_DIR",
        ("esmfold2", "subprocess_envs.esmfold2.python_bin"): "GH_ESMFOLD2_PYTHON",
        ("esmfold2", "subprocess_envs.esmfold2.esm_repo_path"): "GH_ESMFOLD2_ESM_REPO",
        ("esmfold2", "subprocess_envs.esmfold2.transformers_repo_path"): "GH_ESMFOLD2_TRANSFORMERS_REPO",
        ("protenix_assets", "weight_path.protenix"): "GH_WEIGHT_PROTENIX",
        ("protenix_assets", "weight_path.protenix_constraint"): "GH_WEIGHT_PROTENIX_CONSTRAINT",
        ("protenix_assets", "protenix.ccd_components_file"): "GH_PROTENIX_CCD",
        ("protenix_assets", "protenix.ccd_components_rdkit_mol_file"): "GH_PROTENIX_CCD_RDKIT",
    }

    for (env_name, key), env_var in expected.items():
        assert envs[env_name]["export"][key]["env_var"] == env_var


def test_optional_export_default_missing_is_not_verification_error(tmp_path: Path) -> None:
    module = _load_script_module()

    missing, unresolved = module._collect_export_path_issues(
        {"weight_path.alphafold3": str(tmp_path / "default_missing")},
        {
            "weight_path.alphafold3": {
                "optional": True,
                "env_var": "GH_WEIGHT_AF3",
                "used_env_var": False,
            }
        },
    )

    assert missing == []
    assert unresolved == []


def test_optional_export_env_var_missing_is_verification_error(tmp_path: Path) -> None:
    module = _load_script_module()
    missing_path = str(tmp_path / "env_var_missing")

    missing, unresolved = module._collect_export_path_issues(
        {"weight_path.alphafold3": missing_path},
        {
            "weight_path.alphafold3": {
                "optional": True,
                "env_var": "GH_WEIGHT_AF3",
                "used_env_var": True,
            }
        },
    )

    assert missing == [("weight_path.alphafold3", missing_path)]
    assert unresolved == []
