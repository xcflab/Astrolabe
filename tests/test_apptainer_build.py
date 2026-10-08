from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "containers" / "apptainer_build.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("_test_gh_apptainer_build", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_valid_image_metadata(module, image: Path, *, features=()) -> Path:
    features = sorted(set(features))
    contract = {
        "schema_version": 5,
        "context_sha256": "1" * 64,
        "definition_sha256": "2" * 64,
        "external_inputs": [],
        "build_features": features,
        "accepted_licenses": sorted(
            module.BUILD_FEATURE_LICENSES[feature]
            for feature in features
            if feature in module.BUILD_FEATURE_LICENSES
        ),
    }
    metadata = {
        "path": str(image.resolve()),
        "size": image.stat().st_size,
        "sha256": module._sha256_file(image),
        "build_features": features,
        "build_contract": contract,
        "build_fingerprint": module._build_contract_fingerprint(contract),
    }
    path = Path(str(image) + ".build.json")
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_stage_build_context_excludes_large_runtime_artifacts(tmp_path: Path) -> None:
    module = _load_script_module()

    repo_root = tmp_path / "repo"
    stage_root = tmp_path / "stage"
    repo_root.mkdir()
    (repo_root / "outputs" / "old_run").mkdir(parents=True)
    (repo_root / ".git").mkdir()
    (repo_root / ".venv").mkdir()
    (repo_root / ".worktrees").mkdir()
    (repo_root / "logs").mkdir()
    (repo_root / "vendor" / "alphafold3").mkdir(parents=True)
    (repo_root / "scripts" / "installation").mkdir(parents=True)
    (repo_root / "config").mkdir()

    (repo_root / "outputs" / "old_run" / "scores.csv").write_text("1,2,3\n", encoding="utf-8")
    (repo_root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (repo_root / ".venv" / "python").write_text("venv\n", encoding="utf-8")
    (repo_root / ".worktrees" / "meta").write_text("worktree\n", encoding="utf-8")
    (repo_root / "logs" / "build.log").write_text("debug\n", encoding="utf-8")
    (repo_root / "vendor" / "alphafold3" / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    (repo_root / "scripts" / "installation" / "install_pyrosetta.py").write_text("print('ok')\n", encoding="utf-8")
    (repo_root / "config" / "base.yaml").write_text("defaults: []\n", encoding="utf-8")
    (repo_root / "run.py").write_text("print('run')\n", encoding="utf-8")

    tracked_files = (
        Path("vendor/alphafold3/pyproject.toml"),
        Path("scripts/installation/install_pyrosetta.py"),
        Path("config/base.yaml"),
        Path("run.py"),
    )
    stage_repo = module._stage_build_context(repo_root, stage_root, tracked_files)

    assert stage_repo == stage_root / "repo"
    assert not (stage_repo / "outputs").exists()
    assert not (stage_repo / ".git").exists()
    assert not (stage_repo / ".venv").exists()
    assert not (stage_repo / ".worktrees").exists()
    assert not (stage_repo / "logs").exists()
    assert (stage_repo / "vendor" / "alphafold3" / "pyproject.toml").exists()
    assert (stage_repo / "scripts" / "installation" / "install_pyrosetta.py").exists()
    assert (stage_repo / "config" / "base.yaml").exists()
    assert (stage_repo / "run.py").exists()


def test_run_reports_command_failure_without_traceback() -> None:
    module = _load_script_module()

    try:
        module._run(
            [
                "python3",
                "-c",
                "import sys; sys.exit(7)",
            ]
        )
    except SystemExit as exc:
        text = str(exc)
        assert "Command failed with exit code 7" in text
        assert "Traceback" not in text
    else:
        raise AssertionError("non-zero command should fail")


def test_stage_build_context_hardlinks_retained_files(tmp_path: Path) -> None:
    module = _load_script_module()

    repo_root = tmp_path / "repo"
    stage_root = tmp_path / "stage"
    repo_root.mkdir()
    (repo_root / "scripts" / "installation").mkdir(parents=True)
    kept_file = repo_root / "scripts" / "installation" / "install_pyrosetta.py"
    kept_file.write_text("print('ok')\n", encoding="utf-8")

    stage_repo = module._stage_build_context(
        repo_root,
        stage_root,
        (Path("scripts/installation/install_pyrosetta.py"),),
    )
    staged_file = stage_repo / "scripts" / "installation" / "install_pyrosetta.py"

    assert staged_file.exists()
    assert os.stat(kept_file).st_ino == os.stat(staged_file).st_ino


def test_git_build_context_excludes_ignored_and_untracked_files(tmp_path: Path) -> None:
    module = _load_script_module()

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo_root, check=True)
    (repo_root / ".gitignore").write_text(
        "weights/\n*.cache\nastrolabe/generated/\n", encoding="utf-8"
    )
    (repo_root / "src.py").write_text("print('tracked')\n", encoding="utf-8")
    (repo_root / "weights").mkdir()
    (repo_root / "weights" / "model.pt").write_bytes(b"large")
    (repo_root / "weights" / "generated.py").write_text("ignored\n", encoding="utf-8")
    (repo_root / "scratch.cache").write_text("cache\n", encoding="utf-8")
    (repo_root / "scratch.py").write_text("scratch\n", encoding="utf-8")
    (repo_root / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    (repo_root / "docs").mkdir()
    (repo_root / "docs" / "guide.md").write_text("docs\n", encoding="utf-8")
    (repo_root / "tests").mkdir()
    (repo_root / "tests" / "test_build.py").write_text("test\n", encoding="utf-8")
    (repo_root / "logs").mkdir()
    (repo_root / "logs" / "generated.py").write_text("generated\n", encoding="utf-8")
    (repo_root / "outputs").mkdir()
    (repo_root / "outputs" / "generated.py").write_text("generated\n", encoding="utf-8")
    (repo_root / "astrolabe" / "generated").mkdir(parents=True)
    (repo_root / "astrolabe" / "generated" / "module.py").write_text(
        "ignored\n", encoding="utf-8"
    )
    task_link = repo_root / ".codex-tasks" / "task" / "links" / "missing"
    task_link.parent.mkdir(parents=True)
    task_link.symlink_to(tmp_path / "missing-output")
    subprocess.run(
        ["git", "add", ".gitignore", "src.py", ".codex-tasks", "docs", "tests"],
        cwd=repo_root,
        check=True,
    )
    (repo_root / ".codex-tasks" / "generated.py").write_text(
        "generated\n", encoding="utf-8"
    )

    tracked = module._tracked_build_context_files(repo_root)
    stage_repo = module._stage_build_context(repo_root, tmp_path / "stage", tracked)

    assert tracked == (Path(".gitignore"), Path("src.py"))
    assert (stage_repo / "src.py").exists()
    assert not (stage_repo / "weights").exists()
    assert not (stage_repo / "scratch.cache").exists()
    assert not (stage_repo / "scratch.py").exists()
    assert not (stage_repo / "untracked.txt").exists()
    assert not (stage_repo / ".codex-tasks").exists()
    assert not (stage_repo / "docs").exists()
    assert not (stage_repo / "tests").exists()


@pytest.mark.parametrize(
    "offender",
    (
        Path("astrolabe/initializers/rfd3_spec.py"),
        Path("astrolabe/generated module.py"),
        Path("run.py"),
    ),
)
def test_git_build_context_rejects_untracked_runtime_python_sources(
    tmp_path: Path, offender: Path
) -> None:
    module = _load_script_module()

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo_root, check=True)
    (repo_root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo_root, check=True)
    source = repo_root / offender
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("runtime source\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        module._tracked_build_context_files(repo_root)

    message = str(exc_info.value)
    assert "Untracked runtime Python source files" in message
    assert str(repo_root.resolve()) in message
    assert str(offender) in message
    assert "astrolabe/**/*.py and run.py" in message


def test_build_context_excludes_optional_external_backends() -> None:
    module = _load_script_module()

    for path in (
        "vendor/dplm/src/dplm.py",
        "vendor/na_mpnn/run.py",
        "vendor/rpxdock/rpxdock/__init__.py",
        "vendor/protein_sequence_models/sequence_models/__init__.py",
        "vendor/sru/sru/__init__.py",
        "vendor/surfdesign/src/__init__.py",
        "vendor/surfpro/fairseq/__init__.py",
        "vendor/willutil/willutil/__init__.py",
    ):
        assert module._is_build_context_excluded(Path(path))
    for path in (
        "vendor/alphafold2/alphafold/__init__.py",
        "vendor/alphafold3/src/alphafold3/__init__.py",
        "vendor/esm2/esm2/__init__.py",
        "vendor/openfold_esm2/openfold/model/model.py",
        "vendor/protenix/protenix/__init__.py",
        "scripts/containers/apptainer_build.py",
        "scripts/install.py",
    ):
        assert not module._is_build_context_excluded(Path(path))


def test_ananas_build_input_validates_archive_and_inner_binary(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    archive = tmp_path / module.ANANAS_ARCHIVE_NAME
    payload = b"x" * 2_508_152
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("ananas", payload)
    monkeypatch.setattr(module, "ANANAS_ARCHIVE_SHA256", module._sha256_file(archive))
    monkeypatch.setattr(
        module, "ANANAS_BINARY_SHA256", hashlib.sha256(payload).hexdigest()
    )

    assert module._validate_ananas_archive(archive) == archive
    monkeypatch.setattr(module, "ANANAS_BINARY_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="binary checksum mismatch"):
        module._validate_ananas_archive(archive)


def test_rosetta_build_input_requires_exact_manifest_file_set(tmp_path: Path) -> None:
    module = _load_script_module()
    repo = tmp_path / "repo"
    manifest = repo / "container" / "rosetta-tools.manifest.tsv"
    manifest.parent.mkdir(parents=True)
    payload = b"print('molfile')\n"
    rel_path = "source/scripts/python/public/molfile_to_params.py"
    manifest.write_text(
        "sha256\tsize_bytes\tpath\n"
        f"{hashlib.sha256(payload).hexdigest()}\t{len(payload)}\t{rel_path}\n",
        encoding="utf-8",
    )
    archive = tmp_path / "rosetta-tools.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo(rel_path)
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))

    assert module._validate_rosetta_tools_archive(archive, repo) == archive
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo(rel_path)
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
        extra = tarfile.TarInfo("unexpected.txt")
        extra.size = 1
        handle.addfile(extra, io.BytesIO(b"x"))
    with pytest.raises(SystemExit, match="non-exact file set"):
        module._validate_rosetta_tools_archive(archive, repo)


@pytest.mark.parametrize(
    ("member_type", "member_name", "error"),
    [
        (tarfile.SYMTYPE, "unsafe-link", "unsafe/non-regular"),
        (tarfile.LNKTYPE, "unsafe-hardlink", "unsafe/non-regular"),
        (tarfile.CHRTYPE, "unsafe-device", "unsafe/non-regular"),
        (tarfile.REGTYPE, "/absolute", "unsafe/non-regular"),
        (tarfile.REGTYPE, "../traversal", "unsafe/non-regular"),
        (tarfile.REGTYPE, None, "duplicate"),
    ],
)
def test_rosetta_build_input_rejects_unsafe_archive_members(
    tmp_path: Path,
    member_type: bytes,
    member_name: str | None,
    error: str,
) -> None:
    module = _load_script_module()
    repo = tmp_path / "repo"
    manifest = repo / "container" / "rosetta-tools.manifest.tsv"
    manifest.parent.mkdir(parents=True)
    payload = b"print('molfile')\n"
    rel_path = "source/scripts/python/public/molfile_to_params.py"
    manifest.write_text(
        "sha256\tsize_bytes\tpath\n"
        f"{hashlib.sha256(payload).hexdigest()}\t{len(payload)}\t{rel_path}\n",
        encoding="utf-8",
    )
    archive = tmp_path / "rosetta-tools.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        required = tarfile.TarInfo(rel_path)
        required.size = len(payload)
        handle.addfile(required, io.BytesIO(payload))
        bad = tarfile.TarInfo(member_name or rel_path)
        bad.type = member_type
        if member_type in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
            bad.linkname = "../outside"
            handle.addfile(bad)
        elif member_type == tarfile.CHRTYPE:
            handle.addfile(bad)
        else:
            bad.size = len(payload)
            handle.addfile(bad, io.BytesIO(payload))

    with pytest.raises(SystemExit, match=error):
        module._validate_rosetta_tools_archive(archive, repo)


def test_submodule_context_excludes_only_known_transients() -> None:
    module = _load_script_module()

    for path in (
        "alphafold/__pycache__/model.cpython-311.pyc",
        ".pytest_cache/v/cache/nodeids",
        "package.egg-info/PKG-INFO",
        "build/lib/package.py",
        "dist/package.whl",
        "tests/test_model.py",
        "docs/index.md",
        "attn_core_inplace_cuda.cpython-311-x86_64-linux-gnu.so",
    ):
        assert module._is_submodule_transient(Path(path))
    for path in (
        "esm3/esm3/__init__.py",
        "src/package/build/model.py",
    ):
        assert not module._is_submodule_transient(Path(path))


def test_build_context_fingerprint_tracks_only_staged_files(tmp_path: Path) -> None:
    module = _load_script_module()

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo_root, check=True)
    tracked_file = repo_root / "tracked.txt"
    ignored_file = repo_root / "ignored.bin"
    (repo_root / ".gitignore").write_text("ignored.bin\n", encoding="utf-8")
    tracked_file.write_text("v1\n", encoding="utf-8")
    ignored_file.write_text("ignored-v1\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore", "tracked.txt"], cwd=repo_root, check=True)

    tracked = module._tracked_build_context_files(repo_root)
    fingerprint_v1 = module._build_context_fingerprint(repo_root, tracked)
    ignored_file.write_text("ignored-v2\n", encoding="utf-8")
    assert module._build_context_fingerprint(repo_root, tracked) == fingerprint_v1

    tracked_file.write_text("v2\n", encoding="utf-8")
    assert module._build_context_fingerprint(repo_root, tracked) != fingerprint_v1


def test_build_context_fingerprint_reuses_unchanged_file_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    source = repo_root / "source.py"
    source.write_text("v1\n", encoding="utf-8")
    cache = {}
    calls = []
    original_sha256 = module._sha256_file

    def counted_sha256(path):
        calls.append(Path(path))
        return original_sha256(path)

    monkeypatch.setattr(module, "_sha256_file", counted_sha256)
    first = module._build_context_fingerprint(
        repo_root, (Path("source.py"),), cache
    )
    second = module._build_context_fingerprint(
        repo_root, (Path("source.py"),), cache
    )
    source.write_text("v2\n", encoding="utf-8")
    third = module._build_context_fingerprint(
        repo_root, (Path("source.py"),), cache
    )

    assert first == second
    assert third != first
    assert calls == [source, source]


def test_cached_sha256_rehashes_replaced_inode(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    source = tmp_path / "source.py"
    source.write_text("v1\n", encoding="utf-8")
    cache = {}
    first = module._cached_sha256(source, cache)
    original_stat = source.stat()
    replacement = tmp_path / "replacement.py"
    replacement.write_text("v2\n", encoding="utf-8")
    os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    os.replace(replacement, source)

    second = module._cached_sha256(source, cache)

    assert second != first
    assert cache[str(source.resolve())]["inode"] == source.stat().st_ino


def test_main_creates_missing_state_root_before_staging(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()

    home_dir = tmp_path / "home"
    home_dir.mkdir()
    def_file = tmp_path / "Astrolabe.af3.def"
    def_file.write_text("Bootstrap: docker\n", encoding="utf-8")
    output_path = tmp_path / "images" / "Astrolabe_af3.sif"
    fakeroot_bin = tmp_path / "fakeroot" / "bin" / "fakeroot"
    fakeroot_bin.parent.mkdir(parents=True)
    fakeroot_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    fakeroot_bin.chmod(0o755)
    seen_env = {}

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setattr(module, "_detect_apptainer", lambda: "/usr/bin/apptainer")
    monkeypatch.setattr(
        module,
        "_executable_identity",
        lambda _executable, _cache=None: {
            "path": "/usr/bin/apptainer",
            "sha256": "a" * 64,
            "version": "apptainer version 1.5.2",
        },
    )
    monkeypatch.setattr(module, "_inspect_image", lambda **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_stage_build_context",
        lambda _repo_root, stage_root, _tracked_files: Path(stage_root),
    )

    def _fake_run(cmd, cwd=None, env=None):
        del cwd
        seen_env.update(env or {})
        Path(cmd[-2]).write_text("stub sif\n", encoding="utf-8")

    monkeypatch.setattr(module, "_run", _fake_run)

    rc = module.main(
        [
            "--def",
            str(def_file),
            "--output",
            str(output_path),
            "--name",
            "first-run",
            "--fakeroot",
            "--fakeroot-bin",
            str(fakeroot_bin),
            "--without-pyrosetta",
        ]
    )

    state_root = tmp_path / ".gh_container_setup" / "first-run"
    assert rc == 0
    assert state_root.is_dir()
    assert (state_root / "state.json").exists()
    assert output_path.exists()
    assert seen_env["PATH"].split(os.pathsep)[:2] == [
        "/usr/bin",
        str(fakeroot_bin.parent),
    ]


def test_main_reuses_only_matching_build_fingerprint(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()

    def_file = tmp_path / "Astrolabe.af3.def"
    def_file.write_text("Bootstrap: scratch\n", encoding="utf-8")
    output_path = tmp_path / "images" / "Astrolabe_af3.sif"
    state_root = tmp_path / "state"
    build_input = tmp_path / "ccd.pickle"
    build_input.write_bytes(b"ccd-v1")
    apptainer = tmp_path / "apptainer"
    apptainer.write_bytes(b"apptainer-v1")
    apptainer.chmod(0o755)
    calls = []

    monkeypatch.setattr(module, "_detect_apptainer", lambda: str(apptainer))
    monkeypatch.setattr(
        module,
        "_executable_version",
        lambda _executable: "apptainer version 1.5.2",
    )
    monkeypatch.setattr(module, "_inspect_image", lambda **_kwargs: None)
    monkeypatch.setattr(module, "_tracked_build_context_files", lambda _repo_root: (Path("setup.py"),))
    monkeypatch.setattr(
        module,
        "_build_context_fingerprint",
        lambda _repo_root, _files, _cache=None: "1" * 64,
    )
    monkeypatch.setattr(
        module,
        "_stage_build_context",
        lambda _repo_root, stage_root, _files: Path(stage_root),
    )

    def fake_run(cmd, cwd=None, env=None):
        del cwd, env
        calls.append(list(cmd))
        Path(cmd[-2]).write_text("stub sif\n", encoding="utf-8")

    monkeypatch.setattr(module, "_run", fake_run)
    argv = [
        "--def",
        str(def_file),
        "--output",
        str(output_path),
        "--state-root",
        str(state_root),
        "--build-input",
        str(build_input),
        "--without-pyrosetta",
    ]

    assert module.main(argv) == 0
    assert len(calls) == 1
    assert module.main(argv) == 0
    assert len(calls) == 1

    apptainer.write_bytes(b"apptainer-v2")
    assert module.main(argv) == 0
    assert len(calls) == 2
    state = json.loads((state_root / "state.json").read_text(encoding="utf-8"))
    assert state["build_identity"]["apptainer_binary"] == {
        "path": str(apptainer.resolve()),
        "sha256": hashlib.sha256(b"apptainer-v2").hexdigest(),
        "version": "apptainer version 1.5.2",
    }

    build_input.write_bytes(b"ccd-v2")
    assert module.main(argv) == 0
    assert len(calls) == 3

    def_file.write_text("Bootstrap: scratch\n# changed\n", encoding="utf-8")
    assert module.main(argv) == 0
    assert len(calls) == 4


def test_pyrosetta_default_build_feature_is_fingerprinted(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    def_file = tmp_path / "Astrolabe.af3.def"
    def_file.write_text("Bootstrap: scratch\n", encoding="utf-8")
    output_path = tmp_path / "images" / "gh-pyrosetta.sif"
    wheel = tmp_path / module.PYROSETTA_WHEEL_NAME
    wheel.write_bytes(b"approved-wheel")
    monkeypatch.setattr(
        module,
        "PYROSETTA_ALLOWED_SHA256",
        {module._sha256_file(wheel)},
    )
    monkeypatch.setattr(module, "PYROSETTA_WHEEL_SIZE", wheel.stat().st_size)
    state_root = tmp_path / "state"
    build_cache = tmp_path / "build-cache"
    calls = []
    build_arg_payloads = []
    monkeypatch.setattr(module, "_detect_apptainer", lambda: "/usr/bin/apptainer")
    monkeypatch.setattr(
        module,
        "_executable_identity",
        lambda _executable, _cache=None: {
            "path": "/usr/bin/apptainer",
            "sha256": "a" * 64,
            "version": "apptainer version 1.5.2",
        },
    )
    monkeypatch.setattr(module, "_inspect_image", lambda **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_tracked_build_context_files",
        lambda _repo_root: (Path("setup.py"),),
    )
    monkeypatch.setattr(
        module,
        "_build_context_fingerprint",
        lambda _repo_root, _files, _cache=None: "1" * 64,
    )
    monkeypatch.setattr(
        module,
        "_stage_build_context",
        lambda _repo_root, stage_root, _files: Path(stage_root),
    )

    def fake_run(cmd, cwd=None, env=None):
        del cwd, env
        calls.append(list(cmd))
        build_arg_index = cmd.index("--build-arg-file")
        build_arg_payloads.append(
            Path(cmd[build_arg_index + 1]).read_text(encoding="utf-8")
        )
        Path(cmd[-2]).write_text("stub sif\n", encoding="utf-8")

    monkeypatch.setattr(module, "_run", fake_run)
    assert module.main(
        [
            "--def",
            str(def_file),
            "--output",
            str(output_path),
            "--state-root",
            str(state_root),
            "--accept-license",
            "pyrosetta-license",
            "--build-cache",
            str(build_cache),
            "--build-input",
            str(wheel),
        ]
    ) == 0

    assert "--build-arg-file" in calls[0]
    assert "GH_INSTALL_PYROSETTA=1\n" in build_arg_payloads[0]
    assert "GH_BUILD_FEATURES=pyrosetta\n" in build_arg_payloads[0]
    state = json.loads((state_root / "state.json").read_text(encoding="utf-8"))
    assert state["build_identity"]["build_features"] == ["pyrosetta"]
    assert state["build_identity"]["schema_version"] == 5
    assert state["build_identity"]["accepted_licenses"] == ["pyrosetta-license"]
    image_metadata = json.loads(
        Path(str(output_path) + ".build.json").read_text(encoding="utf-8")
    )
    assert image_metadata["build_features"] == ["pyrosetta"]
    assert image_metadata["build_contract"] == state["build_identity"]
    assert image_metadata["build_fingerprint"] == state["build_fingerprint"]
    staged_wheel = build_cache / "wheels" / "pyrosetta" / module.PYROSETTA_WHEEL_NAME
    assert staged_wheel.read_bytes() == b"approved-wheel"
    assert state["build_identity"]["external_inputs"][0]["path"] == str(
        staged_wheel.resolve()
    )


def test_build_arg_file_preserves_multiple_features(tmp_path: Path) -> None:
    module = _load_script_module()
    path = module._write_build_arg_file(
        tmp_path / "build-args.env",
        ("rosetta", "ananas", "pyrosetta"),
    )

    assert path.read_text(encoding="utf-8").splitlines() == [
        "GH_INSTALL_PYROSETTA=1",
        "GH_INSTALL_ANANAS=1",
        "GH_INSTALL_ROSETTA_TOOLS=1",
        "GH_BUILD_FEATURES=ananas,pyrosetta,rosetta",
    ]


def test_pyrosetta_builder_rejects_missing_license_or_verified_wheel(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    monkeypatch.setattr(module, "_detect_apptainer", lambda: "/usr/bin/apptainer")
    with pytest.raises(SystemExit, match="pyrosetta-license"):
        module.main([])
    with pytest.raises(SystemExit, match="exactly one verified"):
        module.main(
            [
                "--accept-license",
                "pyrosetta-license",
            ]
        )
    wheel = tmp_path / module.PYROSETTA_WHEEL_NAME
    wheel.write_bytes(b"approved-wheel")
    monkeypatch.setattr(
        module,
        "PYROSETTA_ALLOWED_SHA256",
        {module._sha256_file(wheel)},
    )
    monkeypatch.setattr(module, "PYROSETTA_WHEEL_SIZE", wheel.stat().st_size)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(SystemExit, match="--build-cache"):
        module.main(
            [
                "--accept-license",
                "pyrosetta-license",
                "--build-input",
                str(wheel),
                "--skip-build",
            ]
        )
    with pytest.raises(SystemExit, match="Astrolabe_af3.sif"):
        module.main(
            [
                "--accept-license",
                "pyrosetta-license",
                "--build-cache",
                str(tmp_path / "build-cache"),
                "--build-input",
                str(wheel),
                "--skip-build",
            ]
        )


def test_record_image_identity_always_hashes_image_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    image = tmp_path / "image.sif"
    image.write_bytes(b"new-bytes")
    stat_result = image.stat()
    state = {
        "output_identity": {
            "path": str(image.resolve()),
            "size": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
            "sha256": "0" * 64,
        }
    }
    calls = []
    original = module._sha256_file

    def counted(path):
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(module, "_sha256_file", counted)
    identity = module._record_image_identity(state, image)

    assert calls == [image.resolve()]
    assert identity["sha256"] == hashlib.sha256(b"new-bytes").hexdigest()
    assert identity["sha256"] != "0" * 64


def test_record_image_identity_rejects_existing_bad_provenance(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    image = tmp_path / "image.sif"
    image.write_bytes(b"image-bytes")
    sidecar = Path(str(image) + ".sha256")
    metadata = Path(str(image) + ".build.json")
    sidecar.write_text("%s  image.sif\n" % ("0" * 64), encoding="ascii")

    with pytest.raises(SystemExit, match="checksum sidecar mismatch"):
        module._record_image_identity({}, image)

    actual = module._sha256_file(image)
    sidecar.write_text("%s  image.sif\n" % actual, encoding="ascii")
    metadata.write_text("[]\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="build metadata does not match"):
        module._record_image_identity({}, image)

    identity = module._record_image_identity(
        {}, image, replace_provenance=True
    )
    assert json.loads(metadata.read_text(encoding="utf-8"))["sha256"] == actual
    assert identity["sha256"] == actual


def test_record_image_identity_rejects_tampered_build_contract(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    image = tmp_path / "image.sif"
    image.write_bytes(b"image-bytes")
    contract = {
        "schema_version": 5,
        "context_sha256": "1" * 64,
        "definition_sha256": "2" * 64,
        "external_inputs": [],
        "build_features": [],
        "accepted_licenses": [],
    }
    fingerprint = module._build_contract_fingerprint(contract)
    module._record_image_identity(
        {},
        image,
        build_contract=contract,
        build_fingerprint=fingerprint,
        replace_provenance=True,
    )
    metadata_path = Path(str(image) + ".build.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["build_contract"]["definition_sha256"] = "3" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(SystemExit, match="contract fingerprint mismatch"):
        module._record_image_identity(
            {},
            image,
            build_contract=contract,
            build_fingerprint=fingerprint,
        )


@pytest.mark.parametrize("suffix", [".sha256", ".build.json"])
def test_record_image_identity_rejects_provenance_directory(
    tmp_path: Path, suffix: str,
) -> None:
    module = _load_script_module()
    image = tmp_path / "image.sif"
    image.write_bytes(b"image-bytes")
    Path(str(image) + suffix).mkdir()

    with pytest.raises(SystemExit, match="not a regular file"):
        module._record_image_identity({}, image)


def test_write_python_runner_is_executable_and_parameterized(tmp_path: Path) -> None:
    module = _load_script_module()

    runner_path = tmp_path / "gh_site_container_python.py"
    module._write_python_runner(
        runner_path,
        apptainer="/opt/apptainer/bin/apptainer",
        sif_path="/images/gh.sif",
        container_repo_root="/opt/Astrolabe",
        binds=[
            "/models/af3:/models/alphafold3",
            "/tools/libgfortran3:/host-libs/libgfortran3",
        ],
        env_defaults={
            "GH_WEIGHT_AF3": "/models/alphafold3",
            "GH_MEEKO_PYTHONPATH": "/opt/tools/lib/python3.11/site-packages",
        },
        env_prepends=[("LD_LIBRARY_PATH", "/host-libs/libgfortran3")],
        container_python="/alphafold3_venv/bin/python",
        use_nv=True,
    )

    text = runner_path.read_text(encoding="utf-8")
    assert os.access(runner_path, os.X_OK)
    assert "APPTAINER_BIN = '/opt/apptainer/bin/apptainer'" in text
    assert "SIF_PATH = '/images/gh.sif'" in text
    assert "CONTAINER_REPO_ROOT = '/opt/Astrolabe'" in text
    assert "CONTAINER_PYTHON = '/alphafold3_venv/bin/python'" in text
    assert "USE_NV = True" in text
    assert "'/models/af3:/models/alphafold3'" in text
    assert "'GH_WEIGHT_AF3': '/models/alphafold3'" in text
    assert "('LD_LIBRARY_PATH', '/host-libs/libgfortran3')" in text
    assert '"GH_DPLM_PYTHON"' in text
    assert '"GH_CUDA_PREINITIALIZE"' in text
    assert '"HUGGINGFACE_HUB_CACHE"' in text
    assert '"HF_TOKEN"' in text
    assert '"HUGGING_FACE_HUB_TOKEN"' in text
    assert "env[f\"APPTAINERENV_{key}\"] = env[key]" in text
    assert '"--containall"' in text
    assert '"--no-mount"' in text
    assert 'source_map_value = host_repo_value or os.environ.get("GH_HOST_REPO_MAP_ROOT")' in text
    assert "env[key] = value" in text
    compile(text, str(runner_path), "exec")


def test_python_runner_preserves_child_error_without_traceback(tmp_path: Path) -> None:
    module = _load_script_module()
    engine = tmp_path / "apptainer"
    engine.write_text("#!/bin/sh\necho 'invalid input: choose an existing file' >&2\nexit 2\n")
    engine.chmod(0o755)
    runner_path = tmp_path / "runner.py"
    module.write_python_runner(runner_path, apptainer=str(engine), sif_path="unused.sif", use_nv=False)
    result = subprocess.run([sys.executable, str(runner_path), "-c", "pass"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "invalid input" in result.stderr
    assert "Traceback" not in result.stderr


def test_python_runner_defaults_to_sif_source_with_mapping_only_repo_root(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    runner_path = tmp_path / "gh_site_container_python.py"
    module._write_python_runner(
        runner_path,
        apptainer="/opt/apptainer/bin/apptainer",
        sif_path="/images/gh.sif",
        container_repo_root="/opt/Astrolabe",
        binds=[],
        env_defaults={
            "GH_WEIGHT_AF3": "/models/alphafold3",
            "HOME": "/cache/gh-runtime/home",
        },
        env_prepends=[],
        container_python="python",
        use_nv=False,
    )
    spec = importlib.util.spec_from_file_location("_generated_sif_runner", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    calls = []
    monkeypatch.delenv("GH_HOST_REPO_ROOT", raising=False)
    monkeypatch.setenv("GH_HOST_REPO_MAP_ROOT", str(repo_root))
    monkeypatch.setenv("HOME", str(tmp_path / "host-home"))
    monkeypatch.setenv("GH_WEIGHT_AF3", str(tmp_path / "ambient-af3"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(
        runner.subprocess,
        "check_call",
        lambda cmd, env=None: calls.append((list(cmd), env)),
    )
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            str(runner_path),
            "-c",
            f"import sys; sys.path.insert(0, {str(repo_root)!r}); print('ok')",
        ],
    )

    assert runner.main() == 0
    cmd, env = calls[0]
    assert cmd[:10] == [
        "/opt/apptainer/bin/apptainer",
        "exec",
        "--containall",
        "--cleanenv",
        "--no-home",
        "--no-mount",
        "cwd",
        "--pwd",
        "/opt/Astrolabe",
        "/images/gh.sif",
    ]
    assert cmd[10:13] == [
        "/usr/bin/env",
        "HOME=/cache/gh-runtime/home",
        "python",
    ]
    assert f"{repo_root.resolve()}:/opt/Astrolabe" not in cmd
    assert str(repo_root) not in cmd[-1]
    assert "/opt/Astrolabe" in cmd[-1]
    assert env["APPTAINERENV_GH_WEIGHT_AF3"] == "/models/alphafold3"
    assert env["APPTAINERENV_HF_HUB_OFFLINE"] == "0"
    assert "APPTAINERENV_HOME" not in env


def test_write_python_runner_forwards_external_envs_and_binds_paths(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()

    runner_path = tmp_path / "gh_site_container_python.py"
    dplm_python = tmp_path / "envs" / "dplm" / "bin" / "python"
    dplm_python.parent.mkdir(parents=True)
    dplm_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    dplm_repo = tmp_path / "repo" / "vendor" / "dplm"
    dplm_repo.mkdir(parents=True)
    hf_cache = tmp_path / "hf" / "hub"
    hf_cache.mkdir(parents=True)
    missing_rpxdock_python = tmp_path / "missing" / "rpxdock" / "bin" / "python"

    module._write_python_runner(
        runner_path,
        apptainer="/opt/apptainer/bin/apptainer",
        sif_path="/images/gh.sif",
        container_repo_root="/opt/Astrolabe",
        binds=[],
        env_defaults={},
        env_prepends=[],
        container_python="python",
        use_nv=False,
    )

    spec = importlib.util.spec_from_file_location("_generated_runner", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    calls = []

    def fake_check_call(cmd, env=None):
        calls.append((cmd, env))

    monkeypatch.setenv("GH_DPLM_PYTHON", str(dplm_python))
    monkeypatch.setenv("GH_DPLM_REPO", str(dplm_repo))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hf_cache))
    monkeypatch.setenv("GH_RPXDOCK_PYTHON", str(missing_rpxdock_python))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("GH_CUDA_PREINITIALIZE", "0")
    monkeypatch.setenv("HF_TOKEN", "secret-native-token")
    monkeypatch.setenv("GH_HOST_REPO_ROOT", str(tmp_path / "repo"))
    monkeypatch.setattr(runner.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(runner.sys, "argv", [str(runner_path), "-c", "print('ok')"])

    assert runner.main() == 0
    cmd, env = calls[0]

    assert f"{(tmp_path / 'repo').resolve()}:/opt/Astrolabe" in cmd
    assert f"{dplm_python.parent.parent.resolve()}:{dplm_python.parent.parent.resolve()}" in " ".join(cmd)
    assert f"{dplm_repo.resolve()}:{dplm_repo.resolve()}" in " ".join(cmd)
    assert f"{hf_cache.resolve()}:{hf_cache.resolve()}" in " ".join(cmd)
    assert str(missing_rpxdock_python) not in " ".join(cmd)
    assert env["APPTAINERENV_GH_DPLM_PYTHON"] == str(dplm_python)
    assert env["APPTAINERENV_GH_DPLM_REPO"] == str(dplm_repo)
    assert env["APPTAINERENV_HUGGINGFACE_HUB_CACHE"] == str(hf_cache)
    assert env["APPTAINERENV_GH_RPXDOCK_PYTHON"] == str(missing_rpxdock_python)
    assert env["APPTAINERENV_HF_HUB_OFFLINE"] == "1"
    assert env["APPTAINERENV_GH_CUDA_PREINITIALIZE"] == "0"
    assert env["APPTAINERENV_HF_TOKEN"] == "secret-native-token"


def test_write_docker_apptainer_runner_is_executable_and_parameterized(tmp_path: Path) -> None:
    module = _load_script_module()

    runner_path = tmp_path / "gh_site_container_python.py"
    module.write_python_runner(
        runner_path,
        apptainer="apptainer",
        runner_backend="docker-apptainer",
        docker_bin="/usr/bin/docker",
        docker_image="ghcr.io/apptainer/apptainer:1.4.5",
        sif_path="/images/gh.sif",
        container_repo_root="/opt/Astrolabe",
        binds=["/models/af3:/models/alphafold3"],
        env_defaults={"GH_WEIGHT_AF3": "/models/alphafold3"},
        env_prepends=[("LD_LIBRARY_PATH", "/host-libs/libgfortran3")],
        container_python="/alphafold3_venv/bin/python",
        use_nv=True,
    )

    text = runner_path.read_text(encoding="utf-8")
    assert os.access(runner_path, os.X_OK)
    assert "DOCKER_BIN = '/usr/bin/docker'" in text
    assert "DOCKER_IMAGE = 'ghcr.io/apptainer/apptainer:1.4.5'" in text
    assert '"apptainer",\n        "exec",\n        "--containall"' in text
    assert 'docker_cmd = [DOCKER_BIN, "run", "--rm", "--privileged"]' in text
    assert 'docker_cmd += ["-w", "/tmp", DOCKER_IMAGE, *inner_cmd]' in text
    assert 'os.environ.get("GH_DOCKER_GPUS", "all" if USE_NV else "")' in text
    assert "'/models/af3:/models/alphafold3'" in text
    assert '"GH_CUDA_PREINITIALIZE"' in text
    assert '"HF_TOKEN"' in text
    assert '"HUGGING_FACE_HUB_TOKEN"' in text
    assert 'docker_cmd += ["-e", f"APPTAINERENV_{key}"]' in text
    assert "env[key] = value" in text
    compile(text, str(runner_path), "exec")


def test_docker_apptainer_runner_preserves_cuda_preinitialize_override(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    runner_path = tmp_path / "gh_site_container_python.py"
    module.write_python_runner(
        runner_path,
        apptainer="apptainer",
        runner_backend="docker-apptainer",
        docker_bin="/usr/bin/docker",
        docker_image="sha256:" + "1" * 64,
        sif_path="/images/gh.sif",
        container_repo_root="/opt/Astrolabe",
        binds=[],
        env_defaults={"GH_CUDA_PREINITIALIZE": "1"},
        env_prepends=[],
        container_python="python",
        use_nv=True,
    )

    spec = importlib.util.spec_from_file_location("_generated_docker_runner", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    calls = []
    monkeypatch.setenv("GH_CUDA_PREINITIALIZE", "0")
    monkeypatch.setenv("GH_DOCKER_GPUS", "none")
    monkeypatch.setattr(
        runner.subprocess,
        "check_call",
        lambda cmd, env=None: calls.append((list(cmd), env)),
    )
    monkeypatch.setattr(runner.sys, "argv", [str(runner_path), "-c", "print('ok')"])

    assert runner.main() == 0
    cmd, env = calls[0]
    assert "--gpus" not in cmd
    assert "-e" in cmd
    assert "APPTAINERENV_GH_CUDA_PREINITIALIZE" in cmd
    assert env["APPTAINERENV_GH_CUDA_PREINITIALIZE"] == "0"


def test_docker_builder_identity_pins_binary_image_and_apptainer(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    docker = tmp_path / "docker"
    docker.write_bytes(b"docker-binary")
    docker.chmod(0o755)
    image = "ghcr.io/apptainer/apptainer:1.4.5"
    commands = []
    output_commands = []
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, cwd=None, env=None: commands.append(cmd),
    )

    def fake_output(cmd, **_kwargs):
        output_commands.append(cmd)
        if cmd[1:3] == ["image", "inspect"]:
            return json.dumps(
                [
                    {
                        "Id": "sha256:image-id",
                        "RepoDigests": [
                            "ghcr.io/apptainer/apptainer@sha256:immutable"
                        ],
                    }
                ]
            )
        if cmd[1:] == ["--version"]:
            return "Docker version 28.0\n"
        return "apptainer version 1.4.5\n"

    monkeypatch.setattr(module.subprocess, "check_output", fake_output)

    identity = module._docker_builder_identity(str(docker), image)

    assert commands == []
    assert identity["image_id"] == "sha256:image-id"
    assert identity["execution_image"] == "sha256:image-id"
    assert identity["repo_digests"] == [
        "ghcr.io/apptainer/apptainer@sha256:immutable"
    ]
    assert identity["docker_binary_sha256"] == hashlib.sha256(
        b"docker-binary"
    ).hexdigest()
    assert identity["apptainer_version"] == "apptainer version 1.4.5"
    assert [
        str(docker),
        "run",
        "--rm",
        "sha256:image-id",
        "apptainer",
        "version",
    ] in output_commands


def test_docker_builder_identity_pulls_only_when_local_image_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    docker = tmp_path / "docker"
    docker.write_bytes(b"docker-binary")
    docker.chmod(0o755)
    image = "ghcr.io/apptainer/apptainer:1.4.5"
    pulls = []
    inspect_calls = 0
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, cwd=None, env=None: pulls.append(cmd),
    )

    def fake_output(cmd, **_kwargs):
        nonlocal inspect_calls
        if cmd[1:3] == ["image", "inspect"]:
            inspect_calls += 1
            if inspect_calls == 1:
                raise subprocess.CalledProcessError(1, cmd)
            return json.dumps([{"Id": "sha256:pulled", "RepoDigests": []}])
        if cmd[1:] == ["--version"]:
            return "Docker version 28.0\n"
        return "apptainer version 1.4.5\n"

    monkeypatch.setattr(module.subprocess, "check_output", fake_output)

    identity = module._docker_builder_identity(str(docker), image)

    assert pulls == [[str(docker), "pull", image]]
    assert inspect_calls == 2
    assert identity["execution_image"] == "sha256:pulled"


def test_docker_builder_mounts_stage_repo_and_output(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()

    repo_root = tmp_path / "repo"
    stage_repo = tmp_path / "stage" / "repo"
    out_path = tmp_path / "images" / "Astrolabe_af3.sif.tmp"
    def_file = repo_root / "container" / "Astrolabe.af3.def"
    def_file.parent.mkdir(parents=True)
    def_file.write_text("Bootstrap: scratch\n", encoding="utf-8")
    stage_repo.mkdir(parents=True)
    out_path.parent.mkdir(parents=True)
    commands = []

    def fake_run(cmd, cwd=None):
        commands.append((cmd, cwd))

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_docker_chown_build_artifacts", lambda **_kwargs: None)

    module._build_with_docker(
        docker_bin="/usr/bin/docker",
        docker_image="ghcr.io/apptainer/apptainer:1.4.5",
        docker_base_image=None,
        docker_base_source="docker",
        docker_network="host",
        build_cache=None,
        stage_repo=stage_repo,
        tmp_path=out_path,
        def_file=def_file,
        repo_root=repo_root,
    )

    cmd, cwd = commands[0]
    assert cwd is None
    assert cmd[:4] == ["/usr/bin/docker", "run", "--rm", "--privileged"]
    assert "--network" in cmd
    assert "host" in cmd
    assert f"{stage_repo.resolve()}:/work" in cmd
    assert f"{out_path.parent.resolve()}:/out" in cmd
    assert f"{stage_repo.parent / 'docker-build-work'}:/apptainer-work" in cmd
    assert "APPTAINER_CACHEDIR=/apptainer-work/apptainer_cache" in cmd
    assert "APPTAINER_TMPDIR=/apptainer-work/apptainer_tmp" in cmd
    assert not any(str(part).startswith("APPTAINER_TMPDIR=/work/") for part in cmd)
    assert ["apptainer", "build"] == cmd[
        cmd.index("apptainer") : cmd.index("apptainer") + 2
    ]
    build_arg_index = cmd.index("--build-arg-file")
    assert cmd[build_arg_index + 1] == "/apptainer-work/build-args.env"
    assert "GH_INSTALL_PYROSETTA=0\n" in (
        stage_repo.parent / "docker-build-work" / "build-args.env"
    ).read_text(encoding="utf-8")
    assert cmd[-2:] == [
        f"/out/{out_path.name}",
        "/work/container/Astrolabe.af3.def",
    ]


def test_prepare_docker_archive_base_rewrites_staged_definition(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()

    repo_root = tmp_path / "repo"
    stage_repo = tmp_path / "stage" / "repo"
    def_file = repo_root / "container" / "Astrolabe.af3.def"
    staged_def = stage_repo / "container" / "Astrolabe.af3.def"
    commands = []

    def_file.parent.mkdir(parents=True)
    staged_def.parent.mkdir(parents=True)
    text = "Bootstrap: docker\nFrom: nvidia/cuda:test\n%post\n    true\n"
    def_file.write_text(text, encoding="utf-8")
    staged_def.write_text(text, encoding="utf-8")

    def fake_run(cmd, cwd=None):
        commands.append(cmd)
        if cmd[:2] == ["/usr/bin/docker", "save"]:
            Path(cmd[cmd.index("-o") + 1]).write_text("archive\n", encoding="utf-8")

    monkeypatch.setattr(module, "_run", fake_run)

    module._prepare_docker_archive_base(
        docker_bin="/usr/bin/docker",
        archive_dir=tmp_path / "archive",
        container_archive_path="/base/base.tar",
        stage_repo=stage_repo,
        def_file=def_file,
        repo_root=repo_root,
        base_image=None,
    )

    rewritten = staged_def.read_text(encoding="utf-8")
    assert "Bootstrap: docker-archive" in rewritten
    assert "From: /base/base.tar" in rewritten
    assert commands[0] == ["/usr/bin/docker", "pull", "nvidia/cuda:test"]
    assert commands[1][:3] == ["/usr/bin/docker", "save", "-o"]
    assert str(stage_repo) not in commands[1]


def test_docker_builder_default_keeps_official_docker_base(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()

    repo_root = tmp_path / "repo"
    stage_repo = tmp_path / "stage" / "repo"
    out_path = tmp_path / "images" / "Astrolabe_af3.sif.tmp"
    def_file = repo_root / "container" / "Astrolabe.af3.def"
    staged_def = stage_repo / "container" / "Astrolabe.af3.def"
    commands = []
    text = "Bootstrap: docker\nFrom: nvidia/cuda:test\n%post\n    true\n"

    def_file.parent.mkdir(parents=True)
    staged_def.parent.mkdir(parents=True)
    out_path.parent.mkdir(parents=True)
    def_file.write_text(text, encoding="utf-8")
    staged_def.write_text(text, encoding="utf-8")

    def fake_run(cmd, cwd=None):
        commands.append(cmd)

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_docker_chown_build_artifacts", lambda **_kwargs: None)

    module._build_with_docker(
        docker_bin="/usr/bin/docker",
        docker_image="ghcr.io/apptainer/apptainer:1.4.5",
        docker_base_image=None,
        docker_base_source="docker",
        docker_network="host",
        build_cache=None,
        stage_repo=stage_repo,
        tmp_path=out_path,
        def_file=def_file,
        repo_root=repo_root,
    )

    assert len(commands) == 1
    assert commands[0][:4] == ["/usr/bin/docker", "run", "--rm", "--privileged"]
    assert staged_def.read_text(encoding="utf-8") == text
    assert not any("docker-archive" in str(part) for part in commands[0])


def test_docker_builder_archive_source_is_explicit_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()

    repo_root = tmp_path / "repo"
    stage_repo = tmp_path / "stage" / "repo"
    out_path = tmp_path / "images" / "Astrolabe_af3.sif.tmp"
    def_file = repo_root / "container" / "Astrolabe.af3.def"
    staged_def = stage_repo / "container" / "Astrolabe.af3.def"
    commands = []
    text = "Bootstrap: docker\nFrom: nvidia/cuda:test\n%post\n    true\n"

    def_file.parent.mkdir(parents=True)
    staged_def.parent.mkdir(parents=True)
    out_path.parent.mkdir(parents=True)
    def_file.write_text(text, encoding="utf-8")
    staged_def.write_text(text, encoding="utf-8")

    def fake_run(cmd, cwd=None):
        commands.append(cmd)
        if cmd[:2] == ["/usr/bin/docker", "save"]:
            Path(cmd[cmd.index("-o") + 1]).write_text("archive\n", encoding="utf-8")

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_docker_chown_build_artifacts", lambda **_kwargs: None)

    module._build_with_docker(
        docker_bin="/usr/bin/docker",
        docker_image="ghcr.io/apptainer/apptainer:1.4.5",
        docker_base_image=None,
        docker_base_source="docker-archive",
        docker_network="host",
        build_cache=None,
        stage_repo=stage_repo,
        tmp_path=out_path,
        def_file=def_file,
        repo_root=repo_root,
    )

    assert commands[0] == ["/usr/bin/docker", "pull", "nvidia/cuda:test"]
    assert commands[1][:3] == ["/usr/bin/docker", "save", "-o"]
    assert f"{stage_repo.parent / 'docker-build-work' / 'base'}:/base:ro" in commands[2]
    assert "Bootstrap: docker-archive" in staged_def.read_text(encoding="utf-8")


def test_docker_builder_mounts_persistent_build_cache(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()

    repo_root = tmp_path / "repo"
    stage_repo = tmp_path / "stage" / "repo"
    out_path = tmp_path / "images" / "Astrolabe_af3.sif.tmp"
    build_cache = tmp_path / "cache" / "container-build"
    def_file = repo_root / "container" / "Astrolabe.af3.def"
    def_file.parent.mkdir(parents=True)
    def_file.write_text("Bootstrap: scratch\n", encoding="utf-8")
    stage_repo.mkdir(parents=True)
    out_path.parent.mkdir(parents=True)
    commands = []

    def fake_run(cmd, cwd=None):
        commands.append((cmd, cwd))

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_docker_chown_build_artifacts", lambda **_kwargs: None)

    module._build_with_docker(
        docker_bin="/usr/bin/docker",
        docker_image="ghcr.io/apptainer/apptainer:1.4.5",
        docker_base_image=None,
        docker_base_source="docker",
        docker_network="host",
        build_cache=build_cache,
        stage_repo=stage_repo,
        tmp_path=out_path,
        def_file=def_file,
        repo_root=repo_root,
    )

    prepare_cmd, _cwd = commands[0]
    cmd, _cwd = commands[1]
    assert f"{build_cache.resolve()}:/cache" in prepare_cmd
    assert prepare_cmd[-3:] == ["sh", "-c", "chown -R 0:0 /cache"]
    assert f"{build_cache.resolve()}:/cache" in cmd
    assert f"{(build_cache / 'apptainer').resolve()}:/apptainer-cache" in cmd
    assert "APPTAINER_CACHEDIR=/apptainer-cache" in cmd
    assert (build_cache / "pip").is_dir()
    assert (build_cache / "mamba" / "pkgs").is_dir()
    assert (build_cache / "apptainer").is_dir()
    assert "build" in cmd
    bind_idx = cmd.index("--bind")
    assert cmd[bind_idx : bind_idx + 2] == ["--bind", "/cache:/cache"]
    build_arg_index = cmd.index("--build-arg-file")
    assert cmd[build_arg_index + 1] == "/apptainer-work/build-args.env"
    assert "GH_INSTALL_PYROSETTA=0\n" in (
        stage_repo.parent / "docker-build-work" / "build-args.env"
    ).read_text(encoding="utf-8")


def test_docker_builder_chowns_persistent_build_cache(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()

    commands = []

    def fake_call(cmd):
        commands.append(cmd)
        return 0

    monkeypatch.setattr(module.subprocess, "call", fake_call)

    build_cache = tmp_path / "cache" / "container-build"
    module._docker_chown_build_artifacts(
        docker_bin="/usr/bin/docker",
        docker_image="ghcr.io/apptainer/apptainer:1.4.5",
        docker_work_dir=tmp_path / "work",
        tmp_path=tmp_path / "images" / "Astrolabe_af3.sif.tmp",
        build_cache=build_cache,
    )

    cmd = commands[0]
    assert f"{build_cache.resolve()}:/cache" in cmd
    assert "chown -R" in cmd[-1]
    assert "/cache" in cmd[-1]


def test_docker_builder_prepares_build_cache_for_container_root(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()

    commands = []

    def fake_run(cmd, cwd=None):
        commands.append((cmd, cwd))

    monkeypatch.setattr(module, "_run", fake_run)

    build_cache = tmp_path / "cache" / "container-build"
    module._docker_prepare_build_cache_for_container_root(
        docker_bin="/usr/bin/docker",
        docker_image="ghcr.io/apptainer/apptainer:1.4.5",
        build_cache=build_cache,
    )

    cmd, _cwd = commands[0]
    assert f"{build_cache.resolve()}:/cache" in cmd
    assert cmd[-3:] == ["sh", "-c", "chown -R 0:0 /cache"]


def test_inspect_image_covers_host_and_docker_backends(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    image_path = tmp_path / "image.sif"
    image_path.write_bytes(b"sif")
    calls = []
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, cwd=None, env=None: calls.append((list(cmd), env)),
    )

    module._inspect_image(
        builder="apptainer",
        apptainer="/opt/apptainer/bin/apptainer",
        docker_bin=None,
        docker_image="unused",
        image_path=image_path,
    )
    module._inspect_image(
        builder="docker",
        apptainer="apptainer",
        docker_bin="/usr/bin/docker",
        docker_image="ghcr.io/apptainer/apptainer:1.4.5",
        image_path=image_path,
    )

    assert calls[0][0] == [
        "/opt/apptainer/bin/apptainer",
        "inspect",
        str(image_path.resolve()),
    ]
    assert calls[0][1]["PATH"].split(os.pathsep)[0] == "/opt/apptainer/bin"
    assert calls[1][0][-3:] == ["apptainer", "inspect", "/image.sif"]
    assert f"{image_path.resolve()}:/image.sif:ro" in calls[1][0]


def test_main_docker_builder_without_runner_does_not_require_host_apptainer(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()

    home_dir = tmp_path / "home"
    repo_stage = tmp_path / "stage" / "repo"
    output_path = tmp_path / "images" / "Astrolabe_af3.sif"
    def_file = REPO_ROOT / "container" / "Astrolabe.af3.def"
    docker = tmp_path / "docker"
    docker.write_bytes(b"docker-client")
    docker.chmod(0o755)
    commands = []
    inspections = []

    home_dir.mkdir()

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setattr(module, "_detect_apptainer", lambda: None)
    monkeypatch.setattr(module, "_detect_docker", lambda explicit=None: str(docker))

    def fake_output(cmd, **_kwargs):
        if cmd[1:3] == ["image", "inspect"]:
            return json.dumps(
                [
                    {
                        "Id": "sha256:test",
                        "RepoDigests": ["example.invalid/builder@sha256:test"],
                    }
                ]
            )
        if cmd[1:] == ["--version"]:
            return "Docker version 28.0\n"
        return "apptainer version 1.4.5\n"

    monkeypatch.setattr(module.subprocess, "check_output", fake_output)
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, **_kwargs: pytest.fail(f"local Docker identity must not pull: {cmd}"),
    )
    monkeypatch.setattr(
        module, "_inspect_image", lambda **kwargs: inspections.append(kwargs)
    )
    monkeypatch.setattr(
        module,
        "_stage_build_context",
        lambda _repo_root, _stage_root, _tracked_files: repo_stage,
    )

    def fake_build_with_docker(**kwargs):
        commands.append(kwargs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.with_suffix(output_path.suffix + ".tmp").write_text("stub sif\n", encoding="utf-8")

    monkeypatch.setattr(module, "_build_with_docker", fake_build_with_docker)

    argv = [
        "--builder",
        "docker",
        "--def",
        str(def_file),
        "--output",
        str(output_path),
        "--without-pyrosetta",
    ]
    rc = module.main(argv)
    assert module.main(argv) == 0

    assert rc == 0
    assert output_path.exists()
    assert len(commands) == 1
    assert commands[0]["docker_bin"] == str(docker)
    assert commands[0]["docker_image"] == "sha256:test"
    assert [item["docker_image"] for item in inspections] == [
        "sha256:test",
        "sha256:test",
    ]
    assert commands[0]["docker_base_source"] == "docker"
    assert commands[0]["docker_network"] == module.DEFAULT_DOCKER_NETWORK
    assert commands[0]["build_cache"] is None


def test_main_skip_build_writes_runner_without_building(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()

    home_dir = tmp_path / "home"
    home_dir.mkdir()
    def_file = tmp_path / "Astrolabe.af3.def"
    def_file.write_text("Bootstrap: docker\n", encoding="utf-8")
    output_path = tmp_path / "images" / "Astrolabe_af3.sif"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("stub sif\n", encoding="utf-8")
    _write_valid_image_metadata(module, output_path)
    runner_path = tmp_path / "tools" / "gh_site_container_python.py"

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setattr(module, "_detect_apptainer", lambda: "/usr/bin/apptainer")
    inspections = []
    monkeypatch.setattr(
        module,
        "_inspect_image",
        lambda **kwargs: inspections.append(kwargs),
    )

    def _fail_build(*_args, **_kwargs):
        raise AssertionError("build should not run with --skip-build")

    monkeypatch.setattr(module, "_run", _fail_build)

    rc = module.main(
        [
            "--skip-build",
            "--def",
            str(def_file),
            "--output",
            str(output_path),
            "--write-runner",
            str(runner_path),
            "--runner-bind",
            "/host/models:/models",
            "--runner-env-default",
            "GH_WEIGHT_AF3=/models/alphafold3",
            "--runner-prepend-env",
            "LD_LIBRARY_PATH=/host-libs/libgfortran3",
            "--runner-no-nv",
            "--without-pyrosetta",
        ]
    )

    text = runner_path.read_text(encoding="utf-8")
    assert rc == 0
    assert inspections and inspections[0]["image_path"] == output_path.resolve()
    assert runner_path.exists()
    assert "USE_NV = False" in text
    assert "'/host/models:/models'" in text
    assert "'GH_WEIGHT_AF3': '/models/alphafold3'" in text
    assert "('LD_LIBRARY_PATH', '/host-libs/libgfortran3')" in text


def test_main_skip_build_rejects_missing_or_contractless_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    image = tmp_path / "image.sif"
    image.write_bytes(b"sif")
    monkeypatch.setattr(module, "_detect_apptainer", lambda: "/usr/bin/apptainer")
    monkeypatch.setattr(
        module,
        "_inspect_image",
        lambda **_kwargs: pytest.fail("invalid provenance must fail before inspect"),
    )
    argv = [
        "--skip-build",
        "--output",
        str(image),
        "--without-pyrosetta",
    ]

    with pytest.raises(SystemExit, match="requires existing schema-v5"):
        module.main(argv)
    metadata_path = Path(str(image) + ".build.json")
    assert not metadata_path.exists()
    assert not Path(str(image) + ".sha256").exists()

    metadata_path.write_text(
        json.dumps(
            {
                "size": image.stat().st_size,
                "sha256": module._sha256_file(image),
                "build_features": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="schema-v5"):
        module.main(argv)
    assert "build_contract" not in json.loads(metadata_path.read_text(encoding="utf-8"))
    assert not Path(str(image) + ".sha256").exists()
