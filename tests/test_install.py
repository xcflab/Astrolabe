from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import multiprocessing
import os
import socket
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "install.py"
DEF_PATH = REPO_ROOT / "container" / "Astrolabe.af3.def"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("_test_gh_bootstrap", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_valid_sif_metadata(module, sif: Path, *, features=()) -> Path:
    features = sorted(set(features))
    license_by_feature = {
        "ananas": "ananas-academic",
        "pyrosetta": "pyrosetta-license",
        "rosetta": "rosetta-license",
    }
    contract = {
        "schema_version": 5,
        "context_sha256": "1" * 64,
        "definition_sha256": "2" * 64,
        "external_inputs": [],
        "build_features": features,
        "accepted_licenses": sorted(
            license_by_feature[feature]
            for feature in features
            if feature in license_by_feature
        ),
    }
    metadata = {
        "path": str(sif),
        "size": sif.stat().st_size,
        "sha256": module._sha256_file(sif),
        "build_features": features,
        "build_contract": contract,
        "build_fingerprint": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    path = Path(str(sif) + ".build.json")
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


def _write_fake_conda_records(module, root: Path, specs) -> None:
    env_root = root / "envs" / module.APPTAINER_BOOTSTRAP_ENV
    conda_meta = env_root / "conda-meta"
    conda_meta.mkdir(parents=True, exist_ok=True)
    (conda_meta / "history").write_text("fake transaction\n", encoding="utf-8")
    for spec in specs:
        name, _separator, version = spec.partition("=")
        (conda_meta / f"{name}-{version}-fake_0.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "version": version,
                    "build": "fake_0",
                    "build_number": 0,
                    "channel": "conda-forge",
                    "subdir": "linux-64",
                    "sha256": "0" * 64,
                    "url": (
                        "https://conda.anaconda.org/conda-forge/linux-64/"
                        f"{name}-{version}-fake_0.conda"
                    ),
                }
            ),
            encoding="utf-8",
        )


def _materialize_fake_apptainer_tiers(module, cmd) -> None:
    root = Path(cmd[cmd.index("--root-prefix") + 1])
    env_root = root / "envs" / module.APPTAINER_BOOTSTRAP_ENV
    installed_specs = []
    if any(spec in cmd for spec in module.APPTAINER_RUNTIME_PACKAGES):
        installed_specs.extend(module.APPTAINER_RUNTIME_PACKAGES)
        for name in ("apptainer", "squashfuse"):
            executable = env_root / "bin" / name
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)
    if any(spec in cmd for spec in module.APPTAINER_BUILD_PACKAGES):
        installed_specs.extend(module.APPTAINER_BUILD_PACKAGES)
        for name in ("meson", "ninja"):
            executable = env_root / "bin" / name
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)
        capability = env_root / "include" / "sys" / "capability.h"
        capability.parent.mkdir(parents=True, exist_ok=True)
        capability.write_text("/* stub */\n", encoding="utf-8")
        for name in ("setcap", "getcap"):
            executable = env_root / "sbin" / name
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)
    _write_fake_conda_records(module, root, installed_specs)


def _concurrent_manifest_worker(
    script_path: str,
    manifest_path: str,
    software: str,
    start,
) -> None:
    spec = importlib.util.spec_from_file_location(
        f"_manifest_worker_{software}", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_write_json = module._write_json

    def slow_write_json(path, payload):
        time.sleep(0.05)
        original_write_json(path, payload)

    module._write_json = slow_write_json
    start.wait()
    module._write_software_manifest(
        Path(manifest_path),
        selected=(software,),
        markers={software: []},
        dry_run=False,
    )


def test_stage_af3_build_data_cache_reuses_gold_pickles(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()

    converters = tmp_path / "gold" / "converters"
    converters.mkdir(parents=True)
    ccd = converters / "ccd.pickle"
    with ccd.open("wb") as handle:
        handle.truncate(100_000_001)
    sets = converters / "chemical_component_sets.pickle"
    sets.write_bytes(b"x" * 1_001)
    components = tmp_path / "components.cif"
    with components.open("wb") as handle:
        handle.truncate(100_000_001)
    monkeypatch.setattr(module, "_gold_af3_converters_dir", lambda _root: converters)

    staged = module._stage_af3_build_data_cache(
        root=tmp_path / "root",
        build_cache=tmp_path / "cache",
        ccd_pickle=None,
        chemical_component_sets_pickle=None,
        components_cif=str(components),
        dry_run=False,
    )

    assert [path.name for path in staged] == [
        "ccd.pickle",
        "chemical_component_sets.pickle",
        "components.cif",
    ]
    assert os.stat(staged[0]).st_ino == os.stat(ccd).st_ino
    assert os.stat(staged[1]).st_ino == os.stat(sets).st_ino
    assert os.stat(staged[2]).st_ino == os.stat(components).st_ino


def test_stage_pyrosetta_cache_reuses_verified_wheel(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    cache = tmp_path / "cache"
    wheel = cache / "wheels" / "pyrosetta" / module.PYROSETTA_WHEEL_NAME
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"gold-wheel")
    monkeypatch.setattr(module, "PYROSETTA_ALLOWED_SHA256", {module._sha256_file(wheel)})
    monkeypatch.setattr(module, "PYROSETTA_WHEEL_SIZE", wheel.stat().st_size)

    staged = module._stage_pyrosetta_wheel_cache(
        root=tmp_path / "root",
        build_cache=cache,
        wheel=None,
        dry_run=False,
    )

    assert staged == wheel


def test_download_url_finishes_after_more_than_twelve_progressing_disconnects(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    payload = b"x" * (20 * 64 * 1024)
    chunk_size = 64 * 1024
    observed_ranges: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            range_header = self.headers.get("Range")
            observed_ranges.append(range_header)
            start = int(range_header.removeprefix("bytes=").removesuffix("-")) if range_header else 0
            remaining = len(payload) - start
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Length", str(remaining))
            if range_header:
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
                )
            self.end_headers()
            end = min(start + chunk_size, len(payload))
            self.wfile.write(payload[start:end])
            self.wfile.flush()
            if end < len(payload):
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()

        def log_message(self, _format: str, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(module, "DOWNLOAD_PROGRESS_DEADLINE_SECONDS", 30.0)
    try:
        destination = tmp_path / "large-wheel.whl"
        module._download_url(
            f"http://127.0.0.1:{server.server_port}/large-wheel.whl",
            destination,
            dry_run=False,
            minimum_size=len(payload),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            attempts=3,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert destination.read_bytes() == payload
    assert len(observed_ranges) == 20
    assert observed_ranges[0] is None
    assert observed_ranges[-1] == f"bytes={19 * chunk_size}-"


def test_download_url_stops_after_bounded_no_progress(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    calls = []

    def fail_without_progress(*_args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("connection closed without bytes")

    monkeypatch.setattr(module, "download_url", fail_without_progress)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="no_progress_attempts=3/3"):
        module._download_url(
            "https://example.invalid/asset",
            tmp_path / "asset.bin",
            dry_run=False,
            attempts=3,
        )

    assert len(calls) == 3
    assert all(call["attempts"] == 1 for call in calls)


def test_download_url_stops_at_total_deadline_despite_progress(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    destination = tmp_path / "asset.bin"
    part = Path(str(destination) + ".part")
    ticks = iter((0.0, 0.0, 2.0))
    calls = []

    def progressing_failure(*_args, **_kwargs):
        calls.append(1)
        part.write_bytes(b"progress")
        raise RuntimeError("connection closed after progress")

    monkeypatch.setattr(module, "download_url", progressing_failure)
    monkeypatch.setattr(module, "DOWNLOAD_PROGRESS_DEADLINE_SECONDS", 1.0)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))

    with pytest.raises(RuntimeError, match="deadline_seconds=1.0"):
        module._download_url(
            "https://example.invalid/asset",
            destination,
            dry_run=False,
        )

    assert calls == [1]


def test_stage_pyrosetta_cache_without_gold_downloads_exact_upstream(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    payload = b"upstream-wheel"
    expected_sha256 = module.hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(module, "PYROSETTA_UPSTREAM_SHA256", expected_sha256)
    monkeypatch.setattr(module, "PYROSETTA_ALLOWED_SHA256", {expected_sha256})
    monkeypatch.setattr(module, "PYROSETTA_WHEEL_SIZE", len(payload))
    downloads = []

    def fake_download(url, dst, *, expected_sha256, **kwargs):
        downloads.append((url, dst, expected_sha256, kwargs))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(payload)

    monkeypatch.setattr(module, "_download_url", fake_download)

    staged = module._stage_pyrosetta_wheel_cache(
        root=tmp_path / "root",
        build_cache=tmp_path / "cache",
        wheel=None,
        dry_run=False,
    )

    assert staged == tmp_path / "cache" / "wheels" / "pyrosetta" / module.PYROSETTA_WHEEL_NAME
    assert downloads == [
        (
            module.PYROSETTA_URL,
            staged,
            expected_sha256,
            {
                "dry_run": False,
                "minimum_size": len(payload),
                "attempts": module.PYROSETTA_MAX_NO_PROGRESS_ATTEMPTS,
            },
        )
    ]


@pytest.mark.parametrize("corruption", ["truncated", "wrong-sha"])
def test_stage_pyrosetta_cache_recovers_invalid_final_target(
    tmp_path: Path, monkeypatch, corruption: str
) -> None:
    module = _load_script_module()
    payload = b"verified-wheel-payload"
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(module, "PYROSETTA_WHEEL_SIZE", len(payload))
    monkeypatch.setattr(module, "PYROSETTA_UPSTREAM_SHA256", expected_sha256)
    monkeypatch.setattr(module, "PYROSETTA_ALLOWED_SHA256", {expected_sha256})
    cache = tmp_path / "cache"
    target = cache / "wheels" / "pyrosetta" / module.PYROSETTA_WHEEL_NAME
    target.parent.mkdir(parents=True)
    corrupt_payload = payload[:5] if corruption == "truncated" else b"z" * len(payload)
    target.write_bytes(corrupt_payload)
    observed_ranges = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            range_header = self.headers.get("Range")
            observed_ranges.append(range_header)
            start = int(range_header.removeprefix("bytes=").removesuffix("-")) if range_header else 0
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Length", str(len(payload) - start))
            if range_header:
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
                )
            self.end_headers()
            self.wfile.write(payload[start:])

        def log_message(self, _format: str, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(
        module,
        "PYROSETTA_URL",
        f"http://127.0.0.1:{server.server_port}/{module.PYROSETTA_WHEEL_NAME}",
    )
    try:
        staged = module._stage_pyrosetta_wheel_cache(
            root=tmp_path / "root",
            build_cache=cache,
            wheel=None,
            dry_run=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert staged == target
    assert target.read_bytes() == payload
    if corruption == "truncated":
        assert observed_ranges == [f"bytes={len(corrupt_payload)}-"]
        assert not Path(str(target) + ".invalid").exists()
    else:
        assert observed_ranges == [None]
        assert Path(str(target) + ".invalid").read_bytes() == corrupt_payload


def test_validate_pyrosetta_wheel_requires_exact_size_and_sha(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    wheel = tmp_path / module.PYROSETTA_WHEEL_NAME
    wheel.write_bytes(b"wheel")
    checksum = module._sha256_file(wheel)
    monkeypatch.setattr(module, "PYROSETTA_ALLOWED_SHA256", {checksum})
    monkeypatch.setattr(module, "PYROSETTA_WHEEL_SIZE", wheel.stat().st_size + 1)
    with pytest.raises(SystemExit, match="size mismatch"):
        module._validate_pyrosetta_wheel(wheel)

    monkeypatch.setattr(module, "PYROSETTA_WHEEL_SIZE", wheel.stat().st_size)
    monkeypatch.setattr(module, "PYROSETTA_ALLOWED_SHA256", {"0" * 64})
    with pytest.raises(SystemExit, match="checksum mismatch"):
        module._validate_pyrosetta_wheel(wheel)


def test_stage_ananas_cache_uses_exact_official_contract(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    calls = []
    monkeypatch.setattr(
        module,
        "_download_url",
        lambda url, dst, **kwargs: calls.append((url, dst, kwargs)),
    )

    staged = module._stage_ananas_archive_cache(
        build_cache=tmp_path / "cache", dry_run=True
    )

    assert staged == tmp_path / "cache" / "sources" / "ananas" / module.ANANAS_ARCHIVE_NAME
    assert calls == [
        (
            module.ANANAS_URL,
            staged,
            {
                "dry_run": True,
                "minimum_size": module.ANANAS_ARCHIVE_SIZE,
                "expected_sha256": module.ANANAS_ARCHIVE_SHA256,
            },
        )
    ]


def test_stage_rosetta_tools_reuses_exact_cached_archive(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    manifest = tmp_path / "rosetta-tools.manifest.tsv"
    checkout = tmp_path / "checkout"
    rel_path = "source/scripts/python/public/molfile_to_params.py"
    source = checkout / rel_path
    source.parent.mkdir(parents=True)
    source.write_text("print('molfile')\n", encoding="utf-8")
    manifest.write_text(
        "sha256\tsize_bytes\tpath\n"
        f"{module._sha256_file(source)}\t{source.stat().st_size}\t{rel_path}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ROSETTA_TOOLS_MANIFEST", manifest)
    build_cache = tmp_path / "cache"
    target = (
        build_cache
        / "sources"
        / "rosetta"
        / module.ROSETTA_TOOLS_ARCHIVE_NAME
    )
    module._write_rosetta_tools_archive(checkout, target)
    monkeypatch.setattr(
        module,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("warm reuse must not run git"),
    )

    assert module._stage_rosetta_tools_cache(
        build_cache=build_cache, dry_run=False
    ) == target


def test_stage_rosetta_tools_quarantines_corrupt_cached_archive(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    payload = b"print('molfile')\n"
    rel_path = "source/scripts/python/public/molfile_to_params.py"
    manifest = tmp_path / "rosetta-tools.manifest.tsv"
    manifest.write_text(
        "sha256\tsize_bytes\tpath\n"
        f"{hashlib.sha256(payload).hexdigest()}\t{len(payload)}\t{rel_path}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ROSETTA_TOOLS_MANIFEST", manifest)
    build_cache = tmp_path / "cache"
    target = (
        build_cache
        / "sources"
        / "rosetta"
        / module.ROSETTA_TOOLS_ARCHIVE_NAME
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not-a-tar")
    checkout = target.parent / f"checkout-{module.ROSETTA_TOOLS_COMMIT}"
    source = checkout / rel_path
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda *_args, **_kwargs: module.ROSETTA_TOOLS_COMMIT + "\n",
    )
    monkeypatch.setattr(module, "_run", lambda *_args, **_kwargs: None)

    assert module._stage_rosetta_tools_cache(
        build_cache=build_cache, dry_run=False
    ) == target
    assert list(target.parent.glob(target.name + ".invalid.*"))
    assert module._validate_rosetta_tools_archive(target) == target


def test_stage_rosetta_tools_recovers_invalid_partial_git_checkout(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    payload = b"print('molfile')\n"
    rel_path = "source/scripts/python/public/molfile_to_params.py"
    manifest = tmp_path / "rosetta-tools.manifest.tsv"
    manifest.write_text(
        "sha256\tsize_bytes\tpath\n"
        f"{hashlib.sha256(payload).hexdigest()}\t{len(payload)}\t{rel_path}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ROSETTA_TOOLS_MANIFEST", manifest)
    build_cache = tmp_path / "cache"
    target = (
        build_cache
        / "sources"
        / "rosetta"
        / module.ROSETTA_TOOLS_ARCHIVE_NAME
    )
    checkout = target.parent / f"checkout-{module.ROSETTA_TOOLS_COMMIT}"
    checkout.mkdir(parents=True)
    (checkout / ".git").write_text("partial\n", encoding="utf-8")
    commands = []

    def fake_run(cmd, *_args, **_kwargs):
        commands.append(cmd)
        if "checkout" in cmd and "FETCH_HEAD" in cmd:
            source = checkout / rel_path
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(payload)

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda *_args, **_kwargs: module.ROSETTA_TOOLS_COMMIT + "\n",
    )

    assert module._stage_rosetta_tools_cache(
        build_cache=build_cache, dry_run=False
    ) == target
    assert any(command[:2] == ["git", "init"] for command in commands)
    assert list(target.parent.glob(checkout.name + ".invalid.*"))


def test_pyrosetta_selection_requires_license_and_uses_unified_sif(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    common = [
        "--root",
        str(tmp_path / "root"),
        "--software",
        "pyrosetta",
        "--skip-build",
        "--skip-submodules",
        "--skip-import-smoke",
        "--skip-doctor",
        "--dry-run",
    ]
    with pytest.raises(SystemExit, match="pyrosetta-license"):
        module.main(common)

    assert module.main([*common, "--accept-license", "pyrosetta-license"]) == 0
    profile = json.loads(
        (tmp_path / "root" / "sites" / "local.json").read_text(encoding="utf-8")
    )
    assert profile["sif"].endswith("Astrolabe_af3.sif")


def test_normalize_af2_params_accepts_params_dir(tmp_path: Path) -> None:
    module = _load_script_module()

    params = tmp_path / "colabfold" / "params"
    params.mkdir(parents=True)
    (params / "params_model_3_multimer_v3.npz").write_text("stub\n", encoding="utf-8")

    assert module._normalize_af2_data_dir(str(params)) == tmp_path / "colabfold"


def test_normalize_af2_params_accepts_parent_data_dir(tmp_path: Path) -> None:
    module = _load_script_module()

    data_dir = tmp_path / "alphafold2"
    params = data_dir / "params"
    params.mkdir(parents=True)
    (params / "params_model_3_multimer_v3.npz").write_text("stub\n", encoding="utf-8")

    assert module._normalize_af2_data_dir(str(data_dir)) == data_dir


def test_af2_validation_checks_every_selected_parameter_digest(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    params = tmp_path / "params"
    params.mkdir()
    payloads = {
        "params_model_1.npz": b"model-one",
        "params_model_2_ptm.npz": b"model-two-ptm",
        "params_model_3_multimer_v3.npz": b"model-three-multimer",
    }
    monkeypatch.setattr(
        module,
        "AF2_PARAM_SPECS",
        {
            name: (len(payload), hashlib.sha256(payload).hexdigest())
            for name, payload in payloads.items()
        },
    )
    for name, payload in payloads.items():
        (params / name).write_bytes(payload)

    module._validate_af2_params_dir(params)

    (params / "params_model_2_ptm.npz").write_bytes(b"badxx-two-ptm")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        module._validate_af2_params_dir(params)


def test_af2_extraction_ignores_unneeded_regular_files_and_rejects_links(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    archive = tmp_path / "params.tar"
    required = "params_model_1.npz"
    with tarfile.open(archive, "w") as handle:
        payload = b"required"
        info = tarfile.TarInfo(required)
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
        extra = tarfile.TarInfo("LICENSE")
        extra.size = 7
        handle.addfile(extra, io.BytesIO(b"license"))
    output = tmp_path / "output"

    module._extract_tar(archive, output, members=(required,), dry_run=False)

    assert (output / required).read_bytes() == b"required"
    assert not (output / "LICENSE").exists()

    with tarfile.open(archive, "w") as handle:
        payload = b"required"
        info = tarfile.TarInfo(required)
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
        link = tarfile.TarInfo("unsafe-link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../outside"
        handle.addfile(link)
    with pytest.raises(RuntimeError, match="non-regular"):
        module._extract_tar(archive, output, members=(required,), dry_run=False)


def test_dry_run_writes_site_profile_and_skips_execution(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load_script_module()

    commands = []

    def fake_run(cmd, *, cwd=None, dry_run=False):
        commands.append((cmd, cwd, dry_run))

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module,
        "_detect_apptainer",
        lambda explicit, dry_run=False, root=None, bootstrap="auto", build_tools=False: explicit or "/usr/bin/apptainer",
    )

    af3 = tmp_path / "af3"
    af3.mkdir()
    af2_params = tmp_path / "localcolabfold" / "colabfold" / "params"
    af2_params.mkdir(parents=True)
    (af2_params / "params_model_3_multimer_v3.npz").write_text("stub\n", encoding="utf-8")

    profile = tmp_path / "sites" / "h100.json"
    runner = tmp_path / "tools" / "gh_h100_container_python.py"

    rc = module.main(
        [
            "--profile",
            "h100",
            "--root",
            str(tmp_path / "root"),
            "--sif",
            str(tmp_path / "images" / "gh.sif"),
            "--runner",
            str(runner),
            "--site-profile",
            str(profile),
            "--software",
            "alphafold3,alphafold2",
            "--af3-weights",
            str(af3),
            "--af2-params",
            str(af2_params),
            "--skip-build",
            "--skip-import-smoke",
            "--skip-doctor",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert "Dry run complete" in output
    assert "Bootstrap complete" not in output
    payload = json.loads(profile.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["profile"] == "h100"
    assert str(af3) + ":/models/alphafold3" in payload["binds"]
    assert str(af2_params.parent) + ":/models/alphafold2" in payload["binds"]
    assert payload["env_defaults"]["GH_WEIGHT_AF3"] == "/models/alphafold3"
    assert payload["env_defaults"]["GH_WEIGHT_AF2"] == "/models/alphafold2"
    assert payload["env_defaults"]["HOME"] == "/cache/gh-runtime/home"
    assert payload["env_defaults"]["TORCH_EXTENSIONS_DIR"] == (
        "/cache/gh-runtime/torch-extensions"
    )
    assert any(bind.endswith(":/cache/gh-runtime") for bind in payload["binds"])
    sync_index = next(
        index
        for index, cmd in enumerate(commands)
        if cmd[0] == ["git", "submodule", "sync", "--recursive"]
    )
    update_index = next(
        index
        for index, cmd in enumerate(commands)
        if cmd[0][:3] == ["git", "submodule", "update"]
    )
    assert sync_index < update_index
    submodule_cmd = next(cmd[0] for cmd in commands if cmd[0][:3] == ["git", "submodule", "update"])
    assert "--depth" in submodule_cmd
    assert "1" in submodule_cmd
    assert "--jobs" in submodule_cmd
    assert "4" in submodule_cmd
    patch_cmd = next(
        cmd[0]
        for cmd in commands
        if any("apply_vendor_patches.py" in part for part in cmd[0])
    )
    assert patch_cmd[:1] == [module.sys.executable]
    assert patch_cmd[-4:] == ["--only", "alphafold2", "--only", "alphafold3"]
    for rel_path in ("vendor/alphafold3", "vendor/alphafold2"):
        assert rel_path in submodule_cmd
    for rel_path in (
        "vendor/na_mpnn",
        "vendor/openfold_esm2",
        "vendor/rhodesign",
        "vendor/rpxdock",
        "vendor/willutil",
    ):
        assert rel_path not in submodule_cmd


def test_openfold_submodule_uses_published_fork() -> None:
    gitmodules = (Path(__file__).parents[1] / ".gitmodules").read_text(encoding="utf-8")

    assert '[submodule "vendor/openfold_esm2"]' in gitmodules
    assert "url = https://github.com/Molaison/openfold.git" in gitmodules
    assert "branch = gh-esmfold-cpu-fallback" in gitmodules


def test_submodule_index_validation_rejects_incomplete_checkout(monkeypatch) -> None:
    module = _load_script_module()

    def fake_run(cmd, **kwargs):
        del kwargs
        if cmd[:3] == ["git", "submodule", "status"]:
            return SimpleNamespace(
                returncode=0,
                stdout=b" 1234567 vendor/rhofold (heads/main)\n",
                stderr=b"",
            )
        assert cmd == ["git", "diff", "--cached", "--quiet", "--"]
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="vendor/rhofold.*clean checkout"):
        module._validate_submodule_indexes(dry_run=False)


def test_bootstrap_reuses_host_weight_env_vars(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()

    commands = []

    def fake_run(cmd, *, cwd=None, dry_run=False):
        commands.append((cmd, cwd, dry_run))

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module,
        "_detect_apptainer",
        lambda explicit, dry_run=False, root=None, bootstrap="auto", build_tools=False: explicit or "/usr/bin/apptainer",
    )

    af3 = tmp_path / "existing" / "af3"
    af3.mkdir(parents=True)
    af2 = tmp_path / "existing" / "alphafold2"
    (af2 / "params").mkdir(parents=True)
    (af2 / "params" / "params_model_3_multimer_v3.npz").write_text("stub\n", encoding="utf-8")
    boltz = tmp_path / "existing" / "boltz"
    boltz.mkdir(parents=True)
    monkeypatch.setenv("GH_WEIGHT_AF3", str(af3))
    monkeypatch.setenv("GH_WEIGHT_AF2", str(af2))
    monkeypatch.setenv("GH_WEIGHT_BOLTZ", str(boltz))

    profile = tmp_path / "sites" / "env_reuse.json"
    runner = tmp_path / "tools" / "gh_env_reuse_container_python.py"

    rc = module.main(
        [
            "--profile",
            "env_reuse",
            "--root",
            str(tmp_path / "root"),
            "--sif",
            str(tmp_path / "images" / "gh.sif"),
            "--runner",
            str(runner),
            "--site-profile",
            str(profile),
            "--software",
            "alphafold3,alphafold2,boltz",
            "--skip-build",
            "--skip-submodules",
            "--skip-import-smoke",
            "--skip-doctor",
            "--dry-run",
        ]
    )

    payload = json.loads(profile.read_text(encoding="utf-8"))
    assert rc == 0
    assert str(af3) + ":/models/alphafold3" in payload["binds"]
    assert str(af2) + ":/models/alphafold2" in payload["binds"]
    assert str(boltz) + ":/models/boltz" in payload["binds"]
    assert payload["env_defaults"]["GH_WEIGHT_AF3"] == "/models/alphafold3"
    assert payload["env_defaults"]["GH_WEIGHT_AF2"] == "/models/alphafold2"
    assert payload["env_defaults"]["GH_WEIGHT_BOLTZ"] == "/models/boltz"


def test_bootstrap_preserves_custom_protenix_env_var_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module()

    monkeypatch.setattr(module, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_detect_apptainer",
        lambda explicit, dry_run=False, root=None, bootstrap="auto", build_tools=False: explicit or "/usr/bin/apptainer",
    )

    protenix = tmp_path / "existing" / "protenix"
    custom_paths = {
        "GH_WEIGHT_PROTENIX": protenix / "release_model" / "custom_default.pt",
        "GH_WEIGHT_PROTENIX_CONSTRAINT": protenix / "release_model" / "custom_constraint.pt",
        "GH_PROTENIX_CCD": protenix / "release_data" / "custom_components.cif",
        "GH_PROTENIX_CCD_RDKIT": protenix / "release_data" / "custom_components.pkl",
    }
    for env_key, path in custom_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stub\n", encoding="utf-8")
        monkeypatch.setenv(env_key, str(path))

    profile = tmp_path / "sites" / "protenix_reuse.json"

    rc = module.main(
        [
            "--profile",
            "protenix_reuse",
            "--root",
            str(tmp_path / "root"),
            "--sif",
            str(tmp_path / "images" / "gh.sif"),
            "--site-profile",
            str(profile),
            "--software",
            "protenix",
            "--allow-missing-af3",
            "--skip-build",
            "--skip-submodules",
            "--skip-import-smoke",
            "--skip-doctor",
            "--dry-run",
        ]
    )

    payload = json.loads(profile.read_text(encoding="utf-8"))
    assert rc == 0
    assert str(protenix) + ":/models/protenix" in payload["binds"]
    for env_key, host_path in custom_paths.items():
        expected = "/models/protenix/" + str(host_path.relative_to(protenix))
        assert payload["env_defaults"][env_key] == expected


def test_protenix_downloads_honor_all_custom_env_targets(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    custom_paths = {
        env_key: tmp_path / "custom" / Path(rel_path).name
        for rel_path, env_key in module.PROTENIX_ENV_BY_REL_PATH.items()
    }
    calls = []
    monkeypatch.setattr(
        module,
        "_download_url",
        lambda url, dst, **kwargs: calls.append((url, dst, kwargs)),
    )

    module._download_protenix_assets(
        tmp_path / "default",
        protenix_env_paths=custom_paths,
        dry_run=True,
    )

    assert [dst for _url, dst, _kwargs in calls] == [
        custom_paths[module.PROTENIX_ENV_BY_REL_PATH[rel_path]]
        for rel_path in module.PROTENIX_ASSET_URLS
    ]
    assert all(call[2]["expected_sha256"] for call in calls)


def test_bootstrap_protenix_env_var_missing_path_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module()
    for key in module.PROTENIX_ENV_BY_REL_PATH.values():
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(module, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_detect_apptainer",
        lambda explicit, dry_run=False, root=None, bootstrap="auto", build_tools=False: explicit or "/usr/bin/apptainer",
    )

    missing = tmp_path / "existing" / "protenix" / "release_model" / "missing_constraint.pt"
    missing.parent.mkdir(parents=True)
    monkeypatch.setenv("GH_WEIGHT_PROTENIX_CONSTRAINT", str(missing))

    try:
        module.main(
            [
                "--profile",
                "protenix_missing",
                "--root",
                str(tmp_path / "root"),
                    "--sif",
                    str(tmp_path / "images" / "gh.sif"),
                    "--software",
                    "protenix",
                "--allow-missing-af3",
                "--skip-build",
                "--skip-submodules",
                "--skip-import-smoke",
                "--skip-doctor",
                "--dry-run",
            ]
        )
    except SystemExit as exc:
        text = str(exc)
        assert "Required resource path does not exist for GH_WEIGHT_PROTENIX_CONSTRAINT" in text
        assert str(missing) in text
    else:
        raise AssertionError("missing Protenix env-var path should fail visibly")


def test_surface_selection_wires_all_external_runtime_resources(
    tmp_path: Path,
) -> None:
    module = _load_script_module()

    def executable(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o755)
        return path

    surfdesign_python = executable(tmp_path / "envs" / "surfdesign" / "bin" / "python")
    surfpro_python = executable(tmp_path / "envs" / "surfpro" / "bin" / "python")
    surfdesign_repo = tmp_path / "repos" / "surfdesign"
    surfdesign_source = surfdesign_repo / "src" / "models" / "generator.py"
    surfdesign_source.parent.mkdir(parents=True)
    surfdesign_source.write_text("# source\n", encoding="utf-8")
    surfdesign_experiment = tmp_path / "weights" / "surfdesign"
    (surfdesign_experiment / ".hydra").mkdir(parents=True)
    (surfdesign_experiment / "checkpoints").mkdir()
    (surfdesign_experiment / ".hydra" / "config.yaml").write_text("model: {}\n")
    (surfdesign_experiment / "checkpoints" / "best.ckpt").write_bytes(b"checkpoint")
    surfpro_repo = tmp_path / "repos" / "surfpro"
    surfpro_source = surfpro_repo / "fairseq" / "models" / "surface_protein_model.py"
    surfpro_source.parent.mkdir(parents=True)
    surfpro_source.write_text("# source\n", encoding="utf-8")
    surfpro_checkpoint = tmp_path / "weights" / "surfpro" / "checkpoint_best.pt"
    surfpro_checkpoint.parent.mkdir(parents=True)
    surfpro_checkpoint.write_bytes(b"checkpoint")
    msms = executable(tmp_path / "tools" / "msms" / "msms")
    pdb_to_xyzrn = executable(tmp_path / "tools" / "xyzrn" / "pdb_to_xyzrn")
    (pdb_to_xyzrn.parent / "atmtypenumbers").write_text(
        "C 1\n", encoding="utf-8"
    )
    profile = tmp_path / "site.json"

    assert module.main(
        [
            "--root",
            str(tmp_path / "root"),
            "--site-profile",
            str(profile),
            "--software",
            "surface",
            "--surfdesign-python",
            str(surfdesign_python),
            "--surfdesign-repo",
            str(surfdesign_repo),
            "--surfdesign-experiment",
            str(surfdesign_experiment),
            "--surfpro-python",
            str(surfpro_python),
            "--surfpro-repo",
            str(surfpro_repo),
            "--surfpro-checkpoint",
            str(surfpro_checkpoint),
            "--surfpro-msms",
            str(msms),
            "--surfpro-pdb-to-xyzrn",
            str(pdb_to_xyzrn),
            "--skip-build",
            "--skip-submodules",
            "--skip-import-smoke",
            "--skip-doctor",
            "--dry-run",
        ]
    ) == 0

    payload = json.loads(profile.read_text(encoding="utf-8"))
    env = payload["env_defaults"]
    assert env["GH_SURFDESIGN_PYTHON"] == "/envs/surfdesign/bin/python"
    assert env["GH_SURFDESIGN_REPO"] == "/ext/surfdesign"
    assert env["GH_WEIGHT_SURFDESIGN"] == "/models/surfdesign"
    assert env["GH_SURFPRO_PYTHON"] == "/envs/surfpro/bin/python"
    assert env["GH_SURFPRO_REPO"] == "/ext/surfpro"
    assert env["GH_SURFPRO_PYTHONPATH"] == (
        "/ext/surfpro:/ext/sru:/ext/protein_sequence_models"
    )
    assert env["GH_WEIGHT_SURFPRO"] == "/models/surfpro"
    assert env["GH_SURFPRO_MSMS"] == "/tools/surfpro/msms/msms"
    assert env["GH_SURFPRO_PDB_TO_XYZRN"] == (
        "/tools/surfpro/pdb_to_xyzrn/pdb_to_xyzrn"
    )
    assert f"{surfdesign_repo}:/ext/surfdesign" in payload["binds"]
    assert f"{surfpro_repo}:/ext/surfpro" in payload["binds"]
    assert f"{REPO_ROOT / 'vendor' / 'sru'}:/ext/sru" in payload["binds"]
    assert (
        f"{REPO_ROOT / 'vendor' / 'protein_sequence_models'}:"
        "/ext/protein_sequence_models"
    ) in payload["binds"]


def test_surfpro_missing_manual_tool_fails_before_side_effects(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    commands = []
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, *, cwd=None, dry_run=False: commands.append(cmd),
    )
    python = tmp_path / "env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    repo = tmp_path / "surfpro"
    source = repo / "fairseq" / "models" / "surface_protein_model.py"
    source.parent.mkdir(parents=True)
    source.write_text("# source\n", encoding="utf-8")
    checkpoint = tmp_path / "weights" / "checkpoint_best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    pdb_to_xyzrn = tmp_path / "tools" / "pdb_to_xyzrn"
    pdb_to_xyzrn.parent.mkdir()
    pdb_to_xyzrn.write_text("#!/bin/sh\n", encoding="utf-8")
    pdb_to_xyzrn.chmod(0o755)

    with pytest.raises(SystemExit, match="SurfPro MSMS"):
        module.main(
            [
                "--root",
                str(tmp_path / "root"),
                "--software",
                "surfpro",
                "--surfpro-python",
                str(python),
                "--surfpro-repo",
                str(repo),
                "--surfpro-checkpoint",
                str(checkpoint),
                "--surfpro-pdb-to-xyzrn",
                str(pdb_to_xyzrn),
                "--skip-build",
                "--skip-submodules",
                "--skip-import-smoke",
                "--skip-doctor",
                "--dry-run",
            ]
        )
    assert commands == []


def test_surfpro_rejects_checkpoint_basename_runtime_cannot_resolve(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    with pytest.raises(SystemExit, match="must be named checkpoint_best.pt"):
        module.main(
            [
                "--root",
                str(tmp_path / "root"),
                "--software",
                "surfpro",
                "--surfpro-checkpoint",
                str(tmp_path / "custom-cath.pt"),
                "--skip-build",
                "--skip-submodules",
                "--skip-import-smoke",
                "--skip-doctor",
                "--dry-run",
            ]
        )


def test_surface_import_preflight_enforces_locked_runtime_contract(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    commands = []
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, *, cwd=None, dry_run=False: commands.append(cmd),
    )

    module._run_selected_host_import_smoke(
        SimpleNamespace(dry_run=True),
        selected=("surfpro", "surfdesign"),
        dplm_python=None,
        opendde_python=None,
        esmfold2_python=None,
        rapidock_python=None,
        rpxdock_python=None,
        rapidock_repo=tmp_path / "rapidock",
        rf3_python=None,
        rf3_repo=None,
        surfdesign_python=tmp_path / "surfdesign" / "bin" / "python",
        surfdesign_repo=tmp_path / "surfdesign-repo",
        surfpro_python=tmp_path / "surfpro" / "bin" / "python",
        surfpro_repo=tmp_path / "surfpro-repo",
        hf_cache=tmp_path / "hf",
    )

    codes = [command[-1] for command in commands]
    assert any("import pymol2" in code for code in codes)
    assert any("sys.version_info[:2] == (3, 8)" in code for code in codes)
    assert any("torch.version.cuda == '11.3'" in code for code in codes)
    assert any("get_edges_batch" in code for code in codes)


def test_submodule_depth_zero_uses_full_history(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()

    commands = []

    def fake_run(cmd, *, cwd=None, dry_run=False):
        commands.append((cmd, cwd, dry_run))

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module,
        "_detect_apptainer",
        lambda explicit, dry_run=False, root=None, bootstrap="auto", build_tools=False: explicit or "/usr/bin/apptainer",
    )

    rc = module.main(
        [
            "--root",
            str(tmp_path / "root"),
            "--software",
            "alphafold3",
            "--skip-build",
            "--skip-import-smoke",
            "--skip-doctor",
            "--allow-missing-af3",
            "--submodule-depth",
            "0",
            "--submodule-jobs",
            "2",
            "--dry-run",
        ]
    )

    submodule_cmd = next(cmd[0] for cmd in commands if cmd[0][:3] == ["git", "submodule", "update"])
    assert rc == 0
    assert "--depth" not in submodule_cmd
    jobs_index = submodule_cmd.index("--jobs")
    assert submodule_cmd[jobs_index : jobs_index + 2] == ["--jobs", "2"]
    assert submodule_cmd[-1:] == ["vendor/alphafold3"]


def test_submodule_set_all_does_not_append_runtime_path_filter(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()

    commands = []

    def fake_run(cmd, *, cwd=None, dry_run=False):
        commands.append((cmd, cwd, dry_run))

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module,
        "_detect_apptainer",
        lambda explicit, dry_run=False, root=None, bootstrap="auto", build_tools=False: explicit or "/usr/bin/apptainer",
    )

    rc = module.main(
        [
            "--root",
            str(tmp_path / "root"),
            "--software",
            "alphafold3",
            "--skip-build",
            "--skip-import-smoke",
            "--skip-doctor",
            "--allow-missing-af3",
            "--submodule-set",
            "all",
            "--dry-run",
        ]
    )

    submodule_cmd = next(cmd[0] for cmd in commands if cmd[0][:3] == ["git", "submodule", "update"])
    assert rc == 0
    assert not any(part.startswith("vendor/") for part in submodule_cmd)


def test_needed_softwares_submodule_set_adds_full_needed_vendor_paths() -> None:
    module = _load_script_module()

    cmd = module._submodule_update_command(
        submodule_set="needed_softwares",
        depth=1,
        jobs=4,
    )

    for rel_path in module.RUNTIME_SUBMODULE_PATHS:
        assert rel_path in cmd
    for rel_path in (
        "vendor/dplm",
        "vendor/na_mpnn",
        "vendor/rpxdock",
        "vendor/willutil",
    ):
        assert rel_path in cmd


def test_selected_only_vendor_patching_uses_initialized_groups(monkeypatch) -> None:
    module = _load_script_module()
    commands = []
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, *, cwd=None, dry_run=False: commands.append(list(cmd)),
    )

    module._apply_vendor_patches(
        dry_run=False,
        selected_paths=("vendor/esm3",),
    )
    assert commands == []

    module._apply_vendor_patches(
        dry_run=False,
        selected_paths=("vendor/esm2", "vendor/openfold_esm2"),
    )
    assert commands[0][-2:] == ["--only", "esm2"]


def test_setup_host_envs_invokes_needed_softwares_env_profile(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()

    commands = []

    def fake_run(cmd, *, cwd=None, dry_run=False):
        commands.append((cmd, cwd, dry_run))

    monkeypatch.setattr(module, "_run", fake_run)

    profile = tmp_path / "sites" / "ty2.json"
    rc = module.main(
        [
            "--profile",
            "ty2",
            "--root",
            str(tmp_path / "root"),
            "--site-profile",
            str(profile),
            "--software",
            "usalign",
            "--skip-build",
            "--skip-submodules",
            "--skip-import-smoke",
            "--skip-doctor",
            "--setup-host-envs",
            "--host-env-yes",
            "--host-env-jobs",
            "3",
        ]
    )

    setup_cmd = next(cmd[0] for cmd in commands if "env_setup.py" in cmd[0][1])
    assert rc == 0
    assert setup_cmd[setup_cmd.index("--profile") + 1] == "needed_softwares"
    assert setup_cmd[setup_cmd.index("--jobs") + 1] == "3"
    assert setup_cmd[setup_cmd.index("--only") + 1] == "usalign"
    assert "-y" in setup_cmd


def test_import_smoke_injects_repo_vendor_import_paths(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    captured = {}

    def fake_run(cmd, *, cwd=None, dry_run=False):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["dry_run"] = dry_run

    monkeypatch.setattr(module, "_run", fake_run)
    args = module.build_parser().parse_args(
        [
            "--allow-missing-af3",
            "--skip-build",
            "--skip-doctor",
        ]
    )
    runner = tmp_path / "runner.py"

    module._run_import_smoke(args, runner)

    code = captured["cmd"][2]
    assert "/opt/Astrolabe" in code
    assert "/opt/Astrolabe/vendor/alphafold2" in code
    assert "/opt/Astrolabe/vendor/dplm/src" in code
    assert "assert gh_source.startswith('/opt/Astrolabe/')" in code
    assert "from astrolabe.app import _load_run_module" in code
    assert "_load_run_module()" in code


def test_esmfold_import_smoke_loads_model_implementation(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    captured = {}
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, *, cwd=None, dry_run=False: captured.update(cmd=cmd),
    )
    args = module.build_parser().parse_args(["--skip-build", "--skip-doctor"])

    module._run_import_smoke(args, tmp_path / "runner.py", selected=("esmfold",))

    code = captured["cmd"][2]
    assert "gh_esmfold._load_esmfold_model_class()" in code


def test_esmif_import_smoke_covers_graph_runtime(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    commands = []
    monkeypatch.setattr(module, "_run", lambda cmd, **_kwargs: commands.append(cmd))
    args = module.argparse.Namespace(srun=None, dry_run=False)

    module._run_import_smoke(args, tmp_path / "runner.py", selected=("esmif",))

    code = commands[0][2]
    assert "esm2.inverse_folding" in code
    assert "torch_geometric" in code


def test_skip_build_import_smoke_allows_prebuilt_feature_superset(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    captured = {}
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, *, cwd=None, dry_run=False: captured.update(cmd=cmd),
    )
    args = module.build_parser().parse_args(["--skip-build", "--skip-doctor"])

    module._run_import_smoke(args, tmp_path / "runner.py", selected=("esm3",))

    assert "'esm3'" in captured["cmd"][2]
    assert "find_spec('pyrosetta')" not in captured["cmd"][2]


def test_reduced_build_import_smoke_rejects_unselected_pyrosetta(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    captured = {}
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, *, cwd=None, dry_run=False: captured.update(cmd=cmd),
    )
    args = module.build_parser().parse_args(["--skip-doctor"])

    module._run_import_smoke(args, tmp_path / "runner.py", selected=("esm3",))

    assert "find_spec('pyrosetta')" in captured["cmd"][2]


def test_doctor_does_not_shadow_installed_vendor_packages(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    captured = {}

    def fake_run(cmd, *, cwd=None, dry_run=False):
        captured["cmd"] = cmd

    monkeypatch.setattr(module, "_run", fake_run)
    args = module.build_parser().parse_args(["--allow-missing-af3", "--skip-build"])

    module._run_doctor(args, tmp_path / "runner.py")

    code = captured["cmd"][2]
    assert "sys.path.insert(0, '/opt/Astrolabe/vendor')" not in code
    assert "sys.path.insert(0, '/opt/Astrolabe')" in code


def test_full_needed_profile_writes_required_tool_and_env_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()

    def write_executable(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o755)
        return path

    fake_repo = tmp_path / "repo"
    for rel_path in (
        "vendor/dplm",
        "vendor/na_mpnn",
        "vendor/rpxdock",
        "vendor/willutil",
    ):
        (fake_repo / rel_path).mkdir(parents=True)
    (fake_repo / "vendor" / "na_mpnn" / "models" / "design_model").mkdir(parents=True)
    (fake_repo / "vendor" / "na_mpnn" / "models" / "design_model" / "s_19137.pt").write_text(
        "stub\n",
        encoding="utf-8",
    )
    write_executable(fake_repo / "vendor" / "usalign" / "USalign")
    write_executable(fake_repo / "astrolabe" / "assets" / "DAlphaBall.gcc")
    monkeypatch.setattr(module, "REPO_ROOT", fake_repo)
    monkeypatch.setattr(module, "_validate_selected_assets", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_preflight_selected_resources", lambda **_kwargs: None)
    monkeypatch.setattr(module, "_download_public_assets", lambda **_kwargs: None)

    root = tmp_path / "root"
    af3 = tmp_path / "af3"
    af3.mkdir()
    af2 = tmp_path / "alphafold2"
    (af2 / "params").mkdir(parents=True)
    (af2 / "params" / module.AF2_SENTINEL).write_text("stub\n", encoding="utf-8")
    ligandmpnn = tmp_path / "ligandmpnn"
    ligandmpnn.mkdir()
    for name in module.LIGANDMPNN_CKPTS:
        (ligandmpnn / name).write_text("stub\n", encoding="utf-8")
    protenix = tmp_path / "protenix"
    (protenix / "release_model").mkdir(parents=True)
    (protenix / "release_data").mkdir(parents=True)
    for rel_path in (
        "release_model/protenix-v2.pt",
        "release_model/protenix_base_constraint_v0.5.0.pt",
        "release_data/components.v20240608.cif",
        "release_data/components.v20240608.cif.rdkit_mol.pkl",
    ):
        (protenix / rel_path).write_text("stub\n", encoding="utf-8")

    dplm_python = write_executable(root / "envs" / "dplm" / "bin" / "python")
    (root / "cache" / "huggingface").mkdir(parents=True)
    rpxdock_python = write_executable(root / "envs" / "rpxdock" / "bin" / "python")
    molfile = tmp_path / "rosetta_public" / "molfile_to_params.py"
    molfile.parent.mkdir()
    molfile.write_text("# rosetta stub\n", encoding="utf-8")
    haddock3 = write_executable(tmp_path / "haddock3")
    autodock_gpu = write_executable(tmp_path / "autodock_gpu_128wi")
    autogrid4 = write_executable(tmp_path / "autogrid4")
    ananas = write_executable(tmp_path / "ananas")
    libgfortran3 = tmp_path / "libgfortran3"
    libgfortran3.mkdir()
    (libgfortran3 / "libgfortran.so.3").write_text("stub\n", encoding="utf-8")

    profile = tmp_path / "sites" / "ty2.json"
    rc = module.main(
        [
            "--profile",
            "ty2",
            "--root",
            str(root),
            "--site-profile",
            str(profile),
            "--skip-build",
            "--skip-submodules",
            "--skip-import-smoke",
            "--skip-doctor",
            "--needed-softwares",
            "full",
            "--accept-license",
            "pyrosetta-license,ananas-academic,rosetta-license",
            "--af3-weights",
            str(af3),
            "--af2-params",
            str(af2),
            "--ligandmpnn-params",
            str(ligandmpnn),
            "--protenix-assets",
            str(protenix),
            "--molfile-to-params",
            str(molfile),
            "--haddock3-bin",
            str(haddock3),
            "--autodock-gpu",
            str(autodock_gpu),
            "--autogrid4-bin",
            str(autogrid4),
            "--ananas-bin",
            str(ananas),
            "--libgfortran3",
            str(libgfortran3),
        ]
    )

    payload = json.loads(profile.read_text(encoding="utf-8"))
    env_defaults = payload["env_defaults"]
    binds = payload["binds"]
    assert rc == 0
    assert str(dplm_python.parent.parent) + ":/envs/dplm" in binds
    assert str(rpxdock_python.parent.parent) + ":/envs/rpxdock" in binds
    assert str(autodock_gpu) + ":/usr/bin/autodock_gpu_128wi" in binds
    assert str(autodock_gpu) + ":/usr/bin/autodock_gpu" in binds
    assert env_defaults["GH_MOLFILE_TO_PARAMS"] == "/rosetta/molfile_to_params.py"
    assert env_defaults["GH_HADDOCK3_BIN"] == "/usr/bin/haddock3"
    assert env_defaults["GH_AUTODOCK_GPU"] == "/usr/bin/autodock_gpu_128wi"
    assert env_defaults["GH_AUTOGRID4"] == "/usr/bin/autogrid4"
    assert env_defaults["GH_ANANAS"] == "/usr/local/bin/AnAnaS"
    assert not any("/tools/ananas" in bind for bind in binds)
    assert not any("rosetta_public" in bind for bind in binds)
    assert env_defaults["GH_DPLM_PYTHON"] == "/envs/dplm/bin/python"
    assert env_defaults["GH_DPLM_REPO"] == "/opt/Astrolabe/vendor/dplm"
    assert env_defaults["GH_NAMPNN_REPO"] == "/opt/Astrolabe/vendor/na_mpnn"
    assert env_defaults["GH_RPXDOCK_PYTHON"] == "/envs/rpxdock/bin/python"
    assert env_defaults["GH_RPXDOCK_REPO"] == "/opt/Astrolabe/vendor/rpxdock"
    assert env_defaults["GH_WILLUTIL_REPO"] == "/opt/Astrolabe/vendor/willutil"
    assert env_defaults["GH_USALIGN"] == "/opt/Astrolabe/vendor/usalign/USalign"
    assert env_defaults["GH_DALPHABALL"] == "/opt/Astrolabe/astrolabe/assets/DAlphaBall.gcc"
    assert env_defaults["GH_WEIGHT_PROTENIX"] == "/models/protenix/release_model/protenix-v2.pt"
    assert env_defaults["APPTAINERENV_LD_LIBRARY_PATH"] == "/host-libs/libgfortran3"


def test_skip_container_steps_does_not_require_apptainer(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()

    commands = []

    def fake_run(cmd, *, cwd=None, dry_run=False):
        commands.append((cmd, cwd, dry_run))

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    profile = tmp_path / "sites" / "ty2.json"
    rc = module.main(
        [
            "--profile",
            "ty2",
            "--root",
            str(tmp_path / "root"),
            "--site-profile",
            str(profile),
            "--software",
            "autodock-vina",
            "--skip-build",
            "--skip-import-smoke",
            "--skip-doctor",
            "--allow-missing-af3",
        ]
    )

    payload = json.loads(profile.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["apptainer_bin"] == "apptainer"
    assert not any(cmd[0][:3] == ["git", "submodule", "update"] for cmd in commands)
    assert not (tmp_path / "root" / "tools" / "gh_ty2_container_python.py").exists()


def test_missing_af3_requires_explicit_allow_missing(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    monkeypatch.delenv("GH_WEIGHT_AF3", raising=False)

    monkeypatch.setattr(
        module,
        "_detect_apptainer",
        lambda explicit, dry_run=False, root=None, bootstrap="auto", build_tools=False: explicit or "/usr/bin/apptainer",
    )

    try:
        module.main(
            [
                "--root",
                str(tmp_path),
                "--skip-build",
                "--skip-import-smoke",
                "--skip-doctor",
            ]
        )
    except SystemExit as exc:
        assert "--af3-weights is required" in str(exc)
    else:
        raise AssertionError("missing AF3 weights should fail without --allow-missing-af3")


def test_gpu_site_profile_sets_nvidia_visibility_default(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    profile = tmp_path / "profile.json"
    runner = tmp_path / "runner.py"
    af3 = tmp_path / "af3"
    af3.mkdir()
    (af3 / "af3.bin").write_text("stub\n", encoding="utf-8")
    sif = tmp_path / "root" / "images" / "Astrolabe_af3.sif"
    sif.parent.mkdir(parents=True)
    sif.write_bytes(b"sif" * 400_000)

    monkeypatch.setattr(
        module,
        "_detect_apptainer",
        lambda explicit, dry_run=False, root=None, bootstrap="auto", build_tools=False: "/usr/bin/apptainer",
    )
    monkeypatch.setattr(module, "_check_apptainer_site_support", lambda **_kwargs: None)
    monkeypatch.setattr(module, "_write_runner_from_profile", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_run_import_smoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_run_doctor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_sif_identity",
        lambda path, *, build_features, require_build_metadata=False: {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": "0" * 64,
            "build_features": [],
            "build_metadata": str(path) + ".build.json",
        },
    )

    rc = module.main(
        [
            "--root",
            str(tmp_path / "root"),
            "--skip-build",
            "--skip-submodules",
            "--software",
            "alphafold3",
            "--af3-weights",
            str(af3),
            "--site-profile",
            str(profile),
            "--runner",
            str(runner),
            "--software-manifest",
            str(tmp_path / "software-manifest.json"),
        ]
    )

    assert rc == 0
    payload = json.loads(profile.read_text(encoding="utf-8"))
    assert payload["env_defaults"]["NVIDIA_VISIBLE_DEVICES"] == "all"
    assert payload["env_defaults"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert payload["env_defaults"]["GH_CUDA_PREINITIALIZE"] == "1"


def test_container_definition_includes_af2_runtime_deps() -> None:
    text = (REPO_ROOT / "container" / "requirements.runtime.in").read_text(encoding="utf-8")

    assert "ml_collections==1.1.0" in text
    assert "tensorflow-cpu==2.15.1" in text


def test_container_definition_installs_vendored_inprocess_packages() -> None:
    text = DEF_PATH.read_text(encoding="utf-8")

    for rel_path in (
        "vendor/protenix",
        "vendor/alphafold2",
        "vendor/rhofold",
        "vendor/chai_lab",
        "vendor/boltz",
        "vendor/esm2",
        "vendor/esm3",
    ):
        assert f"--no-deps -e /opt/Astrolabe/{rel_path}" in text


def test_container_definition_includes_shared_runtime_deps() -> None:
    text = (REPO_ROOT / "container" / "requirements.runtime.in").read_text(encoding="utf-8")

    for dep in (
        "pytorch-lightning==2.5.0",
        "lightning-fabric==2.5.0",
        "einops==0.8.0",
        "einx==0.3.0",
        "fairscale==0.4.13",
        "mashumaro==3.14",
        "numpy==1.26.4",
        "huggingface-hub==0.36.0",
        "transformers==4.52.4",
        "chembl_structure_pipeline==1.2.2",
        "pandera==0.28.1",
        "pyarrow==23.0.0",
        "modelcif==1.6",
        "gemmi==0.7.4",
    ):
        assert dep in text


def test_container_runtime_lock_is_complete_and_hash_pinned() -> None:
    lock = (REPO_ROOT / "container" / "requirements.runtime.lock").read_text(
        encoding="utf-8"
    )

    locked_packages = [
        line for line in lock.splitlines() if line and line[0].isalnum() and "==" in line
    ]
    assert len(locked_packages) == 179
    for dep in ("torch==2.7.1", "triton==3.3.1", "jax==0.4.34"):
        assert dep in lock
    assert lock.count("--hash=sha256:") > 300


def test_download_public_assets_uses_stdlib_for_af2_and_protenix(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    downloaded = []

    def fake_download(url, dst, *, dry_run, **_kwargs):
        downloaded.append((url, dst, dry_run))
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.suffix == ".tar":
            with tarfile.open(dst, "w") as tf:
                for name in module.AF2_PARAM_FILES:
                    payload = tmp_path / name
                    payload.write_text("stub\n", encoding="utf-8")
                    tf.add(payload, arcname=name)
        else:
            dst.write_text("stub\n", encoding="utf-8")

    monkeypatch.setattr(module, "_download_url", fake_download)
    def validate_fake_af2(path: Path) -> None:
        missing = [name for name in module.AF2_PARAM_FILES if not (path / name).is_file()]
        if missing:
            raise RuntimeError(f"missing {missing[0]}")

    monkeypatch.setattr(module, "_validate_af2_params_dir", validate_fake_af2)
    monkeypatch.setattr(module, "_validate_af2_archive", lambda _path: None)

    af2_data_dir = tmp_path / "assets" / "alphafold2"
    ligandmpnn_dir = tmp_path / "assets" / "ligandmpnn"
    protenix_dir = tmp_path / "repos" / "Protenix"

    module._download_public_assets(
        root=tmp_path,
        af2_data_dir=af2_data_dir,
        ligandmpnn_dir=ligandmpnn_dir,
        protenix_dir=protenix_dir,
        dry_run=False,
    )

    assert (af2_data_dir / "params" / module.AF2_SENTINEL).exists()
    assert all((ligandmpnn_dir / name).exists() for name in module.LIGANDMPNN_CKPTS)
    assert (protenix_dir / "release_model" / "protenix-v2.pt").exists()
    assert (protenix_dir / "release_data" / "components.v20240608.cif").exists()
    assert not any("get_model_params.sh" in item[0] for item in downloaded)


def test_af2_sentinel_only_install_reuses_archive_and_self_heals(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    data_dir = tmp_path / "alphafold2"
    params_dir = data_dir / "params"
    params_dir.mkdir(parents=True)
    (params_dir / module.AF2_SENTINEL).write_text("incomplete\n", encoding="utf-8")
    archive = data_dir / ".downloads" / Path(module.AF2_PARAMS_URL).name
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w") as handle:
        for name in module.AF2_PARAM_FILES:
            source = tmp_path / name
            source.write_text("complete\n", encoding="utf-8")
            handle.add(source, arcname=name)

    def validate_fake(path: Path) -> None:
        missing = [name for name in module.AF2_PARAM_FILES if not (path / name).is_file()]
        if missing:
            raise RuntimeError(f"missing {missing[0]}")

    downloads = []

    def reuse_archive(_url, destination, **_kwargs):
        downloads.append(destination)
        assert destination == archive
        assert destination.is_file()

    monkeypatch.setattr(module, "_validate_af2_params_dir", validate_fake)
    monkeypatch.setattr(module, "_validate_af2_archive", lambda _path: None)
    monkeypatch.setattr(module, "_download_url", reuse_archive)

    module._download_af2_params(data_dir, dry_run=False)

    assert downloads == [archive]
    assert all((params_dir / name).is_file() for name in module.AF2_PARAM_FILES)
    assert not archive.exists()


def test_af2_full_size_but_invalid_archive_is_quarantined_and_retried(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    data_dir = tmp_path / "alphafold2"
    archive = data_dir / ".downloads" / Path(module.AF2_PARAMS_URL).name
    archive.parent.mkdir(parents=True)

    def write_archive(path: Path, *, unsafe: bool) -> None:
        with tarfile.open(path, "w") as handle:
            for name in module.AF2_PARAM_FILES:
                payload = b"parameter"
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                handle.addfile(info, io.BytesIO(payload))
            if unsafe:
                link = tarfile.TarInfo("unsafe-link")
                link.type = tarfile.SYMTYPE
                link.linkname = "../outside"
                handle.addfile(link)

    write_archive(archive, unsafe=True)
    stale_invalid = archive.with_name(archive.name + ".invalid.previous-run")
    stale_invalid.write_bytes(b"old-invalid")
    downloads = []

    def fake_download(_url, destination, **_kwargs):
        downloads.append(destination)
        if not destination.exists():
            write_archive(destination, unsafe=False)

    monkeypatch.setattr(module, "_download_url", fake_download)
    monkeypatch.setattr(module, "_validate_af2_archive", lambda _path: None)

    def validate_fake(path: Path) -> None:
        missing = [name for name in module.AF2_PARAM_FILES if not (path / name).is_file()]
        if missing:
            raise RuntimeError(f"missing {missing[0]}")

    monkeypatch.setattr(module, "_validate_af2_params_dir", validate_fake)

    module._download_af2_params(data_dir, dry_run=False)

    assert downloads == [archive, archive]
    assert all(
        (data_dir / "params" / name).is_file() for name in module.AF2_PARAM_FILES
    )
    assert not list(archive.parent.glob(archive.name + ".invalid.*"))


def test_explicit_ligandmpnn_reuse_is_exact_and_skips_unneeded_submodules(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    commands = []
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, *, cwd=None, dry_run=False: commands.append((cmd, cwd, dry_run)),
    )

    ligandmpnn = tmp_path / "ligandmpnn"
    ligandmpnn.mkdir()
    profile = tmp_path / "site.json"
    rc = module.main(
        [
            "--root",
            str(tmp_path / "root"),
            "--software",
            "ligandmpnn",
            "--ligandmpnn-params",
            str(ligandmpnn),
            "--site-profile",
            str(profile),
            "--skip-build",
            "--skip-import-smoke",
            "--skip-doctor",
            "--dry-run",
        ]
    )

    payload = json.loads(profile.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["env_defaults"]["GH_WEIGHT_LIGANDMPNN"] == "/models/ligandmpnn"
    assert "GH_WEIGHT_AF3" not in payload["env_defaults"]
    assert "GH_WEIGHT_AF2" not in payload["env_defaults"]
    assert not any(cmd[:3] == ["git", "submodule", "update"] for cmd, _, _ in commands)


def test_explicit_host_backend_passes_only_resolved_env_to_env_setup(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    commands = []
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, *, cwd=None, dry_run=False: commands.append((cmd, cwd, dry_run)),
    )
    dplm_python = tmp_path / "root" / "envs" / "dplm" / "bin" / "python"
    dplm_python.parent.mkdir(parents=True)
    dplm_python.write_text("#!/bin/sh\n", encoding="utf-8")
    dplm_python.chmod(0o755)
    (tmp_path / "root" / "cache" / "huggingface").mkdir(parents=True)

    module.main(
        [
            "--root",
            str(tmp_path / "root"),
            "--software",
            "dplm",
            "--setup-host-envs",
            "--site-profile",
            str(tmp_path / "site.json"),
            "--skip-build",
            "--skip-submodules",
            "--skip-import-smoke",
            "--skip-doctor",
            "--dry-run",
        ]
    )

    env_cmd = next(cmd for cmd, _, _ in commands if "env_setup.py" in cmd[1])
    assert env_cmd[env_cmd.index("--only") + 1] == "dplm"
    profile = json.loads((tmp_path / "site.json").read_text(encoding="utf-8"))
    relative_snapshot = module.hf_runtime_snapshot_path(
        tmp_path / "root" / "cache" / "huggingface", "dplm"
    ).relative_to(tmp_path / "root" / "cache" / "huggingface")
    assert profile["env_defaults"]["GH_DPLM_MODEL_SNAPSHOT"] == str(
        Path("/models/huggingface") / relative_snapshot
    )
    assert profile["env_defaults"]["HF_HUB_OFFLINE"] == "1"
    assert profile["env_defaults"]["TRANSFORMERS_OFFLINE"] == "1"


def test_default_core_requires_pyrosetta_license_before_auto_download(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()

    monkeypatch.setattr(module, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_detect_apptainer",
        lambda explicit, dry_run=False, root=None, bootstrap="auto", build_tools=False: explicit or "/usr/bin/apptainer",
    )

    with pytest.raises(SystemExit, match="pyrosetta-license"):
        module.main(
            [
                "--root",
                str(tmp_path / "root"),
                "--skip-build",
                "--skip-submodules",
                "--skip-import-smoke",
                "--skip-doctor",
                "--allow-missing-af3",
                "--dry-run",
            ]
        )


def test_implicit_core_download_plan_matches_explicit_core(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    root = tmp_path / "root"
    for path in (
        root / "assets" / "alphafold2",
        root / "assets" / "ligandmpnn",
        root / "assets" / "boltz",
        root / "repos" / "Protenix" / "release_model",
        root / "repos" / "Protenix" / "release_data",
    ):
        path.mkdir(parents=True, exist_ok=True)
    for rel_path in module.PROTENIX_ASSET_URLS:
        (root / "repos" / "Protenix" / rel_path).write_text("stub\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_download(**kwargs):
        captured["download"] = tuple(kwargs["selected"])

    def fake_prefetch(_args, **kwargs):
        captured["prefetch"] = set(kwargs["selected"])

    monkeypatch.setattr(module, "_download_public_assets", fake_download)
    monkeypatch.setattr(module, "_run_prefetch_command", fake_prefetch)
    monkeypatch.setattr(
        module,
        "_detect_apptainer",
        lambda explicit, dry_run=False, root=None, bootstrap="auto", build_tools=False: "/usr/bin/apptainer",
    )

    assert module.main(
        [
            "--root",
            str(root),
            "--download-assets",
            "--accept-license",
            "pyrosetta-license",
            "--allow-missing-af3",
            "--skip-build",
            "--skip-submodules",
            "--skip-import-smoke",
            "--skip-doctor",
            "--dry-run",
        ]
    ) == 0

    assert captured["download"] == module.resolve_software(("core",))
    assert captured["prefetch"] == {"boltz"}


def test_container_prefetch_temporarily_enables_huggingface_network(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    captured = {}
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, *, cwd=None, dry_run=False: captured.update(
            cmd=list(cmd), cwd=cwd, dry_run=dry_run
        ),
    )
    args = module.argparse.Namespace(
        asset_download_jobs=4,
        container_repo_root="/opt/Astrolabe",
        dry_run=False,
        srun=None,
    )

    module._run_prefetch_command(
        args,
        python=tmp_path / "runner.py",
        selected=("esm3",),
        boltz_cache=tmp_path / "boltz",
        chai_cache=tmp_path / "chai",
        torch_cache=tmp_path / "torch",
        hf_cache=tmp_path / "huggingface",
        container_paths=True,
    )

    assert captured["cmd"][:3] == [
        "env",
        "HF_HUB_OFFLINE=0",
        "TRANSFORMERS_OFFLINE=0",
    ]
    assert captured["cmd"][3] == str(tmp_path / "runner.py")


def test_rhofold_checkpoint_env_preserves_custom_filename(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    checkpoint = tmp_path / "approved" / "rhofold_pretrained_params.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("stub\n", encoding="utf-8")
    profile_path = tmp_path / "site.json"
    monkeypatch.setenv("GH_RHOFOLD_CHECKPOINT", str(checkpoint))

    assert module.main(
        [
            "--root",
            str(tmp_path / "root"),
            "--software",
            "rhofold",
            "--site-profile",
            str(profile_path),
            "--skip-build",
            "--skip-submodules",
            "--skip-import-smoke",
            "--skip-doctor",
            "--dry-run",
        ]
    ) == 0

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert f"{checkpoint.parent}:/models/rhofold" in profile["binds"]
    assert profile["env_defaults"]["GH_RHOFOLD_CHECKPOINT"] == (
        "/models/rhofold/rhofold_pretrained_params.pt"
    )


def test_missing_manual_resource_fails_before_sif_build(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    builds = []
    monkeypatch.setattr(module, "_build_sif", lambda *args, **kwargs: builds.append((args, kwargs)))
    monkeypatch.setattr(
        module,
        "_detect_apptainer",
        lambda explicit, dry_run=False, root=None, bootstrap="auto", build_tools=False: "/usr/bin/apptainer",
    )
    monkeypatch.setattr(module, "_check_apptainer_site_support", lambda **_kwargs: None)

    with pytest.raises(SystemExit, match="RF3/RFD3 repository"):
        module.main(
            [
                "--root",
                str(tmp_path / "root"),
                "--software",
                "rf3",
                "--skip-submodules",
                "--skip-import-smoke",
                "--skip-doctor",
            ]
        )
    assert builds == []


def test_rfd3_selection_binds_shared_foundry_runtime_and_only_rfd3_checkpoint(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    foundry = tmp_path / "foundry"
    (foundry / "src").mkdir(parents=True)
    (foundry / "models" / "rfd3" / "src").mkdir(parents=True)
    foundry_python = foundry / ".venv" / "bin" / "python"
    foundry_python.parent.mkdir(parents=True)
    uv_python = tmp_path / "uv" / "cpython-3.12" / "bin" / "python3.12"
    uv_python.parent.mkdir(parents=True)
    uv_python.write_text("#!/bin/sh\n", encoding="utf-8")
    uv_python.chmod(0o755)
    foundry_python.symlink_to(uv_python)
    (foundry / ".venv" / "pyvenv.cfg").write_text(
        f"home = {uv_python.parent}\n",
        encoding="utf-8",
    )
    checkpoint = foundry / "rfd3.ckpt"
    checkpoint.write_text("checkpoint\n", encoding="utf-8")
    profile_path = tmp_path / "site.json"

    assert module.main(
        [
            "--root",
            str(tmp_path / "root"),
            "--software",
            "rfd3",
            "--rf3-repo",
            str(foundry),
            "--rf3-python",
            str(foundry_python),
            "--rfd3-checkpoint",
            str(checkpoint),
            "--site-profile",
            str(profile_path),
            "--skip-build",
            "--skip-submodules",
            "--skip-import-smoke",
            "--skip-doctor",
            "--dry-run",
        ]
    ) == 0

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert f"{foundry}:/ext/rf3" in profile["binds"]
    assert f"{foundry_python.parent.parent}:/envs/rf3" in profile["binds"]
    assert f"{foundry_python.parent.parent}:{foundry_python.parent.parent}" in profile["binds"]
    assert f"{uv_python.parent.parent}:{uv_python.parent.parent}" in profile["binds"]
    assert f"{checkpoint}:/models/rfd3/model.pt" in profile["binds"]
    assert profile["env_defaults"]["GH_RF3_REPO"] == "/ext/rf3"
    assert profile["env_defaults"]["GH_RF3_PYTHON"] == "/envs/rf3/bin/python"
    assert profile["env_defaults"]["GH_WEIGHT_RFD3"] == "/models/rfd3/model.pt"
    assert "GH_WEIGHT_RF3" not in profile["env_defaults"]


def test_rfd3_selection_downloads_exact_official_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    calls = []
    monkeypatch.setattr(
        module,
        "_download_url",
        lambda url, dst, **kwargs: calls.append((url, dst, kwargs)),
    )
    checkpoint = tmp_path / "assets" / "rfd3.ckpt"

    module._download_public_assets(
        root=tmp_path,
        af2_data_dir=None,
        ligandmpnn_dir=tmp_path / "ligandmpnn",
        protenix_dir=tmp_path / "protenix",
        rfd3_checkpoint=checkpoint,
        selected=("rfd3",),
        jobs=1,
        dry_run=True,
    )

    assert calls == [
        (
            module.RFD3_ASSET[0],
            checkpoint,
            {
                "dry_run": True,
                "minimum_size": module.RFD3_ASSET[1],
                "expected_sha256": module.RFD3_ASSET[2],
            },
        )
    ]


def test_missing_manual_resource_fails_before_any_install_side_effect(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    side_effects = []
    monkeypatch.setattr(
        module,
        "_run",
        lambda *args, **kwargs: side_effects.append(("run", args, kwargs)),
    )
    monkeypatch.setattr(
        module,
        "_download_public_assets",
        lambda **kwargs: side_effects.append(("download", kwargs)),
    )
    monkeypatch.setattr(
        module,
        "_detect_apptainer",
        lambda *args, **kwargs: side_effects.append(("apptainer", args, kwargs)) or "/usr/bin/apptainer",
    )

    with pytest.raises(SystemExit, match="RF3/RFD3 repository"):
        module.main(
            [
                "--root",
                str(tmp_path / "root"),
                "--software",
                "rf3",
                "--download-assets",
                "--skip-import-smoke",
                "--skip-doctor",
            ]
        )

    assert side_effects == []


def test_public_hf_models_do_not_require_token_after_license_acceptance(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    monkeypatch.setattr(
        module,
        "_resolve_hf_token",
        lambda _cache: (_ for _ in ()).throw(AssertionError("token lookup is not expected")),
    )
    module._preflight_hf_credentials(
        selected=("esm3", "esmfold2"),
        markers={"esm3": [], "esmfold2": []},
        hf_cache=tmp_path / "huggingface",
        download_assets=True,
        dry_run=False,
    )


def test_hf_runtime_snapshot_manifest_is_exact_and_hashed(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    cache = tmp_path / "hf"
    monkeypatch.setenv("GH_ASSET_VALIDATION_CACHE", str(tmp_path / "validation"))
    snapshot = module.hf_runtime_snapshot_path(cache, "dplm")
    required = module.HF_SNAPSHOT_REQUIRED_FILES["dplm"]
    files = []
    for rel_path in required:
        path = snapshot / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{rel_path}\n", encoding="utf-8")
        files.append(
            {
                "path": rel_path,
                "size": path.stat().st_size,
                "sha256": module._sha256_file(path),
            }
        )
    monkeypatch.setitem(
        module.HF_SNAPSHOT_FILE_SPECS,
        "dplm",
        {
            entry["path"]: (entry["size"], entry["sha256"])
            for entry in files
        },
    )
    manifest = module.hf_snapshot_manifest_path(cache, "dplm")
    module._write_json(
        manifest,
        {
            "schema_version": 2,
            "software": "dplm",
            "repo_id": module.HF_MODEL_REPOS["dplm"],
            "revision": module.SOFTWARE["dplm"].revision,
            "runtime_snapshot_relative": str(snapshot.relative_to(cache)),
            "allow_patterns": list(required),
            "required_files": list(required),
            "files": files,
        },
    )
    markers = {"dplm": [*(snapshot / path for path in required), manifest]}

    module._validate_selected_assets(("dplm",), markers)
    (snapshot / "README.md").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not exact"):
        module._validate_selected_assets(("dplm",), markers)


def test_software_manifest_merges_previous_verified_entries(tmp_path: Path) -> None:
    module = _load_script_module()
    manifest = tmp_path / "software-manifest.json"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("weights\n", encoding="utf-8")

    module._write_software_manifest(
        manifest,
        selected=("rhofold",),
        markers={"rhofold": [checkpoint]},
        accepted_licenses=("first-license",),
        dry_run=False,
    )
    module._write_software_manifest(
        manifest,
        selected=("dssp",),
        markers={"dssp": []},
        accepted_licenses=("second-license",),
        dry_run=False,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 4
    assert payload["last_selection"] == ["dssp"]
    assert set(payload["software"]) == {"rhofold", "dssp"}
    assert payload["installed"] == ["dssp", "rhofold"]
    assert payload["accepted_licenses"] == ["first-license", "second-license"]


def test_software_manifest_binds_pyrosetta_readiness_to_sif_feature(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    manifest = tmp_path / "software-manifest.json"
    identity = {
        "path": str(tmp_path / "pyrosetta.sif"),
        "size": 123,
        "sha256": "a" * 64,
        "build_features": ["pyrosetta"],
        "import_smoke_passed": True,
        "doctor_passed": True,
    }
    module._write_software_manifest(
        manifest,
        selected=("pyrosetta",),
        markers={"pyrosetta": []},
        accepted_licenses=("pyrosetta-license",),
        dry_run=False,
        container_identity=identity,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["container"] == identity
    assert payload["software"]["pyrosetta"]["ready"] is True
    assert payload["software"]["pyrosetta"]["resources"][0]["kind"] == (
        "sif-build-feature"
    )


def test_software_manifest_records_exact_surfpro_variant(tmp_path: Path) -> None:
    module = _load_script_module()
    manifest = tmp_path / "software-manifest.json"

    module._write_software_manifest(
        manifest,
        selected=("surfpro",),
        markers={"surfpro": []},
        dry_run=False,
        selected_asset_variants={"surfpro": "pretrain"},
    )

    selected_asset = json.loads(manifest.read_text(encoding="utf-8"))["software"][
        "surfpro"
    ]["selected_asset"]
    assert selected_asset == {
        "variant": "pretrain",
        "url": module.SURFPRO_PRETRAIN_ASSET[0],
        "size": module.SURFPRO_PRETRAIN_ASSET[1],
        "sha256": module.SURFPRO_PRETRAIN_ASSET[2],
    }


def test_surfpro_validation_rejects_different_approved_variant(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    checkpoint = tmp_path / "checkpoint_best.pt"
    checkpoint.write_bytes(b"cath")
    cath = (
        "https://example.test/cath",
        checkpoint.stat().st_size,
        hashlib.sha256(b"cath").hexdigest(),
    )
    pretrain = (
        "https://example.test/pretrain",
        checkpoint.stat().st_size,
        hashlib.sha256(b"pretrain").hexdigest(),
    )

    module._validate_selected_assets(
        ("surfpro",),
        {"surfpro": [checkpoint]},
        surfpro_asset=cath,
    )
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        module._validate_selected_assets(
            ("surfpro",),
            {"surfpro": [checkpoint]},
            surfpro_asset=pretrain,
        )


def test_software_manifest_concurrent_writers_preserve_all_entries(tmp_path: Path) -> None:
    manifest = tmp_path / "software-manifest.json"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    software = ("dssp", "usalign", "dalphaball", "autodock-vina")
    processes = [
        context.Process(
            target=_concurrent_manifest_worker,
            args=(str(SCRIPT_PATH), str(manifest), name, start),
        )
        for name in software
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert set(payload["software"]) == set(software)
    assert set(payload["installed"]) == set(software)


def test_software_manifest_lock_times_out_without_touching_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    manifest = tmp_path / "software-manifest.json"
    manifest.write_text('{"software": {}}\n', encoding="utf-8")
    original = manifest.read_bytes()
    lock_path = manifest.with_name(manifest.name + ".lock")
    lock_path.touch()
    monkeypatch.setattr(module, "SOFTWARE_MANIFEST_LOCK_TIMEOUT_SECONDS", 0.05)

    with lock_path.open("a+b") as lock_handle:
        module.fcntl.flock(lock_handle.fileno(), module.fcntl.LOCK_EX)
        with pytest.raises(SystemExit, match="Timed out waiting"):
            module._write_software_manifest(
                manifest,
                selected=("dssp",),
                markers={"dssp": []},
                dry_run=False,
            )

    assert manifest.read_bytes() == original


def test_dry_run_does_not_require_apptainer(monkeypatch) -> None:
    module = _load_script_module()

    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    assert module._detect_apptainer(None, dry_run=True) == "apptainer"


def test_micromamba_bootstrap_uses_immutable_conda_forge_build_url(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    captured = {}

    def fake_download(url, target, **kwargs):
        captured.update(url=url, target=target, kwargs=kwargs)

    monkeypatch.setattr(module, "_download_url", fake_download)

    result = module._bootstrap_micromamba(tmp_path / "bin", dry_run=True)

    assert captured["url"] == (
        "https://api.anaconda.org/download/conda-forge/micromamba/2.8.1/"
        "linux-64/micromamba-2.8.1-0.tar.bz2"
    )
    assert captured["target"].name == "micromamba-2.8.1-0.tar.bz2"
    assert captured["kwargs"]["expected_sha256"] == (
        "a934c3709c997feae403a27fd1e321c106d26ffa4f294800ffb11cbc9a3e8515"
    )
    assert result == str(tmp_path / "bin" / "micromamba")


def test_config_only_docker_runner_does_not_require_docker(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()

    def fail_if_docker_is_probed(_explicit):
        raise AssertionError("docker probed")

    def fail_if_apptainer_is_probed(*_args, **_kwargs):
        raise AssertionError("apptainer probed")

    monkeypatch.setattr(module, "_detect_docker", fail_if_docker_is_probed)
    monkeypatch.setattr(module, "_detect_apptainer", fail_if_apptainer_is_probed)
    profile = tmp_path / "site.json"
    runner = tmp_path / "runner.py"

    assert module.main(
        [
            "--root",
            str(tmp_path / "root"),
            "--software",
            "dssp",
            "--runner-backend",
            "docker-apptainer",
            "--site-profile",
            str(profile),
            "--runner",
            str(runner),
            "--skip-build",
            "--skip-submodules",
            "--skip-import-smoke",
            "--skip-doctor",
            "--dry-run",
        ]
    ) == 0
    payload = json.loads(profile.read_text(encoding="utf-8"))
    assert payload["docker_bin"] is None
    assert not runner.exists()


def test_container_inspect_command_covers_apptainer_and_docker(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    sif = tmp_path / "image.sif"
    args = module.build_parser().parse_args([])
    args.docker_bin_resolved = None
    host_cmd = module._container_inspect_command(
        args,
        apptainer_bin="/opt/apptainer/bin/apptainer",
        sif=sif,
    )
    assert host_cmd[2:] == [
        "/opt/apptainer/bin/apptainer",
        "inspect",
        str(sif),
    ]

    args.runner_backend = "docker-apptainer"
    args.docker_bin_resolved = "/usr/bin/docker"
    docker_cmd = module._container_inspect_command(
        args,
        apptainer_bin="unused",
        sif=sif,
    )
    assert docker_cmd[-3:] == ["apptainer", "inspect", "/image.sif"]
    assert f"{sif}:/image.sif:ro" in docker_cmd


def test_dedicated_apptainer_cache_is_cleaned_only_after_success_root(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    root = tmp_path / "root"
    (root / "envs" / module.APPTAINER_BOOTSTRAP_ENV).mkdir(parents=True)
    (root / "pkgs").mkdir()
    micromamba = root / ".bin" / "micromamba"
    micromamba.parent.mkdir()
    micromamba.write_text("#!/bin/sh\n", encoding="utf-8")
    commands = []
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, *, cwd=None, dry_run=False: commands.append((cmd, cwd, dry_run)),
    )

    module._clean_dedicated_bootstrap_cache(root, dry_run=False)

    assert commands == [
        (
            [
                str(micromamba),
                "--root-prefix",
                str(root),
                "clean",
                "--all",
                "--force-pkgs-dirs",
                "--yes",
            ],
            None,
            False,
        )
    ]
    (root / "envs" / "dplm").mkdir()
    commands.clear()
    module._clean_dedicated_bootstrap_cache(root, dry_run=False)
    assert commands == []


def test_runtime_cache_postcondition_covers_every_generated_runner_path(
    tmp_path: Path,
) -> None:
    module = _load_script_module()

    runtime_cache = module._ensure_runtime_cache(tmp_path)

    assert runtime_cache == tmp_path / "cache" / "runtime"
    assert {
        path.name for path in runtime_cache.iterdir() if path.is_dir()
    } == {dirname for _key, dirname in module.RUNTIME_CACHE_DIRS}
    module._validate_runner_bind_sources(
        [f"{runtime_cache}:/cache/gh-runtime", f"{tmp_path}:/workspace"]
    )

    with pytest.raises(SystemExit, match="Runner bind source missing"):
        module._validate_runner_bind_sources(
            [f"{tmp_path / 'missing'}:/models/missing"]
        )


def test_checkout_activation_uses_container_config_and_generated_runner(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    checkout = tmp_path / "checkout"
    config_env = checkout / "config" / "env"
    config_env.mkdir(parents=True)
    (checkout / "config" / "env_container.yaml").write_text(
        "# @package _global_\n", encoding="utf-8"
    )
    env_local = config_env / "env_local.yaml"
    env_local.write_text("old host config\n", encoding="utf-8")
    hint = config_env / ".astrolabe_python_path"
    hint.write_text("/old/host/python\n", encoding="utf-8")
    legacy_hint = config_env / ".gh_python_path"
    legacy_hint.write_text("/legacy/host/python\n", encoding="utf-8")
    runner = tmp_path / "container" / "runner.py"
    runner.parent.mkdir()
    runner.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    runner.chmod(0o755)
    monkeypatch.setattr(module, "REPO_ROOT", checkout)

    module._write_checkout_activation(runner, dry_run=False)

    assert env_local.is_symlink()
    assert os.readlink(env_local) == "../env_container.yaml"
    assert hint.read_text(encoding="utf-8") == f"{runner.resolve()}\n"
    from astrolabe.dev_wrapper import resolve_default_astrolabe_python
    assert resolve_default_astrolabe_python(checkout) == str(runner.resolve())
    assert legacy_hint.read_text() == "/legacy/host/python\n"
    assert not list(config_env.glob(".*.tmp"))


def test_checkout_activation_runs_only_after_validation_and_cleanup(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _load_script_module()
    events = []
    monkeypatch.setattr(module, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_detect_apptainer",
        lambda *_args, **_kwargs: "/usr/bin/apptainer",
    )
    monkeypatch.setattr(
        module,
        "_clean_dedicated_bootstrap_cache",
        lambda *_args, **_kwargs: events.append("cleanup"),
    )
    monkeypatch.setattr(
        module,
        "_write_site_profile",
        lambda *_args, **_kwargs: events.append("profile"),
    )
    monkeypatch.setattr(
        module,
        "_write_runner_from_profile",
        lambda *_args, **_kwargs: events.append("runner"),
    )
    monkeypatch.setattr(
        module,
        "_run_import_smoke",
        lambda *_args, **_kwargs: events.append("import"),
    )
    monkeypatch.setattr(
        module,
        "_run_doctor",
        lambda *_args, **_kwargs: events.append("doctor"),
    )
    monkeypatch.setattr(
        module,
        "_write_software_manifest",
        lambda *_args, **_kwargs: events.append("manifest"),
    )
    monkeypatch.setattr(
        module,
        "_write_checkout_activation",
        lambda *_args, **_kwargs: events.append("activation"),
    )

    assert module.main(
        [
            "--root",
            str(tmp_path / "root"),
            "--software",
            "dssp",
            "--skip-build",
            "--skip-submodules",
            "--activate-checkout",
            "--dry-run",
        ]
    ) == 0

    assert events == [
        "profile",
        "runner",
        "import",
        "doctor",
        "manifest",
        "cleanup",
        "activation",
    ]
    output = capsys.readouterr().out
    assert "Dry run complete" in output
    assert "Bootstrap complete" not in output


def test_sif_identity_hashes_bytes_and_rejects_stale_sidecar(tmp_path: Path) -> None:
    module = _load_script_module()
    sif = tmp_path / "image.sif"
    sif.write_bytes(b"x" * 1_000_000)

    identity = module._sif_identity(sif, build_features=())
    first_sha256 = hashlib.sha256(b"x" * 1_000_000).hexdigest()
    assert identity["sha256"] == first_sha256
    assert Path(str(sif) + ".sha256").read_text().split()[0] == first_sha256

    sif.write_bytes(b"y" * 1_000_000)
    with pytest.raises(SystemExit, match="SIF SHA256 mismatch"):
        module._sif_identity(sif, build_features=())


def test_sif_identity_requires_release_metadata_without_minting_sidecars(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    sif = tmp_path / "image.sif"
    sif.write_bytes(b"x" * 1_000_000)

    with pytest.raises(SystemExit, match="adjacent schema-v5 build metadata"):
        module._sif_identity(
            sif,
            build_features=None,
            require_build_metadata=True,
        )

    assert not Path(str(sif) + ".sha256").exists()


def test_skip_build_rejects_metadata_free_sif_before_runtime_setup(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    sif = tmp_path / "image.sif"
    sif.write_bytes(b"x" * 1_000_000)
    af3 = tmp_path / "af3"
    af3.mkdir()

    with pytest.raises(SystemExit, match="adjacent schema-v5 build metadata"):
        module.main(
            [
                "--root",
                str(tmp_path / "root"),
                "--sif",
                str(sif),
                "--software",
                "alphafold3",
                "--af3-weights",
                str(af3),
                "--skip-build",
                "--skip-submodules",
            ]
        )

    assert not Path(str(sif) + ".sha256").exists()


@pytest.mark.parametrize("suffix", [".sha256", ".build.json"])
def test_sif_identity_rejects_provenance_directory(
    tmp_path: Path, suffix: str,
) -> None:
    module = _load_script_module()
    sif = tmp_path / "image.sif"
    sif.write_bytes(b"x" * 1_000_000)
    provenance = Path(str(sif) + suffix)
    provenance.mkdir()

    with pytest.raises(SystemExit, match="not a regular file"):
        module._sif_identity(sif, build_features=())
    if suffix == ".build.json":
        assert not Path(str(sif) + ".sha256").exists()


def test_sif_identity_validates_metadata_before_writing_sidecar(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    sif = tmp_path / "image.sif"
    sif.write_bytes(b"x" * 1_000_000)
    Path(str(sif) + ".build.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="metadata does not match"):
        module._sif_identity(sif, build_features=())
    assert not Path(str(sif) + ".sha256").exists()


def test_sif_identity_rejects_tampered_licensed_build_contract(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    sif = tmp_path / "image.sif"
    sif.write_bytes(b"x" * 1_000_000)
    contract = {
        "schema_version": 5,
        "context_sha256": "1" * 64,
        "definition_sha256": "2" * 64,
        "external_inputs": [
            {"path": "/cache/wheel.whl", "size": 1, "sha256": "3" * 64}
        ],
        "build_features": ["pyrosetta"],
        "accepted_licenses": ["pyrosetta-license"],
    }
    metadata = {
        "sha256": hashlib.sha256(b"x" * 1_000_000).hexdigest(),
        "size": sif.stat().st_size,
        "build_features": ["pyrosetta"],
        "build_contract": contract,
        "build_fingerprint": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    metadata_path = Path(str(sif) + ".build.json")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert module._sif_identity(sif, build_features=("pyrosetta",))[
        "build_metadata"
    ] == str(metadata_path)
    assert module._sif_identity(sif, build_features=None)["build_features"] == [
        "pyrosetta"
    ]

    metadata["build_contract"]["definition_sha256"] = "4" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(SystemExit, match="contract fingerprint mismatch"):
        module._sif_identity(sif, build_features=("pyrosetta",))


def test_run_reports_command_failure_without_traceback() -> None:
    module = _load_script_module()

    try:
        module._run(
            [
                module.sys.executable,
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


def test_apptainer_bootstrap_uses_micromamba_when_container_steps_need_engine(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()

    commands = []

    def fake_which(name: str) -> str | None:
        if name == "micromamba":
            return "/bin/sh"
        return None

    def fake_run(cmd, *, cwd=None, dry_run=False):
        commands.append((cmd, cwd, dry_run))
        if cmd[:2] == ["/bin/sh", "create"]:
            _materialize_fake_apptainer_tiers(module, cmd)

    monkeypatch.setattr(module.shutil, "which", fake_which)
    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module,
        "_sif_identity",
        lambda sif, *, build_features, require_build_metadata=False: {
            "path": str(sif), "size": 1, "sha256": "0" * 64,
            "build_features": list(build_features),
        },
    )
    monkeypatch.setattr(module, "_missing_uidmap_helpers", lambda: ())

    af3 = tmp_path / "af3"
    af3.mkdir()
    profile = tmp_path / "sites" / "ty2.json"
    runner = tmp_path / "tools" / "gh_ty2_container_python.py"

    rc = module.main(
        [
            "--profile",
            "ty2",
            "--root",
            str(tmp_path / "root"),
            "--software",
            "autodock-vina",
            "--site-profile",
            str(profile),
            "--runner",
            str(runner),
            "--af3-weights",
            str(af3),
            "--skip-submodules",
            "--skip-import-smoke",
            "--skip-doctor",
        ]
    )

    apptainer = tmp_path / "root" / "envs" / module.APPTAINER_BOOTSTRAP_ENV / "bin" / "apptainer"
    payload = json.loads(profile.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["apptainer_bin"] == str(apptainer)
    assert runner.exists()
    assert any(cmd[0][:2] == ["/bin/sh", "create"] for cmd in commands)
    create_cmd = next(cmd[0] for cmd in commands if cmd[0][:2] == ["/bin/sh", "create"])
    assert "--override-channels" in create_cmd
    assert all(package in create_cmd for package in module.APPTAINER_RUNTIME_PACKAGES)
    assert not any(package in create_cmd for package in module.APPTAINER_BUILD_PACKAGES)
    build_cmd = next(
        cmd[0] for cmd in commands if any(str(part).endswith("apptainer_build.py") for part in cmd[0])
    )
    assert build_cmd[build_cmd.index("--apptainer-bin") + 1] == str(apptainer)
    assert build_cmd[build_cmd.index("--build-cache") + 1] == str(
        tmp_path / "root" / "cache" / "container-build"
    )


def test_bootstrapped_apptainer_requires_site_uidmap_helpers(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()

    commands = []

    def fake_which(name: str) -> str | None:
        if name == "micromamba":
            return "/bin/sh"
        return None

    def fake_run(cmd, *, cwd=None, dry_run=False):
        commands.append((cmd, cwd, dry_run))
        if cmd[:2] == ["/bin/sh", "create"]:
            _materialize_fake_apptainer_tiers(module, cmd)

    monkeypatch.setattr(module.shutil, "which", fake_which)
    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_missing_uidmap_helpers", lambda: ("newuidmap", "newgidmap"))

    af3 = tmp_path / "af3"
    af3.mkdir()

    try:
        module.main(
            [
                "--profile",
                "ty2",
                "--root",
                str(tmp_path / "root"),
                "--software",
                "autodock-vina",
                "--af3-weights",
                str(af3),
                "--skip-submodules",
                "--skip-import-smoke",
                "--skip-doctor",
                "--fakeroot",
            ]
        )
    except SystemExit as exc:
        text = str(exc)
        assert "newuidmap" in text
        assert "newgidmap" in text
        assert "--apptainer-bin" in text
        assert "--remote" in text
        assert "--skip-build" in text
        assert "--sif <path>" in text
    else:
        raise AssertionError("bootstrapped Apptainer should fail without uidmap helpers")

    assert any(cmd[0][:2] == ["/bin/sh", "create"] for cmd in commands)
    create_cmd = next(cmd[0] for cmd in commands if cmd[0][:2] == ["/bin/sh", "create"])
    assert all(package in create_cmd for package in module.APPTAINER_BUILD_PACKAGES)
    assert not any(
        any(str(part).endswith("apptainer_build.py") for part in cmd[0])
        for cmd in commands
    )


def test_fakeroot_is_built_from_vendored_source(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()

    repo_root = tmp_path / "repo"
    source = repo_root / "vendor" / "fakeroot"
    source.mkdir(parents=True)
    (source / "meson.build").write_text("project('fakeroot', 'c')\n", encoding="utf-8")
    root = tmp_path / "root"
    engine_env = root / "envs" / module.APPTAINER_BOOTSTRAP_ENV
    meson = engine_env / "bin" / "meson"
    meson.parent.mkdir(parents=True)
    meson.write_text("#!/bin/sh\n", encoding="utf-8")
    meson.chmod(0o755)
    ninja = engine_env / "bin" / "ninja"
    ninja.write_text("#!/bin/sh\n", encoding="utf-8")
    ninja.chmod(0o755)
    capability = engine_env / "include" / "sys" / "capability.h"
    capability.parent.mkdir(parents=True)
    capability.write_text("/* stub */\n", encoding="utf-8")
    for name in ("setcap", "getcap"):
        executable = engine_env / "sbin" / name
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)
    _write_fake_conda_records(module, root, module.APPTAINER_BUILD_PACKAGES)
    module._write_apptainer_tier_marker(
        engine_env,
        micromamba="/bin/sh",
    )
    commands = []

    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    def fake_run(cmd, *, cwd=None, dry_run=False):
        commands.append((list(cmd), cwd, dry_run))
        if "install" in cmd and "-C" in cmd:
            fakeroot = root / "tools" / "fakeroot" / "bin" / "fakeroot"
            fakeroot.parent.mkdir(parents=True)
            fakeroot.write_text("#!/bin/sh\n", encoding="utf-8")
            fakeroot.chmod(0o755)

    monkeypatch.setattr(module, "_run", fake_run)

    fakeroot = module._ensure_fakeroot(root, dry_run=False)

    assert fakeroot == str(root / "tools" / "fakeroot" / "bin" / "fakeroot")
    assert commands[0][0][0:4] == [
        "env",
        f"CPATH={engine_env / 'include'}",
        f"LIBRARY_PATH={engine_env / 'lib'}",
        f"PKG_CONFIG_PATH={engine_env / 'lib' / 'pkgconfig'}",
    ]
    assert "setup" in commands[0][0]
    assert "compile" in commands[1][0]
    assert "install" in commands[2][0]
    assert commands[3][0] == [fakeroot, "--version"]
    assert commands[4][0][:4] == [fakeroot, "--", "sh", "-ec"]
    assert "chown 1234:2345" in commands[4][0][4]
    assert "cap_chown=ep" in commands[4][0][4]


def test_fakeroot_build_bootstraps_only_missing_build_tier(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    repo_root = tmp_path / "repo"
    source = repo_root / "vendor" / "fakeroot"
    source.mkdir(parents=True)
    (source / "meson.build").write_text("project('fakeroot', 'c')\n", encoding="utf-8")
    root = tmp_path / "root"
    commands = []

    def fake_which(name: str) -> str | None:
        return "/bin/sh" if name == "micromamba" else None

    def fake_run(cmd, *, cwd=None, dry_run=False):
        commands.append(list(cmd))
        if cmd[:2] == ["/bin/sh", "create"]:
            _materialize_fake_apptainer_tiers(module, cmd)
        if "install" in cmd and "-C" in cmd:
            fakeroot = root / "tools" / "fakeroot" / "bin" / "fakeroot"
            fakeroot.parent.mkdir(parents=True)
            fakeroot.write_text("#!/bin/sh\n", encoding="utf-8")
            fakeroot.chmod(0o755)

    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    monkeypatch.setattr(module.shutil, "which", fake_which)
    monkeypatch.setattr(module, "_run", fake_run)

    fakeroot = module._ensure_fakeroot(root, dry_run=False)

    create_cmd = commands[0]
    assert create_cmd[:2] == ["/bin/sh", "create"]
    assert all(package in create_cmd for package in module.APPTAINER_BUILD_PACKAGES)
    assert not any(package in create_cmd for package in module.APPTAINER_RUNTIME_PACKAGES)
    assert fakeroot == str(root / "tools" / "fakeroot" / "bin" / "fakeroot")


def test_system_fakeroot_is_executed_before_reuse(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    commands = []
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda name: {
            "fakeroot": "/usr/bin/fakeroot",
            "setcap": "/usr/sbin/setcap",
            "getcap": "/usr/sbin/getcap",
        }.get(name),
    )
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, *, cwd=None, dry_run=False: commands.append(list(cmd)),
    )

    assert module._ensure_fakeroot(tmp_path, dry_run=False) == "/usr/bin/fakeroot"
    assert commands[0] == ["/usr/bin/fakeroot", "--version"]
    assert commands[1][:4] == ["/usr/bin/fakeroot", "--", "sh", "-ec"]
    assert commands[1][-2:] == ["/usr/sbin/setcap", "/usr/sbin/getcap"]


def test_apptainer_runtime_environment_upgrades_build_tools_on_demand(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    root = tmp_path / "root"
    env_bin = root / "envs" / module.APPTAINER_BOOTSTRAP_ENV / "bin"
    env_bin.mkdir(parents=True)
    apptainer = env_bin / "apptainer"
    apptainer.write_text("#!/bin/sh\n", encoding="utf-8")
    (env_bin / "squashfuse").write_text("#!/bin/sh\n", encoding="utf-8")
    _write_fake_conda_records(module, root, module.APPTAINER_RUNTIME_PACKAGES)
    module._write_apptainer_tier_marker(
        env_bin.parent,
        micromamba="/bin/sh",
    )
    commands = []

    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda name: "/bin/sh" if name == "micromamba" else None,
    )
    def fake_run(cmd, *, cwd=None, dry_run=False):
        commands.append(list(cmd))
        if cmd[:2] in (
            ["/bin/sh", "create"],
            ["/bin/sh", "install"],
        ):
            _materialize_fake_apptainer_tiers(module, cmd)

    monkeypatch.setattr(module, "_run", fake_run)

    assert (
        module._ensure_apptainer_via_micromamba(
            root,
            dry_run=False,
            build_tools=False,
        )
        == str(apptainer)
    )
    assert commands == [[str(apptainer), "--version"]]
    marker_path = root / "envs" / module.APPTAINER_BOOTSTRAP_ENV / ".gh-bootstrap-tiers.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["runtime"]["ready"] is True
    assert marker["runtime"]["package_specs"] == list(module.APPTAINER_RUNTIME_PACKAGES)
    assert marker["build_tools"]["ready"] is False

    commands.clear()
    module._ensure_apptainer_via_micromamba(
        root,
        dry_run=False,
        build_tools=True,
    )
    install_cmd = commands[0]
    assert install_cmd[:2] == ["/bin/sh", "install"]
    assert all(package in install_cmd for package in module.APPTAINER_RUNTIME_PACKAGES)
    assert all(package in install_cmd for package in module.APPTAINER_BUILD_PACKAGES)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["runtime"]["ready"] is True
    assert marker["build_tools"]["ready"] is True


def test_apptainer_always_uses_managed_pinned_engine(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    calls = []
    commands = []
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda name: "/usr/bin/apptainer" if name == "apptainer" else None,
    )
    monkeypatch.setattr(
        module,
        "_ensure_apptainer_via_micromamba",
        lambda root, *, dry_run, build_tools: calls.append(
            (root, dry_run, build_tools)
        )
        or str(root / "envs" / module.APPTAINER_BOOTSTRAP_ENV / "bin" / "apptainer"),
    )
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, *, cwd=None, dry_run=False: commands.append(list(cmd)),
    )

    managed = module._detect_apptainer(
        None,
        root=tmp_path,
        bootstrap="always",
    )

    assert managed == str(
        tmp_path / "envs" / module.APPTAINER_BOOTSTRAP_ENV / "bin" / "apptainer"
    )
    assert calls == [(tmp_path, False, False)]
    assert module._detect_apptainer(None, root=tmp_path, bootstrap="auto") == "/usr/bin/apptainer"
    assert commands == [["/usr/bin/apptainer", "--version"]]


def test_apptainer_environment_lock_timeout_is_visible(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    lock_path = module._apptainer_env_lock_path(tmp_path)
    lock_path.parent.mkdir(parents=True)
    lock_path.touch()
    monkeypatch.setattr(module, "APPTAINER_ENV_LOCK_TIMEOUT_SECONDS", 0.05)

    with lock_path.open("a+b") as lock_handle:
        module.fcntl.flock(lock_handle.fileno(), module.fcntl.LOCK_EX)
        with pytest.raises(SystemExit, match="Timed out waiting"):
            module._ensure_apptainer_tiers(
                tmp_path,
                dry_run=False,
                runtime=True,
                build_tools=False,
            )


def test_apptainer_paths_without_conda_records_are_not_a_ready_tier(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    env_root = tmp_path / "envs" / module.APPTAINER_BOOTSTRAP_ENV
    for name in ("apptainer", "squashfuse"):
        path = env_root / "bin" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("partial\n", encoding="utf-8")

    assert not module._apptainer_tier_ready(
        env_root,
        package_specs=module.APPTAINER_RUNTIME_PACKAGES,
        required_paths=module._apptainer_tier_paths(
            env_root,
            runtime=True,
            build_tools=False,
        ),
    )


def test_apptainer_tier_marker_binds_build_channel_and_package_hash(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    root = tmp_path / "root"
    env_root = root / "envs" / module.APPTAINER_BOOTSTRAP_ENV
    for name in ("apptainer", "squashfuse"):
        executable = env_root / "bin" / name
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
    _write_fake_conda_records(module, root, module.APPTAINER_RUNTIME_PACKAGES)
    module._write_apptainer_tier_marker(env_root, micromamba="/bin/sh")
    required_paths = module._apptainer_tier_paths(
        env_root,
        runtime=True,
        build_tools=False,
    )

    assert module._apptainer_tier_ready(
        env_root,
        package_specs=module.APPTAINER_RUNTIME_PACKAGES,
        required_paths=required_paths,
    )
    marker_path = env_root / ".gh-bootstrap-tiers.json"
    marker = json.loads(marker_path.read_text())
    direct = marker["runtime"]["resolved_direct_records"]
    assert all(record["build"] == "fake_0" for record in direct)
    assert all(record["channel"] == "conda-forge" for record in direct)
    assert all(len(record["sha256"]) == 64 for record in direct)

    marker["micromamba"]["sha256"] = "f" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    assert not module._apptainer_tier_ready(
        env_root,
        package_specs=module.APPTAINER_RUNTIME_PACKAGES,
        required_paths=required_paths,
    )
    module._write_apptainer_tier_marker(env_root, micromamba="/bin/sh")

    record_path = next((env_root / "conda-meta").glob("apptainer-*.json"))
    record = json.loads(record_path.read_text())
    record["sha256"] = None
    record["md5"] = None
    record_path.write_text(json.dumps(record), encoding="utf-8")
    assert not module._apptainer_tier_ready(
        env_root,
        package_specs=module.APPTAINER_RUNTIME_PACKAGES,
        required_paths=required_paths,
    )


def test_apptainer_record_rejects_spoofed_channel_url_and_bool_build_number() -> None:
    module = _load_script_module()
    record = {
        "name": "apptainer",
        "version": "1.5.2",
        "build": "fake_0",
        "build_number": 0,
        "channel": "conda-forge",
        "subdir": "linux-64",
        "sha256": "0" * 64,
        "md5": None,
        "url": "https://evil.invalid/conda-forge/linux-64/apptainer.conda",
    }
    assert not module._conda_records_trusted([record])
    valid_url = (
        "https://conda.anaconda.org/conda-forge/linux-64/"
        "apptainer-1.5.2-fake_0.conda"
    )
    for spoofed_url in (
        "https://conda.anaconda.org/evil/cloud/conda-forge/linux-64/"
        "apptainer-1.5.2-fake_0.conda",
        valid_url + "?download=1",
        valid_url + "#fragment",
        valid_url.replace("conda.anaconda.org", "user@conda.anaconda.org"),
        valid_url.replace("conda.anaconda.org", "conda.anaconda.org:444"),
    ):
        record["url"] = spoofed_url
        assert not module._conda_records_trusted([record])
    record["url"] = valid_url
    record["build_number"] = True
    assert not module._conda_records_trusted([record])
    record["build_number"] = 0
    record["channel"] = "https://evil.invalid/conda-forge"
    assert not module._conda_records_trusted([record])
    record["channel"] = "conda-forge"
    record["url"] = (
        "https://conda.anaconda.org/conda-forge/win-64/"
        "not-apptainer-1.5.2-attacker.conda"
    )
    assert not module._conda_records_trusted([record])


@pytest.mark.parametrize("record_bytes", [b"[]\n", b"\xff"])
def test_apptainer_malformed_record_is_untrusted_and_quarantined(
    tmp_path: Path, record_bytes: bytes,
) -> None:
    module = _load_script_module()
    env_root = tmp_path / "envs" / module.APPTAINER_BOOTSTRAP_ENV
    conda_meta = env_root / "conda-meta"
    conda_meta.mkdir(parents=True)
    (conda_meta / "history").write_text("created\n", encoding="utf-8")
    (conda_meta / "broken.json").write_bytes(record_bytes)

    assert module._resolved_conda_records(env_root) == [{}]
    assert module._prepare_apptainer_env_action(env_root, dry_run=False) == "create"
    assert not env_root.exists()
    assert len(list((tmp_path / "envs").glob("_gh_apptainer.incomplete.*"))) == 1


def test_apptainer_missing_or_tampered_marker_forces_verified_install(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    root = tmp_path / "root"
    env_root = root / "envs" / module.APPTAINER_BOOTSTRAP_ENV
    for name in ("apptainer", "squashfuse"):
        executable = env_root / "bin" / name
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
    _write_fake_conda_records(module, root, module.APPTAINER_RUNTIME_PACKAGES)
    commands = []
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda name: "/bin/sh" if name == "micromamba" else None,
    )

    def fake_run(cmd, *, cwd=None, dry_run=False):
        commands.append(list(cmd))
        if cmd[:2] == ["/bin/sh", "install"]:
            _materialize_fake_apptainer_tiers(module, cmd)

    monkeypatch.setattr(module, "_run", fake_run)
    module._ensure_apptainer_via_micromamba(
        root,
        dry_run=False,
        build_tools=False,
    )
    assert commands[0][:2] == ["/bin/sh", "install"]

    commands.clear()
    marker_path = env_root / ".gh-bootstrap-tiers.json"
    marker = json.loads(marker_path.read_text())
    marker["resolved_records_sha256"] = "0" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    module._ensure_apptainer_via_micromamba(
        root,
        dry_run=False,
        build_tools=False,
    )
    assert commands[0][:2] == ["/bin/sh", "install"]

    for non_object_marker in ("[]\n", "null\n"):
        commands.clear()
        marker_path.write_text(non_object_marker, encoding="utf-8")
        module._ensure_apptainer_via_micromamba(
            root,
            dry_run=False,
            build_tools=False,
        )
        assert commands[0][:2] == ["/bin/sh", "install"]


def test_apptainer_failed_create_releases_lock_and_retry_reuses_root(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()
    commands = []
    first_attempt = True

    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda name: "/bin/sh" if name == "micromamba" else None,
    )

    def fake_run(cmd, *, cwd=None, dry_run=False):
        nonlocal first_attempt
        commands.append(list(cmd))
        if cmd[:2] == ["/bin/sh", "create"] and first_attempt:
            first_attempt = False
            env_bin = (
                tmp_path
                / "envs"
                / module.APPTAINER_BOOTSTRAP_ENV
                / "bin"
            )
            env_bin.mkdir(parents=True)
            (env_bin / "apptainer").write_text("partial\n", encoding="utf-8")
            cache_file = tmp_path / "pkgs" / "retained-package.conda"
            cache_file.parent.mkdir()
            cache_file.write_text("retained\n", encoding="utf-8")
            raise SystemExit("simulated micromamba interruption")
        if cmd[:2] in (
            ["/bin/sh", "create"],
            ["/bin/sh", "install"],
        ):
            _materialize_fake_apptainer_tiers(module, cmd)

    monkeypatch.setattr(module, "_run", fake_run)

    with pytest.raises(SystemExit, match="simulated micromamba interruption"):
        module._ensure_apptainer_tiers(
            tmp_path,
            dry_run=False,
            runtime=True,
            build_tools=False,
        )

    env_root = module._ensure_apptainer_tiers(
        tmp_path,
        dry_run=False,
        runtime=True,
        build_tools=False,
    )
    assert commands[0][:2] == ["/bin/sh", "create"]
    assert commands[1][:2] == ["/bin/sh", "create"]
    assert (tmp_path / "pkgs" / "retained-package.conda").read_text() == "retained\n"
    assert not list((tmp_path / "envs").glob("_gh_apptainer.incomplete.*"))
    marker = json.loads((env_root / ".gh-bootstrap-tiers.json").read_text())
    assert marker["runtime"]["ready"] is True
    managed_micromamba = tmp_path / ".bin" / "micromamba"
    managed_micromamba.parent.mkdir(parents=True)
    managed_micromamba.write_text("#!/bin/sh\n", encoding="utf-8")
    module._clean_dedicated_bootstrap_cache(tmp_path, dry_run=False)
    assert any("clean" in command for command in commands)


def test_bootstrapped_apptainer_remote_build_does_not_require_local_uidmap(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()

    commands = []

    def fake_which(name: str) -> str | None:
        if name == "micromamba":
            return "/bin/sh"
        return None

    def fake_run(cmd, *, cwd=None, dry_run=False):
        commands.append((cmd, cwd, dry_run))
        if cmd[:2] == ["/bin/sh", "create"]:
            _materialize_fake_apptainer_tiers(module, cmd)

    monkeypatch.setattr(module.shutil, "which", fake_which)
    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module,
        "_sif_identity",
        lambda sif, *, build_features, require_build_metadata=False: {
            "path": str(sif), "size": 1, "sha256": "0" * 64,
            "build_features": list(build_features),
        },
    )
    monkeypatch.setattr(module, "_missing_uidmap_helpers", lambda: ("newuidmap", "newgidmap"))

    af3 = tmp_path / "af3"
    af3.mkdir()

    rc = module.main(
        [
            "--profile",
            "ty2",
            "--root",
            str(tmp_path / "root"),
            "--software",
            "autodock-vina",
            "--af3-weights",
            str(af3),
            "--skip-submodules",
            "--skip-import-smoke",
            "--skip-doctor",
            "--remote",
        ]
    )

    assert rc == 0
    build_cmd = next(
        cmd[0] for cmd in commands if any(str(part).endswith("apptainer_build.py") for part in cmd[0])
    )
    assert "--remote" in build_cmd
    assert build_cmd[build_cmd.index("--build-cache") + 1] == str(
        tmp_path / "root" / "cache" / "container-build"
    )


def test_bootstrapped_apptainer_skip_build_does_not_require_local_uidmap(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()

    commands = []

    def fake_which(name: str) -> str | None:
        if name == "micromamba":
            return "/bin/sh"
        return None

    def fake_run(cmd, *, cwd=None, dry_run=False):
        commands.append((cmd, cwd, dry_run))
        if cmd[:2] == ["/bin/sh", "create"]:
            _materialize_fake_apptainer_tiers(module, cmd)

    monkeypatch.setattr(module.shutil, "which", fake_which)
    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_missing_uidmap_helpers", lambda: ("newuidmap", "newgidmap"))
    monkeypatch.setattr(module, "_write_runner_from_profile", lambda _profile, dry_run=False: None)

    af3 = tmp_path / "af3"
    af3.mkdir()
    sif = tmp_path / "existing.sif"
    with sif.open("wb") as handle:
        handle.truncate(1_000_000)
    _write_valid_sif_metadata(module, sif)

    rc = module.main(
        [
            "--profile",
            "ty2",
            "--root",
            str(tmp_path / "root"),
            "--software",
            "autodock-vina",
            "--af3-weights",
            str(af3),
            "--sif",
            str(sif),
            "--skip-build",
            "--skip-submodules",
            "--skip-doctor",
        ]
    )

    assert rc == 0
    assert not any(
        any(str(part).endswith("apptainer_build.py") for part in cmd[0])
        for cmd in commands
    )
    assert any(str(cmd[0][0]).endswith("gh_ty2_container_python.py") for cmd in commands)
    create_cmd = next(cmd[0] for cmd in commands if cmd[0][:2] == ["/bin/sh", "create"])
    assert all(package in create_cmd for package in module.APPTAINER_RUNTIME_PACKAGES)
    assert not any(package in create_cmd for package in module.APPTAINER_BUILD_PACKAGES)


def test_docker_container_builder_does_not_require_host_apptainer(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()

    commands = []

    def fake_run(cmd, *, cwd=None, dry_run=False):
        commands.append((cmd, cwd, dry_run))

    monkeypatch.setattr(module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(module, "_detect_docker", lambda explicit: "/usr/bin/docker")
    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module,
        "_sif_identity",
        lambda sif, *, build_features, require_build_metadata=False: {
            "path": str(sif), "size": 1, "sha256": "0" * 64,
            "build_features": list(build_features),
        },
    )

    profile = tmp_path / "sites" / "ty2.json"
    runner = tmp_path / "tools" / "gh_ty2_container_python.py"
    rc = module.main(
        [
            "--profile",
            "ty2",
            "--root",
            str(tmp_path / "root"),
            "--software",
            "autodock-vina",
            "--site-profile",
            str(profile),
            "--runner",
            str(runner),
            "--container-builder",
            "docker",
            "--runner-backend",
            "docker-apptainer",
            "--skip-submodules",
            "--skip-import-smoke",
            "--skip-doctor",
            "--allow-missing-af3",
        ]
    )

    payload = json.loads(profile.read_text(encoding="utf-8"))
    build_cmd = next(
        cmd[0] for cmd in commands if any(str(part).endswith("apptainer_build.py") for part in cmd[0])
    )
    assert rc == 0
    assert payload["runner_backend"] == "docker-apptainer"
    assert payload["docker_bin"] == "/usr/bin/docker"
    assert payload["docker_image"] == module.DEFAULT_DOCKER_APPTAINER_IMAGE
    assert "--builder" in build_cmd
    assert "docker" in build_cmd
    assert "--docker-bin" in build_cmd
    assert "/usr/bin/docker" in build_cmd
    assert build_cmd[build_cmd.index("--docker-base-source") + 1] == "docker"
    assert build_cmd[build_cmd.index("--docker-network") + 1] == "host"
    assert build_cmd[build_cmd.index("--build-cache") + 1] == str(
        tmp_path / "root" / "cache" / "container-build"
    )
    assert runner.exists()
    assert "DOCKER_BIN = '/usr/bin/docker'" in runner.read_text(encoding="utf-8")


def test_container_build_cache_can_be_overridden(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()

    commands = []

    def fake_run(cmd, *, cwd=None, dry_run=False):
        commands.append((cmd, cwd, dry_run))

    monkeypatch.setattr(module, "_detect_docker", lambda explicit: "/usr/bin/docker")
    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module,
        "_sif_identity",
        lambda sif, *, build_features, require_build_metadata=False: {
            "path": str(sif), "size": 1, "sha256": "0" * 64,
            "build_features": list(build_features),
        },
    )

    build_cache = tmp_path / "custom-cache"
    rc = module.main(
        [
            "--profile",
            "ty2",
            "--root",
            str(tmp_path / "root"),
            "--software",
            "autodock-vina",
            "--container-build-cache",
            str(build_cache),
            "--container-builder",
            "docker",
            "--runner-backend",
            "docker-apptainer",
            "--skip-submodules",
            "--skip-import-smoke",
            "--skip-doctor",
            "--allow-missing-af3",
        ]
    )

    build_cmd = next(
        cmd[0] for cmd in commands if any(str(part).endswith("apptainer_build.py") for part in cmd[0])
    )
    assert rc == 0
    assert build_cmd[build_cmd.index("--build-cache") + 1] == str(build_cache)


def test_apptainer_bootstrap_can_be_disabled(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()

    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    try:
        module._detect_apptainer(
            None,
            dry_run=False,
            root=tmp_path / "root",
            bootstrap="never",
        )
    except SystemExit as exc:
        text = str(exc)
        assert "--apptainer-bin" in text
        assert "--bootstrap-apptainer auto" in text
    else:
        raise AssertionError("missing Apptainer should fail when bootstrap is disabled")
