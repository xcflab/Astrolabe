#!/usr/bin/env python3
"""
Build Apptainer/Singularity images with resume.

This is intentionally host-side and low-dependency (works on system Python).
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import pathlib
import stat
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile


DEFAULT_DOCKER_APPTAINER_IMAGE = "ghcr.io/apptainer/apptainer:1.4.5"
DEFAULT_DOCKER_NETWORK = "host"
_METADATA_SETTLE_NS = 2_000_000_000
PYROSETTA_VERSION = "2026.3+releasequarterly.5e498f1409"
PYROSETTA_WHEEL_NAME = (
    "pyrosetta-%s-cp311-cp311-linux_x86_64.whl" % PYROSETTA_VERSION
)
PYROSETTA_ALLOWED_SHA256 = frozenset(
    {
        "fd1a9a51c1d7475b448e010926ed201446a55d20f4c5340455af148914d3cffe",
    }
)
PYROSETTA_WHEEL_SIZE = 1_658_972_065
DEFAULT_BUILD_FEATURES = ("pyrosetta",)
ANANAS_ARCHIVE_NAME = "ananas-linux-1.1.zip"
ANANAS_ARCHIVE_SHA256 = (
    "279a4163e42343599633db71ac2ecd5f20795fe80510ff5f2ec542de35701745"
)
ANANAS_BINARY_SHA256 = (
    "79f470f084f0e98c4e067104e32d879d41d68fc4935ced5c49102bb8666177ab"
)
ROSETTA_TOOLS_COMMIT = "80a0635615099e1b918474a63acba7b1de6fd107"
ROSETTA_TOOLS_ARCHIVE_NAME = "rosetta-tools-%s.tar.gz" % ROSETTA_TOOLS_COMMIT
ROSETTA_TOOLS_MANIFEST_REL = pathlib.Path("container/rosetta-tools.manifest.tsv")
BUILD_FEATURE_LICENSES = {
    "pyrosetta": "pyrosetta-license",
    "ananas": "ananas-academic",
    "rosetta": "rosetta-license",
}

_BUILD_CONTEXT_EXCLUDE_ROOTS = frozenset(
    {
        ".agents",
        ".beads",
        ".claude",
        ".codex",
        ".codex-tasks",
        ".pi",
        ".trellis",
        ".worktrees",
        "docs",
        "logs",
        "outputs",
        "tests",
    }
)
_BUILD_CONTEXT_EXCLUDE_PREFIXES = (
    "vendor/dplm",
    "vendor/libcifpp",
    "vendor/na_mpnn",
    "vendor/pybind11",
    "vendor/protein_sequence_models",
    "vendor/rhodesign",
    "vendor/rpxdock",
    "vendor/sru",
    "vendor/surfdesign",
    "vendor/surfpro",
    "vendor/willutil",
)
_SUBMODULE_TRANSIENT_DIRS = frozenset({"__pycache__", ".pytest_cache"})
_SUBMODULE_NON_RUNTIME_ROOTS = frozenset(
    {".github", "docs", "examples", "imgs", "notebooks", "tests"}
)


def _is_submodule_transient(rel_path):
    rel_path = pathlib.Path(rel_path)
    if rel_path.parts and rel_path.parts[0] in _SUBMODULE_NON_RUNTIME_ROOTS:
        return True
    if any(part in _SUBMODULE_TRANSIENT_DIRS for part in rel_path.parts):
        return True
    if (
        len(rel_path.parts) == 1
        and rel_path.name.startswith("attn_core_inplace_cuda")
        and rel_path.suffix == ".so"
    ):
        return True
    if rel_path.suffix in {".pyc", ".pyo"}:
        return True
    if any(part.endswith(".egg-info") for part in rel_path.parts):
        return True
    return bool(rel_path.parts and rel_path.parts[0] in {"build", "dist"})


def _is_build_context_excluded(rel_path):
    rel_path = pathlib.Path(rel_path)
    if rel_path.parts and rel_path.parts[0] in _BUILD_CONTEXT_EXCLUDE_ROOTS:
        return True
    rel_text = rel_path.as_posix()
    return any(
        rel_text == prefix or rel_text.startswith(prefix + "/")
        for prefix in _BUILD_CONTEXT_EXCLUDE_PREFIXES
    )


def _is_root_runtime_python_source(rel_path):
    rel_path = pathlib.Path(rel_path)
    if rel_path.suffix != ".py":
        return False
    return rel_path == pathlib.Path("run.py") or (
        len(rel_path.parts) > 1 and rel_path.parts[0] == "astrolabe"
    )


def _expand(path_str):
    return pathlib.Path(os.path.expandvars(os.path.expanduser(path_str))).resolve()


def _json_load(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_dump_atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(str(tmp), str(path))


def _detect_apptainer():
    for name in ("apptainer", "singularity"):
        exe = shutil.which(name)
        if exe:
            return exe
    return None


def _detect_docker(explicit=None):
    if explicit:
        return str(_expand(explicit))
    exe = shutil.which("docker")
    if exe:
        return exe
    return None


def _docker_builder_identity(docker_bin, docker_image):
    def inspect_image():
        try:
            raw = subprocess.check_output(
                [docker_bin, "image", "inspect", docker_image], text=True
            )
        except subprocess.CalledProcessError:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                "Could not resolve immutable Docker builder image identity: %s" % exc
            )

    inspect_payload = inspect_image()
    if inspect_payload is None:
        _run([docker_bin, "pull", docker_image])
        inspect_payload = inspect_image()
    if inspect_payload is None:
        raise SystemExit(
            "Could not inspect Docker builder image after pull: %s" % docker_image
        )
    if not isinstance(inspect_payload, list) or len(inspect_payload) != 1:
        raise SystemExit("Docker builder image inspect returned an unexpected payload")
    image_record = inspect_payload[0]
    repo_digests = image_record.get("RepoDigests") if isinstance(image_record, dict) else []
    repo_digests = repo_digests or []
    image_id = image_record.get("Id") if isinstance(image_record, dict) else None
    if (
        not isinstance(repo_digests, list)
        or not all(isinstance(value, str) and "@sha256:" in value for value in repo_digests)
        or not isinstance(image_id, str)
        or not image_id.startswith("sha256:")
    ):
        raise SystemExit(
            "Docker builder image has no immutable image ID: %s" % docker_image
        )
    try:
        docker_version = subprocess.check_output(
            [docker_bin, "--version"], text=True
        ).strip()
        apptainer_version = subprocess.check_output(
            [docker_bin, "run", "--rm", image_id, "apptainer", "version"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit("Could not resolve Docker builder toolchain version: %s" % exc)
    return {
        "image": docker_image,
        "image_id": image_id,
        "execution_image": image_id,
        "repo_digests": sorted(repo_digests),
        "docker_binary": str(pathlib.Path(docker_bin).resolve()),
        "docker_binary_sha256": _sha256_file(pathlib.Path(docker_bin).resolve()),
        "docker_version": docker_version,
        "apptainer_version": apptainer_version,
    }


def _hardlink_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _tracked_build_context_files(repo_root):
    repo_root = pathlib.Path(repo_root).resolve()

    def _git_ls_files(worktree, *, include_untracked):
        cmd = ["git", "ls-files", "--cached"]
        if include_untracked:
            cmd.extend(["--others", "--exclude-standard"])
        cmd.append("-z")
        proc = subprocess.run(
            cmd,
            cwd=str(worktree),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return [pathlib.Path(os.fsdecode(raw)) for raw in proc.stdout.split(b"\0") if raw]

    def _git_ls_untracked(worktree):
        proc = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=str(worktree),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return [pathlib.Path(os.fsdecode(raw)) for raw in proc.stdout.split(b"\0") if raw]

    try:
        root_files = _git_ls_files(repo_root, include_untracked=False)
        root_untracked = _git_ls_untracked(repo_root)
        submodule_proc = subprocess.run(
            ["git", "submodule", "status", "--recursive"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        details = getattr(exc, "stderr", b"") or b""
        details = details.decode("utf-8", errors="replace").strip()
        suffix = ": %s" % details if details else ""
        raise SystemExit(
            "Container builds require an initialized Git checkout so the build "
            "context can exclude weights, caches, outputs, and other untracked data%s"
            % suffix
        )

    untracked_runtime_sources = sorted(
        (
            rel_path
            for rel_path in root_untracked
            if not _is_build_context_excluded(rel_path)
            and _is_root_runtime_python_source(rel_path)
        ),
        key=lambda path: os.fsencode(str(path)),
    )
    if untracked_runtime_sources:
        offenders = "\n".join("  - %s" % path for path in untracked_runtime_sources)
        raise SystemExit(
            "Untracked runtime Python source files would be omitted from the "
            "tracked-only container build context.\n"
            "Repository: %s\n"
            "Runtime boundary: astrolabe/**/*.py and run.py\n"
            "%s\n"
            "Track these files or move/remove them before building the release image."
            % (repo_root, offenders)
        )

    rel_paths = set()
    for rel_path in root_files:
        if _is_build_context_excluded(rel_path):
            continue
        source = repo_root / rel_path
        if source.is_file() or source.is_symlink():
            rel_paths.add(rel_path)

    for raw_line in submodule_proc.stdout.decode("utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or raw_line[0] == "-":
            continue
        fields = line.lstrip(" +-U").split()
        if len(fields) < 2:
            raise SystemExit("Could not parse git submodule status line: %s" % raw_line)
        submodule_rel = pathlib.Path(fields[1])
        if _is_build_context_excluded(submodule_rel):
            continue
        submodule_root = repo_root / submodule_rel
        try:
            submodule_files = _git_ls_files(submodule_root, include_untracked=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            details = getattr(exc, "stderr", b"") or b""
            details = details.decode("utf-8", errors="replace").strip()
            suffix = ": %s" % details if details else ""
            raise SystemExit(
                "Could not enumerate initialized submodule build context %s%s"
                % (submodule_rel, suffix)
            )
        for sub_path in submodule_files:
            if _is_submodule_transient(sub_path):
                continue
            rel_path = submodule_rel / sub_path
            if _is_build_context_excluded(rel_path):
                continue
            source = repo_root / rel_path
            if source.is_file() or source.is_symlink():
                rel_paths.add(rel_path)
    if not rel_paths:
        raise SystemExit("Git checkout has no tracked files; refusing to build an empty image.")
    return tuple(sorted(rel_paths, key=lambda path: os.fsencode(str(path))))


def _hash_file(path, digest):
    with pathlib.Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return
            digest.update(chunk)


def _sha256_file(path):
    digest = hashlib.sha256()
    _hash_file(path, digest)
    return digest.hexdigest()


def _stage_pyrosetta_wheel(source, build_cache, expected_sha256):
    source = pathlib.Path(source).resolve()
    destination = (
        pathlib.Path(build_cache).resolve()
        / "wheels"
        / "pyrosetta"
        / PYROSETTA_WHEEL_NAME
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source == destination:
        return destination
    if destination.is_file() and _sha256_file(destination) == expected_sha256:
        return destination
    temporary = destination.with_name(
        "%s.part.%d" % (destination.name, os.getpid())
    )
    try:
        _hardlink_or_copy(str(source), str(temporary))
        actual = _sha256_file(temporary)
        if actual != expected_sha256:
            raise SystemExit(
                "PyRosetta staged wheel checksum mismatch for %s: %s"
                % (temporary, actual)
            )
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _stage_verified_build_input(source, destination, expected_sha256):
    source = pathlib.Path(source).resolve()
    destination = pathlib.Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source == destination:
        return destination
    if destination.is_file() and _sha256_file(destination) == expected_sha256:
        return destination
    temporary = destination.with_name("%s.part.%d" % (destination.name, os.getpid()))
    try:
        _hardlink_or_copy(str(source), str(temporary))
        actual = _sha256_file(temporary)
        if actual != expected_sha256:
            raise SystemExit(
                "Staged build input checksum mismatch for %s: %s; expected %s"
                % (temporary, actual, expected_sha256)
            )
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _read_rosetta_tools_manifest(repo_root):
    path = pathlib.Path(repo_root) / ROSETTA_TOOLS_MANIFEST_REL
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line_number == 1:
            if raw != "sha256\tsize_bytes\tpath":
                raise SystemExit("Invalid Rosetta tools manifest header: %s" % path)
            continue
        checksum, size_text, rel_path = raw.split("\t")
        rows.append((checksum, int(size_text), rel_path))
    if not rows:
        raise SystemExit("Rosetta tools manifest is empty: %s" % path)
    return tuple(rows)


def _validate_rosetta_tools_archive(path, repo_root):
    path = pathlib.Path(path)
    expected = {
        rel_path: (checksum, size)
        for checksum, size, rel_path in _read_rosetta_tools_manifest(repo_root)
    }
    try:
        with tarfile.open(str(path), mode="r:gz") as archive:
            all_members = archive.getmembers()
            unsafe = [
                member.name
                for member in all_members
                if not member.isfile()
                or pathlib.PurePosixPath(member.name).is_absolute()
                or any(
                    part in ("", ".", "..")
                    for part in pathlib.PurePosixPath(member.name).parts
                )
                or str(pathlib.PurePosixPath(member.name)) != member.name
            ]
            if unsafe:
                raise SystemExit(
                    "Rosetta tools archive contains unsafe/non-regular members: %s"
                    % unsafe
                )
            names = [member.name for member in all_members]
            if len(names) != len(set(names)):
                raise SystemExit(
                    "Rosetta tools archive contains duplicate members: %s" % path
                )
            members = {member.name: member for member in all_members}
            if set(members) != set(expected):
                raise SystemExit(
                    "Rosetta tools archive has non-exact file set: %s" % path
                )
            for rel_path, (expected_sha256, expected_size) in expected.items():
                member = members[rel_path]
                if member.size != expected_size:
                    raise SystemExit(
                        "Rosetta tools archive size mismatch for %s: %d; expected %d"
                        % (rel_path, member.size, expected_size)
                    )
                handle = archive.extractfile(member)
                if handle is None:
                    raise SystemExit(
                        "Could not read Rosetta tools archive member: %s" % rel_path
                    )
                digest = hashlib.sha256()
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                if digest.hexdigest() != expected_sha256:
                    raise SystemExit(
                        "Rosetta tools archive SHA256 mismatch for %s: %s; expected %s"
                        % (rel_path, digest.hexdigest(), expected_sha256)
                    )
    except (OSError, tarfile.TarError) as exc:
        raise SystemExit("Invalid Rosetta tools archive %s: %s" % (path, exc))
    return path


def _validate_ananas_archive(path):
    path = pathlib.Path(path)
    actual = _sha256_file(path)
    if actual != ANANAS_ARCHIVE_SHA256:
        raise SystemExit(
            "AnAnaS archive checksum mismatch for %s: %s; expected %s"
            % (path, actual, ANANAS_ARCHIVE_SHA256)
        )
    try:
        with zipfile.ZipFile(str(path)) as archive:
            regular = [info for info in archive.infolist() if not info.is_dir()]
            if [info.filename for info in regular] != ["ananas"] or regular[0].file_size != 2_508_152:
                raise SystemExit("AnAnaS archive has unexpected contents: %s" % path)
            binary_sha256 = hashlib.sha256(archive.read(regular[0])).hexdigest()
            if binary_sha256 != ANANAS_BINARY_SHA256:
                raise SystemExit(
                    "AnAnaS binary checksum mismatch in %s: %s; expected %s"
                    % (path, binary_sha256, ANANAS_BINARY_SHA256)
                )
    except (OSError, zipfile.BadZipFile) as exc:
        raise SystemExit("Invalid AnAnaS archive %s: %s" % (path, exc))
    return path


def _build_contract_fingerprint(contract):
    if not isinstance(contract, dict):
        raise SystemExit("Container build contract must be a JSON object")
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _validate_build_contract_schema(contract, expected_features=None):
    if not isinstance(contract, dict) or contract.get("schema_version") != 5:
        raise SystemExit("Image build metadata requires a schema-v5 build contract")
    features = contract.get("build_features")
    if not isinstance(features, list) or not all(
        isinstance(feature, str) for feature in features
    ):
        raise SystemExit("Image build contract has invalid build_features")
    if features != sorted(set(features)):
        raise SystemExit("Image build contract features must be sorted and unique")
    if expected_features is not None and features != sorted(set(expected_features)):
        raise SystemExit("Image build contract features do not match selected image features")
    expected_licenses = sorted(
        BUILD_FEATURE_LICENSES[feature]
        for feature in features
        if feature in BUILD_FEATURE_LICENSES
    )
    if contract.get("accepted_licenses") != expected_licenses:
        raise SystemExit("Image build contract license IDs do not match its features")
    for key in ("context_sha256", "definition_sha256"):
        if not _valid_sha256(contract.get(key)):
            raise SystemExit("Image build contract has invalid %s" % key)
    external_inputs = contract.get("external_inputs")
    if not isinstance(external_inputs, list):
        raise SystemExit("Image build contract has invalid external_inputs")
    for entry in external_inputs:
        if not (
            isinstance(entry, dict)
            and isinstance(entry.get("path"), str)
            and isinstance(entry.get("size"), int)
            and entry["size"] > 0
            and _valid_sha256(entry.get("sha256"))
        ):
            raise SystemExit("Image build contract has an invalid external input record")


def _validate_recorded_build_contract(metadata, expected_contract=None, expected_fingerprint=None):
    recorded_contract = metadata.get("build_contract")
    recorded_fingerprint = metadata.get("build_fingerprint")
    if recorded_contract is None and recorded_fingerprint is None:
        if expected_contract is not None or expected_fingerprint is not None:
            raise SystemExit("Image build metadata is missing its build contract/fingerprint")
        return
    if recorded_contract is None or not isinstance(recorded_fingerprint, str):
        raise SystemExit("Image build metadata has an incomplete build contract/fingerprint")
    expected_features = (
        expected_contract.get("build_features")
        if isinstance(expected_contract, dict)
        else metadata.get("build_features")
    )
    _validate_build_contract_schema(recorded_contract, expected_features)
    calculated = _build_contract_fingerprint(recorded_contract)
    if calculated != recorded_fingerprint:
        raise SystemExit(
            "Image build metadata contract fingerprint mismatch: recorded=%s calculated=%s"
            % (recorded_fingerprint, calculated)
        )
    if expected_contract is not None and recorded_contract != expected_contract:
        raise SystemExit("Image build metadata contract does not match current build inputs")
    if expected_fingerprint is not None and recorded_fingerprint != expected_fingerprint:
        raise SystemExit("Image build metadata fingerprint does not match current build inputs")


def _record_image_identity(
    state,
    image_path,
    *,
    build_features=(),
    build_contract=None,
    build_fingerprint=None,
    replace_provenance=False
):
    image_path = pathlib.Path(image_path).resolve()
    stat_result = image_path.stat()
    checksum = _sha256_file(image_path)
    identity = {
        "path": str(image_path),
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "sha256": checksum,
        "build_features": sorted(set(build_features)),
    }
    if build_contract is not None:
        identity["build_contract"] = build_contract
    if build_fingerprint is not None:
        identity["build_fingerprint"] = build_fingerprint
    sidecar = pathlib.Path(str(image_path) + ".sha256")
    metadata_path = pathlib.Path(str(image_path) + ".build.json")
    for provenance_path in (sidecar, metadata_path):
        if provenance_path.exists() and not provenance_path.is_file():
            raise SystemExit(
                "Image provenance path is not a regular file: %s" % provenance_path
            )
    if sidecar.is_file() and not replace_provenance:
        try:
            fields = sidecar.read_text(encoding="ascii").strip().split()
        except (OSError, UnicodeDecodeError) as exc:
            raise SystemExit("Invalid image checksum sidecar %s: %s" % (sidecar, exc))
        if not fields or fields[0] != checksum:
            recorded = fields[0] if fields else "missing"
            raise SystemExit(
                "Image checksum sidecar mismatch for %s: recorded=%s actual=%s"
                % (image_path, recorded, checksum)
            )
    if metadata_path.is_file() and not replace_provenance:
        try:
            existing_metadata = _json_load(metadata_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit("Invalid image build metadata %s: %s" % (metadata_path, exc))
        if not (
            isinstance(existing_metadata, dict)
            and existing_metadata.get("sha256") == checksum
            and existing_metadata.get("size") == stat_result.st_size
            and existing_metadata.get("build_features") == identity["build_features"]
        ):
            raise SystemExit(
                "Image build metadata does not match image/selection: %s"
                % metadata_path
            )
        _validate_recorded_build_contract(
            existing_metadata,
            expected_contract=build_contract,
            expected_fingerprint=build_fingerprint,
        )
    if replace_provenance or not sidecar.exists():
        sidecar_tmp = sidecar.with_name(sidecar.name + ".tmp")
        sidecar_tmp.write_text(
            "%s  %s\n" % (checksum, image_path.name), encoding="ascii"
        )
        os.replace(str(sidecar_tmp), str(sidecar))
    if replace_provenance or not metadata_path.exists():
        _json_dump_atomic(metadata_path, identity)
    validation_cache = os.environ.get("GH_ASSET_VALIDATION_CACHE")
    if validation_cache:
        key = hashlib.sha256(os.fsencode(str(image_path))).hexdigest()
        stamp = pathlib.Path(validation_cache) / (key + ".sha256")
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp_tmp = stamp.with_name(stamp.name + ".tmp")
        stamp_tmp.write_text(
            "%s %d %d\n" % (checksum, stat_result.st_size, stat_result.st_mtime_ns),
            encoding="ascii",
        )
        os.replace(str(stamp_tmp), str(stamp))
    state["output_identity"] = identity
    return identity


def _quick_file_signature(path, size):
    digest = hashlib.sha256()
    chunk_size = 64 * 1024
    with pathlib.Path(path).open("rb") as handle:
        if size <= 4 * chunk_size:
            digest.update(handle.read())
        else:
            for offset in (0, size // 2, size - chunk_size):
                handle.seek(offset)
                digest.update(handle.read(chunk_size))
    return digest.hexdigest()


def _cached_sha256(path, cache=None):
    path = pathlib.Path(path)
    if cache is None:
        return _sha256_file(path)
    stat_result = path.stat()
    key = str(path.resolve())
    metadata = {
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "ctime_ns": stat_result.st_ctime_ns,
        "mode": stat.S_IMODE(stat_result.st_mode),
    }
    newest_timestamp = max(stat_result.st_mtime_ns, stat_result.st_ctime_ns)
    if time.time_ns() - newest_timestamp <= _METADATA_SETTLE_NS:
        metadata["recent_sample_sha256"] = _quick_file_signature(
            path, stat_result.st_size
        )
    cached = cache.get(key)
    if isinstance(cached, dict) and all(
        cached.get(name) == value for name, value in metadata.items()
    ):
        checksum = cached.get("sha256")
        if isinstance(checksum, str) and len(checksum) == 64:
            return checksum
    checksum = _sha256_file(path)
    cache[key] = dict(metadata, sha256=checksum)
    return checksum


def _executable_version(executable):
    if not executable or not pathlib.Path(executable).is_file():
        return None
    return subprocess.check_output([str(executable), "version"], text=True).strip()


def _executable_identity(executable, sha256_cache=None):
    path = pathlib.Path(executable).resolve()
    if not path.is_file():
        raise SystemExit("Executable is missing or not a regular file: %s" % path)
    version = _executable_version(path)
    if not version:
        raise SystemExit("Could not resolve executable version: %s" % path)
    return {
        "path": str(path),
        "sha256": _cached_sha256(path, sha256_cache),
        "version": version,
    }


def _build_context_fingerprint(repo_root, rel_paths=None, sha256_cache=None):
    repo_root = pathlib.Path(repo_root).resolve()
    rel_paths = rel_paths or _tracked_build_context_files(repo_root)
    digest = hashlib.sha256()
    digest.update(b"gh-build-context-v2\0")
    for rel_path in rel_paths:
        source = repo_root / rel_path
        digest.update(os.fsencode(str(rel_path)))
        digest.update(b"\0")
        if source.is_symlink():
            digest.update(b"L\0")
            digest.update(os.fsencode(os.readlink(str(source))))
        else:
            digest.update(b"F\0")
            digest.update(str(stat.S_IMODE(source.stat().st_mode)).encode("ascii"))
            digest.update(b"\0")
            digest.update(_cached_sha256(source, sha256_cache).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _stage_build_context(repo_root, stage_root, rel_paths=None):
    repo_root = pathlib.Path(repo_root).resolve()
    stage_root = pathlib.Path(stage_root).resolve()
    stage_repo = stage_root / "repo"
    rel_paths = rel_paths or _tracked_build_context_files(repo_root)
    stage_repo.mkdir(parents=True)
    for rel_path in rel_paths:
        source = repo_root / rel_path
        destination = stage_repo / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            os.symlink(os.readlink(str(source)), str(destination))
        else:
            _hardlink_or_copy(str(source), str(destination))
    print("Staged %d tracked files for container build." % len(rel_paths))
    return stage_repo


def _run(cmd, cwd=None, env=None):
    if isinstance(cmd, (list, tuple)):
        printable = " ".join([str(x) for x in cmd])
    else:
        printable = str(cmd)
    print("+ " + printable)
    rc = subprocess.call(cmd, cwd=cwd, env=env)
    if rc != 0:
        raise SystemExit(f"Command failed with exit code {rc}: {printable}")


def _write_build_arg_file(path, build_features):
    path = pathlib.Path(path)
    selected = set(build_features)
    values = {
        "GH_INSTALL_PYROSETTA": int("pyrosetta" in selected),
        "GH_INSTALL_ANANAS": int("ananas" in selected),
        "GH_INSTALL_ROSETTA_TOOLS": int("rosetta" in selected),
        "GH_BUILD_FEATURES": ",".join(sorted(selected)) if selected else "core",
    }
    path.write_text(
        "".join("%s=%s\n" % item for item in values.items()),
        encoding="utf-8",
    )
    return path


def _parse_assignment(raw, option_name):
    if "=" not in raw:
        raise SystemExit("%s expects KEY=VALUE, got: %s" % (option_name, raw))
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise SystemExit("%s expects a non-empty KEY, got: %s" % (option_name, raw))
    return key, value


def _render_python_literal(value):
    return repr(value)


def _docker_mount_source(bind):
    return str(bind).split(":", 1)[0]


def _write_docker_apptainer_runner(
    path,
    *,
    docker_bin,
    docker_image,
    sif_path,
    container_repo_root,
    binds,
    env_defaults,
    env_prepends,
    container_python,
    use_nv,
):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """#!/usr/bin/env python3
import os
import pathlib
import subprocess
import sys


DOCKER_BIN = {docker_bin}
DOCKER_IMAGE = {docker_image}
SIF_PATH = {sif_path}
CONTAINER_REPO_ROOT = {container_repo_root}
CONTAINER_PYTHON = {container_python}
USE_NV = {use_nv}
BIND_MOUNTS = {binds}
ENV_DEFAULTS = {env_defaults}
ENV_PREPENDS = {env_prepends}
SENSITIVE_ENV_FORWARD_KEYS = [
    "GH_CUDA_PREINITIALIZE",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
]


def _prepend_env(env, key, value):
    current = env.get(key, "")
    env[key] = value + (os.pathsep + current if current else "")


def _docker_mount_source(bind):
    return str(bind).split(":", 1)[0]


def _add_docker_mount(cmd, seen, host_path, mode="rw"):
    resolved = str(pathlib.Path(host_path).resolve())
    if resolved in seen:
        return
    seen.add(resolved)
    spec = f"{{resolved}}:{{resolved}}"
    if mode:
        spec += f":{{mode}}"
    cmd += ["-v", spec]


def main() -> int:
    host_repo_value = os.environ.get("GH_HOST_REPO_ROOT")
    source_map_value = host_repo_value or os.environ.get("GH_HOST_REPO_MAP_ROOT")
    host_repo_root = (
        str(pathlib.Path(host_repo_value).resolve()) if host_repo_value else None
    )
    source_map_root = (
        str(pathlib.Path(source_map_value).resolve()) if source_map_value else None
    )

    py_args = sys.argv[1:]
    if source_map_root and len(py_args) >= 2 and py_args[0] == "-c":
        py_args = [
            "-c",
            py_args[1].replace(source_map_root, CONTAINER_REPO_ROOT),
            *py_args[2:],
        ]

    env = os.environ.copy()
    env_overrides = {{
        key: env[key]
        for key in SENSITIVE_ENV_FORWARD_KEYS
        if key in env
    }}
    for key, value in ENV_PREPENDS:
        _prepend_env(env, key, value)
    for key, value in ENV_DEFAULTS.items():
        env[key] = value
    env.update(env_overrides)
    container_home = ENV_DEFAULTS.get("HOME")

    inner_cmd = [
        "apptainer",
        "exec",
        "--containall",
        "--cleanenv",
        "--no-home",
        "--no-mount",
        "cwd",
        "--pwd",
        CONTAINER_REPO_ROOT,
    ]
    if USE_NV:
        inner_cmd.append("--nv")
    if host_repo_root:
        inner_cmd += ["-B", f"{{host_repo_root}}:{{CONTAINER_REPO_ROOT}}"]
    for bind in BIND_MOUNTS:
        inner_cmd += ["-B", bind]
    inner_cmd += [SIF_PATH]
    if container_home:
        inner_cmd += ["/usr/bin/env", f"HOME={{container_home}}"]
    inner_cmd += [CONTAINER_PYTHON, *py_args]

    docker_cmd = [DOCKER_BIN, "run", "--rm", "--privileged"]
    docker_gpus = os.environ.get("GH_DOCKER_GPUS", "all" if USE_NV else "")
    if docker_gpus and docker_gpus.lower() not in {{"0", "false", "none", "off"}}:
        docker_cmd += ["--gpus", docker_gpus]

    seen = set()
    if host_repo_root:
        _add_docker_mount(docker_cmd, seen, host_repo_root)
    _add_docker_mount(docker_cmd, seen, SIF_PATH, mode="ro")
    for bind in BIND_MOUNTS:
        source = _docker_mount_source(bind)
        if source:
            _add_docker_mount(docker_cmd, seen, source)

    forward_keys = set(ENV_DEFAULTS)
    forward_keys.update(key for key, _value in ENV_PREPENDS)
    forward_keys.update(SENSITIVE_ENV_FORWARD_KEYS)
    for key in sorted(forward_keys):
        if key == "HOME":
            continue
        if key not in env or not env[key]:
            continue
        env[f"APPTAINERENV_{{key}}"] = env[key]
        docker_cmd += ["-e", f"APPTAINERENV_{{key}}"]
    docker_cmd += ["-w", "/tmp", DOCKER_IMAGE, *inner_cmd]
    subprocess.check_call(docker_cmd, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""".format(
        docker_bin=_render_python_literal(str(docker_bin)),
        docker_image=_render_python_literal(str(docker_image)),
        sif_path=_render_python_literal(str(sif_path)),
        container_repo_root=_render_python_literal(str(container_repo_root)),
        container_python=_render_python_literal(str(container_python)),
        use_nv=repr(bool(use_nv)),
        binds=_render_python_literal([str(x) for x in binds]),
        env_defaults=_render_python_literal(dict(env_defaults)),
        env_prepends=_render_python_literal(list(env_prepends)),
    )
    path.write_text(text, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print("Wrote Docker/Apptainer runner: %s" % path)


def _write_python_runner(
    path,
    *,
    apptainer,
    runner_backend="apptainer",
    docker_bin="docker",
    docker_image=DEFAULT_DOCKER_APPTAINER_IMAGE,
    sif_path,
    container_repo_root,
    binds,
    env_defaults,
    env_prepends,
    container_python,
    use_nv,
):
    if runner_backend == "docker-apptainer":
        _write_docker_apptainer_runner(
            path,
            docker_bin=docker_bin,
            docker_image=docker_image,
            sif_path=sif_path,
            container_repo_root=container_repo_root,
            binds=binds,
            env_defaults=env_defaults,
            env_prepends=env_prepends,
            container_python=container_python,
            use_nv=use_nv,
        )
        return
    if runner_backend != "apptainer":
        raise SystemExit(f"Unknown runner backend: {runner_backend}")
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """#!/usr/bin/env python3
import os
import pathlib
import subprocess
import sys


APPTAINER_BIN = {apptainer}
SIF_PATH = {sif_path}
CONTAINER_REPO_ROOT = {container_repo_root}
CONTAINER_PYTHON = {container_python}
USE_NV = {use_nv}
BIND_MOUNTS = {binds}
ENV_DEFAULTS = {env_defaults}
ENV_PREPENDS = {env_prepends}
ENV_FORWARD_KEYS = [
    "CUDA_VISIBLE_DEVICES",
    "GH_CUDA_PREINITIALIZE",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "GH_DPLM_PYTHON",
    "GH_DPLM_REPO",
    "GH_DPLM_MODEL_SNAPSHOT",
    "GH_ESM3_MODEL_SNAPSHOT",
    "GH_ESMFOLD2_MODEL_SNAPSHOT",
    "GH_ESMFOLD2_ESMC_SNAPSHOT",
    "GH_RPXDOCK_PYTHON",
    "GH_RPXDOCK_REPO",
    "GH_WILLUTIL_REPO",
    "SETUPTOOLS_USE_DISTUTILS",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
]
PATH_ENV_KEYS = [
    "GH_DPLM_PYTHON",
    "GH_DPLM_REPO",
    "GH_RPXDOCK_PYTHON",
    "GH_RPXDOCK_REPO",
    "GH_WILLUTIL_REPO",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
]


def _prepend_env(env, key, value):
    current = env.get(key, "")
    env[key] = value + (os.pathsep + current if current else "")


def _bind_source(bind):
    return str(bind).split(":", 1)[0]


def _host_bind_path(path_value):
    if not path_value:
        return None
    path = pathlib.Path(str(path_value)).expanduser()
    if path.is_file() and path.parent.name == "bin":
        return path.parent.parent
    if path.is_file():
        return path.parent
    return path


def _add_bind(cmd, seen, host_path):
    bind_path = _host_bind_path(host_path)
    if bind_path is None or not bind_path.exists():
        return
    resolved = str(bind_path.resolve())
    if resolved in seen:
        return
    seen.add(resolved)
    cmd += ["-B", f"{{resolved}}:{{resolved}}"]


def main() -> int:
    host_repo_value = os.environ.get("GH_HOST_REPO_ROOT")
    source_map_value = host_repo_value or os.environ.get("GH_HOST_REPO_MAP_ROOT")
    host_repo_root = (
        str(pathlib.Path(host_repo_value).resolve()) if host_repo_value else None
    )
    source_map_root = (
        str(pathlib.Path(source_map_value).resolve()) if source_map_value else None
    )

    py_args = sys.argv[1:]
    if source_map_root and len(py_args) >= 2 and py_args[0] == "-c":
        py_args = [
            "-c",
            py_args[1].replace(source_map_root, CONTAINER_REPO_ROOT),
            *py_args[2:],
        ]

    env = os.environ.copy()
    env_overrides = {{key: env[key] for key in ENV_FORWARD_KEYS if key in env}}
    apptainer_bin_dir = str(pathlib.Path(APPTAINER_BIN).resolve().parent)
    _prepend_env(env, "PATH", apptainer_bin_dir)
    for key, value in ENV_PREPENDS:
        _prepend_env(env, key, value)
    for key, value in ENV_DEFAULTS.items():
        env[key] = value
    env.update(env_overrides)
    container_home = ENV_DEFAULTS.get("HOME")

    cmd = [
        APPTAINER_BIN,
        "exec",
        "--containall",
        "--cleanenv",
        "--no-home",
        "--no-mount",
        "cwd",
        "--pwd",
        CONTAINER_REPO_ROOT,
    ]
    if USE_NV:
        cmd.append("--nv")
    if host_repo_root:
        cmd += ["-B", f"{{host_repo_root}}:{{CONTAINER_REPO_ROOT}}"]
    for bind in BIND_MOUNTS:
        cmd += ["-B", bind]
    seen_binds = set(_bind_source(bind) for bind in BIND_MOUNTS)
    if host_repo_root:
        seen_binds.add(host_repo_root)
    for key in PATH_ENV_KEYS:
        value = env.get(key)
        if value:
            _add_bind(cmd, seen_binds, value)
    forward_keys = set(ENV_DEFAULTS)
    forward_keys.update(key for key, _value in ENV_PREPENDS)
    forward_keys.update(ENV_FORWARD_KEYS)
    for key in forward_keys:
        if key == "HOME":
            continue
        if key in env and env[key]:
            env[f"APPTAINERENV_{{key}}"] = env[key]
    cmd += [SIF_PATH]
    if container_home:
        cmd += ["/usr/bin/env", f"HOME={{container_home}}"]
    cmd += [CONTAINER_PYTHON, *py_args]
    try:
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError as exc:
        # The child already printed its diagnostic; preserve its exit status
        # rather than wrapping an expected CLI failure in a Python traceback.
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""".format(
        apptainer=_render_python_literal(str(apptainer)),
        sif_path=_render_python_literal(str(sif_path)),
        container_repo_root=_render_python_literal(str(container_repo_root)),
        container_python=_render_python_literal(str(container_python)),
        use_nv=repr(bool(use_nv)),
        binds=_render_python_literal([str(x) for x in binds]),
        env_defaults=_render_python_literal(dict(env_defaults)),
        env_prepends=_render_python_literal(list(env_prepends)),
    )
    path.write_text(text, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print("Wrote runner: %s" % path)


def write_python_runner(
    path,
    *,
    apptainer,
    runner_backend="apptainer",
    docker_bin="docker",
    docker_image=DEFAULT_DOCKER_APPTAINER_IMAGE,
    sif_path,
    container_repo_root="/opt/Astrolabe",
    binds=(),
    env_defaults=None,
    env_prepends=(),
    container_python="python",
    use_nv=True,
):
    """Public helper used by bootstrap code and tests."""
    _write_python_runner(
        path,
        apptainer=apptainer,
        runner_backend=runner_backend,
        docker_bin=docker_bin,
        docker_image=docker_image,
        sif_path=sif_path,
        container_repo_root=container_repo_root,
        binds=list(binds),
        env_defaults=dict(env_defaults or {}),
        env_prepends=list(env_prepends),
        container_python=container_python,
        use_nv=use_nv,
    )


def _container_def_path(def_file, repo_root):
    try:
        rel_path = pathlib.Path(def_file).resolve().relative_to(pathlib.Path(repo_root).resolve())
    except ValueError:
        raise SystemExit(
            "Docker builder requires --def to be inside the repository checkout: %s"
            % def_file
        )
    return "/work/%s" % rel_path.as_posix()


def _staged_def_path(stage_repo, def_file, repo_root):
    try:
        rel_path = pathlib.Path(def_file).resolve().relative_to(pathlib.Path(repo_root).resolve())
    except ValueError:
        raise SystemExit(
            "Docker builder requires --def to be inside the repository checkout: %s"
            % def_file
        )
    return pathlib.Path(stage_repo) / rel_path


def _docker_base_from_def(def_file):
    lines = pathlib.Path(def_file).read_text(encoding="utf-8").splitlines()
    bootstrap_idx = None
    from_idx = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        lower = stripped.lower()
        if bootstrap_idx is None and lower == "bootstrap: docker":
            bootstrap_idx = idx
        elif bootstrap_idx is not None and from_idx is None and lower.startswith("from:"):
            from_idx = idx
            break
    if bootstrap_idx is None or from_idx is None:
        return None
    return lines[from_idx].split(":", 1)[1].strip()


def _prepare_docker_archive_base(
    *,
    docker_bin,
    archive_dir,
    container_archive_path,
    stage_repo,
    def_file,
    repo_root,
    base_image,
):
    base_image = base_image or _docker_base_from_def(def_file)
    if not base_image:
        return None
    archive_dir = pathlib.Path(archive_dir).resolve()
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "base.tar"
    _run([docker_bin, "pull", base_image])
    _run([docker_bin, "save", "-o", str(archive_path), base_image])

    staged_def = _staged_def_path(stage_repo, def_file, repo_root)
    lines = staged_def.read_text(encoding="utf-8").splitlines()
    replaced_bootstrap = False
    replaced_from = False
    out_lines = []
    for line in lines:
        stripped = line.strip()
        if (not replaced_bootstrap) and stripped.lower() == "bootstrap: docker":
            out_lines.append("Bootstrap: docker-archive")
            replaced_bootstrap = True
            continue
        if replaced_bootstrap and (not replaced_from) and stripped.lower().startswith("from:"):
            out_lines.append("From: %s" % container_archive_path)
            replaced_from = True
            continue
        out_lines.append(line)
    if not (replaced_bootstrap and replaced_from):
        raise SystemExit("Could not rewrite Docker bootstrap definition for docker-archive: %s" % staged_def)
    staged_def.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return archive_path


def _docker_chown_build_artifacts(*, docker_bin, docker_image, docker_work_dir, tmp_path, build_cache=None):
    uid = os.getuid()
    gid = os.getgid()
    tmp_path = pathlib.Path(tmp_path).resolve()
    docker_work_dir = pathlib.Path(docker_work_dir).resolve()
    mounts = [
        "-v",
        "%s:/apptainer-work" % docker_work_dir,
        "-v",
        "%s:/out" % tmp_path.parent,
    ]
    cache_chown = ""
    if build_cache:
        build_cache = pathlib.Path(build_cache).resolve()
        mounts += [
            "-v",
            "%s:/cache" % build_cache,
        ]
        cache_chown = "chown -R {uid}:{gid} /cache 2>/dev/null || true; "
    cmd = [
        docker_bin,
        "run",
        "--rm",
        "--privileged",
        *mounts,
        docker_image,
        "sh",
        "-c",
        (
            "chown -R {uid}:{gid} /apptainer-work 2>/dev/null || true; "
            + cache_chown
            + "chown {uid}:{gid} /out/{tmp_name} 2>/dev/null || true"
        ).format(uid=uid, gid=gid, tmp_name=tmp_path.name),
    ]
    subprocess.call(cmd)


def _docker_prepare_build_cache_for_container_root(*, docker_bin, docker_image, build_cache):
    build_cache = pathlib.Path(build_cache).resolve()
    cmd = [
        docker_bin,
        "run",
        "--rm",
        "--privileged",
        "-v",
        "%s:/cache" % build_cache,
        docker_image,
        "sh",
        "-c",
        "chown -R 0:0 /cache",
    ]
    _run(cmd)


def _inspect_image(
    *,
    builder,
    apptainer,
    docker_bin,
    docker_image,
    image_path,
):
    image_path = pathlib.Path(image_path).resolve()
    if builder == "apptainer":
        env = os.environ.copy()
        engine_bin = str(pathlib.Path(apptainer).resolve().parent)
        env["PATH"] = engine_bin + (
            os.pathsep + env["PATH"] if env.get("PATH") else ""
        )
        _run([apptainer, "inspect", str(image_path)], env=env)
        return
    _run(
        [
            docker_bin,
            "run",
            "--rm",
            "--privileged",
            "-v",
            "%s:/image.sif:ro" % image_path,
            docker_image,
            "apptainer",
            "inspect",
            "/image.sif",
        ]
    )


def _build_with_docker(
    *,
    docker_bin,
    docker_image,
    docker_base_image,
    docker_base_source,
    docker_network,
    build_cache,
    stage_repo,
    tmp_path,
    def_file,
    repo_root,
    build_features=(),
):
    tmp_path = pathlib.Path(tmp_path).resolve()
    stage_repo = pathlib.Path(stage_repo).resolve()
    docker_work_dir = stage_repo.parent / "docker-build-work"
    build_cache = pathlib.Path(build_cache).resolve() if build_cache else None
    apptainer_cache = (
        build_cache / "apptainer" if build_cache else docker_work_dir / "apptainer_cache"
    )
    apptainer_tmp = docker_work_dir / "apptainer_tmp"
    tmp_dir = docker_work_dir / "tmp"
    for path in (apptainer_cache, apptainer_tmp, tmp_dir):
        path.mkdir(parents=True, exist_ok=True)
    build_arg_file = _write_build_arg_file(
        docker_work_dir / "build-args.env", build_features
    )
    if build_cache:
        (build_cache / "pip").mkdir(parents=True, exist_ok=True)
        (build_cache / "mamba" / "pkgs").mkdir(parents=True, exist_ok=True)
        _docker_prepare_build_cache_for_container_root(
            docker_bin=docker_bin,
            docker_image=docker_image,
            build_cache=build_cache,
        )

    extra_mounts = []
    if docker_base_source == "docker-archive":
        archive_dir = docker_work_dir / "base"
        _prepare_docker_archive_base(
            docker_bin=docker_bin,
            archive_dir=archive_dir,
            container_archive_path="/base/base.tar",
            stage_repo=stage_repo,
            def_file=def_file,
            repo_root=repo_root,
            base_image=docker_base_image,
        )
        extra_mounts.extend(["-v", "%s:/base:ro" % archive_dir])
    elif docker_base_source != "docker":
        raise SystemExit("Unknown Docker base source: %s" % docker_base_source)
    elif docker_base_image:
        raise SystemExit(
            "--docker-base-image requires --docker-base-source docker-archive; "
            "the default Docker builder downloads the base image from the definition's official From: address."
        )

    cmd = [
        docker_bin,
        "run",
        "--rm",
        "--privileged",
    ]
    if docker_network:
        cmd.extend(["--network", docker_network])
    cmd += [
        "-v",
        "%s:/work" % stage_repo,
        "-v",
        "%s:/out" % tmp_path.parent,
        "-v",
        "%s:/apptainer-work" % docker_work_dir,
        "-e",
        "APPTAINER_CACHEDIR=%s"
        % ("/apptainer-cache" if build_cache else "/apptainer-work/apptainer_cache"),
        "-e",
        "APPTAINER_TMPDIR=/apptainer-work/apptainer_tmp",
        "-e",
        "TMPDIR=/apptainer-work/tmp",
    ]
    if build_cache:
        cmd += [
            "-v",
            "%s:/cache" % build_cache,
            "-v",
            "%s:/apptainer-cache" % apptainer_cache,
        ]
    cmd += extra_mounts
    cmd += [
        "-w",
        "/work",
        docker_image,
        "apptainer",
        "build",
    ]
    cmd += [
        "--build-arg-file",
        "/apptainer-work/%s" % build_arg_file.name,
    ]
    if build_cache:
        cmd += [
            "--bind",
            "/cache:/cache",
        ]
    cmd += [
        "/out/%s" % tmp_path.name,
        _container_def_path(def_file, repo_root),
    ]
    try:
        _run(cmd)
    finally:
        _docker_chown_build_artifacts(
            docker_bin=docker_bin,
            docker_image=docker_image,
            docker_work_dir=docker_work_dir,
            tmp_path=tmp_path,
            build_cache=build_cache,
        )


def main(argv=None):
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    p = argparse.ArgumentParser(description="Build Astrolabe Apptainer images (resumable).")
    p.add_argument(
        "--apptainer-bin",
        default=None,
        help="Explicit Apptainer/Singularity executable for build and generated runner.",
    )
    p.add_argument(
        "--builder",
        choices=("apptainer", "docker"),
        default="apptainer",
        help=(
            "Image build backend. 'apptainer' uses a host Apptainer/Singularity "
            "binary. 'docker' runs the official Apptainer image in Docker."
        ),
    )
    p.add_argument(
        "--docker-bin",
        default=None,
        help="Explicit Docker executable for --builder docker or docker-apptainer runner.",
    )
    p.add_argument(
        "--docker-builder-image",
        default=DEFAULT_DOCKER_APPTAINER_IMAGE,
        help="Docker image that provides the apptainer binary for --builder docker.",
    )
    p.add_argument(
        "--docker-network",
        default=DEFAULT_DOCKER_NETWORK,
        help=(
            "Docker network mode for --builder docker. Default: host, so the "
            "inner Apptainer build downloads official OCI bases through the "
            "target server's normal network path. Pass an empty string to omit."
        ),
    )
    p.add_argument(
        "--docker-base-source",
        choices=("docker", "docker-archive"),
        default="docker",
        help=(
            "How --builder docker supplies the definition's base image. Default "
            "'docker' leaves Bootstrap: docker intact so Apptainer downloads "
            "from the official From: address. 'docker-archive' is an explicit "
            "fallback that first pulls with host Docker and mounts that archive."
        ),
    )
    p.add_argument(
        "--docker-base-image",
        default=None,
        help=(
            "Base OCI image for --docker-base-source docker-archive. Default: "
            "parse the first 'From:' line when the def uses 'Bootstrap: docker'."
        ),
    )
    p.add_argument(
        "--def",
        dest="def_file",
        default=str(repo_root / "container" / "Astrolabe.af3.def"),
        help="Apptainer definition file.",
    )
    p.add_argument(
        "--output",
        default=str(pathlib.Path.home() / ".gh_envs" / "images" / "Astrolabe_af3.sif"),
        help="Output .sif path (default: ~/.gh_envs/images/Astrolabe_af3.sif).",
    )
    p.add_argument(
        "--build-cache",
        default=None,
        help=(
            "Persistent host cache mounted read-write at /cache during build. "
            "The AF3 definition uses /cache/pip and /cache/mamba when present."
        ),
    )
    p.add_argument(
        "--build-input",
        action="append",
        default=[],
        help=(
            "External file baked into the image through --build-cache. May be repeated; "
            "its content hash becomes part of the reusable-image fingerprint."
        ),
    )
    p.add_argument(
        "--build-feature",
        action="append",
        choices=("pyrosetta", "ananas", "rosetta"),
        default=[],
        help=(
            "Additional image feature. PyRosetta is part of the default build."
        ),
    )
    p.add_argument(
        "--without-pyrosetta",
        action="store_true",
        help="Explicitly build a reduced image without the default PyRosetta feature.",
    )
    p.add_argument(
        "--accept-license",
        action="append",
        default=[],
        help=(
            "Explicit license identifier required by licensed build features."
        ),
    )
    p.add_argument(
        "--name",
        default="Astrolabe_af3",
        help="State directory name under ~/.gh_envs/.gh_container_setup/.",
    )
    p.add_argument(
        "--state-root",
        default=None,
        help=(
            "Explicit build-state directory. Default: <output-root>/.gh_container_setup/<name>, "
            "where <output-root> is the parent of the output images directory."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if output exists.",
    )
    p.add_argument(
        "--skip-build",
        action="store_true",
        help="Do not build the image; useful when only writing a runner for an existing SIF.",
    )
    p.add_argument(
        "--fakeroot",
        action="store_true",
        help="Use fakeroot mode for build (required on many systems for docker bootstrap).",
    )
    p.add_argument(
        "--fakeroot-bin",
        default=None,
        help=(
            "Explicit fakeroot executable to expose to Apptainer. The bootstrap uses "
            "this on hosts without /etc/subuid mappings or a system fakeroot command."
        ),
    )
    p.add_argument(
        "--remote",
        action="store_true",
        help="Use remote builder (alternative when fakeroot is unavailable).",
    )
    p.add_argument(
        "--write-runner",
        default=None,
        help="Write a GH_PY-compatible Python runner for the built/existing SIF.",
    )
    p.add_argument(
        "--runner-backend",
        choices=("apptainer", "docker-apptainer"),
        default="apptainer",
        help=(
            "Generated runner backend. 'docker-apptainer' runs Apptainer inside "
            "the official Docker image, useful on hosts with Docker but no usable "
            "host Apptainer."
        ),
    )
    p.add_argument(
        "--runner-docker-image",
        default=DEFAULT_DOCKER_APPTAINER_IMAGE,
        help="Docker image used by --runner-backend docker-apptainer.",
    )
    p.add_argument(
        "--container-repo-root",
        default="/opt/Astrolabe",
        help="Repo mount point inside the container used by the generated runner.",
    )
    p.add_argument(
        "--container-python",
        default="python",
        help="Python executable inside the container used by the generated runner.",
    )
    p.add_argument(
        "--runner-bind",
        action="append",
        default=[],
        metavar="HOST:CONTAINER",
        help="Additional bind mount for the generated runner; may be repeated.",
    )
    p.add_argument(
        "--runner-env-default",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Authoritative environment mapping for the generated runner; may be repeated.",
    )
    p.add_argument(
        "--runner-prepend-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Prepend VALUE to an environment variable in the generated runner; may be repeated.",
    )
    p.add_argument(
        "--runner-no-nv",
        action="store_true",
        help="Do not pass --nv in the generated runner.",
    )
    args = p.parse_args(raw_argv)

    if args.builder == "docker" and args.remote:
        raise SystemExit("--builder docker runs a local privileged Docker build; do not combine it with --remote.")

    needs_host_apptainer = args.builder == "apptainer" or (
        bool(args.write_runner) and args.runner_backend == "apptainer"
    )
    apptainer = (
        str(_expand(args.apptainer_bin))
        if args.apptainer_bin
        else (_detect_apptainer() if needs_host_apptainer else "apptainer")
    )
    if needs_host_apptainer and not apptainer:
        raise SystemExit("Could not find `apptainer` or `singularity` in PATH.")
    docker_bin = (
        _detect_docker(args.docker_bin)
        if args.builder == "docker" or (bool(args.write_runner) and args.runner_backend == "docker-apptainer")
        else None
    )
    if (
        args.builder == "docker"
        or (bool(args.write_runner) and args.runner_backend == "docker-apptainer")
    ) and not docker_bin:
        raise SystemExit("Could not find `docker` in PATH. Install Docker or pass --docker-bin.")
    docker_builder_identity = None
    docker_builder_execution_image = args.docker_builder_image

    build_feature_set = set(args.build_feature)
    if not args.without_pyrosetta:
        build_feature_set.update(DEFAULT_BUILD_FEATURES)
    build_features = tuple(sorted(build_feature_set))
    if "rosetta" in build_features and "pyrosetta" not in build_features:
        raise SystemExit(
            "--build-feature rosetta requires the default PyRosetta feature/database; "
            "do not combine it with --without-pyrosetta"
        )
    accepted_licenses = {
        token.strip().lower()
        for raw in args.accept_license
        for token in raw.split(",")
        if token.strip()
    }
    missing_feature_licenses = [
        license_id
        for feature, license_id in BUILD_FEATURE_LICENSES.items()
        if feature in build_features and license_id not in accepted_licenses
    ]
    if missing_feature_licenses:
        raise SystemExit(
            "Explicit license acceptance required for default/selected build features: %s"
            % ", ".join(
                "--accept-license %s" % license_id
                for license_id in missing_feature_licenses
            )
        )
    def_file = _expand(args.def_file)
    out_path = _expand(args.output)
    build_inputs = [_expand(raw) for raw in args.build_input]
    for build_input in build_inputs:
        if not build_input.is_file():
            raise SystemExit("Container build input is missing or not a file: %s" % build_input)
    if "pyrosetta" in build_features:
        wheels = [path for path in build_inputs if path.name == PYROSETTA_WHEEL_NAME]
        if len(wheels) != 1:
            raise SystemExit(
                "--build-feature pyrosetta requires exactly one verified "
                "--build-input %s" % PYROSETTA_WHEEL_NAME
            )
        actual = _sha256_file(wheels[0])
        if wheels[0].stat().st_size != PYROSETTA_WHEEL_SIZE:
            raise SystemExit(
                "PyRosetta build input size mismatch for %s: %d; expected %d"
                % (wheels[0], wheels[0].stat().st_size, PYROSETTA_WHEEL_SIZE)
            )
        if actual not in PYROSETTA_ALLOWED_SHA256:
            raise SystemExit(
                "PyRosetta build input checksum mismatch for %s: %s"
                % (wheels[0], actual)
            )
        if not args.build_cache:
            raise SystemExit(
                "--build-feature pyrosetta requires --build-cache so the verified "
                "wheel is the file consumed by the definition"
            )
        staged_wheel = _stage_pyrosetta_wheel(
            wheels[0], _expand(args.build_cache), actual
        )
        build_inputs = [
            staged_wheel if path == wheels[0] else path for path in build_inputs
        ]
    if "ananas" in build_features:
        archives = [path for path in build_inputs if path.name == ANANAS_ARCHIVE_NAME]
        if len(archives) != 1:
            raise SystemExit(
                "--build-feature ananas requires exactly one verified --build-input %s"
                % ANANAS_ARCHIVE_NAME
            )
        _validate_ananas_archive(archives[0])
        if not args.build_cache:
            raise SystemExit(
                "--build-feature ananas requires --build-cache so the verified archive "
                "is the file consumed by the definition"
            )
        staged_archive = _stage_verified_build_input(
            archives[0],
            _expand(args.build_cache) / "sources" / "ananas" / ANANAS_ARCHIVE_NAME,
            ANANAS_ARCHIVE_SHA256,
        )
        build_inputs = [
            staged_archive if path == archives[0] else path for path in build_inputs
        ]
    if "rosetta" in build_features:
        archives = [path for path in build_inputs if path.name == ROSETTA_TOOLS_ARCHIVE_NAME]
        if len(archives) != 1:
            raise SystemExit(
                "--build-feature rosetta requires exactly one verified --build-input %s"
                % ROSETTA_TOOLS_ARCHIVE_NAME
            )
        _validate_rosetta_tools_archive(archives[0], repo_root)
        if not args.build_cache:
            raise SystemExit(
                "--build-feature rosetta requires --build-cache so the verified archive "
                "is the file consumed by the definition"
            )
        archive_sha256 = _sha256_file(archives[0])
        staged_archive = _stage_verified_build_input(
            archives[0],
            _expand(args.build_cache) / "sources" / "rosetta" / ROSETTA_TOOLS_ARCHIVE_NAME,
            archive_sha256,
        )
        build_inputs = [
            staged_archive if path == archives[0] else path for path in build_inputs
        ]
    fakeroot_bin = _expand(args.fakeroot_bin) if args.fakeroot_bin else None
    if fakeroot_bin and (not fakeroot_bin.is_file() or not os.access(fakeroot_bin, os.X_OK)):
        raise SystemExit("fakeroot executable is missing or not executable: %s" % fakeroot_bin)
    runner_env_defaults = dict(
        _parse_assignment(raw, "--runner-env-default") for raw in args.runner_env_default
    )
    runner_env_prepends = [
        _parse_assignment(raw, "--runner-prepend-env") for raw in args.runner_prepend_env
    ]
    def write_runner_if_requested():
        if not args.write_runner:
            return
        _write_python_runner(
            _expand(args.write_runner),
            apptainer=apptainer,
            runner_backend=args.runner_backend,
            docker_bin=docker_bin or "docker",
            docker_image=args.runner_docker_image,
            sif_path=out_path,
            container_repo_root=args.container_repo_root,
            binds=args.runner_bind,
            env_defaults=runner_env_defaults,
            env_prepends=runner_env_prepends,
            container_python=args.container_python,
            use_nv=not args.runner_no_nv,
        )

    state_root = (
        _expand(args.state_root)
        if args.state_root
        else out_path.parent.parent / ".gh_container_setup" / args.name
    )
    state_root.mkdir(parents=True, exist_ok=True)
    state_path = state_root / "state.json"
    state = _json_load(state_path) or {"completed_steps": [], "updated_at": None}
    for identity_key, fingerprint_key in (
        ("build_identity", "build_fingerprint"),
        ("pending_build_identity", "pending_build_fingerprint"),
    ):
        recorded_identity = state.get(identity_key)
        recorded_fingerprint = state.get(fingerprint_key)
        if recorded_identity is None and recorded_fingerprint is None:
            continue
        if recorded_identity is None or not isinstance(recorded_fingerprint, str):
            raise SystemExit(
                "Container build state has an incomplete %s/%s pair"
                % (identity_key, fingerprint_key)
            )
        _validate_build_contract_schema(recorded_identity)
        if _build_contract_fingerprint(recorded_identity) != recorded_fingerprint:
            raise SystemExit(
                "Container build state contract fingerprint mismatch: %s"
                % state_path
            )

    step = "build"
    if args.skip_build:
        if not out_path.is_file():
            raise SystemExit("--skip-build image does not exist: %s" % out_path)
        metadata_path = pathlib.Path(str(out_path) + ".build.json")
        if not metadata_path.is_file():
            raise SystemExit(
                "--skip-build requires existing schema-v5 image build metadata: %s"
                % metadata_path
            )
        try:
            existing_metadata = _json_load(metadata_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit("Invalid image build metadata %s: %s" % (metadata_path, exc))
        if not isinstance(existing_metadata, dict):
            raise SystemExit("Invalid image build metadata: %s" % metadata_path)
        existing_contract = existing_metadata.get("build_contract")
        existing_fingerprint = existing_metadata.get("build_fingerprint")
        _validate_recorded_build_contract(
            existing_metadata,
            expected_contract=existing_contract,
            expected_fingerprint=existing_fingerprint,
        )
        _validate_build_contract_schema(existing_contract, build_features)
        if args.builder == "docker":
            docker_builder_identity = _docker_builder_identity(
                docker_bin, args.docker_builder_image
            )
            docker_builder_execution_image = docker_builder_identity[
                "execution_image"
            ]
        _inspect_image(
            builder=args.builder,
            apptainer=apptainer,
            docker_bin=docker_bin,
            docker_image=docker_builder_execution_image,
            image_path=out_path,
        )
        _record_image_identity(
            state,
            out_path,
            build_features=build_features,
            build_contract=existing_contract,
            build_fingerprint=existing_fingerprint,
        )
        state["updated_at"] = time.time()
        _json_dump_atomic(state_path, state)
        write_runner_if_requested()
        return 0

    file_hash_cache = state.get("file_hash_cache")
    if not isinstance(file_hash_cache, dict):
        file_hash_cache = {}
    if args.builder == "docker":
        docker_builder_identity = _docker_builder_identity(
            docker_bin, args.docker_builder_image
        )
        docker_builder_execution_image = docker_builder_identity["execution_image"]
    tracked_files = _tracked_build_context_files(repo_root)
    context_fingerprint = _build_context_fingerprint(
        repo_root, tracked_files, file_hash_cache
    )
    apptainer_identity = (
        _executable_identity(apptainer, file_hash_cache)
        if needs_host_apptainer
        else None
    )
    build_identity = {
        "schema_version": 5,
        "context_sha256": context_fingerprint,
        "definition_sha256": _cached_sha256(def_file, file_hash_cache),
        "external_inputs": [
            {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": _cached_sha256(path, file_hash_cache),
            }
            for path in build_inputs
        ],
        "build_features": list(build_features),
        "accepted_licenses": sorted(
            BUILD_FEATURE_LICENSES[feature]
            for feature in build_features
            if feature in BUILD_FEATURE_LICENSES
        ),
        "builder": args.builder,
        "docker_base_source": args.docker_base_source,
        "docker_base_image": args.docker_base_image,
        "docker_network": args.docker_network if args.builder == "docker" else None,
        "docker_builder": docker_builder_identity,
        "apptainer_binary": apptainer_identity,
        "apptainer_version": (
            apptainer_identity["version"] if apptainer_identity else None
        ),
        "fakeroot_sha256": (
            _cached_sha256(fakeroot_bin, file_hash_cache) if fakeroot_bin else None
        ),
    }
    build_fingerprint = _build_contract_fingerprint(build_identity)
    fingerprint_matches = state.get("build_fingerprint") == build_fingerprint
    state["file_hash_cache"] = file_hash_cache

    if (
        step in state.get("completed_steps", [])
        and out_path.exists()
        and fingerprint_matches
        and not args.force
    ):
        _record_image_identity(
            state,
            out_path,
            build_features=build_features,
            build_contract=build_identity,
            build_fingerprint=build_fingerprint,
        )
        state["updated_at"] = time.time()
        _json_dump_atomic(state_path, state)
        print("Already built: %s" % out_path)
        _inspect_image(
            builder=args.builder,
            apptainer=apptainer,
            docker_bin=docker_bin,
            docker_image=docker_builder_execution_image,
            image_path=out_path,
        )
        write_runner_if_requested()
        return 0

    if out_path.exists() and not args.force:
        if state.get("build_fingerprint"):
            print("Build inputs changed; rebuilding stale image: %s" % out_path)
        else:
            print("Image has no verifiable build fingerprint; rebuilding: %s" % out_path)

    state["pending_build_fingerprint"] = build_fingerprint
    state["pending_build_identity"] = build_identity
    state["updated_at"] = time.time()
    _json_dump_atomic(state_path, state)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError:
            pass

    with tempfile.TemporaryDirectory(prefix="gh_apptainer_ctx_", dir=str(state_root)) as stage_dir:
        stage_repo = _stage_build_context(repo_root, stage_dir, tracked_files)
        if args.builder == "docker":
            _build_with_docker(
                docker_bin=docker_bin,
                docker_image=docker_builder_execution_image,
                docker_base_image=args.docker_base_image,
                docker_base_source=args.docker_base_source,
                docker_network=args.docker_network,
                build_cache=args.build_cache,
                stage_repo=stage_repo,
                tmp_path=tmp_path,
                def_file=def_file,
                repo_root=repo_root,
                build_features=build_features,
            )
        else:
            build_arg_file = _write_build_arg_file(
                pathlib.Path(stage_dir) / "build-args.env", build_features
            )
            cmd = [apptainer, "build"]
            cmd += [
                "--build-arg-file",
                str(build_arg_file),
            ]
            if args.fakeroot:
                cmd.append("--fakeroot")
            if args.remote:
                cmd.append("--remote")
            if args.build_cache:
                build_cache = _expand(args.build_cache)
                (build_cache / "pip").mkdir(parents=True, exist_ok=True)
                (build_cache / "mamba" / "pkgs").mkdir(parents=True, exist_ok=True)
                (build_cache / "apptainer").mkdir(parents=True, exist_ok=True)
                cmd += ["--bind", "%s:/cache" % build_cache]
            cmd += [str(tmp_path), str(def_file)]
            build_env = os.environ.copy()
            path_prefixes = [str(pathlib.Path(apptainer).resolve().parent)]
            if fakeroot_bin:
                path_prefixes.append(str(fakeroot_bin.parent))
            build_env["PATH"] = os.pathsep.join(
                path_prefixes + ([build_env["PATH"]] if build_env.get("PATH") else [])
            )
            if args.build_cache:
                build_env["APPTAINER_CACHEDIR"] = str(build_cache / "apptainer")
            _run(cmd, cwd=str(stage_repo), env=build_env)
    os.replace(str(tmp_path), str(out_path))
    _inspect_image(
        builder=args.builder,
        apptainer=apptainer,
        docker_bin=docker_bin,
        docker_image=docker_builder_execution_image,
        image_path=out_path,
    )
    _record_image_identity(
        state,
        out_path,
        build_features=build_features,
        build_contract=build_identity,
        build_fingerprint=build_fingerprint,
        replace_provenance=True,
    )

    state.setdefault("completed_steps", []).append(step)
    state["updated_at"] = time.time()
    state["def_file"] = str(def_file)
    state["output"] = str(out_path)
    state["apptainer"] = str(apptainer)
    state["builder"] = args.builder
    state["build_fingerprint"] = build_fingerprint
    state["build_identity"] = build_identity
    state.pop("pending_build_fingerprint", None)
    state.pop("pending_build_identity", None)
    if args.build_cache:
        state["build_cache"] = str(_expand(args.build_cache))
    if docker_bin:
        state["docker"] = str(docker_bin)
        state["docker_builder_image"] = str(args.docker_builder_image)
    _json_dump_atomic(state_path, state)
    print("Built: %s" % out_path)
    write_runner_if_requested()
    return 0


if __name__ == "__main__":
    sys.exit(main())
