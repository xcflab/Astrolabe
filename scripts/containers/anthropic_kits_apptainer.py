#!/usr/bin/env python3
"""Build the pinned Anthropic kit Dockerfiles as user-owned Apptainer SIFs.

The upstream reference release ships Dockerfiles and definitions that refer to
local Docker daemon tags.  This entry point keeps the Dockerfile commands but
uses Apptainer's OCI bootstrap directly, so the kit remains a container
runtime rather than a host Python environment.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vendor" / "uplifting-biomolecular-modeling"
DEFAULT_APPTAINER = Path(shutil.which("apptainer") or str(Path.home() / ".gh_container/envs/_gh_apptainer/bin/apptainer"))
KITS = ("af3_jax", "protenix_v2", "boltz2", "chai1", "esm_if1", "esmfold2", "opendde")


def logical_lines(path: Path) -> list[str]:
    rows: list[str] = []
    current = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not current:
            current = line
        else:
            current += " " + line.lstrip()
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        rows.append(current)
        current = ""
    if current:
        rows.append(current)
    return rows


def env_assignments(value: str) -> list[str]:
    tokens = shlex.split(value)
    if tokens and "=" not in tokens[0] and len(tokens) >= 2:
        return [f"export {tokens[0]}={shlex.quote(' '.join(tokens[1:]))}"]
    return [f"export {token}" for token in tokens if "=" in token]


def dockerfile_to_def(
    kit: str,
    base_sif: Path | None = None,
    local_wheels: list[Path] | None = None,
    local_files: list[Path] | None = None,
    local_uv: Path | None = None,
    keep_base_apt: bool = False,
    runtime_file: Path | None = None,
    install_cuda_toolkit: bool = False,
    cuda_repo: str = "debian12",
) -> str:
    dockerfile = SOURCE / kit / "environment" / "Dockerfile"
    base = None
    post: list[str] = ["set -e"]
    if base_sif:
        post.extend(["export UV_HTTP_RETRIES=20", "export UV_HTTP_TIMEOUT=600"])
    if runtime_file:
        digest = hashlib.sha256(runtime_file.read_bytes()).hexdigest()
        post.extend([
            "cd /",
            "_gh_run_rc=0",
            (
                f"(cp /opt/anthropic-assets/{runtime_file.name} /tmp/python.tar.gz && "
                f"echo \"{digest}  /tmp/python.tar.gz\" | sha256sum -c - && "
                "tar -xzf /tmp/python.tar.gz -C /tmp && "
                "cp -a /tmp/python/. /usr/local/ && "
                "rm -rf /tmp/python /tmp/python.tar.gz && "
                "ln -sf /usr/local/bin/python3 /usr/local/bin/python && "
                "python -V) || _gh_run_rc=$?"
            ),
            '[ "$_gh_run_rc" -eq 0 ] || exit "$_gh_run_rc"',
        ])
    if install_cuda_toolkit:
        post.extend([
            "cd /",
            "_gh_run_rc=0",
            (
                "(apt-get -o Acquire::Retries=20 update && "
                "apt-get -o Acquire::Retries=20 install -y --no-install-recommends --no-upgrade "
                "ca-certificates curl build-essential && "
                "curl --retry 10 --retry-delay 5 --retry-connrefused -fsSL "
                "-o /tmp/cuda-keyring.deb "
                f"https://developer.download.nvidia.com/compute/cuda/repos/{cuda_repo}/x86_64/cuda-keyring_1.1-1_all.deb && "
                "dpkg -i /tmp/cuda-keyring.deb && "
                "apt-get -o Acquire::Retries=20 update && "
                "apt-get -o Acquire::Retries=20 install -y --no-install-recommends --no-upgrade "
                "cuda-compiler-13-0 cuda-libraries-dev-13-0 cuda-cudart-dev-13-0 "
                "cuda-nvtx-13-0 cuda-profiler-api-13-0 cuda-nvml-dev-13-0 && "
                "ln -sfn /usr/local/cuda-13.0 /usr/local/cuda && "
                "rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb) || _gh_run_rc=$?"
            ),
            '[ "$_gh_run_rc" -eq 0 ] || exit "$_gh_run_rc"',
        ])
    runtime_env: list[str] = []
    workdir = "/"
    copy_rows: list[tuple[Path, str]] = []
    args: dict[str, str] = {}

    for row in logical_lines(dockerfile):
        stripped = row.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("FROM "):
            base = stripped.split(None, 1)[1].strip()
        elif stripped.startswith("ARG "):
            value = stripped[4:].strip()
            name, sep, default = value.partition("=")
            args[name] = default if sep else ""
            if sep:
                post.append(f'export {name}="${{{name}:-{shlex.quote(default)}}}"')
            else:
                post.append(f'export {name}="${{{name}:-}}"')
        elif stripped.startswith("ENV "):
            assignments = env_assignments(stripped[4:].strip())
            post.extend(assignments)
            runtime_env.extend(assignments)
        elif stripped.startswith("WORKDIR "):
            workdir = stripped.split(None, 1)[1].strip()
            post.append(f"cd {shlex.quote(workdir)}")
        elif stripped.startswith("COPY "):
            tokens = shlex.split(stripped[5:].strip())
            if len(tokens) < 2:
                raise SystemExit(f"cannot parse COPY in {dockerfile}: {stripped}")
            source_tokens, destination = tokens[:-1], tokens[-1]
            if any("_jitcache" in token or "*" in token or "[" in token for token in source_tokens):
                continue
            if len(source_tokens) != 1:
                raise SystemExit(f"multiple-source COPY is unsupported: {dockerfile}: {stripped}")
            source = (SOURCE / source_tokens[0]).resolve()
            if not source.exists():
                raise SystemExit(f"COPY source is missing: {source}")
            destination = destination.replace("/tmp/stock", "/opt/anthropic-assets/stock")
            destination = destination.replace("/tmp/requirements.lock", "/opt/anthropic-assets/requirements.lock")
            copy_rows.append((source, destination))
        elif stripped.startswith("RUN "):
            if base_sif and not keep_base_apt and any(
                marker in stripped
                for marker in ("apt-get", "cuda-cudart-dev", "cuda-crt-dev")
            ):
                continue
            post.append(f"cd {shlex.quote(workdir)}")
            command = stripped[4:].strip()
            command = command.replace("/tmp/stock", "/opt/anthropic-assets/stock")
            command = command.replace("/tmp/requirements.lock", "/opt/anthropic-assets/requirements.lock")
            command = command.replace(
                "curl -fsSL",
                "curl --retry 10 --retry-delay 5 --retry-connrefused -fsSL",
            )
            command = command.replace(
                "apt-get update",
                "apt-get -o Acquire::Retries=20 update",
            )
            command = command.replace(
                "apt-get install -y",
                "apt-get -o Acquire::Retries=20 install -y",
            )
            command = command.replace(
                "compute/cuda/repos/debian12/x86_64",
                f"compute/cuda/repos/{cuda_repo}/x86_64",
            )
            if base_sif:
                command = command.replace("uv venv --python", "uv venv --clear --python")
                command = command.replace(
                    "python3 -m pip install --no-cache-dir uv",
                    "/usr/local/bin/python3 -m pip install --no-cache-dir uv",
                )
                uv_install = r"/usr/local/bin/python3 -m pip install --no-cache-dir uv(?:==[^\s&]+)?"
                if local_uv and re.search(uv_install, command):
                    command = re.sub(
                        uv_install,
                        "install -m 0755 /opt/anthropic-assets/uv /usr/local/bin/uv",
                        command,
                        count=1,
                    )
                else:
                    command = re.sub(
                        uv_install,
                        lambda match: f"command -v uv >/dev/null 2>&1 || {match.group(0)}",
                        command,
                        count=1,
                    )
            command = command.replace(
                "python -m pip install --no-cache-dir",
                "python -m pip install --retries 20 --timeout 600 "
                "--trusted-host pypi.org --trusted-host files.pythonhosted.org --no-cache-dir",
            )
            for asset in local_files or []:
                if re.search(r"\b(?:curl|wget)\b.*?\s-[oO]\s+/tmp/python\.tar\.gz", command):
                    command = re.sub(
                        r"\b(?:curl|wget)\b.*?\s-[oO]\s+/tmp/python\.tar\.gz\s+\"[^\"]+\"",
                        f"cp /opt/anthropic-assets/{asset.name} /tmp/python.tar.gz",
                        command,
                        count=1,
                    )
                    break
            local_installs = [
                f"python -m pip install --no-cache-dir --no-deps /opt/anthropic-assets/{wheel.name}"
                for wheel in local_wheels or []
            ]
            if local_installs and "pip install" in command:
                command = " && ".join(local_installs + [command])
            post.append("_gh_run_rc=0")
            post.append(f"({command}) || _gh_run_rc=$?")
            post.append('[ "$_gh_run_rc" -eq 0 ] || exit "$_gh_run_rc"')
        elif stripped.startswith(("CMD ", "ENTRYPOINT ", "EXPOSE ")):
            continue
        else:
            raise SystemExit(f"unsupported Dockerfile directive in {dockerfile}: {stripped}")

    if not base:
        raise SystemExit(f"Dockerfile has no FROM: {dockerfile}")

    files = ["%files"]
    for source, destination in copy_rows:
        files.append(f"    {source} {destination}")
    for wheel in local_wheels or []:
        files.append(f"    {wheel} /opt/anthropic-assets/{wheel.name}")
    for asset in local_files or []:
        files.append(f"    {asset} /opt/anthropic-assets/{asset.name}")
    if runtime_file:
        files.append(f"    {runtime_file} /opt/anthropic-assets/{runtime_file.name}")
    if local_uv:
        files.append(f"    {local_uv} /opt/anthropic-assets/uv")

    environment = ["%environment"]
    environment.extend(f"    {line}" for line in runtime_env)
    runscript = [
        "%runscript",
        f"    cd /kit/{kit}",
        '    exec bash run.sh "$@"',
    ]
    header = (
        ["Bootstrap: localimage", f"From: {base_sif}", ""]
        if base_sif
        else ["Bootstrap: docker", f"From: {base}", ""]
    )
    return "\n".join(header + files + ["", "%post"] + [f"    {line}" for line in post] + ["", *environment, "", *runscript, ""])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", action="append", choices=KITS, required=True)
    parser.add_argument("--output-root", default=str(Path.home() / ".gh_anthropic_kits/f4f62fa6592ae4938d49b1757bea0cfeff9f468e"))
    parser.add_argument("--apptainer-bin", default=os.environ.get("APPTAINER_BIN", str(DEFAULT_APPTAINER)))
    parser.add_argument("--base-sif", help="Use an existing SIF as the localimage base; Docker system RUNs are skipped unless --keep-base-apt is set")
    parser.add_argument(
        "--local-wheel",
        action="append",
        default=[],
        help="Copy a wheel into the build and preinstall it inside %%post before locked requirements.",
    )
    parser.add_argument(
        "--local-file",
        action="append",
        default=[],
        help="Copy a pinned build input into the image and use it instead of its Dockerfile download.",
    )
    parser.add_argument(
        "--local-uv",
        help="Copy a compatible uv executable into the image instead of downloading uv during a localimage build.",
    )
    parser.add_argument(
        "--keep-base-apt",
        action="store_true",
        help="Keep Dockerfile apt/toolchain RUN steps when using --base-sif.",
    )
    parser.add_argument(
        "--runtime-file",
        help="Copy and install a pinned CPython tarball inside a localimage build before the kit steps.",
    )
    parser.add_argument(
        "--install-cuda-toolkit",
        action="store_true",
        help="Install the CUDA 13 compiler/development packages inside the image before kit steps.",
    )
    parser.add_argument(
        "--cuda-repo",
        default="debian12",
        choices=("debian12", "ubuntu2204", "ubuntu2404"),
        help="NVIDIA CUDA repository family used by generated toolkit steps.",
    )
    parser.add_argument("--jobs", default="0", help="Build job count passed as BUILD_JOBS")
    parser.add_argument("--definitions-only", action="store_true", help="Write definitions without starting an Apptainer build")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    apptainer = Path(args.apptainer_bin).expanduser().resolve()
    if not apptainer.is_file() or not os.access(apptainer, os.X_OK):
        raise SystemExit(f"Apptainer executable is missing or not executable: {apptainer}")
    output_root = Path(args.output_root).expanduser().resolve()
    base_sif = Path(args.base_sif).expanduser().resolve() if args.base_sif else None
    if base_sif and not base_sif.is_file():
        raise SystemExit(f"base SIF is missing: {base_sif}")
    local_wheels = [Path(value).expanduser().resolve() for value in args.local_wheel]
    for wheel in local_wheels:
        if not wheel.is_file():
            raise SystemExit(f"local wheel is missing: {wheel}")
    local_files = [Path(value).expanduser().resolve() for value in args.local_file]
    for asset in local_files:
        if not asset.is_file():
            raise SystemExit(f"local file is missing: {asset}")
    local_uv = Path(args.local_uv).expanduser().resolve() if args.local_uv else None
    if local_uv and not local_uv.is_file():
        raise SystemExit(f"local uv executable is missing: {local_uv}")
    runtime_file = Path(args.runtime_file).expanduser().resolve() if args.runtime_file else None
    if runtime_file and not runtime_file.is_file():
        raise SystemExit(f"runtime file is missing: {runtime_file}")
    defs = output_root / "defs"
    images = output_root / "images"
    cache = output_root / "apptainer-cache"
    temp = output_root / "apptainer-tmp"
    for directory in (defs, images, cache, temp):
        directory.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["APPTAINER_CACHEDIR"] = str(cache)
    env["APPTAINER_TMPDIR"] = str(temp)
    # The generated definition carries Docker ARG defaults as shell exports in
    # %post.  Passing --build-arg would make Apptainer reject them as unused
    # because those ARG declarations do not exist in the generated definition.
    env["BUILD_JOBS"] = str(args.jobs)
    path_prefixes = [str(apptainer.parent)]
    managed_fakeroot = Path.home() / ".gh_container/tools/fakeroot/bin"
    if (managed_fakeroot / "fakeroot").is_file():
        path_prefixes.insert(0, str(managed_fakeroot))
    env["PATH"] = os.pathsep.join(path_prefixes + [env.get("PATH", "")])

    for kit in args.kit:
        definition = defs / f"{kit}.def"
        output = images / f"{kit}.sif"
        definition.write_text(
            dockerfile_to_def(
                kit,
                base_sif=base_sif,
                local_wheels=local_wheels,
                local_files=local_files,
                local_uv=local_uv,
                keep_base_apt=args.keep_base_apt,
                runtime_file=runtime_file,
                install_cuda_toolkit=args.install_cuda_toolkit,
                cuda_repo=args.cuda_repo,
            ),
            encoding="utf-8",
        )
        if args.definitions_only:
            print("WRITE", kit, "->", definition, flush=True)
            continue
        if output.exists() and not args.force:
            print(f"SKIP {kit}: {output}", flush=True)
            continue
        command = [str(apptainer), "build", "--fakeroot"]
        command.extend([str(output), str(definition)])
        print("BUILD", kit, "->", output, flush=True)
        subprocess.run(command, cwd=str(ROOT), env=env, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
