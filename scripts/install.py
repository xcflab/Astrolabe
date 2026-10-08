#!/usr/bin/env python3
"""
Server-side bootstrap for the container runtime.

Run this after logging into the target server. It builds/reuses the SIF,
    prepares selected downloadable assets, writes a declarative site profile, and
generates the GH_PY runner from that profile.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Iterable
import urllib.parse
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from astrolabe.asset_download import download_url, validate_file
SCRIPT_HELPER_DIRS = (
    REPO_ROOT / "scripts" / "installation",
    REPO_ROOT / "scripts" / "containers",
    REPO_ROOT / "scripts" / "maintenance",
)
for _helper_dir in SCRIPT_HELPER_DIRS:
    sys.path.insert(0, str(_helper_dir))

from installation.software_catalog import (
    SOFTWARE,
    print_catalog,
    resolve_software,
    selected_host_envs,
    selected_submodules,
)
from installation.asset_prefetch import (
    CHAI_COMPONENTS,
    CHAI_CONFORMERS_SHA256,
    CHAI_ESM_SHA256,
    HF_SNAPSHOT_FILE_SPECS,
    HF_SNAPSHOT_REQUIRED_FILES,
    HF_SNAPSHOT_REPOS,
    hf_snapshot_manifest_path,
    hf_runtime_snapshot_path,
)


DEFAULT_ROOT = Path.home() / ".gh_envs"
DEFAULT_CONTAINER_REPO_ROOT = "/opt/Astrolabe"
DEFAULT_CONTAINER_PYTHON = "python"
APPTAINER_BOOTSTRAP_ENV = "_gh_apptainer"
APPTAINER_RUNTIME_PACKAGES = (
    "apptainer=1.5.2",
    "squashfuse=0.6.2",
)
APPTAINER_BUILD_PACKAGES = (
    "meson=1.11.2",
    "ninja=1.13.2",
    "libcap=2.78",
)
APPTAINER_ENV_LOCK_TIMEOUT_SECONDS = 1_800.0
APPTAINER_TIER_MARKER_SCHEMA_VERSION = 2
DEFAULT_DOCKER_APPTAINER_IMAGE = "ghcr.io/apptainer/apptainer:1.4.5"
MICROMAMBA_VERSION = "2.8.1"
MICROMAMBA_BUILD = "0"
MICROMAMBA_ARCHIVE_NAME = (
    f"micromamba-{MICROMAMBA_VERSION}-{MICROMAMBA_BUILD}.tar.bz2"
)
MICROMAMBA_URL = (
    "https://api.anaconda.org/download/conda-forge/micromamba/"
    f"{MICROMAMBA_VERSION}/linux-64/{MICROMAMBA_ARCHIVE_NAME}"
)
MICROMAMBA_ARCHIVE_SHA256 = (
    "a934c3709c997feae403a27fd1e321c106d26ffa4f294800ffb11cbc9a3e8515"
)
RUNTIME_CACHE_DIRS = (
    ("HOME", "home"),
    ("MPLCONFIGDIR", "matplotlib"),
    ("NUMBA_CACHE_DIR", "numba"),
    ("PROTENIX_ROOT_DIR", "protenix"),
    ("TORCH_EXTENSIONS_DIR", "torch-extensions"),
    ("XDG_CACHE_HOME", "xdg"),
)

PYROSETTA_VERSION = "2026.3+releasequarterly.5e498f1409"
PYROSETTA_WHEEL_NAME = (
    f"pyrosetta-{PYROSETTA_VERSION}-cp311-cp311-linux_x86_64.whl"
)
PYROSETTA_UPSTREAM_SHA256 = "fd1a9a51c1d7475b448e010926ed201446a55d20f4c5340455af148914d3cffe"
PYROSETTA_WHEEL_SIZE = 1_658_972_065
PYROSETTA_MAX_NO_PROGRESS_ATTEMPTS = 12
PYROSETTA_ALLOWED_SHA256 = frozenset({PYROSETTA_UPSTREAM_SHA256})
PYROSETTA_URL = (
    "https://west.rosettacommons.org/pyrosetta/quarterly/release/"
    + PYROSETTA_WHEEL_NAME
)
ANANAS_VERSION = "1.1"
ANANAS_ARCHIVE_NAME = "ananas-linux-1.1.zip"
ANANAS_URL = (
    "https://files.inria.fr/NanoDFiles/Website/Software/AnAnaS/1.1/"
    "Linux/ananas-linux.zip"
)
ANANAS_ARCHIVE_SIZE = 957_099
ANANAS_ARCHIVE_SHA256 = (
    "279a4163e42343599633db71ac2ecd5f20795fe80510ff5f2ec542de35701745"
)
ANANAS_BINARY_SHA256 = (
    "79f470f084f0e98c4e067104e32d879d41d68fc4935ced5c49102bb8666177ab"
)
ROSETTA_TOOLS_COMMIT = "80a0635615099e1b918474a63acba7b1de6fd107"
ROSETTA_TOOLS_REPO_URL = "https://github.com/RosettaCommons/rosetta.git"
ROSETTA_TOOLS_MANIFEST = REPO_ROOT / "container" / "rosetta-tools.manifest.tsv"
ROSETTA_TOOLS_ARCHIVE_NAME = f"rosetta-tools-{ROSETTA_TOOLS_COMMIT}.tar.gz"

DOWNLOAD_PROGRESS_DEADLINE_SECONDS = 6 * 60 * 60
DOWNLOAD_MAX_NO_PROGRESS_ATTEMPTS = 3
DOWNLOAD_REQUEST_TIMEOUT_SECONDS = 30.0

AF2_PARAM_SPECS = {
    "params_model_1.npz": (373_069_562, "f95e453e6a290ddf317ba1c9698d53fa110cf007ea979b0eae43e6ad38b4e364"),
    "params_model_1_ptm.npz": (373_103_340, "5e564f79af5bcd54ccef6e2a6bb0ff01015d01650ebc41d4575e35f0de9ecc84"),
    "params_model_1_multimer_v3.npz": (373_043_148, "611da8fc7478928f68de12e8b226260ef1f4ce62bcc29b008572e52f4f212959"),
    "params_model_2.npz": (373_069_562, "4da37ed3b71f08d935f913dfab0133977cd0d76ad37cd9333ff294d7fae43f87"),
    "params_model_2_ptm.npz": (373_103_340, "23645d9a82c4af2ed54cd48a7b3c1c2575dc6aa9fe931adb4d7203ca5f0dc398"),
    "params_model_2_multimer_v3.npz": (373_043_148, "51362b0844382ae0f5720c59b81dd13a43eea40fbf9995dd2573bdab88865378"),
    "params_model_3.npz": (371_712_506, "3cc4584dc0b0d1bdd1f7af14bd0692d00df3aa0e738d0d1b314821ad7eee4ffe"),
    "params_model_3_ptm.npz": (371_746_284, "b53a724294cdd46a45d530b306bced8e429ffee74a4f9a2a897551b30966c415"),
    "params_model_3_multimer_v3.npz": (373_043_148, "46d9bcad288edc7ad5a6362ee8e5f84307a69712e00e6c36b1ef9daf96ebc9ce"),
    "params_model_4.npz": (371_712_506, "b0a78b7e18661abc6097a905f48505004e6e3993c5dc41fb6820acf56e9c1d12"),
    "params_model_4_ptm.npz": (371_746_284, "07d0e285ad1406ce3d91a2f6633a8ce9ee99155a53d24ee7783f8fd1e89f5d3a"),
    "params_model_4_multimer_v3.npz": (373_043_148, "59bdabd2d69c07fe26b37882d544acbd1b9f89f196828f4220da49e0610b572c"),
    "params_model_5.npz": (371_712_506, "5d261ec6e18ba87784541f5ce113570018b3338df61ae577661b6232358d47e7"),
    "params_model_5_ptm.npz": (371_746_284, "2f8d0bd193e7d6e837958ba92acdc3f02b23b61bf841fa043fc076ee0e1996a5"),
    "params_model_5_multimer_v3.npz": (373_043_148, "917742be5a105d6b80f13f1f13f20459f27ec3fdcd34ea088b359f4502d6177f"),
}
AF2_PARAM_FILES = tuple(AF2_PARAM_SPECS)
AF2_SENTINEL = "params_model_3_multimer_v3.npz"
AF2_SENTINEL_SHA256 = AF2_PARAM_SPECS[AF2_SENTINEL][1]
AF2_PARAMS_URL = "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
AF2_PARAMS_ARCHIVE_SIZE = 5_587_968_000
LIGANDMPNN_CKPTS = (
    "global_label_membrane_mpnn_v_48_020.pt",
    "ligandmpnn_v_32_020_25.pt",
    "per_residue_label_membrane_mpnn_v_48_020.pt",
    "proteinmpnn_v_48_020.pt",
    "solublempnn_v_48_020.pt",
)
LIGANDMPNN_URL_PREFIX = "https://files.ipd.uw.edu/pub/ligandmpnn"
LIGANDMPNN_ASSET_URLS = {
    name: f"{LIGANDMPNN_URL_PREFIX}/{name}" for name in LIGANDMPNN_CKPTS
}
LIGANDMPNN_SHA256 = {
    "global_label_membrane_mpnn_v_48_020.pt": "89ef0abddbfb956c4c7c02dcba6523c0b5152733055f7509292365eaecc38b21",
    "ligandmpnn_v_32_020_25.pt": "42caaa5cdb380867d2b31c30de4ab53ea89171279b3a0732ddf94bc4d3cb6981",
    "per_residue_label_membrane_mpnn_v_48_020.pt": "1e62e193bee12f1b64fff87ae3ca9da95b0f60b15839ae3cbe48e4983ef055ba",
    "proteinmpnn_v_48_020.pt": "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd",
    "solublempnn_v_48_020.pt": "7af52d090172c230c7f0e9d21e02203f6b3a38b16db58d3c7a3960e0a9a6e31a",
}
PROTENIX_ASSET_URLS = {
    "release_model/protenix-v2.pt": (
        "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix-v2.pt"
    ),
    "release_model/protenix_base_constraint_v0.5.0.pt": (
        "https://protenix.tos-cn-beijing.volces.com/checkpoint/"
        "protenix_base_constraint_v0.5.0.pt"
    ),
    "release_data/components.v20240608.cif": (
        "https://af3-dev.tos-cn-beijing.volces.com/release_data/components.v20240608.cif"
    ),
    "release_data/components.v20240608.cif.rdkit_mol.pkl": (
        "https://af3-dev.tos-cn-beijing.volces.com/release_data/"
        "components.v20240608.cif.rdkit_mol.pkl"
    ),
}
PROTENIX_V2_SHA256 = "8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599"
PROTENIX_ASSET_SHA256 = {
    "release_model/protenix-v2.pt": PROTENIX_V2_SHA256,
    "release_model/protenix_base_constraint_v0.5.0.pt": "5358025b20b2212853ad75579be04387859557915f398a1d60f6a1a9a0c8c887",
    "release_data/components.v20240608.cif": "7240b17369ccfbbcc86e2d02dc8c9db59f46c32e0420f58889c6c121c60bfef0",
    "release_data/components.v20240608.cif.rdkit_mol.pkl": "2d6caced2d26c62015115a1d0a50f4106755a300e0c2a2d2b5c101c9038dcbcd",
}
BOLTZ_ASSET_SHA256 = {
    "ccd.pkl": "2d3b2f03a3c5665944adba51e33263511e51b21c9cd05d902f9c4b7c1e58d2f4",
    "boltz1_conf.ckpt": "fea245d912c570ec117b2277c2719f312a6fc109c07b6f6ef741690ee775c2f5",
}
RHOFOLD_REVISION = str(SOFTWARE["rhofold"].revision)
RHOFOLD_CHECKPOINT_URL = (
    f"https://huggingface.co/cuhkaih/rhofold/resolve/{RHOFOLD_REVISION}/rhofold_pretrained_params.pt"
)
RHOFOLD_CHECKPOINT_SHA256 = (
    "3adb621978dfcd7ea0dc0edeb520d249f423d61530df85ff58a1fad2f33e5608"
)
OPENDDE_REVISION = str(SOFTWARE["opendde"].revision)
OPENDDE_ASSETS = {
    "checkpoint/opendde.pt": (
        f"https://huggingface.co/aurekaresearch/OpenDDE/resolve/{OPENDDE_REVISION}/opendde.pt",
        2_625_249_069,
        "7b826620390afad877ee2babc6a4d0df81b94d3a0be030959853d6a7da0807cc",
    ),
    "common/components.cif": (
        f"https://huggingface.co/aurekaresearch/OpenDDE/resolve/{OPENDDE_REVISION}/common/components.cif",
        490_777_362,
        "bb31ae5cf6c8bc669924313077cb4231ee5ffefd3a20118cd14f3ec89f8bb6a5",
    ),
    "common/components.cif.rdkit_mol.pkl": (
        f"https://huggingface.co/aurekaresearch/OpenDDE/resolve/{OPENDDE_REVISION}/common/components.cif.rdkit_mol.pkl",
        142_498_117,
        "d1cfb71f5993a3ebea7c47877022d7f597bbfbaf86e28a4770e957da6c50cd35",
    ),
    "common/release_date_cache.json": (
        f"https://huggingface.co/aurekaresearch/OpenDDE/resolve/{OPENDDE_REVISION}/common/release_date_cache.json",
        12_754_898,
        "8b1ef12ddc01a0d5eb2d388c77ded91aa906eebce7440726c57b6f8d1a3ec142",
    ),
    "common/obsolete_to_successor.json": (
        f"https://huggingface.co/aurekaresearch/OpenDDE/resolve/{OPENDDE_REVISION}/common/obsolete_to_successor.json",
        86_882,
        "2bc08348d0efba438c109bb27be6fa25b611d371c60b8a8da3de387a4a0698ad",
    ),
}
RAPIDOCK_ASSETS = {
    "rapidock_local.pt": (
        "https://zenodo.org/records/14193621/files/rapidock_local.pt?download=1",
        56_648_773,
        "d0f1ebe268354624c345f8730e765e1b21c016f946fffb637461236204919693",
    ),
    "rapidock_global.pt": (
        "https://zenodo.org/records/14193621/files/rapidock_global.pt?download=1",
        56_648_773,
        "a5dfa8f0b20642e26b276d8fd3e7ac87377b5c5150b15b7afcabf9cd8558e0b5",
    ),
}
RF3_ASSET = (
    "https://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt",
    3_038_876_446,
    "364ef592fd8042a9cf4176d045015190f8322f961ccca38d891b20ca578d3bb0",
)
RFD3_ASSET = (
    "https://files.ipd.uw.edu/pub/rfd3/rfd3_foundry_2025_12_01_remapped.ckpt",
    2_690_316_669,
    "9b3f85923e0d51e9453e15cdd2f8c666e7ce096a60577f57d11bbc54ae6d67c1",
)
SURFPRO_CATH42_ASSET = (
    "https://drive.usercontent.google.com/download?"
    "id=1xl_3xUrzjZCYcl5C5eUKjJd5VoRG28NQ&export=download&confirm=t",
    73_345_757,
    "6768f197dd2ef2b8b755d1214c3627dc821b445458c9fc703a5e056f6a34cc53",
)
SURFPRO_PRETRAIN_ASSET = (
    "https://drive.usercontent.google.com/download?"
    "id=1F2M4yMVPlkKY5KXpp6GwJvopl3Qq8gRo&export=download&confirm=t",
    73_345_501,
    "3f037093eaf5906338ac489ab04208c03b64b301c7c9f4d4497d7818a6d684aa",
)
SURFPRO_ASSETS = {
    "cath42": SURFPRO_CATH42_ASSET,
    "pretrain": SURFPRO_PRETRAIN_ASSET,
}
CONTAINER_PREFETCH_SOFTWARE = frozenset(
    {"boltz", "chai1", "esm2", "esmif", "esmfold", "esm3"}
)
HOST_PREFETCH_SOFTWARE = frozenset({"dplm", "esmfold2"})
GATED_HF_SOFTWARE = frozenset()
DOWNLOADABLE_ASSET_SOFTWARE = frozenset(
    {
        "alphafold2",
        "ligandmpnn",
        "protenix",
        "boltz",
        "chai1",
        "esm2",
        "esmif",
        "esmfold",
        "esm3",
        "dplm",
        "esmfold2",
        "rhofold",
        "opendde",
        "rapidock",
        "rf3",
        "rfd3",
        "surfpro",
    }
)
HF_MODEL_REPOS = {
    name: repo_id for name, (repo_id, _revision) in HF_SNAPSHOT_REPOS.items()
}
SOFTWARE_MANIFEST_LOCK_TIMEOUT_SECONDS = 5.0
CONTAINER_IMPORT_SMOKE_MODULES = (
    "alphafold",
    "alphafold3.common.folding_input",
    "boltz",
    "chai_lab",
    "esm2",
    "esm3",
    "jax",
    "ml_collections",
    "protenix",
    "pytorch_lightning",
    "rhofold",
    "tensorflow",
    "torch",
)
SOFTWARE_IMPORT_SMOKE_MODULES = {
    "alphafold3": ("alphafold3.common.folding_input", "jax"),
    "alphafold2": ("alphafold", "tensorflow"),
    "protenix": ("protenix", "ml_collections", "pytorch_lightning"),
    "boltz": ("boltz", "pytorch_lightning"),
    "chai1": ("chai_lab", "torch"),
    "esm2": ("esm2", "torch"),
    "esmif": ("esm2.inverse_folding", "torch_geometric", "torch"),
    "esmfold": ("astrolabe.evaluators.esmfold", "esm2", "torch"),
    "esm3": ("esm3", "torch"),
    "rhofold": ("rhofold", "torch"),
    "pyrosetta": ("pyrosetta",),
    "autodock-vina": ("meeko", "vina"),
}
RUNTIME_SUBMODULE_PATHS = (
    "vendor/fakeroot",
    "vendor/abseil-cpp-source",
    "vendor/pybind11-source",
    "vendor/pybind11-abseil-source",
    "vendor/libcifpp-source",
    "vendor/dssp-source",
    "vendor/libmcfp-source",
    "vendor/eigen-source",
    "vendor/boost-regex-source",
    "vendor/alphafold3",
    "vendor/alphafold2",
    "vendor/protenix",
    "vendor/rhofold",
    "vendor/chai_lab",
    "vendor/boltz",
    "vendor/esm2",
    "vendor/openfold_esm2",
    "vendor/esm3",
)
NEEDED_SOFTWARE_SUBMODULE_PATHS = RUNTIME_SUBMODULE_PATHS + (
    "vendor/dplm",
    "vendor/na_mpnn",
    "vendor/rpxdock",
    "vendor/willutil",
)
LEGACY_FULL_SOFTWARE = (
    "alphafold3",
    "alphafold2",
    "ligandmpnn",
    "protenix",
    "dplm",
    "nampnn",
    "rpxdock",
    "rosetta",
    "haddock3",
    "autodock-gpu",
    "ananas",
    "usalign",
    "dalphaball",
    "pyrosetta",
)


def _expand(raw: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(raw)))).resolve()


def _expand_arg_or_env(raw: str | None, env_key: str) -> tuple[Path | None, bool]:
    if raw:
        return _expand(raw), False
    env_value = os.environ.get(env_key)
    if env_value:
        return _expand(env_value), True
    return None, False


def _expand_python_arg_or_env(
    raw: str | None, env_key: str
) -> tuple[Path | None, bool]:
    value = raw or os.environ.get(env_key)
    if not value:
        return None, False
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.abspath(expanded)), raw is None


PROTENIX_ENV_KEYS = (
    "GH_WEIGHT_PROTENIX",
    "GH_WEIGHT_PROTENIX_CONSTRAINT",
    "GH_PROTENIX_CCD",
    "GH_PROTENIX_CCD_RDKIT",
)
PROTENIX_ENV_BY_REL_PATH = dict(zip(PROTENIX_ASSET_URLS, PROTENIX_ENV_KEYS))


def _protenix_asset_root(path: Path) -> Path:
    if path.parent.name in {"release_model", "release_data"}:
        return path.parent.parent
    return path


def _protenix_root_from_arg_or_env(raw: str | None) -> tuple[Path | None, bool, dict[str, Path]]:
    if raw:
        return _expand(raw), False, {}

    env_paths: dict[str, Path] = {}
    for env_key in PROTENIX_ENV_KEYS:
        env_value = os.environ.get(env_key)
        if not env_value:
            continue
        env_paths[env_key] = _expand(env_value)
    if not env_paths:
        return None, False, {}

    roots = {_protenix_asset_root(path) for path in env_paths.values()}
    if len(roots) != 1:
        details = ", ".join(f"{key}={path}" for key, path in sorted(env_paths.items()))
        raise SystemExit(
            "Protenix env vars must point under one asset root; got: %s" % details
        )
    return next(iter(roots)), True, env_paths


def _container_path_under_root(host_root: Path, host_path: Path, container_root: str) -> str:
    try:
        rel_path = host_path.relative_to(host_root)
    except ValueError:
        return container_root
    if str(rel_path) == ".":
        return container_root
    return str(Path(container_root) / rel_path)


def _quote_cmd(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _run(cmd: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    print("+ " + _quote_cmd(cmd))
    if dry_run:
        return
    rc = subprocess.call(cmd, cwd=str(cwd) if cwd else None)
    if rc != 0:
        raise SystemExit(f"Command failed with exit code {rc}: {_quote_cmd(cmd)}")


def _submodule_update_command(
    *,
    submodule_set: str,
    depth: int,
    jobs: int,
    extra_paths: Iterable[str] = (),
    selected_only: bool = False,
) -> list[str]:
    cmd = [
        "git",
        "submodule",
        "update",
        "--init",
        "--recursive",
        "--jobs",
        str(jobs),
    ]
    if depth > 0:
        cmd.extend(["--depth", str(depth)])
    if selected_only:
        pass
    elif submodule_set == "runtime":
        cmd.extend(RUNTIME_SUBMODULE_PATHS)
    elif submodule_set == "needed_softwares":
        cmd.extend(NEEDED_SOFTWARE_SUBMODULE_PATHS)
    elif submodule_set != "all":
        raise SystemExit(f"Unknown submodule set: {submodule_set}")
    if submodule_set != "all":
        for path in extra_paths:
            if path not in cmd:
                cmd.append(path)
    return cmd


def _prepare_source_archive(*, dry_run: bool) -> None:
    """Give a released source archive the index needed by existing build tools."""
    if (REPO_ROOT / ".git").exists():
        return
    export_path = REPO_ROOT / "SOURCE_EXPORT.json"
    pins_path = REPO_ROOT / "SUBMODULE_PINS.json"
    if not export_path.is_file() or not pins_path.is_file():
        raise SystemExit(
            "Use the Astrolabe source archive (including SOURCE_EXPORT.json and "
            "SUBMODULE_PINS.json), or an initialized Git checkout."
        )
    files = json.loads(export_path.read_text())["files"]
    pins = json.loads(pins_path.read_text())["submodules"]
    print("+ initialize local source-archive index; no upstream history or remote is added")
    if dry_run:
        return
    for relative, expected in files.items():
        source = REPO_ROOT / relative
        if expected.startswith("symlink:"):
            matches = source.is_symlink() and "symlink:" + os.readlink(source) == expected
        else:
            matches = source.is_file() and _sha256_file(source) == expected
        if not matches:
            raise SystemExit(
                f"Source archive file differs from its manifest: {relative}. "
                "Initialize from an unmodified archive before making local edits."
            )
    # Publish the index only when complete, so an interrupted preparation is resumable.
    with tempfile.TemporaryDirectory(prefix=".astrolabe-index-", dir=REPO_ROOT) as stage:
        git_dir = Path(stage) / ".git"
        git = ["git", f"--git-dir={git_dir}", f"--work-tree={REPO_ROOT}"]
        subprocess.run([*git, "init", "--quiet"], check=True)
        subprocess.run(
            [*git, "add", "--force", "--", *files,
             "SOURCE_EXPORT.json", "SUBMODULE_PINS.json", "SOURCE_ARCHIVE_SCOPE.md"],
            cwd=REPO_ROOT, check=True,
        )
        entries = "".join(f"160000 {pin['commit']}\t{pin['path']}\0" for pin in pins)
        subprocess.run(
            [*git, "update-index", "-z", "--index-info"],
            input=entries, text=True, check=True,
        )
        git_dir.rename(REPO_ROOT / ".git")


def _validate_submodule_indexes(*, dry_run: bool) -> None:
    print("+ validate initialized submodule indexes")
    if dry_run:
        return
    status = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if status.returncode != 0:
        details = status.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(f"Could not inspect initialized submodules: {details}")
    for raw_line in status.stdout.decode("utf-8", errors="replace").splitlines():
        if not raw_line or raw_line[0] == "-":
            continue
        fields = raw_line.lstrip(" +-U").split()
        if len(fields) < 2:
            raise SystemExit(f"Could not parse git submodule status line: {raw_line}")
        rel_path = fields[1]
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--"],
            cwd=REPO_ROOT / rel_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if staged.returncode == 1:
            raise SystemExit(
                "Submodule index has staged changes and may be an incomplete checkout: "
                f"{rel_path}. Refusing to overwrite it; use a clean checkout/worktree."
            )
        if staged.returncode != 0:
            details = staged.stderr.decode("utf-8", errors="replace").strip()
            raise SystemExit(f"Could not validate submodule index {rel_path}: {details}")


def _apply_vendor_patches(
    *, dry_run: bool, selected_paths: Iterable[str] | None = None
) -> None:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "maintenance" / "apply_vendor_patches.py")]
    if selected_paths is not None:
        from apply_vendor_patches import VENDOR_PATCHES

        selected = set(selected_paths)
        groups = []
        for name, spec in VENDOR_PATCHES.items():
            rel_path = Path(spec["default_root"]).resolve().relative_to(REPO_ROOT).as_posix()
            if rel_path in selected:
                groups.append(name)
        if not groups:
            print("No selected submodules require repo-owned patches; skipped vendor patching.")
            return
        for name in groups:
            cmd.extend(("--only", name))
    _run(cmd, cwd=REPO_ROOT, dry_run=dry_run)


def _download_url(
    url: str,
    dst: Path,
    *,
    dry_run: bool,
    minimum_size: int = 1,
    expected_sha256: str | None = None,
    attempts: int | None = None,
) -> None:
    kwargs = {
        "dry_run": dry_run,
        "minimum_size": minimum_size,
        "expected_sha256": expected_sha256,
    }
    if dry_run:
        if attempts is not None:
            kwargs["attempts"] = attempts
        download_url(url, dst, **kwargs)
        return

    # ``attempts`` is a consecutive no-progress limit. Disconnects that retain
    # new bytes do not consume it; the wall-clock deadline remains absolute.
    max_no_progress = (
        attempts if attempts is not None else DOWNLOAD_MAX_NO_PROGRESS_ATTEMPTS
    )
    if max_no_progress < 1:
        raise ValueError("download attempts must be at least 1")
    started = time.monotonic()
    part = Path(str(dst) + ".part")
    retained_size = part.stat().st_size if part.is_file() else 0
    no_progress_attempts = 0
    last_error: RuntimeError | None = None

    while True:
        elapsed = time.monotonic() - started
        remaining = DOWNLOAD_PROGRESS_DEADLINE_SECONDS - elapsed
        if remaining <= 0:
            break
        try:
            download_url(
                url,
                dst,
                **kwargs,
                attempts=1,
                timeout_seconds=min(DOWNLOAD_REQUEST_TIMEOUT_SECONDS, remaining),
            )
            return
        except RuntimeError as exc:
            last_error = exc

        current_size = part.stat().st_size if part.is_file() else 0
        if current_size > retained_size:
            no_progress_attempts = 0
        else:
            no_progress_attempts += 1
        retained_size = current_size

        if no_progress_attempts >= max_no_progress:
            break
        if time.monotonic() - started >= DOWNLOAD_PROGRESS_DEADLINE_SECONDS:
            break
        if no_progress_attempts:
            remaining = DOWNLOAD_PROGRESS_DEADLINE_SECONDS - (
                time.monotonic() - started
            )
            if remaining > 0:
                time.sleep(min(2 ** (no_progress_attempts - 1), 30, remaining))

    if last_error is None:
        raise RuntimeError(
            f"Download deadline expired before transfer started: {url}; "
            f"deadline_seconds={DOWNLOAD_PROGRESS_DEADLINE_SECONDS}"
        )
    raise RuntimeError(
        f"Failed to download {url} within the bounded progress policy: "
        f"deadline_seconds={DOWNLOAD_PROGRESS_DEADLINE_SECONDS}, "
        f"no_progress_attempts={no_progress_attempts}/{max_no_progress}; "
        f"partial data remains at {part}: {last_error}"
    ) from last_error


def _extract_tar(
    tar_path: Path,
    dst_dir: Path,
    *,
    members: Iterable[str],
    dry_run: bool,
) -> None:
    print(f"+ extract {tar_path} -> {dst_dir}")
    if dry_run:
        return
    wanted = set(members)
    extracted: set[str] = set()
    dst_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            member_path = PurePosixPath(member.name)
            if (
                member_path.is_absolute()
                or member_path.name in {"", ".", ".."}
                or any(part in {"", ".", ".."} for part in member_path.parts)
            ):
                raise RuntimeError(f"Unsafe AF2 archive member path: {member.name!r}")
            if member.isdir():
                continue
            if str(member_path) != member.name:
                raise RuntimeError(
                    f"Non-canonical AF2 archive member path: {member.name!r}"
                )
            if not member.isfile():
                raise RuntimeError(
                    f"Unsafe non-regular AF2 archive member: {member.name!r}"
                )
            name = str(member_path)
            if name not in wanted:
                continue
            if name in extracted:
                raise RuntimeError(f"Duplicate AF2 archive member: {name}")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read AF2 archive member: {name}")
            destination = dst_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as handle:
                shutil.copyfileobj(source, handle, length=8 * 1024 * 1024)
            extracted.add(name)
    missing = sorted(wanted - extracted)
    if missing:
        raise RuntimeError(
            "AF2 archive is missing required parameter files: " + ", ".join(missing)
        )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _valid_build_data_file(path: Path | None, *, minimum_size: int) -> bool:
    return path is not None and path.is_file() and path.stat().st_size >= minimum_size


def _cache_build_data_file(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and source.samefile(target):
        return target
    tmp = target.with_suffix(target.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        os.link(source, tmp)
    except OSError:
        shutil.copy2(source, tmp)
    os.replace(tmp, target)
    return target


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pyrosetta_wheel(path: Path) -> Path:
    actual_size = path.stat().st_size
    if actual_size != PYROSETTA_WHEEL_SIZE:
        raise SystemExit(
            f"PyRosetta wheel size mismatch for {path}: {actual_size}; "
            f"expected {PYROSETTA_WHEEL_SIZE}"
        )
    actual = _sha256_file(path)
    if actual not in PYROSETTA_ALLOWED_SHA256:
        expected = ", ".join(sorted(PYROSETTA_ALLOWED_SHA256))
        raise SystemExit(
            f"PyRosetta wheel checksum mismatch for {path}: {actual}; expected one of {expected}"
        )
    return path


def _stage_pyrosetta_wheel_cache(
    *,
    root: Path,
    build_cache: Path,
    wheel: str | None,
    dry_run: bool,
) -> Path | None:
    target = build_cache / "wheels" / "pyrosetta" / PYROSETTA_WHEEL_NAME
    if dry_run:
        print(f"+ prepare PyRosetta wheel cache -> {target}")
        return target

    if target.is_file():
        try:
            return _validate_pyrosetta_wheel(target)
        except SystemExit:
            # The shared downloader will resume a truncated target or quarantine
            # a full-size checksum mismatch before retrying the exact URL.
            if wheel:
                invalid = target.with_name(f"{target.name}.invalid.{time.time_ns()}")
                os.replace(target, invalid)
                print(f"Retained invalid PyRosetta wheel at {invalid}; replacing")

    if wheel:
        source = _expand(wheel)
        if not source.is_file():
            raise SystemExit(f"PyRosetta wheel does not exist: {source}")
        _validate_pyrosetta_wheel(source)
        return _validate_pyrosetta_wheel(_cache_build_data_file(source, target))

    _download_url(
        PYROSETTA_URL,
        target,
        dry_run=False,
        minimum_size=PYROSETTA_WHEEL_SIZE,
        expected_sha256=PYROSETTA_UPSTREAM_SHA256,
        attempts=PYROSETTA_MAX_NO_PROGRESS_ATTEMPTS,
    )
    return _validate_pyrosetta_wheel(target)


def _rosetta_tools_manifest() -> tuple[tuple[str, int, str], ...]:
    rows = []
    for line_number, raw in enumerate(
        ROSETTA_TOOLS_MANIFEST.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if line_number == 1:
            if raw != "sha256\tsize_bytes\tpath":
                raise SystemExit(f"Invalid Rosetta tools manifest header: {ROSETTA_TOOLS_MANIFEST}")
            continue
        checksum, size_text, rel_path = raw.split("\t")
        if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
            raise SystemExit(
                f"Invalid Rosetta tools SHA256 on line {line_number}: {ROSETTA_TOOLS_MANIFEST}"
            )
        rows.append((checksum, int(size_text), rel_path))
    if not rows:
        raise SystemExit(f"Rosetta tools manifest is empty: {ROSETTA_TOOLS_MANIFEST}")
    return tuple(rows)


def _validate_rosetta_tools_archive(path: Path) -> Path:
    expected = {rel_path: (checksum, size) for checksum, size, rel_path in _rosetta_tools_manifest()}
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            all_members = archive.getmembers()
            unsafe = [
                member.name
                for member in all_members
                if not member.isfile()
                or PurePosixPath(member.name).is_absolute()
                or any(part in {"", ".", ".."} for part in PurePosixPath(member.name).parts)
                or str(PurePosixPath(member.name)) != member.name
            ]
            if unsafe:
                raise SystemExit(
                    f"Rosetta tools archive contains unsafe/non-regular members: {unsafe}"
                )
            names = [member.name for member in all_members]
            if len(names) != len(set(names)):
                raise SystemExit(f"Rosetta tools archive contains duplicate members: {path}")
            members = {member.name: member for member in all_members}
            if set(members) != set(expected):
                raise SystemExit(
                    f"Rosetta tools archive has non-exact file set: {path}; "
                    f"missing={sorted(set(expected) - set(members))}, "
                    f"unexpected={sorted(set(members) - set(expected))}"
                )
            for rel_path, (expected_sha256, expected_size) in expected.items():
                member = members[rel_path]
                if member.size != expected_size:
                    raise SystemExit(
                        f"Rosetta tools archive size mismatch for {rel_path}: "
                        f"{member.size}; expected {expected_size}"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise SystemExit(f"Could not read Rosetta tools archive member: {rel_path}")
                digest = hashlib.sha256()
                for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(chunk)
                if digest.hexdigest() != expected_sha256:
                    raise SystemExit(
                        f"Rosetta tools archive SHA256 mismatch for {rel_path}: "
                        f"{digest.hexdigest()}; expected {expected_sha256}"
                    )
    except (OSError, tarfile.TarError) as exc:
        raise SystemExit(f"Invalid Rosetta tools archive {path}: {exc}") from exc
    return path


def _write_rosetta_tools_archive(checkout: Path, target: Path) -> None:
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gz_handle:
                with tarfile.open(mode="w", fileobj=gz_handle) as archive:
                    for _checksum, expected_size, rel_path in _rosetta_tools_manifest():
                        source = checkout / rel_path
                        info = tarfile.TarInfo(rel_path)
                        info.size = expected_size
                        info.mode = 0o644
                        info.uid = info.gid = 0
                        info.uname = info.gname = "root"
                        info.mtime = 0
                        with source.open("rb") as source_handle:
                            archive.addfile(info, source_handle)
        _validate_rosetta_tools_archive(temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _stage_rosetta_tools_cache(*, build_cache: Path, dry_run: bool) -> Path:
    target = build_cache / "sources" / "rosetta" / ROSETTA_TOOLS_ARCHIVE_NAME
    if target.is_file():
        try:
            return _validate_rosetta_tools_archive(target)
        except SystemExit:
            if dry_run:
                raise
            invalid = target.with_name(f"{target.name}.invalid.{time.time_ns()}")
            os.replace(target, invalid)
            print(f"Retained invalid Rosetta tools archive at {invalid}; rebuilding")
    checkout = target.parent / f"checkout-{ROSETTA_TOOLS_COMMIT}"
    if dry_run:
        print(
            f"+ sparse fetch {ROSETTA_TOOLS_REPO_URL}@{ROSETTA_TOOLS_COMMIT} "
            f"({len(_rosetta_tools_manifest())} files) -> {target}"
        )
        return target
    checkout.mkdir(parents=True, exist_ok=True)
    git_metadata = checkout / ".git"
    if git_metadata.exists() or git_metadata.is_symlink():
        probe = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "--git-dir"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode != 0:
            invalid = checkout.with_name(f"{checkout.name}.invalid.{time.time_ns()}")
            os.replace(checkout, invalid)
            print(f"Retained invalid Rosetta sparse checkout at {invalid}; rebuilding")
            checkout.mkdir(parents=True)
            git_metadata = checkout / ".git"
    if not git_metadata.exists() and not git_metadata.is_symlink():
        _run(["git", "init", str(checkout)])
    remote = subprocess.run(
        ["git", "-C", str(checkout), "remote", "get-url", "origin"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if remote.returncode != 0:
        _run(["git", "-C", str(checkout), "remote", "add", "origin", ROSETTA_TOOLS_REPO_URL])
    elif remote.stdout.strip() != ROSETTA_TOOLS_REPO_URL:
        raise SystemExit(
            f"Rosetta sparse checkout origin mismatch: {remote.stdout.strip()}; "
            f"expected {ROSETTA_TOOLS_REPO_URL}"
        )
    _run(["git", "-C", str(checkout), "sparse-checkout", "init", "--no-cone"])
    sparse_paths = [rel_path for _checksum, _size, rel_path in _rosetta_tools_manifest()]
    _run(
        ["git", "-C", str(checkout), "sparse-checkout", "set", "--no-cone", *sparse_paths]
    )
    _run(
        [
            "git",
            "-C",
            str(checkout),
            "fetch",
            "--depth=1",
            "--filter=blob:none",
            "origin",
            ROSETTA_TOOLS_COMMIT,
        ]
    )
    _run(["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"])
    actual_commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != ROSETTA_TOOLS_COMMIT:
        raise SystemExit(
            f"Rosetta sparse checkout revision mismatch: {actual_commit}; "
            f"expected {ROSETTA_TOOLS_COMMIT}"
        )
    for expected_sha256, expected_size, rel_path in _rosetta_tools_manifest():
        validate_file(
            checkout / rel_path,
            minimum_size=expected_size,
            expected_sha256=expected_sha256,
            use_adjacent_stamp=False,
        )
    _write_rosetta_tools_archive(checkout, target)
    return _validate_rosetta_tools_archive(target)


def _stage_ananas_archive_cache(*, build_cache: Path, dry_run: bool) -> Path:
    target = build_cache / "sources" / "ananas" / ANANAS_ARCHIVE_NAME
    _download_url(
        ANANAS_URL,
        target,
        dry_run=dry_run,
        minimum_size=ANANAS_ARCHIVE_SIZE,
        expected_sha256=ANANAS_ARCHIVE_SHA256,
    )
    return target


def _gold_af3_converters_dir(root: Path) -> Path | None:
    python = root / "envs" / "gh" / "bin" / "python"
    if not python.is_file():
        return None
    code = (
        "import alphafold3.constants.converters as c; "
        "print(next(iter(c.__path__)))"
    )
    try:
        output = subprocess.check_output(
            [str(python), "-c", code],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return _expand(output) if output else None


def _stage_af3_build_data_cache(
    *,
    root: Path,
    build_cache: Path,
    ccd_pickle: str | None,
    chemical_component_sets_pickle: str | None,
    components_cif: str | None,
    dry_run: bool,
) -> list[Path]:
    cache_dir = build_cache / "af3-build-data"
    if dry_run:
        print(f"+ prepare AF3 build-data cache -> {cache_dir}")
        return []

    converters_dir = _gold_af3_converters_dir(root)
    ccd_source = _expand(ccd_pickle) if ccd_pickle else None
    sets_source = (
        _expand(chemical_component_sets_pickle)
        if chemical_component_sets_pickle
        else None
    )
    if ccd_source is None and converters_dir is not None:
        ccd_source = converters_dir / "ccd.pickle"
    if sets_source is None and converters_dir is not None:
        sets_source = converters_dir / "chemical_component_sets.pickle"

    ccd_target = cache_dir / "ccd.pickle"
    sets_target = cache_dir / "chemical_component_sets.pickle"
    components_target = cache_dir / "components.cif"
    if not _valid_build_data_file(ccd_target, minimum_size=100_000_000):
        if _valid_build_data_file(ccd_source, minimum_size=100_000_000):
            _cache_build_data_file(ccd_source, ccd_target)
    if not _valid_build_data_file(sets_target, minimum_size=1_000):
        if _valid_build_data_file(sets_source, minimum_size=1_000):
            _cache_build_data_file(sets_source, sets_target)

    components_source = _expand(components_cif) if components_cif else None
    if components_source is None:
        candidates = sorted(
            (root / "envs" / "gh" / "lib").glob(
                "python*/site-packages/share/libcifpp/components.cif"
            )
        )
        components_source = candidates[0] if candidates else None
    if not _valid_build_data_file(components_target, minimum_size=100_000_000):
        if _valid_build_data_file(components_source, minimum_size=100_000_000):
            _cache_build_data_file(components_source, components_target)

    staged: list[Path] = []
    if _valid_build_data_file(ccd_target, minimum_size=100_000_000) and _valid_build_data_file(
        sets_target, minimum_size=1_000
    ):
        staged.extend((ccd_target, sets_target))
    if _valid_build_data_file(components_target, minimum_size=100_000_000):
        staged.append(components_target)
    return staged


def _bootstrap_micromamba(bin_dir: Path, *, dry_run: bool) -> str:
    bin_dir.mkdir(parents=True, exist_ok=True)
    micromamba = bin_dir / "micromamba"
    if micromamba.exists():
        return str(micromamba)
    tar_path = bin_dir / MICROMAMBA_ARCHIVE_NAME
    _download_url(
        MICROMAMBA_URL,
        tar_path,
        dry_run=dry_run,
        minimum_size=5_000_000,
        expected_sha256=MICROMAMBA_ARCHIVE_SHA256,
    )
    if dry_run:
        return str(micromamba)
    with tarfile.open(tar_path, "r:bz2") as archive:
        member = next((m for m in archive.getmembers() if m.name.endswith("/micromamba")), None)
        if member is None:
            raise SystemExit("micromamba archive did not contain a micromamba executable.")
        member.name = Path(member.name).name
        archive.extract(member, bin_dir)
    micromamba.chmod(0o755)
    return str(micromamba)


def _apptainer_env_lock_path(root: Path) -> Path:
    return root / ".locks" / f"{APPTAINER_BOOTSTRAP_ENV}.lock"


def _acquire_apptainer_env_lock(lock_handle, lock_path: Path) -> None:
    deadline = time.monotonic() + APPTAINER_ENV_LOCK_TIMEOUT_SECONDS
    announced = False
    while True:
        try:
            fcntl.flock(
                lock_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
            return
        except BlockingIOError:
            if not announced:
                print(f"Waiting for managed Apptainer environment lock: {lock_path}")
                announced = True
            if time.monotonic() >= deadline:
                raise SystemExit(
                    "Timed out waiting for managed Apptainer environment lock after "
                    f"{APPTAINER_ENV_LOCK_TIMEOUT_SECONDS:g}s: {lock_path}"
                )
            time.sleep(0.2)


def _apptainer_tier_paths(
    env_root: Path,
    *,
    runtime: bool,
    build_tools: bool,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    if runtime:
        paths.extend(
            (
                env_root / "bin" / "apptainer",
                env_root / "bin" / "squashfuse",
            )
        )
    if build_tools:
        paths.extend(
            (
                env_root / "bin" / "meson",
                env_root / "bin" / "ninja",
                env_root / "include" / "sys" / "capability.h",
                env_root / "sbin" / "setcap",
                env_root / "sbin" / "getcap",
            )
        )
    return tuple(paths)


def _resolved_conda_records(env_root: Path) -> list[dict[str, object]]:
    conda_meta = env_root / "conda-meta"
    if not conda_meta.is_dir():
        return []
    records: list[dict[str, object]] = []
    for record_path in sorted(conda_meta.glob("*.json")):
        try:
            payload = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            records.append({})
            continue
        if not isinstance(payload, dict):
            records.append({})
            continue
        records.append(
            {
                key: payload.get(key)
                for key in (
                    "name",
                    "version",
                    "build",
                    "build_number",
                    "channel",
                    "subdir",
                    "sha256",
                    "md5",
                    "url",
                )
            }
        )
    return records


def _valid_package_digest(value: object, *, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _conda_forge_record(record: dict[str, object]) -> bool:
    channel = record.get("channel")
    url = record.get("url")
    if not isinstance(channel, str) or not isinstance(url, str):
        return False
    if channel != "conda-forge":
        return False
    parsed = urllib.parse.urlparse(url)
    namespace_by_host = {
        "conda.anaconda.org": "/conda-forge",
        "mirrors.tuna.tsinghua.edu.cn": "/anaconda/cloud/conda-forge",
        "mirrors.bfsu.edu.cn": "/anaconda/cloud/conda-forge",
        "mirrors.ustc.edu.cn": "/anaconda/cloud/conda-forge",
        "mirror.sjtu.edu.cn": "/anaconda/cloud/conda-forge",
        "mirrors.aliyun.com": "/anaconda/cloud/conda-forge",
        "repo.mamba.pm": "/conda-forge",
    }
    name = record.get("name")
    version = record.get("version")
    build = record.get("build")
    subdir = record.get("subdir")
    if not all(isinstance(value, str) and value for value in (name, version, build, subdir)):
        return False
    package_stem = f"{name}-{version}-{build}"
    package_name = Path(parsed.path).name
    try:
        port = parsed.port
    except ValueError:
        return False
    expected_namespace = namespace_by_host.get(parsed.hostname or "")
    return bool(
        parsed.scheme == "https"
        and expected_namespace is not None
        and parsed.username is None
        and parsed.password is None
        and port in (None, 443)
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
        and parsed.path == f"{expected_namespace}/{subdir}/{package_name}"
        and package_name in {f"{package_stem}.conda", f"{package_stem}.tar.bz2"}
    )


def _conda_records_trusted(records: list[dict[str, object]]) -> bool:
    if not records:
        return False
    for record in records:
        if not all(
            isinstance(record.get(key), str) and bool(record.get(key))
            for key in ("name", "version", "build", "subdir")
        ):
            return False
        if type(record.get("build_number")) is not int:
            return False
        if not _conda_forge_record(record):
            return False
        if not (
            _valid_package_digest(record.get("sha256"), length=64)
            or _valid_package_digest(record.get("md5"), length=32)
        ):
            return False
    return True


def _records_sha256(records: list[dict[str, object]]) -> str:
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _direct_conda_records(
    records: list[dict[str, object]], package_specs: tuple[str, ...]
) -> list[dict[str, object]] | None:
    by_name: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_name.setdefault(str(record.get("name")), []).append(record)
    direct: list[dict[str, object]] = []
    for spec in package_specs:
        name, separator, version = spec.partition("=")
        matches = by_name.get(name, [])
        if not separator or len(matches) != 1 or matches[0].get("version") != version:
            return None
        direct.append(matches[0])
    return direct


def _requested_apptainer_specs_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "runtime": APPTAINER_RUNTIME_PACKAGES,
                "build_tools": APPTAINER_BUILD_PACKAGES,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _apptainer_tier_marker_matches(
    env_root: Path,
    *,
    package_specs: tuple[str, ...],
    records: list[dict[str, object]],
) -> bool:
    marker_path = env_root / ".gh-bootstrap-tiers.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(marker, dict):
        return False
    if package_specs == APPTAINER_RUNTIME_PACKAGES:
        required_tiers = (("runtime", APPTAINER_RUNTIME_PACKAGES),)
    elif package_specs == APPTAINER_BUILD_PACKAGES:
        required_tiers = (("build_tools", APPTAINER_BUILD_PACKAGES),)
    elif package_specs == APPTAINER_RUNTIME_PACKAGES + APPTAINER_BUILD_PACKAGES:
        required_tiers = (
            ("runtime", APPTAINER_RUNTIME_PACKAGES),
            ("build_tools", APPTAINER_BUILD_PACKAGES),
        )
    else:
        return False
    if not (
        marker.get("schema_version") == APPTAINER_TIER_MARKER_SCHEMA_VERSION
        and marker.get("channel") == "conda-forge"
        and marker.get("override_channels") is True
        and marker.get("requested_specs_sha256") == _requested_apptainer_specs_sha256()
        and marker.get("resolved_records") == records
        and marker.get("resolved_records_sha256") == _records_sha256(records)
    ):
        return False
    micromamba = marker.get("micromamba")
    if not isinstance(micromamba, dict):
        return False
    micromamba_path_raw = micromamba.get("path")
    micromamba_sha256 = micromamba.get("sha256")
    if not (
        isinstance(micromamba_path_raw, str)
        and _valid_package_digest(micromamba_sha256, length=64)
    ):
        return False
    micromamba_path = Path(micromamba_path_raw)
    if not micromamba_path.is_file() or not os.access(micromamba_path, os.X_OK):
        return False
    try:
        if _sha256_file(micromamba_path) != micromamba_sha256:
            return False
    except OSError:
        return False
    for tier_name, tier_specs in required_tiers:
        tier = marker.get(tier_name)
        direct = _direct_conda_records(records, tier_specs)
        if not (
            direct is not None
            and isinstance(tier, dict)
            and tier.get("package_specs") == list(tier_specs)
            and tier.get("resolved_direct_records") == direct
            and tier.get("ready") is True
        ):
            return False
    return True


def _apptainer_tier_ready(
    env_root: Path,
    *,
    package_specs: tuple[str, ...],
    required_paths: tuple[Path, ...],
    require_marker: bool = True,
) -> bool:
    if not all(path.exists() for path in required_paths):
        return False
    records = _resolved_conda_records(env_root)
    if not _conda_records_trusted(records):
        return False
    if _direct_conda_records(records, package_specs) is None:
        return False
    return not require_marker or _apptainer_tier_marker_matches(
        env_root,
        package_specs=package_specs,
        records=records,
    )


def _apptainer_install_specs(
    env_root: Path,
    *,
    runtime: bool,
    build_tools: bool,
) -> tuple[str, ...]:
    keep_runtime = runtime or _apptainer_tier_ready(
        env_root,
        package_specs=APPTAINER_RUNTIME_PACKAGES,
        required_paths=_apptainer_tier_paths(
            env_root,
            runtime=True,
            build_tools=False,
        ),
    )
    keep_build_tools = build_tools or _apptainer_tier_ready(
        env_root,
        package_specs=APPTAINER_BUILD_PACKAGES,
        required_paths=_apptainer_tier_paths(
            env_root,
            runtime=False,
            build_tools=True,
        ),
    )
    return (
        (APPTAINER_RUNTIME_PACKAGES if keep_runtime else ())
        + (APPTAINER_BUILD_PACKAGES if keep_build_tools else ())
    )


def _prepare_apptainer_env_action(env_root: Path, *, dry_run: bool) -> str:
    if (
        (env_root / "conda-meta" / "history").is_file()
        and _conda_records_trusted(_resolved_conda_records(env_root))
    ):
        return "install"
    if not env_root.exists():
        return "create"
    if dry_run:
        return "create"
    incomplete = env_root.with_name(
        f"{env_root.name}.incomplete.{int(time.time())}.{os.getpid()}"
    )
    os.replace(env_root, incomplete)
    print(f"Quarantined incomplete managed environment: {env_root} -> {incomplete}")
    return "create"


def _remove_incomplete_apptainer_envs(root: Path) -> None:
    envs_root = root / "envs"
    if not envs_root.is_dir():
        return
    for path in sorted(envs_root.glob(f"{APPTAINER_BOOTSTRAP_ENV}.incomplete.*")):
        if path.is_symlink():
            path.unlink()
        else:
            shutil.rmtree(path)
        print(f"Removed superseded incomplete managed environment: {path}")


def _write_apptainer_tier_marker(env_root: Path, *, micromamba: str) -> None:
    records = _resolved_conda_records(env_root)
    runtime_paths = _apptainer_tier_paths(
        env_root,
        runtime=True,
        build_tools=False,
    )
    build_paths = _apptainer_tier_paths(
        env_root,
        runtime=False,
        build_tools=True,
    )
    payload = {
        "schema_version": APPTAINER_TIER_MARKER_SCHEMA_VERSION,
        "channel": "conda-forge",
        "override_channels": True,
        "runtime": {
            "package_specs": list(APPTAINER_RUNTIME_PACKAGES),
            "resolved_direct_records": (
                _direct_conda_records(records, APPTAINER_RUNTIME_PACKAGES) or []
            ),
            "ready": _apptainer_tier_ready(
                env_root,
                package_specs=APPTAINER_RUNTIME_PACKAGES,
                required_paths=runtime_paths,
                require_marker=False,
            ),
        },
        "build_tools": {
            "package_specs": list(APPTAINER_BUILD_PACKAGES),
            "resolved_direct_records": (
                _direct_conda_records(records, APPTAINER_BUILD_PACKAGES) or []
            ),
            "ready": _apptainer_tier_ready(
                env_root,
                package_specs=APPTAINER_BUILD_PACKAGES,
                required_paths=build_paths,
                require_marker=False,
            ),
        },
        "resolved_records": records,
        "resolved_records_sha256": _records_sha256(records),
        "requested_specs_sha256": _requested_apptainer_specs_sha256(),
        "micromamba": {
            "path": str(Path(micromamba).resolve()),
            "sha256": _sha256_file(Path(micromamba)) if Path(micromamba).is_file() else None,
        },
        "cache_policy": (
            "Package archives are retained after failed bootstrap attempts and cleaned only "
            "after a successful bootstrap. A later tier upgrade may require network access."
        ),
    }
    _write_json(env_root / ".gh-bootstrap-tiers.json", payload)


def _ensure_apptainer_tiers(
    root: Path,
    *,
    dry_run: bool,
    runtime: bool,
    build_tools: bool,
) -> Path:
    env_root = root / "envs" / APPTAINER_BOOTSTRAP_ENV
    requested_specs = (
        (APPTAINER_RUNTIME_PACKAGES if runtime else ())
        + (APPTAINER_BUILD_PACKAGES if build_tools else ())
    )
    required_paths = _apptainer_tier_paths(
        env_root,
        runtime=runtime,
        build_tools=build_tools,
    )
    if _apptainer_tier_ready(
        env_root,
        package_specs=requested_specs,
        required_paths=required_paths,
    ):
        return env_root
    if dry_run:
        micromamba = shutil.which("micromamba") or _bootstrap_micromamba(
            root / ".bin",
            dry_run=True,
        )
        install_specs = _apptainer_install_specs(
            env_root,
            runtime=runtime,
            build_tools=build_tools,
        )
        if not _apptainer_tier_ready(
            env_root,
            package_specs=requested_specs,
            required_paths=required_paths,
        ):
            action = _prepare_apptainer_env_action(env_root, dry_run=True)
            _run(
                [
                    micromamba,
                    action,
                    "-y",
                    "-n",
                    APPTAINER_BOOTSTRAP_ENV,
                    "--root-prefix",
                    str(root),
                    "--override-channels",
                    "-c",
                    "conda-forge",
                    *install_specs,
                ],
                dry_run=True,
            )
        return env_root

    lock_path = _apptainer_env_lock_path(root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        _acquire_apptainer_env_lock(lock_handle, lock_path)
        try:
            micromamba = shutil.which("micromamba") or _bootstrap_micromamba(
                root / ".bin",
                dry_run=False,
            )
            install_specs = _apptainer_install_specs(
                env_root,
                runtime=runtime,
                build_tools=build_tools,
            )
            if not _apptainer_tier_ready(
                env_root,
                package_specs=requested_specs,
                required_paths=required_paths,
            ):
                action = _prepare_apptainer_env_action(env_root, dry_run=False)
                _run(
                    [
                        micromamba,
                        action,
                        "-y",
                        "-n",
                        APPTAINER_BOOTSTRAP_ENV,
                        "--root-prefix",
                        str(root),
                        "--override-channels",
                        "-c",
                        "conda-forge",
                        *install_specs,
                    ]
                )
            if not _apptainer_tier_ready(
                env_root,
                package_specs=requested_specs,
                required_paths=required_paths,
                require_marker=False,
            ):
                missing = [str(path) for path in required_paths if not path.exists()]
                raise SystemExit(
                    "Managed Apptainer environment is incomplete after micromamba: "
                    + ", ".join(missing or requested_specs)
                )
            _write_apptainer_tier_marker(env_root, micromamba=micromamba)
            if not _apptainer_tier_ready(
                env_root,
                package_specs=requested_specs,
                required_paths=required_paths,
            ):
                raise SystemExit(
                    "Managed Apptainer environment provenance marker does not match "
                    "the resolved package records."
                )
            _remove_incomplete_apptainer_envs(root)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return env_root


def _ensure_apptainer_via_micromamba(
    root: Path,
    *,
    dry_run: bool,
    build_tools: bool,
) -> str:
    env_root = _ensure_apptainer_tiers(
        root,
        dry_run=dry_run,
        runtime=True,
        build_tools=build_tools,
    )
    apptainer = env_root / "bin" / "apptainer"
    if dry_run:
        return str(apptainer)
    _run([str(apptainer), "--version"], dry_run=False)
    return str(apptainer)


def _fakeroot_capability_tool(
    name: str, *, root: Path, dry_run: bool
) -> str:
    env_root = root / "envs" / APPTAINER_BOOTSTRAP_ENV
    candidates = (
        env_root / "sbin" / name,
        env_root / "bin" / name,
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    system = shutil.which(name)
    if system:
        return system
    if dry_run:
        return str(candidates[0])
    raise SystemExit(
        f"fakeroot validation requires {name}; install libcap or bootstrap the build tier."
    )


def _validate_fakeroot(fakeroot: str, *, root: Path, dry_run: bool) -> str:
    setcap = _fakeroot_capability_tool("setcap", root=root, dry_run=dry_run)
    getcap = _fakeroot_capability_tool("getcap", root=root, dry_run=dry_run)
    _run([fakeroot, "--version"], dry_run=dry_run)
    _run(
        [
            fakeroot,
            "--",
            "sh",
            "-ec",
            (
                'work="$(mktemp -d \"${TMPDIR:-/tmp}/gh-fakeroot.XXXXXX\")"; '
                "trap 'rm -rf \"$work\"' EXIT HUP INT TERM; "
                'file="$work/file"; : >"$file"; chown 1234:2345 "$file"; '
                'test "$(stat -c %u:%g "$file")" = "1234:2345"; '
                '"$1" cap_chown=ep "$file"; '
                '"$2" "$file" | grep -Fq "cap_chown=ep"'
            ),
            "gh-fakeroot-smoke",
            setcap,
            getcap,
        ],
        dry_run=dry_run,
    )
    return fakeroot


def _ensure_fakeroot(root: Path, *, dry_run: bool) -> str:
    system_fakeroot = shutil.which("fakeroot")
    if system_fakeroot:
        return _validate_fakeroot(system_fakeroot, root=root, dry_run=dry_run)

    prefix = root / "tools" / "fakeroot"
    fakeroot = prefix / "bin" / "fakeroot"
    if fakeroot.exists():
        return _validate_fakeroot(str(fakeroot), root=root, dry_run=dry_run)

    source = REPO_ROOT / "vendor" / "fakeroot"
    _require_path(source / "meson.build", label="vendored fakeroot source")
    engine_env = _ensure_apptainer_tiers(
        root,
        dry_run=dry_run,
        runtime=False,
        build_tools=True,
    )
    meson = engine_env / "bin" / "meson"
    if not dry_run:
        _require_path(meson, label="Meson for fakeroot build", executable=True)
        _require_path(
            engine_env / "include" / "sys" / "capability.h",
            label="libcap headers for fakeroot build",
        )

    build_dir = root / "cache" / "fakeroot-build"
    if build_dir.exists() and not dry_run:
        shutil.rmtree(build_dir)
    build_env = [
        "env",
        f"CPATH={engine_env / 'include'}",
        f"LIBRARY_PATH={engine_env / 'lib'}",
        f"PKG_CONFIG_PATH={engine_env / 'lib' / 'pkgconfig'}",
    ]
    _run(
        build_env
        + [
            str(meson),
            "setup",
            "--prefix",
            str(prefix),
            "--buildtype",
            "release",
            str(build_dir),
            str(source),
        ],
        dry_run=dry_run,
    )
    _run(
        build_env
        + [str(meson), "compile", "-C", str(build_dir), "-j", "4"],
        dry_run=dry_run,
    )
    _run([str(meson), "install", "-C", str(build_dir)], dry_run=dry_run)
    if not dry_run:
        _require_path(fakeroot, label="built fakeroot executable", executable=True)
    return _validate_fakeroot(str(fakeroot), root=root, dry_run=dry_run)


def _bootstrapped_apptainer_path(root: Path) -> Path:
    return (root / "envs" / APPTAINER_BOOTSTRAP_ENV / "bin" / "apptainer").resolve()


def _is_bootstrapped_apptainer(apptainer_bin: str, root: Path) -> bool:
    try:
        apptainer_path = Path(apptainer_bin).resolve()
    except OSError:
        return False
    return apptainer_path == _bootstrapped_apptainer_path(root)


def _missing_uidmap_helpers() -> tuple[str, ...]:
    return tuple(name for name in ("newuidmap", "newgidmap") if shutil.which(name) is None)


def _check_apptainer_site_support(
    *,
    apptainer_bin: str,
    root: Path,
    dry_run: bool,
    requires_uidmap: bool,
) -> None:
    if (
        dry_run
        or not requires_uidmap
        or not _is_bootstrapped_apptainer(apptainer_bin, root)
    ):
        return
    missing = _missing_uidmap_helpers()
    if not missing:
        return
    missing_text = ", ".join(missing)
    raise SystemExit(
        "Bootstrapped Apptainer is installed but this site is missing the uidmap "
        f"helpers required by unprivileged Apptainer: {missing_text}.\n"
        f"Bootstrapped engine: {apptainer_bin}\n"
        "Ask the site admin to install uidmap helpers, or provide a working system "
        "Apptainer/Singularity with --apptainer-bin.\n"
        "If remote builds are configured, rerun with --remote. If a SIF already "
        "exists, rerun with --skip-build --sif <path> and a working engine to "
        "generate and validate the runner."
    )


def _detect_apptainer(
    explicit: str | None,
    *,
    dry_run: bool = False,
    root: Path | None = None,
    bootstrap: str = "auto",
    build_tools: bool = False,
) -> str:
    if explicit:
        resolved = str(_expand(explicit))
        if not dry_run:
            _require_path(Path(resolved), label="explicit Apptainer/Singularity", executable=True)
            _run([resolved, "--version"], dry_run=False)
        return resolved
    if bootstrap != "always":
        for name in ("apptainer", "singularity"):
            found = shutil.which(name)
            if found:
                if not dry_run:
                    _run([found, "--version"], dry_run=False)
                return found
    if dry_run:
        if bootstrap == "always" and root is not None:
            return str(_bootstrapped_apptainer_path(root))
        return "apptainer"
    if bootstrap in {"auto", "always"}:
        if root is None:
            raise SystemExit("Internal error: root is required for automatic Apptainer bootstrap.")
        return _ensure_apptainer_via_micromamba(
            root,
            dry_run=dry_run,
            build_tools=build_tools,
        )
    raise SystemExit(
        "Could not find `apptainer` or `singularity` in PATH. "
        "Install one, pass --apptainer-bin, or allow the default "
        "micromamba bootstrap with --bootstrap-apptainer auto/always."
    )


def _apptainer_command(apptainer_bin: str, *args: str) -> list[str]:
    engine_bin = str(Path(apptainer_bin).resolve().parent)
    inherited_path = os.environ.get("PATH", "")
    command_path = engine_bin + (os.pathsep + inherited_path if inherited_path else "")
    return ["env", f"PATH={command_path}", apptainer_bin, *args]


def _container_inspect_command(
    args: argparse.Namespace,
    *,
    apptainer_bin: str,
    sif: Path,
) -> list[str]:
    if args.runner_backend == "apptainer":
        return _apptainer_command(apptainer_bin, "inspect", str(sif))
    if args.runner_backend == "docker-apptainer":
        if not args.docker_bin_resolved:
            raise SystemExit("Docker/Apptainer inspect requires a resolved Docker executable.")
        return [
            args.docker_bin_resolved,
            "run",
            "--rm",
            "--privileged",
            "-v",
            f"{sif}:/image.sif:ro",
            args.runner_docker_image,
            "apptainer",
            "inspect",
            "/image.sif",
        ]
    raise SystemExit(f"Unknown runner backend: {args.runner_backend}")


def _detect_docker(explicit: str | None) -> str:
    if explicit:
        return str(_expand(explicit))
    found = shutil.which("docker")
    if found:
        return found
    raise SystemExit(
        "Could not find `docker` in PATH. Install Docker, pass --docker-bin, "
        "or use --container-builder apptainer."
    )


def _normalize_af2_data_dir(raw: str | None) -> Path | None:
    if not raw:
        return None
    path = _expand(raw)
    if path.name == "params" and (path / AF2_SENTINEL).exists():
        return path.parent
    if (path / "params" / AF2_SENTINEL).exists():
        return path
    # Keep the parent convention even before download/existence checks, because
    # AF2 loads data_dir/params/params_*.npz.
    if path.name == "params":
        return path.parent
    return path


def _resource_bind(host_path: Path, container_path: str) -> str:
    return f"{host_path}:{container_path}"


def _append_bind(binds: list[str], host_path: Path, container_path: str) -> None:
    bind = _resource_bind(host_path, container_path)
    if bind not in binds:
        binds.append(bind)


def _require_path(
    host_path: Path | None,
    *,
    label: str,
    executable: bool = False,
) -> Path:
    if host_path is None:
        raise SystemExit(f"Missing required resource path: {label}")
    if not host_path.exists():
        raise SystemExit(f"Required resource path does not exist for {label}: {host_path}")
    if executable and not os.access(host_path, os.X_OK):
        raise SystemExit(f"Required executable is not executable for {label}: {host_path}")
    return host_path


def _append_bind_and_env(
    *,
    binds: list[str],
    env_defaults: dict[str, str],
    host_path: Path | None,
    container_path: str,
    env_key: str,
    required: bool,
    label: str,
) -> None:
    if host_path is None:
        if required:
            raise SystemExit(f"Missing required resource path: {label}")
        return
    if required:
        _require_path(host_path, label=label)
    _append_bind(binds, host_path, container_path)
    env_defaults[env_key] = container_path


def _append_file_bind_and_env(
    *,
    binds: list[str],
    env_defaults: dict[str, str],
    host_path: Path | None,
    container_path: str,
    env_key: str,
    required: bool,
    label: str,
    executable: bool = False,
) -> None:
    if host_path is None:
        if required:
            raise SystemExit(f"Missing required resource path: {label}")
        return
    if required:
        _require_path(host_path, label=label, executable=executable)
    elif not host_path.exists():
        return
    _append_bind(binds, host_path, container_path)
    env_defaults[env_key] = container_path


def _append_python_env_bind_and_env(
    *,
    binds: list[str],
    env_defaults: dict[str, str],
    host_python: Path | None,
    container_prefix: str,
    env_key: str,
    required: bool,
    label: str,
) -> None:
    if host_python is None:
        if required:
            raise SystemExit(f"Missing required Python executable path: {label}")
        return
    if required:
        _require_path(host_python, label=label, executable=True)
    elif not host_python.exists():
        return
    env_prefix = host_python.parent.parent
    _append_bind(binds, env_prefix, container_prefix)
    env_defaults[env_key] = f"{container_prefix}/bin/{host_python.name}"
    resolved_python = host_python.resolve()
    try:
        resolved_python.relative_to(env_prefix)
    except ValueError:
        # Preserve uv/venv absolute Python symlinks and script shebangs.
        _append_bind(binds, env_prefix, str(env_prefix))
        base_prefix = resolved_python.parent.parent
        _append_bind(binds, base_prefix, str(base_prefix))


def _append_molfile_to_params_bind(
    *,
    binds: list[str],
    env_defaults: dict[str, str],
    rosetta_root: Path | None,
    molfile_to_params: Path | None,
    required: bool,
) -> None:
    if rosetta_root is None and molfile_to_params is None:
        if required:
            raise SystemExit("Missing required resource path: Rosetta molfile_to_params")
        return
    if molfile_to_params is None:
        assert rosetta_root is not None
        molfile_to_params = rosetta_root / "molfile_to_params.py"
    if required:
        _require_path(molfile_to_params, label="Rosetta molfile_to_params")
    elif not molfile_to_params.exists():
        return

    if rosetta_root is not None:
        if required:
            _require_path(rosetta_root, label="Rosetta root")
        _append_bind(binds, rosetta_root, "/rosetta")
        try:
            rel = molfile_to_params.resolve().relative_to(rosetta_root.resolve())
        except ValueError as exc:
            raise SystemExit(
                "--molfile-to-params must be inside --rosetta-root when both are provided."
            ) from exc
        env_defaults["GH_MOLFILE_TO_PARAMS"] = "/rosetta/" + str(rel)
        return

    _append_bind(binds, molfile_to_params.parent, "/rosetta")
    env_defaults["GH_MOLFILE_TO_PARAMS"] = f"/rosetta/{molfile_to_params.name}"


def _ligandmpnn_complete(path: Path) -> bool:
    return all((path / fn).is_file() and (path / fn).stat().st_size >= 1_000_000 for fn in LIGANDMPNN_CKPTS)


def _validate_af2_params_dir(params_dir: Path) -> None:
    for name, (exact_size, expected_sha256) in AF2_PARAM_SPECS.items():
        path = params_dir / name
        validate_file(
            path,
            minimum_size=exact_size,
            expected_sha256=expected_sha256,
        )
        if path.stat().st_size != exact_size:
            raise RuntimeError(
                f"AF2 parameter has unexpected exact size: {path} has "
                f"{path.stat().st_size} bytes; expected {exact_size}"
            )


def _af2_params_complete(params_dir: Path) -> bool:
    try:
        _validate_af2_params_dir(params_dir)
    except RuntimeError:
        return False
    return True


def _validate_af2_archive(path: Path) -> None:
    validate_file(path, minimum_size=AF2_PARAMS_ARCHIVE_SIZE)
    if path.stat().st_size != AF2_PARAMS_ARCHIVE_SIZE:
        raise RuntimeError(
            f"AF2 parameter archive has unexpected exact size: {path} has "
            f"{path.stat().st_size} bytes; expected {AF2_PARAMS_ARCHIVE_SIZE}"
        )


def _download_af2_params(af2_data_dir: Path, *, dry_run: bool) -> None:
    params_dir = af2_data_dir / "params"
    if _af2_params_complete(params_dir):
        return
    tar_path = af2_data_dir / ".downloads" / Path(AF2_PARAMS_URL).name
    partial_dir = af2_data_dir / ".params.partial"
    if dry_run:
        _download_url(
            AF2_PARAMS_URL,
            tar_path,
            dry_run=True,
            minimum_size=AF2_PARAMS_ARCHIVE_SIZE,
        )
        _extract_tar(
            tar_path,
            partial_dir,
            members=AF2_PARAM_FILES,
            dry_run=True,
        )
        return

    quarantined: Path | None = None
    for attempt in range(1, 3):
        _download_url(
            AF2_PARAMS_URL,
            tar_path,
            dry_run=False,
            minimum_size=AF2_PARAMS_ARCHIVE_SIZE,
        )
        try:
            _validate_af2_archive(tar_path)
            shutil.rmtree(partial_dir, ignore_errors=True)
            _extract_tar(
                tar_path,
                partial_dir,
                members=AF2_PARAM_FILES,
                dry_run=False,
            )
            _validate_af2_params_dir(partial_dir)
        except (OSError, RuntimeError, tarfile.TarError) as exc:
            shutil.rmtree(partial_dir, ignore_errors=True)
            for stale in tar_path.parent.glob(f"{tar_path.name}.invalid.*"):
                stale.unlink(missing_ok=True)
            quarantined = tar_path.with_name(
                f"{tar_path.name}.invalid.{time.time_ns()}"
            )
            if tar_path.exists():
                os.replace(tar_path, quarantined)
            if attempt == 2:
                raise RuntimeError(
                    "AF2 parameter archive failed exact extraction twice; "
                    f"retained latest invalid archive at {quarantined}: {exc}"
                ) from exc
            print(
                f"Retained invalid AF2 archive at {quarantined}; downloading once more"
            )
            continue

        params_dir.mkdir(parents=True, exist_ok=True)
        for name in AF2_PARAM_FILES:
            os.replace(partial_dir / name, params_dir / name)
        shutil.rmtree(partial_dir)
        _validate_af2_params_dir(params_dir)
        tar_path.unlink(missing_ok=True)
        for stale in tar_path.parent.glob(f"{tar_path.name}.invalid.*"):
            stale.unlink(missing_ok=True)
        return


def _download_ligandmpnn_params(ligandmpnn_dir: Path, *, dry_run: bool) -> None:
    for name in LIGANDMPNN_CKPTS:
        dst = ligandmpnn_dir / name
        _download_url(
            LIGANDMPNN_ASSET_URLS[name],
            dst,
            dry_run=dry_run,
            minimum_size=1_000_000,
            expected_sha256=LIGANDMPNN_SHA256[name],
        )


def _protenix_download_target(
    protenix_dir: Path,
    rel_path: str,
    protenix_env_paths: dict[str, Path] | None,
) -> Path:
    env_key = PROTENIX_ENV_BY_REL_PATH[rel_path]
    return (protenix_env_paths or {}).get(env_key, protenix_dir / rel_path)


def _download_protenix_assets(
    protenix_dir: Path,
    *,
    protenix_env_paths: dict[str, Path] | None = None,
    dry_run: bool,
) -> None:
    for rel_path, url in PROTENIX_ASSET_URLS.items():
        dst = _protenix_download_target(
            protenix_dir, rel_path, protenix_env_paths
        )
        expected_sha256 = PROTENIX_ASSET_SHA256[rel_path]
        minimum_size = 1_000_000_000 if rel_path.startswith("release_model/") else 50_000_000
        try:
            _download_url(
                url,
                dst,
                dry_run=dry_run,
                minimum_size=minimum_size,
                expected_sha256=expected_sha256,
            )
        except RuntimeError as exc:
            if rel_path.endswith("protenix-v2.pt"):
                raise SystemExit(
                    "The official Protenix v2 endpoint is currently returning HTTP 403 "
                    "(upstream issue bytedance/Protenix#309). Supply an exact existing "
                    "protenix-v2.pt via --protenix-assets or GH_WEIGHT_PROTENIX; "
                    f"expected SHA256 {PROTENIX_V2_SHA256}."
                ) from exc
            raise


def _download_rhofold_checkpoint(rhofold_checkpoint: Path, *, dry_run: bool) -> None:
    _download_url(
        RHOFOLD_CHECKPOINT_URL,
        rhofold_checkpoint,
        dry_run=dry_run,
        minimum_size=500_000_000,
        expected_sha256=RHOFOLD_CHECKPOINT_SHA256,
    )


def _download_exact_assets(
    root: Path,
    assets: dict[str, tuple[str, int, str]],
    *,
    dry_run: bool,
) -> None:
    for rel_path, (url, exact_size, expected_sha256) in assets.items():
        target = root / rel_path
        _download_url(
            url,
            target,
            dry_run=dry_run,
            minimum_size=exact_size,
            expected_sha256=expected_sha256,
        )
        if not dry_run and target.stat().st_size != exact_size:
            raise SystemExit(
                f"Downloaded asset has unexpected exact size: {target} has "
                f"{target.stat().st_size} bytes; expected {exact_size}"
            )


def _download_single_exact_asset(
    target: Path,
    asset: tuple[str, int, str],
    *,
    dry_run: bool,
) -> None:
    url, exact_size, expected_sha256 = asset
    _download_url(
        url,
        target,
        dry_run=dry_run,
        minimum_size=exact_size,
        expected_sha256=expected_sha256,
    )
    if not dry_run and target.stat().st_size != exact_size:
        raise SystemExit(
            f"Downloaded asset has unexpected exact size: {target} has "
            f"{target.stat().st_size} bytes; expected {exact_size}"
        )


def _validate_single_exact_asset(path: Path, asset: tuple[str, int, str]) -> None:
    _url, exact_size, expected_sha256 = asset
    validate_file(
        path,
        minimum_size=exact_size,
        expected_sha256=expected_sha256,
    )
    if path.stat().st_size != exact_size:
        raise SystemExit(
            f"Asset has unexpected exact size: {path} has {path.stat().st_size} bytes; "
            f"expected {exact_size}"
        )


def _download_public_assets(
    *,
    root: Path,
    af2_data_dir: Path | None,
    ligandmpnn_dir: Path,
    protenix_dir: Path,
    protenix_env_paths: dict[str, Path] | None = None,
    rhofold_checkpoint: Path | None = None,
    opendde_assets: Path | None = None,
    rapidock_weights: Path | None = None,
    rf3_checkpoint: Path | None = None,
    rfd3_checkpoint: Path | None = None,
    surfpro_checkpoint: Path | None = None,
    surfpro_asset: tuple[str, int, str] = SURFPRO_CATH42_ASSET,
    selected: Iterable[str] | None = None,
    jobs: int = 4,
    dry_run: bool,
) -> None:
    del root  # Kept in the function signature for compatibility with earlier callers.
    selected_set = set(selected or ("alphafold2", "ligandmpnn", "protenix"))
    protenix_prefetched = False
    if "protenix" in selected_set:
        v2_path = _protenix_download_target(
            protenix_dir,
            "release_model/protenix-v2.pt",
            protenix_env_paths,
        )
        if v2_path.is_file() and not dry_run:
            validate_file(
                v2_path,
                minimum_size=1_000_000_000,
                expected_sha256=PROTENIX_V2_SHA256,
            )
        elif not v2_path.is_file():
            # The known-broken v2 endpoint must fail before multi-GB downloads
            # for other software start in the I/O pool.
            _download_protenix_assets(
                protenix_dir,
                protenix_env_paths=protenix_env_paths,
                dry_run=dry_run,
            )
            protenix_prefetched = True
    actions = []
    if "alphafold2" in selected_set and af2_data_dir is not None:
        actions.append(lambda: _download_af2_params(af2_data_dir, dry_run=dry_run))
    if "ligandmpnn" in selected_set:
        actions.append(lambda: _download_ligandmpnn_params(ligandmpnn_dir, dry_run=dry_run))
    if "protenix" in selected_set and not protenix_prefetched:
        actions.append(
            lambda: _download_protenix_assets(
                protenix_dir,
                protenix_env_paths=protenix_env_paths,
                dry_run=dry_run,
            )
        )
    if "rhofold" in selected_set and rhofold_checkpoint is not None:
        actions.append(
            lambda: _download_rhofold_checkpoint(rhofold_checkpoint, dry_run=dry_run)
        )
    if "opendde" in selected_set and opendde_assets is not None:
        actions.append(
            lambda: _download_exact_assets(
                opendde_assets, OPENDDE_ASSETS, dry_run=dry_run
            )
        )
    if "rapidock" in selected_set and rapidock_weights is not None:
        actions.append(
            lambda: _download_exact_assets(
                rapidock_weights, RAPIDOCK_ASSETS, dry_run=dry_run
            )
        )
    if "rf3" in selected_set and rf3_checkpoint is not None:
        actions.append(
            lambda: _download_single_exact_asset(
                rf3_checkpoint, RF3_ASSET, dry_run=dry_run
            )
        )
    if "rfd3" in selected_set and rfd3_checkpoint is not None:
        actions.append(
            lambda: _download_single_exact_asset(
                rfd3_checkpoint, RFD3_ASSET, dry_run=dry_run
            )
        )
    if "surfpro" in selected_set and surfpro_checkpoint is not None:
        actions.append(
            lambda: _download_single_exact_asset(
                surfpro_checkpoint, surfpro_asset, dry_run=dry_run
            )
        )
    if jobs == 1 or len(actions) < 2:
        for action in actions:
            action()
        return
    with ThreadPoolExecutor(max_workers=min(jobs, len(actions))) as executor:
        tuple(executor.map(lambda action: action(), actions))


def _run_prefetch_command(
    args: argparse.Namespace,
    *,
    python: Path | str,
    selected: Iterable[str],
    boltz_cache: Path,
    chai_cache: Path,
    torch_cache: Path,
    hf_cache: Path,
    container_paths: bool,
) -> None:
    names = tuple(selected)
    if not names:
        return
    cmd = [
        str(python),
        str(REPO_ROOT / "scripts" / "installation" / "asset_prefetch.py")
        if not container_paths
        else f"{args.container_repo_root}/scripts/installation/asset_prefetch.py",
        "--software",
        ",".join(names),
        "--jobs",
        str(args.asset_download_jobs),
        "--boltz-cache",
        "/models/boltz" if container_paths else str(boltz_cache),
        "--chai-cache",
        "/models/chai1" if container_paths else str(chai_cache),
        "--torch-cache",
        "/models/torch" if container_paths else str(torch_cache),
        "--hf-cache",
        "/models/huggingface" if container_paths else str(hf_cache),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    if container_paths:
        cmd = ["env", "HF_HUB_OFFLINE=0", "TRANSFORMERS_OFFLINE=0", *cmd]
    if container_paths and args.srun:
        cmd = ["bash", "-lc", shlex.quote(args.srun) + " " + shlex.quote(_quote_cmd(cmd))]
    _run(cmd, cwd=REPO_ROOT, dry_run=args.dry_run)


def _run_host_asset_prefetch(
    args: argparse.Namespace,
    *,
    selected: Iterable[str],
    dplm_python: Path | None,
    esmfold2_python: Path | None,
    hf_cache: Path,
) -> None:
    for name, python in (("dplm", dplm_python), ("esmfold2", esmfold2_python)):
        if name not in selected:
            continue
        if python is None or (not args.dry_run and not python.is_file()):
            raise SystemExit(
                f"{name} asset download requires its installed host environment; "
                "use --setup-host-envs or pass the matching --*-python path."
            )
        _run_prefetch_command(
            args,
            python=python,
            selected=(name,),
            boltz_cache=Path("."),
            chai_cache=Path("."),
            torch_cache=Path("."),
            hf_cache=hf_cache,
            container_paths=False,
        )


def _run_selected_host_import_smoke(
    args: argparse.Namespace,
    *,
    selected: Iterable[str],
    dplm_python: Path | None,
    opendde_python: Path | None,
    esmfold2_python: Path | None,
    rapidock_python: Path | None,
    rpxdock_python: Path | None,
    rapidock_repo: Path,
    rf3_python: Path | None,
    rf3_repo: Path | None,
    surfdesign_python: Path | None,
    surfdesign_repo: Path,
    surfpro_python: Path | None,
    surfpro_repo: Path,
    hf_cache: Path,
) -> None:
    specs = {
        "dplm": (
            dplm_python,
            [REPO_ROOT / "vendor" / "dplm", REPO_ROOT / "vendor" / "dplm" / "src"],
            "import byprot, torch, transformers",
        ),
        "opendde": (
            opendde_python,
            [REPO_ROOT / "vendor" / "opendde"],
            "import opendde, torch",
        ),
        "esmfold2": (
            esmfold2_python,
            [REPO_ROOT / "vendor" / "biohub_esm", REPO_ROOT / "vendor" / "biohub_transformers"],
            "import esm, torch, transformers",
        ),
        "rapidock": (
            rapidock_python,
            [rapidock_repo],
            "import e3nn, torch",
        ),
        "rpxdock": (
            rpxdock_python,
            [REPO_ROOT / "vendor" / "rpxdock", REPO_ROOT / "vendor" / "willutil"],
            "import rpxdock, willutil",
        ),
        "surfdesign": (
            surfdesign_python,
            [surfdesign_repo],
            "import pymol2; from src import utils; from src.models.generator import IterativeRefinementGenerator",
        ),
        "surfpro": (
            surfpro_python,
            [
                surfpro_repo,
                REPO_ROOT / "vendor" / "sru",
                REPO_ROOT / "vendor" / "protein_sequence_models",
            ],
            (
                "import sys, torch, sru, sequence_models; "
                "assert sys.version_info[:2] == (3, 8), sys.version; "
                "assert torch.__version__.split('+', 1)[0] == '1.10.2', torch.__version__; "
                "assert torch.version.cuda == '11.3', torch.version.cuda; "
                "from fairseq.models.surface_protein_model import get_edges_batch"
            ),
        ),
        "rf3": (
            rf3_python,
            [rf3_repo / "src"] if rf3_repo is not None else [],
            "from rf3.inference_engines.rf3 import RF3InferenceEngine",
        ),
        "rfd3": (
            rf3_python,
            (
                [rf3_repo / "src", rf3_repo / "models" / "rfd3" / "src"]
                if rf3_repo is not None
                else []
            ),
            "from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine",
        ),
    }
    selected_set = set(selected)
    for name, (python, python_paths, code) in specs.items():
        if name not in selected_set:
            continue
        if python is None:
            raise SystemExit(f"Selected host backend {name} has no Python executable.")
        if not args.dry_run:
            _require_path(python, label=f"{name} Python", executable=True)
        cmd = [
            "env",
            "PYTHONNOUSERSITE=1",
            f"PYTHONPATH={os.pathsep.join(str(path) for path in python_paths)}",
            f"HF_HOME={hf_cache}",
            f"HUGGINGFACE_HUB_CACHE={hf_cache / 'hub'}",
            str(python),
            "-c",
            code,
        ]
        _run(cmd, cwd=REPO_ROOT, dry_run=args.dry_run)


def _hf_snapshot_path(cache: Path, software: str) -> Path:
    return hf_runtime_snapshot_path(cache, software)


def _marker_ready(path: Path) -> bool:
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(child.is_file() and child.stat().st_size > 0 for child in path.rglob("*"))
    return False


def _validate_selected_assets(
    selected: Iterable[str],
    markers: dict[str, list[Path]],
    *,
    surfpro_asset: tuple[str, int, str] | None = None,
) -> None:
    validated = {
        "alphafold3",
        "alphafold2",
        "ligandmpnn",
        "protenix",
        "boltz",
        "chai1",
        "esm2",
        "esmif",
        "esmfold",
        "esm3",
        "dplm",
        "esmfold2",
        "rhofold",
        "nampnn",
        "opendde",
        "rapidock",
        "rf3",
        "rfd3",
        "carbondesign",
        "surfdesign",
        "surfpro",
        "haddock3",
        "autodock-gpu",
    }
    missing = [
        f"{name}:{path}"
        for name in selected
        if name in validated
        for path in markers.get(name, [])
        if not _marker_ready(path)
    ]
    empty = [name for name in selected if name in validated and not markers.get(name)]
    if missing or empty:
        details = ", ".join((*empty, *missing))
        raise SystemExit(f"Selected software has missing or empty resources: {details}")

    if "alphafold2" in selected:
        params_dir = markers["alphafold2"][0].parent
        _validate_af2_params_dir(params_dir)
    if "ligandmpnn" in selected:
        for name, expected in LIGANDMPNN_SHA256.items():
            validate_file(markers["ligandmpnn"][0].parent / name, expected_sha256=expected)
    if "protenix" in selected:
        for marker, (rel_path, expected) in zip(
            markers["protenix"], PROTENIX_ASSET_SHA256.items()
        ):
            minimum_size = 1_000_000_000 if rel_path.startswith("release_model/") else 50_000_000
            validate_file(
                marker,
                minimum_size=minimum_size,
                expected_sha256=expected,
            )
    if "boltz" in selected:
        root = markers["boltz"][0].parent
        for name, expected in BOLTZ_ASSET_SHA256.items():
            validate_file(root / name, expected_sha256=expected)
    if "chai1" in selected:
        root = markers["chai1"][0].parent
        validate_file(
            root / "conformers_v1.apkl",
            minimum_size=100_000_000,
            expected_sha256=CHAI_CONFORMERS_SHA256,
        )
        validate_file(
            root / "esm" / "traced_sdpa_esm2_t36_3B_UR50D_fp16.pt",
            minimum_size=5_000_000_000,
            expected_sha256=CHAI_ESM_SHA256,
        )
        for name, (minimum_size, expected) in CHAI_COMPONENTS.items():
            validate_file(
                root / "models_v2" / name,
                minimum_size=minimum_size,
                expected_sha256=expected,
            )
    if "rhofold" in selected:
        validate_file(
            markers["rhofold"][0],
            minimum_size=500_000_000,
            expected_sha256=RHOFOLD_CHECKPOINT_SHA256,
        )
    if "opendde" in selected:
        root = markers["opendde"][0].parents[1]
        for rel_path, asset in OPENDDE_ASSETS.items():
            _validate_single_exact_asset(root / rel_path, asset)
    if "rapidock" in selected:
        root = markers["rapidock"][0].parent
        for name, asset in RAPIDOCK_ASSETS.items():
            _validate_single_exact_asset(root / name, asset)
    if "rf3" in selected:
        _validate_single_exact_asset(markers["rf3"][0], RF3_ASSET)
    if "rfd3" in selected:
        _validate_single_exact_asset(markers["rfd3"][0], RFD3_ASSET)
    if "surfpro" in selected:
        marker = markers["surfpro"][0]
        if surfpro_asset is not None:
            _validate_single_exact_asset(marker, surfpro_asset)
        else:
            approved = {asset[2]: asset for asset in SURFPRO_ASSETS.values()}
            actual = _sha256_file(marker)
            if actual not in approved:
                raise SystemExit(
                    f"SurfPro checkpoint SHA256 mismatch for {marker}: {actual}; "
                    f"expected one of {', '.join(sorted(approved))}"
                )
            _validate_single_exact_asset(marker, approved[actual])
    snapshot_components = {
        "esmfold2": ("esmfold2", "esmfold2-esmc"),
    }
    for selected_name in set(selected) & set(HF_SNAPSHOT_REQUIRED_FILES):
        for name in snapshot_components.get(selected_name, (selected_name,)):
            manifest_path = next(
                path
                for path in markers[selected_name]
                if path.name == f"{name}.json"
                and path.parent.name == "gh-snapshots"
            )
            cache = manifest_path.parents[1]
            snapshot = hf_runtime_snapshot_path(cache, name)
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(
                    f"Invalid Hugging Face snapshot manifest {manifest_path}: {exc}"
                ) from exc
            if not isinstance(manifest, dict):
                raise SystemExit(
                    f"Invalid Hugging Face snapshot manifest object: {manifest_path}"
                )
            repo_id, revision = HF_SNAPSHOT_REPOS[name]
            expected = {
                "schema_version": 2,
                "software": name,
                "repo_id": repo_id,
                "revision": revision,
                "runtime_snapshot_relative": str(snapshot.relative_to(cache)),
                "allow_patterns": list(HF_SNAPSHOT_REQUIRED_FILES[name]),
                "required_files": list(HF_SNAPSHOT_REQUIRED_FILES[name]),
            }
            mismatched = [
                key for key, value in expected.items() if manifest.get(key) != value
            ]
            if mismatched:
                raise SystemExit(
                    f"Hugging Face snapshot manifest mismatch for {name}: "
                    + ", ".join(mismatched)
                )
            required_files = HF_SNAPSHOT_REQUIRED_FILES[name]
            expected_entries = set(required_files)
            for rel_path in required_files:
                parent = Path(rel_path).parent
                while str(parent) != ".":
                    expected_entries.add(str(parent))
                    parent = parent.parent
            actual_entries = {
                str(path.relative_to(snapshot)) for path in snapshot.rglob("*")
            }
            if actual_entries != expected_entries:
                unexpected = sorted(actual_entries - expected_entries)
                absent = sorted(expected_entries - actual_entries)
                raise SystemExit(
                    f"Hugging Face runtime snapshot is not exact for {name}; "
                    f"unexpected={unexpected}, missing={absent}"
                )
            raw_files = manifest.get("files")
            if not isinstance(raw_files, list):
                raise SystemExit(
                    f"Hugging Face snapshot manifest has no file records: {manifest_path}"
                )
            file_records = {
                entry.get("path"): entry
                for entry in raw_files
                if isinstance(entry, dict) and isinstance(entry.get("path"), str)
            }
            if set(file_records) != set(required_files) or len(file_records) != len(
                raw_files
            ):
                raise SystemExit(
                    f"Hugging Face snapshot file manifest is not exact for {name}"
                )
            static_specs = HF_SNAPSHOT_FILE_SPECS.get(name, {})
            for rel_path in required_files:
                entry = file_records[rel_path]
                size = entry.get("size")
                checksum = entry.get("sha256")
                if (
                    not isinstance(size, int)
                    or size < 1
                    or not isinstance(checksum, str)
                    or len(checksum) != 64
                    or any(char not in "0123456789abcdef" for char in checksum)
                ):
                    raise SystemExit(
                        f"Invalid Hugging Face file record for {name}:{rel_path}"
                    )
                static_spec = static_specs.get(rel_path)
                if static_spec is not None and static_spec != (size, checksum):
                    raise SystemExit(
                        f"Hugging Face static asset contract mismatch for {name}:{rel_path}"
                    )
                path = snapshot / rel_path
                if path.stat().st_size != size:
                    raise SystemExit(
                        f"Hugging Face file size mismatch for {name}:{rel_path}"
                    )
                validate_file(
                    path,
                    minimum_size=size,
                    expected_sha256=checksum,
                    use_adjacent_stamp=False,
                )


def _asset_markers(
    *,
    af3_weights: Path | None,
    af2_data_dir: Path | None,
    ligandmpnn_dir: Path,
    protenix_dir: Path,
    protenix_env_paths: dict[str, Path],
    boltz_cache: Path,
    chai_cache: Path,
    torch_cache: Path,
    hf_cache: Path,
    rhofold_checkpoint: Path,
    opendde_assets: Path,
    rapidock_weights: Path,
    surfdesign_python: Path | None,
    surfdesign_repo: Path,
    surfdesign_experiment: Path | None,
    surfpro_python: Path | None,
    surfpro_repo: Path,
    surfpro_msms: Path | None,
    surfpro_pdb_to_xyzrn: Path | None,
    surfpro_checkpoint: Path,
    rf3_checkpoint: Path | None,
    rfd3_checkpoint: Path | None,
    carbondesign_repo: Path | None,
    rosetta_root: Path | None,
    molfile_to_params: Path | None,
    haddock3_bin: Path | None,
    autodock_gpu: Path | None,
    autogrid4_bin: Path | None,
    ananas_bin: Path | None,
) -> dict[str, list[Path]]:
    markers = {
        "alphafold3": [af3_weights] if af3_weights is not None else [],
        "alphafold2": (
            [af2_data_dir / "params" / name for name in AF2_PARAM_FILES]
            if af2_data_dir
            else []
        ),
        "ligandmpnn": [ligandmpnn_dir / name for name in LIGANDMPNN_CKPTS],
        "protenix": [
            protenix_env_paths.get(PROTENIX_ENV_BY_REL_PATH[rel], protenix_dir / rel)
            for rel in PROTENIX_ASSET_URLS
        ],
        "boltz": [boltz_cache / "ccd.pkl", boltz_cache / "boltz1_conf.ckpt"],
        "chai1": [
            chai_cache / "conformers_v1.apkl",
            chai_cache / "esm" / "traced_sdpa_esm2_t36_3B_UR50D_fp16.pt",
            *[chai_cache / "models_v2" / name for name in (
                "feature_embedding.pt",
                "trunk.pt",
                "diffusion_module.pt",
                "confidence_head.pt",
                "token_embedder.pt",
                "bond_loss_input_proj.pt",
            )],
        ],
        "esm2": [
            torch_cache / "hub" / "checkpoints" / "esm2_t33_650M_UR50D.pt",
            torch_cache / "hub" / "checkpoints" / "esm2_t33_650M_UR50D-contact-regression.pt",
        ],
        "esmif": [
            torch_cache / "hub" / "checkpoints" / "esm_if1_gvp4_t16_142M_UR50.pt"
        ],
        "esmfold": [
            torch_cache / "hub" / "checkpoints" / "esmfold_3B_v1.pt",
            torch_cache / "hub" / "checkpoints" / "esm2_t36_3B_UR50D.pt",
            torch_cache
            / "hub"
            / "checkpoints"
            / "esm2_t36_3B_UR50D-contact-regression.pt",
        ],
        "rhofold": [rhofold_checkpoint],
        "nampnn": [REPO_ROOT / "vendor" / "na_mpnn" / "models" / "design_model" / "s_19137.pt"],
        "opendde": [opendde_assets / rel_path for rel_path in OPENDDE_ASSETS],
        "rapidock": [rapidock_weights / name for name in RAPIDOCK_ASSETS],
        "surfdesign": (
            [
                surfdesign_experiment / ".hydra" / "config.yaml",
                surfdesign_experiment / "checkpoints" / "best.ckpt",
                surfdesign_repo / "src" / "models" / "generator.py",
                *([surfdesign_python] if surfdesign_python is not None else []),
            ]
            if surfdesign_experiment is not None
            else []
        ),
        "surfpro": [
            surfpro_checkpoint,
            surfpro_repo / "fairseq" / "models" / "surface_protein_model.py",
            REPO_ROOT / "vendor" / "sru" / "sru" / "__init__.py",
            REPO_ROOT
            / "vendor"
            / "protein_sequence_models"
            / "sequence_models"
            / "__init__.py",
            *([surfpro_python] if surfpro_python is not None else []),
            *([surfpro_msms] if surfpro_msms is not None else []),
            *(
                [surfpro_pdb_to_xyzrn]
                if surfpro_pdb_to_xyzrn is not None
                else []
            ),
            *(
                [surfpro_pdb_to_xyzrn.parent / "atmtypenumbers"]
                if surfpro_pdb_to_xyzrn is not None
                else []
            ),
        ],
        "rf3": [rf3_checkpoint] if rf3_checkpoint is not None else [],
        "rfd3": [rfd3_checkpoint] if rfd3_checkpoint is not None else [],
        "carbondesign": (
            [
                carbondesign_repo / "params" / "carbondesign_default.ckpt",
                carbondesign_repo / "config" / "config_model_mrf_pair_enable_esm_sc.json",
                carbondesign_repo / "config" / "config_data_mrf2.json",
            ]
            if carbondesign_repo is not None
            else []
        ),
        "rosetta": [path for path in (rosetta_root, molfile_to_params) if path is not None],
        "haddock3": [haddock3_bin] if haddock3_bin is not None else [],
        "autodock-gpu": [path for path in (autodock_gpu, autogrid4_bin) if path is not None],
        "ananas": [ananas_bin] if ananas_bin is not None else [],
    }
    for name, required_files in HF_SNAPSHOT_REQUIRED_FILES.items():
        snapshot = _hf_snapshot_path(hf_cache, name)
        markers[name] = [
            *(snapshot / rel_path for rel_path in required_files),
            hf_snapshot_manifest_path(hf_cache, name),
        ]
    markers["esmfold2"] = [
        *markers["esmfold2"],
        *markers["esmfold2-esmc"],
    ]
    return markers


def _preflight_selected_resources(
    *,
    selected: Iterable[str],
    markers: dict[str, list[Path]],
    download_assets: bool,
    dry_run: bool,
    skip_submodules: bool,
    allow_missing_af3: bool,
    opendde_assets: Path,
    rapidock_repo: Path,
    rapidock_weights: Path,
    rf3_repo: Path | None,
    rf3_checkpoint: Path | None,
    rfd3_checkpoint: Path | None,
    rf3_python: Path | None,
    carbondesign_repo: Path | None,
    surfdesign_python: Path | None,
    surfdesign_experiment: Path | None,
    surfpro_python: Path | None,
    surfpro_msms: Path | None,
    surfpro_pdb_to_xyzrn: Path | None,
    rosetta_root: Path | None,
    molfile_to_params: Path | None,
    haddock3_wheel: Path | None,
    haddock3_bin: Path | None,
    autodock_gpu: Path | None,
    autogrid4_bin: Path | None,
    ananas_bin: Path | None,
) -> None:
    selected_set = set(selected)
    if "alphafold3" in selected_set and not allow_missing_af3:
        for path in markers["alphafold3"]:
            _require_path(path, label="AF3 weights")

    if "rapidock" in selected_set:
        _require_path(rapidock_repo, label="RAPiDock repository")
    if selected_set & {"rf3", "rfd3"}:
        _require_path(rf3_repo, label="Foundry RF3/RFD3 repository")
        _require_path(rf3_python, label="Foundry RF3/RFD3 Python", executable=True)
    if "carbondesign" in selected_set:
        _require_path(carbondesign_repo, label="CarbonDesign repository")
        assert carbondesign_repo is not None
        for rel_path in (
            "params/carbondesign_default.ckpt",
            "config/config_model_mrf_pair_enable_esm_sc.json",
            "config/config_data_mrf2.json",
        ):
            _require_path(carbondesign_repo / rel_path, label=f"CarbonDesign {rel_path}")
    if "surfdesign" in selected_set:
        _require_path(
            surfdesign_python,
            label="SurfDesign Python",
            executable=True,
        )
        _require_path(surfdesign_experiment, label="SurfDesign experiment")
        assert surfdesign_experiment is not None
        _require_path(
            surfdesign_experiment / ".hydra" / "config.yaml",
            label="SurfDesign experiment config",
        )
        _require_path(
            surfdesign_experiment / "checkpoints" / "best.ckpt",
            label="SurfDesign checkpoint",
        )
        if skip_submodules:
            _require_path(
                markers["surfdesign"][2],
                label="SurfDesign source",
            )
    if "surfpro" in selected_set:
        _require_path(surfpro_python, label="SurfPro Python", executable=True)
        _require_path(surfpro_msms, label="SurfPro MSMS", executable=True)
        _require_path(
            surfpro_pdb_to_xyzrn,
            label="SurfPro pdb_to_xyzrn",
            executable=True,
        )
        assert surfpro_pdb_to_xyzrn is not None
        if surfpro_pdb_to_xyzrn.name != "pdb_to_xyzrn":
            raise SystemExit(
                "SurfPro requires the annotated pdb_to_xyzrn converter, not pdb_to_xyzr: "
                f"{surfpro_pdb_to_xyzrn}"
            )
        _require_path(
            surfpro_pdb_to_xyzrn.parent / "atmtypenumbers",
            label="SurfPro atmtypenumbers",
        )
        if skip_submodules:
            for source, label in (
                (
                    markers["surfpro"][1],
                    "SurfPro source",
                ),
                (
                    REPO_ROOT / "vendor" / "sru" / "sru" / "__init__.py",
                    "SurfPro SRU source",
                ),
                (
                    REPO_ROOT
                    / "vendor"
                    / "protein_sequence_models"
                    / "sequence_models"
                    / "__init__.py",
                    "SurfPro sequence-models source",
                ),
            ):
                _require_path(source, label=label)
    if "haddock3" in selected_set:
        if haddock3_bin is not None:
            _require_path(haddock3_bin, label="HADDOCK3 executable", executable=True)
        elif haddock3_wheel is not None:
            _require_path(haddock3_wheel, label="HADDOCK3 wheel")
        else:
            raise SystemExit("Missing required resource path: HADDOCK3 wheel or executable")
    if "autodock-gpu" in selected_set:
        _require_path(autodock_gpu, label="AutoDock-GPU executable", executable=True)
        _require_path(autogrid4_bin, label="AutoGrid4 executable", executable=True)
    if download_assets or dry_run:
        return
    labels = {
        "alphafold2": "AF2 data dir",
        "ligandmpnn": "LigandMPNN params",
        "protenix": "Protenix assets",
        "boltz": "Boltz cache",
        "chai1": "Chai-1 cache",
        "esm2": "ESM2 checkpoint",
        "esmif": "ESM-IF checkpoint",
        "esmfold": "ESMFold checkpoint",
        "esm3": "ESM3 snapshot",
        "dplm": "DPLM snapshot",
        "esmfold2": "ESMFold2 snapshot",
        "rhofold": "RhoFold checkpoint",
        "surfpro": "SurfPro runtime/checkpoint",
    }
    missing = [
        f"{labels[name]}:{path}"
        for name in selected_set & DOWNLOADABLE_ASSET_SOFTWARE
        for path in markers.get(name, ())
        if not _marker_ready(path)
    ]
    empty = [
        name
        for name in selected_set & DOWNLOADABLE_ASSET_SOFTWARE
        if not markers.get(name)
    ]
    if missing or empty:
        raise SystemExit(
            "Selected software has missing or empty resources: "
            + ", ".join((*sorted(empty), *sorted(missing)))
            + "; pass --download-assets to fetch public/gated assets."
        )


def _resolve_hf_token(hf_cache: Path) -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(key, "").strip()
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            return token
    token_paths = []
    if os.environ.get("HF_TOKEN_PATH"):
        token_paths.append(_expand(os.environ["HF_TOKEN_PATH"]))
    token_paths.extend(
        (
            hf_cache / "token",
            Path.home() / ".cache" / "huggingface" / "token",
        )
    )
    for path in token_paths:
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if token:
            os.environ["HF_TOKEN"] = token
            return token
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    token = (get_token() or "").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        return token
    return None


def _preflight_hf_credentials(
    *,
    selected: Iterable[str],
    markers: dict[str, list[Path]],
    hf_cache: Path,
    download_assets: bool,
    dry_run: bool,
) -> None:
    if dry_run or not download_assets:
        return
    gated_downloads = []
    for name in selected:
        if name not in GATED_HF_SOFTWARE:
            continue
        try:
            _validate_selected_assets((name,), markers)
        except (RuntimeError, SystemExit):
            gated_downloads.append(name)
    if gated_downloads and _resolve_hf_token(hf_cache) is None:
        raise SystemExit(
            "Hugging Face credential is required before downloading gated software "
            f"{','.join(gated_downloads)}; set HF_TOKEN, run `hf auth login`, or place "
            f"the active token at {hf_cache / 'token'}."
        )


def _expected_marker_sha256(
    name: str, marker: Path, marker_index: int
) -> str | None:
    if name == "alphafold2":
        spec = AF2_PARAM_SPECS.get(marker.name)
        return spec[1] if spec is not None else None
    if name == "ligandmpnn":
        return LIGANDMPNN_SHA256.get(marker.name)
    if name == "protenix":
        return tuple(PROTENIX_ASSET_SHA256.values())[marker_index]
    if name == "boltz":
        return BOLTZ_ASSET_SHA256.get(marker.name)
    if name == "chai1":
        if marker.name == "conformers_v1.apkl":
            return CHAI_CONFORMERS_SHA256
        if marker.name == "traced_sdpa_esm2_t36_3B_UR50D_fp16.pt":
            return CHAI_ESM_SHA256
        component = CHAI_COMPONENTS.get(marker.name)
        return component[1] if component is not None else None
    if name == "rhofold":
        return RHOFOLD_CHECKPOINT_SHA256
    if name == "opendde":
        by_name = {Path(rel_path).name: asset[2] for rel_path, asset in OPENDDE_ASSETS.items()}
        return by_name.get(marker.name)
    if name == "rapidock":
        asset = RAPIDOCK_ASSETS.get(marker.name)
        return asset[2] if asset is not None else None
    if name == "rf3":
        return RF3_ASSET[2]
    if name == "rfd3":
        return RFD3_ASSET[2]
    if name == "surfpro":
        actual = _sha256_file(marker) if marker.is_file() else None
        approved = {asset[2] for asset in SURFPRO_ASSETS.values()}
        return actual if actual in approved else None
    return None


def _validate_sif_build_contract(
    metadata: dict[str, object], *, expected_features: list[str]
) -> None:
    contract = metadata.get("build_contract")
    fingerprint = metadata.get("build_fingerprint")
    if not isinstance(contract, dict) or not isinstance(fingerprint, str):
        raise SystemExit("SIF build metadata is missing its build contract/fingerprint")
    calculated = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if calculated != fingerprint:
        raise SystemExit(
            "SIF build metadata contract fingerprint mismatch: "
            f"recorded={fingerprint} calculated={calculated}"
        )
    if contract.get("schema_version") != 5:
        raise SystemExit("SIF build contract schema_version must be 5")
    if contract.get("build_features") != expected_features:
        raise SystemExit("SIF build contract features do not match selected image features")
    feature_licenses = {
        "ananas": "ananas-academic",
        "pyrosetta": "pyrosetta-license",
        "rosetta": "rosetta-license",
    }
    expected_licenses = sorted(
        feature_licenses[feature]
        for feature in expected_features
        if feature in feature_licenses
    )
    if contract.get("accepted_licenses") != expected_licenses:
        raise SystemExit("SIF build contract license IDs do not match image features")
    for key in ("context_sha256", "definition_sha256"):
        value = contract.get(key)
        if not isinstance(value, str) or not _valid_package_digest(value, length=64):
            raise SystemExit(f"SIF build contract has invalid {key}")
    external_inputs = contract.get("external_inputs")
    if not isinstance(external_inputs, list):
        raise SystemExit("SIF build contract has invalid external_inputs")
    for entry in external_inputs:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("size"), int)
            or entry["size"] < 1
            or not isinstance(entry.get("sha256"), str)
            or not _valid_package_digest(entry["sha256"], length=64)
        ):
            raise SystemExit("SIF build contract has an invalid external input record")


def _sif_identity(
    sif: Path,
    *,
    build_features: Iterable[str] | None,
    require_build_metadata: bool = False,
) -> dict[str, object]:
    try:
        validate_file(sif, minimum_size=1_000_000, use_adjacent_stamp=False)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    actual_sha256 = _sha256_file(sif)
    expected_features = None if build_features is None else sorted(set(build_features))
    sidecar = Path(str(sif) + ".sha256")
    metadata_path = Path(str(sif) + ".build.json")
    if sidecar.exists() and not sidecar.is_file():
        raise SystemExit(f"SIF checksum sidecar is not a regular file: {sidecar}")
    if metadata_path.exists() and not metadata_path.is_file():
        raise SystemExit(f"SIF build metadata is not a regular file: {metadata_path}")
    if sidecar.is_file():
        try:
            fields = sidecar.read_text(encoding="ascii").strip().split()
        except (OSError, UnicodeDecodeError) as exc:
            raise SystemExit(f"Invalid SIF checksum sidecar {sidecar}: {exc}") from exc
        if not fields or not _valid_package_digest(fields[0], length=64):
            raise SystemExit(f"Invalid SIF checksum sidecar: {sidecar}")
        if fields[0] != actual_sha256:
            raise SystemExit(
                f"SIF SHA256 mismatch for {sif}: {actual_sha256}; expected {fields[0]}"
            )
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Invalid SIF build metadata {metadata_path}: {exc}") from exc
        if not isinstance(metadata, dict):
            raise SystemExit(
                f"SIF build metadata does not match image/selection: {metadata_path}"
            )
        recorded_features = metadata.get("build_features")
        if expected_features is None:
            if not isinstance(recorded_features, list) or not all(
                isinstance(feature, str) for feature in recorded_features
            ):
                raise SystemExit(
                    f"SIF build metadata does not match image/selection: {metadata_path}"
                )
            expected_features = sorted(set(recorded_features))
        if (
            metadata.get("sha256") != actual_sha256
            or metadata.get("size") != sif.stat().st_size
            or metadata.get("build_features") != expected_features
        ):
            raise SystemExit(
                f"SIF build metadata does not match image/selection: {metadata_path}"
            )
        _validate_sif_build_contract(metadata, expected_features=expected_features)
        feature_metadata = str(metadata_path)
    elif require_build_metadata or expected_features:
        raise SystemExit(
            "Prebuilt or feature-selected image requires adjacent schema-v5 "
            f"build metadata: {metadata_path}"
        )
    else:
        if expected_features is None:
            expected_features = []
        feature_metadata = None
    if not sidecar.exists():
        tmp = sidecar.with_name(f"{sidecar.name}.{os.getpid()}.tmp")
        tmp.write_text(f"{actual_sha256}  {sif.name}\n", encoding="ascii")
        os.replace(tmp, sidecar)
    assert expected_features is not None
    return {
        "path": str(sif),
        "size": sif.stat().st_size,
        "sha256": actual_sha256,
        "build_features": expected_features,
        "build_metadata": feature_metadata,
    }


def _acquire_software_manifest_lock(lock_handle, lock_path: Path) -> None:
    deadline = time.monotonic() + SOFTWARE_MANIFEST_LOCK_TIMEOUT_SECONDS
    while True:
        try:
            fcntl.flock(
                lock_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
            return
        except BlockingIOError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SystemExit(
                    f"Timed out waiting for software manifest lock after "
                    f"{SOFTWARE_MANIFEST_LOCK_TIMEOUT_SECONDS:g}s: {lock_path}"
                )
            time.sleep(min(0.05, remaining))


def _write_software_manifest(
    path: Path,
    *,
    selected: Iterable[str],
    markers: dict[str, list[Path]],
    accepted_licenses: Iterable[str] = (),
    dry_run: bool,
    container_identity: dict[str, object] | None = None,
    selected_asset_variants: dict[str, str] | None = None,
) -> None:
    selected = tuple(selected)
    if dry_run:
        print(f"+ write software manifest {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+b") as lock_handle:
        _acquire_software_manifest_lock(lock_handle, lock_path)
        try:
            if path.is_file():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid existing software manifest {path}: {exc}") from exc
                if not isinstance(payload.get("software"), dict):
                    raise SystemExit(f"Invalid existing software manifest: {path}")
            else:
                payload = {"software": {}}
            previous_licenses = set(payload.get("accepted_licenses", ()))
            payload.update(
                {
                    "schema_version": 4,
                    "selected": list(selected),
                    "last_selection": list(selected),
                    "accepted_licenses": sorted(previous_licenses | set(accepted_licenses)),
                    "scope": "cumulative asset inventory; ready does not imply model smoke passed",
                    "container": container_identity,
                }
            )
            for name in selected:
                spec = SOFTWARE[name]
                files = []
                if name in {"pyrosetta", "ananas", "rosetta"} and container_identity is not None:
                    files.append(
                        {
                            "path": container_identity["path"],
                            "size": container_identity["size"],
                            "sha256": container_identity["sha256"],
                            "ready": name
                            in set(container_identity.get("build_features", ()))
                            and container_identity.get("import_smoke_passed") is True,
                            "kind": "sif-build-feature",
                        }
                    )
                for marker_index, marker in enumerate(markers.get(name, [])):
                    exists = _marker_ready(marker)
                    entry = {"path": str(marker), "ready": exists}
                    if marker.is_file():
                        entry["size"] = marker.stat().st_size
                        expected_sha256 = _expected_marker_sha256(
                            name, marker, marker_index
                        )
                        if expected_sha256 is not None:
                            entry["sha256"] = expected_sha256
                    files.append(entry)
                software_entry = {
                    "asset_policy": spec.asset_policy,
                    "asset_source": spec.asset_source,
                    "asset_urls": list(spec.asset_urls),
                    "license_id": spec.license_id,
                    "revision": spec.revision,
                    "requires": list(spec.requires),
                    "host_envs": list(spec.host_envs),
                    "submodules": list(spec.submodules),
                    "ready": (
                        not files
                        and spec.asset_policy in {"container", "bundled-build", "bundled-binary"}
                    )
                    or (bool(files) and all(item["ready"] for item in files)),
                    "resources": files,
                }
                variant = (selected_asset_variants or {}).get(name)
                if name == "surfpro" and variant is not None:
                    url, exact_size, expected_sha256 = SURFPRO_ASSETS[variant]
                    software_entry["selected_asset"] = {
                        "variant": variant,
                        "url": url,
                        "size": exact_size,
                        "sha256": expected_sha256,
                    }
                payload["software"][name] = software_entry
            payload["installed"] = sorted(
                name
                for name, entry in payload["software"].items()
                if isinstance(entry, dict) and entry.get("ready") is True
            )
            _write_json(path, payload)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    print(f"Wrote software manifest: {path}")


def _run_host_env_setup(
    args: argparse.Namespace,
    *,
    root: Path,
    only_envs: Iterable[str] | None = None,
    env_overrides: dict[str, Path | None] | None = None,
) -> None:
    if not args.setup_host_envs:
        return
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "installation" / "env_setup.py"),
        "--root-prefix",
        str(root),
        "--profile",
        args.host_env_profile,
        "--jobs",
        str(args.host_env_jobs),
    ]
    if args.host_env_yes:
        cmd.append("-y")
    if args.clone_root:
        cmd.extend(["--clone-root", str(_expand(args.clone_root))])
    if args.host_env_skip_commands:
        cmd.append("--skip-commands")
    selected_envs = tuple(only_envs) if only_envs is not None else None
    if selected_envs == ():
        print("No selected software requires a host environment; skipped env_setup.py.")
        return
    if selected_envs:
        cmd.extend(["--only", ",".join(selected_envs)])
    original_env = {key: os.environ.get(key) for key in (env_overrides or {})}
    try:
        for key, value in (env_overrides or {}).items():
            if value is not None:
                os.environ[key] = str(value)
        _run(cmd, cwd=REPO_ROOT, dry_run=args.dry_run)
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _clean_dedicated_bootstrap_cache(root: Path, *, dry_run: bool) -> None:
    envs_root = root / "envs"
    if not envs_root.is_dir():
        return
    environments = {
        path.name
        for path in envs_root.iterdir()
        if path.is_dir() or path.is_symlink()
    }
    if environments != {APPTAINER_BOOTSTRAP_ENV}:
        return
    package_cache = root / "pkgs"
    micromamba = root / ".bin" / "micromamba"
    if not micromamba.is_file():
        system_micromamba = shutil.which("micromamba")
        if system_micromamba:
            micromamba = Path(system_micromamba)
    if not package_cache.is_dir() or not micromamba.is_file():
        return
    try:
        package_cache.resolve().relative_to(root.resolve())
    except ValueError:
        raise SystemExit(
            f"Refusing to clean package cache outside bootstrap root: {package_cache}"
        )
    cmd = [
        str(micromamba),
        "--root-prefix",
        str(root),
        "clean",
        "--all",
        "--force-pkgs-dirs",
        "--yes",
    ]
    if dry_run:
        _run(cmd, dry_run=True)
        return
    lock_path = _apptainer_env_lock_path(root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        _acquire_apptainer_env_lock(lock_handle, lock_path)
        try:
            _run(cmd, dry_run=False)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _ensure_runtime_cache(root: Path) -> Path:
    runtime_cache = root / "cache" / "runtime"
    for _, dirname in RUNTIME_CACHE_DIRS:
        (runtime_cache / dirname).mkdir(parents=True, exist_ok=True)
    return runtime_cache


def _validate_runner_bind_sources(binds: Iterable[str]) -> None:
    missing = []
    for bind in binds:
        host_path = Path(bind.partition(":")[0])
        if not host_path.exists():
            missing.append(str(host_path))
    if missing:
        raise SystemExit(
            "Runner bind source missing after bootstrap cleanup: "
            + ", ".join(sorted(set(missing)))
        )


def _ensure_haddock3_from_wheel(
    *,
    root: Path,
    wheel: Path | None,
    dry_run: bool,
) -> Path | None:
    if wheel is None:
        return None
    _require_path(wheel, label="HADDOCK3 wheel")
    env_prefix = root / "envs" / "haddock3"
    haddock3_bin = env_prefix / "bin" / "haddock3"
    if haddock3_bin.exists():
        return haddock3_bin

    micromamba = shutil.which("micromamba")
    if not micromamba:
        micromamba = _bootstrap_micromamba(root / ".bin", dry_run=dry_run)
    _run(
        [
            micromamba,
            "create",
            "-y",
            "-n",
            "haddock3",
            "--root-prefix",
            str(root),
            "-c",
            "conda-forge",
            "python=3.11",
            "pip",
        ],
        dry_run=dry_run,
    )
    _run(
        [
            micromamba,
            "run",
            "-n",
            "haddock3",
            "--root-prefix",
            str(root),
            "python",
            "-m",
            "pip",
            "install",
            str(wheel),
        ],
        dry_run=dry_run,
    )
    if not dry_run:
        _require_path(haddock3_bin, label="HADDOCK3 installed binary", executable=True)
    return haddock3_bin


def _write_site_profile(
    path: Path,
    *,
    profile: str,
    apptainer_bin: str,
    sif: Path,
    runner: Path,
    binds: list[str],
    env_defaults: dict[str, str],
    env_prepends: list[tuple[str, str]],
    container_repo_root: str,
    container_python: str,
    use_nv: bool,
    runner_backend: str,
    docker_bin: str | None,
    docker_image: str,
) -> None:
    payload = {
        "schema_version": 1,
        "profile": profile,
        "apptainer_bin": apptainer_bin,
        "runner_backend": runner_backend,
        "docker_bin": docker_bin,
        "docker_image": docker_image,
        "sif": str(sif),
        "runner": str(runner),
        "container_repo_root": container_repo_root,
        "container_python": container_python,
        "use_nv": use_nv,
        "binds": binds,
        "env_defaults": env_defaults,
        "env_prepends": [{"key": key, "value": value} for key, value in env_prepends],
    }
    _write_json(path, payload)
    print(f"Wrote site profile: {path}")


def _write_runner_from_profile(profile_path: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"+ write runner from profile {profile_path}")
        return
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import apptainer_build

    apptainer_build.write_python_runner(
        profile["runner"],
        apptainer=profile["apptainer_bin"],
        runner_backend=profile.get("runner_backend", "apptainer"),
        docker_bin=profile.get("docker_bin") or "docker",
        docker_image=profile.get("docker_image") or apptainer_build.DEFAULT_DOCKER_APPTAINER_IMAGE,
        sif_path=profile["sif"],
        container_repo_root=profile["container_repo_root"],
        binds=profile["binds"],
        env_defaults=profile["env_defaults"],
        env_prepends=[(x["key"], x["value"]) for x in profile["env_prepends"]],
        container_python=profile["container_python"],
        use_nv=bool(profile["use_nv"]),
    )


def _write_checkout_activation(runner: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"+ activate checkout with runner {runner}")
        return
    if not runner.is_file() or not os.access(str(runner), os.X_OK):
        raise SystemExit(f"Cannot activate checkout with a missing or non-executable runner: {runner}")

    config_env = REPO_ROOT / "config" / "env"
    env_local = config_env / "env_local.yaml"
    hint = config_env / ".astrolabe_python_path"
    container_env = REPO_ROOT / "config" / "env_container.yaml"
    if not config_env.is_dir():
        raise SystemExit(f"Checkout env config directory is missing: {config_env}")
    if not container_env.is_file():
        raise SystemExit(f"Checkout container env config is missing: {container_env}")

    resolved_runner = str(runner.resolve())
    env_local_tmp = config_env / f".{env_local.name}.{os.getpid()}.tmp"
    hint_tmp = config_env / f".{hint.name}.{os.getpid()}.tmp"
    try:
        env_local_tmp.unlink(missing_ok=True)
        hint_tmp.unlink(missing_ok=True)
        env_local_tmp.symlink_to("../env_container.yaml")
        hint_tmp.write_text(resolved_runner + "\n", encoding="utf-8")
        os.replace(env_local_tmp, env_local)
        os.replace(hint_tmp, hint)
    finally:
        env_local_tmp.unlink(missing_ok=True)
        hint_tmp.unlink(missing_ok=True)
    print(f"Activated checkout env: {env_local} -> ../env_container.yaml")
    print(f"Activated checkout runner: {hint} -> {resolved_runner}")


def _build_sif(
    args: argparse.Namespace,
    *,
    apptainer_bin: str,
    sif: Path,
    root: Path,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "containers" / "apptainer_build.py"),
        "--output",
        str(sif),
        "--builder",
        args.container_builder,
        "--build-cache",
        str(args.container_build_cache),
        "--state-root",
        str(root / ".gh_container_setup" / sif.stem),
    ]
    if args.container_builder == "apptainer":
        cmd.extend(["--apptainer-bin", apptainer_bin])
    else:
        cmd.extend(
            [
                "--docker-bin",
                args.docker_bin_resolved,
                "--docker-builder-image",
                args.docker_builder_image,
                "--docker-network",
                args.docker_network,
                "--docker-base-source",
                args.docker_base_source,
            ]
        )
        if args.docker_base_image:
            cmd.extend(["--docker-base-image", args.docker_base_image])
    if args.fakeroot:
        cmd.append("--fakeroot")
        if args.fakeroot_bin_resolved:
            cmd.extend(["--fakeroot-bin", args.fakeroot_bin_resolved])
    for build_input in args.af3_build_data_inputs:
        cmd.extend(["--build-input", str(build_input)])
    for feature in args.container_build_features:
        cmd.extend(["--build-feature", feature])
        feature_license = {
            "pyrosetta": "pyrosetta-license",
            "ananas": "ananas-academic",
            "rosetta": "rosetta-license",
        }.get(feature)
        if feature_license:
            cmd.extend(["--accept-license", feature_license])
    if "pyrosetta" not in set(args.container_build_features):
        cmd.append("--without-pyrosetta")
    if args.remote:
        cmd.append("--remote")
    if args.force:
        cmd.append("--force")
    _run(cmd, cwd=REPO_ROOT, dry_run=args.dry_run)


def _run_doctor(args: argparse.Namespace, runner: Path) -> None:
    cmd = [
        str(runner),
        "-c",
        (
            "import sys; "
            "sys.path.insert(0, '/opt/Astrolabe'); "
            "from astrolabe.cli import main; "
            "main(['env','doctor','--task',%r,'--env','config/env_container.yaml','--resolve','--python-deps'])"
            % args.doctor_task
        ),
    ]
    if args.srun:
        cmd = ["bash", "-lc", shlex.quote(args.srun) + " " + shlex.quote(_quote_cmd(cmd))]
    _run(cmd, cwd=REPO_ROOT, dry_run=args.dry_run)


def _run_import_smoke(
    args: argparse.Namespace,
    runner: Path,
    selected: Iterable[str] | None = None,
) -> None:
    modules = CONTAINER_IMPORT_SMOKE_MODULES
    if selected is not None:
        resolved = ["astrolabe"]
        for name in selected:
            for module in SOFTWARE_IMPORT_SMOKE_MODULES.get(name, ()):
                if module not in resolved:
                    resolved.append(module)
        modules = tuple(resolved)
    modules_literal = repr(modules)
    code = (
        "import importlib, pathlib, sys; "
        "sys.path[:0]=['/opt/Astrolabe',"
        "'/opt/Astrolabe/vendor/alphafold2',"
        "'/opt/Astrolabe/vendor/dplm',"
        "'/opt/Astrolabe/vendor/dplm/src']; "
        f"mods={modules_literal}; "
        "loaded=[importlib.import_module(m) for m in mods]; "
        "gh_source=str(pathlib.Path(loaded[0].__file__).resolve()); "
        "assert gh_source.startswith('/opt/Astrolabe/'), gh_source; "
        "from astrolabe.app import _load_run_module; "
        "_load_run_module(); "
        "print('container source:', gh_source); "
        "print('container run graph import ok'); "
        "print('container import smoke ok:', ','.join(mods))"
    )
    if selected is not None and "esmfold" in set(selected):
        code += (
            "; import astrolabe.evaluators.esmfold as gh_esmfold"
            "; gh_esmfold._load_esmfold_model_class()"
            "; print('ESMFold implementation import ok')"
        )
    if (
        selected is not None
        and "pyrosetta" not in set(selected)
        and not getattr(args, "skip_build", False)
    ):
        code += (
            "; import importlib.util"
            "; assert importlib.util.find_spec('pyrosetta') is None, "
            "'core/non-PyRosetta SIF unexpectedly contains PyRosetta'"
            "; print('PyRosetta exclusion ok')"
        )
    cmd = [str(runner), "-c", code]
    if args.srun:
        cmd = ["bash", "-lc", shlex.quote(args.srun) + " " + shlex.quote(_quote_cmd(cmd))]
    _run(cmd, cwd=REPO_ROOT, dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bootstrap Astrolabe on the current server.")
    p.add_argument("--profile", default="local", help="Site profile name.")
    p.add_argument("--root", default=str(DEFAULT_ROOT), help="Bootstrap root, default: ~/.gh_envs.")
    p.add_argument("--apptainer-bin", default=None, help="Explicit apptainer/singularity executable.")
    p.add_argument(
        "--sif",
        default=None,
        help=(
            "SIF path. Default: <root>/images/Astrolabe_af3.sif; exact "
            "build features are recorded in adjacent metadata."
        ),
    )
    p.add_argument(
        "--container-build-cache",
        default=None,
        help=(
            "Persistent cache used during SIF builds, default: <root>/cache/container-build. "
            "It is mounted into the build at /cache so pip/micromamba can reuse target-server downloads."
        ),
    )
    p.add_argument(
        "--pyrosetta-wheel",
        default=None,
        help=(
            "Exact approved PyRosetta wheel to seed the persistent build cache. "
            "When omitted, bootstrap reuses the cached wheel or downloads the pinned "
            "official quarterly wheel with resume and SHA256 validation."
        ),
    )
    p.add_argument("--runner", default=None, help="Runner path, default: <root>/tools/gh_<profile>_container_python.py.")
    p.add_argument("--site-profile", default=None, help="Site profile JSON path, default: <root>/sites/<profile>.json.")
    p.add_argument(
        "--activate-checkout",
        action="store_true",
        help=(
            "After bootstrap validation succeeds, switch this checkout to container mode: "
            "point config/env/env_local.yaml at env_container.yaml and write the generated "
            "runner to config/env/.astrolabe_python_path."
        ),
    )
    p.add_argument("--container-repo-root", default=DEFAULT_CONTAINER_REPO_ROOT)
    p.add_argument("--container-python", default=DEFAULT_CONTAINER_PYTHON)
    p.add_argument("--no-nv", action="store_true", help="Generate a CPU/non-GPU runner.")
    p.add_argument("--skip-build", action="store_true", help="Do not build SIF; generate config/runner only.")
    p.add_argument("--force", action="store_true", help="Force SIF rebuild.")
    p.add_argument("--fakeroot", action="store_true", help="Build SIF with Apptainer --fakeroot.")
    p.add_argument("--remote", action="store_true", help="Build SIF with Apptainer --remote.")
    p.add_argument(
        "--container-builder",
        choices=("apptainer", "docker"),
        default="apptainer",
        help=(
            "SIF build backend. 'docker' runs the official Apptainer Docker image "
            "and is useful when the host has Docker but no usable Apptainer."
        ),
    )
    p.add_argument("--docker-bin", default=None, help="Explicit Docker executable.")
    p.add_argument(
        "--docker-builder-image",
        default=DEFAULT_DOCKER_APPTAINER_IMAGE,
        help="Docker image providing apptainer for --container-builder docker.",
    )
    p.add_argument(
        "--docker-network",
        default="host",
        help=(
            "Docker network mode for --container-builder docker. Default: host, "
            "so Apptainer downloads official OCI bases through the target server."
        ),
    )
    p.add_argument(
        "--docker-base-source",
        choices=("docker", "docker-archive"),
        default="docker",
        help=(
            "Base image source for --container-builder docker. Default: docker, "
            "which keeps the definition's official Bootstrap: docker/From: source. "
            "Use docker-archive only as an explicit site fallback."
        ),
    )
    p.add_argument(
        "--docker-base-image",
        default=None,
        help=(
            "Base OCI image for --docker-base-source docker-archive when "
            "--container-builder docker is used."
        ),
    )
    p.add_argument(
        "--runner-backend",
        choices=("apptainer", "docker-apptainer"),
        default="apptainer",
        help=(
            "Generated runner backend. 'docker-apptainer' runs Apptainer inside "
            "Docker for hosts where SIF exec cannot run directly."
        ),
    )
    p.add_argument(
        "--runner-docker-image",
        default=DEFAULT_DOCKER_APPTAINER_IMAGE,
        help="Docker image used by --runner-backend docker-apptainer.",
    )
    p.add_argument(
        "--bootstrap-apptainer",
        choices=("auto", "always", "never"),
        default="auto",
        help=(
            "When container steps need Apptainer/Singularity and none is in PATH, "
            "automatically create <root>/envs/_gh_apptainer with micromamba "
            "(default: auto). Use 'always' for the pinned managed engine even when "
            "a system engine exists, or 'never' to fail instead."
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="Print the plan without executing commands.")
    p.add_argument(
        "--software",
        action="append",
        default=[],
        metavar="NAME[,NAME...]",
        help=(
            "Select software explicitly. Repeat or use comma-separated names; "
            "profiles core, language, rna, docking, symmetry, surface, public and all are accepted. "
            "Default: core."
        ),
    )
    p.add_argument(
        "--list-software",
        action="store_true",
        help="List selectable software and exit without requiring a container engine.",
    )
    p.add_argument(
        "--list-software-json",
        action="store_true",
        help="Print the selectable software catalog as JSON and exit.",
    )
    p.add_argument(
        "--software-manifest",
        default=None,
        help="Asset audit JSON, default: <root>/assets/software-manifest.json.",
    )
    p.add_argument(
        "--download-assets",
        "--download-public-assets",
        dest="download_public_assets",
        action="store_true",
        default=True,
        help=(
            "Compatibility flag. Missing downloadable assets are fetched automatically "
            "for selected software; licensed and restricted assets still require "
            "explicit acceptance, credentials, or paths."
        ),
    )
    p.add_argument(
        "--asset-download-jobs",
        type=int,
        default=4,
        help="Concurrent public-asset downloads (default: 4).",
    )
    p.add_argument(
        "--accept-license",
        action="append",
        default=[],
        metavar="ID[,ID...]",
        help="Record explicit acceptance for gated assets (esm3-noncommercial, esmfold2-aup).",
    )
    p.add_argument(
        "--needed-softwares",
        choices=("runtime", "full"),
        default="runtime",
        help=(
            "Backward-compatible legacy selection. 'full' selects the historical "
            "external-tool set; prefer --software all for the strict catalog."
        ),
    )
    p.add_argument(
        "--setup-host-envs",
        action="store_true",
        help="Run scripts/installation/env_setup.py before writing the container site profile.",
    )
    p.add_argument(
        "--host-env-profile",
        choices=("runtime", "needed_softwares", "all"),
        default="needed_softwares",
        help="env_setup.py profile used by --setup-host-envs.",
    )
    p.add_argument(
        "--host-env-jobs",
        type=int,
        default=1,
        help="Parallelism for env_setup.py when --setup-host-envs is used.",
    )
    p.add_argument(
        "--host-env-yes",
        action="store_true",
        help="Pass -y to env_setup.py when --setup-host-envs is used.",
    )
    p.add_argument(
        "--host-env-skip-commands",
        action="store_true",
        help="Pass --skip-commands to env_setup.py when only env creation is desired.",
    )
    p.add_argument(
        "--clone-root",
        default=None,
        help="Clone root forwarded to env_setup.py when --setup-host-envs is used.",
    )
    p.add_argument("--skip-submodules", action="store_true", help="Skip git submodule update.")
    p.add_argument(
        "--submodule-set",
        choices=("runtime", "needed_softwares", "all"),
        default=None,
        help=(
            "Official submodule set to initialize. Defaults to runtime, or "
            "needed_softwares when --needed-softwares full is selected. Use "
            "'all' for optional/heavy development backends."
        ),
    )
    p.add_argument(
        "--submodule-depth",
        type=int,
        default=1,
        help="Depth for official submodule clones (default: 1; use 0 for full history).",
    )
    p.add_argument(
        "--submodule-jobs",
        type=int,
        default=4,
        help="Parallel jobs for git submodule update (default: 4).",
    )
    p.add_argument("--skip-import-smoke", action="store_true", help="Skip container Python import smoke.")
    p.add_argument("--skip-doctor", action="store_true", help="Skip final astrolabe env doctor validation.")
    p.add_argument("--doctor-task", default="container_smoke", help="Task used for final env doctor.")
    p.add_argument("--srun", default=None, help="Optional scheduler prefix, e.g. 'srun -p gpu --gres=gpu:h100:1'.")
    p.add_argument("--af3-weights", default=None, help="Required AF3 weights path unless --allow-missing-af3 is set.")
    p.add_argument(
        "--af3-ccd-pickle",
        default=None,
        help="Optional pre-generated AF3 ccd.pickle; otherwise reuse the gold gh env when available.",
    )
    p.add_argument(
        "--af3-chemical-component-sets-pickle",
        default=None,
        help="Optional AF3 chemical_component_sets.pickle paired with --af3-ccd-pickle.",
    )
    p.add_argument(
        "--af3-components-cif",
        default=None,
        help="Optional components.cif fallback when pre-generated AF3 pickles are unavailable.",
    )
    p.add_argument("--allow-missing-af3", action="store_true", help="Allow missing AF3 weights for non-AF3-only smoke setup.")
    p.add_argument("--af2-params", default=None, help="AF2 data dir or direct params/ dir.")
    p.add_argument("--ligandmpnn-params", default=None, help="LigandMPNN model_params dir.")
    p.add_argument("--protenix-assets", default=None, help="Protenix asset root containing release_model/release_data.")
    p.add_argument("--boltz-cache", default=None, help="Boltz model cache root to bind at /models/boltz.")
    p.add_argument("--chai-cache", default=None, help="Chai-1 downloads root to bind at /models/chai1.")
    p.add_argument("--torch-cache", default=None, help="Shared Torch Hub cache used by ESM2.")
    p.add_argument("--hf-cache", default=None, help="Shared Hugging Face cache used by ESM3/DPLM/ESMFold2.")
    p.add_argument(
        "--rhofold-assets",
        default=None,
        help="RhoFold checkpoint file or asset root containing checkpoint.pt.",
    )
    p.add_argument("--opendde-python", default=None, help="OpenDDE subprocess Python executable.")
    p.add_argument("--opendde-assets", default=None, help="OpenDDE checkpoint/database root.")
    p.add_argument("--esmfold2-python", default=None, help="Biohub ESMFold2 subprocess Python executable.")
    p.add_argument("--rapidock-python", default=None, help="RAPiDock subprocess Python executable.")
    p.add_argument("--rapidock-repo", default=None, help="RAPiDock repository path.")
    p.add_argument("--rapidock-weights", default=None, help="RAPiDock trained-model directory.")
    p.add_argument("--rf3-python", default=None, help="RF3 subprocess Python executable.")
    p.add_argument("--rf3-repo", default=None, help="External RF3/Foundry repository path.")
    p.add_argument("--rf3-checkpoint", default=None, help="RF3 checkpoint file.")
    p.add_argument("--rfd3-checkpoint", default=None, help="RFD3 initializer checkpoint file.")
    p.add_argument("--surfdesign-python", default=None, help="SurfDesign external Python executable.")
    p.add_argument("--surfdesign-repo", default=None, help="SurfDesign source repository path.")
    p.add_argument(
        "--surfdesign-experiment",
        default=None,
        help="SurfDesign trained experiment containing .hydra/config.yaml and checkpoints/best.ckpt.",
    )
    p.add_argument("--surfpro-python", default=None, help="SurfPro external Python executable.")
    p.add_argument("--surfpro-repo", default=None, help="SurfPro source repository path.")
    p.add_argument("--surfpro-msms", default=None, help="SurfPro MSMS executable.")
    p.add_argument(
        "--surfpro-pdb-to-xyzrn",
        default=None,
        help="SurfPro pdb_to_xyzrn executable paired with MSMS.",
    )
    p.add_argument(
        "--surfpro-model",
        choices=tuple(SURFPRO_ASSETS),
        default="cath42",
        help="Official SurfPro checkpoint variant to download (default: cath42).",
    )
    p.add_argument(
        "--surfpro-checkpoint",
        default=None,
        help="SurfPro checkpoint file or directory containing checkpoint_best.pt.",
    )
    p.add_argument("--carbondesign-repo", default=None, help="CarbonDesign repository with params/config assets.")
    p.add_argument("--rosetta-root", default=None, help="Rosetta public/tool root to bind at /rosetta.")
    p.add_argument("--molfile-to-params", default=None, help="Explicit Rosetta molfile_to_params.py path.")
    p.add_argument("--haddock3-bin", default=None, help="HADDOCK3 executable to bind at /usr/bin/haddock3.")
    p.add_argument("--haddock3-wheel", default=None, help="HADDOCK3 wheel to install into <root>/envs/haddock3.")
    p.add_argument("--autodock-gpu", default=None, help="AutoDock-GPU executable.")
    p.add_argument("--autogrid4-bin", default=None, help="AutoGrid4 executable.")
    p.add_argument("--dssp-bin", default=None, help="Optional DSSP/mkdssp executable override.")
    p.add_argument("--usalign-bin", default=None, help="Optional USalign executable override.")
    p.add_argument("--obabel-bin", default=None, help="Optional OpenBabel executable override.")
    p.add_argument("--vina-bin", default=None, help="Optional Vina executable override.")
    p.add_argument("--ananas-bin", default=None, help="AnAnaS executable.")
    p.add_argument("--meeko-python", default=None, help="Optional Meeko helper Python executable.")
    p.add_argument("--meeko-pythonpath", default=None, help="Optional Meeko helper PYTHONPATH.")
    p.add_argument("--dplm-python", default=None, help="DPLM subprocess Python executable.")
    p.add_argument("--dplm-repo", default=None, help="DPLM repository path inside the container.")
    p.add_argument("--nampnn-python", default=None, help="NA-MPNN subprocess Python executable.")
    p.add_argument("--nampnn-repo", default=None, help="NA-MPNN repository path inside the container.")
    p.add_argument("--rpxdock-python", default=None, help="RPXDock subprocess Python executable.")
    p.add_argument("--rpxdock-repo", default=None, help="RPXDock repository path inside the container.")
    p.add_argument("--willutil-repo", default=None, help="WillUtil repository path inside the container.")
    p.add_argument("--libgfortran3", default=None, help="Directory containing real libgfortran.so.3.")
    p.add_argument("--extra-bind", action="append", default=[], metavar="HOST:CONTAINER")
    p.add_argument("--env-default", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--prepend-env", action="append", default=[], metavar="KEY=VALUE")
    return p


def _parse_assignment(raw: str, option_name: str) -> tuple[str, str]:
    if "=" not in raw:
        raise SystemExit(f"{option_name} expects KEY=VALUE, got: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise SystemExit(f"{option_name} expects a non-empty KEY, got: {raw}")
    return key, value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_software or args.list_software_json:
        print_catalog(as_json=bool(args.list_software_json))
        return 0
    try:
        selected_software = resolve_software(args.software)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    full_needed = args.needed_softwares == "full"
    software_selection_requested = bool(args.software)
    if full_needed and software_selection_requested:
        raise SystemExit("--needed-softwares full and --software are alternative selection modes; choose one.")
    if not selected_software:
        if full_needed:
            selected_software = resolve_software(LEGACY_FULL_SOFTWARE)
        else:
            selected_software = resolve_software(("core",))
    # Once defaults/profiles are resolved, every plan has exact selection semantics.
    software_selection_explicit = True
    selected_set = set(selected_software)
    accepted_licenses = {
        token.strip().lower()
        for raw in args.accept_license
        for token in raw.split(",")
        if token.strip()
    }
    selected_license_requirements = {
        "pyrosetta": "pyrosetta-license",
        "ananas": "ananas-academic",
        "rosetta": "rosetta-license",
    }
    missing_selected_licenses = [
        license_id
        for name, license_id in selected_license_requirements.items()
        if name in selected_set and license_id not in accepted_licenses
    ]
    if args.download_public_assets and software_selection_explicit:
        license_requirements = {
            "esm3": "esm3-noncommercial",
            "esmfold2": "esmfold2-aup",
        }
        missing_acceptance = [
            license_id
            for name, license_id in license_requirements.items()
            if name in selected_set and license_id not in accepted_licenses
        ]
        if missing_acceptance:
            raise SystemExit(
                "Explicit license acceptance required before download: "
                + ", ".join(f"--accept-license {item}" for item in missing_acceptance)
            )
    root = _expand(args.root)
    default_sif_name = "Astrolabe_af3.sif"
    sif = _expand(args.sif) if args.sif else root / "images" / default_sif_name
    args.container_build_cache = (
        _expand(args.container_build_cache)
        if args.container_build_cache
        else root / "cache" / "container-build"
    )
    runner = _expand(args.runner) if args.runner else root / "tools" / f"gh_{args.profile}_container_python.py"
    site_profile = _expand(args.site_profile) if args.site_profile else root / "sites" / f"{args.profile}.json"
    software_manifest = (
        _expand(args.software_manifest)
        if args.software_manifest
        else root / "assets" / "software-manifest.json"
    )
    if args.container_builder == "docker" and args.remote:
        raise SystemExit(
            "--container-builder docker runs a local privileged Docker build; do not combine it with --remote."
        )
    if args.remote and args.fakeroot:
        raise SystemExit("--remote and --fakeroot are mutually exclusive build modes.")
    args.docker_bin_resolved = None
    args.fakeroot_bin_resolved = None
    args.af3_build_data_inputs = []
    args.container_build_features = tuple(
        feature
        for feature in ("pyrosetta", "ananas", "rosetta")
        if feature in selected_set
    )

    container_prefetch_selected = bool(
        args.download_public_assets
        and software_selection_explicit
        and selected_set & CONTAINER_PREFETCH_SOFTWARE
    )
    runner_needed = (
        (not args.skip_build)
        or (not args.skip_import_smoke)
        or (not args.skip_doctor)
        or container_prefetch_selected
    )
    if args.activate_checkout and not runner_needed:
        raise SystemExit("--activate-checkout requires a generated container runner.")
    docker_needed = (
        (not args.skip_build and args.container_builder == "docker")
        or (runner_needed and args.runner_backend == "docker-apptainer")
    )
    host_apptainer_needed = (
        (not args.skip_build and args.container_builder == "apptainer")
        or (runner_needed and args.runner_backend == "apptainer")
    )
    af3_weights, af3_from_env = _expand_arg_or_env(args.af3_weights, "GH_WEIGHT_AF3")
    af3_required = "alphafold3" in selected_set and not args.allow_missing_af3
    if af3_weights is None and af3_required:
        raise SystemExit("--af3-weights is required unless --allow-missing-af3 is set.")
    prebuilt_container_identity = None
    if args.skip_build and runner_needed and not args.dry_run:
        _require_path(sif, label="prebuilt SIF")
        prebuilt_container_identity = _sif_identity(
            sif,
            build_features=None,
            require_build_metadata=True,
        )
    af2_params, af2_from_env = _expand_arg_or_env(args.af2_params, "GH_WEIGHT_AF2")
    af2_data_dir = _normalize_af2_data_dir(str(af2_params)) if af2_params else None
    if af2_data_dir is None and (
        "alphafold2" in selected_set and (software_selection_explicit or args.download_public_assets)
    ):
        af2_data_dir = root / "assets" / "alphafold2"
    ligandmpnn_dir, ligandmpnn_from_env = _expand_arg_or_env(
        args.ligandmpnn_params,
        "GH_WEIGHT_LIGANDMPNN",
    )
    if ligandmpnn_dir is None:
        ligandmpnn_dir = root / "assets" / "ligandmpnn"
    (
        protenix_dir,
        protenix_from_env,
        protenix_env_paths,
    ) = _protenix_root_from_arg_or_env(args.protenix_assets)
    if protenix_dir is None:
        protenix_dir = root / "repos" / "Protenix"
    boltz_cache, boltz_from_env = _expand_arg_or_env(args.boltz_cache, "GH_WEIGHT_BOLTZ")
    if boltz_cache is None and "boltz" in selected_set and software_selection_explicit:
        boltz_cache = root / "assets" / "boltz"
    chai_cache, _chai_from_env = _expand_arg_or_env(args.chai_cache, "CHAI_DOWNLOADS_DIR")
    if chai_cache is None:
        chai_cache = root / "assets" / "chai1"
    torch_cache, _torch_from_env = _expand_arg_or_env(args.torch_cache, "TORCH_HOME")
    if torch_cache is None:
        torch_cache = root / "cache" / "torch"
    hf_cache, _hf_from_env = _expand_arg_or_env(args.hf_cache, "HF_HOME")
    if hf_cache is None:
        hf_cache = root / "cache" / "huggingface"
    rhofold_asset, _rhofold_from_env = _expand_arg_or_env(
        args.rhofold_assets, "GH_RHOFOLD_CHECKPOINT"
    )
    rhofold_checkpoint = (
        rhofold_asset
        if rhofold_asset is not None
        and (rhofold_asset.is_file() or rhofold_asset.suffix in {".pt", ".pth", ".ckpt"})
        else (rhofold_asset or root / "assets" / "rhofold") / "checkpoint.pt"
    )
    opendde_assets, _opendde_assets_from_env = _expand_arg_or_env(
        args.opendde_assets, "OPENDDE_ROOT_DIR"
    )
    if opendde_assets is None:
        opendde_assets = root / "repos" / "opendde"
    rosetta_root = _expand(args.rosetta_root) if args.rosetta_root else None
    molfile_to_params = _expand(args.molfile_to_params) if args.molfile_to_params else None
    haddock3_wheel = _expand(args.haddock3_wheel) if args.haddock3_wheel else None
    haddock3_bin = _expand(args.haddock3_bin) if args.haddock3_bin else None
    if haddock3_bin is None and full_needed:
        candidate = root / "envs" / "haddock3" / "bin" / "haddock3"
        haddock3_bin = candidate if candidate.exists() else None
    autodock_gpu = _expand(args.autodock_gpu) if args.autodock_gpu else None
    autogrid4_bin = _expand(args.autogrid4_bin) if args.autogrid4_bin else None
    dssp_bin = _expand(args.dssp_bin) if args.dssp_bin else None
    usalign_bin, usalign_from_env = _expand_arg_or_env(args.usalign_bin, "GH_USALIGN")
    obabel_bin = _expand(args.obabel_bin) if args.obabel_bin else None
    vina_bin = _expand(args.vina_bin) if args.vina_bin else None
    ananas_bin = _expand(args.ananas_bin) if args.ananas_bin else None
    meeko_python, _meeko_from_env = _expand_python_arg_or_env(
        args.meeko_python, "GH_MEEKO_PYTHON"
    )
    dplm_python, _dplm_from_env = _expand_python_arg_or_env(
        args.dplm_python, "GH_DPLM_PYTHON"
    )
    nampnn_python, _nampnn_from_env = _expand_python_arg_or_env(
        args.nampnn_python, "GH_NAMPNN_PYTHON"
    )
    rpxdock_python, _rpxdock_from_env = _expand_python_arg_or_env(
        args.rpxdock_python, "GH_RPXDOCK_PYTHON"
    )
    opendde_python, _opendde_from_env = _expand_python_arg_or_env(
        args.opendde_python, "GH_OPENDDE_PYTHON"
    )
    esmfold2_python, _esmfold2_from_env = _expand_python_arg_or_env(
        args.esmfold2_python, "GH_ESMFOLD2_PYTHON"
    )
    surfdesign_python, _surfdesign_python_from_env = _expand_python_arg_or_env(
        args.surfdesign_python, "GH_SURFDESIGN_PYTHON"
    )
    surfdesign_repo, _surfdesign_repo_from_env = _expand_arg_or_env(
        args.surfdesign_repo, "GH_SURFDESIGN_REPO"
    )
    if surfdesign_repo is None:
        surfdesign_repo = REPO_ROOT / "vendor" / "surfdesign"
    surfdesign_experiment, _surfdesign_experiment_from_env = _expand_arg_or_env(
        args.surfdesign_experiment, "GH_WEIGHT_SURFDESIGN"
    )
    if surfdesign_experiment is None:
        surfdesign_experiment = root / "assets" / "surfdesign"
    surfpro_python, _surfpro_python_from_env = _expand_python_arg_or_env(
        args.surfpro_python, "GH_SURFPRO_PYTHON"
    )
    surfpro_repo, _surfpro_repo_from_env = _expand_arg_or_env(
        args.surfpro_repo, "GH_SURFPRO_REPO"
    )
    if surfpro_repo is None:
        surfpro_repo = REPO_ROOT / "vendor" / "surfpro"
    surfpro_msms, _surfpro_msms_from_env = _expand_arg_or_env(
        args.surfpro_msms, "GH_SURFPRO_MSMS"
    )
    surfpro_pdb_to_xyzrn, _surfpro_pdb_to_xyzrn_from_env = _expand_arg_or_env(
        args.surfpro_pdb_to_xyzrn, "GH_SURFPRO_PDB_TO_XYZRN"
    )
    rapidock_python, _rapidock_from_env = _expand_python_arg_or_env(
        args.rapidock_python, "GH_RAPIDOCK_PYTHON"
    )
    rapidock_repo, _rapidock_repo_from_env = _expand_arg_or_env(
        args.rapidock_repo, "GH_RAPIDOCK_REPO"
    )
    if rapidock_repo is None:
        rapidock_repo = root / "repos" / "RAPiDock"
    rapidock_weights, _rapidock_weights_from_env = _expand_arg_or_env(
        args.rapidock_weights, "GH_WEIGHT_RAPIDOCK"
    )
    if rapidock_weights is None:
        rapidock_weights = rapidock_repo / "train_models" / "CGTensorProductEquivariantModel"
    rf3_python, _rf3_python_from_env = _expand_python_arg_or_env(
        args.rf3_python, "GH_RF3_PYTHON"
    )
    rf3_repo, _rf3_repo_from_env = _expand_arg_or_env(args.rf3_repo, "GH_RF3_REPO")
    rf3_checkpoint, _rf3_checkpoint_from_env = _expand_arg_or_env(
        args.rf3_checkpoint, "GH_WEIGHT_RF3"
    )
    if rf3_checkpoint is None:
        rf3_checkpoint = root / "assets" / "foundry" / "rf3_latest.ckpt"
    rfd3_checkpoint, _rfd3_checkpoint_from_env = _expand_arg_or_env(
        args.rfd3_checkpoint, "GH_WEIGHT_RFD3"
    )
    if rfd3_checkpoint is None:
        rfd3_checkpoint = root / "assets" / "foundry" / "rfd3_latest.ckpt"
    surfpro_asset_path, _surfpro_asset_from_env = _expand_arg_or_env(
        args.surfpro_checkpoint, "GH_WEIGHT_SURFPRO"
    )
    if surfpro_asset_path is None:
        surfpro_checkpoint = (
            root / "assets" / "surfpro" / args.surfpro_model / "checkpoint_best.pt"
        )
    elif surfpro_asset_path.is_dir() or not surfpro_asset_path.suffix:
        surfpro_checkpoint = surfpro_asset_path / "checkpoint_best.pt"
    else:
        surfpro_checkpoint = surfpro_asset_path
    if surfpro_checkpoint.name != "checkpoint_best.pt":
        raise SystemExit(
            "SurfPro checkpoint must be named checkpoint_best.pt so the generated "
            f"runtime config resolves it exactly: {surfpro_checkpoint}"
        )
    carbondesign_repo, _carbondesign_from_env = _expand_arg_or_env(
        args.carbondesign_repo, "GH_CARBONDESIGN_REPO"
    )
    libgfortran3 = _expand(args.libgfortran3) if args.libgfortran3 else None

    if args.submodule_jobs < 1:
        raise SystemExit("--submodule-jobs must be >= 1")
    if args.host_env_jobs < 1:
        raise SystemExit("--host-env-jobs must be >= 1")
    if args.asset_download_jobs < 1:
        raise SystemExit("--asset-download-jobs must be >= 1")

    preflight_markers = _asset_markers(
        af3_weights=af3_weights,
        af2_data_dir=af2_data_dir,
        ligandmpnn_dir=ligandmpnn_dir,
        protenix_dir=protenix_dir,
        protenix_env_paths=protenix_env_paths,
        boltz_cache=boltz_cache or root / "assets" / "boltz",
        chai_cache=chai_cache,
        torch_cache=torch_cache,
        hf_cache=hf_cache,
        rhofold_checkpoint=rhofold_checkpoint,
        opendde_assets=opendde_assets,
        rapidock_weights=rapidock_weights,
        surfdesign_python=surfdesign_python,
        surfdesign_repo=surfdesign_repo,
        surfdesign_experiment=surfdesign_experiment,
        surfpro_python=surfpro_python,
        surfpro_repo=surfpro_repo,
        surfpro_msms=surfpro_msms,
        surfpro_pdb_to_xyzrn=surfpro_pdb_to_xyzrn,
        surfpro_checkpoint=surfpro_checkpoint,
        rf3_checkpoint=rf3_checkpoint,
        rfd3_checkpoint=rfd3_checkpoint,
        carbondesign_repo=carbondesign_repo,
        rosetta_root=rosetta_root,
        molfile_to_params=molfile_to_params,
        haddock3_bin=haddock3_bin,
        autodock_gpu=autodock_gpu,
        autogrid4_bin=autogrid4_bin,
        ananas_bin=ananas_bin,
    )
    _preflight_selected_resources(
        selected=selected_software,
        markers=preflight_markers,
        download_assets=args.download_public_assets,
        dry_run=args.dry_run,
        skip_submodules=args.skip_submodules,
        allow_missing_af3=args.allow_missing_af3,
        opendde_assets=opendde_assets,
        rapidock_repo=rapidock_repo,
        rapidock_weights=rapidock_weights,
        rf3_repo=rf3_repo,
        rf3_checkpoint=rf3_checkpoint,
        rfd3_checkpoint=rfd3_checkpoint,
        rf3_python=rf3_python,
        carbondesign_repo=carbondesign_repo,
        surfdesign_python=surfdesign_python,
        surfdesign_experiment=surfdesign_experiment,
        surfpro_python=surfpro_python,
        surfpro_msms=surfpro_msms,
        surfpro_pdb_to_xyzrn=surfpro_pdb_to_xyzrn,
        rosetta_root=rosetta_root,
        molfile_to_params=molfile_to_params,
        haddock3_wheel=haddock3_wheel,
        haddock3_bin=haddock3_bin,
        autodock_gpu=autodock_gpu,
        autogrid4_bin=autogrid4_bin,
        ananas_bin=ananas_bin,
    )
    if missing_selected_licenses:
        raise SystemExit(
            "Explicit license acceptance is required for selected software: "
            + ", ".join(
                f"--accept-license {license_id}"
                for license_id in missing_selected_licenses
            )
        )
    _preflight_hf_credentials(
        selected=selected_software,
        markers=preflight_markers,
        hf_cache=hf_cache,
        download_assets=args.download_public_assets,
        dry_run=args.dry_run,
    )
    os.environ.setdefault(
        "GH_ASSET_VALIDATION_CACHE", str(root / "cache" / "asset-validation")
    )
    if not args.dry_run:
        early_validation = tuple(
            name
            for name in selected_software
            if name not in {"nampnn", "surfdesign", "surfpro"}
            and not (args.download_public_assets and name in DOWNLOADABLE_ASSET_SOFTWARE)
            and not (
                name == "alphafold3"
                and args.allow_missing_af3
                and af3_weights is None
            )
        )
        _validate_selected_assets(
            early_validation,
            preflight_markers,
            surfpro_asset=SURFPRO_ASSETS[args.surfpro_model],
        )

    if not args.skip_build or not args.skip_submodules:
        _prepare_source_archive(dry_run=args.dry_run)

    if not args.skip_submodules:
        submodule_set = args.submodule_set or (
            "needed_softwares" if full_needed and not software_selection_explicit else "runtime"
        )
        selected_only = bool(
            software_selection_explicit and args.skip_build and args.submodule_set is None
        )
        extra_submodules = (
            selected_submodules(selected_software) if software_selection_explicit else ()
        )
        if not selected_only or extra_submodules:
            _run(
                ["git", "submodule", "sync", "--recursive"],
                cwd=REPO_ROOT,
                dry_run=args.dry_run,
            )
            submodule_cmd = _submodule_update_command(
                submodule_set=submodule_set,
                depth=args.submodule_depth,
                jobs=args.submodule_jobs,
                extra_paths=extra_submodules,
                selected_only=selected_only,
            )
            _run(submodule_cmd, cwd=REPO_ROOT, dry_run=args.dry_run)
            _validate_submodule_indexes(dry_run=args.dry_run)
            _apply_vendor_patches(
                dry_run=args.dry_run,
                selected_paths=extra_submodules if selected_only else None,
            )

    args.docker_bin_resolved = _detect_docker(args.docker_bin) if docker_needed else None
    if host_apptainer_needed:
        apptainer_build_tools_needed = bool(
            not args.skip_build
            and args.container_builder == "apptainer"
            and args.fakeroot
            and shutil.which("fakeroot") is None
        )
        apptainer_bin = _detect_apptainer(
            args.apptainer_bin,
            dry_run=args.dry_run,
            root=root,
            bootstrap=args.bootstrap_apptainer,
            build_tools=apptainer_build_tools_needed,
        )
        _check_apptainer_site_support(
            apptainer_bin=apptainer_bin,
            root=root,
            dry_run=args.dry_run,
            requires_uidmap=(
                not args.skip_build
                and args.container_builder == "apptainer"
                and args.fakeroot
                and not args.remote
            ),
        )
    else:
        apptainer_bin = (
            str(_expand(args.apptainer_bin))
            if args.apptainer_bin
            else "apptainer"
        )

    if haddock3_bin is None and haddock3_wheel is not None:
        haddock3_bin = _ensure_haddock3_from_wheel(
            root=root,
            wheel=haddock3_wheel,
            dry_run=args.dry_run,
        )

    if not args.dry_run and "surfdesign" in selected_set:
        _require_path(
            surfdesign_repo / "src" / "models" / "generator.py",
            label="SurfDesign source",
        )
    if not args.dry_run and "surfpro" in selected_set:
        for source, label in (
            (
                surfpro_repo / "fairseq" / "models" / "surface_protein_model.py",
                "SurfPro source",
            ),
            (
                REPO_ROOT / "vendor" / "sru" / "sru" / "__init__.py",
                "SurfPro SRU source",
            ),
            (
                REPO_ROOT
                / "vendor"
                / "protein_sequence_models"
                / "sequence_models"
                / "__init__.py",
                "SurfPro sequence-models source",
            ),
        ):
            _require_path(source, label=label)

    if (
        not args.skip_build
        and args.container_builder == "apptainer"
        and args.fakeroot
    ):
        args.fakeroot_bin_resolved = _ensure_fakeroot(root, dry_run=args.dry_run)

    host_env_only = (
        selected_host_envs(selected_software)
        if software_selection_explicit
        else None
    )
    surface_env = {}
    if "surfdesign" in selected_set:
        surface_env.update(
            {
                "GH_SURFDESIGN_PYTHON": surfdesign_python,
                "GH_SURFDESIGN_REPO": surfdesign_repo,
                "GH_WEIGHT_SURFDESIGN": surfdesign_experiment,
            }
        )
    if "surfpro" in selected_set:
        surface_env.update(
            {
                "GH_SURFPRO_PYTHON": surfpro_python,
                "GH_SURFPRO_REPO": surfpro_repo,
                "GH_SURFPRO_MSMS": surfpro_msms,
                "GH_SURFPRO_PDB_TO_XYZRN": surfpro_pdb_to_xyzrn,
                "GH_WEIGHT_SURFPRO": surfpro_checkpoint.parent,
            }
        )
    _run_host_env_setup(
        args,
        root=root,
        only_envs=host_env_only,
        env_overrides=surface_env,
    )

    if "dplm" in selected_set and dplm_python is None:
        dplm_python = root / "envs" / "dplm" / "bin" / "python"
    if "rpxdock" in selected_set and rpxdock_python is None:
        rpxdock_python = root / "envs" / "rpxdock" / "bin" / "python"
    if "opendde" in selected_set and opendde_python is None:
        opendde_python = root / "envs" / "opendde" / "bin" / "python"
    if "esmfold2" in selected_set and esmfold2_python is None:
        esmfold2_python = root / "envs" / "esmfold2" / "bin" / "python"
    if "rapidock" in selected_set and rapidock_python is None:
        rapidock_python = root / "envs" / "rapidock" / "bin" / "python"

    surface_selection = tuple(
        name for name in selected_software if name in {"surfdesign", "surfpro"}
    )
    if surface_selection and not args.skip_import_smoke:
        _run_selected_host_import_smoke(
            args,
            selected=surface_selection,
            dplm_python=dplm_python,
            opendde_python=opendde_python,
            esmfold2_python=esmfold2_python,
            rapidock_python=rapidock_python,
            rpxdock_python=rpxdock_python,
            rapidock_repo=rapidock_repo,
            rf3_python=rf3_python,
            rf3_repo=rf3_repo,
            surfdesign_python=surfdesign_python,
            surfdesign_repo=surfdesign_repo,
            surfpro_python=surfpro_python,
            surfpro_repo=surfpro_repo,
            hf_cache=hf_cache,
        )

    if args.download_public_assets:
        download_selected = (
            selected_software
            if software_selection_explicit
            else ("alphafold2", "ligandmpnn", "protenix")
        )
        download_selected_set = set(download_selected)
        for cache in (
            boltz_cache if "boltz" in download_selected_set else None,
            chai_cache if "chai1" in download_selected_set else None,
            torch_cache
            if download_selected_set & {"esm2", "esmif", "esmfold"}
            else None,
            hf_cache if download_selected_set & (CONTAINER_PREFETCH_SOFTWARE | HOST_PREFETCH_SOFTWARE) else None,
        ):
            if cache is not None and not args.dry_run:
                cache.mkdir(parents=True, exist_ok=True)
        _download_public_assets(
            root=root,
            af2_data_dir=af2_data_dir,
            ligandmpnn_dir=ligandmpnn_dir,
            protenix_dir=protenix_dir,
            protenix_env_paths=protenix_env_paths,
            rhofold_checkpoint=rhofold_checkpoint,
            selected=download_selected,
            jobs=args.asset_download_jobs,
            dry_run=args.dry_run,
            opendde_assets=opendde_assets,
            rapidock_weights=rapidock_weights,
            rf3_checkpoint=rf3_checkpoint,
            rfd3_checkpoint=rfd3_checkpoint,
            surfpro_checkpoint=surfpro_checkpoint,
            surfpro_asset=SURFPRO_ASSETS[args.surfpro_model],
        )
        _run_host_asset_prefetch(
            args,
            selected=(download_selected_set & HOST_PREFETCH_SOFTWARE),
            dplm_python=dplm_python,
            esmfold2_python=esmfold2_python,
            hf_cache=hf_cache,
        )

    if software_selection_explicit and not args.skip_import_smoke:
        _run_selected_host_import_smoke(
            args,
            selected=tuple(
                name
                for name in selected_software
                if name not in {"surfdesign", "surfpro"}
            ),
            dplm_python=dplm_python,
            opendde_python=opendde_python,
            esmfold2_python=esmfold2_python,
            rapidock_python=rapidock_python,
            rpxdock_python=rpxdock_python,
            rapidock_repo=rapidock_repo,
            rf3_python=rf3_python,
            rf3_repo=rf3_repo,
            surfdesign_python=surfdesign_python,
            surfdesign_repo=surfdesign_repo,
            surfpro_python=surfpro_python,
            surfpro_repo=surfpro_repo,
            hf_cache=hf_cache,
        )

    if full_needed:
        if af2_data_dir is None:
            raise SystemExit("--af2-params is required for --needed-softwares full unless --download-public-assets is used.")
        if dplm_python is None:
            dplm_python = root / "envs" / "dplm" / "bin" / "python"
        if rpxdock_python is None:
            rpxdock_python = root / "envs" / "rpxdock" / "bin" / "python"
        if nampnn_python is None:
            nampnn_python = Path(args.container_python)
        for rel_path in (
            "vendor/dplm",
            "vendor/na_mpnn",
            "vendor/rpxdock",
            "vendor/willutil",
            "vendor/usalign/USalign",
            "astrolabe/assets/DAlphaBall.gcc",
        ):
            _require_path(REPO_ROOT / rel_path, label=rel_path, executable=rel_path.endswith(("USalign", "DAlphaBall.gcc")))
        _require_path(
            REPO_ROOT / "vendor" / "na_mpnn" / "models" / "design_model" / "s_19137.pt",
            label="NA-MPNN design model",
        )
        if libgfortran3 is None:
            raise SystemExit("--libgfortran3 is required for --needed-softwares full.")
        _require_path(libgfortran3 / "libgfortran.so.3", label="libgfortran.so.3")

    if software_selection_explicit and "nampnn" in selected_set:
        if nampnn_python is None:
            nampnn_python = Path(args.container_python)
        _require_path(
            REPO_ROOT / "vendor" / "na_mpnn" / "models" / "design_model" / "s_19137.pt",
            label="NA-MPNN design model",
        )
    if software_selection_explicit and "dalphaball" in selected_set:
        _require_path(
            REPO_ROOT / "astrolabe" / "assets" / "DAlphaBall.gcc",
            label="DAlphaBall executable",
            executable=True,
        )

    binds: list[str] = []
    env_defaults: dict[str, str] = {}
    env_prepends: list[tuple[str, str]] = []
    runtime_cache = root / "cache" / "runtime"
    if not args.dry_run:
        runtime_cache = _ensure_runtime_cache(root)
    _append_bind(binds, runtime_cache, "/cache/gh-runtime")
    env_defaults.update(
        {key: f"/cache/gh-runtime/{dirname}" for key, dirname in RUNTIME_CACHE_DIRS}
    )
    if not args.no_nv:
        # Apptainer's legacy --nv path on this driver exposes devices and
        # nvidia-smi but CUDA initialization needs this standard visibility
        # marker. CUDA_VISIBLE_DEVICES can still select a subset.
        env_defaults["NVIDIA_VISIBLE_DEVICES"] = "all"
        env_defaults["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env_defaults["GH_CUDA_PREINITIALIZE"] = "1"

    selected_required = lambda name: software_selection_explicit and name in selected_set
    if not software_selection_explicit or "alphafold3" in selected_set:
        _append_bind_and_env(
            binds=binds,
            env_defaults=env_defaults,
            host_path=af3_weights,
            container_path="/models/alphafold3",
            env_key="GH_WEIGHT_AF3",
            required=af3_required or af3_from_env,
            label="AF3 weights",
        )
    if af2_data_dir is not None and (
        not software_selection_explicit or "alphafold2" in selected_set
    ):
        _append_bind_and_env(
            binds=binds,
            env_defaults=env_defaults,
            host_path=af2_data_dir,
            container_path="/models/alphafold2",
            env_key="GH_WEIGHT_AF2",
            required=full_needed or selected_required("alphafold2") or af2_from_env,
            label="AF2 data dir",
        )
    if (
        (software_selection_explicit and "ligandmpnn" in selected_set)
        or (
            not software_selection_explicit
            and (
                ligandmpnn_dir.exists()
                or args.download_public_assets
                or full_needed
                or ligandmpnn_from_env
            )
        )
    ):
        _append_bind_and_env(
            binds=binds,
            env_defaults=env_defaults,
            host_path=ligandmpnn_dir,
            container_path="/models/ligandmpnn",
            env_key="GH_WEIGHT_LIGANDMPNN",
            required=full_needed or selected_required("ligandmpnn") or ligandmpnn_from_env,
            label="LigandMPNN params",
        )

    if (
        (software_selection_explicit and "protenix" in selected_set)
        or (
            not software_selection_explicit
            and (protenix_dir.exists() or args.download_public_assets or full_needed or protenix_from_env)
        )
    ):
        if protenix_from_env:
            _require_path(protenix_dir, label="Protenix asset root")
            for env_key, env_path in sorted(protenix_env_paths.items()):
                _require_path(env_path, label=env_key)
        if (full_needed or selected_required("protenix")) and not protenix_from_env:
            _require_path(
                protenix_dir / "release_model" / "protenix-v2.pt",
                label="Protenix v2 checkpoint",
            )
            _require_path(
                protenix_dir / "release_model" / "protenix_base_constraint_v0.5.0.pt",
                label="Protenix constraint checkpoint",
            )
            _require_path(
                protenix_dir / "release_data" / "components.v20240608.cif",
                label="Protenix CCD",
            )
            _require_path(
                protenix_dir / "release_data" / "components.v20240608.cif.rdkit_mol.pkl",
                label="Protenix RDKit CCD cache",
            )
        _append_bind(binds, protenix_dir, "/models/protenix")
        protenix_defaults = {
            "GH_WEIGHT_PROTENIX": "/models/protenix/release_model/protenix-v2.pt",
            "GH_WEIGHT_PROTENIX_CONSTRAINT": "/models/protenix/release_model/protenix_base_constraint_v0.5.0.pt",
            "GH_PROTENIX_CCD": "/models/protenix/release_data/components.v20240608.cif",
            "GH_PROTENIX_CCD_RDKIT": "/models/protenix/release_data/components.v20240608.cif.rdkit_mol.pkl",
        }
        for env_key, env_path in sorted(protenix_env_paths.items()):
            protenix_defaults[env_key] = _container_path_under_root(
                protenix_dir,
                env_path,
                "/models/protenix",
            )
        env_defaults.update(protenix_defaults)

    if boltz_cache is not None and (
        not software_selection_explicit or "boltz" in selected_set
    ):
        _append_bind_and_env(
            binds=binds,
            env_defaults=env_defaults,
            host_path=boltz_cache,
            container_path="/models/boltz",
            env_key="GH_WEIGHT_BOLTZ",
            required=selected_required("boltz") or boltz_from_env,
            label="Boltz cache",
        )

    if selected_required("chai1"):
        _append_bind_and_env(
            binds=binds,
            env_defaults=env_defaults,
            host_path=chai_cache,
            container_path="/models/chai1",
            env_key="CHAI_DOWNLOADS_DIR",
            required=True,
            label="Chai-1 cache",
        )
    if software_selection_explicit and selected_set & {"esm2", "esmif", "esmfold"}:
        _append_bind_and_env(
            binds=binds,
            env_defaults=env_defaults,
            host_path=torch_cache,
            container_path="/models/torch",
            env_key="TORCH_HOME",
            required=True,
            label="Torch model cache",
        )
    if software_selection_explicit and selected_set & {"esm3", "dplm", "esmfold2"}:
        _append_bind_and_env(
            binds=binds,
            env_defaults=env_defaults,
            host_path=hf_cache,
            container_path="/models/huggingface",
            env_key="HF_HOME",
            required=True,
            label="Hugging Face cache",
        )
        env_defaults["HUGGINGFACE_HUB_CACHE"] = "/models/huggingface/hub"
        env_defaults["HF_HUB_OFFLINE"] = "1"
        env_defaults["TRANSFORMERS_OFFLINE"] = "1"
        for name, env_key in (
            ("esm3", "GH_ESM3_MODEL_SNAPSHOT"),
            ("dplm", "GH_DPLM_MODEL_SNAPSHOT"),
            ("esmfold2", "GH_ESMFOLD2_MODEL_SNAPSHOT"),
        ):
            if name in selected_set:
                relative_snapshot = hf_runtime_snapshot_path(
                    hf_cache, name
                ).relative_to(hf_cache)
                env_defaults[env_key] = str(
                    Path("/models/huggingface") / relative_snapshot
                )
        if "esmfold2" in selected_set:
            esmc_snapshot = hf_runtime_snapshot_path(
                hf_cache, "esmfold2-esmc"
            ).relative_to(hf_cache)
            env_defaults["GH_ESMFOLD2_ESMC_SNAPSHOT"] = str(
                Path("/models/huggingface") / esmc_snapshot
            )
    if selected_required("rhofold"):
        _require_path(rhofold_checkpoint, label="RhoFold checkpoint")
        _append_bind(binds, rhofold_checkpoint.parent, "/models/rhofold")
        env_defaults["GH_RHOFOLD_CHECKPOINT"] = (
            f"/models/rhofold/{rhofold_checkpoint.name}"
        )
    if selected_required("opendde"):
        _require_path(opendde_assets / "checkpoint" / "opendde.pt", label="OpenDDE checkpoint")
        _append_bind(binds, opendde_assets, "/models/opendde")
        env_defaults["OPENDDE_ROOT_DIR"] = "/models/opendde"
    if selected_required("surfpro"):
        if not args.dry_run:
            _require_path(surfpro_checkpoint, label="SurfPro checkpoint")
        _append_bind(binds, surfpro_checkpoint.parent, "/models/surfpro")
        env_defaults["GH_WEIGHT_SURFPRO"] = "/models/surfpro"

    if selected_required("rosetta"):
        env_defaults["GH_MOLFILE_TO_PARAMS"] = "/rosetta/molfile_to_params.py"
    else:
        _append_molfile_to_params_bind(
            binds=binds,
            env_defaults=env_defaults,
            rosetta_root=rosetta_root,
            molfile_to_params=molfile_to_params,
            required=False,
        )

    _append_file_bind_and_env(
        binds=binds,
        env_defaults=env_defaults,
        host_path=haddock3_bin,
        container_path="/usr/bin/haddock3",
        env_key="GH_HADDOCK3_BIN",
        required=full_needed or selected_required("haddock3"),
        label="HADDOCK3 executable",
        executable=True,
    )

    _append_file_bind_and_env(
        binds=binds,
        env_defaults=env_defaults,
        host_path=autodock_gpu,
        container_path="/usr/bin/autodock_gpu_128wi",
        env_key="GH_AUTODOCK_GPU",
        required=full_needed or selected_required("autodock-gpu"),
        label="AutoDock-GPU executable",
        executable=True,
    )
    if autodock_gpu is not None and autodock_gpu.exists():
        _append_bind(binds, autodock_gpu, "/usr/bin/autodock_gpu")

    _append_file_bind_and_env(
        binds=binds,
        env_defaults=env_defaults,
        host_path=autogrid4_bin,
        container_path="/usr/bin/autogrid4",
        env_key="GH_AUTOGRID4",
        required=full_needed or selected_required("autodock-gpu"),
        label="AutoGrid4 executable",
        executable=True,
    )
    _append_file_bind_and_env(
        binds=binds,
        env_defaults=env_defaults,
        host_path=dssp_bin,
        container_path="/usr/bin/dssp",
        env_key="GH_DSSP",
        required=False,
        label="DSSP executable",
        executable=True,
    )
    _append_file_bind_and_env(
        binds=binds,
        env_defaults=env_defaults,
        host_path=obabel_bin,
        container_path="/usr/local/bin/obabel",
        env_key="GH_OBABEL",
        required=False,
        label="OpenBabel executable",
        executable=True,
    )
    _append_file_bind_and_env(
        binds=binds,
        env_defaults=env_defaults,
        host_path=vina_bin,
        container_path="/usr/bin/vina",
        env_key="GH_VINA",
        required=False,
        label="Vina executable",
        executable=True,
    )
    if selected_required("ananas"):
        env_defaults["GH_ANANAS"] = "/usr/local/bin/AnAnaS"
    else:
        _append_file_bind_and_env(
            binds=binds,
            env_defaults=env_defaults,
            host_path=ananas_bin,
            container_path="/tools/ananas/ananas",
            env_key="GH_ANANAS",
            required=False,
            label="AnAnaS executable",
            executable=True,
        )
    if usalign_bin is not None:
        _append_file_bind_and_env(
            binds=binds,
            env_defaults=env_defaults,
            host_path=usalign_bin,
            container_path="/usr/local/bin/USalign",
            env_key="GH_USALIGN",
            required=usalign_from_env,
            label="USalign executable",
            executable=True,
        )
    elif full_needed or selected_required("usalign"):
        env_defaults["GH_USALIGN"] = f"{args.container_repo_root}/vendor/usalign/USalign"

    _append_python_env_bind_and_env(
        binds=binds,
        env_defaults=env_defaults,
        host_python=meeko_python,
        container_prefix="/envs/meeko",
        env_key="GH_MEEKO_PYTHON",
        required=False,
        label="Meeko Python",
    )
    if args.meeko_pythonpath:
        env_defaults["GH_MEEKO_PYTHONPATH"] = args.meeko_pythonpath

    dplm_host_python = (
        dplm_python if dplm_python is not None and dplm_python.is_absolute() else None
    )
    _append_python_env_bind_and_env(
        binds=binds,
        env_defaults=env_defaults,
        host_python=dplm_host_python,
        container_prefix="/envs/dplm",
        env_key="GH_DPLM_PYTHON",
        required=selected_required("dplm") or (full_needed and dplm_host_python is not None),
        label="DPLM Python",
    )
    if dplm_python is not None and not dplm_python.is_absolute():
        env_defaults["GH_DPLM_PYTHON"] = str(dplm_python)
    if args.dplm_repo or full_needed or "dplm" in selected_set:
        env_defaults["GH_DPLM_REPO"] = args.dplm_repo or f"{args.container_repo_root}/vendor/dplm"

    nampnn_host_python = (
        nampnn_python if nampnn_python is not None and nampnn_python.is_absolute() else None
    )
    _append_python_env_bind_and_env(
        binds=binds,
        env_defaults=env_defaults,
        host_python=nampnn_host_python,
        container_prefix="/envs/nampnn",
        env_key="GH_NAMPNN_PYTHON",
        required=selected_required("nampnn") and nampnn_host_python is not None,
        label="NA-MPNN Python",
    )
    if nampnn_python is not None and not nampnn_python.is_absolute():
        env_defaults["GH_NAMPNN_PYTHON"] = str(nampnn_python)
    if args.nampnn_repo or full_needed or "nampnn" in selected_set:
        env_defaults["GH_NAMPNN_REPO"] = args.nampnn_repo or f"{args.container_repo_root}/vendor/na_mpnn"

    rpxdock_host_python = (
        rpxdock_python if rpxdock_python is not None and rpxdock_python.is_absolute() else None
    )
    _append_python_env_bind_and_env(
        binds=binds,
        env_defaults=env_defaults,
        host_python=rpxdock_host_python,
        container_prefix="/envs/rpxdock",
        env_key="GH_RPXDOCK_PYTHON",
        required=selected_required("rpxdock") or (full_needed and rpxdock_host_python is not None),
        label="RPXDock Python",
    )
    if rpxdock_python is not None and not rpxdock_python.is_absolute():
        env_defaults["GH_RPXDOCK_PYTHON"] = str(rpxdock_python)
    if args.rpxdock_repo or full_needed or "rpxdock" in selected_set:
        env_defaults["GH_RPXDOCK_REPO"] = args.rpxdock_repo or f"{args.container_repo_root}/vendor/rpxdock"
    if args.willutil_repo or full_needed or "rpxdock" in selected_set:
        env_defaults["GH_WILLUTIL_REPO"] = args.willutil_repo or f"{args.container_repo_root}/vendor/willutil"

    opendde_host_python = (
        opendde_python if opendde_python is not None and opendde_python.is_absolute() else None
    )
    _append_python_env_bind_and_env(
        binds=binds,
        env_defaults=env_defaults,
        host_python=opendde_host_python,
        container_prefix="/envs/opendde",
        env_key="GH_OPENDDE_PYTHON",
        required=selected_required("opendde"),
        label="OpenDDE Python",
    )
    if selected_required("opendde"):
        env_defaults["GH_OPENDDE_REPO"] = f"{args.container_repo_root}/vendor/opendde"

    esmfold2_host_python = (
        esmfold2_python
        if esmfold2_python is not None and esmfold2_python.is_absolute()
        else None
    )
    _append_python_env_bind_and_env(
        binds=binds,
        env_defaults=env_defaults,
        host_python=esmfold2_host_python,
        container_prefix="/envs/esmfold2",
        env_key="GH_ESMFOLD2_PYTHON",
        required=selected_required("esmfold2"),
        label="ESMFold2 Python",
    )
    if selected_required("esmfold2"):
        env_defaults["GH_ESMFOLD2_ESM_REPO"] = f"{args.container_repo_root}/vendor/biohub_esm"
        env_defaults["GH_ESMFOLD2_TRANSFORMERS_REPO"] = (
            f"{args.container_repo_root}/vendor/biohub_transformers"
        )

    if selected_required("surfdesign"):
        _require_path(
            surfdesign_repo / "src" / "models" / "generator.py",
            label="SurfDesign source",
        )
        _require_path(surfdesign_experiment, label="SurfDesign experiment")
        assert surfdesign_experiment is not None
        _append_python_env_bind_and_env(
            binds=binds,
            env_defaults=env_defaults,
            host_python=surfdesign_python,
            container_prefix="/envs/surfdesign",
            env_key="GH_SURFDESIGN_PYTHON",
            required=True,
            label="SurfDesign Python",
        )
        _append_bind(binds, surfdesign_repo, "/ext/surfdesign")
        _append_bind(binds, surfdesign_experiment, "/models/surfdesign")
        env_defaults["GH_SURFDESIGN_REPO"] = "/ext/surfdesign"
        env_defaults["GH_WEIGHT_SURFDESIGN"] = "/models/surfdesign"

    if selected_required("surfpro"):
        _require_path(
            surfpro_repo / "fairseq" / "models" / "surface_protein_model.py",
            label="SurfPro source",
        )
        _require_path(surfpro_msms, label="SurfPro MSMS", executable=True)
        _require_path(
            surfpro_pdb_to_xyzrn,
            label="SurfPro pdb_to_xyzrn",
            executable=True,
        )
        assert surfpro_msms is not None and surfpro_pdb_to_xyzrn is not None
        _require_path(
            surfpro_pdb_to_xyzrn.parent / "atmtypenumbers",
            label="SurfPro atmtypenumbers",
        )
        sru_repo = REPO_ROOT / "vendor" / "sru"
        sequence_models_repo = REPO_ROOT / "vendor" / "protein_sequence_models"
        _require_path(sru_repo / "sru" / "__init__.py", label="SurfPro SRU source")
        _require_path(
            sequence_models_repo / "sequence_models" / "__init__.py",
            label="SurfPro sequence-models source",
        )
        _append_python_env_bind_and_env(
            binds=binds,
            env_defaults=env_defaults,
            host_python=surfpro_python,
            container_prefix="/envs/surfpro",
            env_key="GH_SURFPRO_PYTHON",
            required=True,
            label="SurfPro Python",
        )
        _append_bind(binds, surfpro_repo, "/ext/surfpro")
        _append_bind(binds, sru_repo, "/ext/sru")
        _append_bind(binds, sequence_models_repo, "/ext/protein_sequence_models")
        msms = surfpro_msms.resolve()
        pdb_to_xyzrn = surfpro_pdb_to_xyzrn.resolve()
        _append_bind(binds, msms.parent, "/tools/surfpro/msms")
        _append_bind(
            binds,
            pdb_to_xyzrn.parent,
            "/tools/surfpro/pdb_to_xyzrn",
        )
        env_defaults["GH_SURFPRO_REPO"] = "/ext/surfpro"
        env_defaults["GH_SURFPRO_PYTHONPATH"] = (
            "/ext/surfpro:/ext/sru:/ext/protein_sequence_models"
        )
        env_defaults["GH_SURFPRO_MSMS"] = f"/tools/surfpro/msms/{msms.name}"
        env_defaults["GH_SURFPRO_PDB_TO_XYZRN"] = (
            f"/tools/surfpro/pdb_to_xyzrn/{pdb_to_xyzrn.name}"
        )

    rapidock_host_python = (
        rapidock_python
        if rapidock_python is not None and rapidock_python.is_absolute()
        else None
    )
    _append_python_env_bind_and_env(
        binds=binds,
        env_defaults=env_defaults,
        host_python=rapidock_host_python,
        container_prefix="/envs/rapidock",
        env_key="GH_RAPIDOCK_PYTHON",
        required=selected_required("rapidock"),
        label="RAPiDock Python",
    )
    if selected_required("rapidock"):
        _require_path(rapidock_repo, label="RAPiDock repository")
        _require_path(rapidock_weights, label="RAPiDock checkpoint directory")
        _append_bind(binds, rapidock_repo, "/ext/rapidock")
        _append_bind(binds, rapidock_weights, "/models/rapidock")
        env_defaults["GH_RAPIDOCK_REPO"] = "/ext/rapidock"
        env_defaults["GH_WEIGHT_RAPIDOCK"] = "/models/rapidock"

    if selected_set & {"rf3", "rfd3"}:
        _require_path(rf3_repo, label="Foundry RF3/RFD3 repository")
        _require_path(rf3_python, label="Foundry RF3/RFD3 Python", executable=True)
        assert rf3_repo is not None and rf3_python is not None
        _append_bind(binds, rf3_repo, "/ext/rf3")
        _append_python_env_bind_and_env(
            binds=binds,
            env_defaults=env_defaults,
            host_python=rf3_python,
            container_prefix="/envs/rf3",
            env_key="GH_RF3_PYTHON",
            required=True,
            label="Foundry RF3/RFD3 Python",
        )
        env_defaults["GH_RF3_REPO"] = "/ext/rf3"
    if selected_required("rf3"):
        _require_path(rf3_checkpoint, label="RF3 checkpoint")
        assert rf3_checkpoint is not None
        _append_bind(binds, rf3_checkpoint, "/models/rf3/model.pt")
        env_defaults["GH_WEIGHT_RF3"] = "/models/rf3/model.pt"
    if selected_required("rfd3"):
        _require_path(rfd3_checkpoint, label="RFD3 checkpoint")
        assert rfd3_checkpoint is not None
        _append_bind(binds, rfd3_checkpoint, "/models/rfd3/model.pt")
        env_defaults["GH_WEIGHT_RFD3"] = "/models/rfd3/model.pt"

    if selected_required("carbondesign"):
        _require_path(carbondesign_repo, label="CarbonDesign repository")
        assert carbondesign_repo is not None
        for rel_path in (
            "params/carbondesign_default.ckpt",
            "config/config_model_mrf_pair_enable_esm_sc.json",
            "config/config_data_mrf2.json",
        ):
            _require_path(carbondesign_repo / rel_path, label=f"CarbonDesign {rel_path}")
        _append_bind(binds, carbondesign_repo, "/ext/carbondesign")
        env_defaults["GH_CARBONDESIGN_REPO"] = "/ext/carbondesign"

    if full_needed or selected_required("dalphaball"):
        env_defaults["GH_DALPHABALL"] = f"{args.container_repo_root}/astrolabe/assets/DAlphaBall.gcc"

    if libgfortran3 is not None:
        _append_bind(binds, libgfortran3, "/host-libs/libgfortran3")
        env_prepends.append(("LD_LIBRARY_PATH", "/host-libs/libgfortran3"))
        env_defaults["APPTAINERENV_LD_LIBRARY_PATH"] = "/host-libs/libgfortran3"

    binds.extend(args.extra_bind)
    for raw in args.env_default:
        key, value = _parse_assignment(raw, "--env-default")
        env_defaults[key] = value
    for raw in args.prepend_env:
        env_prepends.append(_parse_assignment(raw, "--prepend-env"))

    resolved_boltz_cache = boltz_cache or root / "assets" / "boltz"
    markers = _asset_markers(
        af3_weights=af3_weights,
        af2_data_dir=af2_data_dir,
        ligandmpnn_dir=ligandmpnn_dir,
        protenix_dir=protenix_dir,
        protenix_env_paths=protenix_env_paths,
        boltz_cache=resolved_boltz_cache,
        chai_cache=chai_cache,
        torch_cache=torch_cache,
        hf_cache=hf_cache,
        rhofold_checkpoint=rhofold_checkpoint,
        opendde_assets=opendde_assets,
        rapidock_weights=rapidock_weights,
        surfdesign_python=surfdesign_python,
        surfdesign_repo=surfdesign_repo,
        surfdesign_experiment=surfdesign_experiment,
        surfpro_python=surfpro_python,
        surfpro_repo=surfpro_repo,
        surfpro_msms=surfpro_msms,
        surfpro_pdb_to_xyzrn=surfpro_pdb_to_xyzrn,
        surfpro_checkpoint=surfpro_checkpoint,
        rf3_checkpoint=rf3_checkpoint,
        rfd3_checkpoint=rfd3_checkpoint,
        carbondesign_repo=carbondesign_repo,
        rosetta_root=rosetta_root,
        molfile_to_params=molfile_to_params,
        haddock3_bin=haddock3_bin,
        autodock_gpu=autodock_gpu,
        autogrid4_bin=autogrid4_bin,
        ananas_bin=ananas_bin,
    )
    if not args.dry_run:
        deferred_prefetch = (
            selected_set & CONTAINER_PREFETCH_SOFTWARE
            if args.download_public_assets
            else set()
        )
        prebuild_selection = tuple(
            name
            for name in selected_software
            if name not in deferred_prefetch
            and (
                name != "alphafold3"
                or not args.allow_missing_af3
                or af3_weights is not None
            )
        )
        _validate_selected_assets(
            prebuild_selection,
            markers,
            surfpro_asset=SURFPRO_ASSETS[args.surfpro_model],
        )

    if not args.skip_build:
        args.af3_build_data_inputs = _stage_af3_build_data_cache(
            root=root,
            build_cache=args.container_build_cache,
            ccd_pickle=args.af3_ccd_pickle,
            chemical_component_sets_pickle=args.af3_chemical_component_sets_pickle,
            components_cif=args.af3_components_cif,
            dry_run=args.dry_run,
        )
        if "pyrosetta" in selected_set:
            pyrosetta_input = _stage_pyrosetta_wheel_cache(
                root=root,
                build_cache=args.container_build_cache,
                wheel=args.pyrosetta_wheel,
                dry_run=args.dry_run,
            )
            if pyrosetta_input is not None:
                args.af3_build_data_inputs.append(pyrosetta_input)
        if "ananas" in selected_set:
            args.af3_build_data_inputs.append(
                _stage_ananas_archive_cache(
                    build_cache=args.container_build_cache,
                    dry_run=args.dry_run,
                )
            )
        if "rosetta" in selected_set:
            args.af3_build_data_inputs.append(
                _stage_rosetta_tools_cache(
                    build_cache=args.container_build_cache,
                    dry_run=args.dry_run,
                )
            )
        _build_sif(args, apptainer_bin=apptainer_bin, sif=sif, root=root)

    if runner_needed:
        _run(
            _container_inspect_command(args, apptainer_bin=apptainer_bin, sif=sif),
            dry_run=args.dry_run,
        )

    _write_site_profile(
        site_profile,
        profile=args.profile,
        apptainer_bin=apptainer_bin,
        sif=sif,
        runner=runner,
        binds=binds,
        env_defaults=env_defaults,
        env_prepends=env_prepends,
        container_repo_root=args.container_repo_root,
        container_python=args.container_python,
        use_nv=not args.no_nv,
        runner_backend=args.runner_backend,
        docker_bin=args.docker_bin_resolved,
        docker_image=args.runner_docker_image,
    )
    if runner_needed:
        _write_runner_from_profile(site_profile, dry_run=args.dry_run)
    else:
        print(
            "Skipped runner generation because build, import smoke, and doctor were skipped; "
            "install a supported runner backend before generating or using the runner."
        )

    prefetch_names = selected_set & CONTAINER_PREFETCH_SOFTWARE
    if container_prefetch_selected and not args.dry_run:
        missing_prefetch = set()
        for name in prefetch_names:
            try:
                _validate_selected_assets(
                    (name,),
                    markers,
                    surfpro_asset=SURFPRO_ASSETS[args.surfpro_model],
                )
            except (RuntimeError, SystemExit):
                missing_prefetch.add(name)
        prefetch_names = missing_prefetch
    if container_prefetch_selected and prefetch_names:
        _run_prefetch_command(
            args,
            python=runner,
            selected=prefetch_names,
            boltz_cache=resolved_boltz_cache,
            chai_cache=chai_cache,
            torch_cache=torch_cache,
            hf_cache=hf_cache,
            container_paths=True,
        )

    if software_selection_explicit and not args.dry_run:
        validation_selection = tuple(
            name
            for name in selected_software
            if name != "alphafold3" or not args.allow_missing_af3 or af3_weights is not None
        )
        _validate_selected_assets(
            validation_selection,
            markers,
            surfpro_asset=SURFPRO_ASSETS[args.surfpro_model],
        )

    if not args.skip_import_smoke:
        _run_import_smoke(
            args,
            runner,
            selected=selected_software if software_selection_explicit else None,
        )

    if not args.skip_doctor:
        _run_doctor(args, runner)

    container_identity = (
        None
        if args.dry_run or not runner_needed
        else prebuilt_container_identity
        or _sif_identity(
            sif,
            build_features=args.container_build_features,
            require_build_metadata=True,
        )
    )
    if container_identity is not None:
        container_identity["import_smoke_passed"] = not args.skip_import_smoke
        container_identity["doctor_passed"] = not args.skip_doctor
    _write_software_manifest(
        software_manifest,
        selected=selected_software,
        markers=markers,
        accepted_licenses=accepted_licenses,
        dry_run=args.dry_run,
        container_identity=container_identity,
        selected_asset_variants={"surfpro": args.surfpro_model},
    )

    _clean_dedicated_bootstrap_cache(root, dry_run=args.dry_run)
    if not args.dry_run:
        _ensure_runtime_cache(root)
        if runner_needed:
            _validate_runner_bind_sources(binds)

    if args.activate_checkout:
        _write_checkout_activation(runner, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run complete. No container was built or runtime validated.")
    elif not runner_needed:
        print("Source/profile preparation complete. No runnable environment was generated; "
              "rerun without the build/import/doctor skip flags to install and validate one.")
    elif args.activate_checkout:
        print("Bootstrap complete. Use: ./scripts/astrolabe ...")
    else:
        print(f"Bootstrap complete. Use: GH_PY={runner} scripts/astrolabe ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
