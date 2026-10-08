#!/usr/bin/env python3
"""Prefetch selected runtime assets into persistent host caches."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from astrolabe.asset_download import download_url, sha256_file, validate_file
from installation.software_catalog import SOFTWARE, resolve_software


CHAI_COMPONENTS = {
    "feature_embedding.pt": (4_000_000, "58600fb282827ee105b4bf33845c2eb2dce2a7d2ba7582951f76c9290c66c019"),
    "trunk.pt": (500_000_000, "b7a08e104455f76491088a5eeaea9bb95a08168c48969b0b892971346e51ccbc"),
    "diffusion_module.pt": (400_000_000, "bd77cf4095ad6b8a4eae7e390d01d24be1ebd819922ca3f28e8744f9945c9cad"),
    "confidence_head.pt": (40_000_000, "82a4ac9f934fdd0e73870150f508cecbe010c1383234bdadcaa70d57323deb6b"),
    "token_embedder.pt": (5_000_000, "60cada19fd9c77d8298131a00368d162b6bc7d3c836dd36c715db04d21f7bbde"),
    "bond_loss_input_proj.pt": (1_000, "98f3b2e415d11e460a684cdbfc213ca80447308022b1953cc5326ad468948851"),
}
CHAI_CONFORMERS_SHA256 = "f2161256b565bbd84198da8b3e3d3978ae2f179f778d0995cd135c4c0c57c41f"
CHAI_ESM_SHA256 = "074673b97e1c1ff9c3cf949294749dd446ad1c83cee50a466b4841bb4d89c27a"
ESM2_MODEL_NAME = "esm2_t33_650M_UR50D"
ESMIF_MODEL_NAME = "esm_if1_gvp4_t16_142M_UR50"
ESMFOLD_MODEL_NAME = "esmfold_3B_v1"
ESMFOLD_ESM_MODEL_NAME = "esm2_t36_3B_UR50D"
FAIR_ESM_CHECKPOINTS = {
    ESM2_MODEL_NAME: (
        2_604_537_549,
        "ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c",
    ),
    f"{ESM2_MODEL_NAME}-contact-regression": (
        3_687,
        "8ffe6edbd4173dc8d45c2cd5cb27d43aad77ec26b4c768200c58ae1f96693575",
    ),
    ESMIF_MODEL_NAME: (
        1_700_450_121,
        "be4ba36edec22a9bfaa4946ff6b2815f1f19d8a3d7e0eada8b796d5a0eae9fd4",
    ),
    ESMFOLD_MODEL_NAME: (
        2_771_653_574,
        "e9a52579027e77d2d2e0a18218e755821f395730e86624cab9413dc117f5ca62",
    ),
    ESMFOLD_ESM_MODEL_NAME: (
        5_678_116_398,
        "7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500",
    ),
    f"{ESMFOLD_ESM_MODEL_NAME}-contact-regression": (
        6_759,
        "4da500eab246481dc9c8c95bc7b1d02f2803d761c380b0e95186d4a07d0fc84e",
    ),
}
BOLTZ_REVISION = str(SOFTWARE["boltz"].revision)
BOLTZ_SHA256 = {
    "ccd.pkl": "2d3b2f03a3c5665944adba51e33263511e51b21c9cd05d902f9c4b7c1e58d2f4",
    "boltz1_conf.ckpt": "fea245d912c570ec117b2277c2719f312a6fc109c07b6f6ef741690ee775c2f5",
}
HF_REPOS = {
    "dplm": ("airkingbd/dplm_150m", str(SOFTWARE["dplm"].revision)),
    "esmfold2": ("biohub/ESMFold2", str(SOFTWARE["esmfold2"].revision)),
}
ESMFOLD2_ESMC_REPO = "biohub/ESMC-6B"
ESMFOLD2_ESMC_REVISION = "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
ESM3_REPO = "EvolutionaryScale/esm3-sm-open-v1"
ESM3_REVISION = str(SOFTWARE["esm3"].revision)
HF_SNAPSHOT_REPOS = {
    **HF_REPOS,
    "esm3": (ESM3_REPO, ESM3_REVISION),
    "esmfold2-esmc": (ESMFOLD2_ESMC_REPO, ESMFOLD2_ESMC_REVISION),
}
HF_SNAPSHOT_REQUIRED_FILES = {
    "dplm": (
        "config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.txt",
    ),
    "esmfold2": (
        "ccd.pkl",
        "config.json",
        "model.safetensors",
    ),
    "esmfold2-esmc": (
        "config.json",
        "model-00001-of-00006.safetensors",
        "model-00002-of-00006.safetensors",
        "model-00003-of-00006.safetensors",
        "model-00004-of-00006.safetensors",
        "model-00005-of-00006.safetensors",
        "model-00006-of-00006.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    "esm3": (
        "data/hyperplanes_8bit_58641.npz",
        "data/uniref90_and_mgnify90_residue_annotations_gt_1k_proteins.csv",
        "data/weights/esm3_function_decoder_v0.pth",
        "data/weights/esm3_sm_open_v1.pth",
        "data/weights/esm3_structure_decoder_v0.pth",
        "data/weights/esm3_structure_encoder_v0.pth",
    ),
}
HF_SNAPSHOT_FILE_SPECS = {
    "dplm": {
        "config.json": (792, "36480be9ea79156026ce374ceb9573e0edde62dfccd22d32b107e4de0fd7f03c"),
        "pytorch_model.bin": (595_359_662, "ea4eaa99536b60ed76f945f71a1a5e604f08447ec3def5104a93ca6001a59961"),
        "special_tokens_map.json": (125, "3aedcd4211c0d43aec4e607ff60a63255f3174ead795e997350f09a5f8cd9ee1"),
        "tokenizer_config.json": (1_126, "7d849f80e0f87303b55f370737e3475390c61ae55f4326db1de310711bd6c7e6"),
        "vocab.txt": (93, "0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03"),
    },
    "esmfold2": {
        "ccd.pkl": (417_306_584, "9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5"),
        "config.json": (2_337, "e9ec2496ec433a1dce18627ed4bf3785b4ce0c1d69e4bb4663dad1ab895da012"),
        "model.safetensors": (939_505_228, "138fd4350d6892b81ce6be7ff9bf5a93ae9d4d3751f46a27438a3f9f0dcefa0e"),
    },
    "esm3": {
        "data/hyperplanes_8bit_58641.npz": (30_026_086, "25e680def9abb366a31bdea8fe26720828c778dce80689294744afdbe341e13e"),
        "data/uniref90_and_mgnify90_residue_annotations_gt_1k_proteins.csv": (109_468, "557a053405eddbdf64f13532eabb4a38aa7a652014185a12406940b7fa61d86d"),
        "data/weights/esm3_function_decoder_v0.pth": (1_299_947_178, "f76d074efcaccfe21365a4fa96f212dadd66798e1e49d809ab7ffbe025d227c9"),
        "data/weights/esm3_sm_open_v1.pth": (2_803_646_068, "5ead5a135c658068db6a4f1b933e72d6110992c4668822e1c0e2dcc53e38acd9"),
        "data/weights/esm3_structure_decoder_v0.pth": (1_237_213_098, "3b726258a44274792b40ce7ea307e10c5da09936368a4ffa2970264d909da65b"),
        "data/weights/esm3_structure_encoder_v0.pth": (62_349_276, "467acbaee703ba3ccde6e75241a912a316952e5ff071355f85c1d33c68704f40"),
    },
    "esmfold2-esmc": {
        "config.json": (341, "c5566fab6a17fd674141331fe75de917b7904d99fb7a410d2b1593c21e576913"),
        "model-00001-of-00006.safetensors": (4_864_457_920, "bd90149ff223e6ac1a0cac6147a5ae0df20d3a21df4f65356a1f19cd14f4aa8a"),
        "model-00002-of-00006.safetensors": (4_971_211_344, "f75e2144d8269fe2eb4b3e0823fb089b94f176d8024153e85b8fb573a42294fa"),
        "model-00003-of-00006.safetensors": (4_863_752_992, "f699f01ecc9691d9c6470492765fe54b8b5d2e9f277c139e89427433ffdfe0b2"),
        "model-00004-of-00006.safetensors": (4_971_211_344, "46add1b7be098bbfdc3073884851ba3057f1b33ea23a158b650a37007dabd13d"),
        "model-00005-of-00006.safetensors": (4_863_752_992, "1e1cb62f060a34e18f54a31a76683ef888b8cec59e73315f5b31d25d45a1f88c"),
        "model-00006-of-00006.safetensors": (873_762_296, "56c73e13ae96e777ce65eee99364056069ef93b646470f352f83c5f1037b1b18"),
        "model.safetensors.index.json": (97_349, "6846456e20e6ee2c37461f7bfc21d316d69bdaf165b925691afcb39e583244da"),
        "special_tokens_map.json": (171, "0b7245ec86c8c3aeaf61523ba70dfa79be137e6283f127bd651adc30b4f15c74"),
        "tokenizer.json": (2_879, "8d3447b278176e65fb3ef0224472927bf5fee3be46ea2bd77fad0111423cee1f"),
        "tokenizer_config.json": (1_392, "e8d8e40c9f92b334f0272e80bb65ed4043cb9836523cbae899e9859e8cbb8833"),
    }
}
HF_SNAPSHOT_MANIFEST_DIR = "gh-snapshots"


def _parallel_download(
    jobs: Iterable[tuple[str, Path, int, str | None]],
    *,
    workers: int,
    dry_run: bool,
) -> None:
    work = tuple(jobs)
    if not work:
        return

    def run(job: tuple[str, Path, int, str | None]) -> None:
        url, path, minimum_size, expected_sha256 = job
        download_url(
            url,
            path,
            minimum_size=minimum_size,
            expected_sha256=expected_sha256,
            dry_run=dry_run,
        )

    if workers == 1 or len(work) == 1:
        for job in work:
            run(job)
        return
    with ThreadPoolExecutor(max_workers=min(workers, len(work))) as executor:
        tuple(executor.map(run, work))


def _prefetch_boltz(cache: Path, *, workers: int, dry_run: bool) -> None:
    from astrolabe.evaluators.wrappers.boltz import CCD_URL, MODEL_URL

    ccd_url = CCD_URL.replace("/resolve/main/", f"/resolve/{BOLTZ_REVISION}/")
    model_url = MODEL_URL.replace("/resolve/main/", f"/resolve/{BOLTZ_REVISION}/")
    _parallel_download(
        (
            (ccd_url, cache / "ccd.pkl", 300_000_000, BOLTZ_SHA256["ccd.pkl"]),
            (
                model_url,
                cache / "boltz1_conf.ckpt",
                3_000_000_000,
                BOLTZ_SHA256["boltz1_conf.ckpt"],
            ),
        ),
        workers=workers,
        dry_run=dry_run,
    )


def _prefetch_chai(cache: Path, *, workers: int, dry_run: bool) -> None:
    previous = os.environ.get("CHAI_DOWNLOADS_DIR")
    os.environ["CHAI_DOWNLOADS_DIR"] = str(cache)
    try:
        from chai_lab.data.dataset.embeddings.esm import ESM_URL
        from chai_lab.utils.paths import COMPONENT_URL, cached_conformers
    finally:
        if previous is None:
            os.environ.pop("CHAI_DOWNLOADS_DIR", None)
        else:
            os.environ["CHAI_DOWNLOADS_DIR"] = previous

    jobs = [
        (
            cached_conformers.url,
            cache / "conformers_v1.apkl",
            100_000_000,
            CHAI_CONFORMERS_SHA256,
        ),
        (
            ESM_URL,
            cache / "esm" / "traced_sdpa_esm2_t36_3B_UR50D_fp16.pt",
            5_000_000_000,
            CHAI_ESM_SHA256,
        ),
    ]
    jobs.extend(
        (COMPONENT_URL.format(comp_key=name), cache / "models_v2" / name, size, expected)
        for name, (size, expected) in CHAI_COMPONENTS.items()
    )
    _parallel_download(jobs, workers=workers, dry_run=dry_run)


def _fair_esm_language_model_jobs(
    model_name: str, checkpoints: Path
) -> list[tuple[str, Path, int, str | None]]:
    model_url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
    model_size, model_sha256 = FAIR_ESM_CHECKPOINTS[model_name]
    jobs = [
        (model_url, checkpoints / Path(model_url).name, model_size, model_sha256)
    ]
    regression_name = f"{model_name}-contact-regression"
    regression_url = (
        "https://dl.fbaipublicfiles.com/fair-esm/regression/"
        f"{regression_name}.pt"
    )
    regression_size, regression_sha256 = FAIR_ESM_CHECKPOINTS[regression_name]
    jobs.append(
        (
            regression_url,
            checkpoints / Path(regression_url).name,
            regression_size,
            regression_sha256,
        )
    )
    return jobs


def _prefetch_esm2(torch_cache: Path, *, workers: int, dry_run: bool) -> None:
    checkpoints = torch_cache / "hub" / "checkpoints"
    jobs = _fair_esm_language_model_jobs(ESM2_MODEL_NAME, checkpoints)
    _parallel_download(jobs, workers=workers, dry_run=dry_run)


def _prefetch_esmfold(torch_cache: Path, *, workers: int, dry_run: bool) -> None:
    checkpoints = torch_cache / "hub" / "checkpoints"
    fold_url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{ESMFOLD_MODEL_NAME}.pt"
    fold_size, fold_sha256 = FAIR_ESM_CHECKPOINTS[ESMFOLD_MODEL_NAME]
    jobs = [
        (fold_url, checkpoints / Path(fold_url).name, fold_size, fold_sha256),
        *_fair_esm_language_model_jobs(ESMFOLD_ESM_MODEL_NAME, checkpoints),
    ]
    _parallel_download(jobs, workers=workers, dry_run=dry_run)


def _prefetch_fair_esm_model(
    model_name: str,
    torch_cache: Path,
    *,
    dry_run: bool,
) -> None:
    url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
    size, expected_sha256 = FAIR_ESM_CHECKPOINTS[model_name]
    download_url(
        url,
        torch_cache / "hub" / "checkpoints" / Path(url).name,
        minimum_size=size,
        expected_sha256=expected_sha256,
        dry_run=dry_run,
    )


def hf_snapshot_manifest_path(cache: Path, software: str) -> Path:
    return cache / HF_SNAPSHOT_MANIFEST_DIR / f"{software}.json"


def hf_runtime_snapshot_path(cache: Path, software: str) -> Path:
    revision = HF_SNAPSHOT_REPOS[software][1]
    selection = hashlib.sha256(
        "\0".join(HF_SNAPSHOT_REQUIRED_FILES[software]).encode("utf-8")
    ).hexdigest()[:12]
    return cache / HF_SNAPSHOT_MANIFEST_DIR / software / f"{revision}-{selection}"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, payload: dict) -> None:
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
    _fsync_directory(path.parent)


def _runtime_snapshot_is_exact(
    runtime_snapshot: Path,
    source_snapshot: Path,
    required_files: tuple[str, ...],
) -> bool:
    if not runtime_snapshot.is_dir():
        return False
    expected_entries = set(required_files)
    for rel_path in required_files:
        parent = Path(rel_path).parent
        while str(parent) != ".":
            expected_entries.add(str(parent))
            parent = parent.parent
    actual_entries = {
        str(path.relative_to(runtime_snapshot)) for path in runtime_snapshot.rglob("*")
    }
    if actual_entries != expected_entries:
        return False
    return all(
        (runtime_snapshot / rel_path).is_file()
        and not (runtime_snapshot / rel_path).is_symlink()
        and os.path.samefile(
            runtime_snapshot / rel_path,
            (source_snapshot / rel_path).resolve(),
        )
        for rel_path in required_files
    )


def _materialize_hf_runtime_snapshot(
    *,
    cache: Path,
    software: str,
    source_snapshot: Path,
    required_files: tuple[str, ...],
) -> Path:
    runtime_snapshot = hf_runtime_snapshot_path(cache, software)
    hub_root = (cache / "hub").resolve()
    try:
        source_snapshot.resolve().relative_to(hub_root)
    except ValueError as exc:
        raise SystemExit(
            f"Hugging Face source snapshot escapes managed Hub cache: {source_snapshot}"
        ) from exc
    if _runtime_snapshot_is_exact(runtime_snapshot, source_snapshot, required_files):
        return runtime_snapshot

    runtime_snapshot.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(
        tempfile.mkdtemp(
            dir=runtime_snapshot.parent,
            prefix=f".{runtime_snapshot.name}.",
            suffix=".tmp",
        )
    )
    try:
        for rel_path in required_files:
            source = (source_snapshot / rel_path).resolve()
            try:
                source.relative_to(hub_root)
            except ValueError as exc:
                raise SystemExit(
                    f"Hugging Face asset escapes managed Hub cache: {source}"
                ) from exc
            target = temp / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
        if runtime_snapshot.exists() or runtime_snapshot.is_symlink():
            invalid = runtime_snapshot.with_name(
                f"{runtime_snapshot.name}.invalid.{time.time_ns()}"
            )
            os.replace(runtime_snapshot, invalid)
            print(f"Retained invalid managed HF runtime snapshot at {invalid}")
        os.replace(temp, runtime_snapshot)
        _fsync_directory(runtime_snapshot.parent)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return runtime_snapshot


def _record_hf_snapshot(
    *,
    software: str,
    repo_id: str,
    revision: str,
    cache: Path,
    snapshot: Path,
) -> None:
    expected_repo_id, expected_revision = HF_SNAPSHOT_REPOS[software]
    if repo_id != expected_repo_id or revision != expected_revision:
        raise SystemExit(
            f"Unexpected Hugging Face identity for {software}: {repo_id}@{revision}; "
            f"expected {expected_repo_id}@{expected_revision}"
        )
    required_files = HF_SNAPSHOT_REQUIRED_FILES[software]
    missing = [
        rel_path
        for rel_path in required_files
        if not (snapshot / rel_path).is_file()
        or (snapshot / rel_path).stat().st_size == 0
    ]
    if missing:
        raise SystemExit(
            f"Incomplete Hugging Face snapshot for {software}: " + ", ".join(missing)
        )
    static_specs = HF_SNAPSHOT_FILE_SPECS.get(software, {})
    files = []
    for rel_path in required_files:
        path = snapshot / rel_path
        expected = static_specs.get(rel_path)
        if expected is None:
            actual_sha256 = sha256_file(path)
        else:
            expected_size, expected_sha256 = expected
            validate_file(
                path,
                minimum_size=expected_size,
                expected_sha256=expected_sha256,
                use_adjacent_stamp=False,
            )
            if path.stat().st_size != expected_size:
                raise SystemExit(
                    f"Hugging Face asset size mismatch for {software}:{rel_path}: "
                    f"{path.stat().st_size}; expected exactly {expected_size}"
                )
            actual_sha256 = expected_sha256
        files.append(
            {
                "path": rel_path,
                "size": path.stat().st_size,
                "sha256": actual_sha256,
            }
        )
    source_files = sorted(
        str(path.relative_to(snapshot))
        for path in snapshot.rglob("*")
        if path.is_file()
    )
    runtime_snapshot = _materialize_hf_runtime_snapshot(
        cache=cache,
        software=software,
        source_snapshot=snapshot,
        required_files=required_files,
    )
    _write_json_atomic(
        hf_snapshot_manifest_path(cache, software),
        {
            "schema_version": 2,
            "software": software,
            "repo_id": repo_id,
            "revision": revision,
            "source_snapshot_relative": str(snapshot.relative_to(cache)),
            "runtime_snapshot_relative": str(runtime_snapshot.relative_to(cache)),
            "allow_patterns": list(required_files),
            "required_files": list(required_files),
            "files": files,
            "source_unmanaged_files": sorted(set(source_files) - set(required_files)),
        },
    )


def _prefetch_hf_repo(
    repo_id: str,
    revision: str,
    cache: Path,
    *,
    software: str,
    dry_run: bool,
) -> None:
    required_files = HF_SNAPSHOT_REQUIRED_FILES[software]
    if dry_run:
        print(
            f"+ huggingface snapshot {repo_id}@{revision} -> {cache / 'hub'} "
            f"[{','.join(required_files)}]"
        )
        return
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(cache / "hub"),
            allow_patterns=list(required_files),
        )
    )
    _record_hf_snapshot(
        software=software,
        repo_id=repo_id,
        revision=revision,
        cache=cache,
        snapshot=snapshot,
    )


def _prefetch_esm3(cache: Path, *, dry_run: bool) -> None:
    os.environ["HF_HOME"] = str(cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
    _prefetch_hf_repo(
        ESM3_REPO,
        ESM3_REVISION,
        cache,
        software="esm3",
        dry_run=dry_run,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--software", action="append", required=True)
    parser.add_argument("--boltz-cache", type=Path)
    parser.add_argument("--chai-cache", type=Path)
    parser.add_argument("--torch-cache", type=Path)
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    try:
        selected = set(resolve_software(args.software))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    actions: list[tuple[str, str, Path | None, Callable[[Path], None]]] = [
        (
            "boltz",
            "--boltz-cache",
            args.boltz_cache,
            lambda path: _prefetch_boltz(path, workers=args.jobs, dry_run=args.dry_run),
        ),
        (
            "chai1",
            "--chai-cache",
            args.chai_cache,
            lambda path: _prefetch_chai(path, workers=args.jobs, dry_run=args.dry_run),
        ),
        (
            "esm2",
            "--torch-cache",
            args.torch_cache,
            lambda path: _prefetch_esm2(path, workers=args.jobs, dry_run=args.dry_run),
        ),
        (
            "esm3",
            "--hf-cache",
            args.hf_cache,
            lambda path: _prefetch_esm3(path, dry_run=args.dry_run),
        ),
        (
            "esmif",
            "--torch-cache",
            args.torch_cache,
            lambda path: _prefetch_fair_esm_model(
                ESMIF_MODEL_NAME, path, dry_run=args.dry_run
            ),
        ),
        (
            "esmfold",
            "--torch-cache",
            args.torch_cache,
            lambda path: _prefetch_esmfold(
                path, workers=args.jobs, dry_run=args.dry_run
            ),
        ),
    ]
    for name, option, cache, action in actions:
        if name not in selected:
            continue
        if cache is None:
            raise SystemExit(f"{option} is required for {name}")
        if not args.dry_run:
            cache.mkdir(parents=True, exist_ok=True)
        action(cache)

    if args.hf_cache is not None:
        for name, (repo_id, revision) in HF_REPOS.items():
            if name in selected:
                if not args.dry_run:
                    args.hf_cache.mkdir(parents=True, exist_ok=True)
                _prefetch_hf_repo(
                    repo_id,
                    revision,
                    args.hf_cache,
                    software=name,
                    dry_run=args.dry_run,
                )
                if name == "esmfold2":
                    _prefetch_hf_repo(
                        ESMFOLD2_ESMC_REPO,
                        ESMFOLD2_ESMC_REVISION,
                        args.hf_cache,
                        software="esmfold2-esmc",
                        dry_run=args.dry_run,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
