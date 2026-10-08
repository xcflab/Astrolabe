from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from installation import asset_prefetch as gh_asset_prefetch


def test_prefetch_exact_selection_does_not_run_unselected_downloaders(
    tmp_path: Path, monkeypatch
) -> None:
    called = []
    monkeypatch.setattr(
        gh_asset_prefetch,
        "_prefetch_boltz",
        lambda path, **_kwargs: called.append(("boltz", path)),
    )
    monkeypatch.setattr(
        gh_asset_prefetch,
        "_prefetch_chai",
        lambda path, **_kwargs: called.append(("chai1", path)),
    )

    rc = gh_asset_prefetch.main(
        [
            "--software",
            "boltz",
            "--boltz-cache",
            str(tmp_path / "boltz"),
            "--dry-run",
        ]
    )

    assert rc == 0
    assert called == [("boltz", tmp_path / "boltz")]


def test_esmfold_prefetch_includes_underlying_esm_language_model(
    tmp_path: Path, monkeypatch
) -> None:
    captured = []
    monkeypatch.setattr(
        gh_asset_prefetch,
        "_parallel_download",
        lambda jobs, **_kwargs: captured.extend(jobs),
    )

    gh_asset_prefetch._prefetch_esmfold(
        tmp_path / "torch", workers=3, dry_run=False
    )

    assert [path.name for _url, path, _size, _sha in captured] == [
        "esmfold_3B_v1.pt",
        "esm2_t36_3B_UR50D.pt",
        "esm2_t36_3B_UR50D-contact-regression.pt",
    ]
    assert all(
        size > 0 and sha is not None and len(sha) == 64
        for _url, _path, size, sha in captured
    )


def test_hf_prefetch_is_pinned_to_catalog_revision(tmp_path: Path, monkeypatch) -> None:
    called = []
    monkeypatch.setattr(
        gh_asset_prefetch,
        "_prefetch_hf_repo",
        lambda repo, revision, cache, **_kwargs: called.append((repo, revision, cache)),
    )

    rc = gh_asset_prefetch.main(
        [
            "--software",
            "dplm",
            "--hf-cache",
            str(tmp_path / "hf"),
            "--dry-run",
        ]
    )

    assert rc == 0
    assert called == [
        (
            "airkingbd/dplm_150m",
            "49b7125a5d28c6418fcc2f3c4fe799352ac1488b",
            tmp_path / "hf",
        )
    ]


def test_hf_prefetch_limits_files_and_writes_exact_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    cache = tmp_path / "hf"
    revision = gh_asset_prefetch.HF_REPOS["dplm"][1]
    snapshot = (
        cache
        / "hub"
        / "models--airkingbd--dplm_150m"
        / "snapshots"
        / revision
    )
    calls = []
    fake_payload = b"asset\n"
    monkeypatch.setitem(
        gh_asset_prefetch.HF_SNAPSHOT_FILE_SPECS,
        "dplm",
        {
            rel_path: (
                len(fake_payload),
                gh_asset_prefetch.hashlib.sha256(fake_payload).hexdigest(),
            )
            for rel_path in gh_asset_prefetch.HF_SNAPSHOT_REQUIRED_FILES["dplm"]
        },
    )

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        for rel_path in gh_asset_prefetch.HF_SNAPSHOT_REQUIRED_FILES["dplm"]:
            path = snapshot / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("asset\n", encoding="utf-8")
        extra = snapshot / "README.md"
        extra.write_text("pre-existing\n", encoding="utf-8")
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )

    gh_asset_prefetch._prefetch_hf_repo(
        "airkingbd/dplm_150m",
        revision,
        cache,
        software="dplm",
        dry_run=False,
    )

    assert calls == [
        {
            "repo_id": "airkingbd/dplm_150m",
            "revision": revision,
            "cache_dir": str(cache / "hub"),
            "allow_patterns": list(
                gh_asset_prefetch.HF_SNAPSHOT_REQUIRED_FILES["dplm"]
            ),
        }
    ]
    manifest = json.loads(
        gh_asset_prefetch.hf_snapshot_manifest_path(cache, "dplm").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["revision"] == revision
    assert manifest["allow_patterns"] == list(
        gh_asset_prefetch.HF_SNAPSHOT_REQUIRED_FILES["dplm"]
    )
    assert manifest["source_unmanaged_files"] == ["README.md"]
    assert [entry["path"] for entry in manifest["files"]] == list(
        gh_asset_prefetch.HF_SNAPSHOT_REQUIRED_FILES["dplm"]
    )
    assert all(entry["size"] > 0 and len(entry["sha256"]) == 64 for entry in manifest["files"])
    runtime_snapshot = gh_asset_prefetch.hf_runtime_snapshot_path(cache, "dplm")
    assert sorted(
        str(path.relative_to(runtime_snapshot))
        for path in runtime_snapshot.rglob("*")
        if path.is_file()
    ) == sorted(gh_asset_prefetch.HF_SNAPSHOT_REQUIRED_FILES["dplm"])
    assert not (runtime_snapshot / "README.md").exists()
    assert all(
        not (runtime_snapshot / rel_path).is_symlink()
        and (runtime_snapshot / rel_path).samefile(snapshot / rel_path)
        for rel_path in gh_asset_prefetch.HF_SNAPSHOT_REQUIRED_FILES["dplm"]
    )
    shutil.rmtree(snapshot)
    assert all(
        (runtime_snapshot / rel_path).is_file()
        for rel_path in gh_asset_prefetch.HF_SNAPSHOT_REQUIRED_FILES["dplm"]
    )


def test_hf_static_contract_covers_every_selected_snapshot_file() -> None:
    for software in ("dplm", "esm3", "esmfold2", "esmfold2-esmc"):
        assert set(gh_asset_prefetch.HF_SNAPSHOT_FILE_SPECS[software]) == set(
            gh_asset_prefetch.HF_SNAPSHOT_REQUIRED_FILES[software]
        )


def test_hf_snapshot_recorder_rejects_wrong_catalog_identity(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Unexpected Hugging Face identity"):
        gh_asset_prefetch._record_hf_snapshot(
            software="dplm",
            repo_id="airkingbd/dplm_150m",
            revision="wrong-revision",
            cache=tmp_path,
            snapshot=tmp_path / "source",
        )


def test_hf_runtime_snapshot_rejects_source_outside_hub_cache(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="escapes managed Hub cache"):
        gh_asset_prefetch._materialize_hf_runtime_snapshot(
            cache=tmp_path / "hf",
            software="dplm",
            source_snapshot=tmp_path / "outside",
            required_files=gh_asset_prefetch.HF_SNAPSHOT_REQUIRED_FILES["dplm"],
        )
