from __future__ import annotations

import importlib
import sys
import urllib.request
from pathlib import Path

import pytest


def test_protenix_wrapper_import_is_lazy_and_does_not_download(monkeypatch) -> None:
    sys.modules.pop("astrolabe.evaluators.wrappers.protenix", None)

    original_exists = Path.exists

    def _patched_exists(self: Path) -> bool:
        # Simulate missing CCD cache files to ensure any import-time auto-download
        # would be exercised.
        name = self.name
        if name in {"components.v20240608.cif", "components.v20240608.cif.rdkit_mol.pkl"}:
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", _patched_exists, raising=True)
    monkeypatch.setattr(
        urllib.request,
        "urlretrieve",
        lambda *_args, **_kwargs: pytest.fail("urlretrieve() must not run at import time"),
    )

    importlib.import_module("astrolabe.evaluators.wrappers.protenix")


def test_protenix_ccd_cache_uses_writable_runtime_root_and_explicit_assets(
    tmp_path: Path, monkeypatch
) -> None:
    module = importlib.import_module("astrolabe.evaluators.wrappers.protenix")
    runtime_root = tmp_path / "runtime" / "protenix"
    external_root = tmp_path / "external"
    external_root.mkdir()
    ccd = external_root / "components.v20240608.cif"
    ccd_rdkit = external_root / "components.v20240608.cif.rdkit_mol.pkl"
    ccd.write_text("ccd\n", encoding="utf-8")
    ccd_rdkit.write_bytes(b"rdkit")
    readonly_vendor = tmp_path / "readonly-vendor"
    readonly_vendor.mkdir()

    monkeypatch.setenv("PROTENIX_ROOT_DIR", str(runtime_root))
    monkeypatch.setenv("GH_PROTENIX_CCD", str(ccd))
    monkeypatch.setenv("GH_PROTENIX_CCD_RDKIT", str(ccd_rdkit))
    monkeypatch.setattr(module, "_VENDORED_PROTENIX", readonly_vendor)
    monkeypatch.setattr(
        urllib.request,
        "urlretrieve",
        lambda *_args, **_kwargs: pytest.fail("configured CCD assets must be reused"),
    )

    module._ensure_protenix_ccd_cache_files_exist()

    common = runtime_root / "common"
    assert (common / "components.cif").resolve() == ccd.resolve()
    assert (common / "components.cif.rdkit_mol.pkl").resolve() == ccd_rdkit.resolve()
    assert not (readonly_vendor / "release_data").exists()
