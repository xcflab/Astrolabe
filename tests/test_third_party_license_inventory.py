from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "release" / "generate_third_party_licenses.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_test_third_party_licenses", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_release_license_inventory_has_no_unreviewed_components() -> None:
    module = _load_module()
    rows = {row["component"]: row for row in module._component_rows(REPO_ROOT)}

    assert "__pycache__" not in rows
    assert rows["protenix"]["status"] == "ok"
    assert "LICENSE" in rows["protenix"]["licenses"]
    assert rows["boost-regex-source"]["status"] == "ok"
    assert "third_party_licenses/boost-regex-source/LICENSE_1_0.txt" in rows[
        "boost-regex-source"
    ]["licenses"]
    assert rows["eigen-source"]["status"] == "ok"
    assert "COPYING.MPL2" in rows["eigen-source"]["licenses"]
    assert rows["surfdesign"]["status"] == "excluded_from_runtime_release"
    assert rows["surfpro"]["status"] == "excluded_from_runtime_release"
    assert rows["willutil"]["status"] == "excluded_from_runtime_release"
    assert rows["ligandmpnn-derived"]["status"] == "ok"
    assert rows["ligandmpnn-derived"]["path"] == (
        "astrolabe/mutators/wrappers/ligandmpnn_utils.py"
    )
    assert rows["ligandmpnn-derived"]["url"].startswith(
        "https://github.com/dauparas/LigandMPNN/"
    )
    assert rows["ligandmpnn-derived"]["licenses"] == (
        "third_party_licenses/ligandmpnn/LICENSE"
    )
    assert not any(
        row["status"] == "license_file_not_found_top_level" for row in rows.values()
    )


def test_ligandmpnn_provenance_header_and_complete_license() -> None:
    source = (REPO_ROOT / "astrolabe/mutators/wrappers/ligandmpnn_utils.py").read_text(
        encoding="utf-8"
    )
    assert "https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de" in source
    assert "data_utils.py, model_utils.py, and sc_utils.py" in source
    assert "Copyright (c) 2024 Justas Dauparas" in source
    assert "MIT License" in source
    assert "third_party_licenses/ligandmpnn/LICENSE" in source
    assert "Local modifications" in source

    license_text = (REPO_ROOT / "third_party_licenses/ligandmpnn/LICENSE").read_text(
        encoding="utf-8"
    )
    license_bytes = license_text.encode("utf-8")
    assert len(license_bytes) == 1072
    assert hashlib.sha256(license_bytes).hexdigest() == (
        "8af7ba31868e08e8949e20ce2ef3c0b1d0ef78c9358c91563a60a9fe3b36c9ea"
    )
    assert license_text.startswith("MIT License\n\nCopyright (c) 2024 Justas Dauparas\n")
    assert "Permission is hereby granted" in license_text
    assert "The above copyright notice and this permission notice shall be included" in license_text
    assert "THE SOFTWARE IS PROVIDED \"AS IS\"" in license_text
    assert license_text.rstrip().endswith("SOFTWARE.")


def test_ligandmpnn_inventory_requires_source_and_license_evidence(tmp_path: Path) -> None:
    module = _load_module()
    source_path = tmp_path / "astrolabe/mutators/wrappers/ligandmpnn_utils.py"
    license_path = tmp_path / "third_party_licenses/ligandmpnn/LICENSE"
    source_path.parent.mkdir(parents=True)
    license_path.parent.mkdir(parents=True)
    source_path.write_bytes(
        (REPO_ROOT / "astrolabe/mutators/wrappers/ligandmpnn_utils.py").read_bytes()
    )
    license_path.write_bytes(
        (REPO_ROOT / "third_party_licenses/ligandmpnn/LICENSE").read_bytes()
    )

    row = next(
        row
        for row in module._component_rows(tmp_path)
        if row["component"] == "ligandmpnn-derived"
    )
    assert row["status"] == "ok"

    source_path.write_bytes(
        source_path.read_bytes().replace(b"# Derived from", b"# Tampered from", 1)
    )
    tampered_header = next(
        row
        for row in module._component_rows(tmp_path)
        if row["component"] == "ligandmpnn-derived"
    )
    assert tampered_header["status"] != "ok"

    source_path.write_bytes(
        (REPO_ROOT / "astrolabe/mutators/wrappers/ligandmpnn_utils.py").read_bytes()
    )
    license_path.write_bytes(license_path.read_bytes() + b"tampered")
    tampered_license = next(
        row
        for row in module._component_rows(tmp_path)
        if row["component"] == "ligandmpnn-derived"
    )
    assert tampered_license["status"] != "ok"

    license_path.unlink()
    missing_license = next(
        row
        for row in module._component_rows(tmp_path)
        if row["component"] == "ligandmpnn-derived"
    )
    assert missing_license["status"] != "ok"

    license_path.write_bytes(
        (REPO_ROOT / "third_party_licenses/ligandmpnn/LICENSE").read_bytes()
    )
    source_path.unlink()
    missing_source = next(
        row
        for row in module._component_rows(tmp_path)
        if row["component"] == "ligandmpnn-derived"
    )
    assert missing_source["status"] != "ok"
