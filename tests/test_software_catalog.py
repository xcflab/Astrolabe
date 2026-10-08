from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "installation"))

from software_catalog import (
    SOFTWARE,
    catalog_payload,
    resolve_software,
    selected_host_envs,
    selected_submodules,
)


def test_profiles_expand_without_duplicates() -> None:
    selected = resolve_software(("core,language", "core"))
    assert selected[:6] == (
        "alphafold3",
        "alphafold2",
        "ligandmpnn",
        "protenix",
        "boltz",
        "pyrosetta",
    )
    assert len(selected) == len(set(selected))
    assert "esmfold2" in selected


def test_dependency_closure_is_explicit() -> None:
    assert resolve_software(("dynamicmpnn",)) == ("dynamicmpnn", "ligandmpnn")
    assert resolve_software(("autodock-gpu",)) == (
        "autodock-gpu",
        "autodock-vina",
    )


def test_selected_install_owners_are_minimal() -> None:
    selected = resolve_software(("dplm,esmfold2",))
    assert selected_host_envs(selected) == ("dplm", "esmfold2")
    assert selected_submodules(selected) == (
        "vendor/dplm",
        "vendor/biohub_esm",
        "vendor/biohub_transformers",
    )


def test_unknown_selection_fails_with_catalog() -> None:
    with pytest.raises(ValueError, match="Unknown software selection"):
        resolve_software(("not-a-backend",))


def test_catalog_json_contains_source_and_license_boundaries() -> None:
    payload = json.loads(json.dumps(catalog_payload()))
    assert set(payload["software"]) == set(SOFTWARE)
    assert payload["software"]["alphafold3"]["asset_policy"] == "restricted"
    assert payload["software"]["esm3"]["license_id"] == "esm3-noncommercial"
    assert payload["software"]["esm3"]["asset_urls"] == [
        "https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1"
    ]
    assert payload["software"]["rhofold"]["asset_urls"]
    assert payload["software"]["rfd3"]["asset_policy"] == "public-checkpoint"
    assert payload["software"]["ananas"]["license_id"] == "ananas-academic"
    assert payload["software"]["rosetta"]["revision"] == (
        "80a0635615099e1b918474a63acba7b1de6fd107"
    )
    assert payload["software"]["surfpro"]["asset_policy"] == "public"
    assert payload["software"]["pyrosetta"]["revision"] == (
        "2026.3+releasequarterly.5e498f1409"
    )
    assert payload["software"]["pyrosetta"]["asset_urls"] == [
        "https://west.rosettacommons.org/pyrosetta/quarterly/release/"
        "pyrosetta-2026.3+releasequarterly.5e498f1409-cp311-cp311-linux_x86_64.whl"
    ]
    assert "pyrosetta" in payload["profiles"]["core"]
    assert "rfd3" in payload["profiles"]["all"]
