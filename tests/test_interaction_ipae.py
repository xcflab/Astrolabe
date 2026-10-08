from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.losses.interaction import (
    compute_interface_pae,
    ipae,
    ipae_max,
    ipae_min,
    ipae_minimal,
)


def _atom_line(
    serial: int,
    atom: str,
    chain: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    *,
    resname: str = "ALA",
) -> str:
    element = atom.strip()[0]
    return (
        f"ATOM  {serial:5d} {atom:^4s} {resname:>3s} {chain}{resseq:4d}"
        f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {element:>2s}"
    )


def _write_pdb(path: Path, atoms: list[tuple[str, str, int, float, float, float]]) -> None:
    lines = [
        _atom_line(i + 1, atom, chain, resseq, x, y, z)
        for i, (atom, chain, resseq, x, y, z) in enumerate(atoms)
    ]
    path.write_text("\n".join(lines + ["TER", "END", ""]), encoding="utf-8")


def _eval_result(path: Path, pae: np.ndarray):
    return SimpleNamespace(
        names=["AB"],
        pae={"AB": [pae]},
        complex_paths={"AB": path},
        chain_pair_scores={},
    )


def test_ipae_variants_use_cross_chain_contact_pairs_not_interface_union(tmp_path: Path) -> None:
    structure = tmp_path / "contacts.pdb"
    _write_pdb(
        structure,
        [
            ("CA", "A", 1, 0.0, 0.0, 0.0),
            ("CA", "A", 2, 100.0, 0.0, 0.0),
            ("CA", "B", 1, 5.0, 0.0, 0.0),
            ("CA", "B", 2, 200.0, 0.0, 0.0),
        ],
    )
    pae = np.array(
        [
            [0.0, 0.0, 2.0, 90.0],
            [0.0, 0.0, 90.0, 90.0],
            [10.0, 90.0, 0.0, 0.0],
            [90.0, 90.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    result = _eval_result(structure, pae)

    assert compute_interface_pae(
        result,
        symmetric="mean",
        chain_a="A",
        chain_b="B",
        contact_method="ca",
    )["AB"] == pytest.approx(6.0)
    assert ipae(result, contact_method="ca")["AB"] == pytest.approx(6.0)
    assert ipae_min(result, contact_method="ca")["AB"] == pytest.approx(2.0)
    assert ipae_minimal(result, contact_method="ca")["AB"] == pytest.approx(2.0)
    assert ipae_max(result, contact_method="ca")["AB"] == pytest.approx(10.0)


def test_ipae_contact_method_can_use_sc_contact_when_ca_is_far(tmp_path: Path) -> None:
    structure = tmp_path / "heavy_contact.pdb"
    _write_pdb(
        structure,
        [
            ("CA", "A", 1, 0.0, 0.0, 0.0),
            ("CB", "A", 1, 0.0, 0.0, 0.0),
            ("CA", "B", 1, 12.0, 0.0, 0.0),
            ("CB", "B", 1, 4.5, 0.0, 0.0),
        ],
    )
    result = _eval_result(structure, np.array([[0.0, 4.0], [8.0, 0.0]], dtype=float))

    assert np.isnan(
        compute_interface_pae(
            result,
            symmetric="mean",
            chain_a="A",
            chain_b="B",
            contact_method="ca",
        )["AB"]
    )
    assert compute_interface_pae(
        result,
        symmetric="mean",
        chain_a="A",
        chain_b="B",
        contact_method="sc",
    )["AB"] == pytest.approx(6.0)


def test_ipae_returns_nan_when_no_interchain_contact_pairs(tmp_path: Path) -> None:
    structure = tmp_path / "no_contacts.pdb"
    _write_pdb(
        structure,
        [
            ("CA", "A", 1, 0.0, 0.0, 0.0),
            ("CA", "B", 1, 50.0, 0.0, 0.0),
        ],
    )
    result = _eval_result(structure, np.array([[0.0, 4.0], [8.0, 0.0]], dtype=float))

    assert np.isnan(ipae(result, contact_method="ca")["AB"])
    assert np.isnan(ipae_min(result, contact_method="sc")["AB"])
