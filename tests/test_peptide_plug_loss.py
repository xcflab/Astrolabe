from pathlib import Path
from types import SimpleNamespace

import pytest

from astrolabe.losses.peptide_plug import (
    compute_peptide_plug_metrics_for_structure,
    peptide_terminal_same_side,
)


def _write_structure(path: Path, peptide_z: list[float]) -> None:
    lines = []
    serial = 1
    for chain_id, x, y in (("B", 8.0, 0.0), ("C", -4.0, 6.928), ("D", -4.0, -6.928)):
        for residue_index, z in enumerate(range(-20, 21, 4), start=1):
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA {chain_id}{residue_index:4d}    "
                f"{x:8.3f}{y:8.3f}{float(z):8.3f}  1.00 50.00           C\n"
            )
            serial += 1
    for residue_index, z in enumerate(peptide_z, start=1):
        lines.append(
            f"ATOM  {serial:5d}  CA  GLY A{residue_index:4d}    "
            f"{0.5 * residue_index:8.3f}{0.0:8.3f}{z:8.3f}  1.00 50.00           C\n"
        )
        serial += 1
    lines.append("END\n")
    path.write_text("".join(lines))


def test_same_side_peptide_is_scaffoldable_and_threaded_peptide_is_not(
    tmp_path: Path,
) -> None:
    same_side = tmp_path / "same_side.pdb"
    threaded = tmp_path / "threaded.pdb"
    _write_structure(same_side, [-14.0, -12.0, -10.0, -8.0, -6.0, -8.0, -10.0, -12.0, -14.0])
    _write_structure(threaded, [-14.0, -10.5, -7.0, -3.5, 0.0, 3.5, 7.0, 10.5, 14.0])

    same_metrics = compute_peptide_plug_metrics_for_structure(
        same_side,
        binder_chain="A",
        target_chains="B,C,D",
    )
    threaded_metrics = compute_peptide_plug_metrics_for_structure(
        threaded,
        binder_chain="A",
        target_chains="B,C,D",
    )

    assert same_metrics["terminal_same_side"] == 1.0
    assert same_metrics["scaffoldability"] > 0.0
    assert threaded_metrics["terminal_same_side"] == 0.0
    assert threaded_metrics["scaffoldability"] == 0.0


def test_registered_loss_reads_complex_path(tmp_path: Path) -> None:
    structure = tmp_path / "same_side.pdb"
    _write_structure(structure, [-14.0, -12.0, -10.0, -8.0, -6.0, -8.0, -10.0, -12.0, -14.0])
    eval_result = SimpleNamespace(
        complex_paths={"ABCD": structure},
        names=["ABCD"],
    )

    result = peptide_terminal_same_side(
        eval_result,
        binder_chain="A",
        target_chains=["B", "C", "D"],
    )

    assert result == {"ABCD": pytest.approx(1.0)}
