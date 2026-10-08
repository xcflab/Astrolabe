from __future__ import annotations

from pathlib import Path

import pytest


def test_merge_pdb_structures_preserves_original_exception(tmp_path) -> None:
    from astrolabe.structure_processors.peptide_docker import PeptideDockingError, merge_pdb_structures

    missing_peptide = tmp_path / "missing_peptide.pdb"
    missing_protein = tmp_path / "missing_protein.pdb"
    out_path = tmp_path / "out.pdb"

    with pytest.raises(PeptideDockingError) as excinfo:
        merge_pdb_structures(
            peptide_file=str(missing_peptide),
            protein_file=str(missing_protein),
            output_file=str(out_path),
            peptide_chain_index=0,
        )

    assert excinfo.value.__cause__ is not None


def test_create_hotspot_mask_preserves_original_exception() -> None:
    import numpy as np

    from astrolabe.core import global_config
    from astrolabe.structure_processors.peptide_docker import PeptideDockingError, create_hotspot_mask

    class _Protomer:
        def select_residues_by_alias(self, _alias):
            raise ValueError("bad alias")

    with pytest.raises(PeptideDockingError) as excinfo:
        global_config.reset()
        global_config.init_const("protomers", {"A": _Protomer()})
        create_hotspot_mask("A", "hotspot")

    assert isinstance(excinfo.value.__cause__, ValueError)


def test_rapidock_predocker_run_impl_preserves_original_exception() -> None:
    from astrolabe.structure_processors.peptide_docker import PeptideDockingError, RAPiDockPeptideDockingProcessor

    predocker = RAPiDockPeptideDockingProcessor.__new__(RAPiDockPeptideDockingProcessor)

    def _boom(_input_file):
        raise RuntimeError("boom")

    predocker._split_complex_chains = _boom  # type: ignore[method-assign]

    with pytest.raises(PeptideDockingError) as excinfo:
        predocker._run_impl("step_000_AB_model.pdb")  # type: ignore[attr-defined]

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_rapidock_predocker_split_complex_chains_preserves_original_exception(tmp_path) -> None:
    from astrolabe.structure_processors.peptide_docker import PeptideDockingError, RAPiDockPeptideDockingProcessor

    predocker = RAPiDockPeptideDockingProcessor.__new__(RAPiDockPeptideDockingProcessor)
    missing_file = tmp_path / "missing.pdb"

    with pytest.raises(PeptideDockingError) as excinfo:
        predocker._split_complex_chains(str(missing_file))  # type: ignore[attr-defined]

    assert excinfo.value.__cause__ is not None


def test_rapidock_predocker_run_impl_uses_runner_bank_and_original_peptide(
    tmp_path,
) -> None:
    from astrolabe.structure_processors.peptide_docker import RAPiDockPeptideDockingProcessor

    predocker = RAPiDockPeptideDockingProcessor.__new__(RAPiDockPeptideDockingProcessor)
    predocker.hotspot = "hotspot"
    predocker.debug_dir = tmp_path / "predocker_existing"
    predocker.debug_dir.mkdir(parents=True, exist_ok=True)

    protein_file = tmp_path / "protein.pdb"
    original_peptide = tmp_path / "original_peptide.pdb"
    candidate_one = tmp_path / "rank1.pdb"
    candidate_two = tmp_path / "rank2.pdb"
    for path, marker in (
        (protein_file, "PROTEIN"),
        (original_peptide, "ORIGINAL"),
        (candidate_one, "CANDIDATE_ONE"),
        (candidate_two, "CANDIDATE_TWO"),
    ):
        path.write_text(f"{marker}\n", encoding="utf-8")

    predocker._split_complex_chains = lambda _input: (  # type: ignore[method-assign]
        str(protein_file),
        str(original_peptide),
        1,
    )
    predocker._run_rapidock_inference = lambda *_args: [  # type: ignore[method-assign]
        str(candidate_one),
        str(candidate_two),
    ]

    observed = {}

    def _evaluate(peptide_files, protein_path, peptide_chain_index, oligo_name):
        observed["peptide_files"] = [Path(p).name for p in peptide_files]
        observed["protein_file"] = Path(protein_path).name
        observed["peptide_chain_index"] = peptide_chain_index
        observed["oligo_name"] = oligo_name
        return str(peptide_files[1])

    predocker._evaluate_docking_results = _evaluate  # type: ignore[method-assign]

    result = predocker._run_impl("step_000_AB_model.pdb")

    assert Path(result).name == "step_000_AB_model__rank2.pdb"
    assert observed == {
        "peptide_files": [
            "step_000_AB_model__rank1.pdb",
            "step_000_AB_model__rank2.pdb",
            "original_peptide.pdb",
        ],
        "protein_file": "protein.pdb",
        "peptide_chain_index": 1,
        "oligo_name": "AB",
    }


def test_rapidock_predocker_run_impl_materializes_runner_candidate_in_debug_dir(
    tmp_path,
) -> None:
    from astrolabe.structure_processors.peptide_docker import RAPiDockPeptideDockingProcessor

    predocker = RAPiDockPeptideDockingProcessor.__new__(RAPiDockPeptideDockingProcessor)
    predocker.hotspot = "hotspot"
    predocker.debug_dir = tmp_path / "predocker"
    predocker.debug_dir.mkdir(parents=True, exist_ok=True)

    protein_file = tmp_path / "protein.pdb"
    original_peptide = tmp_path / "original_peptide.pdb"
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    candidate_one = bank_dir / "rank1.pdb"
    for path, marker in (
        (protein_file, "PROTEIN"),
        (original_peptide, "ORIGINAL"),
        (candidate_one, "CANDIDATE_ONE"),
    ):
        path.write_text(f"{marker}\n", encoding="utf-8")

    predocker._split_complex_chains = lambda _input: (  # type: ignore[method-assign]
        str(protein_file),
        str(original_peptide),
        1,
    )
    predocker._run_rapidock_inference = lambda *_args: [str(candidate_one)]  # type: ignore[method-assign]

    observed = {}

    def _evaluate(peptide_files, protein_path, peptide_chain_index, oligo_name):
        observed["candidate_path"] = Path(peptide_files[0])
        observed["protein_file"] = Path(protein_path).name
        observed["peptide_chain_index"] = peptide_chain_index
        observed["oligo_name"] = oligo_name
        return str(peptide_files[0])

    predocker._evaluate_docking_results = _evaluate  # type: ignore[method-assign]

    result = predocker._run_impl("step_000_AB_model.pdb")

    assert Path(result).parent == predocker.debug_dir
    assert observed["candidate_path"].parent == predocker.debug_dir
    assert observed["candidate_path"].read_text(encoding="utf-8") == "CANDIDATE_ONE\n"


def test_rapidock_predocker_run_impl_merges_candidate_back_to_complex_without_hotspot(
    tmp_path, monkeypatch
) -> None:
    from astrolabe.structure_processors import peptide_docker
    from astrolabe.structure_processors.peptide_docker import RAPiDockPeptideDockingProcessor

    predocker = RAPiDockPeptideDockingProcessor.__new__(RAPiDockPeptideDockingProcessor)
    predocker.hotspot = None
    predocker.debug_dir = tmp_path / "predocker_no_hotspot"
    predocker.debug_dir.mkdir(parents=True, exist_ok=True)

    protein_file = tmp_path / "protein.pdb"
    original_peptide = tmp_path / "original_peptide.pdb"
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    candidate_one = bank_dir / "rank1.pdb"
    for path, marker in (
        (protein_file, "PROTEIN"),
        (original_peptide, "ORIGINAL"),
        (candidate_one, "CANDIDATE_ONE"),
    ):
        path.write_text(f"{marker}\n", encoding="utf-8")

    predocker._split_complex_chains = lambda _input: (  # type: ignore[method-assign]
        str(protein_file),
        str(original_peptide),
        1,
    )
    predocker._run_rapidock_inference = lambda *_args: [str(candidate_one)]  # type: ignore[method-assign]

    observed = {}

    def _fake_merge(peptide_file, protein_path, output_file, peptide_chain_index):
        observed["peptide_file"] = Path(peptide_file)
        observed["protein_file"] = Path(protein_path)
        observed["output_file"] = Path(output_file)
        observed["peptide_chain_index"] = peptide_chain_index
        Path(output_file).write_text("MERGED\n", encoding="utf-8")
        return str(output_file)

    monkeypatch.setattr(peptide_docker, "merge_pdb_structures", _fake_merge)

    result = predocker._run_impl("step_000_AB_model.pdb")

    assert Path(result) == observed["output_file"]
    assert observed["peptide_file"].parent == predocker.debug_dir
    assert observed["protein_file"] == protein_file
    assert observed["peptide_chain_index"] == 1
    assert Path(result).read_text(encoding="utf-8") == "MERGED\n"
