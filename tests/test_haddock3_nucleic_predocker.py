from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.na_mpnn import NAMPNNMutator
from astrolabe.structure_processors.base import (
    StructureProcessorRegistry,
    initialize_predockers,
    run_nested_predocker_on_scores,
)


def _atom(
    serial: int,
    *,
    name: str,
    resname: str,
    chain: str,
    resseq: int,
    xyz: tuple[float, float, float],
    element: str,
    hetatm: bool = False,
) -> str:
    record = "HETATM" if hetatm else "ATOM  "
    x, y, z = xyz
    return (
        f"{record}{serial:5d} {name:^4s} {resname:>3s} {chain:1s}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}"
    )


def _write_far_dna_complex(path: Path) -> Path:
    lines = [
        _atom(1, name="N", resname="ALA", chain="A", resseq=1, xyz=(0.0, 0.0, 0.0), element="N"),
        _atom(2, name="CA", resname="ALA", chain="A", resseq=1, xyz=(1.5, 0.0, 0.0), element="C"),
        _atom(3, name="C", resname="ALA", chain="A", resseq=1, xyz=(2.5, 0.0, 0.0), element="C"),
        "TER",
        _atom(4, name="P", resname="DA", chain="B", resseq=1, xyz=(30.0, 0.0, 0.0), element="P"),
        _atom(5, name="C1'", resname="DA", chain="B", resseq=1, xyz=(31.0, 0.0, 0.0), element="C"),
        _atom(6, name="P", resname="DT", chain="B", resseq=2, xyz=(33.0, 0.0, 0.0), element="P"),
        "TER",
        "END",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_far_dsdna_complex(path: Path) -> Path:
    lines = [
        _atom(1, name="N", resname="ALA", chain="A", resseq=1, xyz=(0.0, 0.0, 0.0), element="N"),
        _atom(2, name="CA", resname="ALA", chain="A", resseq=1, xyz=(1.5, 0.0, 0.0), element="C"),
        _atom(3, name="C", resname="ALA", chain="A", resseq=1, xyz=(2.5, 0.0, 0.0), element="C"),
        "TER",
        _atom(4, name="P", resname="DA", chain="B", resseq=1, xyz=(30.0, 0.0, 0.0), element="P"),
        _atom(5, name="C1'", resname="DA", chain="B", resseq=1, xyz=(31.0, 0.0, 0.0), element="C"),
        _atom(6, name="C2'", resname="DA", chain="B", resseq=1, xyz=(30.0, 1.0, 0.0), element="C"),
        _atom(7, name="P", resname="DT", chain="B", resseq=2, xyz=(33.0, 0.0, 0.0), element="P"),
        "TER",
        _atom(8, name="P", resname="DT", chain="C", resseq=1, xyz=(30.0, 8.0, 0.0), element="P"),
        _atom(9, name="C1'", resname="DT", chain="C", resseq=1, xyz=(31.0, 8.0, 0.0), element="C"),
        _atom(10, name="C2'", resname="DT", chain="C", resseq=1, xyz=(30.0, 9.0, 0.0), element="C"),
        _atom(11, name="P", resname="DA", chain="C", resseq=2, xyz=(33.0, 8.0, 0.0), element="P"),
        "TER",
        "END",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_contact_candidate(path: Path) -> Path:
    lines = [
        _atom(1, name="N", resname="ALA", chain="A", resseq=1, xyz=(0.0, 0.0, 0.0), element="N"),
        _atom(2, name="CA", resname="ALA", chain="A", resseq=1, xyz=(1.5, 0.0, 0.0), element="C"),
        "TER",
        _atom(3, name="P", resname="DA", chain="B", resseq=1, xyz=(2.5, 0.0, 0.0), element="P"),
        _atom(4, name="C1'", resname="DA", chain="B", resseq=1, xyz=(3.0, 0.0, 0.0), element="C"),
        "TER",
        "END",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_fake_haddock3(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from pathlib import Path",
                "import sys",
                "cfg = Path(sys.argv[1])",
                "run = cfg.parent / 'haddock3_run' / '1_rigidbody'",
                "run.mkdir(parents=True, exist_ok=True)",
                "(run / 'rigidbody_1.pdb').write_text('''"
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\\n"
                "ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00 20.00           C\\n"
                "ATOM      3  C   ALA A   1       2.500   0.000   0.000  1.00 20.00           C\\n"
                "TER\\n"
                "ATOM      4  P    DA B   1      20.000   0.000   0.000  1.00 20.00           P\\n"
                "TER\\n"
                "END\\n"
                "''')",
                "(run / 'rigidbody_2.pdb').write_text('''"
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\\n"
                "ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00 20.00           C\\n"
                "ATOM      3  C   ALA A   1       2.500   0.000   0.000  1.00 20.00           C\\n"
                "TER\\n"
                "ATOM      4  P    DA B   1       2.500   0.000   0.000  1.00 20.00           P\\n"
                "ATOM      5  C1'  DA B   1       3.500   0.000   0.000  1.00 20.00           C\\n"
                "ATOM      6  C2'  DA B   1       2.500   1.000   0.000  1.00 20.00           C\\n"
                "ATOM      7  P    DT B   2       5.500   0.000   0.000  1.00 20.00           P\\n"
                "TER\\n"
                "END\\n"
                "''')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _write_fake_haddock3_gz(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from pathlib import Path",
                "import gzip",
                "import sys",
                "cfg = Path(sys.argv[1])",
                "run = cfg.parent / 'haddock3_run' / '1_rigidbody'",
                "run.mkdir(parents=True, exist_ok=True)",
                "pdb = '''"
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\\n"
                "ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00 20.00           C\\n"
                "ATOM      3  C   ALA A   1       2.500   0.000   0.000  1.00 20.00           C\\n"
                "TER\\n"
                "ATOM      4  P    DA B   1       2.500   0.000   0.000  1.00 20.00           P\\n"
                "ATOM      5  C1'  DA B   1       3.500   0.000   0.000  1.00 20.00           C\\n"
                "ATOM      6  C2'  DA B   1       2.500   1.000   0.000  1.00 20.00           C\\n"
                "ATOM      7  P    DT B   2       5.500   0.000   0.000  1.00 20.00           P\\n"
                "TER\\n"
                "END\\n"
                "'''",
                "with gzip.open(run / 'rigidbody_1.pdb.gz', 'wt') as handle:",
                "    handle.write(pdb)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _init_global(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)
    global_config.init_const("dynamic_binding_site", None)
    global_config.init_const("dynamic_binding_site_enabled", False)
    global_config.set_variable("ignored_complexes", [])
    global_config.set_variable("step", 0)


def _metrics(path: Path):
    from astrolabe.structure_processors.haddock3 import _contact_metrics

    return _contact_metrics(
        path,
        binder_chain="A",
        target_chains={"B"},
        contact_cutoff=4.0,
    )


def _chain_ids(path: Path) -> list[str]:
    from astrolabe.structure_processors.haddock3 import _first_model

    return [chain.id for chain in _first_model(path)]


def test_registry_lazy_imports_haddock3_nucleic_predocker() -> None:
    cls = StructureProcessorRegistry.get("haddock3_nucleic")
    assert cls.__name__ == "HADDOCK3NucleicDockingProcessor"


def test_initialize_haddock3_nucleic_uses_canonical_defaults(tmp_path: Path) -> None:
    _init_global(tmp_path)
    fake_haddock3 = _write_fake_haddock3(tmp_path / "fake_haddock3.py")

    initialize_predockers(
        OmegaConf.create(
            {
                "haddock3_nucleic": {
                    "haddock3_bin": str(fake_haddock3),
                    "sampling": 7,
                    "ncores": 3,
                },
                "predocker_terms": {
                    "dock": {"_type": "haddock3_nucleic"},
                },
            }
        )
    )

    predocker = global_config.predockers["dock"]
    assert predocker.haddock3_bin == str(fake_haddock3)
    assert predocker.sampling == 7
    assert predocker.ncores == 3


def test_initialize_haddock3_nucleic_accepts_legacy_default_base(tmp_path: Path) -> None:
    _init_global(tmp_path)
    fake_haddock3 = _write_fake_haddock3(tmp_path / "fake_haddock3.py")

    initialize_predockers(
        OmegaConf.create(
            {
                "haddock3": {
                    "haddock3_bin": str(fake_haddock3),
                    "sampling": 5,
                    "max_restraints": 11,
                },
                "predocker_terms": {
                    "dock": {"_type": "haddock3_nucleic"},
                },
            }
        )
    )

    predocker = global_config.predockers["dock"]
    assert predocker.haddock3_bin == str(fake_haddock3)
    assert predocker.sampling == 5
    assert predocker.max_restraints == 11


def test_initialize_haddock3_nucleic_alias_uses_canonical_defaults(tmp_path: Path) -> None:
    _init_global(tmp_path)
    fake_haddock3 = _write_fake_haddock3(tmp_path / "fake_haddock3.py")

    initialize_predockers(
        OmegaConf.create(
            {
                "haddock3_nucleic": {
                    "haddock3_bin": str(fake_haddock3),
                    "sampling": 6,
                },
                "predocker_terms": {
                    "dock": {"_type": "HADDOCK3_nucleic"},
                },
            }
        )
    )

    predocker = global_config.predockers["dock"]
    assert predocker.haddock3_bin == str(fake_haddock3)
    assert predocker.sampling == 6


def test_initialize_haddock3_nucleic_term_overrides_defaults(tmp_path: Path) -> None:
    _init_global(tmp_path)
    fake_haddock3 = _write_fake_haddock3(tmp_path / "fake_haddock3.py")

    initialize_predockers(
        OmegaConf.create(
            {
                "haddock3_nucleic": {
                    "haddock3_bin": str(fake_haddock3),
                    "sampling": 6,
                    "contact_cutoff": 4.0,
                },
                "predocker_terms": {
                    "dock": {
                        "_type": "haddock3_nucleic",
                        "sampling": 2,
                        "contact_cutoff": 5.5,
                    },
                },
            }
        )
    )

    predocker = global_config.predockers["dock"]
    assert predocker.sampling == 2
    assert predocker.contact_cutoff == 5.5


def test_haddock3_nucleic_predocker_replaces_no_contact_complex(tmp_path: Path) -> None:
    _init_global(tmp_path)
    input_path = _write_far_dna_complex(tmp_path / "step_000_AB_model.pdb")
    fake_haddock3 = _write_fake_haddock3(tmp_path / "fake_haddock3.py")
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [0.9]},
        complex_paths={"AB": input_path},
    )
    predocker = StructureProcessorRegistry.get("haddock3_nucleic")(
        haddock3_bin=str(fake_haddock3),
        sampling=2,
        ncores=1,
        cache_enabled=False,
    )

    updated = predocker.run_on_scores(scores)

    assert updated is not scores
    assert Path(updated.complex_paths["AB"]).exists()
    assert Path(updated.complex_paths["AB"]) != input_path
    assert _metrics(input_path).contact_count == 0
    assert _metrics(Path(updated.complex_paths["AB"])).contact_count > 0

    work_dirs = list((tmp_path / "predocker").glob("haddock3_work_*"))
    assert len(work_dirs) == 1
    cfg_text = (work_dirs[0] / "anchor_B" / "haddock3.cfg").read_text(encoding="utf-8")
    restraints_text = (work_dirs[0] / "anchor_B" / "restraints.tbl").read_text(encoding="utf-8")
    assert "[rigidbody]" in cfg_text
    assert "ambig_fname" in cfg_text
    assert "assign (segid B and resid 1) (segid A)" in restraints_text


def test_haddock3_nucleic_updates_dynamic_binding_site_for_scores(tmp_path: Path) -> None:
    _init_global(tmp_path)
    input_path = _write_far_dna_complex(tmp_path / "step_000_AB_model.pdb")
    fake_haddock3 = _write_fake_haddock3(tmp_path / "fake_haddock3.py")
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [0.9]},
        complex_paths={"AB": input_path},
    )
    predocker = StructureProcessorRegistry.get("haddock3_nucleic")(
        haddock3_bin=str(fake_haddock3),
        sampling=2,
        ncores=1,
        cache_enabled=False,
        max_binding_site_pairs=4,
        binding_site_max_distance=5.5,
    )

    updated = predocker.run_on_scores(scores)

    dynamic = global_config.dynamic_binding_site
    assert dynamic == {
        "AB": {
            "binder": "A:1",
            "target": "B:1,2",
            "mode": "both",
            "max_distance": 5.5,
            "min_distance": 0.0,
        }
    }
    assert Path(updated.complex_paths["AB"]).exists()


def test_haddock3_nucleic_reconstructs_full_target_after_anchor_docking(tmp_path: Path) -> None:
    _init_global(tmp_path)
    input_path = _write_far_dsdna_complex(tmp_path / "step_000_ABC_model.pdb")
    fake_haddock3 = _write_fake_haddock3(tmp_path / "fake_haddock3.py")
    predocker = StructureProcessorRegistry.get("haddock3_nucleic")(
        haddock3_bin=str(fake_haddock3),
        target_chains=["B", "C"],
        anchor_chains=["B"],
        sampling=2,
        ncores=1,
        cache_enabled=False,
    )

    output = Path(predocker.run(str(input_path)))

    assert output.exists()
    assert _chain_ids(output) == ["A", "B", "C"]
    assert _metrics(input_path).contact_count == 0
    assert _metrics(output).contact_count > 0

    work_dirs = list((tmp_path / "predocker").glob("haddock3_work_*"))
    assert len(work_dirs) == 1
    cfg_text = (work_dirs[0] / "anchor_B" / "haddock3.cfg").read_text(encoding="utf-8")
    restraints_text = (work_dirs[0] / "anchor_B" / "restraints.tbl").read_text(encoding="utf-8")
    assert "target_B.pdb" in cfg_text
    assert "segid C" not in restraints_text


def test_haddock3_nucleic_accepts_compressed_rigidbody_candidates(tmp_path: Path) -> None:
    _init_global(tmp_path)
    input_path = _write_far_dsdna_complex(tmp_path / "step_000_ABC_model.pdb")
    fake_haddock3 = _write_fake_haddock3_gz(tmp_path / "fake_haddock3_gz.py")
    predocker = StructureProcessorRegistry.get("haddock3_nucleic")(
        haddock3_bin=str(fake_haddock3),
        target_chains=["B", "C"],
        anchor_chains=["B"],
        sampling=1,
        cache_enabled=False,
    )

    output = Path(predocker.run(str(input_path)))

    assert output.exists()
    assert _chain_ids(output) == ["A", "B", "C"]
    assert _metrics(output).contact_count > 0


def test_haddock3_nucleic_predocker_skips_existing_contact(tmp_path: Path) -> None:
    _init_global(tmp_path)
    input_path = _write_contact_candidate(tmp_path / "contact.pdb")
    fake_haddock3 = _write_fake_haddock3(tmp_path / "fake_haddock3.py")
    predocker = StructureProcessorRegistry.get("haddock3_nucleic")(
        haddock3_bin=str(fake_haddock3),
    )

    output = Path(predocker.run(str(input_path)))
    assert output != input_path
    assert output.exists()
    assert input_path.exists()
    assert output.read_text(encoding="utf-8") == input_path.read_text(encoding="utf-8")
    assert not list((tmp_path / "predocker").glob("haddock3_work_*"))


class _RunnerStub:
    na_shared_tokens = 0

    def design_sequences(self, **_kwargs):
        raise AssertionError("design_sequences should not be called")


class _Protomer(SimpleNamespace):
    mutable_mask = np.asarray([True], dtype=bool)
    tied_positions = []

    def aggregate_bool(self, mask):
        return mask


def test_nampnn_prepare_uses_scores_aware_predocker(monkeypatch, tmp_path: Path) -> None:
    _init_global(tmp_path)
    input_path = _write_far_dna_complex(tmp_path / "input.pdb")
    predocked_path = _write_contact_candidate(tmp_path / "predocked.pdb")
    original_scores = StandardScores(
        names=["AB"],
        plddt={"AB": [0.9]},
        complex_paths={"AB": input_path},
    )
    predocked_scores = original_scores.clone()
    predocked_scores.complex_paths = {"AB": predocked_path}
    calls = []

    def _fake_run_nested(predocker_names, evaluation_scores):
        calls.append((predocker_names, evaluation_scores))
        return predocked_scores

    monkeypatch.setattr(
        "astrolabe.mutators.na_mpnn.run_nested_predocker_on_scores",
        _fake_run_nested,
    )
    monkeypatch.setattr(
        "astrolabe.mutators.na_mpnn.prepare_exclude_sites",
        lambda *_args, **_kwargs: {},
    )
    mutator = NAMPNNMutator(
        sequence_model_runner=_RunnerStub(),
        predocker=["haddock"],
    )
    protomers = {
        "A": _Protomer(type=ProtomerTypes.PROTEIN, length=1),
        "B": _Protomer(
            type=ProtomerTypes.DNA,
            length=1,
            mutable_mask=np.asarray([False], dtype=bool),
        ),
    }

    prepared = mutator.prepare_mutation_input(protomers, original_scores, runtime_config={})

    assert calls == [(["haddock"], original_scores)]
    assert prepared["AB"]["structure_path"] == predocked_path
    assert prepared["AB"]["chains_to_design"] == "A"
    assert prepared["AB"]["fixed_residues"] == ["B1"]


def test_scores_helper_runs_haddock3_nucleic_predocker(tmp_path: Path) -> None:
    _init_global(tmp_path)
    input_path = _write_far_dna_complex(tmp_path / "input.pdb")
    fake_haddock3 = _write_fake_haddock3(tmp_path / "fake_haddock3.py")
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [0.9]},
        complex_paths={"AB": input_path},
    )
    global_config.init_const(
        "predockers",
        {
            "haddock": StructureProcessorRegistry.get("haddock3_nucleic")(
                haddock3_bin=str(fake_haddock3),
                cache_enabled=False,
            )
        },
    )

    updated = run_nested_predocker_on_scores(["haddock"], scores)

    assert Path(updated.complex_paths["AB"]) != input_path
    assert _metrics(Path(updated.complex_paths["AB"])).contact_count > 0
