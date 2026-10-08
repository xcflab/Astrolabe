from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.core.cyclization import (
    derive_cyclic_relative_patches,
    parse_cyclization,
    resolve_cyclization_bonds,
    resolve_cyclization_bond_specs,
)
from astrolabe.losses.cyclic_closure import (
    cyclic_all_bonds_closed,
    cyclic_all_bonds_within_error,
    cyclic_closure_norm,
    cyclic_max_bond_error,
)


@pytest.fixture(autouse=True)
def reset_global_config():
    global_config.reset()
    yield
    global_config.reset()


def _pdb(path: Path, distance: float) -> Path:
    path.write_text(
        "ATOM      1    N ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
        "ATOM      2   CA ALA A   2       0.000   4.000   0.000  1.00 20.00           C\n"
        "ATOM      3   CA ALA A   3       0.000   9.000   0.000  1.00 20.00           C\n"
        f"ATOM      4    C ALA A   4    {distance:8.3f}   0.000   0.000  1.00 20.00           C\n"
        "TER\nEND\n"
    )
    return path


def _context(path: Path | None, *, atom: str = "C", extra_bond: bool = False):
    bonds = [{
        "a": {"proto": "A", "pos": 1, "atom": "N"},
        "b": {"proto": "A", "pos": 4, "atom": atom},
        "order": "single",
        "target_distance": 2.0,
    }]
    if extra_bond:
        bonds.append({
            "a": {"proto": "A", "pos": 2, "atom": "CA"},
            "b": {"proto": "A", "pos": 3, "atom": "CA"},
            "order": "single",
            "target_distance": 1.54,
        })
    global_config.init_const("protomers", {"A": SimpleNamespace(sequence="AAAA")})
    global_config.init_const("cyclization", {
        "bonds": bonds,
        "encoding": {"scope": "head_to_tail_only" if extra_bond else "all_bonds"},
    })
    return SimpleNamespace(names=["A"], complex_paths={} if path is None else {"A": path})


@pytest.mark.parametrize("distance,closed", [(1.5, 0), (1.7, 1), (2.0, 1), (2.3, 1), (2.5, 0)])
def test_publication_endpoint_checks_both_sides_of_target(tmp_path, distance, closed):
    result = _context(_pdb(tmp_path / "geometry.pdb", distance))
    assert cyclic_max_bond_error(result)["A"] == pytest.approx(abs(distance - 2.0))
    assert cyclic_all_bonds_within_error(result)["A"] == closed
    if distance == 1.5:
        # The historical one-sided metric keeps its existing contract.
        assert cyclic_all_bonds_closed(result)["A"] == 1


def test_publication_endpoint_includes_bonds_omitted_from_encoding(tmp_path):
    result = _context(_pdb(tmp_path / "encoding.pdb", 2.0), extra_bond=True)
    assert cyclic_closure_norm(result)["A"] == pytest.approx(1)
    assert cyclic_max_bond_error(result)["A"] == pytest.approx(3.46)
    assert cyclic_all_bonds_within_error(result)["A"] == 0


@pytest.mark.parametrize("missing_structure", [False, True])
def test_missing_endpoint_or_structure_fails_publication_closure(tmp_path, missing_structure):
    result = _context(
        None if missing_structure else _pdb(tmp_path / "missing.pdb", 2.0),
        atom="OXT",
    )
    assert math.isinf(cyclic_max_bond_error(result)["A"])
    assert cyclic_all_bonds_within_error(result)["A"] == 0


def test_paper_inputs_resolve_their_bond_topology():
    root = Path(__file__).resolve().parents[1] / "examples/paper/inputs/cyclic"
    paths = sorted(root.glob("*/*.yaml"))
    assert paths
    for path in paths:
        cfg = yaml.safe_load(path.read_text())
        recipe = "cyclic_peptide_bound" if path.parent.name == "bound" else "cyclic_peptide_multiring"
        composed = OmegaConf.merge(
            OmegaConf.load(root.parents[3] / f"config/paper/{recipe}.yaml"),
            cfg,
        )
        # Representation starters are 20-mers; bound examples use 20 or 9-20.
        length = int(cfg["seqs"]["A"]) if path.parent.name == "multiring" else 20
        spec = parse_cyclization(composed.cyclization)
        bonds = resolve_cyclization_bond_specs(
            spec, protomer_lengths={"A": length}, require_order_for_generic=True
        )
        assert bonds, path
        # Also exercise the simultaneous ring-offset matrix conflict contract.
        derive_cyclic_relative_patches(
            resolve_cyclization_bonds(spec, protomer_lengths={"A": length}),
            oligo_name="A", protomer_lengths={"A": length}
        )
        if cfg.get("paper_protocol", {}).get("topology") in {
            "tail_sidechain_lactone", "tail_sidechain_thiolactone"
        }:
            assert "[CTM]" in cfg["seqs"]["A"], path
            assert "CTM" in cfg["nonstandard_residues"]["aliases"], path


def test_bound_input_overlays_cannot_replace_formal_ligandmpnn_protocol():
    from astrolabe.cli import _compose_hydra_config

    root = Path(__file__).resolve().parents[1]
    formal = _compose_hydra_config(
        config_dir=root / "config", config_name="paper/cyclic_peptide_bound", overrides=["env=env"]
    )
    archived = _compose_hydra_config(
        config_dir=root / "config", config_name="paper/cyclic_peptide_bound_archived", overrides=["env=env"]
    )
    OmegaConf.set_struct(formal, False)
    OmegaConf.set_struct(archived, False)
    for path in (root / "examples/paper/inputs/cyclic/bound").glob("*.yaml"):
        overlay = OmegaConf.load(path)
        cfg = OmegaConf.merge(formal, overlay)
        assert cfg.steps == 30
        assert cfg.stages.default.evaluator == "Alphafold3"
        assert list(cfg.stages.default.mutators) == ["mpnn"], path
        term = cfg.mutator.mutator_terms.mpnn
        assert term._type == "LigandMPNN"
        assert term.get("model_type", cfg.mutator.LigandMPNN.model_type) == "ligand_mpnn"
        assert cfg.stages.default.weighted_score_expression == "iptm + 0.3 * cyclic_closure_norm"
        assert cfg.loss.loss_terms.cyclic_closure_norm.aggregate == "min"
        assert cfg.loss.loss_terms.cyclic_closure_norm.sigma == 0.2
        assert cfg.stages.default.acceptance_threshold == "cyclic_closure_norm >= 0.9"
        if overlay.paper_protocol.get("historical_protocol"):
            old = OmegaConf.merge(archived, overlay)
            assert list(old.stages.default.mutators) == ["LigandMPNN_itf", "LigandMPNN_noitf"]
            assert old.mutator.mutator_terms.LigandMPNN_noitf.model_type == "soluble_mpnn"


def test_multiring_and_archived_recipes_keep_distinct_proposals_and_shared_formula():
    from astrolabe.cli import _compose_hydra_config

    root = Path(__file__).resolve().parents[1]
    for name, mutator in [("cyclic_peptide_multiring", "mpnn"), ("cyclic_peptide_multiring_archived", "MCMC")]:
        cfg = _compose_hydra_config(config_dir=root / "config", config_name=f"paper/{name}", overrides=["env=env"])
        assert cfg.steps == 20
        assert list(cfg.stages.default.mutators) == [mutator]
        assert cfg.stages.default.weighted_score_expression == "plddt + 0.4 * ptm + 0.3 * cyclic_closure_norm"
        assert cfg.loss.loss_terms.cyclic_closure_norm.aggregate == "mean"
        assert cfg.loss.loss_terms.cyclic_closure_norm.sigma == 0.2
        assert cfg.stages.default.acceptance_threshold is None
