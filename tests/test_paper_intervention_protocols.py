from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf, open_dict

from astrolabe.cli import _compose_hydra_config
from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.losses.base import LossEngine
from astrolabe.losses.interaction import ligand_protein_interface_ca90, protein_interface_ca_box
from astrolabe.losses.structure import single_chain_ca_rog
from astrolabe.structure_processors.all_alanine import AllAlanineStructureProcessor
from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue, DockingPenaltyError


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
PROTOCOLS = [
    "ligand_binder_af3", "ligand_binder_protenix", "two_state_af3", "two_state_boltz2",
    "two_state_af3_archived", "two_state_boltz2_archived",
    "pocket_iptm", "pocket_oriented", "pocket_rescue", "cid_iptm", "cid_oriented", "cid_rescue",
    "pocket_iptm_archived", "pocket_oriented_archived", "pocket_rescue_archived",
]


@pytest.mark.parametrize("name", PROTOCOLS)
def test_paper_intervention_config_contract(name, tmp_path):
    cfg = _compose_hydra_config(config_dir=CONFIG_DIR, config_name=f"paper/{name}", overrides=["env=env"])
    with open_dict(cfg):
        cfg.task_name = "contract"
        cfg.out_dir = str(tmp_path)
        cfg.seqs.A = "ACDEFGHIKLMNPQRSTVWY"
        cfg.seqs.B = "ATP"
        cfg.types.B = "c"
        if name.startswith("cid"):
            cfg.seqs.B = "ACDEFGHIKLMNPQRSTVWY"
            cfg.types.B = "protein"
            cfg.seqs.C = "ATP"
            cfg.types.C = "c"
        if name.startswith("pocket"):
            pocket = cfg.loss.loss_terms.target_ligand_in_target_pocket
            pocket.reference_structure_path = "reference.pdb"
            pocket.reference_structure_chain = "A"
            pocket.reference_pocket_positions = [1, 2, 3, 4]
        if name.startswith("two_state"):
            expected = ["-A", "AB"] if name.endswith("_archived") else ["AB"]
            assert list(cfg.stages.af3_refine.oligo) == expected
            assert OmegaConf.is_missing(cfg.paper_protocol, "historical_protenix_checkpoint")
            cfg.paper_protocol.historical_protenix_checkpoint = "protenix_base_default_v0.5.0.pt"
            assert cfg.evaluator.protenix.load_checkpoint_path == "protenix_base_default_v0.5.0.pt"
    global_config.reset()
    engine = LossEngine(OmegaConf.to_container(cfg.loss, resolve=True))
    for stage in cfg.stages.values():
        assert stage.evaluator in cfg.evaluator.evaluator_terms
        assert set(stage.mutators) <= set(cfg.mutator.mutator_terms)
        engine.setup(stage.weighted_score_expression, stage.early_stopping, stage.acceptance_threshold)


def _pdb(path, *, two_chains=True, transform=None):
    points = np.array([[-4, -2, -1], [-2, 1, 0], [0, 0, 2], [2, -1, 0], [5, 2, -1]], dtype=float)
    records = []
    for chain, shift in [("A", np.zeros(3))] + ([("B", np.array([0, 3, 1]))] if two_chains else []):
        for i, coord in enumerate(points + shift, 1):
            records.extend([
                ("ATOM", "CA", "TRP", chain, i, coord),
                ("ATOM", "CB", "TRP", chain, i, coord + [0, 0, 1]),
                ("ATOM", "CG", "TRP", chain, i, coord + [10, 0, 0]),
            ])
    records.extend([
        ("HETATM", "C1", "LIG", "C", 1, np.array([0, 1, 0])),
        ("HETATM", "C2", "LIG", "C", 1, np.array([100, 100, 100])),
        ("HETATM", "CA", "LIG", "C", 1, np.array([200, 200, 200])),
    ])
    lines = []
    for serial, (kind, atom, residue, chain, index, coordinate) in enumerate(records, 1):
        if transform is not None:
            rotation, translation = transform
            coordinate = coordinate @ rotation + translation
        x, y, z = coordinate
        lines.append(f"{kind:<6}{serial:5d} {atom:^4s} {residue:3s} {chain}{index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 80.00           C")
    path.write_text("\n".join(lines) + "\nEND\n")
    return points


@pytest.mark.parametrize("name", ["two_state_af3", "two_state_boltz2"])
def test_twostate_methods_use_apo_gate_and_holo_only_refinement(name):
    from math import exp

    cfg = _compose_hydra_config(config_dir=CONFIG_DIR, config_name=f"paper/{name}", overrides=["env=env"])
    engine = LossEngine(OmegaConf.to_container(cfg.loss, resolve=True))
    rg_ideal = 2.2 * 130 ** 0.38
    descriptors = {
        "A_length": 130,
        "A": SimpleNamespace(binder_rog=1.6 * rg_ideal),
        "AB": SimpleNamespace(plddt=0.85, ptm=0.8, iptm=0.75,
                              binder_rog=0.9 * rg_ideal, ligand_burial=0.7),
        "apo_bound_tmscore": 0.6,
        "binder_x_fraction": 0.05,
    }
    stage = cfg.stages.default
    requested = [stage.weighted_score_expression, cfg.stages.af3_refine._entrance]
    dependencies = engine._ordered_expression_terms_for_expressions(requested)
    values = engine._evaluate_expression_terms(descriptors, engine._basic_operations(), dependencies)
    context = {**descriptors, **values}
    sigmoid = lambda x: 1 / (1 + exp(-x))
    holo_score = (0.85 + 0.8 + 0.75) / 3 + 0.35 * 0.7 - 0.45 * sigmoid((0.9 - 1.1) / 0.08)
    expected = holo_score - 0.0875 * sigmoid((1.6 - 1) / 0.06) - 0.5 * 0.05
    assert engine.weighted_score(context, stage.weighted_score_expression) == pytest.approx(expected)
    # Mean apo/holo compactness is 1.25, but the specified apo-only gate must fail.
    assert not engine.eval_condition(context, cfg.stages.af3_refine._entrance)

    # Holo-only refinement must evaluate without any apo descriptors or TM-score.
    holo_only = {"A_length": 130, "AB": descriptors["AB"]}
    expression = cfg.stages.af3_refine.weighted_score_expression
    dependencies = engine._ordered_expression_terms_for_expressions([expression])
    values = engine._evaluate_expression_terms(holo_only, engine._basic_operations(), dependencies)
    assert engine.weighted_score({**holo_only, **values}, expression) == pytest.approx(holo_score)
    assert "apo_rog_ratio" not in values and "state_divergence" not in values
    assert cfg.evaluator.protenix.model.N_cycle == 4
    assert cfg.evaluator.protenix.sample_diffusion.N_step == 200
    assert cfg.evaluator.protenix.sample_diffusion.N_sample == 5
    if name == "two_state_boltz2":
        assert cfg.evaluator.boltz.recycling_steps == 3
        assert cfg.evaluator.boltz.sampling_steps == 200
        assert cfg.evaluator.boltz.diffusion_samples == 1


def _init(tmp_path):
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)
    global_config.init_const("protomers", {
        "A": SimpleNamespace(type=ProtomerTypes.PROTEIN),
        "B": SimpleNamespace(type=ProtomerTypes.PROTEIN),
        "C": SimpleNamespace(type=ProtomerTypes.SMILES),
    })


def _coordinates(path):
    from Bio.PDB import PDBParser
    return np.array([a.coord for a in PDBParser(QUIET=True).get_structure("s", path).get_atoms()])


def test_ca90_and_docking_share_box_and_coordinate_frame(tmp_path):
    _init(tmp_path)
    source = tmp_path / "complex.pdb"
    points = _pdb(source)
    result = SimpleNamespace(names=["ABC"], complex_paths={"ABC": str(source)})
    assert ligand_protein_interface_ca90(result, interface_chains=["A", "B"], ligand_targets="C")["ABC"] == pytest.approx(1 / 3)

    angle = 0.71
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    moved = tmp_path / "moved.pdb"
    _pdb(moved, transform=(rotation, np.array([12, -8, 3])))
    result.complex_paths["ABC"] = str(moved)
    assert ligand_protein_interface_ca90(result, interface_chains=["A", "B"], ligand_targets="C")["ABC"] == pytest.approx(1 / 3)

    processor = AutoDockVinaLigandPoseRescue(obabel_path="unused", fail_on_error=True)
    protein, ligand = processor._separate_protein_ligand_with_prody(str(source))
    original_protein, original_ligand = _coordinates(protein), _coordinates(ligand)
    site = processor._calculate_binding_site_from_protein_interface(protein, coverage_percent=90, padding=1)
    box = protein_interface_ca_box(points, points + [0, 3, 1])
    assert [site[f"center_{axis}"] for axis in "xyz"] == pytest.approx(box["center"])
    assert [site[f"size_{axis}"] for axis in "xyz"] == pytest.approx(box["size"])
    assert _coordinates(protein) == pytest.approx(original_protein)
    transformed_protein, transformed_ligand = processor._coordinate_transformed_docking_inputs(protein, ligand)
    for path, original in [(transformed_protein, original_protein), (transformed_ligand, original_ligand)]:
        restored = tmp_path / (Path(path).stem + "_restored.pdb")
        processor._apply_coordinate_transform_to_pdb(path, str(restored), processor._last_binding_site_transform, inverse=True)
        assert _coordinates(restored) == pytest.approx(original, abs=0.002)


def test_single_chain_ca_rog_ignores_sidechains_and_pose_backend(tmp_path):
    source = tmp_path / "complex.pdb"
    points = _pdb(source)
    expected = np.sqrt(np.mean(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))
    result = SimpleNamespace(names=["ABC"], complex_paths={"ABC": str(source)}, complex_poses=object())
    assert single_chain_ca_rog(result, "A")["ABC"] == pytest.approx(expected)


def test_allalanine_can_mutate_only_chain_a(tmp_path):
    from Bio.PDB import PDBParser
    _init(tmp_path)
    source = tmp_path / "complex.pdb"
    _pdb(source)
    output = AllAlanineStructureProcessor(included_chains=["A"]).run(str(source))
    model = PDBParser(QUIET=True).get_structure("s", output)[0]
    assert {res.resname for res in model["A"]} == {"ALA"}
    assert {atom.name for atom in model["A"].get_atoms()} == {"CA", "CB"}
    assert {res.resname for res in model["B"]} == {"TRP"}
    assert {atom.name for atom in model["B"].get_atoms()} == {"CA", "CB", "CG"}
    assert len(list(model["C"].get_atoms())) == 3


def test_vina_first_rank_survives_cache(tmp_path):
    _init(tmp_path)
    first, second = tmp_path / "first.pdb", tmp_path / "second.pdb"
    _pdb(first)
    _pdb(second)
    processor = AutoDockVinaLigandPoseRescue(obabel_path="unused", rank_vina_order=True, cache_first_pose_only=True)
    result = processor._filter_ranking({0: (str(first), "C"), 1: (str(second), "C")})
    processor._after_cache_store("key", str(first), result)
    assert {processor.take_cache("key") for _ in range(20)} == {str(first)}


def test_vina_unavailable_interface_fails_without_replacement(tmp_path):
    _init(tmp_path)
    source = tmp_path / "one_chain.pdb"
    _pdb(source, two_chains=False)
    strict = AutoDockVinaLigandPoseRescue(obabel_path="unused", fail_on_error=True)
    with pytest.raises(DockingPenaltyError, match="at least two chains"):
        strict._run_complete_vina_workflow(str(source))
    assert not list((tmp_path / "predocker").glob("*penalty*"))
    legacy = AutoDockVinaLigandPoseRescue(obabel_path="unused")
    replacement = legacy._run_complete_vina_workflow(str(source))
    assert Path(replacement).is_file()


def test_pocket_ordinal_mapping_does_not_realign_sequence():
    reference, current = "A" * 12 + "W" * 12, "W" * 12 + "A" * 12
    mapping, blocks, mismatches = AutoDockVinaLigandPoseRescue._align_reference_sequence_to_current(
        reference, current, allow_mismatches=True, ordinal=True,
    )
    assert mapping == {i: i for i in range(24)}
    assert blocks == [(0, 24)]
    assert len(mismatches) == 24
    with pytest.raises(DockingPenaltyError, match="lengths do not match"):
        AutoDockVinaLigandPoseRescue._align_reference_sequence_to_current(
            reference, current[:-1], allow_mismatches=True, ordinal=True,
        )


def test_packaged_pocket_inputs_resolve_original_reference_and_template(tmp_path, monkeypatch):
    import hashlib
    import json
    import tarfile
    from Bio.PDB import PDBParser

    repo = CONFIG_DIR.parent
    resources = repo / "resources"
    manifest = json.loads((resources / "paper_inputs_interventions.json").read_text())
    with tarfile.open(resources / manifest["archive"]) as archive:
        for item in manifest["files"]:
            data = archive.extractfile(item["path"]).read()
            assert len(data) == item["size_bytes"]
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
            target = tmp_path / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    monkeypatch.setenv("ASTROLABE_PAPER_DATA", str(tmp_path))
    for name in ["pocket_catm", "pocket_liga", "pocket_dsba", "pocket_ecftsz"]:
        cfg = _compose_hydra_config(config_dir=CONFIG_DIR, config_name="paper/pocket_rescue", overrides=["env=env"])
        with open_dict(cfg):
            cfg = OmegaConf.merge(cfg, OmegaConf.load(repo / f"examples/paper/inputs/interventions/{name}.yaml"))
        pocket = cfg.loss.loss_terms.target_ligand_in_target_pocket
        reference = PDBParser(QUIET=True).get_structure("reference", pocket.reference_structure_path)[0]["A"]
        positions = {res.id[1] for res in reference}
        assert set(pocket.reference_pocket_positions) <= positions
        homology = json.loads(Path(str(cfg.homologies.A).removesuffix(":A")).read_text())
        protein = homology["sequences"][0]["protein"]
        assert protein["templates"][0]["mmcif"]
        assert len(protein["sequence"]) == len([res for res in reference if res.id[0] == " "])
        assert cfg.predocker.predocker_terms.vina.reference_structure_path == pocket.reference_structure_path
