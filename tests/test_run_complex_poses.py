from __future__ import annotations

import threading
from pathlib import Path
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import run
import astrolabe.losses.interaction as interaction_losses
from astrolabe.core import global_config
from astrolabe.evaluators.score_snapshot import build_standard_score_snapshot
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import LossEngine
from astrolabe.score_contributions import ScoreContribution


@pytest.fixture(autouse=True)
def _reset_global_config():
    global_config.reset()
    yield
    global_config.reset()


def _hotspot_loss_computer() -> LossEngine:
    loss_computer = LossEngine(
        {
            "loss_terms": {
                "hotspot": {
                    "_type": "hotspot_dsasa",
                    "_for": "AB",
                    "select_expression": "B_1:9",
                }
            }
        },
        validate_config=False,
    )
    loss_computer.setup(
        weighted_score_expression="plddt",
        early_stopping=None,
        acceptance_threshold=None,
        validate_config=False,
    )
    return loss_computer


def _bindcraft_loss_computer() -> LossEngine:
    loss_computer = LossEngine(
        {
            "loss_terms": {
                "interface_dg": {
                    "_type": "bindcraft_interface_dG",
                    "chain_a": "A",
                    "chain_b": "B",
                }
            }
        },
        validate_config=False,
    )
    loss_computer.setup(
        weighted_score_expression="interface_dg",
        early_stopping=None,
        acceptance_threshold=None,
        validate_config=False,
    )
    return loss_computer


def _bindcraft_binder_plddt_loss_computer() -> LossEngine:
    loss_computer = LossEngine(
        {
            "loss_terms": {
                "binder_plddt": {
                    "_type": "bindcraft_binder_plddt_norm",
                    "binder_chain": "B",
                }
            }
        },
        validate_config=False,
    )
    loss_computer.setup(
        weighted_score_expression="binder_plddt",
        early_stopping=None,
        acceptance_threshold=None,
        validate_config=False,
    )
    return loss_computer


def test_bindcraft_binder_plddt_does_not_request_complex_poses() -> None:
    assert not run._loss_computer_needs_complex_poses(
        _bindcraft_binder_plddt_loss_computer()
    )
    assert run._loss_computer_needs_complex_poses(_hotspot_loss_computer())


def test_pyrosetta_loss_promotes_enable_pyrosetta_config() -> None:
    global_config.reset()
    global_config.init_const("enable_pyrosetta", False)
    args = SimpleNamespace(enable_pyrosetta=False)

    assert run._sync_enable_pyrosetta_from_loss_computer(args, _hotspot_loss_computer())
    assert args.enable_pyrosetta is True
    assert global_config.enable_pyrosetta is True


def test_bindcraft_loss_promotes_pyrosetta_and_post_eval_hook() -> None:
    global_config.reset()
    global_config.init_const("enable_pyrosetta", False)
    args = SimpleNamespace(enable_pyrosetta=False)
    loss_computer = _bindcraft_loss_computer()

    assert run._sync_enable_pyrosetta_from_loss_computer(args, loss_computer)
    assert args.enable_pyrosetta is True
    assert run._post_eval_hook_config(args, loss_computer) == {
        "rosetta_minimize": {"enabled": True, "mode": "repack"}
    }


def test_base_config_allows_bindcraft_post_eval_hook_auto_detection() -> None:
    base_cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "config" / "base.yaml")
    args = SimpleNamespace(
        post_eval_hooks=OmegaConf.select(base_cfg, "post_eval_hooks", default=None)
    )

    assert run._post_eval_hook_config(args, _bindcraft_loss_computer()) == {
        "rosetta_minimize": {"enabled": True, "mode": "repack"}
    }


def test_explicit_disabled_post_eval_hook_remains_disabled() -> None:
    args = SimpleNamespace(
        post_eval_hooks={"rosetta_minimize": {"enabled": False, "mode": "repack"}}
    )

    assert run._post_eval_hook_config(args, _bindcraft_loss_computer()) is None


def test_evaluation_work_item_preserves_single_step_runtime_mapping(
    monkeypatch,
) -> None:
    global_config.init_const("stage_manager", SimpleNamespace(current_stage="design"))
    args = SimpleNamespace(
        seed=9,
        post_eval_hooks={"rosetta_minimize": {"enabled": False, "mode": "repack"}},
    )
    evaluator = SimpleNamespace(alias="EvalAlias")

    monkeypatch.setattr(
        run,
        "construct_standard_input",
        lambda oligos, seed: {oligo: f"input:{seed}:{oligo}" for oligo in oligos},
    )

    runtime_config = {"keep": "value"}
    work_items = run._build_evaluation_work_items(
        args,
        evaluator,
        runtime_config,
        4,
        ["AB", "AC"],
        loss_computer=None,
    )

    assert len(work_items) == 1
    item = work_items[0]
    assert item.item_id == "step:4:AB,AC"
    assert item.step == 4
    assert item.stage == "design"
    assert item.evaluator_alias == "EvalAlias"
    assert item.oligo_names == ["AB", "AC"]
    assert item.inputs == {"AB": "input:9:AB", "AC": "input:9:AC"}
    assert item.runtime_config is runtime_config
    assert runtime_config["step"] == 4
    assert runtime_config["keep"] == "value"
    assert item.detail == "step"


def test_evaluation_work_items_keep_incremental_runtime_mapping_isolated(
    monkeypatch,
) -> None:
    global_config.init_const("stage_manager", SimpleNamespace(current_stage="design"))
    args = SimpleNamespace(seed=9, post_eval_hooks=None)
    evaluator = SimpleNamespace(alias="EvalAlias")

    monkeypatch.setattr(
        run,
        "construct_standard_input",
        lambda oligos, seed: {oligo: f"input:{seed}:{oligo}" for oligo in oligos},
    )

    runtime_config = {"keep": "value"}
    work_items = run._build_evaluation_work_items(
        args,
        evaluator,
        runtime_config,
        5,
        ["AB", "AC"],
        loss_computer=_bindcraft_loss_computer(),
    )

    assert [item.item_id for item in work_items] == ["step:5:AB", "step:5:AC"]
    assert [item.oligo_names for item in work_items] == [["AB"], ["AC"]]
    assert [item.detail for item in work_items] == ["incremental", "incremental"]
    assert [item.inputs for item in work_items] == [
        {"AB": "input:9:AB"},
        {"AC": "input:9:AC"},
    ]
    assert all(item.stage == "design" for item in work_items)
    assert all(item.runtime_config is not runtime_config for item in work_items)
    assert [item.runtime_config["step"] for item in work_items] == [5, 5]
    assert runtime_config == {"keep": "value"}


def test_post_eval_hook_updates_complex_paths_and_clears_pose_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_config.init_const("out_dir", tmp_path)
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [1.0]},
        complex_paths={"AB": tmp_path / "ab.pdb"},
        complex_poses=run.ComplexPoses({"AB": object()}),
    )
    new_path = tmp_path / "minimized.pdb"

    def fake_hook(evaluation_result, hook_config, *, out_dir, step):
        evaluation_result.complex_paths = {"AB": new_path}
        evaluation_result.post_eval_hook_reports = {"AB": {"ran": True}}
        return evaluation_result

    monkeypatch.setattr(
        "astrolabe.post_eval_hooks.rosetta_minimize.apply_rosetta_minimize_hook",
        fake_hook,
    )
    monkeypatch.setattr(run, "_init_pyrosetta_runtime", lambda structure_files=None: None)

    out = run._apply_post_eval_hooks(
        SimpleNamespace(post_eval_hooks=None),
        scores,
        runtime_config={"step": 7},
        loss_computer=_bindcraft_loss_computer(),
    )

    assert out.complex_paths["AB"] == new_path
    assert out.complex_poses is None
    assert out.post_eval_hook_reports["AB"]["ran"] is True


def test_post_eval_hook_initializes_pyrosetta_runtime_before_hook(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_config.init_const("out_dir", tmp_path)
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [1.0]},
        complex_paths={"AB": tmp_path / "ab.cif"},
    )
    events = []

    def fake_init(structure_files=None):
        events.append(("init", dict(structure_files or {})))

    def fake_hook(evaluation_result, hook_config, *, out_dir, step):
        events.append(("hook", dict(evaluation_result.complex_paths or {})))
        return evaluation_result

    monkeypatch.setattr(run, "_init_pyrosetta_runtime", fake_init)
    monkeypatch.setattr(
        "astrolabe.post_eval_hooks.rosetta_minimize.apply_rosetta_minimize_hook",
        fake_hook,
    )

    run._apply_post_eval_hooks(
        SimpleNamespace(post_eval_hooks=None),
        scores,
        runtime_config={"step": 7},
        loss_computer=_bindcraft_loss_computer(),
    )

    assert events == [
        ("init", {"AB": tmp_path / "ab.cif"}),
        ("hook", {"AB": tmp_path / "ab.cif"}),
    ]


def test_pyrosetta_runtime_adds_generated_extra_residue_params(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_config.init_const("out_dir", tmp_path)
    init_options = []
    structure_files = {"AB": tmp_path / "ab.cif"}

    def fake_setup_extra_residues(structure_files=None):
        (tmp_path / "raw_model_results").mkdir()
        (tmp_path / "raw_model_results" / "IAI.params").write_text(
            "NAME IAI\n",
            encoding="utf-8",
        )
        assert structure_files == structure_files_arg

    structure_files_arg = structure_files
    fake_pyrosetta = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda opts: init_options.append(opts),
    )

    monkeypatch.setattr(run, "setup_extra_residues", fake_setup_extra_residues)
    monkeypatch.setitem(sys.modules, "pyrosetta", fake_pyrosetta)

    run._init_pyrosetta_runtime(structure_files=structure_files)

    assert len(init_options) == 1
    assert "-extra_res_fa" in init_options[0]
    assert str(tmp_path / "raw_model_results" / "IAI.params") in init_options[0]


def test_post_eval_hook_updates_soft_select_candidates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_config.init_const("out_dir", tmp_path)
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [1.0]},
        complex_paths={"AB": tmp_path / "ab.pdb"},
    )
    candidate = StandardScores(
        names=["AB"],
        plddt={"AB": [1.0]},
        complex_paths={"AB": tmp_path / "candidate.pdb"},
        complex_poses=run.ComplexPoses({"AB": object()}),
    )
    scores.candidate_scores = [candidate]

    def fake_hook(evaluation_result, hook_config, *, out_dir, step):
        name = evaluation_result.names[0]
        evaluation_result.complex_paths = {name: out_dir / f"{name}_{step}.pdb"}
        return evaluation_result

    monkeypatch.setattr(
        "astrolabe.post_eval_hooks.rosetta_minimize.apply_rosetta_minimize_hook",
        fake_hook,
    )
    monkeypatch.setattr(run, "_init_pyrosetta_runtime", lambda structure_files=None: None)

    out = run._apply_post_eval_hooks(
        SimpleNamespace(post_eval_hooks=None),
        scores,
        runtime_config={"step": 7},
        loss_computer=_bindcraft_loss_computer(),
    )

    assert out.complex_paths["AB"] == tmp_path / "structures/post_eval_hooks/AB_7.pdb"
    assert out.candidate_scores[0].complex_paths["AB"] == (
        tmp_path
        / "structures/post_eval_hooks/candidates/candidate_000/AB_7.pdb"
    )
    assert out.candidate_scores[0].complex_poses is None


def test_per_oligomer_hook_runs_after_each_incremental_evaluation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_config.init_const("out_dir", tmp_path)
    events = []

    class _Evaluator:
        def evaluate(self, inputs, runtime_config):
            name = next(iter(inputs))
            events.append(("eval", name))
            path = tmp_path / f"{name}.pdb"
            path.write_text("MODEL\nEND\n", encoding="utf-8")
            return StandardScores(
                names=[name],
                plddt={name: [1.0]},
                complex_paths={name: path},
            )

    def fake_standard_input(oligos, seed):
        return {oligo: object() for oligo in oligos}

    def fake_hook(evaluation_result, hook_config, *, out_dir, step):
        name = evaluation_result.names[0]
        events.append(("hook", name))
        evaluation_result.complex_paths[name] = tmp_path / f"{name}.min.pdb"
        return evaluation_result

    monkeypatch.setattr(run, "construct_standard_input", fake_standard_input)
    monkeypatch.setattr(run, "_ensure_evaluation_complex_poses", lambda args, result, loss_computer=None: result)
    monkeypatch.setattr(run, "_init_pyrosetta_runtime", lambda structure_files=None: None)
    monkeypatch.setattr(
        "astrolabe.post_eval_hooks.rosetta_minimize.apply_rosetta_minimize_hook",
        fake_hook,
    )

    out = run.perform_evaluation(
        SimpleNamespace(seed=1, post_eval_hooks=None, enable_pyrosetta=False, skip_complex_poses=True),
        _Evaluator(),
        {},
        3,
        ["AB", "AC"],
        loss_computer=_bindcraft_loss_computer(),
    )

    assert events == [("eval", "AB"), ("hook", "AB"), ("eval", "AC"), ("hook", "AC")]
    assert out.names == ["AB", "AC"]
    assert out.complex_paths["AB"].name == "AB.min.pdb"
    assert out.complex_paths["AC"].name == "AC.min.pdb"


def test_merge_standard_scores_preserves_score_contributions_from_all_oligomers(
    tmp_path: Path,
) -> None:
    ab_scores = StandardScores(
        names=["AB"],
        plddt={"AB": [1.0]},
        complex_paths={"AB": tmp_path / "ab.pdb"},
        score_contributions=[
            ScoreContribution(
                source="post_eval",
                metric_name="hook_score",
                value=0.7,
                scope="oligo",
                entity="AB",
            )
        ],
    )
    ac_scores = StandardScores(
        names=["AC"],
        plddt={"AC": [0.8]},
        complex_paths={"AC": tmp_path / "ac.pdb"},
        score_contributions=[
            ScoreContribution(
                source="post_eval",
                metric_name="hook_score",
                value=0.2,
                scope="oligo",
                entity="AC",
            )
        ],
    )

    merged = run._merge_standard_scores([ab_scores, ac_scores])
    snapshot = build_standard_score_snapshot(merged)

    assert snapshot["AB"]["hook_score"] == pytest.approx(0.7)
    assert snapshot["AC"]["hook_score"] == pytest.approx(0.2)


def test_per_oligomer_cpu_postprocess_can_overlap_next_evaluation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_config.init_const("out_dir", tmp_path)
    events = []
    hook_started = threading.Event()
    allow_hook_finish = threading.Event()

    class _Evaluator:
        alias = "EvalAlias"

        def evaluate(self, inputs, runtime_config):
            name = next(iter(inputs))
            if name == "AC":
                assert hook_started.wait(timeout=2.0)
                allow_hook_finish.set()
            events.append(("eval", name, int(runtime_config["step"])))
            path = tmp_path / f"{name}.pdb"
            path.write_text("MODEL\nEND\n", encoding="utf-8")
            return StandardScores(
                names=[name],
                plddt={name: [1.0]},
                complex_paths={name: path},
            )

    def fake_standard_input(oligos, seed):
        return {oligo: object() for oligo in oligos}

    def fake_hook(evaluation_result, hook_config, *, out_dir, step):
        name = evaluation_result.names[0]
        events.append(("hook_start", name, step))
        if name == "AB":
            hook_started.set()
            assert allow_hook_finish.wait(timeout=2.0)
        evaluation_result.complex_paths[name] = tmp_path / f"{name}.min.pdb"
        events.append(("hook_end", name, step))
        return evaluation_result

    monkeypatch.setattr(run, "construct_standard_input", fake_standard_input)
    monkeypatch.setattr(run, "_ensure_evaluation_complex_poses", lambda args, result, loss_computer=None: result)
    monkeypatch.setattr(
        "astrolabe.post_eval_hooks.rosetta_minimize.apply_rosetta_minimize_hook",
        fake_hook,
    )

    out = run.perform_evaluation(
        SimpleNamespace(
            seed=1,
            post_eval_hooks=None,
            enable_pyrosetta=False,
            skip_complex_poses=True,
            evaluator={
                "scheduling": {
                    "overlap_cpu_postprocess": True,
                    "cpu_postprocess_workers": 1,
                }
            },
        ),
        _Evaluator(),
        {},
        3,
        ["AB", "AC"],
        loss_computer=_bindcraft_loss_computer(),
    )

    assert events.index(("eval", "AC", 3)) < events.index(("hook_end", "AB", 3))
    assert out.names == ["AB", "AC"]
    assert out.complex_paths["AB"].name == "AB.min.pdb"
    assert out.complex_paths["AC"].name == "AC.min.pdb"


def test_pyrosetta_loss_requests_complex_poses_when_disabled_in_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    scores = StandardScores(
        names=["A"],
        plddt={"A": [1.0]},
        complex_paths={"A": tmp_path / "a.pdb"},
    )
    scores.complex_paths["A"].write_text("MODEL\nEND\n", encoding="utf-8")
    args = SimpleNamespace(enable_pyrosetta=False, skip_complex_poses=False)
    loss_computer = _hotspot_loss_computer()

    monkeypatch.setattr(run, "_init_pyrosetta_runtime", lambda structure_files=None: None)
    monkeypatch.setattr(
        run.ComplexPoses,
        "construct_complex_poses",
        lambda eval_result: run.ComplexPoses(
            poses={name: object() for name in eval_result.names}
        ),
    )
    monkeypatch.setattr(interaction_losses, "_require_pyrosetta", lambda: None)

    out = run._ensure_evaluation_complex_poses(args, scores, loss_computer)

    assert set(out.complex_poses.poses) == {"A"}


def test_per_residue_scores_request_complex_poses_without_pyrosetta_loss(
    monkeypatch,
    tmp_path: Path,
) -> None:
    scores = StandardScores(
        names=["A"],
        plddt={"A": [1.0]},
        complex_paths={"A": tmp_path / "a.pdb"},
    )
    args = SimpleNamespace(
        enable_pyrosetta=False,
        skip_complex_poses=False,
        per_residue_scores={"enabled": True, "terms": {"sap": {"_type": "sap"}}},
    )
    called = {"init": 0}

    def fake_init(structure_files=None):
        called["init"] += 1

    monkeypatch.setattr(run, "_init_pyrosetta_runtime", fake_init)
    monkeypatch.setattr(
        run.ComplexPoses,
        "construct_complex_poses",
        lambda eval_result: run.ComplexPoses(
            poses={name: object() for name in eval_result.names}
        ),
    )

    out = run._ensure_evaluation_complex_poses(args, scores, loss_computer=None)

    assert called["init"] == 1
    assert set(out.complex_poses.poses) == {"A"}


def test_existing_per_residue_score_terms_do_not_request_complex_poses(
    monkeypatch,
    tmp_path: Path,
) -> None:
    scores = StandardScores(
        names=["A"],
        plddt={"A": [1.0]},
        complex_paths={"A": tmp_path / "a.pdb"},
    )
    args = SimpleNamespace(
        enable_pyrosetta=False,
        skip_complex_poses=False,
        per_residue_scores={
            "enabled": True,
            "terms": {"plddt": {"_type": "plddt"}},
            "mutator_score_terms": {"plddt": 1.0},
        },
    )
    called = {"init": 0}

    def fake_init(structure_files=None):
        called["init"] += 1

    monkeypatch.setattr(run, "_init_pyrosetta_runtime", fake_init)

    out = run._ensure_evaluation_complex_poses(args, scores, loss_computer=None)

    assert out.complex_poses is None
    assert called["init"] == 0


def test_skip_complex_poses_remains_explicit_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    scores = StandardScores(
        names=["A"],
        plddt={"A": [1.0]},
        complex_paths={"A": tmp_path / "a.pdb"},
    )
    args = SimpleNamespace(enable_pyrosetta=True, skip_complex_poses=True)
    called = False

    def fake_init(structure_files=None):
        nonlocal called
        called = True

    monkeypatch.setattr(run, "_init_pyrosetta_runtime", fake_init)

    out = run._ensure_evaluation_complex_poses(args, scores, loss_computer=None)

    assert out.complex_poses is None
    assert called is False


def test_single_chain_complex_poses_allows_oligomer_specific_hotspot_loss(
    monkeypatch,
    tmp_path: Path,
) -> None:
    scores = StandardScores(
        names=["A"],
        plddt={"A": [0.8]},
        ptm={"A": [0.6]},
        complex_paths={"A": tmp_path / "a.pdb"},
    )
    scores.complex_paths["A"].write_text("MODEL\nEND\n", encoding="utf-8")
    args = SimpleNamespace(enable_pyrosetta=False, skip_complex_poses=False)
    loss_computer = _hotspot_loss_computer()
    loss_computer.setup(
        weighted_score_expression="plddt + ptm",
        early_stopping=None,
        acceptance_threshold=None,
        validate_config=False,
    )

    monkeypatch.setattr(run, "_init_pyrosetta_runtime", lambda structure_files=None: None)
    monkeypatch.setattr(
        run.ComplexPoses,
        "construct_complex_poses",
        lambda eval_result: run.ComplexPoses(
            poses={name: object() for name in eval_result.names}
        ),
    )
    monkeypatch.setattr(interaction_losses, "_require_pyrosetta", lambda: None)

    out = run._ensure_evaluation_complex_poses(args, scores, loss_computer)
    score, details = loss_computer.compute_loss(out, {})

    assert set(out.complex_poses.poses) == {"A"}
    assert score == 1.4
    assert "hotspot" in details
