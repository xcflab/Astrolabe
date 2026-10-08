from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _write_fake_rapidock_repo(
    repo_root: Path, *, error_once: bool = False, num_candidates: int = 2
) -> Path:
    package_dir = repo_root / "RAPiDock"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "API.py").write_text("class RAPiDockInference:\n    pass\n", encoding="utf-8")
    daemon_path = package_dir / "daemon.py"
    daemon_path.write_text(
        """
import json
import sys
from pathlib import Path


def _write(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()


def main():
    state_path = Path(__file__).resolve().parent / "daemon_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"dock_calls": 0, "error_once_handled": False}
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        msg = json.loads(line)
        mtype = msg.get("type")
        if mtype == "ping":
            _write({"type": "pong", "id": msg.get("id")})
            continue
        if mtype == "shutdown":
            _write({"type": "ok"})
            return 0
        if mtype != "dock":
            _write({"type": "error", "id": msg.get("id"), "message": f"unknown type: {mtype}"})
            continue

        if """ + ("True" if error_once else "False") + """ and not state.get("error_once_handled", False):
            state["error_once_handled"] = True
            state_path.write_text(json.dumps(state), encoding="utf-8")
            _write({"type": "error", "id": msg.get("id"), "error_type": "RuntimeError", "message": "transient failure"})
            continue

        state["dock_calls"] += 1
        state_path.write_text(json.dumps(state), encoding="utf-8")
        generation = state["dock_calls"]
        task = (msg.get("tasks") or [])[0]
        output_dir = Path(task["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        candidates = []
        for rank in range(1, """ + str(num_candidates) + """ + 1):
            pdb_path = output_dir / f"candidate_{generation}_{rank}.pdb"
            pdb_path.write_text(f"MODEL {generation}-{rank}\\nEND\\n", encoding="utf-8")
            candidates.append({"rank": rank, "peptide_pdb": str(pdb_path)})
        _write({"type": "result", "id": msg.get("id"), "results": [{"name": task["name"], "candidates": candidates}]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return daemon_path


def _write_pdb(path: Path, atom_name: str) -> Path:
    path.write_text(
        "\n".join(
            [
                f"ATOM      1  {atom_name:<4}ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
                "TER",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_rapidock_runner_ping_and_shutdown(tmp_path: Path) -> None:
    from astrolabe.structure_processors.wrappers.rapidock_runner import RAPiDockRunner

    repo_root = tmp_path / "rapidock_repo"
    daemon_path = _write_fake_rapidock_repo(repo_root)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "rapidock_global.pt").write_text("stub", encoding="utf-8")

    runner = RAPiDockRunner(
        {
            "python_bin": sys.executable,
            "repo_path": str(repo_root),
            "daemon_script_path": str(daemon_path),
            "model_dir": str(model_dir),
            "ckpt": "rapidock_global.pt",
            "response_cache_dir": str(tmp_path / "cache"),
            "work_dir": str(tmp_path / "work"),
        }
    )

    assert runner.ping() == "pong"
    runner.shutdown()
    assert runner._proc is None


def test_rapidock_runner_default_wrapper_uses_inference_script(tmp_path: Path) -> None:
    from astrolabe.structure_processors.wrappers.rapidock_runner import RAPiDockRunner

    repo_root = tmp_path / "rapidock_repo_default_wrapper"
    (repo_root / "RAPiDock").mkdir(parents=True)
    (repo_root / "RAPiDock" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "inference.py").write_text(
        """
import argparse
from pathlib import Path


def main():
    print("NOISY-LIB-STDOUT")
    parser = argparse.ArgumentParser()
    parser.add_argument("--complex_name", required=True)
    parser.add_argument("--protein_description", required=True)
    parser.add_argument("--peptide_description", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--batch_size")
    parser.add_argument("--inference_steps")
    parser.add_argument("--cpu")
    parser.add_argument("--conformation_type")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(1, args.N + 1):
        (output_dir / f"rank{rank}_confidence.pdb").write_text("MODEL 1\\nEND\\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""".strip()
        + "\n",
        encoding="utf-8",
    )
    model_dir = tmp_path / "model_default_wrapper"
    model_dir.mkdir()
    (model_dir / "rapidock_global.pt").write_text("stub", encoding="utf-8")
    protein_path = _write_pdb(tmp_path / "protein_default_wrapper.pdb", "CA")
    peptide_path = _write_pdb(tmp_path / "peptide_default_wrapper.pdb", "CB")

    runner = RAPiDockRunner(
        {
            "python_bin": sys.executable,
            "repo_path": str(repo_root),
            "model_dir": str(model_dir),
            "ckpt": "rapidock_global.pt",
            "response_cache_dir": str(tmp_path / "cache_default_wrapper"),
            "work_dir": str(tmp_path / "work_default_wrapper"),
            "num_samples_bank": 3,
            "device": "cuda:0",
        }
    )

    candidates = runner.dock_candidates(
        name="default_wrapper",
        protein_pdb=str(protein_path),
        peptide_pdb=str(peptide_path),
    )

    assert [Path(path).name for path in candidates] == [
        "rank1_confidence.pdb",
        "rank2_confidence.pdb",
        "rank3_confidence.pdb",
    ]
    assert runner.ping() == "pong"
    runner.shutdown()


def test_rapidock_default_wrapper_collects_only_final_rank_candidates(
    tmp_path: Path,
) -> None:
    from astrolabe.structure_processors.wrappers.rapidock_daemon import _collect_candidates

    output_dir = tmp_path / "rapidock_outputs"
    output_dir.mkdir()
    for name in [
        "rank1.pdb",
        "rank1_reverseprocess.pdb",
        "rank1_ref2015.pdb",
        "rank1_confidence.pdb",
        "rank2_confidence.pdb",
        "rank2_reverseprocess.pdb",
    ]:
        (output_dir / name).write_text("MODEL 1\nEND\n", encoding="utf-8")

    candidates = _collect_candidates(output_dir)

    assert [Path(path).name for path in candidates] == [
        "rank1_confidence.pdb",
        "rank2_confidence.pdb",
    ]


def test_rapidock_default_wrapper_falls_back_to_exact_rank_pdbs(
    tmp_path: Path,
) -> None:
    from astrolabe.structure_processors.wrappers.rapidock_daemon import _collect_candidates

    output_dir = tmp_path / "rapidock_outputs"
    output_dir.mkdir()
    for name in ["rank10.pdb", "rank2.pdb", "rank2_reverseprocess.pdb"]:
        (output_dir / name).write_text("MODEL 1\nEND\n", encoding="utf-8")

    candidates = _collect_candidates(output_dir)

    assert [Path(path).name for path in candidates] == ["rank2.pdb", "rank10.pdb"]


def test_rapidock_runner_uses_content_cache_and_refreshes_after_bank_exhaustion(
    tmp_path: Path,
) -> None:
    from astrolabe.structure_processors.wrappers.rapidock_runner import RAPiDockRunner

    repo_root = tmp_path / "rapidock_repo"
    daemon_path = _write_fake_rapidock_repo(repo_root)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "rapidock_global.pt").write_text("stub", encoding="utf-8")

    protein_a = _write_pdb(tmp_path / "protein_a.pdb", "CA")
    peptide_a = _write_pdb(tmp_path / "peptide_a.pdb", "CB")
    protein_b = _write_pdb(tmp_path / "protein_b_same_content.pdb", "CA")
    peptide_b = _write_pdb(tmp_path / "peptide_b_same_content.pdb", "CB")

    runner = RAPiDockRunner(
        {
            "python_bin": sys.executable,
            "repo_path": str(repo_root),
            "daemon_script_path": str(daemon_path),
            "model_dir": str(model_dir),
            "ckpt": "rapidock_global.pt",
            "response_cache_dir": str(tmp_path / "cache"),
            "work_dir": str(tmp_path / "work"),
            "num_samples_bank": 2,
        }
    )

    first = runner.dock_candidate(
        name="case",
        protein_pdb=str(protein_a),
        peptide_pdb=str(peptide_a),
    )
    second = runner.dock_candidate(
        name="case",
        protein_pdb=str(protein_b),
        peptide_pdb=str(peptide_b),
    )
    third = runner.dock_candidate(
        name="case",
        protein_pdb=str(protein_a),
        peptide_pdb=str(peptide_a),
    )

    assert Path(first).name == "candidate_1_1.pdb"
    assert Path(second).name == "candidate_1_2.pdb"
    assert Path(third).name == "candidate_2_1.pdb"

    state = (repo_root / "RAPiDock" / "daemon_state.json").read_text(encoding="utf-8")
    assert '"dock_calls": 2' in state

    runner.shutdown()


def test_rapidock_runner_retries_once_after_daemon_error(tmp_path: Path) -> None:
    from astrolabe.structure_processors.wrappers.rapidock_runner import RAPiDockRunner

    repo_root = tmp_path / "rapidock_repo"
    daemon_path = _write_fake_rapidock_repo(repo_root, error_once=True)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "rapidock_global.pt").write_text("stub", encoding="utf-8")

    protein_path = _write_pdb(tmp_path / "protein_retry.pdb", "CA")
    peptide_path = _write_pdb(tmp_path / "peptide_retry.pdb", "CB")

    runner = RAPiDockRunner(
        {
            "python_bin": sys.executable,
            "repo_path": str(repo_root),
            "daemon_script_path": str(daemon_path),
            "model_dir": str(model_dir),
            "ckpt": "rapidock_global.pt",
            "response_cache_dir": str(tmp_path / "cache"),
            "work_dir": str(tmp_path / "work"),
            "num_samples_bank": 2,
        }
    )

    candidate = runner.dock_candidate(
        name="retry_case",
        protein_pdb=str(protein_path),
        peptide_pdb=str(peptide_path),
    )

    assert Path(candidate).name == "candidate_1_1.pdb"
    state = (repo_root / "RAPiDock" / "daemon_state.json").read_text(encoding="utf-8")
    assert '"error_once_handled": true' in state
    assert '"dock_calls": 1' in state
    runner.shutdown()


def test_rapidock_runner_skips_candidates_removed_after_cache_store(tmp_path: Path) -> None:
    from astrolabe.structure_processors.wrappers.rapidock_runner import RAPiDockRunner

    repo_root = tmp_path / "rapidock_repo_evicted"
    daemon_path = _write_fake_rapidock_repo(repo_root)
    model_dir = tmp_path / "model_evicted"
    model_dir.mkdir()
    (model_dir / "rapidock_global.pt").write_text("stub", encoding="utf-8")

    protein_path = _write_pdb(tmp_path / "protein_evicted.pdb", "CA")
    peptide_path = _write_pdb(tmp_path / "peptide_evicted.pdb", "CB")

    runner = RAPiDockRunner(
        {
            "python_bin": sys.executable,
            "repo_path": str(repo_root),
            "daemon_script_path": str(daemon_path),
            "model_dir": str(model_dir),
            "ckpt": "rapidock_global.pt",
            "response_cache_dir": str(tmp_path / "cache_evicted"),
            "work_dir": str(tmp_path / "work_evicted"),
            "num_samples_bank": 2,
        }
    )

    original_store = runner._store_cached_candidates

    def _store_then_evict(key, candidates):
        original_store(key, candidates)
        Path(candidates[0]).unlink()

    runner._store_cached_candidates = _store_then_evict  # type: ignore[method-assign]

    candidate = runner.dock_candidate(
        name="evicted_case",
        protein_pdb=str(protein_path),
        peptide_pdb=str(peptide_path),
    )

    assert Path(candidate).name == "candidate_1_2.pdb"
    runner.shutdown()


def test_rapidock_runner_regenerates_bank_when_cached_candidates_are_all_gone(
    tmp_path: Path,
) -> None:
    from astrolabe.structure_processors.wrappers.rapidock_runner import RAPiDockRunner

    repo_root = tmp_path / "rapidock_repo_all_evicted"
    daemon_path = _write_fake_rapidock_repo(repo_root)
    model_dir = tmp_path / "model_all_evicted"
    model_dir.mkdir()
    (model_dir / "rapidock_global.pt").write_text("stub", encoding="utf-8")

    protein_path = _write_pdb(tmp_path / "protein_all_evicted.pdb", "CA")
    peptide_path = _write_pdb(tmp_path / "peptide_all_evicted.pdb", "CB")

    runner = RAPiDockRunner(
        {
            "python_bin": sys.executable,
            "repo_path": str(repo_root),
            "daemon_script_path": str(daemon_path),
            "model_dir": str(model_dir),
            "ckpt": "rapidock_global.pt",
            "response_cache_dir": str(tmp_path / "cache_all_evicted"),
            "work_dir": str(tmp_path / "work_all_evicted"),
            "num_samples_bank": 2,
        }
    )

    first = runner.dock_candidate(
        name="all_evicted_case",
        protein_pdb=str(protein_path),
        peptide_pdb=str(peptide_path),
    )
    for candidate_path in runner._load_cached_candidates(
        runner._cache_key(protein_pdb=str(protein_path), peptide_pdb=str(peptide_path))
    ):
        Path(candidate_path).unlink()

    second = runner.dock_candidate(
        name="all_evicted_case",
        protein_pdb=str(protein_path),
        peptide_pdb=str(peptide_path),
    )

    assert Path(first).name == "candidate_1_1.pdb"
    assert Path(second).name == "candidate_2_1.pdb"
    state = (repo_root / "RAPiDock" / "daemon_state.json").read_text(encoding="utf-8")
    assert '"dock_calls": 2' in state
    runner.shutdown()


def test_rapidock_runner_consumes_large_bank_in_numeric_order(tmp_path: Path) -> None:
    from astrolabe.structure_processors.wrappers.rapidock_runner import RAPiDockRunner

    repo_root = tmp_path / "rapidock_repo"
    daemon_path = _write_fake_rapidock_repo(repo_root, num_candidates=10)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "rapidock_global.pt").write_text("stub", encoding="utf-8")

    protein_path = _write_pdb(tmp_path / "protein_large_bank.pdb", "CA")
    peptide_path = _write_pdb(tmp_path / "peptide_large_bank.pdb", "CB")

    runner = RAPiDockRunner(
        {
            "python_bin": sys.executable,
            "repo_path": str(repo_root),
            "daemon_script_path": str(daemon_path),
            "model_dir": str(model_dir),
            "ckpt": "rapidock_global.pt",
            "response_cache_dir": str(tmp_path / "cache"),
            "work_dir": str(tmp_path / "work"),
            "num_samples_bank": 10,
        }
    )

    observed = [
        Path(
            runner.dock_candidate(
                name="large_bank",
                protein_pdb=str(protein_path),
                peptide_pdb=str(peptide_path),
            )
        ).name
        for _ in range(10)
    ]

    assert observed == [f"candidate_1_{rank}.pdb" for rank in range(1, 11)]
    runner.shutdown()


def test_rapidock_runner_cache_key_changes_with_confidence_checkpoint(
    tmp_path: Path,
) -> None:
    from astrolabe.structure_processors.wrappers.rapidock_runner import RAPiDockRunner

    repo_root = tmp_path / "rapidock_repo"
    daemon_path = _write_fake_rapidock_repo(repo_root)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "rapidock_global.pt").write_text("stub", encoding="utf-8")

    protein_path = _write_pdb(tmp_path / "protein_confidence.pdb", "CA")
    peptide_path = _write_pdb(tmp_path / "peptide_confidence.pdb", "CB")

    common = {
        "python_bin": sys.executable,
        "repo_path": str(repo_root),
        "daemon_script_path": str(daemon_path),
        "model_dir": str(model_dir),
        "ckpt": "rapidock_global.pt",
        "response_cache_dir": str(tmp_path / "cache"),
        "work_dir": str(tmp_path / "work"),
        "num_samples_bank": 2,
        "confidence_model_dir": str(tmp_path / "confidence"),
    }

    runner_a = RAPiDockRunner({**common, "confidence_ckpt": "confidence_a.pt"})
    first = runner_a.dock_candidate(
        name="confidence_case",
        protein_pdb=str(protein_path),
        peptide_pdb=str(peptide_path),
    )
    runner_a.shutdown()

    runner_b = RAPiDockRunner({**common, "confidence_ckpt": "confidence_b.pt"})
    second = runner_b.dock_candidate(
        name="confidence_case",
        protein_pdb=str(protein_path),
        peptide_pdb=str(peptide_path),
    )
    runner_b.shutdown()

    assert Path(first).name == "candidate_1_1.pdb"
    assert Path(second).name == "candidate_2_1.pdb"

    state = (repo_root / "RAPiDock" / "daemon_state.json").read_text(encoding="utf-8")
    assert '"dock_calls": 2' in state


def test_rapidock_runner_cache_key_changes_with_repo_path(tmp_path: Path) -> None:
    from astrolabe.structure_processors.wrappers.rapidock_runner import RAPiDockRunner

    repo_a = tmp_path / "rapidock_repo_a"
    repo_b = tmp_path / "rapidock_repo_b"
    daemon_a = _write_fake_rapidock_repo(repo_a)
    daemon_b = _write_fake_rapidock_repo(repo_b)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "rapidock_global.pt").write_text("stub", encoding="utf-8")

    protein_path = _write_pdb(tmp_path / "protein_repo_cache.pdb", "CA")
    peptide_path = _write_pdb(tmp_path / "peptide_repo_cache.pdb", "CB")
    common = {
        "python_bin": sys.executable,
        "model_dir": str(model_dir),
        "ckpt": "rapidock_global.pt",
        "response_cache_dir": str(tmp_path / "cache"),
        "work_dir": str(tmp_path / "work"),
        "num_samples_bank": 2,
    }

    runner_a = RAPiDockRunner(
        {**common, "repo_path": str(repo_a), "daemon_script_path": str(daemon_a)}
    )
    first = runner_a.dock_candidate(
        name="repo_cache_case",
        protein_pdb=str(protein_path),
        peptide_pdb=str(peptide_path),
    )
    runner_a.shutdown()

    runner_b = RAPiDockRunner(
        {**common, "repo_path": str(repo_b), "daemon_script_path": str(daemon_b)}
    )
    second = runner_b.dock_candidate(
        name="repo_cache_case",
        protein_pdb=str(protein_path),
        peptide_pdb=str(peptide_path),
    )
    runner_b.shutdown()

    assert Path(first).name == "candidate_1_1.pdb"
    assert Path(second).name == "candidate_1_1.pdb"
    state_a = (repo_a / "RAPiDock" / "daemon_state.json").read_text(encoding="utf-8")
    state_b = (repo_b / "RAPiDock" / "daemon_state.json").read_text(encoding="utf-8")
    assert '"dock_calls": 1' in state_a
    assert '"dock_calls": 1' in state_b


def test_rapidock_predocker_scores_full_bank_and_keeps_original_peptide(
    tmp_path: Path,
) -> None:
    from astrolabe.structure_processors.peptide_docker import RAPiDockPeptideDockingProcessor

    protein_path = _write_pdb(tmp_path / "protein.pdb", "CA")
    original_peptide_path = _write_pdb(tmp_path / "original_peptide.pdb", "CB")
    candidate_paths = [
        _write_pdb(tmp_path / "candidate_1.pdb", "CG"),
        _write_pdb(tmp_path / "candidate_2.pdb", "CD"),
    ]

    class _Runner:
        def __init__(self):
            self.calls = []

        def dock_candidates(self, *, name: str, protein_pdb: str, peptide_pdb: str) -> list[str]:
            self.calls.append((name, protein_pdb, peptide_pdb))
            return [str(path) for path in candidate_paths]

    predocker = RAPiDockPeptideDockingProcessor.__new__(RAPiDockPeptideDockingProcessor)
    predocker.runner = _Runner()
    predocker.hotspot = "B_1:7,12"
    predocker.debug_dir = tmp_path / "predocker_runner"
    predocker.debug_dir.mkdir(parents=True, exist_ok=True)
    predocker._split_complex_chains = lambda _input: (
        str(protein_path),
        str(original_peptide_path),
        0,
    )

    captured = {}

    def _fake_evaluate(peptide_files, protein_file, peptide_chain_index, oligo_name):
        captured["peptide_files"] = [str(p) for p in peptide_files]
        captured["protein_file"] = protein_file
        captured["peptide_chain_index"] = peptide_chain_index
        captured["oligo_name"] = oligo_name
        return str(peptide_files[1])

    predocker._evaluate_docking_results = _fake_evaluate

    result = predocker._run_impl("step_000_AB_model.pdb")

    assert result == captured["peptide_files"][1]
    assert predocker.runner.calls == [
        ("step_000_AB_model", str(protein_path), str(original_peptide_path))
    ]
    assert Path(captured["peptide_files"][0]).parent == predocker.debug_dir
    assert Path(captured["peptide_files"][0]).name == "step_000_AB_model__candidate_1.pdb"
    assert Path(captured["peptide_files"][1]).parent == predocker.debug_dir
    assert Path(captured["peptide_files"][1]).name == "step_000_AB_model__candidate_2.pdb"
    assert captured["peptide_files"][2] == str(original_peptide_path)
    assert captured["protein_file"] == str(protein_path)
    assert captured["peptide_chain_index"] == 0
    assert captured["oligo_name"] == "AB"
