from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from omegaconf import OmegaConf, open_dict

from astrolabe.experimental.candidate_batch_policy import (
    CandidateCollection,
    CandidateExecutionError,
    CandidatePoolConfig,
    CandidatePoolScheduler,
    CandidateResult,
    CandidateSelector,
    CandidateTask,
    select_highest_score_candidate,
)


@dataclass(frozen=True)
class CandidateBatchRun:
    collections: tuple[CandidateCollection, ...]
    decision_collection_count: int

    @property
    def decision_collections(self) -> tuple[CandidateCollection, ...]:
        return self.collections[: self.decision_collection_count]

    @property
    def drain_collections(self) -> tuple[CandidateCollection, ...]:
        return self.collections[self.decision_collection_count :]


class CandidateBatchConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ProcessCandidateRunnerConfig:
    config_yaml: Path
    out_root: Path
    runner: Path
    num_candidates: int = 4
    min_completed: int = 2
    max_wait_s: float = 30.0
    max_in_flight: int | None = None
    decision_rounds: int = 1
    candidate_steps: int = 2
    candidate_timeout_s: float | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class ProcessCandidatePayload:
    config_yaml: str
    out_dir: str
    task_name: str
    stdout_log: str
    stderr_log: str
    sequences: dict[str, str] = field(default_factory=dict)

    def as_summary(self) -> dict[str, object]:
        return {
            "config_yaml": self.config_yaml,
            "out_dir": self.out_dir,
            "task_name": self.task_name,
            "stdout_log": self.stdout_log,
            "stderr_log": self.stderr_log,
            "sequences": dict(self.sequences),
        }


@dataclass(frozen=True)
class ProcessCandidateScore:
    weighted_score: float
    sequences: dict[str, str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_base_cfg(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"candidate batch config not found: {path}")
    try:
        return OmegaConf.load(path)
    except Exception as exc:
        raise CandidateBatchConfigError(
            f"failed to load candidate batch config {path}: {exc}"
        ) from exc


def _write_candidate_cfg(
    *,
    base_cfg: Any,
    path: Path,
    task_name: str,
    out_dir: Path,
    steps: int,
    seed: int,
    sequences: dict[str, str],
) -> None:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    with open_dict(cfg):
        cfg.task_name = task_name
        cfg.out_dir = str(out_dir)
        cfg.steps = int(steps)
        cfg.seed = int(seed)
        if sequences:
            if "seqs" not in cfg or cfg.seqs is None:
                cfg.seqs = {}
            for chain, sequence in sequences.items():
                cfg.seqs[str(chain)] = str(sequence)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(cfg, resolve=False), encoding="utf-8")


def _find_scores_csv(out_dir: Path, task_name: str) -> Path:
    task_root = out_dir / task_name
    candidates = sorted(task_root.rglob("scores.csv"))
    if not candidates:
        raise FileNotFoundError(f"No scores.csv found under {task_root}")
    return candidates[-1]


def _best_score_row(scores_csv: Path) -> dict[str, str]:
    with scores_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in scores.csv: {scores_csv}")

    def _score(row: dict[str, str]) -> float:
        value = row.get("weighted_score")
        if value in (None, ""):
            return float("-inf")
        return float(value)

    accepted = [
        row
        for row in rows
        if str(row.get("accepted", "")).strip().lower() in {"true", "1", "yes"}
    ]
    return max(accepted or rows, key=_score)


def _extract_sequences(row: dict[str, str]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    for key, value in row.items():
        if not key.endswith("_sequence") or not value:
            continue
        chain_key = key[: -len("_sequence")]
        chain = chain_key[1:] if chain_key[:1] in {"p", "r", "d", "s", "c"} else chain_key
        if chain:
            sequences[chain] = value
    return sequences


class CandidateDecisionRun:
    """Own the candidate generation state for a multi-round decision run."""

    def __init__(
        self,
        *,
        pool_config: CandidatePoolConfig,
        worker: Callable[[CandidateTask], CandidateResult | float],
        decision_rounds: int,
        initial_sequences: dict[str, str] | None = None,
        selector: CandidateSelector | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.pool_config = pool_config
        self.worker = worker
        self.num_candidates = int(pool_config.num_candidates)
        self.decision_rounds = int(decision_rounds)
        self.current_sequences = dict(initial_sequences or {})
        self.next_candidate = 1
        self.selector = selector
        self.logger = logger

    def run(self) -> CandidateBatchRun:
        collections: list[CandidateCollection] = []
        with CandidatePoolScheduler(
            self.pool_config,
            self.worker,
            selector=self.selector,
            logger=self.logger,
        ) as scheduler:
            scheduler.submit_generation(1, self._tasks_for_generation(1))
            for generation in range(1, self.decision_rounds + 1):
                collection = scheduler.collect_until_ready(target_generation=generation)
                if collection is None:
                    break
                collections.append(collection)
                self._adopt_selected_sequences(collection, generation=generation)
                if generation < self.decision_rounds:
                    scheduler.submit_generation(
                        generation + 1,
                        self._tasks_for_generation(generation + 1),
                    )
            decision_collection_count = len(collections)
            collections.extend(scheduler.drain())
        return CandidateBatchRun(
            collections=tuple(collections),
            decision_collection_count=decision_collection_count,
        )

    def _tasks_for_generation(self, generation: int) -> list[CandidateTask]:
        tasks: list[CandidateTask] = []
        for _ in range(self.num_candidates):
            tasks.append(
                CandidateTask(
                    candidate_id=f"cand_{self.next_candidate:04d}",
                    generation=int(generation),
                    payload={"sequences": dict(self.current_sequences)},
                )
            )
            self.next_candidate += 1
        return tasks

    def _adopt_selected_sequences(
        self,
        collection: CandidateCollection,
        *,
        generation: int,
    ) -> None:
        if collection.selected is None:
            raise CandidateExecutionError(
                "No successful candidate was available for decision "
                f"round {generation}; failed={len(collection.failed)}"
            )
        payload = collection.selected.payload
        if isinstance(payload, ProcessCandidatePayload):
            self.current_sequences = dict(payload.sequences)


class ProcessCandidateOutputReader:
    def __init__(self, out_root: Path) -> None:
        self.out_root = Path(out_root)

    def read_best_score(self, task_name: str) -> ProcessCandidateScore:
        row = _best_score_row(_find_scores_csv(self.out_root, task_name))
        return ProcessCandidateScore(
            weighted_score=float(row["weighted_score"]),
            sequences=_extract_sequences(row),
        )


class ProcessCandidateWorker:
    def __init__(self, config: ProcessCandidateRunnerConfig, base_cfg: Any) -> None:
        self.config = config
        self.base_cfg = base_cfg
        self.output_reader = ProcessCandidateOutputReader(config.out_root)

    def __call__(self, task: CandidateTask) -> CandidateResult:
        payload = dict(task.payload or {})
        sequences = dict(payload.get("sequences") or {})
        task_name = f"{self.config.config_yaml.stem}__g{task.generation:03d}__{task.candidate_id}"
        work_root = self.config.out_root / "candidate_pool" / task.candidate_id
        cfg_path = work_root / "config.yaml"
        stdout_log = work_root / "stdout.log"
        stderr_log = work_root / "stderr.log"
        seed_base = int(getattr(self.base_cfg, "seed", 0) or 0)
        seed = seed_base + _candidate_seed_offset(task.candidate_id)
        _write_candidate_cfg(
            base_cfg=self.base_cfg,
            path=cfg_path,
            task_name=task_name,
            out_dir=self.config.out_root,
            steps=self.config.candidate_steps,
            seed=seed,
            sequences=sequences,
        )
        payload_info = ProcessCandidatePayload(
            config_yaml=str(cfg_path),
            out_dir=str(self.config.out_root),
            task_name=task_name,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            sequences=sequences,
        )
        if self.config.dry_run:
            return CandidateResult(
                task=task,
                status="completed",
                score=0.0,
                payload=payload_info,
            )

        work_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(self.config.runner),
            "run",
            "--standalone",
            str(cfg_path),
            "--no-chdir",
        ]
        try:
            with stdout_log.open("w", encoding="utf-8") as stdout, stderr_log.open(
                "w", encoding="utf-8"
            ) as stderr:
                proc = subprocess.run(
                    cmd,
                    cwd=str(_repo_root()),
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    check=False,
                    timeout=(
                        None
                        if self.config.candidate_timeout_s is None
                        else float(self.config.candidate_timeout_s)
                    ),
                )
            if proc.returncode != 0:
                return CandidateResult(
                    task=task,
                    status="failed",
                    error=(
                        f"candidate command failed rc={proc.returncode}; "
                        f"stderr={stderr_log}"
                    ),
                    payload=payload_info,
                )
            score = self.output_reader.read_best_score(task_name)
        except Exception as exc:
            return CandidateResult(
                task=task,
                status="failed",
                error=str(exc) or exc.__class__.__name__,
                payload=payload_info,
            )
        return CandidateResult(
            task=task,
            status="completed",
            score=score.weighted_score,
            payload=ProcessCandidatePayload(
                config_yaml=payload_info.config_yaml,
                out_dir=payload_info.out_dir,
                task_name=payload_info.task_name,
                stdout_log=payload_info.stdout_log,
                stderr_log=payload_info.stderr_log,
                sequences=score.sequences,
            ),
        )


ProcessCandidateWorkerFactory = Callable[
    [ProcessCandidateRunnerConfig, Any],
    Callable[[CandidateTask], CandidateResult | float],
]


@dataclass(frozen=True)
class ProcessCandidateBatchAdapters:
    worker_factory: ProcessCandidateWorkerFactory = ProcessCandidateWorker
    selector: CandidateSelector = select_highest_score_candidate


def _candidate_seed_offset(candidate_id: str) -> int:
    suffix = str(candidate_id).rsplit("_", 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    digest = hashlib.blake2b(str(candidate_id).encode("utf-8"), digest_size=4)
    return int.from_bytes(digest.digest(), "big")


def run_process_candidate_batch(
    config: ProcessCandidateRunnerConfig,
    *,
    adapters: ProcessCandidateBatchAdapters | None = None,
) -> dict[str, Any]:
    active_adapters = adapters or ProcessCandidateBatchAdapters()
    base_cfg = _load_base_cfg(config.config_yaml)
    pool_cfg = CandidatePoolConfig(
        num_candidates=config.num_candidates,
        min_completed=config.min_completed,
        max_wait_s=config.max_wait_s,
        max_in_flight=config.max_in_flight,
    )
    worker = active_adapters.worker_factory(config, base_cfg)
    run = CandidateDecisionRun(
        pool_config=pool_cfg,
        worker=worker,
        decision_rounds=config.decision_rounds,
        selector=active_adapters.selector,
    ).run()

    return {
        "config_yaml": str(config.config_yaml),
        "out_root": str(config.out_root),
        "runner": str(config.runner),
        "dry_run": bool(config.dry_run),
        "num_candidates": int(config.num_candidates),
        "min_completed": int(config.min_completed),
        "max_wait_s": float(config.max_wait_s),
        "max_in_flight": int(config.max_in_flight or config.num_candidates),
        "candidate_timeout_s": (
            None
            if config.candidate_timeout_s is None
            else float(config.candidate_timeout_s)
        ),
        "decision_rounds": int(config.decision_rounds),
        "collections": [collection.as_row() for collection in run.collections],
        "selected_candidate_ids": [
            collection.selected.candidate_id
            for collection in run.decision_collections
            if collection.selected is not None
        ],
        "drain_selected_candidate_ids": [
            collection.selected.candidate_id
            for collection in run.drain_collections
            if collection.selected is not None
        ],
        "failed_candidates": [
            failed
            for collection in run.collections
            for failed in collection.as_row()["failed_candidates"]
        ],
    }


def add_candidate_batch_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config-yaml", required=True, help="Base standalone GH YAML.")
    parser.add_argument("--out-root", required=True, help="Output root for candidate runs.")
    parser.add_argument(
        "--runner",
        default=str(_repo_root() / "scripts" / "gh_dev"),
        help="GH CLI runner executable (default: scripts/astrolabe).",
    )
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--min-completed", type=int, default=2)
    parser.add_argument("--max-wait-s", type=float, default=30.0)
    parser.add_argument("--max-in-flight", type=int, default=None)
    parser.add_argument("--decision-rounds", type=int, default=1)
    parser.add_argument("--candidate-steps", type=int, default=2)
    parser.add_argument(
        "--candidate-timeout-s",
        type=float,
        default=None,
        help=(
            "Optional per-candidate subprocess hard timeout. "
            "Unset by default so slow candidates can carry forward past collection windows."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gh_candidate_batch_run",
        description="Experimental candidate-pool runner for process-isolated GH candidates.",
    )
    return add_candidate_batch_arguments(parser)


def _config_from_namespace(args: argparse.Namespace) -> ProcessCandidateRunnerConfig:
    return ProcessCandidateRunnerConfig(
        config_yaml=Path(args.config_yaml).expanduser().resolve(),
        out_root=Path(args.out_root).expanduser().resolve(),
        runner=Path(args.runner).expanduser().resolve(),
        num_candidates=int(args.num_candidates),
        min_completed=int(args.min_completed),
        max_wait_s=float(args.max_wait_s),
        max_in_flight=args.max_in_flight,
        decision_rounds=int(args.decision_rounds),
        candidate_steps=int(args.candidate_steps),
        candidate_timeout_s=args.candidate_timeout_s,
        dry_run=bool(args.dry_run),
    )


def _print_summary(summary: dict[str, Any], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(f"config_yaml: {summary['config_yaml']}")
    print(f"out_root: {summary['out_root']}")
    print(f"dry_run: {str(summary['dry_run']).lower()}")
    print(
        "pool: "
        f"num_candidates={summary['num_candidates']} "
        f"min_completed={summary['min_completed']} "
        f"max_wait_s={summary['max_wait_s']} "
        f"max_in_flight={summary['max_in_flight']} "
        f"candidate_timeout_s={summary['candidate_timeout_s']}"
    )
    for row in summary["collections"]:
        print(
            "collection: "
            f"index={row['collection_index']} trigger={row['trigger']} "
            f"selected={row['selected_candidate_id']} "
            f"completed={row['completed']} failed={row['failed']} "
            f"carried_forward={row['carried_forward']}"
        )
        for failed in row.get("failed_candidates", []):
            print(
                "failed_candidate: "
                f"collection={row['collection_index']} "
                f"candidate={failed['candidate_id']} "
                f"error={failed['error']}"
            )


def run_from_namespace(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> int:
    try:
        summary = run_process_candidate_batch(_config_from_namespace(args))
    except (CandidateExecutionError, CandidateBatchConfigError, FileNotFoundError) as exc:
        if parser is not None:
            parser.exit(1, f"error: {exc}\n")
        raise
    _print_summary(summary, str(args.format))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return run_from_namespace(args, parser=parser)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
