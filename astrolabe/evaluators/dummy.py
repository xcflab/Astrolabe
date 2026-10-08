from __future__ import annotations

import pathlib
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import StandardScores


class DummyEvaluator(Evaluator):
    """A lightweight evaluator for smoke tests.

    This does not run any structure prediction. It produces deterministic,
    minimal `StandardScores` so the pipeline can exercise mutator/stage logic.
    """

    def __init__(
        self,
        plddt: float = 0.9,
        *,
        write_structure: bool = False,
        structure_dir: str | pathlib.Path | None = None,
    ):
        self._plddt = float(plddt)
        self._write_structure = bool(write_structure)
        self._structure_dir = (
            pathlib.Path(structure_dir) if structure_dir is not None else None
        )

    @staticmethod
    def _ensure_pyrosetta_initialized():
        import os
        from contextlib import redirect_stderr, redirect_stdout

        import pyrosetta  # type: ignore

        if hasattr(pyrosetta, "is_initialized") and pyrosetta.is_initialized():
            return pyrosetta

        opts = "-ignore_zero_occupancy false -mute all -beta"
        dalpha = (
            pathlib.Path(__file__).resolve().parents[1]
            / "assets"
            / "DAlphaBall.gcc"
        )
        if dalpha.exists():
            opts = f"{opts} -holes:dalphaball {dalpha}"
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                pyrosetta.init(opts)
        return pyrosetta

    def _write_input_structure(
        self,
        name: str,
        input_obj: Any,
        runtime_config: Dict[str, Any],
    ) -> pathlib.Path:
        structure_dir = self._structure_dir
        if structure_dir is None:
            raw_dir = (
                runtime_config.get("raw_model_results_dir")
                if isinstance(runtime_config, dict)
                else None
            )
            structure_dir = (
                pathlib.Path(raw_dir)
                / "dummy"
                if raw_dir
                else pathlib.Path("dummy_structures")
            )
        structure_dir.mkdir(parents=True, exist_ok=True)

        protein_chains = list(getattr(input_obj, "protein_chains", []) or [])
        if not protein_chains:
            raise ValueError("DummyEvaluator(write_structure=True) requires protein chains.")

        pyrosetta = self._ensure_pyrosetta_initialized()
        pose = None
        for chain in protein_chains:
            chain_pose = pyrosetta.pose_from_sequence(str(chain.sequence))
            if pose is None:
                pose = chain_pose
            else:
                pyrosetta.rosetta.core.pose.append_pose_to_pose(pose, chain_pose, True)
        out_path = structure_dir / f"{name}_dummy.pdb"
        pose.dump_pdb(str(out_path))
        return out_path

    @staticmethod
    def _protomers_from_global_or_inputs(inputs: Dict[str, Any]) -> dict[str, Any]:
        try:
            protomers = getattr(global_config, "protomers", {}) or {}
        except RuntimeError:
            protomers = {}
        if protomers:
            return protomers

        inferred: dict[str, Any] = {}
        for input_obj in (inputs or {}).values():
            for chain in getattr(input_obj, "protein_chains", []) or []:
                inferred[str(chain.id)] = SimpleNamespace(
                    length=len(str(chain.sequence)),
                    sequence=str(chain.sequence),
                    type=ProtomerTypes.PROTEIN,
                )
        return inferred

    def prepare_input(
        self, inputs: Dict[str, Any], runtime_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        return inputs

    def run_function(self, input_feat: Dict[str, Any]) -> Dict[str, Any]:
        return input_feat

    def prepare_output(
        self,
        output: Dict[str, Any],
        inputs: Dict[str, Any],
        runtime_config: Dict[str, Any],
    ) -> StandardScores:
        names = list(inputs.keys())
        protomers = self._protomers_from_global_or_inputs(inputs)
        per_proto_scores = {}
        for proto_name, protomer in protomers.items():
            try:
                length = int(protomer.length)
            except Exception:
                length = len(getattr(protomer, "sequence", "")) or 1
            per_proto_scores[proto_name] = np.full((length,), self._plddt, dtype=float)

        plddt = {name: [self._plddt] for name in names}
        complex_paths = {}
        for name in names:
            if self._write_structure:
                complex_paths[name] = self._write_input_structure(
                    name,
                    inputs[name],
                    runtime_config or {},
                )
            else:
                complex_paths[name] = pathlib.Path("")
        return StandardScores(
            names=names,
            plddt=plddt,
            complex_paths=complex_paths,
            per_proto_scores=per_proto_scores,
            ptm={name: [0.0] for name in names},
            iptm={name: [0.0] for name in names},
        )
