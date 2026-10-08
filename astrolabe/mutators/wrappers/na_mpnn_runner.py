from __future__ import annotations

import logging
import os
import pathlib
import re
import subprocess
import sys
from typing import Dict, List, Sequence

import numpy as np


logger = logging.getLogger(__name__)


class NAMPNNRunner:
    """Wrapper around baker-laboratory/NA-MPNN inference script."""

    def __init__(self, args, *, output_root: pathlib.Path) -> None:
        self.args = args
        self.repo_path = self._resolve_repo_path(self._get("repo_path"))
        self.python = str(self._get("python_bin", sys.executable))
        self.checkpoint = self._resolve_checkpoint(self._get("checkpoint"))
        self.temperature = float(self._get("temperature", 0.1))
        self.batch_size = int(self._get("batch_size", 1))
        self.number_of_batches = int(self._get("number_of_batches", 1))
        self.design_na_only = int(self._get("design_na_only", 0))
        self.parse_na_only = int(self._get("parse_na_only", 0))
        self.na_shared_tokens = int(self._get("na_shared_tokens", 0))
        self.output_pdbs = int(self._get("output_pdbs", 0))
        self.job_name = self._get("job_name")
        self.extra_args = dict(self._get("extra_args", {}) or {})
        self.output_root = pathlib.Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

    def design_sequences(
        self,
        *,
        structure_path: pathlib.Path,
        job_name: str,
        chains_to_design: str | Sequence[str] | None = None,
        fixed_residues: Sequence[str] | None = None,
        parse_these_chains_only: str | Sequence[str] | None = None,
    ) -> Dict[str, object]:
        structure_path = pathlib.Path(structure_path).expanduser().resolve()
        if not structure_path.exists():
            raise FileNotFoundError(f"NA-MPNN input structure not found: {structure_path}")

        job_dir = self.output_root / self._safe_name(job_name)
        job_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python,
            str(self.repo_path / "inference" / "run.py"),
            "--mode",
            "design",
            "--model_type",
            "na_mpnn",
            "--pdb_path",
            str(structure_path),
            "--out_folder",
            str(job_dir),
            "--checkpoint_na_mpnn",
            str(self.checkpoint),
            "--temperature",
            str(self.temperature),
            "--batch_size",
            str(self.batch_size),
            "--number_of_batches",
            str(self.number_of_batches),
            "--design_na_only",
            str(self.design_na_only),
            "--parse_na_only",
            str(self.parse_na_only),
            "--na_shared_tokens",
            str(self.na_shared_tokens),
            "--output_pdbs",
            str(self.output_pdbs),
            # Always output sequence + specificity when running as a mutator.
            "--output_sequences",
            "1",
            "--output_specificity",
            "1",
        ]

        if chains_to_design:
            cmd.extend(["--chains_to_design", self._normalize_chain_list(chains_to_design)])
        if fixed_residues:
            cmd.extend(["--fixed_residues", self._normalize_residue_list(fixed_residues)])
        if parse_these_chains_only:
            cmd.extend(
                [
                    "--parse_these_chains_only",
                    self._normalize_parse_chain_spec(parse_these_chains_only),
                ]
            )

        for key, value in self.extra_args.items():
            if value is None:
                continue
            key = str(key).strip().replace("_", "-")
            if not key:
                continue
            cmd.extend([f"--{key}", str(value)])

        logger.info("Running NA-MPNN design for %s", structure_path.name)
        subprocess.run(
            cmd,
            check=True,
            cwd=str(self.repo_path),
            env=self._patched_env(),
        )

        expected_fasta = job_dir / "seqs" / f"{self._structure_basename(structure_path)}.fa"
        fasta_files = [expected_fasta]
        if not expected_fasta.exists():
            raise RuntimeError(f"NA-MPNN did not produce expected FASTA output: {expected_fasta}.")

        best_entry, best_fasta_path = self._select_best_entry(fasta_files)
        best_entry["job_dir"] = job_dir
        best_entry["fasta_path"] = best_fasta_path
        best_entry["probabilities"] = self._load_specificity_matrix(job_dir, structure_path)
        return best_entry

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _get(self, key, default=None):
        if hasattr(self.args, key):
            value = getattr(self.args, key)
            if value is None:
                return default
            return value
        if isinstance(self.args, dict):
            value = self.args.get(key, default)
            if value is None:
                return default
            return value
        return default

    def _resolve_repo_path(self, path_value) -> pathlib.Path:
        if path_value is None:
            raise ValueError("NAMPNN `repo_path` must be provided in the configuration.")
        path = pathlib.Path(path_value).expanduser()
        if not path.is_absolute():
            path = (pathlib.Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"NAMPNN repository path '{path}' not found. "
                "Initialize vendor submodules (recommended) or point `na_mpnn.repo_path` to a valid checkout."
            )
        return path

    def _resolve_checkpoint(self, checkpoint_value):
        if checkpoint_value is None:
            checkpoint_value = self.repo_path / "models" / "design_model" / "s_19137.pt"
        checkpoint = pathlib.Path(checkpoint_value).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (self.repo_path / checkpoint).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"NAMPNN checkpoint file '{checkpoint}' does not exist.")
        return checkpoint

    def _patched_env(self):
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        repo_str = str(self.repo_path)
        env["PYTHONPATH"] = f"{repo_str}:{pythonpath}" if pythonpath else repo_str
        return env

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()) or "job"

    def _select_best_entry(
        self, fasta_files: List[pathlib.Path]
    ) -> tuple[Dict[str, object], pathlib.Path]:
        best = None
        best_score = float("-inf")
        best_path = None
        for fasta_path in fasta_files:
            for entry in self._parse_fasta(fasta_path):
                confidence = entry["metadata"].get("overall_confidence")
                if confidence is None:
                    continue
                try:
                    score = float(confidence)
                except (TypeError, ValueError):
                    continue
                if score > best_score:
                    best = entry
                    best_score = score
                    best_path = fasta_path
        if best is None or best_path is None:
            raise RuntimeError(
                "No designed sequences with `overall_confidence` found in NA-MPNN FASTA outputs."
            )
        return best, best_path

    def _parse_fasta(self, fasta_path: pathlib.Path) -> List[Dict[str, object]]:
        entries: List[Dict[str, object]] = []
        header = None
        seq_lines: List[str] = []
        with open(fasta_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None:
                        entries.append(
                            {
                                "header": header,
                                "sequence": "".join(seq_lines),
                                "metadata": self._parse_header(header),
                            }
                        )
                    header = line[1:].strip()
                    seq_lines = []
                else:
                    seq_lines.append(line)
        if header is not None:
            entries.append(
                {
                    "header": header,
                    "sequence": "".join(seq_lines),
                    "metadata": self._parse_header(header),
                }
            )
        return entries

    @staticmethod
    def _parse_header(header: str) -> Dict[str, object]:
        metadata: Dict[str, object] = {}
        parts = [part.strip() for part in header.split(",")]
        for part in parts[1:]:
            tokens = part.replace(",", " ").split()
            for token in tokens:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                key = key.strip().lower()
                value = value.strip().strip(",")
                try:
                    metadata[key] = float(value)
                except ValueError:
                    metadata[key] = value
        return metadata

    def _load_specificity_matrix(
        self, job_dir: pathlib.Path, structure_path: pathlib.Path
    ) -> Dict[str, object] | None:
        spec_dir = job_dir / "specificity"
        spec_path = spec_dir / f"{self._structure_basename(structure_path)}.npz"
        if not spec_path.exists():
            raise FileNotFoundError(f"NA-MPNN specificity output missing at {spec_path}.")
        data = np.load(spec_path, allow_pickle=True)
        bundle: Dict[str, object] = {}
        for key in data.files:
            value = data[key]
            if key == "restype_to_int":
                bundle[key] = value.item()
            elif key in {"encoded_residues", "chain_letters"}:
                bundle[key] = value.tolist()
            elif isinstance(value, np.ndarray):
                bundle[key] = value.copy()
            else:
                bundle[key] = value
        return bundle

    @staticmethod
    def _structure_basename(structure_path: pathlib.Path) -> str:
        name = structure_path.name
        for suffix in (".pdb", ".cif", ".mmcif"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    @staticmethod
    def _normalize_chain_list(value: str | Sequence[str]) -> str:
        if isinstance(value, str):
            return value
        return ",".join(str(item).strip() for item in value if str(item).strip())

    @staticmethod
    def _normalize_parse_chain_spec(value: str | Sequence[str]) -> str:
        if isinstance(value, str):
            return value
        return "".join(str(item).strip() for item in value if str(item).strip())

    @staticmethod
    def _normalize_residue_list(value: Sequence[str]) -> str:
        return " ".join(str(item).strip() for item in value if str(item).strip())
