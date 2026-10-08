from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.folding_input import DnaChain, Ligand, ProteinChain, RnaChain
from astrolabe.evaluators.x_masking import preserve_unknown_residue_sequence


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_OPENDDE_REPO = _REPO_ROOT / "vendor" / "opendde"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _bool_cli(value: Any) -> str:
    return "true" if _as_bool(value) else "false"


def _resolve_path(value: str | os.PathLike[str] | None) -> str | None:
    if value is None:
        return None
    text = os.fspath(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return os.path.abspath(os.path.expandvars(os.path.expanduser(text)))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def _chain_id(chain: Any) -> str:
    return str(getattr(chain, "id", "")).strip()


def _chain_kind(chain: Any) -> str:
    if isinstance(chain, ProteinChain) or type(chain).__name__ == "ProteinChain":
        return "protein"
    if isinstance(chain, DnaChain) or type(chain).__name__ == "DnaChain":
        return "dna"
    if isinstance(chain, RnaChain) or type(chain).__name__ == "RnaChain":
        return "rna"
    if isinstance(chain, Ligand) or type(chain).__name__ == "Ligand":
        return "ligand"
    raise TypeError(f"Unsupported OpenDDE chain type: {type(chain).__name__}")


def _modifications(chain: Any, *, polymer: str) -> list[dict[str, object]]:
    raw = (
        getattr(chain, "ptms", None)
        if polymer == "protein"
        else getattr(chain, "modifications", None)
    )
    if callable(raw):
        raw = raw()
    out: list[dict[str, object]] = []
    for mod, site in raw or []:
        code = str(mod)
        if not code.startswith("CCD_"):
            code = f"CCD_{code}"
        if polymer == "protein":
            out.append({"ptmType": code, "ptmPosition": int(site)})
        else:
            out.append({"modificationType": code, "basePosition": int(site)})
    return out


def _msa_dir(*, oligo_name: str, entity_id: int) -> Path:
    return (
        Path(global_config.out_dir)
        / "sequences"
        / "opendde_msa"
        / str(oligo_name)
        / str(entity_id)
    )


@dataclass(frozen=True)
class OpenDDEPreparedInput:
    json_path: Path
    output_dir: Path
    job_name: str
    chain_order: list[dict[str, Any]]
    protein_lengths: dict[str, int]


def opendde_json_from_folding_input(
    cplx_input: Any,
    oligo_name: str,
    *,
    job_name: str,
    include_msa_paths: bool = True,
    use_template: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    """Convert GH's lightweight folding input to OpenDDE's inference JSON job."""

    job: dict[str, Any] = {
        "name": job_name,
        "modelSeeds": [
            int(s) for s in (getattr(cplx_input, "rng_seeds", None) or [101])
        ],
        "sequences": [],
    }
    chain_id_to_entities: dict[str, list[int]] = defaultdict(list)
    chain_order: list[dict[str, Any]] = []
    protein_lengths: dict[str, int] = {}

    def _single_entity_id(chain_id: str, *, purpose: str) -> int:
        entities = chain_id_to_entities.get(str(chain_id)) or []
        if len(entities) != 1:
            raise ValueError(
                f"OpenDDE {purpose} requires chain {chain_id!r} to map to exactly one entity, "
                f"got {len(entities)} entities {entities!r} in {oligo_name!r}."
            )
        return int(entities[0])

    for chain in getattr(cplx_input, "chains", ()) or ():
        kind = _chain_kind(chain)
        cid = _chain_id(chain)
        if not cid:
            raise ValueError(f"OpenDDE {kind} chain in {oligo_name!r} has an empty id.")
        entity_id = len(job["sequences"]) + 1
        chain_id_to_entities[cid].append(entity_id)
        chain_order.append({"id": cid, "kind": kind, "entity_id": entity_id})

        if kind == "protein":
            sequence = preserve_unknown_residue_sequence(
                str(getattr(chain, "sequence", "")), evaluator="opendde"
            )
            protein_lengths[cid] = len(sequence)
            payload: dict[str, Any] = {
                "sequence": sequence,
                "count": 1,
                "id": [cid],
            }
            mods = _modifications(chain, polymer="protein")
            if mods:
                payload["modifications"] = mods
            if include_msa_paths:
                unpaired = str(getattr(chain, "unpaired_msa", "") or "")
                paired = str(getattr(chain, "paired_msa", "") or "")
                if unpaired.strip():
                    msa_path = (
                        _msa_dir(oligo_name=oligo_name, entity_id=entity_id)
                        / "non_pairing.a3m"
                    )
                    _write_text_atomic(msa_path, unpaired)
                    payload["unpairedMsaPath"] = str(msa_path)
                if paired.strip():
                    msa_path = (
                        _msa_dir(oligo_name=oligo_name, entity_id=entity_id)
                        / "pairing.a3m"
                    )
                    _write_text_atomic(msa_path, paired)
                    payload["pairedMsaPath"] = str(msa_path)
            templates = getattr(chain, "templates", None)
            if templates and use_template:
                raise ValueError(
                    "OpenDDE evaluator does not accept inline AF3 Template objects; "
                    "provide OpenDDE templatesPath preprocessing artifacts instead."
                )
            job["sequences"].append({"proteinChain": payload})
            continue

        if kind == "dna":
            payload = {
                "sequence": str(getattr(chain, "sequence", "")),
                "count": 1,
                "id": [cid],
            }
            mods = _modifications(chain, polymer="dna")
            if mods:
                payload["modifications"] = mods
            job["sequences"].append({"dnaSequence": payload})
            continue

        if kind == "rna":
            payload = {
                "sequence": str(getattr(chain, "sequence", "")),
                "count": 1,
                "id": [cid],
            }
            mods = _modifications(chain, polymer="rna")
            if mods:
                payload["modifications"] = mods
            if include_msa_paths:
                unpaired = str(getattr(chain, "unpaired_msa", "") or "")
                if unpaired.strip():
                    msa_path = (
                        _msa_dir(oligo_name=oligo_name, entity_id=entity_id)
                        / "rna_unpaired.a3m"
                    )
                    _write_text_atomic(msa_path, unpaired)
                    payload["unpairedMsaPath"] = str(msa_path)
            job["sequences"].append({"rnaSequence": payload})
            continue

        if kind == "ligand":
            smiles = getattr(chain, "smiles", None)
            ccd_ids = getattr(chain, "ccd_ids", None)
            if smiles is not None:
                ligand_value = str(smiles)
            elif ccd_ids is not None:
                ids = [str(v).removeprefix("CCD_") for v in ccd_ids]
                if not ids:
                    raise ValueError("OpenDDE ligand ccd_ids is empty.")
                ligand_value = "CCD_" + "_".join(ids)
            else:
                raise ValueError(
                    "OpenDDE ligand chain must have either smiles or ccd_ids."
                )
            job["sequences"].append(
                {
                    "ligand": {
                        "ligand": ligand_value,
                        "count": 1,
                        "id": [cid],
                    }
                }
            )
            continue

    bonded = getattr(cplx_input, "bonded_atom_pairs", None) or []
    if bonded:
        covalent_bonds: list[dict[str, object]] = []
        for atom1, atom2 in bonded:
            c1, r1, a1 = atom1
            c2, r2, a2 = atom2
            covalent_bonds.append(
                {
                    "entity1": _single_entity_id(str(c1), purpose="covalent bond"),
                    "copy1": 1,
                    "position1": int(r1),
                    "atom1": str(a1),
                    "entity2": _single_entity_id(str(c2), purpose="covalent bond"),
                    "copy2": 1,
                    "position2": int(r2),
                    "atom2": str(a2),
                }
            )
        job["covalent_bonds"] = covalent_bonds

    return job, chain_order, protein_lengths


@dataclass
class OpenDDERunner:
    python_bin: str
    repo_path: Path = _DEFAULT_OPENDDE_REPO
    opendde_root_dir: Path | None = None
    model_name: str = "opendde_v1"
    load_checkpoint_path: str | None = None
    num_cycles: int = 10
    num_steps: int = 200
    num_samples: int = 5
    dtype: str = "fp32"
    use_msa: bool = False
    use_template: bool = False
    use_rna_msa: bool = False
    need_atom_confidence: bool = False
    triangle_multiplicative: str = "auto"
    triangle_attention: str = "auto"
    enable_cache: bool = True
    enable_fusion: bool = True
    enable_tf32: bool = True
    deterministic: bool = False
    use_tfg_guidance: bool = False
    kalign_binary_path: str | None = None
    hmmsearch_binary_path: str | None = None
    hmmbuild_binary_path: str | None = None
    seqres_database_path: str | None = None
    nhmmer_binary_path: str | None = None
    hmmalign_binary_path: str | None = None
    hmmbuild_rna_binary_path: str | None = None
    ntrna_database_path: str | None = None
    rfam_database_path: str | None = None
    rna_central_database_path: str | None = None
    nhmmer_n_cpu: int | None = None
    foldcp_mode: str = "single"
    foldcp_size_dp: int = 1
    foldcp_size_cp: int = 1
    foldcp_devices: str = ""
    foldcp_metrics_jsonl: str = ""
    timeout_seconds: float | None = None

    @classmethod
    def from_config(cls, conf: Any) -> "OpenDDERunner":
        def get(name: str, default: Any = None) -> Any:
            if conf is None:
                return default
            if isinstance(conf, Mapping):
                return conf.get(name, default)
            return getattr(conf, name, default)

        python_bin = _resolve_path(get("python_bin")) or sys.executable
        repo_path = Path(_resolve_path(get("repo_path", _DEFAULT_OPENDDE_REPO)) or _DEFAULT_OPENDDE_REPO)
        opendde_root = _resolve_path(get("opendde_root_dir"))
        return cls(
            python_bin=python_bin,
            repo_path=repo_path,
            opendde_root_dir=Path(opendde_root) if opendde_root else None,
            model_name=str(get("model_name", "opendde_v1")),
            load_checkpoint_path=_resolve_path(get("load_checkpoint_path")),
            num_cycles=int(get("num_cycles", get("cycle", 10))),
            num_steps=int(get("num_steps", get("step", 200))),
            num_samples=int(get("num_samples", get("sample", 5))),
            dtype=str(get("dtype", "fp32")),
            use_msa=_as_bool(get("use_msa", False)),
            use_template=_as_bool(get("use_template", False)),
            use_rna_msa=_as_bool(get("use_rna_msa", False)),
            need_atom_confidence=_as_bool(get("need_atom_confidence", False)),
            triangle_multiplicative=str(get("triangle_multiplicative", get("trimul_kernel", "auto"))),
            triangle_attention=str(get("triangle_attention", get("triatt_kernel", "auto"))),
            enable_cache=_as_bool(get("enable_cache", True)),
            enable_fusion=_as_bool(get("enable_fusion", True)),
            enable_tf32=_as_bool(get("enable_tf32", True)),
            deterministic=_as_bool(get("deterministic", False)),
            use_tfg_guidance=_as_bool(get("use_tfg_guidance", False)),
            kalign_binary_path=_resolve_path(get("kalign_binary_path")),
            hmmsearch_binary_path=_resolve_path(get("hmmsearch_binary_path")),
            hmmbuild_binary_path=_resolve_path(get("hmmbuild_binary_path")),
            seqres_database_path=_resolve_path(get("seqres_database_path")),
            nhmmer_binary_path=_resolve_path(get("nhmmer_binary_path")),
            hmmalign_binary_path=_resolve_path(get("hmmalign_binary_path")),
            hmmbuild_rna_binary_path=_resolve_path(get("hmmbuild_rna_binary_path")),
            ntrna_database_path=_resolve_path(get("ntrna_database_path")),
            rfam_database_path=_resolve_path(get("rfam_database_path")),
            rna_central_database_path=_resolve_path(get("rna_central_database_path")),
            nhmmer_n_cpu=int(get("nhmmer_n_cpu")) if get("nhmmer_n_cpu") is not None else None,
            foldcp_mode=str(get("foldcp_mode", "single")),
            foldcp_size_dp=int(get("foldcp_size_dp", 1)),
            foldcp_size_cp=int(get("foldcp_size_cp", 1)),
            foldcp_devices=str(get("foldcp_devices", "") or ""),
            foldcp_metrics_jsonl=str(get("foldcp_metrics_jsonl", "") or ""),
            timeout_seconds=float(get("timeout_seconds")) if get("timeout_seconds") is not None else None,
        )

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        repo_str = str(self.repo_path)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = repo_str if not existing else f"{repo_str}{os.pathsep}{existing}"
        if self.opendde_root_dir is not None:
            env["OPENDDE_ROOT_DIR"] = str(self.opendde_root_dir)
        return env

    def _command(self, *, input_json: Path, out_dir: Path) -> list[str]:
        cmd = [
            self.python_bin,
            "-m",
            "runner.batch_inference",
            "pred",
            "-i",
            str(input_json),
            "-o",
            str(out_dir),
            "-n",
            self.model_name,
            "--cycle",
            str(self.num_cycles),
            "--step",
            str(self.num_steps),
            "--sample",
            str(self.num_samples),
            "--dtype",
            self.dtype,
            "--use_msa",
            _bool_cli(self.use_msa),
            "--use_template",
            _bool_cli(self.use_template),
            "--use_rna_msa",
            _bool_cli(self.use_rna_msa),
            "--need_atom_confidence",
            _bool_cli(self.need_atom_confidence),
            "--trimul_kernel",
            self.triangle_multiplicative,
            "--triatt_kernel",
            self.triangle_attention,
            "--enable_cache",
            _bool_cli(self.enable_cache),
            "--enable_fusion",
            _bool_cli(self.enable_fusion),
            "--enable_tf32",
            _bool_cli(self.enable_tf32),
            "--deterministic",
            _bool_cli(self.deterministic),
            "--use_tfg_guidance",
            _bool_cli(self.use_tfg_guidance),
            "--foldcp_mode",
            self.foldcp_mode,
            "--foldcp_size_dp",
            str(self.foldcp_size_dp),
            "--foldcp_size_cp",
            str(self.foldcp_size_cp),
        ]
        optional = {
            "--load_checkpoint_path": self.load_checkpoint_path,
            "--kalign_binary_path": self.kalign_binary_path,
            "--hmmsearch_binary_path": self.hmmsearch_binary_path,
            "--hmmbuild_binary_path": self.hmmbuild_binary_path,
            "--seqres_database_path": self.seqres_database_path,
            "--nhmmer_binary_path": self.nhmmer_binary_path,
            "--hmmalign_binary_path": self.hmmalign_binary_path,
            "--hmmbuild_rna_binary_path": self.hmmbuild_rna_binary_path,
            "--ntrna_database_path": self.ntrna_database_path,
            "--rfam_database_path": self.rfam_database_path,
            "--rna_central_database_path": self.rna_central_database_path,
            "--nhmmer_n_cpu": None if self.nhmmer_n_cpu is None else str(self.nhmmer_n_cpu),
            "--foldcp_devices": self.foldcp_devices or None,
            "--foldcp_metrics_jsonl": self.foldcp_metrics_jsonl or None,
        }
        for flag, value in optional.items():
            if value is not None and str(value).strip():
                cmd.extend([flag, str(value)])
        return cmd

    def run_inference(self, *, input_json: Path, out_dir: Path) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"Vendored OpenDDE repo not found: {self.repo_path}. Initialize vendor/opendde."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._command(input_json=input_json, out_dir=out_dir)
        result = subprocess.run(
            cmd,
            cwd=str(self.repo_path),
            env=self._env(),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        (out_dir / "opendde.stdout.log").write_text(result.stdout or "", encoding="utf-8")
        (out_dir / "opendde.stderr.log").write_text(result.stderr or "", encoding="utf-8")
        if result.returncode != 0:
            tail = "\n".join((result.stderr or result.stdout or "").splitlines()[-20:])
            raise RuntimeError(
                f"OpenDDE subprocess failed with exit code {result.returncode}. "
                f"See {out_dir / 'opendde.stderr.log'}.\n{tail}"
            )


def write_opendde_input(
    cplx_input: Any,
    *,
    oligo_name: str,
    job_name: str,
    input_dir: Path,
    output_dir: Path,
    include_msa_paths: bool = True,
    use_template: bool = False,
) -> OpenDDEPreparedInput:
    job, chain_order, protein_lengths = opendde_json_from_folding_input(
        cplx_input,
        oligo_name,
        job_name=job_name,
        include_msa_paths=include_msa_paths,
        use_template=use_template,
    )
    input_dir.mkdir(parents=True, exist_ok=True)
    json_path = input_dir / f"{job_name}.json"
    json_path.write_text(json.dumps([job], indent=2), encoding="utf-8")
    return OpenDDEPreparedInput(
        json_path=json_path,
        output_dir=output_dir,
        job_name=job_name,
        chain_order=chain_order,
        protein_lengths=protein_lengths,
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"OpenDDE summary JSON must be an object: {path}")
    return data


def _as_float(value: Any, *, field: str) -> float:
    if value is None:
        raise ValueError(f"OpenDDE summary field {field!r} is missing.")
    arr = np.asarray(value, dtype=float)
    if arr.size != 1:
        raise ValueError(f"OpenDDE summary field {field!r} expected scalar, got shape {arr.shape}.")
    return float(arr.reshape(-1)[0])


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def collect_opendde_outputs(prepared: OpenDDEPreparedInput, *, structures_dir: Path) -> dict[str, Any]:
    summary_paths = sorted(
        prepared.output_dir.glob(f"{prepared.job_name}/seed_*/predictions/{prepared.job_name}_summary_confidence_sample_*.json"),
        key=lambda p: (p.parent.parent.name, p.name),
    )
    if not summary_paths:
        err_paths = sorted((prepared.output_dir / "ERR").glob("*.txt"))
        err_text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in err_paths)
        raise FileNotFoundError(
            f"OpenDDE produced no summary confidence files for {prepared.job_name} under {prepared.output_dir}."
            + (f"\n{err_text}" if err_text else "")
        )

    summaries = [_read_json(path) for path in summary_paths]
    structure_paths = [
        path.with_name(path.name.replace("_summary_confidence", "")).with_suffix(".cif")
        for path in summary_paths
    ]
    missing_structures = [str(path) for path in structure_paths if not path.exists()]
    if missing_structures:
        raise FileNotFoundError(
            f"OpenDDE summary exists but structure CIF is missing for {prepared.job_name}: {missing_structures}"
        )

    structures_dir.mkdir(parents=True, exist_ok=True)
    best_structure = structures_dir / f"{prepared.job_name}_model.cif"
    shutil.copy2(structure_paths[0], best_structure)

    plddt = [_as_float(s.get("plddt"), field="plddt") for s in summaries]
    ptm = [_as_float(s.get("ptm"), field="ptm") for s in summaries if s.get("ptm") is not None]
    iptm = [_as_float(s.get("iptm"), field="iptm") for s in summaries if s.get("iptm") is not None]
    gpde = [_as_float(s.get("gpde"), field="gpde") for s in summaries if s.get("gpde") is not None]

    per_proto_scores: dict[str, list[np.ndarray]] = defaultdict(list)
    chain_pair_scores: dict[str, list[np.ndarray]] = {}
    matrix_fields = (
        "chain_pair_iptm",
        "chain_pair_iptm_global",
        "chain_pair_plddt",
        "chain_pair_gpde",
    )
    for field in matrix_fields:
        values = [s.get(field) for s in summaries if s.get(field) is not None]
        if values:
            chain_pair_scores[field] = [_as_array(v) for v in values]

    for summary in summaries:
        chain_plddt = summary.get("chain_plddt")
        if chain_plddt is None:
            continue
        chain_values = list(np.asarray(chain_plddt, dtype=float).reshape(-1))
        for idx, chain_info in enumerate(prepared.chain_order):
            if chain_info["kind"] != "protein":
                continue
            if idx >= len(chain_values):
                raise ValueError(
                    f"OpenDDE chain_plddt has {len(chain_values)} entries but chain index {idx} "
                    f"is needed for {prepared.job_name}."
                )
            cid = str(chain_info["id"])
            length = int(prepared.protein_lengths[cid])
            per_proto_scores[cid].append(np.full(length, chain_values[idx], dtype=float))

    return {
        "plddt": plddt,
        "ptm": ptm or None,
        "iptm": iptm or None,
        "pde": gpde or None,
        "per_proto_scores": dict(per_proto_scores) or None,
        "chain_pair_scores": chain_pair_scores or None,
        "complex_path": best_structure,
    }
