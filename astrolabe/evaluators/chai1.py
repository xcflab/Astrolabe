from __future__ import annotations

import logging
import os
import shutil
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.folding_input import Ligand, ProteinChain
from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import StandardScores
from astrolabe.evaluators.x_masking import (
    patch_chai1_get_esm_embedding_context,
    preserve_unknown_residue_sequence,
)


_CHAI1_MODULE = None
_CHAI1_SUPPORTS_CONSTRAINTS: bool | None = None
_CHAI1_MODULE_PATH: str | None = None
_CCD_SMILES_CACHE: dict[tuple[str, str], str] = {}


def _ccd_components_file() -> Path:
    candidates = [
        os.environ.get("GH_PROTENIX_CCD"),
        os.environ.get("CHAI_CCD_COMPONENTS_FILE"),
    ]
    try:
        protenix_conf = getattr(global_config, "protenix")
        candidates.append(getattr(protenix_conf, "ccd_components_file", None))
    except Exception:
        pass

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.exists():
            return path
    raise FileNotFoundError(
        "No CCD components CIF found for Chai1 ligand conversion. "
        "Set GH_PROTENIX_CCD or CHAI_CCD_COMPONENTS_FILE."
    )


def _smiles_from_ccd(code: str) -> str:
    ccd_code = str(code or "").strip().upper()
    if not ccd_code:
        raise ValueError("Chai1 ligand CCD code cannot be empty.")

    ccd_path = _ccd_components_file()
    cache_key = (str(ccd_path), ccd_code)
    if cache_key in _CCD_SMILES_CACHE:
        return _CCD_SMILES_CACHE[cache_key]

    import gemmi

    doc = gemmi.cif.read_file(str(ccd_path))
    block = doc.find_block(ccd_code)
    if block is None:
        raise KeyError(f"CCD code {ccd_code!r} not found in {ccd_path}.")
    loop_item = block.find_loop("_pdbx_chem_comp_descriptor.type")
    if loop_item is None:
        raise ValueError(f"CCD code {ccd_code!r} has no descriptor loop in {ccd_path}.")
    loop = loop_item.get_loop()
    tags = list(loop.tags)
    type_idx = tags.index("_pdbx_chem_comp_descriptor.type")
    descriptor_idx = tags.index("_pdbx_chem_comp_descriptor.descriptor")

    fallback = ""
    for row_idx in range(loop.length()):
        typ = str(loop[row_idx, type_idx]).strip()
        descriptor = str(loop[row_idx, descriptor_idx]).strip().strip('"')
        if not descriptor or descriptor in {"?", ".", "None"}:
            continue
        if "SMILES_CANONICAL" in typ.upper():
            _CCD_SMILES_CACHE[cache_key] = descriptor
            return descriptor
        if not fallback and "SMILES" in typ.upper():
            fallback = descriptor

    if fallback:
        _CCD_SMILES_CACHE[cache_key] = fallback
        return fallback
    raise ValueError(f"CCD code {ccd_code!r} has no SMILES descriptor in {ccd_path}.")


def _chai_ligand_sequence(chain: Ligand) -> str:
    if chain.smiles is not None:
        smiles = str(chain.smiles).strip()
        if not smiles:
            raise ValueError(f"Chai1 ligand chain {chain.id!r} has empty SMILES.")
        return smiles

    if chain.ccd_ids is not None:
        ccd_ids = [str(ccd).strip().upper() for ccd in chain.ccd_ids if str(ccd).strip()]
        if len(ccd_ids) != 1:
            raise ValueError(
                f"Chai1 ligand chain {chain.id!r} requires exactly one CCD id; got {ccd_ids!r}."
            )
        return _smiles_from_ccd(ccd_ids[0])

    raise ValueError(f"Chai1 ligand chain {chain.id!r} must have smiles or ccd_ids.")


def _load_chai1_module():
    """Prefer vendored chai_lab (supports constraints) over site-packages chai_lab."""
    global _CHAI1_MODULE, _CHAI1_SUPPORTS_CONSTRAINTS, _CHAI1_MODULE_PATH
    if _CHAI1_MODULE is not None and _CHAI1_SUPPORTS_CONSTRAINTS is not None:
        return _CHAI1_MODULE

    import importlib
    import inspect
    import sys

    def _supports_constraints(mod: Any) -> bool:
        try:
            sig = inspect.signature(mod.run_inference)
        except Exception:
            return False
        return "constraint_path" in sig.parameters

    # If a chai_lab package is already imported but lacks constraint support, purge it so we
    # can re-import from vendor/ (if available).
    def _purge_chai_lab() -> None:
        for k in list(sys.modules.keys()):
            if k == "chai_lab" or k.startswith("chai_lab."):
                del sys.modules[k]

    repo_root = Path(__file__).resolve().parents[2]
    vendor_root = repo_root / "vendor" / "chai_lab"

    # Try vendored first.
    if vendor_root.exists():
        sys.path.insert(0, str(vendor_root))
        _purge_chai_lab()
        try:
            chai1_mod = importlib.import_module("chai_lab.chai1")
            _CHAI1_MODULE_PATH = getattr(chai1_mod, "__file__", None)
            _CHAI1_SUPPORTS_CONSTRAINTS = _supports_constraints(chai1_mod)
            _CHAI1_MODULE = chai1_mod
            return chai1_mod
        except Exception:
            # Fall back to site-packages below.
            pass

    # Fallback: whatever chai_lab is installed in the environment.
    chai1_mod = importlib.import_module("chai_lab.chai1")
    _CHAI1_MODULE_PATH = getattr(chai1_mod, "__file__", None)
    _CHAI1_SUPPORTS_CONSTRAINTS = _supports_constraints(chai1_mod)
    _CHAI1_MODULE = chai1_mod
    return chai1_mod


def _maybe_write_aligned_pqt(
    *,
    msas_dir: Path,
    query_sequence: str,
    unpaired_a3m: str | None,
    paired_a3m: str | None,
    max_unpaired_rows: int | None = None,
    max_paired_rows: int | None = None,
) -> Path | None:
    """Materialize Chai MSAs for a protein sequence when MSA strings exist.

    Chai expects `msa_directory/<sha>.aligned.pqt` files (schema: sequence/source_database/pairing_key/comment).
    Writing these requires a Parquet engine (pyarrow/fastparquet) in the runtime env.

    We currently materialize:
      - unpaired MSA (when provided) as an `.aligned.pqt` file.

    Notes:
      - `paired_a3m` is accepted for API completeness but not yet used (we don't know
        the provenance/source for AF3 paired MSAs; mislabeling it would be subtle).
    """
    query_sequence = str(query_sequence or "").strip()
    if not query_sequence:
        raise ValueError("Chai MSA writing requires a non-empty query_sequence.")

    unpaired_a3m = None if unpaired_a3m is None else str(unpaired_a3m).strip()
    paired_a3m = None if paired_a3m is None else str(paired_a3m).strip()
    if not unpaired_a3m and not paired_a3m:
        return None

    def _ensure_parquet_engine() -> None:
        try:
            import pyarrow  # noqa: F401
            return
        except Exception:
            pass
        try:
            import fastparquet  # noqa: F401
            return
        except Exception:
            pass
        raise ModuleNotFoundError(
            "Chai precomputed MSA writing requires a Parquet engine (pyarrow or fastparquet). "
            "Install it in the `gh` env (see docs/third_party_envs.md)."
        )

    def _expected_basename(seq: str) -> str:
        import hashlib

        seqhash = hashlib.sha256(seq.upper().encode()).hexdigest()
        return f"{seqhash}.aligned.pqt"

    def _parse_a3m_records(text: str) -> list[tuple[str, str]]:
        records: list[tuple[str, str]] = []
        header = ""
        seq_lines: list[str] = []
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_lines:
                    records.append((header, "".join(seq_lines)))
                header = line[1:].strip()
                seq_lines = []
                continue
            seq_lines.append(line)
        if seq_lines:
            records.append((header, "".join(seq_lines)))
        return records

    msas_dir.mkdir(parents=True, exist_ok=True)
    out_path = msas_dir / _expected_basename(query_sequence)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    if not unpaired_a3m:
        # Paired MSAs are intentionally ignored for now.
        return None

    _ensure_parquet_engine()
    records = _parse_a3m_records(unpaired_a3m)
    if not records:
        return None

    # Ensure query row exists and is first.
    if records[0][1].replace("-", "").replace(".", "").upper() != query_sequence.upper():
        records.insert(0, ("query", query_sequence))

    if max_unpaired_rows is not None and max_unpaired_rows > 0:
        records = records[: max(1, int(max_unpaired_rows))]

    import pandas as pd

    out_rows: list[dict[str, str]] = []
    for i, (header, seq) in enumerate(records):
        out_rows.append(
            {
                "sequence": str(seq),
                "source_database": "query" if i == 0 else "uniref90",
                "pairing_key": "",
                "comment": str(header or ""),
            }
        )

    aligned_df = pd.DataFrame.from_records(out_rows)
    tmp_out = out_path.with_suffix(out_path.suffix + f".{os.getpid()}.tmp")
    aligned_df.to_parquet(tmp_out, index=False)
    os.replace(str(tmp_out), str(out_path))
    return out_path


def _normalize_explicit_chai_restraints(
    raw_cfg: Any,
    *,
    oligo_name: str,
    seq_by_chain: Mapping[str, str],
) -> list[dict[str, object]]:
    """Normalize explicit Chai restraint rows from config.

    Supported forms:
      chai_restraints:
        - {chainA: A, res_idxA: Q76, chainB: B, res_idxB: C15, connection_type: contact}

      chai_restraints:
        AB:
          - {chainA: A, res_idxA: Q76, chainB: B, res_idxB: C15, connection_type: contact}
    """
    if raw_cfg is None:
        return []

    def _to_plain(value: Any) -> Any:
        if hasattr(value, "items") and not isinstance(value, dict):
            return {k: _to_plain(value[k]) for k in value.keys()}
        if isinstance(value, list):
            return [_to_plain(x) for x in value]
        return value

    cfg = _to_plain(raw_cfg)
    direct_keys = {
        "chainA",
        "res_idxA",
        "chainB",
        "res_idxB",
        "connection_type",
        "confidence",
        "min_distance_angstrom",
        "max_distance_angstrom",
        "comment",
        "restraint_id",
    }

    if isinstance(cfg, Mapping) and not (set(cfg.keys()) & direct_keys):
        cfg = cfg.get(str(oligo_name))
        if cfg is None:
            return []

    if isinstance(cfg, Mapping):
        raw_rows: list[Any] = [cfg]
    elif isinstance(cfg, Sequence) and not isinstance(cfg, (str, bytes)):
        raw_rows = list(cfg)
    else:
        raise ValueError(
            "chai_restraints must be a row mapping, a list of row mappings, or a per-oligomer mapping."
        )

    def _normalize_res_idx(raw_value: Any, *, chain_id: str, field_name: str) -> str:
        if raw_value is None:
            raise ValueError(f"chai_restraints.{field_name} is required")
        text = str(raw_value).strip()
        if not text:
            raise ValueError(f"chai_restraints.{field_name} must be non-empty")
        if text.isdigit():
            pos = int(text)
            seq = seq_by_chain.get(chain_id)
            if seq is None:
                raise ValueError(f"chai_restraints references unknown chain {chain_id!r}")
            if pos <= 0 or pos > len(seq):
                raise ValueError(
                    f"chai_restraints.{field_name} position {pos} is out of range for chain {chain_id!r} (1..{len(seq)})."
                )
            return f"{seq[pos - 1]}{pos}"
        return text

    rows: list[dict[str, object]] = []
    for idx, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"chai_restraints row #{idx} must be a mapping, got {type(raw_row)!r}")
        missing = [key for key in ("chainA", "res_idxA", "chainB", "res_idxB", "connection_type") if key not in raw_row]
        if missing:
            raise ValueError(f"chai_restraints row #{idx} is missing keys: {missing}")

        chain_a = str(raw_row["chainA"]).strip()
        chain_b = str(raw_row["chainB"]).strip()
        if chain_a not in seq_by_chain or chain_b not in seq_by_chain:
            raise ValueError(
                f"chai_restraints row #{idx} references unknown chains {(chain_a, chain_b)!r}; present={sorted(seq_by_chain)}"
            )

        rows.append(
            {
                "chainA": chain_a,
                "res_idxA": _normalize_res_idx(raw_row["res_idxA"], chain_id=chain_a, field_name="res_idxA"),
                "chainB": chain_b,
                "res_idxB": _normalize_res_idx(raw_row["res_idxB"], chain_id=chain_b, field_name="res_idxB"),
                "connection_type": str(raw_row["connection_type"]).strip(),
                "confidence": float(raw_row.get("confidence", 1.0)),
                "min_distance_angstrom": float(raw_row.get("min_distance_angstrom", 0.0)),
                "max_distance_angstrom": float(raw_row.get("max_distance_angstrom", 6.0)),
                "comment": str(raw_row.get("comment", "user_chai_restraint")),
                "restraint_id": str(raw_row.get("restraint_id", f"user_restraint_{idx}")),
            }
        )
    return rows


class Chai1Evaluator(Evaluator):
    def __init__(self, model_config, output_dir):
        self.output_dir = output_dir
        self.model_config = model_config

    def run_function(self, input_feat):
        output = {}
        start = time.time()
        import torch

        model_kwargs = dict(self.model_config)
        if model_kwargs.get("device") is None:
            model_kwargs["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
        chai1 = _load_chai1_module()
        with patch_chai1_get_esm_embedding_context(chai1):
            for oligo_name, cplx_input in input_feat.items():
                if (
                    cplx_input.get("constraint_path") is not None
                    and not (_CHAI1_SUPPORTS_CONSTRAINTS is True)
                ):
                    raise TypeError(
                        "This chai_lab installation does not support `constraint_path` "
                        f"(chai1={_CHAI1_MODULE_PATH}). Use the vendored version under "
                        f"{Path(__file__).resolve().parents[2] / 'vendor' / 'chai_lab'}."
                    )
                output[oligo_name] = chai1.run_inference(**cplx_input, **model_kwargs)
        end = time.time()
        logging.info(f"Chai1 inference took {end - start:.2f} seconds")
        return output

    def prepare_input(self, inputs, runtime_config):
        step = global_config.step
        result = {}
        from astrolabe.core.binding_site import parse_binding_site

        for oligo_name, inp in inputs.items():
            fasta_file = (
                global_config.out_dir
                / "sequences"
                / f"step_{step:03d}_{oligo_name}_input.fasta"
            )
            output_dir = self.output_dir / f"step_{step:03d}_{oligo_name}"
            msas_dir = (
                global_config.out_dir
                / "sequences"
                / "chai1_msas"
                / str(oligo_name)
            )

            fasta_file.parent.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Chai expects a fasta on disk; write it from the current chains.
            lines = []
            chains_with_msa: list[str] = []
            use_msa_server = bool(dict(self.model_config).get("use_msa_server", False))
            for fallback_chain_id, chain in zip(inp.name, inp.chains):
                chain_id = str(getattr(chain, "id", fallback_chain_id))
                if isinstance(chain, ProteinChain):
                    seq = getattr(chain, "sequence", None)
                    if not seq:
                        raise ValueError(f"Empty sequence for chain {chain_id}")
                    seq = preserve_unknown_residue_sequence(str(seq), evaluator="chai1")
                    lines.append(f">protein|{chain_id}")
                    lines.append(seq)
                    if not use_msa_server and str(getattr(chain, "unpaired_msa", "") or "").strip():
                        chains_with_msa.append(str(chain_id))
                    continue

                if isinstance(chain, Ligand):
                    lines.append(f">ligand|{chain_id}")
                    lines.append(_chai_ligand_sequence(chain))
                    continue

                if not isinstance(chain, ProteinChain):
                    raise TypeError(
                        f"Chai1Evaluator only supports ProteinChain and Ligand, got {type(chain)}"
                    )

            tmp_path = fasta_file.with_suffix(fasta_file.suffix + f".{os.getpid()}.tmp")
            tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            os.replace(str(tmp_path), str(fasta_file))

            seq_by_chain = {}
            for fallback_chain_id, chain in zip(inp.name, inp.chains):
                if not isinstance(chain, ProteinChain):
                    continue
                seq_by_chain[str(getattr(chain, "id", fallback_chain_id))] = (
                    preserve_unknown_residue_sequence(
                        str(getattr(chain, "sequence")), evaluator="chai1"
                    )
                )

            def _aa_at(chain_id: str, pos: int) -> str:
                seq = seq_by_chain.get(str(chain_id))
                if seq is None:
                    raise ValueError(f"Unknown chain {chain_id!r}")
                if pos <= 0 or pos > len(seq):
                    raise ValueError(f"Residue position {pos} out of range (1..{len(seq)})")
                return seq[pos - 1]

            rows: list[dict[str, object]] = []

            spec = parse_binding_site(
                getattr(global_config, "binding_site", None),
                oligo_name=oligo_name,
            )
            if spec is not None:
                from astrolabe.evaluators.binding_site_constraints import (
                    build_chai_binding_site_rows,
                )

                rows.extend(
                    build_chai_binding_site_rows(
                        spec,
                        seq_by_chain=seq_by_chain,
                        context=oligo_name,
                        oligo_name=oligo_name,
                    )
                )

            explicit_chai_restraints = None
            try:
                explicit_chai_restraints = getattr(global_config, "chai_restraints")
            except Exception:
                explicit_chai_restraints = None

            rows.extend(
                _normalize_explicit_chai_restraints(
                    explicit_chai_restraints,
                    oligo_name=oligo_name,
                    seq_by_chain=seq_by_chain,
                )
            )

            bonded = getattr(inp, "bonded_atom_pairs", None) or []
            if bonded:
                offset = len(rows)
                for idx, (atom1, atom2) in enumerate(bonded):
                    c1, r1, a1 = atom1
                    c2, r2, a2 = atom2
                    c1s = str(c1)
                    c2s = str(c2)
                    # Skip constraints for single-chain stages (e.g. "A" before "AB").
                    if c1s not in seq_by_chain or c2s not in seq_by_chain:
                        logging.info(
                            "Skip cyclization covalent bond for '%s': missing chains (c1=%s present=%s, c2=%s present=%s)",
                            oligo_name,
                            c1s,
                            c1s in seq_by_chain,
                            c2s,
                            c2s in seq_by_chain,
                        )
                        continue

                    rows.append(
                        {
                            "chainA": c1s,
                            "res_idxA": f"{_aa_at(c1s, int(r1))}{int(r1)}@{str(a1)}",
                            "chainB": c2s,
                            "res_idxB": f"{_aa_at(c2s, int(r2))}{int(r2)}@{str(a2)}",
                            "connection_type": "covalent",
                            "confidence": 1.0,
                            "min_distance_angstrom": 0.0,
                            "max_distance_angstrom": 0.0,
                            "comment": "auto_cyclization",
                            "restraint_id": f"restraint_{offset + idx}",
                        }
                    )

            constraint_path: Path | None = None
            if rows:
                constraint_path = (
                    global_config.out_dir
                    / "sequences"
                    / "chai1_constraints"
                    / f"step_{step:03d}_{oligo_name}.restraints"
                )
                constraint_path.parent.mkdir(parents=True, exist_ok=True)
                import csv

                header = [
                    "chainA",
                    "res_idxA",
                    "chainB",
                    "res_idxB",
                    "connection_type",
                    "confidence",
                    "min_distance_angstrom",
                    "max_distance_angstrom",
                    "comment",
                    "restraint_id",
                ]
                tmp = constraint_path.with_suffix(
                    constraint_path.suffix + f".{os.getpid()}.tmp"
                )
                with tmp.open("w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=header)
                    w.writeheader()
                    for row in rows:
                        w.writerow(row)
                os.replace(str(tmp), str(constraint_path))

            msa_directory: Path | None = None
            if not use_msa_server:
                wrote_any = False
                for chain_id, chain in zip(inp.name, inp.chains):
                    if not isinstance(chain, ProteinChain):
                        continue
                    unpaired = getattr(chain, "unpaired_msa", None)
                    paired = getattr(chain, "paired_msa", None)
                    query_sequence = preserve_unknown_residue_sequence(
                        str(getattr(chain, "sequence")), evaluator="chai1"
                    )
                    out = _maybe_write_aligned_pqt(
                        msas_dir=msas_dir,
                        query_sequence=query_sequence,
                        unpaired_a3m=unpaired,
                        paired_a3m=paired,
                    )
                    wrote_any = wrote_any or (out is not None)
                if wrote_any:
                    msa_directory = msas_dir
                    logging.info(
                        "Chai1Evaluator using msa_directory=%s (chains_with_unpaired_msa=%s)",
                        msa_directory,
                        ",".join(chains_with_msa) if chains_with_msa else "<none>",
                    )

            result[oligo_name] = {
                "fasta_file": fasta_file,
                "output_dir": output_dir,
                "constraint_path": constraint_path,
                "msa_directory": msa_directory,
                "cyclic_subchains": [
                    str(proto_chain_id)
                    for proto_chain_id, chain in zip(inp.name, inp.chains)
                    if str(getattr(chain, "id", "")) in {
                        str(cid)
                        for cid in (getattr(inp, "cyclic_chain_ids", None) or ())
                    }
                ]
                or None,
                "cyclic_relative_patches": [
                    {
                        "chain_id": str(patch.chain_id),
                        "offsets": [
                            (int(rel.residue_i), int(rel.residue_j), int(rel.offset))
                            for rel in patch.offsets
                        ],
                    }
                    for patch in (getattr(inp, "cyclic_relative_patches", None) or ())
                ]
                or None,
            }

        return result

    def prepare_output(self, output, inputs, runtime_config):
        per_chain_plddt = defaultdict(list)
        pae = {}
        ptm = {}
        iptm = {}
        pde = {}
        plddt_per_complex = {}
        chain_pair_scores = {}
        complex_paths = {}
        for oligo_name, input in inputs.items():
            res_idx = 0
            this_score = output[oligo_name]
            for chain_id, chain in zip(input.name, input.chains):
                if isinstance(chain, ProteinChain):
                    per_chain_plddt[chain_id] = [
                        plddt
                        for plddt in this_score.plddt[
                            :, res_idx : res_idx + len(chain.sequence)
                        ]
                        * 100
                    ]
                    res_idx += len(chain.sequence)

            plddt_per_complex[oligo_name] = np.mean(np.array(this_score.plddt))
            ptm[oligo_name] = [
                score.ptm_scores.complex_ptm for score in this_score.ranking_data
            ]
            iptm[oligo_name] = [
                score.ptm_scores.interface_ptm for score in this_score.ranking_data
            ]
            chain_pair_scores[oligo_name] = {
                "chain_ptm": [
                    np.asarray(score.ptm_scores.per_chain_ptm, dtype=float)
                    for score in this_score.ranking_data
                ],
                "chain_pair_iptm": [
                    np.asarray(score.ptm_scores.per_chain_pair_iptm, dtype=float)
                    for score in this_score.ranking_data
                ],
                "chain_plddt": [
                    np.asarray(score.plddt_scores.per_chain_plddt, dtype=float)
                    for score in this_score.ranking_data
                ],
            }
            pde[oligo_name] = this_score.pde
            pae[oligo_name] = this_score.pae
            ranking = [score.aggregate_score for score in this_score.ranking_data]
            which_best = np.argmax(ranking)
            complex_paths[oligo_name] = (
                global_config.out_dir
                / "structures"
                / f"step_{global_config.step:03d}_{oligo_name}_model.cif"
            )
            shutil.copy(this_score.cif_paths[which_best], complex_paths[oligo_name])

        return StandardScores(
            names=list(inputs.keys()),
            plddt=plddt_per_complex,
            ptm=ptm,
            pae=pae,
            iptm=iptm,
            per_proto_scores=per_chain_plddt,
            pde=pde,
            chain_pair_scores=chain_pair_scores,
            complex_paths=complex_paths,
        )
