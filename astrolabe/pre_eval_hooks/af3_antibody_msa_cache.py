from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Mapping

from astrolabe.core.folding_input import (
    DnaChain,
    Input,
    Ligand,
    ProteinChain,
    RnaChain,
    Template,
)
from astrolabe.core.msa_utils import load_af3_chain_data


def _to_plain_mapping(value: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return dict(value or {})


def _is_enabled(raw_config: Any) -> bool:
    cfg = _to_plain_mapping(raw_config)
    return bool(cfg.get("enabled", False))


def _selected_chain_id(raw_config: Any) -> str:
    cfg = _to_plain_mapping(raw_config)
    raw = cfg.get("chain_ids", ("A",))
    if raw in (None, "null", "None"):
        selected = ["A"]
    elif isinstance(raw, str):
        selected = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        selected = [str(x).strip() for x in raw if str(x).strip()]
    if selected != ["A"]:
        raise ValueError("pre_eval_hooks.af3_antibody_msa_cache only supports chain_ids: [A]")
    return "A"


def _residue_difference(left: str, right: str, *, max_distance: int) -> int | None:
    """Return same-length residue differences, or None when already too far."""
    left = str(left or "")
    right = str(right or "")
    if len(left) != len(right):
        return None
    distance = 0
    for a, b in zip(left, right):
        if a != b:
            distance += 1
            if distance > max_distance:
                return None
    return distance


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_payload(chain: Any) -> bool:
    return bool(
        _has_text(getattr(chain, "unpaired_msa", None))
        or _has_text(getattr(chain, "paired_msa", None))
        or list(getattr(chain, "templates", None) or [])
    )


def _has_unpaired_msa(chain: Any) -> bool:
    return _has_text(getattr(chain, "unpaired_msa", None))


def _entry_id(*, chain_id: str, sequence: str) -> str:
    digest = hashlib.sha256(f"{chain_id}\0{sequence}".encode("utf-8")).hexdigest()
    return digest[:24]


def _validate_template_mapping(
    *,
    query_indices: list[int],
    template_indices: list[int],
    sequence: str,
) -> None:
    if len(query_indices) != len(template_indices):
        raise ValueError(
            "AF3 antibody MSA template queryIndices and templateIndices must have the same length"
        )
    query_length = len(sequence)
    for query_index in query_indices:
        if query_index < 0 or query_index >= query_length:
            raise ValueError(
                "AF3 antibody MSA template query index "
                f"{query_index} is outside query sequence length {query_length}"
            )
    for template_index in template_indices:
        if template_index < 0:
            raise ValueError(
                f"AF3 antibody MSA template index {template_index} must be non-negative"
            )


def _template_to_json(template: Any, *, sequence: str | None = None) -> dict[str, Any]:
    if isinstance(template, Template):
        mapping = dict(template.query_to_template_map)
        query_indices = [int(k) for k in mapping.keys()]
        template_indices = [int(v) for v in mapping.values()]
        if sequence is not None:
            _validate_template_mapping(
                query_indices=query_indices,
                template_indices=template_indices,
                sequence=sequence,
            )
        return {
            "mmcif": template.mmcif,
            "queryIndices": query_indices,
            "templateIndices": template_indices,
        }
    if isinstance(template, dict):
        if "query_to_template_map" in template:
            mapping = dict(template.get("query_to_template_map") or {})
            data = {
                "mmcif": template.get("mmcif"),
                "queryIndices": [int(k) for k in mapping.keys()],
                "templateIndices": [int(v) for v in mapping.values()],
            }
        else:
            data = dict(template)
        if sequence is not None:
            query_indices = [int(x) for x in data.get("queryIndices") or []]
            template_indices = [int(x) for x in data.get("templateIndices") or []]
            _validate_template_mapping(
                query_indices=query_indices,
                template_indices=template_indices,
                sequence=sequence,
            )
        return data
    mapping = getattr(template, "query_to_template_map", None)
    if mapping is not None:
        mapping = dict(mapping)
        query_indices = [int(k) for k in mapping.keys()]
        template_indices = [int(v) for v in mapping.values()]
        if sequence is not None:
            _validate_template_mapping(
                query_indices=query_indices,
                template_indices=template_indices,
                sequence=sequence,
            )
        return {
            "mmcif": getattr(template, "mmcif", None),
            "queryIndices": query_indices,
            "templateIndices": template_indices,
        }
    raise TypeError(f"Unsupported template payload: {type(template).__name__}")


def _sha256_text(value: str | None) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _sha256_templates(templates: Any, *, sequence: str) -> str:
    payload = [
        _template_to_json(template, sequence=sequence)
        for template in list(templates or [])
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _template_from_json(data: dict[str, Any], *, sequence: str) -> Template:
    query_indices = [int(x) for x in data.get("queryIndices") or []]
    template_indices = [int(x) for x in data.get("templateIndices") or []]
    _validate_template_mapping(
        query_indices=query_indices,
        template_indices=template_indices,
        sequence=sequence,
    )
    return Template(
        mmcif=data.get("mmcif"),
        query_to_template_map=dict(zip(query_indices, template_indices)),
    )


def _coerce_templates(templates: Any, *, sequence: str) -> tuple[Template, ...]:
    return tuple(
        t if isinstance(t, Template) else _template_from_json(_template_to_json(t, sequence=sequence), sequence=sequence)
        for t in templates or ()
    )


def _a3m_first_sequence_length(msa_text: str) -> int | None:
    sequence_lines: list[str] = []
    in_first = False
    for raw_line in str(msa_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if in_first:
                break
            in_first = True
            continue
        if in_first:
            sequence_lines.append(line)
    if not sequence_lines:
        return None
    clean = "".join(
        char for char in "".join(sequence_lines) if char.isupper() or char == "-"
    )
    return len(clean.replace("-", ""))


def _retarget_a3m_query(msa_text: str | None, sequence: str) -> str | None:
    if not _has_text(msa_text):
        return None
    expected = _a3m_first_sequence_length(str(msa_text))
    if expected != len(sequence):
        raise ValueError(
            f"Cannot reuse MSA with query length {expected} for sequence length {len(sequence)}"
        )
    lines = str(msa_text).splitlines()
    header_index = None
    next_header = None
    for idx, line in enumerate(lines):
        if line.startswith(">"):
            if header_index is None:
                header_index = idx
                continue
            next_header = idx
            break
    if header_index is None:
        raise ValueError("Cannot reuse MSA without a query header")
    end = len(lines) if next_header is None else next_header
    new_lines = lines[: header_index + 1] + [sequence] + lines[end:]
    return "\n".join(new_lines) + "\n"


@dataclasses.dataclass(frozen=True, slots=True)
class MsaCacheEntry:
    chain_id: str
    sequence: str
    unpaired_msa: str | None
    paired_msa: str | None
    templates: tuple[Template, ...]
    path: Path | None = None

    @property
    def id(self) -> str:
        return _entry_id(chain_id=self.chain_id, sequence=self.sequence)


class AntibodyMsaCache:
    def __init__(self, *, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._loaded = False
        self._entries: dict[tuple[str, str], MsaCacheEntry] = {}

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.cache_dir.exists():
            return
        for path in sorted(self.cache_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            sequence = str(payload["sequence"])
            entry = MsaCacheEntry(
                chain_id=str(payload["chain_id"]),
                sequence=sequence,
                unpaired_msa=payload.get("unpaired_msa"),
                # Antibody-only searches cannot establish antibody-target
                # pairing. Older cache files may contain such rows, but they
                # are deliberately ignored on load.
                paired_msa=None,
                templates=tuple(
                    _template_from_json(t, sequence=sequence)
                    for t in list(payload.get("templates") or [])
                    if isinstance(t, dict)
                ),
                path=path,
            )
            self._entries[(entry.chain_id, entry.sequence)] = entry

    def add_from_chain(self, *, chain_id: str, chain: Any) -> MsaCacheEntry | None:
        if not _has_payload(chain):
            return None
        sequence = str(getattr(chain, "sequence", "") or "")
        if not sequence:
            return None
        entry = MsaCacheEntry(
            chain_id=str(chain_id),
            sequence=sequence,
            unpaired_msa=getattr(chain, "unpaired_msa", None) or None,
            paired_msa=None,
            templates=_coerce_templates(getattr(chain, "templates", None), sequence=sequence),
        )
        self.add_entry(entry)
        return entry

    def add_entry(self, entry: MsaCacheEntry) -> None:
        if entry.paired_msa:
            entry = dataclasses.replace(entry, paired_msa=None)
        key = (entry.chain_id, entry.sequence)
        self._entries[key] = entry
        self._write(entry)

    def find_nearest(
        self,
        *,
        chain_id: str,
        sequence: str,
        max_distance: int,
    ) -> tuple[MsaCacheEntry, int] | None:
        best: tuple[MsaCacheEntry, int] | None = None
        for entry in self._entries.values():
            if entry.chain_id != str(chain_id):
                continue
            distance = _residue_difference(
                sequence,
                entry.sequence,
                max_distance=max_distance,
            )
            if distance is None:
                continue
            if best is None or distance < best[1]:
                best = (entry, distance)
                if distance == 0:
                    break
        return best

    def _write(self, entry: MsaCacheEntry) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{entry.id}.json"
        payload = {
            "chain_id": entry.chain_id,
            "sequence": entry.sequence,
            "unpaired_msa": entry.unpaired_msa,
            "paired_msa": "",
            "templates": [
                _template_to_json(t, sequence=entry.sequence) for t in entry.templates
            ],
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)


def _replace_chain_payload(
    chain: ProteinChain,
    entry: MsaCacheEntry,
    *,
    preserve_existing_templates: bool,
    expected_template_count: int | None,
) -> ProteinChain:
    sequence = str(getattr(chain, "sequence", "") or "")
    template_source = chain.templates if preserve_existing_templates else entry.templates
    templates = list(_coerce_templates(template_source, sequence=sequence))
    if expected_template_count is not None and len(templates) != expected_template_count:
        raise ValueError(
            "AF3 antibody framework template count must be "
            f"{expected_template_count}, got {len(templates)}"
        )
    return dataclasses.replace(
        chain,
        unpaired_msa=_retarget_a3m_query(entry.unpaired_msa, sequence),
        paired_msa="",
        templates=templates,
    )


def _clear_all_paired_msa(inputs: dict[str, Input]) -> dict[str, Input]:
    updated: dict[str, Input] = {}
    for oligo_name, input_obj in inputs.items():
        chains = [
            dataclasses.replace(chain, paired_msa="")
            if isinstance(chain, ProteinChain) and _has_text(chain.paired_msa)
            else chain
            for chain in input_obj.chains
        ]
        updated[oligo_name] = (
            dataclasses.replace(input_obj, chains=chains)
            if any(new is not old for new, old in zip(chains, input_obj.chains))
            else input_obj
        )
    return updated


def _a3m_sequence_count(msa_text: str | None) -> int:
    if not _has_text(msa_text):
        return 0
    return sum(1 for line in str(msa_text).splitlines() if line.startswith(">"))


def _validate_final_chain_payloads(
    inputs: dict[str, Input],
    *,
    expected_template_count: int | None,
    minimum_unpaired_msa_sequences: int,
) -> None:
    for oligo_name, input_obj in inputs.items():
        for chain in input_obj.chains:
            chain_id = str(chain.id)
            if _has_text(chain.paired_msa):
                raise ValueError(
                    f"AF3 input oligo={oligo_name} chain={chain_id} retains paired MSA rows"
                )
            if minimum_unpaired_msa_sequences:
                count = _a3m_sequence_count(chain.unpaired_msa)
                if count < minimum_unpaired_msa_sequences:
                    raise ValueError(
                        f"AF3 input oligo={oligo_name} chain={chain_id} has "
                        f"{count} unpaired MSA sequences; expected at least "
                        f"{minimum_unpaired_msa_sequences}"
                    )
            templates = _coerce_templates(chain.templates, sequence=str(chain.sequence))
            if expected_template_count is not None and len(templates) != expected_template_count:
                raise ValueError(
                    f"AF3 input oligo={oligo_name} chain={chain_id} has "
                    f"{len(templates)} templates; expected {expected_template_count}"
                )


def _replace_input_chain(input_obj: Input, chain_index: int, chain: ProteinChain) -> Input:
    chains = list(input_obj.chains)
    chains[chain_index] = chain
    return dataclasses.replace(input_obj, chains=chains)


def _current_semantic_protomer_sequence(protomer: Any, *, semantic_id: str) -> str:
    sequence = getattr(protomer, "mutated_sequence", None)
    if sequence is None:
        sequence = getattr(protomer, "sequence", None)
    if sequence is None:
        raise ValueError(
            f"pre_eval_hooks.af3_antibody_msa_cache semantic protomer {semantic_id} has no current sequence"
        )
    return str(sequence)


def _semantic_protomer_expectation(raw_protomers: Mapping[str, Any], semantic_id: str) -> dict[str, Any]:
    if semantic_id not in raw_protomers:
        raise ValueError(
            f"pre_eval_hooks.af3_antibody_msa_cache missing semantic protomer expectation for {semantic_id}"
        )
    protomer = raw_protomers[semantic_id]
    sequence = _current_semantic_protomer_sequence(
        protomer,
        semantic_id=semantic_id,
    )
    homology = _to_plain_mapping(getattr(protomer, "homology", {}) or {})
    return {
        "semantic_id": semantic_id,
        "sequence": sequence,
        "unpaired_msa": homology.get("unpairedMsa"),
        "templates": list(homology.get("templates") or []),
    }


def _validate_antibody_chain_a_sequence(
    *,
    oligo_name: str,
    chain: ProteinChain,
    expected: dict[str, Any],
) -> str:
    sequence = str(chain.sequence)
    expected_sequence = str(expected["sequence"])
    if len(sequence) != len(expected_sequence):
        raise ValueError(
            "pre_eval_hooks.af3_antibody_msa_cache "
            f"oligo={oligo_name} semantic antibody A length mismatch: "
            f"expected {len(expected_sequence)}, got {len(sequence)}"
        )
    if sequence != expected_sequence:
        raise ValueError(
            "pre_eval_hooks.af3_antibody_msa_cache "
            f"oligo={oligo_name} semantic antibody A sequence mismatch"
        )
    return sequence


def _validate_semantic_target_payload(
    *,
    oligo_name: str,
    target_chain: ProteinChain,
    semantic_target_id: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    sequence = str(target_chain.sequence)
    expected_sequence = str(expected["sequence"])
    if len(sequence) != len(expected_sequence):
        raise ValueError(
            "pre_eval_hooks.af3_antibody_msa_cache "
            f"oligo={oligo_name} semantic target {semantic_target_id} length mismatch: "
            f"expected {len(expected_sequence)}, got {len(sequence)}"
        )
    if sequence != expected_sequence:
        raise ValueError(
            "pre_eval_hooks.af3_antibody_msa_cache "
            f"oligo={oligo_name} semantic target {semantic_target_id} sequence mismatch"
        )

    target_unpaired_msa_sha256 = _sha256_text(target_chain.unpaired_msa)
    expected_unpaired_msa_sha256 = _sha256_text(expected.get("unpaired_msa"))
    if target_unpaired_msa_sha256 != expected_unpaired_msa_sha256:
        raise ValueError(
            "pre_eval_hooks.af3_antibody_msa_cache "
            f"oligo={oligo_name} semantic target {semantic_target_id} "
            "unpaired MSA SHA256 mismatch"
        )

    target_template_sha256 = _sha256_templates(
        target_chain.templates,
        sequence=sequence,
    )
    expected_template_sha256 = _sha256_templates(
        expected.get("templates"),
        sequence=expected_sequence,
    )
    if target_template_sha256 != expected_template_sha256:
        raise ValueError(
            "pre_eval_hooks.af3_antibody_msa_cache "
            f"oligo={oligo_name} semantic target {semantic_target_id} "
            "template SHA256 mismatch"
        )

    return {
        "target_length": len(sequence),
        "target_sequence_sha256": _sha256_text(sequence),
        "target_unpaired_msa_sha256": target_unpaired_msa_sha256,
        "target_template_sha256": target_template_sha256,
    }


def _validate_antibody_multitarget_inputs(
    inputs: dict[str, Input],
    *,
    antibody_chain_id: str,
    semantic_protomers: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    allowed_semantic_states = {"A", "AB", "AC"}
    state_reports: dict[str, dict[str, Any]] = {}
    antibody_sequences: dict[str, str] = {}
    antibody_expectation = (
        _semantic_protomer_expectation(semantic_protomers, antibody_chain_id)
        if semantic_protomers is not None
        else None
    )
    for oligo_name, input_obj in inputs.items():
        if not isinstance(input_obj, Input):
            raise TypeError(
                "pre_eval_hooks.af3_antibody_msa_cache expects hallucination.core.folding_input.Input values"
            )
        oligo_text = str(oligo_name)
        if oligo_text not in allowed_semantic_states:
            raise ValueError(
                "pre_eval_hooks.af3_antibody_msa_cache only supports semantic states A, AB and AC"
            )
        if str(input_obj.name) != oligo_text:
            raise ValueError(
                "pre_eval_hooks.af3_antibody_msa_cache "
                f"input.name must match semantic state key {oligo_text}, got {input_obj.name!r}"
            )
        semantic_chain_ids = list(oligo_text)
        physical_chain_ids = [antibody_chain_id] if oligo_text == "A" else [antibody_chain_id, "B"]
        if semantic_chain_ids[0] != antibody_chain_id:
            raise ValueError(
                "pre_eval_hooks.af3_antibody_msa_cache semantic states must start with antibody A"
            )
        chains = list(getattr(input_obj, "chains", ()) or ())
        chain_ids = [str(getattr(chain, "id", "") or "") for chain in chains]
        if len(chain_ids) != len(set(chain_ids)):
            raise ValueError(
                f"pre_eval_hooks.af3_antibody_msa_cache oligo={oligo_name} has duplicate chain ids: {chain_ids}"
            )
        if len(chains) != len(physical_chain_ids):
            chain_count = "one" if oligo_text == "A" else "two"
            raise ValueError(
                f"pre_eval_hooks.af3_antibody_msa_cache oligo={oligo_name} must contain exactly {chain_count} protein chains"
            )
        if chain_ids != physical_chain_ids:
            raise ValueError(
                f"pre_eval_hooks.af3_antibody_msa_cache oligo={oligo_name} physical chain order must be {physical_chain_ids}, got {chain_ids}"
            )
        if not all(isinstance(chain, ProteinChain) for chain in chains):
            raise TypeError(
                f"pre_eval_hooks.af3_antibody_msa_cache oligo={oligo_name} expects protein chains only"
            )
        antibody_chain = chains[0]
        antibody_sequence = str(antibody_chain.sequence)
        if antibody_expectation is not None:
            antibody_sequence = _validate_antibody_chain_a_sequence(
                oligo_name=oligo_text,
                chain=antibody_chain,
                expected=antibody_expectation,
            )
        antibody_sequences[oligo_text] = antibody_sequence
        if oligo_text == "A":
            state_reports[oligo_text] = {
                "action": "semantic_state_validation",
                "semantic_state": oligo_text,
                "semantic_chain_ids": semantic_chain_ids,
                "physical_chain_ids": chain_ids,
                "physical_to_semantic_mapping": {antibody_chain_id: antibody_chain_id},
            }
            continue
        target_chain = chains[1]
        semantic_target_id = semantic_chain_ids[1]
        target_payload_report: dict[str, Any]
        if semantic_protomers is not None:
            target_payload_report = _validate_semantic_target_payload(
                oligo_name=oligo_text,
                target_chain=target_chain,
                semantic_target_id=semantic_target_id,
                expected=_semantic_protomer_expectation(
                    semantic_protomers,
                    semantic_target_id,
                ),
            )
        else:
            target_payload_report = {
                "target_length": len(str(target_chain.sequence)),
                "target_sequence_sha256": _sha256_text(str(target_chain.sequence)),
                "target_unpaired_msa_sha256": _sha256_text(target_chain.unpaired_msa),
                "target_template_sha256": _sha256_templates(
                    target_chain.templates,
                    sequence=str(target_chain.sequence),
                ),
            }
        state_reports[oligo_text] = {
            "action": "semantic_state_validation",
            "semantic_state": oligo_text,
            "semantic_chain_ids": semantic_chain_ids,
            "physical_chain_ids": chain_ids,
            "physical_to_semantic_mapping": {
                antibody_chain_id: antibody_chain_id,
                "B": semantic_target_id,
            },
            "semantic_target_id": semantic_target_id,
            "target_physical_chain_id": "B",
            **target_payload_report,
        }

    if len(set(antibody_sequences.values())) > 1:
        raise ValueError(
            "pre_eval_hooks.af3_antibody_msa_cache A/AB/AC antibody A sequences must be identical"
        )
    return state_reports


def _default_cache_dir(output_dir: Path) -> Path:
    return Path(output_dir).parent / "af3_antibody_msa_cache"


def _cache_dir_from_config(raw_config: Any, *, output_dir: Path) -> Path:
    cfg = _to_plain_mapping(raw_config)
    raw = cfg.get("cache_dir")
    if raw in (None, "", "null", "None"):
        return _default_cache_dir(output_dir)
    return Path(os.path.expandvars(os.path.expanduser(str(raw)))).resolve()


def _path_from_config(value: Any, *, field_name: str, must_exist: bool) -> Path | None:
    if value in (None, "", "null", "None"):
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(
            f"pre_eval_hooks.af3_antibody_msa_cache.{field_name} does not exist: {path}"
        )
    return path


_AF3_TURBOAB_REQUIRED_DB_FILES = (
    "uniref90_2022_05.fa",
    "bfd-first_non_consensus_sequences.fasta",
    "mgy_clusters_2022_05.fa",
    "uniprot_all_2021_04.fa",
    "pdb_seqres_2022_09_28.fasta",
)

_AF3_TURBOAB_DB_SUBDIRS = {
    "vhh": ("VHH",),
    "nb": ("VHH",),
    "nanobody": ("VHH",),
    "igg": ("IgG",),
    "fv": ("IgG",),
}


def _missing_af3_turboab_db_files(path: Path) -> list[str]:
    return [name for name in _AF3_TURBOAB_REQUIRED_DB_FILES if not (path / name).is_file()]


def _resolve_af3_turboab_db_dir(raw_path: Any, *, subset: Any) -> Path | None:
    root = _path_from_config(
        raw_path,
        field_name="af3_turboab_db_dir",
        must_exist=True,
    )
    if root is None:
        return None
    if not _missing_af3_turboab_db_files(root):
        return root

    subset_key = str(subset or "VHH").strip().lower()
    candidates = _AF3_TURBOAB_DB_SUBDIRS.get(subset_key)
    if candidates is None:
        raise ValueError(
            "pre_eval_hooks.af3_antibody_msa_cache.af3_turboab_db_subset must be one of VHH, IgG, Nb, nanobody, or Fv"
        )
    for name in candidates:
        candidate = root / name
        if candidate.is_dir() and not _missing_af3_turboab_db_files(candidate):
            return candidate

    missing = _missing_af3_turboab_db_files(root)
    checked = ", ".join(str(root / name) for name in candidates)
    raise FileNotFoundError(
        "AF3-TurboAb custom MSA database is missing required files "
        f"{missing} under {root}; also checked subset directories: {checked}. "
        "Use the Zenodo AF3TuboAb/AF3_customMSA/VHH or IgG directory, or pass "
        "af3_turboab_db_dir to the AF3_customMSA root with af3_turboab_db_subset."
    )


def _as_shell_command(value: Any) -> list[str] | None:
    if value in (None, "", "null", "None"):
        return None
    if isinstance(value, str):
        return shlex.split(value)
    return [str(part) for part in value]


def _chain_to_af3_sequence(chain: Any) -> dict[str, Any]:
    chain_id = str(getattr(chain, "id", "") or "")
    if isinstance(chain, ProteinChain):
        payload: dict[str, Any] = {
            "id": chain_id,
            "sequence": str(chain.sequence),
            "pairedMsa": str(chain.paired_msa or ""),
        }
        if getattr(chain, "ptms", None):
            payload["modifications"] = [
                {"ptmType": str(ptm), "ptmPosition": int(pos)}
                for ptm, pos in (chain.ptms or [])
            ]
        if getattr(chain, "description", None):
            payload["description"] = chain.description
        if _has_text(chain.unpaired_msa):
            payload["unpairedMsa"] = chain.unpaired_msa
        if getattr(chain, "templates", None):
            payload["templates"] = [
                _template_to_json(t, sequence=str(chain.sequence))
                for t in chain.templates or []
            ]
        return {"protein": payload}
    if isinstance(chain, RnaChain):
        payload = {
            "id": chain_id,
            "sequence": str(chain.sequence),
        }
        if getattr(chain, "modifications", None):
            payload["modifications"] = [
                {"modificationType": str(mod), "basePosition": int(pos)}
                for mod, pos in (chain.modifications or [])
            ]
        if _has_text(chain.unpaired_msa):
            payload["unpairedMsa"] = chain.unpaired_msa
        if _has_text(chain.paired_msa):
            payload["pairedMsa"] = chain.paired_msa
        return {"rna": payload}
    if isinstance(chain, DnaChain):
        payload = {
            "id": chain_id,
            "sequence": str(chain.sequence),
        }
        if getattr(chain, "modifications", None):
            payload["modifications"] = [
                {"modificationType": str(mod), "basePosition": int(pos)}
                for mod, pos in (chain.modifications or [])
            ]
        if _has_text(chain.unpaired_msa):
            payload["unpairedMsa"] = chain.unpaired_msa
        if _has_text(chain.paired_msa):
            payload["pairedMsa"] = chain.paired_msa
        return {"dna": payload}
    if isinstance(chain, Ligand):
        payload = {"id": chain_id}
        if chain.smiles is not None:
            payload["smiles"] = chain.smiles
        elif chain.ccd_ids is not None:
            payload["ccdCodes"] = list(chain.ccd_ids)
        else:
            raise ValueError(f"Ligand chain {chain_id!r} must have smiles or ccd_ids")
        return {"ligand": payload}
    raise TypeError(f"Unsupported AF3 chain type for data pipeline JSON: {type(chain).__name__}")


def af3_input_to_json_payload(input_obj: Input, *, name: str) -> dict[str, Any]:
    """Serialize the exact post-hook AF3 input for audit and replay."""
    return {
        "dialect": "alphafold3",
        "version": 1,
        "name": str(name),
        "modelSeeds": [int(seed) for seed in (input_obj.rng_seeds or [1])],
        "sequences": [_chain_to_af3_sequence(chain) for chain in input_obj.chains],
        "bondedAtomPairs": input_obj.bonded_atom_pairs,
        "userCCD": input_obj.user_ccd,
    }


def _single_chain_af3_json_payload(
    input_obj: Input,
    *,
    chain: ProteinChain,
    name: str,
) -> dict[str, Any]:
    payload = af3_input_to_json_payload(
        dataclasses.replace(input_obj, chains=[chain], bonded_atom_pairs=None),
        name=name,
    )
    return payload


def _safe_job_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return safe or "af3_antibody_msa"


def _format_command_template(template: list[str], *, json_path: Path, output_dir: Path, db_dir: Path) -> list[str]:
    values = {
        "json_path": str(json_path),
        "json_dir": str(json_path.parent),
        "json_name": json_path.name,
        "output_dir": str(output_dir),
        "db_dir": str(db_dir),
    }
    return [part.format(**values) for part in template]


def _candidate_data_json_paths(run_dir: Path, *, job_name: str, oligo_name: str) -> list[Path]:
    stems = [
        _safe_job_name(job_name),
        _safe_job_name(oligo_name),
        _safe_job_name(job_name).lower(),
        _safe_job_name(oligo_name).lower(),
    ]
    paths: list[Path] = []
    for stem in stems:
        paths.extend(
            [
                run_dir / f"{stem}_data.json",
                run_dir / stem / f"{stem}_data.json",
                run_dir / stem.lower() / f"{stem.lower()}_data.json",
            ]
        )
    paths.extend(sorted(run_dir.glob("*_data.json")))
    paths.extend(sorted(run_dir.glob("*/**/*_data.json")))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _load_entry_from_af3_data_json(
    data_path: Path,
    *,
    chain_id: str,
    sequence: str,
) -> MsaCacheEntry:
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    _polymer_key, chain_data = load_af3_chain_data(payload, chain_id)
    if str(chain_data.get("sequence") or "") != str(sequence):
        raise ValueError(
            f"AF3 data-pipeline output {data_path} chain {chain_id!r} sequence does not match requested sequence"
        )
    entry = MsaCacheEntry(
        chain_id=str(chain_id),
        sequence=str(sequence),
        unpaired_msa=chain_data.get("unpairedMsa"),
        paired_msa=None,
        templates=tuple(
            _template_from_json(t, sequence=sequence)
            for t in list(chain_data.get("templates") or [])
            if isinstance(t, dict)
        ),
        path=data_path,
    )
    if not (entry.unpaired_msa or entry.paired_msa or entry.templates):
        raise ValueError(
            f"AF3 data-pipeline output {data_path} contains no MSA/template payload for chain {chain_id!r}"
        )
    if not _has_text(entry.unpaired_msa):
        raise ValueError(
            f"AF3 data-pipeline output {data_path} contains no unpaired MSA payload for chain {chain_id!r}"
        )
    return entry


def _entry_has_unpaired_msa(entry: MsaCacheEntry) -> bool:
    return _has_text(entry.unpaired_msa)


@dataclasses.dataclass(frozen=True, slots=True)
class Af3TurboAbDataPipeline:
    """Runs AF3 data-pipeline with the AF3-TurboAb custom MSA database."""

    db_dir: Path
    command_template: tuple[str, ...]
    work_dir: Path

    @classmethod
    def from_config(cls, cfg: dict[str, Any], *, output_dir: Path) -> "Af3TurboAbDataPipeline | None":
        command = _as_shell_command(cfg.get("data_pipeline_command"))
        db_dir = _resolve_af3_turboab_db_dir(
            cfg.get("af3_turboab_db_dir") or cfg.get("custom_msa_db_dir"),
            subset=cfg.get("af3_turboab_db_subset"),
        )
        if command is None and db_dir is None:
            return None
        if command is None:
            raise ValueError(
                "pre_eval_hooks.af3_antibody_msa_cache.data_pipeline_command is required when using AF3-TurboAb MSA data"
            )
        if db_dir is None:
            raise ValueError(
                "pre_eval_hooks.af3_antibody_msa_cache.af3_turboab_db_dir is required when data_pipeline_command is set"
            )
        work_dir = _path_from_config(
            cfg.get("data_pipeline_work_dir"),
            field_name="data_pipeline_work_dir",
            must_exist=False,
        )
        if work_dir is None:
            work_dir = Path(output_dir).parent / "af3_antibody_msa_pipeline"
        return cls(
            db_dir=db_dir,
            command_template=tuple(command),
            work_dir=work_dir,
        )

    def materialize(
        self,
        *,
        oligo_name: str,
        input_obj: Input,
        chain: ProteinChain,
        chain_id: str,
        sequence: str,
    ) -> MsaCacheEntry:
        job_name = _safe_job_name(f"{oligo_name}_{chain_id}_{_entry_id(chain_id=chain_id, sequence=sequence)}")
        run_dir = self.work_dir / job_name
        input_dir = run_dir / "input"
        output_dir = run_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = input_dir / f"{job_name}.json"
        json_path.write_text(
            json.dumps(
                _single_chain_af3_json_payload(
                    input_obj,
                    chain=chain,
                    name=job_name,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

        command = _format_command_template(
            list(self.command_template),
            json_path=json_path,
            output_dir=output_dir,
            db_dir=self.db_dir,
        )
        logging.info(
            "pre_eval af3_antibody_msa_cache running AF3-TurboAb data pipeline oligo=%s chain=%s db_dir=%s",
            oligo_name,
            chain_id,
            self.db_dir,
        )
        subprocess.run(command, cwd=run_dir, check=True)
        for data_path in _candidate_data_json_paths(output_dir, job_name=job_name, oligo_name=oligo_name):
            if not data_path.is_file():
                continue
            try:
                return _load_entry_from_af3_data_json(
                    data_path,
                    chain_id=chain_id,
                    sequence=sequence,
                )
            except Exception as exc:
                logging.debug("Ignoring AF3 data-pipeline candidate %s: %s", data_path, exc)
        raise FileNotFoundError(
            f"AF3-TurboAb data-pipeline command completed but no usable *_data.json for chain {chain_id!r} was found under {output_dir}"
        )


def apply_af3_antibody_msa_cache_hook(
    inputs: dict[str, Input],
    raw_config: Any,
    *,
    output_dir: Path,
    cache: AntibodyMsaCache | None = None,
    semantic_protomers: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Input], dict[str, list[dict[str, Any]]], AntibodyMsaCache | None]:
    """Reuse chain A AF3 MSA/template payloads for monomer A and complex AB/AC inputs."""
    if not _is_enabled(raw_config):
        return inputs, {}, cache

    cfg = _to_plain_mapping(raw_config)
    max_distance = int(cfg.get("max_residue_differences", 4))
    if max_distance < 0:
        raise ValueError("pre_eval_hooks.af3_antibody_msa_cache.max_residue_differences must be >= 0")
    antibody_chain_id = _selected_chain_id(cfg)
    if semantic_protomers is None:
        raise ValueError(
            "pre_eval_hooks.af3_antibody_msa_cache requires semantic protomer expectations"
        )
    state_reports = _validate_antibody_multitarget_inputs(
        inputs,
        antibody_chain_id=antibody_chain_id,
        semantic_protomers=semantic_protomers,
    )
    overwrite_existing = bool(cfg.get("overwrite_existing", False))
    require_payload = bool(cfg.get("require_msa_payload", False))
    clear_paired_msa = bool(cfg.get("clear_paired_msa", True))
    preserve_existing_templates = bool(cfg.get("preserve_existing_templates", False))
    raw_expected_template_count = cfg.get("expected_template_count")
    expected_template_count = (
        None
        if raw_expected_template_count in (None, "", "null", "None")
        else int(raw_expected_template_count)
    )
    if expected_template_count is not None and expected_template_count < 0:
        raise ValueError(
            "pre_eval_hooks.af3_antibody_msa_cache.expected_template_count must be >= 0"
        )
    minimum_unpaired_msa_sequences = int(
        cfg.get("minimum_unpaired_msa_sequences", 0)
    )
    if minimum_unpaired_msa_sequences < 0:
        raise ValueError(
            "pre_eval_hooks.af3_antibody_msa_cache.minimum_unpaired_msa_sequences must be >= 0"
        )
    require_unpaired_msa = bool(
        require_payload or minimum_unpaired_msa_sequences > 0
    )

    working_inputs = _clear_all_paired_msa(inputs) if clear_paired_msa else inputs

    if cache is None:
        cache = AntibodyMsaCache(cache_dir=_cache_dir_from_config(cfg, output_dir=output_dir))
    cache.load()
    data_pipeline = Af3TurboAbDataPipeline.from_config(cfg, output_dir=output_dir)

    updated = dict(working_inputs)
    reports: dict[str, list[dict[str, Any]]] = {}

    for oligo_name, input_obj in working_inputs.items():
        oligo_reports: list[dict[str, Any]] = []
        current_input = updated[oligo_name]
        chain = current_input.chains[0]
        if not isinstance(chain, ProteinChain):
            raise TypeError(
                f"pre_eval_hooks.af3_antibody_msa_cache oligo={oligo_name} chain A must be a protein chain"
            )

        chain_id = str(getattr(chain, "id", "") or "")
        sequence = str(chain.sequence)
        has_payload = _has_payload(chain)
        has_unpaired_msa = _has_unpaired_msa(chain)
        if (
            has_payload
            and not overwrite_existing
            and (has_unpaired_msa or not require_unpaired_msa)
        ):
            entry = cache.add_from_chain(chain_id=chain_id, chain=chain)
            oligo_reports.append(
                {
                    "chain_id": chain_id,
                    "action": "seed",
                    "sequence": sequence,
                    "entry_id": entry.id if entry is not None else None,
                }
            )
            reports[str(oligo_name)] = oligo_reports
            continue

        hit = cache.find_nearest(
            chain_id=chain_id,
            sequence=sequence,
            max_distance=max_distance,
        )
        if hit is not None and require_unpaired_msa and not _entry_has_unpaired_msa(hit[0]):
            hit = None
        if hit is None:
            if data_pipeline is not None:
                entry = data_pipeline.materialize(
                    oligo_name=str(oligo_name),
                    input_obj=current_input,
                    chain=chain,
                    chain_id=chain_id,
                    sequence=sequence,
                )
                cache.add_entry(entry)
                replacement = _replace_chain_payload(
                    chain,
                    entry,
                    preserve_existing_templates=preserve_existing_templates,
                    expected_template_count=expected_template_count,
                )
                updated[oligo_name] = _replace_input_chain(current_input, 0, replacement)
                action = "materialize"
                entry_id = entry.id
            elif has_payload and (has_unpaired_msa or not require_unpaired_msa):
                entry = cache.add_from_chain(chain_id=chain_id, chain=chain)
                action = "seed"
                entry_id = entry.id if entry is not None else None
            else:
                if require_unpaired_msa:
                    if has_payload and not has_unpaired_msa:
                        raise RuntimeError(
                            f"Existing AF3 antibody payload for oligo={oligo_name} chain={chain_id} has no unpaired MSA and no data pipeline is configured"
                        )
                    raise RuntimeError(
                        f"No cached AF3 antibody MSA payload for oligo={oligo_name} chain={chain_id} and no data pipeline is configured"
                    )
                action = "miss"
                entry_id = None
            oligo_reports.append(
                {
                    "chain_id": chain_id,
                    "action": action,
                    "sequence": sequence,
                    "entry_id": entry_id,
                    **(
                        {"source": "af3_turboab_data_pipeline"}
                        if action == "materialize"
                        else {}
                    ),
                }
            )
            reports[str(oligo_name)] = oligo_reports
            continue

        entry, distance = hit
        replacement = _replace_chain_payload(
            chain,
            entry,
            preserve_existing_templates=preserve_existing_templates,
            expected_template_count=expected_template_count,
        )
        updated[oligo_name] = _replace_input_chain(current_input, 0, replacement)
        oligo_reports.append(
            {
                "chain_id": chain_id,
                "action": "hit",
                "sequence": sequence,
                "entry_id": entry.id,
                "distance": distance,
            }
        )
        logging.info(
            "pre_eval af3_antibody_msa_cache oligo=%s chain=%s hit distance=%d entry=%s",
            oligo_name,
            chain_id,
            distance,
            entry.id,
        )
        reports[str(oligo_name)] = oligo_reports

    _validate_final_chain_payloads(
        updated,
        expected_template_count=expected_template_count,
        minimum_unpaired_msa_sequences=minimum_unpaired_msa_sequences,
    )
    for oligo_name, state_report in state_reports.items():
        reports.setdefault(oligo_name, []).append(state_report)
    return updated, reports, cache
