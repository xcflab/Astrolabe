from __future__ import annotations
from functools import partial, lru_cache
from io import StringIO
import pathlib
from typing import Any, Dict, List, Mapping, Tuple, Literal

import numpy as np
try:
    from Bio import SeqIO  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SeqIO = None
from omegaconf import DictConfig, ListConfig, OmegaConf

from astrolabe.core.folding_input import Template

SequenceDict = Dict[str, str]
IndexDict = Dict[str, List[int]]
AF3_POLYMER_KEYS = ("protein", "rna", "dna")


@lru_cache(maxsize=4)
def parse_json(file):
    import json

    with open(file, "r") as f:
        data = json.load(f)
    return data


def _iter_af3_polymer_chain_dicts(data):
    for item in data.get("sequences", []) or []:
        if not isinstance(item, dict):
            continue
        for polymer_key in AF3_POLYMER_KEYS:
            chain_data = item.get(polymer_key)
            if isinstance(chain_data, dict):
                yield polymer_key, chain_data


def _af3_chain_matches(chain_data: dict, chain_id: str) -> bool:
    ids = chain_data.get("id")
    if isinstance(ids, str):
        return ids == chain_id
    if isinstance(ids, list):
        return chain_id in ids
    return False


def load_af3_chain_data(data: dict, chain_id: str) -> tuple[str, dict]:
    for polymer_key, chain_data in _iter_af3_polymer_chain_dicts(data):
        if _af3_chain_matches(chain_data, chain_id):
            return polymer_key, chain_data
    raise ValueError(f"Chain {chain_id!r} not found in AF3 JSON")


def _to_plain_data(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _is_inline_msa_string(value: Any) -> bool:
    text = str(value or "")
    return text.lstrip().startswith(">")


def _is_antibody_framework_homology_spec(value: Any) -> bool:
    plain = _to_plain_data(value)
    return isinstance(plain, dict) and str(plain.get("_type", "")).strip() == "antibody_framework_homology"


def _normalize_antibody_layout(layout_cfg: Any) -> dict[str, Any]:
    plain = _to_plain_data(layout_cfg)
    if not isinstance(plain, dict):
        raise ValueError("antibody framework homology layout must be a mapping")
    fixed_segments = [str(part) for part in plain.get("fixed_segments") or []]
    loop_order = [str(loop) for loop in plain.get("loop_order") or []]
    source_loop_lengths_raw = plain.get("source_loop_lengths") or {}
    source_loop_lengths = {
        str(loop): int(source_loop_lengths_raw[loop]) for loop in loop_order
    }
    if len(fixed_segments) != len(loop_order) + 1:
        raise ValueError(
            "antibody framework homology layout requires len(fixed_segments) == len(loop_order) + 1"
        )
    return {
        "fixed_segments": fixed_segments,
        "loop_order": loop_order,
        "source_loop_lengths": source_loop_lengths,
    }


def _build_layout_slices(
    fixed_segments: List[str],
    loop_order: List[str],
    loop_lengths: Mapping[str, int],
) -> dict[str, Any]:
    fixed_slices = []
    loop_slices = {}
    cursor = 0
    for idx, fixed_segment in enumerate(fixed_segments[:-1]):
        fixed_len = len(fixed_segment)
        fixed_slices.append((cursor, cursor + fixed_len))
        cursor += fixed_len
        loop_label = loop_order[idx]
        loop_len = int(loop_lengths[loop_label])
        loop_slices[loop_label] = (cursor, cursor + loop_len)
        cursor += loop_len
    fixed_slices.append((cursor, cursor + len(fixed_segments[-1])))
    cursor += len(fixed_segments[-1])
    return {
        "fixed_slices": fixed_slices,
        "loop_slices": loop_slices,
        "total_length": cursor,
    }


def _derive_target_loop_lengths(
    current_seq: str,
    fixed_residues: List[int],
    fixed_segments: List[str],
    loop_order: List[str],
) -> dict[str, int]:
    fixed_set = {int(pos) for pos in fixed_residues}
    target_lengths = {}
    cursor = 0
    for idx, fixed_segment in enumerate(fixed_segments[:-1]):
        next_cursor = cursor + len(fixed_segment)
        if current_seq[cursor:next_cursor] != fixed_segment:
            raise ValueError(
                f"Current antibody sequence does not match fixed framework segment {idx} "
                f"(expected {fixed_segment!r}, got {current_seq[cursor:next_cursor]!r})"
            )
        expected_positions = set(range(cursor, next_cursor))
        if not expected_positions.issubset(fixed_set):
            raise ValueError(
                f"Current fixed residues do not cover framework segment {idx}"
            )
        cursor = next_cursor
        loop_len = 0
        while cursor + loop_len < len(current_seq) and (cursor + loop_len) not in fixed_set:
            loop_len += 1
        target_lengths[loop_order[idx]] = loop_len
        cursor += loop_len

    tail = fixed_segments[-1]
    if current_seq[cursor:cursor + len(tail)] != tail:
        raise ValueError(
            f"Current antibody sequence does not match final fixed framework segment "
            f"(expected {tail!r}, got {current_seq[cursor:cursor + len(tail)]!r})"
        )
    expected_tail_positions = set(range(cursor, cursor + len(tail)))
    if not expected_tail_positions.issubset(fixed_set):
        raise ValueError("Current fixed residues do not cover final framework segment")
    if cursor + len(tail) != len(current_seq):
        raise ValueError("Current antibody sequence has extra residues after final framework segment")
    return target_lengths


def _load_af3_chain_raw(homology_arg: str) -> dict[str, Any]:
    file_str, *chain_parts = str(homology_arg or "").split(":")
    if not file_str.endswith(".json"):
        raise ValueError(f"Not an AF3 JSON homology: {homology_arg!r}")
    if not chain_parts:
        raise ValueError(f"Chain ID must be specified for JSON homology files: {homology_arg!r}")
    chain_id = str(chain_parts[0]).strip()
    data = parse_json(file_str)
    if "dialect" not in data:
        raise ValueError(f"{file_str} is not a valid AlphaFold3 JSON file")
    _polymer_key, chain_data = load_af3_chain_data(data, chain_id)
    return {
        "sequence": chain_data.get("sequence"),
        "unpairedMsa": chain_data.get("unpairedMsa"),
        "pairedMsa": chain_data.get("pairedMsa"),
        "templates": list(chain_data.get("templates") or []),
    }


def _resolve_framework_homology_path(
    path_str: str,
    *,
    base_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    path = pathlib.Path(str(path_str).strip()).expanduser()
    if not path.is_absolute():
        root = pathlib.Path.cwd() if base_dir is None else pathlib.Path(base_dir)
        path = root / path
    return path.resolve()


def _resolve_framework_homology_msa_value(
    value: Any,
    *,
    base_dir: pathlib.Path | None = None,
) -> str | None:
    plain = _to_plain_data(value)
    if plain is None:
        return None
    if not isinstance(plain, str):
        raise ValueError("Framework homology MSA source must be a string or null")
    text = plain.strip()
    if text == "":
        return None
    if _is_inline_msa_string(text):
        return plain
    path = _resolve_framework_homology_path(text, base_dir=base_dir)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".a3m":
        raise ValueError(f"Unsupported framework homology MSA file format: {path}")
    return path.read_text(encoding="utf-8")


def _load_framework_homology_source(
    source: Any,
    *,
    base_dir: pathlib.Path | None = None,
) -> dict[str, Any]:
    plain = _to_plain_data(source)
    if isinstance(plain, str):
        text = str(plain).strip()
        if ":" in text and text.split(":", 1)[0].endswith(".json"):
            file_str, chain = text.split(":", 1)
            path = _resolve_framework_homology_path(file_str, base_dir=base_dir)
            return _load_af3_chain_raw(f"{path}:{chain}")
        path = _resolve_framework_homology_path(text, base_dir=base_dir)
        if not path.is_file():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".a3m":
            return {"unpairedMsa": path.read_text(encoding="utf-8")}
        if suffix in {".yaml", ".yml", ".json"}:
            loaded = _to_plain_data(OmegaConf.load(path))
            return _load_framework_homology_source(loaded, base_dir=path.parent)
        raise ValueError(f"Unsupported framework homology file format: {path}")
    if isinstance(plain, list):
        return {"templates": list(plain)}
    if isinstance(plain, dict):
        if "dialect" in plain and "sequences" in plain:
            raise ValueError(
                "AF3 JSON framework homology must use '.json:<chain>' form, not a raw JSON mapping"
            )
        result = {}
        if "unpairedMsa" in plain:
            result["unpairedMsa"] = _resolve_framework_homology_msa_value(
                plain["unpairedMsa"],
                base_dir=base_dir,
            )
        if "pairedMsa" in plain:
            result["pairedMsa"] = _resolve_framework_homology_msa_value(
                plain["pairedMsa"],
                base_dir=base_dir,
            )
        if "templates" in plain:
            result["templates"] = _to_plain_data(plain["templates"])
        if result:
            return result
    raise ValueError("Unsupported framework homology source")


def _transform_antibody_msa(
    *,
    current_seq: str,
    source_msa: str,
    fixed_segments: List[str],
    loop_order: List[str],
    source_loop_lengths: Mapping[str, int],
    target_loop_lengths: Mapping[str, int],
) -> str | None:
    if source_msa is None or str(source_msa).strip() == "":
        return None
    msa = parse_a3m_sequences(source_msa)
    if not msa:
        return None
    source_layout = _build_layout_slices(fixed_segments, loop_order, source_loop_lengths)
    source_total_length = int(source_layout["total_length"])
    query_seq = msa[0][1]
    if len(query_seq) != source_total_length:
        raise ValueError(
            f"Source antibody framework homology length mismatch: expected {source_total_length}, got {len(query_seq)}"
        )
    for idx, fixed_segment in enumerate(fixed_segments):
        seg_start, seg_end = source_layout["fixed_slices"][idx]
        if query_seq[seg_start:seg_end] != fixed_segment:
            raise ValueError(
                f"Source antibody framework homology fixed framework segment {idx} does not match source framework"
            )
    for loop_label in loop_order:
        for _, seq in msa:
            if len(seq) != source_total_length:
                raise ValueError(
                    f"Source antibody framework homology record length mismatch in loop {loop_label}"
                )

    rewritten = []
    for header, seq in msa:
        parts = []
        for idx, fixed_segment in enumerate(fixed_segments[:-1]):
            seg_start, seg_end = source_layout["fixed_slices"][idx]
            parts.append(seq[seg_start:seg_end])
            parts.append("-" * int(target_loop_lengths[loop_order[idx]]))
        last_start, last_end = source_layout["fixed_slices"][-1]
        parts.append(seq[last_start:last_end])
        rewritten.append((header, "".join(parts)))
    if rewritten:
        rewritten[0] = (rewritten[0][0], current_seq)
    return "\n".join([f">{header}\n{seq}" for header, seq in rewritten])


def _build_source_to_target_index_map(
    fixed_segments: List[str],
    loop_order: List[str],
    source_loop_lengths: Mapping[str, int],
    target_loop_lengths: Mapping[str, int],
) -> dict[int, int]:
    source_pos = 0
    target_pos = 0
    index_map = {}
    for idx, fixed_segment in enumerate(fixed_segments[:-1]):
        for offset in range(len(fixed_segment)):
            index_map[source_pos + offset] = target_pos + offset
        source_pos += len(fixed_segment)
        target_pos += len(fixed_segment)
        loop_label = loop_order[idx]
        source_pos += int(source_loop_lengths[loop_label])
        target_pos += int(target_loop_lengths[loop_label])
    for offset in range(len(fixed_segments[-1])):
        index_map[source_pos + offset] = target_pos + offset
    return index_map


def _transform_antibody_templates(
    *,
    raw_templates: Any,
    fixed_segments: List[str],
    loop_order: List[str],
    source_loop_lengths: Mapping[str, int],
    target_loop_lengths: Mapping[str, int],
) -> list[dict[str, Any]] | None:
    plain_templates = _to_plain_data(raw_templates) or []
    if not plain_templates:
        return None
    index_map = _build_source_to_target_index_map(
        fixed_segments,
        loop_order,
        source_loop_lengths,
        target_loop_lengths,
    )
    templates = []
    for raw_template in plain_templates:
        query_indices = list(raw_template.get("queryIndices") or [])
        template_indices = list(raw_template.get("templateIndices") or [])
        if len(query_indices) != len(template_indices):
            raise ValueError("Template queryIndices/templateIndices length mismatch")
        new_query = []
        new_template = []
        for query_index, template_index in zip(query_indices, template_indices):
            q = int(query_index)
            if q not in index_map:
                continue
            new_query.append(int(index_map[q]))
            new_template.append(int(template_index))
        if not new_query:
            continue
        templates.append(
            {
                "mmcif": raw_template.get("mmcif"),
                "queryIndices": new_query,
                "templateIndices": new_template,
            }
        )
    return templates or None


def _stretch_antibody_framework_homology(
    current_seq: str,
    fixed_residues: List[int],
    homologies: Any,
) -> dict[str, Any]:
    plain = _to_plain_data(homologies)
    layout = _normalize_antibody_layout(plain.get("layout"))
    fixed_segments = list(layout["fixed_segments"])
    loop_order = list(layout["loop_order"])
    source_loop_lengths = dict(layout["source_loop_lengths"])
    target_loop_lengths = _derive_target_loop_lengths(
        current_seq,
        fixed_residues,
        fixed_segments,
        loop_order,
    )
    source_data = _load_framework_homology_source(plain.get("source"))
    stretched = {}
    if "unpairedMsa" in source_data:
        stretched_msa = _transform_antibody_msa(
            current_seq=current_seq,
            source_msa=source_data.get("unpairedMsa"),
            fixed_segments=fixed_segments,
            loop_order=loop_order,
            source_loop_lengths=source_loop_lengths,
            target_loop_lengths=target_loop_lengths,
        )
        if stretched_msa is not None:
            stretched["unpairedMsa"] = stretched_msa
    if "pairedMsa" in source_data:
        stretched_msa = _transform_antibody_msa(
            current_seq=current_seq,
            source_msa=source_data.get("pairedMsa"),
            fixed_segments=fixed_segments,
            loop_order=loop_order,
            source_loop_lengths=source_loop_lengths,
            target_loop_lengths=target_loop_lengths,
        )
        if stretched_msa is not None:
            stretched["pairedMsa"] = stretched_msa
    if "templates" in source_data:
        stretched_templates = _transform_antibody_templates(
            raw_templates=source_data.get("templates"),
            fixed_segments=fixed_segments,
            loop_order=loop_order,
            source_loop_lengths=source_loop_lengths,
            target_loop_lengths=target_loop_lengths,
        )
        if stretched_templates is not None:
            stretched["templates"] = stretched_templates
    return stretched


def parse_homology_arg(
    sequence,
    fixed_residues,
    homologies: str | DictConfig | ListConfig | list,
    key: Literal["unpairedMsa", "pairedMsa", "templates"] | None = None,
    mask_mutable_msa: bool = True,
):
    if isinstance(homologies, (dict, DictConfig)) and "maskMutableResidues" in homologies:
        mask_mutable_msa = bool(homologies["maskMutableResidues"])
    retained_msa_positions = (
        fixed_residues if mask_mutable_msa else list(range(len(sequence)))
    )
    results = {}
    mask_msa = partial(
        mask_msa_fixpos,
        sequence,
        retained_msa_positions,
        remove_mainly_gaps=True,
    )
    msa_templates = partial(
        extract_template_fixpos,
        fixed_residues,
    )

    if _is_antibody_framework_homology_spec(homologies):
        stretched = _stretch_antibody_framework_homology(
            sequence,
            fixed_residues,
            homologies,
        )
        return parse_homology_arg(
            sequence,
            fixed_residues,
            stretched,
            key=key,
            mask_mutable_msa=mask_mutable_msa,
        )

    if isinstance(homologies, str):
        if key in {"unpairedMsa", "pairedMsa"} and _is_inline_msa_string(homologies):
            results[key] = mask_msa(homologies)
            return results
        file, *chain = homologies.split(":")
        if file.endswith(".json"):
            data = parse_json(file)
            if "dialect" not in data:
                raise ValueError(f"{file} is not a valid AlphaFold3 JSON file")
            if len(chain) == 0:
                raise ValueError("Chain ID must be specified for JSON homology files")
            chain = chain[0]
            _polymer_key, chain_data = load_af3_chain_data(data, chain)
            if key is not None:
                if key == "unpairedMsa" or key == "pairedMsa":
                    results[key] = mask_msa(chain_data.get(key, None))
                elif key == "templates":
                    results[key] = msa_templates(
                        chain_data.get("templates", [])
                    )
                else:
                    raise ValueError(f"Unsupported key: {key}")
                return results
            else:
                results["unpairedMsa"] = mask_msa(
                    chain_data.get("unpairedMsa", None)
                )
                results["pairedMsa"] = mask_msa(
                    chain_data.get("pairedMsa", None)
                )
                results["templates"] = msa_templates(
                    chain_data.get("templates", [])
                )
                return results
        elif file.endswith(".a3m"):
            with open(file, "r") as f:
                msa_string = f.read()
            results["unpairedMsa"] = mask_msa(msa_string)
            return results
        else:
            raise ValueError(f"Unsupported homology file format: {file}")
    elif isinstance(homologies, (list, ListConfig)):
        # Support direct AF3-style template entries, e.g.:
        #   homologies:
        #     B:
        #       templates:
        #         - mmcif: "<mmcif string>"
        #           queryIndices: [0, 1, 2, ...]
        #           templateIndices: [0, 1, 2, ...]
        raw_templates = list(homologies)
        if key is not None and key != "templates":
            raise ValueError(f"List homology input only supports templates, got key={key}")
        results["templates"] = msa_templates(raw_templates)
        return results
    elif isinstance(homologies, (dict, DictConfig)):
        if "unpairedMsa" in homologies:
            value = homologies["unpairedMsa"]
            if isinstance(value, str) and _is_inline_msa_string(value):
                results["unpairedMsa"] = mask_msa(value)
            else:
                results.update(
                    parse_homology_arg(
                        sequence,
                        fixed_residues,
                        value,
                        key="unpairedMsa",
                        mask_mutable_msa=mask_mutable_msa,
                    )
                )
        if "pairedMsa" in homologies:
            value = homologies["pairedMsa"]
            if isinstance(value, str) and _is_inline_msa_string(value):
                results["pairedMsa"] = mask_msa(value)
            else:
                results.update(
                    parse_homology_arg(
                        sequence,
                        fixed_residues,
                        value,
                        key="pairedMsa",
                        mask_mutable_msa=mask_mutable_msa,
                    )
                )
        if "templates" in homologies:
            value = homologies["templates"]
            if value is None:
                return results
            results.update(
                parse_homology_arg(
                    sequence,
                    fixed_residues,
                    value,
                    key="templates",
                    mask_mutable_msa=mask_mutable_msa,
                )
            )
        return results


def extract_template_fixpos(fix_res, raw_templates):
    if not raw_templates:
        templates = None
    else:
        templates = []
        for raw_template in raw_templates:
            mmcif = raw_template.get("mmcif", None)
            query_to_template_map = dict(
                zip(
                    raw_template["queryIndices"],
                    raw_template["templateIndices"],
                )
            )
            include_mutable = bool(raw_template.get("includeMutableResidues", False))
            if not include_mutable:
                query_to_template_map = {
                    k: v for k, v in query_to_template_map.items() if k in fix_res
                }
            templates.append(
                Template(mmcif=mmcif, query_to_template_map=query_to_template_map)
            )
    return templates


def mask_msa_fixpos(current_seq, fix_res, msa_string, remove_mainly_gaps=False):
    if msa_string is None or msa_string.strip() == "":
        return None
    msa = parse_a3m_sequences(msa_string)
    headers, msa_seqs = zip(*msa)
    if len(current_seq) != len(msa_seqs[0]):
        raise ValueError(
            f"Sequence length mismatch: expected {len(current_seq)}, got {len(msa_seqs[0])}"
        )
    seqs = apply_position_mask(msa_seqs, fix_res)
    seqs[0] = current_seq
    if remove_mainly_gaps:
        seqs = [seq for seq in seqs if seq.count("-") / len(current_seq) < 0.7]
    masked_msa = "\n".join([f">{header}\n{seq}" for header, seq in zip(headers, seqs)])
    return masked_msa


def parse_a3m_sequences(msa_string: str) -> List[Tuple[str, str]]:
    """
    Parse sequences from an A3M alignment file, removing insertion characters.

    Args:
        msa_string: Path to the A3M file

    Returns:
        List of tuples containing (sequence_id, aligned_sequence)
    """
    if SeqIO is None:
        raise RuntimeError(
            "Biopython is required to parse A3M sequences (`pip install biopython`)."
        )
    aligned_sequences = []
    for record in SeqIO.parse(StringIO(msa_string), "fasta"):
        # Filter out lowercase insertion characters while keeping gaps
        clean_seq = "".join(
            char if char.isupper() or char == "-" else "" for char in str(record.seq)
        )
        aligned_sequences.append((record.id, clean_seq))
    return aligned_sequences


def apply_position_mask(sequences: List[str], positions: List[int]) -> List[str]:
    """
    Mask sequences by keeping only specified positions, replacing others with gaps.

    Args:
        sequences: List of sequence strings to mask
        positions: List of positions to keep (0-based indexing)

    Returns:
        List of masked sequences
    """
    if len(positions) == 0:
        return list(sequences)
    seq_array = np.array([list(seq) for seq in sequences], dtype="U1")
    position_mask = np.zeros(seq_array.shape, dtype=bool)
    position_mask[:, positions] = True
    seq_array[~position_mask] = "-"
    return ["".join(seq) for seq in seq_array]
