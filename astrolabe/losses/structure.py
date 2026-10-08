import logging
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from typing import List, Optional

import Bio
import numpy as np
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB import PDBIO, PDBParser, MMCIFParser, Select
from astrolabe.utils.mmcif_ids import int_id_to_str_id

from astrolabe.core import global_config
from astrolabe.losses.base import loss_registry


def _looks_like_mkdssp(path: str) -> bool:
    name = pathlib.Path(str(path)).name
    return name == "mkdssp" or name.startswith("mkdssp_")


def _protein_only_pdb_for_dssp(file: str) -> tuple[str, Optional[str]]:
    """Return a protein-only PDB path for DSSP tools that reject ligands."""
    if not str(file).endswith(".pdb"):
        return file, None

    with open(file, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    if not any(line.startswith("HETATM") for line in lines):
        return file, None

    fd, tmp_path = tempfile.mkstemp(
        suffix=".protein_only.pdb",
        prefix=pathlib.Path(file).name + ".dssp.",
        dir=str(pathlib.Path(file).parent),
        text=True,
    )
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for line in lines:
            if line.startswith("ATOM") or line.startswith(
                ("TER", "END", "MODEL", "ENDMDL")
            ):
                handle.write(line)
        if not lines or not lines[-1].startswith("END"):
            handle.write("END\n")
    return tmp_path, tmp_path


class _ProteinChainSelect(Select):
    def __init__(self, chain_ids: set[str]):
        self.chain_ids = set(chain_ids)

    def accept_chain(self, chain):
        return chain.id in self.chain_ids

    def accept_residue(self, residue):
        return residue.id[0] == " " and is_aa(residue, standard=True)


def _write_protein_chain_subset_pdb(
    structure_path: pathlib.Path,
    chain_ids: list[str],
    output_path: pathlib.Path,
) -> None:
    if structure_path.suffix.lower() == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein_subset", str(structure_path))
    io = PDBIO()
    io.set_structure(structure)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(output_path), select=_ProteinChainSelect(set(chain_ids)))


def _dssp_list_for_chain_subset(path: pathlib.Path, chain_ids: list[str]) -> list[str]:
    if not chain_ids:
        return []
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        suffix=".protein_subset.pdb",
        prefix=path.name + ".",
        dir=str(path.parent),
    ) as tmp:
        tmp_path = pathlib.Path(tmp.name)
    try:
        _write_protein_chain_subset_pdb(path, chain_ids, tmp_path)
        return dssp_wrapper(str(tmp_path))
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _dssp_list_from_biotite_geometry(file):
    """Estimate coarse SSE from CA geometry when external DSSP is unavailable."""
    import biotite.structure as struc
    import biotite.structure.io as strucio

    atoms = strucio.load_structure(file)
    if hasattr(atoms, "stack_depth"):
        atoms = atoms[0]
    sse = struc.annotate_sse(atoms)
    dssp_list = []
    for sym in np.asarray(sse).tolist():
        sym = str(sym or "").lower()
        if sym == "a":
            dssp_list.append("H")
        elif sym == "b":
            dssp_list.append("E")
        else:
            dssp_list.append("L")
    return dssp_list


def dssp_wrapper(file):
    """Compute DSSP string on structure."""
    file = str(file)
    dssp_exec = getattr(global_config, "dssp_path", None)
    dssp_input = file
    dssp_tmp_file = None
    try:
        dssp_input, dssp_tmp_file = _protein_only_pdb_for_dssp(str(file))
    except Exception as exc:
        logging.warning("Failed to create protein-only DSSP input for %s: %s", file, exc)
    if dssp_exec and str(file).endswith((".pdb", ".cif")):
        try:
            import biotite.structure.io as strucio
            from biotite.application.dssp import DsspApp

            atoms = strucio.load_structure(dssp_input)
            sse = DsspApp.annotate_sse(atoms, bin_path=str(dssp_exec))
            result = [str(x) for x in np.asarray(sse).tolist()]
            if dssp_tmp_file is not None:
                os.remove(dssp_tmp_file)
                dssp_tmp_file = None
            return result
        except Exception as exc:
            logging.warning(
                "Biotite DSSP path failed for %s via %s: %s; falling back to legacy DSSP wrapper",
                file,
                dssp_exec,
                exc,
            )
    tmp_file = None
    if file.endswith(".cif"):
        from astrolabe.utils.mmcif_fallback import write_centered_pdb_from_mmcif

        cif_path = pathlib.Path(file)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            suffix=".pdb",
            prefix=cif_path.name + ".dssp_tmp.",
            dir=str(cif_path.parent),
        ) as tmp:
            tmp_file = tmp.name
        write_centered_pdb_from_mmcif(cif_path, pathlib.Path(tmp_file))
        file = tmp_file
    elif not file.endswith(".pdb"):
        raise ValueError("Unsupported file format: {}".format(file))

    if dssp_tmp_file is not None:
        os.remove(dssp_tmp_file)
        dssp_tmp_file = None
    try:
        dssp_input, dssp_tmp_file = _protein_only_pdb_for_dssp(str(file))
    except Exception as exc:
        logging.warning("Failed to create protein-only DSSP input for %s: %s", file, exc)
        dssp_input = file

    dssp_exec = getattr(global_config, "dssp_path", None)
    dssp_candidates = []
    if dssp_exec:
        cand = str(dssp_exec)
        dssp_candidates.append(cand)
        # Treat relative paths as repo-root relative (Hydra often chdirs into the
        # run directory, so relative paths would otherwise break).
        try:
            p = pathlib.Path(cand)
            if not p.is_absolute():
                repo_root = pathlib.Path(__file__).resolve().parents[2]
                dssp_candidates.append(str(repo_root / p))
        except Exception:
            pass
    for name in ("dssp", "mkdssp"):
        p = shutil.which(name)
        if p:
            dssp_candidates.append(p)
    dssp_candidates.extend(("/usr/bin/dssp", "/usr/bin/mkdssp"))

    dssp_bin = None
    for cand in dssp_candidates:
        if not cand:
            continue
        try:
            import pathlib

            if pathlib.Path(cand).exists():
                dssp_bin = cand
                break
        except Exception:
            continue

    if not dssp_bin:
        raise FileNotFoundError(
            "DSSP executable not found. Install dssp/mkdssp or set global_config.dssp_path."
        )

    # mkdssp has a different CLI from classic DSSP; some builds don't support
    # extra flags (e.g. --output-format). Use the mkdssp-native interface.
    if _looks_like_mkdssp(str(dssp_bin)):
        import importlib
        import warnings
        from io import StringIO

        dssp_mod = importlib.import_module("Bio.PDB.DSSP")

        # 'mkdssp -o stdout' is not reliably treated as stdout across builds;
        # write to a temp file and parse it.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            suffix=".dssp",
            prefix="mkdssp.",
        ) as tmp_out:
            out_path = tmp_out.name

        try:
            cmd = [str(dssp_bin), "-i", dssp_input, "-o", out_path]
            p = subprocess.Popen(
                cmd,
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _stdout, err = p.communicate()
            if err and err.strip():
                warnings.warn(err)

            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    out = f.read()
            except Exception:
                out = ""

            if not out.strip():
                logging.warning(
                    "DSSP failed to produce output for %s via %s; using Biotite geometry SSE fallback",
                    file,
                    dssp_bin,
                )
                try:
                    return _dssp_list_from_biotite_geometry(dssp_input)
                finally:
                    if tmp_file is not None:
                        os.remove(tmp_file)
                    if dssp_tmp_file is not None:
                        os.remove(dssp_tmp_file)

            dssp_tuple = dssp_mod._make_dssp_dict(StringIO(out))[0]
        finally:
            try:
                os.remove(out_path)
            except Exception:
                pass
    else:
        dssp_tuple = dssp_dict_from_pdb_file(dssp_input, DSSP=str(dssp_bin))[0]

    if tmp_file is not None:
        # remove the temporary pdb file
        os.remove(tmp_file)
    if dssp_tmp_file is not None:
        os.remove(dssp_tmp_file)

    dssp_list = []
    for key in dssp_tuple.keys():
        dssp_list.append(dssp_tuple[key][1])

    return dssp_list


def _collapse_dssp_symbol(sym: str) -> str:
    """
    Map DSSP symbols to coarse E/H/L classes.
    """
    sym = (sym or "-").upper()
    if sym in {"G", "H", "I"}:
        return "H"
    if sym in {"E", "B"}:
        return "E"
    return "L"


def _protomer_sse_fraction_from_dssp_list(
    dssp_list: list[str],
    oligo_name: str,
    proto_name: str,
    sse_type: str,
) -> float:
    """
    Compute per-protomer SSE fraction by slicing the concatenated DSSP list.

    NOTE: This assumes the residue order in the structure file matches the protomer
    order in `oligo_name` (common for our generated mmCIF/PDB outputs).
    """
    sse_type = sse_type.upper()
    if sse_type not in ("E", "H", "L"):
        raise ValueError("sse_type must be one of 'E', 'H', or 'L'")

    offset = 0
    for proto in oligo_name:
        length = int(getattr(global_config.protomers.get(proto), "length", 0) or 0)
        if length <= 0:
            continue
        if proto == proto_name:
            chunk = dssp_list[offset : offset + length]
            if not chunk:
                return 0.0
            collapsed = [_collapse_dssp_symbol(x) for x in chunk]
            return float(sum(1 for x in collapsed if x == sse_type) / len(collapsed))
        offset += length
    return 0.0


def _longest_sse_run_fraction_from_chunk(chunk: list[str], sse_type: str) -> float:
    """Return longest contiguous run fraction for a coarse DSSP class."""
    if not chunk:
        return 0.0
    collapsed = [_collapse_dssp_symbol(x) for x in chunk]
    longest = 0
    current = 0
    for code in collapsed:
        if code == sse_type:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest / len(collapsed))


def _protomer_longest_sse_run_fractions_from_dssp_list(
    dssp_list: list[str],
    oligo_name: str,
    included_protomers: set[str],
    sse_type: str,
) -> list[float]:
    """Return longest-run fractions for every included protomer copy."""
    fractions = []
    offset = 0
    for proto in oligo_name:
        length = int(getattr(global_config.protomers.get(proto), "length", 0) or 0)
        if length <= 0:
            continue
        chunk = dssp_list[offset : offset + length]
        if proto in included_protomers:
            fractions.append(_longest_sse_run_fraction_from_chunk(chunk, sse_type))
        offset += length
    return fractions


def calculate_dssp_fractions(dssp_list):
    """Compute DSSP fraction based on a DSSP list."""
    # todo hotspot sse fraction: since we
    sym2full = {
        "G": "helix",
        "H": "helix",
        "I": "helix",
        "E": "strand",
        "B": "strand",
        "S": "loop",
        "T": "loop",
        "C": "loop",
        "-": "loop",
    }
    dssp_list = [sym2full[sym] for sym in dssp_list]

    N_residues = len(dssp_list)
    fraction_beta = float(dssp_list.count("strand")) / float(N_residues)
    fraction_helix = float(dssp_list.count("helix")) / float(N_residues)
    fraction_other = float(1.0 - fraction_beta - fraction_helix)
    # print(dssp_list, fraction_beta, fraction_helix, fraction_other)

    return fraction_beta, fraction_helix, fraction_other


_USALIGN_RMSD_RE = re.compile(r"RMSD=\s*([0-9]+(?:\.[0-9]+)?)")
_USALIGN_TMSCORE_RE = re.compile(
    r"TM-score=\s*([0-9]+(?:\.[0-9]+)?)\s*\(normalized by length of Structure_(\d):"
)


def _parse_usalign_output(
    out: str, *, tm_normalize_by: str = "reference"
) -> tuple[float, float]:
    rmsd = None
    tm_scores: dict[int, float] = {}

    for line in (out or "").splitlines():
        if rmsd is None and "RMSD=" in line:
            m = _USALIGN_RMSD_RE.search(line)
            if m:
                rmsd = float(m.group(1))

        if "TM-score=" in line and "Structure_" in line:
            m = _USALIGN_TMSCORE_RE.search(line)
            if m:
                tm = float(m.group(1))
                struct_id = int(m.group(2))
                tm_scores[struct_id] = tm

        if rmsd is not None and len(tm_scores) >= 2:
            break

    # USalign prints: "Structure_1 ... (to be superimposed onto Structure_2)" and
    # recommends using the TM-score normalized by the reference structure.
    normalize = str(tm_normalize_by or "reference").strip().lower()
    preferred = None
    if normalize in {"reference", "ref", "template", "structure_2", "2"}:
        preferred = tm_scores.get(2)
    elif normalize in {"query", "pred", "prediction", "structure_1", "1"}:
        preferred = tm_scores.get(1)
    else:
        preferred = tm_scores.get(2) or tm_scores.get(1)

    if rmsd is None or preferred is None:
        snippet = "\\n".join((out or "").splitlines()[:60])
        raise RuntimeError(
            "Failed to parse USalign output (tm_normalize_by=%s). Output (first lines):\\n%s"
            % (tm_normalize_by, snippet)
        )

    return float(rmsd), float(preferred)


def usalign_wrapper(
    query_structure,
    reference_structure,
    force_alignment=None,
    *,
    multimer: bool = True,
    tm_normalize_by: str = "reference",
):
    try:
        usalign_exec = global_config.usalign_path
    except Exception:
        usalign_exec = None
    usalign_exec = usalign_exec or "USalign"
    mm = "1" if bool(multimer) else "0"
    cmd = [str(usalign_exec), "-mm", mm, str(query_structure), str(reference_structure)]
    if force_alignment is not None:
        cmd += ["-I", str(force_alignment)]

    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out = res.stdout or ""
    return _parse_usalign_output(out, tm_normalize_by=tm_normalize_by)


def tmalign_wrapper(template, temp_pdbfile, force_alignment=None):
    # Backwards-compatible name: this project uses USalign for TM-score computation.
    # We treat `template` as the reference structure (Structure_2).
    return usalign_wrapper(
        temp_pdbfile,
        template,
        force_alignment,
        multimer=True,
        tm_normalize_by="reference",
    )


def _prepare_chain_selected_structure(
    structure_path: str | pathlib.Path,
    chain_id: str | None,
) -> tuple[pathlib.Path, bool]:
    path = pathlib.Path(structure_path)
    if chain_id is None:
        return path, False
    return _extract_single_chain_to_pdb(path, str(chain_id)), True


def tmalign(
    path,
    template_files: list,
    *,
    query_chain: str | None = None,
    reference_chain: str | None = None,
):
    # write temporary pdbfile to compute tmscore.
    # todo: align all chains instead of just the first one
    # todo: specify the alignment for task like antibody-antigen
    score = []
    query_path, cleanup_query = _prepare_chain_selected_structure(path, query_chain)
    try:
        for template_file in template_files:
            reference_path, cleanup_reference = _prepare_chain_selected_structure(
                template_file,
                reference_chain,
            )
            try:
                tm_rmsd, tm_score = tmalign_wrapper(reference_path, query_path, None)
                logging.debug(f"tm_RMSD, tmscore {tm_rmsd}, {tm_score}")
                score.append(tm_score)
            finally:
                if cleanup_reference:
                    try:
                        reference_path.unlink()
                    except FileNotFoundError:
                        pass
    finally:
        if cleanup_query:
            try:
                query_path.unlink()
            except FileNotFoundError:
                pass
    return np.mean(score)


def _extract_single_chain_to_pdb(structure_path: pathlib.Path, chain_id: str) -> pathlib.Path:
    import tempfile

    from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select

    structure_path = pathlib.Path(structure_path)
    parser = MMCIFParser(QUIET=True) if structure_path.suffix.lower() == ".cif" else PDBParser(QUIET=True)
    structure = parser.get_structure("binder_pair", str(structure_path))

    class _ChainSelect(Select):
        def accept_chain(self, chain):
            return chain.id == chain_id

    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".pdb")
    handle.close()
    io = PDBIO()
    io.set_structure(structure)
    io.save(handle.name, _ChainSelect())
    return pathlib.Path(handle.name)


@loss_registry.register(
    name="binder_pair_tmscore",
    description="Mean binder-chain TM-score between two predicted oligomers using USalign",
    required_params=["oligo_a", "oligo_b", "binder_chain"],
)
def binder_pair_tmscore(eval_result, oligo_a: str, oligo_b: str, binder_chain: str):
    complex_paths = getattr(eval_result, "complex_paths", {}) or {}
    path_a_str = complex_paths.get(str(oligo_a))
    path_b_str = complex_paths.get(str(oligo_b))
    if path_a_str is None or path_b_str is None:
        return {name: 0.0 for name in getattr(eval_result, "names", [])}

    path_a = pathlib.Path(path_a_str)
    path_b = pathlib.Path(path_b_str)

    chain_a = _extract_single_chain_to_pdb(path_a, str(binder_chain))
    chain_b = _extract_single_chain_to_pdb(path_b, str(binder_chain))
    try:
        _, tm_query = usalign_wrapper(chain_a, chain_b, None, multimer=True, tm_normalize_by="query")
        _, tm_reference = usalign_wrapper(chain_a, chain_b, None, multimer=True, tm_normalize_by="reference")
        value = float((tm_query + tm_reference) / 2.0)
    finally:
        for tmp_path in (chain_a, chain_b):
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

    return {name: value for name in eval_result.names}


@loss_registry.register(
    optional_params=["template_files", "template_file_alias", "query_chain", "reference_chain"],
    description="USalign TM-score against template/reference structures",
    supports_oligomer_specific=True,
)
def tmscore(
    eval_result,
    template_files: Optional[List[str]] = None,
    template_file_alias: str | List[str] | None = None,
    query_chain: str | None = None,
    reference_chain: str | None = None,
):
    """Calculate TM-scores using USalign.

    Provide either:
      - `template_files`: list of structure paths, or
      - `template_file_alias`: alias(es) from config `named_files` (e.g. 'initial').
    """
    if template_files is None:
        if template_file_alias is None:
            raise ValueError("tmscore requires template_files or template_file_alias")

        aliases = (
            [template_file_alias]
            if isinstance(template_file_alias, str)
            else list(template_file_alias)
        )
        named_files = getattr(global_config, "named_files", {}) or {}
        resolved: list[str] = []
        for alias in aliases:
            pf = named_files.get(str(alias))
            if pf is None:
                raise ValueError(f"tmscore template_file_alias '{alias}' not found in named_files")
            path = getattr(pf, "file_path", None)
            if not path:
                raise ValueError(f"tmscore named_files['{alias}'] missing file_path")
            resolved.append(str(path))
        template_files = resolved

    scores = {}
    for name in eval_result.names:
        scores[name] = tmalign(
            eval_result.complex_paths[name],
            template_files,
            query_chain=query_chain,
            reference_chain=reference_chain,
        )
    return scores


def calculate_contact_order(coordinates, cutoff=6.0):
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    N = len(coordinates)
    distances = squareform(pdist(coordinates))
    contacts = np.where(distances < cutoff)
    contact_pairs = [(i, j) for i, j in zip(*contacts) if i < j]

    contact_order = sum(abs(i - j) for i, j in contact_pairs) / len(contact_pairs) / N
    return contact_order


def _radius_of_gyration_from_coords(coords: np.ndarray) -> float:
    if coords.size == 0:
        return 0.0
    centroid = coords.mean(axis=0)
    squared_distances = np.sum((coords - centroid) ** 2, axis=1)
    return float(np.sqrt(squared_distances.mean()))


def _chain_coords_from_structure_file(path: str, chain_id: str, *, ca_only: bool = False) -> np.ndarray:
    # gemmi supports both mmCIF and PDB.
    import gemmi

    st = gemmi.read_structure(path)
    coords = []
    for model in st:
        for chain in model:
            if chain.name != chain_id:
                continue
            for res in chain:
                for atom in res:
                    if ca_only and atom.name != "CA":
                        continue
                    p = atom.pos
                    coords.append([p.x, p.y, p.z])
    return np.asarray(coords, dtype=np.float64)


def compute_single_chain_rog(eval_result, included_protomers):
    result = {}

    complex_poses = getattr(eval_result, "complex_poses", None)
    complex_paths = getattr(eval_result, "complex_paths", {}) or {}

    for oligo_name in eval_result.names:
        included_chains = [int_id_to_str_id(i + 1) for i, p in enumerate(oligo_name) if p in included_protomers]

        # Fast path: PyRosetta ComplexPoses available.
        if complex_poses is not None and getattr(complex_poses, "poses", None) is not None and oligo_name in complex_poses.poses:
            pose = complex_poses.poses[oligo_name]
            chains = pose.split_by_chain()
            chain_id_to_index = {}
            for i in range(1, len(chains) + 1):  # split_by_chain is 1-based
                sub_pose = chains[i]
                chain_id = sub_pose.pdb_info().chain(1)  # each sub pose has one chain
                chain_id_to_index[chain_id] = i

            chain2rog = {}
            for chain in included_chains:
                chain_idx = chain_id_to_index.get(chain)
                if chain_idx is None:
                    continue
                # PyRosetta-based Rg from all atoms.
                coords = []
                ch_pose = chains[chain_idx]
                for r in range(1, ch_pose.size() + 1):
                    res = ch_pose.residue(r)
                    for a in range(1, res.natoms() + 1):
                        atom_xyz = res.xyz(a)
                        coords.append([atom_xyz.x, atom_xyz.y, atom_xyz.z])
                rog = _radius_of_gyration_from_coords(np.asarray(coords, dtype=np.float64))
                chain2rog[chain] = rog
            result[oligo_name] = float(np.mean(list(chain2rog.values()))) if chain2rog else 0.0
            continue

        # Fallback: compute Rg from the predicted structure file (mmCIF/PDB), no PyRosetta.
        path = complex_paths.get(oligo_name)
        if not path:
            result[oligo_name] = 0.0
            continue

        chain2rog = {}
        for chain in included_chains:
            coords = _chain_coords_from_structure_file(str(path), chain)
            chain2rog[chain] = _radius_of_gyration_from_coords(coords)
        result[oligo_name] = float(np.mean(list(chain2rog.values()))) if chain2rog else 0.0

    return result


@loss_registry.register(required_params=["included_protomers"], description="Calculate radius of gyration for included protomers", )
def single_chain_rog(eval_result, included_protomers: str):
    """Calculate single chain radius of gyration."""
    return compute_single_chain_rog(eval_result, included_protomers)


@loss_registry.register(
    required_params=["included_protomers"],
    description="Mean C-alpha radius of gyration of included protein chains",
    supports_oligomer_specific=True,
)
def single_chain_ca_rog(eval_result, included_protomers: str):
    scores = {}
    for name in eval_result.names:
        radii = []
        for index, protomer in enumerate(name):
            if protomer not in included_protomers:
                continue
            chain_id = int_id_to_str_id(index + 1)
            coords = _chain_coords_from_structure_file(
                str(eval_result.complex_paths[name]), chain_id, ca_only=True
            )
            if not len(coords):
                raise ValueError(f"single_chain_ca_rog found no CA atoms in {name} chain {chain_id}")
            radii.append(_radius_of_gyration_from_coords(coords))
        if not radii:
            raise ValueError(f"single_chain_ca_rog selected no chains in {name}")
        scores[name] = float(np.mean(radii))
    return scores


def _protomer_sse_fraction(proto_ss, proto_name: str, sse_type: str) -> float:
    chains = proto_ss.get(proto_name, [])
    if not chains:
        return 0.0

    target_sets = {
        "H": {"H", "G", "I"},
        "E": {"E", "B"},
    }

    target_codes = target_sets.get(sse_type, None)
    total = sum(len(chain) for chain in chains)
    if total == 0:
        return 0.0

    if target_codes is None:  # loop/other
        helix = target_sets["H"]
        strand = target_sets["E"]
        count = sum(
            sum(1 for res in chain if res not in helix and res not in strand)
            for chain in chains
        )
    else:
        count = sum(
            sum(1 for res in chain if res in target_codes)
            for chain in chains
        )

    return count / total


@loss_registry.register(
    required_params=["included_protomers", "sse_type"],
    description="Compute DSSP fraction of a specific SSE type (E/H/L) for protomers",
)
def secondary_structure_fraction(
    eval_result,
    included_protomers: str,
    sse_type: str,
):
    """Return the DSSP fraction for the requested secondary structure type."""

    sse_type = sse_type.upper()
    if sse_type not in ("E", "H", "L"):
        raise ValueError("sse_type must be one of 'E', 'H', or 'L'")

    include = [p.strip() for p in included_protomers.split(",") if p.strip()]

    result = {}
    for oligo_name in eval_result.names:
        path = eval_result.complex_paths.get(oligo_name)
        if path is None:
            result[oligo_name] = 0.0
            continue
        path_obj = pathlib.Path(str(path))
        fractions = []
        for idx, proto in enumerate(oligo_name):
            if proto not in include:
                continue
            chain_id = int_id_to_str_id(idx + 1)
            dssp_list = _dssp_list_for_chain_subset(path_obj, [chain_id])
            if not dssp_list:
                fractions.append(0.0)
                continue
            collapsed = [_collapse_dssp_symbol(x) for x in dssp_list]
            fractions.append(
                float(sum(1 for x in collapsed if x == sse_type) / len(collapsed))
            )
        result[oligo_name] = float(sum(fractions) / len(fractions)) if fractions else 0.0

    return result


@loss_registry.register(
    required_params=["included_protomers", "sse_type"],
    description="Compute longest contiguous DSSP run fraction for protomers",
)
def longest_secondary_structure_run_fraction(
    eval_result,
    included_protomers: str,
    sse_type: str,
):
    """Return the max contiguous SSE run fraction across included protomer copies."""

    sse_type = sse_type.upper()
    if sse_type not in ("E", "H", "L"):
        raise ValueError("sse_type must be one of 'E', 'H', or 'L'")

    include = {p.strip() for p in included_protomers.split(",") if p.strip()}

    result = {}
    for oligo_name in eval_result.names:
        path = eval_result.complex_paths.get(oligo_name)
        if path is None:
            result[oligo_name] = 0.0
            continue
        path_obj = pathlib.Path(str(path))
        fractions = []
        for idx, proto in enumerate(oligo_name):
            if proto not in include:
                continue
            chain_id = int_id_to_str_id(idx + 1)
            dssp_list = _dssp_list_for_chain_subset(path_obj, [chain_id])
            fractions.append(_longest_sse_run_fraction_from_chunk(dssp_list, sse_type))
        result[oligo_name] = float(max(fractions)) if fractions else 0.0

    return result


@loss_registry.register(required_params=["EHL_fractions"], description="Minimum DSSP fractions for helices, beta strands, and other structures", )
def min_dssp_fractions(eval_result, EHL_fractions):
    """Compute minimum DSSP fraction for each oligo."""
    result = {}
    min_frac_helix, min_frac_beta, min_frac_other = EHL_fractions.split(",")
    for oligo_name, path in eval_result.complex_paths.items():
        # calculate dssp for each pose
        dssp_list = dssp_wrapper(str(path))
        fraction_beta, fraction_helix, fraction_other = calculate_dssp_fractions(dssp_list)
        frac_delta = []
        if min_frac_helix is not None and min_frac_helix != "None":
            frac_delta.append(abs(fraction_helix - float(min_frac_helix)))
        if min_frac_beta is not None and min_frac_beta != "None":
            frac_delta.append(abs(fraction_beta - float(min_frac_beta)))
        if min_frac_other is not None and min_frac_other != "None":
            frac_delta.append(abs(fraction_other - float(min_frac_other)))
        result[oligo_name] = np.mean(frac_delta)
    return result
