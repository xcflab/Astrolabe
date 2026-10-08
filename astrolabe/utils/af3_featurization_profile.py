from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
import time
from typing import Any


def _ensure_af3_on_sys_path() -> None:
    """Ensure vendored AlphaFold3 sources are importable without external installs."""
    try:
        import alphafold3  # noqa: F401

        return
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    af3_src = repo_root / "vendor" / "alphafold3" / "src"
    if af3_src.exists():
        af3_src_str = str(af3_src)
        if af3_src_str not in sys.path:
            sys.path.insert(0, af3_src_str)

    # Retry import; if it still fails, let the exception bubble up for a clear error.
    import alphafold3  # noqa: F401


@dataclass(frozen=True)
class FeaturizationProfile:
    sequence_length: int
    num_tokens: int
    bucket_tokens: int
    total_s: float
    times_s: Mapping[str, float]


def _nan_check(np_example: Mapping[str, Any], *, fold_name: str, random_seed: int) -> None:
    import numpy as np

    for name, value in np_example.items():
        if (
            hasattr(value, "dtype")
            and value.dtype.kind not in {"U", "S"}
            and value.dtype.name != "object"
            and np.isnan(np.sum(value))
        ):
            raise RuntimeError(
                "AF3 featurisation produced NaNs: "
                f"nan feature={name} fold_input={fold_name} random_seed={random_seed}"
            )


def _profile_single_run(
    *,
    fold_input,
    ccd,
    random_seed: int,
    buckets: Sequence[int] | None,
    deterministic_frames: bool,
    msa_crop_size: int,
    max_templates: int,
    pad_num_chains: int,
    average_num_atoms_per_token: int,
    atom_cross_att_queries_subset_size: int,
    atom_cross_att_keys_subset_size: int,
    flatten_non_standard_residues: bool,
    remove_nonsymmetric_bonds: bool,
    drop_ligand_leaving_atoms: bool,
    filter_clashes: bool,
    filter_crystal_aids: bool,
    max_paired_sequence_per_species: int,
    resolve_msa_overlaps: bool,
    ref_max_modified_date,
    conformer_max_iterations: int | None,
) -> FeaturizationProfile:
    import numpy as np

    _ensure_af3_on_sys_path()

    from alphafold3.model import feat_batch, features
    from alphafold3.model.pipeline import inter_chain_bonds, pipeline as pipeline_lib
    from alphafold3.model.pipeline import structure_cleaning
    from alphafold3.constants import mmcif_names
    from alphafold3.structure import chemical_components as struc_chem_comps

    def measure(step: str, fn):
        t0 = time.perf_counter()
        out = fn()
        times_s[step] = float(time.perf_counter() - t0)
        return out

    times_s: dict[str, float] = {}
    start_total = time.perf_counter()

    random_state = np.random.RandomState(seed=random_seed)

    struct = measure("to_structure", lambda: fold_input.to_structure(ccd=ccd))
    cleaned_struc, cleaning_metadata = measure(
        "clean_structure",
        lambda: structure_cleaning.clean_structure(
            struct,
            ccd=ccd,
            drop_non_standard_atoms=True,
            drop_missing_sequence=True,
            filter_clashes=filter_clashes,
            filter_crystal_aids=filter_crystal_aids,
            filter_waters=True,
            filter_hydrogens=True,
            filter_leaving_atoms=drop_ligand_leaving_atoms,
            only_glycan_ligands_for_leaving_atoms=True,
            covalent_bonds_only=True,
            remove_polymer_polymer_bonds=True,
            remove_bad_bonds=True,
            remove_nonsymmetric_bonds=remove_nonsymmetric_bonds,
        ),
    )
    num_clashing_chains_removed = cleaning_metadata.get("num_clashing_chains_removed", 0)
    if num_clashing_chains_removed and cleaned_struc.num_chains == 0:
        raise RuntimeError("AF3 clean_structure removed all chains unexpectedly.")
    if cleaned_struc.num_chains == 0:
        raise RuntimeError("AF3 featurisation: no chains in cleaned structure.")

    polymer_ligand_bonds, ligand_ligand_bonds = measure(
        "inter_chain_bonds",
        lambda: inter_chain_bonds.get_polymer_ligand_and_ligand_ligand_bonds(
            cleaned_struc,
            only_glycan_ligands=False,
            allow_multiple_bonds_per_atom=True,
        ),
    )
    if ligand_ligand_bonds is not None and not ligand_ligand_bonds.atom_name.size:
        ligand_ligand_bonds = None
    if polymer_ligand_bonds is not None and not polymer_ligand_bonds.atom_name.size:
        polymer_ligand_bonds = None

    empty_output_struc, flat_output_layout = measure(
        "create_empty_output_struc_and_layout",
        lambda: structure_cleaning.create_empty_output_struc_and_layout(
            struc=cleaned_struc,
            ccd=ccd,
            polymer_ligand_bonds=polymer_ligand_bonds,
            ligand_ligand_bonds=ligand_ligand_bonds,
            drop_ligand_leaving_atoms=drop_ligand_leaving_atoms,
        ),
    )

    all_tokens, all_token_atoms_layout, standard_token_idxs = measure(
        "tokenizer",
        lambda: features.tokenizer(
            flat_output_layout,
            ccd=ccd,
            max_atoms_per_token=24,
            flatten_non_standard_residues=flatten_non_standard_residues,
            logging_name=f"{fold_input.name}, random_seed={random_seed}",
        ),
    )
    num_tokens = len(all_tokens.atom_name)

    bucket_tokens = measure(
        "calculate_bucket_size",
        lambda: pipeline_lib.calculate_bucket_size(num_tokens, buckets),
    )

    num_atoms = bucket_tokens * average_num_atoms_per_token
    num_atoms = int(
        np.ceil(num_atoms / atom_cross_att_queries_subset_size) * atom_cross_att_queries_subset_size
    )
    padding_shapes = measure(
        "padding_shapes",
        lambda: features.PaddingShapes(
            num_tokens=bucket_tokens,
            msa_size=msa_crop_size,
            num_chains=pad_num_chains,
            num_templates=max_templates,
            num_atoms=num_atoms,
        ),
    )

    batch_atom_cross_att = measure(
        "atom_cross_att",
        lambda: features.AtomCrossAtt.compute_features(
            all_token_atoms_layout=all_token_atoms_layout,
            queries_subset_size=atom_cross_att_queries_subset_size,
            keys_subset_size=atom_cross_att_keys_subset_size,
            padding_shapes=padding_shapes,
        ),
    )

    batch_token_features = measure(
        "token_features",
        lambda: features.TokenFeatures.compute_features(
            all_tokens=all_tokens,
            padding_shapes=padding_shapes,
        ),
    )

    chemical_components_data = measure(
        "populate_missing_ccd_data",
        lambda: struc_chem_comps.populate_missing_ccd_data(
            ccd=ccd,
            chemical_components_data=cleaned_struc.chemical_components_data,
            populate_pdbx_smiles=True,
        ),
    )
    empty_output_struc = empty_output_struc.copy_and_update_globals(
        chemical_components_data=chemical_components_data
    )

    batch_convert_model_output = measure(
        "convert_model_output",
        lambda: features.ConvertModelOutput.compute_features(
            all_token_atoms_layout=all_token_atoms_layout,
            padding_shapes=padding_shapes,
            cleaned_struc=cleaned_struc,
            flat_output_layout=flat_output_layout,
            empty_output_struc=empty_output_struc,
            polymer_ligand_bonds=polymer_ligand_bonds,
            ligand_ligand_bonds=ligand_ligand_bonds,
        ),
    )

    batch_predicted_structure_info = measure(
        "predicted_structure_info",
        lambda: features.PredictedStructureInfo.compute_features(
            all_tokens=all_tokens,
            all_token_atoms_layout=all_token_atoms_layout,
            padding_shapes=padding_shapes,
        ),
    )

    batch_msa = measure(
        "msa_features",
        lambda: features.MSA.compute_features(
            all_tokens=all_tokens,
            standard_token_idxs=standard_token_idxs,
            padding_shapes=padding_shapes,
            fold_input=fold_input,
            logging_name=f"{fold_input.name}, random_seed={random_seed}",
            max_paired_sequence_per_species=max_paired_sequence_per_species,
            resolve_msa_overlaps=resolve_msa_overlaps,
        ),
    )

    batch_templates = measure(
        "template_features",
        lambda: features.Templates.compute_features(
            all_tokens=all_tokens,
            standard_token_idxs=standard_token_idxs,
            padding_shapes=padding_shapes,
            fold_input=fold_input,
            max_templates=max_templates,
            logging_name=f"{fold_input.name}, random_seed={random_seed}",
        ),
    )

    batch_ref_structure, ligand_ligand_bonds = measure(
        "ref_structure",
        lambda: features.RefStructure.compute_features(
            all_token_atoms_layout=all_token_atoms_layout,
            ccd=ccd,
            padding_shapes=padding_shapes,
            chemical_components_data=chemical_components_data,
            random_state=random_state,
            ref_max_modified_date=ref_max_modified_date,
            conformer_max_iterations=conformer_max_iterations,
            ligand_ligand_bonds=ligand_ligand_bonds,
        ),
    )

    deterministic_ref_structure = None
    if deterministic_frames:
        has_non_polymer_chains = np.isin(
            all_tokens.chain_type, list(mmcif_names.NON_POLYMER_CHAIN_TYPES)
        ).any()
        if has_non_polymer_chains:
            deterministic_ref_structure, _ = measure(
                "ref_structure_deterministic",
                lambda: features.RefStructure.compute_features(
                    all_token_atoms_layout=all_token_atoms_layout,
                    ccd=ccd,
                    padding_shapes=padding_shapes,
                    chemical_components_data=chemical_components_data,
                    random_state=np.random.RandomState(12312837),
                    ref_max_modified_date=ref_max_modified_date,
                    conformer_max_iterations=None,
                    ligand_ligand_bonds=ligand_ligand_bonds,
                ),
            )
        else:
            deterministic_ref_structure = batch_ref_structure

    polymer_ligand_bond_info = measure(
        "polymer_ligand_bond_info",
        lambda: features.PolymerLigandBondInfo.compute_features(
            all_tokens=all_tokens,
            all_token_atoms_layout=all_token_atoms_layout,
            bond_layout=polymer_ligand_bonds,
            padding_shapes=padding_shapes,
        ),
    )
    ligand_ligand_bond_info = measure(
        "ligand_ligand_bond_info",
        lambda: features.LigandLigandBondInfo.compute_features(
            all_tokens,
            ligand_ligand_bonds,
            padding_shapes,
        ),
    )

    batch_pseudo_beta_info = measure(
        "pseudo_beta_info",
        lambda: features.PseudoBetaInfo.compute_features(
            all_token_atoms_layout=all_token_atoms_layout,
            ccd=ccd,
            padding_shapes=padding_shapes,
            logging_name=f"{fold_input.name}, random_seed={random_seed}",
        ),
    )
    batch_frames = measure(
        "frames",
        lambda: features.Frames.compute_features(
            all_tokens=all_tokens,
            all_token_atoms_layout=all_token_atoms_layout,
            ref_structure=deterministic_ref_structure
            if deterministic_frames
            else batch_ref_structure,
            padding_shapes=padding_shapes,
        ),
    )

    batch = feat_batch.Batch(
        msa=batch_msa,
        templates=batch_templates,
        token_features=batch_token_features,
        ref_structure=batch_ref_structure,
        predicted_structure_info=batch_predicted_structure_info,
        polymer_ligand_bond_info=polymer_ligand_bond_info,
        ligand_ligand_bond_info=ligand_ligand_bond_info,
        pseudo_beta_info=batch_pseudo_beta_info,
        atom_cross_att=batch_atom_cross_att,
        convert_model_output=batch_convert_model_output,
        frames=batch_frames,
    )
    np_example = measure("as_data_dict", batch.as_data_dict)
    measure(
        "nan_check",
        lambda: _nan_check(np_example, fold_name=fold_input.name, random_seed=random_seed),
    )

    total_s = float(time.perf_counter() - start_total)
    return FeaturizationProfile(
        sequence_length=len(fold_input.chains[0].sequence),
        num_tokens=num_tokens,
        bucket_tokens=bucket_tokens,
        total_s=total_s,
        times_s=times_s,
    )


def profile_featurization_steps(
    *,
    sequence: str,
    name: str = "gh_af3_featur",
    chain_id: str = "A",
    random_seed: int = 0,
    buckets: Sequence[int] | None = None,
    deterministic_frames: bool = True,
    msa_crop_size: int = 16384,
    max_templates: int = 4,
    pad_num_chains: int = 1000,
    average_num_atoms_per_token: int = 24,
    atom_cross_att_queries_subset_size: int = 32,
    atom_cross_att_keys_subset_size: int = 128,
    flatten_non_standard_residues: bool = True,
    remove_nonsymmetric_bonds: bool = False,
    drop_ligand_leaving_atoms: bool = True,
    filter_clashes: bool = False,
    filter_crystal_aids: bool = False,
    max_paired_sequence_per_species: int = 600,
    resolve_msa_overlaps: bool = True,
    ref_max_modified_date=None,
    conformer_max_iterations: int | None = None,
    runs: int = 3,
    warmup: int = 1,
) -> dict[str, Any]:
    """Profile AF3 featurisation (no inference) and return per-step timings.

    Notes:
    - This measures the vendored AlphaFold3 featurisation pipeline (WholePdbPipeline.process_item)
      broken down into the major steps used to build the model input BatchDict.
    - MSAs/templates are set to a minimal query-only A3M and empty templates unless
      the caller modifies this function.
    """
    if runs <= 0:
        raise ValueError("runs must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    _ensure_af3_on_sys_path()

    from alphafold3.common.folding_input import Input, ProteinChain
    from alphafold3.constants import chemical_components, mmcif_names
    from alphafold3.data import msa as msa_module

    # Minimal query-only MSAs, so the benchmark is self-contained (no DB search).
    unpaired = msa_module.Msa.from_empty(
        query_sequence=sequence,
        chain_poly_type=mmcif_names.PROTEIN_CHAIN,
    ).to_a3m()
    paired = msa_module.Msa.from_empty(
        query_sequence=sequence,
        chain_poly_type=mmcif_names.PROTEIN_CHAIN,
    ).to_a3m()

    fold_input = Input(
        name=name,
        chains=[
            ProteinChain(
                id=chain_id,
                sequence=sequence,
                ptms=[],
                unpaired_msa=unpaired,
                paired_msa=paired,
                templates=[],
            )
        ],
        rng_seeds=[random_seed],
        bonded_atom_pairs=None,
        user_ccd=None,
    )
    ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)

    # Warm-up runs (populate caches, filesystem reads, etc).
    for _ in range(warmup):
        _profile_single_run(
            fold_input=fold_input,
            ccd=ccd,
            random_seed=random_seed,
            buckets=buckets,
            deterministic_frames=deterministic_frames,
            msa_crop_size=msa_crop_size,
            max_templates=max_templates,
            pad_num_chains=pad_num_chains,
            average_num_atoms_per_token=average_num_atoms_per_token,
            atom_cross_att_queries_subset_size=atom_cross_att_queries_subset_size,
            atom_cross_att_keys_subset_size=atom_cross_att_keys_subset_size,
            flatten_non_standard_residues=flatten_non_standard_residues,
            remove_nonsymmetric_bonds=remove_nonsymmetric_bonds,
            drop_ligand_leaving_atoms=drop_ligand_leaving_atoms,
            filter_clashes=filter_clashes,
            filter_crystal_aids=filter_crystal_aids,
            max_paired_sequence_per_species=max_paired_sequence_per_species,
            resolve_msa_overlaps=resolve_msa_overlaps,
            ref_max_modified_date=ref_max_modified_date,
            conformer_max_iterations=conformer_max_iterations,
        )

    profiles: list[FeaturizationProfile] = []
    for _ in range(runs):
        profiles.append(
            _profile_single_run(
                fold_input=fold_input,
                ccd=ccd,
                random_seed=random_seed,
                buckets=buckets,
                deterministic_frames=deterministic_frames,
                msa_crop_size=msa_crop_size,
                max_templates=max_templates,
                pad_num_chains=pad_num_chains,
                average_num_atoms_per_token=average_num_atoms_per_token,
                atom_cross_att_queries_subset_size=atom_cross_att_queries_subset_size,
                atom_cross_att_keys_subset_size=atom_cross_att_keys_subset_size,
                flatten_non_standard_residues=flatten_non_standard_residues,
                remove_nonsymmetric_bonds=remove_nonsymmetric_bonds,
                drop_ligand_leaving_atoms=drop_ligand_leaving_atoms,
                filter_clashes=filter_clashes,
                filter_crystal_aids=filter_crystal_aids,
                max_paired_sequence_per_species=max_paired_sequence_per_species,
                resolve_msa_overlaps=resolve_msa_overlaps,
                ref_max_modified_date=ref_max_modified_date,
                conformer_max_iterations=conformer_max_iterations,
            )
        )

    mean_total = statistics.mean(p.total_s for p in profiles)
    # Union of all step names (some may be absent depending on config).
    all_steps: set[str] = set()
    for p in profiles:
        all_steps.update(p.times_s.keys())
    mean_times = {
        step: statistics.mean(p.times_s.get(step, 0.0) for p in profiles)
        for step in sorted(all_steps)
    }

    seq_len = len(sequence)
    # All profiles should agree on token counts for fixed inputs.
    num_tokens = profiles[0].num_tokens
    bucket_tokens = profiles[0].bucket_tokens

    return {
        "sequence_length": seq_len,
        "num_tokens": num_tokens,
        "bucket_tokens": bucket_tokens,
        "total_s": float(mean_total),
        "times_s": mean_times,
        "runs": runs,
        "warmup": warmup,
    }
