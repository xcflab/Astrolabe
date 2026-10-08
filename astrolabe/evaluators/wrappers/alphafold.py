from __future__ import annotations

import copy
import io
import logging
from unittest import mock
from typing import Sequence

import numpy as np
from Bio.PDB import MMCIFParser

from alphafold.common import residue_constants
from alphafold.data import pipeline, templates
from alphafold.data import msa_pairing, feature_processing
from alphafold.data.parsers import Msa
from alphafold.data.pipeline_multimer import (
    convert_monomer_features,
    add_assembly_features,
    pad_msa,
)
from astrolabe.core.msa_utils import parse_a3m_sequences
from alphafold.model import model, config, data


def alphafold_setup(
    oligo_names: list,
    model_id=3,
    recycles=1,
    msa_clusters=1,
    suffix="multimer_v3",
    data_dir=None,
    initial_guess=None,
) -> model.RunModel:
    """Setup AlphaFold2 models."""

    mod = f"model_{model_id}_{suffix}"  # _ptm series of models necessary for returning pTM and PAE.
    logging.info("Setting up model: %s", mod)
    model_config = config.model_config(mod)
    model_config.model.num_recycle = (
        recycles  # AF2 default is 3. Effect on computing time is linear.
    )
    # model 1,2 do not accept msa_clusters < 512, manually correct:
    min_msa_clusters_model1and2 = 5
    if (
        any(str(m) in mod for m in [1, 2])
        and msa_clusters < min_msa_clusters_model1and2
    ):
        logging.info(
            "Setting MSA clusters to %s as using: %s",
            min_msa_clusters_model1and2,
            mod,
        )
        msa_clusters = int(min_msa_clusters_model1and2)
    if suffix.startswith("ptm"):
        model_config.data.common.num_recycle = (
            recycles  # AF2 default is 3. Effect on computing time is linear.
        )
        model_config.data.common.max_extra_msa = (
            msa_clusters  # AF2 default is 5120. Turning off is about 8x faster.
        )
        model_config.data.eval.max_msa_clusters = (
            msa_clusters  # AF2 default is 512. Turning off is about 8x faster.
        )
        model_config.data.eval.num_ensemble = 1
    else:
        model_config.model.global_config.multimer_mode = True
    model_params = data.get_model_haiku_params(model_name=mod, data_dir=data_dir)
    model_runner = model.RunModel(model_config, model_params)
    # Keep evaluator-facing AF2 extras on runner without changing vendor API.
    model_runner._gh_initial_guess_cfg = initial_guess  # type: ignore[attr-defined]
    # model_runner.model_params = model_params

    # Make separate model runners for each oligomer to avoid re-compilation and save time.
    model_runners = {}
    for oligo in oligo_names:
        model_runners[oligo] = model_runner

    return model_runners


def _a3m_to_msa(sequence: str, msa_string: str | None, label: str) -> Msa | None:
    if not msa_string:
        return None
    records = parse_a3m_sequences(msa_string)
    valid_records = [
        (header if header else f"{label}_{idx}", seq)
        for idx, (header, seq) in enumerate(records)
        if len(seq) == len(sequence)
    ]
    if not valid_records:
        return None
    sequences = [seq for _, seq in valid_records]
    deletion_matrix = [[0] * len(sequence) for _ in sequences]
    descriptions = [header for header, _ in valid_records]
    return Msa(
        sequences=sequences,
        deletion_matrix=deletion_matrix,
        descriptions=descriptions,
    )


def _make_msa_features_from_homology(chain_sequence: str, chain_data: dict) -> dict | None:
    msa_objects = []
    for label in ("unpaired_msa", "paired_msa"):
        msa_obj = _a3m_to_msa(chain_sequence, chain_data.get(label), label)
        if msa_obj is not None:
            msa_objects.append(msa_obj)
    if not msa_objects:
        return None
    return pipeline.make_msa_features(msa_objects)


def _parse_template_residues(mmcif_string: str) -> dict[int, any]:
    parser = MMCIFParser(QUIET=True)
    handle = io.StringIO(mmcif_string)
    structure = parser.get_structure("template", handle)
    chain = next(structure.get_chains(), None)
    if chain is None:
        return {}
    residues = {}
    for residue in chain.get_residues():
        hetflag, resseq, icode = residue.id
        if hetflag.strip():
            continue
        residues[resseq] = residue
    return residues


def _make_template_features_from_af3(
    sequence: str, templates_af3: Sequence | None
) -> dict | None:
    if not templates_af3:
        return None
    num_res = len(sequence)
    atom_type_num = residue_constants.atom_type_num
    aa_to_idx = {
        aa: idx for idx, aa in enumerate(residue_constants.restypes_with_x_and_gap)
    }
    template_positions = []
    template_masks = []
    template_aatype = []
    template_sequences = []
    template_confidence = []
    domain_names = []
    release_dates = []
    sum_probs = []

    for template_idx, template in enumerate(templates_af3):
        mmcif_string = getattr(template, "mmcif", None)
        mapping = getattr(template, "query_to_template_map", {})
        if not mmcif_string or not mapping:
            continue
        residues = _parse_template_residues(mmcif_string)
        if not residues:
            continue
        positions = np.zeros((num_res, atom_type_num, 3), dtype=np.float32)
        masks = np.zeros((num_res, atom_type_num), dtype=np.float32)
        aatype = np.zeros(
            (num_res, len(residue_constants.restypes_with_x_and_gap)),
            dtype=np.float32,
        )
        seq_chars = ["-"] * num_res
        conf = np.zeros(num_res, dtype=np.float32)
        filled = False
        for query_idx, template_idx_res in mapping.items():
            q = int(query_idx) - 1
            if not (0 <= q < num_res):
                continue
            residue = residues.get(int(template_idx_res))
            if residue is None:
                continue
            resname = residue.get_resname()
            aa = residue_constants.restype_3to1.get(resname, "X")
            aa_idx = aa_to_idx.get(aa, aa_to_idx["X"])
            aatype[q, aa_idx] = 1.0
            seq_chars[q] = aa
            conf[q] = 1.0
            for atom_name, atom_index in residue_constants.atom_order.items():
                if atom_name in residue:
                    atom = residue[atom_name]
                    positions[q, atom_index] = atom.get_coord().astype(np.float32)
                    masks[q, atom_index] = 1.0
            filled = True
        if not filled:
            continue
        template_positions.append(positions)
        template_masks.append(masks)
        template_aatype.append(aatype)
        template_sequences.append("".join(seq_chars).encode())
        template_confidence.append(conf)
        domain_names.append(f"template_{template_idx}".encode())
        release_dates.append("1970-01-01".encode())
        sum_probs.append(1.0)

    if not template_positions:
        return None

    return {
        "template_all_atom_positions": np.stack(template_positions, axis=0),
        "template_all_atom_masks": np.stack(template_masks, axis=0),
        "template_aatype": np.stack(template_aatype, axis=0),
        "template_sequence": np.array(template_sequences, dtype=object),
        "template_confidence_scores": np.stack(template_confidence, axis=0),
        "template_domain_names": np.array(domain_names, dtype=object),
        "template_release_date": np.array(release_dates, dtype=object),
        "template_sum_probs": np.array(sum_probs, dtype=np.float32),
    }


def make_AF2_features(query_sequence: str):
    data_pipeline_mock = mock.Mock()
    data_pipeline_mock.process.return_value = {
        **pipeline.make_sequence_features(
            sequence=query_sequence, description="none", num_res=len(query_sequence)
        ),  # **pipeline.make_msa_features(Msa(msas=[[query_sequence]],
        #                              deletion_matrices=[[[0] * len(query_sequence)]])),
        **pipeline.make_msa_features(
            [
                Msa(
                    sequences=[query_sequence],
                    deletion_matrix=[[[0] * len(query_sequence)]],
                    descriptions=["none"],
                )
            ]
        ),
        **mk_mock_template(query_sequence),
    }

    # Get features.
    feature_dict = data_pipeline_mock.process()

    return feature_dict


def make_AFM_features(chain_id_map, is_homomer_or_monomer, chain_homology=None):
    # split sequence into monomer chunks

    all_chain_features = {}
    sequence_features = {}
    for chain_id, fasta_chain in chain_id_map.items():
        chain_sequence = fasta_chain.sequence
        chain_data = chain_homology.get(chain_id, {}) if chain_homology else {}
        msa_feat = _make_msa_features_from_homology(chain_sequence, chain_data)
        template_features = chain_data.get("template_features")
        if template_features is None:
            template_features = _make_template_features_from_af3(
                chain_sequence, chain_data.get("templates")
            )
        if template_features is None:
            template_features = mk_mock_template(chain_sequence)
        if chain_sequence in sequence_features:
            all_chain_features[chain_id] = copy.deepcopy(
                sequence_features[chain_sequence]
            )
            all_chain_features[chain_id].update(
                convert_monomer_features(template_features, chain_id)
            )
            continue
        # single chain features
        data_pipeline_mock = mock.Mock()
        if msa_feat is None:
            msa_feat = pipeline.make_msa_features(
                [
                    Msa(
                        sequences=[chain_sequence],
                        deletion_matrix=[[0] * len(chain_sequence)],
                        descriptions=["query"],
                    )
                ]
            )
        data_pipeline_mock.process.return_value = {
            **pipeline.make_sequence_features(
                sequence=chain_sequence,
                description="none",
                num_res=len(chain_sequence),
            ),
            **msa_feat,
            **template_features,
        }

        # Get features.
        chain_features = data_pipeline_mock.process()
        if not is_homomer_or_monomer:
            valid_feats = msa_pairing.MSA_FEATURES + ("msa_species_identifiers",)
            feats = {
                f"{k}_all_seq": v for k, v in chain_features.items() if k in valid_feats
            }
            chain_features.update(feats)

        chain_features = convert_monomer_features(chain_features, chain_id=chain_id)
        all_chain_features[chain_id] = chain_features
        sequence_features[chain_sequence] = chain_features

    all_chain_features_asm = add_assembly_features(all_chain_features)

    np_example = feature_processing.pair_and_merge(
        all_chain_features=all_chain_features_asm
    )

    # Pad MSA to avoid zero-sized extra_msa.
    feature_dict = pad_msa(np_example, 512)
    return feature_dict


def mk_mock_template(query_sequence):
    """Generate mock template features from the input sequence."""

    output_templates_sequence = []
    output_confidence_scores = []
    templates_all_atom_positions = []
    templates_all_atom_masks = []

    for _ in query_sequence:
        templates_all_atom_positions.append(
            np.zeros((templates.residue_constants.atom_type_num, 3))
        )
        templates_all_atom_masks.append(
            np.zeros(templates.residue_constants.atom_type_num)
        )
        output_templates_sequence.append("-")
        output_confidence_scores.append(-1)

    output_templates_sequence = "".join(output_templates_sequence)
    templates_aatype = templates.residue_constants.sequence_to_onehot(
        output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
    )

    template_placeholder = np.zeros((1,), dtype=np.int32)

    template_features = {
        "template_all_atom_positions": np.array(templates_all_atom_positions)[None],
        "template_all_atom_masks": np.array(templates_all_atom_masks)[None],
        "template_sequence": template_placeholder,
        "template_aatype": np.array(templates_aatype)[None],
        "template_confidence_scores": np.array(output_confidence_scores)[None],
        "template_domain_names": template_placeholder,
        "template_release_date": template_placeholder,
        "template_sum_probs": np.zeros((1,), dtype=np.float32),
    }

    return template_features
