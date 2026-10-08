import json
import logging
import math
import os.path
import random
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from astrolabe.mutators.wrappers.ligandmpnn_utils import (
    ProteinMPNN,
    alphabet,
    element_dict_rev,
    featurize,
    get_score,
    get_seq_rec,
    parse_PDB,
    restype_int_to_str,
    restype_str_to_int,
)


class LigandMPNN:
    def __init__(self, args):

        # persistent parameters
        self.omit_AA_list = args.omit_AA
        self.bias_AA = args.bias_AA
        self.temperature = args.temperature
        self.verbose = args.verbose
        self.ligand_mpnn_use_atom_context = args.ligand_mpnn_use_atom_context
        self.ligand_mpnn_cutoff_for_score = args.ligand_mpnn_cutoff_for_score
        self.parse_atoms_with_zero_occupancy = args.parse_atoms_with_zero_occupancy
        self.number_of_batches = args.number_of_batches
        self.zero_indexed = args.zero_indexed
        self.fasta_seq_separation = args.fasta_seq_separation
        self.save_stats = args.save_stats
        self.file_ending = args.file_ending
        self.model_type = args.model_type
        self.bias_mode = getattr(args, "bias_mode", "additive")
        self.ligand_mpnn_use_side_chain_context = (
            args.ligand_mpnn_use_side_chain_context
        )
        self.seed = args.seed
        self.bias_AA_per_residue_multi = args.bias_AA_per_residue_multi
        self.bias_AA_per_residue = args.bias_AA_per_residue
        self.omit_AA_per_residue_multi = args.omit_AA_per_residue_multi
        self.omit_AA_per_residue = args.omit_AA_per_residue
        self.base_folder = args.out_folder
        if args.seed:
            seed = args.seed
        else:
            seed = int(np.random.randint(0, high=99999, size=1, dtype=int)[0])
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{int(torch.cuda.current_device())}")
        else:
            device = torch.device("cpu")
        self.device = device
        folder_for_outputs = args.out_folder
        base_folder = folder_for_outputs
        if base_folder[-1] != "/":
            base_folder = base_folder + "/"
        if not os.path.exists(base_folder):
            os.makedirs(base_folder, exist_ok=True)
        if not os.path.exists(base_folder + "seqs"):
            os.makedirs(base_folder + "seqs", exist_ok=True)
        if not os.path.exists(base_folder + "backbones"):
            os.makedirs(base_folder + "backbones", exist_ok=True)
        if args.model_type == "protein_mpnn":
            checkpoint_path = args.checkpoint_protein_mpnn
        elif args.model_type == "ligand_mpnn":
            checkpoint_path = args.checkpoint_ligand_mpnn
        elif args.model_type == "per_residue_label_membrane_mpnn":
            checkpoint_path = args.checkpoint_per_residue_label_membrane_mpnn
        elif args.model_type == "global_label_membrane_mpnn":
            checkpoint_path = args.checkpoint_global_label_membrane_mpnn
        elif args.model_type == "soluble_mpnn":
            checkpoint_path = args.checkpoint_soluble_mpnn
        else:
            logging.error("Choose one of the available models")
            sys.exit()
        self.checkpoint_path = checkpoint_path
        if (args.design_pattern_range is None) != (args.design_pattern is None):
            raise ValueError(
                "Both --design_pattern_range and --design_pattern must be provided or omitted together."
            )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if args.model_type == "ligand_mpnn":
            atom_context_num = checkpoint["atom_context_num"]
            ligand_mpnn_use_side_chain_context = args.ligand_mpnn_use_side_chain_context
            k_neighbors = checkpoint["num_edges"]
        else:
            atom_context_num = 1
            ligand_mpnn_use_side_chain_context = 0
            k_neighbors = checkpoint["num_edges"]
        self.ligand_mpnn_use_side_chain_context = ligand_mpnn_use_side_chain_context
        model = ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=k_neighbors,
            device=device,
            atom_context_num=atom_context_num,
            model_type=args.model_type,
            ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
        )
        self.atom_context_num = atom_context_num
        if args.design_pattern is not None:
            setattr(model, "design_pattern", args.design_pattern)
            setattr(model, "design_pattern_range", args.design_pattern_range)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        self.model = model

    @staticmethod
    def _load_json_mapping(value):
        if not value:
            return value
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        if isinstance(value, dict):
            return dict(value)
        with open(value, "r") as fh:
            return json.load(fh)

    def _convert_bias_value(self, raw_value: float) -> float:
        value = float(raw_value)
        if self.bias_mode == "additive":
            return value
        if self.bias_mode == "multiplicative":
            if value <= 0:
                raise ValueError(
                    f"Multiplicative aa_bias requires positive weights, got {value!r}."
                )
            if float(self.temperature) <= 0:
                raise ValueError(
                    f"Multiplicative aa_bias requires positive temperature, got {self.temperature!r}."
                )
            return float(self.temperature) * math.log(value)
        raise ValueError(f"Unsupported LigandMPNN bias_mode: {self.bias_mode!r}")

    def design_sequences(
        self,
        pdb_path=None,
        fixed_residues=None,
        symmetry_residues=None,
        symmetry_weights="",
        fixed_residues_multi="",
        pdb_path_multi="",
        redesigned_residues_multi="",
        redesigned_residues="",
        bias_AA_per_residue_multi="",
        bias_AA_per_residue="",
        omit_AA_per_residue_multi="",
        omit_AA_per_residue="",
        parse_these_chains_only="",
        transmembrane_buried="",
        transmembrane_interface="",
        transmembrane_buried_multi="",
        transmembrane_interface_multi="",
        global_transmembrane_label="",
        chains_to_design=None,
        homo_oligomer="",
        batch_size=1,
    ):
        """
        Inference function
        """
        device = self.device
        if bias_AA_per_residue_multi in ("", None):
            bias_AA_per_residue_multi = self.bias_AA_per_residue_multi
        if bias_AA_per_residue in ("", None):
            bias_AA_per_residue = self.bias_AA_per_residue
        if omit_AA_per_residue_multi in ("", None):
            omit_AA_per_residue_multi = self.omit_AA_per_residue_multi
        if omit_AA_per_residue in ("", None):
            omit_AA_per_residue = self.omit_AA_per_residue
        if global_transmembrane_label in ("", None):
            membrane_label_value = 0
        else:
            membrane_label_value = int(global_transmembrane_label)
        multi_mode = bool(pdb_path_multi)
        if pdb_path_multi:
            with open(pdb_path_multi, "r") as fh:
                pdb_paths = list(json.load(fh))
        else:
            pdb_paths = [pdb_path]
        outputs_by_path = {}

        if fixed_residues_multi:
            with open(fixed_residues_multi, "r") as fh:
                fixed_residues_multi = json.load(fh)
        else:
            fixed_residues = [item for item in fixed_residues.split()]
            fixed_residues_multi = {}
            for pdb in pdb_paths:
                fixed_residues_multi[pdb] = fixed_residues

        if redesigned_residues_multi:
            with open(redesigned_residues_multi, "r") as fh:
                redesigned_residues_multi = json.load(fh)
        else:
            redesigned_residues = [item for item in redesigned_residues.split()]
            redesigned_residues_multi = {}
            for pdb in pdb_paths:
                redesigned_residues_multi[pdb] = redesigned_residues

        bias_AA = torch.zeros([21], device=device, dtype=torch.float32)
        if self.bias_AA:
            tmp = [item.split(":") for item in self.bias_AA.split(",")]
            a1 = [b[0] for b in tmp]
            a2 = [float(b[1]) for b in tmp]
            for i, AA in enumerate(a1):
                bias_AA[restype_str_to_int[AA]] = self._convert_bias_value(a2[i])

        if bias_AA_per_residue_multi:
            bias_AA_per_residue_multi = self._load_json_mapping(
                bias_AA_per_residue_multi
            )  # {"pdb_path" : {"A12": {"G": 1.1}}}
        else:
            if bias_AA_per_residue:
                bias_AA_per_residue = self._load_json_mapping(
                    bias_AA_per_residue
                )  # {"A12": {"G": 1.1}}
                bias_AA_per_residue_multi = {}
                for pdb in pdb_paths:
                    bias_AA_per_residue_multi[pdb] = bias_AA_per_residue

        if omit_AA_per_residue_multi:
            omit_AA_per_residue_multi = self._load_json_mapping(
                omit_AA_per_residue_multi
            )  # {"pdb_path" : {"A12": "PQR", "A13": "QS"}}
        else:
            if omit_AA_per_residue:
                omit_AA_per_residue = self._load_json_mapping(
                    omit_AA_per_residue
                )  # {"A12": "PG"}
                omit_AA_per_residue_multi = {}
                for pdb in pdb_paths:
                    omit_AA_per_residue_multi[pdb] = omit_AA_per_residue

        if transmembrane_buried_multi:
            transmembrane_buried_multi = self._load_json_mapping(
                transmembrane_buried_multi
            )
        else:
            transmembrane_buried_multi = {
                pdb: transmembrane_buried for pdb in pdb_paths
            }
        if transmembrane_interface_multi:
            transmembrane_interface_multi = self._load_json_mapping(
                transmembrane_interface_multi
            )
        else:
            transmembrane_interface_multi = {
                pdb: transmembrane_interface for pdb in pdb_paths
            }

        omit_AA = torch.tensor(
            np.array([AA in self.omit_AA_list for AA in alphabet]).astype(np.float32),
            device=device,
        )

        # loop over PDB paths
        for pdb in pdb_paths:
            if self.verbose:
                logging.info("Designing protein from this path:", pdb)
            fixed_residues = fixed_residues_multi[pdb]
            redesigned_residues = redesigned_residues_multi[pdb]
            transmembrane_buried = transmembrane_buried_multi[pdb]
            transmembrane_interface = transmembrane_interface_multi[pdb]
            protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
                pdb,
                device=device,
                chains=parse_these_chains_only,
                parse_all_atoms=self.ligand_mpnn_use_side_chain_context,
                parse_atoms_with_zero_occupancy=self.parse_atoms_with_zero_occupancy,
            )
            # make chain_letter + residue_idx + insertion_code mapping to integers
            R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
            chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
            encoded_residues = []
            for i, R_idx_item in enumerate(R_idx_list):
                tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
                encoded_residues.append(tmp)
            encoded_residue_dict = dict(
                zip(encoded_residues, range(len(encoded_residues)))
            )
            encoded_residue_dict_rev = dict(
                zip(list(range(len(encoded_residues))), encoded_residues)
            )

            bias_AA_per_residue = torch.zeros(
                [len(encoded_residues), 21], device=device, dtype=torch.float32
            )
            if bias_AA_per_residue_multi:
                bias_dict = bias_AA_per_residue_multi[pdb]
                for residue_name, v1 in bias_dict.items():
                    if residue_name in encoded_residues:
                        i1 = encoded_residue_dict[residue_name]
                        for amino_acid, v2 in v1.items():
                            if amino_acid in alphabet:
                                j1 = restype_str_to_int[amino_acid]
                                bias_AA_per_residue[i1, j1] = self._convert_bias_value(v2)

            omit_AA_per_residue = torch.zeros(
                [len(encoded_residues), 21], device=device, dtype=torch.float32
            )
            if omit_AA_per_residue_multi:
                omit_dict = omit_AA_per_residue_multi[pdb]
                for residue_name, v1 in omit_dict.items():
                    if residue_name in encoded_residues:
                        i1 = encoded_residue_dict[residue_name]
                        for amino_acid in v1:
                            if amino_acid in alphabet:
                                j1 = restype_str_to_int[amino_acid]
                                omit_AA_per_residue[i1, j1] = 1.0

            fixed_positions = torch.tensor(
                [int(item not in fixed_residues) for item in encoded_residues],
                device=device,
            )
            redesigned_positions = torch.tensor(
                [int(item not in redesigned_residues) for item in encoded_residues],
                device=device,
            )

            # specify which residues are buried for checkpoint_per_residue_label_membrane_mpnn model
            if transmembrane_buried:
                buried_residues = [item for item in transmembrane_buried.split()]
                buried_positions = torch.tensor(
                    [int(item in buried_residues) for item in encoded_residues],
                    device=device,
                )
            else:
                buried_positions = torch.zeros_like(fixed_positions)

            if transmembrane_interface:
                interface_residues = [item for item in transmembrane_interface.split()]
                interface_positions = torch.tensor(
                    [int(item in interface_residues) for item in encoded_residues],
                    device=device,
                )
            else:
                interface_positions = torch.zeros_like(fixed_positions)
            protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
                1 - interface_positions
            ) + 1 * interface_positions * (1 - buried_positions)

            if self.model_type == "global_label_membrane_mpnn":
                protein_dict["membrane_per_residue_labels"] = (
                    membrane_label_value + 0 * fixed_positions
                )
            if type(chains_to_design) == str:
                chains_to_design_list = chains_to_design.split(",")
            else:
                chains_to_design_list = protein_dict["chain_letters"]
            chain_mask = torch.tensor(
                np.array(
                    [
                        item in chains_to_design_list
                        for item in protein_dict["chain_letters"]
                    ],
                    dtype=np.int32,
                ),
                device=device,
            )

            # create chain_mask to notify which residues are fixed (0) and which need to be designed (1)
            if redesigned_residues:
                protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
            elif fixed_residues:
                protein_dict["chain_mask"] = chain_mask * fixed_positions
            else:
                protein_dict["chain_mask"] = chain_mask

            if self.verbose:
                PDB_residues_to_be_redesigned = [
                    encoded_residue_dict_rev[item]
                    for item in range(protein_dict["chain_mask"].shape[0])
                    if protein_dict["chain_mask"][item] == 1
                ]
                PDB_residues_to_be_fixed = [
                    encoded_residue_dict_rev[item]
                    for item in range(protein_dict["chain_mask"].shape[0])
                    if protein_dict["chain_mask"][item] == 0
                ]
                logging.info(
                    "These residues will be redesigned: ", PDB_residues_to_be_redesigned
                )
                logging.info("These residues will be fixed: ", PDB_residues_to_be_fixed)

            # specify which residues are linked
            if symmetry_residues:
                symmetry_residues_list_of_lists = [
                    x.split(",") for x in symmetry_residues.split("|")
                ]
                remapped_symmetry_residues = []
                for t_list in symmetry_residues_list_of_lists:
                    tmp_list = []
                    for t in t_list:
                        tmp_list.append(encoded_residue_dict[t])
                    remapped_symmetry_residues.append(tmp_list)
            else:
                remapped_symmetry_residues = [[]]

            # specify linking weights
            if symmetry_weights:
                parsed_symmetry_weights = [
                    [float(item) for item in x.split(",")]
                    for x in str(symmetry_weights).split("|")
                ]
            else:
                parsed_symmetry_weights = [[]]

            if homo_oligomer:
                if self.verbose:
                    logging.info("Designing HOMO-OLIGOMER")
                chain_letters_set = list(set(chain_letters_list))
                reference_chain = chain_letters_set[0]
                lc = len(reference_chain)
                residue_indices = [
                    item[lc:]
                    for item in encoded_residues
                    if item[:lc] == reference_chain
                ]
                remapped_symmetry_residues = []
                parsed_symmetry_weights = []
                for res in residue_indices:
                    tmp_list = []
                    tmp_w_list = []
                    for chain in chain_letters_set:
                        name = chain + res
                        tmp_list.append(encoded_residue_dict[name])
                        tmp_w_list.append(1 / len(chain_letters_set))
                    remapped_symmetry_residues.append(tmp_list)
                    parsed_symmetry_weights.append(tmp_w_list)

            # set other atom bfactors to 0.0
            if other_atoms:
                other_bfactors = other_atoms.getBetas()
                other_atoms.setBetas(other_bfactors * 0.0)

            # adjust input PDB name by dropping .pdb if it does exist
            name = pdb[pdb.rfind("/") + 1 :]
            if name[-4:] == ".pdb":
                name = name[:-4]

            with torch.no_grad():
                # run featurize to remap R_idx and add batch dimension
                if self.verbose:
                    if "Y" in list(protein_dict):
                        atom_coords = protein_dict["Y"].cpu().numpy()
                        atom_types = list(protein_dict["Y_t"].cpu().numpy())
                        atom_mask = list(protein_dict["Y_m"].cpu().numpy())
                        number_of_atoms_parsed = np.sum(atom_mask)
                    else:
                        logging.info("No ligand atoms parsed")
                        number_of_atoms_parsed = 0
                        atom_types = ""
                        atom_coords = []
                    if number_of_atoms_parsed == 0:
                        logging.info("No ligand atoms parsed")
                    elif self.model_type == "ligand_mpnn":
                        logging.info(
                            f"The number of ligand atoms parsed is equal to: {number_of_atoms_parsed}"
                        )
                        for i, atom_type in enumerate(atom_types):
                            logging.info(
                                f"Type: {element_dict_rev[atom_type]}, Coords {atom_coords[i]}, Mask {atom_mask[i]}"
                            )
                feature_dict = featurize(
                    protein_dict,
                    cutoff_for_score=self.ligand_mpnn_cutoff_for_score,
                    use_atom_context=self.ligand_mpnn_use_atom_context,
                    number_of_ligand_atoms=self.atom_context_num,
                    model_type=self.model_type,
                )
                feature_dict["batch_size"] = batch_size
                B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
                # add additional keys to the feature dictionary
                feature_dict["temperature"] = self.temperature
                feature_dict["bias"] = (
                    (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
                    + bias_AA_per_residue[None]
                    - 1e8 * omit_AA_per_residue[None]
                )
                feature_dict["symmetry_residues"] = remapped_symmetry_residues
                feature_dict["symmetry_weights"] = parsed_symmetry_weights

                sampling_probs_list = []
                log_probs_list = []
                decoding_order_list = []
                S_list = []
                loss_list = []
                loss_per_residue_list = []
                loss_XY_list = []
                for _ in range(self.number_of_batches):
                    feature_dict["randn"] = torch.randn(
                        [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                        device=device,
                    )
                    output_dict = self.model.sample(feature_dict)
                    # compute confidence scores
                    loss, loss_per_residue = get_score(
                        output_dict["S"],
                        output_dict["log_probs"],
                        feature_dict["mask"] * feature_dict["chain_mask"],
                    )
                    if self.model_type == "ligand_mpnn":
                        combined_mask = (
                            feature_dict["mask"]
                            * feature_dict["mask_XY"]
                            * feature_dict["chain_mask"]
                        )
                    else:
                        combined_mask = (
                            feature_dict["mask"] * feature_dict["chain_mask"]
                        )
                    loss_XY, _ = get_score(
                        output_dict["S"], output_dict["log_probs"], combined_mask
                    )
                    # -----
                    S_list.append(output_dict["S"])
                    log_probs_list.append(output_dict["log_probs"])
                    sampling_probs_list.append(output_dict["sampling_probs"])
                    decoding_order_list.append(output_dict["decoding_order"])
                    loss_list.append(loss)
                    loss_per_residue_list.append(loss_per_residue)
                    loss_XY_list.append(loss_XY)
                S_stack = torch.cat(S_list, 0)
                log_probs_stack = torch.cat(log_probs_list, 0)
                sampling_probs_stack = torch.cat(sampling_probs_list, 0)
                decoding_order_stack = torch.cat(decoding_order_list, 0)
                loss_stack = torch.cat(loss_list, 0)
                loss_per_residue_stack = torch.cat(loss_per_residue_list, 0)
                loss_XY_stack = torch.cat(loss_XY_list, 0)
                rec_mask = feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
                rec_stack = get_seq_rec(feature_dict["S"][:1], S_stack, rec_mask)

                native_seq = "".join(
                    [
                        restype_int_to_str[AA]
                        for AA in feature_dict["S"][0].cpu().numpy()
                    ]
                )
                seq_np = np.array(list(native_seq))
                seq_out_str = []
                for mask in protein_dict["mask_c"]:
                    seq_out_str += list(seq_np[mask.cpu().numpy()])
                    seq_out_str += [self.fasta_seq_separation]
                seq_out_str = "".join(seq_out_str)[:-1]

                out_dict = {}
                out_dict["generated_sequences"] = S_stack.cpu()
                out_dict["sampling_probs"] = sampling_probs_stack.cpu()
                out_dict["log_probs"] = log_probs_stack.cpu()
                out_dict["decoding_order"] = decoding_order_stack.cpu()
                out_dict["native_sequence"] = feature_dict["S"][0].cpu()
                out_dict["mask"] = feature_dict["mask"][0].cpu()
                out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu()
                out_dict["seed"] = self.seed
                out_dict["temperature"] = self.temperature
                output_dict["chain_letters"] = tuple(
                    str(chain) for chain in chain_letters_list
                )
                outputs_by_path[str(pdb)] = output_dict

        if multi_mode:
            return {"multi_outputs": outputs_by_path}
        return output_dict
