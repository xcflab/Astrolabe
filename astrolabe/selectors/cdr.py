import logging
from collections import defaultdict

import numpy as np
import pyrosetta
from pyrosetta.rosetta.core.pose import get_chain_from_chain_id

from astrolabe.evaluators.utils import ComplexPoses
from astrolabe.selectors.base import PositionSelector


class CDRPositionSelector(PositionSelector):
    def __init__(
        self, complex_poses: ComplexPoses, cdr_proto_name: str | list[str], merge="any"
    ):
        self.cdr_proto_name = cdr_proto_name
        super().__init__(complex_poses, merge=merge)

    def select_positions(self, **kwargs):
        self.sites = self.cdr_selector(
            self.complex_poses.poses,
            self.cdr_proto_name,
            self.merge_func,
            self.complex_poses.chain2proto,
        )
        return self.sites

    def cdr_selector(
        self, poses, cdr_proto_name="A", merge_func=np.any, chain2proto=None
    ):
        proto2cdr = defaultdict(list)
        for oligo_name, pose in poses.items():
            cdr_sele = (
                pyrosetta.rosetta.protocols.antibody.residue_selector.CDRResidueSelector()
            )
            for chain in pose.split_by_chain():
                chain_name = get_chain_from_chain_id(chain.chain(1), chain)
                logging.info(chain_name)
                proto_name = chain2proto[chain_name]
                if proto_name == cdr_proto_name or proto_name in cdr_proto_name:
                    chain.pdb_info().set_chains("H")
                    # cdr_sele.set_cdr(pyrosetta.rosetta.protocols.antibody.CDRName.H3)
                    cdr_sele.apply(chain)
                    cdr_site = np.array(cdr_sele.selection_positions(chain))
                    cdr_site_mask = np.zeros(len(chain), dtype=bool)
                    cdr_site_mask[cdr_site] = True
                    proto2cdr[chain_name].append(cdr_site_mask)
                else:
                    proto2cdr[chain_name].append(np.zeros(len(chain), dtype=bool))
        # Select sites based on secondary structure.
        selected_sites = {}
        for proto, value in proto2cdr.items():
            selected_sites[proto] = merge_func(value, axis=0)
        return selected_sites
