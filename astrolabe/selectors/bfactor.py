from collections import defaultdict

import numpy as np
import pyrosetta
from pyrosetta.rosetta.core.pose import get_chain_from_chain_id

from astrolabe.utils.mmcif_ids import int_id_to_str_id
from astrolabe.evaluators.utils import ComplexPoses
from astrolabe.selectors.base import PositionSelector


class BFactorPositionSelector(PositionSelector):
    def __init__(
        self,
        complex_poses: ComplexPoses,
        lower_bound=None,
        upper_bound=None,
        merge="any",
    ):
        self.lower_bound, self.upper_bound = self.parse_cutoff(lower_bound, upper_bound)

        super().__init__(complex_poses, merge=merge)

    def select_positions(self, **kwargs):

        self.sites = self.bfactor_selector(
            self.complex_poses.poses,
            self.lower_bound,
            self.upper_bound,
            self.merge_func,
        )
        return self.sites

    def parse_cutoff(self, lower_bound, upper_bound):
        if lower_bound is None:
            lower_bound = 0.0

        if upper_bound is None:
            upper_bound = 100.0

        return lower_bound, upper_bound

    def bfactor_selector(
        self, poses,  lower_bound=0.0, higher_bound=1.0, merge_func=np.any
    ):
        proto2bfactor = defaultdict(list)
        for oligo_name, pose in poses.items():
            bfactor_sele = (
                pyrosetta.rosetta.core.select.residue_selector.BFactorSelector()
            )
            chain2proto = {int_id_to_str_id(i+1):proto for i, proto in enumerate(oligo_name)}
            for chain in pose.split_by_chain():
                chain_name = chain2proto[get_chain_from_chain_id(chain.chain(1), chain)]
                bfactor_sele.set_lower_bfactor_threshold(lower_bound)
                bfactor_sele.set_upper_bfactor_threshold(higher_bound)
                bfactor_sele.apply(chain)
                bfactor_site = np.array(bfactor_sele.apply(chain))
                proto2bfactor[chain_name].append(bfactor_site)
        selected_sites = {}
        for proto, value in proto2bfactor.items():
            selected_sites[proto] = merge_func(value, axis=0)
        return selected_sites
