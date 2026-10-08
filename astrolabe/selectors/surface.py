import numpy as np
import pyrosetta
from pyrosetta.rosetta.core.pose import get_chain_from_chain_id

from astrolabe.evaluators.utils import ComplexPoses
from astrolabe.selectors.base import PositionSelector


class SurfacePositionSelector(PositionSelector):
    def __init__(self, complex_poses: ComplexPoses, merge="any"):
        super().__init__(complex_poses, merge=merge)

    def select_positions(self):
        self.sites = self.surface_selector(
            self.complex_poses.poses
        )
        return self.sites

    def surface_selector(self, poses, ):
        surface_sele = pyrosetta.rosetta.core.select.residue_selector.LayerSelector()
        surface_sele.set_layers(False, False, True)
        surface_sele.set_use_sc_neighbors(False)
        selected_sites = {}
        for oligo_name, pose in poses.items():
            for chain in pose.split_by_chain():
                chain_name = get_chain_from_chain_id(chain.chain(1), chain)
                selected_sites[chain_name] = np.array(surface_sele.apply(chain))
        return selected_sites
