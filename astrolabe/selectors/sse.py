from collections import defaultdict

import numpy as np
import pyrosetta
from pyrosetta.rosetta.core.pose import get_chain_from_chain_id

from astrolabe.utils.mmcif_ids import int_id_to_str_id
from astrolabe.core import global_config
from astrolabe.evaluators.utils import ComplexPoses
from astrolabe.selectors.base import PositionSelector


class SecondaryStructurePositionSelector(PositionSelector):
    def __init__(self, complex_poses: ComplexPoses, sec_struct_sign, merge="any"):

        if sec_struct_sign is None:
            raise ValueError("Secondary structure sign must be specified.")
        elif sec_struct_sign not in ["H", "E", "L"]:
            raise ValueError(
                f"Invalid secondary structure sign: {sec_struct_sign}, must be 'H', 'E', or 'L'."
            )
        else:
            self.sec_struct_sign = sec_struct_sign

        super().__init__(complex_poses, merge=merge)

    def select_positions(self, **kwargs):

        self.sites = self.make_protomer_ss(
            self.complex_poses.poses,
            self.sec_struct_sign,
            self.merge_func,
        )
        return self.sites

    def make_protomer_ss(
        self, poses, sec_struct_sign=None, merge_func=np.any,
    ):
        if sec_struct_sign is None:
            return {
                name: range(protomer.length) for name, protomer in global_config.protomers.items()
            }
        elif sec_struct_sign not in ["H", "E", "L"]:
            raise ValueError(
                f"Invalid secondary structure sign: {sec_struct_sign}, must be 'H', 'E', or 'L'."
            )

        proto2dssp = defaultdict(list)
        for oligo_name, pose in poses.items():
            chain2proto = {int_id_to_str_id(i+1):proto for i, proto in enumerate(oligo_name)}
            dssp = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
            dssp.insert_ss_into_pose(pose)
            # Compute DSSP for each chain, and append to the corresponding protomer.
            for chain in pose.split_by_chain():
                chain_name = get_chain_from_chain_id(chain.chain(1), chain)
                proto2dssp[chain2proto[chain_name]].append(chain.secstruct())

        # Select sites based on secondary structure.
        selected_sites = {}
        for proto, value in proto2dssp.items():
            dssp_arr_sub = np.array([list(v) for v in value]).T
            selected_sites[proto] = merge_func(dssp_arr_sub == sec_struct_sign, axis=1)

        return selected_sites


def _proto_secstruct_map(complex_poses):
    proto2ss = defaultdict(list)
    for oligo_name, pose in complex_poses.poses.items():
        chain2proto = {
            int_id_to_str_id(i + 1): proto for i, proto in enumerate(oligo_name)
        }
        dssp = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
        dssp.insert_ss_into_pose(pose)
        for chain in pose.split_by_chain():
            chain_name = get_chain_from_chain_id(chain.chain(1), chain)
            proto2ss[chain2proto[chain_name]].append(chain.secstruct())
    return proto2ss


def _empty_mask(proto_name: str) -> np.ndarray:
    length = len(global_config.protomers[proto_name].sequence)
    return np.zeros(length, dtype=bool)


class LongHelixCorePositionSelector(PositionSelector):
    """Select core residues of helices longer than ``min_length``."""

    def __init__(self, complex_poses, min_length: int = 12, core_length: int = 4, merge="any"):
        self.min_length = max(1, int(min_length))
        self.core_length = max(1, int(core_length))
        super().__init__(complex_poses, merge=merge)

    def select_positions(self, **kwargs):
        proto2ss = _proto_secstruct_map(self.complex_poses)
        sites = {}
        for proto_name in global_config.protomers.keys():
            secstructs = proto2ss.get(proto_name, [])
            if not secstructs:
                sites[proto_name] = _empty_mask(proto_name)
                continue
            per_chain_masks = [self._mask_for_chain(ss, proto_name) for ss in secstructs]
            stacked = np.stack(per_chain_masks, axis=0)
            sites[proto_name] = self.merge_func(stacked, axis=0)
        self.sites = sites
        return sites

    def _mask_for_chain(self, secstruct: str, proto_name: str) -> np.ndarray:
        mask = _empty_mask(proto_name)
        i = 0
        limit = len(mask)
        while i < min(len(secstruct), limit):
            if secstruct[i] == "H":
                start = i
                while i < len(secstruct) and secstruct[i] == "H":
                    i += 1
                run_len = i - start
                if run_len >= self.min_length:
                    core_size = min(self.core_length, run_len)
                    offset = (run_len - core_size) // 2
                    core_start = start + offset
                    core_end = min(start + offset + core_size, limit)
                    mask[core_start:core_end] = True
            else:
                i += 1
        return mask


class LongLoopPositionSelector(PositionSelector):
    """Select all residues belonging to loops longer than ``min_length``."""

    def __init__(self, complex_poses, min_length: int = 10, merge="any"):
        self.min_length = max(1, int(min_length))
        super().__init__(complex_poses, merge=merge)

    def select_positions(self, **kwargs):
        proto2ss = _proto_secstruct_map(self.complex_poses)
        sites = {}
        for proto_name in global_config.protomers.keys():
            secstructs = proto2ss.get(proto_name, [])
            if not secstructs:
                sites[proto_name] = _empty_mask(proto_name)
                continue
            per_chain_masks = [self._mask_for_chain(ss, proto_name) for ss in secstructs]
            stacked = np.stack(per_chain_masks, axis=0)
            sites[proto_name] = self.merge_func(stacked, axis=0)
        self.sites = sites
        return sites

    def _mask_for_chain(self, secstruct: str, proto_name: str) -> np.ndarray:
        mask = _empty_mask(proto_name)
        limit = len(mask)
        i = 0
        while i < min(len(secstruct), limit):
            if secstruct[i] == "L":
                start = i
                while i < len(secstruct) and secstruct[i] == "L":
                    i += 1
                run_len = i - start
                if run_len >= self.min_length:
                    loop_end = min(start + run_len, limit)
                    mask[start:loop_end] = True
            else:
                i += 1
        return mask


class LongLoopTerminalPositionSelector(PositionSelector):
    """Select central residues for loops longer than ``min_length`` (legacy behavior)."""

    def __init__(
        self,
        complex_poses,
        min_length: int = 10,
        terminal_span: int = 3,
        merge="any",
    ):
        self.min_length = max(1, int(min_length))
        self.terminal_span = max(1, int(terminal_span))
        super().__init__(complex_poses, merge=merge)

    def select_positions(self, **kwargs):
        proto2ss = _proto_secstruct_map(self.complex_poses)
        sites = {}
        for proto_name in global_config.protomers.keys():
            secstructs = proto2ss.get(proto_name, [])
            if not secstructs:
                sites[proto_name] = _empty_mask(proto_name)
                continue
            per_chain_masks = [self._mask_for_chain(ss, proto_name) for ss in secstructs]
            stacked = np.stack(per_chain_masks, axis=0)
            sites[proto_name] = self.merge_func(stacked, axis=0)
        self.sites = sites
        return sites

    def _mask_for_chain(self, secstruct: str, proto_name: str) -> np.ndarray:
        mask = _empty_mask(proto_name)
        limit = len(mask)
        i = 0
        while i < min(len(secstruct), limit):
            if secstruct[i] == "L":
                start = i
                while i < len(secstruct) and secstruct[i] == "L":
                    i += 1
                run_len = i - start
                if run_len >= self.min_length:
                    core_size = min(self.terminal_span, run_len)
                    offset = (run_len - core_size) // 2
                    core_start = start + offset
                    core_end = min(start + offset + core_size, limit)
                    mask[core_start:core_end] = True
            else:
                i += 1
        return mask
