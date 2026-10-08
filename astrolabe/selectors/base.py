import numpy as np

from astrolabe.evaluators.utils import ComplexPoses


class PositionSelector:
    def __init__(self, complex_poses: ComplexPoses | None, sites=None, merge="any"):
        self.complex_poses = complex_poses
        if sites is not None:
            self.sites = sites
        else:
            self.set_merge_func(merge)
            self.select_positions()

    def set_merge_func(self, merge):
        if merge == "any":
            self.merge_func = np.any
        elif merge == "all":
            self.merge_func = np.all
        else:
            raise ValueError(f"Invalid merge method: {merge}, must be 'any' or 'all'.")

    def select_positions(self, **kwargs): ...

    def __add__(self, other):
        sites = {
            proto: self.sites[proto] & other.sites[proto] for proto in self.sites.keys()
        }
        return PositionSelector(self.complex_poses or other.complex_poses, sites)

    def __and__(self, other):
        sites = {
            proto: self.sites[proto] & other.sites[proto] for proto in self.sites.keys()
        }
        return PositionSelector(self.complex_poses or other.complex_poses, sites)

    def __rand__(self, other):
        return self.__and__(other)

    def __sub__(self, other):
        sites = {
            proto: self.sites[proto] & ~other.sites[proto]
            for proto in self.sites.keys()
        }
        return PositionSelector(self.complex_poses or other.complex_poses, sites)

    def __or__(self, other):
        sites = {
            proto: self.sites[proto] | other.sites[proto] for proto in self.sites.keys()
        }
        return PositionSelector(self.complex_poses or other.complex_poses, sites)

    def __ror__(self, other):
        return self.__or__(other)

    def __xor__(self, other):
        sites = {
            proto: self.sites[proto] ^ other.sites[proto] for proto in self.sites.keys()
        }
        return PositionSelector(self.complex_poses or other.complex_poses, sites)

    def __rxor__(self, other):
        return self.__xor__(other)

    def __invert__(self):
        sites = {proto: ~self.sites[proto] for proto in self.sites.keys()}
        return PositionSelector(self.complex_poses , sites)

class ExplicitPositionSelector(PositionSelector):
    def __init__(self, sites=None, merge="any"):
        super().__init__(None, sites, merge)
