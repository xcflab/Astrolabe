from astrolabe.mutators.ligandmpnn import LigandMPNNMutator


class SurfDesignMutator(LigandMPNNMutator):
    """Structure-conditioned mutator backed by smiles724/SurfDesign."""

    name = "SurfDesign"
    capabilities = LigandMPNNMutator.capabilities

    def shutdown(self) -> None:
        self.runner.close()
