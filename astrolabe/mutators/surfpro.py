from astrolabe.mutators.ligandmpnn import LigandMPNNMutator


class SurfProMutator(LigandMPNNMutator):
    """Structure-conditioned mutator backed by JocelynSong/SurfPro."""

    name = "SurfPro"
    capabilities = LigandMPNNMutator.capabilities

    def shutdown(self) -> None:
        self.runner.close()
