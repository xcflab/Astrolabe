"""Compatibility imports for the former multistate ProteinMPNN name."""

from astrolabe.mutators.proteinmpnn import MultistateProteinMPNNMutator


FarApartTiedMultitargetLigandMPNNMutator = MultistateProteinMPNNMutator
FarApartTiedMultitargetLigandMPNN = "FarApartTiedMultitargetLigandMPNN"


__all__ = [
    "FarApartTiedMultitargetLigandMPNN",
    "FarApartTiedMultitargetLigandMPNNMutator",
]
