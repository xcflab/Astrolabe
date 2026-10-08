import functools


restype_str_to_int = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "X": 20,
}
restype_int_to_str = {
    0: "A",
    1: "C",
    2: "D",
    3: "E",
    4: "F",
    5: "G",
    6: "H",
    7: "I",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "Q",
    14: "R",
    15: "S",
    16: "T",
    17: "V",
    18: "W",
    19: "Y",
    20: "X",
}

aa_order = [restype_int_to_str[i] for i in range(len(restype_int_to_str))][:20]

aa_freq = {
    "A": 0.07421620506799341,
    "R": 0.05161448614128464,
    "N": 0.044645808512757915,
    "D": 0.05362600083855441,
    "C": 0.02468745716794485,
    "Q": 0.03425965059141602,
    "E": 0.0543119256845875,
    "G": 0.074146941452645,
    "H": 0.026212984805266227,
    "I": 0.06791736761895376,
    "L": 0.09890786849715096,
    "K": 0.05815568230307968,
    "M": 0.02499019757964311,
    "F": 0.04741845974228475,
    "P": 0.038538003320306206,
    "S": 0.05722902947649442,
    "T": 0.05089136455028703,
    "W": 0.013029956129972148,
    "Y": 0.03228151231375858,
    "V": 0.07291909820561925,
}


def _normalize_exclude_aas(exclude) -> tuple[str, ...]:
    if exclude is None:
        return ()
    if isinstance(exclude, str):
        return tuple(aa.upper() for aa in exclude if aa.strip())
    return tuple(str(aa).strip().upper() for aa in exclude if str(aa).strip())


@functools.cache
def _cached_aa_freq(exclude: tuple[str, ...] = ()) -> dict[str, float]:
    aa_freq = {
        "A": 0.07421620506799341,
        "R": 0.05161448614128464,
        "N": 0.044645808512757915,
        "D": 0.05362600083855441,
        "C": 0.02468745716794485,
        "Q": 0.03425965059141602,
        "E": 0.0543119256845875,
        "G": 0.074146941452645,
        "H": 0.026212984805266227,
        "I": 0.06791736761895376,
        "L": 0.09890786849715096,
        "K": 0.05815568230307968,
        "M": 0.02499019757964311,
        "F": 0.04741845974228475,
        "P": 0.038538003320306206,
        "S": 0.05722902947649442,
        "T": 0.05089136455028703,
        "W": 0.013029956129972148,
        "Y": 0.03228151231375858,
        "V": 0.07291909820561925,
    }
    if exclude:
        aa_freq = {k: v for k, v in aa_freq.items() if k not in exclude}
        aa_freq = {key: f / sum(aa_freq.values()) for key, f in aa_freq.items()}
    return aa_freq


def cached_aa_freq(exclude=None) -> dict[str, float]:
    return dict(_cached_aa_freq(_normalize_exclude_aas(exclude)))


DEFAULT_SEEDS = 1024


FULL_TYPES = {"p": "protein", "l": "ligand", "r": "rna", "d": "dna"}

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

restype_3to1 = {v: k for k, v in restype_1to3.items()}
