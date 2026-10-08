#!/usr/bin/env python3
"""Declarative software selection for the GH container bootstrap."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Iterable


@dataclass(frozen=True)
class SoftwareSpec:
    name: str
    summary: str
    asset_policy: str
    asset_source: str
    host_envs: tuple[str, ...] = ()
    submodules: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    license_id: str | None = None
    asset_urls: tuple[str, ...] = ()
    revision: str | None = None


SOFTWARE: dict[str, SoftwareSpec] = {
    "alphafold3": SoftwareSpec(
        "alphafold3",
        "AlphaFold 3 evaluator",
        "restricted",
        "Google DeepMind AF3 model-parameter access request; supply an existing models directory",
        submodules=("vendor/alphafold3",),
        license_id="af3-terms",
        asset_urls=("https://github.com/google-deepmind/alphafold3",),
    ),
    "alphafold2": SoftwareSpec(
        "alphafold2",
        "AlphaFold 2/Multimer evaluator",
        "public",
        "Google DeepMind alphafold_params_2022-12-06.tar",
        submodules=("vendor/alphafold2",),
        asset_urls=("https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar",),
    ),
    "ligandmpnn": SoftwareSpec(
        "ligandmpnn",
        "LigandMPNN, ProteinMPNN, membrane and soluble mutators",
        "public",
        "Institute for Protein Design LigandMPNN checkpoint archive",
        asset_urls=("https://files.ipd.uw.edu/pub/ligandmpnn/",),
    ),
    "protenix": SoftwareSpec(
        "protenix",
        "Protenix v2 evaluator and constraint model",
        "public-upstream-currently-broken",
        "ByteDance Protenix dependency_url.py; the v2 endpoint currently returns HTTP 403",
        submodules=("vendor/protenix", "vendor/esm2"),
        asset_urls=(
            "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix-v2.pt",
            "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_constraint_v0.5.0.pt",
            "https://af3-dev.tos-cn-beijing.volces.com/release_data/components.v20240608.cif",
            "https://af3-dev.tos-cn-beijing.volces.com/release_data/components.v20240608.cif.rdkit_mol.pkl",
        ),
    ),
    "boltz": SoftwareSpec(
        "boltz",
        "Boltz evaluator",
        "public",
        "URLs owned by astrolabe.evaluators.wrappers.boltz",
        submodules=("vendor/boltz",),
        asset_urls=("https://huggingface.co/boltz-community/boltz-1",),
        revision="7c1d83b779e4c65ecc37dfdf0c6b2788076f31e1",
    ),
    "chai1": SoftwareSpec(
        "chai1",
        "Chai-1 evaluator",
        "public",
        "Chai upstream component URLs in chai_lab.utils.paths",
        submodules=("vendor/chai_lab",),
        asset_urls=("https://chaiassets.com/chai1-inference-depencencies/",),
    ),
    "esm2": SoftwareSpec(
        "esm2",
        "ESM2 sequence mutator",
        "public",
        "Meta fair-esm model and regression URLs from vendor/esm2",
        submodules=("vendor/esm2",),
        asset_urls=("https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",),
    ),
    "esmif": SoftwareSpec(
        "esmif",
        "ESM inverse-folding mutator",
        "public",
        "Meta fair-esm ESM-IF1 checkpoint used by vendor/esm2",
        submodules=("vendor/esm2",),
        asset_urls=("https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt",),
    ),
    "esmfold": SoftwareSpec(
        "esmfold",
        "In-process ESMFold evaluator",
        "public-explicit-prefetch",
        "Meta fair-esm ESMFold checkpoint; GH intentionally disables lazy download",
        submodules=("vendor/esm2", "vendor/openfold_esm2"),
        asset_urls=("https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt",),
    ),
    "esm3": SoftwareSpec(
        "esm3",
        "ESM3 sequence/structure mutator",
        "gated",
        "EvolutionaryScale Hugging Face model; license acceptance and HF_TOKEN may be required",
        submodules=("vendor/esm3",),
        license_id="esm3-noncommercial",
        asset_urls=("https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1",),
        revision="47f0545b2b6daf26a93439a3cd610f4f7f3d5478",
    ),
    "rhofold": SoftwareSpec(
        "rhofold",
        "RhoFold RNA evaluator",
        "public",
        "ml4bio RhoFold checkpoint on Hugging Face",
        submodules=("vendor/rhofold",),
        asset_urls=("https://huggingface.co/cuhkaih/rhofold",),
        revision="4458c1c5484a3a10a7f3059b9f0e8ca0447b23ac",
    ),
    "dplm": SoftwareSpec(
        "dplm",
        "DPLM subprocess mutator",
        "public",
        "vendor/dplm scripts/install.sh plus airkingbd/dplm_150m on Hugging Face",
        host_envs=("dplm",),
        submodules=("vendor/dplm",),
        asset_urls=("https://huggingface.co/airkingbd/dplm_150m",),
        revision="49b7125a5d28c6418fcc2f3c4fe799352ac1488b",
    ),
    "nampnn": SoftwareSpec(
        "nampnn",
        "NA-MPNN mutator",
        "bundled",
        "Checkpoint and code are carried by vendor/na_mpnn",
        submodules=("vendor/na_mpnn",),
    ),
    "opendde": SoftwareSpec(
        "opendde",
        "OpenDDE evaluator",
        "public",
        "Pinned official OpenDDE checkpoint and common chemistry data on Hugging Face; search databases are optional",
        host_envs=("opendde", "opendde_assets"),
        submodules=("vendor/opendde",),
        asset_urls=(
            "https://huggingface.co/aurekaresearch/OpenDDE/resolve/eddd563ce96571f784012edd8f045181c8f8627d/opendde.pt",
            "https://huggingface.co/aurekaresearch/OpenDDE/resolve/eddd563ce96571f784012edd8f045181c8f8627d/common/components.cif",
            "https://huggingface.co/aurekaresearch/OpenDDE/resolve/eddd563ce96571f784012edd8f045181c8f8627d/common/components.cif.rdkit_mol.pkl",
            "https://huggingface.co/aurekaresearch/OpenDDE/resolve/eddd563ce96571f784012edd8f045181c8f8627d/common/release_date_cache.json",
            "https://huggingface.co/aurekaresearch/OpenDDE/resolve/eddd563ce96571f784012edd8f045181c8f8627d/common/obsolete_to_successor.json",
        ),
        revision="eddd563ce96571f784012edd8f045181c8f8627d",
    ),
    "esmfold2": SoftwareSpec(
        "esmfold2",
        "Biohub ESMFold2 subprocess evaluator",
        "gated",
        "Vendored Biohub code plus biohub/ESMFold2 on Hugging Face",
        host_envs=("esmfold2",),
        submodules=("vendor/biohub_esm", "vendor/biohub_transformers"),
        license_id="esmfold2-aup",
        asset_urls=("https://huggingface.co/biohub/ESMFold2",),
        revision="1ebf0e3481a5184eb6171d40615c79e384b48796",
    ),
    "rapidock": SoftwareSpec(
        "rapidock",
        "RAPiDock predocker",
        "public",
        "Pinned RAPiDock source plus official Zenodo local/global checkpoints",
        host_envs=("rapidock", "rapidock_assets"),
        asset_urls=(
            "https://zenodo.org/records/14193621/files/rapidock_local.pt?download=1",
            "https://zenodo.org/records/14193621/files/rapidock_global.pt?download=1",
        ),
        revision="36b2f780cf60b6cd80c2ed128b79a7d5c3b29469",
    ),
    "rpxdock": SoftwareSpec(
        "rpxdock",
        "RPXDock symmetry canonicalizer",
        "bundled-build",
        "Build vendor/rpxdock and vendor/willutil in an isolated host environment",
        host_envs=("rpxdock",),
        submodules=("vendor/rpxdock", "vendor/willutil"),
    ),
    "autodock-vina": SoftwareSpec(
        "autodock-vina",
        "AutoDock Vina, Meeko and OpenBabel predocker",
        "container",
        "Already installed in the unified container runtime",
    ),
    "autodock-gpu": SoftwareSpec(
        "autodock-gpu",
        "AutoDock-GPU predocker",
        "manual-binary",
        "Supply licensed/site-compatible AutoDock-GPU and AutoGrid4 executables",
        requires=("autodock-vina",),
    ),
    "haddock3": SoftwareSpec(
        "haddock3",
        "HADDOCK3 nucleic-acid predocker",
        "manual-wheel-or-binary",
        "Supply a reviewed HADDOCK3 wheel or executable",
    ),
    "ananas": SoftwareSpec(
        "ananas",
        "AnAnaS repeat-symmetry loss",
        "licensed-build-download",
        "INRIA AnAnaS 1.1 Linux academic binary; downloaded only into a private accepted build",
        license_id="ananas-academic",
        asset_urls=(
            "https://files.inria.fr/NanoDFiles/Website/Software/AnAnaS/1.1/Linux/ananas-linux.zip",
        ),
        revision="1.1",
    ),
    "rosetta": SoftwareSpec(
        "rosetta",
        "Rosetta command-line tools and molfile_to_params",
        "licensed-build-source",
        "Minimal molfile_to_params tree fetched from an exact official Rosetta commit",
        license_id="rosetta-license",
        asset_urls=("https://github.com/RosettaCommons/rosetta.git",),
        revision="80a0635615099e1b918474a63acba7b1de6fd107",
        requires=("pyrosetta",),
    ),
    "pyrosetta": SoftwareSpec(
        "pyrosetta",
        "PyRosetta losses, hooks and selectors",
        "licensed-build-download",
        "Exact official cp311 wheel installed in the default private SIF build",
        license_id="pyrosetta-license",
        asset_urls=(
            "https://west.rosettacommons.org/pyrosetta/quarterly/release/"
            "pyrosetta-2026.3+releasequarterly.5e498f1409-cp311-cp311-linux_x86_64.whl",
        ),
        revision="2026.3+releasequarterly.5e498f1409",
    ),
    "dynamicmpnn": SoftwareSpec(
        "dynamicmpnn",
        "DynamicMPNN mutator",
        "bundled",
        "GH implementation reuses the selected LigandMPNN checkpoints",
        requires=("ligandmpnn",),
    ),
    "carbondesign": SoftwareSpec(
        "carbondesign",
        "CarbonDesign subprocess mutator",
        "manual-assets",
        "Supply a reviewed CarbonDesign repository, checkpoint and feature configs",
    ),
    "rf3": SoftwareSpec(
        "rf3",
        "RoseTTAFold3 evaluator",
        "public-checkpoint",
        "Official Foundry RF3 checkpoint; Foundry runtime source/environment remains explicit",
        asset_urls=(
            "https://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt",
        ),
    ),
    "rfd3": SoftwareSpec(
        "rfd3",
        "RFD3 backbone initializer",
        "public-checkpoint",
        "Official Foundry RFD3 checkpoint; Foundry runtime source/environment remains explicit",
        asset_urls=(
            "https://files.ipd.uw.edu/pub/rfd3/rfd3_foundry_2025_12_01_remapped.ckpt",
        ),
    ),
    "surfdesign": SoftwareSpec(
        "surfdesign",
        "SurfDesign surface-conditioned mutator",
        "manual-trained-checkpoint",
        "Vendored source is fixed, but upstream does not publish the trained SurfDesign checkpoint",
        host_envs=("surfdesign",),
        submodules=("vendor/surfdesign",),
    ),
    "surfpro": SoftwareSpec(
        "surfpro",
        "SurfPro surface-conditioned mutator",
        "public",
        "Official SurfPro CATH 4.2 and pretraining checkpoints on Google Drive",
        host_envs=("surfpro",),
        submodules=("vendor/surfpro", "vendor/protein_sequence_models", "vendor/sru"),
        asset_urls=(
            "https://drive.usercontent.google.com/download?id=1xl_3xUrzjZCYcl5C5eUKjJd5VoRG28NQ&export=download&confirm=t",
            "https://drive.usercontent.google.com/download?id=1F2M4yMVPlkKY5KXpp6GwJvopl3Qq8gRo&export=download&confirm=t",
        ),
        revision="upstream source gitlink; checkpoint variant recorded separately",
    ),
    "dssp": SoftwareSpec(
        "dssp",
        "DSSP-based losses",
        "container",
        "Built from the vendored DSSP source in the unified SIF",
        submodules=("vendor/dssp-source",),
    ),
    "usalign": SoftwareSpec(
        "usalign",
        "USalign/TM-score losses",
        "bundled-build",
        "Build the vendored USalign source through gh_env_setup",
        host_envs=("usalign",),
    ),
    "dalphaball": SoftwareSpec(
        "dalphaball",
        "DAlphaBall-backed packing losses",
        "bundled-binary",
        "Executable is carried by astrolabe/assets/DAlphaBall.gcc",
    ),
}


SOFTWARE_PROFILES: dict[str, tuple[str, ...]] = {
    "core": (
        "alphafold3",
        "alphafold2",
        "ligandmpnn",
        "protenix",
        "boltz",
        "pyrosetta",
    ),
    "language": ("chai1", "esm2", "esmif", "esmfold", "esm3", "dplm", "esmfold2"),
    "rna": ("rhofold", "nampnn"),
    "docking": (
        "autodock-vina",
        "autodock-gpu",
        "rapidock",
        "haddock3",
    ),
    "symmetry": ("rpxdock", "ananas"),
    "surface": ("surfdesign", "surfpro"),
    "public": ("alphafold2", "ligandmpnn", "boltz", "chai1", "esm2", "esmif", "esmfold", "rhofold"),
}
SOFTWARE_PROFILES["all"] = tuple(SOFTWARE)


def resolve_software(values: Iterable[str] | None) -> tuple[str, ...]:
    """Expand repeatable/comma-separated software names and profiles."""
    if not values:
        return ()
    selected: list[str] = []
    for raw in values:
        for token in str(raw).split(","):
            name = token.strip().lower()
            if not name:
                continue
            expanded = SOFTWARE_PROFILES.get(name, (name,))
            for item in expanded:
                if item not in SOFTWARE:
                    allowed = ", ".join(sorted(set(SOFTWARE) | set(SOFTWARE_PROFILES)))
                    raise ValueError(f"Unknown software selection {item!r}; choose from: {allowed}")
                if item not in selected:
                    selected.append(item)
    index = 0
    while index < len(selected):
        name = selected[index]
        for dependency in SOFTWARE[name].requires:
            if dependency not in selected:
                selected.append(dependency)
        index += 1
    return tuple(selected)


def selected_host_envs(selected: Iterable[str]) -> tuple[str, ...]:
    names: list[str] = []
    for software in selected:
        for name in SOFTWARE[software].host_envs:
            if name not in names:
                names.append(name)
    return tuple(names)


def selected_submodules(selected: Iterable[str]) -> tuple[str, ...]:
    paths: list[str] = []
    for software in selected:
        for path in SOFTWARE[software].submodules:
            if path not in paths:
                paths.append(path)
    return tuple(paths)


def catalog_payload() -> dict:
    return {
        "schema_version": 1,
        "profiles": {key: list(value) for key, value in SOFTWARE_PROFILES.items()},
        "software": {key: asdict(value) for key, value in SOFTWARE.items()},
    }


def print_catalog(*, as_json: bool) -> None:
    if as_json:
        print(json.dumps(catalog_payload(), indent=2, sort_keys=True))
        return
    for name, spec in SOFTWARE.items():
        print(f"{name:16s} {spec.asset_policy:32s} {spec.summary}")
