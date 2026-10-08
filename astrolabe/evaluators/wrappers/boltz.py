from __future__ import annotations

import inspect
import logging
import pathlib
import pickle
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import click
import torch
from lightning_fabric.utilities import rank_zero_only
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm

from astrolabe.core import global_config
from astrolabe.core.folding_input import ProteinChain, Ligand, RnaChain, DnaChain
from astrolabe.evaluators.x_masking import preserve_unknown_residue_sequence
try:
    from alphafold3.common.folding_input import (
        ProteinChain as AF3ProteinChain,
        Ligand as AF3Ligand,
        RnaChain as AF3RnaChain,
        DnaChain as AF3DnaChain,
    )
except ModuleNotFoundError:  # pragma: no cover
    AF3ProteinChain = None
    AF3Ligand = None
    AF3RnaChain = None
    AF3DnaChain = None

ProteinChainTypes = (ProteinChain,) if AF3ProteinChain is None else (ProteinChain, AF3ProteinChain)
LigandTypes = (Ligand,) if AF3Ligand is None else (Ligand, AF3Ligand)
RnaChainTypes = (RnaChain,) if AF3RnaChain is None else (RnaChain, AF3RnaChain)
DnaChainTypes = (DnaChain,) if AF3DnaChain is None else (DnaChain, AF3DnaChain)
from astrolabe.utils.mmcif_ids import int_id_to_str_id
from boltz.data import const
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml
from boltz.data.types import Manifest, Record, MSA
from boltz.data.write.writer import BoltzWriter
try:
    # Newer Boltz versions
    from boltz.model.model import Boltz1
except ModuleNotFoundError:
    # Older Boltz versions
    from boltz.model.models.boltz1 import Boltz1

CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MODEL_URL = (
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt"
)


@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


@rank_zero_only
def download(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        click.echo(
            f"Downloading the CCD dictionary to {ccd}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(CCD_URL, str(ccd))  # noqa: S310

    # Download model
    model = cache / "boltz1_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the model weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(MODEL_URL, str(model))  # noqa: S310


def check_inputs(
    data: Path,
    outdir: Path,
    override: bool = False,
) -> list[Path]:
    """Check the input data and output directory.

    If the input data is a directory, it will be expanded
    to all files in this directory. Then, we check if there
    are any existing predictions and remove them from the
    list of input data, unless the override flag is set.

    Parameters
    ----------
    data : Path
        The input data.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    click.echo("Checking input data.")

    # Check if data is a directory
    if data.is_dir():
        data: list[Path] = list(data.glob("*"))

        # Filter out non .fasta or .yaml files, raise
        # an error on directory and other file types
        filtered_data = []
        for d in data:
            if d.suffix in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                filtered_data.append(d)
            elif d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            else:
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)

        data = filtered_data
    else:
        data = [data]

    # Check if existing predictions are found
    existing = (outdir / "predictions").rglob("*")
    existing = {e.name for e in existing if e.is_dir()}

    # Remove them from the input data
    if existing and not override:
        data = [d for d in data if d.stem not in existing]
        num_skipped = len(existing) - len(data)
        msg = (
            f"Found some existing predictions ({num_skipped}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override:
        msg = "Found existing predictions, will override."
        click.echo(msg)

    return data


def compute_msa(
    data: dict[str, str],
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
) -> list[Path]:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    msa_dir : Path
        The msa temp directory.

    Returns
    -------
    list[Path]
        The list of MSA files.

    """
    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            msa_dir / "tmp",
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        msa_dir / "tmp",
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
    )

    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[: const.max_paired_seqs]

        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{name}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))


@rank_zero_only
def process_inputs(  # noqa: C901, PLR0912, PLR0915
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int = 4096,
    use_msa_server: bool = False,
) -> None:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA seuqneces, by default 4096.
    use_msa_server : bool, optional
        Whether to use the MMSeqs2 server for MSA generation, by default False.

    Returns
    -------
    BoltzProcessedInput
        The processed input data.

    """
    click.echo("Processing input data.")

    # Create output directories
    msa_dir = out_dir / "msa"
    mol_dir = out_dir / "mols"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    mol_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    with ccd_path.open("rb") as file:
        ccd = pickle.load(file)  # noqa: S301

    # Parse input data
    records: list[Record] = []
    for path in tqdm(data):
        # Parse data
        if path.suffix in (".fa", ".fas", ".fasta"):
            if len(inspect.signature(parse_fasta).parameters) >= 3:
                target = parse_fasta(path, ccd, mol_dir)
            else:
                target = parse_fasta(path, ccd)
        elif path.suffix in (".yml", ".yaml"):
            if len(inspect.signature(parse_yaml).parameters) >= 3:
                target = parse_yaml(path, ccd, mol_dir)
            else:
                target = parse_yaml(path, ccd)
        elif path.is_dir():
            msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
            raise RuntimeError(msg)
        else:
            msg = (
                f"Unable to parse filetype {path.suffix}, "
                "please provide a .fasta or .yaml file."
            )
            raise RuntimeError(msg)

        # Get target id
        target_id = target.record.id

        # Get all MSA ids and decide whether to generate MSA
        to_generate = {}
        prot_id = const.chain_type_ids["PROTEIN"]
        for chain in target.record.chains:
            # Add to generate list, assigning entity id
            if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                entity_id = chain.entity_id
                msa_id = f"{target_id}_{entity_id}"
                to_generate[msa_id] = target.sequences[entity_id]
                chain.msa_id = msa_dir / f"{msa_id}.csv"

            # We do not support msa generation for non-protein chains
            elif chain.msa_id == 0:
                chain.msa_id = -1

        # Generate MSA
        if to_generate and not use_msa_server:
            msg = "Missing MSA's in input and --use_msa_server flag not set."
            raise RuntimeError(msg)

        if to_generate:
            msg = f"Generating MSA for {path} with {len(to_generate)} protein entities."
            click.echo(msg)
            compute_msa(
                to_generate,
                msa_dir,
                msa_server_url=msa_server_url,
                msa_pairing_strategy=msa_pairing_strategy,
            )

        # Parse MSA data
        msas = {c.msa_id for c in target.record.chains if c.msa_id != -1}
        msa_id_map = {}
        for msa_idx, msa_id in enumerate(msas):
            # Check that raw MSA exists
            msa_path = Path(msa_id)
            if not msa_path.exists():
                msg = f"MSA file {msa_path} not found."
                raise FileNotFoundError(msg)

            # Dump processed MSA
            processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
            msa_id_map[msa_id] = f"{target_id}_{msa_idx}"
            if not processed.exists():
                # Parse A3M
                if msa_path.suffix == ".a3m":
                    msa: MSA = parse_a3m(
                        msa_path,
                        taxonomy=None,
                        max_seqs=max_msa_seqs,
                    )
                elif msa_path.suffix == ".csv":
                    msa: MSA = parse_csv(msa_path)
                else:
                    msg = f"MSA file {msa_path} not supported, only a3m or csv."
                    raise RuntimeError(msg)

                msa.dump(processed)

        # Modify records to point to processed MSA
        for c in target.record.chains:
            if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                c.msa_id = msa_id_map[c.msa_id]

        # Keep record
        records.append(target.record)

        # Dump structure
        struct_path = structure_dir / f"{target.record.id}.npz"
        target.structure.dump(struct_path)

    # Dump manifest
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")


@click.group()
def cli() -> None:
    """Boltz1."""
    return


def _normalize_msa_text(msa: str | None) -> str:
    text = str(msa or "").strip()
    if not text:
        return ""
    return text if text.endswith("\n") else f"{text}\n"


def _protein_msa_path_for_boltz(*, cplx_name: str, chain_id: str) -> Path:
    return (
        Path(global_config.out_dir)
        / "sequences"
        / "boltz_msa"
        / str(cplx_name)
        / f"{chain_id}.a3m"
    )


def _resolve_protein_msa_for_boltz(*, chain, cplx_name: str, chain_id: str) -> str:
    for raw in (getattr(chain, "unpaired_msa", None), getattr(chain, "paired_msa", None)):
        candidate = str(raw or "").strip()
        if not candidate or "\n" in candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.is_file() and candidate_path.suffix in {".a3m", ".csv"}:
            return str(candidate_path)

    msa_text = _normalize_msa_text(getattr(chain, "unpaired_msa", None))
    if not msa_text:
        msa_text = _normalize_msa_text(getattr(chain, "paired_msa", None))
    if not msa_text:
        return "empty"

    msa_path = _protein_msa_path_for_boltz(cplx_name=cplx_name, chain_id=chain_id)
    msa_path.parent.mkdir(parents=True, exist_ok=True)
    if msa_path.exists() and msa_path.read_text(encoding="utf-8") == msa_text:
        return str(msa_path)

    msa_path.write_text(msa_text, encoding="utf-8")
    return str(msa_path)


def _collect_head_to_tail_cyclic_chains_for_boltz(*, input_obj, protein_lengths: dict[str, int]) -> set[str]:
    bonded = list(getattr(input_obj, "bonded_atom_pairs", None) or [])
    bond_meta = list(getattr(input_obj, "bonded_atom_pair_meta", None) or [])
    if not bonded:
        return set()

    out: set[str] = set()
    for idx, pair in enumerate(bonded):
        if (
            not isinstance(pair, (tuple, list))
            or len(pair) != 2
            or not isinstance(pair[0], (tuple, list))
            or not isinstance(pair[1], (tuple, list))
            or len(pair[0]) != 3
            or len(pair[1]) != 3
        ):
            continue
        atom1, atom2 = pair
        c1, r1, a1 = atom1
        c2, r2, a2 = atom2
        c1 = str(c1)
        c2 = str(c2)
        if c1 != c2:
            continue
        if c1 not in protein_lengths:
            continue

        chain_len = int(protein_lengths[c1])
        try:
            ends = {int(r1), int(r2)}
        except Exception:
            continue
        atoms = {str(a1).strip().upper(), str(a2).strip().upper()}
        inferred_head_to_tail = ends == {1, chain_len} and atoms == {"N", "C"}

        meta = bond_meta[idx] if idx < len(bond_meta) else None
        meta_head_to_tail = (
            bool(meta.get("is_head_to_tail")) if isinstance(meta, dict) else None
        )
        if meta_head_to_tail is True and inferred_head_to_tail:
            out.add(c1)
        elif meta_head_to_tail is None and inferred_head_to_tail:
            out.add(c1)

    return out


def from_af3_to_boltz(input_obj):
    sequences = []
    # fixme: boltz cannot handle multiple ccd codes for a single ligand, try to specify the chain name
    chain_idx = 1
    chain2input_chain = {}
    cplx_name = str(getattr(input_obj, "name", "") or "default")
    if hasattr(input_obj, "chains"):
        chains = list(input_obj.chains)
    else:
        chains = (
            input_obj.protein_chains
            + input_obj.rna_chains
            + input_obj.dna_chains
            + input_obj.ligands
        )
    cyclic_chain_ids = {
        str(chain_id)
        for chain_id in (getattr(input_obj, "cyclic_chain_ids", None) or ())
    }

    protein_lengths: dict[str, int] = {}
    for chain in chains:
        if isinstance(chain, ProteinChainTypes):
            protein_sequence = preserve_unknown_residue_sequence(
                chain.sequence, evaluator="boltz"
            )
            protein_lengths[str(chain.id)] = int(len(protein_sequence))
    cyclic_chain_ids.update(
        _collect_head_to_tail_cyclic_chains_for_boltz(
            input_obj=input_obj,
            protein_lengths=protein_lengths,
        )
    )

    for chain in chains:
        if isinstance(chain, ProteinChainTypes):
            protein_sequence = preserve_unknown_residue_sequence(
                chain.sequence, evaluator="boltz"
            )
            input_chain_id = int_id_to_str_id(chain_idx)
            modifications = []
            for ccd, pos in chain.ptms:
                modifications.append({"position": pos, "ccd": ccd})
            protein_payload = {
                "id": input_chain_id,
                "sequence": protein_sequence,
                "modifications": modifications,
                "msa": _resolve_protein_msa_for_boltz(
                    chain=chain,
                    cplx_name=cplx_name,
                    chain_id=input_chain_id,
                ),
            }
            if str(getattr(chain, "id", "")) in cyclic_chain_ids:
                protein_payload["cyclic"] = True
            sequences.append(
                {
                    "protein": protein_payload
                }
            )
            chain2input_chain[input_chain_id] = chain.id
            chain_idx += 1
        elif isinstance(chain, LigandTypes):
            if chain.smiles is not None:
                sequences.append(
                    {
                        "ligand": {
                            "id": int_id_to_str_id(chain_idx),
                            "smiles": chain.smiles,
                        }
                    }
                )
                chain2input_chain[int_id_to_str_id(chain_idx)] = chain.id
                chain_idx += 1
            elif chain.ccd_ids is not None:
                for ccd in chain.ccd_ids:
                    sequences.append(
                        {
                            "ligand": {
                                "id": int_id_to_str_id(chain_idx),
                                "ccd": ccd,
                            }
                        }
                    )
                    chain2input_chain[int_id_to_str_id(chain_idx)] = chain.id
                    chain_idx += 1
            else:
                raise ValueError("Ligand must have either CCD or SMILES")

        elif isinstance(chain, RnaChainTypes):
            sequences.append(
                {
                    "rna": {
                        "id": int_id_to_str_id(chain_idx),
                        "sequence": chain.sequence,
                    }
                }
            )
            chain2input_chain[int_id_to_str_id(chain_idx)] = chain.id
            chain_idx += 1
        elif isinstance(chain, DnaChainTypes):
            sequences.append(
                {
                    "dna": {
                        "id": int_id_to_str_id(chain_idx),
                        "sequence": chain.sequence,
                    }
                }
            )
            chain2input_chain[int_id_to_str_id(chain_idx)] = chain.id
            chain_idx += 1
        else:
            raise ValueError(f"Unknown chain type {type(chain)}")
    result = {"sequences": sequences, "constraints": [], "version": 1}

    from astrolabe.core.binding_site import parse_binding_site
    from astrolabe.evaluators.binding_site_constraints import (
        build_boltz_binding_site_constraints,
    )

    spec = parse_binding_site(
        getattr(global_config, "binding_site", None),
        oligo_name=getattr(input_obj, "name", None),
    )
    if spec is not None:
        result["constraints"].extend(
            build_boltz_binding_site_constraints(
                spec,
                chain2input_chain=chain2input_chain,
                context=getattr(input_obj, "name", "") or "<boltz_input>",
                oligo_name=getattr(input_obj, "name", None),
            )
        )

    bonded = getattr(input_obj, "bonded_atom_pairs", None) or []
    if bonded:
        from astrolabe.evaluators.binding_site_constraints import (
            invert_chain2input_chain_unique,
        )

        orig_to_input = invert_chain2input_chain_unique(
            chain2input_chain,
            context=getattr(input_obj, "name", "") or "<boltz_input>",
        )
        for atom1, atom2 in bonded:
            c1, r1, a1 = atom1
            c2, r2, a2 = atom2
            nc1 = orig_to_input.get(str(c1))
            nc2 = orig_to_input.get(str(c2))
            if nc1 is None or nc2 is None:
                raise ValueError(
                    f"Boltz bond references unknown chains: {c1!r}->{nc1!r}, {c2!r}->{nc2!r}"
                )
            result["constraints"].append(
                {
                    "bond": {
                        "atom1": [str(nc1), int(r1), str(a1)],
                        "atom2": [str(nc2), int(r2), str(a2)],
                    }
                }
            )
    cyclic_relative_patches = getattr(input_obj, "cyclic_relative_patches", None) or ()
    if cyclic_relative_patches:
        result["inference_options"] = {
            "cyclic_relative_entries": [
                {
                    "chain": str(patch.chain_id),
                    "residue_index_1": int(rel.residue_i),
                    "residue_index_2": int(rel.residue_j),
                    "offset": int(rel.offset),
                }
                for patch in cyclic_relative_patches
                for rel in patch.offsets
            ]
        }
    return result, chain2input_chain


class Boltz1Runner:
    def __init__(self, config):
        self.config = config
        self.accelerator = config["accelerator"]
        self.cache = pathlib.Path(config["cache"])
        self.checkpoint = config["checkpoint"]
        self.devices = config["devices"]
        self.diffusion_samples = config["diffusion_samples"]
        self.num_workers = config["num_workers"]
        self.output_format = config["output_format"]
        self.recycling_steps = config["recycling_steps"]
        self.sampling_steps = config["sampling_steps"]
        self.write_full_pae = config["write_full_pae"]
        self.write_full_pde = config["write_full_pde"]
        self.use_msa_server = config["use_msa_server"]
        self.msa_server_url = config["msa_server_url"] or "https://api.colabfold.com"
        self.msa_pairing_strategy = config["msa_pairing_strategy"]

    def run_inference(self, data, out_dir):
        self.run_prediction(data, out_dir)

    def run_prediction(
        self,
        data,
        out_dir,
    ):
        # Load processed data
        processed_dir = out_dir / "processed"
        processed = BoltzProcessedInput(
            manifest=Manifest.load(processed_dir / "manifest.json"),
            targets_dir=processed_dir / "structures",
            msa_dir=processed_dir / "msa",
        )
        # Create data module
        data_module = BoltzInferenceDataModule(
            manifest=processed.manifest,
            target_dir=processed.targets_dir,
            msa_dir=processed.msa_dir,
            num_workers=self.num_workers,
        )

        strategy = "auto"
        devices = self.devices
        if (isinstance(devices, int) and devices > 1) or (
            isinstance(devices, list) and len(devices) > 1
        ):
            strategy = DDPStrategy()
            if len(data) < devices:
                msg = (
                    "Number of requested devices is greater "
                    "than the number of predictions."
                )
                raise ValueError(msg)
        self.load_model_once()
        # Create prediction writer
        pred_writer = BoltzWriter(
            data_dir=processed.targets_dir,
            output_dir=out_dir / "predictions",
            output_format=self.output_format,
        )
        trainer = Trainer(
            default_root_dir=out_dir,
            strategy=strategy,
            callbacks=[pred_writer],
            accelerator=self.accelerator,
            devices=devices,
            precision=32,
        )
        # Compute predictions
        trainer.predict(
            self.model_module,
            datamodule=data_module,
            return_predictions=False,
        )

    def load_model_once(
        self,
    ):
        logging.info("Loading Boltz model.")
        # Load model
        if self.checkpoint is None:
            checkpoint = self.cache / "boltz1_conf.ckpt"
        else:
            checkpoint = pathlib.Path(self.checkpoint)
        if not hasattr(self, "model_module"):
            predict_args = {
                "recycling_steps": self.recycling_steps,
                "sampling_steps": self.sampling_steps,
                "diffusion_samples": self.diffusion_samples,
                "write_confidence_summary": True,
                "write_full_pae": self.write_full_pae,
                "write_full_pde": self.write_full_pde,
            }
            map_location = "cpu"
            if self.accelerator != "cpu" and torch.cuda.is_available():
                map_location = torch.device("cuda:0")
            model_module: Boltz1 = Boltz1.load_from_checkpoint(
                checkpoint,
                strict=True,
                predict_args=predict_args,
                map_location=map_location,
                diffusion_process_args=asdict(BoltzDiffusionParams()),
                ema=False,
            )
            model_module.eval()
            self.model_module = model_module

    def featurize_data(self, data, out_dir):
        # Validate inputs
        data = check_inputs(data, out_dir, False)
        if not data:
            click.echo("No predictions to run, exiting.")
            return []

        msg = f"Running predictions for {len(data)} structure"
        msg += "s" if len(data) > 1 else ""
        click.echo(msg)

        cache = Path(self.cache).expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        download(cache)
        # Process inputs
        ccd_path = cache / "ccd.pkl"
        process_inputs(
            data=data,
            out_dir=out_dir,
            ccd_path=ccd_path,
            use_msa_server=self.use_msa_server,
            msa_server_url=self.msa_server_url,
            msa_pairing_strategy=self.msa_pairing_strategy,
        )
        return data

    def setup_model(self):
        # If cpu, write a friendly warning
        if self.accelerator == "cpu":
            msg = "Running on CPU, this will be slow. Consider using a GPU."
            click.echo(msg)
        # Set no grad
        torch.set_grad_enabled(False)
        # Ignore matmul precision warning
        torch.set_float32_matmul_precision("highest")
        # Set cache path
        cache = Path(self.cache).expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        download(cache)

        return cache
