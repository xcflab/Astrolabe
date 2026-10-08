#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import sys

# Allow running without `pip install -e .`
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from alphafold3.common.folding_input import Input, ProteinChain

from astrolabe.core import global_config
from astrolabe.core.msa_utils import parse_homology_arg
from astrolabe.evaluators.wrappers.protenix import (
    from_af3_to_protenix,
    protenix_from_dict_to_feature,
)


def _count_a3m_records(a3m_text: str | None) -> int:
    if not a3m_text:
        return 0
    return sum(1 for line in a3m_text.splitlines() if line.lstrip().startswith(">"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke-test AF3 JSON homology/MSA -> Protenix precomputed MSA + features."
    )
    p.add_argument(
        "--json",
        required=True,
        help="Path to AlphaFold3 JSON (dialect) containing unpairedMsa/pairedMsa.",
    )
    p.add_argument(
        "--chain",
        default="A",
        help="Chain id within the AF3 JSON to use (e.g. A).",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output dir for generated MSA files (default: temp under /tmp).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    json_path = Path(args.json).expanduser().resolve()
    if not json_path.is_file():
        raise FileNotFoundError(json_path)

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else Path(tempfile.mkdtemp(prefix="gh_af3json_msa_", dir="/tmp"))
    )

    # Extract chain sequence + MSAs from AF3 JSON via the repo's parser.
    # In binder benchmarks, the target chain is fully fixed, so we mimic that by
    # marking all residues as fixed for masking logic.
    import json as _json

    data = _json.loads(json_path.read_text(encoding="utf-8"))
    seq = None
    for item in data.get("sequences", []):
        if "protein" not in item:
            continue
        prot = item["protein"]
        if args.chain in (prot.get("id") or []):
            seq = prot.get("sequence")
            break
    if not seq:
        raise ValueError(f"Chain {args.chain!r} not found in {json_path}")

    fixed = list(range(len(seq)))
    homo = parse_homology_arg(
        seq,
        fixed,
        f"{json_path}:{args.chain}",
    )
    unpaired = homo.get("unpairedMsa")
    paired = homo.get("pairedMsa")

    if _count_a3m_records(unpaired) <= 1:
        raise AssertionError("unpairedMsa missing/too shallow in AF3 JSON extract")
    if _count_a3m_records(paired) <= 1:
        raise AssertionError("pairedMsa missing/too shallow in AF3 JSON extract")

    # Minimal global_config for the Protenix wrappers (only used for output paths).
    global_config.reset()
    global_config.init_const("out_dir", out_dir)
    global_config.set_variable("step", 0)

    # Build an AF3 Input (single protein chain) and stage it for Protenix.
    chain = ProteinChain(
        id="B",
        sequence=seq,
        ptms=[],
        unpaired_msa=unpaired or "",
        paired_msa=paired or "",
        templates=homo.get("templates") or [],
    )
    inp = Input(
        chains=[chain],
        name="B",
        rng_seeds=[0],
        bonded_atom_pairs=None,
        user_ccd=None,
    )

    sample_dict = from_af3_to_protenix(inp, "B")
    msa_dir = Path(sample_dict["sequences"][0]["proteinChain"]["msa"]["precomputed_msa_dir"])
    non_pair = (msa_dir / "non_pairing.a3m").read_text(encoding="utf-8")
    pair = (msa_dir / "pairing.a3m").read_text(encoding="utf-8")

    if _count_a3m_records(non_pair) <= 1:
        raise AssertionError("Protenix non_pairing.a3m not written or too shallow")
    if _count_a3m_records(pair) <= 1:
        raise AssertionError("Protenix pairing.a3m not written or too shallow")

    data_dict, *_ = protenix_from_dict_to_feature(sample_dict)
    msa_feat = data_dict["input_feature_dict"].get("msa")
    if msa_feat is None:
        raise AssertionError("Protenix feature dict missing 'msa'")
    if int(msa_feat.shape[0]) <= 1:
        raise AssertionError(f"Protenix msa feature depth too small: {tuple(msa_feat.shape)}")

    print("OK")
    print("out_dir:", out_dir)
    print("json:", json_path)
    print("chain:", args.chain)
    print("msa_dir:", msa_dir)
    print("unpaired_records:", _count_a3m_records(unpaired))
    print("paired_records:", _count_a3m_records(paired))
    print("msa_feat_shape:", tuple(msa_feat.shape))


if __name__ == "__main__":
    main()
