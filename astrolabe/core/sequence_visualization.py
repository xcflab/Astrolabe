from __future__ import annotations

import logging
from typing import Dict, List

import termcolor


def log_fixed_regions(fix_idx: Dict[str, List[int]], seqs: Dict[str, str]) -> None:
    """Print sequences with fixed regions highlighted."""
    for key, idx_ls in fix_idx.items():
        colored_fix_region = "".join(
            [
                termcolor.colored(aa, "red") if idx in idx_ls else aa
                for idx, aa in enumerate(seqs[key])
            ]
        )
        logging.info(f"fix regions: {key} => {idx_ls} => {colored_fix_region}")


def log_colored_sequences(colored_seqs: Dict[str, List[str]]) -> None:
    """Print sequences with color coding."""
    for key, seq in colored_seqs.items():
        logging.info(f"{key} => {'/'.join(seq)}")
