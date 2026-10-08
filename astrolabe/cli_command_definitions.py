from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any


OUTPUT_FORMAT_CHOICES = ("text", "json")
COMPLETION_SHELL_CHOICES = ("bash", "zsh")
APTAMER_TYPE_CHOICES = ("rna", "dna")
DOMAIN_SPLIT_MODE_CHOICES = ("full", "template")
RUN_SELECTION_CHOICES = ("all-passing", "representative-per-run")
DELETE_SCOPE_CHOICES = ("raw_model_results", "run_dir")
CONFIG_BASE_CHOICES = ("base", "template")
SYMMETRY_PREDOCKER_BACKEND_CHOICES = ("legacy", "rfdiffusion", "rpxdock")
SYMMETRY_INTERFACE_SELECTION_CHOICES = ("auto", "manual")
REFINE_LIGAND_POCKET_SCORING_MODE_CHOICES = ("vina", "retarget")

CYCLIC_ENCODING_CHOICES = (
    "head-to-tail",
    "head_to_tail",
    "headtail",
    "all",
    "all_bonds",
    "all-bonds",
    "global",
    "global-equivalent",
    "global_equivalent",
    "global-generic",
    "global_generic",
    "rfpeptide",
    "rfpeptide-like",
    "rfpeptide_like",
    "highfold",
    "highfold-cycpoem",
    "highfold_cycpoem",
    "cycpoem",
    "off",
    "none",
    "false",
    "0",
)


def set_handler(
    parser: argparse.ArgumentParser,
    handler: Callable[[argparse.Namespace], int],
    *,
    parser_ref: bool = False,
    **defaults: Any,
) -> None:
    values = {**defaults, "_handler": handler, "_handler_registered": handler}
    if parser_ref:
        values["_parser"] = parser
    parser.set_defaults(**values)


def add_completion_choices(
    action: argparse.Action,
    choices: Sequence[str],
) -> argparse.Action:
    action.completion_choices = tuple(choices)
    return action


def add_output_format_argument(parser: argparse.ArgumentParser) -> argparse.Action:
    return parser.add_argument(
        "--format",
        choices=OUTPUT_FORMAT_CHOICES,
        default="text",
        help="Output format.",
    )


def add_run_selection_argument(
    parser: argparse.ArgumentParser,
    *,
    help: str,
) -> argparse.Action:
    return parser.add_argument(
        "--run-selection",
        choices=RUN_SELECTION_CHOICES,
        default="all-passing",
        help=help,
    )


def add_cyclic_encoding_argument(parser: argparse.ArgumentParser) -> argparse.Action:
    return parser.add_argument(
        "--cyclic-encoding",
        default=None,
        choices=CYCLIC_ENCODING_CHOICES,
        help=(
            "Cyclic encoding scope for cyclization offsets: "
            "head-to-tail|all-bonds|rfpeptide-like|highfold-cycpoem|off."
        ),
    )


def add_aptamer_type_argument(parser: argparse.ArgumentParser) -> argparse.Action:
    return parser.add_argument(
        "--aptamer-type",
        default="rna",
        choices=APTAMER_TYPE_CHOICES,
        help="Aptamer chain A polymer type: rna|dna (default: rna).",
    )
