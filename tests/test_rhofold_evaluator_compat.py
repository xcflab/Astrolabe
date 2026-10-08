from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "vendor" / "rhofold"))

from astrolabe.core.folding_input import RnaChain
from astrolabe.evaluators import rhofold as rhofold_evaluator


def test_rhofold_accepts_lightweight_gh_rna_chains() -> None:
    assert RnaChain in rhofold_evaluator.RnaChainTypes


def test_rhofold_does_not_import_optional_relaxation_stack_eagerly() -> None:
    assert "rhofold.relax.relax" not in sys.modules
