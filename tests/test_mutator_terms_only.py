from pathlib import Path

from omegaconf import OmegaConf

from astrolabe.mutators.base import build_mutators


def test_build_mutators_ignores_mutator_names(tmp_path: Path) -> None:
    # `mutator.names` is legacy/ignored. Mutators should be built from `mutator_terms`
    # (optionally restricted via enabled_aliases derived from stages).
    mutator_conf = OmegaConf.create(
        {
            "names": ["MCMC"],  # should not gate which aliases are built
            "enabled_aliases": ["X"],
            "mutator_terms": {
                "X": {
                    "_type": "MCMC",
                }
            },
            "MCMC": {
                "mutation_method": "frequency_adjusted",
                "select_positions": "plddt",
            },
        }
    )

    built = build_mutators(mutator_conf, tmp_path)
    assert "X" in built

