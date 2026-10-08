from __future__ import annotations

from pathlib import Path

import astrolabe.cli as cli


CONFIG_DIR = Path(cli.__file__).resolve().parents[1] / "config"

SYMMETRY_CONFIGS = (
    "benchmark_pseudocycle_v2",
    "benchmark_tandem_repeat_v2",
    "benchmark_oligomer_v4_refine",
    "benchmark_dihedral_d2_dplm_protenix_af2_rpxdock_v1",
    "benchmark_dihedral_d3_dplm_protenix_af2_rpxdock_v1",
    "benchmark_oligomer_morphology_cyclic_v1",
    "benchmark_dihedral_d4_dplm_protenix_af2_rpxdock_v1",
)


def _compose(name: str, overrides: list[str] | None = None):
    return cli._compose_hydra_config(
        config_dir=CONFIG_DIR,
        config_name=name,
        overrides=overrides or ["env=env"],
    )


def test_symmetry_configs_compose_with_resolved_stage_aliases() -> None:
    for config_name in SYMMETRY_CONFIGS:
        cfg = _compose(config_name)
        evaluator_aliases = set((cfg.evaluator.evaluator_terms or {}).keys())
        mutator_aliases = set((cfg.mutator.mutator_terms or {}).keys())

        for stage in cfg.stages.values():
            if getattr(stage, "evaluator", None) is not None:
                assert stage.evaluator in evaluator_aliases, config_name
            if getattr(stage, "mutators", None) is not None:
                assert set(stage.mutators) <= mutator_aliases, config_name


def test_oligomer_refine_symmetry_predocker_does_not_pin_one_oligo() -> None:
    cfg = _compose(
        "benchmark_oligomer_v4_refine",
        overrides=[
            "env=env",
            "stages.default.oligo=[AA,AAA]",
            "stages.refine.oligo=[AA,AAA]",
        ],
    )

    assert getattr(cfg.predocker.predocker_terms.symmetry, "oligo", None) is None
