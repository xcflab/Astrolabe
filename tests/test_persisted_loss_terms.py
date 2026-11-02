from omegaconf import OmegaConf

from astrolabe.losses.base import LossEngine, loss_registry
from astrolabe.run_config_lifecycle import build_persisted_run_config


@loss_registry.register(name="__persisted_config_test_loss__", optional_params=[])
def _test_loss(result):
    return {name: 0.25 for name in result.names}


class _Scores:
    names = ["AB"]


def test_persisted_config_preserves_losses_used_outside_stage_expression():
    config = OmegaConf.create({
        "stages": {"default": {"weighted_score_expression": "primary"}},
        "loss": {"loss_terms": {
            "primary": {"_type": "__persisted_config_test_loss__"},
            "report_only": {"_type": "__persisted_config_test_loss__"},
            "disabled": None,
        }},
    })
    saved = build_persisted_run_config(config)
    reloaded = OmegaConf.create(OmegaConf.to_yaml(saved))
    engine = LossEngine(reloaded.loss, validate_config=False)
    engine.setup("primary", None, None, validate_config=False)
    _, scores = engine._extract_scores_with_registry(_Scores())

    assert scores["primary"] == 0.25
    assert scores["report_only"] == 0.25
    assert "disabled" not in scores
