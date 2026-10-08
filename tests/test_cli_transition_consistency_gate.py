from __future__ import annotations

from astrolabe.cli import _compose_validate_basic


def test_compose_validate_basic_accepts_stage_transition_consistency_gate() -> None:
    overlay = {
        "evaluator": {"evaluator_terms": {"Eval": {"_type": "esmfold"}}},
        "mutator": {"mutator_terms": {"M": {"_type": "mcmc"}}},
        "stages": {
            "default": {
                "evaluator": "Eval",
                "mutators": ["M"],
            },
            "stage_1": {
                "evaluator": "Eval",
                "mutators": ["M"],
                "transition_consistency_gate": {
                    "weight": 0.5,
                    "consistency_threshold": 0.8,
                    "max_attempts": 3,
                },
            },
        },
    }

    assert _compose_validate_basic(overlay) == []


def test_compose_validate_basic_accepts_stage_transition_consistency_gate_without_weight() -> None:
    overlay = {
        "evaluator": {"evaluator_terms": {"Eval": {"_type": "esmfold"}}},
        "mutator": {"mutator_terms": {"M": {"_type": "mcmc"}}},
        "stages": {
            "default": {
                "evaluator": "Eval",
                "mutators": ["M"],
            },
            "stage_1": {
                "evaluator": "Eval",
                "mutators": ["M"],
                "transition_consistency_gate": {
                    "consistency_threshold": 0.8,
                    "max_attempts": 3,
                },
            },
        },
    }

    assert _compose_validate_basic(overlay) == []


def test_compose_validate_basic_accepts_stage_transition_consistency_gate_evaluator() -> None:
    overlay = {
        "evaluator": {
            "evaluator_terms": {
                "AF3": {"_type": "Alphafold3"},
                "BaseProtenix": {"_type": "Protenix"},
            }
        },
        "mutator": {"mutator_terms": {"M": {"_type": "mcmc"}}},
        "stages": {
            "default": {
                "evaluator": "BaseProtenix",
                "mutators": ["M"],
            },
            "stage_1": {
                "evaluator": "AF3",
                "mutators": ["M"],
                "transition_consistency_gate": {
                    "evaluator": "BaseProtenix",
                    "consistency_threshold": 0.8,
                    "max_attempts": 3,
                },
            },
        },
    }

    assert _compose_validate_basic(overlay) == []


def test_compose_validate_basic_rejects_unknown_transition_consistency_gate_evaluator() -> None:
    overlay = {
        "evaluator": {"evaluator_terms": {"AF3": {"_type": "Alphafold3"}}},
        "mutator": {"mutator_terms": {"M": {"_type": "mcmc"}}},
        "stages": {
            "default": {
                "evaluator": "AF3",
                "mutators": ["M"],
            },
            "stage_1": {
                "evaluator": "AF3",
                "mutators": ["M"],
                "transition_consistency_gate": {
                    "evaluator": "MissingProtenix",
                    "consistency_threshold": 0.8,
                    "max_attempts": 3,
                },
            },
        },
    }

    assert (
        "stages.stage_1.transition_consistency_gate.evaluator references unknown alias: MissingProtenix"
        in _compose_validate_basic(overlay)
    )


def test_compose_validate_basic_rejects_default_stage_transition_consistency_gate() -> None:
    overlay = {
        "evaluator": {"evaluator_terms": {"Eval": {"_type": "esmfold"}}},
        "mutator": {"mutator_terms": {"M": {"_type": "mcmc"}}},
        "stages": {
            "default": {
                "evaluator": "Eval",
                "mutators": ["M"],
                "transition_consistency_gate": {
                    "weight": 0.5,
                    "consistency_threshold": 0.8,
                },
            },
        },
    }

    assert (
        "stages.default.transition_consistency_gate is not allowed"
        in _compose_validate_basic(overlay)
    )


def test_compose_validate_basic_rejects_invalid_transition_consistency_gate_max_attempts() -> None:
    overlay = {
        "evaluator": {"evaluator_terms": {"Eval": {"_type": "esmfold"}}},
        "mutator": {"mutator_terms": {"M": {"_type": "mcmc"}}},
        "stages": {
            "default": {
                "evaluator": "Eval",
                "mutators": ["M"],
            },
            "stage_1": {
                "evaluator": "Eval",
                "mutators": ["M"],
                "transition_consistency_gate": {
                    "weight": 0.5,
                    "consistency_threshold": 0.8,
                    "max_attempts": 0,
                },
            },
        },
    }

    assert (
        "stages.stage_1.transition_consistency_gate.max_attempts must be an integer >= 1"
        in _compose_validate_basic(overlay)
    )


def test_compose_validate_basic_rejects_legacy_extra_template_key() -> None:
    overlay = {
        "evaluator": {"evaluator_terms": {"Eval": {"_type": "esmfold"}}},
        "mutator": {"mutator_terms": {"M": {"_type": "mcmc"}}},
        "stages": {
            "default": {
                "evaluator": "Eval",
                "mutators": ["M"],
            },
            "stage_1": {
                "evaluator": "Eval",
                "mutators": ["M"],
                "extra_template": {
                    "weight": 0.5,
                    "consistency_threshold": 0.8,
                },
            },
        },
    }

    errors = _compose_validate_basic(overlay)
    assert any("extra_template" in err for err in errors)
