from __future__ import annotations

from astrolabe.core import global_config


def test_runtime_context_defaults_for_uninitialized_state() -> None:
    global_config.reset()
    ctx = global_config.build_runtime_context()
    assert ctx.protomers == {}
    assert ctx.mutable_protomers == {}
    assert ctx.mutators == {}
    assert ctx.evaluators == {}
    assert ctx.stage_manager is None
    assert ctx.step == 0


def test_runtime_context_reflects_global_state() -> None:
    global_config.reset()
    ctx = global_config.build_runtime_context()
    global_config.init_const("protomers", {"A": object()})
    global_config.init_const("mutable_protomers", {"A": object()})
    ctx.set_variable("step", 7)
    assert set(ctx.protomers.keys()) == {"A"}
    assert set(ctx.mutable_protomers.keys()) == {"A"}
    assert ctx.step == 7
