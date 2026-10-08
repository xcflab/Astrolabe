from __future__ import annotations

from collections.abc import Callable

from astrolabe.initializers.base import Initializer, InitializerResult


class RandomInitializer(Initializer):
    def apply(self) -> InitializerResult:
        return InitializerResult(mode="random")


class _CallbackInitializer(Initializer):
    mode = ""

    def __init__(self, apply_fn: Callable[[], None]):
        self.apply_fn = apply_fn

    def apply(self) -> InitializerResult:
        self.apply_fn()
        return InitializerResult(mode=self.mode)


class DPLMInitializer(_CallbackInitializer):
    mode = "dplm"


class AminoAcidSetInitializer(_CallbackInitializer):
    mode = "aa_set"


class MaskMixInitializer(_CallbackInitializer):
    mode = "mask_mix"


class AllAInitializer(_CallbackInitializer):
    mode = "all_a"


class PSSMInitializer(_CallbackInitializer):
    mode = "pssm"


class XMixInitializer(_CallbackInitializer):
    mode = "x_mix"
