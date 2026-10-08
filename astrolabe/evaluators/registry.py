from __future__ import annotations

import logging
from typing import Callable, Dict, List


EvaluatorFactory = Callable[..., object]


class EvaluatorRegistry:
    """Registry for evaluator factories.

    Evaluators often require backend-specific runners and optional dependencies.
    To keep CLI startup fast and allow extension, we register lightweight factory
    callables keyed by evaluator `_type` (case-insensitive).
    """

    _registry: Dict[str, EvaluatorFactory] = {}

    @staticmethod
    def _key(name: str) -> str:
        return str(name).strip().lower()

    @classmethod
    def register(cls, name: str, factory: EvaluatorFactory) -> None:
        key = cls._key(name)
        previous = cls._registry.get(key)
        if previous is not None and previous is not factory:
            logging.debug(
                "Overriding evaluator factory for %s: %s -> %s",
                key,
                getattr(previous, "__name__", type(previous).__name__),
                getattr(factory, "__name__", type(factory).__name__),
            )
        cls._registry[key] = factory
        logging.debug(
            "Registered evaluator factory: %s -> %s",
            key,
            getattr(factory, "__name__", type(factory).__name__),
        )

    @classmethod
    def get(cls, name: str) -> EvaluatorFactory:
        key = cls._key(name)
        if key not in cls._registry:
            raise KeyError(f"Evaluator type '{name}' not found in registry")
        return cls._registry[key]

    @classmethod
    def list_available(cls) -> List[str]:
        return sorted(cls._registry.keys())


def register_evaluator_factory(name: str):
    """Decorator to register an evaluator factory under a given `_type`."""

    def decorator(factory: EvaluatorFactory) -> EvaluatorFactory:
        EvaluatorRegistry.register(name, factory)
        return factory

    return decorator
