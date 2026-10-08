from __future__ import annotations

import logging
from typing import Callable, Dict, List


MutatorFactory = Callable[..., object]


class MutatorRegistry:
    """Registry for mutator factories keyed by mutator `_type` (case-insensitive)."""

    _registry: Dict[str, MutatorFactory] = {}

    @staticmethod
    def _key(name: str) -> str:
        return str(name).strip().lower()

    @classmethod
    def register(cls, name: str, factory: MutatorFactory) -> None:
        key = cls._key(name)
        previous = cls._registry.get(key)
        if previous is not None and previous is not factory:
            logging.debug(
                "Overriding mutator factory for %s: %s -> %s",
                key,
                getattr(previous, "__name__", type(previous).__name__),
                getattr(factory, "__name__", type(factory).__name__),
            )
        cls._registry[key] = factory
        logging.debug(
            "Registered mutator factory: %s -> %s",
            key,
            getattr(factory, "__name__", type(factory).__name__),
        )

    @classmethod
    def get(cls, name: str) -> MutatorFactory:
        key = cls._key(name)
        if key not in cls._registry:
            raise KeyError(f"Mutator type '{name}' not found in registry")
        return cls._registry[key]

    @classmethod
    def list_available(cls) -> List[str]:
        return sorted(cls._registry.keys())


def register_mutator_factory(name: str):
    """Decorator to register a mutator factory under a given `_type`."""

    def decorator(factory: MutatorFactory) -> MutatorFactory:
        MutatorRegistry.register(name, factory)
        return factory

    return decorator
