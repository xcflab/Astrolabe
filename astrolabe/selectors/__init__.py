"""Selector package.

Many selector implementations depend on optional heavy deps (e.g. PyRosetta).
Avoid eager imports here; selectors are imported lazily where needed.
"""

__all__ = []

