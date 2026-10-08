from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class InitializerResult:
    mode: str
    sequences: dict[str, str] = field(default_factory=dict)
    structure_path: Path | None = None
    manifest_path: Path | None = None


class Initializer(ABC):
    @abstractmethod
    def apply(self) -> InitializerResult:
        """Apply the initializer to the already parsed protomers."""
