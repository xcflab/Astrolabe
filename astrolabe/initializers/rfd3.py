from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from astrolabe.initializers.rfd3_spec import RFD3SpecificationBuilder
from astrolabe.initializers.structure import (
    PreparedStructure,
    StructureSequenceInitializer,
    resolve_structure_path,
)
from astrolabe.initializers.wrappers.rfd3 import RFD3Runner


class RFD3Initializer(StructureSequenceInitializer):
    """Generate an RFD3 structure, then use the shared structure initializer."""

    def __init__(
        self,
        *,
        args: Any,
        mutators: dict[str, Any],
        segment_plan: dict[str, list[tuple[str, slice, list[int]]]],
        runner_cls=RFD3Runner,
    ):
        initializer_cfg = getattr(args, "initializer", None)
        rfd3_cfg = (
            initializer_cfg.get("rfd3")
            if isinstance(initializer_cfg, dict) or OmegaConf.is_dict(initializer_cfg)
            else None
        )
        super().__init__(
            mode="rfd3",
            args=args,
            mutators=mutators,
            segment_plan=segment_plan,
            mode_config=rfd3_cfg,
        )

        plan = RFD3SpecificationBuilder(
            mode_config=self.mode_config,
            chain_entries=self.chain_entries,
            resolve_path=lambda value: resolve_structure_path(
                value,
                field="initializer.rfd3.specification.input",
            ),
        ).build()
        self.specification = plan.specification
        self.specification_details = plan.details
        self.n_batches = int(self.mode_config.get("n_batches", 1))
        self.sample_index = int(self.mode_config.get("sample_index", 0))
        if self.n_batches < 1:
            raise ValueError("initializer.rfd3.n_batches must be at least 1.")
        if self.sample_index < 0:
            raise ValueError("initializer.rfd3.sample_index must be non-negative.")
        self.runner = runner_cls(plan.engine_config)

    def prepare_structure(self) -> PreparedStructure:
        outputs = self.runner.run(
            specification=self.specification,
            output_dir=self.output_dir,
            n_batches=self.n_batches,
        )
        if self.sample_index >= len(outputs):
            raise ValueError(
                f"initializer.rfd3.sample_index={self.sample_index} is out of range for "
                f"{len(outputs)} RFD3 output(s)."
            )
        selected = outputs[self.sample_index]
        return PreparedStructure(
            path=Path(selected.structure_path),
            details={
                "specification": self.specification,
                "specification_plan": self.specification_details,
                "n_batches": self.n_batches,
                "sample_index": self.sample_index,
                "outputs": [
                    {
                        "example_id": output.example_id,
                        "structure_path": str(output.structure_path),
                        "metadata_path": (
                            str(output.metadata_path)
                            if output.metadata_path is not None
                            else None
                        ),
                    }
                    for output in outputs
                ],
            },
        )
