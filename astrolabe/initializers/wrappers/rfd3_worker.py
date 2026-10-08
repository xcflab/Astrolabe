#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip() or "rfd3")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        raise SystemExit("usage: rfd3_worker.py REQUEST.json")

    request = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    script_dir = Path(__file__).resolve().parent
    sys.path = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != script_dir
    ]
    sys.modules.pop("rfd3", None)

    from biotite.structure.io.pdb import PDBFile
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine

    config = RFD3InferenceConfig(**dict(request["config"]))
    engine = RFD3InferenceEngine(**config)
    outputs_by_example = engine.run(
        inputs=str(Path(request["input_path"])),
        n_batches=int(request["n_batches"]),
        out_dir=None,
    )
    if not outputs_by_example:
        raise RuntimeError("RFD3 inference produced no outputs.")

    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_outputs = []
    for output_list in outputs_by_example.values():
        for output in output_list:
            example_id = str(output.example_id)
            base_path = output_dir / _safe_label(example_id)
            structure_path = base_path.with_suffix(".pdb")
            pdb_file = PDBFile()
            pdb_file.set_structure(output.atom_array)
            pdb_file.write(structure_path)

            metadata_path = None
            if output.metadata:
                metadata_path = base_path.with_suffix(".json")
                metadata_path.write_text(
                    json.dumps(output.metadata, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )
            manifest_outputs.append(
                {
                    "example_id": example_id,
                    "structure_path": str(structure_path),
                    "metadata_path": str(metadata_path) if metadata_path else None,
                }
            )

    Path(request["response_path"]).write_text(
        json.dumps({"outputs": manifest_outputs}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
