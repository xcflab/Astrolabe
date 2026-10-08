from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def _normalize_config(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        normalized = dict(config)
    else:
        normalized = OmegaConf.to_container(config, resolve=True)
    if not isinstance(normalized, dict):
        raise ValueError(
            f"RFD3 runner config must resolve to a mapping, got {type(normalized).__name__}."
        )
    return normalized


def _required_path(value: Any, *, field: str, executable: bool = False) -> Path:
    if value in (None, "", "null", "None"):
        raise ValueError(f"initializer.rfd3.{field} is required.")
    path = Path(os.path.abspath(Path(str(value)).expanduser()))
    if not path.exists():
        raise FileNotFoundError(f"initializer.rfd3.{field} does not exist: {path}")
    if executable and not os.access(path, os.X_OK):
        raise PermissionError(f"initializer.rfd3.{field} is not executable: {path}")
    return path


@dataclass(frozen=True, slots=True)
class RFD3StructureOutput:
    example_id: str
    structure_path: Path
    metadata_path: Path | None


class RFD3Runner:
    """Run one RFD3 initialization request in the isolated Foundry environment."""

    def __init__(self, config: Any):
        normalized = _normalize_config(config)
        self.repo_path = _required_path(normalized.pop("repo_path", None), field="repo_path")
        self.python_bin = _required_path(
            normalized.pop("python_bin", None),
            field="python_bin",
            executable=True,
        )
        checkpoint = normalized.get("ckpt_path")
        if checkpoint in (None, "", "null", "None"):
            raise ValueError("initializer.rfd3.ckpt_path is required.")
        checkpoint_path = Path(str(checkpoint)).expanduser()
        if checkpoint_path.is_absolute() or checkpoint_path.parent != Path("."):
            normalized["ckpt_path"] = str(
                _required_path(checkpoint_path, field="ckpt_path")
            )
        self.engine_config = normalized
        self.worker_path = Path(__file__).resolve().with_name("rfd3_worker.py")

    def run(
        self,
        *,
        specification: dict[str, Any],
        output_dir: Path,
        n_batches: int,
    ) -> list[RFD3StructureOutput]:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = output_dir / "input.json"
        request_path = output_dir / "request.json"
        response_path = output_dir / "response.json"

        input_path.write_text(
            json.dumps({"gh_initializer": specification}, indent=2) + "\n",
            encoding="utf-8",
        )
        request_path.write_text(
            json.dumps(
                {
                    "config": self.engine_config,
                    "input_path": str(input_path),
                    "n_batches": int(n_batches),
                    "output_dir": str(output_dir),
                    "response_path": str(response_path),
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

        import_roots = [
            str(self.repo_path / "src"),
            str(self.repo_path / "models" / "rfd3" / "src"),
        ]
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            import_roots.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(import_roots)
        subprocess.run(
            [str(self.python_bin), "-u", str(self.worker_path), str(request_path)],
            cwd=str(self.repo_path),
            env=env,
            check=True,
        )

        response = json.loads(response_path.read_text(encoding="utf-8"))
        outputs = []
        for item in response.get("outputs") or []:
            structure_path = Path(str(item["structure_path"]))
            if not structure_path.is_file() or structure_path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"RFD3 worker did not produce a non-empty structure: {structure_path}"
                )
            metadata_raw = item.get("metadata_path")
            metadata_path = Path(str(metadata_raw)) if metadata_raw else None
            outputs.append(
                RFD3StructureOutput(
                    example_id=str(item["example_id"]),
                    structure_path=structure_path,
                    metadata_path=metadata_path,
                )
            )
        if not outputs:
            raise RuntimeError("RFD3 returned no structure outputs.")
        return outputs
