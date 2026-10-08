#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path


def _patch_env() -> None:
    original = os._Environ.__setitem__

    def patched(self, key, value):
        if key == "USE_DEEPSPEED_EVO_ATTENTION":
            value = "false"
        elif key == "LAYERNORM_TYPE" and str(value) == "fast_layernorm":
            value = ""
        return original(self, key, value)

    os._Environ.__setitem__ = patched
    os.environ["USE_DEEPSPEED_EVO_ATTENTION"] = "false"
    os.environ.pop("LAYERNORM_TYPE", None)


def _write_dummy_msa_dir(base_dir: Path, *, sequence: str, entity_idx: int) -> Path:
    digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]
    msa_dir = base_dir / ("entity_%02d_len_%03d_%s" % (entity_idx + 1, len(sequence), digest))
    msa_dir.mkdir(parents=True, exist_ok=True)
    text = ">query\n%s\n" % sequence
    for name in ("pairing.a3m", "non_pairing.a3m"):
        path = msa_dir / name
        if not path.exists():
            path.write_text(text, encoding="utf-8")
    return msa_dir


def _materialize_missing_protein_msas(input_json_path: str | Path) -> Path:
    input_path = Path(input_json_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Expected Protenix input JSON list: %s" % input_path)
    changed = False
    dummy_root = input_path.parent / "_binder_dummy_msa"
    for item in payload:
        sequences = item.get("sequences") or []
        for entity_idx, sequence in enumerate(sequences):
            protein = sequence.get("proteinChain")
            if not protein:
                continue
            if protein.get("msa"):
                continue
            msa_dir = _write_dummy_msa_dir(
                dummy_root,
                sequence=str(protein["sequence"]),
                entity_idx=entity_idx,
            )
            protein["msa"] = {
                "precomputed_msa_dir": str(msa_dir),
                "pairing_db": "uniref100",
            }
            changed = True
    if not changed:
        return input_path
    patched_path = input_path.with_name("%s__patched_msa%s" % (input_path.stem, input_path.suffix))
    patched_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return patched_path


def _patch_protenix_inferrer(runner_inference) -> None:
    inferrer = getattr(runner_inference, "ProtenixInferrer", None)
    if inferrer is None:
        return
    if getattr(inferrer, "_binder_missing_methods_patch_applied", False):
        return
    original_predict = inferrer.predict

    def patched_predict(self, *args, **kwargs):
        if "input_json_path" in kwargs:
            kwargs["input_json_path"] = str(
                _materialize_missing_protein_msas(kwargs["input_json_path"])
            )
        elif args:
            new_args = list(args)
            new_args[0] = str(_materialize_missing_protein_msas(new_args[0]))
            args = tuple(new_args)
        return original_predict(self, *args, **kwargs)

    inferrer.predict = patched_predict
    inferrer._binder_missing_methods_patch_applied = True


def _patch_protenix_defaults(target_script: str) -> None:
    script_path = Path(target_script).resolve()
    protenix_root = script_path.parent / "Protenix"
    if str(protenix_root) not in sys.path:
        sys.path.insert(0, str(protenix_root))
    try:
        import runner.inference as runner_inference
    except Exception:
        return
    try:
        runner_inference.configs_base["use_deepspeed_evo_attention"] = False
    except Exception:
        pass
    _patch_protenix_inferrer(runner_inference)


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: halludesign_protenix_entry.py <script> [args...]")
    script = sys.argv[1]
    sys.argv = [script] + sys.argv[2:]
    _patch_env()
    _patch_protenix_defaults(script)
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
