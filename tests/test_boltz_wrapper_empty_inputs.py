from __future__ import annotations

from pathlib import Path

import pytest


def test_boltz_featurize_data_skips_work_when_no_inputs(tmp_path: Path, monkeypatch) -> None:
    from astrolabe.evaluators.wrappers import boltz as boltz_wrapper

    runner = boltz_wrapper.Boltz1Runner(
        {
            "cache": str(tmp_path / "cache"),
            "checkpoint": None,
            "devices": 1,
            "accelerator": "cpu",
            "recycling_steps": 1,
            "sampling_steps": 1,
            "diffusion_samples": 1,
            "write_full_pae": False,
            "write_full_pde": False,
            "output_format": "mmcif",
            "num_workers": 0,
            "override": False,
            "use_msa_server": False,
            "msa_server_url": None,
            "msa_pairing_strategy": "greedy",
        }
    )

    monkeypatch.setattr(boltz_wrapper, "check_inputs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        boltz_wrapper,
        "download",
        lambda *_args, **_kwargs: pytest.fail("download() should not run for empty inputs"),
    )
    monkeypatch.setattr(
        boltz_wrapper,
        "process_inputs",
        lambda *_args, **_kwargs: pytest.fail("process_inputs() should not run for empty inputs"),
    )

    assert runner.featurize_data(data=[], out_dir=tmp_path) == []

