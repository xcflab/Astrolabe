from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from Bio.PDB import MMCIFParser
from omegaconf import OmegaConf

from astrolabe.losses.base import loss_registry
from astrolabe.losses.channel_hole import (
    ChannelScanInput,
    FRAME_SOURCE_EXACT_CYCLIC,
    _parser_for_path,
    _run_single_hole_scan,
    build_channel_scan_inputs,
    channel_hole_score,
    channel_hole_metrics_from_scan_rows,
    load_channel_scan_summary,
    parse_hole_profile,
    radius_score_from_min_radius,
    softplus_radius_penalty_from_min_radii,
)
from astrolabe.posthoc.collect import collect_scores
from astrolabe.posthoc.derive import derive_channel_hole


REPO_ROOT = Path(__file__).resolve().parents[1]


def _example_scan_rows() -> list[dict[str, object]]:
    return [
        {
            "scan_id": "x03_y04__dx+0_dy+0",
            "parent_scan_id": "x03_y04",
            "ix": 3,
            "iy": 4,
            "cx": 0.0,
            "cy": 0.0,
            "cz": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 1.0,
            "status": "ok",
            "profile_length": 54.5,
            "effective_path_len": 46.0,
            "effective_start_s": -21.70394,
            "effective_end_s": 24.29606,
            "n_profile_points": 437,
            "min_radius": 0.88035,
            "min_radius_profile": 0.79606,
        },
        {
            "scan_id": "x03_y04__dx-1_dy+1",
            "parent_scan_id": "x03_y04",
            "ix": 2,
            "iy": 5,
            "cx": -1.0,
            "cy": 1.0,
            "cz": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 1.0,
            "status": "ok",
            "profile_length": 54.5,
            "effective_path_len": 46.0,
            "effective_start_s": -21.70394,
            "effective_end_s": 24.29606,
            "n_profile_points": 437,
            "min_radius": 1.03676,
            "min_radius_profile": 0.79606,
        },
    ]


def _write_example_scan_summary(path: Path) -> Path:
    fieldnames = [
        "scan_id",
        "parent_scan_id",
        "ix",
        "iy",
        "cx",
        "cy",
        "cz",
        "vx",
        "vy",
        "vz",
        "status",
        "profile_length",
        "effective_path_len",
        "effective_start_s",
        "effective_end_s",
        "n_profile_points",
        "min_radius",
        "min_radius_profile",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(_example_scan_rows())
    return path


def _write_example_hole_output(path: Path) -> Path:
    start_s = -27.20394
    step = 54.5 / 436.0
    min_s = 0.79606
    rows: list[tuple[float, float]] = []
    for index in range(437):
        s_value = start_s + index * step
        if abs(s_value - min_s) < step / 2:
            s_value = min_s
            radius = 0.88035
        elif s_value < -21.70394 or s_value > 24.29606:
            radius = 4.25
        else:
            radius = 1.20
        rows.append((s_value, radius))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            ["HEADER", " cenxyz.cvec"] + [f"{s:10.5f} {r:8.5f} 0.0 0.0" for s, r in rows] + [" Minimum radius found:"]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_ca_pdb(path: Path, chain_coords: dict[str, np.ndarray]) -> Path:
    lines: list[str] = []
    serial = 1
    for chain_id, coords in chain_coords.items():
        for res_i, xyz in enumerate(np.asarray(coords, dtype=float), start=1):
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA {chain_id}{res_i:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00           C"
            )
            serial += 1
        lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _rotation_z(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def test_channel_hole_score_uses_binary_path_gate_and_nonlinear_radius_score() -> None:
    assert radius_score_from_min_radius(1.5) == pytest.approx(0.9)
    assert radius_score_from_min_radius(1.8) == pytest.approx(1.0)
    assert radius_score_from_min_radius(0.3) > 0.0
    assert channel_hole_score(effective_path_len=40.0, min_radius=1.8) == pytest.approx(0.0)
    assert channel_hole_score(effective_path_len=40.1, min_radius=1.5) == pytest.approx(0.9)
    assert channel_hole_score(effective_path_len=40.1, min_radius=0.3) > 0.0
    assert channel_hole_score(effective_path_len=40.1, min_radius=1.2) == pytest.approx(
        channel_hole_score(effective_path_len=80.0, min_radius=1.2)
    )


def test_channel_hole_metrics_match_example_scan_summary() -> None:
    metrics = channel_hole_metrics_from_scan_rows(_example_scan_rows())

    assert metrics["best_channel_scan_id"] == "x03_y04__dx-1_dy+1"
    assert metrics["best_channel_effective_path_len"] == pytest.approx(46.0)
    assert metrics["best_channel_min_radius"] == pytest.approx(1.03676)
    assert metrics["channel_hole"] == pytest.approx(0.42994851904)
    assert metrics["channel_softplus_radius"] == pytest.approx(-0.25237608073361273)
    assert metrics["max_geometric_channel_scan_id"] == "x03_y04__dx-1_dy+1"
    assert metrics["max_geometric_channel_min_radius"] == pytest.approx(1.03676)


def test_channel_hole_metrics_accepts_softplus_target_override() -> None:
    default_metrics = channel_hole_metrics_from_scan_rows(_example_scan_rows())
    stricter_metrics = channel_hole_metrics_from_scan_rows(
        _example_scan_rows(),
        softplus_target_min_radius=1.5,
    )

    assert stricter_metrics["channel_hole"] == pytest.approx(default_metrics["channel_hole"])
    assert stricter_metrics["channel_softplus_radius"] < default_metrics["channel_softplus_radius"]


def test_softplus_radius_penalty_smoothly_scores_all_scan_min_radii() -> None:
    very_narrow = softplus_radius_penalty_from_min_radii([0.4, 0.8, 1.0])
    partly_open = softplus_radius_penalty_from_min_radii([0.4, 1.2, 1.4])
    open_channel = softplus_radius_penalty_from_min_radii([1.25, 1.35, 1.5])

    assert very_narrow < partly_open < open_channel < 0.0


def test_channel_hole_metrics_zero_score_tie_uses_largest_radius_then_path_len() -> None:
    rows = [
        {
            "scan_id": "scan_a",
            "status": "ok",
            "effective_path_len": 36.0,
            "min_radius": 0.75,
        },
        {
            "scan_id": "scan_b",
            "status": "ok",
            "effective_path_len": 38.75,
            "min_radius": 2.70,
        },
        {
            "scan_id": "scan_c",
            "status": "ok",
            "effective_path_len": 39.00,
            "min_radius": 2.68,
        },
    ]
    metrics = channel_hole_metrics_from_scan_rows(rows)

    assert metrics["channel_hole"] == pytest.approx(0.0)
    assert metrics["best_channel_scan_id"] == "scan_b"
    assert metrics["best_channel_min_radius"] == pytest.approx(2.70)
    assert metrics["max_geometric_channel_scan_id"] == "scan_b"
    assert metrics["max_geometric_channel_min_radius"] == pytest.approx(2.70)


def test_posthoc_derive_channel_hole_reads_scan_summary_artifacts(tmp_path: Path) -> None:
    scan_root = tmp_path / "raw_model_results" / "channel_hole" / "step_000_AB"
    _write_example_scan_summary(scan_root / "scan_summary.tsv")

    derived = derive_channel_hole(tmp_path)
    metrics = derived[(0, "AB")]

    assert metrics["best_channel_scan_id"] == "x03_y04__dx-1_dy+1"
    assert metrics["best_channel_effective_path_len"] == pytest.approx(46.0)
    assert metrics["best_channel_min_radius"] == pytest.approx(1.03676)
    assert metrics["channel_hole"] == pytest.approx(0.42994851904)


def test_posthoc_collect_merges_channel_hole_metrics_into_exported_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "task_a" / "001_run"
    run_dir.mkdir(parents=True)
    (run_dir / "scores.csv").write_text(
        "step,weighted_score,accepted,stage\n0,1.25,True,default\n",
        encoding="utf-8",
    )
    scan_root = run_dir / "raw_model_results" / "channel_hole" / "step_000_AB"
    _write_example_scan_summary(scan_root / "scan_summary.tsv")

    export_csv = tmp_path / "collected.csv"
    summary = collect_scores(
        out_root=tmp_path,
        db_path=tmp_path / "scores.db",
        workers=1,
        export_csv=export_csv,
        fail_on_missing_required=False,
    )

    assert summary["runs_processed"] == 1
    rows = export_csv.read_text(encoding="utf-8").splitlines()
    assert rows[0].count("channel_hole") == 1
    assert any("x03_y04__dx-1_dy+1" in row for row in rows[1:])
    assert any("0.42994851904" in row for row in rows[1:])


def test_parse_hole_profile_reconstructs_summary_row_from_raw_hole_output(tmp_path: Path) -> None:
    metrics = parse_hole_profile(
        _write_example_hole_output(tmp_path / "x03_y04__dx+0_dy+0.hole.txt"),
        effective_radius_cutoff=4.0,
    )

    assert metrics["profile_length"] == pytest.approx(54.5)
    assert metrics["effective_path_len"] == pytest.approx(46.0)
    assert metrics["effective_start_s"] == pytest.approx(-21.70394)
    assert metrics["effective_end_s"] == pytest.approx(24.29606)
    assert metrics["n_profile_points"] == 437
    assert metrics["min_radius"] == pytest.approx(0.88035)
    assert metrics["min_radius_profile"] == pytest.approx(0.79606)


def test_build_channel_scan_inputs_aligns_structure_before_sampling(tmp_path: Path) -> None:
    ref_coords = np.array(
        [
            [-2.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [2.0, 0.0, 3.0],
        ],
        dtype=float,
    )
    ref_structure = _write_ca_pdb(tmp_path / "reference.pdb", {"A": ref_coords})

    rotation = _rotation_z(90.0)
    translation = np.array([12.0, -7.0, 4.0], dtype=float)
    moved_coords = ref_coords @ rotation + translation
    pred_structure = _write_ca_pdb(tmp_path / "prediction.pdb", {"A": moved_coords})

    ref_spheres = tmp_path / "reference_spheres.pdb"
    ref_spheres.write_text(
        "\n".join(
            [
                "ATOM      1  QSS SPH S   0       0.000   0.000  -3.000  2.40  2.40",
                "ATOM      1  QSS SPH S   1       0.000   0.000  -1.500  1.80  1.80",
                "ATOM      1  QSS SPH S   2       0.000   0.000   0.000  1.20  1.20",
                "ATOM      1  QSS SPH S   3       0.000   0.000   1.500  1.80  1.80",
                "ATOM      1  QSS SPH S   4       0.000   0.000   3.000  2.40  2.40",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    prepared = build_channel_scan_inputs(
        structure_path=pred_structure,
        reference_structure_path=ref_structure,
        reference_spheres_path=ref_spheres,
        output_dir=tmp_path / "prepared",
        lateral_step=2.0,
    )

    aligned_path = prepared.aligned_structure_path
    assert aligned_path.exists()

    aligned_lines = [
        line
        for line in aligned_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("ATOM")
    ]
    first_ca = np.array(
        [
            float(aligned_lines[0][30:38]),
            float(aligned_lines[0][38:46]),
            float(aligned_lines[0][46:54]),
        ],
        dtype=float,
    )
    assert first_ca == pytest.approx(ref_coords[0], abs=1e-3)

    central = next(item for item in prepared.scans if item.scan_id.endswith("dx+0_dy+0"))
    dx_plus = next(item for item in prepared.scans if item.scan_id.endswith("dx+1_dy+0"))
    dy_plus = next(item for item in prepared.scans if item.scan_id.endswith("dx+0_dy+1"))

    assert central.axis == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)
    assert central.center == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert np.array(dx_plus.center) - np.array(central.center) == pytest.approx(
        (2.0, 0.0, 0.0),
        abs=1e-6,
    )
    assert np.array(dy_plus.center) - np.array(central.center) == pytest.approx(
        (0.0, 2.0, 0.0),
        abs=1e-6,
    )


def test_build_channel_scan_inputs_can_use_current_exact_cyclic_axis(tmp_path: Path) -> None:
    base = np.array(
        [
            [10.0, 0.0, -6.0],
            [11.0, 0.0, 0.0],
            [10.0, 0.0, 6.0],
        ],
        dtype=float,
    )
    chain_coords = {}
    for chain_id, angle in zip(
        ["A", "B", "C", "D"],
        [0.0, 90.0, 180.0, 270.0],
    ):
        chain_coords[chain_id] = base @ _rotation_z(angle)
    pred_structure = _write_ca_pdb(tmp_path / "prediction_c4.pdb", chain_coords)

    prepared = build_channel_scan_inputs(
        structure_path=pred_structure,
        output_dir=tmp_path / "prepared",
        frame_source=FRAME_SOURCE_EXACT_CYCLIC,
        oligo_name="AAAA",
        lateral_step=2.0,
    )

    assert prepared.aligned_structure_path.exists()
    central = next(item for item in prepared.scans if item.scan_id.endswith("dx+0_dy+0"))
    assert central.parent_scan_id == "AAAA__exact_cyclic"
    assert np.abs(np.array(central.axis)) == pytest.approx(
        (0.0, 0.0, 1.0),
        abs=1e-6,
    )
    assert central.center == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)


def test_channel_hole_metadata_accepts_reference_and_exact_cyclic_frames() -> None:
    metadata = loss_registry.get_metadata("channel_hole")

    assert metadata.required_params == []
    assert {
        "alignment_chains",
        "frame_source",
        "reference_structure_path",
        "reference_spheres_path",
    }.issubset(set(metadata.optional_params))


def test_benchmark_channel_template_uses_alignment_chains_for_channel_hole() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "config" / "benchmark_channel_v1_refine.yaml")
    term = OmegaConf.to_container(cfg.loss.loss_terms.channel_hole, resolve=False)

    assert term["alignment_chains"] == "???"
    assert "reference_structure_chain" not in term


def test_run_single_hole_scan_uses_short_local_input_filenames(
    monkeypatch,
    tmp_path: Path,
) -> None:
    long_dir = tmp_path / ("deep_" * 20)
    long_dir.mkdir()
    aligned_structure_path = long_dir / "step_000_AB_model_with_a_very_long_name_for_hole.pdb"
    aligned_structure_path.write_text("ATOM\nEND\n", encoding="utf-8")
    radius_path = long_dir / "simple_radius_file_with_a_very_long_name_for_hole.rad"
    radius_path.write_text("remark radius\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_run(*args, **kwargs):
        captured["input"] = kwargs["input"]
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr("astrolabe.losses.channel_hole.subprocess.run", _fake_run)
    monkeypatch.setattr(
        "astrolabe.losses.channel_hole.parse_hole_profile",
        lambda *args, **kwargs: {
            "profile_length": 1.0,
            "effective_path_len": 1.0,
            "effective_start_s": 0.0,
            "effective_end_s": 1.0,
            "n_profile_points": 2,
            "min_radius": 1.0,
            "min_radius_profile": 0.0,
        },
    )

    _run_single_hole_scan(
        aligned_structure_path=aligned_structure_path,
        scan=ChannelScanInput(
            scan_id="scan_a",
            parent_scan_id="parent",
            center=(1.0, 2.0, 3.0),
            axis=(0.0, 0.0, 1.0),
            ix=3,
            iy=4,
        ),
        output_dir=tmp_path / "hole",
        hole_executable="/usr/bin/hole",
        hole_radius_file=str(radius_path),
        end_radius=5.0,
        effective_radius_cutoff=4.0,
    )

    input_lines = str(captured["input"]).splitlines()
    assert next(line for line in input_lines if line.startswith("COORD ")) == "COORD input.pdb"
    assert next(line for line in input_lines if line.startswith("RADIUS ")) == "RADIUS radius.rad"
    assert (tmp_path / "hole" / "scan_a" / "radius.rad").read_text(encoding="utf-8") == "remark radius\n"


def test_parser_for_path_uses_mmcif_parser_for_mmcif_suffix() -> None:
    assert isinstance(_parser_for_path(Path("example.mmcif")), MMCIFParser)
