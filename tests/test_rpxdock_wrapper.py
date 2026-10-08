from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from astrolabe.structure_processors.wrappers.rpxdock import (
    RpxDockRunConfig,
    RpxDockOutputError,
    _build_run_env,
    _collect_docked_candidates,
    _extract_transform_from_result_pickle,
    _extract_transforms_from_result_pickle_all,
    _find_result_pickle,
    _list_overall_docked_candidates,
    _pick_docked_structure,
    _write_rpxdock_compat_sitecustomize,
    build_cyclic_dock_command,
)


class _FakeDataArray:
    def __init__(self, data):
        self.data = data


class _FakeResult:
    def __init__(self, xforms):
        self._data = {"xforms": _FakeDataArray(xforms)}

    def __getitem__(self, key):
        return self._data[key]


@pytest.fixture(autouse=True)
def _restore_vendor_module_fakes():
    prefixes = ("rpxdock", "willutil", "tqdm", "xarray")
    saved = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    }
    yield
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            del sys.modules[name]
    sys.modules.update(saved)


def test_build_cyclic_dock_command_includes_geometry_fit_options(tmp_path: Path) -> None:
    monomer = tmp_path / "monomer.pdb"
    monomer.write_text("END\n", encoding="utf-8")
    config = RpxDockRunConfig(
        nfold=4,
        output_dir=tmp_path / "out",
        python_bin="/abs/python",
        docking_method="hier",
        extra_args=("--beam_size", "5000"),
        cart_bounds=(18.5, 32.0),
        max_longaxis_dot_z=0.42,
        components_aligned_to_sym_axes=True,
        hscore_data_dir=tmp_path / "hscore",
        hscore_files=("small_ilv_h",),
    )

    cmd = build_cyclic_dock_command(monomer, config)

    assert cmd[:3] == ["/abs/python", "-m", "rpxdock.app.dock"]
    assert "--cart_bounds" in cmd
    cart_idx = cmd.index("--cart_bounds")
    assert cmd[cart_idx + 1 : cart_idx + 3] == ["18.5", "32.0"]
    dot_idx = cmd.index("--max_longaxis_dot_z")
    assert cmd[dot_idx + 1] == "0.42"
    hscore_dir_idx = cmd.index("--hscore_data_dir")
    assert cmd[hscore_dir_idx + 1] == str((tmp_path / "hscore").resolve())
    hscore_files_idx = cmd.index("--hscore_files")
    assert cmd[hscore_files_idx + 1] == "small_ilv_h"
    assert "--components_already_aligned_to_sym_axes" in cmd
    assert "--limit_rotation_to_z" in cmd
    tarball_idx = cmd.index("--save_results_as_tarball")
    assert cmd[tarball_idx + 1] == "False"
    assert cmd[-2:] == ["--beam_size", "5000"]


def test_build_cyclic_dock_command_supports_dihedral_architecture(tmp_path: Path) -> None:
    monomer = tmp_path / "monomer.pdb"
    monomer.write_text("END\n", encoding="utf-8")
    config = RpxDockRunConfig(
        nfold=3,
        architecture="D3_3",
        output_dir=tmp_path / "out",
        python_bin="/abs/python",
        cart_bounds=(-4.0, 4.0),
        limit_rotation_to_z=False,
    )

    cmd = build_cyclic_dock_command(monomer, config)

    arch_idx = cmd.index("--architecture")
    assert cmd[arch_idx + 1] == "D3_3"
    cart_idx = cmd.index("--cart_bounds")
    assert cmd[cart_idx + 1 : cart_idx + 3] == ["-4.0", "4.0"]
    assert "--limit_rotation_to_z" not in cmd


def test_extract_transform_from_result_pickle_reads_first_xform(tmp_path: Path) -> None:
    xform = np.array(
        [
            [0.0, -1.0, 0.0, 12.0],
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, -1.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    ds = _FakeResult(np.stack([xform], axis=0))
    path = tmp_path / "result.pickle"
    with path.open("wb") as fh:
        pickle.dump(ds, fh)

    rotation, translation = _extract_transform_from_result_pickle(path)

    # RPXDock stores transforms for column-vector application; our callers use
    # row-vector math, so the rotation must be transposed on extraction.
    assert np.allclose(rotation, xform[:3, :3].T)
    assert np.allclose(translation, xform[:3, 3])


def test_build_run_env_prepends_vendored_rpxdock_and_willutil(tmp_path: Path) -> None:
    rpxdock_repo = tmp_path / "rpxdock"
    willutil_repo = tmp_path / "willutil"
    env = _build_run_env(
        rpxdock_repo,
        willutil_repo,
        {"PYTHONPATH": "/existing/site-packages"},
    )

    assert env["PYTHONPATH"].split(":")[:3] == [
        str(rpxdock_repo),
        str(willutil_repo),
        "/existing/site-packages",
    ]
    assert env["SETUPTOOLS_USE_DISTUTILS"] == "stdlib"
    assert env["PYTHONSAFEPATH"] == "1"


def test_build_run_env_preserves_explicit_setuptools_distutils_override(tmp_path: Path) -> None:
    env = _build_run_env(
        tmp_path / "rpxdock",
        None,
        {"SETUPTOOLS_USE_DISTUTILS": "local"},
    )

    assert env["SETUPTOOLS_USE_DISTUTILS"] == "local"


def test_rpxdock_compat_sitecustomize_patches_body_intersect_to_float64(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    compat_dir = _write_rpxdock_compat_sitecustomize(tmp_path / "out")
    package_root = tmp_path / "fake_pkg"
    body_dir = package_root / "rpxdock" / "body"
    body_dir.mkdir(parents=True)
    (package_root / "rpxdock" / "__init__.py").write_text(
        """
import numpy as np

class _Bvh:
    def _check(self, xself, xother, dtype):
        if xself.dtype != dtype or xother.dtype != dtype:
            raise RuntimeError(f"expected {dtype}, got {xself.dtype}/{xother.dtype}")
        if not xself.flags.c_contiguous or not xother.flags.c_contiguous:
            raise RuntimeError("expected contiguous xforms")

    def bvh_isect_vec(self, bb1, bb2, xself, xother, mindis):
        self._check(xself, xother, np.float64)
        return np.zeros(len(xself), dtype=bool)

    def bvh_collect_pairs_range_vec(self, bb1, bb2, xself, xother, maxdist, *bounds):
        self._check(xself, xother, np.float32)
        return np.zeros((0, 2), dtype=np.int32), np.zeros((len(xself), 2), dtype=np.int32)

bvh = _Bvh()
""".lstrip(),
        encoding="utf-8",
    )
    (body_dir / "__init__.py").write_text("", encoding="utf-8")
    (body_dir / "body.py").write_text(
        """
import numpy as np
import rpxdock as rp

_CLASHRAD = 1.0

class Body:
    bvh_bb = object()
    pos = np.eye(4, dtype=np.float64)

    def intersect(self, other, xself=None, xother=None, mindis=2 * _CLASHRAD, **kw):
        xself = self.pos if xself is None else xself
        xother = other.pos if xother is None else xother
        return rp.bvh.bvh_isect_vec(
            self.bvh_bb,
            other.bvh_bb,
            np.ascontiguousarray(xself.reshape(-1, 4, 4), dtype=np.float32),
            np.ascontiguousarray(xother.reshape(-1, 4, 4), dtype=np.float32),
            mindis,
        )
""".lstrip(),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(compat_dir), str(package_root)])

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import numpy as np; "
                "from rpxdock.body.body import Body; "
                "x=np.arange(6*4*4,dtype=np.float64).reshape(6,4,4)[::2]; "
                "print(Body().intersect(Body(), xself=x, xother=x, mindis=1.5).shape); "
                "import rpxdock as rp; "
                "print(rp.bvh.bvh_collect_pairs_range_vec(None,None,x,x,8,0,1,-1,0,1,-1)[1].shape); "
                "z=np.empty((0,4,4),dtype=np.float64); "
                "print(rp.bvh.bvh_collect_pairs_range_vec(None,None,z,z,8,0,1,-1,0,1,-1)[1].shape)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.stdout.splitlines() == ["(3,)", "(3, 2)", "(0, 2)"]


def test_pick_docked_structure_prefers_overall_top_result(tmp_path: Path) -> None:
    monomer = tmp_path / "monomer.pdb"
    monomer.write_text("END\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    latest_non_top = output_dir / "rpxdock_cyclicC4__monomer__job0_top1_1.pdb"
    latest_non_top.write_text("END\n", encoding="utf-8")
    top0 = output_dir / "rpxdock_cyclicC4__monomer__top0_0.pdb"
    top0.write_text("END\n", encoding="utf-8")
    oldest = output_dir / "rpxdock_cyclicC4__monomer__top4_4.pdb"
    oldest.write_text("END\n", encoding="utf-8")

    picked = _pick_docked_structure(output_dir, monomer)

    assert picked == top0.resolve()


def test_find_result_pickle_supports_dihedral_architecture_names(tmp_path: Path) -> None:
    output_dir = tmp_path / "out_pickle"
    output_dir.mkdir()
    result_pickle = output_dir / "rpxdock_cyclicD3_3__Result.pickle"
    result_pickle.write_bytes(b"pickle")

    assert _find_result_pickle(output_dir, "D3_3") == result_pickle


def test_extract_transforms_from_result_pickle_reads_all_xforms(tmp_path: Path) -> None:
    xforms = np.stack(
        [
            np.array(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0.0, -1.0, 0.0, 4.0],
                    [1.0, 0.0, 0.0, 5.0],
                    [0.0, 0.0, 1.0, 6.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )
    ds = _FakeResult(xforms)
    path = tmp_path / "result_all.pickle"
    with path.open("wb") as fh:
        pickle.dump(ds, fh)

    transforms = _extract_transforms_from_result_pickle_all(path)

    assert len(transforms) == 2
    assert np.allclose(transforms[0][0], xforms[0, :3, :3].T)
    assert np.allclose(transforms[0][1], xforms[0, :3, 3])
    assert np.allclose(transforms[1][0], xforms[1, :3, :3].T)
    assert np.allclose(transforms[1][1], xforms[1, :3, 3])


def test_list_overall_docked_candidates_prefers_overall_results_in_rank_order(tmp_path: Path) -> None:
    monomer = tmp_path / "monomer.pdb"
    monomer.write_text("END\n", encoding="utf-8")
    output_dir = tmp_path / "out_candidates"
    output_dir.mkdir()
    (output_dir / "rpxdock_cyclicC4__monomer__job0_top0_0.pdb").write_text("END\n", encoding="utf-8")
    top2 = output_dir / "rpxdock_cyclicC4__monomer__top2_2.pdb"
    top2.write_text("END\n", encoding="utf-8")
    top0 = output_dir / "rpxdock_cyclicC4__monomer__top0_0.pdb"
    top0.write_text("END\n", encoding="utf-8")
    top1 = output_dir / "rpxdock_cyclicC4__monomer__top1_1.pdb"
    top1.write_text("END\n", encoding="utf-8")

    candidates = _list_overall_docked_candidates(output_dir, monomer)

    assert candidates == (top0.resolve(), top1.resolve(), top2.resolve())


def test_collect_docked_candidates_matches_top_rank_to_result_transform(tmp_path: Path) -> None:
    monomer = tmp_path / "monomer.pdb"
    monomer.write_text("END\n", encoding="utf-8")
    output_dir = tmp_path / "out_collect"
    output_dir.mkdir()
    top0 = output_dir / "rpxdock_cyclicC4__monomer__top0_0.pdb"
    top1 = output_dir / "rpxdock_cyclicC4__monomer__top1_1.pdb"
    top0.write_text("END\n", encoding="utf-8")
    top1.write_text("END\n", encoding="utf-8")

    xforms = np.stack(
        [
            np.array(
                [
                    [1.0, 0.0, 0.0, 10.0],
                    [0.0, 1.0, 0.0, 11.0],
                    [0.0, 0.0, 1.0, 12.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0.0, -1.0, 0.0, 20.0],
                    [1.0, 0.0, 0.0, 21.0],
                    [0.0, 0.0, 1.0, 22.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )
    result_pickle = output_dir / "rpxdock_cyclicC4__Result.pickle"
    with result_pickle.open("wb") as fh:
        pickle.dump(_FakeResult(xforms), fh)

    candidates = _collect_docked_candidates(
        output_dir,
        monomer,
        result_pickle,
        python_bin="python",
        vendored_rpxdock=tmp_path,
        vendored_willutil=None,
        env=None,
    )

    assert [candidate.rank for candidate in candidates] == [0, 1]
    assert [candidate.docked_structure_path for candidate in candidates] == [top0.resolve(), top1.resolve()]
    assert np.allclose(candidates[0].translation, [10.0, 11.0, 12.0])
    assert np.allclose(candidates[1].rotation, xforms[1, :3, :3].T)


def test_collect_docked_candidates_uses_result_pickle_when_no_pdb_was_dumped(tmp_path: Path) -> None:
    monomer = tmp_path / "monomer.pdb"
    monomer.write_text("END\n", encoding="utf-8")
    output_dir = tmp_path / "out_pickle_only"
    output_dir.mkdir()
    xforms = np.stack(
        [
            np.array(
                [
                    [1.0, 0.0, 0.0, 3.0],
                    [0.0, 1.0, 0.0, 4.0],
                    [0.0, 0.0, 1.0, 5.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        ],
        axis=0,
    )
    result_pickle = output_dir / "rpxdock_cyclicD3_3__Result.pickle"
    with result_pickle.open("wb") as fh:
        pickle.dump(_FakeResult(xforms), fh)

    candidates = _collect_docked_candidates(
        output_dir,
        monomer,
        result_pickle,
        python_bin="python",
        vendored_rpxdock=tmp_path,
        vendored_willutil=None,
        env=None,
    )

    assert len(candidates) == 1
    assert candidates[0].rank == 0
    assert candidates[0].docked_structure_path == result_pickle.resolve()
    assert np.allclose(candidates[0].translation, [3.0, 4.0, 5.0])


def test_collect_docked_candidates_reports_empty_result_pickle_as_output_error(tmp_path: Path) -> None:
    monomer = tmp_path / "monomer.pdb"
    monomer.write_text("END\n", encoding="utf-8")
    output_dir = tmp_path / "out_empty_pickle"
    output_dir.mkdir()
    result_pickle = output_dir / "rpxdock_cyclicD3_3__Result.pickle"
    with result_pickle.open("wb") as fh:
        pickle.dump(_FakeResult(np.empty((0, 4, 4), dtype=np.float32)), fh)

    with pytest.raises(RpxDockOutputError, match="Unexpected xforms shape"):
        _collect_docked_candidates(
            output_dir,
            monomer,
            result_pickle,
            python_bin=sys.executable,
            vendored_rpxdock=tmp_path,
            vendored_willutil=None,
            env=None,
        )


def test_rpxdock_body_intersect_normalizes_noncontiguous_xforms(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    import sys
    import types

    repo_root = Path(__file__).resolve().parents[1]
    body_py = repo_root / "vendor" / "rpxdock" / "rpxdock" / "body" / "body.py"

    captured = []

    def _fake_bvh_isect_vec(bb1, bb2, xself, xother, mindis):
        captured.append((xself.flags.c_contiguous, xother.flags.c_contiguous, xself.shape, xother.shape))
        return np.zeros(len(xself), dtype=bool)

    rp_module = types.ModuleType("rpxdock")
    rp_module.bvh = types.SimpleNamespace(bvh_isect_vec=_fake_bvh_isect_vec)
    sys.modules["rpxdock"] = rp_module
    sys.modules["rpxdock.filter"] = types.ModuleType("rpxdock.filter")
    sscount_module = types.ModuleType("rpxdock.filter.sscount")
    sscount_module.secondary_structure_map = lambda *args, **kwargs: None
    sys.modules["rpxdock.filter.sscount"] = sscount_module

    wu_module = types.ModuleType("willutil")
    wu_module.timed = lambda fn: fn

    class _Bunch(dict):
        pass

    wu_module.Bunch = _Bunch
    sys.modules["willutil"] = wu_module

    spec = importlib.util.spec_from_file_location("body_module", body_py)
    body_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(body_module)
    Body = body_module.Body

    class _DummyBody:
        bvh_bb = object()
        pos = np.eye(4, dtype=np.float32)

    xself = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4)[::2]
    xother = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4)[1::2]

    result = Body.intersect(_DummyBody(), _DummyBody(), xself=xself, xother=xother, mindis=1.5)

    assert result.shape == (3,)
    assert captured == [(True, True, (3, 4, 4), (3, 4, 4))]


def test_rpxdock_cyclic_evaluator_skips_empty_clash_batches() -> None:
    import importlib.util
    import sys
    import types

    repo_root = Path(__file__).resolve().parents[1]
    cyclic_py = repo_root / "vendor" / "rpxdock" / "rpxdock" / "search" / "cyclic.py"

    class _Bunch(dict):
        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, value):
            self[name] = value

        def sub(self, **updates):
            out = _Bunch(self)
            out.update(updates)
            return out

    rp_module = types.ModuleType("rpxdock")
    rp_module.search = types.SimpleNamespace(trim_ok=lambda trim, nres, **kw: (trim, np.array([], dtype=bool)))
    sys.modules["rpxdock"] = rp_module
    hm_module = types.ModuleType("rpxdock.homog")
    hm_module.hrot = lambda axis, angle, degrees=True: np.eye(4, dtype=float)
    sys.modules["rpxdock.homog"] = hm_module
    search_module = types.ModuleType("rpxdock.search")
    search_module.hier_search = None
    search_module.grid_search = None
    sys.modules["rpxdock.search"] = search_module
    filter_module = types.ModuleType("rpxdock.filter")
    filter_module.filters = None
    sys.modules["rpxdock.filter"] = filter_module
    wu_module = types.ModuleType("willutil")
    wu_module.Bunch = _Bunch
    sys.modules["willutil"] = wu_module

    spec = importlib.util.spec_from_file_location("cyclic_module", cyclic_py)
    cyclic_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cyclic_module)
    CyclicEvaluator = cyclic_module.CyclicEvaluator

    class _Body:
        pcavecs = [np.array([0.0, 0.0, 1.0, 0.0], dtype=float)]
        nres = 10

        def clash_ok(self, *args, **kwargs):
            raise AssertionError("clash_ok should not be called when no poses survive the flatness filter")

    class _HScore:
        def scorepos(self, *args, **kwargs):
            raise AssertionError("scorepos should not be called when no poses survive the flatness filter")

    evaluator = CyclicEvaluator(
        _Body(),
        "C4",
        _HScore(),
        max_longaxis_dot_z=0.0,
        clash_distances=[1.5],
        max_trim=0,
    )

    scores, extra = evaluator(np.eye(4, dtype=float)[None, ...], iresl=0)

    assert scores.shape == (1,)
    assert np.allclose(scores, 0.0)


def test_rpxdock_result_top_each_handles_empty_scores() -> None:
    import importlib.util
    import sys
    import types

    repo_root = Path(__file__).resolve().parents[1]
    result_py = repo_root / "vendor" / "rpxdock" / "rpxdock" / "search" / "result.py"

    rp_module = types.ModuleType("rpxdock")
    sys.modules["rpxdock"] = rp_module
    util_module = types.ModuleType("rpxdock.util")
    util_module.sanitize_for_storage = lambda value: value
    util_module.num_digits = lambda value: 1
    sys.modules["rpxdock.util"] = util_module
    wu_module = types.ModuleType("willutil")
    sys.modules["willutil"] = wu_module

    class _Dataset(dict):
        def __init__(self, mapping, attrs=None):
            super().__init__({k: (v[1] if isinstance(v, tuple) else v) for k, v in mapping.items()})
            self.attrs = attrs or {}

        def __getattr__(self, name):
            return self[name]

    xr_module = types.ModuleType("xarray")
    xr_module.Dataset = _Dataset
    sys.modules["xarray"] = xr_module

    spec = importlib.util.spec_from_file_location("result_module", result_py)
    result_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(result_module)
    Result = result_module.Result

    class _FakeScores:
        def __len__(self):
            return 0

        @property
        def data(self):
            return np.asarray([], dtype=float)

        def groupby(self, group):
            raise AssertionError("groupby should not be called for empty scores")

    result = Result.__new__(Result)
    result.data = types.SimpleNamespace(scores=_FakeScores(), ijob=np.asarray([], dtype=int))

    assert result.top_each(1) == {}


def test_rpxdock_result_dump_pdbs_top_score_each_ignores_empty_results() -> None:
    import importlib.util
    import sys
    import types

    repo_root = Path(__file__).resolve().parents[1]
    result_py = repo_root / "vendor" / "rpxdock" / "rpxdock" / "search" / "result.py"

    rp_module = types.ModuleType("rpxdock")
    sys.modules["rpxdock"] = rp_module
    util_module = types.ModuleType("rpxdock.util")
    util_module.sanitize_for_storage = lambda value: value
    util_module.num_digits = lambda value: 1
    sys.modules["rpxdock.util"] = util_module
    wu_module = types.ModuleType("willutil")
    sys.modules["willutil"] = wu_module

    class _Dataset(dict):
        def __init__(self, mapping, attrs=None):
            super().__init__({k: (v[1] if isinstance(v, tuple) else v) for k, v in mapping.items()})
            self.attrs = attrs or {}

        def __getattr__(self, name):
            return self[name]

    xr_module = types.ModuleType("xarray")
    xr_module.Dataset = _Dataset
    sys.modules["xarray"] = xr_module

    spec = importlib.util.spec_from_file_location("result_module", result_py)
    result_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(result_module)
    Result = result_module.Result

    result = Result.__new__(Result)
    result.top_each = lambda neach=1: {}

    assert result.dump_pdbs_top_score_each(nout_each=1) == set()


def test_rpxdock_dock_cyclic_injects_threadpool_executor_when_nthread_gt_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import concurrent.futures
    import importlib.util
    import sys
    import types

    repo_root = Path(__file__).resolve().parents[1]
    dock_py = repo_root / "vendor" / "rpxdock" / "rpxdock" / "app" / "dock.py"

    class _Bunch(dict):
        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, value):
            self[name] = value

    class _InProcessExecutor:
        def __init__(self, _ncpu):
            self._ncpu = _ncpu

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            future = concurrent.futures.Future()
            future.set_result(fn(*args, **kwargs))
            return future

    class _FakeThreadPoolExecutor:
        def __init__(self, max_workers):
            self._max_workers = max_workers
            self.shutdown_calls = []

        def shutdown(self, wait=True, *, cancel_futures=False):
            self.shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})

    captured: dict[str, object] = {}

    def _make_cyclic(body, sym, hscore, **kw):
        captured["body"] = body
        captured["sym"] = sym
        captured["hscore"] = hscore
        captured["executor"] = kw.get("executor")
        return {"ok": True}

    rp_module = types.ModuleType("rpxdock")
    rp_module.util = types.SimpleNamespace(InProcessExecutor=_InProcessExecutor)
    rp_module.Body = lambda inp, allowed_res=None, required_res_sets=None, **kw: {"input": inp}
    rp_module.search = types.SimpleNamespace(make_cyclic=_make_cyclic, make_cyclic_stack=_make_cyclic)
    rp_module.concat_results = lambda results: results
    sys.modules["rpxdock"] = rp_module

    wu_module = types.ModuleType("willutil")
    wu_module.Bunch = _Bunch
    sys.modules["willutil"] = wu_module

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, *args, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_module

    spec = importlib.util.spec_from_file_location("dock_module", dock_py)
    dock_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(dock_module)
    monkeypatch.setattr(
        dock_module.concurrent.futures,
        "ThreadPoolExecutor",
        _FakeThreadPoolExecutor,
    )

    result = dock_module.dock_cyclic(
        "fake-hscore",
        inputs1=["monomer.pdb"],
        allowed_residues1=[None],
        allowed_residues_also1=[None],
        architecture="C4",
        ncpu=8,
        nthread=4,
        quiet=True,
    )

    executor = captured.get("executor")
    assert isinstance(executor, _FakeThreadPoolExecutor)
    assert executor._max_workers == 4
    assert executor.shutdown_calls == [{"wait": False, "cancel_futures": True}]
    assert result == [{"ok": True}]


def test_rpxdock_result_strips_executor_from_nested_dockinfo_attrs() -> None:
    import importlib.util
    import sys
    import types

    repo_root = Path(__file__).resolve().parents[1]
    result_py = repo_root / "vendor" / "rpxdock" / "rpxdock" / "search" / "result.py"

    rp_module = types.ModuleType("rpxdock")
    rp_module.Body = type("Body", (), {})
    sys.modules["rpxdock"] = rp_module

    util_module = types.ModuleType("rpxdock.util")
    util_module.sanitize_for_storage = lambda value: value
    util_module.num_digits = lambda value: 1
    sys.modules["rpxdock.util"] = util_module

    wu_module = types.ModuleType("willutil")
    sys.modules["willutil"] = wu_module

    class _Dataset(dict):
        def __init__(self, mapping, attrs=None):
            super().__init__({k: (v[1] if isinstance(v, tuple) else v) for k, v in mapping.items()})
            self.attrs = attrs or {}

        def __getattr__(self, name):
            return self[name]

    xr_module = types.ModuleType("xarray")
    xr_module.Dataset = _Dataset
    sys.modules["xarray"] = xr_module

    spec = importlib.util.spec_from_file_location("result_module", result_py)
    result_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(result_module)
    Result = result_module.Result

    result = Result(
        attrs={
            "dockinfo": [{"arg": {"executor": object(), "keep": 1}}],
            "sym": "C4",
        },
        scores=(["model"], np.asarray([1.0], dtype=float)),
        xforms=(["model", "hrow", "hcol"], np.eye(4, dtype=float)[None, ...]),
    )

    assert "executor" not in result.data.attrs["dockinfo"][0]["arg"]
    assert result.data.attrs["dockinfo"][0]["arg"]["keep"] == 1


def test_rpxdock_dock_cyclic_shuts_down_created_threadpool_when_body_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util
    import sys
    import types

    repo_root = Path(__file__).resolve().parents[1]
    dock_py = repo_root / "vendor" / "rpxdock" / "rpxdock" / "app" / "dock.py"

    class _Bunch(dict):
        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, value):
            self[name] = value

    class _FakeThreadPoolExecutor:
        def __init__(self, max_workers):
            self._max_workers = max_workers
            self.shutdown_calls = []

        def shutdown(self, wait=True, *, cancel_futures=False):
            self.shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})

    created = []

    rp_module = types.ModuleType("rpxdock")
    rp_module.util = types.SimpleNamespace(InProcessExecutor=object)

    def _body(*args, **kwargs):
        raise RuntimeError("boom")

    rp_module.Body = _body
    rp_module.search = types.SimpleNamespace(make_cyclic=lambda *a, **k: None, make_cyclic_stack=lambda *a, **k: None)
    rp_module.concat_results = lambda results: results
    sys.modules["rpxdock"] = rp_module

    wu_module = types.ModuleType("willutil")
    wu_module.Bunch = _Bunch
    sys.modules["willutil"] = wu_module

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, *args, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_module

    spec = importlib.util.spec_from_file_location("dock_module", dock_py)
    dock_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(dock_module)
    monkeypatch.setattr(
        dock_module.concurrent.futures,
        "ThreadPoolExecutor",
        lambda max_workers: (
            created.append(_FakeThreadPoolExecutor(max_workers)) or created[-1]
        ),
    )

    with pytest.raises(RuntimeError, match="boom"):
        dock_module.dock_cyclic(
            "fake-hscore",
            inputs1=["monomer.pdb"],
            allowed_residues1=[None],
            allowed_residues_also1=[None],
            architecture="C4",
            ncpu=8,
            nthread=4,
            quiet=True,
        )

    assert len(created) == 1
    assert created[0].shutdown_calls == [{"wait": False, "cancel_futures": True}]
