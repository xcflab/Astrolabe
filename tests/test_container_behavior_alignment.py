from __future__ import annotations

import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
VALIDATION_SCRIPTS_DIR = SCRIPTS_DIR / "validation"
CONFIG_DIR = REPO_ROOT / "config"
CONTAINER_DEF = REPO_ROOT / "container" / "Astrolabe.af3.def"
RUNTIME_REQUIREMENTS = REPO_ROOT / "container" / "requirements.runtime.in"


def _load_script_module(script_name: str):
    script_path = VALIDATION_SCRIPTS_DIR / script_name
    spec = importlib.util.spec_from_file_location(f"_test_{script_path.stem}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_container_smoke_overrides_env_group() -> None:
    text = (CONFIG_DIR / "container_smoke.yaml").read_text(encoding="utf-8")

    assert "- override env: env_container" in text


def test_gh_dev_does_not_shadow_installed_vendor_packages() -> None:
    text = (SCRIPTS_DIR / "astrolabe").read_text(encoding="utf-8")

    assert "sys.path.insert(0, '$REPO_ROOT/vendor')" not in text
    assert "sys.path.insert(0, sys.argv.pop(1))" in text
    assert '\"$REPO_ROOT\" \"$@\"' in text


def test_gh_dev_auto_resolved_runtime_binds_current_checkout() -> None:
    text = (SCRIPTS_DIR / "astrolabe").read_text(encoding="utf-8")

    assert 'GH_HOST_REPO_MAP_ROOT="${GH_HOST_REPO_MAP_ROOT:-$ASTROLABE_HOST_REPO_MAP_ROOT}"' in text
    assert 'GH_HOST_REPO_ROOT="${GH_HOST_REPO_ROOT:-$ASTROLABE_HOST_REPO_ROOT}"' in text


@pytest.mark.parametrize(
    ("script_name", "task_name"),
    [
        ("sif_pipeline_smoketest.py", "container_smoke"),
        ("sif_af3_smoketest.py", "container_af3_smoke"),
    ],
)
def test_sif_smoketests_invoke_gh_run(monkeypatch, tmp_path: Path, script_name: str, task_name: str) -> None:
    module = _load_script_module(script_name)
    sif_path = tmp_path / "Astrolabe_af3.sif"
    sif_path.write_text("placeholder", encoding="utf-8")

    captured = {}

    def _fake_run(cmd, env=None):
        captured["cmd"] = list(cmd)
        captured["env"] = env

    monkeypatch.setattr(module, "_detect_engine", lambda: "/usr/bin/singularity")
    monkeypatch.setattr(module, "_run", _fake_run)
    if script_name == "sif_af3_smoketest.py":
        monkeypatch.setattr(module, "_verify_outputs", lambda _run_dir: None)

    argv = ["--sif", str(sif_path)]
    if script_name == "sif_af3_smoketest.py":
        af3_models = tmp_path / "af3_models"
        af3_models.mkdir()
        (af3_models / "af3.bin").write_text("placeholder", encoding="utf-8")
        argv.extend(["--af3-models", str(af3_models), "--nv"])

    rc = int(module.main(argv))

    assert rc == 0
    assert captured["cmd"][0:2] == ["/usr/bin/singularity", "exec"]
    assert f"{REPO_ROOT}:/opt/Astrolabe" in captured["cmd"]
    assert str(sif_path) in captured["cmd"]
    assert captured["cmd"].index(str(sif_path)) < captured["cmd"].index("astrolabe")
    assert "/opt/Astrolabe/run.py" not in captured["cmd"]
    assert "astrolabe" in captured["cmd"]
    assert "run" in captured["cmd"]
    assert captured["cmd"].count("--task") == 1
    assert captured["cmd"][captured["cmd"].index("--task") + 1] == task_name
    if script_name == "sif_af3_smoketest.py":
        assert captured["env"]["NVIDIA_VISIBLE_DEVICES"] == "all"
        assert captured["env"]["GH_CUDA_PREINITIALIZE"] == "1"


def test_container_definition_uses_repo_af3_and_default_pyrosetta_feature() -> None:
    text = CONTAINER_DEF.read_text(encoding="utf-8")

    assert "https://github.com/google-deepmind/alphafold3" not in text
    assert "/opt/Astrolabe/vendor/alphafold3" in text
    assert "AF3_THIRD_PARTY_DIR=/opt/Astrolabe/vendor" in text
    assert "GH_INSTALL_PYROSETTA=1" in text
    assert "if [ '{{ GH_INSTALL_PYROSETTA }}' = '1' ]" in text


def test_container_default_config_uses_portable_container_env() -> None:
    text = CONTAINER_DEF.read_text(encoding="utf-8")

    assert "ln -sfn ../env_container.yaml" in text
    assert "/opt/Astrolabe/config/env/env_local.yaml" in text
    assert '"../env_container.yaml"' in text


def test_container_definition_imports_complete_run_graph_during_build_test() -> None:
    text = CONTAINER_DEF.read_text(encoding="utf-8")

    assert "%test" in text
    assert "from astrolabe.app import _load_run_module; _load_run_module()" in text


def test_container_uses_minimal_cuda_base_without_duplicate_devel_toolkit() -> None:
    text = CONTAINER_DEF.read_text(encoding="utf-8")
    requirements = (REPO_ROOT / "container" / "requirements.runtime.lock").read_text(
        encoding="utf-8"
    )

    assert "From: nvidia/cuda@sha256:6b3201183858bad08441837f5a5efc2c75290135cc25fcc87d9ff763190cfd09" in text
    assert "nvidia/cuda:12.6.3-base-ubuntu24.04" in text
    assert "cudnn-devel" not in text
    assert "nvidia-cudnn-cu12==9.5.1.17" in requirements
    assert "nvidia-cuda-runtime-cu12==12.6.77" in requirements


def test_container_definition_excludes_unused_hmmer_and_autodock_gpu_downloads() -> None:
    text = CONTAINER_DEF.read_text(encoding="utf-8")

    assert "hmmer-3.4.tar.gz" not in text
    assert "jackhmmer_seq_limit.patch" not in text
    assert "eddylab.org/software/hmmer" not in text
    assert "/hmmer/bin" not in text
    assert "github.com/ccsb-scripps/AutoDock-GPU" not in text
    assert "/usr/bin/autodock_gpu" not in text


def test_container_installs_one_gold_runtime_resolution() -> None:
    text = CONTAINER_DEF.read_text(encoding="utf-8")
    requirements = RUNTIME_REQUIREMENTS.read_text(encoding="utf-8")

    assert text.count("-r /opt/Astrolabe/container/requirements.runtime.lock") == 1
    assert "vendor/alphafold3/dev-requirements.txt" not in text
    assert "torch==2.7.1" in requirements
    assert "torch-geometric==2.8.0" in requirements
    for dep in ("fairscale==0.4.13", "lightning-fabric==2.5.0", "pytorch-lightning==2.5.0"):
        assert dep in requirements


def test_container_resumes_and_retries_large_wheel_downloads() -> None:
    text = CONTAINER_DEF.read_text(encoding="utf-8")
    conda_lock = (REPO_ROOT / "container" / "conda-linux-64.lock").read_text(
        encoding="utf-8"
    )

    assert "export PIP_RESUME_RETRIES=10" in text
    assert "pip-25.3" in conda_lock
    assert 'if [ "$attempt" -ge 3 ]' in text


def test_container_builds_af3_after_final_numpy_without_isolation() -> None:
    text = CONTAINER_DEF.read_text(encoding="utf-8")

    final_numpy = text.index("requirements.runtime.lock")
    af3_install = text.index(
        "pip3 wheel --no-deps --no-build-isolation --wheel-dir"
    )
    assert final_numpy < af3_install
    assert "cmake ninja-build" in text
    assert "scikit-build-core==0.11.6" in RUNTIME_REQUIREMENTS.read_text(encoding="utf-8")


def test_container_caches_af3_compiled_wheel_by_source_and_toolchain() -> None:
    text = CONTAINER_DEF.read_text(encoding="utf-8")

    assert "AF3_WHEEL_KEY" in text
    assert "/cache/wheels/alphafold3/$AF3_WHEEL_KEY" in text
    assert "requirements.runtime.lock" in text
    assert "conda-linux-64.lock" in text
    assert "subprocess.check_output(['g++', '--version'])" in text
    assert "subprocess.check_output(['cmake', '--version'])" in text
    assert "subprocess.check_output(['ninja', '--version'])" in text
    assert "subprocess.check_output(['ldd', '--version']" in text
    assert "Reusing cached AlphaFold3 wheel" in text
    assert 'pip3 install --no-deps "$AF3_CACHED_WHEEL"' in text


def test_container_externalizes_mamba_cache_and_pins_bootstrap_binary() -> None:
    text = CONTAINER_DEF.read_text(encoding="utf-8")

    assert "MICROMAMBA_VERSION=2.8.1" in text
    assert "MICROMAMBA_SHA256=a934c3709c997feae403a27fd1e321c106d26ffa4f294800ffb11cbc9a3e8515" in text
    assert "--root-prefix /cache/mamba" in text
    assert "CONDA_PKGS_DIRS=/cache/mamba/pkgs" in text
    assert "MAMBA_PKGS_DIRS" not in text
    assert "rm -rf /var/lib/apt/lists/* /opt/micromamba/pkgs" in text


def test_container_uses_stable_cuda_device_enumeration() -> None:
    text = CONTAINER_DEF.read_text(encoding="utf-8")

    assert "export CUDA_DEVICE_ORDER=PCI_BUS_ID" in text


def test_container_pins_quarterly_pyrosetta_wheel_and_hash() -> None:
    text = CONTAINER_DEF.read_text(encoding="utf-8")

    assert "PYROSETTA_VERSION='2026.3+releasequarterly.5e498f1409'" in text
    assert (
        "PYROSETTA_UPSTREAM_SHA256="
        "fd1a9a51c1d7475b448e010926ed201446a55d20f4c5340455af148914d3cffe"
    ) in text
    assert "PYROSETTA_GOLD_REPACK_SHA256=" not in text
    assert "/cache/wheels/pyrosetta/$PYROSETTA_WHEEL_NAME" in text
    assert 'pip3 install --no-deps "$PYROSETTA_WHEEL"' in text
    assert "pyrosetta-installer" not in (
        REPO_ROOT / "container" / "requirements.runtime.in"
    ).read_text(encoding="utf-8")


def test_container_builds_af3_cpp_dependencies_offline_and_reuses_gold_data() -> None:
    text = CONTAINER_DEF.read_text(encoding="utf-8")

    assert "-DFETCHCONTENT_FULLY_DISCONNECTED=ON" in text
    for source in ("libmcfp-source", "eigen-source", "boost-regex-source"):
        assert f"/opt/Astrolabe/vendor/{source}" in text
    assert "-DCIFPP_DOWNLOAD_CCD=OFF" in text
    assert "AF3_BUILD_DATA_CACHE=/cache/af3-build-data" in text
    assert '"$AF3_BUILD_DATA_CACHE/ccd.pickle"' in text
    assert '"$AF3_BUILD_DATA_CACHE/components.cif"' in text
    assert 'rm -f "$AF3_COMPONENTS_DIR/components.cif"' not in text
    assert "print(next(iter(c.__path__)))" in text
    assert "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz" in text
    assert "apply_vendor_patches.py" not in text


def test_container_uses_persistent_download_cache() -> None:
    text = CONTAINER_DEF.read_text(encoding="utf-8")

    assert "pip3 install --upgrade pip" not in text
    assert "--no-cache-dir" not in text
    assert "PIP_DISABLE_PIP_VERSION_CHECK=1" in text
    assert "PIP_DEFAULT_TIMEOUT=600" in text
    assert "PIP_RETRIES=10" in text
    assert "PIP_CACHE_DIR=/cache/pip" in text
    assert "CONDA_PKGS_DIRS=/cache/mamba/pkgs" in text
    assert "--root-prefix /cache/mamba" in text
    assert "Dir::Cache::archives=/cache/apt/archives" in text
    assert "APT::Keep-Downloaded-Packages=true" in text
    assert 'mkdir -p "${APPTAINER_ROOTFS}/cache"' in text


def test_container_consolidates_python_and_command_tools_in_one_prefix() -> None:
    text = CONTAINER_DEF.read_text(encoding="utf-8")
    conda_lock = (REPO_ROOT / "container" / "conda-linux-64.lock").read_text(
        encoding="utf-8"
    )

    assert "python-3.11.14" in conda_lock
    for tool in ("openbabel-3.1.1", "vina-1.2.7", "dssp-4.5.7"):
        assert tool in conda_lock
    assert "https://conda.anaconda.org/conda-forge/" in conda_lock
    assert "mirrors." not in conda_lock
    assert "tensorflow-cpu==2.15.1" in RUNTIME_REQUIREMENTS.read_text(encoding="utf-8")
    assert text.count("micromamba create") == 1
    assert "-p /alphafold3_venv" in text
    assert "/opt/micromamba/envs/tools" not in text


@pytest.mark.parametrize(
    "config_name",
    ["container_af3_smoke.yaml", "container_af3_smiles_smoke.yaml"],
)
def test_container_af3_smokes_disable_official_data_pipeline(config_name: str) -> None:
    text = (CONFIG_DIR / config_name).read_text(encoding="utf-8")

    assert "run_data_pipeline: false" in text
    assert "steps: 1" not in text


def test_sif_smoketest_uses_task_scoped_doctor(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("sif_smoketest.py")
    sif_path = tmp_path / "Astrolabe_af3.sif"
    sif_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.delenv("GH_AUTODOCK_GPU", raising=False)
    monkeypatch.delenv("GH_WEIGHT_RAPIDOCK", raising=False)

    captured = {}

    def _fake_run(cmd, env=None):
        captured["cmd"] = list(cmd)
        captured["env"] = env

    monkeypatch.setattr(module, "_detect_singularity", lambda: "/usr/bin/singularity")
    monkeypatch.setattr(module, "_run", _fake_run)

    rc = int(module.main(["--sif", str(sif_path)]))

    assert rc == 0
    assert captured["cmd"][0:2] == ["/usr/bin/singularity", "exec"]
    assert "doctor" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--task") + 1] == "container_smoke"
    assert "GH_AUTODOCK_GPU" not in (captured["env"] or {})
    assert "GH_WEIGHT_RAPIDOCK" not in (captured["env"] or {})


def test_sif_cli_smoketest_manifest_covers_all_commands() -> None:
    module = _load_script_module("sif_cli_smoketest.py")

    manifest = module._command_manifest()
    leaves = [node for node in manifest["nodes"] if node["leaf"]]

    assert len(manifest["nodes"]) == 67
    assert len(leaves) == 57
    assert len(manifest["aliases"]) == 6
    assert len(manifest["legacy_aliases"]) == 4
    assert all(node["handler"] for node in leaves)
    assert module._completion_errors(manifest) == []


def test_sif_cli_execution_matrix_parses_every_canonical_leaf(tmp_path: Path) -> None:
    from astrolabe import cli

    module = _load_script_module("sif_cli_execution_smoketest.py")
    module._prepare_fixtures(tmp_path)
    cases = module._case_specs(tmp_path, REPO_ROOT)

    assert len(cases) == 57
    assert len({case.path for case in cases}) == 57
    assert {case.path for case in cases} == module._canonical_leaf_paths()
    parser = cli._build_parser()
    for case in cases:
        assert "--help" not in case.argv
        assert "-h" not in case.argv
        parsed = parser.parse_args(list(case.argv))
        assert callable(parsed._handler)

    model_paths = {
        case.path
        for case in cases
        if case.path[0]
        in {
            "run",
            "bench-binder-v1-4",
            "bench-idr-binder-v1",
            "bench-peptide-binder-v1",
            "bench-aptamer-v1",
            "bench-antibody-v1",
            "design",
            "refine-oligomer-v3",
            "refine",
            "bench-pseudocycle-v1",
            "bench-tandem-repeat-v1",
            "refine-heterooligomer-v1",
        }
    }
    model_paths.update({("paper", "run"), ("paper", "baseline")})
    assert model_paths
    assert all("--dry-run" in case.argv for case in cases if case.path in model_paths)


def test_paper_cli_execution_matrix_runs_analyses_and_model_previews(tmp_path: Path) -> None:
    import yaml

    module = _load_script_module("sif_cli_execution_smoketest.py")
    module._prepare_fixtures(tmp_path)
    cases = [case for case in module._case_specs(tmp_path, REPO_ROOT) if case.path[0] == "paper"]
    env = dict(os.environ, GH_CUDA_PREINITIALIZE="0", CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    payloads = {}
    for case in cases:
        result = module._invoke_case(case, repo_root=REPO_ROOT, env=env)
        stdout = result["stdout"]
        failures = module._validate_case(case, result)
        assert not failures, (failures, result)
        if case.json_stdout:
            payloads[case.path] = json.loads(stdout)

    assert len(cases) == 14
    paper = tmp_path / "paper"
    preview = yaml.safe_load((paper / "evaluation_resolved.yaml").read_text())
    assert preview["seqs"] == {"A": "ACGS", "B": "GASG"}
    assert preview["evaluator"]["alphafold3"]["num_samples"] == 5
    assert payloads["paper", "baseline"]["executed"] is False
    assert not (paper / "baseline").exists()
    collected = payloads["paper", "collect"]
    assert (collected["planned_units"], collected["observed_units"]) == (2, 1)
    ensemble = json.loads((paper / "two_state/per_design.json").read_text())
    assert ensemble["prediction_rows"] == 50
    assert ensemble["designs"][0]["status"] == "ok"
    assert ensemble["designs"][0]["displacement_count"] == 0
    with (paper / "agreement/02_designs.csv").open() as handle:
        design = next(csv.DictReader(handle))
    assert design["geometry_status"] == "complete"
    assert float(design["pose_rmsd"]) == pytest.approx(1)
    assert payloads["paper", "analyze", "cre"]["distance_angstrom"] == pytest.approx(30)
    assert payloads["paper", "analyze", "rna"]["c4_cyclic_rmsd_angstrom"] == pytest.approx(0, abs=1e-6)


def test_sif_cli_smoketest_outer_uses_engine_helpers_and_writes_report(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_script_module("sif_cli_smoketest.py")
    sif_path = tmp_path / "image.sif"
    sif_path.write_text("placeholder", encoding="utf-8")
    report_path = tmp_path / "report.json"
    captured = {}

    class Result:
        returncode = 0
        stdout = json.dumps(
            {
                "canonical_leaves": 57,
                "parser_aliases": 6,
                "legacy_aliases": 4,
                "failures": [],
            }
        )
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return Result()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    engine = tmp_path / "apptainer-env" / "bin" / "apptainer"
    engine.parent.mkdir(parents=True)
    engine.write_text("placeholder", encoding="utf-8")

    rc = module.main(
        [
            "--sif",
            str(sif_path),
            "--engine",
            str(engine),
            "--report",
            str(report_path),
        ]
    )

    assert rc == 0
    assert captured["command"][0] == str(engine.resolve())
    assert captured["env"]["PATH"].split(os.pathsep)[0] == str(engine.parent.resolve())
    assert json.loads(report_path.read_text(encoding="utf-8"))["canonical_leaves"] == 57


@pytest.mark.parametrize(
    ("config_name", "evaluator_type"),
    [
        ("container_af2_smoke.yaml", "alphafold2"),
        ("container_boltz_smoke.yaml", "Boltz"),
        ("container_protenix_smoke.yaml", "Protenix"),
    ],
)
def test_weighted_container_smoke_configs_use_container_env(
    config_name: str, evaluator_type: str
) -> None:
    text = (CONFIG_DIR / config_name).read_text(encoding="utf-8")

    assert "- override env: env_container" in text
    assert f"_type: {evaluator_type}" in text
    assert "steps: 2" in text


def test_ligandmpnn_container_smoke_uses_bound_weights() -> None:
    text = (CONFIG_DIR / "container_ligandmpnn_smoke.yaml").read_text(encoding="utf-8")

    assert "- override env: env_container" in text
    assert "_type: LigandMPNN" in text
    assert "model_type: soluble_mpnn" in text
    assert "mutators: [\"LigandMPNN\"]" in text
    assert "steps: 2" in text


def test_container_model_smoketest_uses_generated_runner_and_verifies_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_script_module("container_model_smoketest.py")
    runner = tmp_path / "runner.py"
    runner.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    runner.chmod(0o755)
    out_dir = tmp_path / "outputs"
    task_name = "af2_smoke"
    captured = {}

    def fake_check_call(command, *, cwd, env):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        run_dir = out_dir / task_name / "001_run"
        (run_dir / "structures").mkdir(parents=True)
        (run_dir / "scores.csv").write_text("step,score\n0,0\n", encoding="utf-8")
        (run_dir / "structures" / "model.pdb").write_text("ATOM\n", encoding="utf-8")

    monkeypatch.setattr(module.subprocess, "check_call", fake_check_call)
    rc = module.main(
        [
            "--runner",
            str(runner),
            "--config-name",
            "container_af2_smoke",
            "--out-dir",
            str(out_dir),
            "--task-name",
            task_name,
        ]
    )

    assert rc == 0
    assert captured["env"]["GH_PY"] == str(runner.resolve())
    assert captured["env"]["NVIDIA_VISIBLE_DEVICES"] == "all"
    assert captured["command"][1:4] == ["run", "--task", "container_af2_smoke"]
    report = json.loads((out_dir / "smoke_report.json").read_text(encoding="utf-8"))
    assert report["scores_bytes"] > 0


def test_container_runtime_smoketest_uses_gpu_runner_and_writes_report(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_script_module("container_runtime_smoketest.py")
    runner = tmp_path / "runner.py"
    runner.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    runner.chmod(0o755)
    report_path = tmp_path / "runtime.json"
    captured = {}
    payload = {
        "jax_version": "0.4.34",
        "python_version": "3.11.14",
        "tensorflow_version": "2.15.1",
        "torch_version": "2.7.1+cu126",
    }

    class Result:
        returncode = 0
        stdout = module.REPORT_PREFIX + json.dumps(payload) + "\n"
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return Result()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.main(["--runner", str(runner), "--report", str(report_path)]) == 0
    assert captured["command"][0] == str(runner.resolve())
    assert captured["env"]["GH_CUDA_PREINITIALIZE"] == "1"
    assert captured["env"]["NVIDIA_VISIBLE_DEVICES"] == "all"
    assert captured["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert "import pyrosetta" in captured["command"][-1]
    assert "if True:" in captured["command"][-1]
    assert "from astrolabe.app import _load_run_module" in captured["command"][-1]
    assert "_load_run_module()" in captured["command"][-1]
    tree = ast.parse(captured["command"][-1])
    preinitialize_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_preinitialize_cuda_driver"
    ]
    assert len(preinitialize_calls) == 1
    namespace_call = preinitialize_calls[0].args[0]
    assert isinstance(namespace_call, ast.Call)
    assert isinstance(namespace_call.func, ast.Attribute)
    assert isinstance(namespace_call.func.value, ast.Name)
    assert (namespace_call.func.value.id, namespace_call.func.attr) == (
        "argparse",
        "Namespace",
    )
    namespace_values = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in namespace_call.keywords
    }
    assert namespace_values == {"_cuda_preinitialize": True, "dry_run": False}
    assert json.loads(report_path.read_text(encoding="utf-8")) == payload


def test_container_runtime_smoketest_pyrosetta_default_and_reduced_opt_out(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_script_module("container_runtime_smoketest.py")

    core_code = module._inner_code(expect_pyrosetta=False)
    pyrosetta_code = module._inner_code(expect_pyrosetta=True)

    assert "if False:" in core_code
    assert "if True:" in pyrosetta_code
    assert 'importlib.util.find_spec("pyrosetta")' in core_code

    runner = tmp_path / "runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    runner.chmod(0o755)
    captured = []

    class Result:
        returncode = 0
        stdout = module.REPORT_PREFIX + "{}\n"
        stderr = ""

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **_kwargs: captured.append(command) or Result(),
    )
    assert module.main(
        ["--runner", str(runner), "--report", str(tmp_path / "default.json")]
    ) == 0
    assert "if True:" in captured[-1][-1]
    assert module.main(
        [
            "--runner",
            str(runner),
            "--report",
            str(tmp_path / "reduced.json"),
            "--expect-no-pyrosetta",
        ]
    ) == 0
    assert "if False:" in captured[-1][-1]
