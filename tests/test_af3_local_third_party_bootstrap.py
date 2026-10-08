from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_SPEC_PATH = REPO_ROOT / "config" / "third_party_envs.yaml"
AF3_CMAKE_PATH = REPO_ROOT / "vendor" / "alphafold3" / "CMakeLists.txt"


def test_gh_env_exports_af3_bootstrap_sources_into_cmake_args() -> None:
    spec = OmegaConf.to_container(OmegaConf.load(str(ENV_SPEC_PATH)), resolve=True)
    gh_env = next(env for env in spec["envs"] if env["name"] == "gh")

    cmake_args = gh_env["pip_env"]["CMAKE_ARGS"]

    assert "-DAF3_THIRD_PARTY_DIR={{ROOT_PREFIX}}/bootstrap_sources" in cmake_args


def test_gh_env_installs_tensorflow_for_alphafold2() -> None:
    spec = OmegaConf.to_container(OmegaConf.load(str(ENV_SPEC_PATH)), resolve=True)
    gh_env = next(env for env in spec["envs"] if env["name"] == "gh")

    assert "flatbuffers>=23.5.26" in gh_env["pip"]
    assert "tensorflow-cpu==2.15.1" in gh_env["pip"]


def test_alphafold3_cmake_accepts_local_libcifpp_bootstrap_dir() -> None:
    cmake_text = AF3_CMAKE_PATH.read_text(encoding="utf-8")

    assert 'set(AF3_CIFPP_SOURCE_DIR "${AF3_THIRD_PARTY_DIR}/libcifpp")' in cmake_text
    assert 'FetchContent_Declare(cifpp SOURCE_DIR "${AF3_CIFPP_SOURCE_DIR}")' in cmake_text


def test_alphafold3_cmake_forces_libcifpp_ccd_download() -> None:
    cmake_text = AF3_CMAKE_PATH.read_text(encoding="utf-8")

    assert 'set(CIFPP_DOWNLOAD_CCD ON CACHE BOOL' in cmake_text
    assert cmake_text.index('set(CIFPP_DOWNLOAD_CCD ON CACHE BOOL') < cmake_text.index(
        "FetchContent_MakeAvailable(pybind11 abseil-cpp pybind11_abseil cifpp)"
    )
