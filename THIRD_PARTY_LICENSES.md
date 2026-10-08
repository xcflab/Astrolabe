# THIRD_PARTY_LICENSES

Auto-generated third-party inventory for release compliance.

- generated_at_utc: `2026-08-30 16:06:43Z`
- policy: release artifacts must include this file and review status for each `vendor/*` component and tracked in-tree derived component.

| component | path | source | license evidence | status |
|---|---|---|---|---|
| `abseil-cpp-source` | `vendor/abseil-cpp-source` | `https://github.com/abseil/abseil-cpp.git` | `LICENSE` | `ok` |
| `alphafold2` | `vendor/alphafold2` | `https://github.com/google-deepmind/alphafold.git` | `LICENSE` | `ok` |
| `alphafold3` | `vendor/alphafold3` | `https://github.com/Molaison/alphafold3.git` | `LICENSE` | `ok` |
| `biohub_esm` | `vendor/biohub_esm` | `https://github.com/Biohub/esm.git` | `LICENSE.md` | `ok` |
| `biohub_transformers` | `vendor/biohub_transformers` | `https://github.com/Biohub/transformers.git` | `LICENSE` | `ok` |
| `boltz` | `vendor/boltz` | `https://github.com/jwohlwend/boltz.git` | `LICENSE` | `ok` |
| `boost-regex-source` | `vendor/boost-regex-source` | `https://github.com/boostorg/regex.git` | `third_party_licenses/boost-regex-source/LICENSE_1_0.txt` | `ok` |
| `chai_lab` | `vendor/chai_lab` | `https://github.com/chaidiscovery/chai-lab.git` | `LICENSE` | `ok` |
| `dplm` | `vendor/dplm` | `https://github.com/Molaison/dplm.git` | `LICENSE` | `ok` |
| `dssp` | `vendor/dssp` | `-` | `LICENSE` | `ok` |
| `dssp-source` | `vendor/dssp-source` | `https://github.com/PDB-REDO/dssp.git` | `LICENSE` | `ok` |
| `eigen-source` | `vendor/eigen-source` | `https://gitlab.com/libeigen/eigen.git` | `COPYING.APACHE, COPYING.BSD, COPYING.GPL, COPYING.LGPL, COPYING.MINPACK, COPYING.MPL2, COPYING.README` | `ok` |
| `esm2` | `vendor/esm2` | `https://github.com/Molaison/gh-esm2.git` | `LICENSE` | `ok` |
| `esm3` | `vendor/esm3` | `https://github.com/Molaison/gh-esm3.git` | `LICENSE.md` | `ok` |
| `fakeroot` | `vendor/fakeroot` | `https://salsa.debian.org/clint/fakeroot.git` | `COPYING` | `ok` |
| `libcifpp` | `vendor/libcifpp` | `-` | `LICENSE` | `ok` |
| `libcifpp-source` | `vendor/libcifpp-source` | `https://github.com/PDB-REDO/libcifpp.git` | `LICENSE` | `ok` |
| `libmcfp-source` | `vendor/libmcfp-source` | `https://github.com/mhekkel/libmcfp.git` | `LICENSE` | `ok` |
| `na_mpnn` | `vendor/na_mpnn` | `https://github.com/baker-laboratory/NA-MPNN.git` | `LICENSE` | `ok` |
| `opendde` | `vendor/opendde` | `https://github.com/aurekaresearch/OpenDDE.git` | `LICENSE` | `ok` |
| `openfold_esm2` | `vendor/openfold_esm2` | `https://github.com/Molaison/openfold.git` | `LICENSE` | `ok` |
| `protein_sequence_models` | `vendor/protein_sequence_models` | `https://github.com/microsoft/protein-sequence-models.git` | `LICENSE` | `ok` |
| `protenix` | `vendor/protenix` | `https://github.com/bytedance/Protenix.git` | `LICENSE` | `ok` |
| `pybind11` | `vendor/pybind11` | `-` | `LICENSE` | `ok` |
| `pybind11-abseil-source` | `vendor/pybind11-abseil-source` | `https://github.com/pybind/pybind11_abseil.git` | `LICENSE` | `ok` |
| `pybind11-source` | `vendor/pybind11-source` | `https://github.com/pybind/pybind11.git` | `LICENSE` | `ok` |
| `rhodesign` | `vendor/rhodesign` | `https://github.com/ml4bio/RhoDesign.git` | `LICENSE` | `ok` |
| `rhofold` | `vendor/rhofold` | `https://github.com/ml4bio/RhoFold.git` | `LICENSE` | `ok` |
| `rpxdock` | `vendor/rpxdock` | `https://github.com/Molaison/gh-rpxdock.git` | `LICENSE` | `ok` |
| `sru` | `vendor/sru` | `https://github.com/asappresearch/sru.git` | `LICENSE` | `ok` |
| `surfdesign` | `vendor/surfdesign` | `https://github.com/smiles724/SurfDesign.git` | `-` | `excluded_from_runtime_release` |
| `surfpro` | `vendor/surfpro` | `https://github.com/JocelynSong/SurfPro.git` | `-` | `excluded_from_runtime_release` |
| `usalign` | `vendor/usalign` | `-` | `LICENSE` | `ok` |
| `willutil` | `vendor/willutil` | `https://github.com/willsheffler/willutil.git` | `-` | `excluded_from_runtime_release` |
| `ligandmpnn-derived` | `astrolabe/mutators/wrappers/ligandmpnn_utils.py` | `https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de` | `third_party_licenses/ligandmpnn/LICENSE` | `ok` |

## Notes

- `not_checked_out`: submodule directory exists but content is not present in current workspace.
- `license_file_not_found_top_level`: required license or source evidence is missing; review required.
- `excluded_from_runtime_release`: source is not copied into the runtime SIF; the parent repository distributes only its submodule pointer.
