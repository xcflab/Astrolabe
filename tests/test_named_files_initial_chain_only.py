from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import unittest

from astrolabe.core import global_config


def _load_protomer_parser():
    import importlib

    constant_module_name = "astrolabe.utils.constant"
    constant_module = sys.modules.get(constant_module_name)
    if constant_module is not None and not hasattr(constant_module, "aa_freq"):
        del sys.modules[constant_module_name]

    module_name = "astrolabe.core.protomer_config_parser"
    module = sys.modules.get(module_name)
    if module is not None and not hasattr(module, "register_files"):
        del sys.modules[module_name]
    return importlib.import_module(module_name)


class TestNamedFilesInitialChainOnly(unittest.TestCase):
    def test_initial_chain_only_statement_reads_full_chain(self) -> None:
        cif = "\n".join(
            [
                "data_test",
                "#",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.id",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_seq_id",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.occupancy",
                "_atom_site.B_iso_or_equiv",
                "_atom_site.pdbx_PDB_ins_code",
                "_atom_site.auth_asym_id",
                "_atom_site.auth_seq_id",
                "_atom_site.auth_comp_id",
                "ATOM 1 C CA ALA A 1 0 0 0 1 20 ? A 1 ALA",
                "ATOM 2 C CA GLY A 2 1 0 0 1 20 ? A 2 GLY",
                "#",
                "",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "toy.cif"
            path.write_text(cif, encoding="utf-8")

            global_config.reset()
            parser = _load_protomer_parser()
            register_files = parser.register_files
            protein_file_handler = parser.ProteinFileHandler
            register_files({"initial": str(path)})

            seq, ptm = protein_file_handler().handle("initial:A")

        self.assertEqual(seq, "AG")
        self.assertEqual(ptm, {})

    def test_initialize_global_config_preserves_registered_named_files(self) -> None:
        cif = "\n".join(
            [
                "data_test",
                "#",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.id",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_seq_id",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.occupancy",
                "_atom_site.B_iso_or_equiv",
                "_atom_site.pdbx_PDB_ins_code",
                "_atom_site.auth_asym_id",
                "_atom_site.auth_seq_id",
                "_atom_site.auth_comp_id",
                "ATOM 1 C CA ALA A 1 0 0 0 1 20 ? A 1 ALA",
                "ATOM 2 C CA GLY A 2 1 0 0 1 20 ? A 2 GLY",
                "#",
                "",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "toy.cif"
            path.write_text(cif, encoding="utf-8")

            script = f"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProteinFileHandler, register_files

run_path = Path({str(pathlib.Path(__file__).resolve().parents[1] / "run.py")!r})
spec = importlib.util.spec_from_file_location("gh_run_module_test", run_path)
assert spec is not None and spec.loader is not None
run_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_mod)

path = Path({str(path)!r})
tmpdir = Path({tmpdir!r})
global_config.reset()
register_files({{"initial": str(path)}})
parsed_before = global_config.named_files["initial"]
run_mod.initialize_global_config(
    SimpleNamespace(
        exclude_AA=[],
        initial_temperature=0.0,
        out_dir=tmpdir,
        raw_model_results_dir=tmpdir / "raw",
        use_metropolis_acceptance=False,
        enable_pyrosetta=False,
        skip_complex_poses=False,
        allow_mmcif_pdb_fallback=False,
        protenix_constraint=None,
        binding_site=None,
        chai_restraints=None,
        cyclization=None,
        ligand_resnames=None,
        nonstandard_residues=None,
    )
)
assert global_config.named_files["initial"] is parsed_before
seq, ptm = ProteinFileHandler().handle("initial:A")
assert seq == "AG"
assert ptm == {{}}
"""
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=pathlib.Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
