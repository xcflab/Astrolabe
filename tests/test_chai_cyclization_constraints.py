from __future__ import annotations

from pathlib import Path

from astrolabe.core import global_config
from astrolabe.core.folding_input import Input, ProteinChain
from astrolabe.evaluators.chai1 import Chai1Evaluator


def test_chai_prepare_input_writes_covalent_constraints(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("binding_site", None)
    global_config.set_variable("step", 0)

    ev = Chai1Evaluator(model_config={}, output_dir=tmp_path / "chai_out")

    inp = Input(
        chains=[
            ProteinChain(
                id="A",
                sequence="ACDEFGHIK",
                ptms=[],
                unpaired_msa="",
                paired_msa="",
                templates=[],
            ),
        ],
        name="A",
        rng_seeds=[0],
        bonded_atom_pairs=[(("A", 1, "N"), ("A", 9, "C"))],
        user_ccd=None,
    )

    runtime: dict[str, object] = {}
    out = ev.prepare_input({"A": inp}, runtime)
    constraint_path = out["A"]["constraint_path"]
    assert constraint_path is not None
    p = Path(constraint_path)
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "connection_type" in text
    assert "covalent" in text
    assert "A,A1@N,A,K9@C" in text.replace("\r\n", "\n").replace("\n", "")

