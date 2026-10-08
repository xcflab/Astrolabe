from __future__ import annotations

from pathlib import Path

from astrolabe.core import global_config
from astrolabe.core.folding_input import Input, ProteinChain
from astrolabe.evaluators.chai1 import Chai1Evaluator


def test_chai_prepare_input_writes_explicit_restraints(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("binding_site", None)
    global_config.init_const(
        "chai_restraints",
        {
            "AB": [
                {
                    "chainA": "A",
                    "res_idxA": 2,
                    "chainB": "B",
                    "res_idxB": 3,
                    "connection_type": "contact",
                    "max_distance_angstrom": 5.5,
                }
            ]
        },
    )
    global_config.set_variable("step", 0)

    ev = Chai1Evaluator(model_config={}, output_dir=tmp_path / "chai_out")

    inp = Input(
        chains=[
            ProteinChain(
                id="A",
                sequence="MQQL",
                ptms=[],
                unpaired_msa="",
                paired_msa="",
                templates=[],
            ),
            ProteinChain(
                id="B",
                sequence="ACDE",
                ptms=[],
                unpaired_msa="",
                paired_msa="",
                templates=[],
            ),
        ],
        name="AB",
        rng_seeds=[0],
        user_ccd=None,
    )

    runtime: dict[str, object] = {}
    out = ev.prepare_input({"AB": inp}, runtime)
    constraint_path = out["AB"]["constraint_path"]
    assert constraint_path is not None
    text = Path(constraint_path).read_text(encoding="utf-8")
    compact = text.replace("\r\n", "\n").replace("\n", "")
    assert "connection_type" in text
    assert "contact" in text
    assert "A,Q2,B,D3,contact,1.0,0.0,5.5,user_chai_restraint,user_restraint_0" in compact
