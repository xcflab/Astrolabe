from __future__ import annotations

import pytest

from astrolabe.core.binding_site import BindingSiteSpec, ChainPositions
from astrolabe.evaluators.binding_site_constraints import (
    build_boltz_binding_site_constraints,
    build_chai_binding_site_rows,
    build_protenix_binding_site_constraint,
)


def test_build_boltz_binding_site_constraints_pocket_only() -> None:
    spec = BindingSiteSpec(
        binder=ChainPositions(chain="A", positions=(1,)),
        target=ChainPositions(chain="B", positions=(2, 3)),
        mode="both",
        max_distance=6.0,
        min_distance=0.0,
    )
    chain2input_chain = {"A": "A", "B": "B"}  # input_chain -> original_chain

    constraints = build_boltz_binding_site_constraints(
        spec,
        chain2input_chain=chain2input_chain,
        context="AB",
    )
    assert constraints == [
        {
            "pocket": {
                "binder": "A",
                "contacts": [["B", 2], ["B", 3]],
                "max_distance": 6.0,
                "force": True,
            }
        }
    ]


def test_build_boltz_binding_site_constraints_paratope_only_reverses_pocket() -> None:
    spec = BindingSiteSpec(
        binder=ChainPositions(chain="A", positions=(1, 3)),
        target=ChainPositions(chain="B", positions=()),
        mode="both",
        max_distance=6.0,
        min_distance=0.0,
    )
    chain2input_chain = {"A": "A", "B": "B"}

    constraints = build_boltz_binding_site_constraints(
        spec,
        chain2input_chain=chain2input_chain,
        context="AB",
    )
    assert constraints == [
        {
            "pocket": {
                "binder": "B",
                "contacts": [["A", 1], ["A", 3]],
                "max_distance": 6.0,
                "force": True,
            }
        }
    ]


def test_build_protenix_binding_site_constraint_includes_contact() -> None:
    spec = BindingSiteSpec(
        binder=ChainPositions(chain="A", positions=(1, 3)),
        target=ChainPositions(chain="B", positions=(2,)),
        mode="both",
        max_distance=5.5,
        min_distance=1.25,
    )
    chain_id_to_entities = {"A": [1], "B": [2]}

    constraint = build_protenix_binding_site_constraint(
        spec,
        chain_id_to_entities=chain_id_to_entities,
        context="AB",
    )
    assert constraint is not None
    assert constraint["pocket"] == {
        "binder_chain": {"entity": 1, "copy": 1},
        "contact_residues": [{"entity": 2, "copy": 1, "position": 2}],
        "max_distance": 5.5,
    }
    assert constraint["contact"] == [
        {
            "entity1": 1,
            "copy1": 1,
            "position1": 1,
            "entity2": 2,
            "copy2": 1,
            "position2": 2,
            "max_distance": 5.5,
            "min_distance": 1.25,
        },
        {
            "entity1": 1,
            "copy1": 1,
            "position1": 3,
            "entity2": 2,
            "copy2": 1,
            "position2": 2,
            "max_distance": 5.5,
            "min_distance": 1.25,
        },
    ]


def test_build_protenix_binding_site_constraint_paratope_only_reverses_pocket() -> None:
    spec = BindingSiteSpec(
        binder=ChainPositions(chain="A", positions=(1, 3)),
        target=ChainPositions(chain="B", positions=()),
        mode="both",
        max_distance=5.5,
        min_distance=1.25,
    )
    chain_id_to_entities = {"A": [1], "B": [2]}

    constraint = build_protenix_binding_site_constraint(
        spec,
        chain_id_to_entities=chain_id_to_entities,
        context="AB",
    )
    assert constraint is not None
    assert constraint == {
        "pocket": {
            "binder_chain": {"entity": 2, "copy": 1},
            "contact_residues": [
                {"entity": 1, "copy": 1, "position": 1},
                {"entity": 1, "copy": 1, "position": 3},
            ],
            "max_distance": 5.5,
        }
    }


def test_build_protenix_binding_site_constraint_rejects_multicopy_selector() -> None:
    spec = BindingSiteSpec(
        binder=ChainPositions(chain="protomer:A", positions=(1,)),
        target=ChainPositions(chain="protomer:B", positions=(2,)),
        mode="pocket",
        max_distance=6.0,
        min_distance=0.0,
    )
    chain_id_to_entities = {"A": [1], "B": [2], "C": [3]}

    with pytest.raises(ValueError, match="exactly one chain"):
        build_protenix_binding_site_constraint(
            spec,
            chain_id_to_entities=chain_id_to_entities,
            context="AAB",
            oligo_name="AAB",
        )


def test_build_chai_binding_site_rows_contains_expected_fields() -> None:
    spec = BindingSiteSpec(
        binder=ChainPositions(chain="A", positions=(1, 3)),
        target=ChainPositions(chain="B", positions=(2,)),
        mode="both",
        max_distance=6.0,
        min_distance=0.0,
    )
    seq_by_chain = {"A": "ACDE", "B": "FGHI"}

    rows = build_chai_binding_site_rows(spec, seq_by_chain=seq_by_chain, context="AB")
    assert len(rows) == 3
    assert rows[0]["connection_type"] == "pocket"
    assert rows[0]["chainA"] == "A"
    assert rows[0]["res_idxA"] == ""
    assert rows[0]["chainB"] == "B"
    assert rows[0]["res_idxB"] == "G2"
    assert rows[0]["restraint_id"] == "restraint_0"
    assert rows[1]["connection_type"] == "contact"
    assert rows[2]["connection_type"] == "contact"


def test_build_chai_binding_site_rows_paratope_only_reverses_pocket() -> None:
    spec = BindingSiteSpec(
        binder=ChainPositions(chain="A", positions=(1, 3)),
        target=ChainPositions(chain="B", positions=()),
        mode="both",
        max_distance=6.0,
        min_distance=0.0,
    )
    seq_by_chain = {"A": "ACDE", "B": "FGHI"}

    rows = build_chai_binding_site_rows(spec, seq_by_chain=seq_by_chain, context="AB")
    assert len(rows) == 2
    assert rows[0]["connection_type"] == "pocket"
    assert rows[0]["chainA"] == "B"
    assert rows[0]["res_idxA"] == ""
    assert rows[0]["chainB"] == "A"
    assert rows[0]["res_idxB"] == "A1"
    assert rows[1]["connection_type"] == "pocket"
    assert rows[1]["chainA"] == "B"
    assert rows[1]["res_idxA"] == ""
    assert rows[1]["chainB"] == "A"
    assert rows[1]["res_idxB"] == "D3"


def test_build_chai_binding_site_rows_rejects_multicopy_selector() -> None:
    spec = BindingSiteSpec(
        binder=ChainPositions(chain="protomer:A", positions=(1,)),
        target=ChainPositions(chain="protomer:B", positions=(2,)),
        mode="pocket",
        max_distance=6.0,
        min_distance=0.0,
    )
    seq_by_chain = {"A": "ACDE", "B": "FGHI", "C": "KLMN"}

    with pytest.raises(ValueError, match="exactly one chain"):
        build_chai_binding_site_rows(
            spec,
            seq_by_chain=seq_by_chain,
            context="AAB",
            oligo_name="AAB",
        )


def test_build_boltz_binding_site_constraints_rejects_multicopy_selector() -> None:
    spec = BindingSiteSpec(
        binder=ChainPositions(chain="protomer:A", positions=(1,)),
        target=ChainPositions(chain="protomer:B", positions=(2,)),
        mode="pocket",
        max_distance=6.0,
        min_distance=0.0,
    )
    chain2input_chain = {"A": "A", "B": "B", "C": "C"}  # input_chain -> original_chain

    with pytest.raises(ValueError, match="exactly one chain"):
        build_boltz_binding_site_constraints(
            spec,
            chain2input_chain=chain2input_chain,
            context="AAB",
            oligo_name="AAB",
        )


def test_build_boltz_binding_site_constraints_supports_explicit_chain_override() -> None:
    spec = BindingSiteSpec(
        binder=ChainPositions(chain="chain:B", positions=(1,)),
        target=ChainPositions(chain="protomer:B", positions=(2,)),
        mode="pocket",
        max_distance=6.0,
        min_distance=0.0,
    )
    chain2input_chain = {"A": "A", "B": "B", "C": "C"}  # input_chain -> original_chain

    constraints = build_boltz_binding_site_constraints(
        spec,
        chain2input_chain=chain2input_chain,
        context="AAB",
        oligo_name="AAB",
    )
    assert constraints == [
        {
            "pocket": {
                "binder": "B",
                "contacts": [["C", 2]],
                "max_distance": 6.0,
                "force": True,
            }
        }
    ]


def test_build_boltz_binding_site_constraints_rejects_multicopy_self_selector() -> None:
    spec = BindingSiteSpec(
        binder=ChainPositions(chain="protomer:A", positions=(1,)),
        target=ChainPositions(chain="protomer:A", positions=(2,)),
        mode="pocket",
        max_distance=6.0,
        min_distance=0.0,
    )
    chain2input_chain = {"A": "A", "B": "B", "C": "C"}  # input_chain -> original_chain

    with pytest.raises(ValueError, match="exactly one chain"):
        build_boltz_binding_site_constraints(
            spec,
            chain2input_chain=chain2input_chain,
            context="AAA",
            oligo_name="AAA",
        )
