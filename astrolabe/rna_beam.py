"""NA-MPNN/AlphaFold 3 refinement of synchronized RNA oligomers.

The search bookkeeping is independent of model execution. ``run_rna_beam``
uses the native mutator, shared segments and folding inputs; importing this
module does not load a predictor or provide a substitute predictor.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BeamCandidate:
    evaluation_id: int
    sequence: str
    score: float
    structure_path: str
    generation: int
    parent_id: int | None
    metrics: dict = field(default_factory=dict)


class BeamSearchState:
    """Budget, global sequence deduplication and elitist beam selection.

    Reserve an evaluation immediately before calling the predictor. A failed
    call consumes its reservation and must be propagated by the caller.
    Scores are maximized; exact ties retain earlier evaluation order.
    """

    def __init__(self, *, beam_width: int = 8, max_evaluations: int = 1000):
        for name, value in (("beam_width", beam_width), ("max_evaluations", max_evaluations)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.beam_width = beam_width
        self.max_evaluations = max_evaluations
        self.seen: set[str] = set()
        self.records: list[dict] = []
        self.beam: list[BeamCandidate] = []

    @property
    def remaining(self) -> int:
        return self.max_evaluations - len(self.records)

    def reserve(self, sequence: str, *, generation: int, parent_id: int | None) -> int | None:
        if sequence in self.seen or self.remaining == 0:
            return None
        evaluation_id = len(self.records)
        self.seen.add(sequence)
        self.records.append({
            "evaluation_id": evaluation_id, "sequence": sequence,
            "generation": generation, "parent_id": parent_id,
            "status": "running", "score": None, "structure_path": None, "metrics": {},
            "entered_beam": None, "in_final_beam": False,
        })
        return evaluation_id

    def complete(self, evaluation_id: int, *, score: float, structure_path: str, metrics: dict | None = None) -> BeamCandidate:
        score = float(score)
        if not math.isfinite(score):
            raise ValueError(f"Non-finite score for evaluation {evaluation_id}: {score}")
        row = self.records[evaluation_id]
        if row["status"] != "running":
            raise ValueError(f"Evaluation {evaluation_id} is already complete")
        metrics = dict(metrics or {})
        row.update(status="completed", score=score, structure_path=str(structure_path), metrics=metrics)
        return BeamCandidate(evaluation_id, row["sequence"], score, str(structure_path),
                             row["generation"], row["parent_id"], metrics)

    def select(self, children: list[BeamCandidate]) -> list[BeamCandidate]:
        candidates = {c.sequence: c for c in self.beam}
        for child in children:
            previous = candidates.get(child.sequence)
            if previous is None or (-child.score, child.evaluation_id) < (-previous.score, previous.evaluation_id):
                candidates[child.sequence] = child
        self.beam = sorted(candidates.values(), key=lambda c: (-c.score, c.evaluation_id))[:self.beam_width]
        selected = {c.evaluation_id for c in self.beam}
        for child in children:
            self.records[child.evaluation_id]["entered_beam"] = child.evaluation_id in selected
        for row in self.records:
            row["in_final_beam"] = row["evaluation_id"] in selected
        return self.beam


def _single_component(terms, expected_type: str) -> str:
    active = [(str(alias), term) for alias, term in terms.items() if term is not None]
    if len(active) != 1 or str(active[0][1].get("_type", "")).lower() != expected_type:
        raise ValueError(f"RNA beam search requires exactly one {expected_type} component")
    return active[0][0]


def _validate_sequence_statements(seqs) -> None:
    # The native RNA random-segment parser does not set initialized_from_random.
    for statement in seqs.values():
        for token in str(statement).split():
            segment = token.split("@", 1)[0].lstrip("+")
            if segment and set(segment) <= set("0123456789-+"):
                raise ValueError("RNA beam refinement requires a supplied candidate sequence, not random-length segments")


def _validate_shared_rna(protomers, segments, oligo: str | None = None) -> str:
    from astrolabe.core.protomer_config_parser import ProtomerTypes

    oligo = oligo or "".join(protomers)
    if oligo not in ("AAAA", "AAAAA", "ABCD", "ABCDE") or set(oligo) != set(protomers):
        raise ValueError("RNA oligomers require oligo: [AAAA]/[AAAAA], or shared protomers ABCD/ABCDE")
    first = protomers["A"]
    component_ids = [id(segments[name]) for name in first.components]
    if len(set(component_ids)) != len(component_ids):
        raise ValueError("This oligomer workflow requires interchain sharing, without intrachain repeats")
    for proto in protomers.values():
        if proto.type != ProtomerTypes.RNA or len(proto.sequence) != 87 or set(proto.sequence) - set("ACGU"):
            raise ValueError("Every oligomer chain must be an explicit, unmodified 87-nucleotide A/C/G/U RNA sequence")
        if [id(segments[name]) for name in proto.components] != component_ids:
            raise ValueError("All RNA chains must reference the same ordered segments (e.g. A: '<sequence>@shared'; B: shared)")
        if proto.ptm_dict or not proto.mutable_mask.all():
            raise ValueError("RNA oligomer refinement requires all 87 standard nucleotides to be mutable")
    return oligo


def _expand_repeated_protomer(oligo: str) -> str:
    """Map AF3's distinct chain IDs to native shared-segment protomer views.

    NAMPNN addresses physical chains by protomer name; repeating A directly
    would address only chain A of AF3's A/B/C/D[/E] structure.
    """
    if set(oligo) != {"A"}:
        return oligo
    from astrolabe.core import global_config
    from astrolabe.core.protomer_config_parser import Protomer

    physical_oligo = "ABCDE"[:len(oligo)]
    statement = " ".join(global_config.protomers["A"].components)
    for name in physical_oligo[1:]:
        proto = Protomer(name, statement, "rna")
        global_config.protomers[name] = proto
        global_config.mutable_protomers[name] = proto
    return physical_oligo


def _restore_sequence(sequence: str, protomer, segments) -> None:
    """Restore the unique shared sequence; repeated segment references update once."""
    offset = 0
    restored = set()
    for name in protomer.components:
        segment = segments[name]
        if id(segment) in restored:
            continue
        restored.add(id(segment))
        length = len(segment.sequence)
        segment.sequence = sequence[offset:offset + length]
        segment.clear_mutations()
        segment.clean_up()
        offset += length


def _sequence_key(protomer, segments, *, mutated: bool = False) -> str:
    seen = set()
    parts = []
    for name in protomer.components:
        segment = segments[name]
        if id(segment) not in seen:
            parts.append(segment.mutated_sequence if mutated else segment.sequence)
            seen.add(id(segment))
    return "".join(parts)


def _validate_bivalent_rna(protomers, segments, oligo: str) -> str:
    from astrolabe.core.protomer_config_parser import ProtomerTypes

    if oligo != "ABC" or list(protomers) != list("ABC"):
        raise ValueError("Bivalent RNA requires ordered RNA A and U1A proteins B/C, oligo: [ABC]")
    rna = protomers["A"]
    if rna.type != ProtomerTypes.RNA or len(rna.components) != 2:
        raise ValueError("RNA A must contain two references to one shared 20nt segment")
    first, second = (segments[name] for name in rna.components)
    if first is not second or len(first.sequence) != 20 or set(first.sequence) - set("ACGU"):
        raise ValueError("RNA A must contain two identical shared 20nt repeats, e.g. '20@repeat repeat'")
    if rna.ptm_dict or not rna.mutable_mask.all():
        raise ValueError("Both RNA repeats must be unmodified and mutable")
    for chain in ("B", "C"):
        proto = protomers[chain]
        if proto.type != ProtomerTypes.PROTEIN or proto.length != 96 or proto.mutable_mask.any():
            raise ValueError("Bivalent RNA requires fixed 96-residue U1A protein chains B/C")
    if protomers["B"].sequence != protomers["C"].sequence:
        raise ValueError("The two fixed U1A proteins must have identical sequences")
    return oligo


def _normalize_bivalent_sequence_statement(statement: str) -> str:
    """The native RNA random initializer expects a range, including fixed lengths."""
    tokens = statement.split()
    for index, token in enumerate(tokens):
        definition, *aliases = token.split("@")
        if definition == "20":
            tokens[index] = "@".join(["20-20", *aliases])
    return " ".join(tokens)


def validate_bivalent_objective_weights(objective_weights) -> dict[str, float]:
    keys = {"weak_interface", "repeat_contacts", "repeat_plddt", "clashes"}
    if objective_weights is None or set(objective_weights) != keys:
        raise ValueError("objective_weights must explicitly provide weak_interface, repeat_contacts, repeat_plddt and clashes")
    weights = {key: float(objective_weights[key]) for key in sorted(keys)}
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("All objective weights must be finite and nonnegative; clashes is subtracted")
    return weights


def score_bivalent_rna(metrics: dict, objective_weights) -> float:
    """Archived contact/PAE objective; clash penalty saturates at one atom pair.

    Weight keys weak_interface/repeat_contacts/repeat_plddt multiply archived
    dual_interface/repeat_cohesion/weak_arm_plddt, respectively.
    """
    weights = validate_bivalent_objective_weights(objective_weights)
    values = {key: float(metrics[key]) for key in ("dual_interface", "repeat_cohesion", "weak_arm_plddt", "interchain_clashes")}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Bivalent objective metrics must be finite")
    return (weights["weak_interface"] * values["dual_interface"]
            + weights["repeat_contacts"] * values["repeat_cohesion"]
            + weights["repeat_plddt"] * values["weak_arm_plddt"]
            - weights["clashes"] * min(values["interchain_clashes"], 1))


def proposal_seed(run_seed: int, proposal_step: int) -> int:
    return (run_seed * 100003 + proposal_step) % 2147483647 or 1


def beam_expansion_order(parents, proposals_per_parent):
    """Archived round-robin order keeps a partial final generation balanced."""
    return [(parent, index) for index in range(proposals_per_parent) for parent in parents]


def _write_table(path: Path, state: BeamSearchState) -> None:
    metric_names = sorted({key for row in state.records for key in row["metrics"]})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "evaluation_id", "proposal_step", "sequence", "generation", "parent_id", "status",
            "score", "structure_path", "entered_beam", "in_final_beam",
        ] + metric_names)
        writer.writeheader()
        for row in state.records:
            writer.writerow({**{key: value for key, value in row.items() if key != "metrics"}, **row["metrics"]})


def run_rna_beam(
    cfg, *, out_dir, seed: int, beam_width: int = 8,
    proposals_per_parent: int = 4, max_evaluations: int = 1000,
) -> dict:
    """Run the paper's 4/5-chain, 87-nt RNA oligomer refinement.

    ``cfg`` is a composed Astrolabe DictConfig with ``seqs``, ``types``,
    ``homologies`` and exactly one enabled NA-MPNN/AF3 term in their respective
    component maps. ``oligo: [AAAA]``/``[AAAAA]`` repeats candidate protomer A;
    alternatively, A defines a shared segment alias and B--D/E reuse it.
    The archived protocol uses five diffusion samples per evaluation: its
    pLDDT objective averages samples/residues while the retained structure is
    selected by AF3 ranking_score. These distinct aggregation rules are
    recorded in the summary. No random initializer, cached initial score or
    structure-start acceptance rule is used. ``out_dir`` is the exact new run
    directory, not its parent; it must not already exist.

    One AF3 model and one mutator wrapper are reused. The existing NA-MPNN
    wrapper launches its native subprocess for each proposal. Seeds for both
    proposals/native sampling use (seed*100003+proposal_step)%2147483647 or 1;
    AF3 uses the unchanged run seed for every sequence.

    Returns the JSON-serializable final beam and budget summary. Artifacts are
    01_evaluations.jsonl (append-only events, including failed calls),
    02_evaluations.csv (scores in [0,1], entry/final selection flags),
    03_final_beam.json, 04_config.yaml and native structures/raw_model_results.
    Predictor and proposal failures propagate unchanged, without replacement.
    """
    return _run_native_rna_beam(cfg, out_dir=out_dir, seed=seed, beam_width=beam_width,
                               proposals_per_parent=proposals_per_parent,
                               max_evaluations=max_evaluations, objective_weights=None)


def run_bivalent_rna_beam(
    cfg, *, out_dir, seed: int, beam_width: int, proposals_per_parent: int,
    objective_weights, max_evaluations: int = 200,
) -> dict:
    """Optimize RNA A='20@repeat repeat' with two fixed 96-aa U1A chains B/C.

    The shared 20nt unit is initialized once and is the sequence
    deduplication key; every prediction contains both copies. Explicit starting
    repeat sequences are also supported. ``beam_search.initialization`` selects
    ``archived`` (archived_starts[archived_start]), ``random`` (seeded 20nt), or
    ``configured`` (use seqs.A). The original 40nt run used three contact-guided
    shortened candidates; random is a distinct manuscript-described protocol.
    The initial AF3 prediction counts
    toward the 200-evaluation budget. Independent starts belong to the caller.

    Archived width/proposals are 8/4 and weights are 2/1/0.5/0.5. The first
    three terms are PAE-based weaker-interface quality, repeat cohesion and
    lower repeat pLDDT; the clash term is min(interchain_clashes,1).

    Native configuration/lifecycle match ``run_rna_beam``. With the archived
    five samples, the objective uses the ranking-selected CIF and its own PAE
    confidence JSON, not the five-sample mean. The sampled ensemble mean is
    retained as a separately named diagnostic.
    The output contains raw metrics, weights, score and separate geometry
    review flags; geometry flags do not silently reject evaluated candidates.
    """
    weights = validate_bivalent_objective_weights(objective_weights)
    return _run_native_rna_beam(cfg, out_dir=out_dir, seed=seed, beam_width=beam_width,
                               proposals_per_parent=proposals_per_parent,
                               max_evaluations=max_evaluations, objective_weights=weights)


def _run_native_rna_beam(
    cfg, *, out_dir, seed, beam_width, proposals_per_parent, max_evaluations,
    objective_weights,
) -> dict:
    bivalent = objective_weights is not None
    state = BeamSearchState(beam_width=beam_width, max_evaluations=max_evaluations)
    if type(proposals_per_parent) is not int or proposals_per_parent < 1:
        raise ValueError("proposals_per_parent must be a positive integer")
    if type(seed) is not int or not 1 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in 1..2**32-1")

    import numpy as np
    from omegaconf import OmegaConf, open_dict
    from astrolabe.argparser import set_random_seeds
    from astrolabe.core import global_config
    from astrolabe.evaluators.utils import construct_evaluators, _merge_conf_with_defaults
    from astrolabe.mutators.base import build_mutators
    from astrolabe.utils.base import construct_standard_input
    from run import initialize_global_config, setup_protomers

    args = copy.deepcopy(cfg)
    initialization = "configured"
    if bivalent:
        initialization = str(args.beam_search.get("initialization", "configured"))
        if initialization == "archived":
            start = args.beam_search.archived_starts[int(args.beam_search.get("archived_start", 0))]
            args.seqs.A = f"{start['sequence']}@repeat repeat"
        elif initialization == "random":
            args.seqs.A = "20@repeat repeat"
        elif initialization != "configured":
            raise ValueError("Bivalent initialization must be archived, random, or configured")
    _validate_sequence_statements({name: statement for name, statement in args.seqs.items()
                                   if not bivalent or name != "A"})
    evaluator_alias = _single_component(args.evaluator.evaluator_terms, "alphafold3")
    mutator_alias = _single_component(args.mutator.mutator_terms, "nampnn")
    af3_conf = _merge_conf_with_defaults(args.evaluator, "alphafold3",
                                       args.evaluator.evaluator_terms[evaluator_alias], alias=evaluator_alias)
    if int(af3_conf.num_seeds) != 1:
        raise ValueError("Archived RNA beam search uses exactly one fixed AF3 seed")
    if args.get("structure_start") is not None:
        raise ValueError("RNA beam search re-predicts its starting sequence; remove structure_start")

    output = Path(out_dir).expanduser().resolve()
    args._set_flag("allow_objects", True)
    with open_dict(args):
        args.out_dir = output
        args.raw_model_results_dir = output / "raw_model_results"
        args.seed = seed
        args.mutator.enabled_aliases = [mutator_alias]
        args.evaluator.enabled_aliases = [evaluator_alias]
        if bivalent:
            args.seqs.A = _normalize_bivalent_sequence_statement(str(args.seqs.A))
    global_config.reset()
    initialize_global_config(args)
    set_random_seeds(seed)
    setup_protomers(args)
    oligos = args.get("oligo") or ["".join(global_config.protomers)]
    if len(oligos) != 1:
        raise ValueError("RNA beam search refines exactly one configured oligomer")
    if bivalent:
        requested_oligo = _validate_bivalent_rna(global_config.protomers, global_config.segments, str(oligos[0]))
        oligo = requested_oligo
    else:
        requested_oligo = _validate_shared_rna(global_config.protomers, global_config.segments, str(oligos[0]))
        oligo = _expand_repeated_protomer(requested_oligo)
    protomer = global_config.protomers["A"]
    initial_sequence = _sequence_key(protomer, global_config.segments)
    global_config.init_const("num_tokens", {oligo: sum(global_config.protomers[name].length for name in oligo)})
    global_config.init_const("structure_start", None)
    global_config.set_variable("ignored_complexes", [])
    global_config.set_variable("negative_design_complexes", [])
    global_config.set_variable("mutation_num", sum(proto.mutable_length for proto in global_config.protomers.values()))
    output.mkdir(parents=True, exist_ok=False)
    args.raw_model_results_dir.mkdir()
    OmegaConf.save(args, output / "04_config.yaml")

    score_payloads = {}
    generation = 0
    proposal_count = 0
    duplicate_count = 0
    stop_reason = "budget_exhausted"
    mutator = None
    pending_children: list[BeamCandidate] = []

    with (output / "01_evaluations.jsonl").open("x", encoding="utf-8") as journal:
        def event(kind: str, **data) -> None:
            journal.write(json.dumps({"event": kind, **data}, allow_nan=False) + "\n")
            journal.flush()

        def evaluate(sequence: str, parent_id: int | None) -> BeamCandidate | None:
            evaluation_id = state.reserve(sequence, generation=generation, parent_id=parent_id)
            if evaluation_id is None:
                return None
            prediction_seed = seed
            state.records[evaluation_id]["proposal_step"] = proposal_count
            event("evaluation_started", **state.records[evaluation_id], seed=prediction_seed)
            _restore_sequence(sequence, protomer, global_config.segments)
            global_config.set_variable("step", proposal_count)
            try:
                scores = evaluator.evaluate(construct_standard_input([oligo], prediction_seed), {})
                all_plddt = np.asarray(scores.plddt[oligo], dtype=float)
                metrics = {"mean_plddt": float(all_plddt.mean()), "rna_sequence": protomer.sequence}
                score = metrics["mean_plddt"]
                if bivalent:
                    import shutil
                    from astrolabe.analysis.rna_geometry import measure_bivalent_model
                    model_path = Path(scores.complex_paths[oligo])
                    identifier = model_path.stem.removesuffix("_model")
                    confidence_path = args.raw_model_results_dir / "alphafold3" / identifier / f"{identifier}_confidences.json"
                    metrics.update(measure_bivalent_model(model_path, confidence_path))
                    sidecar = model_path.parent / confidence_path.name
                    shutil.copy2(confidence_path, sidecar)
                    metrics["confidence_path"] = str(sidecar)
                    score = score_bivalent_rna(metrics, objective_weights)
                candidate = state.complete(evaluation_id, score=score,
                                           structure_path=str(scores.complex_paths[oligo]), metrics=metrics)
            except BaseException as exc:
                state.records[evaluation_id]["status"] = "failed"
                event("evaluation_failed", **state.records[evaluation_id], error=repr(exc))
                raise
            score_payloads[evaluation_id] = scores
            event("evaluation_completed", **state.records[evaluation_id])
            return candidate

        def select_generation() -> None:
            state.select(pending_children)
            event("beam_selected", generation=generation,
                  candidates=[c.evaluation_id for c in pending_children],
                  beam=[asdict(c) for c in state.beam])
            _write_table(output / "02_evaluations.csv", state)
            keep = {c.evaluation_id for c in state.beam}
            for evaluation_id in list(score_payloads):
                if evaluation_id not in keep:
                    del score_payloads[evaluation_id]

        try:
            mutator = build_mutators(args.mutator, args.raw_model_results_dir)[mutator_alias]
            if mutator.position_selector not in (None, "None") or mutator.max_mutations_per_proto is not None:
                raise ValueError("RNA proposals require all mutable positions; remove selector and mutation limits")
            mutator.runner.batch_size = 1
            mutator.runner.number_of_batches = 1
            evaluator = construct_evaluators(args.evaluator, args.raw_model_results_dir,
                                             [oligo], enabled_aliases=[evaluator_alias])[evaluator_alias]
            event("search_started", seed=seed, beam_width=beam_width,
                  proposals_per_parent=proposals_per_parent, max_evaluations=max_evaluations,
                  oligo=requested_oligo, structure_chains=oligo, initial_sequence=initial_sequence,
                  workflow="bivalent_rna" if bivalent else "rna_oligomer", objective_weights=objective_weights)
            pending_children = [evaluate(initial_sequence, None)]
            select_generation()

            while state.remaining:
                generation += 1
                pending_children = []
                for parent, proposal_index in beam_expansion_order(list(state.beam), proposals_per_parent):
                    if not state.remaining:
                        break
                    proposal_count += 1
                    if proposal_count > max_evaluations * 10:
                        raise RuntimeError("Distinct-sequence search exhausted the archived 10x proposal cap")
                    _restore_sequence(parent.sequence, protomer, global_config.segments)
                    current_proposal_seed = proposal_seed(seed, proposal_count)
                    set_random_seeds(current_proposal_seed)
                    global_config.set_variable("step", proposal_count)
                    mutator.runner.extra_args["seed"] = current_proposal_seed
                    mutator.runner.job_name = f"generation_{generation:04d}_parent_{parent.evaluation_id:04d}_proposal_{proposal_index:02d}"
                    mutator.mutate(global_config.protomers, score_payloads[parent.evaluation_id], {})
                    sequence = _sequence_key(protomer, global_config.segments, mutated=True)
                    candidate = evaluate(sequence, parent.evaluation_id)
                    if candidate is None:
                        duplicate_count += 1
                        event("duplicate_proposal", generation=generation, parent_id=parent.evaluation_id,
                              proposal_index=proposal_index, sequence=sequence, seed=current_proposal_seed)
                    else:
                        pending_children.append(candidate)
                select_generation()

            result = {"out_dir": str(output), "seed": seed, "oligo": requested_oligo,
                      "structure_chains": oligo,
                      "beam_width": beam_width, "proposals_per_parent": proposals_per_parent,
                      "max_evaluations": max_evaluations, "evaluations": len(state.records),
                      "proposals": proposal_count, "duplicate_proposals": duplicate_count,
                      "stop_reason": stop_reason, "budget_complete": state.remaining == 0,
                      "score": "weighted_bivalent_score" if bivalent else "mean_plddt",
                      "score_scale": None if bivalent else [0, 1], "objective_weights": objective_weights,
                      "sequence_key_length": len(initial_sequence),
                      "initialization": initialization,
                      "af3_samples_per_evaluation": int(af3_conf.num_samples),
                      "score_aggregation": "ranking-selected sample contact/PAE objective" if bivalent else "mean pLDDT over all diffusion samples and residues",
                      "structure_selection": "AF3 ranking_score",
                      "proposal_order": "round_robin", "proposal_cap": 10 * max_evaluations,
                      "beam": [asdict(c) for c in state.beam]}
            (output / "03_final_beam.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            _restore_sequence(state.beam[0].sequence, protomer, global_config.segments)
            return result
        except BaseException as exc:
            event("search_failed", generation=generation, evaluations=len(state.records), budget_complete=False, error=repr(exc))
            raise
        finally:
            _write_table(output / "02_evaluations.csv", state)
            if mutator is not None:
                mutator.shutdown()
