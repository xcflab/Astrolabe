"""Constructible typed data for the unified design framework.

The module deliberately contains no model, runtime, or coordinator imports.
Identity-bearing factories hash canonical content. Python object identity is
never part of a framework key.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from pathlib import Path
import math
from typing import Any, Generic, Literal, Protocol, TypeVar


I = TypeVar("I")
O = TypeVar("O")
T = TypeVar("T")
S = TypeVar("S")


@dataclass(frozen=True)
class Key:
    digest: str


@dataclass(frozen=True)
class Blob:
    identity: Key
    codec: str


@dataclass(frozen=True)
class Polymer:
    kind: Literal["protein", "rna", "dna"]
    tokens: tuple[str, ...]
    modifications: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class Chemical:
    graph: Blob


@dataclass(frozen=True)
class Assignment:
    variable: str
    value: Polymer | Chemical


@dataclass(frozen=True)
class Occurrence:
    identity: Key
    state: str
    assembly: str
    chain: str
    component: str
    variable: str
    index_map: Blob


@dataclass(frozen=True)
class Domain:
    variable: str
    fixed_tokens: tuple[int, ...]
    fixed_atoms: tuple[str, ...]
    allowed_values: Blob | None


@dataclass(frozen=True)
class System:
    identity: Key
    variables: tuple[str, ...]
    occurrences: tuple[Occurrence, ...]
    topology: Blob
    domains: tuple[Domain, ...]

    def edit(
        self,
        base: Candidate,
        replacements: tuple[Assignment, ...],
        *,
        data: DataStore | None = None,
    ) -> Candidate:
        """Return a new candidate after validated whole-variable replacements.

        ``data`` is required when an affected domain has owner restrictions or
        chemical fixed atoms. It is never converted into a hidden global store.
        """

        if not isinstance(base, Candidate):
            raise TypeError("base must be a Candidate")
        if base.system != self.identity:
            raise ValueError("base candidate belongs to a different system")
        if self.identity != _system_identity(
            self.variables,
            self.occurrences,
            self.topology,
            self.domains,
        ):
            raise ValueError("system identity does not match its content")

        current = _validated_assignments(self, base.assignments)
        if base.identity != _candidate_identity(self.identity, current):
            raise ValueError("base candidate identity does not match its content")

        replacement_map: dict[str, Assignment] = {}
        for replacement in replacements:
            if not isinstance(replacement, Assignment):
                raise TypeError("replacements must contain Assignment values")
            if replacement.variable not in self.variables:
                raise ValueError(
                    f"replacement references unknown variable {replacement.variable!r}"
                )
            if replacement.variable in replacement_map:
                raise ValueError(
                    f"multiple replacements supplied for variable {replacement.variable!r}"
                )
            replacement_map[replacement.variable] = _normalize_assignment(replacement)

        updated = {
            assignment.variable: replacement_map.get(assignment.variable, assignment)
            for assignment in current
        }
        normalized = tuple(
            updated[variable]
            for variable in self.variables
        )
        self._validate_edit(current, normalized, replacement_map, data=data)
        return Candidate(
            identity=_candidate_identity(self.identity, normalized),
            system=self.identity,
            assignments=normalized,
        )

    def _validate_edit(
        self,
        base: tuple[Assignment, ...],
        updated: tuple[Assignment, ...],
        replacements: dict[str, Assignment],
        *,
        data: DataStore | None,
    ) -> None:
        base_map = {assignment.variable: assignment for assignment in base}
        updated_map = {assignment.variable: assignment for assignment in updated}
        domains = {domain.variable: domain for domain in self.domains}

        for variable in self.variables:
            old = base_map[variable]
            new = updated_map[variable]
            domain = domains.get(variable)
            if type(old.value) is not type(new.value):
                raise ValueError(f"variable {variable!r} cannot change molecule kind")

            if isinstance(new.value, Polymer):
                _validate_polymer(variable, new.value)
                if old.value.kind != new.value.kind:
                    raise ValueError(f"variable {variable!r} cannot change polymer kind")
                if len(old.value.tokens) != len(new.value.tokens):
                    raise ValueError(
                        f"variable {variable!r} cannot change token count in an edit"
                    )
                if old.value.modifications != new.value.modifications:
                    raise ValueError(
                        f"variable {variable!r} cannot change modification topology (PTM) in an assignment edit"
                    )
                for index, _ in old.value.modifications:
                    if old.value.tokens[index] != new.value.tokens[index]:
                        raise ValueError(
                            f"modified token {variable}[{index}] needs an owned chemical transformation"
                        )
                if domain is None:
                    if old.value.modifications != new.value.modifications:
                        raise ValueError(
                            f"variable {variable!r} cannot change modification "
                            "topology without an explicit Domain"
                        )
                    continue
                modification_map = dict(new.value.modifications)
                old_modification_map = dict(old.value.modifications)
                for index in domain.fixed_tokens:
                    if old.value.tokens[index] != new.value.tokens[index]:
                        raise ValueError(
                            f"fixed token at {variable}[{index}] cannot change"
                        )
                    if old_modification_map.get(index) != modification_map.get(index):
                        raise ValueError(
                            f"modification at fixed token {variable}[{index}] cannot change"
                        )
                if domain.fixed_atoms:
                    raise ValueError(
                        f"polymer variable {variable!r} cannot declare fixed atoms"
                    )
                if domain.allowed_values is not None:
                    allowed = _read_allowed_tokens(domain.allowed_values, data)
                    for index, tokens in allowed.items():
                        if index >= len(new.value.tokens):
                            raise ValueError(
                                f"allowed-value index {index} is outside {variable!r}"
                            )
                        if new.value.tokens[index] not in tokens:
                            raise ValueError(
                                f"token {new.value.tokens[index]!r} at {variable}[{index}] "
                                "is not allowed"
                            )
                continue

            if not isinstance(new.value, Chemical):
                raise TypeError(
                    f"variable {variable!r} has unsupported assignment type "
                    f"{type(new.value).__name__}"
                )
            _validate_chemical(variable, new.value)
            if domain is None:
                if old.value.graph != new.value.graph:
                    raise ValueError(
                        f"variable {variable!r} cannot change chemical graph "
                        "topology without an explicit Domain"
                    )
                continue
            if old.value.graph != new.value.graph:
                if data is None:
                    raise ValueError("Chemical graph edits require DataStore owner validation")
                data.validate_chemical_edit(self, variable, old.value.graph, new.value.graph)
            if domain.fixed_tokens:
                raise ValueError(
                    f"chemical variable {variable!r} cannot declare fixed tokens"
                )
            if domain.fixed_atoms:
                if data is None:
                    raise ValueError(
                        f"variable {variable!r} requires DataStore validation "
                        "for fixed chemical atoms"
                    )
                validator = getattr(data, "validate_chemical_reference", None)
                if validator is None:
                    raise TypeError(
                        "data must provide validate_chemical_reference() "
                        "for fixed chemical atoms"
                    )
                old_records = validator(old.value.graph, domain.fixed_atoms)
                new_records = validator(new.value.graph, domain.fixed_atoms)
                if old_records != new_records:
                    raise ValueError(
                        f"fixed chemical atoms changed for variable {variable!r}"
                    )
            if domain.allowed_values is not None:
                raise ValueError(
                    f"allowed_tokens restrictions do not apply to chemical "
                    f"variable {variable!r}"
                )


@dataclass(frozen=True)
class Candidate:
    identity: Key
    system: Key
    assignments: tuple[Assignment, ...]


@dataclass(frozen=True)
class Dependency:
    role: str
    identity: Key


@dataclass(frozen=True)
class InputManifest:
    identity: Key
    dependencies: tuple[Dependency, ...]


@dataclass(frozen=True)
class Sample:
    protocol: Key
    replicate: str
    seed: int
    index: str


@dataclass(frozen=True)
class Artifact:
    identity: Key
    candidate: Key
    occurrences: tuple[Occurrence, ...]
    sample: Sample
    content: Blob
    derived_from: tuple[Key, ...]
    inputs: InputManifest


@dataclass(frozen=True)
class Axis:
    name: str
    kind: str
    index_map: Blob


@dataclass(frozen=True)
class Subject:
    candidate: Key
    state: str | None
    assembly: str | None
    occurrences: tuple[Key, ...]
    artifact: Key | None
    sample: Sample | None


@dataclass(frozen=True)
class Measurement:
    identity: Key
    quantity: str
    definition: Key
    units: str
    producer: Key
    subjects: tuple[Subject, ...]
    axes: tuple[Axis, ...]
    values: Blob | None
    missing: Literal["present", "partial", "unavailable", "failed"]
    valid_mask: Blob | None
    inputs: InputManifest


@dataclass(frozen=True)
class Selection:
    subject: Subject
    axis_name: str
    index_map: Blob
    mask: Blob


@dataclass(frozen=True)
class Reduction:
    axes: tuple[str, ...]
    method: str
    parameters: Blob
    missing: Literal["fail", "omit", "propagate"]


class MeasurementView(Protocol):
    def select(self, selection: Selection) -> MeasurementView: ...

    def between(
        self,
        left: Selection,
        right: Selection,
        *,
        left_axis: str,
        right_axis: str,
    ) -> MeasurementView: ...

    def reduce(self, reduction: Reduction) -> Measurement: ...

    def materialize(self) -> Measurement: ...


@dataclass(frozen=True)
class Evidence:
    identity: Key
    evaluated_candidate: Key
    artifacts: tuple[Artifact, ...]
    measurements: tuple[Measurement, ...]
    inputs: InputManifest

    def with_measurements(self, measurements: tuple[Measurement, ...]) -> Evidence:
        merged = list(self.measurements)
        positions = {
            measurement.identity.digest: index
            for index, measurement in enumerate(merged)
        }
        for measurement in tuple(measurements):
            if not isinstance(measurement, Measurement):
                raise TypeError("measurements must contain Measurement values")
            if not measurement.subjects:
                raise ValueError("measurement must contain at least one subject")
            if any(
                subject.candidate != self.evaluated_candidate
                for subject in measurement.subjects
            ):
                raise ValueError(
                    "measurement subjects must remain the evaluated candidate; "
                    "reference observations require a separate Evidence"
                )
            digest = measurement.identity.digest
            existing_index = positions.get(digest)
            if existing_index is None:
                positions[digest] = len(merged)
                merged.append(measurement)
                continue
            if merged[existing_index] != measurement:
                raise ValueError(
                    f"measurement identity collision for {digest!r}"
                )
        measurements = tuple(merged)
        return Evidence(
            identity=_evidence_identity(
                self.evaluated_candidate,
                self.artifacts,
                measurements,
                self.inputs,
            ),
            evaluated_candidate=self.evaluated_candidate,
            artifacts=self.artifacts,
            measurements=measurements,
            inputs=self.inputs,
        )


@dataclass(frozen=True)
class EvidenceUse:
    evidence: Evidence
    relation: Literal["current", "base_guidance", "reference"]
    working_candidate: Key

    def __post_init__(self):
        if self.relation not in ("current", "base_guidance", "reference"):
            raise ValueError("Unknown evidence-use relation")
        if self.relation == "current" and self.evidence.evaluated_candidate != self.working_candidate:
            raise ValueError("Current evidence must belong to the working candidate")


@dataclass(frozen=True)
class Proposal:
    identity: Key
    parents: tuple[Key, ...]
    base: Key
    candidate: Candidate
    used_evidence: tuple[Key, ...]
    protocol: Key
    artifacts: tuple[Artifact, ...] = ()


@dataclass(frozen=True)
class Capability:
    namespace: str
    name: str
    version: str


@dataclass(frozen=True)
class Requirement:
    component: str
    capability: Capability
    enforcement: Literal["required", "preferred"]


@dataclass(frozen=True)
class SelectedOptionsSchema:
    owner: str
    version: str
    json_schema: str
    effective_options_json: str


class Codec(Protocol[T]):
    identity: str

    def encode(self, value: T) -> bytes: ...

    def decode(self, data: bytes) -> T: ...

    def references(self, value: T) -> tuple[Blob, ...]: ...


class DataStore(Protocol):
    """Type-only view of the concrete persistent store."""

    def put(self, value: Any, codec: Codec[Any]) -> Blob: ...

    def get(self, reference: Blob, codec: Codec[T] | None = None) -> T: ...

    def persist(self, references: tuple[Blob, ...]) -> None: ...

    def validate_chemical_edit(self, system: System, variable: str, old: Blob, new: Blob) -> None: ...

    def validate_chemical_reference(
        self,
        reference: Blob,
        fixed_atoms: tuple[str, ...],
    ) -> Any: ...


@dataclass(frozen=True)
class ComponentDescriptor(Generic[I, O]):
    identity: Key
    alias: str
    request_type: type[I]
    result_type: type[O]
    options_schema: SelectedOptionsSchema
    request_codec: Codec[I]
    result_codec: Codec[O]
    capabilities: tuple[Capability, ...]


class Operation(Protocol[I, O]):
    descriptor: ComponentDescriptor[I, O]


@dataclass(frozen=True)
class Conditioning:
    identity: Key
    owner: Key
    intent: Blob
    mechanism: str
    effective_parameters: Blob
    approximations: tuple[str, ...]
    inputs: InputManifest


@dataclass(frozen=True)
class InitializeRequest:
    system: System
    sample: Sample


@dataclass(frozen=True)
class Initialized:
    candidate: Candidate
    evidence: Evidence | None


@dataclass(frozen=True)
class PredictRequest:
    system: System
    candidate: Candidate
    assemblies: tuple[str, ...]
    conditioning: tuple[Conditioning, ...]
    sources: tuple[Dependency, ...]
    samples: tuple[Sample, ...]


@dataclass(frozen=True)
class TransformRequest:
    artifact: Artifact


@dataclass(frozen=True)
class ArtifactBank:
    artifacts: tuple[Artifact, ...]


@dataclass(frozen=True)
class ChooseArtifactRequest:
    bank: ArtifactBank
    index: int


@dataclass(frozen=True)
class SelectRequest:
    system: System
    candidate: Candidate
    evidence: tuple[EvidenceUse, ...]


@dataclass(frozen=True)
class PackRequest:
    artifacts: tuple[Artifact, ...]
    selections: tuple[Selection, ...]
    ties: Blob
    state_weights: tuple[float, ...]


@dataclass(frozen=True)
class PackedStates:
    artifact: Artifact
    mapping: Blob
    ties: Blob
    state_weights: tuple[float, ...]


@dataclass(frozen=True)
class SamplingPolicy:
    identity: Key
    options: Blob


@dataclass(frozen=True)
class Applied:
    request: Key
    operation: str
    parameters: Key
    decoding_boundary: str


@dataclass(frozen=True)
class IndependentDistribution:
    identity: Key
    alphabet: tuple[str, ...]
    sites: Blob
    samples: tuple[Sample, ...]
    values: Blob
    semantics: Literal["logits", "log_probs", "sampling_probs", "empirical"]
    conditioning: InputManifest
    applied: tuple[Applied, ...]
    ancillary_evidence: tuple[Key, ...]


@dataclass(frozen=True)
class ProposeRequest:
    system: System
    parents: tuple[Candidate, ...]
    working: Candidate
    evidence: tuple[EvidenceUse, ...]
    selection: tuple[Selection, ...]
    packed: PackedStates | None
    sampling: SamplingPolicy
    sample: Sample


@dataclass(frozen=True)
class Proposals:
    items: tuple[Proposal, ...]


@dataclass(frozen=True)
class SampleIndependentRequest:
    distribution: IndependentDistribution
    context: ProposeRequest
    count: int


@dataclass(frozen=True)
class FuseRequest:
    distributions: tuple[IndependentDistribution, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class MeasureRequest:
    system: System
    candidate: Candidate
    evidence: tuple[EvidenceUse, ...]
    selections: tuple[Selection, ...]


@dataclass(frozen=True)
class AssessRequest:
    candidate: Candidate
    evidence: Evidence
    history: History
    objective_revision: Key


@dataclass(frozen=True)
class Assessment:
    evidence: Key
    objective_revision: Key
    history_revision: Key
    values: tuple[float, ...]
    gates: tuple[tuple[str, bool], ...]
    inputs: InputManifest


@dataclass(frozen=True)
class Scored:
    candidate: Candidate
    evidence: Evidence
    assessment: Assessment


@dataclass(frozen=True)
class Work:
    identity: str
    parents: tuple[Scored, ...]
    stage: str
    sample: Sample
    candidate: Candidate | None = None


@dataclass(frozen=True)
class Continuation:
    """Successor chosen by a recorded scientific acceptance operation.

    Ranking policies need not use this. A continuation policy follows the
    decision as recorded, including accepted moves to a lower objective.
    """

    successor: Scored
    stop_reason: str | None = None


@dataclass(frozen=True)
class TrialOutcome:
    work: str
    status: Literal["completed", "failed", "unknown", "refused"]
    proposals: tuple[Proposal, ...]
    children: tuple[Scored, ...]
    attempts: tuple[str, ...]
    continuation: Continuation | None = None


@dataclass(frozen=True)
class History:
    revision: Key
    records: tuple[TrialOutcome, ...]


@dataclass(frozen=True)
class CallKey:
    work_id: str
    named_round: str
    component: Key
    input_manifest: Key


@dataclass(frozen=True)
class RNGReceipt:
    before: Blob
    after: Blob
    protocol: Key


@dataclass(frozen=True)
class Attempt:
    identity: str
    key: CallKey
    status: Literal["reserved", "issued", "completed", "failed", "unknown"]
    result: Blob | None
    rng: RNGReceipt | None


@dataclass(frozen=True)
class Reconciliation:
    attempt: str
    status: Literal["completed", "failed", "unknown"]
    evidence: Blob


@dataclass(frozen=True)
class Preflight:
    identity: Key
    requirements: tuple[Requirement, ...]
    scientific_contract: Key


class Runtime(Protocol):
    history: History
    data: DataStore

    def call(
        self,
        named_round: str,
        operation: Operation[I, O],
        request: I,
    ) -> O: ...

    def preflight(self, requirements: tuple[Requirement, ...]) -> Preflight: ...

    def reconcile(self, attempt: str) -> Reconciliation: ...


class Trial(Protocol):
    requirements: tuple[Requirement, ...]

    def root(self, candidate: Candidate, runtime: Any) -> Scored: ...

    def perform(self, work: Work, runtime: Any) -> TrialOutcome: ...


class SearchPolicy(Protocol[S]):
    codec: Codec[S]

    def start(self, initial: tuple[Scored, ...]) -> S: ...

    def ask(
        self,
        state: S,
        history: History,
        capacity: int,
    ) -> tuple[Work, ...]: ...

    def tell(
        self,
        state: S,
        outcomes: tuple[TrialOutcome, ...],
    ) -> S: ...

    def finished(self, state: S) -> bool: ...

    def selected(self, state: S) -> tuple[Scored, ...]: ...


@dataclass(frozen=True)
class RunManifest:
    scientific_contract: Key
    preflight: Key
    policy_codec: str
    result_codec: str
    journal: Path
    checkpoint: Blob
    selected_summary: Path


@dataclass(frozen=True)
class RunResult:
    selected: tuple[Scored, ...]
    manifest: RunManifest


class Components(Protocol):
    @classmethod
    def from_config(cls, config: Path) -> Components: ...

    def operation(
        self,
        alias: str,
        request: type[I],
        result: type[O],
    ) -> Operation[I, O]: ...

    def sampling(self, alias: str) -> SamplingPolicy: ...

    def requirements(self, aliases: tuple[str, ...]) -> tuple[Requirement, ...]: ...


@dataclass(frozen=True)
class _Prepared(Generic[I]):
    request: I
    inputs: InputManifest
    requirements: tuple[Requirement, ...]
    paid_boundary: str
    references: tuple[Blob, ...] = ()


class _Adapter(Protocol[I, O]):
    def prepare(self, request: I) -> _Prepared[I]: ...

    def invoke(self, prepared: _Prepared[I]) -> O: ...


def key_for(value: Any) -> Key:
    """Return a deterministic SHA-256 content key for a constructible value."""

    payload = _canonical_value(value, path="$")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return Key(hashlib.sha256(encoded).hexdigest())


def manifest(dependencies: tuple[Dependency, ...]) -> InputManifest:
    """Build an input manifest with actual dependencies in canonical order."""

    normalized = tuple(
        sorted(
            set(dependencies),
            key=lambda item: (item.role, item.identity.digest),
        )
    )
    return InputManifest(
        identity=key_for(("input-manifest/v1", normalized)),
        dependencies=normalized,
    )


def make_system(
    variables: tuple[str, ...],
    occurrences: tuple[Occurrence, ...],
    topology: Blob,
    domains: tuple[Domain, ...] = (),
) -> System:
    """Construct and validate a system with a content-derived identity."""

    variables = tuple(variables)
    occurrences = tuple(occurrences)
    domains = tuple(domains)
    if not variables:
        raise ValueError("system must declare at least one variable")
    if len(set(variables)) != len(variables):
        raise ValueError("system variables must be unique")
    for variable in variables:
        if not isinstance(variable, str) or not variable:
            raise ValueError("system variable names must be non-empty strings")
    if not isinstance(topology, Blob):
        raise TypeError("topology must be a Blob")

    occurrence_ids: set[str] = set()
    for occurrence in occurrences:
        if not isinstance(occurrence, Occurrence):
            raise TypeError("occurrences must contain Occurrence values")
        if occurrence.identity.digest in occurrence_ids:
            raise ValueError(
                f"duplicate occurrence identity {occurrence.identity.digest!r}"
            )
        occurrence_ids.add(occurrence.identity.digest)
        if occurrence.variable not in variables:
            raise ValueError(
                f"occurrence {occurrence.identity.digest!r} references unknown "
                f"variable {occurrence.variable!r}"
            )
        for name in (
            occurrence.state,
            occurrence.assembly,
            occurrence.chain,
            occurrence.component,
        ):
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "occurrence state/assembly/chain/component must be non-empty strings"
                )
        if not isinstance(occurrence.index_map, Blob):
            raise TypeError("occurrence index_map must be a Blob")

    domain_variables: set[str] = set()
    for domain in domains:
        if not isinstance(domain, Domain):
            raise TypeError("domains must contain Domain values")
        if domain.variable not in variables:
            raise ValueError(
                f"domain references unknown variable {domain.variable!r}"
            )
        if domain.variable in domain_variables:
            raise ValueError(
                f"multiple domains supplied for variable {domain.variable!r}"
            )
        domain_variables.add(domain.variable)
        if len(set(domain.fixed_tokens)) != len(domain.fixed_tokens):
            raise ValueError(
                f"fixed token indices must be unique for {domain.variable!r}"
            )
        if any(type(index) is not int or index < 0 for index in domain.fixed_tokens):
            raise ValueError(
                f"fixed token indices must be non-negative integers for "
                f"{domain.variable!r}"
            )
        if len(set(domain.fixed_atoms)) != len(domain.fixed_atoms):
            raise ValueError(
                f"fixed atom IDs must be unique for {domain.variable!r}"
            )
        if any(not isinstance(atom, str) or not atom for atom in domain.fixed_atoms):
            raise ValueError(
                f"fixed atom IDs must be non-empty strings for {domain.variable!r}"
            )
        if domain.allowed_values is not None and not isinstance(
            domain.allowed_values, Blob
        ):
            raise TypeError("domain allowed_values must be a Blob or None")

    return System(
        identity=_system_identity(variables, occurrences, topology, domains),
        variables=variables,
        occurrences=occurrences,
        topology=topology,
        domains=domains,
    )


def make_candidate(
    system: System,
    assignments: tuple[Assignment, ...],
    *,
    data: DataStore | None = None,
) -> Candidate:
    """Construct a complete assignment and validate declared allowed values."""

    if not isinstance(system, System):
        raise TypeError("system must be a System")
    if system.identity != _system_identity(
        system.variables,
        system.occurrences,
        system.topology,
        system.domains,
    ):
        raise ValueError("system identity does not match its content")
    normalized = _validated_assignments(system, tuple(assignments))
    values = {assignment.variable: assignment.value for assignment in normalized}
    for domain in system.domains:
        if domain.allowed_values is None:
            continue
        value = values[domain.variable]
        if not isinstance(value, Polymer):
            raise ValueError(
                f"allowed_tokens restrictions do not apply to chemical variable {domain.variable!r}"
            )
        for index, tokens in _read_allowed_tokens(domain.allowed_values, data).items():
            if index >= len(value.tokens):
                raise ValueError(f"allowed-value index {index} is outside {domain.variable!r}")
            if value.tokens[index] not in tokens:
                raise ValueError(
                    f"token {value.tokens[index]!r} at {domain.variable}[{index}] is not allowed"
                )
    return Candidate(
        identity=_candidate_identity(system.identity, normalized),
        system=system.identity,
        assignments=normalized,
    )


def _system_identity(
    variables: tuple[str, ...],
    occurrences: tuple[Occurrence, ...],
    topology: Blob,
    domains: tuple[Domain, ...],
) -> Key:
    return key_for(
        (
            "system/v1",
            variables,
            occurrences,
            topology,
            domains,
        )
    )


def _candidate_identity(
    system: Key,
    assignments: tuple[Assignment, ...],
) -> Key:
    return key_for(("candidate/v1", system, assignments))


def _evidence_identity(
    evaluated_candidate: Key,
    artifacts: tuple[Artifact, ...],
    measurements: tuple[Measurement, ...],
    inputs: InputManifest,
) -> Key:
    return key_for(
        (
            "evidence/v1",
            evaluated_candidate,
            artifacts,
            measurements,
            inputs,
        )
    )


def _validated_assignments(
    system: System,
    assignments: tuple[Assignment, ...],
) -> tuple[Assignment, ...]:
    by_variable: dict[str, Assignment] = {}
    for assignment in assignments:
        if not isinstance(assignment, Assignment):
            raise TypeError("assignments must contain Assignment values")
        if assignment.variable not in system.variables:
            raise ValueError(
                f"assignment references unknown variable {assignment.variable!r}"
            )
        if assignment.variable in by_variable:
            raise ValueError(
                f"assignment supplied more than once for {assignment.variable!r}"
            )
        normalized = _normalize_assignment(assignment)
        _validate_assignment_value(normalized.variable, normalized.value)
        by_variable[normalized.variable] = normalized

    missing = [variable for variable in system.variables if variable not in by_variable]
    if missing:
        raise ValueError(f"candidate is missing assignments for {missing!r}")

    domain_map = {domain.variable: domain for domain in system.domains}
    for variable, assignment in by_variable.items():
        domain = domain_map.get(variable)
        if domain is None:
            continue
        if isinstance(assignment.value, Polymer):
            for index in domain.fixed_tokens:
                if index >= len(assignment.value.tokens):
                    raise ValueError(
                        f"fixed token index {index} is outside variable {variable!r}"
                    )
            if domain.fixed_atoms:
                raise ValueError(
                    f"polymer variable {variable!r} cannot declare fixed atoms"
                )
        elif domain.fixed_tokens:
            raise ValueError(
                f"chemical variable {variable!r} cannot declare fixed tokens"
            )

    return tuple(by_variable[variable] for variable in system.variables)


def _normalize_assignment(assignment: Assignment) -> Assignment:
    value = assignment.value
    if isinstance(value, Polymer):
        modifications = tuple(sorted(value.modifications))
        if modifications != value.modifications:
            value = Polymer(value.kind, value.tokens, modifications)
    return Assignment(assignment.variable, value)


def _validate_assignment_value(variable: str, value: Polymer | Chemical) -> None:
    if isinstance(value, Polymer):
        _validate_polymer(variable, value)
        return
    if isinstance(value, Chemical):
        _validate_chemical(variable, value)
        return
    raise TypeError(
        f"variable {variable!r} has unsupported assignment type "
        f"{type(value).__name__}"
    )


def _validate_polymer(variable: str, value: Polymer) -> None:
    if value.kind not in {"protein", "rna", "dna"}:
        raise ValueError(f"unknown polymer kind {value.kind!r} for {variable!r}")
    if not isinstance(value.tokens, tuple) or any(
        not isinstance(token, str) or not token for token in value.tokens
    ):
        raise ValueError(
            f"polymer tokens for {variable!r} must be non-empty strings"
        )
    if not isinstance(value.modifications, tuple):
        raise ValueError(f"modifications for {variable!r} must be a tuple")
    seen: set[int] = set()
    for modification in value.modifications:
        if (
            not isinstance(modification, tuple)
            or len(modification) != 2
            or type(modification[0]) is not int
        ):
            raise ValueError(f"invalid modification for {variable!r}")
        index, name = modification
        if index < 0 or index >= len(value.tokens):
            raise ValueError(
                f"modification index {index} is outside variable {variable!r}"
            )
        if not isinstance(name, str) or not name:
            raise ValueError(f"modification name for {variable!r} must be non-empty")
        if index in seen:
            raise ValueError(
                f"multiple modifications at {variable}[{index}] are not allowed"
            )
        seen.add(index)


def _validate_chemical(variable: str, value: Chemical) -> None:
    if not isinstance(value.graph, Blob):
        raise TypeError(f"chemical graph for {variable!r} must be a Blob")


def _read_allowed_tokens(reference: Blob, data: Any) -> dict[int, frozenset[str]]:
    if data is None:
        raise ValueError(
            "DataStore is required to resolve domain allowed_values"
        )
    getter = getattr(data, "get", None)
    if getter is None:
        raise TypeError("data must provide get(reference, codec)")

    from .store import JSONCodec

    payload = getter(reference, JSONCodec())
    if not isinstance(payload, dict) or set(payload) != {"allowed_tokens"}:
        raise ValueError(
            "restriction JSON must contain exactly the allowed_tokens field"
        )
    raw_tokens = payload["allowed_tokens"]
    if not isinstance(raw_tokens, dict):
        raise ValueError("allowed_tokens must be a JSON object")

    result: dict[int, frozenset[str]] = {}
    for raw_index, tokens in raw_tokens.items():
        if not isinstance(raw_index, str) or not raw_index.isdecimal():
            raise ValueError("allowed_tokens keys must be decimal indices")
        index = int(raw_index)
        if str(index) != raw_index:
            raise ValueError("allowed_tokens indices must use canonical decimal form")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("each allowed_tokens entry must be a non-empty list")
        if any(not isinstance(token, str) or not token for token in tokens):
            raise ValueError("allowed token values must be non-empty strings")
        if len(set(tokens)) != len(tokens):
            raise ValueError("allowed token values must be unique")
        result[index] = frozenset(tokens)
    return result


def _canonical_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float at {path}")
        return value
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Path):
        return {"$path": value.as_posix()}
    if is_dataclass(value) and not isinstance(value, type):
        type_name = f"{type(value).__module__}.{type(value).__qualname__}"
        return {
            "$dataclass": type_name,
            "fields": {
                field.name: _canonical_value(
                    getattr(value, field.name),
                    path=f"{path}.{field.name}",
                )
                for field in fields(value)
            },
        }
    if isinstance(value, tuple):
        return {
            "$tuple": [
                _canonical_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        }
    if isinstance(value, list):
        return {
            "$list": [
                _canonical_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        }
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"dictionary keys at {path} must be strings")
        return {
            "$dict": {
                key: _canonical_value(item, path=f"{path}[{key!r}]")
                for key, item in value.items()
            }
        }
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item, path=f"{path}{{set}}") for item in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return {"$frozenset" if isinstance(value, frozenset) else "$set": items}
    raise TypeError(
        f"unsupported value at {path}: {type(value).__module__}."
        f"{type(value).__qualname__}; arrays must be referenced through Blob"
    )


__all__ = [
    "Applied",
    "Artifact",
    "ArtifactBank",
    "AssessRequest",
    "Assessment",
    "Assignment",
    "Attempt",
    "Axis",
    "Blob",
    "CallKey",
    "Candidate",
    "Capability",
    "Chemical",
    "ChooseArtifactRequest",
    "Codec",
    "ComponentDescriptor",
    "Components",
    "Conditioning",
    "Continuation",
    "Dependency",
    "DataStore",
    "Domain",
    "Evidence",
    "EvidenceUse",
    "FuseRequest",
    "History",
    "IndependentDistribution",
    "InitializeRequest",
    "Initialized",
    "InputManifest",
    "Key",
    "MeasureRequest",
    "Measurement",
    "MeasurementView",
    "Occurrence",
    "Operation",
    "PackRequest",
    "PackedStates",
    "Polymer",
    "PredictRequest",
    "Preflight",
    "Proposal",
    "Proposals",
    "ProposeRequest",
    "Reconciliation",
    "Reduction",
    "Requirement",
    "RNGReceipt",
    "RunManifest",
    "RunResult",
    "Runtime",
    "Sample",
    "SampleIndependentRequest",
    "SamplingPolicy",
    "Scored",
    "SearchPolicy",
    "SelectRequest",
    "SelectedOptionsSchema",
    "Selection",
    "Subject",
    "System",
    "TransformRequest",
    "Trial",
    "TrialOutcome",
    "Work",
    "key_for",
    "make_candidate",
    "make_system",
    "manifest",
]
