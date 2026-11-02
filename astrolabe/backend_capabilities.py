"""Conservative observed backend lifecycle inventory at Astrolabe b537b8a.

This inventory does not certify a backend or enable a split by itself.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendCapability:
    identity: str
    legacy: str
    split: str
    lifecycle: str


BACKENDS = {entry.identity: entry for entry in (
    BackendCapability("af2", "adapter", "unsupported", "in-process"),
    BackendCapability("af3", "adapter", "daemon-forward", "one resident runner per client"),
    BackendCapability("protenix-v0.5", "adapter", "unsupported", "in-process mutable runner"),
    BackendCapability("protenix-v0.5-constraint", "adapter", "unsupported", "in-process mutable runner"),
    BackendCapability("protenix-v1", "adapter", "unsupported", "in-process mutable runner"),
    BackendCapability("protenix-v2", "adapter", "unsupported", "in-process mutable runner"),
    BackendCapability("esmfold1", "adapter", "unsupported", "in-process lazy model"),
    BackendCapability("esmfold2", "adapter", "resident-atomic", "one resident model and builder per client"),
    BackendCapability("opendde", "adapter", "unsupported", "one subprocess per complex"),
    BackendCapability("mol-1", "no adapter", "unsupported", "unavailable"),
    BackendCapability("chai-1", "adapter", "unsupported", "component cache; components move between cpu and device"),
    BackendCapability("boltz1", "adapter", "unsupported", "in-process"),
    BackendCapability("boltz2", "no adapter", "unsupported", "unavailable"),
)}


def resolve_protenix_identity(requested: str, guidance: bool, policy: str = "compat") -> tuple[str, str]:
    """Report the legacy constraint-checkpoint override, or reject it in strict mode."""
    if policy not in ("compat", "strict"):
        raise ValueError("Protenix identity_policy must be compat or strict")
    effective = "protenix_base_constraint_v0.5.0" if guidance else requested
    if policy == "strict" and requested != effective:
        raise ValueError(f"Protenix strict identity forbids switching {requested!r} to {effective!r}")
    return requested, effective
