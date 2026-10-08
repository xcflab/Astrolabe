from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from astrolabe.posthoc import refold_backends
from astrolabe.posthoc.io_utils import sanitize_token

GroupKey = tuple[str, str, str]
BackendResourceKey = tuple[str, str]
ResourceKey = tuple[BackendResourceKey, ...]
RuntimeSignatureLoader = Callable[[Path, str], Mapping[str, Any] | None]


@dataclass(frozen=True)
class RefoldManifestItem:
    payload: dict[str, Any]

    @classmethod
    def from_mapping(
        cls,
        item: Mapping[str, Any] | "RefoldManifestItem",
    ) -> "RefoldManifestItem":
        if isinstance(item, RefoldManifestItem):
            return item
        return cls(dict(item))

    @property
    def task(self) -> str:
        return str(self.payload["task"])

    @property
    def run(self) -> str:
        return str(self.payload["run"])

    @property
    def normalized_source_run_dir(self) -> str:
        raw = str(self.payload.get("src_run_dir") or "").strip()
        if not raw:
            return ""
        return str(Path(raw).expanduser().resolve())

    @property
    def group_key(self) -> GroupKey:
        return (self.task, self.run, self.normalized_source_run_dir)

    @property
    def explicit_output_run_name(self) -> str:
        return str(self.payload.get("_refold_dst_run_name") or "").strip()

    def source_run_dir_path(self) -> Path | None:
        raw = str(self.payload.get("src_run_dir") or "").strip()
        if not raw:
            return None
        return Path(raw).expanduser().resolve()

    def target_config_path(self, config_yaml: Path | None) -> Path | None:
        if config_yaml is not None:
            return Path(config_yaml).expanduser().resolve()
        raw = str(self.payload.get("src_config") or "").strip()
        if not raw:
            return None
        return Path(raw).expanduser().resolve()

    def needed_backends(self, requested: set[str]) -> set[str]:
        return refold_backends.item_needed_backends(
            self.payload.get("needed_evaluators"),
            requested,
        )

    def as_manifest_record(self, *, output_run_name: str | None = None) -> dict[str, Any]:
        record = dict(self.payload)
        if output_run_name:
            record["_refold_dst_run_name"] = output_run_name
        return record


@dataclass(frozen=True)
class RefoldGroupPlan:
    key: GroupKey
    items: tuple[RefoldManifestItem, ...]
    output_run_name: str
    resource_key: ResourceKey

    def manifest_items(self) -> list[dict[str, Any]]:
        return [
            item.as_manifest_record(output_run_name=self.output_run_name)
            for item in self.items
        ]


@dataclass(frozen=True)
class RefoldPlan:
    groups: tuple[RefoldGroupPlan, ...]
    resource_group_count: int

    def group_entries(self) -> list[tuple[GroupKey, list[dict[str, Any]]]]:
        return [(group.key, group.manifest_items()) for group in self.groups]


def _manifest_item(
    item: Mapping[str, Any] | RefoldManifestItem,
) -> RefoldManifestItem:
    return RefoldManifestItem.from_mapping(item)


def hash_manifest_inputs(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]


def canonical_refold_backend_name(raw: str | None) -> str | None:
    return refold_backends.canonical_refold_backend_name(raw)


def requested_refold_backends(evaluators: Sequence[str]) -> set[str]:
    return refold_backends.requested_refold_backends(evaluators)


def item_needed_backends(
    item: Mapping[str, Any] | RefoldManifestItem,
    requested: set[str],
) -> set[str]:
    return _manifest_item(item).needed_backends(requested)


def normalized_refold_source_run_dir(item: Mapping[str, Any] | RefoldManifestItem) -> str:
    return _manifest_item(item).normalized_source_run_dir


def refold_group_key(item: Mapping[str, Any] | RefoldManifestItem) -> GroupKey:
    return _manifest_item(item).group_key


def refold_source_run_dir_path(item: Mapping[str, Any] | RefoldManifestItem) -> Path | None:
    return _manifest_item(item).source_run_dir_path()


def refold_target_config_path(
    *,
    item: Mapping[str, Any] | RefoldManifestItem,
    config_yaml: Path | None,
) -> Path | None:
    return _manifest_item(item).target_config_path(config_yaml)


def refold_output_run_name(*, run: str, src_run_dir: str, collides: bool) -> str:
    base = sanitize_token(run)
    if not collides:
        return base

    suffix_parts: list[str] = []
    if src_run_dir:
        src_path = Path(src_run_dir)
        seed_hint = next((part.name for part in src_path.parents if part.name.startswith("seed_")), "")
        if seed_hint:
            suffix_parts.append(sanitize_token(seed_hint))
        suffix_parts.append(f"src_{hash_manifest_inputs(src_run_dir)[:8]}")
    else:
        suffix_parts.append("src_unknown")
    return "__".join([base, *suffix_parts])


def planned_refold_output_run_name(
    *,
    group_key: GroupKey,
    items: Sequence[Mapping[str, Any] | RefoldManifestItem],
    run_name_counts: Mapping[tuple[str, str], int],
) -> str:
    explicit_names = {
        item.explicit_output_run_name
        for item in (RefoldManifestItem.from_mapping(raw) for raw in items)
        if item.explicit_output_run_name
    }
    if len(explicit_names) > 1:
        raise ValueError(f"Conflicting _refold_dst_run_name values for refold group {group_key}")
    if explicit_names:
        return next(iter(explicit_names))

    task, run, src_run_dir = group_key
    return refold_output_run_name(
        run=run,
        src_run_dir=src_run_dir,
        collides=run_name_counts.get((task, run), 0) > 1,
    )


def refold_backend_resource_key(
    *,
    item: Mapping[str, Any] | RefoldManifestItem,
    backend: str,
    config_yaml: Path | None,
    load_runtime_signature: RuntimeSignatureLoader,
) -> BackendResourceKey:
    backend_name = canonical_refold_backend_name(backend)
    if backend_name is None:
        return (str(backend), "unsupported")

    target_config_path = _manifest_item(item).target_config_path(config_yaml)
    if target_config_path is None:
        return (backend_name, "missing-config")

    signature = load_runtime_signature(target_config_path, backend_name)
    if signature is None:
        return (backend_name, f"unresolved:{target_config_path}")
    return (backend_name, hash_manifest_inputs(signature))


def refold_item_resource_keys(
    *,
    item: Mapping[str, Any] | RefoldManifestItem,
    requested_backends: set[str],
    config_yaml: Path | None,
    load_runtime_signature: RuntimeSignatureLoader,
) -> ResourceKey:
    needed_backends = item_needed_backends(item, requested_backends)
    keys = [
        refold_backend_resource_key(
            item=item,
            backend=backend,
            config_yaml=config_yaml,
            load_runtime_signature=load_runtime_signature,
        )
        for backend in sorted(needed_backends)
        if backend in requested_backends
    ]
    return tuple(keys) or (("none", "none"),)


def refold_group_resource_key(
    *,
    items: Sequence[Mapping[str, Any] | RefoldManifestItem],
    requested_backends: set[str],
    config_yaml: Path | None,
    load_runtime_signature: RuntimeSignatureLoader,
) -> ResourceKey:
    keys: set[BackendResourceKey] = set()
    for item in items:
        keys.update(
            refold_item_resource_keys(
                item=item,
                requested_backends=requested_backends,
                config_yaml=config_yaml,
                load_runtime_signature=load_runtime_signature,
            )
        )
    return tuple(sorted(keys)) or (("none", "none"),)


def build_refold_plan(
    *,
    items: Sequence[Mapping[str, Any] | RefoldManifestItem],
    requested_backends: set[str],
    config_yaml: Path | None,
    load_runtime_signature: RuntimeSignatureLoader,
) -> RefoldPlan:
    grouped: dict[GroupKey, list[RefoldManifestItem]] = {}
    for raw_item in items:
        item = RefoldManifestItem.from_mapping(raw_item)
        grouped.setdefault(item.group_key, []).append(item)

    run_name_counts: dict[tuple[str, str], int] = {}
    for task, run, _src_run_dir in grouped:
        run_name_counts[(task, run)] = run_name_counts.get((task, run), 0) + 1

    groups: list[RefoldGroupPlan] = []
    resource_keys: set[ResourceKey] = set()
    for group_key, group_items in sorted(grouped.items()):
        output_run_name = planned_refold_output_run_name(
            group_key=group_key,
            items=group_items,
            run_name_counts=run_name_counts,
        )
        resource_key = refold_group_resource_key(
            items=group_items,
            requested_backends=requested_backends,
            config_yaml=config_yaml,
            load_runtime_signature=load_runtime_signature,
        )
        resource_keys.add(resource_key)
        groups.append(
            RefoldGroupPlan(
                key=group_key,
                items=tuple(group_items),
                output_run_name=output_run_name,
                resource_key=resource_key,
            )
        )

    return RefoldPlan(
        groups=tuple(groups),
        resource_group_count=len(resource_keys),
    )


def assign_refold_plan_to_worker_shards(
    *,
    plan: RefoldPlan,
    worker_count: int,
) -> tuple[list[list[dict[str, Any]]], int]:
    shards: list[list[dict[str, Any]]] = [[] for _ in range(worker_count)]
    if worker_count <= 0:
        return shards, 0

    resource_buckets: dict[ResourceKey, list[RefoldGroupPlan]] = {}
    for group in plan.groups:
        resource_buckets.setdefault(group.resource_key, []).append(group)

    shard_loads = [0 for _ in range(worker_count)]
    bucket_entries = sorted(
        resource_buckets.items(),
        key=lambda kv: (
            -sum(len(group.items) for group in kv[1]),
            repr(kv[0]),
        ),
    )
    for _resource_key, bucket_groups in bucket_entries:
        shard_idx = min(range(worker_count), key=lambda idx: (shard_loads[idx], idx))
        bucket_load = 0
        for group in bucket_groups:
            bucket_load += len(group.items)
            for enriched in group.manifest_items():
                shards[shard_idx].append(enriched)
        shard_loads[shard_idx] += bucket_load

    return shards, plan.resource_group_count


def assign_refold_groups_to_worker_shards(
    *,
    groups: list[tuple[GroupKey, list[dict[str, Any]]]],
    planned_run_names: Mapping[GroupKey, str],
    requested_backends: set[str],
    config_yaml: Path | None,
    worker_count: int,
    load_runtime_signature: RuntimeSignatureLoader,
) -> tuple[list[list[dict[str, Any]]], int]:
    plan_groups: list[RefoldGroupPlan] = []
    resource_keys: set[ResourceKey] = set()
    for group_key, items in groups:
        manifest_items = tuple(RefoldManifestItem.from_mapping(item) for item in items)
        resource_key = refold_group_resource_key(
            items=manifest_items,
            requested_backends=requested_backends,
            config_yaml=config_yaml,
            load_runtime_signature=load_runtime_signature,
        )
        resource_keys.add(resource_key)
        plan_groups.append(
            RefoldGroupPlan(
                key=group_key,
                items=manifest_items,
                output_run_name=planned_run_names[group_key],
                resource_key=resource_key,
            )
        )
    return assign_refold_plan_to_worker_shards(
        plan=RefoldPlan(
            groups=tuple(plan_groups),
            resource_group_count=len(resource_keys),
        ),
        worker_count=worker_count,
    )
