#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import datetime as dt
import hashlib
from pathlib import Path


_LICENSE_CANDIDATES = (
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "COPYING",
    "COPYING.txt",
    "NOTICE",
    "NOTICE.txt",
)
_LICENSE_GLOBS = ("COPYING.*",)
_IGNORED_COMPONENT_DIRS = frozenset({"__pycache__", ".pytest_cache"})
_LICENSE_EVIDENCE_OVERRIDES = {
    "boost-regex-source": "third_party_licenses/boost-regex-source/LICENSE_1_0.txt",
}
_EXCLUDED_FROM_RUNTIME_RELEASE = frozenset({"surfdesign", "surfpro", "willutil"})
_LIGANDMPNN_PROVENANCE_HEADER = (
    "# Derived from dauparas/LigandMPNN at commit\n"
    "# 26ec57ac976ade5379920dbd43c7f97a91cf82de:\n"
    "# https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de\n"
    "# Upstream source files: data_utils.py, model_utils.py, and sc_utils.py\n"
    "# (including parse_PDB, featurize, ProteinMPNN, ProteinFeatures*, and related\n"
    "# helpers).\n"
    "# Copyright (c) 2024 Justas Dauparas; licensed under the MIT License.\n"
    "# Full license text: third_party_licenses/ligandmpnn/LICENSE\n"
    "# Local modifications: this wrapper contains GH-specific adaptations and\n"
    "# additions around the upstream-derived helpers.\n\n"
)
_LIGANDMPNN_LICENSE_BYTES = 1072
_LIGANDMPNN_LICENSE_SHA256 = "8af7ba31868e08e8949e20ce2ef3c0b1d0ef78c9358c91563a60a9fe3b36c9ea"
_IN_TREE_DERIVED_COMPONENTS = (
    {
        "component": "ligandmpnn-derived",
        "path": "astrolabe/mutators/wrappers/ligandmpnn_utils.py",
        "url": (
            "https://github.com/dauparas/LigandMPNN/tree/"
            "26ec57ac976ade5379920dbd43c7f97a91cf82de"
        ),
        "license": "third_party_licenses/ligandmpnn/LICENSE",
    },
)


def _load_gitmodules(gitmodules_path: Path) -> dict[str, str]:
    if not gitmodules_path.exists():
        return {}
    parser = configparser.ConfigParser()
    parser.read(gitmodules_path, encoding="utf-8")
    mapping: dict[str, str] = {}
    for section in parser.sections():
        if not section.startswith("submodule "):
            continue
        path = parser.get(section, "path", fallback="").strip()
        url = parser.get(section, "url", fallback="").strip()
        if path:
            mapping[path] = url
    return mapping


def _find_license_files(component_dir: Path) -> list[str]:
    found: list[str] = []
    for name in _LICENSE_CANDIDATES:
        p = component_dir / name
        if p.is_file():
            found.append(str(p.relative_to(component_dir)))
    for pattern in _LICENSE_GLOBS:
        found.extend(
            str(path.relative_to(component_dir))
            for path in sorted(component_dir.glob(pattern))
            if path.is_file()
        )
    return sorted(set(found))


def _component_rows(repo_root: Path) -> list[dict[str, str]]:
    vendor_root = repo_root / "vendor"
    gitmodules_map = _load_gitmodules(repo_root / ".gitmodules")
    rows: list[dict[str, str]] = []
    if vendor_root.exists():
        components = [
            path
            for path in vendor_root.iterdir()
            if path.is_dir()
            and path.name not in _IGNORED_COMPONENT_DIRS
            and not path.name.startswith(".")
        ]
        for component in sorted(components):
            rel = str(component.relative_to(repo_root))
            license_files = _find_license_files(component)
            override = _LICENSE_EVIDENCE_OVERRIDES.get(component.name)
            if override is not None:
                override_path = repo_root / override
                if override_path.is_file():
                    license_files.append(override)
            status = "ok"
            if not any(component.iterdir()):
                status = "not_checked_out"
            elif component.name in _EXCLUDED_FROM_RUNTIME_RELEASE:
                status = "excluded_from_runtime_release"
            elif not license_files:
                status = "license_file_not_found_top_level"

            rows.append(
                {
                    "component": component.name,
                    "path": rel,
                    "url": gitmodules_map.get(rel, ""),
                    "licenses": ", ".join(license_files) if license_files else "",
                    "status": status,
                }
            )

    for component in _IN_TREE_DERIVED_COMPONENTS:
        source_path = repo_root / component["path"]
        license_path = repo_root / component["license"]
        source_ok = source_path.is_file() and source_path.read_bytes().startswith(
            _LIGANDMPNN_PROVENANCE_HEADER.encode("utf-8")
        )
        license_bytes = license_path.read_bytes() if license_path.is_file() else b""
        license_ok = (
            len(license_bytes) == _LIGANDMPNN_LICENSE_BYTES
            and hashlib.sha256(license_bytes).hexdigest() == _LIGANDMPNN_LICENSE_SHA256
        )
        evidence = component["license"] if license_path.is_file() else ""
        rows.append(
            {
                "component": component["component"],
                "path": component["path"],
                "url": component["url"],
                "licenses": evidence,
                # Keep the existing review-needed status so gh_release_audit
                # rejects missing source or license evidence automatically.
                "status": (
                    "ok"
                    if source_ok and license_ok
                    else "license_file_not_found_top_level"
                ),
            }
        )
    return rows


def _status_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "")).strip()
        if not status:
            continue
        counts[status] = counts.get(status, 0) + 1
    return counts


def _render_markdown(rows: list[dict[str, str]]) -> str:
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# THIRD_PARTY_LICENSES",
        "",
        "Auto-generated third-party inventory for release compliance.",
        "",
        f"- generated_at_utc: `{generated_at}`",
        "- policy: release artifacts must include this file and review status for each `vendor/*` component and tracked in-tree derived component.",
        "",
        "| component | path | source | license evidence | status |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        source = row["url"] or "-"
        licenses = row["licenses"] or "-"
        lines.append(
            f"| `{row['component']}` | `{row['path']}` | `{source}` | `{licenses}` | `{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `not_checked_out`: submodule directory exists but content is not present in current workspace.",
            "- `license_file_not_found_top_level`: required license or source evidence is missing; review required.",
            "- `excluded_from_runtime_release`: source is not copied into the runtime SIF; the parent repository distributes only its submodule pointer.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate THIRD_PARTY_LICENSES.md from the third-party inventory."
    )
    parser.add_argument(
        "--out",
        default="THIRD_PARTY_LICENSES.md",
        help="Output markdown path (default: THIRD_PARTY_LICENSES.md).",
    )
    parser.add_argument(
        "--fail-on-review-needed",
        action="store_true",
        help="Exit non-zero when any inventory component needs manual license review.",
    )
    ns = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    out_path = (repo_root / ns.out).resolve()
    rows = _component_rows(repo_root)
    payload = _render_markdown(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(out_path)
    print(f"wrote: {out_path}")
    counts = _status_counts(rows)
    if counts:
        print("status_counts:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    review_needed = int(counts.get("license_file_not_found_top_level", 0))
    if ns.fail_on_review_needed and review_needed > 0:
        print(
            "FAIL: inventory components with missing or invalid license/source evidence."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
