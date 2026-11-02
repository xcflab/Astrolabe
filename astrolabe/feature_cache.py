"""Content identities for CPU-side input artifacts, never model state."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

# These fields are deliberately mandatory, including explicit nulls: a caller
# cannot accidentally omit a dependency when constructing a cache key.
FEATURE_DEPENDENCIES = frozenset({
    "schema", "source", "chains", "chain_order", "topology", "constraints",
    "seed", "msa", "templates", "ccd", "options",
})


def feature_key(dependencies: Mapping) -> str:
    if set(dependencies) != FEATURE_DEPENDENCIES:
        raise ValueError(f"feature dependencies must be exactly {sorted(FEATURE_DEPENDENCIES)}")
    encoded = json.dumps(dependencies, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
