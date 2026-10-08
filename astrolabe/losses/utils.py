import inspect
import logging
import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Dict, List, Optional

import numpy as np


def pae_residue_aggregation(token_res_ids, token_chain_ids, pae_matrix):
    """Aggregate PAE data by residue level."""
    token_res_ids, token_chain_ids = np.array(token_res_ids), np.array(token_chain_ids)
    pae_matrix = np.array(pae_matrix)

    unique_residues, res_idx = np.unique(
        np.stack([token_chain_ids, token_res_ids], axis=1), axis=0, return_inverse=True
    )
    n_residues = len(unique_residues)

    pae_residue = np.zeros((n_residues, n_residues), dtype=np.float64)
    count_matrix = np.zeros((n_residues, n_residues), dtype=np.int32)

    for i in range(len(token_res_ids)):
        for j in range(len(token_res_ids)):
            pae_residue[res_idx[i], res_idx[j]] += pae_matrix[i, j]
            count_matrix[res_idx[i], res_idx[j]] += 1

    pae_residue /= np.where(count_matrix > 0, count_matrix, 1)
    return pae_residue, unique_residues


@dataclass
class LossMetadata:
    """Loss function metadata."""

    name: str
    params: List[str] = None
    required_params: List[str] = None
    optional_params: List[str] = None
    description: str = ""
    supports_oligomer_specific: bool = False


class LossRegistry:
    """Registry for loss functions with decorator-based registration."""

    def __init__(self):
        self._loss_functions: Dict[str, Callable] = {}
        self._metadata: Dict[str, LossMetadata] = {}

    def register(
        self,
        name: Optional[str] = None,
        params: Optional[List[str]] = None,
        required_params: Optional[List[str]] = None,
        optional_params: Optional[List[str]] = None,
        description: str = "",
        supports_oligomer_specific: bool = False,
    ):
        """
        Decorator to register a loss function.

        Args:
            name: Loss function name (defaults to function name)
            params: All parameter names
            required_params: Required parameter names
            optional_params: Optional parameter names
            description: Function description
            supports_oligomer_specific: Whether supports <oligomer>.<loss> format
        """

        def decorator(func: Callable) -> Callable:
            loss_name = name or func.__name__

            # Auto-detect parameters from function signature if not provided
            if params is None and required_params is None and optional_params is None:
                sig = inspect.signature(func)
                auto_params = []
                auto_required = []
                auto_optional = []

                for param_name, param in sig.parameters.items():
                    if param_name in ["eval_result", "runtime_config", "oligo2complex"]:
                        continue

                    auto_params.append(param_name)
                    if param.default == inspect.Parameter.empty:
                        auto_required.append(param_name)
                    else:
                        auto_optional.append(param_name)

                all_params = auto_params
                required_params_final = auto_required
                optional_params_final = auto_optional
            else:
                all_params = params or []
                required_params_final = required_params or []
                optional_params_final = optional_params or []

            # Store metadata
            self._metadata[loss_name] = LossMetadata(
                name=loss_name,
                params=all_params,
                required_params=required_params_final,
                optional_params=optional_params_final,
                description=description,
                supports_oligomer_specific=supports_oligomer_specific,
            )

            # Store function
            self._loss_functions[loss_name] = func

            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed_time = time.perf_counter() - start_time
                logging.info(
                    f"{name or func.__name__} executed in {elapsed_time:.6f} seconds"
                )
                return result

            return wrapper

        return decorator

    def get_function(self, name: str) -> Callable:
        """Get registered loss function."""
        if name not in self._loss_functions:
            raise ValueError(f"Loss function '{name}' not registered")
        return self._loss_functions[name]

    def get_metadata(self, name: str) -> LossMetadata:
        """Get loss function metadata."""
        if name not in self._metadata:
            raise ValueError(f"Loss function '{name}' not registered")
        return self._metadata[name]

    @property
    def valid_losses(self) -> List[str]:
        """Get list of all registered loss names."""
        return list(self._loss_functions.keys())

    def validate_params(self, name: str, provided_params: List[str]) -> List[str]:
        """Validate parameters for a loss function."""
        metadata = self.get_metadata(name)
        issues = []

        # Check required parameters
        for required in metadata.required_params or []:
            if required not in provided_params:
                issues.append(
                    f"Missing required parameter '{required}' for loss '{name}'"
                )

        # Check for unknown parameters
        all_valid_params = (metadata.required_params or []) + (
            metadata.optional_params or []
        )
        for param in provided_params:
            if param not in all_valid_params:
                issues.append(f"Unknown parameter '{param}' for loss '{name}'")

        return issues

    @property
    def param_free_losses(self) -> List[str]:
        """Get list of losses with no required parameters."""
        return [
            name
            for name, metadata in self._metadata.items()
            if not metadata.required_params
        ]
