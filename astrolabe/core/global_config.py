# astrolabe/core/global_config.py
from dataclasses import dataclass

DEFAULT_MUTATION_FRACTION = 0.1


# Track initialization state for each constant and variable
_initialized_constants: dict[str, bool] = {}
constants_names = [
    "num_tokens",
    "PARSED_FILE",
    "predefined_statement",
    "segments",
    "named_files",
    "exclude_AA",
    "covalent_bonds",
    "protomers",
    "mutable_protomers",
    "segment_manager",
    "initial_temperature",
    "out_dir",
    "raw_model_results_dir",
    "is_debug",
    "use_metropolis_acceptance",
    "stage_manager",
    "loss_computer",
    "predockers",
    "mutators",
    "initializer_result",
    "evaluators",
    "decisions",
    "obabel_path",
    "molfile_to_params",
    "dssp_path",
    "usalign_path",
    "ananas_path",
    "enable_pyrosetta",
    "skip_complex_poses",
    "allow_mmcif_pdb_fallback",
    "per_residue_scores",
    "pre_eval_hooks",
    "protenix_constraint",
    "binding_site",
    "dynamic_binding_site",
    "dynamic_binding_site_enabled",
    "chai_restraints",
    "channel",
    "cyclization",
    "ligand_resnames",
    "nonstandard_aliases",
    "user_ccd",
    "structure_start",
    # Add more constant names here as needed
]

variable_names = [
    "step",
    "ignored_complexes",
    "negative_design_complexes",
    "mutation_num",
    "pore_bottleneck_activation_index",
]

# Initialize all constants and variables to None
for name in constants_names:
    globals()[f"_{name}"] = None
    _initialized_constants[name] = False

for name in variable_names:
    globals()[f"_{name}"] = None


@dataclass(frozen=True)
class RuntimeContext:
    """Explicit runtime-state access object over global_config storage."""

    _MISSING = object()

    def get_const(self, name: str, default=_MISSING):
        try:
            return get_const(name)
        except RuntimeError:
            if default is not self._MISSING:
                return default
            raise

    @property
    def protomers(self):
        return self.get_const("protomers", {})

    @property
    def mutable_protomers(self):
        return self.get_const("mutable_protomers", {})

    @property
    def mutators(self):
        return self.get_const("mutators", {})

    @property
    def evaluators(self):
        return self.get_const("evaluators", {})

    @property
    def stage_manager(self):
        return self.get_const("stage_manager", None)

    @property
    def step(self):
        value = get_variable("step")
        return 0 if value is None else value

    def set_variable(self, name: str, value) -> None:
        set_variable(name, value)

    def get_variable(self, name: str):
        return get_variable(name)


def build_runtime_context() -> RuntimeContext:
    return RuntimeContext()


def init_const(name, value):
    """Initialize a constant with the given value.

    Args:
        name: The name of the constant to initialize
        value: The value to assign to the constant

    Raises:
        ValueError: If the constant name is not in the predefined list
    """
    if name not in constants_names:
        raise ValueError(f"'{name}' is not a recognized constant name")

    if _initialized_constants[name]:
        current = globals()[f"_{name}"]
        if current == value:
            return
        raise RuntimeError(
            f"Constant '{name}' is already initialized. Call `global_config.reset()` before re-initializing constants."
        )
    globals()[f"_{name}"] = value
    _initialized_constants[name] = True


def update_const(name, value):
    """Update an already initialized constant with an explicit runtime value."""
    if name not in constants_names:
        raise ValueError(f"'{name}' is not a recognized constant name")
    if not _initialized_constants[name]:
        raise RuntimeError(f"Constant '{name}' has not been initialized yet")
    globals()[f"_{name}"] = value


def set_variable(name, value):
    """Set a variable value.

    Args:
        name: The name of the variable to set
        value: The value to assign to the variable

    Raises:
        ValueError: If the variable name is not in the predefined list
    """
    if name not in variable_names:
        raise ValueError(f"'{name}' is not a recognized variable name")

    globals()[f"_{name}"] = value


def get_const(name):
    """Get the value of a constant.

    Args:
        name: The name of the constant to retrieve

    Returns:
        The value of the constant

    Raises:
        ValueError: If the constant name is not in the predefined list
        RuntimeError: If the constant has not been initialized yet
    """
    if name not in constants_names:
        raise ValueError(f"'{name}' is not a recognized constant name")

    if not _initialized_constants[name]:
        raise RuntimeError(f"Constant '{name}' has not been initialized yet")

    return globals()[f"_{name}"]


def get_variable(name):
    """Get the value of a variable.

    Args:
        name: The name of the variable to retrieve

    Returns:
        The value of the variable

    Raises:
        ValueError: If the variable name is not in the predefined list
    """
    if name not in variable_names:
        raise ValueError(f"'{name}' is not a recognized variable name")

    return globals()[f"_{name}"]


def __getattr__(name):
    """Custom attribute access to make constants and variables available as module attributes."""
    if name in constants_names:
        return get_const(name)
    if name in variable_names:
        return get_variable(name)
    raise AttributeError(f"module 'global_config' has no attribute '{name}'")


def reset():
    """Reset all constants and variables to their uninitialized state."""
    for name in constants_names:
        globals().pop(name, None)
        globals()[f"_{name}"] = None
        _initialized_constants[name] = False
    for name in variable_names:
        globals().pop(name, None)
        globals()[f"_{name}"] = None
