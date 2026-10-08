import logging
import os
import pathlib
import string
from typing import Any

from filelock import FileLock
from omegaconf import DictConfig, OmegaConf, open_dict

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import register_files
from astrolabe import run_config_lifecycle
from astrolabe.utils.run_dirs import format_run_dir_name, parse_run_dir_index
from astrolabe.utils.logging_utils import normalize_logging_config


_AUTO_MANAGED_RAW_DIR_MARKER_TEXT = "auto-managed raw_model_results_dir (safe to delete)\n"

_FALSE_VALUES = {False, 0, "false", "False", "FALSE", "0", "no", "No", "NO"}
_TRUE_VALUES = {True, 1, "true", "True", "TRUE", "1", "yes", "Yes", "YES"}


def _repo_relative_roots(repo_root: pathlib.Path) -> list[pathlib.Path]:
    repo_root = pathlib.Path(repo_root).resolve()
    roots = [repo_root]
    if repo_root.parent.name == ".worktrees":
        roots.append(repo_root.parent.parent.resolve())
    return roots


def _resolve_repo_relative_path(value, *, repo_root: pathlib.Path | None = None):
    if value is None:
        return None
    s = str(value).strip()
    if not s or s in {"null", "none"}:
        return value
    expanded = os.path.expanduser(os.path.expandvars(s))
    p = pathlib.Path(expanded)
    if p.is_absolute():
        return str(p)

    root = pathlib.Path(repo_root).resolve() if repo_root is not None else pathlib.Path(__file__).resolve().parents[1]
    for candidate_root in _repo_relative_roots(root):
        candidate = (candidate_root / expanded).resolve()
        if candidate.exists():
            return str(candidate)
    return value


def _validate_schema_compatibility(args) -> None:
    deprecated: list[str] = []
    if getattr(args, "evaluator", None) is not None:
        evaluator_name = getattr(args.evaluator, "name", None)
        if evaluator_name not in (None, "alphafold3"):
            deprecated.append("evaluator.name")
    if getattr(args, "mutator", None) is not None:
        mutation_num = getattr(args.mutator, "mutation_num", None)
        if mutation_num is not None:
            deprecated.append("mutator.mutation_num")
        mutation_rate = getattr(args.mutator, "mutation_rate", None)
        if mutation_rate not in (None, 0.2):
            deprecated.append("mutator.mutation_rate")
        mutator_weights = getattr(args.mutator, "mutator_weights", None)
        if mutator_weights not in (None,):
            deprecated.append("mutator.mutator_weights")
    if deprecated:
        raise ValueError(
            "Deprecated config keys are no longer accepted: "
            + ", ".join(deprecated)
            + ". Use stages.* fields and component alias maps instead."
        )

    if getattr(args, "reinitialize_random_segments_with_dplm", False):
        logging.warning(
            "`reinitialize_random_segments_with_dplm` is a deprecated compatibility "
            "field; use `initializer.mode: dplm` instead."
        )


def set_random_seeds(seed: int):
    # Set random seeds
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np

    np.random.seed(seed)
    try:
        import torch  # type: ignore
    except ImportError:
        torch = None

    if torch is not None:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_arguments(args):
    # -------------------------------------------------
    # Configure the output directory, and create the necessary subdirectories
    # -------------------------------------------------
    if args.task_name is None or args.task_name == "":
        raise ValueError("Task name is not provided")

    args.task_name = sanitised_name(args.task_name)
    args.out_dir = create_run_directory(args.out_dir, args.task_name)

    (args.out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "structures").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sequences").mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Normalize logging policy (sink levels + color behavior).
    # -------------------------------------------------
    try:
        raw_logging_cfg = OmegaConf.to_container(getattr(args, "logging", None), resolve=True)
    except Exception:
        raw_logging_cfg = getattr(args, "logging", None)
    normalized_logging_cfg = normalize_logging_config(raw_logging_cfg)
    with open_dict(args):
        args.logging = OmegaConf.create(normalized_logging_cfg)

    # -------------------------------------------------
    # raw_model_results_dir: can create massive numbers of small files.
    #
    # Behavior:
    # - If explicitly provided via config/CLI: use it and do NOT auto-delete.
    # - If not provided: default to `/tmp/gh_<run_name>` and auto-delete on success.
    #   If `/tmp` is not usable, fall back to `<out_dir>/raw_model_results`.
    # -------------------------------------------------
    explicit_raw_dir = getattr(args, "raw_model_results_dir", None)
    explicit_raw_dir = None if explicit_raw_dir in ("", "null", "None") else explicit_raw_dir

    auto_managed = False
    marker_path = None

    if explicit_raw_dir is not None:
        raw_dir = pathlib.Path(str(explicit_raw_dir)).expanduser()
        raw_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_name = f"{args.task_name}_{pathlib.Path(args.out_dir).name}"
        auto_managed = True
        raw_dir, marker_path = create_auto_managed_raw_model_results_dir(
            run_name=run_name,
            fallback_out_dir=args.out_dir,
        )

    with open_dict(args):
        args.raw_model_results_dir = raw_dir
        args._raw_model_results_dir_auto_managed = auto_managed
        args._raw_model_results_dir_marker = str(marker_path) if marker_path else None

    # -------------------------------------------------
    # Configure mutator config
    # -------------------------------------------------
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return list(value)
        if OmegaConf.is_list(value):
            return list(value)
        return [value]

    def _collect_stage_mutator_aliases(stage_cfg) -> list[str]:
        if stage_cfg is None:
            return []
        if not (isinstance(stage_cfg, dict) or OmegaConf.is_dict(stage_cfg)):
            raise ValueError(
                f"stages must be a mapping, got {type(stage_cfg).__name__}"
            )
        ordered: list[str] = []
        seen: set[str] = set()

        default_unit = stage_cfg.get("default")
        if default_unit is not None:
            for alias in _as_list(default_unit.get("mutators")):
                alias = str(alias)
                if alias and alias not in seen:
                    ordered.append(alias)
                    seen.add(alias)

        for stage_name, unit in stage_cfg.items():
            if unit is None:
                continue
            if str(stage_name) == "default":
                continue
            mutators = unit.get("mutators")
            for alias in _as_list(mutators):
                alias = str(alias)
                if alias and alias not in seen:
                    ordered.append(alias)
                    seen.add(alias)

        return ordered

    # Determine which mutator aliases should be built for this run.
    #
    # We treat `mutator.mutator_terms` as the source of truth (defines available aliases),
    # and stages reference those aliases under `stages.*.mutators`.
    required_aliases = _collect_stage_mutator_aliases(getattr(args, "stages", None))
    initializer_cfg = getattr(args, "initializer", None)
    initializer_mode = str(
        getattr(initializer_cfg, "mode", "") or ""
    ).strip().lower()
    if initializer_mode in {
        "structure",
        "start_from_structure",
        "start-from-structure",
        "existing_structure",
        "existing-structure",
        "rfd3",
        "rfd-3",
    }:
        initializer_alias = str(
            getattr(initializer_cfg, "sequence_designer", "") or ""
        ).strip()
        if initializer_alias and initializer_alias not in required_aliases:
            required_aliases.append(initializer_alias)
    mutator_terms = getattr(args.mutator, "mutator_terms", None) or {}
    missing_defs = [a for a in required_aliases if mutator_terms.get(a) is None]
    if missing_defs:
        available = sorted(str(k) for k, v in mutator_terms.items() if v is not None)
        raise ValueError(
            "Stages reference mutators that are not defined under mutator.mutator_terms "
            "(including initializer sequence_designer aliases): "
            f"{missing_defs}. Available mutator_terms: {available}."
        )
    enabled_aliases = required_aliases or sorted(
        str(k) for k, v in mutator_terms.items() if v is not None
    )
    with open_dict(args.mutator):
        args.mutator.enabled_aliases = enabled_aliases

    # -------------------------------------------------
    # Configure evaluator config
    # -------------------------------------------------
    def _collect_stage_evaluator_aliases(stage_cfg) -> list[str]:
        if stage_cfg is None:
            return []
        if not (isinstance(stage_cfg, dict) or OmegaConf.is_dict(stage_cfg)):
            raise ValueError(
                f"stages must be a mapping, got {type(stage_cfg).__name__}"
            )
        ordered: list[str] = []
        seen: set[str] = set()

        default_alias = None
        default_unit = stage_cfg.get("default")
        if default_unit is not None:
            default_alias = default_unit.get("evaluator")
            if default_alias:
                alias = str(default_alias)
                if alias and alias not in seen:
                    ordered.append(alias)
                    seen.add(alias)

        for stage_name, unit in stage_cfg.items():
            if unit is None or str(stage_name) == "default":
                continue
            alias = unit.get("evaluator")
            alias = alias or default_alias
            aliases = [alias]
            gate = unit.get("transition_consistency_gate")
            if gate is not None:
                gate_eval = gate.get("evaluator")
                if gate_eval:
                    aliases.append(gate_eval)
            for alias in aliases:
                if not alias:
                    continue
                alias = str(alias)
                if alias and alias not in seen:
                    ordered.append(alias)
                    seen.add(alias)

        return ordered

    if getattr(args, "evaluator", None) is not None:
        required_eval_aliases = _collect_stage_evaluator_aliases(getattr(args, "stages", None))
        evaluator_terms = getattr(args.evaluator, "evaluator_terms", None) or {}
        missing_eval_defs = [a for a in required_eval_aliases if evaluator_terms.get(a) is None]
        if missing_eval_defs:
            available = sorted(str(k) for k, v in evaluator_terms.items() if v is not None)
            raise ValueError(
                "Stages reference evaluators that are not defined under evaluator.evaluator_terms: "
                f"{missing_eval_defs}. Available evaluator_terms: {available}."
            )
        enabled_eval_aliases = required_eval_aliases or sorted(
            str(k) for k, v in evaluator_terms.items() if v is not None
        )
        with open_dict(args.evaluator):
            args.evaluator.enabled_aliases = enabled_eval_aliases

    # -------------------------------------------------
    # Configure the mutator config
    # selector_expression, separated by ";" for multiple selectors. Each selector is a list of strings separated by ":", e.g. "MCMC: expression"
    # -------------------------------------------------
    name2expression = {name: "None" for name in enabled_aliases}
    if (
        args.mutator.selector_expression is not None
        and ":" in args.mutator.selector_expression
    ):
        selectors = args.mutator.selector_expression.split(";")
        name2expression = {}
        for selector in selectors:
            name, *expression = selector.split(":")
            # in case the expression contains ":", join the rest of the string
            name2expression[name.strip()] = "".join(expression).strip()
    elif args.mutator.selector_expression is not None:
        for name in enabled_aliases:
            name2expression[name] = args.mutator.selector_expression
    else:
        pass

    with open_dict(args.mutator):
        args.mutator.selectors = name2expression

    # -------------------------------------------------
    # Resolve tool paths (repo-relative -> absolute).
    #
    # When using `gh run --config-file ...`, the CLI may `chdir` into the config
    # directory. Relative paths in env.yaml (e.g. `vendor/mkdssp`) should still
    # resolve relative to the repo root, not the current working directory.
    # -------------------------------------------------
    repo_root = pathlib.Path(__file__).resolve().parents[1]

    with open_dict(args):
        args.dssp_path = _resolve_repo_relative_path(
            getattr(args, "dssp_path", None),
            repo_root=repo_root,
        )
        args.usalign_path = _resolve_repo_relative_path(
            getattr(args, "usalign_path", None),
            repo_root=repo_root,
        )
        args.ananas_path = _resolve_repo_relative_path(
            getattr(args, "ananas_path", None),
            repo_root=repo_root,
        )

    # -------------------------------------------------
    # Configure random seed
    # -------------------------------------------------
    raw_seed = args.seed
    seed_error = (
        f"Invalid seed value {raw_seed!r}; expected a positive decimal integer "
        f"within 1..{2**32 - 1} from the repeat lifecycle."
    )
    seed_is_decimal_string = False
    if type(raw_seed) is str:
        digits = raw_seed[1:] if raw_seed[:1] in "+-" else raw_seed
        seed_is_decimal_string = (
            bool(digits) and digits.isascii() and digits.isdecimal()
        )
    if type(raw_seed) is int:
        seed = raw_seed
    elif seed_is_decimal_string:
        seed = int(raw_seed)
    else:
        raise ValueError(seed_error)
    if not 1 <= seed <= 2**32 - 1:
        raise ValueError(seed_error)
    args.seed = seed
    set_random_seeds(seed)

    # -------------------------------------------------
    # Reject deprecated/ignored config keys
    # -------------------------------------------------
    _validate_schema_compatibility(args)

    # -------------------------------------------------
    # Configure the msa
    # -------------------------------------------------
    # if args.proto.extra_msa:
    #     if not isinstance(args.proto.extra_msa, DictConfig):
    #         raise ValueError("extra_msa must be a dictionary")
    #     keys = ["unpairedMsa", "pairedMsa", "templates"]
    #     for key, value in args.proto.extra_msa.items():
    #         if isinstance(value, str) and os.path.exists(value):
    #             if value.endswith(".json"):
    #                 with open(value) as f:
    #                     homology_dict = json.load(f)
    #                 args.proto.extra_msa[key] = {k: homology_dict.get(k) for k in keys}
    #
    #             else:
    #                 args.proto.extra_msa[key] = {
    #                     "unpairedMsa": open(value).read(),
    #                     "pairedMsa": None,
    #                     "templates": None,
    #                 }
    #         elif (
    #             isinstance(value, DictConfig)
    #             and all(k in value for k in keys)
    #             and all(os.path.exists(value[k]) for k in keys)
    #         ):
    #             args.proto.extra_msa[key] = {
    #                 homo_key: open(value[homo_key], "r").read() for homo_key in keys
    #             }

    # ------------------------------------------------
    # Configure the initial positions
    # ------------------------------------------------
    # if args.proto.init_positions:
    #     if not isinstance(args.proto.init_positions, str):
    #         raise ValueError(
    #             "Initial positions must be a string, `<chains>:<path_to_structure>"
    #         )
    #     init_positions = {}
    #     for statement in args.proto.init_positions.split(","):
    #
    #         to_initialized_proto, initialized_path = [
    #             i.strip() for i in statement.split(":")
    #         ]
    #
    #         if len(to_initialized_proto) > 1:
    #             raise NotImplementedError(
    #                 f"Currently, initial positions can only be specified for one chain, but {len(to_initialized_proto)} chains are given."
    #             )
    #
    #         init_positions[to_initialized_proto] = {
    #             "init_guess_path": initialized_path,
    #         }
    #     args.proto.init_positions = init_positions

    # ------------------------------------------------
    # Configure the file aliases
    # ------------------------------------------------
    if args.named_files is not None:
        if not isinstance(args.named_files, DictConfig):
            raise ValueError("named_files must be a dictionary")
        named_files = {k: os.path.abspath(v) for k, v in args.named_files.items()}
        flag = False
        for k, v in named_files.items():
            if not os.path.exists(v):
                logging.error(f"File {v} does not exist, please check the path.")
                flag = True
            named_files[k] = v
        if flag:
            raise ValueError("Some named files do not exist, please check the paths.")
        register_files(
            named_files,
        )
    else:
        register_files({})
    if os.getenv("DEBUG") == "1" or args.debug:
        global_config.init_const("is_debug", True)
    else:
        global_config.init_const("is_debug", False)

    if args.types is None:
        args.types = {key: "p" for key in args.seqs.keys()}

    # Dump a pruned persisted view after injecting derived fields (selectors, enabled_aliases, etc).
    OmegaConf.save(
        run_config_lifecycle.build_persisted_run_config(args),
        args.out_dir / "config.yaml",
    )
    # Record machine-readable provenance for this run directory (best-effort; the
    # config digest above is only available once the persisted config exists).
    run_config_lifecycle.start_run_provenance(args)
    return args


def create_auto_managed_raw_model_results_dir(
    *,
    run_name: str,
    tmp_root: pathlib.Path | str = pathlib.Path("/tmp"),
    fallback_out_dir: pathlib.Path | str | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    tmp_root = pathlib.Path(tmp_root).expanduser()
    base_name = f"gh_{run_name}"
    last_error: OSError | None = None

    for suffix_idx in range(1, 1001):
        suffix = "" if suffix_idx == 1 else f"_{suffix_idx}"
        candidate = tmp_root / f"{base_name}{suffix}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        except OSError as exc:
            last_error = exc
            break

        marker = candidate / ".gh_auto_managed"
        marker.write_text(_AUTO_MANAGED_RAW_DIR_MARKER_TEXT, encoding="utf-8")
        return candidate, marker

    if fallback_out_dir is None:
        if last_error is not None:
            raise last_error
        raise FileExistsError(
            f"Failed to allocate auto-managed raw_model_results_dir under {tmp_root} "
            f"for run {run_name!r}: exhausted collision retries."
        )

    fallback_dir = pathlib.Path(fallback_out_dir).expanduser() / "raw_model_results"
    fallback_dir.mkdir(parents=True, exist_ok=False)
    marker = fallback_dir / ".gh_auto_managed"
    marker.write_text(_AUTO_MANAGED_RAW_DIR_MARKER_TEXT, encoding="utf-8")
    return fallback_dir, marker


def create_run_directory(out_dir, task_name):
    if out_dir is None or out_dir == "":
        out_dir = pathlib.Path(".").resolve() / task_name
    else:
        out_dir = pathlib.Path(out_dir).resolve() / task_name
    os.makedirs(out_dir, exist_ok=True)

    # Following the convention of naming the run folders as 001_run, 002_run, etc.
    # Use a lock to ensure that the run folders are created in a thread-safe manner.
    with FileLock(out_dir / "runs.lock"):
        modifier = ""
        existing_run_indices = []
        for entry in out_dir.iterdir():
            if not entry.is_dir():
                continue
            run_index = parse_run_dir_index(entry.name)
            if run_index is None:
                continue
            existing_run_indices.append(run_index)
        next_run_index = max(existing_run_indices, default=0) + 1
        next_run = format_run_dir_name(next_run_index)
        if "SLURM_ARRAY_JOB_ID" in os.environ:
            modifier += f'_{os.environ["SLURM_ARRAY_JOB_ID"]}'
            modifier += f'_{os.environ["SLURM_ARRAY_TASK_ID"]}'
        elif "SLURM_JOB_ID" in os.environ:
            modifier += f'_{os.environ["SLURM_JOB_ID"]}'

        run_dir = out_dir / (next_run + modifier)
        os.mkdir(run_dir)

    return run_dir


def sanitised_name(name) -> str:
    """Returns sanitised version of the name that can be used as a filename."""
    lower_spaceless_name = name.lower().replace(" ", "_")
    allowed_chars = set(string.ascii_lowercase + string.digits + "_-.")
    return "".join(l for l in lower_spaceless_name if l in allowed_chars)
