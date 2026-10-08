from __future__ import annotations

import ast
import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

from astrolabe.runtime_usage import analyze_runtime_usage


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _extract_selector_vars(expr: Optional[str]) -> set[str]:
    if not expr or expr == "None":
        return set()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid position_selector expression: {expr!r}") from exc
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


@lru_cache(maxsize=1)
def _pyrosetta_loss_types_from_source() -> set[str]:
    """
    Infer which loss names are PyRosetta-based without importing PyRosetta.
    Heuristic: losses declared in modules that import/use PyRosetta.
    """
    losses_dir = _repo_root() / "astrolabe" / "losses"
    if not losses_dir.exists():
        return set()

    out: set[str] = set()
    skip = {"__init__.py", "base.py", "utils.py"}
    for path in sorted(losses_dir.glob("*.py")):
        if path.name in skip:
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        if "pyrosetta" not in src and ".rosetta." not in src:
            continue
        tree = ast.parse(src, filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Call):
                    continue
                fn = dec.func
                if not (isinstance(fn, ast.Attribute) and fn.attr == "register"):
                    continue
                if not (isinstance(fn.value, ast.Name) and fn.value.id == "loss_registry"):
                    continue
                loss_name = None
                for kw in dec.keywords:
                    if kw.arg == "name":
                        v = _ast_literal(kw.value)
                        if isinstance(v, str) and v.strip():
                            loss_name = v.strip()
                node_src = ast.get_source_segment(src, node) or ""
                arg_names = [a.arg for a in list(node.args.args) + list(node.args.kwonlyargs)]
                uses_pyrosetta = (
                    path.name == "interaction.py"
                    or "pyrosetta_init" in arg_names
                    or "_ensure_pyrosetta" in node_src
                    or "_bindcraft_interface_field" in node_src
                    or "_sap_score_single" in node_src
                    or "_align_chain" in node_src
                    or "_chain_rmsd" in node_src
                    or "_bindcraft_binder_alone_score_single" in node_src
                    or "pyrosetta." in node_src
                    or "pr.rosetta" in node_src
                )
                if not uses_pyrosetta:
                    break
                out.add(loss_name or node.name)
                break
    return out


def _ast_literal(node: Any) -> Any:
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError, SyntaxError):
        return None


def task_requirements(task_cfg: Any) -> dict:
    req_env_keys: set[str] = set()
    req_repo_files: set[str] = set()
    req_env_repo_files: set[tuple[str, str, str]] = set()
    req_imports: set[str] = set()

    def _as_bool(value: Any, *, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off", "none", "null"}:
                return False
        return bool(value)

    # Always needed by `run.py` (used for logging control).
    req_imports.add("absl")

    enable_pyrosetta = bool(task_cfg.get("enable_pyrosetta", True)) if isinstance(task_cfg, dict) else True
    post_eval_hooks = (task_cfg.get("post_eval_hooks") or {}) if isinstance(task_cfg, dict) else {}
    binding_site_cfg = task_cfg.get("binding_site") if isinstance(task_cfg, dict) else None
    protenix_constraint_cfg = task_cfg.get("protenix_constraint") if isinstance(task_cfg, dict) else None
    dynamic_binding_site_cfg = (
        task_cfg.get("dynamic_binding_site_enabled") if isinstance(task_cfg, dict) else None
    )
    initializer_cfg = task_cfg.get("initializer") if isinstance(task_cfg, dict) else {}
    initializer_mode = (
        str((initializer_cfg or {}).get("mode") or "").strip().lower()
        if isinstance(initializer_cfg, dict)
        else ""
    )

    usage = analyze_runtime_usage(task_cfg)
    evaluator_terms_used = usage.evaluator_terms
    mutator_terms_used = usage.mutator_terms
    predocker_terms_used = usage.predocker_terms
    loss_types = usage.loss_types
    evaluator_types_used = usage.evaluator_types
    mutator_types_used = usage.mutator_types
    predocker_types_used = usage.predocker_types

    if initializer_mode in {"rfd3", "rfd-3"}:
        req_env_keys |= {
            "weight_path.rfd3",
            "rfd3.repo_path",
            "rfd3.ckpt_path",
            "subprocess_envs.rf3.python_bin",
        }

    selector_exprs: list[str] = []
    for conf in mutator_terms_used.values():
        if not isinstance(conf, dict):
            continue
        expr = conf.get("position_selector")
        if isinstance(expr, str) and expr.strip():
            selector_exprs.append(expr.strip())

    # Evaluator requirements
    af3_terms_used = [
        conf
        for conf in evaluator_terms_used.values()
        if str(conf.get("_type") or "").strip().lower() == "alphafold3"
    ]
    af3_data_pipeline_used = any(
        _as_bool(conf.get("run_data_pipeline"), default=False) for conf in af3_terms_used
    )

    if any(t.lower() == "alphafold3" for t in evaluator_types_used):
        req_env_keys.add("weight_path.alphafold3")
        if af3_data_pipeline_used:
            req_env_keys |= {
                "hmmer.jackhmmer",
                "hmmer.nhmmer",
                "hmmer.hmmalign",
                "hmmer.hmmsearch",
                "hmmer.hmmbuild",
            }
        # Validate the package Python will actually import, not a fresh vendor
        # checkout that can hide the container's already-built data files.
        spec = importlib.util.find_spec("alphafold3")
        af3_root = (
            Path(next(iter(spec.submodule_search_locations)))
            if spec is not None and spec.submodule_search_locations
            else _repo_root() / "vendor" / "alphafold3" / "src" / "alphafold3"
        )
        for name in ("ccd.pickle", "chemical_component_sets.pickle"):
            req_repo_files.add(str(af3_root / "constants" / "converters" / name))
        # AlphaFold3 python deps (we install vendor/alphafold3 with --no-deps).
        # These imports are cheap and catch missing deps before a long run.
        req_imports |= {
            "jax",
            "jaxlib",
            "jax_triton",
            "haiku",
            "tree",
            "jaxtyping",
            "typeguard",
            "zstandard",
        }

    if any(t.lower() == "alphafold2" for t in evaluator_types_used):
        req_env_keys.add("weight_path.alphafold")
        # AF2 still imports TensorFlow in-process through the vendored stack.
        req_imports.add("tensorflow")

    if any(t.lower() == "protenix" for t in evaluator_types_used):
        req_env_keys.add("weight_path.protenix")
        req_env_keys.add("protenix.ccd_components_file")
        req_env_keys.add("protenix.ccd_components_rdkit_mol_file")
        protenix_cfg = (task_cfg.get("evaluator") or {}).get("protenix") or {}
        protenix_dynamic_binding_site = False
        if isinstance(protenix_cfg, dict):
            protenix_dynamic_binding_site = bool(protenix_cfg.get("dynamic_binding_site", False))
        if (
            binding_site_cfg not in (None, {}, "null", "None")
            or protenix_constraint_cfg not in (None, {}, "null", "None")
            or bool(dynamic_binding_site_cfg)
            or protenix_dynamic_binding_site
        ):
            req_env_keys.add("weight_path.protenix_constraint")
        if isinstance(protenix_cfg, dict):
            triangle_attention = str(protenix_cfg.get("triangle_attention") or "").lower()
            triangle_multiplicative = str(protenix_cfg.get("triangle_multiplicative") or "").lower()
            if protenix_cfg.get("use_deepspeed_evo_attention") or triangle_attention == "deepspeed":
                req_env_keys.add("CUTLASS_PATH")
                req_imports.add("deepspeed")
            if triangle_attention == "cuequivariance" or triangle_multiplicative == "cuequivariance":
                req_imports.add("cuequivariance_torch")

    opendde_terms_used = [
        conf
        for conf in evaluator_terms_used.values()
        if str(conf.get("_type") or "").strip().lower() == "opendde"
    ]
    opendde_cfg = (task_cfg.get("evaluator") or {}).get("opendde") or {}
    if not isinstance(opendde_cfg, dict):
        opendde_cfg = {}
    if any(t.lower() == "boltz" for t in evaluator_types_used):
        req_env_keys.add("weight_path.boltz")
        req_imports |= {"boltz", "pytorch_lightning"}

    if any(t.lower() == "opendde" for t in evaluator_types_used):
        req_env_keys |= {
            "weight_path.opendde",
            "subprocess_envs.opendde.repo_path",
        }
        if not (
            opendde_cfg.get("load_checkpoint_path")
            or any(conf.get("load_checkpoint_path") for conf in opendde_terms_used)
        ):
            req_env_repo_files.add(
                ("weight_path.opendde", "checkpoint/opendde.pt", "opendde_checkpoint")
            )
        if _as_bool(opendde_cfg.get("use_template"), default=False) or any(
            _as_bool(conf.get("use_template"), default=False) for conf in opendde_terms_used
        ):
            req_env_keys |= {"hmmer.hmmsearch", "hmmer.hmmbuild"}
        if _as_bool(opendde_cfg.get("use_rna_msa"), default=False) or any(
            _as_bool(conf.get("use_rna_msa"), default=False) for conf in opendde_terms_used
        ):
            req_env_keys |= {"hmmer.nhmmer", "hmmer.hmmalign", "hmmer.hmmbuild"}

    if any(t.lower() == "chai1" for t in evaluator_types_used):
        req_imports.add("chai_lab")

    if any(t.lower() == "esmfold2" for t in evaluator_types_used):
        req_env_keys |= {
            "subprocess_envs.esmfold2.python_bin",
            "subprocess_envs.esmfold2.esm_repo_path",
            "subprocess_envs.esmfold2.transformers_repo_path",
        }
    elif any(t.lower() == "esmfold" for t in evaluator_types_used):
        req_imports |= {"esm2", "torch"}

    if any(t.lower() == "rhofold" for t in evaluator_types_used):
        req_env_keys.add("rhofold.checkpoint")
        req_imports |= {"rhofold", "torch"}

    # Mutator requirements
    if any(
        t.lower()
        in {
            "ligandmpnn",
            "grouped_ligandmpnn",
            "dynamicmpnn",
            "proteinmpnn",
            "multistateproteinmpnn",
            "multistate_proteinmpnn",
            "faraparttiedmultitargetligandmpnn",
            "far_apart_tied_multitarget_ligandmpnn",
        }
        for t in mutator_types_used
    ):
        req_env_keys.add("weight_path.ligandmpnn")
        req_imports.add("prody")

    if any(t.lower() == "dplm" for t in mutator_types_used):
        req_env_keys |= {"subprocess_envs.dplm.python_bin", "subprocess_envs.dplm.repo_path"}

    if any(t.lower() == "surfdesign" for t in mutator_types_used):
        req_env_keys |= {
            "weight_path.surfdesign",
            "subprocess_envs.surfdesign.python_bin",
            "subprocess_envs.surfdesign.repo_path",
        }
        req_env_repo_files.add(
            ("weight_path.surfdesign", ".hydra/config.yaml", "surfdesign_config")
        )
        surfdesign_cfg = (task_cfg.get("mutator") or {}).get("SurfDesign") or {}
        checkpoint = next(
            (
                conf.get("checkpoint")
                for conf in mutator_terms_used.values()
                if str(conf.get("_type") or "").strip().lower() == "surfdesign"
                and conf.get("checkpoint")
            ),
            None,
        ) or (
            surfdesign_cfg.get("checkpoint", "best.ckpt")
            if isinstance(surfdesign_cfg, dict)
            else "best.ckpt"
        )
        req_env_repo_files.add(
            (
                "weight_path.surfdesign",
                f"checkpoints/{checkpoint}",
                "surfdesign_checkpoint",
            )
        )

    if any(t.lower() == "surfpro" for t in mutator_types_used):
        req_env_keys |= {
            "weight_path.surfpro",
            "subprocess_envs.surfpro.python_bin",
            "subprocess_envs.surfpro.repo_path",
            "subprocess_envs.surfpro.msms_bin",
            "subprocess_envs.surfpro.pdb_to_xyzrn_bin",
        }
        req_repo_files.add("vendor/sru/sru/__init__.py")
        req_repo_files.add(
            "vendor/protein_sequence_models/sequence_models/structure.py"
        )
        req_env_repo_files.add(
            (
                "subprocess_envs.surfpro.repo_path",
                "fairseq/models/surface_protein_model.py",
                "surfpro_source",
            )
        )
        surfpro_cfg = (task_cfg.get("mutator") or {}).get("SurfPro") or {}
        checkpoint = next(
            (
                conf.get("checkpoint")
                for conf in mutator_terms_used.values()
                if str(conf.get("_type") or "").strip().lower() == "surfpro"
                and conf.get("checkpoint")
            ),
            None,
        ) or (
            surfpro_cfg.get("checkpoint", "checkpoint_best.pt")
            if isinstance(surfpro_cfg, dict)
            else "checkpoint_best.pt"
        )
        checkpoint_path = Path(str(checkpoint))
        if not checkpoint_path.is_absolute():
            req_env_repo_files.add(
                (
                    "weight_path.surfpro",
                    str(checkpoint_path),
                    "surfpro_checkpoint",
                )
            )

    if any(t.lower() == "nampnn" for t in mutator_types_used):
        req_env_keys |= {"subprocess_envs.nampnn.python_bin", "subprocess_envs.nampnn.repo_path"}
        mutator_cfg = task_cfg.get("mutator") if isinstance(task_cfg, dict) else {}
        nampnn_cfg = (mutator_cfg or {}).get("NAMPNN") if isinstance(mutator_cfg, dict) else {}
        checkpoint = None
        if isinstance(nampnn_cfg, dict):
            checkpoint = nampnn_cfg.get("checkpoint")
        checkpoint_path = str(checkpoint or "models/design_model/s_19137.pt")
        req_env_repo_files.add(
            (
                "subprocess_envs.nampnn.repo_path",
                checkpoint_path,
                "nampnn_checkpoint",
            )
        )

    if any(t.lower() == "carbondesign" for t in mutator_types_used):
        req_env_keys |= {
            "carbondesign.repo_path",
            "carbondesign.model_ckpt",
            "carbondesign.model_config",
            "carbondesign.model_features",
        }

    if any(t.lower() == "esm3" for t in mutator_types_used):
        # Vendored ESM3 pulls weights via huggingface_hub (import-time dependency).
        req_imports |= {
            "huggingface_hub",
            "zstd",
            "transformers",
            "tenacity",
            "cloudpathlib",
            "msgpack_numpy",
        }

    # Predocker requirements
    if any(t.lower() == "autodock_gpu" for t in predocker_types_used):
        req_env_keys |= {
            "obabel_path",
            "autodock.autodock_gpu_path",
            "autodock.autogrid4_path",
            "autodock.mk_prepare_receptor",
            "autodock.mk_prepare_ligand",
            "subprocess_envs.meeko.python_bin",
        }

    if any(t.lower() == "autodock_vina" for t in predocker_types_used):
        req_env_keys |= {
            "obabel_path",
            "autodock.vina_path",
            "autodock.mk_prepare_ligand",
            "subprocess_envs.meeko.python_bin",
        }

    if any(t.lower() == "rapidock" for t in predocker_types_used):
        req_env_keys |= {
            "weight_path.rapidock",
            "subprocess_envs.rapidock.python_bin",
            "subprocess_envs.rapidock.repo_path",
        }

    if any(t.lower() in {"haddock3_nucleic", "haddock3-nucleic"} for t in predocker_types_used):
        req_env_keys.add("subprocess_envs.haddock3.haddock3_bin")

    if any(t.lower() == "symmetry_canonicalizer" for t in predocker_types_used):
        symmetry_terms = [
            conf for conf in predocker_terms_used.values()
            if isinstance(conf, dict) and str(conf.get("_type") or "").lower() == "symmetry_canonicalizer"
        ]
        if any(str(conf.get("backend") or "").lower() == "rpxdock" for conf in symmetry_terms):
            req_env_keys |= {
                "subprocess_envs.rpxdock.python_bin",
                "subprocess_envs.rpxdock.repo_path",
                "subprocess_envs.rpxdock.willutil_repo_path",
            }

    # Loss requirements (external tools)
    if "tmscore" in loss_types:
        req_env_keys.add("usalign_path")
    if "min_dssp_fractions" in loss_types:
        req_env_keys.add("dssp_path")
    if loss_types & {"ananas_rmsd_raw", "ananas_rmsd_norm"}:
        req_env_keys.add("ananas_path")

    # PyRosetta requirement inference
    pyro_losses = _pyrosetta_loss_types_from_source()
    needs_pyrosetta = any(t in pyro_losses for t in loss_types)
    if isinstance(post_eval_hooks, dict):
        rosetta_hook = post_eval_hooks.get("rosetta_minimize") or post_eval_hooks.get("rosetta_repack")
        if isinstance(rosetta_hook, dict) and bool(rosetta_hook.get("enabled", False)):
            needs_pyrosetta = True
    selector_vars: set[str] = set()
    for expr in selector_exprs:
        selector_vars |= _extract_selector_vars(expr)
    if selector_vars & {"sse", "cdr", "bfactor", "interface", "surface", "long_helix_core", "long_loop"}:
        needs_pyrosetta = True

    if needs_pyrosetta:
        req_imports.add("pyrosetta")

    env_key_reasons: dict[str, str] = {}

    def _note_env(keys: Iterable[str], reason: str) -> None:
        for key in keys:
            if key in req_env_keys and key not in env_key_reasons:
                env_key_reasons[key] = reason

    if any(t.lower() == "alphafold3" for t in evaluator_types_used):
        _note_env(["weight_path.alphafold3"], "required by Alphafold3 evaluator")
        if af3_data_pipeline_used:
            _note_env(
                [
                    "hmmer.jackhmmer",
                    "hmmer.nhmmer",
                    "hmmer.hmmalign",
                    "hmmer.hmmsearch",
                    "hmmer.hmmbuild",
                ],
                "required by explicit Alphafold3 MSA/data pipeline",
            )
    if any(t.lower() == "alphafold2" for t in evaluator_types_used):
        _note_env(["weight_path.alphafold"], "required by Alphafold2 evaluator")
    if any(t.lower() == "protenix" for t in evaluator_types_used):
        _note_env(["weight_path.protenix"], "required by Protenix evaluator checkpoint")
        _note_env(
            ["weight_path.protenix_constraint"],
            "required by Protenix binding-site/constraint guidance checkpoint",
        )
        _note_env(
            ["protenix.ccd_components_file", "protenix.ccd_components_rdkit_mol_file"],
            "required by Protenix chemical component lookup",
        )
        _note_env(["CUTLASS_PATH"], "required by Protenix DeepSpeed Evoformer attention")
    if any(t.lower() == "boltz" for t in evaluator_types_used):
        _note_env(["weight_path.boltz"], "required by Boltz evaluator cache/models")
    if any(t.lower() == "esmfold2" for t in evaluator_types_used):
        _note_env(
            ["subprocess_envs.esmfold2.python_bin"],
            "required by Biohub ESMFold2 subprocess runtime",
        )
        _note_env(
            [
                "subprocess_envs.esmfold2.esm_repo_path",
                "subprocess_envs.esmfold2.transformers_repo_path",
            ],
            "required by vendored Biohub ESMFold2 source",
        )
    if any(t.lower() == "opendde" for t in evaluator_types_used):
        _note_env(["weight_path.opendde"], "required by OpenDDE OPENDDE_ROOT_DIR assets")
        _note_env(
            ["subprocess_envs.opendde.repo_path"],
            "required by OpenDDE subprocess evaluator",
        )
        _note_env(
            ["hmmer.hmmsearch", "hmmer.hmmbuild"],
            "required by OpenDDE template preprocessing",
        )
        _note_env(
            ["hmmer.nhmmer", "hmmer.hmmalign"],
            "required by OpenDDE RNA MSA preprocessing",
        )
    if any(t.lower() == "rhofold" for t in evaluator_types_used):
        _note_env(["rhofold.checkpoint"], "required by RhoFold evaluator")
    if initializer_mode in {"rfd3", "rfd-3"}:
        _note_env(
            [
                "rfd3.repo_path",
                "rfd3.ckpt_path",
                "weight_path.rfd3",
                "subprocess_envs.rf3.python_bin",
            ],
            "required by RFD3 initializer subprocess",
        )
    if any(
        t.lower() in {"ligandmpnn", "grouped_ligandmpnn", "dynamicmpnn"}
        for t in mutator_types_used
    ):
        _note_env(["weight_path.ligandmpnn"], "required by MPNN mutator checkpoints")
    if any(t.lower() == "dplm" for t in mutator_types_used):
        _note_env(
            ["subprocess_envs.dplm.python_bin", "subprocess_envs.dplm.repo_path"],
            "required by DPLM mutator subprocess",
        )
    if any(t.lower() == "surfdesign" for t in mutator_types_used):
        _note_env(
            [
                "weight_path.surfdesign",
                "subprocess_envs.surfdesign.python_bin",
                "subprocess_envs.surfdesign.repo_path",
            ],
            "required by SurfDesign persistent mutator",
        )
    if any(t.lower() == "surfpro" for t in mutator_types_used):
        _note_env(
            [
                "weight_path.surfpro",
                "subprocess_envs.surfpro.python_bin",
                "subprocess_envs.surfpro.repo_path",
                "subprocess_envs.surfpro.msms_bin",
                "subprocess_envs.surfpro.pdb_to_xyzrn_bin",
            ],
            "required by SurfPro persistent mutator and MSMS preprocessing",
        )
    if any(t.lower() == "nampnn" for t in mutator_types_used):
        _note_env(
            ["subprocess_envs.nampnn.python_bin", "subprocess_envs.nampnn.repo_path"],
            "required by NAMPNN mutator subprocess",
        )
    if any(t.lower() == "carbondesign" for t in mutator_types_used):
        _note_env(
            [
                "carbondesign.repo_path",
                "carbondesign.model_ckpt",
                "carbondesign.model_config",
                "carbondesign.model_features",
            ],
            "required by CarbonDesign mutator",
        )
    if any(t.lower() == "autodock_gpu" for t in predocker_types_used):
        _note_env(
            [
                "obabel_path",
                "autodock.autodock_gpu_path",
                "autodock.autogrid4_path",
                "autodock.mk_prepare_receptor",
                "autodock.mk_prepare_ligand",
                "subprocess_envs.meeko.python_bin",
            ],
            "required by AutoDock-GPU predocker",
        )
    if any(t.lower() == "autodock_vina" for t in predocker_types_used):
        _note_env(
            [
                "obabel_path",
                "autodock.vina_path",
                "autodock.mk_prepare_ligand",
                "subprocess_envs.meeko.python_bin",
            ],
            "required by AutoDock Vina predocker",
        )
    if any(t.lower() == "rapidock" for t in predocker_types_used):
        _note_env(
            [
                "weight_path.rapidock",
                "subprocess_envs.rapidock.python_bin",
                "subprocess_envs.rapidock.repo_path",
            ],
            "required by RAPiDock predocker",
        )
    if any(t.lower() in {"haddock3_nucleic", "haddock3-nucleic"} for t in predocker_types_used):
        _note_env(
            ["subprocess_envs.haddock3.haddock3_bin"],
            "required by HADDOCK3 nucleic-acid predocker",
        )
    if any(t.lower() == "symmetry_canonicalizer" for t in predocker_types_used):
        _note_env(
            [
                "subprocess_envs.rpxdock.python_bin",
                "subprocess_envs.rpxdock.repo_path",
                "subprocess_envs.rpxdock.willutil_repo_path",
            ],
            "required by symmetry_canonicalizer rpxdock backend",
        )
    if "tmscore" in loss_types:
        _note_env(["usalign_path"], "required by TM-score style loss terms")
    if "min_dssp_fractions" in loss_types:
        _note_env(["dssp_path"], "required by DSSP-based loss terms")
    if loss_types & {"ananas_rmsd_raw", "ananas_rmsd_norm"}:
        _note_env(["ananas_path"], "required by AnAnaS symmetry RMSD loss terms")

    return {
        "enable_pyrosetta": enable_pyrosetta,
        "evaluator_types": sorted([t for t in evaluator_types_used if t]),
        "mutator_types": sorted([t for t in mutator_types_used if t]),
        "predocker_types": sorted([t for t in predocker_types_used if t]),
        "loss_types": sorted([t for t in loss_types if t]),
        "env_key_reasons": env_key_reasons,
        "required_env_keys": sorted(req_env_keys),
        "required_env_repo_files": [
            {
                "base_env_key": base_env_key,
                "path": path,
                "label": label,
            }
            for base_env_key, path, label in sorted(req_env_repo_files)
        ],
        "required_repo_files": sorted(req_repo_files),
        "required_imports": sorted(req_imports),
    }
