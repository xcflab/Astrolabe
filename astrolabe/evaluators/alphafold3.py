from __future__ import annotations

import dataclasses
import itertools
import json
import os
from pathlib import Path
import logging
import shutil
import time
from collections import defaultdict
from collections.abc import Mapping
from typing import Sequence

import jax
import numpy as np

# from alphafold.model import features
from alphafold3.common.folding_input import ProteinChain
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from astrolabe.core.cyclization import serialize_cyclic_relative_patches
from astrolabe.utils.mmcif_ids import int_id_to_str_id
from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import StandardScores
from astrolabe.evaluators.wrappers.alphafold3 import (
    BUCKETS,
    ResultsForSeed,
    get_af3_scores,
    write_outputs,
)
from astrolabe.utils.af3_compat import (
    allow_all_x_protein_chains_during_featurisation,
)
from astrolabe.utils.extra_residues import (
    resolve_ligand_resname_for_protomer,
)

from astrolabe.pre_eval_hooks.af3_antibody_msa_cache import (
    apply_af3_antibody_msa_cache_hook,
    af3_input_to_json_payload,
)


class AlphaFold3Evaluator(Evaluator):
    def __init__(self, model_runner, output_dir, af3_conf=None):
        self.model_runner = model_runner
        self.af3_conf = af3_conf
        self._pre_eval_hook_reports = {}
        self._af3_antibody_msa_cache = None
        self._bucket_pad_tolerance = 50
        self._default_buckets: Sequence[int] = BUCKETS
        if af3_conf is not None:
            try:
                self._bucket_pad_tolerance = int(
                    getattr(af3_conf, "bucket_pad_tolerance", 50) or 50
                )
            except Exception:
                self._bucket_pad_tolerance = 50
            try:
                override_buckets = getattr(af3_conf, "buckets", None)
            except Exception:
                override_buckets = None
            if override_buckets:
                buckets = tuple(int(x) for x in override_buckets)
                if any(b <= 0 for b in buckets):
                    logging.warning(
                        "Ignoring invalid AF3 buckets override (non-positive): %s",
                        buckets,
                    )
                elif any(prev >= curr for prev, curr in zip(buckets, buckets[1:])):
                    logging.warning(
                        "Ignoring invalid AF3 buckets override (not strictly increasing): %s",
                        buckets,
                    )
                else:
                    self._default_buckets = buckets
        self.buckets = {}
        self.output_dir = output_dir
        self.save_all_samples = (
            bool(getattr(af3_conf, "save_all_samples", False))
            if af3_conf is not None
            else False
        )
        self.kwargs = defaultdict(dict)
        self.query_residue_index_jumps = self._normalise_query_residue_index_jumps(
            getattr(af3_conf, "query_residue_index_jumps", None)
            if af3_conf is not None
            else None
        )

    @staticmethod
    def _get_config_value(config, key, default=None):
        if config is None:
            return default
        if isinstance(config, Mapping):
            return config.get(key, default)
        try:
            return getattr(config, key)
        except (AttributeError, RuntimeError):
            return default

    def _apply_pre_eval_hooks(self, inputs):
        hook_cfg = None
        root_hook_cfg = self._get_config_value(
            getattr(self, "pre_eval_hooks_config", None),
            "af3_antibody_msa_cache",
        )
        if root_hook_cfg is None:
            global_hooks = self._get_config_value(global_config, "pre_eval_hooks")
            root_hook_cfg = self._get_config_value(
                global_hooks,
                "af3_antibody_msa_cache",
            )
        hook_cfg = root_hook_cfg

        af3_hook_cfg = self._get_config_value(self.af3_conf, "pre_eval_hooks")
        if af3_hook_cfg is not None:
            hook_cfg = self._get_config_value(
                af3_hook_cfg,
                "af3_antibody_msa_cache",
                hook_cfg,
            )
        if hook_cfg is None:
            self._pre_eval_hook_reports = {}
            return inputs

        original_inputs = inputs
        updated_inputs, reports, cache = apply_af3_antibody_msa_cache_hook(
            original_inputs,
            hook_cfg,
            output_dir=self.output_dir,
            cache=self._af3_antibody_msa_cache,
            semantic_protomers=self._semantic_protomer_expectations(),
        )
        if isinstance(original_inputs, dict) and updated_inputs is not original_inputs:
            # Preserve the existing evaluator contract: prepare_input mutates the
            # caller's mapping so prepare_output receives the same prepared inputs.
            original_inputs.clear()
            original_inputs.update(updated_inputs)
            inputs = original_inputs
        else:
            inputs = updated_inputs
        self._pre_eval_hook_reports = reports
        self._af3_antibody_msa_cache = cache
        dump_paths = self._dump_final_af3_input_jsons(
            inputs,
            hook_cfg,
            self._pre_eval_hook_reports,
        )
        for oligo_name, paths in dump_paths.items():
            self._pre_eval_hook_reports.setdefault(oligo_name, []).append(
                {
                    "action": "dump_final_input_json",
                    "path": str(paths["input_json"]),
                    "hook_report_path": str(paths["hook_report"]),
                }
            )
        return inputs

    @staticmethod
    def _semantic_protomer_expectations():
        return global_config.protomers

    @staticmethod
    def _semantic_state_report(reports, oligo_name):
        for report in reports.get(str(oligo_name), []) or []:
            if report.get("action") == "semantic_state_validation":
                return dict(report)
        raise RuntimeError(
            f"No semantic AF3 antibody MSA hook report found for oligo={oligo_name}"
        )

    def _dump_final_af3_input_jsons(self, inputs, hook_cfg, reports):
        if not bool(self._get_config_value(hook_cfg, "dump_final_input_json", False)):
            return {}
        raw_dir = self._get_config_value(hook_cfg, "final_input_json_dir")
        if raw_dir in (None, "", "null", "None"):
            output_dir = self.output_dir / "final_input_json"
        else:
            output_dir = Path(
                os.path.expandvars(os.path.expanduser(str(raw_dir)))
            ).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        step = int(self._get_config_value(global_config, "step", 0))
        paths = {}
        for oligo_name, input_obj in inputs.items():
            path = output_dir / f"step_{step:03d}_{oligo_name}_input.json"
            report_path = output_dir / f"step_{step:03d}_{oligo_name}_hook_report.json"
            payload = af3_input_to_json_payload(
                input_obj,
                name=f"step_{step:03d}_{oligo_name}",
            )
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary, path)
            report_payload = self._semantic_state_report(reports, oligo_name)
            report_payload["step"] = step
            report_payload["final_input_json_path"] = str(path)
            report_temporary = report_path.with_suffix(".json.tmp")
            report_temporary.write_text(
                json.dumps(report_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(report_temporary, report_path)
            paths[str(oligo_name)] = {
                "input_json": path,
                "hook_report": report_path,
            }
        return paths

    @staticmethod
    def _normalise_query_residue_index_jumps(raw) -> tuple[dict[str, int], ...]:
        if raw is None:
            return ()
        if isinstance(raw, Mapping):
            raw = [raw]
        specs = []
        for item in raw or []:
            if not isinstance(item, Mapping):
                raise TypeError("query_residue_index_jumps entries must be mappings")
            after = item.get("after") or item.get("jump_after") or item.get("break_after")
            offset = item.get("offset") or item.get("jump") or item.get("gap")
            if after is None or offset is None:
                raise ValueError("query_residue_index_jumps entries require after and offset")
            specs.append({"after": int(after), "offset": int(offset)})
        return tuple(specs)

    @staticmethod
    def _structure_start_soft_select_enabled() -> bool:
        try:
            state = global_config.structure_start
        except RuntimeError:
            return False
        config = getattr(state, "config", None)
        return bool(config is not None and getattr(config, "mode", None) == "soft_select")

    def _should_use_default_buckets(self, num_tokens: int | None) -> bool:
        """Whether to pad to a nearby default bucket size (vs exact token length)."""
        if num_tokens is None:
            return True
        next_bucket = None
        for b in self._default_buckets:
            if b >= num_tokens:
                next_bucket = b
                break
        if next_bucket is None:
            return False
        return (next_bucket - num_tokens) <= self._bucket_pad_tolerance

    def run_function(self, input_feat):
        output = {}
        for oligo, feat in input_feat.items():
            if getattr(self, "resident", None) is not None:
                from astrolabe.constraints import Constraint, ConstraintKind, validate_constraints
                declared = self._get_config_value(self.af3_conf, "daemon_constraints", ()) or ()
                constraints = [Constraint(ConstraintKind(item["kind"]), item.get("payload"), item["mechanism"]) for item in declared]
                for mechanism in ("cyclic_asym_ids", "cyclic_relative_patches", "query_residue_index_jumps"):
                    if self.kwargs[oligo].get(mechanism):
                        constraints.append(Constraint(ConstraintKind.TOPOLOGY, None, mechanism))
                allowed = {"af3": {(ConstraintKind.TOPOLOGY, "cyclic_asym_ids"),
                                   (ConstraintKind.TOPOLOGY, "cyclic_relative_patches"),
                                   (ConstraintKind.TOPOLOGY, "query_residue_index_jumps")}}
                validate_constraints("af3", constraints, allowed)
                prepared = None
                cache_enabled = bool(self._get_config_value(self.af3_conf, "feature_cache", False))
                if cache_enabled:
                    from copy import deepcopy
                    from astrolabe.feature_cache import feature_key
                    from astrolabe.evaluators.worker_identity import digest_path
                    import alphafold3
                    raw = json.loads(feat.to_json())
                    raw.pop("name", None)
                    options = dict(self.kwargs[oligo])
                    if "init_guess_path" in options:
                        options["init_guess_path"] = digest_path(options["init_guess_path"])
                    dependencies = {"schema": 1, "source": digest_path(Path(alphafold3.__file__).parent),
                        "chains": raw.get("sequences"), "chain_order": [str(c.id) for c in feat.chains],
                        "topology": raw.get("bondedAtomPairs"), "constraints": options,
                        "seed": list(feat.rng_seeds), "msa": raw.get("sequences"),
                        "templates": raw.get("sequences"), "ccd": [raw.get("userCCD"), digest_path(chemical_components._CCD_PICKLE_FILE)],
                        "options": {"buckets": self.buckets.get(oligo)}}
                    key = feature_key(dependencies)
                    cached = getattr(self, "_last_feature_cache", None)
                    if cached and cached[0] == key:
                        prepared = deepcopy(cached[1])
                if prepared is None:
                    prepared = self.prepare_features(feat, self.buckets.get(oligo), **self.kwargs[oligo])
                    if cache_enabled:
                        self._last_feature_cache = (key, deepcopy(prepared))
                examples, cyclic, patches = prepared
                results = self.resident.forward(feat.rng_seeds, examples, cyclic, patches)
                from astrolabe.evaluators.wrappers.alphafold3 import ModelRunner
                class Extractor:
                    def extract_structures(self, **kwargs):
                        return ModelRunner.extract_structures(None, **kwargs)
                output[oligo] = self.finalize_features(Extractor(), feat, examples, results)
                continue
            output[oligo] = self.predict_structure(
                fold_input=feat,
                model_runner=self.model_runner,
                buckets=self.buckets.get(oligo),
                **self.kwargs[oligo],
            )
        return output

    def prepare_input(self, inputs, runtime_config=None):
        self.kwargs = defaultdict(dict)
        inputs = self._apply_pre_eval_hooks(inputs)
        # Convert lightweight folding inputs to AF3 folding_input.Input for featurisation.
        # Mutates the `inputs` mapping so `prepare_output` sees AF3 chain types too.
        from astrolabe.utils.af3_compat import to_af3_input

        # Use per-step token counts derived from the *current* inputs (post-mutation),
        # not the one-time `global_config.num_tokens` snapshot.
        num_tokens_by_oligo = {}
        for oligo, v in inputs.items():
            token_count = 0
            try:
                chains = getattr(v, "chains", None) or []
                for chain in chains:
                    seq = getattr(chain, "sequence", None)
                    if isinstance(seq, str):
                        token_count += len(seq)
            except Exception:
                token_count = 0
            baseline = global_config.num_tokens.get(oligo)
            if baseline is not None:
                # AF3 tokenization includes non-polymer chains (e.g. ligands). Sequence
                # lengths alone can undercount tokens when small molecules are present,
                # which in turn can disable bucket sharing and cause unnecessary recompiles.
                try:
                    token_count = max(int(token_count), int(baseline))
                except Exception:
                    token_count = int(baseline)
            elif token_count <= 0:
                token_count = baseline
            num_tokens_by_oligo[oligo] = token_count

        for k, v in list(inputs.items()):
            cyclic_chain_ids = {
                str(chain_id)
                for chain_id in (getattr(v, "cyclic_chain_ids", None) or ())
            }
            if cyclic_chain_ids:
                self.kwargs[k]["cyclic_asym_ids"] = tuple(
                    chain_idx
                    for chain_idx, chain in enumerate(
                        getattr(v, "chains", ()) or (), start=1
                    )
                    if str(getattr(chain, "id", "")) in cyclic_chain_ids
                )
            cyclic_relative_patches = tuple(
                serialize_cyclic_relative_patches(
                    getattr(v, "cyclic_relative_patches", None),
                    chain_ids=[str(getattr(chain, "id", "")) for chain in getattr(v, "chains", ()) or ()],
                )
            )
            if cyclic_relative_patches:
                self.kwargs[k]["cyclic_relative_patches"] = cyclic_relative_patches
            if self.query_residue_index_jumps:
                self.kwargs[k]["query_residue_index_jumps"] = self.query_residue_index_jumps
            inputs[k] = to_af3_input(v)
        buckets_by_oligo = {}
        for oligo in inputs.keys():
            num_tokens = num_tokens_by_oligo.get(oligo)
            use_defaults = self._should_use_default_buckets(num_tokens)
            buckets_by_oligo[oligo] = self._default_buckets if use_defaults else None
            logging.debug(
                "AF3 buckets for %s: num_tokens=%s use_defaults=%s pad_tolerance=%s",
                oligo,
                num_tokens,
                use_defaults,
                self._bucket_pad_tolerance,
            )
        self.buckets = buckets_by_oligo
        return inputs

    def prepare_output(self, outputs, inputs, runtime_config):

        per_proto_plddt = defaultdict(list)
        step = global_config.step
        structure_path = global_config.out_dir / "structures"
        structure_path.mkdir(parents=True, exist_ok=True)

        complex_dump_paths = {}
        soft_select_candidates = self._structure_start_soft_select_enabled()
        candidate_payloads: dict[tuple[int, int], dict[str, dict]] = defaultdict(
            lambda: {
                "plddt": {},
                "ptm": {},
                "pae": {},
                "pde": {},
                "iptm": {},
                "per_proto_scores": defaultdict(list),
                "chain_pair_scores": {},
                "complex_paths": {},
            }
        )
        candidate_keys_by_oligo: dict[str, set[tuple[int, int]]] = {}
        scores_by_oligo = {}

        def write_model_with_ligand_names(source_path, dest_path, lig_name_mapping):
            with open(source_path, "r") as f:
                lines = f.read()
            for source_name, resolved_name in lig_name_mapping.items():
                lines = lines.replace(source_name, resolved_name)
            with open(dest_path, "w") as f:
                f.write(lines)
                f.write('_dummy.title ""\n')

        def ligand_resname_mapping_for_oligo(oligo_name):
            mapping = {}
            for i, proto in enumerate(oligo_name):
                protomer = global_config.protomers[proto]
                if protomer.type not in {ProtomerTypes.SMILES, ProtomerTypes.CCD}:
                    continue

                cid = int_id_to_str_id(i + 1)
                protomer_name = str(getattr(protomer, "name", proto))
                ccd_code = None
                if protomer.type == ProtomerTypes.CCD:
                    ccd_values = [
                        str(x).strip().upper()
                        for x in str(getattr(protomer, "sequence", "")).split("+")
                        if str(x).strip()
                    ]
                    if len(ccd_values) == 1:
                        ccd_code = ccd_values[0]

                resolved_name = resolve_ligand_resname_for_protomer(
                    protomer_key=proto,
                    protomer_name=protomer_name,
                    ccd_code=ccd_code,
                )
                mapping[f"LIG_{cid}"] = resolved_name
                if ccd_code:
                    mapping[ccd_code] = resolved_name
            return mapping

        for oligo_name, input in inputs.items():
            # raw model output
            jobname = f"step_{step:03d}_{oligo_name}"
            lig_name_mapping = ligand_resname_mapping_for_oligo(oligo_name)

            write_outputs(
                all_inference_results=outputs[oligo_name],
                output_dir=self.output_dir / jobname,
                job_name=jobname,
                save_all_samples=self.save_all_samples,
            )
            write_model_with_ligand_names(
                self.output_dir / jobname / f"{jobname}_model.cif",
                structure_path / f"{jobname}_model.cif",
                lig_name_mapping,
            )
            complex_dump_paths[oligo_name] = structure_path / f"{jobname}_model.cif"

            scores = get_af3_scores(outputs[oligo_name])
            scores_by_oligo[oligo_name] = scores
            plddts = np.array(scores["per_res_scores"]["per_res_plddt"])
            plddt_per_res = plddts.mean(axis=0) / 100.0
            res_idx = 0
            for chain_id, chain in zip(input.name, input.chains):
                if isinstance(chain, ProteinChain):
                    per_proto_plddt[chain_id].append(
                        plddt_per_res[res_idx : res_idx + len(chain.sequence)]
                    )
                    res_idx += len(chain.sequence)

            if not soft_select_candidates:
                continue
            candidate_keys_by_oligo[oligo_name] = set()
            candidate_dir = self.output_dir / jobname / "structure_start_candidates"
            for results_for_seed in outputs[oligo_name]:
                seed = int(results_for_seed.seed)
                for sample_idx, inference_result in enumerate(results_for_seed.inference_results):
                    key = (seed, int(sample_idx))
                    candidate_jobname = (
                        f"{jobname}_candidate_seed{seed}_sample{sample_idx:03d}"
                    )
                    single_result = ResultsForSeed(
                        seed=seed,
                        inference_results=[inference_result],
                        full_fold_input=results_for_seed.full_fold_input,
                        embeddings=results_for_seed.embeddings,
                    )
                    candidate_output_dir = candidate_dir / f"seed-{seed}_sample-{sample_idx}"
                    write_outputs(
                        all_inference_results=[single_result],
                        output_dir=candidate_output_dir,
                        job_name=candidate_jobname,
                    )
                    candidate_model = candidate_output_dir / f"{candidate_jobname}_model.cif"
                    if not candidate_model.exists():
                        continue
                    candidate_path = structure_path / f"{candidate_jobname}_model.cif"
                    write_model_with_ligand_names(
                        candidate_model,
                        candidate_path,
                        lig_name_mapping,
                    )
                    candidate_scores = get_af3_scores([single_result])
                    payload = candidate_payloads[key]
                    payload["complex_paths"][oligo_name] = candidate_path
                    payload["plddt"][oligo_name] = [
                        np.array(candidate_scores["per_res_scores"]["per_res_plddt"][0])
                        / 100.0
                    ]
                    payload["ptm"][oligo_name] = candidate_scores["per_str_scores"]["ptm"]
                    payload["iptm"][oligo_name] = candidate_scores["per_str_scores"]["iptm"]
                    payload["pde"][oligo_name] = candidate_scores["per_str_scores"][
                        "predicted_distance_error"
                    ]
                    payload["pae"][oligo_name] = candidate_scores["inter_o_cross_scores"][
                        "pae_xchain"
                    ]
                    payload["chain_pair_scores"][oligo_name] = candidate_scores[
                        "chain_pair_scores"
                    ]
                    plddt_per_res = payload["plddt"][oligo_name][0]
                    res_idx = 0
                    for chain_id, chain in zip(input.name, input.chains):
                        if isinstance(chain, ProteinChain):
                            payload["per_proto_scores"][chain_id].append(
                                plddt_per_res[res_idx : res_idx + len(chain.sequence)]
                            )
                            res_idx += len(chain.sequence)
                    candidate_keys_by_oligo[oligo_name].add(key)
        plddt = {}
        ptm = {}
        pae = {}
        pde = {}
        iptm = {}
        chain_pair_scores = {}
        for oligo_name in inputs.keys():
            scores = scores_by_oligo[oligo_name]
            plddt[oligo_name] = [
                np.array(score) / 100
                for score in scores["per_res_scores"]["per_res_plddt"]
            ]
            ptm[oligo_name] = scores["per_str_scores"]["ptm"]
            iptm[oligo_name] = scores["per_str_scores"]["iptm"]
            pde[oligo_name] = scores["per_str_scores"]["predicted_distance_error"]
            pae[oligo_name] = scores["inter_o_cross_scores"]["pae_xchain"]
            chain_pair_scores[oligo_name] = scores["chain_pair_scores"]

        scores = StandardScores(
            names=list(inputs.keys()),
            plddt=plddt,
            ptm=ptm,
            pae=pae,
            iptm=iptm,
            per_proto_scores=per_proto_plddt,
            pde=pde,
            chain_pair_scores=chain_pair_scores,
            complex_paths=complex_dump_paths,
        )
        if self._pre_eval_hook_reports:
            scores.pre_eval_hook_reports = self._pre_eval_hook_reports
        if soft_select_candidates and candidate_keys_by_oligo:
            common_keys = set.intersection(*candidate_keys_by_oligo.values())
            candidates = []
            for key in sorted(common_keys):
                payload = candidate_payloads[key]
                candidates.append(
                    StandardScores(
                        names=list(inputs.keys()),
                        plddt=payload["plddt"],
                        ptm=payload["ptm"],
                        pae=payload["pae"],
                        iptm=payload["iptm"],
                        per_proto_scores=dict(payload["per_proto_scores"]),
                        pde=payload["pde"],
                        chain_pair_scores=payload["chain_pair_scores"],
                        complex_paths=payload["complex_paths"],
                    )
                )
            if candidates:
                scores.candidate_scores = candidates
        return scores

    @staticmethod
    def _apply_query_residue_index_jumps(featurised_examples, fold_input, specs) -> list:
        if not specs:
            return featurised_examples
        protein_lengths = [
            len(chain.sequence) for chain in fold_input.chains if isinstance(chain, ProteinChain)
        ]
        if not protein_lengths:
            raise ValueError("query_residue_index_jumps require at least one protein chain")
        starts = np.cumsum([0] + protein_lengths[:-1])
        total_len = sum(protein_lengths)
        patched = []
        for example in featurised_examples:
            if isinstance(example, dict):
                residue_index = np.asarray(example["residue_index"])
                seq_mask = np.asarray(example["seq_mask"]).astype(bool)
                is_protein = np.asarray(example["is_protein"]).astype(bool)
            else:
                token_features = example.token_features
                residue_index = np.asarray(token_features.residue_index)
                seq_mask = np.asarray(token_features.mask).astype(bool)
                is_protein = np.asarray(token_features.is_protein).astype(bool)
            protein_positions = np.where(seq_mask & is_protein)[0]
            if protein_positions.size < total_len:
                raise ValueError(
                    "Cannot apply AF3 query_residue_index_jumps: protein token count "
                    f"{protein_positions.size} < expected {total_len}"
                )
            updated = residue_index.copy()
            protein_positions = protein_positions[:total_len]
            for spec in specs:
                after = spec["after"]
                offset = spec["offset"]
                for start, length in zip(starts, protein_lengths):
                    if after <= 0 or after >= length:
                        raise ValueError(
                            f"query_residue_index_jumps after={after} is outside chain length {length}"
                        )
                    updated[protein_positions[start + after : start + length]] += offset
            if isinstance(example, dict):
                patched_example = dict(example)
                patched_example["residue_index"] = updated.astype(residue_index.dtype, copy=False)
                patched.append(patched_example)
            else:
                patched_token_features = dataclasses.replace(
                    token_features,
                    residue_index=updated.astype(residue_index.dtype, copy=False),
                )
                patched.append(dataclasses.replace(example, token_features=patched_token_features))
        logging.info("AF3 query residue_index jump applied: specs=%s", specs)
        return patched

    @staticmethod
    def prepare_features(fold_input, buckets, **kwargs):
        """Featurise on the caller CPU; preserve the original per-seed inputs."""
        kwargs = dict(kwargs)
        cyclic_asym_ids = tuple(int(x) for x in (kwargs.pop("cyclic_asym_ids", ()) or ()))
        cyclic_relative_patches = tuple(
            tuple(int(y) for y in x)
            for x in (kwargs.pop("cyclic_relative_patches", ()) or ())
        )
        ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
        if "init_guess_path" in kwargs:
            if "init_guess_chain" not in kwargs:
                raise ValueError("init_guess_chain not found in kwargs")
            kwargs["init_guess_positions"] = featurisation.generate_initial_guess(
                init_guess_path=kwargs["init_guess_path"], ccd=ccd
            )
        jumps = tuple(kwargs.pop("query_residue_index_jumps", ()) or ())
        with allow_all_x_protein_chains_during_featurisation(fold_input):
            examples = featurisation.featurise_input(
                fold_input=fold_input, buckets=buckets, ccd=ccd, verbose=True, **kwargs
            )
        examples = AlphaFold3Evaluator._apply_query_residue_index_jumps(
            examples, fold_input, jumps
        )
        if len(examples) != len(fold_input.rng_seeds):
            raise ValueError("AF3 featurisation returned a different number of examples than seeds")
        return examples, cyclic_asym_ids, cyclic_relative_patches

    @staticmethod
    def forward_features(model_runner, seeds, examples, cyclic_asym_ids=(), cyclic_relative_patches=()):
        """Run only inference, with the request's exact seeds and topology."""
        if len(seeds) != len(examples):
            raise ValueError("AF3 seeds and features differ in length")
        if hasattr(model_runner, "set_cyclic_asym_ids"):
            model_runner.set_cyclic_asym_ids(cyclic_asym_ids)
        if hasattr(model_runner, "set_cyclic_relative_patches"):
            model_runner.set_cyclic_relative_patches(cyclic_relative_patches)
        return [model_runner.run_inference(example, jax.random.PRNGKey(seed))
                for seed, example in zip(seeds, examples)]

    @staticmethod
    def finalize_features(model_runner, fold_input, examples, results):
        """Extract structures on the caller from CPU inference results."""
        if len(results) != len(examples) or len(results) != len(fold_input.rng_seeds):
            raise ValueError("AF3 result, feature and seed counts differ")
        return [ResultsForSeed(
            seed=seed,
            inference_results=model_runner.extract_structures(
                batch=example, result=result, target_name=fold_input.name
            ),
            full_fold_input=fold_input,
        ) for seed, example, result in zip(fold_input.rng_seeds, examples, results)]

    @staticmethod
    def predict_structure(fold_input, model_runner, buckets, **kwargs):
        """Run the same stages in-process for the legacy evaluator."""
        examples, cyclic_ids, patches = AlphaFold3Evaluator.prepare_features(
            fold_input, buckets, **kwargs
        )
        results = AlphaFold3Evaluator.forward_features(
            model_runner, fold_input.rng_seeds, examples, cyclic_ids, patches
        )
        return AlphaFold3Evaluator.finalize_features(model_runner, fold_input, examples, results)
