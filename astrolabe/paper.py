"""Manuscript workflow entry points; model imports remain execution-only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import tarfile

from astrolabe import cli_command_definitions as cli_defs

# One inventory supplies parser choices, completion, and the reader-facing list.
PROTOCOLS = {
    "protein-binder-af3": "Protein binder benchmark, AF3",
    "protein-binder-protenix": "Protein binder benchmark, Protenix",
    "protein-binder-sap": "SAP-aware protein binder design",
    "idr-binder": "Flexible IDR target design",
    "cre-binder-af3": "Split-domain Cre binder design, AF3",
    "cre-binder-protenix2": "Split-domain Cre binder design, Protenix v2",
    "dual-target-af3": "Shared minibinder in two target complexes, AF3",
    "dual-target-protenix2": "Shared minibinder, Protenix v2 optimization",
    "dual-target-esmfold2": "Shared minibinder, ESMFold2 optimization",
    "single-target-control": "Single-target minibinder control",
    "antibody-framework": "Fixed VHH/scFv framework and mutable CDRs",
    "cross-reactive-vhh": "Shared VHH against two targets",
    "single-target-vhh-control": "Single-target VHH control",
    "solublempnn-control": "Interface-preserving post-design SolubleMPNN control",
    "ligand-binder-af3": "Small-molecule binder design, AF3",
    "ligand-binder-protenix": "Small-molecule binder design, Protenix",
    "two-state-af3": "Protenix apo/holo optimization and holo-only AF3 refinement",
    "two-state-boltz2": "Protenix apo/holo optimization and holo-only Boltz-2 refinement",
    "pocket-iptm": "Pocket design: ipTM-only control",
    "pocket-oriented": "Pocket design: ipTM and target-pocket occupancy",
    "pocket-rescue": "Pocket design with conditional Vina rescue",
    "cid-iptm": "Chemically induced dimer: ipTM-only control",
    "cid-oriented": "Chemically induced dimer: interface occupancy",
    "cid-rescue": "Chemically induced dimer with conditional Vina rescue",
    "pseudo-c3": "Shared protein cores and independent interface patches",
    "tandem-repeat-3": "Three synchronized protein repeat units",
    "tandem-repeat-4": "Four synchronized protein repeat units",
    "tandem-repeat-5": "Five synchronized protein repeat units",
    "cyclic-peptide-bound": "Target-bound single-ring peptide design",
    "cyclic-peptide-representation": "Peptide-only linkage/encoding comparison",
    "cyclic-peptide-multiring": "Multiring peptide design, manuscript LigandMPNN recipe",
    "cyclic-peptide-multiring-archived": "Multiring peptide design, archived MCMC protocol",
    "rna-aptamer": "RNA aptamer design, 20 states",
    "dna-aptamer": "DNA aptamer design, 40 states; summarize first 20",
    "protein-binder-rna": "Protein binder against a fixed RNA target",
    "protein-binder-dna": "Protein binder against a fixed DNA target",
    "protein-binder-rna-haddock": "RNA-target binder with contact-triggered HADDOCK context",
    "protein-binder-dna-haddock": "DNA-target binder with contact-triggered HADDOCK context",
    "rna-oligomer-4": "Four identical 87-nt RNAs, NA-MPNN/AF3 beam search",
    "rna-oligomer-5": "Five identical 87-nt RNAs, NA-MPNN/AF3 beam search",
    "bivalent-rna": "Two shared 20-nt RNA repeats and two U1A proteins",
    "evaluate-af3": "Fixed-sequence AlphaFold 3 evaluation",
    "evaluate-protenix2": "Fixed-sequence Protenix v2 evaluation",
    "evaluate-esmfold2": "Fixed-sequence ESMFold2 evaluation",
    "evaluate-af2": "Fixed-sequence AlphaFold 2 evaluation",
}
for _symmetry in ("c3", "c4", "c5", "c6", "d2", "d3"):
    PROTOCOLS[f"symmetry-{_symmetry}"] = f"{_symmetry.upper()} assembly with RPXDock rescue"
    PROTOCOLS[f"symmetry-{_symmetry}-no-rescue"] = f"{_symmetry.upper()} matched control without rescue"
for _name in ("protein-binder-sap", "idr-binder", "dual-target-af3", "single-target-control",
              "pocket-iptm", "pocket-oriented", "pocket-rescue", "two-state-af3", "two-state-boltz2",
              "protein-binder-rna", "protein-binder-dna", "bivalent-rna",
              "tandem-repeat-3", "tandem-repeat-4", "tandem-repeat-5", "cyclic-peptide-bound",
              "protein-binder-af3", "protein-binder-protenix", "cross-reactive-vhh",
              "single-target-vhh-control"):
    PROTOCOLS[f"{_name}-archived"] = f"Archived run configuration: {PROTOCOLS[_name]}"

SCORE_ENDPOINTS = (
    "protein-binder", "ligand-binder", "dual-target", "vhh", "idr",
    "pocket", "cid", "symmetry", "cyclic-bound", "cyclic-multiring", "aptamer",
)
BUNDLES = ("all", "protein", "nucleic", "cyclic", "interventions")


def register_commands(subparsers) -> None:
    parser = subparsers.add_parser("paper", help="Manuscript design, evaluation, inputs and analysis.")
    sub = parser.add_subparsers(dest="paper_command", required=True)
    listing = sub.add_parser("list", help="List executable manuscript protocols.")
    cli_defs.add_output_format_argument(listing)
    cli_defs.set_handler(listing, _list, _cuda_preinitialize=False)

    show = sub.add_parser("show", help="Print a protocol, sources and execution settings without loading models.")
    show.add_argument("protocol", choices=tuple(PROTOCOLS))
    show.add_argument("--config-env", default=None, help="Hydra environment group, e.g. env or env_container.")
    cli_defs.set_handler(show, _show, _cuda_preinitialize=False)

    prepare = sub.add_parser("prepare-inputs", help="Extract bundled paper templates and chemical inputs.")
    prepare.add_argument("--bundle", choices=BUNDLES, default="all")
    prepare.add_argument("-o", "--out-dir", required=True, help="Destination for the paper input data root.")
    cli_defs.set_handler(prepare, _prepare_inputs, _cuda_preinitialize=False)

    run = sub.add_parser(
        "run", help="Execute a paper design, beam search or fixed-sequence evaluation.",
        description="Layer protocol < input YAML < named options < --overrides. Relative paths in the input YAML are resolved from its directory. --dry-run prints the configuration without model execution. One run is the default; manuscript cohort sizes are documented per protocol.",
    )
    run.add_argument("protocol", choices=tuple(PROTOCOLS))
    run.add_argument("--inputs", help="Input/overlay YAML path, not a built-in name.")
    run.add_argument("--data-dir", help="Data root produced by paper prepare-inputs; sets ASTROLABE_PAPER_DATA for this call.")
    run.add_argument("--config-env", default=None, help="Hydra environment group; normal site-local selection applies if omitted.")
    run.add_argument("-t", "--task-name", default=None)
    run.add_argument("-o", "--out-dir", default=None, help="Output root (default: outputs/paper/<protocol>).")
    run.add_argument("-n", "--num-runs", type=int, default=1, help="Independent trajectories or evaluation seeds (default: 1).")
    run.add_argument("--seed", type=int, default=None, help="Override the archived/configured initial seed.")
    run.add_argument("--target-seq", default=None, help="Fixed protein target B sequence.")
    run.add_argument("--target-homology", default=None, help="Target B AF3 JSON:chain or A3M input.")
    run.add_argument("--target-structure", default=None, help="Target B PDB/mmCIF; requires --target-chain.")
    run.add_argument("--target-chain", default=None)
    run.add_argument("--parent-structure", default=None, help="Parent complex for the interface-preserving SolubleMPNN control.")
    run.add_argument("--binder-chain", default="A", help="Binder chain in --parent-structure (default: A); target defaults to B.")
    run.add_argument("--ligand", default=None, help="Ligand CCD code or SMILES; uses ligand B, or C for CID.")
    run.add_argument("--ligand-type", choices=("ccd", "smiles"), default="ccd")
    run.add_argument("--framework", choices=("vhh_7eow", "scfv4d5_vhvl"), default=None)
    run.add_argument("--cdr-initializer", choices=("all-x", "random", "ser-tyr", "all-ala"), default=None)
    run.add_argument("--overrides", default=None, help="Quoted Hydra-style key=value overrides applied last.")
    run.add_argument("--overrides-file", default=None, help="Text file with one key=value override per line.")
    run.add_argument("--dump-config", default=None, help="Write the composed YAML to this path.")
    run.add_argument("--print-config", action="store_true", help="Print the composed YAML before execution.")
    run.add_argument("--dry-run", action="store_true", help="Compose/print only; no model loading or inference.")
    cli_defs.set_handler(run, _run)

    from astrolabe.paper_baselines import METHODS
    baseline = sub.add_parser("baseline", help="Run a source-grounded external comparator or study adapter.")
    baseline.add_argument("method", choices=METHODS)
    baseline.add_argument("--inputs", required=True, help="Launcher YAML naming the method's native inputs.")
    baseline.add_argument("--tool-dir", default=None, help="External source checkout (default: vendor/<method>).")
    baseline.add_argument("-o", "--out-dir", default=None)
    baseline.add_argument("--seed", type=int, default=1)
    baseline.add_argument("--dry-run", action="store_true", help="Print native commands, seed support and missing assets; no writes or subprocesses.")
    cli_defs.set_handler(baseline, _baseline, _cuda_preinitialize=False)

    collect = sub.add_parser("collect", help="Join native scores.csv files to a planned design-unit manifest.")
    collect.add_argument("--planned", required=True, help="CSV: unit_id,method,group,run_dir; every planned unit has a row.")
    collect.add_argument("--output", required=True, help="Output evaluation-record CSV.")
    cli_defs.set_handler(collect, _collect, _cuda_preinitialize=False)

    analyze = sub.add_parser("analyze", help="Calculate manuscript endpoints with planned denominators.")
    analyze_sub = analyze.add_subparsers(dest="analysis", required=True)
    scores = analyze_sub.add_parser("scores", help="Representative selection and cumulative/any-step endpoints.")
    scores.add_argument("--endpoint", choices=SCORE_ENDPOINTS, required=True)
    scores.add_argument("--planned", required=True)
    scores.add_argument("--records", required=True)
    scores.add_argument("--column", action="append", default=[], metavar="METRIC=COLUMN", help="Explicit canonical metric to input CSV column mapping; repeatable.")
    scores.add_argument("-o", "--out-dir", required=True)
    cli_defs.set_handler(scores, _analyze_scores, _cuda_preinitialize=False)
    ranking = analyze_sub.add_parser("ligand-rank", help="Seven-metric ligand ranks from fixed design representatives.")
    ranking.add_argument("--records", required=True, help="CSV with one fixed representative per design unit.")
    cli_defs.set_handler(ranking, _analyze_ligand_rank, _cuda_preinitialize=False)
    sensitivity = analyze_sub.add_parser("sensitivity", help="Target-balanced threshold curves and design-unit bootstrap intervals.")
    sensitivity.add_argument("--endpoint", choices=("protein-binder", "ligand-binder"), required=True)
    sensitivity.add_argument("--planned", required=True)
    sensitivity.add_argument("--records", required=True)
    sensitivity.add_argument("--denominator", choices=("planned", "available"), default="planned",
                             help="Keep planned units (default), or use the explicitly available-unit sensitivity cohort.")
    sensitivity.add_argument("--comparison-methods", nargs=2, default=("Astrolabe-AF3", "ProteinHunter"),
                             metavar=("METHOD_A", "METHOD_B"), help="Exact CSV method labels for the ligand rate difference A minus B.")
    sensitivity.add_argument("--bootstrap-samples", type=int, default=20000)
    sensitivity.add_argument("--seed", type=int, default=0)
    cli_defs.set_handler(sensitivity, _analyze_sensitivity, _cuda_preinitialize=False)
    ddg = analyze_sub.add_parser("ddg", help="Signed target-2 minus target-1 ddG and median bootstrap intervals.")
    ddg.add_argument("--records", required=True, help="CSV of fixed paired design representatives, with condition and design class.")
    ddg.add_argument("--bootstrap-samples", type=int, default=5000)
    ddg.add_argument("--seed", type=int, default=0)
    cli_defs.set_handler(ddg, _analyze_ddg, _cuda_preinitialize=False)
    agreement = analyze_sub.add_parser("agreement", help="Matched predictor binding modes, confidence and design-unit uncertainty.")
    agreement.add_argument("--manifest", required=True, help="CSV: one row per design and target side, with structures, sequences and confidence.")
    agreement.add_argument("--confidence-column", action="append", default=[], metavar="PREDICTOR=COLUMN")
    agreement.add_argument("--astrolabe-method", default="Astrolabe", help="Exact method label for distributional win probabilities.")
    agreement.add_argument("--bootstrap-samples", type=int, default=2000)
    agreement.add_argument("--seed", type=int, default=20260930)
    cli_defs.set_handler(agreement, _analyze_agreement, _cuda_preinitialize=False)
    for analysis_parser in (ranking, sensitivity, ddg, agreement):
        analysis_parser.add_argument("--column", action="append", default=[], metavar="METRIC=COLUMN",
                                     help="Explicit canonical field to input CSV column mapping; repeatable.")
        analysis_parser.add_argument("-o", "--out-dir", required=True)
    twostate = analyze_sub.add_parser("two-state", help="5-seed x 5-sample apo/holo ensemble response analysis.")
    twostate.add_argument("--manifest", required=True, help="Per-sample structure/confidence manifest CSV.")
    twostate.add_argument("-o", "--out-dir", required=True)
    cli_defs.set_handler(twostate, _analyze_twostate, _cuda_preinitialize=False)
    cre = analyze_sub.add_parser("cre", help="Reference-fitted all-atom Cre domain-centroid distance.")
    cre.add_argument("--reference", required=True)
    cre.add_argument("--structure", required=True)
    cre.add_argument("--reference-chain", default="B")
    cre.add_argument("--target-chain", default="B")
    cli_defs.set_handler(cre, _analyze_cre, _cuda_preinitialize=False)
    rna = analyze_sub.add_parser("rna", help="RNA cyclic fit/connectivity or bivalent contacts/clashes.")
    rna.add_argument("--kind", choices=("oligomer", "bivalent"), required=True)
    rna.add_argument("--structure", required=True)
    cli_defs.set_handler(rna, _analyze_rna, _cuda_preinitialize=False)


def _list(ns):
    rows = [{"protocol": name, "description": title, "config": f"paper/{name.replace('-', '_')}"}
            for name, title in PROTOCOLS.items()]
    if ns.format == "json":
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(f"{row['protocol']:<38} {row['description']}")
        print("\nInspect: astrolabe paper show <protocol>\nRun:     astrolabe paper run <protocol> --inputs <input.yaml> --out-dir <outputs>")
    return 0


def _compose(ns):
    from astrolabe import cli
    config_dir = Path(ns.config_dir).resolve() if ns.config_dir else cli._default_config_dir()
    overrides = [f"env={ns.config_env}"] if ns.config_env else []
    return cli._compose_hydra_config(config_dir=config_dir,
                                    config_name=f"paper/{ns.protocol.replace('-', '_')}", overrides=overrides)


def _show(ns):
    from omegaconf import OmegaConf
    print(OmegaConf.to_yaml(_compose(ns), resolve=False))
    return 0


@contextmanager
def _input_context(directory: Path, data_dir: str | None):
    previous = Path.cwd()
    old_data = os.environ.get("ASTROLABE_PAPER_DATA")
    if data_dir:
        os.environ["ASTROLABE_PAPER_DATA"] = str(Path(data_dir).expanduser().resolve())
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)
        if old_data is None:
            os.environ.pop("ASTROLABE_PAPER_DATA", None)
        else:
            os.environ["ASTROLABE_PAPER_DATA"] = old_data


def _framework(cfg, ns):
    from omegaconf import OmegaConf, open_dict
    name = ns.framework or OmegaConf.select(cfg, "paper_protocol.framework")
    if not name:
        return
    from astrolabe.antibody_framework_presets import get_antibody_framework_preset, build_framework_homology_config
    with open_dict(cfg):
        if OmegaConf.is_missing(cfg.seqs, "A") or ns.framework:
            cfg.seqs.A = get_antibody_framework_preset(name).binder_seq()
        if ns.framework or not cfg.homologies.get("A"):
            cfg.homologies.A = build_framework_homology_config(framework=name)
        if ns.framework and OmegaConf.select(cfg, "pre_eval_hooks.af3_antibody_msa_cache") is not None:
            cfg.pre_eval_hooks.af3_antibody_msa_cache.af3_turboab_db_subset = "VHH" if name == "vhh_7eow" else "IgG"
        initialization = ns.cdr_initializer or OmegaConf.select(cfg, "paper_protocol.cdr_initializer")
        if initialization == "all-x":
            cfg.initializer.mode = "x_mix"
            cfg.initializer.x_fraction = 1.0
        elif initialization == "ser-tyr":
            cfg.initializer.mode = "aa_set"
            cfg.initializer.aa_set = "SY"
        elif initialization == "all-ala":
            cfg.initializer.mode = "all_a"
        elif initialization == "random":
            cfg.initializer.mode = "random"


def _prepare_parent_redesign(cfg, path, binder_chain="A", target_chain="B"):
    """Fix parent-interface residues using strict heavy-atom distance <4 Å."""
    import gzip
    import numpy as np
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.Polypeptide import is_aa
    from Bio.SeqUtils import seq1
    from omegaconf import open_dict
    path = Path(path).expanduser().resolve()
    parser = MMCIFParser(QUIET=True) if ".cif" in path.suffixes else PDBParser(QUIET=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        model = parser.get_structure("parent", handle)[0]
    binder = [res for res in model[binder_chain] if is_aa(res, standard=False)]
    target = [res for res in model[target_chain] if is_aa(res, standard=False)]
    if not binder or not target:
        raise ValueError("Parent redesign requires two nonempty protein chains")
    def heavy(res):
        return [atom.coord for atom in res if atom.element not in {"H", "D"}]
    target_atoms = np.asarray([xyz for residue in target for xyz in heavy(residue)])
    fixed = []
    statement = []
    current = ""
    previous_fixed = None
    for index, residue in enumerate(binder, 1):
        xyz = np.asarray(heavy(residue))
        is_interface = bool(np.any(np.sum((xyz[:, None] - target_atoms[None, :]) ** 2, axis=-1) < 16.0))
        letter = seq1(residue.resname, custom_map={"MSE": "M"})
        if previous_fixed is not None and previous_fixed != is_interface:
            statement.append(("+" if previous_fixed else "") + current)
            current = ""
        current += letter
        previous_fixed = is_interface
        if is_interface:
            fixed.append(index)
    statement.append(("+" if previous_fixed else "") + current)
    with open_dict(cfg):
        cfg.seqs.A = " ".join(statement)
        cfg.seqs.B = "+" + "".join(seq1(res.resname, custom_map={"MSE": "M"}) for res in target)
        cfg.initializer.structure.source = str(path)
        cfg.initializer.structure.chain_map = {"A": binder_chain, "B": target_chain}
        cfg.paper_protocol.fixed_parent_interface_positions = fixed
        cfg.paper_protocol.parent_interface_cutoff_angstrom = 4.0


def _run(ns):
    from omegaconf import OmegaConf, open_dict
    from astrolabe import cli
    if ns.num_runs < 1:
        raise ValueError("--num-runs must be positive")
    cfg = _compose(ns)
    invocation_dir = Path.cwd()
    input_path = Path(ns.inputs).expanduser().resolve() if ns.inputs else None
    output = Path(ns.out_dir or f"outputs/paper/{ns.protocol}").expanduser().resolve()
    dump = Path(ns.dump_config).expanduser().resolve() if ns.dump_config else None
    overrides = cli._read_overrides_file(ns.overrides_file) + cli._split_overrides_string(ns.overrides)
    if input_path:
        overlay = OmegaConf.load(input_path)
        if "defaults" in overlay:
            del overlay["defaults"]
        with open_dict(cfg):
            cfg = OmegaConf.merge(cfg, overlay)
    with _input_context(input_path.parent if input_path else invocation_dir, ns.data_dir):
        with open_dict(cfg):
            cfg.task_name = ns.task_name or ns.protocol.replace("-", "_")
            cfg.out_dir = str(output)
            cfg.repeats = ns.num_runs
            if ns.seed is not None:
                cfg.seed = ns.seed
            if ns.parent_structure:
                if not OmegaConf.select(cfg, "paper_protocol.parent_interface_redesign", default=False):
                    raise ValueError("--parent-structure applies to solublempnn-control")
                _prepare_parent_redesign(cfg, ns.parent_structure, ns.binder_chain, ns.target_chain or "B")
            elif any((ns.target_seq, ns.target_homology, ns.target_structure, ns.target_chain)):
                seq, homology = cli._resolve_target_inputs(
                    target_seq=ns.target_seq, target_homology=ns.target_homology,
                    target_structure=ns.target_structure, target_chain=ns.target_chain)
                cfg.seqs.B = seq
                cfg.homologies.B = homology
            if ns.ligand:
                chain = "C" if ns.protocol.startswith("cid-") else "B"
                cfg.seqs[chain] = ns.ligand
                if cfg.get("types") is None:
                    cfg.types = {}
                cfg.types[chain] = "c" if ns.ligand_type == "ccd" else "s"
        _framework(cfg, ns)
        if overrides:
            with open_dict(cfg):
                cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([item.lstrip("+") for item in overrides]))
        if ns.print_config or ns.dry_run:
            print(OmegaConf.to_yaml(cfg, resolve=False))
        if dump:
            dump.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(cfg, dump)
        if ns.dry_run:
            return 0
        required = list(OmegaConf.select(cfg, "paper_protocol.required_input_paths", default=[]))
        required += [f"seqs.{key}" for key in cfg.seqs]
        for key in required:
            if OmegaConf.select(cfg, key, throw_on_missing=True) is None:
                raise ValueError(f"Protocol {ns.protocol} requires {key} in --inputs or --overrides")
        if ns.protocol.startswith("evaluate-"):
            # Evaluation must not silently turn an unspecified length into a new design.
            import re
            for value in cfg.seqs.values():
                if re.search(r"(?:^|\s)\+?\d+(?:-\d+)?(?:@|\s|$)", str(value)):
                    raise ValueError("Fixed-sequence evaluation requires literal sequences, not random-length segments")
        engine = OmegaConf.select(cfg, "paper_protocol.engine", default="trajectory")
        print(f"[Astrolabe] {ns.protocol}: {ns.num_runs} run(s), output {output}", flush=True)
        if engine in ("rna_beam", "bivalent_rna_beam"):
            from astrolabe.rna_beam import run_rna_beam, run_bivalent_rna_beam
            cli._maybe_autoselect_cuda_visible_devices(enabled=bool(ns.autoselect_gpu))
            settings = OmegaConf.to_container(cfg.beam_search, resolve=True)
            for index in range(ns.num_runs):
                current = copy.deepcopy(cfg)
                seed = int(cfg.seed) + index
                if engine == "bivalent_rna_beam" and settings.get("initialization") == "archived":
                    selected = int(settings.get("archived_start", 0)) + index
                    starts = settings["archived_starts"]
                    if selected >= len(starts):
                        raise ValueError("--num-runs exceeds the supplied archived bivalent starts")
                    current.beam_search.archived_start = selected
                    if ns.seed is None:
                        seed = int(starts[selected]["seed"])
                kwargs = dict(out_dir=output / f"{index + 1:03d}_seed_{seed}", seed=seed,
                              beam_width=int(settings["beam_width"]),
                              proposals_per_parent=int(settings["proposals_per_parent"]),
                              max_evaluations=int(settings["max_evaluations"]))
                if engine == "bivalent_rna_beam":
                    result = run_bivalent_rna_beam(current, objective_weights=settings["objective_weights"], **kwargs)
                else:
                    result = run_rna_beam(current, **kwargs)
                print(json.dumps(result, default=str))
        elif OmegaConf.select(cfg, "paper_protocol.initial_sequence_generator") == "python_random":
            import random
            lower, upper = cfg.paper_protocol.initial_sequence_length
            alphabet = str(cfg.paper_protocol.initial_sequence_alphabet)
            for index in range(ns.num_runs):
                current = copy.deepcopy(cfg)
                current.seed = int(cfg.seed) + index
                current.repeats = 1
                rng = random.Random(current.seed)
                current.seqs.A = "".join(rng.choices(alphabet, k=rng.randint(int(lower), int(upper))))
                cli._run_from_config(current, autoselect_gpu=bool(ns.autoselect_gpu))
        else:
            cli._run_from_config(cfg, autoselect_gpu=bool(ns.autoselect_gpu))
    return 0


def _prepare_inputs(ns):
    from astrolabe import cli
    root = cli._repo_root() / "resources"
    paths = sorted(root.glob("paper_inputs_*.tar.gz")) if ns.bundle == "all" else [root / f"paper_inputs_{ns.bundle}.tar.gz"]
    if not paths:
        raise FileNotFoundError("No bundled paper inputs found in this distribution")
    destination = Path(ns.out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    for path in paths:
        manifest_path = path.with_name(path.name.removesuffix(".tar.gz") + ".json")
        manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError(f"Input archive checksum mismatch: {path.name}")
        entries = {item["path"]: item for item in manifest["files"]}
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if member.isdir():
                    continue
                target = (destination / member.name).resolve()
                if not member.isfile() or not target.is_relative_to(destination) or member.name not in entries:
                    raise ValueError(f"Invalid input archive member: {member.name}")
                stream = archive.extractfile(member)
                data = stream.read()
                expected = entries.pop(member.name)
                if hashlib.sha256(data).hexdigest() != expected["sha256"]:
                    raise ValueError(f"Input checksum mismatch: {member.name}")
                if target.exists():
                    if hashlib.sha256(target.read_bytes()).hexdigest() != expected["sha256"]:
                        raise FileExistsError(f"Existing input differs: {target}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(data)
                extracted += 1
        if entries:
            raise ValueError(f"Missing archive members: {', '.join(entries)}")
    print(json.dumps({"data_dir": str(destination), "files_extracted": extracted,
                      "input_overlays": str(cli._repo_root() / "examples" / "paper" / "inputs"),
                      "usage": f"astrolabe paper run <protocol> --data-dir {destination} --inputs <input.yaml>"}, indent=2))
    return 0


def _collect(ns):
    from astrolabe.analysis.paper_scores import collect_records
    print(json.dumps(collect_records(ns.planned, ns.output), default=str))
    return 0


def _baseline(ns):
    from astrolabe.paper_baselines import run_baseline
    import subprocess
    try:
        result = run_baseline(ns.method, inputs=ns.inputs,
                              tool_dir=ns.tool_dir or f"vendor/{ns.method}",
                              out_dir=ns.out_dir or f"outputs/baselines/{ns.method}",
                              seed=ns.seed, dry_run=ns.dry_run)
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    print(json.dumps(result, indent=2))
    return 0


def _column_mapping(expressions, option="--column"):
    columns = {}
    for expression in expressions:
        key, separator, value = expression.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"{option} requires NAME=COLUMN")
        columns[key] = value
    return columns


def _analyze_scores(ns):
    from astrolabe.analysis.paper_scores import analyze_scores
    print(json.dumps(analyze_scores(ns.planned, ns.records, ns.out_dir, ns.endpoint,
                                   columns=_column_mapping(ns.column)), default=str))
    return 0


def _analyze_ligand_rank(ns):
    from astrolabe.analysis.paper_statistics import rank_ligands
    print(json.dumps(rank_ligands(ns.records, ns.out_dir, columns=_column_mapping(ns.column)), default=str))
    return 0


def _analyze_sensitivity(ns):
    from astrolabe.analysis.paper_statistics import threshold_sensitivity
    result = threshold_sensitivity(ns.planned, ns.records, ns.out_dir, ns.endpoint,
                                   denominator=ns.denominator, columns=_column_mapping(ns.column),
                                   seed=ns.seed, bootstrap_resamples=ns.bootstrap_samples,
                                   comparison_methods=tuple(ns.comparison_methods))
    print(json.dumps(result, default=str))
    return 0


def _analyze_ddg(ns):
    from astrolabe.analysis.paper_statistics import bootstrap_ddg
    print(json.dumps(bootstrap_ddg(ns.records, ns.out_dir, columns=_column_mapping(ns.column),
                                   seed=ns.seed, bootstrap_resamples=ns.bootstrap_samples), default=str))
    return 0


def _analyze_agreement(ns):
    from astrolabe.analysis.paper_agreement import analyze_agreement
    result = analyze_agreement(ns.manifest, ns.out_dir, columns=_column_mapping(ns.column),
                               confidence_columns=_column_mapping(ns.confidence_column, "--confidence-column") or None,
                               astrolabe_method=ns.astrolabe_method,
                               bootstrap_samples=ns.bootstrap_samples, seed=ns.seed)
    print(json.dumps(result, default=str))
    return 0


def _analyze_twostate(ns):
    from astrolabe.analysis.paper_twostate import analyze_twostate
    print(json.dumps(analyze_twostate(ns.manifest, ns.out_dir), default=str))
    return 0


def _analyze_cre(ns):
    from astrolabe.analysis.paper_geometry import cre_domain_distance
    print(json.dumps(cre_domain_distance(ns.reference, ns.structure,
                                        reference_chain=ns.reference_chain, predicted_chain=ns.target_chain)))
    return 0


def _analyze_rna(ns):
    from astrolabe.analysis.rna_geometry import analyze_rna_oligomer, analyze_bivalent_rna_geometry
    fn = analyze_rna_oligomer if ns.kind == "oligomer" else analyze_bivalent_rna_geometry
    print(json.dumps(fn(ns.structure)))
    return 0
