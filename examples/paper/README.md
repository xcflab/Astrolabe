# Paper workflows

Use `astrolabe paper list` to discover the design and evaluation protocols.
The detailed [workflow guide](../../docs/paper_workflows.md) explains their model
requirements, budgets, input conventions and analysis rules.

Prepare the supplied target templates, MSAs and chemical definitions once:

```bash
astrolabe paper prepare-inputs --out-dir data/paper
```

Then run a protocol with its target input, in the configured model environment:

```bash
astrolabe paper run rna-aptamer \
  --inputs examples/paper/inputs/nucleic/rna_aptamer_1urn_u1a.yaml \
  --data-dir data/paper --seed 1 --out-dir outputs/u1a-rna
```

`--dry-run` prints the resulting configuration without inference. One invocation
runs one trajectory; `--num-runs` requests independent runs. `steps: N` includes
state 0 and permits states 0 through N−1, as defined in the SI.

| Inputs | Content |
| --- | --- |
| [protein](inputs/protein/) | The protein target panel, IDRs, split-template Cre, dual targets and antibody frameworks |
| [nucleic](inputs/nucleic/) | Four RNA/DNA aptamer targets and six nucleic-acid targets for protein binders |
| [cyclic](inputs/cyclic/README.md) | Atom-level linkage topologies, modified residues and target-bound/free peptide inputs |
| [assemblies](inputs/assemblies/manifest.csv) | Twenty original pseudo-C3 starting structures, shared-segment inputs and archived seeds |
| [interventions](inputs/interventions/) | Target-pocket definitions and reference structures |
| [baselines](baselines/README.md) | Native comparator commands and study-specific adapters |

Target overlays reference `ASTROLABE_PAPER_DATA`; `--data-dir` sets it for one
invocation. Relative paths inside an overlay are interpreted from that overlay's
directory. Input archives and their source/checksum manifests are under
`resources/paper_inputs_*`. They contain required small inputs, not model weights
or complete design-output archives.

Archive-derived variants retain their original settings and state provenance.
Where manuscript text and historical execution differ, `*_archived` protocols
and each configuration's `paper_protocol.notes` make that distinction explicit.
