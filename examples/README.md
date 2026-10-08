# Astrolabe examples

Start with the [paper workflows](paper/README.md). They include the original
RNA beam implementations, manuscript/archived design protocols, target and
chemistry input bundles, fixed-sequence evaluation and analysis commands.

```bash
astrolabe paper list
astrolabe paper prepare-inputs --out-dir data/paper
```

Use the [workflow guide](../docs/paper_workflows.md) to choose a model-backed
protocol and its input. Performance helpers remain under `benchmarks/`.

## CPU demo

```bash
python examples/01_cpu_reproduce.py --out-dir outputs/cpu-example
```

The script reads `input/9TP_cid.cif`, an existing AF3 prediction. For chains A
(85 residues) and B (87 residues), an 8 Å cutoff gives 19 Cα and 205 all-atom
residue contacts among 7,395 pairs. Minimum distances are 5.622136 Å and
2.376993 Å, respectively. No new structure prediction is performed.

Outputs include `scores.csv`, `scores.sqlite`, a selected CIF,
`selected/filter_report.csv`, `report/posthoc_report.md` and per-run provenance.
Use a new output directory on each run. See [reproducibility](../docs/reproducibility.md)
for the complete output contract and numerical reference.

## Model-backed design

Follow the [AF3 container setup](../container/README.md#install-af3), then run:

```bash
./scripts/astrolabe env doctor --task release_protein --strict
./scripts/astrolabe run --task release_protein
```

Inspect the native `scores.csv`, structures, saved configuration and terminal
provenance. This is an installation example. Paper figures require their own
frozen inputs, cohorts and analysis, as described in the reproducibility guide.
