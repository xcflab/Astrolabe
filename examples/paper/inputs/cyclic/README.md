# Cyclic peptide paper inputs

Prepare the bundled target homologies and modified-residue components once:

```bash
astrolabe paper prepare-inputs --out-dir data/paper
astrolabe paper run cyclic-peptide-multiring \
  --inputs examples/paper/inputs/cyclic/multiring/01_plecanatide.yaml \
  --data-dir data/paper --seed 1 --out-dir outputs/plecanatide
astrolabe paper run cyclic-peptide-bound \
  --inputs examples/paper/inputs/cyclic/bound/01_MDM2_head_tail.yaml \
  --data-dir data/paper --seed 1 --out-dir outputs/mdm2_cyclic
```

The `bound/` overlays cover MDM2, MCL1, KEAP1, and PD-L1 with head-to-tail,
head-to-side-chain, tail-to-side-chain, disulfide, lactam, lactone,
thiolactone, thioether, and hydrocarbon-stapled inputs. Target sequences and
the original AF3 homology inputs are retained. Standard-residue overlays retain
the original interface/non-interface LigandMPNN mixture and require PyRosetta;
modified-residue overlays use the source runner's global LigandMPNN proposals.
Manifest-derived postbond examples use length 20, within the original length
range recorded in each overlay. Stapled bound examples reuse a chemically
defined peptide-only starter with the target-bound paper protocol.

`multiring/` contains all five publication topologies, including the head-to-tail
bond of kalata B1. Sequence positions at the configured endpoints retain the
required amino acids. The latest manuscript specifies LigandMPNN; the original
run script used MCMC. `cyclic-peptide-multiring` follows the manuscript, while
`cyclic-peptide-multiring-archived` retains that original MCMC execution recipe.
Neither command claims that newly generated runs are the archived cohort.

Use `cyclic-peptide-representation` with an overlay from `representation/` for
the peptide-only input comparison. Each file preserves one original sample's
sequence, seed, endpoint atoms, target lengths, modified residues, and raw
method name. These are runnable examples of the source methods, not the full
40-start cohort. The source `generic_cyclic_index_offset` usually emits **both**
explicit bonds and cyclic offsets; only `head_tail_index_offset` emits offsets
alone. Source methods are kept separate instead of choosing a method from its
measured success rate.

Tail-to-side-chain lactone and thiolactone require a chemically prepared C
terminus. The CTM component retains the terminal carbonyl C=O and removes OXT
for the specified external ester/thioester bond; SAM/CAM provide the configured
donor state where present. Four archived controls with an ordinary C terminus
are excluded from runnable examples. Their source rows and reasons are in
`01_provenance.json`. S5/R8 inputs preserve the original stereochemistry and
named reactive atoms for hydrocarbon staples.

The optimization descriptor `cyclic_closure_norm` uses a Gaussian with sigma
0.20 Å over bonds selected by `encoding.scope`. The bound objective uses the
minimum and requires a value of at least 0.9; the free multiring objective uses
the mean without that gate. The separate publication endpoint is
`closure_all_bonds` (`cyclic_all_bonds_within_error`): **every configured bond**
must have both endpoint atoms and absolute distance error at most 0.30 Å,
including bonds omitted from the encoding. `closure_max_error` reports the
maximum absolute error. Missing endpoints give infinity and fail closure.
The representative is the state with the highest optimization objective;
closure and confidence must be read from that same state. Aggregate success
fractions retain all planned trajectories in the denominator (40 per free
multiring topology), including missing or failed trajectories.

`resources/paper_inputs_cyclic.json` records the archive and component hashes,
sizes, and source-relative paths. The archive contains four target homology
JSON files and six original postbond CCD files; model weights and result
structures are not part of it.
