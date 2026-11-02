# Proposal: a composable scientific execution architecture

**Status:** design proposal for discussion. The interfaces below describe a
target, not capabilities already available in this repository. Migration must
preserve existing recipes and be qualified against the corresponding original
implementations.

## Goals and boundaries

Astrolabe should let a user compose molecular systems, prediction and mutation
methods, objectives, and search policies without creating a separate search
engine for each scientific task. Common interfaces must expose actual backend
capabilities rather than assume that a successful AlphaFold 3 case applies to
other evaluators. Model-specific inputs, algorithms and outputs remain available.

The ordinary configuration and CLI should continue to work. Advanced users may
compose typed components directly. Configuration should show the selected
components and their effective parameters; summaries should show selected
results, with detailed data and provenance available on request.

## Scientific data model

* **System** defines design variables, molecular composition, scientific states,
  and the physical occurrences of each variable. A shared sequence is one
  variable even when it occurs in multiple chains or assemblies; equal strings
  alone do not establish sharing.
* **Candidate** is an immutable assignment to the variables, including chemical
  identity and modifications. A **proposal** records the parent, edit, method
  and guidance; two proposal events may produce the same candidate.
* A scientific **state**, a search **stage**, and a model **sample** are
  independent identities. Component aliases identify producers, not states.
* **Artifacts and evidence** bind observations to the candidate, state,
  assembly, occurrence and actual model sample. A prediction batch records
  both the requested sampling plan and the members actually produced. Each
  member links its own structure, metrics, seed and native metadata.
* A **measurement** records its scientific definition, producer, units,
  subjects, named axes and dependencies. An **assessment** applies an explicit
  objective, sample/state reduction, gates and comparison context to
  measurements; it does not overwrite them.

For example, in two binding states `AB` and `CB`, B may be the same design
variable. Editing B changes the input for both predictions, while the two
complexes retain independent structures and measurements. In an `AAB` assembly,
the A copies share sequence but retain distinct physical observations.

Mutators may use an earlier structure to guide an updated working sequence.
That relationship must identify both the candidate evaluated by the evidence
and the candidate being edited. Structural processing creates a derived
artifact and records the transformation; confidence from the original model
remains associated with the original prediction.

## Component contracts

Keep the actual family registries as the source of component implementations.
Each component should declare its configuration schema, supported input and
output semantics, scientific asset requirements, execution resources and
recovery boundary. Before loading a model, resolve the selected configuration
and check requested capabilities. Validate input-dependent constraints again
when actual inputs are known. Unsupported required behavior must fail explicitly;
approximations must be selected and recorded.

The families have different natural results:

| Component | Input and result |
| --- | --- |
| Initializer | System and initialization settings → initial candidate and optional artifacts |
| Evaluator | Candidate, state and sampling plan → prediction batch and evidence |
| Mutator | Candidate, guidance and sampling policy → proposals or a specified distribution |
| Structure processor | Input artifact and settings → derived artifact, mapping and optional measurements |
| Measurement | Candidate, evidence and selection → measurements |
| Assessment | Measurements and objective protocol → scores, gates and comparison context |

An evaluator may produce multiple samples per call. The contract must preserve
the mapping between logical requests, native batches and actual samples; it
must not silently select the first or best sample. Guidance selection, sample
reduction and search selection are separate explicit decisions. A backend that
cannot expose a requested result must report the limitation.

Sampling records must distinguish explicit per-call seeds from continuous RNG
state, including native model/member/sample indices where applicable. Existing
recipes keep their current random-number consumption and selection semantics
during migration. Recycle counts, diffusion steps, initial guesses, templates
and constraints retain their backend-specific meanings.

Measurements retain their producer and definitions: a pair matrix has separate
ordered axes; a residue, token or atom mapping must be explicit; missing values
are not zero. Similar metric names do not establish cross-model comparability.
Legacy aggregated data should remain marked as aggregated or unknown when the
original provenance cannot be recovered.

## Search, execution and recovery

Use one shared search coordinator with interchangeable policies for trajectory,
branching and collection. Policies own search decisions, budgets and stopping;
scientific components own their calculations. Model owners and existing backend
services remain responsible for their resource lifecycles. Asynchronous
collection of external jobs retains their actual job and result types.

Each named operation records its prepared inputs, component and implementation
identities, relevant asset contents, sampling protocol and resource requirements.
Cache reuse requires matching actual dependencies, including sequence projection,
model, MSA, templates, conditioning and sampling where relevant. A change in
objective weights may reuse measurements but requires a new assessment.

Separate resource availability from recoverability. A resident model is not
proof that an operation has a durable result. For durable execution, associate a
stable attempt with its input and budget; commit the complete result and RNG
receipt before reporting completion. An issued call whose outcome is unknown is
not automatically replayed. A deliberate retry is a new, accounted attempt.
Objects without a faithful persistent representation, such as live poses in some
runtimes, must retain an explicit process-lifetime boundary.

## Migration and acceptance

1. Inventory the actual registered implementations, entry points and output
   consumers. Record effective parameters, sample selection, supported input,
   assets, scientific behavior and existing evidence for each component.
2. Establish candidate, state, occurrence, proposal and prediction-member
   identities and retain compatibility with existing scientific recipes.
3. Migrate backend input and result production, then measurements, selections
   and assessments. Preserve native information and clearly identify unsupported
   or historically aggregated output. A cold registry construction is not an
   execution or scientific equivalence test.
4. Connect mutators, structure transformations, resource owners and durable
   receipts to the same contracts. Retire legacy shared mutable state and
   duplicate search control only after their consumers have migrated.
5. Verify each registered component against its applicable baseline: input and
   output semantics, seeds and samples, relevant scientific results, error
   handling and recovery. Separately measure actual device activity and
   end-to-end throughput before making performance claims. Small canaries set
   the order of work; they do not reduce the intended coverage.

Compatibility tests should cover shared variables, repeated chain copies,
fixed residues and modifications, parent/sibling isolation, asymmetric pair
axes, missing data, selection and reduction, and cache invalidation. Preserve
the scientific settings, model assets, input identities and random state used
for each comparison.

## Decisions for review

Existing recipes retain their current behavior. New recipes must explicitly
specify how prediction members guide proposals, how samples and states are
reduced for objectives and success criteria, whether samples across states are
paired, and whether scores from different models are comparable or calibrated.
Changes to those choices are scientific protocol changes, not consequences of
the refactor.
