# Command Design Rules

This file records the repository-local rules for adding or changing `astrolabe` commands.
It is intentionally concrete and biased toward the behaviors already enforced in tests.

## 1. One Meaning, One Name

- Reuse the same option name for the same concept everywhere.
- Current canonical examples:
  - task/run name: `-t` / `--task-name`
  - output root: `-o` / `--out-dir`
  - repeated runs: `-n` / `--num-runs`
  - reproducibility seed: `--seed`
- If an old name is ambiguous but already used, keep it only as a compatibility alias and add a clearer canonical name.
  - Example: `astrolabe design protein-binder` with alias `binder`
  - Example: `astrolabe design ligand-binder` with alias `ligand`

## 2. Empty Invocation Must Not Surprise

- A top-level executable command with no meaningful inputs must print help and exit non-zero.
- Do not fall through into pipeline execution and let lower layers raise unrelated errors.

## 3. Help Must Explain Layering and Selection

- If a command composes multiple config sources, the help text must state the layering order explicitly.
  - Example: `astrolabe run` explains `--task` / `--standalone` / `--overlay`.
- If a selector is a file path rather than a built-in name, say so directly in the help.
  - Example: `astrolabe posthoc filter --preset` is a YAML path.
- If a flag changes how rows are selected, document the exact rule, not just the marketing name.
  - Example: `representative-per-run` = choose one row from the last stage, highest `weighted_score`, ties prefer later step.

## 4. Defaults Should Reduce Boilerplate, Not Hide State

- Use sensible derived defaults when the destination is mechanical.
  - Example: `astrolabe posthoc filter` auto-generates `--dst` next to the DB.
- Do not add silent fallback behavior that changes semantics invisibly.
- If a default is derived, print the resolved value in normal command output.

## 5. Dry-Run and Preview Must Be Real CLI Paths

- `--dry-run`, `--print-config`, and preview-style commands must work without requiring unrelated environment interpolation to resolve.
- Preview/inspection modes must not require execution-only arguments.
  - Example: `--preview-preset` should not require `--db` or `--dst`.

## 6. Interactive Commands Need a Clean Non-TTY Failure Mode

- Interactive commands must detect non-TTY stdin early.
- On non-TTY, print one clear stderr line and return exit code `2`.
- Do not raise raw Python tracebacks for a normal “wrong execution mode” case.

## 7. Path Validation Must Respect Where Config Lives

- Validate relative env/config paths relative to the owning file first, then repo-root-aware fallbacks when appropriate.
- Path validators must know whether a key expects a file, directory, or executable.
- If a value is `${oc.env:VAR,???}`, diagnostics should name `VAR` explicitly and mention the shell/`.env` path.

## 8. Task-Aware Diagnostics Must Explain Why

- `env doctor` / `env wizard` should not only say a key is needed; they should explain which evaluator, mutator, predocker, or loss term requires it.
- Return structured reason data from the requirement inference layer instead of duplicating prose at the UI layer.

## 9. Strict Failure Beats Hidden Fallback

- Missing required metrics in post-hoc workflows should fail loudly unless the user explicitly opts into exploratory mode.
- Commands must not silently skip meaningful checks just because upstream data is incomplete.

## 10. Prefer Shared Helpers Over Repeated Parser Drift

- Shared option semantics belong in helper functions, not copied across many subparsers.
- If two commands call the same backend handler, they should expose the same relevant surface unless there is a deliberate documented reason not to.

## 11. Shell Completion Is Part of the CLI Contract

- `astrolabe` completion must be updated together with CLI surface changes.
- Prefer deriving static completion from the `argparse` parser tree instead of maintaining separate shell-only command maps.
- Dynamic providers are allowed only for values that cannot be known statically.
  - Examples: SQLite metric names for `posthoc filter --expr`, task templates under `config/`, preset YAML files near `--db`.
- Dynamic providers must be lazy and argument-gated.
  - Do not scan DBs or the filesystem on unrelated completion paths.
- Keep Bash and Zsh thin. Shared logic belongs in Python; shell scripts should only forward words/cursor state.
- Completion performance matters because shells invoke it repeatedly. Treat the current Python startup dominated path as a ceiling, not a target to exceed.
- Any completion change must add or update:
  - parser-level/unit tests in `tests/test_cli_completion.py`
  - at least one real Bash round-trip check
  - at least one real Zsh round-trip check

## 12. Script Entrypoints Need an Owner

- A reusable workflow should have a canonical `astrolabe` subcommand or `just` recipe.
- `scripts/` may keep thin legacy wrappers, smoke helpers, and host/bootstrap tools, but wrappers should delegate to the canonical owner instead of carrying separate parser logic.
- Task- or figure-specific analysis scripts belong under `examples/` or the owning project workspace, not as new top-level `scripts/` entrypoints.
- Do not maintain a second static script inventory. A public reusable workflow
  should normally be an `astrolabe` subcommand; if a top-level script remains the
  canonical public entrypoint, document only that entrypoint in
  `docs/user_guide.md` and its owner in `docs/developer_guide.md`.
