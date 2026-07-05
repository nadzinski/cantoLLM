# Development

This project uses [uv](https://docs.astral.sh/uv/) for Python + dependency management.

## Setup

```
uv sync
```

Installs dependencies and the `cantollm` package itself (editable) into `.venv/`.
After that, activate the venv:

```
source .venv/bin/activate
```

## Running tests

```
python -m pytest tests/ -v
```

## Adding dependencies

```
uv add <package>              # runtime dependency
uv add --dev <package>        # dev-only dependency
```

# viz/ — architecture explorer maintenance

`viz/` holds an interactive architecture visualization (see `viz/README.md`):
trace harnesses that instrument the real code at runtime plus a static
`viz/index.html`. It is a snapshot of the architecture, so architectural
changes can make it stale or break the harnesses — e.g. renaming/moving
modules or methods that the harnesses wrap (`StandardBackend.forward/sample`,
`attention_method.build_mask`, `apply_rotary_emb`, `SequentialEngine.submit`,
the CB prototype's `_plan_step`/`_build_input_ids`), changing engine/queue
mechanics, or landing new subsystems (e.g. the continuous-batching engine).

**Do not update viz/ by default.** When you make (or are asked to plan) an
architectural change that would affect it, always ask the user whether the
visualization should be updated too. Trace data regenerates via the commands
in `viz/README.md`; diagram/structure changes mean editing `viz/index.html`.
The same rule covers the two design-content tabs: the Roadmap tab mirrors
PLAN.md's phase Status lines and the CB-wiring tab mirrors
`continuous_batching.md` — when either document changes, ask whether the
corresponding tab should be updated.

# PLAN.md

`PLAN.md` is the long-horizon roadmap for the project and is written as a
forward-looking spec. The bullets describe intent; they should stay readable
as "what we're going to build" long after the work lands.

Each phase carries a single **Status (YYYY-MM-DD)** line right under its
**Goal:** that captures what's done, what's been explicitly descoped or
moved, and what's still open. That line is the single source of truth for
progress — a future agent should be able to read PLAN.md alone and know
where things stand, without grepping git history or the code.

When you make progress that affects a phase:

1. Update that phase's Status line. Bump the date to today. Summarize in
   one or two sentences, same shape as the existing lines: "Done: …. Open: ….".
   Mention deferrals/descopes explicitly ("moved to Phase N", "dropped,
   reason …") so nothing silently disappears.
2. Do **not** add checkmarks, strikethroughs, or "DONE" tags to individual
   bullets. Keep the phase body forward-looking; let the Status line carry
   the state.
3. If an individual bullet is being moved to another phase, edit the bullet
   out of the current phase and add it to the target phase — then note the
   move in both phases' Status lines.
4. If a whole phase finishes, the Status line becomes "Complete (YYYY-MM-DD)."
   — keep the phase body as historical reference for why it was shaped that way.
