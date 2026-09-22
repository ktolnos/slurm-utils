# Per-cluster rules

One directory per cluster, holding that cluster's own facts — account, GPU
flags, walltime tiers, quotas, paths. They live here, in git, and reach
sessions through **user-level memory** rather than a file in a project:

```bash
echo "@<devbox>/clusters/<cluster>/AGENTS.md" >> "$CLAUDE_CONFIG_DIR/CLAUDE.md"
```

User level, because the agents start in whichever project is active rather
than in one fixed root. A per-project copy would have to be repeated in, and
would litter, every repo they are ever pointed at — and would be a second
place the same cluster facts are written down.

Two rules for anything in here:

- **Import shared rules, never copy them.** Use a fully absolute
  `@/path/to/devbox/AGENTS.shared.md`, not a relative one and not `@~/...`:
  on a cluster whose `$HOME` is node-local, `~` resolves to a different
  filesystem depending on which node reads it.
- **Only cluster facts.** Anything project-specific belongs in that project's
  own `CLAUDE.md`, which Claude Code reads on top of these.

The devbox hooks (`pin-session`, `active-project --hook`, `slurm-guard`) are
installed once per cluster in `$CLAUDE_CONFIG_DIR/settings.json`, for the same
reason — see `SETUP_INSTRUCTIONS.md` step 6. A project that needs settings of
its own still uses its own `.claude/settings.local.json`, which Claude Code
merges over the user-level file.
