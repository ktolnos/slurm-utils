# Per-cluster session-root files

One directory per cluster, holding the files that belong in that cluster's
devbox session root. They live here, in git, and are **symlinked** into the
root rather than copied — a copy in the root drifts from this one silently, and
on a cluster whose root is a project repo it would also be a second place the
same cluster facts are written down.

```bash
ln -sfn ~/slurm-utils/devbox/clusters/<cluster>/AGENTS.md  "$(devbox-up config | awk '/^root/{print $3}')/AGENTS.md"
ln -sfn ~/slurm-utils/devbox/settings.json                 "<root>/.claude/settings.json"
```

Two rules for anything in here:

- **Import shared rules, never copy them.** Use `@~/slurm-utils/devbox/AGENTS.shared.md`
  with an absolute `~/` path, not a relative one: these files are read through a
  symlink, and a relative import would resolve against whichever path the reader
  happened to arrive by.
- **Only cluster facts.** Anything project-specific belongs in the project's own
  `CLAUDE.md`, which is what imports `AGENTS.md`.

`settings.json` is shared by every cluster, so it holds only the devbox pin
hook. A root that needs its own settings uses `.claude/settings.local.json`
alongside the symlink, which Claude Code merges over it.
