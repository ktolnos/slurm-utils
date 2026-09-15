# Template: a new cluster's AGENTS.md

Copy this into the devbox session root as `AGENTS.md` (default `~/devbox`) and
fill in the blanks. Everything portable is imported, so this file should stay
short — if a section here has no cluster-specific number or name in it, it
probably belongs in `AGENTS.shared.md` instead.

```bash
cp ~/slurm-utils/devbox/AGENTS.template.md ~/devbox/AGENTS.md
# then edit the TODOs; verify with:  head -40 ~/devbox/AGENTS.md
```

Claude Code reads `CLAUDE.md` by default, so also add a one-line `CLAUDE.md`
next to it that imports this file. Two files rather than one keeps the name
working for both Claude Code and other agents:

```bash
echo '@AGENTS.md' > ~/devbox/CLAUDE.md
```

Both files must live **inside the session root**, because that is the directory
whose trust is persisted and whose project settings are honoured. A file in
`$HOME` is not read as project instructions.

Delete everything above the line when you copy it.

---

# Cluster rules (TODO: cluster name)

Keep here only rules required for each session, and only things that are true of
*this* cluster. Portable rules are imported below. Deep detail — full partition
ladder, node layout, queue notes — belongs in a separate `CLUSTER.md` that
nobody reads unless they need it.

@../slurm-utils/devbox/AGENTS.shared.md

## This cluster

| | |
|---|---|
| Cluster name (`$CC_CLUSTER`) | TODO |
| Account | TODO (`echo $SBATCH_ACCOUNT`, or `sacctmgr show assoc user=$USER format=account%30 -n`; without the width it truncates) |
| Devbox names | TODO-dev (tunnel), TODO-dev-1/2/3 (remote control) |
| Login node → compute | TODO: does `sbatch` work from `/home`? |
| Python | TODO: `uv run python …` / module load / conda — and any repo that is the exception |

## GPUs: ask for the smallest that fits

TODO: the flag table for this cluster. Get it from `sinfo -o '%G %N'` and
`scontrol show node <gpu node>`; list MIG profiles if the site has them.

| Need | Flag |
|---|---|
| smallest slice | `--gpus=TODO` |
| … | … |
| full card | `--gpus=TODO:1` |
| n cards, one node | `--gpus-per-node=TODO:n` |

TODO: node shape — cores / memory / GPUs per node, and the cores-per-GPU ratio
to stay near.

## Walltime and partitions

TODO: how walltime routes to partitions here, the tier boundaries, and the
site's minimum and maximum job length. Note whether a partition must be named
explicitly or Slurm resolves it from `--time` and the resources requested.

## Storage

TODO: replace with this cluster's real paths and a dated `diskusage_report`
snapshot.

| Path | Quota (measured TODO: date) | Use |
|---|---|---|
| TODO home | | repos, dotfiles, logs |
| TODO scratch | | caches, checkpoints, outputs |
| TODO project/shared | | shared group data |

## Local quirks

TODO: anything that has surprised someone here — a `~/.bashrc` that overrides
`cd` or ends with a `cd` into a project, a job-private `/tmp`, a login node that
refuses `sbatch` from `/home`, a module system that must be loaded first. If it
turns out to be true on every cluster, move it to `AGENTS.shared.md`.
