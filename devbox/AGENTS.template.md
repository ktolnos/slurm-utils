# Template: a new cluster's AGENTS.md

Copy this into `clusters/<cluster>/AGENTS.md` — in this repo, in git — and
symlink it into the devbox session root. It lives here rather than in the root
so there is one copy: a copy in the root drifts from this repo silently, and
where the root is a project repo it would also be a second place the same
cluster facts are written down. Everything portable is imported, so this file
should stay short — if a section here has no cluster-specific number or name in
it, it probably belongs in `AGENTS.shared.md` instead.

```bash
ROOT=$(devbox-up config | awk '/^root/{print $3}')
mkdir -p ~/slurm-utils/devbox/clusters/$CC_CLUSTER
cp ~/slurm-utils/devbox/AGENTS.template.md ~/slurm-utils/devbox/clusters/$CC_CLUSTER/AGENTS.md
# edit the TODOs, then:
ln -sfn ~/slurm-utils/devbox/clusters/$CC_CLUSTER/AGENTS.md "$ROOT/AGENTS.md"
head -40 "$ROOT/AGENTS.md"      # verify it reads through the symlink
```

Claude Code reads `CLAUDE.md` by default, so the root also needs a `CLAUDE.md`
that imports this file. Two files rather than one keeps the name working for
both Claude Code and other agents:

```bash
echo '@AGENTS.md' > "$ROOT/CLAUDE.md"   # or add that line to an existing CLAUDE.md
```

Both must be reachable **inside the session root**, because that is the
directory whose trust is persisted and whose project settings are honoured — a
file in `$HOME` is not read as project instructions. The symlink satisfies that;
its target does not have to be in the root.

Where the root is an existing project repo, git-ignore the symlink and add
`@AGENTS.md` to that repo's own `CLAUDE.md` rather than overwriting it.

Delete everything above the line when you copy it.

---

# Cluster rules (TODO: cluster name)

Keep here only rules required for each session, and only things that are true of
*this* cluster. Portable rules are imported below. Deep detail — full partition
ladder, node layout, queue notes — belongs in a separate `CLUSTER.md` that
nobody reads unless they need it.

@~/slurm-utils/devbox/AGENTS.shared.md

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
