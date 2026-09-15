# Cluster rules (shared)

Portable rules that hold on every Slurm cluster we run a devbox on. Import this
from the cluster's own `AGENTS.md` rather than copying it, so a fix lands
everywhere at once:

```markdown
@~/slurm-utils/devbox/AGENTS.shared.md
```

**Absolute, not relative.** A cluster's `AGENTS.md` lives in
`clusters/<cluster>/` and is read through a symlink from the session root, so a
relative path resolves against whichever directory the reader arrived by, not
against this repo. On a root outside `$HOME` it points somewhere that does not
exist — and a `@`-import that resolves to nothing fails silently, so the shared
rules are simply absent with no dialog and no error.

Anything with a number in it that differs per cluster — account name, partition
ladder, GPU flags, quotas, the venv layout — belongs in that cluster's
`AGENTS.md`, not here. See `AGENTS.template.md`.

## The dev box is not a compute node

This session usually runs inside the `claude-dev` Slurm job
(`~/slurm-utils/devbox/devbox.sh`, driven by `devbox-up`). It is sized to host
the agent sessions and a VS Code tunnel, nothing else — a couple of cores and a
few GB, no GPU. `devbox-up config` prints what this cluster asked for.

**Never run training, inference, vLLM, or a test suite in it.** A vLLM import
alone will exceed the memory cap. The cap is per job, not per session, so an
OOM kills every agent slot *and* the tunnel at once, not just the offender.

To get real hardware, either submit a batch job, or hold an allocation and reuse
it for every command:

```bash
sbatch --time=3:00:00 --gpus=<smallest that fits> --cpus-per-task=6 --mem=16G job.sh

salloc --time=3:00:00 --gpus=<smallest that fits> --cpus-per-task=6 --mem=16G
srun --jobid=<id> --overlap python -m pytest tests/
srun --jobid=<id> --overlap python src/train.py
```

`--overlap` is required — without it the second `srun` blocks waiting for the
first step's resources. `srun --jobid=<id> --overlap` is also the only way into
the dev box's tmux where `/tmp` is job-private; `ssh <node>` cannot see the
socket. `devbox-up attach [slot]` does this for you.

## You are not the only agent here

Several agent slots share one devbox and one source tree, and any of them may
have jobs in the queue. This is the normal case, not the exception.

- **Don't cancel jobs you didn't submit** without explicit permission. Use
  `devbox-up restart` for the devbox chain itself, which only touches the
  pending successor and leaves running agents alone.
- Expect concurrent edits in the same checkout. Re-read a file before assuming
  its contents; don't "clean up" work you cannot account for.

## One source tree

Work lives in `$HOME` and is edited in place. **Do not make a second copy** to
work around a full filesystem or a busy node — on one cluster a fork like that
drifted for days, with jobs running against one copy while the other held older
code. If a second copy ever seems necessary, say so and get agreement first.

## Submitting jobs

- **Ask for the smallest GPU that fits.** Where a cluster partitions GPUs (MIG),
  a partition request reaches strictly more nodes than a whole card, so it
  usually starts sooner *and* leaves the big cards alone. The cluster's
  `AGENTS.md` has the exact flags.
- **Ask for the shortest walltime you can.** A job is eligible for every tier at
  or above its `--time`, so a shorter request reaches more nodes.
- **Stay near the node's core:memory:GPU ratio** unless more genuinely speeds
  the job up.
- `sbatch --test-only` is a pessimistic backfill bound, not a prediction.
  Submit the real job and watch `squeue`.
- **Editing a job script does not change an already-queued job** — Slurm
  snapshots the script at submit time. Cancel and resubmit to apply a change.

## Storage

Keep large artifacts (caches, checkpoints, outputs) on scratch, which is fast
and usually not backed up or purged on a schedule; keep repos and anything you
would miss on home, which is usually backed up. A shared `/project`-style
allocation is the one to avoid for new work: it is shared with the whole group
and typically runs out of *inodes* long before space.

Run `diskusage_report` (or the cluster's equivalent) rather than trusting any
table — the numbers in a cluster's `AGENTS.md` are a snapshot from when someone
last looked.

Cache redirects (`HF_HOME`, `UV_CACHE_DIR`, `TMPDIR`, `WANDB_*`, …) belong in
`~/.bashrc` **above** any interactive guard, so `sbatch` jobs inherit them.

## Shell traps that have bitten us

- A `~/.bashrc` that ends with a `cd` into the current project means every new
  tmux window and every job step starts *there*, not in `$HOME`.
