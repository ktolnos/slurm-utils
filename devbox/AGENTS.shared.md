# Cluster rules (shared)

These hold on every cluster. The `AGENTS.md` that imports this one adds that
cluster's own facts — account, GPU flags, walltime tiers, quotas, paths.

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
You can run CPU-only jobs to get more memory or cores.
Clusters use fairshare, so requesting more than you need lowers your priority
for the subsequent jobs.
`--overlap` is required — without it the second `srun` blocks waiting for the
first step's resources. `srun --jobid=<id> --overlap` is also the only way into
the dev box's tmux where `/tmp` is job-private; `ssh <node>` cannot see the
socket. `devbox-up attach [slot]` does this for you. 

## The session root is not the project

This session's root holds the devbox's own docs and config. The work is
elsewhere, and the path is injected into every session (including after each
`/clear`) as an `Active project:` line — `active-project` prints it, `project`
cds to it. Read and edit code there, not in the root.

The root is what it is because workspace trust and conversation history are
keyed to it, not because anyone thinks the work lives there. If the user starts
on a different project, run `active-project <dir>` so the next session and the
other slots agree.

## You are not the only agent here

Several agent slots share one devbox and one source tree, and any of them may
have jobs in the queue. This is the normal case, not the exception.

- **Don't cancel jobs you didn't submit** without explicit permission. Use
  `devbox-up restart` for the devbox chain itself, which only touches the
  pending successor and leaves running agents alone.
- Expect concurrent edits in the same checkout. Re-read a file before assuming
  its contents; don't "clean up" work you cannot account for.

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

## Storage: what goes where

Keep large artifacts (caches, checkpoints, outputs) on scratch, which is fast
and usually not backed up or purged on a schedule; keep repos and anything you
would miss on home, which is usually backed up. 

Cache redirects (`HF_HOME`, `UV_CACHE_DIR`, `TMPDIR`, `WANDB_*`, …) are set in
`~/.bashrc` **above** any interactive guard, so `sbatch` jobs inherit them.
