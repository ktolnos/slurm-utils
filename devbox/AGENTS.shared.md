# Cluster rules (shared)

These hold on every cluster. The `AGENTS.md` that imports this one adds that
cluster's own facts — account, GPU flags, walltime tiers, quotas, paths.

## The dev box is not a compute node

This session runs inside the devbox (`devbox.sh`, driven by `devbox-up`) — on
most clusters as the `claude-dev` Slurm job, on a site whose login node is
meant to be used directly as a plain background process on that machine.
`devbox-up config` prints which, and what this cluster asked for. Either way it
is sized to host the agent sessions and a VS Code tunnel, nothing else.

**Never run training, inference, vLLM, or a test suite in it.** A vLLM import
alone will exceed the memory cap. In a job the cap is per job, not per session,
so an OOM kills every agent slot, the codex daemon *and* the tunnel at once,
not just the offender. In local mode there is no cgroup and so no cap — the
same rule holds, but nothing enforces it and the damage lands on everyone else
sharing that machine instead.

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

## You start in the active project

This session's working directory **is** the active project: `active-project`
prints it, `project` cds to it, and it is injected into every session —
including after each `/clear` — as an `Active project:` line.

Workspace trust and conversation history are both keyed to that path, so each
project keeps its own set of slot conversations; switching away and back
resumes the ones belonging to it rather than stranding them.

`active-project <dir>` repoints it, granting workspace trust as part of the
move (it asks first, and lists any `.claude/settings.json`, `.mcp.json` or
hooks in the tree, because those execute on session start). **Running slots
keep their old working directory**: a process cannot change its own cwd, so no
amount of `/clear` moves them — they follow the pointer when next relaunched.

## You are not the only agent here

Several agent slots share one devbox and one source tree, and any of them may
have jobs in the queue. This is the normal case, not the exception. Nor are they
all Claude Code: codex sessions started from the ChatGPT app share the box and
the tree without appearing as tmux windows.

- **Don't cancel jobs you didn't submit** without explicit permission. Use
  `devbox-up restart` for the devbox chain itself: on a Slurm-hosted box it
  replaces only the pending successor and leaves running agents alone, so the
  change lands when the current job hits its walltime — up to three days away.
- **`devbox-up restart --now` kills every live slot.** It cancels the running
  job (or, in local mode, the running process) so the change applies
  immediately, taking the other agents' sessions, the codex daemon and the
  tunnel with it. Their conversations resume afterwards; their in-flight work
  does not. Ask first. In local mode plain `restart` already does this, since
  there is no chain to hide behind — `devbox-up config` prints the mode.
- `up` and `restart` update claude/codex/code first (`--no-update` to skip).
  A restart can therefore change the CLI version under a resumed conversation.
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
