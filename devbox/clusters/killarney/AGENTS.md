# Cluster rules (Killarney)

Only rules needed in every session, and only things true of *this* cluster.
Portable rules are imported, not copied. The absolute `~/` path is deliberate:
this file is read through a symlink from the devbox session root, so a relative
import would resolve against the wrong directory.

@~/slurm-utils/devbox/AGENTS.shared.md

## This cluster

| | |
|---|---|
| Cluster name (`$CC_CLUSTER`) | `killarney` |
| Account | `aip-gigor` — the only association for this user. `$SBATCH_ACCOUNT` is **unset** and Slurm resolves no default, so `--account` must be named on every job. |
| Devbox names | `killarney-dev` (tunnel), `killarney-dev-1` / `-2` / `-3` (remote control) |
| Devbox root | the project repo, `/project/6101830/eop/unlearning-reward-hacking`. `$HOME` and `/scratch/eop` arrive via `--add-dir`; `/project` outside the repo does not. |
| Login node → compute | **`sbatch` is rejected from `/home`.** The check is on the submitting *directory*, not on where the script lives — `cd /scratch/eop; sbatch ~/some/script.sh` is fine. `devbox-up` handles this via `DEVBOX_SUBMIT_DIR`. |
| Python | `uv`. The repo has `uv.lock` and a `.venv`; use `uv run python …`, which resolves the environment without activating it. |

## GPUs: ask for the smallest that fits

**No MIG here** — the only GRES strings are `gpu:l40s:<n>` and `gpu:h100:<n>`, so
a whole card is the smallest unit. Prefer an L40S: there are 168 L40S nodes and
only 10 H100 nodes, so an L40S job starts far sooner and leaves the big cards for
work that genuinely needs 80 GB.

| Need | Flag |
|---|---|
| 48 GB, one L40S | `--gres=gpu:l40s:1` |
| n L40S, one node | `--gres=gpu:l40s:n` (n ≤ 4) |
| 80 GB, one H100 | `--gres=gpu:h100:1` |
| n H100, one node | `--gres=gpu:h100:n` (n ≤ 8) |

Node shapes (measured 2026-09-15 with `scontrol show node`):

| | nodes | CPUs | memory | GPUs | pair one GPU with |
|---|---|---|---|---|---|
| L40S (`kn001-168`) | 168 | 64 | 515 GB | 4 | ~16 CPUs, ~128 GB |
| H100 (`kn169-178`) | 10 | 48 | 2060 GB | 8 | ~6 CPUs, ~257 GB |

H100 nodes have **48** cores, not 96 — six per GPU. `--cpus-per-task=8` with 8
H100s cannot be satisfied on one node.

## Walltime and partitions

**Never name a partition.** Slurm routes on `--time` and the GPU type requested.
Every partition is `gpubase_*`; there is no CPU-only partition, so a CPU-only job
(the devbox included) also lands on a GPU node without holding a GPU. Don't read
"I'm on kn030, an L40S node" as "I have an L40S".

| `--time` ≤ | L40S nodes | H100 nodes |
|---|---|---|
| `3:00:00` | 168 | 10 |
| `12:00:00` | 126 | 8 |
| `1-00:00:00` | 84 | 6 |
| `3-00:00:00` | 42 | 4 |
| `7-00:00:00` | 17 | 2 |

A job is eligible for its tier and every tier above it, so **a shorter walltime
reaches strictly more nodes** — `3:00:00` reaches four times the L40S nodes that
`3-00:00:00` does. `gpubase_interac` (3 h, 25 nodes) serves `salloc`.

For **small CPU-only jobs this does not matter**: measured 2026-09-11, a
2-CPU/4 GB/no-GPU job started in 30 s at 1-, 3- and 7-day walltimes alike. Don't
shorten one hoping to start sooner. It matters a great deal for GPU jobs.

`sbatch --test-only` is a pessimistic backfill bound here, not a prediction — it
said +32.6 h for a job that started in 30 s. Submit the real job and watch
`squeue`.

## Storage

Measured with `diskusage_report` on 2026-09-15 — rerun it rather than trusting
this table.

| Path | Space | Inodes | Use |
|---|---|---|---|
| `/home/eop` | 19 / 50 GiB | 27K / 500K | dotfiles, `~/bin`, `~/.claude`, logs |
| `/scratch/eop` | 1254 / 2000 GiB | 172K / 10M | **caches, checkpoints, outputs** |
| `/project/6101830` (`aip-gigor`) | 4431 / 6000 GiB | 19M / 30M | the repo |

**Do not write large artifacts to `/project`** — it is shared with the rest of
`aip-gigor` and has been near its ceiling before (5891/6000 GiB on 2026-09-11,
since cleared). That headroom is not yours to spend.

**`/scratch` is the one to watch**: 16 GB → 903 GB → 1254 GB over 2026-09-11..15.

Cache redirects (`HF_HOME`, `UV_CACHE_DIR`, `TMPDIR`, `WANDB_*`, …) already point
at `/scratch/eop/cache` in `~/.bashrc`, above the interactive guard, so `sbatch`
jobs inherit them. Don't re-point them at `$HOME` or `/project`.

## Local quirks

- **`~/.bashrc` ends with `module load gcc` and `module load cuda/13.2`**, so every
  tmux window and job step gets `nvcc` (needed by vLLM/Triton JIT).
- **Nothing overrides `cd`.** `slurm_utils.sh` used to, to auto-activate a
  `.venv`, and it was removed on 2026-09-15: the first version returned
  `activate`'s exit status, so `cd x && y` silently skipped `y` while reporting
  success, and even fixed it stayed a surprise for every caller.
  `SLURM_UTILS_AUTO_ACTIVATE` is inert. Run `activate` explicitly, or `uv run`,
  which needs no activation.
- **`/tmp` is job-private** (`job_container/tmpfs`), so `ssh <node> tmux attach`
  cannot see the devbox's tmux socket. Use `devbox-up attach [slot]`, which goes
  through `srun --jobid=<id> --overlap`.
