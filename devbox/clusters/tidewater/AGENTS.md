@/h/eop/slurm-utils/devbox/AGENTS.shared.md

# Cluster rules (Tidewater)

Only rules needed in every session, and only things true of *this* cluster.
Portable rules are imported, not copied.

## This cluster

| | |
|---|---|
| Cluster name | `tidewater` (Slurm `ClusterName`). **`$CC_CLUSTER` says `killarney` and is wrong**: the site profile sets it on purpose to borrow Alliance modules. `config.sh` special-cases this. |
| Account | `humanpref-igor`, the user's own project (not the whole group's). It is the only association and also the Slurm default. **Pass `-A humanpref-igor` on every job anyway.** |
| Allocation | **57,000 GPU-hours** (`sshare`'s RawShares). Queue priority follows the part of the allocation that is still unused: usage counts against it with a ~4-day half-life, so burning GPU-hours early pushes every later job further back. `sshare -A humanpref-igor -l` shows the current standing. Billing is `MAX_TRES` per node: **1 CPU = 1/14 of a B200** (1521 vs 21300) and 1 GB RAM ≈ 1/257 of a B200, so a CPU-only job is not free. |
| Devbox | **runs on the login node** (`DEVBOX_LOCAL=1`), not in a job, because a CPU job would cost GPU-hours. It is pinned to the login node it was started on (`tlogin01`/`tlogin02`, no ssh between them). Names: `tidewater-dev` (tunnel), `tidewater-dev-1/2/3` (remote control). |
| Devbox root | `/project/humanpref-igor/shared`, where the projects live. `$HOME` (as `/home/eop`) and `/scratch/eop` come in via `--add-dir`. `$HOME` is `/h/eop`, and `/h` is a symlink to `/home`; claude keys history by the resolved path. |
| VS Code tunnel | **off by default**. Start it only when the user asks: `devbox-up tunnel start`. Stop it when they are done: `devbox-up tunnel stop`. |
| Login node → compute | `sbatch`/`srun` are rejected when the *current directory* is under `/home`. They are accepted from `/project` and `/scratch`, which is where the agents work. |

## What to run in the devbox, and what needs a job

This overrides the shared "the dev box is not a compute node" rule. CPUs are
expensive here (see Allocation), so light CPU work runs in the devbox: unit
tests, data processing, plotting, small scripts. **Allocate a job when it is
actually needed:** anything that needs a GPU, or more than the login node's
per-user cap of **4 CPUs and 12 GB** (a systemd user slice).

That cap is shared by every agent, their tools and the VS Code server.
Exceeding the memory cap OOM-kills something in the slice, possibly an agent.
Measured 2026-11-24 with four agents and VS Code up: about 0.9 GB anonymous
memory; the rest of `memory.current` (~10.5 GB) was reclaimable page cache. So
importing torch, transformers or vLLM (a few GB) is fine here; it only proves
the venv installs, though, since there is no GPU. **Keep model loading and
engine construction out of the devbox**: weights land in process memory.
Check the headroom first with `anon` in
`/sys/fs/cgroup/user.slice/user-$UID.slice/memory.stat`, not `memory.current`.
An interactive `srun` of 3 h or less gets priority.

## GPUs

The only GPU type is B200. Use `--gpus-per-node b200:N` or `--gres gpu:b200:N`.
Any other type string is rejected.

Each node (`tn001-022`) has 8 B200s, 112 CPUs and 2 TB of RAM. Pair one GPU with
**≤14 CPUs and ≤257 GB**. Billing takes the max, so CPUs or memory beyond that
ratio are charged as extra GPUs. `DefMemPerGPU` is 240 GB, but always state
`--mem` or `--mem-per-cpu`: with no memory request a job is given (and billed
for) the whole node's memory.

## GPU efficiency is monitored

The site audits how well allocated GPUs are used, so an idle GPU costs the
allocation **and** counts against the project. For every new GPU job:

1. **Check it once it is past startup** (a few minutes in): `gpuutil <jobid>`
   for Slurm's figure, `gpuutil -l <jobid>` for live `nvidia-smi` inside the
   job. `gpuutil` with no arguments lists your running jobs and those that
   ended in the last 24 h. Steps under 50% are flagged `LOW`.
2. **Check again when it ends** and report utilization and GPU memory to the
   user, next to the results.
3. **If it is low, find out why before scaling up.** Common causes are a
   dataloader or tokenizer bound by CPU (raise workers, pre-tokenize), reading
   from NFS (stage data to `$SLURM_TMPDIR`), a batch too small for a
   B200, waiting on generation or evaluation between steps, or more GPUs than
   the job can use. Propose a fix. Several small runs can often share one GPU
   instead of taking one each.

Slurm samples every 30 s (`gres/gpuutil`, `gres/gpumem` in `TRESUsageInAve`).
That accounting figure is what an audit sees.

## Walltime and partitions

**`--time` is required, and never pass `-p`.** Routing is by walltime, and a
job is eligible for its tier and every tier above it:

| `--time` ≤ | partition | nodes |
|---|---|---|
| `3:00:00` interactive `srun` | `interac` (QoS `interactive`: 2 jobs / 2 nodes per user get priority) | 5 |
| `8:00:00` | `b200_b1` | 22 |
| `3-00:00:00` | `b200_b2` | 16 |
| `7-00:00:00` | `b200_b3` | 10 |
| `14-00:00:00` | `b200_b4` | 4 |

Shorter jobs reach more nodes. Chain long runs with `--dependency=afterok:<id>`.
`sbatch --test-only` shows the resolved partition and start estimate without
queueing.

## Storage

Measured 2026-11-24 with `df` (rerun `diskusage_report` before trusting it):

| Path | Quota | Use |
|---|---|---|
| `/home/eop` (= `/h/eop`) | 50 GB | dotfiles, `~/bin`, `~/.claude`, logs |
| `/scratch/eop` | 2 TB | **caches, tmp, checkpoints, outputs** |
| `/project/humanpref-igor/shared` | 10 TB pool across all of `/project` | **the repos** (devbox root) |
| `/project/humanpref-igor/eop` | same pool | private to this user |

`~/.bashrc` points `HF_HOME`, `UV_CACHE_DIR`, `PIP_CACHE_DIR`, `WANDB_*`,
`TORCH_HOME`, `TRITON_CACHE_DIR`, `XDG_CACHE_HOME`, `ENROOT_*`, … at
`/scratch/eop`, above any interactive guard, so jobs inherit them.

`TMPDIR` is `/scratch/eop/tmp` on the login node. Inside a job it is the job's
own `/tmp`, which is private and node-local (`SLURM_TMPDIR=/tmp`, cleaned at job
end), but only if the job's shell sources `.bashrc`. A bare `sbatch` script
inherits the scratch one, so set `TMPDIR=$SLURM_TMPDIR` in the script when temp
I/O matters. Slurm sets no `TMPDIR` of its own here.

## Local quirks

- **`/etc/profile.d/ai-tool-redirects.sh` moves `XDG_CACHE_HOME` and
  `VSCODE_EXTENSIONS` to `/run/user/$UID`**, a RAM tmpfs that is removed when the
  last ssh session ends. `~/.bashrc` overrides both to scratch because the
  devbox outlives logout.
- **Containers:** pass `--container-image=/scratch/...sqsh --export=NONE`, then
  the image provides the tools (`#!/bin/sh` unless it ships bash). Build images
  with `enroot import` on a login node. Never keep images or outputs in `/tmp`.
- The Alliance module tree is loaded (`module avail`).
- **No CUDA toolkit on the nodes** (`/usr/local/cuda` is absent). On B200,
  vLLM 0.29 picks the FlashInfer attention backend, which JIT-builds kernels
  with `nvcc`, so the engine fails at startup ("Ninja build failed ... nvcc: not
  found"). Pass `attention_backend=` to `vllm.LLM` (the `VLLM_ATTENTION_BACKEND`
  variable no longer exists): `FLASH_ATTN` (FlashAttention 4) for head dim 128
  (Qwen3); **not for Gemma 4**, whose head-dim-256 layers fail an FA4 SM100
  paged-KV assertion at engine start, so `TRITON_ATTN` there. `module load
  cuda/12.9` or `cuda/13.2` would also provide `nvcc` if FlashInfer is wanted.
- **Moving data between clusters:** the site has no Globus endpoint. The user
  runs a personal one, **`tidewater-eop`** (id
  `0d14ff71-e853-11f1-8855-0effc68fde15`). It is **off by default**: start it on
  `tlogin01` when the user wants a transfer (`globus-up`), check with
  `globus-up status`, and `globus-up stop` once that says idle. It reaches only
  `/scratch/eop` and `/project/humanpref-igor/shared`, read-write. Use it
  together with the Alliance clusters' managed collections.
- **Per-connection network throughput is low.** On 2026-11-24, downloads ran at
  ~1 MB/s per TCP connection on the login node and ~2.5 MB/s on a compute node,
  and total throughput grew with the number of connections. For anything big,
  use Globus or several transfers in parallel, never a single `scp`.
