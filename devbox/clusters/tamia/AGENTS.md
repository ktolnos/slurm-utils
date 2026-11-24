@/home/e/eop/slurm-utils/devbox/AGENTS.shared.md

# Cluster rules (tamIA)

Only rules needed in every session, and only things true of *this* cluster.
Portable rules are imported, not copied.

## This cluster

| | |
|---|---|
| Cluster name (`$CC_CLUSTER`) | `tamia` (Calcul Québec / PAICE) |
| Account | **`--account=aip-gigor`, always** — the only association, and `$SBATCH_ACCOUNT` is not set |
| Devbox names | `tamia-dev` (tunnel), `tamia-dev-1/2/3` (remote control) |
| `sbatch` from `/home` | works here |
| Partition | don't name one; it routes on `--time` and on what you request |
| Python | `uv` 0.13.1 is installed (`~/bin/uv`); project venvs come from the project's `uv.lock`. The Alliance wheelhouse (vLLM ≤ 0.25 as of 2026-10-15) is the fallback |

## GPUs: whole nodes only

A GPU job gets an entire node. Anything smaller is **rejected** at submit
("The h100 GPUs are only allocated by node").

| Need | Flag | Node |
|---|---|---|
| 4× H100 80 GB | `--gpus-per-node=h100:4` | 48 cores, 500 GB (53 nodes) |
| 8× H200 141 GB | `--gpus-per-node=h200:8` | 64 cores, 1 TB (12 nodes) |

Ask for the node's cores and memory too; nobody else can use them. Because
the smallest GPU job is a whole node, and FairShare is charged for the whole
node, **do debugging and small experiments on CPU**, or pack several runs into
one node job — never request a node for a single-GPU test without asking.

FairShare is almost all of a job's priority here (weight 5,000,000 vs. 2,500
for age), shared across all of `aip-gigor`, and **the budget is small**.
Measured 2026-10-16 with `sshare -l` and `scontrol show config`:

- Shares are **not** equal: `aip-gigor` has the minimum 1150 raw shares
  (NormShares 0.00028, i.e. 0.028% of the cluster), like most `aip-*` groups;
  some have 20k–41k (18–35× more), presumably via an allocation.
- Usage decays with a **7-day half-life** (`PriorityDecayHalfLife`), Fair Tree.
- "Fair" usage (LevelFS 1) is ≈ 0.5 node-hours/day. Active groups with the same
  1150 shares sit at LevelFS 0.2–0.3, which in steady state is
  **≈ 2 H100-node-hours/day (≈ 8 GPU-hours)**.
- Observed: ~30 node-hours on 2026-10-15 took the user's FairShare from 0.29 to
  0.056, and the next full-node job's estimated start was ~32 h out. Getting
  back to the peer level takes ~3 days with no usage.

So treat Tamia as occasional capacity, not a daily resource: put sustained GPU
work on another cluster, ask before more than a few node-hours here, and
request the shortest `--time` (a 3 h job backfills far sooner than 12 h).
(An earlier estimate of ~270 GPU-hours/day assumed equal shares; it was wrong.)

## Walltime and partitions

`b1` 3 h, `b2` 12 h, `b3` 1 d — **1 day is the maximum for any job**, so
anything longer must checkpoint and resubmit. Same ladder for
`gpubase_bynode_*`, `cpubase_bycore_*` and `cpubase_bynode_*`. A job is
eligible for every tier at or above its `--time`, so ask for the shortest you
can. `*_interac` (6 h, QoS `interac`, **one job per user**) serves `salloc`.

CPU nodes: 8 × 64 cores / 500 GB, `cpubase_bycore_*` shares them by core.

## No internet on compute nodes — use the site proxy

Compute nodes have no direct internet. The site's squid passes general HTTPS
(Anthropic, GitHub, PyPI, Hugging Face, wandb all verified 2026-10-14):

```bash
module load httpproxy/1.0     # sets http_proxy/https_proxy to squid.tamia.ecpia.ca:3128
```

The devbox job exports this itself (`DEVBOX_HTTP_PROXY` in `config.sh`), and
`sbatch` passes the environment on, so jobs submitted from a devbox pane
already have it. A job submitted from a fresh login shell needs the
`module load` in its script.

## Storage

Measured with `diskusage_report` on 2026-10-14 — rerun it rather than trusting
this table.

| Path | Space | Inodes | Use |
|---|---|---|---|
| `/home/e/eop` | 1.6 / 25 GiB | 10K / 250K | repos, dotfiles, `~/bin`, `~/.claude`, logs |
| `/scratch/e/eop` (`~/links/scratch`) | 0 / 500 GiB | 0 / 500K | **caches, checkpoints, outputs** — not backed up, purged |
| `/project/6101830` (`~/links/projects/aip-gigor`) | 0 / 500 GiB | 14 / 250K | shared group data |

`/home` is only 25 GiB: keep datasets, caches and checkpoints on scratch.
`~/.bashrc` points `HF_HOME`, `UV_CACHE_DIR`, `PIP_CACHE_DIR` and `WANDB_*`
at `/scratch/e/eop/` (added 2026-10-15). `UV_LINK_MODE=copy` because the cache
and the venvs are on different filesystems.

## Local quirks

- **A uv venv on `/usr/bin/python3.12` (Ubuntu's) can't compile with the
  Alliance toolchain** that StdEnv puts first on `PATH`: source builds and
  Triton/inductor's runtime compiles fail ("`pyconfig.h`: No such file", or
  "`GLIBC_2.38` not found" from the Alliance `ld`). In builds and job scripts,
  use `export PATH=/usr/bin:$PATH CC=/usr/bin/gcc CXX=/usr/bin/g++`
  (found 2026-10-15).
- **H100s and H200s are billed differently** (36,696 vs 40,240 per GPU), so an
  8×H200 node costs about 2.2× an H100 node in FairShare.
