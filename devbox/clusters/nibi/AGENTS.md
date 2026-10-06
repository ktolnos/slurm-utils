@/home/eop/slurm-utils/devbox/AGENTS.shared.md

# Cluster rules (Nibi)

Only rules needed in every session, and only things true of *this* cluster.
Portable rules are imported, not copied.

## This cluster

| | |
|---|---|
| Cluster name (`$CC_CLUSTER`) | `nibi` (SHARCNET) |
| Accounts | **two, and which one depends on the job** — see below |
| Devbox names | `nibi-dev` (tunnel), `nibi-dev-1/2/3` (remote control) |
| `sbatch` from `/home` | works here |
| Partition | don't name one; it routes on `--time` and on what you request |

## Accounts: GPU jobs on `rrg-gigor`, everything else on `def-gigor`

| Job | Account | Why |
|---|---|---|
| requests a GPU | `rrg-gigor` (the default: `$SBATCH_ACCOUNT`) | the group's RAC allocation |
| CPU only | **`--account=def-gigor`, always** | `rrg-gigor` has only a `_gpu` association |

`$SBATCH_ACCOUNT` is `rrg-gigor` in every shell and job, set by the site
profile. A CPU-only job submitted without `--account=def-gigor` is **rejected**
with "You are not a member of the specified account rrg-gigor" — that message
means a missing `--account`, not a membership problem. Slurm adds the
`_cpu`/`_gpu` suffix itself; never write it. The devbox itself runs on
`def-gigor` (`config.sh`).

## GPUs: ask for the smallest that fits

The type must always be named — a bare `--gpus=1` is rejected.

| Need | Flag |
|---|---|
| 10 GB, 1/8 H100 | `--gpus=nvidia_h100_80gb_hbm3_1g.10gb:1` |
| 20 GB, 2/8 H100 | `--gpus=nvidia_h100_80gb_hbm3_2g.20gb:1` |
| 40 GB, 3/8 H100 | `--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1` |
| full 80 GB H100 | `--gpus=h100:1` |
| n full H100s, one node | `--gpus-per-node=h100:n` (n ≤ 8) |

Also present but rarely the right choice: `t4` (16 GB, 7 nodes), `a5000`,
`a100` (1 node), `mi300a` (AMD — CUDA code will not run).

Node shapes (`sinfo`, 2026-10-05): H100 nodes are 112 cores / 2 TB / 8 GPUs,
so stay near **14 cores per full H100** and ~3 per 1g.10gb slice. 8 H100 nodes
are MIG-partitioned. CPU nodes are 192 cores / 750 GB (712 of them).

## Walltime and partitions

`b1` 3 h, `b2` 12 h, `b3` 1 d, `b4` 3 d, `b5` 7 d — the same ladder for
`cpubase_bycore_*`, `cpubase_bynode_*` and `gpubase_bygpu_*`. A job is eligible
for every tier at or above its `--time`, so ask for the shortest you can.
`*_interac` partitions (8 h) serve `salloc`.

## Storage

Measured with `diskusage_report` on 2026-10-05 — rerun it rather than trusting
this table.

| Path | Space | Inodes | Use |
|---|---|---|---|
| `/home/eop` | 21 / 50 GiB | 105K / 500K | repos, dotfiles, `~/bin`, `~/.claude`, logs |
| `/scratch/eop` (`~/scratch`) | 409 / 1024 GiB | 138K / 1M | **caches, checkpoints, outputs** — not backed up, purged |
| `/project/def-gigor` (`~/projects/def-gigor`) | **8845 / 9313 GiB** | 1.4M / 2M | shared group data — nearly full |

**Do not write to `/project`**: it is shared with the rest of `def-gigor` and
is within 500 GiB of its quota. `/home` is small (50 GiB) — keep large files on
scratch.

## Local quirks

- **Cache redirects** (`HF_HOME`, `TMPDIR`, `TORCHINDUCTOR_*`, …) point at
  `/scratch/eop` in `~/.bashrc`, so jobs inherit them.
- **`pip` is a shell function** in `~/.bashrc` that calls `uv pip` when `uv` is
  on `PATH` (`~/.local/bin/uv`).
