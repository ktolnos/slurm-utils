# Cluster rules (Fir)

Only fir's own facts. The portable rules are imported below; deeper detail —
full partition ladder, MIG node layout, queue notes — is in `~/devbox/CLUSTER.md`,
and the devbox internals are in `~/slurm-utils/devbox/README.md`. Read either
only if you need it.

@~/slurm-utils/devbox/AGENTS.shared.md

## This cluster

| | |
|---|---|
| Cluster name (`$CC_CLUSTER`) | `fir` |
| Account | `def-gigor`, already in `$SBATCH_ACCOUNT`. Slurm resolves the `_cpu`/`_gpu` suffix itself |
| Devbox names | `fir-dev` (tunnel), `fir-dev-1/2/3` (remote control) |
| `sbatch` from `/home` | works here |
| Partition | don't name one; it routes on `--time` and on what you request |
| Python | `uv run`, below |

## Python: `uv run`

```bash
uv run python src/train.py         # uses the repo's .venv; nothing to activate
uv run pytest tests/               # in a job, not on the dev box
uv pip install -r requirements.txt
uv venv                            # repo has no .venv yet
```

`uv` is `~/.local/bin/uv`. One exception: `~/reasoning-distillation` is not set
up for `uv run` — see `CLUSTER.md`.

## GPUs: ask for the smallest that fits

Roughly half the GPU nodes are MIG-partitioned, and a MIG request reaches
strictly more nodes than a full card, so it usually starts sooner *and* leaves
the big cards alone.

| Need | Flag |
|---|---|
| 10 GB, 1/8 H100 | `--gpus=nvidia_h100_80gb_hbm3_1g.10gb:1` |
| 20 GB, 2/8 H100 | `--gpus=nvidia_h100_80gb_hbm3_2g.20gb:1` |
| 40 GB, 3/8 H100 | `--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1` |
| full 80 GB H100 | `--gpus=h100:1` |
| n full H100s, one node | `--gpus-per-node=h100:n` (n ≤ 4) |

GPU nodes are 1 × EPYC 9454: 48 cores / 1125 GB / 4 H100. Stay near the ratio —
6 cores (one CCD) per MIG, 12 per full GPU. CPU nodes are 192 cores / 750 GB,
where `--cpus-per-task=8` keeps a task inside one CCD.

## Walltime and partitions

Walltime routes the job: `3:00:00` reaches 60 MIG nodes, `7-00:00:00` only 20,
so ask for the shortest you can. Site policy is minimum **1 h** per job (5 min
for test jobs), maximum **7 d**. No crontab. Compute nodes have full internet
access. Sub-node CPU jobs land in `cpubase_bycore_*`, whole-node ones in
`cpubase_bynode_*`.

## Storage: fir's paths and quotas

| Path | Quota (measured 2026-09-15) | Use |
|---|---|---|
| `/home/eop` | 19 TB, 151 GB used | repos, dotfiles, `~/bin`, logs — roomy, backed up daily |
| `/scratch/eop` (`~/scratch`) | 19 TB, 182 GB used | caches, checkpoints, outputs — **no backup, auto-purged** |
| `/project/def-gigor` | 9537 GB, 2741 GB used — but **498K/500K inodes** | shared group data; avoid |

`/project` is the tight one and will fail on file *count* long before space, and
it is shared with the rest of `def-gigor`, so keep new work off it. Run
`diskusage_report` rather than trusting this table.

## Local quirks

- **Work lives in `$HOME`**: `~/Reward-tampering`, `~/reasoning-distillation`,
  `~/jailbraking-teacher-OPD`, `~/devbox`.
- **`~/.bashrc` starts a shell in the active project**, not `$HOME` — it `cd`s
  to whatever `active-project` says. So a fresh tmux window and every job step
  begin there.
- **`/tmp` is job-private** (`job_container/tmpfs`), so `ssh <node> tmux attach`
  cannot see the devbox's tmux socket, and the socket race at job start is real.
  Use `devbox-up attach [slot]`.
- **Nothing overrides `cd`** (removed from `slurm_utils.sh` on 2026-09-15). Run
  `activate` when you want a venv; `uv run` needs none.
