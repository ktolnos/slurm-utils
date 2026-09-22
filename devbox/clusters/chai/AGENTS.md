# Cluster rules (CHAI / ist.berkeley.edu)

@/nas/ucb/eop/slurm-utils/devbox/AGENTS.shared.md

Absolute import, not `@~/...`: `$HOME` is node-local here (see below), so a
`~`-relative path resolves somewhere different depending on the node.

## Storage: $HOME is a trap here

`$HOME` (`/home/<user>`) is **node-local** — rnn's and each compute node's are
different disks. A file written to `$HOME` on rnn does not exist inside a job,
and rnn's `/` regularly sits at 100% full. Never put anything durable there.

| Path | What it is |
|---|---|
| `/nas/ucb/eop` (`$NAS_PERSIST`) | the real home. Repos, checkpoints, wandb runs, agent config. Mounted on every node. |
| `/nas/ttl=60d/eop` (`$NAS_TTL`) | scratch. **Deleted 60 days after creation** — touching a file does *not* extend it. Only things a rerun can rebuild. |
| `$HOME` | node-local, small, often full. Nothing that must survive. |

Caches (`HF_HOME`, `TMPDIR`, `TORCHINDUCTOR_*`, `WANDB_*`, …) are already
redirected to `$NAS_TTL` in `/nas/ucb/eop/.bashrc`, above the interactive
guard, so batch jobs inherit them.

The agent CLIs live on NAS too (`/nas/ucb/eop/.local/bin/{claude,codex}`), and
that is what the devbox launches. `claude` resolves its install root from
`$HOME`, so an update run from a plain login shell installs a *second*, newer
copy under `/home/<user>` that nothing launches — `claude --version` then
disagrees with the version your slots are actually running, and a model
released that morning shows up disabled in the picker. `devbox-up` updates the
NAS copy with `HOME` pointed there; to do it by hand, do the same:

```bash
HOME=/nas/ucb/eop /nas/ucb/eop/.local/bin/claude update
```

## Slurm

Account `chai`. Nodes are 256 cores / ~1 TB RAM / 8 GPUs each.

| | |
|---|---|
| `--partition=main` | default, **not** preemptible |
| `--partition=scavenger --qos=scavenger` | preemptible (`REQUEUE`), priority 0 — only for restartable work |
| `--qos=default` | priority 1, max walltime **3 days**, 32 GPUs/user |
| `--qos=high` | priority 2, max walltime **7 days**, but capped at 8 GPUs / 256 CPUs / 1 TB per user |

The QOS walltime caps (3d / 7d) bind well before the partition's 30-day limit,
so a `--time` over 3 days needs `--qos=high`.

GPUs, since `--gpus=1` gets you whichever is free and they are not equivalent:

- `A100-SXM4-80GB` — airl, sac (fastest interconnect; use for multi-GPU)
- `A100-PCI-80GB` — cirl, rlhf
- `A6000` (48 GB) — ddpg, dqn, gan (gail is down)
- `A4000` (16 GB) — ppo, vae

**Prefer a shard to a whole card.** ddpg/dqn/ppo/vae expose `shard`, and one
shard is **1 GB of GPU memory** (384 shards across 8×48 GB A6000s; 128 across
8×16 GB A4000s). So `--gres=shard:20` asks for 20 GB and packs beside other
jobs, where `--gpus=1` takes a whole card. Check with
`sinfo -N -o '%N %G'`.

## rnn is a "wild west" box, not Slurm compute

rnn is the submit node and has 8×A6000 + 256 cores + 1 TB, but it is **not
part of the Slurm cluster** — nothing schedules it and nothing caps you, so
resource discipline is manual and other people are logged in right now.

- `gpustat` first; **never** use a GPU that already has memory on it.
- Pick explicitly: `CUDA_VISIBLE_DEVICES=3 python ...`. Everything defaults to
  GPU 0, which is how two jobs collide.
- Cap threads: `OMP_NUM_THREADS=<n>`, or torch grabs all 256 cores.
- Anything over ~50% of the machine: say so in `#compute` with an end date.
- Real training belongs in a Slurm job. rnn is for quick checks.

The devbox itself runs here, as a plain background process rather than a job
(`devbox-up config` shows `mode: local`), which is why it costs no allocation.

## Do not crash the controller

This cluster runs **Slurm 23.02.1**, the version where
`scontrol update ... NodeList=` / `ReqNodes=` with an *empty* value kills
`slurmctld` and every queued and running job with it. A `PreToolUse` hook
blocks it, but do not go looking for a way around it: use `ExcNodeList=` to
relax a node constraint, or cancel and resubmit.
