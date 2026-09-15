# Devbox: Claude Code + VS Code tunnel on a Slurm cluster

A long-lived, self-chaining CPU-only Slurm job that hosts **three** Claude Code
sessions and a `code tunnel`, so you get a persistent dev box with three
independently drivable agents, reachable from desktop VS Code, phone or
claude.ai.

```bash
devbox-up            # submit it
devbox-up status     # what is running, which slots, where to reach them
devbox-up attach 2   # tmux, straight to slot 2
devbox-up restart    # apply an edit to config.sh or devbox.sh
devbox-up stop       # stop the chain
devbox-up config     # resolved config + the exact sbatch flags
```

| File | What it is |
|---|---|
| `config.sh` | every knob, and the only place a cluster name appears |
| `devbox.sh` | the job: tmux session, agent slots, tunnel, watchdog |
| `devbox-up` | submit / status / attach / restart / stop, with preflight |
| `pin-session` | `SessionStart` hook that keeps each slot's uuid honest |
| `SETUP_INSTRUCTIONS.md` | step-by-step for a new cluster, written for an agent |
| `AGENTS.shared.md` | portable cluster rules, imported by each cluster's `AGENTS.md` |
| `AGENTS.template.md` | skeleton for a new cluster's `AGENTS.md`, with the blanks marked |
| `clusters/<cluster>/AGENTS.md` | that cluster's real rules, **symlinked** into its session root |
| `settings.json` | the pin hook, symlinked into every root's `.claude/` |

Defaults: root `~/devbox` with `--add-dir $HOME`, tunnel `<cluster>-dev`,
3 slots, 2 cores / 6 GB / no GPU, 3-day walltime, account from
`$SBATCH_ACCOUNT`. Override any of them from the environment
(`DEVBOX_SLOTS="1 2" devbox-up`) or in `config.sh`, whose per-cluster `case`
runs before the defaults — precedence is `environment > stanza > default`.

Where the work is **not** under `$HOME` — repos on `/project`, outputs on
`/scratch` — set `DEVBOX_ADD_DIRS` to the trees the agents need. The root is
implicit and the default list is `$HOME` alone, so an otherwise healthy box
comes up with agents that cannot read the repo they exist to work on.

The root does not have to be `~/devbox`. Pointing `DEVBOX_ROOT` at a project
repo you already trust skips the interactive trust step entirely and keeps any
pre-devbox conversation resumable — at the cost of two symlinks inside that
repo, which is what `.gitignore` is for. Killarney's stanza does exactly this.

## Porting to a new cluster

**`SETUP_INSTRUCTIONS.md` is the step-by-step version of this section** —
follow that when actually doing it; the summary here is for deciding whether
it is worth doing.

Most of this needs nothing: the cluster name comes from `$CC_CLUSTER` (or
`scontrol`), every name and path derives from it, and the sbatch flags are
built in one function. In practice:

1. `git clone` this repo into `$HOME` and put `devbox/` on your `PATH` (or call
   `~/slurm-utils/devbox/devbox-up` directly).
2. Install the two binaries (step 1 below) and authenticate them (step 3).
3. `mkdir ~/devbox`, then **run `claude` there once, interactively, and accept
   the trust dialog.** This is the one step that cannot be automated -- see the
   home-trust section -- and `devbox-up` refuses to submit without it.
4. Symlink the pin hook: `<root>/.claude/settings.json` -> `devbox/settings.json`.
5. Fill in `clusters/<cluster>/AGENTS.md` from `AGENTS.template.md` and symlink
   it to `<root>/AGENTS.md`; the portable rules arrive by importing
   `AGENTS.shared.md` through an **absolute** `@~/slurm-utils/...` path, which a
   file read through a symlink needs.
6. `devbox-up config` to see what it resolved, then `devbox-up`.

Add a stanza to the `case` in `config.sh` only for what a cluster genuinely
gets wrong. From two clusters so far that is: the account, the walltime,
whether `sbatch` is allowed from `/home`, and — where work lives off `$HOME` —
the root and the `--add-dir` list. Everything below marked
**fir-specific** is a note, not a dependency.

## What is NOT portable, and why

- **Workspace trust** is per-machine and per-directory, and must be accepted
  interactively once per cluster (step 3). A batch job cannot answer it.
- **`claude` and `code` logins** are per-machine; do them on the login node.
- **Tunnel and Remote Control names must differ between clusters**, which is why
  they derive from the cluster name -- VS Code tunnel names are globally unique
  per account, and two boxes both called `dev` are indistinguishable in the app.
- **Pinned conversation uuids live under `~/.devbox/<cluster>/`**, not a single
  file in `$HOME`. If a site shares `$HOME` between clusters, a shared pin would
  have two clusters resuming one conversation -- two agents on one conversation
  corrupts it.

## The three slots

`DEVBOX_SLOTS` in `config.sh` drives everything; add a `4` and you get a fourth
agent with no other edits. Names below are for `fir`; on another cluster the
prefix is that cluster's name.

| Slot | Remote Control name | tmux window | pinned conversation |
|---|---|---|---|
| 1 | `fir-dev-1` | `agent1` | `~/.devbox/fir/session-id-1` |
| 2 | `fir-dev-2` | `agent2` | `~/.devbox/fir/session-id-2` |
| 3 | `fir-dev-3` | `agent3` | `~/.devbox/fir/session-id-3` |

Each slot's uuid is minted once and then kept forever, so the same three
conversations come back across `/clear`, node hops and the 3-day chain. Pick one
from any device by its Remote Control name.

**All three share the root `~/devbox`** (plus `--add-dir $HOME`). That is not a
simplification -- it is the only directory whose workspace trust is persisted
(see the home-trust section below), so giving each slot its own root would mean
answering a trust dialog per slot, per job, forever. The trade-off is that the
three agents work in one tree and can collide; `AGENTS.md` tells them so.

Slot 1 was the original single-agent devbox: its id file was migrated into
`~/.devbox/fir/session-id-1` and its Remote Control name went from `fir-dev` to
`fir-dev-1`. Only the tunnel is still plain `fir-dev`.

## 1. The two binaries (login node)

Both self-contained; no node/npm needed.

```bash
curl -fsSL https://claude.ai/install.sh | bash          # -> ~/.local/bin/claude
mkdir -p ~/bin
curl -fsSL 'https://update.code.visualstudio.com/latest/cli-linux-x64/stable' \
  | tar xz -C ~/bin                                     # -> ~/bin/code
```

**Not** the Killarney URL (`code.visualstudio.com/sha/download?...`) -- that
endpoint now returns 404.

## 2. `~/.bashrc`

`sbatch`/`srun` shells are non-login, so they read `.bashrc` and never
`.bash_profile` (which is where `$HOME/bin` was on the PATH). Added near the top:

```bash
case ":$PATH:" in
  *":$HOME/bin:"*) ;;
  *) export PATH="$HOME/bin:$HOME/.local/bin:$PATH" ;;
esac
export VSCODE_CLI_USE_FILE_KEYCHAIN=1
export VSCODE_CLI_DISABLE_KEYCHAIN_ENCRYPT=1
export VSCODE_CLI_DATA_DIR="$HOME/.vscode-cli"
```

The cache redirects (`HF_HOME`, `TMPDIR`, `UV_CACHE_DIR`, `WANDB_*`, ...) were
already there and already point at `/scratch`. Home is 19 TiB here, so nothing
else needed moving. Backup of the original: `~/.bashrc.bak-devbox`.

## 3. Authenticate once, on the login node

```bash
claude                                    # /login, browser flow
~/bin/code tunnel user login --provider github
```

## 4. Start it

```bash
devbox-up
```

`devbox-up` runs a preflight first (binaries present, root exists, trust
accepted) and refuses to submit a second chain if one is already queued.

**fir-specific:** unlike Killarney, fir accepts `sbatch` from a `/home`
directory -- verified. Where it does not, set `DEVBOX_SUBMIT_DIR`.

## 5. Connect

Desktop VS Code -> *Remote Tunnels* extension -> sign in with the same GitHub
account -> pick `fir-dev`. Or open `https://vscode.dev/tunnel/fir-dev`.

From a phone or any browser, the three agents show up separately in Remote
Control as `fir-dev-1`, `fir-dev-2` and `fir-dev-3`; each also prints its own
`https://claude.ai/code/session_...` link in its pane at startup.

To reach the tmux session directly (`ssh <node>` does **not** work here -- see
below):

```bash
JOB=$(squeue -u $USER -h -n claude-dev -t RUNNING -o %i | head -1)
srun --jobid=$JOB --overlap --pty tmux attach -t claude   # then ctrl-b 1/2/3
srun --jobid=$JOB --overlap --pty tmux attach -t claude:agent2   # straight to a slot
```

One-shot look at every slot without attaching:

```bash
for w in agent1 agent2 agent3; do echo "== $w"; \
  srun --jobid=$JOB --overlap tmux capture-pane -t claude:$w -p | tail -5; done
```

---

## Fir-specific notes (examples of what a cluster stanza is for)

- **`.bashrc` overrides `cd` with a function** that runs `activate` and returns
  *its* exit status. In a directory with no venv that status is non-zero, so
  `cd somewhere && thing` silently skips `thing`. It also breaks plain
  `cd ~/bin && curl ...` on the login node. Two consequences: the job script
  uses `cd '$REPO'; claude ...` with a semicolon, never `&&`; and interactively
  you want `builtin cd` in any `&&` chain.
- **`.bashrc` ends with a `cd` into the current project** (`~/jailbraking-teacher-OPD`
  as of 2026-09-15; it changes when the work does). Every tmux window therefore
  starts in that repo, not `$HOME`. The job script re-`cd`s to `$REPO` in the
  `send-keys` line after the shell has finished its init -- do not remove it, or
  the agent opens the wrong history.
- **fir also has `job_container/tmpfs`** (`/tmp` shows up twice in
  `/proc/self/mountinfo`), so the tmux socket race from Killarney applies here
  too. The retry loop and the watchdog are load-bearing. A probe doing four
  immediate `new-session` calls at job start succeeded 4/4, same as Killarney --
  it is rare, not absent.
- **The per-slot watchdog must check that the window exists first.**
  `tmux display-message -p -t claude:agentN '#{pane_current_command}'` does *not*
  fail when window `agentN` is gone -- it silently answers for the session's
  *active* window instead (measured: probing a nonexistent `agent9` returned
  `claude`, the running `agent1`). Without the preceding
  `list-windows | grep -qx`, a dead slot looks healthy forever. The watchdog also
  relaunches only when the pane's foreground process is a bare shell, and only
  after a 180 s grace period: `send-keys` into a pane that *is* running an agent
  would type the command line into that agent's prompt.
- **GPUs: ask for the smallest instance that fits.** The devbox itself has no
  GPU. Get hardware in a separate job. Roughly half the GPU nodes are MIG-
  partitioned, and only three instance sizes exist:

  | Need | Flag |
  |---|---|
  | 10 GB, 1/8 H100 | `--gpus=nvidia_h100_80gb_hbm3_1g.10gb:1` |
  | 20 GB, 2/8 H100 | `--gpus=nvidia_h100_80gb_hbm3_2g.20gb:1` |
  | 40 GB, 3/8 H100 | `--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1` |
  | full 80 GB H100 | `--gpus=h100:1` |
  | n full H100s | `--gpus-per-node=h100:n` (n <= 4) |

  MIG requests reach both the MIG and (for `interac`) the interactive nodes; a
  full-H100 request can only land on the non-MIG half, so a `1g.10gb` job
  generally starts sooner *and* leaves the big cards for jobs that need them.
  GPU nodes are 1 x EPYC 9454, 48 cores, NPS=4 -- pair a MIG with
  `--cpus-per-task=6` (one CCD) rather than a whole socket.
- **Walltime routes the job to a partition**, as on Killarney, but fir exposes
  the ladder explicitly: `b1` 3 h, `b2` 12 h, `b3` 1 d, `b4` 3 d, `b5` 7 d
  (`b6` 28 d is down). A job is eligible for every tier at or above its
  `--time`, so a shorter walltime reaches strictly more nodes. The devbox asks
  for 3 days (`b4`+`b5`); drop to `1-00:00:00` if it ever queues.
  Sub-node CPU jobs land in `cpubase_bycore_*`, whole-node ones in
  `cpubase_bynode_*`.
- **Account suffixes are automatic.** Submit with `--account=def-gigor`; Slurm
  resolves `def-gigor_cpu` / `def-gigor_gpu` from what the job requests.
- **Quota:** `/home` 150 GiB of 19 TiB, `/scratch` 182 GiB of 19 TiB -- both
  roomy. `/project` is the tight one: 2741/9537 GiB but **497K of 500K inodes**,
  so it will fail on file *count* long before space. Keep new work off
  `/project`. `diskusage_report` shows where you stand.

## Two things that only showed up on fir

### `ssh <node>` cannot reach the tmux session -- use `srun --overlap`

`pam_slurm_adopt` does let you `ssh` to a node you hold a job on, but the ssh
session lands in the node's *real* `/tmp`, while the job's tmux socket lives in
the job-private `/tmp` that `job_container/tmpfs` bind-mounts over it. So:

```console
$ ssh fc30354 tmux attach -t claude
error connecting to /tmp/tmux-3146987/default (No such file or directory)
```

Counting `/tmp` in `/proc/self/mountinfo` shows it plainly: **1** over ssh,
**2** inside the job. Attach through a job step instead, which runs in the
job's namespace:

```bash
JOB=$(squeue -u $USER -h -n claude-dev -t RUNNING -o %i | head -1)
srun --jobid=$JOB --overlap --pty tmux attach -t claude
```

`--overlap` is required, or the step blocks waiting on the batch step's
resources. The same form works for one-shot inspection:
`srun --jobid=$JOB --overlap tmux capture-pane -t claude:agent -p`.

### Home-directory trust is session-only, so the devbox is NOT rooted at $HOME

Rooted at `/home/eop`, Claude Code shows the workspace-trust dialog and, when
you accept, **does not persist it** -- `hasTrustDialogAccepted` stays `false` in
`~/.claude.json` forever. It is a deliberate special case for the home
directory, not a bug and not an NFS write failure. From the binary:

> not available to a session rooted at the home directory without a person
> present (home trust is session-only): run Claude Code interactively here and
> accept the trust dialog for that session, or work from a project directory
> you have trusted

and the lookup short-circuits before the stored flag is read:

```js
function uue(e){ if(jVn(e)) return !1; return oe().projects?.[e]?.hasTrustDialogAccepted===!0 }
```

Measured: `/home/eop` stays `false` across repeated accepts; `/home/eop/devbox`
flips to `true` on the first one.

Two consequences for a devbox, which is why the root moved:

1. Every job in the chain would stop at the trust dialog -- every 3 days, plus
   every node failure -- and sit there until a human answered it.
2. Persisted trust also gates project-scoped settings, hooks and MCP servers.
   A home-rooted session silently drops them ("workspace not yet trusted").

The fix is the current config: root at `~/devbox` (trusted once, persisted) and
pull the rest of home in with `--add-dir $HOME`. Roaming still works, nothing
re-prompts. `devbox-up` preflights this by reading `hasTrustDialogAccepted` for
`$DEVBOX_ROOT` out of `~/.claude.json`, so a new cluster fails fast with
instructions instead of hanging on a dialog nobody can see. If you ever move the
root, the history key moves with it
(`~/.claude/projects/<root with non-alphanumerics replaced by ->`), so clear
`~/.devbox/<cluster>/session-id-*` at the same time or the slots resume
conversations that the new root cannot see.

### `/clear` mints a new conversation id, so it drifts off the pin (not cluster-specific)

`/clear` does not reset the pinned conversation -- it starts a *new* one inside
the same process. Measured here: the job launched slot 1 as
`--session-id 5a9ce381-...`, a `/clear` moved the live conversation to
`8ea4b2a3-...`, and the slot's id file still said `5a9ce381`. Left
alone, the next job in the chain would have resumed the abandoned pre-`/clear`
conversation and silently dropped everything done after it.

So `/clear` is safe to use, but the pin has to follow it. The fix is a
`SessionStart` hook, `devbox/pin-session`, that rewrites the slot's id
file with whatever conversation the slot is actually in; the job exports
`DEVBOX_SLOT=N` and `DEVBOX_STATE`, so the hook knows which file to touch, and it no-ops
everywhere else (any session without that variable). Every pin it writes is
logged to `~/.devbox/<cluster>/pins.log`.

The script alone does nothing -- what activates it is the entry in
`~/devbox/.claude/settings.json`:

```json
{"hooks":{"SessionStart":[{"hooks":[
   {"type":"command","command":"$HOME/slurm-utils/devbox/pin-session"}]}]}}
```

Confirm with `cat ~/devbox/.claude/settings.json`; if that file is absent the
hook is not installed and the pins are only as fresh as the last job launch.

Without that hook, re-pin by hand after a `/clear`:

```bash
ls -t ~/.claude/projects/-home-eop-devbox/*.jsonl | head -1 \
  | xargs -n1 basename | sed 's/\.jsonl$//' > ~/.devbox/$CC_CLUSTER/session-id-1
```

## Three things that bit the first real run of `devbox.sh`

Killarney job 5466901 was the first job to actually run the *ported* script --
every earlier green run, on either cluster, was still the pre-port
`~/claude-dev.sh`. Both traps below are portable: neither is a property of
Killarney, and both would have hit fir on its first `devbox-up` too.

### `sbatch` runs a *copy*, so `$BASH_SOURCE` does not find `config.sh`

Slurm stages the batch script into the node's spool directory and executes that,
so inside a job `${BASH_SOURCE[0]}` is
`/cm/local/apps/slurm/var/spool/job<id>/slurm_script` -- and `config.sh` is not
next to it. `source` failed, and because nothing checked, the script ran on with
**every `DEVBOX_*` variable empty**: no slots, no `--add-dir`, no tunnel, a
`mkdir ''`, and a successor `sbatch` that could not open its own script, which
silently broke the chain as well. One line in the log, then 20 minutes of
looking healthy.

`devbox-up` now exports `DEVBOX_DIR` at submit time (it rides the chain on
sbatch's default `--export=ALL`, and the job re-exports it for its successor),
with `scontrol show job $SLURM_JOB_ID`'s `Command=` as the fallback for a job
submitted some other way -- a site defaulting to `--export=NONE` still resolves.
`$BASH_SOURCE` remains only for a direct `bash devbox.sh`. If none of the three
find `config.sh` the job now exits 1 with a FATAL instead of coming up empty.

### Submitting from inside tmux exports `$TMUX`, and tmux then never starts a server

Every shell in a tmux pane has `$TMUX` set, and `sbatch` exports the submitting
environment wholesale -- so restarting the devbox *from inside the devbox* hands
the job a socket path belonging to the old node. A tmux client that sees `$TMUX`
believes a server is already there and reuses that path instead of creating it,
so on a fresh node where `/tmp/tmux-$UID` does not exist yet, every
`new-session` dies with

```
error creating /tmp/tmux-3146987/default (No such file or directory)
```

and still exits 0 -- which is why `start_tmux` confirms with `has-session`
rather than trusting an exit status. Measured on kn057: identical command, dir
absent, fails with `$TMUX` set and succeeds under `env -u TMUX`; `mkdir` the
directory by hand and the running job's watchdog recovers on its next pass.

It is only fatal when no server exists yet, so `devbox-up status` and `attach`
are unaffected -- but `--export=ALL` would have handed the same poisoned `$TMUX`
to every job in the chain, so one restart from a pane breaks the box forever.
`devbox.sh` and `devbox-up submit` both `unset TMUX TMUX_PANE`.

### `@`-imports in `AGENTS.md` need a one-time approval per root

`AGENTS.md` pulls in the shared rules with an absolute
`@~/slurm-utils/devbox/AGENTS.shared.md`, which has to be absolute because the
file is read through a symlink -- but a path outside the root is an *external*
include, and Claude Code gates those behind a dialog:

```
❯ No, disable external imports
  Yes, allow external imports
```

All three slots came up on it and sat there. It looks like a healthy box from
the outside: the job is RUNNING, tmux has its windows, every pane's foreground
process is `claude`, so the watchdog is satisfied -- and not one agent is
reachable.

It is the trust dialog's shape (interactive, per-root, unanswerable from a batch
job) but not its substance: `hasClaudeMdExternalIncludesApproved` is a plain
persisted boolean in `~/.claude.json`'s per-project entry, with no home-directory
special case and no `settings.json` or environment equivalent -- the dialog and
`/config` are the only other ways to set it (checked against 2.1.273). So unlike
trust it can simply be seeded, and `devbox-up` preflight now does, atomically and
idempotently, touching one key of one project because live agents write that file
too. A new cluster never meets the dialog.

## Caveats inherited from Killarney (all still apply)

- **The dev box is not a compute node.** 2 CPU / 6 GB / no GPU. A vLLM import
  alone OOM-kills the job. Use a separate `sbatch`, or `salloc` +
  `srun --jobid=<id> --overlap <cmd>` -- `--overlap` is required.
- **Both VS Code keychain vars are load-bearing.** Without them
  `code tunnel user show` says *logged in* on the login node and *not logged in*
  on every compute node. Cost: `~/.vscode-cli/token.json` holds the GitHub token
  in plaintext (mode 0600, on NFS).
- **Never pipe `code tunnel` through `tee` unattended.** Unauthenticated it
  falls back to an interactive provider picker and redraws forever. Hence the
  `user show` guard and the log-truncation loop.
- **Tunnel won't start, name conflict** -> delete `~/.vscode-cli/tunnel-stable.lock`.
- **Codex remote control does not work here; don't re-add it to the job.** The
  CLI itself runs fine on the box (use the VS Code Codex extension over the
  tunnel), but `codex remote-control start` dies at startup on
  `401 refresh_token_invalidated` -- reproduced 2026-09-15 with a 60-second-old
  `--device-auth` token and nothing else holding the credentials, while plain
  HTTPS to the API answered normally. Its real error only shows up in
  `~/.codex/app-server-daemon/app-server.stderr.log`, not in the "the connection
  is errored" message it prints.
- **Pin the session id; don't use `--continue`**, which means "most recent
  conversation in this directory" and lets any other session hijack the next job
  in the chain. Never run two agents on the same conversation. With three slots
  rooted in the same directory this is no longer a theoretical risk: all three
  would resolve `--continue` to the same conversation and fight over it.
- **History is keyed by absolute path.** `/home/eop/devbox` here -- not through
  a symlink, and not `~/devbox` expanded somewhere else.
- **Set `--autocompact` explicitly** (we run `500k`). Takes `auto` or 100k-1M;
  anything else is a parse error, so a typo fails loudly.
- **Name the Remote Control session** (`--remote-control fir-dev`); auto names
  are hostname-prefixed and change on every node hop.
- **Queue the successor at job start**, not from a USR1 trap -- a trap never
  fires when the node dies or you `scancel`.
- **Stop the chain with the stop file**, not a bare `scancel`: `devbox-up stop`
  writes `~/.devbox/<cluster>/stop` and then cancels. A plain `scancel` just
  hands over to the successor that was queued at job start.
- **Editing the script doesn't change the queued successor** -- Slurm snapshots
  it at submit time. `devbox-up restart` cancels the pending job and resubmits
  with the current code; the running job and its live agents are left alone.
  This also applies to a `git pull`: the chain picks up new code at the next
  restart or node hop, not immediately.
