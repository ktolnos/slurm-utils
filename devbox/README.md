# Devbox: Claude Code + codex + VS Code tunnel on a Slurm cluster

A long-lived, self-chaining CPU-only Slurm job that hosts **three** Claude Code
sessions, a `codex` app-server daemon and a `code tunnel`, so you get a
persistent dev box with three independently drivable agents plus codex sessions
you start from the ChatGPT app, reachable from desktop VS Code, phone,
claude.ai or chatgpt.com.

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
| `devbox.sh` | the job: tmux session, agent slots, tunnel, codex daemon, watchdog |
| `devbox-up` | submit / status / attach / restart / stop, with preflight |
| `pin-session` | `SessionStart` hook that keeps each slot's uuid honest |
| `active-project` | the active project pointer, and the hook that tells sessions about it |
| `slurm-guard` | `PreToolUse` hook: blocks the `scontrol update` that crashes slurmctld |
| `SETUP_INSTRUCTIONS.md` | step-by-step for a new cluster, written for an agent |
| `AGENTS.shared.md` | portable cluster rules, imported by each cluster's `AGENTS.md` |
| `AGENTS.template.md` | skeleton for a new cluster's `AGENTS.md`, with the blanks marked |
| `clusters/<cluster>/AGENTS.md` | that cluster's real rules, **symlinked** into its session root |
| `settings.json` | both `SessionStart` hooks, symlinked into every root's `.claude/` |

Defaults: root `~/devbox` with `--add-dir $HOME`, tunnel `<cluster>-dev`,
3 slots, codex remote control on, 2 cores / 6 GB / no GPU, 3-day walltime,
account from `$SBATCH_ACCOUNT`. Override any of them from the environment
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

## Editing the shared rules

`AGENTS.shared.md` is imported into the instruction context of every session on
every cluster, so treat its length as a running cost and keep maintenance notes
out of it — an `@`-import splices the **whole** file, with no way to include
only part, so anything in there is read by every agent on every request and is
addressed to them, not to you. (An earlier version carried 21 lines of "import
this, don't copy it" preamble, about 17% of the file, aimed at whoever edits it.)

- **Import it, never copy it**, with an absolute `~/` path:
  `@~/slurm-utils/devbox/AGENTS.shared.md`. A cluster's `AGENTS.md` is read
  through a symlink from the session root, so a relative path resolves against
  whichever directory the reader arrived by, not against this repo — and an
  `@`-import that resolves to nothing **fails silently**: no dialog, no error,
  the rules simply absent.
- **An absolute path is an external include**, gated per root by
  `hasClaudeMdExternalIncludesApproved` in `~/.claude.json`. Unapproved, the
  session receives the literal `@…` line as its entire project instructions and
  none of the file — also silently. `devbox-up` preflight seeds the flag; run
  `devbox-up config` once on a cluster before wiring the import.
- **Anything with a number in it that differs per cluster** — account, partition
  ladder, GPU flags, quotas, venv layout — belongs in `clusters/<cluster>/AGENTS.md`.

To check what a session really received, ask one: `claude -p "quote the section
headings in your instructions"`. That is the only way to catch a silent
non-expansion.

## The active project

The session root cannot be the project you are working on: it is fixed by
workspace trust and by conversation history being keyed to an absolute path. So
the agents live in `~/devbox` while the work lives somewhere else — and because
`/clear` starts a conversation with no memory of the last one, every clear used
to mean telling each slot again where the work is.

One pointer fixes that, per cluster:

```bash
active-project ~/my-repo     # set it
active-project               # print it
project                      # shell helper: cd there (or `project <dir>` to set)
```

It is read, not inherited, at three points:

- **A `SessionStart` hook** injects it into every new conversation, including
  each `/clear`, via `additionalContext`. This is the part that matters: the
  environment of an already-running agent process cannot change, but a hook is
  re-read every single time a session starts.
- **`devbox.sh` adds it to `--add-dir`** when launching each slot, so an agent
  can actually read the tree it has just been told to work in.
- **`devbox-up status`** prints it, and the job log has a `project:` line.

Stored in `$DEVBOX_STATE/active-project` (so `~/.devbox/<cluster>/`), because
what you have in flight on one cluster has nothing to do with another's.
`$DEVBOX_PROJECT` overrides the file for one command or one shell — but nothing
exports it, deliberately: `slurm_utils.sh` is sourced by `~/.bashrc`, which
`devbox.sh` also sources, so an exported value would be inherited by every
agent and frozen at launch, and the stale environment would then beat the file
the hook reads.

A pointer at a directory that no longer exists is reported as such rather than
passed over in silence — a stale pointer does its damage precisely when nobody
notices it.

**Cost.** The injected line is deliberately one sentence, ~30 tokens, because it
rides along on every request for the life of the conversation; the reasoning
behind it lives in `AGENTS.shared.md`, which is loaded anyway, instead of being
paid for twice. An **idle slot costs nothing at all**: a session that is never
spoken to makes no model calls, and measured on 2026-09-15 a live session left
sitting at an empty prompt for 75 s had not even written a transcript file. The
hook only runs a local script; the text it returns is not billed until you
actually send something.

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

## 1. The three binaries (login node)

All self-contained; no node/npm needed.

```bash
curl -fsSL https://claude.ai/install.sh | bash          # -> ~/.local/bin/claude
mkdir -p ~/bin
curl -fsSL 'https://update.code.visualstudio.com/latest/cli-linux-x64/stable' \
  | tar xz -C ~/bin                                     # -> ~/bin/code
curl -fsSL https://chatgpt.com/codex/install.sh | sh    # -> ~/.local/bin/codex
```

**Not** the Killarney URL (`code.visualstudio.com/sha/download?...`) -- that
endpoint now returns 404.

Codex installs as a statically linked musl release under
`~/.codex/packages/standalone/releases/<version>-x86_64-unknown-linux-musl`,
with `~/.local/bin/codex` -> `packages/standalone/current` -> that release, and
it keeps itself updated from the same URL (`codex update` forces it). Point
`DEVBOX_CODEX_BIN` at the symlink, never at a release path.

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
codex login                               # ChatGPT account; needs MFA enabled
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

Codex is reached from the ChatGPT app instead, where the box appears under the
**current node's hostname** (`fc30354`) rather than `fir-dev`, and sessions are
created there rather than waiting in slots -- see the codex section below.

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

## Codex remote control

`codex` is the third binary, and it joins the box differently from the other
two: **one app-server daemon for the whole devbox, not one per slot.** Sessions
are created from the ChatGPT app and run inside that daemon; `codex agents`
lists them from a shell on the box. So there is no name to choose, no
conversation id to pin, and nothing in tmux -- `DEVBOX_SLOTS` has no codex
equivalent and does not need one.

`devbox.sh` starts it with one call:

```bash
codex remote-control start --json
```

which forks the daemon, prints a JSON line and exits in ~0.3 s. It is
idempotent -- a second call answers `"daemon":{"status":"alreadyRunning"}` -- and
against a daemon that is already up it re-enables remote control, so the *same*
call is both the start path and the repair path. The watchdog therefore just
re-issues it every 5 min, and logs only when the state changes:

```
codex: remote control connected as fc30354 (env env_e_6aac66ec...)
```

Set `DEVBOX_CODEX=0` to leave codex out of the box entirely.

### The app shows the node name, and that is fine

The `serverName` the app displays is `gethostname()`, not `$DEVBOX_NAME`: it is
`fc30354`, it changes on every node hop, and nothing overrides it -- there is no
config key for it and `HOSTNAME=fir-dev codex remote-control start` still
reports the node.

What does *not* change is the identity behind that label. The enrollment
(`server_id` + `environment_id`) is persisted under `~/.codex` and reused across
nodes: the box enrolled on `fc30355` on 2026-09-17 and came back up on
`fc30354` on 2026-09-21 with the same `srv_e_...` **and** the same `env_e_...`
(`reusing persisted remote control enrollment` in `~/.codex/logs_2.sqlite`). One
stable environment in the app, wearing whatever node name it woke up on.

### Never run `codex remote-control start` on the login node

Measured: the control socket is
`~/.codex/app-server-control/app-server-control.sock` and the pid file is
`app-server-daemon/app-server.pid` beside it -- both on **shared** `$HOME`,
while the pid inside the file (plus a `processStartTime`) only means anything on
the node that wrote it. Codex does recover from a stale one: this box
bootstrapped cleanly today over a pid file left by the 2026-09-17 job on another
node.

Which is exactly the hazard. A codex started on the login node while the box is
up sees a pid it cannot verify against its own process table, and the recovery
path that makes node hops work is then free to take the socket out from under
the daemon in the live job. Not worth reproducing to find out how gracefully.

That is why `devbox-up` never starts codex -- only `devbox.sh` does -- and why
`devbox-up status` reads the last `codex:` line back out of the job log instead
of asking codex anything. `codex login status` touches no daemon and is safe
anywhere, which is what preflight uses; plain `codex` on the login node is not,
while the box is up.

### The processes, and the memory cap

Two of them, both reparented to init but both still inside the job's cgroup
(checked in `/proc/<pid>/cgroup`), so they die with the job and never leak onto
the node:

| Process | RSS | What it is |
|---|---|---|
| `codex app-server --remote-control --listen unix://` | ~110 MB | the daemon |
| `codex app-server daemon pid-update-loop` | ~25 MB | its auto-updater sidecar |

~135 MB against the box's 6 GiB, next to ~800 MB for three Claude sessions --
so codex costs about a sixth of one agent slot. Note that `remote-control stop`
stops the daemon and leaves the updater running.

That updater reflows `~/.codex/packages/standalone/current`, which is why
`DEVBOX_CODEX_BIN` points at the `~/.local/bin/codex` symlink (itself a link
into `current`) rather than at a versioned release path that the next update
would strand.

**A codex session is an agent doing real work, so every devbox rule applies to
it unchanged**: no training, no inference, no test suites. The memory cap is per
*job*, so a codex session that imports vLLM takes the three Claude slots and the
tunnel down with it.

### The auth failure to expect

The first attempt at this came up with the daemon running and remote control
never connecting, and the reason was only ever written to
`~/.codex/app-server-daemon/app-server.stderr.log`:

```
ERROR codex_login::auth::manager: Failed to refresh token: 401 Unauthorized:
  "Your refresh token has been invalidated. Please try signing in again."
  code: refresh_token_invalidated
```

The fix was enabling MFA on the account and re-running `codex login` on the
login node. Note that `codex login status` still reports `Logged in using
ChatGPT` in that state, so it is not a usable health check -- the `"status"`
field of `remote-control start` is the only honest signal, and that is what
`launch_codex` tests.

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
