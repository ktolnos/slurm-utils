# Setting up a devbox on a new cluster

For an agent doing this on a cluster where the devbox has never run. Work
through it in order; each step says how to verify it before moving on. Expect
~20 minutes, most of it waiting on downloads and the queue.

**Read `README.md` first** if you need to know what any of this is for. This
file is the procedure; that one is the explanation.

**Two steps need a human** and cannot be worked around, so surface them early
rather than at the end (step 3 and step 5). Ask for both at once.

---

## 0. Check where you are

```bash
echo "$CC_CLUSTER"                 # Alliance clusters set this
scontrol show config | grep -i '^ClusterName'
echo "$SBATCH_ACCOUNT"             # may already hold the account
squeue -u $USER -n claude-dev      # is a devbox already running here?
```

If a devbox is already queued or running, stop: you are not setting up, you are
changing an existing one. Use `devbox-up status` and `devbox-up restart`.

If the cluster name comes back empty from every source, everything derived from
it (tunnel name, state directory) would be wrong, and `config.sh` deliberately
hard-fails. Set `DEVBOX_CLUSTER` explicitly instead of working around it.

## 1. Clone the repo and put `devbox-up` on PATH

```bash
git clone https://github.com/ktolnos/slurm-utils.git ~/slurm-utils
grep -q 'slurm_utils.sh' ~/.bashrc || echo 'source ~/slurm-utils/slurm_utils.sh' >> ~/.bashrc
source ~/slurm-utils/slurm_utils.sh
command -v devbox-up            # -> ~/slurm-utils/devbox/devbox-up
```

`slurm_utils.sh` adds `devbox/` to `PATH`, so nothing else needs installing.

## 2. Install the three binaries (on the login node)

All are self-contained; no node or npm needed.

```bash
curl -fsSL https://claude.ai/install.sh | bash          # -> ~/.local/bin/claude
mkdir -p ~/bin
curl -fsSL 'https://update.code.visualstudio.com/latest/cli-linux-x64/stable' \
  | tar xz -C ~/bin                                     # -> ~/bin/code
curl -fsSL https://chatgpt.com/codex/install.sh | sh    # -> ~/.local/bin/codex
claude --version && ~/bin/code --version && codex --version
```

Codex is optional -- `DEVBOX_CODEX=0` skips it and the box comes up without it.
It lands as a static musl binary in `~/.codex/packages/standalone/releases/` with
`~/.local/bin/codex` symlinked through `packages/standalone/current`, and its own
updater reflows that symlink, so leave `DEVBOX_CODEX_BIN` pointing at the
symlink.

Add to `~/.bashrc`, near the top and **above any interactive guard** so batch
jobs inherit it:

```bash
case ":$PATH:" in
  *":$HOME/bin:"*) ;;
  *) export PATH="$HOME/bin:$HOME/.local/bin:$PATH" ;;
esac
export VSCODE_CLI_USE_FILE_KEYCHAIN=1
export VSCODE_CLI_DISABLE_KEYCHAIN_ENCRYPT=1
export VSCODE_CLI_DATA_DIR="$HOME/.vscode-cli"
```

Both keychain variables are load-bearing. Without them `code tunnel user show`
reports *logged in* on the login node and *not logged in* on every compute node,
so the tunnel silently never starts. The cost is that
`~/.vscode-cli/token.json` holds the GitHub token in plaintext (mode 0600).

Verify: `sbatch` a one-line job that runs `which claude code` and check the
output file. A non-login shell reads `.bashrc` and never `.bash_profile`, which
is a common reason a binary is found interactively and not in a job.

## 3. HUMAN STEP: authenticate all three tools

```bash
claude                                      # /login, browser flow
~/bin/code tunnel user login --provider github
codex login                                 # ChatGPT account
```

You cannot do this for them; it is an account credential flow. Ask, then verify:

```bash
~/bin/code tunnel user show | grep -i 'logged in with'
codex login status                          # "Logged in using ChatGPT"
```

**Codex remote control needs MFA enabled on the ChatGPT account.** Without it the
daemon starts and the remote-control websocket never connects, failing with
`refresh_token_invalidated` in `~/.codex/app-server-daemon/app-server.stderr.log`
while `codex login status` keeps claiming it is logged in. Ask them to enable MFA
and run `codex login` again. Do **not** run `codex remote-control start` here to
test it -- that is a login-node daemon on a shared-`$HOME` socket; see the codex
section of `README.md`. The real check is the `codex:` line in the job log once
the box is up.

## 4. Create the session root

```bash
mkdir -p ~/devbox ~/logs
```

The root must **not** be `$HOME`. Home-directory workspace trust is
session-only and is never persisted — a home-rooted job stops at the trust
dialog on every node hop forever, and project settings and hooks are silently
dropped. `$HOME` is reachable anyway: the job passes `--add-dir $HOME`.

## 5. HUMAN STEP: accept the trust dialog once

```bash
cd ~/devbox && claude      # accept the workspace-trust prompt, then /exit
```

This is the other step a batch job cannot do — nobody is there to answer the
dialog, and the job would sit on it for its whole walltime. `devbox-up` refuses
to submit until it sees `hasTrustDialogAccepted` for the root in
`~/.claude.json`, so a missed step fails fast with instructions rather than
hanging.

Verify:

```bash
devbox-up config | tail -3        # preflight must say "ok"
```

## 6. Install the pin hook

Symlink the shared one; don't write a second copy:

```bash
ROOT=$(devbox-up config | awk '/^root/{print $3}')
mkdir -p "$ROOT/.claude"
ln -sfn ~/slurm-utils/devbox/settings.json "$ROOT/.claude/settings.json"
python3 -m json.tool "$ROOT/.claude/settings.json" >/dev/null && echo ok
```

If that root needs Claude Code settings of its own, put them in
`.claude/settings.local.json` next to the symlink — Claude Code merges it over
the shared file, so the two do not fight.

Without this, `/clear` in a slot silently orphans that slot's conversation: it
mints a new conversation id inside the same process, the pinned id goes stale,
and the next job in the chain resumes the abandoned pre-`/clear` conversation.
The hook rewrites the slot's id file with whatever conversation it is actually
in, and no-ops in any session that is not a devbox slot.

Note this edits the agent's own hook configuration. If you are an agent running
under a permission policy, expect that write to need approval — ask rather than
routing around it.

## 6b. Install the slurm guard

This one goes in `~/.claude/settings.json` — **user** level, not the root's
`.claude/` — so it covers every session under this `$HOME`, not only the devbox
slots. Merge the `hooks` key into whatever is already in that file; do not
overwrite it:

```json
{"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command",
  "command": "$HOME/slurm-utils/devbox/slurm-guard",
  "timeout": 10, "statusMessage": "slurm guard"}]}]}}
```

Verify, with a command that is only an `echo` and so is harmless if the hook is
*not* working — it must come back denied:

```bash
echo 'scontrol update JobId=999999 NodeList='
```

It blocks `scontrol update ... NodeList=`/`ReqNodes=` with an empty value, which
crashes `slurmctld` for the whole cluster on Slurm 23.02.1 (2026-08-19, and again
2026-09-18 from a workflow subagent). Install it on every cluster whatever the
version — one regex per Bash call against a failure nobody can recover from
inside a session. `ExcNodeList=` is deliberately still allowed, empty or not,
since that is the field you are meant to use instead.

It is a hook rather than a rule in `AGENTS.md`, on purpose: a prompt-level rule
is advice a subagent can reason past, and a subagent is what caused the second
crash. Nothing is added to any `AGENTS.md` or `CLAUDE.md` for this.

## 7. Install `AGENTS.md`

The cluster's rules live in git under `clusters/<cluster>/AGENTS.md` and are
**symlinked** into the root — a copy in the root drifts from the repo silently.

```bash
mkdir -p ~/slurm-utils/devbox/clusters/$CC_CLUSTER
cp ~/slurm-utils/devbox/AGENTS.template.md ~/slurm-utils/devbox/clusters/$CC_CLUSTER/AGENTS.md
# fill in the TODOs, then:
ln -sfn ~/slurm-utils/devbox/clusters/$CC_CLUSTER/AGENTS.md "$ROOT/AGENTS.md"
echo '@AGENTS.md' > "$ROOT/CLAUDE.md"    # or add that line to an existing CLAUDE.md
```

Import the portable rules with the **absolute** path
`@~/slurm-utils/devbox/AGENTS.shared.md`, not a relative one: the file is read
through a symlink, so a relative import resolves against the wrong directory.
Leave a TODO rather than guessing a number you have not measured.

That absolute path points outside the root, which makes it an **external**
include -- gated behind a one-time per-root dialog that a batch job has nobody
to answer, so without it every slot comes up parked on *"Yes, allow external
imports"*. `devbox-up` preflight approves it for this root before submitting, so
there is nothing to do here; it is called out only because the failure looks
like the agents started fine.

Where the root is an existing project repo, git-ignore the two symlinks — they
point into `$HOME` and mean nothing in a fresh clone:

```
/AGENTS.md
/.claude/settings.json
.claude/settings.local.json
```

**If the work is not under `$HOME`,** set `DEVBOX_ADD_DIRS` in this cluster's
`config.sh` stanza to the trees the agents need (`"$HOME /scratch/$USER"`, say).
The root is implicit; the default list is `$HOME` alone, and an agent that cannot
read its own repo is useless. `devbox-up config` does not print it — check the
`add-dir:` line in the job log, or the dry run at the end of this file.

## 7b. Point at the active project

```bash
active-project ~/the-repo-being-worked-on
active-project                     # verify
```

Optionally make the shell follow the same pointer, so there is one source of
truth rather than a path hardcoded in `~/.bashrc` too:

```bash
__devbox_project=$("$HOME/slurm-utils/devbox/active-project" 2>/dev/null)
[ -d "${__devbox_project:-}" ] && cd "$__devbox_project"
unset __devbox_project
```

Test the directory, then `cd` — never `cd "$p" || cd ~`. Where `.bashrc`
overrides `cd`, the override returns the *other* command's status, so a fallback
chained on `cd` fires even when the `cd` succeeded.

The root is not the work (trust and history pin it), and `/clear` starts a
conversation that remembers nothing — so without this, every clear means telling
each slot again where the code is. A `SessionStart` hook injects the pointer into
each new conversation and `devbox.sh` passes it to `--add-dir`. Skip it if you
genuinely do not know yet; sessions then start with no project line rather than a
wrong one, and `active-project <dir>` can be set at any time.

## 8. Launch

```bash
devbox-up               # preflight, then submit
devbox-up status        # jobs, slots, tunnel URL, live tmux windows
```

`status` shows nothing under `windows` until the job actually starts. Watch
`~/logs/claude-dev-<jobid>.out` for the slot lines; each should print its
Remote Control name and either `--session-id` (first run) or `--resume`.

Then connect: desktop VS Code → *Remote Tunnels* → the `<cluster>-dev` tunnel,
or `https://vscode.dev/tunnel/<cluster>-dev`. The agents appear separately in
Remote Control as `<cluster>-dev-1`, `-2`, `-3`.

## 9. Record what you learned

If this cluster needed anything the defaults got wrong, add a stanza to the
`case` in `config.sh` — account, walltime, `DEVBOX_SUBMIT_DIR`, `DEVBOX_ROOT`,
`DEVBOX_ADD_DIRS`, a partition — and keep it to values, not logic. The `case`
runs **before** the portable defaults, so precedence is
`environment > stanza > default`; write every assignment as `${VAR:-...}` or you
break the environment override. If you discover something true of *every*
cluster, put it in `AGENTS.shared.md` so the other clusters get it too.

Commit and push both, or the next cluster starts from where you did.

---

## If it does not come up

Diagnose in this order; each rules out the one below.

| Symptom | Check |
|---|---|
| `devbox-up` refuses to submit | it printed the reason — trust, a missing binary, or an existing chain |
| Job queued but never starts | `squeue -u $USER --start`; try a shorter `DEVBOX_TIME`, which reaches more partitions |
| Job runs, exits at once | `~/logs/claude-dev-<jobid>.out`; a rejected sbatch flag or a missing `$DEVBOX_ROOT` |
| Job runs, no tmux session | the socket race — the watchdog retries; if it never recovers, check whether `/tmp` is job-private |
| tmux is up, agent panes sit at a shell | the agent exited; capture the pane (`devbox-up attach N`) and look for the trust dialog or a login prompt |
| Agents fine, no tunnel | `code tunnel user show` on a *compute* node, not the login node — this is the keychain-variable symptom |
| Everything fine, wrong conversations | the slot id files in `~/.devbox/<cluster>/`; never let two sessions share one uuid |

Things that have wasted time before, worth knowing up front:

- **A queued job runs the code it was submitted with.** Slurm snapshots the
  script, so editing `devbox.sh` or `config.sh` changes nothing until
  `devbox-up restart`. Verify a change took effect by the *next* job, not the
  running one.
- **`cd x && y` can silently skip `y`** where `.bashrc` overrides `cd` with a
  function returning another command's status. Use `;`. This has produced a
  patch that reported success and did nothing, with `bash -n` then validating
  the *old* file.
- **Never pipe `code tunnel` through `tee` unattended.** Unauthenticated it
  falls back to an interactive picker and redraws forever — hundreds of MB in
  under a minute. Hence the auth guard and the log truncation in `devbox.sh`.
- **A daemon being alive does not mean it works.** Prefer a health check that
  proves function over one that proves a process exists, and read the tool's own
  stderr log before believing its summary message.
- **Codex remote control does not work on this setup** (as of 2026-09-15): the
  CLI runs fine, but its remote-control daemon dies on
  `401 refresh_token_invalidated`. It was tried and removed; don't re-add it to
  the job. Use the VS Code Codex extension over the tunnel instead.

---

## Dry run: what will each slot actually execute?

Worth doing before the first submit — it resolves the root, the `--add-dir`
list and each slot's `--resume` / `--session-id` without launching anything.

```bash
set -u; source ~/slurm-utils/devbox/config.sh
REPO="$DEVBOX_ROOT"
HIST="$HOME/.claude/projects/$(echo "$REPO" | sed 's/[^A-Za-z0-9]/-/g')"
ADD=""; for d in $DEVBOX_ADD_DIRS; do [ "$d" = "$REPO" ] && continue
  [ -d "$d" ] && ADD="$ADD --add-dir '$d'" || echo "skip $d"; done
echo "root:$REPO"; echo "add :$ADD"
for s in $DEVBOX_SLOTS; do id=$(cat "$DEVBOX_STATE/session-id-$s" 2>/dev/null)
  if [ -z "$id" ]; then a="--session-id <minted>"
  elif [ -f "$HIST/$id.jsonl" ]; then a="--resume $id"
  else a="--session-id $id"; fi
  echo "slot $s: $DEVBOX_NAME-$s $a$ADD"; done
```

A slot showing `--session-id` for an id you expected to resume means its history
is not under **this** root — the root moved, and that conversation is stranded
(history is keyed by the root's absolute path). Either point the root back, or
clear that slot's id file and let it mint a fresh one.
