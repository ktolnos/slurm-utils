# Setting up a devbox on a new cluster

For an agent doing this on a cluster where the devbox has never run. Work
through it in order; each step says how to verify it before moving on. Expect
~20 minutes, most of it waiting on downloads and the queue.

**Read `README.md` first** if you need to know what any of this is for. This
file is the procedure; that one is the explanation.

**One step needs a human** and cannot be worked around: step 3, the account
logins. Surface it early rather than at the end. (Accepting the workspace-trust
dialog used to be a second such step; `active-project` now grants trust
directly, so it is not.)

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
command -v devbox-up            # -> <checkout>/devbox/devbox-up
```

`slurm_utils.sh` adds `devbox/` to `PATH`, derived from its own location, so
the checkout does not have to be `~/slurm-utils` — and on a cluster with a
node-local `$HOME` (step 4) it must not be: clone it onto shared storage and
source it from there instead.

## 2. Install the three binaries (on the login node)

All are self-contained; no node or npm needed. Install them under
`$DEVBOX_CONFIG_HOME` rather than `$HOME` if step 4 says `$HOME` is node-local
-- a binary in a node-local `$HOME` does not exist on the compute node, and
`HOME=<shared> curl ... | sh` is enough to redirect an installer that insists.

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

## 4. Where does durable state live?

Answer this before anything else, because it decides every path below.

```bash
# Is $HOME the same filesystem on the login node and a compute node?
ls ~/.claude.json                                   # here
srun -t 2 --mem=1G bash -c 'ls ~/.claude.json; df -h $HOME | tail -1'
```

If the compute node cannot see it, `$HOME` is **node-local** and the defaults
are all wrong: workspace trust, conversation history, the tunnel token and the
codex enrollment would each be missing on the other side of a node hop, quietly.
Set `DEVBOX_CONFIG_HOME` in this cluster's stanza to a filesystem every node
mounts, and move the existing config there:

```bash
rsync -a ~/.claude/ /shared/you/.claude/
cp -p ~/.claude.json /shared/you/.claude/.claude.json   # NOTE: moves INSIDE the dir
```

`CLAUDE_CONFIG_DIR` relocates the whole directory *including* `.claude.json`
(verified against 2.1.278), so trust, history and the login travel as one unit.
Export it (plus `CODEX_HOME`, `VSCODE_CLI_DATA_DIR` and the two keychain
variables) from the shell profile too, **above any interactive guard**, so
plain `claude` on the login node and a batch job agree about which config they
are using.

## 5. Point at the active project

The agents start **in** the active project — it is each slot's working
directory, and so what Claude Code treats as the session root.

```bash
active-project /path/to/the/repo        # grants trust, asking first
active-project                          # verify
```

This is also the step that used to be an unavoidable human one. `active-project`
writes `hasTrustDialogAccepted` itself rather than sending you off to
`cd <dir> && claude` to answer a dialog you already answered by naming the
directory. It prompts first and lists any `.claude/settings.json`, `.mcp.json`
or `.claude/hooks/` in the tree, because those execute on session start — that,
not reading the code, is what trust is actually about. `--trust` skips the
prompt for scripts; it refuses rather than assuming when not on a tty.

Slot conversations are keyed by **(project, slot)** under
`$DEVBOX_STATE/projects/<project>/`, so switching projects gives that project
its own three conversations and switching back resumes them. One consequence
worth knowing: a running slot cannot change its own working directory, so
repointing does *not* move live agents however many times you `/clear` — they
follow on the next `devbox-up restart`.

If you genuinely do not know the project yet, skip this; the box falls back to
`DEVBOX_ROOT`.

## 6. Install the hooks (user level)

All three go in `$CLAUDE_CONFIG_DIR/settings.json` — **user** level, not a
project's `.claude/`. The agents move between projects, so a project-level
install would have to be repeated in, and would litter, every repo they are
ever pointed at. Merge into whatever is already in that file:

```json
{"hooks": {
  "SessionStart": [{"hooks": [
    {"type": "command", "command": "<devbox>/pin-session"},
    {"type": "command", "command": "<devbox>/active-project --hook"}]}],
  "PreToolUse": [{"matcher": "Bash", "hooks": [
    {"type": "command", "command": "<devbox>/slurm-guard",
     "timeout": 10, "statusMessage": "slurm guard"}]}]}}
```

- **`pin-session`** keeps each slot's id pointing at the conversation it is
  actually in. Without it `/clear` silently orphans that slot: it mints a new
  id inside the same process, the pinned one goes stale, and the next restart
  resumes the abandoned pre-`/clear` conversation. No-ops outside a slot.
- **`active-project --hook`** tells every new conversation which project is
  active, and warns when the pointer is stale.
- **`slurm-guard`** blocks `scontrol update ... NodeList=`/`ReqNodes=` with an
  empty value, which crashes `slurmctld` for the whole cluster on Slurm
  23.02.1. Install it on every cluster whatever the version. It is a hook and
  not a rule in `AGENTS.md` on purpose: a prompt-level rule is advice a
  subagent can reason past, and a subagent is what caused the second crash.
  `ExcNodeList=` stays allowed, empty or not — that is the field to use instead.

Verify with a command that is only an `echo`, so it is harmless if the hook is
*not* working. It must come back denied:

```bash
echo 'scontrol update JobId=999999 NodeList='
```

## 7. Install the cluster rules

Cluster facts live in git at `clusters/<cluster>/AGENTS.md` and reach sessions
through **user-level memory**, for the same reason the hooks do — the agents
are not rooted in one fixed directory any more:

```bash
mkdir -p "$DEVBOX_DIR/clusters/<cluster>"
cp "$DEVBOX_DIR/AGENTS.template.md" "$DEVBOX_DIR/clusters/<cluster>/AGENTS.md"
# fill in the TODOs, then:
echo "@$DEVBOX_DIR/clusters/<cluster>/AGENTS.md" >> "$CLAUDE_CONFIG_DIR/CLAUDE.md"
```

Import the portable rules from inside that file with an **absolute** path
(`@/path/to/devbox/AGENTS.shared.md`), never `@~/...`: on a node-local-`$HOME`
site `~` resolves somewhere different depending on the node.

Those absolute paths are **external** includes, gated behind a one-time
per-project dialog a batch job has nobody to answer. `active-project` and
`devbox-up` preflight both approve it, so there is nothing to do here; it is
called out only because the failure looks like the agents started fine.

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
