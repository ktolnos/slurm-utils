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

## 2. Install the two binaries (on the login node)

Both are self-contained; no node or npm needed.

```bash
curl -fsSL https://claude.ai/install.sh | bash          # -> ~/.local/bin/claude
mkdir -p ~/bin
curl -fsSL 'https://update.code.visualstudio.com/latest/cli-linux-x64/stable' \
  | tar xz -C ~/bin                                     # -> ~/bin/code
claude --version && ~/bin/code --version
```

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

## 3. HUMAN STEP: authenticate both tools

```bash
claude                                      # /login, browser flow
~/bin/code tunnel user login --provider github
```

You cannot do this for them; it is an account credential flow. Ask, then verify:

```bash
~/bin/code tunnel user show | grep -i 'logged in with'
```

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

```bash
mkdir -p ~/devbox/.claude
cat > ~/devbox/.claude/settings.json <<'EOF'
{
  "hooks": {
    "SessionStart": [
      { "hooks": [ { "type": "command", "command": "$HOME/slurm-utils/devbox/pin-session" } ] }
    ]
  }
}
EOF
```

Without this, `/clear` in a slot silently orphans that slot's conversation: it
mints a new conversation id inside the same process, the pinned id goes stale,
and the next job in the chain resumes the abandoned pre-`/clear` conversation.
The hook rewrites the slot's id file with whatever conversation it is actually
in, and no-ops in any session that is not a devbox slot.

Note this edits the agent's own hook configuration. If you are an agent running
under a permission policy, expect that write to need approval — ask rather than
routing around it.

## 7. Install `AGENTS.md`

See `AGENTS.template.md` in this directory: copy it to `~/devbox/AGENTS.md`,
add a one-line `~/devbox/CLAUDE.md` that imports it, and fill in the TODOs. The
portable rules come from `AGENTS.shared.md` via an import, so only
cluster-specific facts go in the copy — and leave a TODO rather than guessing a
number you have not measured.

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
`case` in `config.sh` — account, walltime, `DEVBOX_SUBMIT_DIR`, a partition —
and keep it to values, not logic. If you discover something true of *every*
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
