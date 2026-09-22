#!/bin/bash
# Long-lived, self-chaining CPU-only devbox: one Claude Code session per slot
# plus one `code tunnel`, all as windows of a single tmux session, so you can
# drive several agents from any device and keep them across node hops. Plus one
# codex app-server daemon, which is not a tmux window -- see launch_codex.
#
# Launch it with `devbox-up`, never with a bare `sbatch` -- all resource flags
# come from config.sh (see devbox_sbatch_flags), because a #SBATCH --output
# directive cannot contain $HOME and so cannot be portable.
#
# This is NOT a compute node: it is sized to host the agent sessions and the
# tunnel, nothing else. Never run training, inference or a test suite in it --
# a vLLM import alone will exceed the memory cap and get the job OOM-killed,
# which takes down every agent AND the tunnel at once.
#
# In local mode (DEVBOX_LOCAL, for a site whose login node is meant to be used
# directly and is not Slurm compute) there is no cgroup and so no cap: the same
# rule holds, but nothing enforces it and the damage lands on everyone else
# using that machine rather than on this job.

# Where devbox/ actually lives. NOT $BASH_SOURCE: sbatch copies the script into
# the node's spool directory and runs that copy, so inside a job $BASH_SOURCE is
# /.../spool/job<id>/slurm_script and config.sh is not next to it. That failed
# silently -- `source` printed one line and the script carried on with every
# DEVBOX_* variable empty: no slots, no tunnel, no successor, and a tmux called
# with empty arguments.
#
# devbox-up exports DEVBOX_DIR at submit time and it rides the chain on sbatch's
# default --export=ALL; scontrol is the fallback for a job submitted some other
# way, and $BASH_SOURCE for a direct `bash devbox.sh` outside Slurm.
devbox_dir() {
    [ -f "${DEVBOX_DIR:-}/config.sh" ] && { printf '%s\n' "$DEVBOX_DIR"; return; }
    local cmd
    cmd=$(scontrol show job "${SLURM_JOB_ID:-}" 2>/dev/null \
          | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)
    [ -n "$cmd" ] && [ -f "$(dirname "$cmd")/config.sh" ] && { dirname "$cmd"; return; }
    ( cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd )
}
DEVBOX_DIR="$(devbox_dir)"
[ -f "$DEVBOX_DIR/config.sh" ] || {
    echo "FATAL: config.sh not found in '$DEVBOX_DIR' -- set DEVBOX_DIR to the devbox/ directory" >&2
    exit 1
}
export DEVBOX_DIR          # so the successor this job queues resolves it too
# shellcheck source=config.sh
source "$DEVBOX_DIR/config.sh"

# $TMUX is set in every shell inside a tmux pane, and sbatch exports the whole
# submitting environment, so submitting the devbox from inside the devbox's own
# tmux carries it into the job. A tmux client that sees $TMUX believes it is
# already inside a server and tries to reuse that socket path instead of
# creating one: on a fresh node /tmp/tmux-$UID does not exist yet, tmux never
# makes it, and every new-session dies with
#   error creating /tmp/tmux-<uid>/default (No such file or directory)
# while still exiting 0. Worse, --export=ALL hands the same poisoned $TMUX to
# the successor, so one submit from inside a pane breaks the entire chain.
unset TMUX TMUX_PANE

SCRIPT="$DEVBOX_DIR/devbox.sh"
STOP_FILE="$DEVBOX_STATE/stop"
# Names the log files. Outside Slurm there is no job id, and every run would
# otherwise share one "tunnel-.log" and interleave into it.
RUN_ID="${SLURM_JOB_ID:-local-$(hostname -s)-$$}"
TUNNEL_LOG="$DEVBOX_LOG_DIR/tunnel-$RUN_ID.log"

# A login shell's .bashrc may override `cd` with a function that returns some
# other command's exit status (a venv `activate`, say), which makes `cd x && y`
# silently skip y. Never chain on cd below; use `;`.
source "$HOME/.bashrc" 2>/dev/null

export PATH="$HOME/bin:$HOME/.local/bin:$PATH"

# Every piece of durable agent state, pointed at whatever filesystem this
# cluster keeps it on -- $HOME by default, shared storage where $HOME is
# node-local. Exported AFTER sourcing .bashrc so config.sh wins over whatever
# the shell profile happens to set.
#
# CLAUDE_CONFIG_DIR is the load-bearing one: it carries .claude.json, and so
# workspace trust and the history that every --resume depends on. Get it wrong
# on a node-local-$HOME site and the box comes up with three slots parked on a
# trust dialog and every pinned conversation silently replaced by an empty one.
export CLAUDE_CONFIG_DIR="$DEVBOX_CLAUDE_CONFIG_DIR"
export CODEX_HOME="$DEVBOX_CODEX_HOME"
# Both keychain vars are load-bearing: without them `code tunnel user show`
# reports logged in on the login node and not logged in on every compute node.
# Cost: $VSCODE_CLI_DATA_DIR/token.json holds the GitHub token in plaintext (0600).
export VSCODE_CLI_USE_FILE_KEYCHAIN=1 VSCODE_CLI_DISABLE_KEYCHAIN_ENCRYPT=1
export VSCODE_CLI_DATA_DIR="$DEVBOX_VSCODE_DATA_DIR" VSCODE_CLI_NONINTERACTIVE=1

# --- where the agents work ----------------------------------------------------
# The agents start IN the active project. Claude Code keys both workspace trust
# and conversation history to the working directory, so that one path decides
# which conversations resume and whether the slots come up at all -- which is
# why active-project grants trust when it is set rather than leaving it to a
# dialog nobody is here to answer. DEVBOX_ROOT is now only the fallback for
# when no project has been named.
ACTIVE_PROJECT="$("$DEVBOX_DIR/active-project" 2>/dev/null)"
WORKDIR="${ACTIVE_PROJECT:-$DEVBOX_ROOT}"
if [ ! -d "$WORKDIR" ]; then
    echo "active project '$WORKDIR' is not a directory -- falling back to $DEVBOX_ROOT"
    WORKDIR="$DEVBOX_ROOT"
fi

# Slot ids are keyed by (project, slot), not by slot alone. History lives under
# the working directory, so a pinned id only resolves inside the project it was
# minted in: with one flat set, switching projects would hand --resume an id
# whose transcript is under the old path, and the slot would come up in a fresh
# empty conversation wearing a used id. Per project, switching away and back
# resumes that project's own conversations instead.
slot_dir() {
    printf '%s/projects/%s\n' "$DEVBOX_STATE" "$(printf '%s' "$1" | sed 's/[^A-Za-z0-9]/-/g')"
}
SLOT_DIR="$(slot_dir "$WORKDIR")"
mkdir -p "$SLOT_DIR"

mkdir -p "$DEVBOX_STATE" "$DEVBOX_LOG_DIR"
if [ "$DEVBOX_LOCAL" != 0 ]; then
    echo "=== $DEVBOX_JOB_NAME local on $(hostname) ($DEVBOX_CLUSTER) at $(date) ==="
    # Claimed here, not by devbox-up: `setsid nohup ... &` reports the pid of a
    # process that may already have forked away, so $! there is not reliably
    # this box. Written with the hostname because a bare pid checked from
    # another node would match some unrelated process.
    printf '%s %s\n' "$$" "$(hostname -s)" > "$DEVBOX_STATE/local.pid"
else
    echo "=== $DEVBOX_JOB_NAME job $SLURM_JOB_ID on $(hostname) ($DEVBOX_CLUSTER) at $(date) ==="
fi
echo "    project=$WORKDIR  slots=[$DEVBOX_SLOTS]  name=$DEVBOX_NAME"
echo "    state=$SLOT_DIR  config=$CLAUDE_CONFIG_DIR"

# Queue the successor NOW: survives node failure, OOM and scancel, and accrues
# queue age. A USR1 trap would not fire in any of those cases. Resubmitting from
# $SCRIPT means a `git pull` in this repo reaches every cluster's next job.
if [ "$DEVBOX_LOCAL" != 0 ]; then
    # Nothing to chain to: a local box has no walltime to be evicted by, so it
    # runs until the machine reboots or someone stops it.
    echo "local mode -- no successor queued"
elif [ -e "$STOP_FILE" ]; then
    echo "stop file present ($STOP_FILE) -- not chaining"
else
    mapfile -t FLAGS < <(devbox_sbatch_flags)
    ( cd "$DEVBOX_SUBMIT_DIR" 2>/dev/null; sbatch "${FLAGS[@]}" --dependency=singleton "$SCRIPT" )
fi

tmux has-session -t claude 2>/dev/null && tmux kill-session -t claude

HIST_DIR="$CLAUDE_CONFIG_DIR/projects/$(echo "$WORKDIR" | sed 's/[^A-Za-z0-9]/-/g')"

# Pin one conversation per slot, minted once and then kept forever:
# --resume needs the id to exist, --session-id needs it not to. A slot whose id
# has never been used yet keeps its uuid and starts it -- do not re-mint, or the
# "persistent uuid" promise breaks on the first restart before you used the slot.
session_arg() {
    local f="$SLOT_DIR/session-id-$1" id
    id=$(cat "$f" 2>/dev/null)
    if [ -z "$id" ]; then
        id=$(uuidgen); printf '%s\n' "$id" > "$f"; chmod 600 "$f"
    fi
    if [ -f "$HIST_DIR/$id.jsonl" ]; then
        echo "--resume $id"
    else
        echo "--session-id $id"
    fi
}

# Slurm's job_container/tmpfs can bind-mount a job-private /tmp between tmux's
# mkdir of /tmp/tmux-$UID and the bind() of the socket inside it, so retry and
# confirm with has-session -- never trust a tmux command's exit status.
start_tmux() {
    local i
    for i in $(seq 1 10); do
        tmux has-session -t claude 2>/dev/null && return 0
        tmux new-session -d -s claude -n "agent$(set -- $DEVBOX_SLOTS; echo "$1")" -c "$WORKDIR"
        tmux has-session -t claude 2>/dev/null && return 0
        sleep 3
    done
    return 1
}

# --add-dir flags, built once. Quoted per directory so a path with a space
# survives the trip through tmux send-keys, and skipped when absent so a
# cluster's stanza can name a tree that only some nodes mount.
ADD_DIRS=""
# The project no longer needs naming here -- it is the working directory, so it
# is implicit. What does need naming is everything OUTSIDE it: $HOME, this
# cluster's shared trees, and DEVBOX_ROOT, so the devbox's own scripts and docs
# stay readable from whichever project the agents are working in.
for d in $DEVBOX_ADD_DIRS "$DEVBOX_ROOT"; do
    [ "$d" = "$WORKDIR" ] && continue       # the working directory is implicit
    [ -d "$d" ] || { echo "add-dir: skipping $d (not a directory)"; continue; }
    case "$ADD_DIRS" in *"--add-dir '$d'"*) continue ;; esac   # named twice
    ADD_DIRS="$ADD_DIRS --add-dir '$d'"
done
echo "    add-dir:$ADD_DIRS"
[ -n "$ACTIVE_PROJECT" ] || \
    echo "    project: none set -- run 'active-project <dir>'; using $DEVBOX_ROOT"

declare -A LAST_LAUNCH
launch_agent() {
    local slot=$1 win="agent$1" arg
    arg=$(session_arg "$slot")
    tmux list-windows -t claude -F '#{window_name}' | grep -qx "$win" \
        || tmux new-window -t claude -n "$win" -c "$WORKDIR"
    # cd explicitly: the window's shell sources .bashrc, which may cd elsewhere,
    # and the agent must start in $WORKDIR or it opens a different history
    # (history is keyed by absolute path) and an untrusted workspace.
    # DEVBOX_SLOT/DEVBOX_SLOT_DIR tell the SessionStart pin hook which slot file
    # to update when /clear or /resume changes the conversation id underneath
    # us; DEVBOX_SLOT_DIR is per project, so a pin never lands in another
    # project's set. DEVBOX_STATE still goes through for pins.log.
    tmux send-keys -t "claude:$win" \
        "cd '$WORKDIR'; DEVBOX_SLOT=$slot DEVBOX_STATE='$DEVBOX_STATE' DEVBOX_SLOT_DIR='$SLOT_DIR' '$DEVBOX_CLAUDE_BIN' --remote-control $DEVBOX_NAME-$slot $arg$ADD_DIRS --autocompact $DEVBOX_AUTOCOMPACT" C-m
    LAST_LAUNCH[$slot]=$SECONDS
    echo "slot $slot: $DEVBOX_NAME-$slot  $arg"
}
launch_agents() { local s; for s in $DEVBOX_SLOTS; do launch_agent "$s"; done; }

launch_tunnel() {
    [ -x "$DEVBOX_CODE_BIN" ] || { echo "tunnel: no code CLI at $DEVBOX_CODE_BIN -- skipping"; return 1; }
    "$DEVBOX_CODE_BIN" tunnel user show 2>&1 | grep -qi "logged in with" || {
        echo "tunnel: code CLI not authenticated on this node -- skipping"; return 1; }
    rm -f "$VSCODE_CLI_DATA_DIR/tunnel-stable.lock"   # stale lock from a killed job
    tmux list-windows -t claude -F '#{window_name}' | grep -qx tunnel \
        || tmux new-window -t claude -n tunnel -c "$HOME"
    # cd first: the folder the shell is in becomes the default folder in the
    # vscode.dev/tunnel/... link. Never pipe `code tunnel` through tee
    # unattended -- unauthenticated it redraws an interactive picker forever,
    # hundreds of MB in under a minute. Hence the guard above and the
    # truncation below.
    tmux send-keys -t claude:tunnel \
        "cd '$WORKDIR'; '$DEVBOX_CODE_BIN' tunnel --accept-server-license-terms --name $DEVBOX_NAME >> '$TUNNEL_LOG' 2>&1" C-m
}

# Codex remote control. Deliberately NOT a tmux window: `remote-control start`
# forks the app-server daemon, prints one JSON line and exits, so a pane would
# just sit at a shell prompt. It is a call, it is idempotent (`alreadyRunning`),
# it takes ~0.3 s, and on a daemon that is already up it re-enables remote
# control -- which makes the same call the start path AND the repair path, so the
# watchdog below has nothing else to do.
#
# One daemon serves the whole box; there is no per-slot equivalent of
# DEVBOX_SLOTS. The ChatGPT app creates and drives sessions inside it (`codex
# agents` lists them from here), so there is no conversation id to pin either.
#
# Node hops are free: the enrollment (server_id + environment_id) is persisted
# under ~/.codex and reused, so fir's box kept one environment when it moved
# fc30355 -> fc30354. Only the display name follows gethostname(), which is not
# overridable -- the app shows the bare node name, not $DEVBOX_NAME.
#
# The daemon and its updater reparent to init but stay in the job's cgroup
# (verified), so they die with the job rather than leaking onto the node.
codex_field() { printf '%s' "$2" | grep -o "\"$1\":\"[^\"]*\"" | head -1 | cut -d'"' -f4; }

# "" until the first check, then up/down/absent. Only a CHANGE is logged, so a
# healthy box costs one line per job rather than one line per minute.
CODEX_STATE=""
launch_codex() {
    [ "$DEVBOX_CODEX" != 0 ] || return 0
    if [ ! -x "$DEVBOX_CODEX_BIN" ]; then
        [ "$CODEX_STATE" = absent ] || echo "codex: no CLI at $DEVBOX_CODEX_BIN -- skipping"
        CODEX_STATE=absent; return 1
    fi
    # timeout: `start` has its own connect deadline (it reports timedOut) but a
    # wedged daemon must not stall the per-slot watchdog queued behind it. 60 s
    # is ~200x the measured call (0.3 s warm, similar cold), and the two failure
    # modes are asymmetric: too short costs one spurious "down" line and a retry
    # 5 min later, too long delays relaunching a dead agent slot -- which has
    # only a 180 s grace of its own.
    local out state
    out=$(timeout 60 "$DEVBOX_CODEX_BIN" remote-control start --json 2>&1)
    # Matched against the whole blob on purpose. There are two "status" fields --
    # top-level connection and nested daemon lifecycle -- but only the former is
    # ever "connected" (the latter is started/bootstrapped/alreadyRunning), so
    # this does not depend on serde's field order.
    case "$out" in *'"status":"connected"'*) state=up ;; *) state=down ;; esac
    [ "$state" = "$CODEX_STATE" ] && return 0
    CODEX_STATE=$state
    if [ "$state" = up ]; then
        echo "codex: remote control connected as $(codex_field serverName "$out") (env $(codex_field environmentId "$out"))"
    else
        # ${out:-...}: a `timeout` kill leaves $out empty, and a bare
        # "NOT connected -- " with nothing after it reads like a parsing bug.
        echo "codex: NOT connected -- ${out:-no output (timed out after 60s?)}"
        # The one failure seen so far: refresh_token_invalidated, which needs a
        # fresh `codex login` on the login node and MFA enabled on the account.
        # `codex login status` still says "Logged in" in that state, so the
        # status field above is the only honest signal.
        echo "codex: re-run 'codex login' on the login node (the account needs MFA enabled)"
    fi
    [ "$state" = up ]
}

if start_tmux; then launch_agents; launch_tunnel
else echo "FATAL: no tmux server; watchdog below keeps retrying"; fi
launch_codex     # independent of tmux: no window, no session to wait for
CODEX_CHECKED=$SECONDS

while true; do
    sleep 60
    # Codex first, and before the tmux checks: it does not live in the tmux
    # session, and the branch below `continue`s on a dead session, which would
    # otherwise stop the daemon ever being re-checked on the box that needs it
    # most. Every 5 min rather than every minute -- one idempotent call is both
    # the liveness probe and the repair, so the only cost of a longer period is
    # how stale a "down" line can be.
    if [ $(( SECONDS - CODEX_CHECKED )) -ge 300 ]; then
        CODEX_CHECKED=$SECONDS; launch_codex
    fi
    [ -f "$TUNNEL_LOG" ] && [ "$(stat -c %s "$TUNNEL_LOG")" -gt 52428800 ] && : > "$TUNNEL_LOG"
    # Watchdog: agents and tunnel are windows of this one session, so losing it
    # means idling for the rest of the walltime unless something rebuilds it.
    tmux has-session -t claude 2>/dev/null \
        || { start_tmux && { launch_agents; launch_tunnel; }; continue; }
    # Per-slot watchdog: a window whose foreground process is back to a shell
    # lost its agent. Checking the pane command (not just the window) is what
    # makes a single dead slot recoverable instead of idle until the job ends.
    for slot in $DEVBOX_SLOTS; do
        [ $(( SECONDS - ${LAST_LAUNCH[$slot]:-0} )) -lt 180 ] && continue
        tmux list-windows -t claude -F '#{window_name}' | grep -qx "agent$slot" \
            || { launch_agent "$slot"; continue; }
        # That existence check is load-bearing: `display-message -t claude:agentN`
        # on a window that does not exist does not fail -- it silently reports the
        # session's ACTIVE window instead (measured: probing agent9 returned
        # agent1's `claude`), so a dead slot would look healthy forever.
        # Relaunch only into a bare shell, and only after a grace period:
        # send-keys into a pane that IS running an agent would type the command
        # line into that agent's prompt.
        case "$(tmux display-message -p -t "claude:agent$slot" '#{pane_current_command}')" in
            bash|sh|zsh) launch_agent "$slot" ;;
        esac
    done
done
