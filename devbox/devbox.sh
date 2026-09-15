#!/bin/bash
# Long-lived, self-chaining CPU-only devbox: one Claude Code session per slot
# plus one `code tunnel`, all as windows of a single tmux session, so you can
# drive several agents from any device and keep them across node hops.
#
# Launch it with `devbox-up`, never with a bare `sbatch` -- all resource flags
# come from config.sh (see devbox_sbatch_flags), because a #SBATCH --output
# directive cannot contain $HOME and so cannot be portable.
#
# This is NOT a compute node: it is sized to host the agent sessions and the
# tunnel, nothing else. Never run training, inference or a test suite in it --
# a vLLM import alone will exceed the memory cap and get the job OOM-killed,
# which takes down every agent AND the tunnel at once.

DEVBOX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$DEVBOX_DIR/config.sh"

SCRIPT="$DEVBOX_DIR/devbox.sh"
REPO="$DEVBOX_ROOT"
STOP_FILE="$DEVBOX_STATE/stop"
TUNNEL_LOG="$DEVBOX_LOG_DIR/tunnel-${SLURM_JOB_ID}.log"

# A login shell's .bashrc may override `cd` with a function that returns some
# other command's exit status (a venv `activate`, say), which makes `cd x && y`
# silently skip y. Never chain on cd below; use `;`.
source "$HOME/.bashrc" 2>/dev/null

export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
# Both keychain vars are load-bearing: without them `code tunnel user show`
# reports logged in on the login node and not logged in on every compute node.
# Cost: ~/.vscode-cli/token.json holds the GitHub token in plaintext (0600).
export VSCODE_CLI_USE_FILE_KEYCHAIN=1 VSCODE_CLI_DISABLE_KEYCHAIN_ENCRYPT=1
export VSCODE_CLI_DATA_DIR="$HOME/.vscode-cli" VSCODE_CLI_NONINTERACTIVE=1

mkdir -p "$DEVBOX_STATE" "$DEVBOX_LOG_DIR"
echo "=== $DEVBOX_JOB_NAME job $SLURM_JOB_ID on $(hostname) ($DEVBOX_CLUSTER) at $(date) ==="
echo "    root=$REPO  slots=[$DEVBOX_SLOTS]  name=$DEVBOX_NAME  state=$DEVBOX_STATE"

# Queue the successor NOW: survives node failure, OOM and scancel, and accrues
# queue age. A USR1 trap would not fire in any of those cases. Resubmitting from
# $SCRIPT means a `git pull` in this repo reaches every cluster's next job.
if [ -e "$STOP_FILE" ]; then
    echo "stop file present ($STOP_FILE) -- not chaining"
else
    mapfile -t FLAGS < <(devbox_sbatch_flags)
    ( cd "$DEVBOX_SUBMIT_DIR" 2>/dev/null; sbatch "${FLAGS[@]}" --dependency=singleton "$SCRIPT" )
fi

tmux has-session -t claude 2>/dev/null && tmux kill-session -t claude

HIST_DIR="$HOME/.claude/projects/$(echo "$REPO" | sed 's/[^A-Za-z0-9]/-/g')"

# Pin one conversation per slot, minted once and then kept forever:
# --resume needs the id to exist, --session-id needs it not to. A slot whose id
# has never been used yet keeps its uuid and starts it -- do not re-mint, or the
# "persistent uuid" promise breaks on the first restart before you used the slot.
session_arg() {
    local f="$DEVBOX_STATE/session-id-$1" id
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
        tmux new-session -d -s claude -n "agent$(set -- $DEVBOX_SLOTS; echo "$1")" -c "$REPO"
        tmux has-session -t claude 2>/dev/null && return 0
        sleep 3
    done
    return 1
}

# --add-dir flags, built once. Quoted per directory so a path with a space
# survives the trip through tmux send-keys, and skipped when absent so a
# cluster's stanza can name a tree that only some nodes mount.
ADD_DIRS=""
for d in $DEVBOX_ADD_DIRS; do
    [ "$d" = "$REPO" ] && continue          # the root is already implicit
    [ -d "$d" ] || { echo "add-dir: skipping $d (not a directory)"; continue; }
    ADD_DIRS="$ADD_DIRS --add-dir '$d'"
done
echo "    add-dir:$ADD_DIRS"

declare -A LAST_LAUNCH
launch_agent() {
    local slot=$1 win="agent$1" arg
    arg=$(session_arg "$slot")
    tmux list-windows -t claude -F '#{window_name}' | grep -qx "$win" \
        || tmux new-window -t claude -n "$win" -c "$REPO"
    # cd explicitly: the window's shell sources .bashrc, which may cd elsewhere,
    # and the agent must start in $REPO or it opens a different history (history
    # is keyed by absolute path) and an untrusted workspace.
    # DEVBOX_SLOT/DEVBOX_STATE tell the SessionStart pin hook which slot file to
    # update when /clear or /resume changes the conversation id underneath us.
    tmux send-keys -t "claude:$win" \
        "cd '$REPO'; DEVBOX_SLOT=$slot DEVBOX_STATE='$DEVBOX_STATE' '$DEVBOX_CLAUDE_BIN' --remote-control $DEVBOX_NAME-$slot $arg$ADD_DIRS --autocompact $DEVBOX_AUTOCOMPACT" C-m
    LAST_LAUNCH[$slot]=$SECONDS
    echo "slot $slot: $DEVBOX_NAME-$slot  $arg"
}
launch_agents() { local s; for s in $DEVBOX_SLOTS; do launch_agent "$s"; done; }

launch_tunnel() {
    [ -x "$DEVBOX_CODE_BIN" ] || { echo "tunnel: no code CLI at $DEVBOX_CODE_BIN -- skipping"; return 1; }
    "$DEVBOX_CODE_BIN" tunnel user show 2>&1 | grep -qi "logged in with" || {
        echo "tunnel: code CLI not authenticated on this node -- skipping"; return 1; }
    rm -f "$HOME/.vscode-cli/tunnel-stable.lock"   # stale lock from a killed job
    tmux list-windows -t claude -F '#{window_name}' | grep -qx tunnel \
        || tmux new-window -t claude -n tunnel -c "$HOME"
    # cd first: the folder the shell is in becomes the default folder in the
    # vscode.dev/tunnel/... link. Never pipe `code tunnel` through tee
    # unattended -- unauthenticated it redraws an interactive picker forever,
    # hundreds of MB in under a minute. Hence the guard above and the
    # truncation below.
    tmux send-keys -t claude:tunnel \
        "cd '$REPO'; '$DEVBOX_CODE_BIN' tunnel --accept-server-license-terms --name $DEVBOX_NAME >> '$TUNNEL_LOG' 2>&1" C-m
}

if start_tmux; then launch_agents; launch_tunnel
else echo "FATAL: no tmux server; watchdog below keeps retrying"; fi

while true; do
    sleep 60
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
