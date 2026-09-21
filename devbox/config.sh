#!/bin/bash
# Devbox configuration: everything that differs between clusters lives here, and
# nothing else in devbox/ has a cluster name or an absolute path in it.
#
# Sourced by both devbox.sh (the job) and devbox-up (the launcher), so the
# self-chaining job submits its successor with exactly the flags you launched
# with. Override any DEVBOX_* value from the environment without editing this
# file: `DEVBOX_SLOTS="1 2" devbox-up`.

# --- which cluster are we on? -------------------------------------------------
# CC_CLUSTER is set on every Alliance cluster (fir, narval, rorqual, ...).
# SLURM_CLUSTER_NAME only exists inside a job, so scontrol is the better
# fallback; the hostname strip is a last resort for a site with neither.
# Every expansion here is ${VAR:-} guarded: this file is sourced by devbox-up,
# which runs under `set -u`, and an unguarded $CC_CLUSTER on a non-Alliance
# cluster aborts the function mid-way. That failed quietly the first time and
# resolved the cluster to the empty string, which would have created a VS Code
# tunnel named "-dev".
devbox_cluster() {
    [ -n "${DEVBOX_CLUSTER:-}" ] && { printf '%s\n' "$DEVBOX_CLUSTER"; return; }
    [ -n "${CC_CLUSTER:-}" ]     && { printf '%s\n' "$CC_CLUSTER"; return; }
    local n
    n=$(scontrol show config 2>/dev/null | awk '/^ClusterName/{print $3; exit}')
    [ -n "$n" ] && [ "$n" != "(null)" ] && { printf '%s\n' "$n"; return; }
    [ -n "${SLURM_CLUSTER_NAME:-}" ] && { printf '%s\n' "$SLURM_CLUSTER_NAME"; return; }
    hostname -s | sed 's/[0-9]*$//'
}
DEVBOX_CLUSTER="$(devbox_cluster)"
[ -n "$DEVBOX_CLUSTER" ] || { echo "config.sh: could not determine the cluster name; set DEVBOX_CLUSTER" >&2; return 1 2>/dev/null || exit 1; }

# --- per-cluster overrides ----------------------------------------------------
# Runs BEFORE the portable defaults below, so precedence is
#   environment  >  this stanza  >  portable default
# and every assignment here must be ${VAR:-...} to keep the environment on top.
# It used to run after, which meant a stanza could only ever set a value the
# defaults had left empty: DEVBOX_ROOT and DEVBOX_ADD_DIRS were already non-empty
# by then, so `DEVBOX_ROOT="${DEVBOX_ROOT:-/some/path}"` silently did nothing and
# the box came up rooted at the default.
#
# Add a stanza when a cluster needs something the defaults get wrong.
# Keep it to the few things that genuinely differ -- if you find yourself adding
# logic here, it probably belongs in devbox.sh guarded by a capability test
# instead of a cluster name.
case "$DEVBOX_CLUSTER" in
    fir)
        # Account arrives via SBATCH_ACCOUNT (def-gigor). Slurm resolves the
        # _cpu/_gpu suffix itself, and the partition routes on --time, so
        # nothing to set. sbatch from /home works here.
        :
        ;;
    killarney)
        # sbatch is rejected from /home on this cluster: the check is on the
        # submitting *directory*, not on where the script lives, so submitting
        # from scratch with the script still in ~/slurm-utils works (verified
        # 2026-09-15 with --test-only from both).
        DEVBOX_SUBMIT_DIR="${DEVBOX_SUBMIT_DIR:-${SCRATCH:-$HOME}}"
        # No SBATCH_ACCOUNT in the environment here, and Killarney does not
        # resolve a default, so the account has to be named.
        DEVBOX_ACCOUNT="${DEVBOX_ACCOUNT:-aip-gigor}"
        # Home is 50 GB here and holds no work, so the root is the project repo
        # itself -- which is already trust-accepted, so there is no interactive
        # step, and the pre-devbox single-agent conversation stays resumable
        # (history is keyed by the root's absolute path, so a root move would
        # strand it). $HOME and $SCRATCH come in via --add-dir.
        DEVBOX_ROOT="${DEVBOX_ROOT:-/project/6101830/eop/unlearning-reward-hacking}"
        DEVBOX_ADD_DIRS="${DEVBOX_ADD_DIRS:-$HOME ${SCRATCH:-/scratch/$USER}}"
        ;;
    *)
        # Unknown cluster: the defaults are the portable ones. If the first
        # `devbox-up` fails, the flag it rejects is the thing to add above.
        :
        ;;
esac


# --- portable defaults (filled in only where neither the environment nor the
# cluster stanza above set a value) --------------------------------------------------------
DEVBOX_JOB_NAME="${DEVBOX_JOB_NAME:-claude-dev}"
DEVBOX_SLOTS="${DEVBOX_SLOTS:-1 2 3}"   # one Claude session per slot; add a 4 for a fourth
DEVBOX_CPUS="${DEVBOX_CPUS:-2}"
DEVBOX_MEM="${DEVBOX_MEM:-6G}"
DEVBOX_TIME="${DEVBOX_TIME:-3-00:00:00}"
DEVBOX_AUTOCOMPACT="${DEVBOX_AUTOCOMPACT:-500k}"

# Session root. NOT $HOME: home-directory workspace trust is session-only and
# never persists, so a home-rooted job stops at the trust dialog on every node
# hop, forever. This must be a real project directory you have accepted the
# trust dialog in once. $HOME comes in via --add-dir instead.
DEVBOX_ROOT="${DEVBOX_ROOT:-$HOME/devbox}"

# Extra directories each agent gets via --add-dir, space separated. The root is
# implicit; this is for the trees that live outside it. $HOME is the default
# because that is where work sits on a cluster with a roomy home -- but on a
# site where the repos are in /project and the outputs in /scratch, neither is
# reachable from a $HOME-only list, and an agent that cannot read its own repo
# is useless. A directory that does not exist is skipped rather than passed.
DEVBOX_ADD_DIRS="${DEVBOX_ADD_DIRS:-$HOME}"

# Names: `<cluster>-dev` for the tunnel, `<cluster>-dev-<slot>` for Remote
# Control. Deriving them from the cluster is what makes several clusters usable
# from one account at once -- VS Code tunnel names are globally unique, and two
# boxes both called `dev` would collide and be indistinguishable in the app.
DEVBOX_NAME="${DEVBOX_NAME:-${DEVBOX_CLUSTER}-dev}"

# Per-cluster state (pinned conversation uuids, stop file). Keyed by cluster so
# that a site which shares $HOME between clusters cannot end up resuming the
# same conversation in two places -- two agents on one conversation corrupts it.
DEVBOX_STATE="${DEVBOX_STATE:-$HOME/.devbox/$DEVBOX_CLUSTER}"
DEVBOX_LOG_DIR="${DEVBOX_LOG_DIR:-$HOME/logs}"

# Binaries. All three are self-contained downloads; see README.md.
DEVBOX_CLAUDE_BIN="${DEVBOX_CLAUDE_BIN:-$HOME/.local/bin/claude}"
DEVBOX_CODE_BIN="${DEVBOX_CODE_BIN:-$HOME/bin/code}"
# codex's own installer symlinks this at ~/.codex/packages/standalone/current,
# which its auto-updater reflows -- so point at the symlink, never at a
# versioned path that the next update invalidates.
DEVBOX_CODEX_BIN="${DEVBOX_CODEX_BIN:-$HOME/.local/bin/codex}"

# Codex remote control: one app-server daemon for the whole box, so sessions
# started from the ChatGPT app run here. Exactly `0` disables it and any other
# value enables it -- tested `!= 0` rather than `= 1` so that DEVBOX_CODEX=true
# or =yes cannot silently mean "off". Unlike the agent slots there is nothing
# per-slot to configure: the daemon is a singleton and the app creates sessions
# inside it.
DEVBOX_CODEX="${DEVBOX_CODEX:-1}"

# Where to submit from, and anything else this cluster needs on the sbatch line.
DEVBOX_SUBMIT_DIR="${DEVBOX_SUBMIT_DIR:-$HOME}"
DEVBOX_ACCOUNT="${DEVBOX_ACCOUNT:-${SBATCH_ACCOUNT:-}}"
DEVBOX_PARTITION="${DEVBOX_PARTITION:-}"
DEVBOX_EXTRA_SBATCH="${DEVBOX_EXTRA_SBATCH:-}"

# --- sbatch flags -------------------------------------------------------------
# Printed one per line so callers can read them into an array and keep quoting
# intact. No #SBATCH directives live in devbox.sh: Slurm does not expand $HOME
# in a directive, so an --output path there cannot be portable.
devbox_sbatch_flags() {
    printf '%s\n' \
        "--job-name=$DEVBOX_JOB_NAME" \
        "--time=$DEVBOX_TIME" \
        "--cpus-per-task=$DEVBOX_CPUS" \
        "--mem=$DEVBOX_MEM" \
        "--nodes=1" \
        "--ntasks=1" \
        "--output=$DEVBOX_LOG_DIR/$DEVBOX_JOB_NAME-%j.out"
    [ -n "$DEVBOX_ACCOUNT" ]   && printf '%s\n' "--account=$DEVBOX_ACCOUNT"
    [ -n "$DEVBOX_PARTITION" ] && printf '%s\n' "--partition=$DEVBOX_PARTITION"
    for f in $DEVBOX_EXTRA_SBATCH; do printf '%s\n' "$f"; done
    return 0
}
