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

# --- portable defaults --------------------------------------------------------
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

# Binaries. Both are self-contained downloads; see README.md.
DEVBOX_CLAUDE_BIN="${DEVBOX_CLAUDE_BIN:-$HOME/.local/bin/claude}"
DEVBOX_CODE_BIN="${DEVBOX_CODE_BIN:-$HOME/bin/code}"

# Where to submit from, and anything else this cluster needs on the sbatch line.
DEVBOX_SUBMIT_DIR="${DEVBOX_SUBMIT_DIR:-$HOME}"
DEVBOX_ACCOUNT="${DEVBOX_ACCOUNT:-${SBATCH_ACCOUNT:-}}"
DEVBOX_PARTITION="${DEVBOX_PARTITION:-}"
DEVBOX_EXTRA_SBATCH="${DEVBOX_EXTRA_SBATCH:-}"

# --- per-cluster overrides ----------------------------------------------------
# Add a stanza when a cluster needs something the defaults above get wrong.
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
    # killarney)
    #     # sbatch is rejected from /home on this cluster -- submit from scratch.
    #     DEVBOX_SUBMIT_DIR="${SCRATCH:-$HOME}"
    #     ;;
    *)
        # Unknown cluster: the defaults are the portable ones. If the first
        # `devbox-up` fails, the flag it rejects is the thing to add above.
        :
        ;;
esac

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
