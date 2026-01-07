# ==============================================================================
# GPU CLUSTER UTILITIES
# ==============================================================================

# ------------------------------------------------------------------------------
# Function: avail_gpus
# ------------------------------------------------------------------------------
# Description:
#   Scans the cluster to find GPU types that are currently available for
#   immediate use. It parses `sinfo` output to calculate true availability
#   (Total - Used) and applies strict state filtering.
#
# Logic:
#   1. Runs `sinfo` with extended column widths to handle long MIG strings.
#   2. Filters out nodes that are:
#      - Drained (drain, drng)
#      - Down (down, fail)
#      - Maintenance (maint)
#      - Unreachable/Not Responding (*)
#   3. Parses 'Gres' and 'GresUsed' columns using Python regex.
#   4. Returns a sorted, space-separated list of GPU types that have at
#      least one free slot on a valid, active node.
#
# Usage:
#   avail_gpus
#
# Output Example:
#   h100 nvidia_h100_80gb_hbm3_1g.10gb nvidia_h100_80gb_hbm3_3g.40gb
# ------------------------------------------------------------------------------
avail_gpus() {
    # 1. We include NodeList again to help with debugging/logging.
    local sinfo_cmd="sinfo -N -h -t idle,mix,alloc -O NodeList:20,StateCompact:20,Gres:5000,GresUsed:5000"
    
    $sinfo_cmd | python3 -c "
import sys, re

pattern = re.compile(r'gpu:([^:\s]+):(\d+)')
available_types = set()

# Debug counters
skipped_bad_state = 0
skipped_no_gres = 0

# States that are definitely broken/unusable
# (Note: We do NOT filter 'mix-' specifically, only explicit drain/fail flags)
bad_states = ['drain', 'drng', 'down', 'fail', 'maint']

for line in sys.stdin:
    parts = line.strip().split()
    if len(parts) < 3: continue

    # 2. Dynamic Column Parsing
    # We explicitly look for the columns containing 'gpu:' to handle weird spacing
    # default structure: Node [0], State [1], Gres [?], GresUsed [?]
    node = parts[0]
    state = parts[1].lower()
    
    gres_total_str = next((p for p in parts[2:] if 'gpu:' in p), '(null)')
    # Used is the *last* gpu-like column, or (null)
    gres_used_str = parts[-1] if 'gpu:' in parts[-1] and parts[-1] != gres_total_str else '(null)'
    
    # 3. Filter: Check for locked nodes
    # We check for '*' (non-responding) or specific bad keywords
    if any(b in state for b in bad_states) or '*' in state:
        # Print to stderr so user sees it, but doesn't corrupt stdout for scripts
        # sys.stderr.write(f'DEBUG: Skipping {node} ({state})\n')
        skipped_bad_state += 1
        continue

    if gres_total_str == '(null)': 
        skipped_no_gres += 1
        continue
    
    # 4. Math
    totals = {}
    for m in pattern.finditer(gres_total_str): 
        totals[m.group(1)] = int(m.group(2))
    
    used = {}
    if gres_used_str != '(null)':
        for m in pattern.finditer(gres_used_str): 
            used[m.group(1)] = used.get(m.group(1), 0) + int(m.group(2))

    for g_type, total_count in totals.items():
        if total_count - used.get(g_type, 0) > 0:
            available_types.add(g_type)

# Output results
print(' '.join(sorted(available_types)))

# Optional: Print summary to stderr if list is suspiciously empty
if not available_types and skipped_bad_state > 0:
    sys.stderr.write(f'Warning: No GPUs found. {skipped_bad_state} nodes were skipped due to state (drain/maint/down).\n')
"
}
# ------------------------------------------------------------------------------
# Function: sbt (Smart Batch Tail)
# ------------------------------------------------------------------------------
# Description:
#   Submits a Slurm job with dynamic GPU selection capabilities and immediately
#   tails the output file.
#
# Features:
#   1. Dynamic GPU Selection:
#      Scans the submission script for a custom header:
#      #SELECTGPU type1, type2, type3
#      It checks availability using `avail_gpus`. It modifies the `sbatch`
#      command to request the first available type from your list.
#      If none are free, it defaults to the first preference in the list.
#   2. Auto-Tailing:
#      Parses the Job ID and runs `tail -F` on the output file.
#
# Usage:
#   sbt my_script.slurm
#   sbt my_script.slurm --time=2:00:00
# ------------------------------------------------------------------------------
sbt() {
    local script_file="$1"
    
    if [[ -z "$script_file" ]]; then
        echo "Usage: sbt <sbatch_script> [sbatch_options...]" >&2
        return 1
    fi

    # -- Step A: Parse the script for #SELECTGPU --
    local gpu_priority_line
    gpu_priority_line=$(grep "^#SELECTGPU" "$script_file" | head -n 1 | sed 's/^#SELECTGPU //')
    
    local selected_gres=""

    if [[ -n "$gpu_priority_line" ]]; then
        echo "--> Found dynamic GPU request: $gpu_priority_line"
        
        # -- Step B: Call the helper function --
        local available_types
        available_types=$(avail_gpus)
        
        echo "--> Currently available types: [ $available_types ]"

        # -- Step C: Find first match --
        IFS=',' read -ra ADDR <<< "$gpu_priority_line"
        for candidate in "${ADDR[@]}"; do
            candidate=$(echo "$candidate" | xargs) # trim whitespace
            
            # Check if candidate is in the available string
            if [[ " $available_types " =~ " $candidate " ]]; then
                selected_gres="gpu:$candidate:1"
                echo "--> Match found! Overriding selection to: $selected_gres"
                break
            fi
        done

        # -- Step D: Fallback to first preference if nothing is free --
        if [[ -z "$selected_gres" ]]; then
             local first_choice=$(echo "${ADDR[0]}" | xargs)
             selected_gres="gpu:$first_choice:1"
             echo "--> No GPUs from list are currently free. Queueing for first preference: $selected_gres"
        fi
    fi

    # -- Step E: Construct args and submit --
    local args=("$@")
    if [[ -n "$selected_gres" ]]; then
        # Prepend the GRES requirement
        args=("--gres=$selected_gres" "${args[@]}")
    fi

    local jobid
    jobid=$(sbatch --parsable "${args[@]}")

    if [[ -z "$jobid" ]]; then
        echo "sbatch command failed." >&2
        return 1
    fi

    echo "--> Submitted Job ID: $jobid"
    local outfile="slurm-${jobid}.out"
    echo "--> Tailing ${outfile} (Ctrl+C to stop)"
    echo "--------------------------------------------------"
    tail -F "${outfile}"
}

# ------------------------------------------------------------------------------
# Function: psbt (Pull, Submit, Tail)
# ------------------------------------------------------------------------------
# Description:
#   A safety wrapper for `sbt` that ensures the code is up-to-date before
#   running.
#
# Logic:
#   1. Checks if the current directory is a valid git repository.
#   2. Executes `git pull`. If this fails (conflicts/network), it aborts
#      the submission to prevent running stale or broken code.
#   3. Logs the current commit hash and message for record-keeping.
#   4. Calls `sbt` to submit the job.
#
# Usage:
#   psbt my_job.sh
#   psbt my_job.sh --partition=gpu
# ------------------------------------------------------------------------------
psbt() {
    # 1. Check if we are inside a git repository.
    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "Error: Not a git repository. Cannot run 'git pull'." >&2
        echo "Aborting job submission." >&2
        return 1
    fi

    # 2. Run 'git pull' and check if it succeeds.
    echo "--> Attempting to pull latest changes with 'git pull'..."
    if ! git pull; then
        echo "Error: 'git pull' failed. Please resolve conflicts or check your connection." >&2
        echo "Aborting job submission." >&2
        return 1
    fi
    echo "--> Git pull successful."

    # 3. Get and print the latest commit information.
    #    %h = abbreviated commit hash, %s = subject (commit message first line)
    local latest_commit
    latest_commit=$(git log -1 --pretty=format:"%h - %s")
    echo -e "--> Running from commit: \033[0;35m${latest_commit}\033[0m"                                                                                                                                                                                                                                            206,1         85%
                                                                                                                                                      
    # 4. If git pull was successful, call the sbt function with all arguments.                                                                                 
    echo "--> Proceeding with job submission..."                                                                                                                
    sbt "$@"
}          

alias pip='if command -v uv &> /dev/null; then uv pip; else python -m pip; fi'
alias activate='source venv/bin/activate &> /dev/null; source .venv/bin/activate &> /dev/null' # tries to activate a virtual environment inside current folder
function cd { # activates virtual environment after changing directory
    builtin cd "$@"; activate
}