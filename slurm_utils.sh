# ==============================================================================
# GPU CLUSTER UTILITIES
# ==============================================================================

# ------------------------------------------------------------------------------
# Function: avail_gpus
# ------------------------------------------------------------------------------
# Description:
#   Scans the cluster to find GPU types that are currently available for
#   immediate use. It parses `sinfo` output to calculate true availability
#   (Total - Used), enforces minimum CPU/RAM availability, and applies strict
#   state filtering.
#
# Logic:
#   1. Runs `sinfo` with extended column widths to handle long MIG strings.
#   2. Filters out nodes that are:
#      - Drained (drain, drng)
#      - Down (down, fail)
#      - Maintenance (maint)
#      - Unreachable/Not Responding (*)
#   3. Optionally filters partitions by MaxTime (must be >= requested).
#   4. Requires at least 10 idle CPU cores and 32 GB available RAM.
#   5. Parses 'Gres' and 'GresUsed' columns using Python regex.
#   6. Returns a sorted, space-separated list of GPU types that have at
#      least one free slot on a valid, active node.
#
# Usage:
#   avail_gpus [-n] [-t <max_time>]
#
# Output Example:
#   h100 nvidia_h100_80gb_hbm3_1g.10gb nvidia_h100_80gb_hbm3_3g.40gb
#   h100:4 nvidia_h100_80gb_hbm3_1g.10gb:2 nvidia_h100_80gb_hbm3_3g.40gb:1
# ------------------------------------------------------------------------------
avail_gpus() {
    local show_counts=0
    local opt
    local OPTIND=1
    local max_time=""

    while getopts ":nt:" opt; do
        case "$opt" in
            n) show_counts=1 ;;
            t) max_time="$OPTARG" ;;
            *) echo "Usage: avail_gpus [-n] [-t <max_time>]" >&2; return 2 ;;
        esac
    done
    shift $((OPTIND - 1))

    if [[ $# -ne 0 ]]; then
        echo "Usage: avail_gpus [-n] [-t <max_time>]" >&2
        return 2
    fi

    local partition_filter=""
    if [[ -n "$max_time" ]]; then
        partition_filter=$(scontrol show partition -o 2>/dev/null | python3 -c "
import sys, re, math

def parse_time_to_seconds(value):
    value = value.strip()
    if not value:
        return None
    lower = value.lower()
    if lower in ('infinite', 'unlimited'):
        return math.inf
    days = 0
    if '-' in value:
        days_part, value = value.split('-', 1)
        try:
            days = int(days_part)
        except ValueError:
            return None
    parts = value.split(':')
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (int(p) for p in parts)
        elif len(parts) == 2:
            hours, minutes = (int(p) for p in parts)
            seconds = 0
        elif len(parts) == 1:
            hours = 0
            minutes = int(parts[0])
            seconds = 0
        else:
            return None
    except ValueError:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds

req = parse_time_to_seconds(sys.argv[1])
if req is None:
    sys.stderr.write(f'Invalid max-time: {sys.argv[1]}\\n')
    sys.exit(2)

allowed = []
for line in sys.stdin:
    m_name = re.search(r'PartitionName=(\\S+)', line)
    m_time = re.search(r'MaxTime=(\\S+)', line)
    if not m_name or not m_time:
        continue
    max_time = parse_time_to_seconds(m_time.group(1))
    if max_time is None:
        continue
    if max_time >= req:
        allowed.append(m_name.group(1))

print(','.join(allowed))
" "$max_time")
        local partition_status=$?
        if [[ $partition_status -eq 2 ]]; then
            echo "Usage: avail_gpus [-n] [-t <max_time>]" >&2
            return 2
        elif [[ $partition_status -ne 0 ]]; then
            echo "Warning: failed to parse partition MaxTime; proceeding without partition filter." >&2
            partition_filter=""
        elif [[ -z "$partition_filter" ]]; then
            echo "Warning: no partitions with MaxTime >= $max_time" >&2
            return 0
        fi
    fi

    # 1. We include NodeList again to help with debugging/logging.
    local mem_mode="alloc"
    local sinfo_cmd="sinfo -N -h -t idle,mix,alloc"
    if [[ -n "$partition_filter" ]]; then
        sinfo_cmd+=" -p $partition_filter"
    fi
    sinfo_cmd+=" -O NodeList:40,StateCompact:20,CPUsState:20,Memory:20,AllocMem:20,Gres:5000,GresUsed:5000"
    local sinfo_output

    if ! sinfo_output=$($sinfo_cmd 2>/dev/null); then
        mem_mode="total"
        sinfo_cmd="sinfo -N -h -t idle,mix,alloc"
        if [[ -n "$partition_filter" ]]; then
            sinfo_cmd+=" -p $partition_filter"
        fi
        sinfo_cmd+=" -O NodeList:40,StateCompact:20,CPUsState:20,Memory:20,Gres:5000,GresUsed:5000"
        sinfo_output=$($sinfo_cmd) || return 1
    fi
    
    printf '%s\n' "$sinfo_output" | python3 -c "
import sys, re

show_counts = (len(sys.argv) > 1 and sys.argv[1] == '1')
mem_mode = sys.argv[2] if len(sys.argv) > 2 else 'alloc'

pattern = re.compile(r'gpu:([^:\s]+):(\d+)')
available_types = set()
available_counts = {}
seen_nodes = set()

# Debug counters
skipped_bad_state = 0
skipped_no_gres = 0
skipped_cpu = 0
skipped_mem = 0

MIN_IDLE_CPU = 10
MIN_AVAIL_MEM_MB = 32768

# States that are definitely broken/unusable
# (Note: We do NOT filter 'mix-' specifically, only explicit drain/fail flags)
bad_states = ['drain', 'drng', 'down', 'fail', 'maint']

def parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

for line in sys.stdin:
    parts = line.strip().split()
    if len(parts) < 4: continue

    # 2. Dynamic Column Parsing
    # We explicitly look for the columns containing 'gpu:' to handle weird spacing
    # default structure: Node [0], State [1], Gres [?], GresUsed [?]
    node = parts[0]
    if node in seen_nodes:
        continue
    state = parts[1].lower()

    cpu_state_str = parts[2] if len(parts) > 2 else None
    mem_total = parse_int(parts[3]) if len(parts) > 3 else None
    alloc_mem = parse_int(parts[4]) if mem_mode == 'alloc' and len(parts) > 4 else None
    
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

    # 3. Filter: Require minimum CPU and memory availability
    idle_cpu = None
    if cpu_state_str and re.match(r'^\d+/\d+/\d+/\d+$', cpu_state_str):
        idle_cpu = int(cpu_state_str.split('/')[1])
    if idle_cpu is None or idle_cpu < MIN_IDLE_CPU:
        skipped_cpu += 1
        continue
        

    mem_available = None
    if mem_total is not None:
        if mem_mode == 'alloc':
            mem_available = mem_total - alloc_mem if alloc_mem is not None else mem_total
        else:
            mem_available = mem_total
    if mem_available is None or mem_available < MIN_AVAIL_MEM_MB:
        skipped_mem += 1
        continue

    if gres_total_str == '(null)':
        skipped_no_gres += 1
        continue

    seen_nodes.add(node)
    
    # 4. Math
    totals = {}
    for m in pattern.finditer(gres_total_str): 
        totals[m.group(1)] = int(m.group(2))
    
    used = {}
    if gres_used_str != '(null)':
        for m in pattern.finditer(gres_used_str): 
            used[m.group(1)] = used.get(m.group(1), 0) + int(m.group(2))

    for g_type, total_count in totals.items():
        free_count = total_count - used.get(g_type, 0)
        if free_count > 0:
            if show_counts:
                available_counts[g_type] = available_counts.get(g_type, 0) + free_count
            else:
                available_types.add(g_type)

# Output results
if show_counts:
    print(' '.join(f'{g}:{available_counts[g]}' for g in sorted(available_counts)))
else:
    print(' '.join(sorted(available_types)))

# Optional: Print summary to stderr if list is suspiciously empty
found_any = bool(available_counts) if show_counts else bool(available_types)
if not found_any and (skipped_bad_state + skipped_cpu + skipped_mem) > 0:
    sys.stderr.write(
        'Warning: No GPUs found. '
        f'{skipped_bad_state} nodes skipped for state, '
        f'{skipped_cpu} for CPU, {skipped_mem} for memory.\n'
    )
" "$show_counts" "$mem_mode"
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
    local time_limit
    time_limit=$(grep -E "^#SBATCH[[:space:]]+--time(=|[[:space:]])" "$script_file" | head -n 1 | sed -E "s/^#SBATCH[[:space:]]+--time(=|[[:space:]])//; s/[[:space:]].*$//")
    
    local selected_gres=""

    if [[ -n "$gpu_priority_line" ]]; then
        echo "--> Found dynamic GPU request: $gpu_priority_line"
        
        # -- Step B: Call the helper function --
        local available_types
        if [[ -n "$time_limit" ]]; then
            echo "--> Using partition MaxTime filter: $time_limit"
            available_types=$(avail_gpus -t "$time_limit")
        else
            available_types=$(avail_gpus)
        fi
        
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
    echo "tail -F ${outfile}"
    echo "(Ctrl+C to stop)--------------------------------------------------"
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
