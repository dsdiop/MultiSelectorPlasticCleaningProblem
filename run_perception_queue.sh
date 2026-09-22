#!/usr/bin/env bash
set -uo pipefail

# =============================================================================
# Run the 3 queued "perception fixes" experiments (see
# FIXES_AND_TRAINING_QUEUE.md) back to back on ONE GPU, so they never contend
# for the same device. Each run must finish (done.json present) before the
# next one launches; if a run crashes, the queue stops instead of burning GPU
# time on runs built on top of a broken baseline.
#
# Usage:
#   ./run_perception_queue.sh [device] [seed]
#
# Meant to be started inside its own tmux session, e.g.:
#   tmux new-session -d -s malaga_port_F4_PerceptionQueue './run_perception_queue.sh 1 0'
#   tmux attach -t malaga_port_F4_PerceptionQueue
#
# Each individual run still gets its own nested tmux session + log file from
# run_hard_role_training.sh, exactly as if launched by hand -- this script
# only adds sequencing on top.
# =============================================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

DEVICE="${1:-1}"
SEED="${2:-0}"
OUTPUT_DIR="Learning/ctde_ram/outputs"
PRESET="ppot1wpoppreftokenmb512"

run_one() {
    local label="$1"
    shift
    echo "============================================================"
    echo "[queue] launching: ${label}"
    echo "[queue] env overrides: $*"
    echo "============================================================"

    local out
    out="$(env "$@" EPISODES=20000 ./run_hard_role_training.sh "${PRESET}" "${DEVICE}" "${SEED}" 2>&1)"
    echo "${out}"

    local run_name session
    run_name="$(echo "${out}" | sed -n 's/^\[launch\] run_name[[:space:]]*: //p')"
    session="$(echo "${out}" | sed -n 's/^\[launch\] tmux[[:space:]]*: //p')"
    if [[ -z "${run_name}" || -z "${session}" ]]; then
        echo "[queue][error] could not parse run_name/tmux session for '${label}' from launcher output."
        echo "[queue] stopping queue -- remaining runs NOT launched."
        exit 1
    fi

    echo "[queue] waiting for ${run_name} (tmux session ${session}) to finish..."
    while tmux has-session -t "=${session}" 2>/dev/null; do
        sleep 60
    done

    if [[ ! -f "${OUTPUT_DIR}/${run_name}/done.json" ]]; then
        echo "[queue][error] ${run_name} ended without done.json -- it likely crashed."
        echo "[queue][error] check: ${OUTPUT_DIR}/_tmux_logs/${run_name}.log"
        echo "[queue] stopping queue -- remaining runs NOT launched."
        exit 1
    fi
    echo "[queue] ${run_name} finished OK ($(date))."
    echo
}

run_one "Run A: Perception_CoordConv (fix 2+4, vs existing WPOP_MB512 baseline)"
run_one "Run B: Perception_WorldAgentSplit (fix 5, vs Run A)" WORLD_AGENT_SPLIT=1
run_one "Run C: Perception_VectorCriticFixed (fix 1, vs Run B)" WORLD_AGENT_SPLIT=1 PPO_CRITIC_MODE=vector PPO_ADVANTAGE_SCALARIZATION=ws

echo "============================================================"
echo "[queue] ALL 3 RUNS FINISHED"
echo "============================================================"
