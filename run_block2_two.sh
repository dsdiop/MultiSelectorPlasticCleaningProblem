#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run ONE Block-2 CTDE-RAM improvement experiment in tmux.
#
# Usage:
#   ./run_block2_two.sh <experiment> <device> [seed]
#
# Examples:
#   ./run_block2_two.sh SAttnHard 1
#   ./run_block2_two.sh SAttnFull 1
#   ./run_block2_two.sh HFactoredQMIX 1
#   ./run_block2_two.sh HFactoredFull 1 2
#
# device:
#   -1 = CPU
#    0 = cuda:0
#    1 = cuda:1
# ============================================================

if [[ $# -lt 2 ]]; then
    echo "Usage:"
    echo "  $0 <experiment> <device> [seed]"
    echo
    echo "Soft-attention experiments:"
    echo "  SAttnT10          T_role=10 baseline"
    echo "  SAttnHard         hard_argmax execution"
    echo "  SAttnGumbel       straight-through Gumbel execution"
    echo "  SAttnPenalty      switch penalty (lambda=0.1)"
    echo "  SAttnHPR          hindsight preference replay"
    echo "  SAttnFilm         FiLM preference conditioning"
    echo "  SAttnFull         hard execution + penalty + HPR + FiLM"
    echo
    echo "Hard-factored experiments:"
    echo "  HFactoredT10      T_role=10 baseline"
    echo "  HFactoredDueling  dueling RAM head"
    echo "  HFactoredQMIX     monotonic QMIX mixer"
    echo "  HFactoredFull     penalty + HPR + FiLM + dueling + QMIX"
    echo
    echo "Examples:"
    echo "  $0 SAttnHard 1"
    echo "  $0 SAttnFull 1"
    echo "  $0 HFactoredQMIX 1"
    exit 1
fi

EXP_RAW="$1"
DEVICE="$2"
SEED="${3:-0}"

# ---- User config -------------------------------------------------------------

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

RUN_SCRIPT="${PROJECT_ROOT}/Learning/ctde_ram/run_experiment.py"
OUTPUT_DIR="Learning/ctde_ram/outputs"
PATH_PLANNER_FOLDER="Experimento_clean28_malaga_port_macro_plastic_random_nus_nsteps5_distbudget100_old_reward"

# Block 2 uses a genuine high-level window instead of per-step role decisions.
EPISODES=20000
T_ROLE=10
EVAL_EVERY=2000
EVAL_EPISODES=10
SAVE_EVERY=15000
EVAL_POINTS=10
PROBE_POINTS=10
PROBE_EPISODES=10
SWITCH_PENALTY=0.1

# ---- Resolve experiment ------------------------------------------------------

EXP="$(echo "${EXP_RAW}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"

RAM_ARGS=()
RUN_NAME=""

SATTN_BASE=(
    --ram-mode soft_v2
    --soft-ram-arch attention
    --global-agg attention
    --role-state-mode pooled
    --soft-ram-temperature 0.3
    --role-scalarization ewc
    --q-scalarization ewc
)

HFACTORED_BASE=(
    --ram-mode factored
    --global-agg attention
    --role-state-mode flat
)

case "${EXP}" in
    sattnt10|sattn_t10|soft_baseline)
        RUN_NAME="malaga_B2_SAttn_T10_tau03_beta04_s${SEED}"
        RAM_ARGS=("${SATTN_BASE[@]}")
        ;;

    sattnhard|sattn_hard|hard_argmax)
        RUN_NAME="malaga_B2_SAttn_hard_T10_tau03_beta04_s${SEED}_EP_${EPISODES}"
        RAM_ARGS=("${SATTN_BASE[@]}" --w-execution hard_argmax)
        ;;

    sattngumbel|sattn_gumbel|st_gumbel)
        RUN_NAME="malaga_B2_SAttn_stgumbel_T10_tau03_beta04_s${SEED}"
        RAM_ARGS=("${SATTN_BASE[@]}" --w-execution st_gumbel)
        ;;

    sattnpenalty|sattn_penalty|switch_penalty)
        RUN_NAME="malaga_B2_SAttn_penalty01_T10_tau03_beta04_s${SEED}"
        RAM_ARGS=("${SATTN_BASE[@]}" --role-switch-penalty "${SWITCH_PENALTY}")
        ;;

    sattnhpr|sattn_hpr|hpr)
        RUN_NAME="malaga_B2_SAttn_hpr05_k1_T10_tau03_beta04_s${SEED}"
        RAM_ARGS=("${SATTN_BASE[@]}" --hpr --hpr-fraction 0.5 --hpr-kappa 1.0)
        ;;

    sattnfilm|sattn_film|film)
        RUN_NAME="malaga_B2_SAttn_film_T10_tau03_beta04_s${SEED}"
        RAM_ARGS=("${SATTN_BASE[@]}" --w-conditioning film)
        ;;

    sattnfull|sattn_full|soft_full)
        RUN_NAME="malaga_B2_SAttn_hard_penalty01_hpr_film_T10_tau03_beta04_s${SEED}"
        RAM_ARGS=(
            "${SATTN_BASE[@]}"
            --w-execution hard_argmax
            --role-switch-penalty "${SWITCH_PENALTY}"
            --hpr
            --hpr-fraction 0.5
            --hpr-kappa 1.0
            --w-conditioning film
        )
        ;;

    hfactoredt10|hfactored_t10|factored_baseline)
        RUN_NAME="malaga_B2_HFactored_T10_beta04_s${SEED}"
        RAM_ARGS=("${HFACTORED_BASE[@]}")
        ;;

    hfactoreddueling|hfactored_dueling|dueling)
        RUN_NAME="malaga_B2_HFactored_dueling_T10_beta04_s${SEED}"
        RAM_ARGS=("${HFACTORED_BASE[@]}" --ram-dueling)
        ;;

    hfactoredqmix|hfactored_qmix|qmix)
        RUN_NAME="malaga_B2_HFactored_qmix_T10_beta04_s${SEED}"
        RAM_ARGS=("${HFACTORED_BASE[@]}" --hfactored-mixer qmix)
        ;;

    hfactoredfull|hfactored_full|factored_full)
        RUN_NAME="malaga_B2_HFactored_penalty01_hpr_film_dueling_qmix_T10_beta04_s${SEED}"
        RAM_ARGS=(
            "${HFACTORED_BASE[@]}"
            --role-switch-penalty "${SWITCH_PENALTY}"
            --hpr
            --hpr-fraction 0.5
            --hpr-kappa 1.0
            --w-conditioning film
            --ram-dueling
            --hfactored-mixer qmix
        )
        ;;

    *)
        echo "[error] Unknown experiment: ${EXP_RAW}"
        echo "Run without arguments to see the valid Block-2 experiments."
        exit 1
        ;;
esac

SESSION_NAME="${RUN_NAME}"
LOG_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}/_tmux_logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

# ---- Pre-flight checks -------------------------------------------------------

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[error] tmux session already exists:"
    echo "  ${SESSION_NAME}"
    echo
    echo "Attach with:"
    echo "  tmux attach -t ${SESSION_NAME}"
    exit 1
fi

if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "[error] run_experiment.py not found:"
    echo "  ${RUN_SCRIPT}"
    exit 1
fi

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "[error] No active python found in PATH."
    echo "Activate the training environment and rerun."
    exit 1
fi

mkdir -p "${LOG_DIR}"

# ---- Build command -----------------------------------------------------------

COMMON_ARGS=(
    --env project
    --project-control expert_nu
    --path-planner-folder "${PATH_PLANNER_FOLDER}"
    --map-name malaga_port
    --N 4
    --episodes "${EPISODES}"
    --T-role "${T_ROLE}"

    --role-reward-norm minmax
    --role-scalarization ws
    --q-scalarization ws
    --ram-reward-mode component_rewards

    --warmup-episodes 10
    --seed "${SEED}"
    --device "${DEVICE}"

    --weight-sampling beta
    --weight-alpha 0.4

    --eval-every "${EVAL_EVERY}"
    --eval-episodes "${EVAL_EPISODES}"
    --save-every "${SAVE_EVERY}"
    --eval-points "${EVAL_POINTS}"

    --probe-preference-sensitivity
    --probe-points "${PROBE_POINTS}"
    --probe-episodes "${PROBE_EPISODES}"

    --check-aggregator-grad
    --check-frozen-popart

    --run-name "${RUN_NAME}"
    --output-dir "${OUTPUT_DIR}"
)

CMD_FILE="${LOG_DIR}/${RUN_NAME}.cmd.sh"

{
    echo "#!/usr/bin/env bash"
    echo "cd '${PROJECT_ROOT}'"
    printf "%q " "${PYTHON_BIN}" "${RUN_SCRIPT}" "${COMMON_ARGS[@]}" "${RAM_ARGS[@]}"
    echo
} > "${CMD_FILE}"
chmod +x "${CMD_FILE}"

# ---- Launch ------------------------------------------------------------------

echo "[launch] experiment: ${EXP_RAW}"
echo "[launch] run_name:   ${RUN_NAME}"
echo "[launch] device:     ${DEVICE}"
echo "[launch] T_role:     ${T_ROLE}"
echo "[launch] tmux:       ${SESSION_NAME}"
echo "[launch] log:        ${LOG_FILE}"
echo "[launch] command:    ${CMD_FILE}"
echo

tmux new-session -d -s "${SESSION_NAME}" bash -lc "
    set -uo pipefail
    cd '${PROJECT_ROOT}'

    echo '============================================================'
    echo '[run] ${RUN_NAME}'
    echo '[start]' \$(date)
    echo '[host]' \$(hostname)
    echo '[project] ${PROJECT_ROOT}'
    echo '[python] ${PYTHON_BIN}'
    echo '[device] ${DEVICE}'
    echo '[T_role] ${T_ROLE}'
    echo '============================================================'
    echo

    set +e
    '${CMD_FILE}' 2>&1 | tee '${LOG_FILE}'
    TRAIN_EXIT_CODE=\${PIPESTATUS[0]}
    set -e

    echo
    echo '============================================================'
    echo '[done] ${RUN_NAME} exit code' \${TRAIN_EXIT_CODE}
    echo '[finish]' \$(date)
    echo '============================================================'
    exit \${TRAIN_EXIT_CODE}
"

echo "[ok] launched."
echo
echo "Attach while training:"
echo "  tmux attach -t ${SESSION_NAME}"
echo
echo "Detach inside tmux:"
echo "  Ctrl+B, then D"
echo
echo "Tail the persistent log:"
echo "  tail -f '${LOG_FILE}'"
