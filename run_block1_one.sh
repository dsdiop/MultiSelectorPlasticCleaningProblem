#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run ONE Block-1 CTDE-RAM experiment inside ONE tmux session.
#
# Usage:
#   ./run_block1_one.sh <experiment> <device> [seed]
#
# Examples:
#   ./run_block1_one.sh SAttn 1
#   ./run_block1_one.sh SMLP 1
#   ./run_block1_one.sh SMean 1
#   ./run_block1_one.sh HDiscrete 1
#   ./run_block1_one.sh Random -1
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
    echo "Experiments:"
    echo "  SAttn      main soft_v2 attention selector + attention aggregator + pooled"
    echo "  SMLP       soft_v2 MLP selector"
    echo "  SMean      soft_v2 attention selector + mean_pool aggregator"
    echo "  SFlat      soft_v2 attention selector + flat role state"
    echo "  HDiscrete  hard discrete joint RAM"
    echo "  HFactored  hard factored RAM"
    echo "  Random     random RAM control"
    echo
    echo "Examples:"
    echo "  $0 SAttn 1"
    echo "  $0 HDiscrete 1"
    echo "  $0 Random -1"
    exit 1
fi

EXP_RAW="$1"
DEVICE="$2"
SEED="${3:-0}"

# ---- User config -------------------------------------------------------------

# Put this script in the repository root, or override:
#   PROJECT_ROOT=/path/to/repo ./run_block1_one.sh SAttn 1
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"

# Use the current active Python environment from the terminal.
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

RUN_SCRIPT="${PROJECT_ROOT}/Learning/ctde_ram/run_experiment.py"
OUTPUT_DIR="Learning/ctde_ram/outputs"

PATH_PLANNER_FOLDER="Experimento_clean28_malaga_port_macro_plastic_random_nus_nsteps5_distbudget100_old_reward"

# Serious training protocol.
EPISODES=5000
T_ROLE=1
EVAL_EVERY=500
EVAL_EPISODES=10
SAVE_EVERY=500
EVAL_POINTS=10
PROBE_POINTS=10
PROBE_EPISODES=10

# ---- Resolve experiment ------------------------------------------------------

EXP="$(echo "${EXP_RAW}" | tr '[:upper:]' '[:lower:]' | tr '-' '_' )"

RAM_ARGS=()
RUN_NAME=""

case "${EXP}" in
    sattn|main|soft_attention)
        RUN_NAME="malaga_SAttn_comp_wsws_T1_tau03_beta04_s${SEED}"
        RAM_ARGS=(
            --ram-mode soft_v2
            --soft-ram-arch attention
            --global-agg attention
            --role-state-mode pooled
            --soft-ram-temperature 0.3
        )
        ;;

    smlp|mlp|soft_mlp)
        RUN_NAME="malaga_SMLP_comp_wsws_T1_tau03_beta04_s${SEED}"
        RAM_ARGS=(
            --ram-mode soft_v2
            --soft-ram-arch mlp
            --global-agg attention
            --role-state-mode pooled
            --soft-ram-temperature 0.3
        )
        ;;

    smean|mean|mean_pool|soft_mean)
        RUN_NAME="malaga_SMean_comp_wsws_T1_tau03_beta04_s${SEED}"
        RAM_ARGS=(
            --ram-mode soft_v2
            --soft-ram-arch attention
            --global-agg mean_pool
            --role-state-mode pooled
            --soft-ram-temperature 0.3
        )
        ;;

    sflat|flat|soft_flat)
        RUN_NAME="malaga_SFlat_comp_wsws_T1_tau03_beta04_s${SEED}"
        RAM_ARGS=(
            --ram-mode soft_v2
            --soft-ram-arch attention
            --global-agg attention
            --role-state-mode flat
            --soft-ram-temperature 0.3
        )
        ;;

    hdiscrete|discrete|hard_discrete)
        RUN_NAME="malaga_HDiscrete_comp_ws_T1_beta04_s${SEED}"
        RAM_ARGS=(
            --ram-mode discrete
            --global-agg attention
            --role-state-mode flat
        )
        ;;

    hfactored|factored|hard_factored)
        RUN_NAME="malaga_HFactored_comp_ws_T1_beta04_s${SEED}"
        RAM_ARGS=(
            --ram-mode factored
            --global-agg attention
            --role-state-mode flat
        )
        ;;

    random|rand)
        RUN_NAME="malaga_Random_comp_T1_beta04_s${SEED}"
        RAM_ARGS=(
            --ram-mode random
        )
        ;;

    *)
        echo "[error] Unknown experiment: ${EXP_RAW}"
        echo
        echo "Valid experiments:"
        echo "  SAttn"
        echo "  SMLP"
        echo "  SMean"
        echo "  SFlat"
        echo "  HDiscrete"
        echo "  HFactored"
        echo "  Random"
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
    echo "Activate your environment first, then rerun:"
    echo "  conda activate envDRL0"
    echo "  ./run_block1_one.sh SAttn 1"
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

    --run-name "${RUN_NAME}"
    --output-dir "${OUTPUT_DIR}"
)

# Save exact command for reproducibility.
CMD_FILE="${LOG_DIR}/${RUN_NAME}.cmd.sh"

{
    echo "#!/usr/bin/env bash"
    echo "cd '${PROJECT_ROOT}'"
    printf "%q " "${PYTHON_BIN}" "${RUN_SCRIPT}" "${COMMON_ARGS[@]}" "${RAM_ARGS[@]}"
    echo
} > "${CMD_FILE}"
chmod +x "${CMD_FILE}"

# ---- Launch one tmux session, one training process, kill tmux terminal after completion ---------------------------

echo "[launch] experiment: ${EXP_RAW}"
echo "[launch] run_name:   ${RUN_NAME}"
echo "[launch] device:     ${DEVICE}"
echo "[launch] tmux:       ${SESSION_NAME}"
echo "[launch] log:        ${LOG_FILE}"
echo "[launch] command:    ${CMD_FILE}"
echo

# 1. Ejecutamos el bloque de comandos. Al terminar el último comando, la sesión morirá sola.
tmux new-session -d -s "${SESSION_NAME}" bash -lc "
    # Nota: Quitamos 'set -e' temporalmente para que el bloque final de 'echo' 
    # se ejecute incluso si tu script de entrenamiento falla (exit code > 0).
    set -uo pipefail
    cd '${PROJECT_ROOT}'

    echo '============================================================'
    echo '[run] ${RUN_NAME}'
    echo '[start]' \$(date)
    echo '[host]' \$(hostname)
    echo '[project] ${PROJECT_ROOT}'
    echo '[python] ${PYTHON_BIN}'
    echo '[device] ${DEVICE}'
    echo '============================================================'
    echo

    # Guardamos el código de salida del entrenamiento
    set +e
    '${CMD_FILE}' 2>&1 | tee '${LOG_FILE}'
    TRAIN_EXIT_CODE=\${PIPESTATUS[0]}
    set -e

    echo
    echo '============================================================'
    echo '[done] ${RUN_NAME} con estado \${TRAIN_EXIT_CODE}'
    echo '[finish]' \$(date)
    echo '============================================================'

    # 2. Forzamos la salida devolviendo el estado real del entrenamiento
    exit \${TRAIN_EXIT_CODE}
"

echo "[ok] launched."
echo
echo "Attach (solo mientras entrene):"
echo "  tmux attach -t ${SESSION_NAME}"
echo
echo "Detach inside tmux:"
echo "  Ctrl+B, then D"
echo
echo "Tail log (disponible siempre):"
echo "  tail -f '${LOG_FILE}'"
