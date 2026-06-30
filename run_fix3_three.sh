#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run ONE F3 CTDE-RAM experiment in tmux.
#
# F3 = post reward-normalization fix (per-window cumsum; no norm for
# delta_metrics) + live/normalized reward-state channel. Numbers from
# nonlinear scalarizations are NOT comparable to pre-fix (B2/B1) runs.
#
# Usage:
#   ./run_fix3_three.sh <experiment> <device> [seed]
#
# device:  -1 = CPU,  0 = cuda:0,  1 = cuda:1
# ============================================================

if [[ $# -lt 2 ]]; then
    echo "Usage:"
    echo "  $0 <experiment> <device> [seed]"
    echo
    echo "PHASE 1 - main method (soft learning + hard execution):"
    echo "  SAttnHardWS        SAttn, hard_argmax, ws/ws        <- expected main method"
    echo "  SAttnHardEWC       SAttn, hard_argmax, ewc/ewc      <- nonlinear ablation"
    echo "  SAttnHardWPOP      SAttn, hard_argmax, wpop/wpop    <- nonlinear ablation"
    echo "  SAttnSoftWS        SAttn, soft execution, ws/ws     <- execution ablation"
    echo "  SAttnGumbelWS      SAttn, st_gumbel, ws/ws          <- execution ablation"
    echo
    echo "PHASE 2 - single-tweak ablations ON TOP of the SAttnHardWS base (one at a time):"
    echo "  SAttnHardHPR       + hindsight preference replay"
    echo "  SAttnHardFilm      + FiLM preference conditioning"
    echo "  SAttnHardPenalty   + relative switch penalty (lambda_rel=0.05)"
    echo
    echo "PHASE 2A - single-tweak ablations ON TOP of the SAttnHardEWC base (one at a time):" 
    echo "  SAttnHardEWC_HPR   + hindsight preference replay"
    echo "  SAttnHardEWC_Film  + FiLM preference conditioning"
    echo "  SAttnHardEWC_Penalty + relative switch penalty (lambda_rel=0.05)"
    echo
    echo "PHASE 3 - competitors / controls:"
    echo "  HFactoredWS        hard factored RAM, ws"
    echo "  HFactoredEWC       hard factored RAM, ewc"
    echo "  HDiscreteWS        joint discrete RAM, ws (negative result, M2-fail)"
    echo "  SMeanHardWS        mean-pool aggregator, hard_argmax, ws"
    echo "  RandomCtrl         random role control (floor)"
    echo
    echo "PHASE 4 - reward-mode contrast (label-only, not mixed in main table):"
    echo "  SAttnHardWS_DM     SAttnHardWS but ram-reward-mode delta_metrics"
    echo
    echo "Examples:"
    echo "  $0 SAttnHardWS 1"
    echo "  $0 SAttnHardEWC 1 2"
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

# ---- Shared schedule ---------------------------------------------------------
EPISODES=5000
T_ROLE=10
EVAL_EVERY=1000
EVAL_EPISODES=30
SAVE_EVERY=15000
EVAL_POINTS=20
PROBE_POINTS=20
PROBE_EPISODES=100
SWITCH_PENALTY_REL=0.05
WEIGHT_ALPHA=0.4
TAU=0.3

# ---- Per-experiment knobs (defaults; cases override) -------------------------
REWARD_MODE="component_rewards"   # component_rewards | delta_metrics
REWARD_TAG="comp"                 # short tag for the run name
ROLE_SCAL="ws"
Q_SCAL="ws"
W_EXEC="soft"                     # soft | hard_argmax | st_gumbel
EXEC_TAG="soft"

RAM_ARGS=()
RUN_NAME=""

SATTN_BASE=(
    --ram-mode soft_v2
    --soft-ram-arch attention
    --global-agg attention
    --role-state-mode pooled
    --soft-ram-temperature "${TAU}"
)

HFACTORED_BASE=(
    --ram-mode factored
    --global-agg attention
    --role-state-mode flat
)

# helper: build a descriptive run name with all the things that matter
#   <map>_F3_<method>_<exec>_<roleScal><qScal>_<rewardTag>_T<Trole>_ep<EP>_beta<alpha>_s<seed>
mkname () {
    local method="$1"
    echo "malaga_F3_${method}_${EXEC_TAG}_${ROLE_SCAL}${Q_SCAL}_${REWARD_TAG}_T${T_ROLE}_ep${EPISODES}_beta${WEIGHT_ALPHA}_s${SEED}"
}

EXP="$(echo "${EXP_RAW}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"

case "${EXP}" in
    # ---------------- PHASE 1: main + execution/scalarization ----------------
    sattnhardws|sattn_hard_ws)
        ROLE_SCAL="ws"; Q_SCAL="ws"; W_EXEC="hard_argmax"; EXEC_TAG="hard"
        RAM_ARGS=("${SATTN_BASE[@]}" --w-execution hard_argmax --role-scalarization ws --q-scalarization ws)
        RUN_NAME="$(mkname SAttn)"
        ;;

    sattnhardewc|sattn_hard_ewc)
        ROLE_SCAL="ewc"; Q_SCAL="ewc"; W_EXEC="hard_argmax"; EXEC_TAG="hard"
        RAM_ARGS=("${SATTN_BASE[@]}" --w-execution hard_argmax --role-scalarization ewc --q-scalarization ewc)
        RUN_NAME="$(mkname SAttn)"
        ;;
    sattnhardwpop|sattn_hard_wpop)
        ROLE_SCAL="wpop"; Q_SCAL="wpop"; W_EXEC="hard_argmax"; EXEC_TAG="hard"
        RAM_ARGS=("${SATTN_BASE[@]}" --w-execution hard_argmax --role-scalarization wpop --q-scalarization wpop)
        RUN_NAME="$(mkname SAttn)"
        ;;
    sattnsoftws|sattn_soft_ws)
        ROLE_SCAL="ws"; Q_SCAL="ws"; W_EXEC="soft"; EXEC_TAG="soft"
        RAM_ARGS=("${SATTN_BASE[@]}" --w-execution soft --role-scalarization ws --q-scalarization ws)
        RUN_NAME="$(mkname SAttn)"
        ;;

    sattngumbelws|sattn_gumbel_ws)
        ROLE_SCAL="ws"; Q_SCAL="ws"; W_EXEC="st_gumbel"; EXEC_TAG="stgumbel"
        RAM_ARGS=("${SATTN_BASE[@]}" --w-execution st_gumbel --role-scalarization ws --q-scalarization ws)
        RUN_NAME="$(mkname SAttn)"
        ;;

    # ---------------- PHASE 2: single-tweak ablations on SAttnHardWS ----------------
    sattnhardhpr|sattn_hard_hpr)
        ROLE_SCAL="ws"; Q_SCAL="ws"; W_EXEC="hard_argmax"; EXEC_TAG="hard"
        RAM_ARGS=("${SATTN_BASE[@]}" --w-execution hard_argmax --role-scalarization ws --q-scalarization ws \
                  --hpr --hpr-fraction 0.5 --hpr-kappa 1.0)
        RUN_NAME="$(mkname SAttn)_hpr05k1"
        ;;

    sattnhardfilm|sattn_hard_film)
        ROLE_SCAL="ws"; Q_SCAL="ws"; W_EXEC="hard_argmax"; EXEC_TAG="hard"
        RAM_ARGS=("${SATTN_BASE[@]}" --w-execution hard_argmax --role-scalarization ws --q-scalarization ws \
                  --w-conditioning film)
        RUN_NAME="$(mkname SAttn)_film"
        ;;

    sattnhardpenalty|sattn_hard_penalty)
        ROLE_SCAL="ws"; Q_SCAL="ws"; W_EXEC="hard_argmax"; EXEC_TAG="hard"
        RAM_ARGS=("${SATTN_BASE[@]}" --w-execution hard_argmax --role-scalarization ws --q-scalarization ws \
                  --role-switch-penalty-rel "${SWITCH_PENALTY_REL}")
        RUN_NAME="$(mkname SAttn)_penrel05"
        ;;
    # ---------------- PHASE 2A: single-tweak ablations on SAttnHardEWC ----------------
    sattnhardewc_hpr|sattn_hard_ewc_hpr)
        ROLE_SCAL="ewc"; Q_SCAL="ewc"; W_EXEC="hard_argmax"; EXEC_TAG="hard"
        RAM_ARGS=("${SATTN_BASE[@]}" --w-execution hard_argmax --role-scalarization ewc --q-scalarization ewc \
                  --hpr --hpr-fraction 0.5 --hpr-kappa 1.0)
        RUN_NAME="$(mkname SAttn)_hpr05k1"
        ;;
    sattnhardewc_film|sattn_hard_ewc_film)
        ROLE_SCAL="ewc"; Q_SCAL="ewc"; W_EXEC="hard_argmax"; EXEC_TAG="hard"
        RAM_ARGS=("${SATTN_BASE[@]}" --w-execution hard_argmax --role-scalarization ewc --q-scalarization ewc \
                  --w-conditioning film)
        RUN_NAME="$(mkname SAttn)_film"
        ;;
    sattnhardewc_penalty|sattn_hard_ewc_penalty)
        ROLE_SCAL="ewc"; Q_SCAL="ewc"; W_EXEC="hard_argmax"; EXEC_TAG="hard"
        RAM_ARGS=("${SATTN_BASE[@]}" --w-execution hard_argmax --role-scalarization ewc --q-scalarization ewc \
                  --role-switch-penalty-rel "${SWITCH_PENALTY_REL}")
        RUN_NAME="$(mkname SAttn)_penrel05"
        ;;
    # ---------------- PHASE 3: competitors / controls ----------------
    hfactoredws|hfactored_ws)
        ROLE_SCAL="ws"; Q_SCAL="ws"; W_EXEC="hard_argmax"; EXEC_TAG="hard"
        RAM_ARGS=("${HFACTORED_BASE[@]}" --role-scalarization ws --q-scalarization ws)
        RUN_NAME="$(mkname HFactored)"
        ;;

    hfactoredewc|hfactored_ewc)
        ROLE_SCAL="ewc"; Q_SCAL="ewc"; W_EXEC="hard_argmax"; EXEC_TAG="hard"
        RAM_ARGS=("${HFACTORED_BASE[@]}" --role-scalarization ewc --q-scalarization ewc)
        RUN_NAME="$(mkname HFactored)"
        ;;

    hdiscretews|hdiscrete_ws)
        ROLE_SCAL="ws"; Q_SCAL="ws"; W_EXEC="hard_argmax"; EXEC_TAG="hard"
        RAM_ARGS=(--ram-mode discrete --global-agg attention --role-state-mode flat \
                  --role-scalarization ws --q-scalarization ws)
        RUN_NAME="$(mkname HDiscrete)"
        ;;

    smeanhardws|smean_hard_ws)
        ROLE_SCAL="ws"; Q_SCAL="ws"; W_EXEC="hard_argmax"; EXEC_TAG="hard"
        RAM_ARGS=(--ram-mode soft_v2 --soft-ram-arch attention --global-agg mean_pool \
                  --role-state-mode pooled --soft-ram-temperature "${TAU}" \
                  --w-execution hard_argmax --role-scalarization ws --q-scalarization ws)
        RUN_NAME="$(mkname SMean)"
        ;;

    randomctrl|random)
        ROLE_SCAL="ws"; Q_SCAL="ws"; W_EXEC="soft"; EXEC_TAG="soft"
        RAM_ARGS=(--ram-mode random --global-agg attention --role-state-mode flat \
                  --role-scalarization ws --q-scalarization ws)
        RUN_NAME="$(mkname Random)"
        ;;

    # ---------------- PHASE 4: reward-mode contrast (labelled, not mixed) ----------------
    sattnhardws_dm|sattn_hard_ws_dm|delta_metrics)
        REWARD_MODE="delta_metrics"; REWARD_TAG="metrics"
        ROLE_SCAL="ws"; Q_SCAL="ws"; W_EXEC="hard_argmax"; EXEC_TAG="hard"
        RAM_ARGS=("${SATTN_BASE[@]}" --w-execution hard_argmax --role-scalarization ws --q-scalarization ws)
        RUN_NAME="$(mkname SAttn)"
        ;;

    *)
        echo "[error] Unknown experiment: ${EXP_RAW}"
        echo "Run without arguments to see the valid F3 experiments."
        exit 1
        ;;
esac

SESSION_NAME="${RUN_NAME}"
LOG_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}/_tmux_logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

# ---- Pre-flight checks -------------------------------------------------------
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[error] tmux session already exists: ${SESSION_NAME}"
    echo "Attach with: tmux attach -t ${SESSION_NAME}"
    exit 1
fi
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "[error] No active python found in PATH. Activate the env and rerun."
    exit 1
fi
mkdir -p "${LOG_DIR}"

# ---- Build command -----------------------------------------------------------
if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "[error] run_experiment.py not found: ${RUN_SCRIPT}"
    exit 1
fi
    COMMON_ARGS=(
        --env project
        --project-control expert_nu
        --path-planner-folder "${PATH_PLANNER_FOLDER}"
        --map-name malaga_port
        --N 4
        --episodes "${EPISODES}"
        --T-role "${T_ROLE}"

        --freeze

        --role-reward-norm minmax
        --ram-reward-mode "${REWARD_MODE}"

        --warmup-episodes 10
        --seed "${SEED}"
        --device "${DEVICE}"

        --weight-sampling beta
        --weight-alpha "${WEIGHT_ALPHA}"

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
    # NOTE: role/q scalarization come from RAM_ARGS per case, so they are NOT in
    # COMMON_ARGS (the old script hardcoded ws/ws there and made run names lie).
BUILT_CMD=( "${PYTHON_BIN}" "${RUN_SCRIPT}" "${COMMON_ARGS[@]}" "${RAM_ARGS[@]}" )

CMD_FILE="${LOG_DIR}/${RUN_NAME}.cmd.sh"
{
    echo "#!/usr/bin/env bash"
    echo "cd '${PROJECT_ROOT}'"
    printf "%q " "${BUILT_CMD[@]}"
    echo
} > "${CMD_FILE}"
chmod +x "${CMD_FILE}"

# ---- Launch ------------------------------------------------------------------
echo "[launch] experiment:  ${EXP_RAW}"
echo "[launch] run_name:     ${RUN_NAME}"
echo "[launch] reward_mode:  ${REWARD_MODE}"
echo "[launch] scalarization:${ROLE_SCAL}/${Q_SCAL}   execution: ${W_EXEC}"
echo "[launch] episodes:     ${EPISODES}    T_role: ${T_ROLE}"
echo "[launch] device:       ${DEVICE}     tmux: ${SESSION_NAME}"
echo "[launch] log:          ${LOG_FILE}"
echo

tmux new-session -d -s "${SESSION_NAME}" bash -lc "
    set -uo pipefail
    cd '${PROJECT_ROOT}'
    echo '============================================================'
    echo '[run] ${RUN_NAME}'
    echo '[reward] ${REWARD_MODE}  [scal] ${ROLE_SCAL}/${Q_SCAL}  [exec] ${W_EXEC}'
    echo '[episodes] ${EPISODES}  [T_role] ${T_ROLE}'
    echo '[start]' \$(date)  '[host]' \$(hostname)
    echo '============================================================'
    set +e
    '${CMD_FILE}' 2>&1 | tee '${LOG_FILE}'
    TRAIN_EXIT_CODE=\${PIPESTATUS[0]}
    set -e
    echo '============================================================'
    echo '[done] ${RUN_NAME} exit code' \${TRAIN_EXIT_CODE}
    echo '[finish]' \$(date)
    echo '============================================================'
    exit \${TRAIN_EXIT_CODE}
"

echo "[ok] launched."
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Tail:   tail -f '${LOG_FILE}'"
