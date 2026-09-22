#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Train ONE current hard-role RAM experiment in ONE tmux session.
#
# Usage:
#   ./run_hard_role_training.sh <experiment> <device> [seed]
#
# device:
#   -1 = CPU
#    0 = cuda:0
#    1 = cuda:1
#
# Main paper runs (run these first, with the same seeds):
#   PPO              PPO_RAM_FiLM_Attn
#   HardQ            HardRoleQ_RAM_FiLM_Attn, factorized-sum baseline
#   HardQQMIX        HardRoleQ_RAM_FiLM_Attn_QMIX
#
# Focused alternatives/ablations:
#   PPOT1            PPO with role decision every environment step
#   PPOPrefToken     replace FiLM with a preference token and two attention layers
#   PPOT1WPOP        PPO T1 with WPOP role-reward scalarization
#   PPOT1WPOPPrefToken  PPO T1, WPOP reward, preference token, two attention layers
#   PPOT1VectorCriticAdvWPOP  PPO T1, vector critic, PopArt, WPOP advantages, FiLM
#   PPOT1VectorCriticDeltaAdvWPOP  same vector PPO using raw mission-metric deltas
#   PPOT1D3PODelta  D3PO with delta metrics, vector critic, PopArt, and diversity loss
#   PPOT1D3POComponents  same D3PO setup using component rewards
#   PPOT1LogWPrefToken  PPO T1 with terminal weighted-log utility and preference token
#   PPOT1VectorWPOP  short alias for PPOT1VectorCriticAdvWPOP
#   PPOT1VectorWPOPToken  PPO T1 with vector critic, WPOP advantages, preference token and two attention layers, no popart
#   PPOT1WP          PPO T1 with WP role-reward scalarization,
#   HardQT1          HardRoleQ with role decision every environment step
#   PPOT20           longer role commitment
#   HardQT20         longer role commitment
#   PPONoEntropy     entropy coefficient = 0
#   PPOEntropyHigh   entropy coefficient = 0.03
#   PPOClipSmall     PPO clip epsilon = 0.10
#   PPOTwoLayers     two fleet-attention layers
#   HardQPERUniform  alpha = 0 (uniform replay control)
#   HardQPERStrong   alpha = 0.8
#   HardQTargetSlow  target synchronization every 200 updates
#   HardQTwoLayers   two fleet-attention layers
#   PPOPrefBias      optional preference-to-role-output bias ablation
#   HardQPrefBias    optional preference-to-role-output bias ablation
#   PPOEWC           nonlinear macro-reward scalarization ablation
#   HardQEWC         nonlinear macro-reward scalarization ablation
#
# Examples:
#   ./run_hard_role_training.sh PPO 0 0
#   ./run_hard_role_training.sh HardQ 1 0
#   ./run_hard_role_training.sh HardQQMIX 0 1
#
# Environment overrides:
#   EPISODES=20000 T_ROLE=10 MAP_NAME=malaga_port \
#   PATH_PLANNER_FOLDER=<folder> ./run_hard_role_training.sh PPO 0 0
#
# Inspect the exact generated command without launching tmux:
#   DRY_RUN=1 ./run_hard_role_training.sh PPO 0 0
# =============================================================================

if [[ $# -lt 2 ]]; then
    sed -n '4,48p' "$0"
    exit 1
fi

EXP_RAW="$1"
DEVICE="$2"
SEED="${3:-0}"

# ---- User configuration ------------------------------------------------------

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
RUN_SCRIPT="${PROJECT_ROOT}/Learning/ctde_ram/run_experiment.py"
OUTPUT_DIR="${OUTPUT_DIR:-Learning/ctde_ram/outputs}"

# Defaults follow the latest Malaga launcher. Override both for Alamillo.
MAP_NAME="${MAP_NAME:-malaga_port}"
# IF DIFFERENT MAP, MAKE SURE TO CHANGE THE PATH_PLANNER_FOLDER ACCORDINGLY. THE FOLLOWING CODE CHECK THE MAP NAME
if [[ "${MAP_NAME}" == "malaga_port" ]]; then
    PATH_PLANNER_FOLDER="${PATH_PLANNER_FOLDER:-Experimento_clean28_malaga_port_macro_plastic_random_nus_nsteps5_distbudget100_old_reward}"
elif [[ "${MAP_NAME}" == "alamillo_lake" ]]; then
    PATH_PLANNER_FOLDER="${PATH_PLANNER_FOLDER:-Experimento_clean26_alamillo_lake_macro_plastic_random_nus_nsteps5}"
else
    echo "[error] Unknown map name: ${MAP_NAME}"
    exit 1
fi

N_AGENTS="${N_AGENTS:-4}"
EPISODES="${EPISODES:-10000}"
T_ROLE="${T_ROLE:-10}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
EVAL_POINTS="${EVAL_POINTS:-20}"
PROBE_POINTS="${PROBE_POINTS:-20}"
PROBE_EPISODES="${PROBE_EPISODES:-100}"
WEIGHT_ALPHA="${WEIGHT_ALPHA:-0.4}"
D3PO_DIVERSITY_COEF="${D3PO_DIVERSITY_COEF:-0.1}"
D3PO_DIVERSITY_ALPHA="${D3PO_DIVERSITY_ALPHA:-0.5}"
ENTROPY_COEF="${ENTROPY_COEF:-0.03}"
CHEBYSHEV_REF_CLEAN="${CHEBYSHEV_REF_CLEAN:-1.0}"
CHEBYSHEV_REF_COV="${CHEBYSHEV_REF_COV:-1.0}"
CHEBYSHEV_RHO="${CHEBYSHEV_RHO:-0.05}"
REF_AUTOCALIBRATE="${REF_AUTOCALIBRATE:-0}"
STCH_MU="${STCH_MU:-0.1}"
WPOP_EPS="${WPOP_EPS:-0.01}"
PREF_WEIGHT_CLAMP="${PREF_WEIGHT_CLAMP:-0.05}"
DENSE_REWARD_MODE="${DENSE_REWARD_MODE:-none}"
DENSE_SCALARIZATION="${DENSE_SCALARIZATION:-sum}"
DENSE_TERMINAL_RATIO="${DENSE_TERMINAL_RATIO:-0.2}"
DENSE_WARMUP_EPISODES="${DENSE_WARMUP_EPISODES:-100}"
DRY_RUN="${DRY_RUN:-0}"
CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"

# Shared architecture from the hard-role specification.
D_MODEL="${D_MODEL:-64}"
N_ATTN_HEADS="${N_ATTN_HEADS:-4}"
N_ATTN_LAYERS="${N_ATTN_LAYERS:-1}"
ATTN_FF_DIM="${ATTN_FF_DIM:-128}"
HARD_ROLE_PREFERENCE_CONDITIONING="${HARD_ROLE_PREFERENCE_CONDITIONING:-film}"
WORLD_AGENT_SPLIT="${WORLD_AGENT_SPLIT:-0}"
PPO_MINIBATCH_SIZE="${PPO_MINIBATCH_SIZE:-256}"
PPO_ROLLOUT_MACRO_STEPS="${PPO_ROLLOUT_MACRO_STEPS:-2048}"
DETERMINISTIC="${DETERMINISTIC:-1}"

# ---- Resolve experiment ------------------------------------------------------

EXP="$(echo "${EXP_RAW}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
METHOD_TAG=""
MODE_ARGS=()
EXTRA_ARGS=()
ROLE_SCALARIZATION="${ROLE_SCALARIZATION:-ws}"
RAM_REWARD_MODE="${RAM_REWARD_MODE:-component_rewards}"
PPO_CRITIC_MODE="${PPO_CRITIC_MODE:-scalar}"
PPO_CRITIC_POPART="${PPO_CRITIC_POPART:-0}"
PPO_ADVANTAGE_SCALARIZATION="${PPO_ADVANTAGE_SCALARIZATION:-ws}"
HARD_ROLE_DEEP_INPUT_PROJECTIONS="${HARD_ROLE_DEEP_INPUT_PROJECTIONS:-1}"
RUN_T_ROLE="${T_ROLE}"

case "${EXP}" in
    # ---------------- Main comparison ----------------------------------------
    ppo|ppo_main|ppo_ram)
        METHOD_TAG="PPO_RAM_FiLM_Attn"
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    hardq|hard_q|hard_role_q)
        METHOD_TAG="HardRoleQ_RAM_FiLM_Attn"
        MODE_ARGS=(--ram-mode hard_role_q --role-q-mixer none)
        ;;
    hardqqmix|hardq_qmix|hard_q_qmix|qmix)
        METHOD_TAG="HardRoleQ_RAM_FiLM_Attn_QMIX"
        MODE_ARGS=(--ram-mode hard_role_q --role-q-mixer qmix)
        ;;

    # ---------------- Macro-duration sensitivity -----------------------------
    ppot1|ppo_t1)
        METHOD_TAG="PPO_RAM_FiLM_Attn_T1"
        RUN_T_ROLE=1
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    ppot1wpop|ppo_t1_wpop)
        METHOD_TAG="PPO_RAM_FiLM_Attn_T1_WPOP"
        RUN_T_ROLE=1
        ROLE_SCALARIZATION="wpop"
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    ppot1wpoppreftoken|ppo_t1_wpop_pref_token)
        METHOD_TAG="PPO_RAM_PrefToken_AttnL2_T1_WPOP"
        RUN_T_ROLE=1
        ROLE_SCALARIZATION="wpop"
        HARD_ROLE_PREFERENCE_CONDITIONING="pref_token"
        N_ATTN_LAYERS=2
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    ppot1wpoppreftokenmb512|ppo_t1_wpop_pref_token_mb512)
        METHOD_TAG="PPO_RAM_PrefToken_AttnL2_T1_WPOP_MB512"
        RUN_T_ROLE=1
        ROLE_SCALARIZATION="wpop"
        HARD_ROLE_PREFERENCE_CONDITIONING="pref_token"
        HARD_ROLE_DEEP_INPUT_PROJECTIONS=1
        N_ATTN_LAYERS=2
        PPO_MINIBATCH_SIZE=512
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    ppot1wpoppreftokenl3d128h8ff256|ppo_t1_wpop_pref_token_l3_d128_h8_ff256)
        METHOD_TAG="PPO_RAM_PrefToken_AttnL3_T1_WPOP_D128_H8_FF256"
        RUN_T_ROLE=1
        ROLE_SCALARIZATION="wpop"
        RAM_REWARD_MODE="component_rewards"
        ENTROPY_COEF=0.03
        PPO_CRITIC_MODE="scalar"
        PPO_CRITIC_POPART=0
        PPO_ADVANTAGE_SCALARIZATION="ws"
        HARD_ROLE_PREFERENCE_CONDITIONING="pref_token"
        HARD_ROLE_DEEP_INPUT_PROJECTIONS=1
        N_ATTN_LAYERS=3
        D_MODEL=128
        N_ATTN_HEADS=8
        ATTN_FF_DIM=256
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    ppot1wpoppreftokenentropy005|ppo_t1_wpop_pref_token_entropy005)
        METHOD_TAG="PPO_RAM_PrefToken_AttnL2_T1_WPOP_Entropy005"
        RUN_T_ROLE=1
        ROLE_SCALARIZATION="wpop"
        ENTROPY_COEF=0.05
        HARD_ROLE_PREFERENCE_CONDITIONING="pref_token"
        HARD_ROLE_DEEP_INPUT_PROJECTIONS=0
        N_ATTN_LAYERS=2
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    ppot1wp|ppo_t1_wp)
        METHOD_TAG="PPO_RAM_FiLM_Attn_T1_WP"
        RUN_T_ROLE=1
        ROLE_SCALARIZATION="wp"
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    ppot1chebyshev|ppo_t1_chebyshev|chebyshev_terminal)
        METHOD_TAG="PPO_RAM_FiLM_Attn_T1_ChebyshevTerminal"
        RUN_T_ROLE=1
        RAM_REWARD_MODE="chebyshev_terminal"
        PPO_CRITIC_MODE="scalar"
        PPO_CRITIC_POPART=0
        PPO_ADVANTAGE_SCALARIZATION="ws"
        HARD_ROLE_PREFERENCE_CONDITIONING="film"
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    ppot1chebyshevautopreftoken|ppo_t1_chebyshev_auto_pref_token|chebyshev_augmented_terminal)
        METHOD_TAG="PPO_RAM_PrefToken_AttnL2_T1_ChebyshevAugTerminal_AutoRef"
        RUN_T_ROLE=1
        RAM_REWARD_MODE="chebyshev_augmented_terminal"
        PPO_CRITIC_MODE="scalar"
        PPO_CRITIC_POPART=0
        PPO_ADVANTAGE_SCALARIZATION="ws"
        HARD_ROLE_PREFERENCE_CONDITIONING="pref_token"
        N_ATTN_LAYERS=2
        HARD_ROLE_DEEP_INPUT_PROJECTIONS=1
        REF_AUTOCALIBRATE=1
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    ppot1stchpreftoken|ppo_t1_stch_pref_token|stch_terminal)
        METHOD_TAG="PPO_RAM_PrefToken_AttnL2_T1_STCHTerminal_AutoRef"
        RUN_T_ROLE=1
        RAM_REWARD_MODE="stch_terminal"
        PPO_CRITIC_MODE="scalar"
        PPO_CRITIC_POPART=0
        PPO_ADVANTAGE_SCALARIZATION="ws"
        HARD_ROLE_PREFERENCE_CONDITIONING="pref_token"
        N_ATTN_LAYERS=2
        HARD_ROLE_DEEP_INPUT_PROJECTIONS=1
        REF_AUTOCALIBRATE=1
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    ppot1logwpreftoken|ppo_t1_logw_pref_token|logw_terminal)
        METHOD_TAG="PPO_RAM_PrefToken_AttnL2_T1_LogWTerminal"
        RUN_T_ROLE=1
        RAM_REWARD_MODE="logw_terminal"
        PPO_CRITIC_MODE="scalar"
        PPO_CRITIC_POPART=0
        PPO_ADVANTAGE_SCALARIZATION="ws"
        HARD_ROLE_PREFERENCE_CONDITIONING="pref_token"
        N_ATTN_LAYERS=2
        HARD_ROLE_DEEP_INPUT_PROJECTIONS=1
        REF_AUTOCALIBRATE=0
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    ppot1logwcontribution|ppo_t1_logw_contribution|logw_terminal_contribution)
        METHOD_TAG="PPO_RAM_PrefToken_AttnL2_T1_LogWTerminal_DenseContribution_Rho02"
        RUN_T_ROLE=1
        RAM_REWARD_MODE="logw_terminal"
        DENSE_REWARD_MODE="contribution"
        DENSE_SCALARIZATION="sum"
        DENSE_TERMINAL_RATIO=0.2
        PPO_CRITIC_MODE="scalar"
        PPO_CRITIC_POPART=0
        PPO_ADVANTAGE_SCALARIZATION="ws"
        HARD_ROLE_PREFERENCE_CONDITIONING="pref_token"
        N_ATTN_LAYERS=2
        HARD_ROLE_DEEP_INPUT_PROJECTIONS=1
        REF_AUTOCALIBRATE=0
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    ppot1logwdelta|ppo_t1_logw_delta|logw_terminal_delta)
        METHOD_TAG="PPO_RAM_PrefToken_AttnL2_T1_LogWTerminal_DenseDelta_Rho02"
        RUN_T_ROLE=1
        RAM_REWARD_MODE="logw_terminal"
        DENSE_REWARD_MODE="delta"
        DENSE_SCALARIZATION="sum"
        DENSE_TERMINAL_RATIO=0.2
        PPO_CRITIC_MODE="scalar"
        PPO_CRITIC_POPART=0
        PPO_ADVANTAGE_SCALARIZATION="ws"
        HARD_ROLE_PREFERENCE_CONDITIONING="pref_token"
        N_ATTN_LAYERS=2
        HARD_ROLE_DEEP_INPUT_PROJECTIONS=1
        REF_AUTOCALIBRATE=0
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    ppot1logwpreftokenentropy001|ppo_t1_logw_pref_token_entropy001|logw_terminal_entropy001)
        METHOD_TAG="PPO_RAM_PrefToken_AttnL2_T1_LogWTerminal_Entropy001"
        RUN_T_ROLE=1
        RAM_REWARD_MODE="logw_terminal"
        ENTROPY_COEF=0.01
        PPO_CRITIC_MODE="scalar"
        PPO_CRITIC_POPART=0
        PPO_ADVANTAGE_SCALARIZATION="ws"
        HARD_ROLE_PREFERENCE_CONDITIONING="pref_token"
        N_ATTN_LAYERS=2
        HARD_ROLE_DEEP_INPUT_PROJECTIONS=1
        REF_AUTOCALIBRATE=0
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    ppot1vectorwpop|ppo_t1_vector_wpop|ppot1vectorcriticadvwpop|ppo_t1_vector_critic_adv_wpop)
        METHOD_TAG="PPO_RAM_FiLM_Attn_T1_VectorCritic_PopArt_AdvWPOP"
        RUN_T_ROLE=1
        PPO_CRITIC_MODE="vector"
        PPO_CRITIC_POPART=1
        PPO_ADVANTAGE_SCALARIZATION="wpop"
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    ppot1vectorcriticdeltaadvwpop|ppo_t1_vector_critic_delta_adv_wpop)
        METHOD_TAG="PPO_RAM_FiLM_Attn_T1_VectorCritic_PopArt_DeltaMetrics_AdvWPOP"
        RUN_T_ROLE=1
        RAM_REWARD_MODE="delta_metrics"
        PPO_CRITIC_MODE="vector"
        PPO_CRITIC_POPART=1
        PPO_ADVANTAGE_SCALARIZATION="wpop"
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    ppot1d3po|ppo_t1_d3po|d3po|ppot1d3podelta|ppo_t1_d3po_delta|d3podelta|d3po_delta)
        METHOD_TAG="PPO_RAM_FiLM_Attn_T1_D3PO_VectorCritic_PopArt_DeltaMetrics_Fair2048"
        RUN_T_ROLE=1
        RAM_REWARD_MODE="delta_metrics"
        PPO_CRITIC_MODE="vector"
        PPO_CRITIC_POPART=1
        PPO_ADVANTAGE_SCALARIZATION="d3po"
        HARD_ROLE_PREFERENCE_CONDITIONING="film"
        HARD_ROLE_DEEP_INPUT_PROJECTIONS=0
        PPO_MINIBATCH_SIZE=256
        PPO_ROLLOUT_MACRO_STEPS=2048
        DETERMINISTIC=1
        EXTRA_ARGS=(--d3po-diversity-coef "${D3PO_DIVERSITY_COEF}" --d3po-diversity-alpha "${D3PO_DIVERSITY_ALPHA}")
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    ppot1d3pocomponents|ppo_t1_d3po_components|d3pocomponents|d3po_components)
        METHOD_TAG="PPO_RAM_FiLM_Attn_T1_D3PO_VectorCritic_PopArt_ComponentRewards_Fair2048"
        RUN_T_ROLE=1
        RAM_REWARD_MODE="component_rewards"
        PPO_CRITIC_MODE="vector"
        PPO_CRITIC_POPART=1
        PPO_ADVANTAGE_SCALARIZATION="d3po"
        HARD_ROLE_PREFERENCE_CONDITIONING="film"
        HARD_ROLE_DEEP_INPUT_PROJECTIONS=0
        PPO_MINIBATCH_SIZE=256
        PPO_ROLLOUT_MACRO_STEPS=2048
        DETERMINISTIC=1
        EXTRA_ARGS=(--d3po-diversity-coef "${D3PO_DIVERSITY_COEF}" --d3po-diversity-alpha "${D3PO_DIVERSITY_ALPHA}")
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    ppot1vectorwpoptoken|ppo_t1_vector_wpop_token)
        METHOD_TAG="PPO_RAM_PrefToken_AttnL2_T1_VectorCritic_AdvWPOP"
        RUN_T_ROLE=1
        PPO_CRITIC_MODE="vector"
        PPO_CRITIC_POPART=0
        PPO_ADVANTAGE_SCALARIZATION="wpop"
        HARD_ROLE_PREFERENCE_CONDITIONING="pref_token"
        N_ATTN_LAYERS=2
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    hardqt1|hardq_t1)
        METHOD_TAG="HardRoleQ_RAM_FiLM_Attn_T1"
        RUN_T_ROLE=1
        MODE_ARGS=(--ram-mode hard_role_q --role-q-mixer none)
        ;;
    ppot20|ppo_t20)
        METHOD_TAG="PPO_RAM_FiLM_Attn_T20"
        RUN_T_ROLE=20
        MODE_ARGS=(--ram-mode ppo_ram)
        ;;
    hardqt20|hardq_t20)
        METHOD_TAG="HardRoleQ_RAM_FiLM_Attn_T20"
        RUN_T_ROLE=20
        MODE_ARGS=(--ram-mode hard_role_q --role-q-mixer none)
        ;;

    # ---------------- PPO-focused ablations ----------------------------------
    pponoentropy|ppo_no_entropy)
        METHOD_TAG="PPO_RAM_NoEntropy"
        MODE_ARGS=(--ram-mode ppo_ram)
        EXTRA_ARGS=(--entropy-coef 0.0)
        ;;
    ppoentropyhigh|ppo_entropy_high)
        METHOD_TAG="PPO_RAM_Entropy003"
        MODE_ARGS=(--ram-mode ppo_ram)
        EXTRA_ARGS=(--entropy-coef 0.03)
        ;;
    ppoclipsmall|ppo_clip_small)
        METHOD_TAG="PPO_RAM_Clip010"
        MODE_ARGS=(--ram-mode ppo_ram)
        EXTRA_ARGS=(--ppo-clip-eps 0.10)
        ;;
    ppotwolayers|ppo_two_layers)
        METHOD_TAG="PPO_RAM_AttnL2"
        MODE_ARGS=(--ram-mode ppo_ram)
        N_ATTN_LAYERS=2
        ;;
    ppopreftoken|ppo_pref_token)
        METHOD_TAG="PPO_RAM_PrefToken_AttnL2"
        MODE_ARGS=(--ram-mode ppo_ram)
        HARD_ROLE_PREFERENCE_CONDITIONING="pref_token"
        N_ATTN_LAYERS=2
        ;;
    ppoprefbias|ppo_pref_bias)
        METHOD_TAG="PPO_RAM_PrefBias"
        MODE_ARGS=(--ram-mode ppo_ram)
        EXTRA_ARGS=(--preference-role-bias)
        ;;
    ppoewc|ppo_ewc)
        METHOD_TAG="PPO_RAM_EWC"
        MODE_ARGS=(--ram-mode ppo_ram)
        ROLE_SCALARIZATION="ewc"
        ;;

    # ---------------- HardRoleQ-focused ablations ----------------------------
    hardqperuniform|hardq_per_uniform)
        METHOD_TAG="HardRoleQ_RAM_PERalpha0"
        MODE_ARGS=(--ram-mode hard_role_q --role-q-mixer none)
        EXTRA_ARGS=(--per-alpha 0.0)
        ;;
    hardqperstrong|hardq_per_strong)
        METHOD_TAG="HardRoleQ_RAM_PERalpha08"
        MODE_ARGS=(--ram-mode hard_role_q --role-q-mixer none)
        EXTRA_ARGS=(--per-alpha 0.8)
        ;;
    hardqtargetslow|hardq_target_slow)
        METHOD_TAG="HardRoleQ_RAM_Target200"
        MODE_ARGS=(--ram-mode hard_role_q --role-q-mixer none)
        EXTRA_ARGS=(--role-q-target-update 200)
        ;;
    hardqtwolayers|hardq_two_layers)
        METHOD_TAG="HardRoleQ_RAM_AttnL2"
        MODE_ARGS=(--ram-mode hard_role_q --role-q-mixer none)
        N_ATTN_LAYERS=2
        ;;
    hardqprefbias|hardq_pref_bias)
        METHOD_TAG="HardRoleQ_RAM_PrefBias"
        MODE_ARGS=(--ram-mode hard_role_q --role-q-mixer none)
        EXTRA_ARGS=(--preference-role-bias)
        ;;
    hardqewc|hardq_ewc)
        METHOD_TAG="HardRoleQ_RAM_EWC"
        MODE_ARGS=(--ram-mode hard_role_q --role-q-mixer none)
        ROLE_SCALARIZATION="ewc"
        ;;
    *)
        echo "[error] Unknown experiment: ${EXP_RAW}"
        echo "Run without arguments for the exhaustive experiment list."
        exit 2
        ;;
esac

if [[ "${HARD_ROLE_DEEP_INPUT_PROJECTIONS}" == "1" ]]; then
    METHOD_TAG="${METHOD_TAG}_DeepInputProj"
fi
if [[ "${WORLD_AGENT_SPLIT}" == "1" ]]; then
    METHOD_TAG="${METHOD_TAG}_WorldAgentSplit"
fi
if [[ "${PPO_CRITIC_MODE}" == "vector" ]]; then
    METHOD_TAG="${METHOD_TAG}_VectorCriticFixed_Adv${PPO_ADVANTAGE_SCALARIZATION^^}"
fi

RUN_NAME_BASE="${MAP_NAME}_F4_${METHOD_TAG}_T${RUN_T_ROLE}_ep${EPISODES}_beta${WEIGHT_ALPHA}_s${SEED}"
LOG_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}/_tmux_logs"

# Resolve the run suffix before naming any artifact. Python also protects its
# output directory against collisions, but doing it only there leaves tmux,
# logs, and command files using the unsuffixed name. tmux maps dots in session
# names to underscores, so use that normalized form consistently when checking.
RUN_SUFFIX=0
while true; do
    if (( RUN_SUFFIX == 0 )); then
        RUN_NAME="${RUN_NAME_BASE}"
    else
        RUN_NAME="${RUN_NAME_BASE}_${RUN_SUFFIX}"
    fi
    SESSION_NAME="${RUN_NAME//./_}"
    RUN_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}/${RUN_NAME}"
    LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
    CMD_FILE="${LOG_DIR}/${RUN_NAME}.cmd.sh"

    SESSION_EXISTS=0
    if command -v tmux >/dev/null 2>&1 && tmux has-session -t "=${SESSION_NAME}" 2>/dev/null; then
        SESSION_EXISTS=1
    fi
    if [[ ! -e "${RUN_DIR}" && ! -e "${LOG_FILE}" && ! -e "${CMD_FILE}" && "${SESSION_EXISTS}" == "0" ]]; then
        break
    fi
    RUN_SUFFIX=$((RUN_SUFFIX + 1))
done

# ---- Preflight ---------------------------------------------------------------

if [[ ! "${DEVICE}" =~ ^-1$|^[0-9]+$ ]]; then
    echo "[error] device must be -1 or a non-negative CUDA index; got '${DEVICE}'"
    exit 2
fi
if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "[error] run_experiment.py not found: ${RUN_SCRIPT}"
    exit 1
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "[error] No executable Python found. Activate your environment first."
    exit 1
fi
if [[ "${DRY_RUN}" != "1" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "[error] tmux is not installed or not in PATH."
        exit 1
    fi
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "[error] tmux session already exists: ${SESSION_NAME}"
        echo "Attach with: tmux attach -t '${SESSION_NAME}'"
        exit 1
    fi
fi
mkdir -p "${LOG_DIR}"

# ---- Build exact command -----------------------------------------------------

COMMON_ARGS=(
    --env project
    --project-control expert_nu
    --path-planner-folder "${PATH_PLANNER_FOLDER}"
    --map-name "${MAP_NAME}"
    --N "${N_AGENTS}"
    --episodes "${EPISODES}"
    --T-role "${RUN_T_ROLE}"
    --gamma 0.99
    --role-reward-norm minmax
    --role-scalarization "${ROLE_SCALARIZATION}"
    --ram-reward-mode "${RAM_REWARD_MODE}"
    --chebyshev-ref-clean "${CHEBYSHEV_REF_CLEAN}"
    --chebyshev-ref-cov "${CHEBYSHEV_REF_COV}"
    --chebyshev-rho "${CHEBYSHEV_RHO}"
    --stch-mu "${STCH_MU}"
    --wpop-eps "${WPOP_EPS}"
    --pref-weight-clamp "${PREF_WEIGHT_CLAMP}"
    --dense-reward-mode "${DENSE_REWARD_MODE}"
    --dense-scalarization "${DENSE_SCALARIZATION}"
    --dense-terminal-ratio "${DENSE_TERMINAL_RATIO}"
    --dense-warmup-episodes "${DENSE_WARMUP_EPISODES}"
    --d-model "${D_MODEL}"
    --n-attn-heads "${N_ATTN_HEADS}"
    --n-attn-layers "${N_ATTN_LAYERS}"
    --attn-ff-dim "${ATTN_FF_DIM}"
    --hard-role-preference-conditioning "${HARD_ROLE_PREFERENCE_CONDITIONING}"
    --ppo-epochs 4
    --ppo-minibatch-size "${PPO_MINIBATCH_SIZE}"
    --ppo-rollout-macro-steps "${PPO_ROLLOUT_MACRO_STEPS}"
    --ppo-clip-eps 0.1
    --gae-lambda 0.95
    --entropy-coef "${ENTROPY_COEF}"
    --value-coef 0.5
    --actor-lr 0.0001
    --critic-lr 0.0001
    --ppo-critic-mode "${PPO_CRITIC_MODE}"
    --ppo-advantage-scalarization "${PPO_ADVANTAGE_SCALARIZATION}"
    --ppo-target-kl 0.02
    --ppo-max-grad-norm 0.5
    --ppo-preference-sampling random
    --role-q-lr 0.0003
    --role-q-target-update 50
    --role-q-epsilon-start 1.0
    --role-q-epsilon-end 0.05
    --role-q-epsilon-fraction 0.75
    --per-alpha 0.6
    --per-beta-start 0.4
    --per-beta-end 1.0
    --per-eps 0.000001
    --weight-sampling beta
    --weight-alpha "${WEIGHT_ALPHA}"
    --eval-every "${EVAL_EVERY}"
    --eval-episodes "${EVAL_EPISODES}"
    --eval-points "${EVAL_POINTS}"
    --save-every "${SAVE_EVERY}"
    --probe-preference-sensitivity
    --probe-points "${PROBE_POINTS}"
    --probe-episodes "${PROBE_EPISODES}"
    --same-state-preference-check
    --freeze
    --seed "${SEED}"
    --device "${DEVICE}"
    --run-name "${RUN_NAME}"
    --output-dir "${OUTPUT_DIR}"
)

if [[ "${PPO_CRITIC_POPART}" == "1" ]]; then
    COMMON_ARGS+=(--ppo-critic-popart)
fi
if [[ "${REF_AUTOCALIBRATE}" == "1" ]]; then
    COMMON_ARGS+=(--ref-autocalibrate)
fi
if [[ "${HARD_ROLE_DEEP_INPUT_PROJECTIONS}" == "1" ]]; then
    COMMON_ARGS+=(--hard-role-deep-input-projections)
fi
if [[ "${WORLD_AGENT_SPLIT}" == "1" ]]; then
    COMMON_ARGS+=(--world-agent-split)
fi
if [[ "${DETERMINISTIC}" == "0" ]]; then
    COMMON_ARGS+=(--no-deterministic)
fi

{
    echo "#!/usr/bin/env bash"
    printf "cd %q\n" "${PROJECT_ROOT}"
    printf "export CUBLAS_WORKSPACE_CONFIG=%q\n" "${CUBLAS_WORKSPACE_CONFIG}"
    printf "export PYTHONHASHSEED=%q\n" "${PYTHONHASHSEED}"
    printf "%q " "${PYTHON_BIN}" "${RUN_SCRIPT}" "${COMMON_ARGS[@]}" "${MODE_ARGS[@]}" "${EXTRA_ARGS[@]}"
    echo
} > "${CMD_FILE}"
chmod +x "${CMD_FILE}"

# ---- Launch and persist logs -------------------------------------------------

echo "[launch] experiment : ${EXP_RAW}"
echo "[launch] method     : ${METHOD_TAG}"
echo "[launch] run_name   : ${RUN_NAME}"
echo "[launch] device     : ${DEVICE}"
echo "[launch] map        : ${MAP_NAME}"
echo "[launch] T_role     : ${RUN_T_ROLE}"
echo "[launch] critic     : ${PPO_CRITIC_MODE} (popart=${PPO_CRITIC_POPART})"
echo "[launch] advantage  : ${PPO_ADVANTAGE_SCALARIZATION}"
echo "[launch] deep input : ${HARD_ROLE_DEEP_INPUT_PROJECTIONS}"
echo "[launch] deterministic: ${DETERMINISTIC} (CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG})"
echo "[launch] episodes   : ${EPISODES}"
echo "[launch] tmux       : ${SESSION_NAME}"
echo "[launch] log        : ${LOG_FILE}"
echo "[launch] command    : ${CMD_FILE}"
echo

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] No tmux session was started. Exact command:"
    sed -n '1,5p' "${CMD_FILE}"
    rm -f "${CMD_FILE}"
    exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" bash -lc "
    set -uo pipefail
    cd '${PROJECT_ROOT}'
    echo '============================================================'
    echo '[run] ${RUN_NAME}'
    echo '[start]' \$(date)
    echo '[host]' \$(hostname)
    echo '[python] ${PYTHON_BIN}'
    echo '[device] ${DEVICE}'
    echo '[map] ${MAP_NAME}'
    echo '[T_role] ${RUN_T_ROLE}'
    echo '============================================================'
    set +e
    '${CMD_FILE}' 2>&1 | tee '${LOG_FILE}'
    TRAIN_EXIT_CODE=\${PIPESTATUS[0]}
    set -e
    echo
    echo '============================================================'
    echo '[done] ${RUN_NAME} exit='\${TRAIN_EXIT_CODE}
    echo '[finish]' \$(date)
    echo '============================================================'
    exit \${TRAIN_EXIT_CODE}
"

echo "Started successfully."
echo "Attach: tmux attach -t '${SESSION_NAME}'"
echo "Follow: tail -f '${LOG_FILE}'"
