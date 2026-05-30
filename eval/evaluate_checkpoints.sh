#!/usr/bin/env bash
# Runs N rollouts at a single action horizon, for every checkpoint found under
# the given path (file or directory of .ckpt files).
# Results are written to: outputs/<experiment_name>/T_a_<x>/<checkpoint_stem>/
# The experiment name is derived from the directory before "checkpoints/".
#
# Example usage (single checkpoint):
#   ./eval/evaluate_checkpoints.sh /path/to/ckpts/epoch=095.ckpt 8 4 20 100 4
#
# Example usage (directory of checkpoints):
#   ./eval/evaluate_checkpoints.sh /path/to/checkpoints/ 8 4 20 100 4
#
# Debug mode (1 rollout, 1 env, no video, not headless):
#   ./eval/evaluate_checkpoints.sh <ckpt_or_dir> 8 --debug
#
# Resume a previous run (skips completed checkpoints):
#   ./eval/evaluate_checkpoints.sh <ckpt_or_dir> 8 4 20 100 4 --resume
set -euo pipefail
ulimit -s unlimited
ulimit -c unlimited

CHECKPOINT_PATH="${1:?Usage: $0 <ckpt_or_dir> <n_action_steps> [n_envs] [n_video_trials] [n_rollouts] [n_required_subtasks] [--debug] [--resume]}"
N_ACTION_STEPS="${2:?Usage: $0 <ckpt_or_dir> <n_action_steps> [n_envs] [n_video_trials] [n_rollouts] [n_required_subtasks] [--debug] [--resume]}"
N_ENVS="${3:-1}"
N_VIDEO_TRIALS="${4:-0}"
N_ROLLOUTS="${5:-100}"
N_REQUIRED_SUBTASKS="${6:-4}"
DEBUG=false
RESUME=false
TASK_TIMEOUT=""
_ARGS=("$@")
for (( i=0; i<${#_ARGS[@]}; i++ )); do
    [[ "${_ARGS[$i]}" == "--debug" ]] && DEBUG=true
    [[ "${_ARGS[$i]}" == "--resume" ]] && RESUME=true
    if [[ "${_ARGS[$i]}" == "--task-timeout" ]]; then
        TASK_TIMEOUT="${_ARGS[$((i+1))]}"
    fi
done

# Derive experiment name from the directory before "checkpoints/".
if [[ -f "${CHECKPOINT_PATH}" ]]; then
    EXPERIMENT_NAME="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
else
    EXPERIMENT_NAME="$(basename "$(dirname "${CHECKPOINT_PATH}")")"
fi

if [[ "${DEBUG}" == "true" ]]; then
    echo "DEBUG mode enabled: n_envs=1, n_rollouts=1, n_video_trials=0, headless=false"
    N_ENVS=1 N_ROLLOUTS=1 N_VIDEO_TRIALS=0
fi

HEADLESS_FLAG="--headless"
[[ "${DEBUG}" == "true" ]] && HEADLESS_FLAG=""
RESUME_FLAG=""
[[ "${RESUME}" == "true" ]] && RESUME_FLAG="--resume"
TASK_TIMEOUT_FLAG=""
[[ -n "${TASK_TIMEOUT}" ]] && TASK_TIMEOUT_FLAG="--task-timeout ${TASK_TIMEOUT}"

# Collect checkpoints: single file or all .ckpt files in a directory
# (excluding 'latest.ckpt' since it's typically a duplicate of the most recent topk).
if [[ -f "${CHECKPOINT_PATH}" ]]; then
    CHECKPOINTS=("${CHECKPOINT_PATH}")
elif [[ -d "${CHECKPOINT_PATH}" ]]; then
    mapfile -t CHECKPOINTS < <(find "${CHECKPOINT_PATH}" -maxdepth 1 -name "*.ckpt" ! -name "latest.ckpt" | sort)
    if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
        echo "No .ckpt files found in ${CHECKPOINT_PATH}" >&2
        exit 1
    fi
else
    echo "Error: ${CHECKPOINT_PATH} is neither a file nor a directory." >&2
    exit 1
fi

BASE_OUT="outputs/${EXPERIMENT_NAME}"
echo "Base output directory: ${BASE_OUT}"
echo "Checkpoints found:     ${#CHECKPOINTS[@]}"

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CKPT_STEM="$(basename "${CHECKPOINT}" .ckpt)"
    OUT_DIR="${BASE_OUT}/T_a_${N_ACTION_STEPS}/${CKPT_STEM}"
    echo ""
    echo "##########################################"
    echo "Checkpoint:      ${CKPT_STEM}"
    echo "Action horizon:  ${N_ACTION_STEPS}  ->  ${OUT_DIR}"
    echo "##########################################"
    if ! python -u eval/evaluate_kitchen.py \
        --checkpoint "${CHECKPOINT}" \
        --n-rollouts "${N_ROLLOUTS}" \
        --n-envs "${N_ENVS}" \
        --n-action-steps "${N_ACTION_STEPS}" \
        --n-video-trials "${N_VIDEO_TRIALS}" \
        --n-required-subtasks "${N_REQUIRED_SUBTASKS}" \
        --output-dir "${OUT_DIR}" \
        ${RESUME_FLAG} \
        ${HEADLESS_FLAG} \
        ${TASK_TIMEOUT_FLAG}; then
        echo "ERROR: evaluate_kitchen.py failed for T_a=${N_ACTION_STEPS}, checkpoint=${CKPT_STEM}" >&2
        continue
    fi
done
