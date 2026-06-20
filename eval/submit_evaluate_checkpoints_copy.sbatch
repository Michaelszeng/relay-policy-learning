#!/bin/bash
#SBATCH --job-name=evaluate_kitchen
#SBATCH --account=locomotion
#SBATCH --partition=vision-shared-rtx3090
#SBATCH --exclude=freeman-titanrtx-2,freeman-titanrtx-1
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --requeue

# Usage:
#   sbatch eval/submit_evaluate_checkpoints.sbatch <action_horizon>
#
# SLURM runner for `eval/evaluate_checkpoints.sh` (Franka Kitchen eval).

# ============================================================================
# Checkpoint to evaluate — edit this constant.
# Path to a .ckpt file or a directory of .ckpt files.
# ============================================================================
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/franka_kitchen/2_obs_human_expert/checkpoints"
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/franka_kitchen/2_obs_human_expert_attention_double_enc/checkpoints"
CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/franka_kitchen/4_obs_human_expert_attention_double_enc/checkpoints"
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/franka_kitchen/8_obs_human_expert_attention_double_enc/checkpoints"
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/franka_kitchen/12_obs_human_expert_attention_double_enc/checkpoints"
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/franka_kitchen/16_obs_human_expert_attention_double_enc/checkpoints"
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/franka_kitchen/20_obs_human_expert_attention_double_enc/checkpoints"

ACTION_HORIZON="${1:?Usage: sbatch eval/submit_evaluate_checkpoints.sbatch <action_horizon>}"

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
echo "DATE: $DATE"
echo "TIME: $TIME"
echo "ACTION_HORIZON: $ACTION_HORIZON"

# ============================================================================
# Evaluation parameters
# ============================================================================
N_ENVS=8
N_ROLLOUTS=100
N_VIDEO_TRIALS=20
N_REQUIRED_SUBTASKS=4
TASK_TIMEOUT=""  # leave empty to use DEFAULT_TASK_TIMEOUT (280) from evaluate_kitchen.py
# ============================================================================

# Extract the experiment name as the directory immediately before "checkpoints".
EXPERIMENT_NAME="$(basename "$(dirname "${CHECKPOINT_PATH}")")"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "SLURM Job ID:    $SLURM_JOB_ID"
echo "SLURM Node:      $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

# --- Environment setup ---
echo "Changing directory to $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to change directory to $SLURM_SUBMIT_DIR"; exit 1; }

# Conda env that has the diffusion-policy package + relay-policy-learning extras
# installed (per README steps 5-6). Must be Python 3.9.
source /data/locomotion/michzeng/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-relay}"
conda activate "${CONDA_ENV}"
# Conda activation can clear CUDA_VISIBLE_DEVICES on some setups — restore it
# from SLURM's GPU assignment if it was unset.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-0}}}"
echo "CUDA_VISIBLE_DEVICES after conda activate: ${CUDA_VISIBLE_DEVICES}"

mkdir -p logs

# mujoco-py needs the mujoco210 binary on LD_LIBRARY_PATH (matches README step 3).
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

# Compute nodes are headless (no X11 DISPLAY), so dm_control's default GLFW
# backend can't create a GL context ("gladLoadGL error"). Force EGL, which does
# offscreen rendering directly on the GPU. Inherited by the spawned env workers.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# evaluate_kitchen.py already bootstraps adept_envs onto sys.path, but adding
# it to PYTHONPATH is harmless and keeps standalone usage consistent.
export PYTHONPATH="${PWD}/adept_envs:${PYTHONPATH:-}"

echo "CHECKPOINT_PATH:     $CHECKPOINT_PATH"
echo "N_ENVS:              $N_ENVS"
echo "N_ROLLOUTS:          $N_ROLLOUTS"
echo "N_VIDEO_TRIALS:      $N_VIDEO_TRIALS"
echo "N_REQUIRED_SUBTASKS: $N_REQUIRED_SUBTASKS"
echo "ACTION_HORIZON:      $ACTION_HORIZON"
echo "TASK_TIMEOUT:        ${TASK_TIMEOUT:-(default)}"
echo "=========================================="

TASK_TIMEOUT_ARG=()
[[ -n "${TASK_TIMEOUT}" ]] && TASK_TIMEOUT_ARG=(--task-timeout "${TASK_TIMEOUT}")

bash eval/evaluate_checkpoints.sh \
    "${CHECKPOINT_PATH}" \
    "${ACTION_HORIZON}" \
    "${N_ENVS}" \
    "${N_VIDEO_TRIALS}" \
    "${N_ROLLOUTS}" \
    "${N_REQUIRED_SUBTASKS}" \
    --resume "${TASK_TIMEOUT_ARG[@]}"
