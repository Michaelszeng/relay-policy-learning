#!/usr/bin/env bash
# Submit one SLURM job per action horizon. Edit `submit_evaluate_checkpoints.sbatch`
# to set CHECKPOINT_PATH / N_ENVS / N_ROLLOUTS / N_REQUIRED_SUBTASKS before running this.
for T_a in 1 2 3 4 5 6 8 10 12 15; do
    sbatch eval/submit_evaluate_checkpoints_copy.sbatch "${T_a}"
done
