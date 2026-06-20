# Policy Evaluation

This repo contains scripts to run policy evaluations on a number of checkpoints for a number of execution horizons (assuming an action-chunking policy).

The following installation instructions pertain to training a policy using [diffusion-policy-experiments](https://github.com/michaelszeng/diffusion-policy), but similar instructions may apply when using a different training pipeline.


## Installation

### 1. Install the `diffusion-policy` package and its pinned dependencies
These instructions are only necessary for my training/eval pipeline. The eval pipeline (`eval/evaluate_kitchen.py`) loads checkpoints trained with my [diffusion-policy-experiments](https://github.com/michaelszeng/diffusion-policy) repo, so we install both its pinned dependencies and the editable package itself. **Python must be 3.9**.
```bash
git clone https://github.com/michaelszeng/diffusion-policy ~/diffusion-policy

# install pinned deps + editable package
pip install -r ~/diffusion-policy/requirements.txt
pip install -e ~/diffusion-policy
```

### 2. Install the kitchen environment extras
`mujoco` must be pinned to **2.3.5** (not the latest patch release): `dm_control==1.0.12`'s onscreen viewer indexes into `mjVISSTRING`, whose layout changed in mujoco 2.3.6+ — using 2.3.7 makes the viewer import crash with `ord() expected a character, but string of length 0 found`. Offscreen rendering still works at 2.3.7, but the live viewer used by `evaluate_kitchen.py --no-headless` does not.
```bash
pip install \
  setuptools \
  mujoco-py \
  "gym==0.26.2" \
  "mujoco==2.3.5" \
  "dm_control==1.0.12" \
  click \
  termcolor \
  statsmodels \
  git+https://github.com/aravindr93/mjrl.git
```

### 3. Add `adept_envs` to your PYTHONPATH
There is no `setup.py`, so add the package directory directly (also add to `~/.zshrc` or `~/.bashrc`):
```bash
export PYTHONPATH=$PYTHONPATH:~/relay-policy-learning/adept_envs
```



## Installation on the MIT CSAIL cluster

These instructions pertain to using SLURM clusters like the MIT CSAIL cluster for evaluation.


### 1. Clone the repo and create a Python 3.9 conda env
```bash
git clone https://github.com/google-research/relay-policy-learning ~/relay-policy-learning
cd ~/relay-policy-learning

source /data/locomotion/michzeng/miniconda3/etc/profile.d/conda.sh
conda create -n relay python=3.9 -y
conda activate relay
```

### 2. Install system libs via conda (replaces `apt-get` from step 4)
```bash
conda install -n base -c conda-forge conda-libmamba-solver -y
conda config --set solver libmamba
conda install -c conda-forge -y mesalib glew glfw patchelf
```

### 3. Install MuJoCo 2.1.0 binary for mujoco-py
```bash
mkdir -p ~/.mujoco
wget -qO /tmp/mujoco.tar.gz https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf /tmp/mujoco.tar.gz -C ~/.mujoco
echo 'export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 4. Install `diffusion-policy` + kitchen extras (same pins as steps 5-6)
Two things must be pinned *before* the requirements install (see gotchas): pip (the requirements pull a pip that's too new for Python 3.9) and `mujoco` (`dm-control==1.0.9` requires `mujoco>=2.3.1.post1` unpinned, so pip grabs 3.5.0, which has no 3.9 wheel and fails to build from source). Installing `mujoco==2.3.5` first satisfies the constraint with a real wheel:
```bash
pip install "pip<24.1"
pip install "mujoco==2.3.5"
git clone https://github.com/michaelszeng/diffusion-policy
pip install -r /data/locomotion/michzeng/diffusion-policy-experiments/requirements.txt
pip install -e /data/locomotion/michzeng/diffusion-policy-experiments
pip install setuptools mujoco-py "gym==0.26.2" "dm_control==1.0.12" click termcolor statsmodels git+https://github.com/aravindr93/mjrl.git
```

### 5. PYTHONPATH
```bash
echo 'export PYTHONPATH=$HOME/relay-policy-learning/adept_envs:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

### 6. Submit eval jobs via SLURM
The sbatch script (`eval/submit_evaluate_checkpoints.sbatch`) activates the `relay` conda env, sets `LD_LIBRARY_PATH` for `mujoco210`, and calls `eval/evaluate_checkpoints.sh`. Edit the `readonly CHECKPOINT_PATH=...` line at the top before first submission.
```bash
cd ~/relay-policy-learning
mkdir -p logs
sbatch eval/submit_evaluate_checkpoints.sbatch 8          # single horizon
bash   eval/batch_submit.sh                               # sweep horizons 1..15
```
Override the env name with `CONDA_ENV=foo sbatch ...` if you named the conda env something other than `relay`.



## Running Evaluation

A trial counts as a `success` if at least `--n-required-subtasks` distinct kitchen subtasks (out of the 7: `bottomknob`, `topknob`, `light`, `slide`, `hinge`, `microwave`, `kettle`) are achieved at any point before the timeout — order doesn't matter, and "achieved" means within D4RL's 0.3 distance threshold to that subtask's goal joint configuration. The default matches D4RL's `kitchen-complete-v0`-style rubric: any 4 of the 7 subtasks complete.
```bash
python eval/evaluate_kitchen.py \
  --checkpoint /path/to/run/checkpoints/epoch=050-val_loss=0.1234.ckpt \
  --n-rollouts 10
```
Use `--n-required-subtasks N` to change the success threshold (e.g. `--n-required-subtasks 3` for a lower bar).

Outputs land in `outputs/<date>/<time>/` (or `--output-dir <path>`) with the same structure as `evaluate_model_custom.py`: `results.csv` (per-trial: trial, result, reward, trial_time), `results.pkl`, `summary.txt`, and `videos/trial_NNNN_<result>.mp4`. Use `--resume` together with `--output-dir` to pick up a partial run.
