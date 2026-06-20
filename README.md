# Relay Policy Learning Environments

This is a fork of the [original Franka Kitchen MuJoCo environments repo](https://github.com/google-research/relay-policy-learning).

This contains 2 new features beyond the original repo:
1) Dataset post-processing to add image keys for a static scene camera and wrist camera.
2) A Markovian, scripted expert and data generation pipeline.

I maintained relatively clean code so this can hopefully be reused in other research projects.


## Installation

### 1. Clone the repository
```bash
git clone https://github.com/google-research/relay-policy-learning
cd relay-policy-learning
```

### 2. Create and activate a virtual environment
`mujoco_py` requires **Python 3.9 or earlier** (it is incompatible with Python 3.10+):
```bash
python3.9 -m venv env
source env/bin/activate
```

### 3. Install MuJoCo 2.1.0
`mujoco-py` requires the MuJoCo binary to be installed at `~/.mujoco/mujoco210`:
```bash
mkdir -p ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz
tar -xzf /tmp/mujoco.tar.gz -C ~/.mujoco
```

Add MuJoCo and NVIDIA libraries to your library path (also add these lines to `~/.zshrc` or `~/.bashrc`):
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
```

### 4. Install system dependencies
```bash
sudo apt-get install -y libosmesa6-dev libgl1 libglfw3 patchelf
```

### 5. Install the `diffusion-policy` package and its pinned dependencies
The eval pipeline (`eval/evaluate_kitchen.py`) loads checkpoints trained with the [diffusion-policy](https://github.com/michaelszeng/diffusion-policy) repo, so we install both its pinned dependencies and the editable package itself. **Python must be 3.9**.
```bash
# clone if you don't have it yet
git clone https://github.com/michaelszeng/diffusion-policy ~/diffusion-policy

# install pinned deps + editable package
pip install -r ~/diffusion-policy/requirements.txt
pip install -e ~/diffusion-policy
```

### 6. Install the kitchen environment extras
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

### 7. Add `adept_envs` to your PYTHONPATH
There is no `setup.py`, so add the package directory directly (also add to `~/.zshrc` or `~/.bashrc`):
```bash
export PYTHONPATH=$PYTHONPATH:~/relay-policy-learning/adept_envs
```

## Installation on the MIT CSAIL cluster

The local install above assumes you can `sudo apt-get` for system libraries (mesa, glew, glfw, patchelf). On CSAIL you can't, but `conda` supplies equivalents without root. **Vulkan is not required** — every renderer in this stack (mujoco-py, `mujoco==2.3.5`, `dm_control==1.0.12`) uses OpenGL, so a conda env with the conda-forge mesa stack is sufficient.

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


## Getting Started (User)

1. Clone the repository
```
$ git clone https://github.com/google-research/relay-policy-learning
```

2. Use the environments in your code (After including in the PYTHONPATH)
```
#!/usr/bin/env python3

import adept_envs
import gym

env = gym.make('kitchen_relax-v1')
```

## Download pre-collected Human Teleoperation Dataset


1. To use the demos, clone the puppet VR repository and add its `vive/source` directory to the PYTHONPATH:

```
$ git clone https://github.com/vikashplus/puppet
$ export PYTHONPATH=$PYTHONPATH:/PATH/TO/puppet/vive/source
```

2. Use parse_demos to parse the data into pkl format with 12.5 Hz observations/actions. Unzip the kitchen_demos_multitask.zip and then run
```bash
for dir in kitchen_demos_multitask/kitchen_demos_multitask/*/; do
  python adept_envs/adept_envs/utils/parse_demos.py --env kitchen_relax-v1 \
    --demo_dir "$dir" --view playback --skip 40 --render None
done
```

3. Convert pkl files into zarr dataset (recursively scans `--input_dir` for `*.pkl`):
```bash
python adept_envs/adept_envs/utils/process_pickles.py \
  --input_dir kitchen_demos_multitask/kitchen_demos_multitask \
  --output kitchen_demos.zarr
```

4. For debugging: visualize the zarr dataset. Add `--state` to also show a time-series panel of the non-camera arrays beneath the camera tiles.
```bash
python adept_envs/adept_envs/utils/visualize_dataset.py kitchen_demos.zarr \
  --episode 0 --fps 12.5 --state
```
Controls: `k`/`l` step 1/10 frames forward, `j`/`h` step 1/10 frames backward, `n`/`p` next/previous episode, `space` play/pause, `q` quit.



## Generating Data Using Markovian Expert

1. 


## Evaluating Trained Policies

7. Evaluate a trained diffusion policy checkpoint (trained using my [diffusion policy repo](https://github.com/Michaelszeng/diffusion-policy-experiments)). A trial counts as a `success` if at least `--n-required-subtasks` distinct kitchen subtasks (out of the 7: `bottomknob`, `topknob`, `light`, `slide`, `hinge`, `microwave`, `kettle`) are achieved at any point before the timeout — order doesn't matter, and "achieved" means within D4RL's 0.3 distance threshold to that subtask's goal joint configuration. The default matches D4RL's `kitchen-complete-v0`-style rubric: any 4 of the 7 subtasks complete.
```bash
python eval/evaluate_kitchen.py \
  --checkpoint /path/to/run/checkpoints/epoch=050-val_loss=0.1234.ckpt \
  --n-rollouts 10
```
Use `--n-required-subtasks N` to change the success threshold (e.g. `--n-required-subtasks 3` for a lower bar).

Outputs land in `outputs/<date>/<time>/` (or `--output-dir <path>`) with the same structure as `evaluate_model_custom.py`: `results.csv` (per-trial: trial, result, reward, trial_time), `results.pkl`, `summary.txt`, and `videos/trial_NNNN_<result>.mp4`. Use `--resume` together with `--output-dir` to pick up a partial run.
