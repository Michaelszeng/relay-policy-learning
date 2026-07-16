## Installation on a SLURM cluster

These instructions pertain to using SLURM clusters like the MIT CSAIL cluster.

### 1. Clone the repo and create a Python 3.9 conda env
```bash
git clone https://github.com/google-research/relay-policy-learning ~/relay-policy-learning
cd ~/relay-policy-learning

source /data/locomotion/michzeng/miniconda3/etc/profile.d/conda.sh
conda create -n relay python=3.9 -y
conda activate relay
```

### 2. Install system libs via conda
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


**For running evaluations using my [diffusion-policy-experiments](https://github.com/michaelszeng/diffusion-policy) repo only (see `eval/README.md` for more details):**

### 4. Install `diffusion-policy-experiments` + kitchen extras (same pins as steps 5-6)
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

### 6. Submitting eval jobs via SLURM
The sbatch script (`eval/submit_evaluate_checkpoints.sbatch`) activates the `relay` conda env, sets `LD_LIBRARY_PATH` for `mujoco210`, and calls `eval/evaluate_checkpoints.sh` to launch evaluations. Set your `CHECKPOINT_PATH`, and SLURM cluster settings in `eval/submit_evaluate_checkpoints.sbatch` before running.

```bash
cd ~/relay-policy-learning
sbatch eval/submit_evaluate_checkpoints.sbatch 8          # single execution horizon (8 in this case)
bash   eval/batch_submit.sh                               # sweep multiple execution horizons
```