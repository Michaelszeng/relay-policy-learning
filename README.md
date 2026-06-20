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

### 5. Python dependencies

TODO

### (Optional) Policy evaluation installation

To use my training and policy eval pipeline, additional installs are required. See `eval/README.md`.



## Getting Started (User)

Use the environments in your code (After including in the PYTHONPATH)
```
#!/usr/bin/env python3

import adept_envs
import gym

env = gym.make('kitchen_relax-v1')
```

### Download pre-collected Human Teleoperation Dataset


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



### Generating Data Using Markovian Expert

An automated data generation script runs a markovian scripted expert across random 4-subtask sequences and collects successful episodes. This scripted expert is roughly 94% successful, which should be sufficient for data generation.

Given the multi-modal, multi-task nature of Franka Kitchen, developing a true "Markovian" (in the sense that the expert's action depends only on the environment state) expert is very challenging. To make such a Markovian expert possible, we record one-hot encodings containing the 4-subtask sequence (which is constant throughout a recorded episode) as well the current active subtask (which will change 4 times throughout the episode) as observation keys in the .zarr dataset. Thus, the scripted expert *is* Markovian w.r.t. the combined environment state + subtask labels.

```bash
# Generate random 4-subtask sequences
python experts/record_demos.py --randomize --chain-len 4 \
      --n-episodes 500 --seed 0 \
      --out kitchen_demos_markovian_scripted_expert.zarr
```


### Evaluating Trained Policies

See `eval/README.md`.