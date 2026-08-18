# Relay Policy Learning Environments

This is a fork of the [original Franka Kitchen MuJoCo environments repo](https://github.com/google-research/relay-policy-learning).

This contains 2 new features beyond the original repo:
1) Dataset post-processing script to add image keys for a scene camera and wrist camera, useful for training image-based Behavior Cloning policies.
2) A Markovian, scripted expert and data generation pipeline for mass data generation and studying nature-of-expert in Behavior Cloning.

Please use in your project if helpful!


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

`mujoco-py` builds a Cython extension against NumPy on its first import, so install those two **first**:
```bash
pip install Cython==0.29.32 numpy==1.26.4   # build prerequisites for mujoco-py (Cython must be <3)
```

Then install everything else:
```bash
pip install mujoco-py==2.1.2.14 mujoco==2.3.5 gym==0.26.2 zarr==2.12.0 \
  numcodecs==0.10.2 imageio==2.22.0 imageio-ffmpeg==0.4.7 matplotlib==3.7.0 \
  opencv-python==4.11.0.86 tqdm==4.64.1 click==8.1.8 mjrl==1.0.0 sk-video==1.1.11
```


### (Optional) Policy evaluation installation

To use my training and policy eval pipeline, additional installs are required. See `eval/README.md`.



## Getting Started (User)

Use the environments in your code (After including in the PYTHONPATH)
```bash
#!/usr/bin/env python3

import adept_envs
import gym

env = gym.make('kitchen_relax-v1')
```

### Download pre-collected Human Teleoperation Dataset


1. To use the demos, clone the puppet VR repository and add its `vive/source` directory to the PYTHONPATH:

```bash
git clone https://github.com/vikashplus/puppet
export PYTHONPATH=$PYTHONPATH:/PATH/TO/puppet/vive/source
```

2. Use parse_demos to parse the data into pkl format with 12.5 Hz observations/actions. Unzip the kitchen_demos_multitask.zip and then run:
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

An automated data generation script runs a markovian scripted expert across random 4-subtask sequences and collects successful episodes. This scripted expert is roughly 95% successful, which should be sufficient for data generation.

Given the multi-modal, multi-task nature of Franka Kitchen, developing a true "Markovian" (in the sense that the expert's action depends only on the environment state) expert is very challenging. To make such a Markovian expert possible, we additionally record one-hot encodings containing the 4-subtask sequence (which is constant throughout a recorded episode) as observation keys in the .zarr dataset. Thus, the scripted expert *is* Markovian w.r.t. the combined environment state + subtask labels.

```bash
# Generate random 4-subtask sequences
# 4 workers is appropriate on desktop with 32 GB of RAM (and at least 4 cores)
python experts/record_demos.py --randomize --chain-len 4 \
      --seed 0 \
      --n-workers 4 \
      --out kitchen_demos_markovian_scripted_expert.zarr
```


### Evaluating Trained Policies

See `eval/README.md`.


### Installation on a SLURM Cluster

See `SLURM_README.md`.



## Citation

If you use this repo in your work, please:


1. The original `relay-policy-learning` repo that developed the Franka Kitchen environment:

```
@misc{gupta2019relaypolicylearningsolving,
      title={Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning}, 
      author={Abhishek Gupta and Vikash Kumar and Corey Lynch and Sergey Levine and Karol Hausman},
      year={2019},
      eprint={1910.11956},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1910.11956}, 
}
```

2. [Revisiting Open-Loop Execution in Robotics: Toward Reactive, Higher-Performing Policies](https://arxiv.org/abs/2608.15938):

```
@misc{zeng2026revisitingopenloopexecutionrobotics,
      title={Revisiting Open-Loop Execution in Robotics: Toward Reactive, Higher-Performing Policies}, 
      author={Michael Zeng and Abhinav Agarwal and Ajay Bati and Brian Lee and Siddharth Ancha and Russ Tedrake},
      year={2026},
      eprint={2608.15938},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2608.15938}, 
}
```
