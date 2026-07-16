# Policy Evaluation.

The following installation instructions pertain to evaluating a policy trained using [diffusion-policy-experiments](https://github.com/michaelszeng/diffusion-policy). The evaluation script `eval/evaluate_kitchen.py` may be easily adapted to other training pipelines.

## Installation

### 1. Install the `diffusion-policy` package and its pinned dependencies
The eval pipeline (`eval/evaluate_kitchen.py`) loads checkpoints trained with my [diffusion-policy-experiments](https://github.com/michaelszeng/diffusion-policy) repo, so we install both its pinned dependencies and the editable package itself. **Python must be 3.9**.
```bash
git clone https://github.com/michaelszeng/diffusion-policy ~/diffusion-policy

# install pinned deps + editable package
pip install -r ~/diffusion-policy/requirements.txt
pip install -e ~/diffusion-policy
```

### 2. Install the kitchen environment extras
`mujoco` must be pinned to **2.3.5**: `dm_control==1.0.12`'s onscreen viewer indexes into `mjVISSTRING`, whose layout changed in mujoco 2.3.6+ — using 2.3.7 makes the viewer import crash with `ord() expected a character, but string of length 0 found`. Offscreen rendering still works at 2.3.7, but the live viewer used by `evaluate_kitchen.py --no-headless` does not.
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




## Running Evaluation

A trial counts as a `success` if at least `--n-required-subtasks` distinct kitchen subtasks (out of the 7: `bottomknob`, `topknob`, `light`, `slide`, `hinge`, `microwave`, `kettle`) are achieved at any point before the timeout — order doesn't matter, and "achieved" means within D4RL's 0.3 distance threshold to that subtask's goal joint configuration.

```bash
python eval/evaluate_kitchen.py \
  --checkpoint /path/to/run/checkpoints/epoch=050-val_loss=0.1234.ckpt \
  --n-rollouts 10
```

Outputs land in `outputs/<date>/<time>/` (or `--output-dir <path>`). Use `--resume` together with `--output-dir` to resume from an interrupted evaluation run.
