"""Shared infrastructure for the per-subtask Markovian FSMs.

This lives apart from both expert.py (the top-level API) and the individual
subtask modules so that neither has to import the other: subtask modules import
from here, and expert.py imports the registry, with no import cycle.
"""

import enum

import numpy as np


class GripperCmd(enum.Enum):
    OPEN = 1
    CLOSE = 2


# --------------------------------------------------------------------------- #
# Fixed home pose (a function of the env's fixed init_qpos, so it is constant).
# Measured at reset; the EE z-axis points roughly +world-y (the approach
# direction) and the fingers open along +-world-x.
# --------------------------------------------------------------------------- #
HOME_EE_POS = np.array([-0.432, 0.119, 2.025])
# HOME_EE_POS = np.array([-0.432, 0.219, 2.225])
HOME_EE_ROT = np.array(
    [
        [-0.036, 0.997, 0.073],
        [-0.049, -0.075, 0.996],
        [0.998, 0.032, 0.052],
    ]
)

# The 7 arm joint positions at the env's true reset qpos. EE pose alone doesn't
# pin down the arm's (redundant) null-space configuration, so two subtasks can
# each consider themselves "home" at the same EE pose via different joint
# configs -- one of which may be collision-prone for the next subtask's
# approach. return-to-home drives the arm to this exact joint configuration
# instead, so every subtask hands off from the identical posture.
RESET_ARM_QPOS = np.array([0.14838802, -1.76848573, 1.84390296, -2.4768576, 0.26025203, 0.7125331, 1.59515394])

# Subtask completion criteria — indices into env.sim.data.qpos[:30], goals and
# threshold identical to the D4RL franka_kitchen / eval/evaluate_kitchen.py.
SUBTASK_INFO = {
    "bottomknob": {"indices": np.array([11, 12]), "goal": np.array([-0.88, -0.01])},
    "topknob": {"indices": np.array([15, 16]), "goal": np.array([-0.92, -0.01])},
    "light": {"indices": np.array([17, 18]), "goal": np.array([-0.69, -0.05])},
    "slide": {"indices": np.array([19]), "goal": np.array([0.37])},
    "hinge": {"indices": np.array([20, 21]), "goal": np.array([0.0, 1.45])},
    "microwave": {"indices": np.array([22]), "goal": np.array([-0.75])},
    "kettle": {
        "indices": np.array([23, 24, 25, 26, 27, 28, 29]),
        "goal": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
    },
}
BONUS_THRESH = 0.3


# --------------------------------------------------------------------------- #
# Shared helpers (all read current sim state only).
# --------------------------------------------------------------------------- #
def qpos(env):
    """Current full qpos (robot + objects) of the sim."""
    return env.sim.data.qpos


def rz(theta):
    """Rotation matrix for a yaw of `theta` radians about the world z axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rx(theta):
    """Rotation matrix for a pitch of `theta` radians about the world x axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ry(theta):
    """Rotation matrix for a roll of `theta` radians about the world y axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def is_done(env, subtask_name):
    """True when subtask's joint(s) are within BONUS_THRESH of their goal."""
    info = SUBTASK_INFO[subtask_name]
    err = qpos(env)[info["indices"]] - info["goal"]
    return float(np.linalg.norm(err)) < BONUS_THRESH


def at_home(env, qpos_tol=0.06):
    """True when the arm's joints are back at the env's true reset configuration."""
    return float(np.linalg.norm(qpos(env)[:7] - RESET_ARM_QPOS)) < qpos_tol


class Subtask:
    """Interface every subtask FSM implements.

    Each subtask is its own finite state machine whose state is a pure function
    of the current observation (keeping the expert Markovian):

        fsm_state(env) -> str
            The current phase, recomputed from the live obs each step.
        targets(env, state) -> (GripperCmd, target_pos (3,), target_rot (3, 3))
            The EE pose + gripper command to command in that phase. The caller
            turns this into a joint action via controller.pose_to_action.
    """

    def fsm_state(self, env):
        raise NotImplementedError

    def targets(self, env, state):
        raise NotImplementedError
