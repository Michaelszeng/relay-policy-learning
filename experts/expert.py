"""Markovian scripted expert for the Franka kitchen env (top-level API).

Design:

  * The expert conditions on a fixed sequence of subtasks plus a scalar `label`
    (which subtask it is currently working on). The `label` is supplied by an
    *oracle in the environment* (see run_expert.py): it advances to the next
    subtask only once the current one is complete AND the arm has returned to a
    fixed home pose, so the arm never starts the next subtask from a weird,
    collision-prone configuration.

  * Each subtask has its own finite state machine, one per file under
    `subtasks/` (see subtasks/base.py for the interface). Every FSM state's
    entry condition is a function of the *current observation only*, so the
    state can be recomputed from scratch each step and never relies on history —
    that is what makes the expert Markovian.

Two top-level entry points:

    compute_state(env, label)  -> (subtask_name, fsm_state)
    compute_action(env, subtask_name, fsm_state) -> (gripper_cmd, target_pos, target_rot)

The caller turns (gripper_cmd, target_pos, target_rot) into a joint action with
controller.pose_to_action (a delta-pose move toward the target).
"""

from subtasks import (  # noqa: F401 (re-exported for callers/back-compat)
    BONUS_THRESH,
    GripperCmd,
    HOME_EE_POS,
    HOME_EE_ROT,
    MAX_SEQUENCE_LEN,
    N_SUBTASKS,
    RESET_ARM_QPOS,
    SUBTASK_IDS,
    SUBTASK_INFO,
    SUBTASKS,
    at_home,
    is_done,
    sequence_onehot,
    subtask_onehot,
)

# Default ordering used when none is supplied.
DEFAULT_SEQUENCE = ["microwave", "kettle", "slide", "hinge"]


def compute_state(env, subtask_name):
    """Return (subtask_name, fsm_state) for the subtask the oracle selected.

    The subtask is chosen by the oracle-provided label (passed in as
    `subtask_name`); the FSM state within it is a function of the current
    observation only.
    """
    if subtask_name not in SUBTASKS:
        raise NotImplementedError(f"subtask '{subtask_name}' not implemented yet")
    return subtask_name, SUBTASKS[subtask_name].fsm_state(env)


def compute_action(env, subtask_name, fsm_state):
    """Return (gripper_cmd, target_pos, target_rot) to command this step."""
    gripper_cmd, target_pos, target_rot = SUBTASKS[subtask_name].targets(env, fsm_state)
    gripper_str = "open" if gripper_cmd == GripperCmd.OPEN else "close"
    return gripper_str, target_pos, target_rot
