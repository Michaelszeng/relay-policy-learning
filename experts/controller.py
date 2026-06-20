"""Kinematics + operational-space controller for the Franka kitchen env.

The kitchen env (`KitchenTaskRelaxV1`) only accepts a 9-D joint command through
`env.step(a)`: 7 arm joints + 2 gripper fingers, interpreted by `Robot_VelAct`
as joint *velocities* (`q_next = q_now + clip(vel)*dt`, normalized by `act_amp`).

There is no Cartesian/operational-space actuator, so this module provides the
thin layer the scripted expert needs to think in end-effector space:

  * read the end-effector pose and the site Jacobian from the live sim,
  * read arbitrary body-frame points in world coordinates (object handles),
  * convert a desired EE pose (position [+ orientation]) and a gripper command
    into the 9-D normalized joint-velocity action via damped least squares.

Everything reads the *current* sim state, so the resulting action is a pure
function of the current observation (Markovian).
"""

import mujoco
import numpy as np

# --- env-specific constants -------------------------------------------------
EE_SITE = "end_effector"  # site on panda0_link7, ~at the fingertips
N_ARM = 7  # arm joints occupy qpos/qvel/jac cols 0..6
GRIPPER_QPOS = (7, 8)  # finger_joint1, finger_joint2 in qpos
FINGER_OPEN = 0.04  # per-finger joint upper limit (fully open)
FINGER_CLOSED = 0.0  # per-finger joint lower limit (closed)
ACT_AMP = 2.0  # env multiplies the action by this (act_amp)

# control gains / limits
KP_POS = 8.0  # cartesian P gain (1/s) on position error
KP_ROT = 3.0  # cartesian P gain (1/s) on orientation error
V_LIN_MAX = 0.4  # m/s cap on commanded linear EE velocity
V_ANG_MAX = 2.0  # rad/s cap on commanded angular EE velocity
DLS_LAMBDA = 0.1  # damped-least-squares damping (rad)
# Large enough that a "close"/"open" command saturates the finger velocity and
# the position-actuator target clips to its limit (0 or 0.04). That matters for
# grasping: the env's Robot_VelAct recomputes the finger target from the current
# (blocked) finger position each step, so a gentle gain leaves only a tiny
# position error and a weak squeeze (~6 N) that slips; saturating clips the
# target to 0 and gives the full ~10 N needed to hold the handle under load.
KP_GRIP = 25.0  # P gain (1/s) driving fingers to their target opening
KP_JOINT = 4.0  # joint-space P gain (1/s) on joint position error
V_JOINT_MAX = 1.5  # rad/s cap per arm joint, used by joint_pose_to_action


def _mjmodel(env):
    return env.sim.model


def _mjdata(env):
    return env.sim.data


def ee_pos(env):
    """World position (3,) of the end-effector site."""
    d = _mjdata(env)
    sid = _mjmodel(env).site_name2id(EE_SITE)
    return d.site_xpos[sid].copy()


def ee_rot(env):
    """World rotation matrix (3, 3) of the end-effector site."""
    d = _mjdata(env)
    sid = _mjmodel(env).site_name2id(EE_SITE)
    return d.site_xmat[sid].reshape(3, 3).copy()


def gripper_width(env):
    """Sum of the two finger joint openings (0 closed .. ~0.08 fully open)."""
    qp = _mjdata(env).qpos
    return float(qp[GRIPPER_QPOS[0]] + qp[GRIPPER_QPOS[1]])


def body_point_world(env, body_name, local_offset=(0.0, 0.0, 0.0)):
    """World position of a point given in a body's local frame."""
    m, d = _mjmodel(env), _mjdata(env)
    bid = m.body_name2id(body_name)
    xpos = d.xpos[bid]
    xmat = d.xmat[bid].reshape(3, 3)
    return xpos + xmat @ np.asarray(local_offset, dtype=float)


def joint_anchor_axis(env, joint_name):
    """World-frame (anchor, axis) of a hinge joint (both (3,))."""
    m, d = _mjmodel(env), _mjdata(env)
    jid = m.joint_name2id(joint_name)
    return d.xanchor[jid].copy(), d.xaxis[jid].copy()


def site_jacobian(env):
    """6x7 end-effector site Jacobian (linear stacked over angular, arm cols)."""
    m, d = _mjmodel(env), _mjdata(env)
    sid = m.site_name2id(EE_SITE)
    nv = m.ptr.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    mujoco.mj_jacSite(m.ptr, d.ptr, jacp, jacr, sid)
    return jacp[:, :N_ARM], jacr[:, :N_ARM]


def control_dt(env):
    """Duration (s) of one env.step: frame_skip * sim timestep (here 0.08 s)."""
    return env.skip * _mjmodel(env).ptr.opt.timestep


def gravity_feedforward(env):
    """Joint-velocity feedforward (N_ARM,) that holds the arm against gravity.

    The env integrates the action (a joint *velocity*) into the position-actuator
    setpoint once per control step: ctrl = qpos + vel*dt. A MuJoCo position
    actuator supplies torque kp*(ctrl - qpos), so to balance the gravity/bias
    torque tau (= data.qfrc_bias, gravity + Coriolis) it needs a standing offset
    ctrl - qpos = tau/kp. Feeding vel_ff = tau/(kp*dt) produces exactly that
    offset, so the cartesian P term no longer has to leave a residual pose error
    to supply the holding torque -- that residual *is* the steady-state droop,
    worst at high/extended reach where the weak wrist actuators (kp=120) near
    saturate. tau is a function of the current state, so this stays Markovian.
    """
    m, d = _mjmodel(env), _mjdata(env)
    kp = np.array([m.ptr.actuator_gainprm[i, 0] for i in range(N_ARM)])
    return d.qfrc_bias[:N_ARM] / (kp * control_dt(env))


def orientation_error(R_cur, R_des):
    """Angular error (3,) in world frame that rotates R_cur toward R_des.

    Geometric axis-angle error: the rotation vector of R_des @ R_cur^T. Unlike
    the small-angle sum-of-cross-products form, this stays accurate for large
    targets (e.g. the 45 deg open-phase yaw), so the commanded angular velocity
    points along the true error axis and the EE doesn't pitch out of plane.
    """
    R_err = R_des @ R_cur.T
    w = np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]
    )
    s = np.linalg.norm(w) / 2.0  # sin(angle)
    c = (np.trace(R_err) - 1.0) / 2.0  # cos(angle)
    if s < 1e-6:
        return 0.5 * w  # small angle: w/2 ~= angle * axis
    angle = np.arctan2(s, c)
    return (angle / (2.0 * s)) * w  # = angle * unit_axis


def pose_to_action(env, target_pos, target_rot=None, gripper="open", gravity_comp=True):
    """Map a desired EE pose + gripper command to the 9-D normalized action.

    Args:
        target_pos: (3,) desired world position of the EE site.
        target_rot: optional (3, 3) desired world rotation of the EE site.
                    If None, orientation is left uncontrolled.
        gripper: "open" or "close" (or a float per-finger target in [0, 0.04]).
        gravity_comp: add a joint-velocity feedforward that holds the arm against
                    gravity (see gravity_feedforward), removing the steady-state
                    droop of the pure-P cartesian loop. On by default.

    Returns:
        (9,) action in [-1, 1] suitable for env.step.
    """
    # --- cartesian twist from pose error ---
    pos_err = np.asarray(target_pos, dtype=float) - ee_pos(env)
    v_lin = np.clip(KP_POS * pos_err, -V_LIN_MAX, V_LIN_MAX)

    jacp, jacr = site_jacobian(env)
    if target_rot is not None:
        rot_err = orientation_error(ee_rot(env), np.asarray(target_rot, float))
        v_ang = np.clip(KP_ROT * rot_err, -V_ANG_MAX, V_ANG_MAX)
        J = np.vstack([jacp, jacr])  # 6x7
        twist = np.concatenate([v_lin, v_ang])
    else:
        J = jacp  # 3x7
        twist = v_lin

    # --- damped least squares: dq = J^T (J J^T + lam^2 I)^-1 twist ---
    m = J.shape[0]
    JJt = J @ J.T + (DLS_LAMBDA**2) * np.eye(m)
    dq_arm = J.T @ np.linalg.solve(JJt, twist)  # (7,)

    # --- gravity-compensation feedforward (holds the arm, kills steady-state droop) ---
    if gravity_comp:
        dq_arm = dq_arm + gravity_feedforward(env)

    dq_grip = _gripper_dq(env, gripper)
    qvel_cmd = np.concatenate([dq_arm, dq_grip])  # (9,)
    action = np.clip(qvel_cmd / ACT_AMP, -1.0, 1.0)
    return action


def _gripper_dq(env, gripper):
    """Finger joint velocities (2,) driving toward the gripper command."""
    if gripper == "open":
        finger_target = FINGER_OPEN
    elif gripper == "close":
        finger_target = FINGER_CLOSED
    else:
        finger_target = float(gripper)
    qp = _mjdata(env).qpos
    return np.array(
        [
            KP_GRIP * (finger_target - qp[GRIPPER_QPOS[0]]),
            KP_GRIP * (finger_target - qp[GRIPPER_QPOS[1]]),
        ]
    )


def joint_pose_to_action(env, target_arm_qpos, gripper, gravity_comp=True):
    """Map a desired arm joint configuration + gripper command to the 9-D action.

    Drives the 7 arm joints directly in joint space (P control on joint
    position error) rather than through the Cartesian operational-space
    controller. Matching EE pose alone leaves the arm's redundant null-space
    configuration unconstrained -- two subtasks can both call themselves "home"
    at the same EE pose via different joint configurations, one of which may be
    collision-prone for the next subtask's approach. Driving to a specific joint
    configuration (the env's reset qpos) removes that ambiguity.
    """
    qp = _mjdata(env).qpos
    q_err = np.asarray(target_arm_qpos, dtype=float) - qp[:N_ARM]
    dq_arm = np.clip(KP_JOINT * q_err, -V_JOINT_MAX, V_JOINT_MAX)

    if gravity_comp:
        dq_arm = dq_arm + gravity_feedforward(env)

    dq_grip = _gripper_dq(env, gripper)
    qvel_cmd = np.concatenate([dq_arm, dq_grip])  # (9,)
    action = np.clip(qvel_cmd / ACT_AMP, -1.0, 1.0)
    return action
