"""Bottomknob subtask: grasp the lower-left stove knob and turn it ~90 deg.

Phases (each entered purely from the current obs, so the FSM is Markovian):
    move-to-pre-grasp -> move-to-grasp -> grasp -> rotate -> release
    -> return-to-home

Structurally identical to the topknob (see topknob.py): "knob 2" is the same
vertical-bar stove knob, just lower on the panel, rotating about the *world -y
axis*. `knob_Joint_2` (qpos[11], range -1.57..0) is the turn angle and the D4RL
goal is -0.88 ("turned clockwise"). We side-grasp the bar (approach +world-y,
fingers along +-world-x) and, because the EE approach axis is collinear with the
knob axis, roll the wrist about world-y to turn it. The grasp point (knob2_site)
lies on the rotation axis, so it is rotation-invariant: the EE holds that point
and leads the current orientation a fixed roll each frame until the joint reaches
the goal.
"""

import numpy as np
from controller import body_point_world, ee_pos, ee_rot, gripper_width

from .base import HOME_EE_POS, HOME_EE_ROT, GripperCmd, Subtask, qpos, ry


class BottomknobSubtask(Subtask):
    """Turn the lower-left stove knob ~90 deg by side-grasping its bar and rolling."""

    # knob2_site in the "knob 2" body frame -- the bar center, which sits *on* the
    # knob's rotation axis (so it stays put as the knob turns).
    KNOB_BODY = "knob 2"
    SITE_LOCAL = (0.0, 0.0, 0.038)
    KNOB_QPOS = 11  # knob_Joint_2 turn angle
    # Approach is along +world-y (the EE moves +y onto the bar), same gripper
    # orientation as the microwave (fingers open along +-world-x).
    APPROACH = np.array([0.0, 1.0, 0.0])
    # Target depths along the approach axis, measured from the bar center.
    # Negative = on the approach side (in front of the bar). The panel blocks the
    # EE ~0.03-0.05 short of the center, so we aim a touch past the front face
    # (GRASP_DEPTH < 0) to seat the fingers firmly on the bar before closing.
    PREGRASP_DEPTH = -0.12
    GRASP_DEPTH = -0.04
    # Turning: roll the wrist about world-y, leading the current orientation by
    # ROLL_STEP each frame (a continuous velocity-style turn, like the microwave's
    # tangential open lead). ROTATE_DIR fixes which way around drives the joint
    # toward its negative goal.
    ROLL_STEP = np.deg2rad(20)
    ROTATE_DIR = +1.0  # +y roll drives knob_Joint negative (verified in sim)

    # Turn thresholds (hysteresis, like microwave's door angle): turned past
    # ROTATE_DONE (goal -0.88, bonus needs <= -0.58, so -0.75 leaves margin), and
    # still "un-turned" (so we may approach/grasp) only while above ROTATE_CLOSED.
    ROTATE_DONE = -0.75
    ROTATE_CLOSED = -0.3

    # FSM thresholds in approach-axis coordinates (see _axis_coords), same as
    # microwave/light: a loose lateral tol to *stay* in grasp gives contact-jitter
    # hysteresis so finger-closing wobble can't bounce the phase back to approach.
    LAT_TOL = 0.045  # off-axis tol to first line up before advancing to grasp
    GRASP_LAT_TOL = 0.07  # looser tol to *stay* in grasp
    DEPTH_TOL = 0.04  # slack on reaching the pre-grasp depth
    GRASP_DEPTH_TOL = 0.04  # slack on reaching the grasp depth
    # A grasp is on the bar when the fingers are partially closed (band measured
    # empirically; the ~0.028-wide bar holds them part-way).
    GRASP_WIDTH_MIN = 0.012
    GRASP_WIDTH_MAX = 0.060
    GRASP_OPEN_WIDTH = 0.06

    # --- geometry helpers (read current sim state only) ---
    def knob(self, env):
        """World position of the bar center (== knob2_site, on the turn axis)."""
        return body_point_world(env, self.KNOB_BODY, self.SITE_LOCAL)

    def pregrasp_pose(self, env):
        return self.knob(env) + self.PREGRASP_DEPTH * self.APPROACH

    def grasp_pose(self, env):
        return self.knob(env) + self.GRASP_DEPTH * self.APPROACH

    def rotate_rot(self, env):
        """Orientation carrot: lead the current gripper roll about world-y."""
        return ry(self.ROTATE_DIR * self.ROLL_STEP) @ ee_rot(env)

    def _axis_coords(self, env):
        """(depth, lateral) of the EE relative to the bar along the approach.

        depth is negative on the approach side and grows toward 0 as the EE
        advances onto the bar: pre-grasp sits at PREGRASP_DEPTH, grasp at
        GRASP_DEPTH.
        """
        rel = ee_pos(env) - self.knob(env)
        depth = float(self.APPROACH @ rel)
        lateral = float(np.linalg.norm(rel - depth * self.APPROACH))
        return depth, lateral

    # --- predicates (current obs only) ---
    def _turned(self, env):
        return float(qpos(env)[self.KNOB_QPOS]) < self.ROTATE_DONE

    def _not_turned(self, env):
        return float(qpos(env)[self.KNOB_QPOS]) > self.ROTATE_CLOSED

    def _grasping(self, env):
        # Generous "near": while rolling, the EE stays on the (axis-fixed) bar
        # center; a tight threshold would momentarily read "not grasping", drop to
        # an approach phase, and re-open the gripper.
        near = np.linalg.norm(ee_pos(env) - self.knob(env)) < 0.13
        return near and self.GRASP_WIDTH_MIN < gripper_width(env) < self.GRASP_WIDTH_MAX

    def _grasp_opened(self, env):
        return gripper_width(env) > self.GRASP_OPEN_WIDTH

    def fsm_state(self, env):
        # Evaluated in priority order so the partition is exhaustive & disjoint.
        if self._turned(env):
            return "release" if not self._grasp_opened(env) else "return-to-home"
        if self._grasping(env):
            return "rotate"
        if self._not_turned(env):
            depth, lateral = self._axis_coords(env)
            # Stay in grasp under a *loose* lateral tol so finger-closing contact
            # jitter can't bounce the phase all the way back to pre-grasp.
            if lateral < self.GRASP_LAT_TOL and depth >= self.GRASP_DEPTH - self.GRASP_DEPTH_TOL:
                return "grasp"
            if lateral < self.LAT_TOL and depth >= self.PREGRASP_DEPTH - self.DEPTH_TOL:
                return "move-to-grasp"
            return "move-to-pre-grasp"
        return "return-to-home"

    def targets(self, env, state):
        rot = HOME_EE_ROT
        if state == "move-to-pre-grasp":
            return GripperCmd.OPEN, self.pregrasp_pose(env), rot
        if state == "move-to-grasp":
            return GripperCmd.OPEN, self.grasp_pose(env), rot
        if state == "grasp":
            return GripperCmd.CLOSE, self.grasp_pose(env), rot
        if state == "rotate":
            # Hold the (axis-fixed) bar center and roll the wrist to turn the knob.
            return GripperCmd.CLOSE, self.knob(env), self.rotate_rot(env)
        if state == "release":
            return GripperCmd.OPEN, self.grasp_pose(env), ee_rot(env)
        if state == "return-to-home":
            return GripperCmd.OPEN, HOME_EE_POS, rot
        raise ValueError(f"unknown bottomknob FSM state: {state}")
