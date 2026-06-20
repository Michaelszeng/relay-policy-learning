"""Light subtask: grab the oven light switch's lever and flip it left.

Phases (each entered purely from the current obs, so the FSM is Markovian):
    move-to-pre-grasp -> move-to-grasp -> grasp -> flip -> release
    -> return-to-home

The switch is a small lever (a capsule) that pivots about a *vertical* (+world-z)
axis at the switch base; `lightswitch_joint` (qpos[17], range -0.7..0) is the flip
angle and the D4RL goal is -0.69 ("flipped left"). We side-grasp the lever tip
exactly like the microwave/kettle handles (approach +world-y, fingers opening
along +-world-x), then push the tip *tangentially* around its pivot -- the same
hinge-arc push microwave.py uses to open the door, only here the hinge axis is
vertical, so the lever swings horizontally toward -world-x. Reading the pivot's
anchor/axis live (joint_anchor_axis) makes the push a continuous "out -> arc".
"""

import numpy as np
from controller import body_point_world, ee_pos, gripper_width, joint_anchor_axis

from .base import HOME_EE_POS, HOME_EE_ROT, GripperCmd, Subtask, qpos


class LightSubtask(Subtask):
    """Flip the oven light switch left by side-grasping its lever and arcing it."""

    # Lever tip in the lightswitchroot body frame == the `light_site` position
    # (from oven_chain.xml). The lever pivots about +world-z at the switch base.
    SWITCH_BODY = "lightswitchroot"
    TIP_LOCAL = (0.0315, -0.075, 0.0)
    SWITCH_QPOS = 17  # lightswitch_joint flip angle
    HINGE_JOINT = "lightswitch_joint"
    # Z-height to grasp and flip at (the lever stays at constant world z).
    WORK_Z = 2.28
    # Approach is along +world-y (the EE moves +y onto the lever), same gripper
    # orientation as the microwave (fingers open along +-world-x).
    APPROACH = np.array([0.0, 1.0, 0.0])
    # Target depths along the approach axis, measured from the lever tip.
    # Negative = on the approach side (in front of the lever); grasp targets the
    # tip so it sits between the finger pads.
    PREGRASP_DEPTH = -0.12
    GRASP_DEPTH = 0.0
    # The switch flips by nudging the lever tip *tangentially* around its vertical
    # pivot (not by pulling toward a fixed point, which drags the EE off the
    # lever). The opening tangent is +-(axis x r); PULL fixes the sign so the
    # flip goes "left" (the tip arcs toward -world-x, driving the joint negative).
    PULL = np.array([-0.075, -0.0315, 0.0])
    FLIP_STEP = 0.1  # tangential lead of the flip carrot from the EE position

    # Flip thresholds (hysteresis, exactly like microwave's door angle): the
    # switch counts as flipped past FLIP_OPEN, and is still "un-flipped" (so we
    # may approach/grasp) only while above FLIP_CLOSED. The wide gap keeps a
    # small post-release rebound from bouncing the FSM back into a re-grasp.
    FLIP_OPEN = -0.5  # joint below this == flipped far enough (goal -0.69)
    FLIP_CLOSED = -0.2  # joint above this == still essentially un-flipped

    # FSM thresholds in approach-axis coordinates (see _axis_coords), same as
    # microwave: a loose lateral tol to *stay* in grasp gives contact-jitter
    # hysteresis so finger-closing wobble can't bounce the phase back to approach.
    LAT_TOL = 0.05  # off-axis tol to first line up before advancing to grasp
    GRASP_LAT_TOL = 0.07  # looser tol to *stay* in grasp
    DEPTH_TOL = 0.04  # slack on reaching the pre-grasp depth
    GRASP_DEPTH_TOL = 0.04  # slack on reaching the grasp depth
    # A grasp is on the lever when the fingers are partially closed. Measured: the
    # gripper fully open reads ~0.077-0.085, and closed *onto the lever* settles
    # ~0.055-0.065 (the lever is fatter than the handles -- its collision capsule
    # holds the fingers wider than a 0.04 bar would). So the grasp band tops out
    # just under the open width, well above the held width.
    GRASP_WIDTH_MIN = 0.018
    GRASP_WIDTH_MAX = 0.070
    GRASP_OPEN_WIDTH = 0.07

    # --- geometry helpers (read current sim state only) ---
    def tip(self, env):
        """World position of the lever tip, with z pinned to WORK_Z."""
        p = body_point_world(env, self.SWITCH_BODY, self.TIP_LOCAL)
        p[2] = self.WORK_Z
        return p

    def pregrasp_pose(self, env):
        return self.tip(env) + self.PREGRASP_DEPTH * self.APPROACH

    def grasp_pose(self, env):
        return self.tip(env) + self.GRASP_DEPTH * self.APPROACH

    def flip_pose(self, env):
        """EE target that arcs the (grasped) lever around its vertical pivot.

        The carrot's tangential lead is taken from the EE's *own* position (a
        velocity-style push), not a fixed offset off the lever: if the switch
        stalls, a lever-anchored carrot would keep pulling the EE past the
        (stationary) lever until it slips out of the pinch. The lateral position
        is anchored halfway to the lever so the gripper stays centered on it. The
        tangent at angle~0 points straight out and curves as the lever swings, so
        this is a continuous "out -> arc" push.
        """
        h = self.tip(env)
        anchor, axis = joint_anchor_axis(env, self.HINGE_JOINT)
        tangent = np.cross(axis, h - anchor)
        tangent /= np.linalg.norm(tangent)
        if tangent @ self.PULL < 0:  # orient toward the flip-left direction
            tangent = -tangent
        target = 0.5 * (ee_pos(env) + h) + self.FLIP_STEP * tangent
        target[2] = self.WORK_Z
        return target

    def _axis_coords(self, env):
        """(depth, lateral) of the EE relative to the lever along the approach.

        depth is negative on the approach side and grows toward 0 as the EE
        advances onto the lever: pre-grasp sits at PREGRASP_DEPTH, grasp at
        GRASP_DEPTH.
        """
        rel = ee_pos(env) - self.tip(env)
        depth = float(self.APPROACH @ rel)
        lateral = float(np.linalg.norm(rel - depth * self.APPROACH))
        return depth, lateral

    # --- predicates (current obs only) ---
    def _flipped(self, env):
        return float(qpos(env)[self.SWITCH_QPOS]) < self.FLIP_OPEN

    def _not_flipped(self, env):
        return float(qpos(env)[self.SWITCH_QPOS]) > self.FLIP_CLOSED

    def _grasping(self, env):
        # Generous "near": while arcing the lever the EE leads the tip by up to
        # ~FLIP_STEP. A tight threshold would momentarily read "not grasping",
        # drop to an approach phase, and re-open the gripper.
        near = np.linalg.norm(ee_pos(env) - self.tip(env)) < 0.13
        return near and self.GRASP_WIDTH_MIN < gripper_width(env) < self.GRASP_WIDTH_MAX

    def _grasp_opened(self, env):
        return gripper_width(env) > self.GRASP_OPEN_WIDTH

    def fsm_state(self, env):
        # Evaluated in priority order so the partition is exhaustive & disjoint.
        if self._flipped(env):
            return "release" if not self._grasp_opened(env) else "return-to-home"
        if self._grasping(env):
            return "flip"
        if self._not_flipped(env):
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
        if state == "flip":
            return GripperCmd.CLOSE, self.flip_pose(env), rot
        if state == "release":
            return GripperCmd.OPEN, self.grasp_pose(env), rot
        if state == "return-to-home":
            return GripperCmd.OPEN, HOME_EE_POS, rot
        raise ValueError(f"unknown light FSM state: {state}")
