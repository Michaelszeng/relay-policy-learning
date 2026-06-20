"""Hinge subtask: open the right hinge-cabinet door by side-grasping its handle.

Phases (each entered purely from the current obs, so the FSM is Markovian):
    move-to-pre-grasp -> move-to-grasp -> grasp -> open -> release
    -> post-release -> return-to-home

This is the microwave subtask mirrored (see microwave.py): the door is a vertical
handle bar that swings about a *vertical* axis, opened by hooking the bar (a world-z
yaw swings a finger behind it for form closure) and dragging it along its hinge arc.
The two differences from the microwave are:
  * the door opens the *other* way -- `rightdoorhinge` (qpos[21], range 0..1.57)
    grows *positive* to the D4RL goal 1.45, so the pull tangent and the hook yaw
    are flipped in sign;
  * the handle sits high (world z~2.6, out of reach), so -- like the slide -- we
    grasp low on the bar at a reachable WORK_Z (the bar spans z~2.44..2.76).
"""

import numpy as np
from controller import body_point_world, ee_pos, gripper_width, joint_anchor_axis

from .base import HOME_EE_POS, HOME_EE_ROT, GripperCmd, Subtask, qpos, rz


class HingeSubtask(Subtask):
    """Open the right hinge-cabinet door by side-grasping its vertical handle bar."""

    # Handle bar (hinge_site2) in the hingerightdoor body frame. The bar is
    # vertical, so its xy is height-independent; we grasp low at WORK_Z.
    HANDLE_BODY = "hingerightdoor"
    HANDLE_LOCAL = (-0.302, -0.128, 0.0)
    DOOR_QPOS = 21  # rightdoorhinge
    # Height to grasp/open at: the bar tops out of reach (~2.6), so grasp near its
    # bottom (the bar spans z~2.44..2.76; the EE tops out near z~2.47).
    WORK_Z = 2.45
    # Approach is along +world-y (the EE moves +y to reach the handle).
    APPROACH = np.array([0.0, 1.0, 0.0])
    # Target depths along the approach axis, measured from the handle center.
    PREGRASP_DEPTH = -0.14
    GRASP_DEPTH = 0.0
    # The door is opened by dragging the handle along its hinge arc. Unlike the
    # microwave (small ~40 deg arc, where an instantaneous tangent lead is fine),
    # this door swings ~83 deg, so a straight tangent diverges from the curved
    # path and the door stalls mid-arc. Instead we lead along the *true arc*:
    # rotate the current handle point about the (vertical) hinge anchor by
    # OPEN_LEAD_ANGLE in the opening direction. We also co-rotate the gripper yaw
    # with the door angle so the form-closure hook stays behind the bar through
    # the whole swing (a fixed world-z yaw would slide off over 83 deg).
    HINGE_JOINT = "rightdoorhinge"
    OPEN_LEAD_ANGLE = np.deg2rad(15)  # how far along the arc to lead the handle
    OPEN_YAW = np.deg2rad(-45)  # hook yaw at door=0 (then tracks the door angle)

    # FSM thresholds in approach-axis coordinates (see _axis_coords), same idea as
    # the microwave: a loose lateral tol to *stay* in grasp gives contact-jitter
    # hysteresis so finger-closing wobble can't bounce the phase back to pre-grasp.
    LAT_TOL = 0.035  # off-axis tol to first line up before advancing to grasp
    GRASP_LAT_TOL = 0.07  # looser tol to *stay* in grasp
    DEPTH_TOL = 0.03  # slack on reaching a depth waypoint
    GRASP_DEPTH_TOL = 0.04  # slack on reaching the grasp depth

    # A grasp is on the bar when the fingers are partially closed. The upper bound
    # is a touch generous (0.06): near the end of the ~83 deg swing the grip can
    # drift open to ~0.056, and a tighter bound read that as "let go" and released
    # ~1 deg short of the done threshold (door reached 1.145, needs 1.15).
    GRASP_WIDTH_MIN = 0.022
    GRASP_WIDTH_MAX = 0.06
    GRASP_OPEN_WIDTH = 0.07  # released gripper near the bar only reaches ~0.078
    # Door considered "open" once the angle is past this (D4RL goal 1.45, bonus
    # needs within 0.3 -> >= 1.15. _door_closed sits a wide
    # 0.3 below it (hysteresis): only while essentially shut do we approach/grasp,
    # so a small post-release rebound can't bounce the FSM back into a re-grasp.
    DOOR_OPEN_ANGLE = 1.14
    # The door is "substantially open" (task essentially done) past this lower
    # angle. The finish sequence (release -> post-release -> home) is gated on
    # *this*, not DOOR_OPEN_ANGLE, so the small dip below the open target right
    # after release can't kick the FSM out of the finish and back into a re-grasp.
    DOOR_DONE_ANGLE = 0.9
    # After releasing, back the gripper *straight out in -y* (a pure retreat
    # directly away from the door) so it never sweeps sideways across the door and
    # knocks it shut. We exit the retreat by distance from the handle, not by a y
    # threshold: home sits at y~0.42, so a y-gate would be re-crossed (and chatter)
    # the moment homing moves back up, whereas distance-from-handle grows
    # monotonically under both the retreat and the subsequent home move -> no
    # chatter, and the door is never approached again.
    POST_RELEASE_TARGET_Y = 0.10  # pure -y point the retreat backs straight out to
    POST_RELEASE_CLEAR_DIST = 0.15  # once this far from the handle, the EE is clear

    def handle(self, env):
        """World position of the handle bar center, with z pinned to WORK_Z."""
        p = body_point_world(env, self.HANDLE_BODY, self.HANDLE_LOCAL)
        p[2] = self.WORK_Z
        return p

    def pregrasp_pose(self, env):
        return self.handle(env) + self.PREGRASP_DEPTH * self.APPROACH

    def grasp_pose(self, env):
        return self.handle(env) + self.GRASP_DEPTH * self.APPROACH

    def open_pose(self, env):
        # Lead the handle along its *true* hinge arc: rotate the current handle
        # point about the (vertical) hinge anchor by OPEN_LEAD_ANGLE in the
        # opening direction. Following the actual curved path (rather than a
        # straight tangent, which diverges over this big arc and stalls the door)
        # drives the door all the way open.
        h = self.handle(env)
        anchor, _ = joint_anchor_axis(env, self.HINGE_JOINT)
        led = anchor + rz(self.OPEN_LEAD_ANGLE) @ (h - anchor)
        led[2] = self.WORK_Z
        return led

    def _axis_coords(self, env):
        """(depth, lateral) of the EE relative to the handle along approach."""
        rel = ee_pos(env) - self.handle(env)
        depth = float(self.APPROACH @ rel)
        lateral = float(np.linalg.norm(rel - depth * self.APPROACH))
        return depth, lateral

    # -- predicates (current obs only) --
    def _door_open(self, env):
        return qpos(env)[self.DOOR_QPOS] > self.DOOR_OPEN_ANGLE

    def _door_closed(self, env):
        return qpos(env)[self.DOOR_QPOS] < self.DOOR_OPEN_ANGLE - 0.3

    def _grasping(self, env):
        # Generous "near": while pulling the door open the EE leads the handle by
        # up to ~OPEN_STEP. A tight threshold would momentarily read "not
        # grasping", drop to an approach phase, and re-open the gripper.
        near = np.linalg.norm(ee_pos(env) - self.handle(env)) < 0.13
        return near and self.GRASP_WIDTH_MIN < gripper_width(env) < self.GRASP_WIDTH_MAX

    def _grasp_opened(self, env):
        return self.GRASP_OPEN_WIDTH < gripper_width(env)

    def fsm_state(self, env):
        # Evaluated in priority order so the partition is exhaustive & disjoint.
        door = float(qpos(env)[self.DOOR_QPOS])
        # Finish sequence: the door is substantially open and we are no longer
        # actively gripping (the gripper has begun opening) -> let go fully,
        # retreat clear of the door, then home. Gated on the *low* DOOR_DONE_ANGLE
        # (not the open target) so the dip below the target right after release
        # can't bounce the FSM back out. The retreat happens before homing so the
        # arm doesn't sweep across the door and knock it shut.
        if door > self.DOOR_DONE_ANGLE and not self._grasping(env):
            if not self._grasp_opened(env):
                return "release"
            # Retreat in -y until clear of the door, then home. Distance from the
            # handle grows monotonically under *both* the retreat and the
            # joint-space home (which pulls the arm down and toward the robot, away
            # from the high handle), so a single distance gate is a clean one-way
            # latch -- no chatter, and the door is never re-approached. (The old
            # z-margin guard flickered as the EE z wavered a few mm around WORK_Z;
            # it was only needed for the former Cartesian home, which moved back
            # toward the handle's y.)
            if np.linalg.norm(ee_pos(env) - self.handle(env)) < self.POST_RELEASE_CLEAR_DIST:
                return "post-release"
            return "return-to-home"
        # Reached the open target while still holding -> start releasing.
        if self._door_open(env):
            return "release"
        # Holding the bar but not yet at the target -> keep pulling it open.
        if self._grasping(env):
            return "open"
        # Essentially shut -> line up and grasp (hysteresis: only while quite shut
        # do we approach, so a partly-open door we let go of doesn't re-grasp).
        if self._door_closed(env):
            depth, lateral = self._axis_coords(env)
            # Stay in grasp under a *loose* lateral tol so the finger-closing
            # contact jitter can't bounce the phase all the way back to pre-grasp.
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
        if state == "open":
            # Co-rotate the hook yaw with the door so it stays behind the bar.
            door = float(qpos(env)[self.DOOR_QPOS])
            return GripperCmd.CLOSE, self.open_pose(env), rz(self.OPEN_YAW + door) @ HOME_EE_ROT
        if state == "release":
            return GripperCmd.OPEN, self.grasp_pose(env), rot
        if state == "post-release":
            # Back straight out in -y (hold x and the work height): a pure retreat
            # directly away from the door so the gripper never sweeps across it.
            ee = ee_pos(env)
            target = np.array([ee[0], self.POST_RELEASE_TARGET_Y, self.WORK_Z])
            return GripperCmd.OPEN, target, rot
        if state == "return-to-home":
            return GripperCmd.OPEN, HOME_EE_POS, rot
        raise ValueError(f"unknown hinge FSM state: {state}")
