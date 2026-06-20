"""Slide subtask: grab the sliding cabinet door's vertical handle and slide it
to the right (along +world-x).

Phases (each entered purely from the current obs, so the FSM is Markovian):
    pre-pre-grasp -> pre-grasp -> grasp -> slide -> release -> post-release
    -> pre-return-to-home -> return-to-home

The door is a 1-DOF prismatic joint (`slidedoor_joint`, qpos[19], axis +world-x,
range 0..0.44; the D4RL goal is 0.37). We side-grasp the vertical handle post
exactly like the microwave/kettle handles (approach +world-y, fingers opening
along +-world-x). Once grasped we push the post a small bounded lead in +x each
step until the door is open. While sliding the gripper is yawed -45 deg about
world z, which eases the push.

pre-pre-grasp exists to dodge the hinge cabinet door: if the hinge subtask runs
before slide, its door can be swung open, sweeping through roughly the same
height band the slide handle sits in. Going straight from home to the (high)
pre-grasp pose cuts diagonally through that band and can clip the open door.
pre-pre-grasp instead squares up in x/y at the pre-grasp pose's column while
staying well below the door's swing band, then pre-grasp rises straight up
that column to the actual stand-off. It reaches the column in two stages --
rising straight up from home before sweeping sideways -- since cutting the x/y
and z move from home together swings the arm through its own base.

pre-return-to-home applies the same dodge in reverse on the way out: slide
finishes at the handle's high reach, inside the hinge door's swing band, so
heading straight home would sweep laterally through that band again. It first
descends straight down (same x/y) to home's height, then sweeps laterally to
home's x/y at that now-safe height, before handing off to the joint-space
return-to-home controller.
"""

import numpy as np
from controller import body_point_world, ee_pos, gripper_width

from .base import HOME_EE_POS, HOME_EE_ROT, GripperCmd, Subtask, qpos, rz


class SlideSubtask(Subtask):
    """Slide the cabinet door open by side-grasping its vertical handle post."""

    # Vertical handle post in the slidelink body frame (from the kitchen XML: a
    # vertical cylinder centered at local (-0.183, -0.123, 0), ~0.044 dia and
    # ~0.32 tall, the same kind of bar as the microwave/kettle handles). We aim
    # below its center: the post center sits at world z~2.6, just out of the
    # arm's reach (the EE tops out near z~2.47), so we grasp low on the bar at a
    # height the EE can actually reach -- still firmly on the post, which spans
    # z~2.44..2.76.
    HANDLE_BODY = "slidelink"
    HANDLE_LOCAL = (-0.183, -0.123, -0.05)
    SLIDE_QPOS = 19  # prismatic door joint
    # Approach is along +world-y (the EE moves +y onto the post), same gripper
    # orientation as the microwave/kettle (fingers open along +-world-x).
    APPROACH = np.array([0.0, 1.0, 0.0])
    # Retreat direction after releasing slide handle
    RETREAT = np.array([0.0, 1.0, 1.0])
    # The door slides along +world-x; "to the right" = increasing qpos[19].
    SLIDE_AXIS = np.array([1.0, 0.0, 0.0])
    # Target depths along the approach axis, measured from the post center.
    # Negative = on the approach side (in front of the post, i.e. -world-y);
    # grasp targets the post center (0) so the bar sits between the finger pads.
    # Pre-grasp first parks at PREGRASP_DEPTH (a stand-off a little -y of the
    # post) and only drives straight in to the grasp pose once squared away, so
    # the post seats between the finger pads before they close.
    PREGRASP_DEPTH = -0.10
    GRASP_DEPTH = 0.0
    # pre-pre-grasp parks at the pre-grasp pose's x/y column, staying well below
    # the hinge door's swing band (so the later rise to the actual pre-grasp
    # stand-off is a straight-up move at a column clear of the door rather than
    # a diagonal cut through it from home). It gets there in two stages -- rise
    # straight up from home first, then sweep sideways to the column at that
    # raised height -- rather than moving x/y and z together: a direct diagonal
    # from home swings the arm through its own base (self-collision). The rise
    # height is still well clear of the hinge band (~z 2.44+).
    PRE_PRE_GRASP_RISE_Z = HOME_EE_POS[2] + 0.3
    PRE_PRE_GRASP_RISE_TOL = 0.05
    PRE_PRE_GRASP_XY_TOL = 0.10

    # The door is considered open (slid far enough) once it is past SLIDE_OPEN
    # and still essentially shut while below SLIDE_CLOSED (joint range 0..0.44,
    # D4RL goal 0.37; SLIDE_OPEN ~0.33 leaves the door ~75% open). The wide gap
    # between the two is hysteresis, exactly as in microwave.py: the approach /
    # grasp phases fire only while _door_closed, so once the door has moved, the
    # small rebound back toward shut after release (the door springs in a touch
    # when let go) can't drop the FSM back into a re-grasp -- it falls through to
    # return-to-home instead.
    SLIDE_OPEN = 0.33
    SLIDE_CLOSED = 0.10
    # While sliding we lead the post by a small fixed SLIDE_STEP in +x rather
    # than aiming at the far goal: a goal-sized lead would yank the EE far ahead
    # of the (friction-resisted) post, the grasp would read "not near", and the
    # FSM would drop the push and re-open. A bounded lead keeps the EE just
    # ahead of the post -- a continuous push the grasp's "near" tol always passes.
    SLIDE_STEP = 0.07
    # While sliding, yaw the gripper -45 deg about world z. The handle sits high
    # (world z~2.6) and this wrist yaw eases the push along the rail.
    SLIDE_YAW = np.deg2rad(-45)
    # Approach/grasp with a +15 deg world-z yaw instead of flat. When slide runs
    # right after hinge (hinge door left swung open), the flat-gripper approach
    # swings just wide enough to nudge the hinge door a few degrees on the way
    # in to grasp -- enough to fail the hinge subtask's already-met goal. The
    # small yaw narrows the swept profile on that side and clears the door.
    GRASP_YAW = np.deg2rad(20)
    # Grasp "near" tolerance. Generous: while sliding, the EE leads the post by
    # the slide lead and the pitch nudges it a few cm, so a tight tol would read
    # "not grasping" mid-push, drop to an approach phase, and re-open the grip.
    GRASP_NEAR = 0.17
    # After releasing, back the gripper out in RETREAT direction
    POST_RELEASE_RETREAT_DEPTH = -0.20
    POST_RELEASE_CLEAR_DEPTH = -0.15  # once past this far in -y, clear to go home

    # pre-return-to-home: once clear of the post, heading straight home from the
    # handle's high reach sweeps laterally through the hinge door's swing band
    # (same hazard pre-pre-grasp dodges on the way in). Descend to home's height
    # at the current x/y first, then sweep laterally at that safe height, before
    # the joint-space return-to-home controller takes over.
    PRE_RETURN_TOL = 0.07

    # FSM thresholds (in approach-axis coordinates, see _axis_coords).
    LAT_TOL = 0.035  # off-axis tol to line up before grasping
    DEPTH_TOL = 0.04  # slack on reaching the grasp depth
    # A grasp is on the post when the fingers are partially closed: fully open is
    # ~0.08, closed-on-nothing is ~0, the ~0.044-dia post holds them in a band.
    GRASP_WIDTH_MIN = 0.022
    GRASP_WIDTH_MAX = 0.055
    # Released gripper held at the -45 deg slide yaw only reaches ~0.075-0.079
    # (a finger catches the cabinet frame just shy of the true 0.08 joint
    # limit) -- same lesson as microwave.py/hinge.py's GRASP_OPEN_WIDTH. Set
    # below that plateau so release still detects "opened" while yawed.
    GRASP_OPEN_WIDTH = 0.07

    PRE_RETURN_TO_HOME_Z = HOME_EE_POS[2] + 0.3

    # --- geometry helpers (read current sim state only) ---
    def handle(self, env):
        """World position of the vertical handle post (tracks the sliding door)."""
        return body_point_world(env, self.HANDLE_BODY, self.HANDLE_LOCAL)

    def pregrasp_pose(self, env):
        """Stand-off a little -y in front of the post (open-gripper park)."""
        return self.handle(env) + self.PREGRASP_DEPTH * self.APPROACH

    def pre_pre_grasp_pose(self, env):
        """Rise straight up from home, then sweep over to the pre-grasp column.

        Until the EE has risen to PRE_PRE_GRASP_RISE_Z it targets straight up
        at home's x/y (clearing the arm's own base); once risen it targets the
        pre-grasp column at that same height. Monotonic in z, so it can't
        chatter back to the rise target once the sweep has started.
        """
        if ee_pos(env)[2] < self.PRE_PRE_GRASP_RISE_Z - self.PRE_PRE_GRASP_RISE_TOL:
            return np.array([HOME_EE_POS[0], HOME_EE_POS[1], self.PRE_PRE_GRASP_RISE_Z])
        p = self.pregrasp_pose(env).copy()
        p[2] = self.PRE_PRE_GRASP_RISE_Z
        return p

    def pre_return_to_home_pose(self, env):
        """Descend straight down at the current x/y, clearing the swing band."""
        target = self.handle(env) + self.POST_RELEASE_RETREAT_DEPTH * self.RETREAT
        return np.array([target[0] - 0.5, target[1], self.PRE_RETURN_TO_HOME_Z])

    def grasp_pose(self, env):
        return self.handle(env) + self.GRASP_DEPTH * self.APPROACH

    def approach_pose(self, env):
        """Target while in pre-grasp: park at the stand-off, then drive in.

        Until the EE is lined up laterally and back at the stand-off depth it
        aims at the stand-off (parking a little -y of the post); once squared
        away it drives straight in to the grasp pose (still open) so the post
        seats between the finger pads before they close. The hand-off is purely
        a function of the current pose (Markovian) and monotone in depth -- once
        the EE moves in past the stand-off the test latches on the grasp pose,
        so it cannot chatter back to the stand-off.
        """
        depth, lateral = self._axis_coords(env)
        if lateral > self.LAT_TOL or depth < self.PREGRASP_DEPTH - self.DEPTH_TOL:
            return self.pregrasp_pose(env)
        return self.grasp_pose(env)

    def slide_pose(self, env):
        """EE target that pushes the (grasped) door open, to the right.

        The carrot is the post itself led a small fixed SLIDE_STEP in +x. Since
        the post tracks the door, this is a continuous, bounded push: the EE
        stays ~SLIDE_STEP ahead of the post (well inside the grasp's "near"
        tolerance) and keeps driving +x until the door is open.
        """
        return self.grasp_pose(env) + self.SLIDE_STEP * self.SLIDE_AXIS

    def _axis_coords(self, env):
        """(depth, lateral) of the EE relative to the post along the approach."""
        rel = ee_pos(env) - self.handle(env)
        depth = float(self.APPROACH @ rel)
        lateral = float(np.linalg.norm(rel - depth * self.APPROACH))
        return depth, lateral

    def _xy_dist(self, env, target):
        """Horizontal (x, y only) distance from the EE to `target`.

        Ignores z so it can be used to check column alignment independently of
        height, while the EE is still down at PRE_PRE_GRASP_Z.
        """
        return float(np.linalg.norm(ee_pos(env)[:2] - target[:2]))

    # --- predicates (current obs only) ---
    def _door_open(self, env):
        return float(qpos(env)[self.SLIDE_QPOS]) > self.SLIDE_OPEN

    def _door_closed(self, env):
        # A wide gap below SLIDE_OPEN (hysteresis): only while the door is still
        # essentially shut do we approach/grasp, so a small post-release rebound
        # can't bounce the FSM back into a re-grasp.
        return float(qpos(env)[self.SLIDE_QPOS]) < self.SLIDE_CLOSED

    def _grasping(self, env):
        # Generous "near": while sliding, the EE and post move together so the
        # EE stays on the post; a tight threshold would momentarily read "not
        # grasping", drop to an approach phase, and re-open the gripper.
        near = np.linalg.norm(ee_pos(env) - self.handle(env)) < self.GRASP_NEAR
        return near and self.GRASP_WIDTH_MIN < gripper_width(env) < self.GRASP_WIDTH_MAX

    def _grasp_opened(self, env):
        return gripper_width(env) > self.GRASP_OPEN_WIDTH

    def _go_home(self, env):
        """Route into return-to-home via a safe vertical-clear-first dodge.

        Only gates on height: once the EE is back down near home's z (below
        the hinge door's swing band), the rest of the trip -- including the
        lateral sweep -- is handed to the fast joint-space return-to-home
        controller, which is now safe to take. Height shrinks monotonically
        while descending, so this can't chatter back once past the gate.
        """
        pre_return_to_home_pose = self.pre_return_to_home_pose(env)
        if (
            ee_pos(env)[2] > pre_return_to_home_pose[2] + self.PRE_RETURN_TOL
            or ee_pos(env)[0] > pre_return_to_home_pose[0] + self.PRE_RETURN_TOL
        ):
            return "pre-return-to-home"
        return "return-to-home"

    def fsm_state(self, env):
        # Evaluated in priority order so the partition is exhaustive & disjoint.
        # Door slid far enough: let go, retreat clear of the post, then go home.
        if self._door_open(env):
            if not self._grasp_opened(env):
                return "release"
            depth, _ = self._axis_coords(env)
            if depth > self.POST_RELEASE_CLEAR_DEPTH and ee_pos(env)[2] >= self.handle(env)[2] - 0.05:
                return "post-release"
            return self._go_home(env)

        # Holding the post: slide the door to the right.
        if self._grasping(env):
            return "slide"

        # Still essentially shut: line up in front of the post, then grasp. (The
        # wide _door_closed gate is the anti-chatter trick from microwave.py --
        # a partially-open door that is no longer grasped falls through to the
        # final return-to-home rather than re-approaching.)
        if self._door_closed(env):
            # Square up in x/y at the pre-grasp column before rising into the
            # hinge door's swing band: horizontal distance shrinks monotonically
            # as the EE approaches, so this can't chatter back once passed.
            if self._xy_dist(env, self.pregrasp_pose(env)) > self.PRE_PRE_GRASP_XY_TOL:
                return "pre-pre-grasp"
            depth, lateral = self._axis_coords(env)
            if lateral < self.LAT_TOL and depth >= self.GRASP_DEPTH - self.DEPTH_TOL:
                return "grasp"
            return "pre-grasp"
        return self._go_home(env)

    def targets(self, env, state):
        # Approach and grasp with a small +15 deg yaw (see GRASP_YAW); once the
        # post is held, yaw -45 deg about world z to slide (eases the push at
        # the handle's high reach). Release keeps that same -45 deg yaw rather
        # than un-yawing -- GRASP_OPEN_WIDTH is set below the yawed-open
        # plateau so _grasp_opened still fires.
        yawed = rz(self.SLIDE_YAW) @ HOME_EE_ROT
        grasp_yawed = rz(self.GRASP_YAW) @ HOME_EE_ROT
        if state == "pre-pre-grasp":
            return GripperCmd.OPEN, self.pre_pre_grasp_pose(env), HOME_EE_ROT
        if state == "pre-grasp":
            return GripperCmd.OPEN, self.approach_pose(env), grasp_yawed
        if state == "grasp":
            return GripperCmd.CLOSE, self.grasp_pose(env), grasp_yawed
        if state == "slide":
            return GripperCmd.CLOSE, self.slide_pose(env), yawed
        if state == "release":
            return GripperCmd.OPEN, ee_pos(env), HOME_EE_ROT
        if state == "post-release":
            target = self.handle(env) + self.POST_RELEASE_RETREAT_DEPTH * self.RETREAT
            return GripperCmd.OPEN, target, HOME_EE_ROT
        if state == "pre-return-to-home":
            return GripperCmd.OPEN, self.pre_return_to_home_pose(env), HOME_EE_ROT
        if state == "return-to-home":
            return GripperCmd.OPEN, HOME_EE_POS, HOME_EE_ROT
        raise ValueError(f"unknown slide FSM state: {state}")
