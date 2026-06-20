"""Microwave subtask: open the door by side-grasping its vertical handle bar.

Phases (each entered purely from the current obs, so the FSM is Markovian):
    move-to-pre-grasp -> move-to-grasp -> grasp -> open -> release
    -> post-release -> return-to-home

The open phase hooks the bar (a ~45 deg world-z yaw swings a finger behind it,
turning a slippy friction pinch into form closure) and drags the handle along
its hinge arc. See the surrounding comments for the why behind each trick.
"""

import numpy as np
from controller import body_point_world, ee_pos, gripper_width, joint_anchor_axis

from .base import HOME_EE_POS, HOME_EE_ROT, GripperCmd, Subtask, qpos, rx, rz


class MicrowaveSubtask(Subtask):
    """Open the microwave door by side-grasping its vertical handle bar."""

    # Handle bar center in the microdoorroot body frame (from the door XML).
    HANDLE_LOCAL = (0.475, -0.108, 0.0)
    # Z-height to grasp and open the microwave
    WORK_Z = 1.78
    # Approach is along +world-y (the EE moves +y to reach the handle).
    APPROACH = np.array([0.0, 1.0, 0.0])
    # Target depths along the approach axis, measured from the handle center.
    # Negative = on the approach side (in front of the handle); grasp targets
    # the handle center so the bar sits between the finger pads.
    PREGRASP_DEPTH = -0.14
    GRASP_DEPTH = 0.0
    # The door is opened by nudging the handle *tangentially* along its hinge
    # arc, not by pulling toward a far carrot (which drags the EE off the bar
    # and slips). The opening tangent is -(hinge_axis x r); PULL just fixes the
    # sign (which way around the arc counts as "opening").
    HINGE_JOINT = "microjoint"
    PULL = np.array([-0.105, -0.342, 0.0])
    OPEN_STEP = 0.06  # tangential lead of the open carrot from the EE position
    # While pulling, yaw the gripper about world z so a finger swings *behind*
    # the bar: the grasp stops being a pure friction pinch (which slips) and
    # becomes a form-closure hook that also pushes the handle around the arc.
    OPEN_YAW = np.deg2rad(45)
    # ...and pitch the gripper nose-down during the pull. The hook + arc-pull
    # otherwise rides the gripper *up* the bar (the contact force has an upward
    # component) until it reaches the very top and nearly slides off -- fragile for
    # a learned policy that makes small errors. A -30 deg downward pitch cancels
    # that ride so the gripper holds its grasp height (~5 cm of margin below the
    # bar top) through the whole open. The grasp itself stays un-pitched: tilting
    # the grasp reaches lower on the bar but the tilted grip is far less reliable,
    # and steeper open pitches break the form-closure hook, so -30 is the max.
    OPEN_PITCH = np.deg2rad(-30)

    # FSM thresholds. The approach phases are expressed in *approach-axis
    # coordinates* (depth = signed distance from the handle along the approach
    # axis; lateral = off-axis distance) so the phases tile monotonically with
    # depth and cannot chatter back and forth (a pure 3D distance-ball test
    # flips back the moment the EE leaves the pre-grasp ball toward the grasp).
    GRASP_LAT_TOL = 0.05  # off-axis (horizontal x) tol to advance to / stay in /
    # complete the grasp. Measured in the horizontal plane only (see _axis_coords):
    # the handle is a *vertical* bar, so the only centering that matters is the x
    # offset between the finger gap and the bar -- the EE's z error (large here,
    # the arm sags well below WORK_Z at this deep/high reach) is irrelevant. The x
    # offset stays small (~0.02), so this commits promptly and never chatters.
    DEPTH_TOL = 0.03  # slack on reaching a depth waypoint
    GRASP_DEPTH_TOL = 0.04  # slack on reaching a depth waypoint

    # A grasp is on the bar when the fingers are *partially* closed: fully open
    # is ~0.08, closed-on-nothing is ~0, the ~0.04-dia bar holds them in a band.
    GRASP_WIDTH_MIN = 0.022
    GRASP_WIDTH_MAX = 0.055
    GRASP_OPEN_WIDTH = 0.07  # released gripper near the bar only reaches ~0.078
    DOOR_OPEN_ANGLE = -0.5  # door considered "open" (reached target) once below this
    # The door is "substantially open" (task essentially done) past this less-open
    # angle. The finish sequence (release -> post-release -> home) is gated on this,
    # not the open target, so the small dip below the target right after release
    # can't kick the FSM back out into a re-grasp.
    DOOR_DONE_ANGLE = -0.3
    # Velocity index of the door hinge (qvel[DOOR_DOF]). Every joint before the
    # kettle's free joint is a 1-dof hinge/slide, so the velocity index coincides
    # with the qpos index 22. Used to tell the opening pull (door driving more
    # negative, vel < 0) apart from the post-release rebound (door springing back
    # toward closed, vel > 0) -- the one signal that distinguishes them, since both
    # pass through the same door angle (~-0.5) while the fingers are still in the
    # grasp band. See fsm_state.
    DOOR_DOF = 22
    # After releasing, retreat the gripper straight out in -y (away from the door,
    # toward the robot) before homing, so the home sweep doesn't cut across the
    # door and knock it shut (that knock, not the door's own tiny settle, was
    # closing the door and causing re-grasp cycling). Exit the retreat on a one-way
    # *y* latch: stay in post-release while the EE is still forward (y above
    # POST_RELEASE_Y_CLEAR), home once it has backed past it. The home pose sits at
    # y~0.12 (below the clear threshold), so once homing starts the EE can never
    # come forward enough to re-enter post-release. (A z latch chatters here: the
    # joint-space home's z oscillates across any z threshold.)
    POST_RELEASE_Y_CLEAR = 0.15  # back past this EE y before homing
    POST_RELEASE_Y_TARGET = 0.05  # -y point the retreat backs straight out to

    def handle(self, env):
        """World position of the handle bar center, with z pinned to WORK_Z."""
        p = body_point_world(env, "microdoorroot", self.HANDLE_LOCAL)
        p[2] = self.WORK_Z
        return p

    def pregrasp_pose(self, env):
        return self.handle(env) + self.PREGRASP_DEPTH * self.APPROACH

    def grasp_pose(self, env):
        return self.handle(env) + self.GRASP_DEPTH * self.APPROACH

    def open_pose(self, env):
        # Drag the handle along its hinge arc. The carrot's tangential lead is
        # taken from the EE's *own* position (a velocity-style pull), not from a
        # fixed offset off the handle: if the door stalls against friction, a
        # handle-anchored carrot would keep pulling the EE past the (stationary)
        # handle until the bar slips out of the pinch. The lateral position is
        # anchored halfway to the handle so the gripper stays centered on the
        # bar. The tangent at angle~0 already points straight out and curves as
        # the door swings, so this is a continuous "out -> arc" pull.
        h = self.handle(env)
        anchor, axis = joint_anchor_axis(env, self.HINGE_JOINT)
        tangent = np.cross(axis, h - anchor)
        tangent /= np.linalg.norm(tangent)
        if tangent @ self.PULL < 0:  # orient toward the opening direction
            tangent = -tangent
        target = 0.5 * (ee_pos(env) + h) + self.OPEN_STEP * tangent
        target[2] = self.WORK_Z
        return target

    def _axis_coords(self, env):
        """(depth, lateral) of the EE relative to the handle along approach.

        depth is negative on the approach side and grows toward 0 as the EE
        advances into the handle: pre-grasp sits at PREGRASP_DEPTH, grasp at
        GRASP_DEPTH.
        """
        rel = ee_pos(env) - self.handle(env)
        depth = float(self.APPROACH @ rel)
        # Off-axis distance in the *horizontal* plane only (drop z). The handle is
        # a vertical bar, so only the x offset between the finger gap and the bar
        # matters for centering; the EE's z error is large here (the arm sags well
        # below WORK_Z at this reach) but harmless -- the gripper just slides along
        # the bar. Counting z would let that sag dominate "lateral" and make the
        # approach idle/chatter at every tolerance.
        horiz = rel.copy()
        horiz[2] = 0.0
        lateral = float(np.linalg.norm(horiz - depth * self.APPROACH))
        return depth, lateral

    # -- predicates (current obs only) --
    def _door_open(self, env):
        return qpos(env)[22] < self.DOOR_OPEN_ANGLE

    def _door_closed(self, env):
        return qpos(env)[22] > self.DOOR_OPEN_ANGLE + 0.3

    def _grasping(self, env):
        # Generous "near", measured in the *horizontal* plane only (drop z): while
        # pulling the door open the EE leads the handle by up to ~OPEN_STEP and
        # also sags in z, and the bar is vertical so the z sag doesn't mean we let
        # go. Counting z (the sag reaches ~0.13 by the time the door is open) would
        # momentarily read "not grasping", drop to an approach phase mid-open, and
        # re-grab. The horizontal distance stays small (~0.03-0.06) throughout.
        rel = ee_pos(env) - self.handle(env)
        rel[2] = 0.0
        near = np.linalg.norm(rel) < 0.13
        return near and self.GRASP_WIDTH_MIN < gripper_width(env) < self.GRASP_WIDTH_MAX

    def _grasp_opened(self, env):
        return self.GRASP_OPEN_WIDTH < gripper_width(env)

    def fsm_state(self, env):
        # Evaluated in priority order so the partition is exhaustive & disjoint.
        door = float(qpos(env)[22])
        # Finish sequence (door substantially open and the gripper has begun
        # opening): let go fully, retreat clear of the door, then home. Gated on
        # the low DOOR_DONE_ANGLE so the dip below the open target right after
        # release can't bounce us back into a re-grasp, and on a one-way y latch
        # (home backs the EE to y~0.12, below POST_RELEASE_Y_CLEAR) so homing can't
        # re-enter post-release.
        if door < self.DOOR_DONE_ANGLE and not self._grasping(env):
            if not self._grasp_opened(env):
                return "release"
            if ee_pos(env)[1] > self.POST_RELEASE_Y_CLEAR:
                return "post-release"
            return "return-to-home"
        # Reached the open target while still holding -> start releasing.
        if self._door_open(env):
            return "release"
        if self._grasping(env):
            # Open <-> release jitter guard: right after release, the gripper
            # takes ~2 steps to spread past the grasp band, and in that window the
            # door springs back above the open target (vel > 0). A position-only
            # test reads "still grasping, door not open -> open" and re-pulls for a
            # single step (the cosmetic bounce). The opening pull instead drives
            # the door *more* negative (vel < 0). So once the door is substantially
            # open, a positive door velocity means the post-release rebound, not a
            # stalled pull -- keep releasing rather than re-grabbing.
            door_vel = float(env.sim.data.qvel[self.DOOR_DOF])
            if door < self.DOOR_DONE_ANGLE and door_vel > 0.0:
                return "release"
            return "open"
        if self._door_closed(env):
            depth, lateral = self._axis_coords(env)
            # Stay in grasp under a *loose* lateral tol so the finger-closing
            # contact jitter can't bounce the phase all the way back to
            # pre-grasp
            if lateral < self.GRASP_LAT_TOL and depth >= self.GRASP_DEPTH - self.GRASP_DEPTH_TOL:
                return "grasp"
            # Drive straight in once the EE has reached the standoff depth and is
            # laterally lined up. Both tols are the loose grasp tols: the EE
            # settles ~1-2 cm short in depth and ~4-5 cm laterally (controller
            # droop at this reach), so a tighter commit gate parked the arm at the
            # standoff for tens of steps waiting for tracking noise to dip under
            # it. Once in move-to-grasp the depth only increases (the standoff test
            # stays true) and the lateral stays well under GRASP_LAT_TOL, so this
            # can't bounce back to pre-grasp.
            if depth >= self.PREGRASP_DEPTH - self.DEPTH_TOL and lateral < self.GRASP_LAT_TOL:
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
            # World-z yaw hooks a finger behind the bar; world-x downward pitch
            # cancels the upward ride along the bar (see OPEN_PITCH).
            return GripperCmd.CLOSE, self.open_pose(env), rx(self.OPEN_PITCH) @ rz(self.OPEN_YAW) @ HOME_EE_ROT
        if state == "release":
            return GripperCmd.OPEN, self.grasp_pose(env), rot
        if state == "post-release":
            # Back straight out in -y (hold x and the current z) so the gripper
            # pulls clear of the door before the home move sweeps the arm back.
            ee = ee_pos(env)
            target = np.array([ee[0], self.POST_RELEASE_Y_TARGET, ee[2]])
            return GripperCmd.OPEN, target, rot
        if state == "return-to-home":
            return GripperCmd.OPEN, HOME_EE_POS, rot
        raise ValueError(f"unknown microwave FSM state: {state}")
