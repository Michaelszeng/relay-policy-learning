"""Kettle subtask: pick the kettle up by its left handle post and place it on
the back burner.

Phases (each entered purely from the current obs, so the FSM is Markovian):
    pre-grasp -> grasp -> lift -> transport -> lower -> release -> post-release
    -> return-to-home

The kettle is a free body (a freejoint at qpos[23:30]); the goal is to move it to
the back-burner pose. We grasp the left handle post (a vertical bar, like the
microwave handle, with the same gripper orientation). Once grasped, the EE holds
the post rigidly, so to move the kettle by some displacement we simply command
the EE by that *same* displacement: target = ee + (kettle_desired - kettle_now).
"""

import numpy as np
from controller import body_point_world, ee_pos, ee_rot, gripper_width, orientation_error

from .base import HOME_EE_POS, HOME_EE_ROT, GripperCmd, Subtask, qpos, rx


class KettleSubtask(Subtask):
    """Pick the kettle up by its left handle post and place it on the back burner."""

    KETTLE_BODY = "kettleroot"
    KETTLE_QPOS = slice(23, 26)  # freejoint x, y, z of the kettle
    # Left handle post center in the kettleroot body frame (from the kettle XML:
    # a vertical capsule at local (-0.092, 0, 0.186)).
    HANDLE_LOCAL = (-0.092, 0.0, 0.2)
    # Approach is along +world-y (the EE moves +y onto the post), same gripper
    # orientation as the microwave (fingers open along +-world-x).
    APPROACH = np.array([0.0, 1.0, 0.0])
    GRASP_DEPTH = 0.1

    # Place target = the D4RL kettle goal position (back burner).
    GOAL_POS = np.array([-0.23, 0.75, 1.62])
    # Carry height. This is roughly where the arm bottoms out at the back-burner
    # xy (it can't set the kettle all the way down to the burner at ~1.62 when
    # extended), but at this height it reaches the goal xy *accurately*. A higher
    # carry can't reach the far xy; a lower one isn't achievable there. Because
    # the arm can't descend further once over the goal, "lower" doesn't drag the
    # xy back, so there is no lower<->transport limit cycle.
    LIFT_Z = 1.72

    # Loose "has descended below the carry height" ceiling: once the kettle is
    # under this it has left the transport height and is on its way down. This is
    # *not* the release trigger on its own (the arm bottoms out at a height that
    # varies with the approach config -- ~1.64 alone, ~1.65 when chained after
    # another subtask -- so an absolute release height is fragile: too low and the
    # chained kettle never reaches it and is stuck in "lower" forever; too high and
    # we let go in mid-air). Release is gated on the *tilt completion* instead (see
    # fsm_state), with this only as a sanity floor.
    LOWER_Z = 1.70

    # FSM thresholds.
    LAT_TOL = 0.035  # off-axis tol to line up before grasping
    DEPTH_TOL = 0.1  # slack on reaching the grasp depth
    XY_TOL = 0.06  # kettle within this of the goal in the xy-plane
    LIFT_TOL = 0.10  # slack on reaching the carry height before transporting
    # The arm can't *translate* the kettle down onto the burner at full
    # extension, so "lower" instead tilts the gripper down about world-x, which
    # rotates the held kettle down to the surface (the wrist can rotate even when
    # the arm can't reach lower). We only leave "lower" once the gripper has
    # actually *reached* LOWER_TILT -- otherwise we'd release mid-tilt, before
    # the kettle has been set down.
    LOWER_TILT = np.deg2rad(-25)  # downward pitch of the gripper during lower
    LOWER_TILT_MAG = abs(LOWER_TILT)
    TILT_TOL = 0.10  # rad; orientation error below this == tilt reached -> release
    DONE_XY_TOL = 0.20  # once near the goal & not gripping, finish (don't re-grab)
    # After releasing, back the gripper straight out in -y so it fully clears the
    # kettle before the arm sweeps home (otherwise homing can knock the kettle).
    POST_RELEASE_RETREAT = 0.20  # -y target offset from the kettle for the retreat
    POST_RELEASE_CLEAR = 0.15  # once the EE is this far in -y of the kettle, go home
    # A grasp is on the post when the fingers are partially closed: fully open is
    # ~0.08, closed-on-nothing is ~0, the ~0.04-dia post holds them in a band.
    GRASP_WIDTH_MIN = 0.022
    GRASP_WIDTH_MAX = 0.055
    GRASP_OPEN_WIDTH = 0.075

    # --- geometry helpers (read current sim state only) ---
    def handle(self, env):
        """World position of the left handle post."""
        return body_point_world(env, self.KETTLE_BODY, self.HANDLE_LOCAL)

    def grasp_pose(self, env):
        return self.handle(env) + self.GRASP_DEPTH * self.APPROACH

    def kettle_pos(self, env):
        return np.asarray(qpos(env)[self.KETTLE_QPOS], dtype=float)

    def _ee_for_kettle(self, env, kettle_desired):
        """EE target that moves the (grasped) kettle toward `kettle_desired`.

        The EE holds the post rigidly, so the EE displacement equals the kettle
        displacement -- command the EE by the kettle's remaining offset.
        """
        return ee_pos(env) + (np.asarray(kettle_desired, float) - self.kettle_pos(env))

    def _axis_coords(self, env):
        """(depth, lateral) of the EE relative to the post along the approach."""
        rel = ee_pos(env) - self.handle(env)
        depth = float(self.APPROACH @ rel)
        lateral = float(np.linalg.norm(rel - depth * self.APPROACH))
        return depth, lateral

    # --- predicates (current obs only) ---
    def _grasping(self, env):
        near = np.linalg.norm(ee_pos(env) - self.handle(env)) < 0.13
        return near and self.GRASP_WIDTH_MIN < gripper_width(env) < self.GRASP_WIDTH_MAX

    def _grasp_opened(self, env):
        return gripper_width(env) > self.GRASP_OPEN_WIDTH

    def _tilt_err(self, env):
        """Orientation error (rad) between the gripper and the LOWER_TILT pose.

        ~LOWER_TILT_MAG at the home orientation, ~0 once the tilt is reached.
        """
        target = rx(self.LOWER_TILT) @ HOME_EE_ROT
        return float(np.linalg.norm(orientation_error(ee_rot(env), target)))

    def fsm_state(self, env):
        # Evaluated in priority order so the partition is exhaustive & disjoint.
        kpos = self.kettle_pos(env)
        xy_err = float(np.linalg.norm(kpos[:2] - self.GOAL_POS[:2]))

        # Holding the kettle: lift -> transport -> lower, then release. The xy
        # guard (XY_TOL) is what commits us to placing: only once the kettle is
        # over the goal do we lower/release, and from there we never fall back to
        # transport, which kills the lower<->transport limit cycle (the tilt drags
        # the xy back a touch).
        if self._grasping(env):
            tilt_err = self._tilt_err(env)

            if xy_err < self.XY_TOL:
                # Set the kettle down, then let go once the gripper has *fully
                # tilted down* (tilt_err small) and the kettle has dropped below the
                # carry height. The wrist tilt is what rotates the held kettle onto
                # the burner, so tilt completion coincides with the kettle being
                # set down -- and unlike an absolute release height it is reachable
                # whether run alone or chained (the arm bottoms out ~1 cm higher
                # when chained). Crucially it is gated on tilt_err, a stable
                # controller-tracked signal, *not* the kettle's z-velocity: opening
                # the gripper in "release" nudges the kettle and spikes its z-vel,
                # which a velocity-settle gate misreads as "not settled" and flips
                # back to "lower" (re-closing the gripper) -- the lower<->release
                # jitter. tilt_err barely moves through the release, so it doesn't.
                if tilt_err < self.TILT_TOL and kpos[2] <= self.LOWER_Z:
                    return "release"
                return "lower"
            else:
                if kpos[2] >= self.LIFT_Z - self.LIFT_TOL:
                    return "transport"
                return "lift"

        # Not holding the kettle. If it is already near the goal it has been
        # placed (it drops the last few cm onto the burner) -> go home and never
        # re-approach it; the generous tol keeps a small settling drift from
        # flickering back to the approach. Otherwise it is still at the start ->
        # approach the post and grasp.
        if xy_err < self.DONE_XY_TOL:
            # Just placed. Back the gripper straight out in -y to clear the kettle
            # before homing; only head home once it has retreated far enough from
            # the (now stationary) kettle.
            if ee_pos(env)[1] > self.kettle_pos(env)[1] - self.POST_RELEASE_CLEAR:
                return "post-release"
            return "return-to-home"
        depth, lateral = self._axis_coords(env)
        if lateral < self.LAT_TOL and depth >= self.GRASP_DEPTH - self.DEPTH_TOL:
            return "grasp"
        return "pre-grasp"

    def targets(self, env, state):
        rot = HOME_EE_ROT
        if state == "pre-grasp":
            return GripperCmd.OPEN, self.grasp_pose(env), rot
        if state == "grasp":
            return GripperCmd.CLOSE, self.grasp_pose(env), rot
        if state == "lift":
            kpos = self.kettle_pos(env)
            return GripperCmd.CLOSE, self._ee_for_kettle(env, [kpos[0], kpos[1], self.LIFT_Z]), rot
        if state == "transport":
            return GripperCmd.CLOSE, self._ee_for_kettle(env, [self.GOAL_POS[0], self.GOAL_POS[1], self.LIFT_Z]), rot
        if state == "lower":
            # Hold position and tilt the gripper down: the wrist rotates the held
            # kettle down onto the burner (the arm can't reach lower here).
            # (Holding the goal xy instead tips the kettle past upright.)
            return (
                GripperCmd.CLOSE,
                self._ee_for_kettle(env, [self.GOAL_POS[0], self.GOAL_POS[1], self.LOWER_Z]),
                rx(self.LOWER_TILT) @ HOME_EE_ROT,
            )
        if state == "release":
            return (
                GripperCmd.OPEN,
                self._ee_for_kettle(env, [self.GOAL_POS[0], self.GOAL_POS[1], self.LOWER_Z]),
                rx(self.LOWER_TILT) @ HOME_EE_ROT,
            )
        if state == "post-release":
            # Same open, tilted-down pose as release, but with the EE target
            # backed off in -y so the gripper pulls clear of the kettle.
            ee = ee_pos(env)
            target = np.array([ee[0], self.kettle_pos(env)[1] - self.POST_RELEASE_RETREAT, ee[2]])
            return GripperCmd.OPEN, target, rx(self.LOWER_TILT) @ HOME_EE_ROT
        if state == "return-to-home":
            return GripperCmd.OPEN, HOME_EE_POS, rot
        raise ValueError(f"unknown kettle FSM state: {state}")
