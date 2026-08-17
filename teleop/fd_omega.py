"""
High-level wrapper around a Force Dimension omega device for teleoperation.

Runs a fast background thread that continuously reads the device pose
(position, wrist orientation, gripper gap, user button) and commands a force
so the device either floats under gravity compensation, or -- for rate-control
teleop -- is pulled back toward a "center" by a virtual spring. Consumers read
the latest cached state via ``get_state()`` at whatever rate they like (e.g.
the 20 Hz robosuite control loop) without stalling the haptic loop.

The single user button is overloaded:
  * short press  -> increments ``short_press_count``  (used to toggle gripper)
  * long  press  -> increments ``long_press_count``   (used to reset episode)
"""

import threading
import time

import numpy as np

import fdsdk


def advance_reflected_force(filtered, applied, target, *, dt_s, tau_s, rate_n_s):
    """One measured-time filter/slew step for the independent device servo."""
    filtered = np.asarray(filtered, dtype=float).copy()
    applied = np.asarray(applied, dtype=float).copy()
    target = np.asarray(target, dtype=float)
    if tau_s > 0.0:
        alpha = 1.0 - np.exp(-float(dt_s) / float(tau_s))
    else:
        alpha = 1.0
    filtered += alpha * (target - filtered)
    step = filtered - applied
    step_norm = np.linalg.norm(step)
    budget = float(rate_n_s) * float(dt_s)
    if rate_n_s > 0.0 and step_norm > budget:
        step *= budget / max(step_norm, 1e-12)
    applied += step
    return filtered, applied


def advance_grip_force(filtered, applied, target, *, dt_s, tau_s, rate_n_s):
    """Causal scalar filter and slew limit for the omega.7 gripper channel."""
    if tau_s > 0.0:
        alpha = 1.0 - np.exp(-float(dt_s) / float(tau_s))
    else:
        alpha = 1.0
    filtered = float(filtered) + alpha * (float(target) - float(filtered))
    step = filtered - float(applied)
    budget = float(rate_n_s) * float(dt_s)
    if rate_n_s > 0.0:
        step = float(np.clip(step, -budget, budget))
    return filtered, float(applied) + step


class FDOmega:
    def __init__(self, poll_hz=1000.0, long_press_s=0.7, auto_init=False,
                 read_orientation=True, spring_k=0.0, max_force=10.0,
                 home_pos=None, wall_k=0.0, wall_half=None, damping_b=15.0,
                 grip_damping=6.0, grip_tau_s=0.010, grip_rate_n_s=60.0,
                 max_grip_force=None, spring_max_force=None,
                 reflected_tau_s=0.0, reflected_rate=0.0):
        self.poll_dt = 1.0 / poll_hz
        self.long_press_s = long_press_s
        self.auto_init = auto_init
        self.read_orientation = read_orientation
        # Spring-damper + wall force model, mirroring the SDK's hold.cpp:
        #   f = -spring_k*(pos-center)  gentle centering spring within workspace
        #       -wall_k*over            stiff walls, only outside the +/-wall_half box
        #       -damping_b*velocity     damping everywhere (crisp, stable feel)
        #   then the force *vector magnitude* is clamped to max_force.
        self.spring_k = spring_k          # N/m
        self.spring_max_force = (
            None if spring_max_force is None else float(spring_max_force)
        )
        if self.spring_max_force is not None and self.spring_max_force < 0.0:
            raise ValueError("spring_max_force cannot be negative")
        self._centering_enabled = bool(self.spring_k > 0.0)
        self.wall_k = wall_k              # N/m
        self.damping_b = damping_b        # N/(m/s)
        self.max_force = max_force        # N (omega.6 peaks ~12 N)
        self.reflected_tau_s = float(reflected_tau_s)
        self.reflected_rate = float(reflected_rate)
        if self.reflected_tau_s < 0.0:
            raise ValueError("reflected_tau_s cannot be negative")
        if self.reflected_rate < 0.0:
            raise ValueError("reflected_rate cannot be negative")
        self.wall_half = None if wall_half is None else np.asarray(wall_half, dtype=float)
        # optional Cartesian home the device is driven to at startup and that the
        # spring/rate-control origin sits at (device coords, metres). None -> use
        # wherever the handle rests at startup.
        self.home_pos = None if home_pos is None else np.asarray(home_pos, dtype=float)

        self.id = -1
        self.system_name = ""
        self.serial = 0
        self.has_wrist = False            # orientation sensing (active or passive wrist)
        self.has_gripper = False          # True on omega.7 (active force gripper)
        self._grip_force = 0.0            # commanded gripper force (N), omega.7
        self.grip_damping = grip_damping # gripper velocity damping N/(m/s) (anti-buzz)
        self.grip_tau_s = float(grip_tau_s)
        self.grip_rate_n_s = float(grip_rate_n_s)
        self.max_grip_force = float(
            max_force if max_grip_force is None else max_grip_force
        )
        if self.grip_tau_s < 0.0:
            raise ValueError("grip_tau_s cannot be negative")
        if self.grip_rate_n_s < 0.0:
            raise ValueError("grip_rate_n_s cannot be negative")
        if self.max_grip_force < 0.0:
            raise ValueError("max_grip_force cannot be negative")

        # shared state (protected by _lock)
        self._lock = threading.Lock()
        self._pos = np.zeros(3)
        self._vel = np.zeros(3)
        self._velocity_valid = False
        self._rot = np.eye(3)
        self._orientation_valid = False
        self._orientation_sample_count = 0
        self._orientation_error_count = 0
        self._gripper = 0.0
        self._center = np.zeros(3)
        self._force_cmd = np.zeros(3)    # force we command (the felt resistance)
        self._force_meas = np.zeros(3)   # force the device reports applying
        self._reflected = np.zeros(3)    # externally-supplied force (e.g. sim contact)
        self._reflected_filtered = np.zeros(3)
        self._reflected_applied = np.zeros(3)  # filtered + slew-limited servo force
        self._servo_sequence = 0
        self._servo_timestamp_ns = 0
        self._servo_dt_s = self.poll_dt
        self._grip_filtered = 0.0
        self._grip_applied = 0.0
        self._grip_cmd = 0.0
        self.short_press_count = 0
        self.long_press_count = 0

        self._thread = None
        self._running = False

        # button state machine (thread-local use)
        self._btn_down = False
        self._btn_t0 = 0.0
        self._long_fired = False

    # ------------------------------------------------------------------ open
    def open(self):
        fdsdk.Init()
        # retry: after an unclean exit the device can briefly report "not found"
        self.id = -1
        for attempt in range(5):
            self.id = fdsdk.Open()
            if self.id >= 0:
                break
            if attempt == 0:
                print(">>> device busy/settling, retrying...")
            time.sleep(1.0)
        if self.id < 0:
            raise RuntimeError(
                "No Force Dimension device found. Is it powered on, and is any "
                "other program (HapticDesk, gravity, sphere, ...) still holding it? "
                "If it was just closed, wait a couple seconds and retry."
            )
        self.system_name = fdsdk.GetSystemName(self.id)
        self.serial = fdsdk.GetSerialNumber(self.id)
        self.has_wrist = fdsdk.HasWrist(self.id)
        self.has_gripper = fdsdk.HasActiveGripper(self.id)
        if self.read_orientation and not self.has_wrist:
            fdsdk.Close(self.id)
            self.id = -1
            raise RuntimeError(
                f"{self.system_name or 'This Force Dimension device'} has no wrist "
                "orientation sensor; rotation control cannot be enabled."
            )

        # Bring the DRD (robotics) layer into a known state and, crucially, leave
        # FORCES ENABLED for our manual haptic loop. This mirrors the SDK's
        # autocenter example: after drdOpen you must run the regulation thread
        # and then drdStop(frc=True) -- "stop regulation but leave forces
        # enabled". Calling dhdEnableForce alone is NOT enough after a drdOpen.
        if not fdsdk.IsInitialized(self.id):
            if not self.auto_init:
                fdsdk.Close(self.id)
                raise RuntimeError(
                    "Device is not calibrated. Re-run with auto_init=True "
                    "(the arm will move through its workspace -- keep it clear), "
                    "or calibrate first with bin/HapticInit."
                )
            print(">>> Auto-calibrating: the device will move on its own. "
                  "Keep the workspace clear...")
            if fdsdk.AutoInit(self.id) < 0:  # leaves regulation active
                raise RuntimeError("drdAutoInit failed")
            print(">>> Calibration complete.")
        else:
            # already calibrated: start regulation (holds current pose) so we
            # can stop it with forces kept on
            if fdsdk.Start(self.id) < 0:
                raise RuntimeError("drdStart failed")

        # While regulation is still running, drive the base to the home position
        # so the handle starts where we want it (e.g. raised, for maximum
        # downward travel) rather than wherever it happened to droop.
        if self.home_pos is not None:
            print(f">>> Moving device to home {np.round(self.home_pos, 3)} "
                  f"(keep the workspace clear)...")
            # slow the homing move down so it glides in instead of slamming
            gret, amax, vmax, jerk = fdsdk.GetPosMoveParam(self.id)
            if gret >= 0 and amax > 0 and vmax > 0:
                fdsdk.SetPosMoveParam(0.25 * amax, 0.25 * vmax, 0.5 * jerk, self.id)
            hx, hy, hz = (float(v) for v in self.home_pos)
            if fdsdk.MoveToPos(hx, hy, hz, True, self.id) < 0:
                print(">>> warning: drdMoveToPos failed; starting from rest instead")

        # stop the regulation thread but KEEP FORCES ON -> manual force control
        if fdsdk.Stop(True, self.id) < 0:
            raise RuntimeError("drdStop failed")
        fdsdk.EnableForce(1, self.id)  # belt-and-suspenders

        # NOTE: on the omega.7 the user "button" is emulated by fully squeezing
        # the force gripper -- but we use the gripper for ANALOG grasp, so we do
        # NOT enable button emulation (it would fire a "button" every time you
        # close the gripper). Reset is handled by a key in the viewer instead.
        if self.has_gripper:
            fdsdk.EmulateButton(0, self.id)

        # seed pose + center from a first read
        ret, x, y, z = fdsdk.GetPosition(self.id)
        if ret >= 0:
            self._pos = np.array([x, y, z])
        if self.read_orientation:
            rret, rotation = fdsdk.GetOrientationFrame(self.id)
            if rret < 0 or not self._is_rotation_matrix(rotation):
                self.close()
                raise RuntimeError(
                    "The device has a wrist, but its initial orientation frame could "
                    "not be read. Check device calibration and the SDK connection."
                )
            # Seed synchronously before consumers can call get_state(). Previously
            # _rot stayed at identity until the background thread happened to run,
            # so the first teleop iteration could capture a false wrist home.
            self._rot = np.asarray(rotation, dtype=float).copy()
            self._orientation_valid = True
            self._orientation_sample_count = 1
        self._center = (self.home_pos.copy() if self.home_pos is not None
                        else self._pos.copy())

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    # ------------------------------------------------------------ haptic loop
    def _loop(self):
        last_tick = time.perf_counter() - self.poll_dt
        next_tick = time.perf_counter()
        while self._running:
            tick = time.perf_counter()
            raw_servo_dt = max(tick - last_tick, 1e-6)
            # A delayed Python wake-up means the previous force was physically
            # held for longer; do not compensate with one unsafe force jump.
            servo_dt = min(raw_servo_dt, 0.005)
            last_tick = tick
            ret, x, y, z = fdsdk.GetPosition(self.id)
            pos = np.array([x, y, z]) if ret >= 0 else None

            rot = None
            if self.read_orientation:
                rret, R = fdsdk.GetOrientationFrame(self.id)
                if rret >= 0 and self._is_rotation_matrix(R):
                    rot = R
                else:
                    with self._lock:
                        self._orientation_error_count += 1

            gret, gap = fdsdk.GetGripperGap(self.id)
            vret, vx, vy, vz = fdsdk.GetLinearVelocity(self.id)
            velocity_sample = np.array([vx, vy, vz]) if vret >= 0 else None
            vel = velocity_sample if velocity_sample is not None else np.zeros(3)

            # force command: elastic (spring + walls + sim contact) + damping.
            f = np.zeros(3)
            if pos is not None:
                with self._lock:
                    center = self._center.copy()
                    ref_target = self._reflected.copy()
                    spring_k = self.spring_k if self._centering_enabled else 0.0
                    spring_max_force = self.spring_max_force
                    (
                        self._reflected_filtered,
                        self._reflected_applied,
                    ) = advance_reflected_force(
                        self._reflected_filtered,
                        self._reflected_applied,
                        ref_target,
                        dt_s=servo_dt,
                        tau_s=self.reflected_tau_s,
                        rate_n_s=self.reflected_rate,
                    )
                    reflected_applied = self._reflected_applied.copy()
                e = pos - center
                # Sim contact reflection, filtered and slew-limited on this
                # independent servo schedule. The producer may update more or
                # less regularly without changing the N/s force-rate setting.
                elastic = reflected_applied
                if spring_k > 0.0:                     # gentle centering spring
                    spring_force = -spring_k * e
                    spring_magnitude = np.linalg.norm(spring_force)
                    if (
                        spring_max_force is not None
                        and spring_max_force >= 0.0
                        and spring_magnitude > spring_max_force
                    ):
                        spring_force *= spring_max_force / max(
                            spring_magnitude, 1e-12
                        )
                    elastic = elastic + spring_force
                if self.wall_k > 0.0 and self.wall_half is not None:
                    over = e - np.clip(e, -self.wall_half, self.wall_half)
                    elastic = elastic - self.wall_k * over   # walls (outside box)

                # Damping stabilizes the elastic forces but only needs to act
                # when they are engaged -- ramp it in with the elastic magnitude
                # so FREE SPACE stays effortless (no drag) yet contact is stable.
                f = elastic
                if self.damping_b > 0.0:
                    scale = min(1.0, np.linalg.norm(elastic) / 2.0)
                    f = f - self.damping_b * scale * vel

                # clamp the force VECTOR magnitude (preserves direction)
                mag = np.linalg.norm(f)
                if mag > self.max_force:
                    f = f * (self.max_force / mag)
            if self.has_gripper:
                # The grasp channel gets the same measured-time filtering and
                # slew limiting as translation.  This prevents a contact onset
                # or simulator catch-up batch from snapping the user's fingers.
                with self._lock:
                    fg_target = self._grip_force
                    self._grip_filtered, self._grip_applied = advance_grip_force(
                        self._grip_filtered,
                        self._grip_applied,
                        fg_target,
                        dt_s=servo_dt,
                        tau_s=self.grip_tau_s,
                        rate_n_s=self.grip_rate_n_s,
                    )
                    grip_applied = self._grip_applied
                # gripper velocity damping -- ALWAYS on (a little viscosity on
                # the gripper axis is unnoticeable but kills residual buzz that a
                # force-scaled damping would leave undamped at small forces)
                gret2, gvel = fdsdk.GetGripperLinearVelocity(self.id)
                fg = grip_applied
                if self.grip_damping > 0.0 and gret2 >= 0:
                    fg = fg - self.grip_damping * gvel
                fg = float(
                    np.clip(fg, -self.max_grip_force, self.max_grip_force)
                )
                with self._lock:
                    self._grip_cmd = fg
                fdsdk.SetForceAndTorqueAndGripperForce(
                    float(f[0]), float(f[1]), float(f[2]), 0.0, 0.0, 0.0,
                    fg, self.id)
            else:
                fdsdk.SetForce(float(f[0]), float(f[1]), float(f[2]), self.id)

            # read back the force the device reports it is applying
            fret, mfx, mfy, mfz = fdsdk.GetForce(self.id)
            fmeas = np.array([mfx, mfy, mfz]) if fret >= 0 else None

            self._update_button()

            with self._lock:
                if pos is not None:
                    self._pos = pos
                if velocity_sample is not None:
                    self._vel = velocity_sample
                    self._velocity_valid = True
                if rot is not None:
                    self._rot = rot
                    self._orientation_valid = True
                    self._orientation_sample_count += 1
                if gret >= 0:
                    self._gripper = gap
                self._force_cmd = f
                if fmeas is not None:
                    self._force_meas = fmeas
                self._servo_sequence += 1
                self._servo_timestamp_ns = time.perf_counter_ns()
                self._servo_dt_s = raw_servo_dt

            # Absolute scheduling avoids the old period of SDK-work + 1 ms.
            next_tick += self.poll_dt
            remaining = next_tick - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            elif remaining < -5.0 * self.poll_dt:
                next_tick = time.perf_counter()

    def _update_button(self):
        pressed = bool(fdsdk.GetButton(0, self.id))
        now = time.time()
        if pressed and not self._btn_down:
            self._btn_down = True
            self._btn_t0 = now
            self._long_fired = False
        elif pressed and self._btn_down:
            if (not self._long_fired) and (now - self._btn_t0 >= self.long_press_s):
                self._long_fired = True
                with self._lock:
                    self.long_press_count += 1
        elif (not pressed) and self._btn_down:
            self._btn_down = False
            if not self._long_fired:
                with self._lock:
                    self.short_press_count += 1

    # --------------------------------------------------------------- consume
    def set_reflected_force(self, f):
        """Set an external force (device frame, N) added into the haptic loop,
        e.g. reflecting the newest force from the simulation control loop."""
        with self._lock:
            self._reflected = np.asarray(f, dtype=float)

    def clear_reflected_force(self):
        """Immediately discard both pending and smoothed reflected force.

        ``set_reflected_force(0)`` intentionally decays the device-side filter,
        which is desirable during contact but can carry a stale impulse across
        an episode reset. Collection boundaries need a hard reset instead.
        """
        with self._lock:
            self._reflected = np.zeros(3)
            self._reflected_filtered = np.zeros(3)
            self._reflected_applied = np.zeros(3)
            self._force_cmd = np.zeros(3)

    def set_grip_force(self, fg):
        """Set the gripper force (N) applied on the omega.7 force gripper.
        Positive closes / resists opening; sign depends on hand config."""
        with self._lock:
            self._grip_force = float(fg)

    def clear_grip_force(self):
        """Hard-clear pending and filtered omega.7 squeeze feedback."""
        with self._lock:
            self._grip_force = 0.0
            self._grip_filtered = 0.0
            self._grip_applied = 0.0
            self._grip_cmd = 0.0

    def set_centering_enabled(self, enabled):
        """Enable or disable the configured Cartesian home spring at runtime."""
        with self._lock:
            self._centering_enabled = bool(enabled and self.spring_k > 0.0)

    def recenter(self):
        """Reset the spring/rate-control origin. If a home position was given,
        the origin stays fixed at home (the spring holds the handle there);
        otherwise it snaps to the current handle position."""
        with self._lock:
            self._center = (self.home_pos.copy() if self.home_pos is not None
                            else self._pos.copy())

    def get_state(self):
        with self._lock:
            return {
                "pos": self._pos.copy(),
                "vel": self._vel.copy(),
                "velocity_valid": self._velocity_valid,
                "rot": self._rot.copy(),
                "has_wrist": self.has_wrist,
                "orientation_valid": self._orientation_valid,
                "orientation_sample_count": self._orientation_sample_count,
                "orientation_error_count": self._orientation_error_count,
                "center": self._center.copy(),
                "centering_enabled": self._centering_enabled,
                "gripper": self._gripper,
                "grip_force_target": self._grip_force,
                "grip_force_applied": self._grip_cmd,
                "force_cmd": self._force_cmd.copy(),
                "force_meas": self._force_meas.copy(),
                "reflected_target": self._reflected.copy(),
                "reflected_applied": self._reflected_applied.copy(),
                "servo_sequence": self._servo_sequence,
                "servo_timestamp_ns": self._servo_timestamp_ns,
                "servo_dt_s": self._servo_dt_s,
                "short_press_count": self.short_press_count,
                "long_press_count": self.long_press_count,
            }

    @staticmethod
    def _is_rotation_matrix(rotation):
        """Reject missing/corrupt SDK frames before they enter pose control."""
        rotation = np.asarray(rotation, dtype=float)
        return (
            rotation.shape == (3, 3)
            and np.all(np.isfinite(rotation))
            and np.linalg.norm(rotation.T @ rotation - np.eye(3)) < 1e-4
            and abs(np.linalg.det(rotation) - 1.0) < 1e-4
        )

    # ----------------------------------------------------------------- close
    def close(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.id >= 0:
            try:
                # zero the commanded force, then disable forces -> the device
                # engages its electromagnetic brakes (viscous damping) so the
                # arm settles gently instead of dropping when we let go.
                fdsdk.SetForce(0.0, 0.0, 0.0, self.id)
                fdsdk.EnableForce(0, self.id)
                fdsdk.SetBrakes(1, self.id)
                time.sleep(0.2)
            except Exception:
                pass
            fdsdk.Close(self.id)
            self.id = -1
