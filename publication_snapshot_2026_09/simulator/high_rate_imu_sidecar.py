from __future__ import annotations

import types
import numpy as np
import mujoco

try:
    from scipy.signal import resample_poly
except ImportError as exc:
    raise ImportError("scipy is required for 450->100 Hz resampling") from exc


MJOBJ_BODY = (
    mujoco.mjtObj.mjOBJ_BODY
    if hasattr(mujoco, "mjtObj")
    else mujoco.mjOBJ_BODY
)


class HighRateIMUSidecar:
    """
    Passive IMU observer.

    Important:
      - Does NOT change MuJoCo physics.
      - Does NOT change the 30 Hz control policy.
      - Observes every existing MuJoCo physics substep.
      - Native physics: expected 450 Hz.
      - Target wearable output: 100 Hz.
    """

    def __init__(
        self,
        model,
        data,
        sensor_body="Torso",
        sensor_offset_local=None,
        proxy_fraction_torso_to_pelvis=0.40,
        output_hz=100.0,
    ):
        self.model = model
        self.data = data

        self.physics_dt = float(model.opt.timestep)
        self.physics_hz = 1.0 / self.physics_dt

        self.output_hz = float(output_hz)
        self.output_dt = 1.0 / self.output_hz

        body_id = mujoco.mj_name2id(
            model, MJOBJ_BODY, sensor_body
        )

        if body_id < 0:
            sensor_body = "Pelvis"
            body_id = mujoco.mj_name2id(
                model, MJOBJ_BODY, sensor_body
            )

        if body_id < 0:
            raise RuntimeError(
                "Could not find Torso or Pelvis for IMU."
            )

        self.body_id = int(body_id)
        self.body_name = (
            mujoco.mj_id2name(
                model, MJOBJ_BODY, self.body_id
            )
            or sensor_body
        )

        self.pelvis_id = mujoco.mj_name2id(
            model, MJOBJ_BODY, "Pelvis"
        )

        if sensor_offset_local is None:
            self.sensor_offset_local = (
                self._infer_l1l2_proxy_offset(
                    proxy_fraction_torso_to_pelvis
                )
            )
        else:
            self.sensor_offset_local = np.asarray(
                sensor_offset_local, dtype=float
            ).copy()

        self.buffer = {
            "timestamp": [],
            "accel_true": [],
            "gyro_true": [],
            "sensor_world_pos": [],
            "sensor_world_vel": [],
            "pelvis_height": [],
            "impact_force": [],
        }

    def _rotation(self):
        return np.asarray(
            self.data.xmat[self.body_id],
            dtype=float
        ).reshape(3, 3)

    def _infer_l1l2_proxy_offset(self, fraction):
        if (
            self.body_name.lower() == "torso"
            and self.pelvis_id >= 0
        ):
            R = self._rotation()

            torso = np.asarray(
                self.data.xpos[self.body_id],
                dtype=float
            )

            pelvis = np.asarray(
                self.data.xpos[self.pelvis_id],
                dtype=float
            )

            offset_world = float(fraction) * (
                pelvis - torso
            )

            return R.T @ offset_world

        return np.zeros(3, dtype=float)

    def _sensor_world_position(self):
        R = self._rotation()

        origin = np.asarray(
            self.data.xpos[self.body_id],
            dtype=float
        )

        return (
            origin
            + R @ self.sensor_offset_local
        )

    def _body_kinematics_local(self):
        vel6 = np.zeros(6, dtype=float)
        acc6 = np.zeros(6, dtype=float)

        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            MJOBJ_BODY,
            self.body_id,
            vel6,
            1,
        )

        mujoco.mj_rnePostConstraint(
            self.model,
            self.data,
        )

        mujoco.mj_objectAcceleration(
            self.model,
            self.data,
            MJOBJ_BODY,
            self.body_id,
            acc6,
            1,
        )

        return vel6, acc6

    def _impact_force(self):
        peak = 0.0

        for i in range(self.data.ncon):
            c = self.data.contact[i]

            b1 = int(
                self.model.geom_bodyid[int(c.geom1)]
            )
            b2 = int(
                self.model.geom_bodyid[int(c.geom2)]
            )

            if b1 != self.body_id and b2 != self.body_id:
                continue

            force = np.zeros(6, dtype=float)

            mujoco.mj_contactForce(
                self.model,
                self.data,
                i,
                force,
            )

            peak = max(
                peak,
                float(max(0.0, force[0]))
            )

        return peak

    def capture(self, sim_time=None):
        if sim_time is None:
            sim_time = float(self.data.time)

        R = self._rotation()

        vel6, acc6 = (
            self._body_kinematics_local()
        )

        omega_local = vel6[:3].copy()
        v_origin_local = vel6[3:].copy()

        alpha_local = acc6[:3].copy()
        a_origin_local = acc6[3:].copy()

        r = self.sensor_offset_local

        # Rigid-body point velocity.
        v_sensor_local = (
            v_origin_local
            + np.cross(omega_local, r)
        )

        # Rigid-body point acceleration.
        a_sensor_local = (
            a_origin_local
            + np.cross(alpha_local, r)
            + np.cross(
                omega_local,
                np.cross(omega_local, r)
            )
        )

        gravity_world = np.array(
            [0.0, 0.0, -9.81],
            dtype=float
        )

        # Specific force in the local sensor frame.
        accel_true = (
            a_sensor_local
            - R.T @ gravity_world
        )

        gyro_true = omega_local.copy()

        sensor_world_vel = (
            R @ v_sensor_local
        )

        pelvis_height = (
            float(
                self.data.xpos[
                    self.pelvis_id
                ][2]
            )
            if self.pelvis_id >= 0
            else np.nan
        )

        self.buffer["timestamp"].append(
            float(sim_time)
        )

        self.buffer["accel_true"].append(
            accel_true.copy()
        )

        self.buffer["gyro_true"].append(
            gyro_true.copy()
        )

        self.buffer[
            "sensor_world_pos"
        ].append(
            self._sensor_world_position()
        )

        self.buffer[
            "sensor_world_vel"
        ].append(
            sensor_world_vel.copy()
        )

        self.buffer[
            "pelvis_height"
        ].append(
            pelvis_height
        )

        self.buffer[
            "impact_force"
        ].append(
            self._impact_force()
        )

        return accel_true, gyro_true

    @property
    def n_native(self):
        return len(
            self.buffer["timestamp"]
        )

    def to_100hz_true_stream(self):
        """
        Anti-aliased 450 -> 100 Hz conversion.

        450 * (2/9) = 100 exactly.

        This returns the ideal physical signal only.
        Noise/bias/saturation/hardware transfer functions
        will be applied later as a separate sensor layer.
        """

        if self.n_native < 10:
            raise RuntimeError(
                "Not enough high-rate samples."
            )

        t = np.asarray(
            self.buffer["timestamp"],
            dtype=float
        )

        dt = np.diff(t)

        native_hz = (
            1.0 / np.median(dt)
        )

        # Our measured HumEnv physics rate is 450 Hz.
        if not np.isclose(
            native_hz,
            450.0,
            rtol=0.01,
            atol=0.5,
        ):
            raise RuntimeError(
                f"Unexpected native rate: "
                f"{native_hz:.6f} Hz"
            )

        def rp_vec(key):
            x = np.asarray(
                self.buffer[key],
                dtype=float
            )

            cols = [
                resample_poly(
                    x[:, k],
                    up=2,
                    down=9,
                    padtype="line",
                )
                for k in range(x.shape[1])
            ]

            return np.column_stack(cols)

        def rp_scalar(key):
            x = np.asarray(
                self.buffer[key],
                dtype=float
            )

            return resample_poly(
                x,
                up=2,
                down=9,
                padtype="line",
            )

        accel = rp_vec("accel_true")
        gyro = rp_vec("gyro_true")

        pos = rp_vec("sensor_world_pos")
        vel = rp_vec("sensor_world_vel")

        pelvis_height = rp_scalar(
            "pelvis_height"
        )

        n = len(accel)

        # Uniform hardware-style 100-Hz timestamps.
        t100 = (
            t[0]
            + np.arange(n, dtype=float)
            / self.output_hz
        )

        return {
            "timestamp": t100,
            "accel_true": accel,
            "gyro_true": gyro,
            "sensor_world_pos": pos,
            "sensor_world_vel": vel,
            "pelvis_height": pelvis_height,
            "source_hz": native_hz,
            "output_hz": self.output_hz,
            "resampling": "scipy.signal.resample_poly up=2 down=9",
        }


class HumEnvSubstepHook:
    """
    Patches only this HumEnv instance.

    env.step(action) still goes through the normal Gymnasium
    TimeLimit wrapper, but HumEnv's internal nstep=15 call is
    exposed as 15 identical individual mj_step calls.

    The user's equivalence test already demonstrated that these
    two integration forms are numerically identical.
    """

    def __init__(
        self,
        env,
        callback,
    ):
        self.env = env
        self.base = env.unwrapped
        self.callback = callback

        self._original_step = None

    def install(self):
        if self._original_step is not None:
            raise RuntimeError(
                "Hook is already installed."
            )

        base = self.base
        callback = self.callback

        self._original_step = (
            base.step
        )

        def hooked_step(this, action):
            this.data.ctrl[:] = action

            for substep in range(
                int(this.action_repeat)
            ):
                mujoco.mj_step(
                    this.model,
                    this.data,
                )

                callback(
                    float(this.data.time)
                )

            if this.data.warning.number.any():
                warning_index = int(
                    np.nonzero(
                        this.data.warning.number
                    )[0][0]
                )

                warning = mujoco.mjtWarning(
                    warning_index
                ).name

                raise ValueError(
                    "UNSTABLE MUJOCO. "
                    "Stopped due to divergence "
                    f"({warning}).\n"
                )

            # Preserve original HumEnv behavior.
            mujoco.mj_step1(
                this.model,
                this.data,
            )

            observation = (
                this.get_obs()
            )

            reward = this.task.compute(
                this.model,
                this.data,
            )

            terminated = (
                this.is_terminated()
            )

            truncated = False

            info = this.get_info()

            return (
                observation,
                reward,
                terminated,
                truncated,
                info,
            )

        base.step = types.MethodType(
            hooked_step,
            base,
        )

        return self

    def uninstall(self):
        if self._original_step is not None:
            self.base.step = (
                self._original_step
            )

            self._original_step = None

    def __enter__(self):
        return self.install()

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ):
        self.uninstall()
        return False
