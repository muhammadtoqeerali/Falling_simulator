from __future__ import annotations

"""
Common high-rate IMU bridge for the fall simulator.

Purpose
-------
- Preserve existing 30 Hz controller/policy behavior.
- Preserve existing legacy IMU and CSV output.
- Observe each already-existing MuJoCo physics substep at 450 Hz.
- Produce a separate anti-aliased 100 Hz physics-truth IMU CSV.
- Do NOT add undocumented hardware filtering/noise to the truth stream.

The legacy simulator remains responsible for its original CSV.
This bridge writes a sibling file:

    original.csv
    original_highrate_truth.csv
"""

from pathlib import Path
import csv
import numpy as np

from high_rate_imu_sidecar import (
    HighRateIMUSidecar,
    HumEnvSubstepHook,
)


_SIDECAR_ATTR = "_protechto_highrate_imu_sidecar"
_EXPORT_WRAP_ATTR = "_protechto_highrate_export_wrapped"


def _resolve_mount_from_legacy(legacy_imu):
    """
    Later IMUValidator implementations expose their actual resolved
    body + local L1-L2 offset. Reuse those exactly.

    Early LeanIMU implementations do not expose sensor_offset_local;
    in that case return Torso/None so HighRateIMUSidecar constructs
    the shared Torso->Pelvis L1-L2 proxy.
    """
    sensor_body = "Torso"
    sensor_offset = None
    mount_source = "shared_l1l2_proxy"

    if legacy_imu is None:
        return sensor_body, sensor_offset, mount_source

    offset = getattr(
        legacy_imu,
        "sensor_offset_local",
        None,
    )

    body_name = getattr(
        legacy_imu,
        "body_name",
        None,
    )

    if offset is not None:
        arr = np.asarray(offset, dtype=float)

        if arr.shape == (3,) and np.all(np.isfinite(arr)):
            sensor_offset = arr.copy()

            if body_name:
                sensor_body = str(body_name)

            mount_source = "reused_legacy_resolved_mount"

    return sensor_body, sensor_offset, mount_source


def get_highrate_sidecar(env):
    return getattr(
        env.unwrapped,
        _SIDECAR_ATTR,
        None,
    )


def _make_highrate_sidecar(
    env,
    legacy_imu=None,
):
    base = env.unwrapped

    sensor_body, sensor_offset, mount_source = (
        _resolve_mount_from_legacy(
            legacy_imu
        )
    )

    sidecar = HighRateIMUSidecar(
        base.model,
        base.data,
        sensor_body=sensor_body,
        sensor_offset_local=sensor_offset,
        output_hz=100.0,
    )

    sidecar.mount_source = mount_source

    setattr(
        base,
        _SIDECAR_ATTR,
        sidecar,
    )

    print(
        "      [HighRateIMU] "
        f"physics={sidecar.physics_hz:.1f} Hz | "
        f"output={sidecar.output_hz:.1f} Hz | "
        f"body={sidecar.body_name} | "
        f"mount={mount_source} | "
        f"offset={np.round(sidecar.sensor_offset_local, 6)}"
    )

    return sidecar


def _truth_filename_from_legacy(
    legacy_filename,
):
    p = Path(legacy_filename)

    if p.suffix:
        return p.with_name(
            p.stem
            + "_highrate_truth"
            + p.suffix
        )

    return Path(
        str(p)
        + "_highrate_truth.csv"
    )


def export_highrate_truth_csv(
    sidecar,
    filename,
    metadata=None,
):
    """
    Export anti-aliased 100 Hz physics truth.

    Units
    -----
    acceleration : m/s^2
    gyroscope    : rad/s
    position     : m
    velocity     : m/s

    This is intentionally NOT called accel_raw or gyro_raw.
    It is the ideal inertial truth derived from MuJoCo physics.
    """
    metadata = dict(metadata or {})

    stream = (
        sidecar.to_100hz_true_stream()
    )

    t = np.asarray(
        stream["timestamp"],
        dtype=float,
    )

    acc = np.asarray(
        stream["accel_true"],
        dtype=float,
    )

    gyro = np.asarray(
        stream["gyro_true"],
        dtype=float,
    )

    pos = np.asarray(
        stream["sensor_world_pos"],
        dtype=float,
    )

    vel = np.asarray(
        stream["sensor_world_vel"],
        dtype=float,
    )

    pelvis_h = np.asarray(
        stream["pelvis_height"],
        dtype=float,
    )

    acc_mag = np.linalg.norm(
        acc,
        axis=1,
    )

    gyro_mag = np.linalg.norm(
        gyro,
        axis=1,
    )

    filename = Path(filename)

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with filename.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        # Machine-readable provenance comments.
        f.write(
            "# pipeline: "
            "highrate_physics_truth_v1\n"
        )
        f.write(
            "# signal_type: "
            "physics_truth_not_hardware_filtered\n"
        )
        f.write(
            "# acceleration_pipeline: "
            "mj_rnePostConstraint_then_mj_objectAcceleration_v1\n"
        )
        f.write(
            f"# physics_sampling_hz: "
            f"{sidecar.physics_hz:.12g}\n"
        )
        f.write(
            f"# output_sampling_hz: "
            f"{sidecar.output_hz:.12g}\n"
        )
        f.write(
            "# resampling: "
            "scipy.signal.resample_poly "
            "up=2 down=9\n"
        )
        f.write(
            f"# sensor_body: "
            f"{sidecar.body_name}\n"
        )
        f.write(
            "# sensor_offset_local_m: "
            + ",".join(
                f"{float(x):.12g}"
                for x in
                sidecar.sensor_offset_local
            )
            + "\n"
        )
        f.write(
            f"# mount_source: "
            f"{getattr(sidecar, 'mount_source', 'unknown')}\n"
        )
        f.write(
            "# accel_unit: m/s^2\n"
        )
        f.write(
            "# gyro_unit: rad/s\n"
        )

        for key, value in sorted(
            metadata.items()
        ):
            f.write(
                f"# legacy_metadata_{key}: "
                f"{value}\n"
            )

        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "accel_true_x",
            "accel_true_y",
            "accel_true_z",
            "gyro_true_x",
            "gyro_true_y",
            "gyro_true_z",
            "accel_true_mag",
            "gyro_true_mag",
            "sensor_pos_x",
            "sensor_pos_y",
            "sensor_pos_z",
            "sensor_vel_x",
            "sensor_vel_y",
            "sensor_vel_z",
            "pelvis_height",
        ])

        for i in range(len(t)):
            writer.writerow([
                f"{t[i]:.8f}",
                f"{acc[i,0]:.9f}",
                f"{acc[i,1]:.9f}",
                f"{acc[i,2]:.9f}",
                f"{gyro[i,0]:.9f}",
                f"{gyro[i,1]:.9f}",
                f"{gyro[i,2]:.9f}",
                f"{acc_mag[i]:.9f}",
                f"{gyro_mag[i]:.9f}",
                f"{pos[i,0]:.9f}",
                f"{pos[i,1]:.9f}",
                f"{pos[i,2]:.9f}",
                f"{vel[i,0]:.9f}",
                f"{vel[i,1]:.9f}",
                f"{vel[i,2]:.9f}",
                f"{pelvis_h[i]:.9f}",
            ])

    return {
        "filename": str(filename),
        "frames": len(t),
        "physics_hz": float(
            sidecar.physics_hz
        ),
        "output_hz": float(
            sidecar.output_hz
        ),
    }


def _install_export_wrapper(
    legacy_imu,
    sidecar,
):
    """
    Wrap ONLY the legacy IMU instance's existing export call.

    The original exporter executes first and is not modified.
    Then a sibling *_highrate_truth.csv is written.

    This avoids changing all 27 scenario-specific export blocks.
    """
    if legacy_imu is None:
        return

    if getattr(
        legacy_imu,
        _EXPORT_WRAP_ATTR,
        False,
    ):
        return

    method_name = None

    if callable(
        getattr(
            legacy_imu,
            "export_to_csv",
            None,
        )
    ):
        method_name = "export_to_csv"

    elif callable(
        getattr(
            legacy_imu,
            "export_csv",
            None,
        )
    ):
        method_name = "export_csv"

    if method_name is None:
        print(
            "      [HighRateIMU] WARNING: "
            "legacy IMU has no supported "
            "CSV export method"
        )
        return

    original_export = getattr(
        legacy_imu,
        method_name,
    )

    def wrapped_export(
        *args,
        **kwargs,
    ):
        # Legacy output first.
        result = original_export(
            *args,
            **kwargs,
        )

        try:
            legacy_filename = None

            if (
                isinstance(result, dict)
                and result.get("filename")
            ):
                legacy_filename = (
                    result["filename"]
                )

            elif args:
                legacy_filename = args[0]

            if not legacy_filename:
                raise RuntimeError(
                    "Could not determine "
                    "legacy IMU filename"
                )

            metadata = (
                kwargs.get("metadata")
                or kwargs.get("meta")
                or {}
            )

            truth_filename = (
                _truth_filename_from_legacy(
                    legacy_filename
                )
            )

            hr_result = (
                export_highrate_truth_csv(
                    sidecar,
                    truth_filename,
                    metadata=metadata,
                )
            )

            print(
                "      [HighRateIMU] "
                "truth CSV saved -> "
                f"{hr_result['filename']} "
                f"({hr_result['frames']} "
                "frames)"
            )

        except Exception as exc:
            # Never break the historical export.
            print(
                "      [HighRateIMU] "
                f"truth export ERROR: {exc}"
            )

        return result

    setattr(
        legacy_imu,
        method_name,
        wrapped_export,
    )

    setattr(
        legacy_imu,
        _EXPORT_WRAP_ATTR,
        True,
    )


def highrate_env_step(
    env,
    action,
    legacy_imu=None,
):
    """
    Drop-in replacement for the ONE main-loop:

        env.step(action)

    It keeps Gymnasium's wrapper path intact.

    For this call only, HumEnv's bundled nstep integration is
    exposed as individual MuJoCo physics steps. The user's
    equivalence test established identical final qpos/qvel/qacc.

    The callback passively observes each physics state.
    """
    sidecar = get_highrate_sidecar(
        env
    )

    if sidecar is None:
        sidecar = (
            _make_highrate_sidecar(
                env,
                legacy_imu=legacy_imu,
            )
        )

        _install_export_wrapper(
            legacy_imu,
            sidecar,
        )

    # Patch HumEnv.step only for this one call.
    # TimeLimit/Gymnasium wrapper behavior is preserved because
    # env.step(...) itself is still invoked.
    with HumEnvSubstepHook(
        env,
        sidecar.capture,
    ):
        return env.step(action)
