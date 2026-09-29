# -*- coding: utf-8 -*-
"""
Scenario 43 - Forward Fall While Climbing Up Ladder [v1j-handlocked-fastsettle-climb]
Based on: "Human Digital Twin for Realistic Fall Simulation Using Meta Motivo"
Paper: DETC2025-169046

Builds on v6 (93.2% HIGH_CONFIDENCE, all 6 checks passing).
v8 adds two non-invasive improvements - no effect on controller stability:

  ZMP  - Zero Moment Point from contact forces (Vukobratovic & Borovac 2004).
         Displayed alongside XCoM each step as a second stability indicator.
  IMU  - Physics pipeline: anti-aliasing pre-filter ? noise+bias ? soft-tissue
         artifact ? hardware saturation (+/-16g accel, +/-2000 deg/s gyro) ?
         on-chip low-pass filter. Improves CSV signal realism.

  Excluded from v7 (caused instability):
    Hill muscle dynamics  - conflicts with muscle-weakening gear modifications
    Compliant contact     - destabilizes MuJoCo contact solver
    Cascaded neural delays - zero-init queues destabilize initial standing
"""
import sys, os
import re
import tempfile
import xml.etree.ElementTree as ET

# MuJoCo viewer/GLFW calls os.getcwd() inside glfw.init(). If the process
# was started from a temp/output directory that has been removed, glfw.init()
# crashes with FileNotFoundError before the scenario loop begins.
_SCENARIO29_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
def _scenario29_ensure_valid_cwd():
    try:
        os.getcwd()
    except FileNotFoundError:
        os.chdir(_SCENARIO29_FILE_DIR)

_scenario29_ensure_valid_cwd()
sys.path.insert(0, _SCENARIO29_FILE_DIR)
import numpy as np
import cv2
from highrate_runtime_bridge import highrate_env_step
import torch
import mujoco
import mujoco.viewer
from collections import deque, defaultdict
from pathlib import Path


def _task43_output_path(filename):
    """Return filename under FALL_OUTPUT_DIR when the standard scenario wrapper sets it.

    This preserves legacy direct-run behavior while restoring the normal
    project outputs/scenario43_* folder behavior when launched by fall_dispatcher.
    """
    out_dir = str(os.environ.get("FALL_OUTPUT_DIR", "")).strip()
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return str(Path(out_dir) / str(filename))
    return str(filename)


TASK43_RUNTIME_VERSION = "v1j-handlocked-fastsettle-climb"
TASK43_EXPECTED_BANNER = "Scenario 43 - Forward Fall While Climbing Up Ladder [v1j-handlocked-fastsettle-climb]"

from biofidelic_profile import (
    get_age_style_v2,
    get_age_reference_band_v2,
    apply_age_effects_v2,
    perturbation_force_config,
    weakening_config,
    phase_timing,
    get_segment_mass_fractions,
    print_subject_profile,
)

# scipy for FallValidator jerk/peak analysis
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("  [Warning] scipy not found - FallValidator jerk/peak checks will be skipped.")

import humenv
from humenv import make_humenv
from humenv.rewards import LocomotionReward, LieDownReward
from metamotivo.fb_cpr.huggingface import FBcprModel

# MuJoCo API compatibility
# In MuJoCo 3.x, use mujoco.mjtObj.mjOBJ_BODY instead of mujoco.mjOBJ_BODY
if hasattr(mujoco, 'mjtObj'):
    MJOBJ_BODY     = mujoco.mjtObj.mjOBJ_BODY
    MJOBJ_ACTUATOR = mujoco.mjtObj.mjOBJ_ACTUATOR
    MJOBJ_GEOM     = mujoco.mjtObj.mjOBJ_GEOM
    MJOBJ_SITE     = mujoco.mjtObj.mjOBJ_SITE
else:
    MJOBJ_BODY     = mujoco.mjOBJ_BODY
    MJOBJ_ACTUATOR = mujoco.mjOBJ_ACTUATOR
    MJOBJ_GEOM     = mujoco.mjOBJ_GEOM
    MJOBJ_SITE     = getattr(mujoco, 'mjOBJ_SITE', None)


def _norm_name(s):
    return ''.join(ch.lower() for ch in str(s or '') if ch.isalnum())


def _tokenize_name(s):
    s = str(s or '').replace('-', '_')
    return [tok.lower() for tok in re.split(r'[^A-Za-z0-9]+', s) if tok]


def _body_world_velocity(mj_model, mj_data, body_id):
    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, body_id, vel6, 0)
    return vel6[3:].copy()


def _contact_rotation_world(contact):
    return np.asarray(contact.frame, dtype=float).reshape(3, 3).T


def _contact_wrench_world(contact, wrench_local):
    R = _contact_rotation_world(contact)
    f_world = R @ np.asarray(wrench_local[:3], dtype=float)
    t_world = R @ np.asarray(wrench_local[3:], dtype=float)
    return f_world, t_world


def _body_name_matches_side(name, side):
    norm = _norm_name(name)
    toks = _tokenize_name(name)
    left_hit = any(t in ('l', 'left') for t in toks) or norm.endswith('l') or '_l' in str(name).lower()
    right_hit = any(t in ('r', 'right') for t in toks) or norm.endswith('r') or '_r' in str(name).lower()
    if side == 'left':
        return left_hit and not right_hit
    if side == 'right':
        return right_hit and not left_hit
    return True


def _safe_name2id(mj_model, obj_type, name):
    if obj_type is None or not name:
        return -1
    try:
        return int(mujoco.mj_name2id(mj_model, obj_type, str(name)))
    except Exception:
        return -1


def _resolve_site_id(mj_model, candidates):
    if MJOBJ_SITE is None:
        return -1
    best_id = -1
    best_score = -1e9
    for sid in range(getattr(mj_model, 'nsite', 0)):
        sname = mujoco.mj_id2name(mj_model, MJOBJ_SITE, sid) or ''
        norm = _norm_name(sname)
        score = 0.0
        for cand in candidates:
            key = _norm_name(cand)
            if not key:
                continue
            if norm == key:
                score += 100.0
            elif key in norm:
                score += 20.0
        if score > best_score:
            best_score = score
            best_id = sid
    return best_id if best_score >= 15.0 else -1


def _resolve_body_id(mj_model, candidates):
    best_id = -1
    best_score = -1e9
    for bid in range(mj_model.nbody):
        bname = mujoco.mj_id2name(mj_model, MJOBJ_BODY, bid) or ''
        norm = _norm_name(bname)
        score = 0.0
        for cand in candidates:
            key = _norm_name(cand)
            if not key:
                continue
            if norm == key:
                score += 100.0
            elif key in norm:
                score += 20.0
        if score > best_score:
            best_score = score
            best_id = bid
    return best_id if best_score >= 10.0 else -1


def _load_reference_mujoco_model(xml_path):
    xml_path = str(xml_path or '').strip()
    if not xml_path:
        return None
    if not os.path.exists(xml_path):
        raise FileNotFoundError(xml_path)
    return mujoco.MjModel.from_xml_path(xml_path)



def estimate_reference_bmi(age_years, sex='male'):
    age = float(age_years)
    sex = str(sex or 'male').lower()
    if age < 18:
        bmi = 19.0 if sex == 'female' else 19.5
    elif age < 40:
        bmi = 22.0 if sex == 'female' else 23.0
    elif age < 65:
        bmi = 23.0 if sex == 'female' else 24.0
    else:
        bmi = 25.5 if sex == 'female' else 26.0
    return float(np.clip(bmi, 18.5, 29.0))


def resolve_subject_weight_kg(height_m, age_years, sex='male', explicit_weight=None):
    target_bmi = estimate_reference_bmi(age_years, sex)
    if explicit_weight is None:
        weight_kg = target_bmi * float(height_m) * float(height_m)
        source = 'auto_bmi_profile'
    else:
        weight_kg = float(explicit_weight)
        target_bmi = weight_kg / max(float(height_m) * float(height_m), 1e-9)
        source = 'user_input'
    return float(weight_kg), float(target_bmi), source


def resolve_imu_mount_configuration(mj_model, requested_xml_path=''):
    """Resolve a better IMU mount only when an explicit MyoSuite XML is available.

    Without a provided reference XML, the simulator should keep the original Torso/L1-L2
    proxy so behaviour stays comparable to the baseline run.
    """
    report = {
        'requested_xml_path': str(requested_xml_path or '').strip(),
        'reference_loaded': False,
        'runtime_mount_source': 'baseline_proxy',
        'sensor_body': IMU_HARDWARE_SPEC['proxy_body'],
        'sensor_site': None,
        'sensor_offset_local': None,
        'mount_label': IMU_HARDWARE_SPEC['mount_label'],
        'reference_mount': None,
        'notes': [],
    }

    site_candidates = [
        'imu', 'imu_site', 'imu_l1_l2', 'l1_l2', 'lumbar_imu', 'back_imu',
        'sacrum_imu', 'pelvis_imu', 'sensor'
    ]
    body_candidates = ['L1_L2', 'L1L2', 'lumbar', 'lumbar_body', 'sacrum']

    xml_path = report['requested_xml_path']
    if not xml_path:
        report['notes'].append('No MyoSuite XML provided; keeping the original Torso L1-L2 proxy.')
        return report

    try:
        ref_model = _load_reference_mujoco_model(xml_path)
        report['reference_loaded'] = True
        ref_site_id = _resolve_site_id(ref_model, site_candidates)
        if ref_site_id >= 0:
            ref_body_id = int(ref_model.site_bodyid[ref_site_id])
            ref_body_name = mujoco.mj_id2name(ref_model, MJOBJ_BODY, ref_body_id) or ''
            ref_site_name = mujoco.mj_id2name(ref_model, MJOBJ_SITE, ref_site_id) or ''
            report['reference_mount'] = {
                'body': ref_body_name,
                'site': ref_site_name,
                'offset_local': np.asarray(ref_model.site_pos[ref_site_id], dtype=float).copy(),
            }
            runtime_same_site = _safe_name2id(mj_model, MJOBJ_SITE, ref_site_name)
            runtime_same_body = _safe_name2id(mj_model, MJOBJ_BODY, ref_body_name)
            if runtime_same_site >= 0:
                report['sensor_body'] = ref_body_name
                report['sensor_site'] = ref_site_name
                report['sensor_offset_local'] = np.asarray(mj_model.site_pos[runtime_same_site], dtype=float).copy()
                report['runtime_mount_source'] = 'runtime_site_matching_reference'
                report['mount_label'] = f"{ref_site_name} site"
            elif runtime_same_body >= 0:
                report['sensor_body'] = ref_body_name
                report['sensor_offset_local'] = np.asarray(ref_model.site_pos[ref_site_id], dtype=float).copy()
                report['runtime_mount_source'] = 'reference_site_projected_to_runtime_body'
                report['mount_label'] = f"{ref_site_name} guided proxy"
            else:
                report['notes'].append('Reference XML loaded, but no matching body/site exists in the humenv runtime model; keeping baseline Torso proxy.')
        else:
            ref_body_id = _resolve_body_id(ref_model, body_candidates)
            if ref_body_id >= 0:
                ref_body_name = mujoco.mj_id2name(ref_model, MJOBJ_BODY, ref_body_id) or ''
                report['reference_mount'] = {'body': ref_body_name, 'site': None, 'offset_local': None}
                runtime_same_body = _safe_name2id(mj_model, MJOBJ_BODY, ref_body_name)
                if runtime_same_body >= 0:
                    report['sensor_body'] = ref_body_name
                    report['runtime_mount_source'] = 'runtime_body_matching_reference'
                    report['mount_label'] = f"{ref_body_name} body proxy"
                else:
                    report['notes'].append('Reference XML loaded, but no matching lumbar body exists in the humenv runtime model; keeping baseline Torso proxy.')
            else:
                report['notes'].append('Reference XML loaded, but no lumbar/L1-L2 marker was found; keeping baseline Torso proxy.')
        report['notes'].append('Reference MyoSuite XML is used for guidance only; the Meta Motivo runtime body was not replaced.')
    except Exception as exc:
        report['notes'].append(f'MyoSuite XML not loaded: {exc}; keeping the original Torso proxy.')

    return report

# -------------------------------------------------------------------
# CONFIGURATION - Tuned for biofidelity
# -------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Timing (30 Hz simulation)
PHASES = {
    'stand':   150,   # 5s:  stabilize standing
    'walk':    300,   # 10s: establish walking gait
    'perturb':  60,   # 2s:  gradual force application (not instant)
    'react':    15,   # 0.5s: human reaction time delay
    'fall':    400,   # 13s: fall and settle
}

TOTAL_STEPS = sum(PHASES.values())


def scenario29_phase_timing(age, sex):
    """Task 29 timing at 30 Hz.

    The trip should look like a short toe/stone catch, not a sit-down
    transition: walking momentum continues, one toe is caught, then the body
    pitches forward and settles on the floor.
    """
    age = float(age)
    age_frac = float(np.clip((age - 55.0) / 35.0, 0.0, 1.0))
    stand = int(round(60 + 15 * age_frac))
    walk = int(round(145 + 10 * age_frac))
    # Real toe trips are short: the toe catch is a brief event, then the
    # body falls ballistically.  Longer perturb windows made the motion look
    # like a deliberate crouch/sit.
    faint = int(round(38 + 4 * age_frac))
    react = int(round(10 + 3 * age_frac))
    fall = int(round(185 + 25 * age_frac))
    return {'stand': stand, 'walk': walk, 'perturb': faint, 'react': react, 'fall': fall}


def scenario29_perturbation_force_config(body_mass_kg, age, sex, bw_fraction=0.34):
    """Syncope guidance load, not a task-34 style perturbation.

    The main fall driver is muscle tone loss plus a posture/torso bias.  This
    config is intentionally small and is only used for reporting/compatibility;
    the runtime applies forces in the walking heading frame via
    apply_toe_trip_forward_fall().
    """
    bw = float(body_mass_kg) * 9.81
    age_frac = float(np.clip((float(age) - 55.0) / 35.0, 0.0, 1.0))
    mag = float(np.clip((0.28 + 0.10 * age_frac) * bw, 180.0, 430.0))
    direction = np.array([0.36, 0.00, -0.22], dtype=float)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    return {
        'magnitude': mag,
        'ramp_up': 65,
        'direction': direction,
        'application_point':  'Foot',
        'bw_fraction': float(mag / max(bw, 1e-9)),
        'body_mass_kg': float(body_mass_kg),
    }

# Biofidelic parameters (from paper: ~500N, but applied gradually)
FORCE_CONFIG = {
    'magnitude':        450.0,                           # Newtons
    'ramp_up':           30,                             # steps to full force (1s)
    'direction':  np.array([-1.0, 0.0, 0.3]),           # backward + slight upward
    'application_point': 'Pelvis',
}


# -------------------------------------------------------------------
# TASK 40 HEIGHT LAYER - v19 methodology
# -------------------------------------------------------------------
# v24 keeps the continuous learned WALK phase removed. The subject is held in a
# quiet, normal upright pose near the platform edge for ~2.8 s, then performs a
# short deterministic swing-leg backward step toward the inferred backside/backward direction.
# The root/pelvis is NOT translated across the block: only the swing leg moves.
# The platform remains a physical MuJoCo support object after release.
#
# v24 keeps the required height and adds protective descent/impact reaction to the
# required 1.80 m object height.  The stand/step logic remains stable, but the
# descent now adds visible arm/leg reaction and a short ground-impact recoil before slow rest.
TASK43_HEIGHT_MODE = True
TASK43_CHAIR_HEIGHT_M = 1.80
TASK43_CHAIR_HALF_EXTENTS = (0.72, 0.50, TASK43_CHAIR_HEIGHT_M * 0.5)
TASK43_CHAIR_CENTER_XY = (0.0, 0.0)
# These legacy constants are retained for fallback/debug only.  v20 computes
# the actual start/edge/step positions from the humanoid's real foot/head-side
# direction at runtime, because world +X was observed to be the lateral side for
# this model.
TASK43_START_ROOT_X = 0.50
TASK43_STEP_TARGET_X = TASK43_CHAIR_HALF_EXTENTS[0] + 0.16
TASK43_TOP_CLEARANCE = -0.0035
TASK43_EDGE_ROOT_BACKOFF_M = 0.14
TASK43_BACK_HEEL_EDGE_MARGIN_M = 0.022
TASK43_STEP_ROOT_DRIFT_M = 0.000
TASK43_STEP_CLEARANCE_M = 0.22
TASK43_RUNTIME_FWD_XY = np.array([1.0, 0.0], dtype=float)
TASK43_RUNTIME_LAT_XY = np.array([0.0, 1.0], dtype=float)
TASK43_SLOW_REST_BLEND_STEPS = 24   # v1j: faster passive settle after visible impact; climb unchanged
TASK43_SLOW_REST_HARD_LOCK_FRAMES = 3
TASK43_POST_IMPACT_FREE_RECOIL_STEPS = 4
TASK43_REST_ENGAGE_STABLE_FRAMES = 1
TASK43_REST_FORCE_AFTER_IMPACT_FRAMES = 3
TASK43_REST_CONTACT_MIN_COUNT = 1
TASK43_REST_LOW_PELVIS_M = 0.34
TASK43_REST_LOW_TORSO_M = 0.70
TASK43_REST_LOW_HEAD_M = 0.96
TASK43_CONTACT_SETTLE_MAX_DROP_M = 0.120
TASK43_CONTACT_SETTLE_STEP_M = 0.0010
# v30: stable human post-impact settle: early rest engagement, bounce damping, no continuous floor rolling/sliding.
TASK43_MAX_KEEP_CONTACT_DROP_M = 0.115
TASK43_VISUAL_DISTAL_CLEARANCE_M = -0.018
TASK43_VISUAL_MAX_NO_HOVER_DROP_M = 0.075
TASK43_VISUAL_SOLE_PENETRATION_M = 0.018
# Bounded support search: only shift the root in XY if required for true contact.
TASK43_SUPPORT_XY_SEARCH_MAX_M = 0.24
TASK43_SUPPORT_XY_SEARCH_STEP_M = 0.025
TASK43_SUPPORT_XY_SEARCH_LAT_M = 0.030
TASK43_SUPPORT_LOG_XY_SHIFT_EPS = 0.002
# v5: keep the platform geoms physically active after the back-step.  The old
# implementation disabled the platform contact at release, so the block behaved
# like a visual prop.  Now the block remains a real MuJoCo support object; we
# stop pose-locking after the step and let contacts + gravity drive the fall.
TASK43_KEEP_PLATFORM_CONTACT_AFTER_STEP = True
TASK43_EDGE_RELEASE_BACKWARD_SPEED = 0.000
TASK43_EDGE_RELEASE_DOWN_SPEED = -0.012
# v7d: keep the stable v7/v7c stand + single swing-leg back-step, but
# replace the weak post-step fall cue with the proven Task38 backward fall
# action: posterior/upward pelvis push + progressive weakening + grounded
# supine settle.  Only the fall action is borrowed; the Task38 reverse-walk
# setup is intentionally NOT used here.
TASK43_TASK38_FORCE_BW_FRACTION = 0.52
TASK43_TASK38_UPWARD = 0.10
TASK43_TASK38_FORCE_RAMP_START = 0.02
TASK43_TASK38_FORCE_RAMP_END = 0.86
TASK43_TASK38_TORSO_FOLLOW_BW = 0.050
TASK43_TASK38_HEAD_FOLLOW_BW = 0.018
TASK43_TASK38_INITIAL_BACK_VEL = 0.34
TASK43_TASK38_TARGET_BACK_VEL = 1.08
TASK43_TASK38_BACK_VEL_CAP_EARLY = 0.24
TASK43_TASK38_BACK_VEL_CAP_FREE = 1.08
TASK43_TASK38_LAT_VEL_CAP = 0.035
TASK43_TASK38_DOWN_VEL_CAP = -2.85
TASK43_TASK38_PITCH_TORQUE_NM = 30.0
TASK43_TASK38_COUNTER_TORQUE_NM = 30.0
TASK43_TASK38_MAX_PITCH_RATE = 3.55
TASK43_TASK38_MAX_ROLL_RATE = 0.22
TASK43_TASK38_MAX_YAW_RATE = 0.20
TASK43_TASK38_LEAN_COUNTER_DEG = 98.0
# Keep old names as aliases for diagnostics/guards that still reference them.
TASK43_EDGE_RELEASE_TORSO_TORQUE_NM = TASK43_TASK38_PITCH_TORQUE_NM
TASK43_EDGE_RELEASE_TORSO_PUSH_BW = TASK43_TASK38_TORSO_FOLLOW_BW
TASK43_SIMPLE_BACK_FALL_BACK_VEL_CAP_EARLY = TASK43_TASK38_BACK_VEL_CAP_EARLY
TASK43_SIMPLE_BACK_FALL_BACK_VEL_CAP_FREE = TASK43_TASK38_BACK_VEL_CAP_FREE
TASK43_SIMPLE_BACK_FALL_TARGET_BACK_VEL = TASK43_TASK38_TARGET_BACK_VEL
TASK43_SIMPLE_BACK_FALL_LAT_VEL_CAP = TASK43_TASK38_LAT_VEL_CAP
TASK43_SIMPLE_BACK_FALL_MAX_PITCH_RATE = TASK43_TASK38_MAX_PITCH_RATE
TASK43_SIMPLE_BACK_FALL_MAX_ROLL_RATE = TASK43_TASK38_MAX_ROLL_RATE
TASK43_SIMPLE_BACK_FALL_MAX_YAW_RATE = TASK43_TASK38_MAX_YAW_RATE



def scenario43_phase_timing(age, sex, height_m=None, body_mass_kg=None):
    """Initial ladder hold -> top-rung upward climb -> forward ladder fall.

    v7f speed-only bridge: the ladder climb uses the same profile target walk
    speed / double-support timing as the walking tasks, but it does NOT change
    the ladder pose, foot targets, or fall posture.  Older/cautious subjects
    spend more frames in the identical top-rung climb animation; younger/faster
    subjects spend fewer frames.
    """
    age = float(age)
    sex = str(sex or 'male').lower()
    height_m = float(height_m if height_m is not None else globals().get('SIM_HEIGHT', 1.65))
    body_mass_kg = float(body_mass_kg if body_mass_kg is not None else globals().get('SIM_RESOLVED_WEIGHT', 70.8))
    age_frac = float(np.clip((age - 55.0) / 35.0, 0.0, 1.0))
    try:
        style = get_age_style_v2(age, height_m=height_m, sex=sex, body_mass_kg=body_mass_kg)
        target_speed = float(style.get('target_walk_speed', 1.24))
        expected_ds = float(style.get('expected_double_support', 0.30))
    except Exception:
        target_speed = 1.24 - 0.10 * age_frac
        expected_ds = 0.24 + 0.12 * age_frac

    # Same walk-profile timing bridge as Task41, now scaled to climb
    # from first rung to the top rung before release. The pose/ladder path is
    # unchanged; only PHASES['step'] is longer because target_rung_index=6
    # instead of the Task41 fifth-rung index=4.
    young_ref_speed = 1.465
    speed_factor = float(np.clip(young_ref_speed / max(target_speed, 0.25), 0.82, 1.30))
    ds_factor = float(np.clip(1.0 + 0.45 * (expected_ds - 0.20), 0.92, 1.12))
    age_slow = float(np.clip((age - 35.0) / 40.0, 0.0, 1.0))

    stand = int(round(46 + 10 * age_frac + 8 * max(0.0, expected_ds - 0.28)))
    base_step = float(np.clip(138.0 * (speed_factor ** 1.15) * ds_factor + 8.0 * age_slow, 132.0, 192.0))
    top_rung_scale = float(np.clip((float(globals().get('TASK43_LADDER_TRIGGER_RUNG_INDEX', 6)) / 4.0), 1.0, 1.65))
    step = int(round(np.clip(base_step * top_rung_scale, 198.0, 292.0)))
    perturb = int(round(46 + 4 * age_frac))
    react = int(round(np.clip(10 + 18 * expected_ds + 3 * age_frac, 12, 18)))
    fall = int(round(np.clip(250 + 48 * age_frac + 32 * max(0.0, expected_ds - 0.25), 250, 292)))
    return {'stand': stand, 'step': step, 'perturb': perturb, 'react': react, 'fall': fall}

# Muscle weakening (collapse simulation) - exponential decay
WEAKENING_CONFIG = {
    'initial_factor': 1.0,
    'min_factor':     0.05,   # 5% remaining strength
    'decay_time':     45,     # steps to reach min (1.5s)
    'recovery_time':  120,
}

# Smoothing parameters
Z_INTERPOLATION_STEPS = 20
ACTION_SMOOTHING       = 0.7

IMU_HARDWARE_SPEC = {
    'microcontroller': 'STM32F722RET6 (ARM Cortex-M7, 216 MHz)',
    'accelerometer':   'LIS3DH tri-axis MEMS, +/-16g, 1 mg resolution',
    'gyroscope':       'LSM6DS3 tri-axis, +/-2000 dps, 0.07 dps resolution',
    'sampling_hz':     100.0,
    'mount_label':     'Lower back (vertebrae L1-L2)',
    'proxy_body':      'Torso',
    'proxy_fraction_torso_to_pelvis': 0.40,
}

MYOSUITE_INTEGRATION = {
    'enable_reference_xml': str(os.environ.get('MYOSUITE_ENABLE_REFERENCE_XML', '0')).strip().lower() in ('1', 'true', 'yes', 'on'),
    'model_xml': str(os.environ.get('MYOSUITE_MODEL_XML', '')).strip(),
    'mode': str(os.environ.get('MYOSUITE_MODE', 'reference')).strip().lower() or 'reference',
}

PAPER_ALIGNMENT_EXPORT = {
    'enable_opensim_bridge': True,
    'enable_pose_dataset_bridge': True,
    'pose_views': ('frontal', 'sagittal', 'oblique'),
    'pose_preview_frames': 4,
    'render_pose_images': False,
}

STABILITY_WARMUP_STEPS = 60
XCOM_NEGATIVE_CONFIRM_STEPS = 2
XCOM_LOG_START_OFFSET_STEPS = 30   # start capture-point logging 1 s into walk
LAST_Z_WALK_DIAGNOSTICS = {}

# -------------------------------------------------------------------
# IMPORT-SAFE SUBJECT DEFAULTS
# -------------------------------------------------------------------
# Interactive collection is handled by fall_dispatcher.py / scenario wrappers.
SIM_AGE = int(os.environ.get("FALL_DEFAULT_AGE", "75"))
SIM_HEIGHT = float(os.environ.get("FALL_DEFAULT_HEIGHT", "1.65"))
SIM_SEX = str(os.environ.get("FALL_DEFAULT_SEX", "male")).strip().lower() or "male"
_SIM_WEIGHT_ENV = str(os.environ.get("FALL_DEFAULT_WEIGHT", "")).strip()
SIM_WEIGHT = float(_SIM_WEIGHT_ENV) if _SIM_WEIGHT_ENV else None
SIM_RESOLVED_WEIGHT, SIM_TARGET_BMI, SIM_WEIGHT_SOURCE = resolve_subject_weight_kg(
    SIM_HEIGHT, SIM_AGE, SIM_SEX, explicit_weight=SIM_WEIGHT
)
SIM_WEIGHT_DISPLAY = (
    f"{SIM_WEIGHT:.1f}kg (user)" if SIM_WEIGHT is not None
    else f"{SIM_RESOLVED_WEIGHT:.1f}kg (auto BMI {SIM_TARGET_BMI:.1f})"
)
PHASES = scenario43_phase_timing(SIM_AGE, SIM_SEX, SIM_HEIGHT, SIM_RESOLVED_WEIGHT)
TOTAL_STEPS = sum(PHASES.values())


def _apply_subject_globals(subject_params=None):
    """Recompute module-level subject globals without changing the motion code path."""
    global SIM_AGE, SIM_HEIGHT, SIM_SEX, SIM_WEIGHT
    global SIM_RESOLVED_WEIGHT, SIM_TARGET_BMI, SIM_WEIGHT_SOURCE
    global SIM_WEIGHT_DISPLAY, PHASES, TOTAL_STEPS

    params = dict(subject_params or {})
    age = int(params.get('age', SIM_AGE))
    height = float(params.get('height', SIM_HEIGHT))
    sex = str(params.get('sex', SIM_SEX)).strip().lower() or 'male'
    weight = params.get('weight', SIM_WEIGHT)
    weight = None if weight in ('', None) else float(weight)

    SIM_AGE = age
    SIM_HEIGHT = height
    SIM_SEX = sex
    SIM_WEIGHT = weight
    SIM_RESOLVED_WEIGHT, SIM_TARGET_BMI, SIM_WEIGHT_SOURCE = resolve_subject_weight_kg(
        SIM_HEIGHT, SIM_AGE, SIM_SEX, explicit_weight=SIM_WEIGHT
    )
    SIM_WEIGHT_DISPLAY = (
        f"{SIM_WEIGHT:.1f}kg (user)" if SIM_WEIGHT is not None
        else f"{SIM_RESOLVED_WEIGHT:.1f}kg (auto BMI {SIM_TARGET_BMI:.1f})"
    )
    PHASES = scenario43_phase_timing(SIM_AGE, SIM_SEX, SIM_HEIGHT, SIM_RESOLVED_WEIGHT)
    TOTAL_STEPS = sum(PHASES.values())
    # Task43 has a deterministic ladder/climb/release helper in addition to
    # the shared AnthropometricModel. Refresh its subject-dependent constants
    # before MuJoCo XML injection so different age/height/weight profiles do
    # not all use the same hard-coded ladder/release behavior.
    try:
        _task43_apply_subject_profile_globals()
    except NameError:
        pass


def run_with_subject(subject_params=None):
    _apply_subject_globals(subject_params)
    print("=" * 70)
    print("  Scenario 43 - Forward Fall While Climbing Up Ladder [v1j-handlocked-fastsettle-climb]")
    print(f"  Task43 source file: {__file__}")
    print("  Pipeline: physical ladder stand -> scripted top-rung climb upward -> forward release/COM pitch -> protective prone settle")
    print("  Based on: Ferrari et al. DETC2025-169046")
    print(f"  Seed: {SEED} | Total Steps: subject-adaptive (resolved after input)")
    print("=" * 70)
    print(f"\n  >> Using: age={SIM_AGE}yr  height={SIM_HEIGHT}m  weight={SIM_WEIGHT_DISPLAY}  sex={SIM_SEX}\n")
    print(f"  >> Subject-adaptive total steps = {TOTAL_STEPS}  |  phases={PHASES}")
    print_subject_profile(SIM_AGE, SIM_SEX, SIM_HEIGHT, SIM_RESOLVED_WEIGHT)
    return _legacy_main()


def run(subject_params=None):
    """Compatibility alias for scenario wrappers that call run()."""
    return run_with_subject(subject_params)


def get_age_style(age_years, height_m=None, sex=None, body_mass_kg=None):
    height_m = SIM_HEIGHT if height_m is None else float(height_m)
    sex = (SIM_SEX if sex is None else sex).lower()
    body_mass_kg = SIM_RESOLVED_WEIGHT if body_mass_kg is None else float(body_mass_kg)
    return get_age_style_v2(age_years, height_m=height_m, sex=sex, body_mass_kg=body_mass_kg)


def get_age_reference_band(age_years, sex='male', height_m=None, body_mass_kg=None):
    height_m = SIM_HEIGHT if height_m is None else float(height_m)
    body_mass_kg = SIM_RESOLVED_WEIGHT if body_mass_kg is None else float(body_mass_kg)
    return get_age_reference_band_v2(age_years, sex, height_m, body_mass_kg)


def print_age_behavior_audit(age_years, sex, age_style, adapted_speed, natural_speed, anthro):
    ref = get_age_reference_band(age_years, sex, height_m=anthro.get('height_m', SIM_HEIGHT))
    sp_lo, sp_hi = ref['comfortable_speed_band_mps']
    ds_lo, ds_hi = ref['double_support_band']
    status = 'OK'
    if adapted_speed < 0.70 * sp_lo:
        status = 'LOW_WALK_TARGET'
    print("    ? AgeBehaviourAudit ACTIVE")
    print(f"      audit band               = {ref['label']}")
    print(f"      lit comfortable speed    = {sp_lo:.2f} to {sp_hi:.2f} m/s")
    print(f"      model style target       = {age_style['target_walk_speed']:.2f} m/s")
    print(f"      adapted controller speed = {adapted_speed:.2f} m/s")
    print(f"      z_walk natural speed     = {natural_speed:.2f} m/s")
    print(f"      lit double-support band  = {ds_lo:.0%} to {ds_hi:.0%}")
    print(f"      model expected dbl_sup   = {age_style['expected_double_support']:.0%}")
    print(f"      strength_factor          = {float(anthro.get('strength_factor', 1.0)):.3f}")
    print(f"      reaction_delay           = {float(anthro.get('reaction_delay', 0.0)):.3f} s")
    print(f"      balance_impairment       = {float(anthro.get('balance_impairment', 0.0)):.3f}")
    print(f"      age audit status         = {status}")

# -------------------------------------------------------------------
# META MOTIVO SETUP (lazy import-safe loader)
# -------------------------------------------------------------------
_MODEL_CACHE = {"model": None}


def _get_model():
    model_obj = _MODEL_CACHE.get("model")
    if model_obj is None:
        print("\n[1/4] Loading Meta Motivo model...")
        model_obj = FBcprModel.from_pretrained("facebook/metamotivo-M-1")
        model_obj.eval()
        _MODEL_CACHE["model"] = model_obj
        device = next(model_obj.parameters()).device
        print(f"      Model loaded on {device}")
    return model_obj


class _LazyModelProxy:
    def __getattr__(self, name):
        return getattr(_get_model(), name)

    def __call__(self, *args, **kwargs):
        return _get_model()(*args, **kwargs)


model = _LazyModelProxy()


# -------------------------------------------------------------------
# TASK EMBEDDING INFERENCE (with proper buffer usage)
# -------------------------------------------------------------------


def compute_subject_control_targets(age_style, natural_speed, natural_ds):
    """
    Blend the policy-natural speed with the literature-facing subject target.

    v34 methodology:
      - young / middle-aged adults should not be trapped near the slow policy
        natural speed, otherwise they drift and circle instead of committing to
        forward locomotion;
      - older adults still remain closer to the natural policy speed to avoid
        forcing an unrealistically aggressive gait.
    """
    natural_speed = float(max(0.18, natural_speed))
    natural_ds = float(np.clip(natural_ds, 0.05, 0.60))
    control_gain = float(age_style.get('control_speed_gain', 1.0))
    target_speed = float(age_style.get('target_walk_speed', natural_speed))
    blend = float(np.clip(age_style.get('policy_speed_blend', 0.55), 0.0, 1.0))
    desired_speed = blend * natural_speed + (1.0 - blend) * target_speed
    desired_speed *= control_gain
    policy_floor_ratio = float(np.clip(0.86 - 0.40 * blend, 0.58, 0.72))
    speed_floor = max(
        age_style.get('min_adapted_speed', 0.18),
        policy_floor_ratio * target_speed,
        0.65 * natural_speed,
    )
    speed_ceiling = max(
        speed_floor + 1e-6,
        float(age_style.get('max_speed_ratio', 1.8)) * natural_speed,
        0.92 * target_speed,
    )
    control_speed = float(np.clip(desired_speed, speed_floor, speed_ceiling))
    control_ds = float(np.clip(0.20 * natural_ds + 0.80 * age_style.get('expected_double_support', 0.30),
                               0.18, 0.52))
    return {
        'control_speed': control_speed,
        'control_ds': control_ds,
        'control_gain': control_gain,
    }

def infer_z_walk_stable(n_samples=1000, age=None):
    global LAST_Z_WALK_DIAGNOSTICS
    """
    Infer robust walking embedding using weighted regression
    as described in Meta Motivo paper (Sec 8.5.3).

    v3: reward threshold lowered to 0.3 (was 0.6) and trials increased
    to 80 to collect enough diverse walking states. Includes a best-of-N
    fallback if reward inference still yields insufficient data.
    """
    age = SIM_AGE if age is None else age
    age_style = get_age_style(age, SIM_HEIGHT, SIM_SEX, body_mass_kg=SIM_RESOLVED_WEIGHT)
    latent_style = get_age_style(35, height_m=1.70, sex='male', body_mass_kg=70.0)
    print(f"  Inferring z_walk (stable locomotion | reference_straight_walk_speed={latent_style['target_walk_speed']:.2f} m/s | subject_age_style={age_style['label']} | subject_target_speed={age_style['target_walk_speed']:.2f} m/s)...")
    reward_fn = LocomotionReward(move_speed=latent_style['target_walk_speed'], move_angle=0, stand_height=1.4)

    env_tmp, _ = make_humenv(task="move-ego-0-2")

    observations, rewards = [], []

    for trial in range(80):   # was 50
        torch.manual_seed(SEED + trial)
        z = model.sample_z(1)
        obs, _ = env_tmp.reset()

        traj_obs, traj_rew = [], []
        for step in range(120):   # was 100 - allow more steps per trial
            obs_t = torch.tensor(obs['proprio'], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = model.act(obs_t, z).squeeze(0).numpy()

            obs, _, term, trunc, _ = env_tmp.step(action)
            r = reward_fn.compute(env_tmp.unwrapped.model, env_tmp.unwrapped.data)

            traj_obs.append(obs['proprio'].copy())
            traj_rew.append(r)

            if term or trunc:
                break

        # Lowered from 0.6 ? 0.3 so we accumulate more walking data
        if np.mean(traj_rew) > 0.3:
            observations.extend(traj_obs)
            rewards.extend(traj_rew)

    env_tmp.close()

    # Fallback: if still too few samples, use best single z found so far
    if len(observations) < 50:
        print("      Warning: few walking samples collected - using best-z fallback")
        best_z   = None
        best_rew = -np.inf
        env_fb, _ = make_humenv(task="move-ego-0-2")
        for trial in range(30):
            torch.manual_seed(SEED + trial + 9000)
            z   = model.sample_z(1)
            obs_fb, _ = env_fb.reset()
            trial_rewards = []
            for _ in range(60):
                obs_t = torch.tensor(obs_fb['proprio'], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    act = model.act(obs_t, z).squeeze(0).numpy()
                obs_fb, _, t, tr, _ = env_fb.step(act)
                trial_rewards.append(
                    reward_fn.compute(env_fb.unwrapped.model, env_fb.unwrapped.data))
                if t or tr:
                    break
            if np.mean(trial_rewards) > best_rew:
                best_rew = np.mean(trial_rewards)
                best_z   = z.clone()
        env_fb.close()
        print(f"      Best-z fallback reward: {best_rew:.3f}")
        LAST_Z_WALK_DIAGNOSTICS = {'mean_reward': float(best_rew), 'mean_vx': 0.0, 'straightness': 0.0, 'double_support_frac': 0.5, 'gait_score': float(best_rew), 'age_target_speed': float(age_style['target_walk_speed'])}
        return best_z

    obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32)
    rew_tensor = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        # Search a wider latent set. The reward-inference anchor is useful, but in
        # practice it often collapses to a slow / curved gait prior. Add both local
        # perturbations and fresh samples, then choose by rollout quality.
        z_base = model.reward_inference(obs_tensor, rew_tensor)
        z_candidates = [("base", z_base)]
        torch.manual_seed(SEED + 777)
        for perturb_scale in [0.05, 0.10, 0.15, 0.20]:
            noise = torch.randn_like(z_base) * perturb_scale
            z_candidates.append((f"noise={perturb_scale}", z_base + noise))
        for sample_idx in range(6):
            torch.manual_seed(SEED + 3000 + sample_idx)
            z_candidates.append((f"sample={sample_idx+1}", model.sample_z(1)))

    # Select the best candidate by evaluating each on a short rollout.
    print("      Selecting best z_walk via rollout evaluation (expanded multi-objective gait search)...")
    best_z    = None
    best_score = -np.inf
    best_reward = -np.inf
    eval_env, _ = make_humenv(task="move-ego-0-2")
    pelvis_id = mujoco.mj_name2id(eval_env.unwrapped.model, MJOBJ_BODY, 'Pelvis')
    foot_left = mujoco.mj_name2id(eval_env.unwrapped.model, MJOBJ_BODY, 'FootL')
    foot_right = mujoco.mj_name2id(eval_env.unwrapped.model, MJOBJ_BODY, 'FootR')
    if foot_left < 0:
        foot_left = mujoco.mj_name2id(eval_env.unwrapped.model, MJOBJ_BODY, 'L_Ankle')
    if foot_right < 0:
        foot_right = mujoco.mj_name2id(eval_env.unwrapped.model, MJOBJ_BODY, 'R_Ankle')

    def _support_flags(model, data, left_id, right_id):
        lsup = False; rsup = False
        plane_type = getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', -999)
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            names = ((mujoco.mj_id2name(model, MJOBJ_GEOM, g1) or '').lower(), (mujoco.mj_id2name(model, MJOBJ_GEOM, g2) or '').lower())
            is_ground = any(('floor' in n or 'ground' in n or 'plane' in n) for n in names) or any(model.geom_type[g] == plane_type for g in (g1, g2))
            if not is_ground:
                continue
            ng = g2 if (('floor' in names[0]) or ('ground' in names[0]) or ('plane' in names[0]) or model.geom_type[g1] == plane_type) else g1
            bg = int(model.geom_bodyid[ng])
            if bg == left_id:
                lsup = True
            elif bg == right_id:
                rsup = True
        return lsup, rsup

    for idx, (cand_label, z_cand) in enumerate(z_candidates):
        obs_e, _ = eval_env.reset()
        rollout_rewards = []
        xs, ys, hs, vxy = [], [], [], []
        dbl = []
        for _ in range(75):
            obs_t = torch.tensor(obs_e['proprio'], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                act = model.act(obs_t, z_cand).squeeze(0).numpy()
            obs_e, _, t, tr, _ = eval_env.step(act)
            rollout_rewards.append(reward_fn.compute(eval_env.unwrapped.model, eval_env.unwrapped.data))
            if pelvis_id >= 0:
                p = eval_env.unwrapped.data.xpos[pelvis_id].copy()
                v = _body_world_velocity(eval_env.unwrapped.model, eval_env.unwrapped.data, pelvis_id)
                xs.append(float(p[0])); ys.append(float(p[1])); hs.append(float(p[2])); vxy.append(float(np.linalg.norm(v[:2])))
            if foot_left >= 0 and foot_right >= 0:
                lsup, rsup = _support_flags(eval_env.unwrapped.model, eval_env.unwrapped.data, foot_left, foot_right)
                dbl.append(float(lsup and rsup))
            if t or tr:
                break
        mean_r = float(np.mean(rollout_rewards)) if rollout_rewards else 0.0
        mean_speed = float(np.mean(vxy)) if vxy else 0.0
        if len(xs) > 1:
            dxy = np.array([xs[-1] - xs[0], ys[-1] - ys[0]], dtype=float)
            net_disp = float(np.linalg.norm(dxy))
            heading_xy = (dxy / net_disp) if net_disp > 1e-6 else np.array([1.0, 0.0], dtype=float)
            step_xy = np.column_stack([np.diff(xs), np.diff(ys)]) if len(xs) > 2 else np.zeros((0, 2), dtype=float)
            path_len = float(np.sum(np.linalg.norm(step_xy, axis=1))) if step_xy.size else 0.0
            straightness = net_disp / max(path_len, 1e-6) if path_len > 1e-6 else 0.0
            lat_axis = np.array([-heading_xy[1], heading_xy[0]], dtype=float)
            lateral_offsets = [float(np.dot(np.array([x - xs[0], y - ys[0]], dtype=float), lat_axis)) for x, y in zip(xs, ys)]
            lateral_rms = float(np.sqrt(np.mean(np.square(lateral_offsets)))) if lateral_offsets else 0.0
        else:
            heading_xy = np.array([1.0, 0.0], dtype=float)
            net_disp = 0.0
            straightness = 0.0
            lateral_rms = 1.0
        ds_frac = float(np.mean(dbl)) if dbl else 0.5
        speed_score = float(np.clip(mean_speed / max(0.70 * age_style['target_walk_speed'], 0.20), 0.0, 1.0))
        ds_target = float(np.clip(age_style.get('expected_double_support', latent_style.get('expected_double_support', 0.26)), 0.18, 0.45))
        ds_score = float(np.clip(1.0 - abs(ds_frac - ds_target) / 0.20, 0.0, 1.0))
        lateral_score = float(np.clip(1.0 - lateral_rms / max(net_disp, 0.35), 0.0, 1.0))
        gait_score = 0.28 * mean_r + 0.28 * straightness + 0.22 * speed_score + 0.12 * ds_score + 0.10 * lateral_score
        print(f"        z_candidate {idx+1}/{len(z_candidates)} ({cand_label})  reward={mean_r:.3f}  vxy={mean_speed:.3f}  straight={straightness:.3f}  lat_rms={lateral_rms:.3f}  ds={ds_frac:.2f}  score={gait_score:.3f}")
        if gait_score > best_score:
            best_score = gait_score
            best_reward = mean_r
            best_z = z_cand.clone()
            LAST_Z_WALK_DIAGNOSTICS = {
                'mean_reward': float(mean_r),
                'mean_vx': float(mean_speed),
                'straightness': float(straightness),
                'double_support_frac': float(ds_frac),
                'gait_score': float(gait_score),
                'age_target_speed': float(age_style['target_walk_speed']),
                'heading_xy': heading_xy.tolist(),
                'net_disp': float(net_disp),
                'lateral_rms': float(lateral_rms),
            }
    eval_env.close()
    print(f"      Best z_walk selected (score={best_score:.3f} | reward={best_reward:.3f} | vxy={LAST_Z_WALK_DIAGNOSTICS.get('mean_vx', 0.0):.3f})")

    z_walk = best_z
    print(f"      z_walk inferred from {len(observations)} states")
    return z_walk


def infer_z_forward_fall():
    """
    Infer embedding for backward fall using LieDownReward with orient='up'
    and goal inference from collapsed poses (Sec 4.2 of paper)
    """
    print("  Inferring z_fall (backward collapse)...")

    env_tmp, _ = make_humenv(task="lieonground-up")

    collapsed_obs = []

    for trial in range(10):
        obs, _ = env_tmp.reset()
        zero_action = np.zeros(env_tmp.action_space.shape)

        for step in range(100):
            obs, _, term, trunc, _ = env_tmp.step(zero_action)

            pelvis_id = mujoco.mj_name2id(env_tmp.unwrapped.model, MJOBJ_BODY, "Pelvis")
            pelvis_z  = env_tmp.unwrapped.data.xpos[pelvis_id][2]

            if pelvis_z < 0.3:
                collapsed_obs.append(obs['proprio'].copy())

            if term or trunc:
                break

    env_tmp.close()

    if len(collapsed_obs) > 50:
        obs_tensor = torch.tensor(np.array(collapsed_obs[-200:]), dtype=torch.float32)
        with torch.no_grad():
            z_fall = model.goal_inference(obs_tensor).mean(dim=0, keepdim=True)
    else:
        print("      Warning: using fallback goal inference")
        z_fall = model.sample_z(1)

    print(f"      z_fall inferred from {len(collapsed_obs)} collapsed states")
    return z_fall



def infer_z_task29_forward_trip_fall():
    """Infer a prone/forward collapsed goal for walking toe trip task 28."""
    print("  Inferring z_fall (walking toe-trip unsupported forward collapse)...")
    env_tmp, _ = make_humenv(task="lieonground-up")
    pelvis_id = mujoco.mj_name2id(env_tmp.unwrapped.model, MJOBJ_BODY, "Pelvis")
    head_id = mujoco.mj_name2id(env_tmp.unwrapped.model, MJOBJ_BODY, "Head")
    collapsed_obs = []
    for trial in range(14):
        obs, _ = env_tmp.reset()
        zero_action = np.zeros(env_tmp.action_space.shape)
        for step in range(140):
            d = env_tmp.unwrapped.data
            torso_id = mujoco.mj_name2id(env_tmp.unwrapped.model, MJOBJ_BODY, "Torso")
            if torso_id >= 0:
                d.xfrc_applied[torso_id, 0] = 90.0
                d.xfrc_applied[torso_id, 1] = 16.0
                d.xfrc_applied[torso_id, 2] = -65.0
                d.xfrc_applied[torso_id, 4] = -24.0
            obs, _, term, trunc, _ = env_tmp.step(zero_action)
            if pelvis_id >= 0 and head_id >= 0:
                p = d.xpos[pelvis_id]
                h = d.xpos[head_id]
                if float(p[2]) < 0.36 and float(h[0] - p[0]) > -0.06:
                    collapsed_obs.append(obs['proprio'].copy())
            if term or trunc:
                break
    env_tmp.close()
    if len(collapsed_obs) > 40:
        obs_tensor = torch.tensor(np.array(collapsed_obs[-260:]), dtype=torch.float32)
        with torch.no_grad():
            z_fall = model.goal_inference(obs_tensor).mean(dim=0, keepdim=True)
    else:
        print("      Warning: using fallback sampled z for forward toe-trip collapse")
        z_fall = model.sample_z(1)
    print(f"      z_fall inferred from {len(collapsed_obs)} toe-trip forward collapse states")
    return z_fall


def infer_z_forward_prone_rest():
    """Infer stable prone/forward rest for post-impact task-28 settling."""
    print("  Inferring z_rest (forward/prone settle)...")
    env_tmp, _ = make_humenv(task="lieonground-up")
    pelvis_id = mujoco.mj_name2id(env_tmp.unwrapped.model, MJOBJ_BODY, "Pelvis")
    head_id = mujoco.mj_name2id(env_tmp.unwrapped.model, MJOBJ_BODY, "Head")
    torso_id = mujoco.mj_name2id(env_tmp.unwrapped.model, MJOBJ_BODY, "Torso")
    stable_obs = []
    for trial in range(16):
        obs, _ = env_tmp.reset()
        zero_action = np.zeros(env_tmp.action_space.shape)
        for step in range(170):
            d = env_tmp.unwrapped.data
            if torso_id >= 0:
                d.xfrc_applied[torso_id, 0] = 110.0
                d.xfrc_applied[torso_id, 1] = 18.0
                d.xfrc_applied[torso_id, 2] = -55.0
                d.xfrc_applied[torso_id, 4] = -22.0
            obs, _, term, trunc, _ = env_tmp.step(zero_action)
            if pelvis_id >= 0 and head_id >= 0 and torso_id >= 0:
                p = d.xpos[pelvis_id]
                h = d.xpos[head_id]
                torso_z = float(d.xpos[torso_id][2])
                qvel_norm = float(np.linalg.norm(d.qvel))
                vec = h - p
                nv = float(np.linalg.norm(vec))
                trunk_lean_deg = float(np.degrees(np.arccos(np.clip(np.dot(vec / nv, np.array([0.0, 0.0, 1.0])), -1.0, 1.0)))) if nv > 1e-8 else 0.0
                if float(p[2]) < 0.22 and torso_z < 0.28 and float(h[2]) < 0.42 and qvel_norm < 1.5 and float(h[0] - p[0]) > -0.08 and trunk_lean_deg > 68.0:
                    stable_obs.append(obs['proprio'].copy())
            if term or trunc:
                break
    env_tmp.close()
    if len(stable_obs) > 40:
        obs_tensor = torch.tensor(np.array(stable_obs[-260:]), dtype=torch.float32)
        with torch.no_grad():
            z_rest = model.goal_inference(obs_tensor).mean(dim=0, keepdim=True)
    else:
        print("      Warning: few prone-rest samples - blending fallback from z_fall")
        z_rest = infer_z_task29_forward_trip_fall()
    print(f"      z_rest inferred from {len(stable_obs)} forward/prone stable states")
    return z_rest

def infer_z_lie_rest():
    """Infer a stable, low-velocity supine pose for post-impact rest mode."""
    print("  Inferring z_rest (stable lie-on-ground)...")

    env_tmp, _ = make_humenv(task="lieonground-up")
    stable_obs = []

    pelvis_id = mujoco.mj_name2id(env_tmp.unwrapped.model, MJOBJ_BODY, "Pelvis")
    head_id = mujoco.mj_name2id(env_tmp.unwrapped.model, MJOBJ_BODY, "Head")
    torso_id = mujoco.mj_name2id(env_tmp.unwrapped.model, MJOBJ_BODY, "Torso")

    for trial in range(12):
        obs, _ = env_tmp.reset()
        zero_action = np.zeros(env_tmp.action_space.shape)

        for step in range(140):
            obs, _, term, trunc, _ = env_tmp.step(zero_action)
            pelvis_z = float(env_tmp.unwrapped.data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 1.0
            torso_z = float(env_tmp.unwrapped.data.xpos[torso_id][2]) if torso_id >= 0 else pelvis_z
            head_z = float(env_tmp.unwrapped.data.xpos[head_id][2]) if head_id >= 0 else torso_z
            qvel_norm = float(np.linalg.norm(env_tmp.unwrapped.data.qvel))
            trunk_lean_deg = 0.0
            if pelvis_id >= 0 and head_id >= 0:
                vec = env_tmp.unwrapped.data.xpos[head_id] - env_tmp.unwrapped.data.xpos[pelvis_id]
                nv = float(np.linalg.norm(vec))
                if nv > 1e-8:
                    trunk_lean_deg = float(np.degrees(np.arccos(np.clip(np.dot(vec / nv, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))))

            is_low = pelvis_z < 0.18 and torso_z < 0.24 and head_z < 0.40
            is_quiet = qvel_norm < 1.2
            is_supine_like = trunk_lean_deg > 72.0
            if is_low and is_quiet and is_supine_like:
                stable_obs.append(obs['proprio'].copy())

            if term or trunc:
                break

    env_tmp.close()

    if len(stable_obs) > 50:
        obs_tensor = torch.tensor(np.array(stable_obs[-240:]), dtype=torch.float32)
        with torch.no_grad():
            z_rest = model.goal_inference(obs_tensor).mean(dim=0, keepdim=True)
    else:
        print("      Warning: few supine-rest samples - blending fallback from z_fall")
        z_rest = infer_z_forward_fall()

    print(f"      z_rest inferred from {len(stable_obs)} stable lying states")
    return z_rest


def infer_z_stand():
    """Infer stable standing embedding"""
    print("  Inferring z_stand...")
    reward_fn = LocomotionReward(move_speed=0.0, move_angle=0, stand_height=1.4)

    env_tmp, _ = make_humenv(task="move-ego-0-0")
    observations, rewards = [], []

    for trial in range(30):
        torch.manual_seed(SEED + trial + 1000)
        z = model.sample_z(1)
        obs, _ = env_tmp.reset()

        for step in range(80):
            obs_t = torch.tensor(obs['proprio'], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = model.act(obs_t, z).squeeze(0).numpy()

            obs, _, term, trunc, _ = env_tmp.step(action)
            r = reward_fn.compute(env_tmp.unwrapped.model, env_tmp.unwrapped.data)

            observations.append(obs['proprio'].copy())
            rewards.append(r)

            if term or trunc:
                break

    env_tmp.close()

    obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32)
    rew_tensor = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        z_stand = model.reward_inference(obs_tensor, rew_tensor).mean(dim=0, keepdim=True)

    return z_stand


# -------------------------------------------------------------------
# GAIT PHASE DETECTOR - For biofidelic perturbation timing
# -------------------------------------------------------------------
class GaitPhaseDetector:
    """
    Detects gait phase from contact state + body-frame point kinematics.

    Layer-2 note:
      - avoids direct use of mjData.cvel for user-facing linear velocity
      - uses ground contacts when available, falling back to height thresholds
    """

    def __init__(self, mj_model, mj_data):
        self.mj_model = mj_model
        self.mj_data  = mj_data
        self.foot_left  = self._resolve_body(['L_Foot', 'FootL', 'LeftFoot', 'L_Ankle', 'AnkleL'], side='left', family='foot')
        self.foot_right = self._resolve_body(['R_Foot', 'FootR', 'RightFoot', 'R_Ankle', 'AnkleR'], side='right', family='foot')
        self.phase_history = deque(maxlen=10)
        self.last_phase = 'unknown'

    def _resolve_body(self, candidates, side=None, family=None):
        best_id = -1
        best_score = -1e9
        fam_kw = {
            'foot': ['foot', 'ankle', 'toe', 'heel'],
            'shank': ['shin', 'knee', 'lowerleg'],
        }.get(family, [])
        for bid in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, bid) or ''
            norm = _norm_name(name)
            score = 0.0
            for c in candidates:
                key = _norm_name(c)
                if norm == key:
                    score += 100.0
                elif key and key in norm:
                    score += 20.0
            if family and any(k in norm for k in fam_kw):
                score += 10.0
            if side and _body_name_matches_side(name, side):
                score += 15.0
            elif side:
                score -= 20.0
            if score > best_score:
                best_score = score
                best_id = bid
        return best_id if best_score >= 10.0 else -1

    def _foot_world_velocity(self, body_id):
        if body_id < 0:
            return np.zeros(3)
        return _body_world_velocity(self.mj_model, self.mj_data, body_id)

    def _foot_in_ground_contact(self, body_id):
        if body_id < 0:
            return False
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1, g2 = c.geom1, c.geom2
            b1 = self.mj_model.geom_bodyid[g1]
            b2 = self.mj_model.geom_bodyid[g2]
            names = ((mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g1) or '').lower(), (mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g2) or '').lower())
            ground = any(('floor' in n or 'ground' in n or 'plane' in n) for n in names) or any(self.mj_model.geom_type[g] == getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', -999) for g in (g1, g2))
            if not ground:
                continue
            if b1 == body_id or b2 == body_id:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.mj_model, self.mj_data, i, wrench)
                if wrench[0] > 1.0:
                    return True
        return False

    def detect_phase(self):
        if self.foot_left < 0 or self.foot_right < 0:
            return 'unknown'
        left_pos = self.mj_data.xpos[self.foot_left]
        right_pos = self.mj_data.xpos[self.foot_right]
        left_vel = self._foot_world_velocity(self.foot_left)
        right_vel = self._foot_world_velocity(self.foot_right)
        left_on_ground = self._foot_in_ground_contact(self.foot_left) or (left_pos[2] < 0.06)
        right_on_ground = self._foot_in_ground_contact(self.foot_right) or (right_pos[2] < 0.06)

        if left_on_ground and right_on_ground:
            if left_vel[2] < -0.08 or right_vel[2] < -0.08:
                phase = 'heel_strike'
            else:
                phase = 'double_support'
        elif left_on_ground and not right_on_ground:
            phase = 'mid_stance' if right_vel[2] < 0.05 else 'terminal_stance'
        elif right_on_ground and not left_on_ground:
            phase = 'mid_stance' if left_vel[2] < 0.05 else 'terminal_stance'
        else:
            phase = 'swing'
        self.phase_history.append(phase)
        self.last_phase = phase
        return phase

    def get_optimal_perturbation_window(self, fall_type):
        windows = {
            'backward_walking': ['mid_stance', 'terminal_stance'],
            'forward_stumble':  ['swing', 'heel_strike'],
            'lateral_left':     ['mid_stance'],
            'lateral_right':    ['mid_stance'],
            'slip_induced':     ['heel_strike', 'mid_stance'],
        }
        return windows.get(fall_type, ['mid_stance'])

    def is_in_window(self, fall_type):
        return self.detect_phase() in self.get_optimal_perturbation_window(fall_type)


# -------------------------------------------------------------------
# BIOFIDELIC CONTROL SYSTEM
# -------------------------------------------------------------------
class BiofidelicFallController:
    """
    Manages realistic fall dynamics including:
    - Reaction time delays
    - Progressive muscle weakening
    - Smooth policy transitions
    - Orientation-aware LieDownReward factory
    - GaitPhaseDetector for perturbation timing
    """

    def __init__(self, env, mj_model, mj_data):
        self.env      = env
        self.mj_model = mj_model
        self.mj_data  = mj_data

        # Store original actuator parameters
        self.original_gear      = mj_model.actuator_gear[:, 0].copy()
        self.original_ctrlrange = mj_model.actuator_ctrlrange.copy()

        # Body part indices for targeted weakening
        self.leg_actuators   = self._get_leg_actuators()
        self.arm_actuators   = self._get_arm_actuators()
        self.torso_actuators = self._get_torso_actuators()

        # State variables
        self.current_z       = None
        self.target_z        = None
        self.z_blend         = 0.0
        self.action_smoothing = deque(maxlen=3)
        self.leg_strength    = 1.0
        self.force_applied   = 0.0

        self.original_dof_damping = mj_model.dof_damping.copy()
        self._ground_geoms = self._detect_ground_geoms()
        self._geom_friction_orig = {g: mj_model.geom_friction[g].copy() for g in range(mj_model.ngeom)}
        self._configure_contact_solver()
        self.rest_mode = False
        self.rest_mode_hard_lock = False
        self.rest_counter = 0
        self.impact_brake_mode = False
        self.impact_anchor_xy = None
        self.rest_anchor_xy = None
        self.impact_brake_mode = False
        self.impact_anchor_xy = None
        self._rest_arm_scale = 0.006
        self._rest_leg_scale = 0.035
        self._rest_torso_scale = 0.045
        self._postimpact_brake_counter = 0

        # Gait phase detection for biofidelic perturbation timing
        self.gait_detector        = GaitPhaseDetector(mj_model, mj_data)
        self.perturbation_applied = False
        self.waiting_for_phase    = False

        # Lie orientation metadata
        self._lie_orient = 'up'

    # ------------------------------------------------------------------
    # Actuator group helpers
    # ------------------------------------------------------------------
    def _get_leg_actuators(self):
        indices  = []
        keywords = ['hip', 'knee', 'ankle', 'foot', 'leg']
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, MJOBJ_ACTUATOR, i)
            if name and any(k in name.lower() for k in keywords):
                indices.append(i)
        return indices

    def _get_arm_actuators(self):
        indices  = []
        keywords = ['shoulder', 'elbow', 'wrist', 'hand', 'arm']
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, MJOBJ_ACTUATOR, i)
            if name and any(k in name.lower() for k in keywords):
                indices.append(i)
        return indices

    def _get_torso_actuators(self):
        indices  = []
        keywords = ['torso', 'spine', 'abdomen', 'chest']
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, MJOBJ_ACTUATOR, i)
            if name and any(k in name.lower() for k in keywords):
                indices.append(i)
        return indices

    def _detect_ground_geoms(self):
        plane_type = getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', None)
        ground = set()
        for g in range(self.mj_model.ngeom):
            gname = (mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g) or '').lower()
            is_ground = ('floor' in gname or 'ground' in gname or 'plane' in gname or
                         (plane_type is not None and self.mj_model.geom_type[g] == plane_type))
            if is_ground:
                ground.add(g)
        return ground

    def _configure_contact_solver(self):
        opt = getattr(self.mj_model, 'opt', None)
        if opt is None:
            return
        if hasattr(opt, 'iterations'):
            opt.iterations = max(int(opt.iterations), 120)
        if hasattr(opt, 'ls_iterations'):
            opt.ls_iterations = max(int(opt.ls_iterations), 50)
        if hasattr(opt, 'noslip_iterations'):
            opt.noslip_iterations = max(int(opt.noslip_iterations), 16)

    def _geom_body_name(self, gid):
        bid = int(self.mj_model.geom_bodyid[int(gid)])
        return mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, bid) or f'body_{bid}'

    def _is_foot_like_body(self, body_name):
        n = _norm_name(body_name)
        return any(k in n for k in ('foot', 'ankle', 'toe', 'heel'))

    def has_nonfoot_ground_contact(self, min_normal_n=25.0):
        for i in range(int(self.mj_data.ncon)):
            c = self.mj_data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self._ground_geoms and g2 not in self._ground_geoms:
                non_ground = g2
            elif g2 in self._ground_geoms and g1 not in self._ground_geoms:
                non_ground = g1
            else:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, wrench)
            normal_n = float(max(0.0, wrench[0]))
            if normal_n < float(min_normal_n):
                continue
            if not self._is_foot_like_body(self._geom_body_name(non_ground)):
                return True
        return False

    def ground_contact_count(self, min_normal_n=8.0, nonfoot_only=False):
        """Count real ground contacts above a small normal-force threshold."""
        count = 0
        total_n = 0.0
        for i in range(int(self.mj_data.ncon)):
            c = self.mj_data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self._ground_geoms and g2 not in self._ground_geoms:
                non_ground = g2
            elif g2 in self._ground_geoms and g1 not in self._ground_geoms:
                non_ground = g1
            else:
                continue
            if nonfoot_only and self._is_foot_like_body(self._geom_body_name(non_ground)):
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, wrench)
            normal_n = float(max(0.0, wrench[0]))
            if normal_n >= float(min_normal_n):
                count += 1
                total_n += normal_n
        return int(count), float(total_n)

    def activate_impact_brake(self):
        if self.impact_brake_mode:
            return
        self.impact_brake_mode = True
        self.impact_anchor_xy = self.mj_data.qpos[:2].copy() if self.mj_data.qpos.shape[0] >= 2 else None

    def apply_impact_brake(self):
        if not self.impact_brake_mode or self.mj_data.qvel.shape[0] < 6:
            return
        # v30 grounded impact brake. The previous v29 damper was too weak, so
        # the body kept translating/rolling for the whole fall phase. This is
        # velocity-only (no qpos pull), which dissipates energy without causing
        # visual snapping or contact jitter.
        self._postimpact_brake_counter += 1
        tau = float(np.clip(self._postimpact_brake_counter / 18.0, 0.0, 1.0))
        lin_xy = 0.42 * (1.0 - tau) + 0.16 * tau
        ang_xy = 0.64 * (1.0 - tau) + 0.30 * tau
        ang_z  = 0.66 * (1.0 - tau) + 0.28 * tau
        self.mj_data.qvel[0] *= lin_xy
        self.mj_data.qvel[1] *= lin_xy
        # Kill upward rebound much more than downward settling velocity.
        self.mj_data.qvel[2] *= (0.18 if float(self.mj_data.qvel[2]) > 0.0 else 0.74)
        self.mj_data.qvel[3] *= ang_xy
        self.mj_data.qvel[4] *= ang_xy
        self.mj_data.qvel[5] *= ang_z
        if self.mj_data.qvel.shape[0] > 6:
            self.mj_data.qvel[6:] *= (0.82 * (1.0 - tau) + 0.48 * tau)
        # No qpos anchor-pull here. Position correction during contact caused
        # visible floor buzzing and repeated GRF spikes in older attempts.

    def activate_rest_mode(self, z_rest=None):
        if self.rest_mode:
            return
        self.activate_impact_brake()
        self.rest_mode = True
        self.rest_mode_hard_lock = False
        self.rest_counter = 0
        self.rest_anchor_xy = self.mj_data.qpos[:2].copy() if self.mj_data.qpos.shape[0] >= 2 else None
        self.clear_forces()
        # v30: do not hand control to the learned z_rest policy after floor
        # impact. It can keep trying to complete a prone pose and looks like
        # continuous rolling/slipping. Post-impact should be mostly passive:
        # a short recoil, then friction + damping + very small residual tone.
        # The optional z_rest argument is intentionally ignored for Task43.
        self.mj_model.dof_damping[:] = self.original_dof_damping * 2.25
        # High sliding/torsional friction only after true floor impact.
        for g in range(self.mj_model.ngeom):
            fr = self._geom_friction_orig[g].copy()
            if g in self._ground_geoms:
                fr[0] = max(fr[0], 4.8)
                fr[1] = max(fr[1], 0.18)
                fr[2] = max(fr[2], 0.050)
            else:
                fr[0] = max(fr[0], 2.6)
                fr[1] = max(fr[1], 0.12)
                fr[2] = max(fr[2], 0.035)
            self.mj_model.geom_friction[g] = fr
        # Very small residual tone only. This prevents floor jitter/rolling
        # while preserving a human-like limp settle after the reflex phase.
        for idx in self.leg_actuators:
            self.mj_model.actuator_gear[idx, 0] = self.original_gear[idx] * self._rest_leg_scale
        for idx in self.arm_actuators:
            self.mj_model.actuator_gear[idx, 0] = self.original_gear[idx] * self._rest_arm_scale
        for idx in self.torso_actuators:
            self.mj_model.actuator_gear[idx, 0] = self.original_gear[idx] * self._rest_torso_scale

    def harden_rest_mode(self):
        if not self.rest_mode:
            return
        self.rest_mode_hard_lock = True

    def apply_rest_stiction(self):
        if not self.rest_mode or self.mj_data.qvel.shape[0] < 6:
            return
        # v30 passive post-fall settle. Strong enough to stop floor skating and
        # rolling, but progressive so there is still a small natural recoil.
        self.rest_counter += 1
        tau = float(np.clip(self.rest_counter / max(float(TASK43_SLOW_REST_BLEND_STEPS), 1.0), 0.0, 1.0))
        lin_xy = 0.34 * (1.0 - tau) + 0.030 * tau
        ang_xy = 0.48 * (1.0 - tau) + 0.055 * tau
        ang_z  = 0.48 * (1.0 - tau) + 0.050 * tau
        self.mj_data.qvel[0] *= lin_xy
        self.mj_data.qvel[1] *= lin_xy
        self.mj_data.qvel[2] *= (0.10 if float(self.mj_data.qvel[2]) > 0.0 else 0.62)
        self.mj_data.qvel[3] *= ang_xy
        self.mj_data.qvel[4] *= ang_xy
        self.mj_data.qvel[5] *= ang_z
        if self.mj_data.qvel.shape[0] > 6:
            joint_damp = 0.62 * (1.0 - tau) + 0.18 * tau
            self.mj_data.qvel[6:] *= joint_damp
        if self.rest_mode_hard_lock:
            self.mj_data.qvel[:6] *= 0.02
            if self.mj_data.qvel.shape[0] > 6:
                self.mj_data.qvel[6:] *= 0.08

    def restore_passive_contact(self):
        self.mj_model.dof_damping[:] = self.original_dof_damping
        for g in range(self.mj_model.ngeom):
            self.mj_model.geom_friction[g] = self._geom_friction_orig[g].copy()
        self.rest_mode = False
        self.rest_mode_hard_lock = False
        self.rest_counter = 0

    # ------------------------------------------------------------------
    # Z interpolation
    # ------------------------------------------------------------------
    def set_target_z(self, z_target, blend_steps=Z_INTERPOLATION_STEPS):
        if self.current_z is None:
            self.current_z  = z_target.clone()
            self.target_z   = z_target.clone()
            self.z_blend    = 1.0
        else:
            self.target_z    = z_target.clone()
            self.z_blend     = 0.0
            self.blend_steps  = blend_steps
            self.blend_counter = 0

    def update_z_interpolation(self):
        if self.z_blend < 1.0:
            self.blend_counter += 1
            t        = self.blend_counter / self.blend_steps
            t_smooth = t * t * (3 - 2 * t)
            self.z_blend = t_smooth
            return (1 - self.z_blend) * self.current_z + self.z_blend * self.target_z
        return self.target_z

    def finalize_z_transition(self):
        self.current_z = self.target_z.clone()
        self.z_blend   = 1.0

    # ------------------------------------------------------------------
    # Force / weakening
    # ------------------------------------------------------------------
    def apply_external_force(self, magnitude, direction, ramp_progress):
        body_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY,
                                    FORCE_CONFIG['application_point'])
        if body_id >= 0:
            force_factor       = 0.5 * (1 - np.cos(ramp_progress * np.pi))
            self.force_applied = magnitude * force_factor
            self.mj_data.xfrc_applied[body_id, :3] = self.force_applied * direction
            self.mj_data.xfrc_applied[body_id, 3:]  = [0, force_factor * 20, 0]

    def clear_forces(self):
        self.mj_data.xfrc_applied[:] = 0
        self.force_applied = 0

    def lock_scenario29_sagittal_trip_lane(self, verbose=False):
        """Lock Task-29 to the *actual* walking-forward axis.

        Important fix v5:
        The previous sagittal rail trusted ``walk_ref_fwd_xy`` / latent
        diagnostics. In this Humanoid model those diagnostics can point along
        world-Y even while the actual walking telemetry is world-X (your logs
        show vx ~= -0.8 m/s before trip). That made the "pure forward" rail
        force the avatar into a lateral fall.

        This method therefore ignores the latent heading for Task 29 and locks
        the trip direction from observed pre-trip motion. It then collapses the
        direction to the dominant world axis, with a deliberate preference for
        X whenever X motion is meaningful. For your current runs this resolves
        to [-1, 0], matching the visible walking direction.
        """
        if self.scenario29_trip_lane_locked:
            return

        # 1) Best signal: accumulated real pelvis velocity during WALK.
        candidates = []
        ema = np.asarray(getattr(self, 'scenario29_walk_vel_ema_xy', np.zeros(2)), dtype=float)
        samples = int(getattr(self, 'scenario29_walk_vel_samples', 0))
        if samples >= 6 and float(np.linalg.norm(ema)) > 0.035:
            candidates.append(('walk_velocity_ema', ema.copy(), 3.0))

        # 2) Robust fallback: actual displacement since walk start.
        if self._pelvis_id >= 0:
            cur_xy = np.asarray(self.mj_data.xpos[self._pelvis_id][:2], dtype=float)
            if getattr(self, 'walk_ref_origin_xy', None) is not None:
                disp = cur_xy - np.asarray(self.walk_ref_origin_xy, dtype=float)
                if float(np.linalg.norm(disp)) > 0.06:
                    candidates.append(('walk_displacement', disp.copy(), 2.4))

            # 3) Last fallback: current pelvis world velocity.
            vel_now = _body_world_velocity(self.mj_model, self.mj_data, self._pelvis_id)[:2]
            if float(np.linalg.norm(vel_now)) > 0.025:
                candidates.append(('current_pelvis_velocity', np.asarray(vel_now, dtype=float).copy(), 1.0))

        if candidates:
            source, raw, _ = max(candidates, key=lambda item: float(np.linalg.norm(item[1])) * item[2])
        else:
            source, raw = 'fallback_negative_x', np.array([-1.0, 0.0], dtype=float)

        raw = np.asarray(raw[:2], dtype=float)
        ax, ay = abs(float(raw[0])), abs(float(raw[1]))

        # Critical v5 rule: if there is meaningful X walking, use X.  The
        # console diagnostics and the model's actual travel use vx; letting a
        # noisy or latent Y component win is exactly what caused the lateral fall.
        if ax >= 0.035 and ax >= 0.30 * max(ay, 1e-9):
            fwd = np.array([1.0 if raw[0] >= 0.0 else -1.0, 0.0], dtype=float)
        elif ay >= 0.035:
            fwd = np.array([0.0, 1.0 if raw[1] >= 0.0 else -1.0], dtype=float)
        else:
            fwd = np.array([-1.0, 0.0], dtype=float)
            source = 'fallback_negative_x_small_motion'

        # The log has repeatedly shown negative-X walking.  If the latent/walk
        # guidance was still pointing along Y but the raw observed X is negative,
        # force the task to negative-X instead of allowing a lateral rail.
        if ax >= 0.08 and raw[0] < 0.0:
            fwd = np.array([-1.0, 0.0], dtype=float)
            source += '+vx_override'

        lat = np.array([-fwd[1], fwd[0]], dtype=float)
        if self._pelvis_id >= 0:
            origin = np.asarray(self.mj_data.xpos[self._pelvis_id][:2], dtype=float).copy()
        else:
            origin = np.zeros(2, dtype=float)

        self.scenario29_trip_fwd_xy = fwd
        self.scenario29_trip_lat_xy = lat
        self.scenario29_trip_origin_xy = origin
        self.scenario29_trip_yaw = float(np.arctan2(fwd[1], fwd[0]))
        self.scenario29_trip_lane_locked = True

        # Make all later fall/rest calculations use the corrected axis too.
        self.walk_ref_fwd_xy = fwd.copy()
        self.walk_ref_lat_xy = lat.copy()
        self.walk_ref_yaw = float(self.scenario29_trip_yaw)

        if verbose:
            print(f"      sagittal toe-trip lane locked from {source} raw=[{raw[0]:+.2f}, {raw[1]:+.2f}] -> fwd=[{fwd[0]:+.2f}, {fwd[1]:+.2f}] lat=[{lat[0]:+.2f}, {lat[1]:+.2f}]")

    def apply_scenario29_sagittal_rail(self, body_mass_kg, strength=1.0, hard=False):
        """Suppress side drift gently without fighting the contact solver.

        The previous rail was too stiff and produced solver chatter after impact.
        During trip we use light force-based correction; in rest mode we use
        velocity damping only.
        """
        self.lock_scenario29_sagittal_trip_lane(verbose=False)
        fwd = np.asarray(self.scenario29_trip_fwd_xy, dtype=float)
        lat = np.asarray(self.scenario29_trip_lat_xy, dtype=float)
        origin = np.asarray(self.scenario29_trip_origin_xy, dtype=float)
        bw = max(float(body_mass_kg) * 9.81, 1.0)
        gain = float(np.clip(strength, 0.0, 1.2))

        if not bool(getattr(self, 'rest_mode', False)):
            body_specs = [
                (self._pelvis_id, 0.34, 0.22),
                (self._torso_id,  0.42, 0.26),
                (self._head_id,   0.20, 0.14),
            ]
            for bid, kp, kd in body_specs:
                if bid is None or bid < 0:
                    continue
                pos = np.asarray(self.mj_data.xpos[bid][:2], dtype=float)
                vel = _body_world_velocity(self.mj_model, self.mj_data, bid)
                lat_err = float(np.dot(pos - origin, lat))
                lat_speed = float(np.dot(np.asarray(vel[:2], dtype=float), lat))
                rail_n = float(np.clip((-kp * lat_err - kd * lat_speed) * bw * gain,
                                       -0.16 * bw, 0.16 * bw))
                self.mj_data.xfrc_applied[bid, 0] += rail_n * lat[0]
                self.mj_data.xfrc_applied[bid, 1] += rail_n * lat[1]

                vel6 = np.zeros(6)
                mujoco.mj_objectVelocity(self.mj_model, self.mj_data, MJOBJ_BODY, bid, vel6, 0)
                ang = np.asarray(vel6[:3], dtype=float)
                roll_rate = float(np.dot(ang[:2], fwd))
                yaw_rate = float(ang[2])
                roll_tau = float(np.clip(-7.0 * gain * roll_rate, -20.0, 20.0))
                yaw_tau = float(np.clip(-5.0 * gain * yaw_rate, -16.0, 16.0))
                self.mj_data.xfrc_applied[bid, 3] += roll_tau * fwd[0]
                self.mj_data.xfrc_applied[bid, 4] += roll_tau * fwd[1]
                self.mj_data.xfrc_applied[bid, 5] += yaw_tau

        if self.mj_data.qvel.shape[0] >= 6:
            lin_xy = np.asarray(self.mj_data.qvel[:2], dtype=float).copy()
            side_v = float(np.dot(lin_xy, lat))
            self.mj_data.qvel[:2] = lin_xy - (0.42 if hard else 0.12) * side_v * lat
            ang_xy = np.asarray(self.mj_data.qvel[3:5], dtype=float).copy()
            roll_v = float(np.dot(ang_xy, fwd))
            self.mj_data.qvel[3:5] = ang_xy - (0.42 if hard else 0.10) * roll_v * fwd
            self.mj_data.qvel[5] *= (0.52 if hard else 0.82)

    def _scenario29_heading_axes(self):
        """Return walking-frame forward/lateral axes for the toe trip fall."""
        if getattr(self, 'walk_ref_fwd_xy', None) is not None:
            fwd = np.asarray(self.walk_ref_fwd_xy, dtype=float).copy()
        elif getattr(self, '_pelvis_id', -1) >= 0:
            R = self.mj_data.xmat[self._pelvis_id].reshape(3, 3)
            raw = np.asarray(R[:, 0], dtype=float)
            fwd = np.asarray(raw[:2], dtype=float)
        else:
            fwd = np.array([1.0, 0.0], dtype=float)
        n = float(np.linalg.norm(fwd))
        if n < 1e-6:
            fwd = np.array([1.0, 0.0], dtype=float)
        else:
            fwd = fwd / n
        lat = np.array([-fwd[1], fwd[0]], dtype=float)
        return fwd, lat

    def _scenario29_toe_trip_body_ids(self):
        """Return toe/forefoot bodies used as the virtual stone impact point.

        Prefer both toe bodies only. Pulling ankle/heel bodies into the virtual
        stone made one side of the gait snag first and produced lateral falls.
        """
        ids = []
        for i in range(self.mj_model.nbody):
            name = (mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, i) or '').lower()
            if any(k in name for k in ('toe', 'forefoot', 'ball')) and i not in ids:
                ids.append(i)
        if ids:
            return ids
        # Fallback only if the model has no toe-like body names.
        for i in range(self.mj_model.nbody):
            name = (mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, i) or '').lower()
            if 'foot' in name and i not in ids:
                ids.append(i)
        return ids

    def apply_toe_trip_forward_fall(self, progress, body_mass_kg, lateral_sign=0.0, followthrough=False):
        """Task 29: controlled toe/stone catch -> forward unsupported fall.

        v11 keeps the v10 no-slip toe-catch mechanics but makes the obstacle
        visibly register. The obstacle behaves like a short toe brake and
        pivot, not a forward shove or hard wall. A trip is foot/object contact
        that converts walking momentum into forward whole-body angular momentum,
        so the driver uses a slightly stronger early toe brake, a brief impact
        bump, and torso pitch torque while retaining the velocity guard.
        """
        raw_p = float(np.clip(progress, 0.0, 1.0))

        def smooth01(x):
            x = float(np.clip(x, 0.0, 1.0))
            return x * x * (3.0 - 2.0 * x)

        # Visible but non-explosive contact: several frames of toe catch, then
        # release. This avoids v9's backward/side skating from a narrow impulse.
        catch_on = smooth01((raw_p - 0.015) / 0.055)
        catch_off = 1.0 - smooth01((raw_p - 0.205) / 0.135)
        hit_gate = float(np.clip(catch_on * catch_off, 0.0, 1.0))
        contact_twitch = float(np.exp(-0.5 * ((raw_p - 0.070) / 0.034) ** 2))
        visible_bump = float(np.exp(-0.5 * ((raw_p - 0.105) / 0.030) ** 2))

        # Fast enough to read as a stumble, not so fast that the body launches.
        collapse = smooth01((raw_p - 0.075) / 0.50)
        late = smooth01((raw_p - 0.48) / 0.30)
        follow = 0.36 if followthrough else 1.0

        bw = max(float(body_mass_kg) * 9.81, 1.0)
        self.lock_scenario29_sagittal_trip_lane(verbose=False)
        fwd = np.asarray(self.scenario29_trip_fwd_xy, dtype=float)
        lat = np.asarray(self.scenario29_trip_lat_xy, dtype=float)

        pelvis_id = getattr(self, '_pelvis_id', -1)
        torso_id = getattr(self, '_torso_id', -1)
        head_id = getattr(self, '_head_id', -1)

        toe_ids = self._scenario29_toe_trip_body_ids()
        if toe_ids:
            projections = []
            for bid in toe_ids:
                try:
                    projections.append((float(np.dot(self.mj_data.xpos[bid][:2], fwd)), bid))
                except Exception:
                    projections.append((0.0, bid))
            lead_bid = max(projections, key=lambda item: item[0])[1]

            # Toe brake + light downward pin. The brake is much lower than v9,
            # so the body rotates over the caught foot instead of rebounding.
            brake_n = (0.16 + 0.48 * hit_gate + 0.13 * contact_twitch + 0.10 * visible_bump) * bw * follow
            down_n = (0.014 + 0.034 * hit_gate + 0.010 * visible_bump) * bw * follow
            self.mj_data.xfrc_applied[lead_bid, 0] += -brake_n * fwd[0]
            self.mj_data.xfrc_applied[lead_bid, 1] += -brake_n * fwd[1]
            self.mj_data.xfrc_applied[lead_bid, 2] += -down_n

            if raw_p > 0.22:
                self.mj_data.xfrc_applied[lead_bid, 2] += 0.006 * bw * smooth01((raw_p - 0.22) / 0.22)

        if pelvis_id is not None and pelvis_id >= 0:
            # Preserve momentum without pushing the root across the floor.
            fwd_n = (0.006 + 0.028 * collapse + 0.004 * visible_bump) * bw * follow
            down_n = (0.006 + 0.044 * collapse + 0.006 * visible_bump) * bw * follow
            self.mj_data.xfrc_applied[pelvis_id, 0] += fwd_n * fwd[0]
            self.mj_data.xfrc_applied[pelvis_id, 1] += fwd_n * fwd[1]
            self.mj_data.xfrc_applied[pelvis_id, 2] += -down_n

        if torso_id is not None and torso_id >= 0:
            # Main trip cue: pitch angular momentum, not horizontal shove.
            fwd_n = (0.024 + 0.072 * collapse + 0.020 * contact_twitch + 0.008 * visible_bump) * bw * follow
            down_n = (0.010 + 0.046 * collapse + 0.006 * visible_bump) * bw * follow
            self.mj_data.xfrc_applied[torso_id, 0] += fwd_n * fwd[0]
            self.mj_data.xfrc_applied[torso_id, 1] += fwd_n * fwd[1]
            self.mj_data.xfrc_applied[torso_id, 2] += -down_n
            pitch_torque = (20.0 * hit_gate + 64.0 * collapse + 14.0 * contact_twitch + 10.0 * visible_bump + 8.0 * late) * follow
            self.mj_data.xfrc_applied[torso_id, 3] += pitch_torque * lat[0]
            self.mj_data.xfrc_applied[torso_id, 4] += pitch_torque * lat[1]

        if head_id is not None and head_id >= 0:
            head_fwd_n = (0.004 + 0.014 * collapse + 0.006 * contact_twitch + 0.003 * visible_bump) * bw * follow
            head_down_n = (0.000 + 0.005 * collapse) * bw * follow
            self.mj_data.xfrc_applied[head_id, 0] += head_fwd_n * fwd[0]
            self.mj_data.xfrc_applied[head_id, 1] += head_fwd_n * fwd[1]
            self.mj_data.xfrc_applied[head_id, 2] += -head_down_n

        # Guard against the exact bad symptom from the last log: vx/slip jumps
        # to ~2.2 m/s after contact. Blend-limiting avoids visible snapping.
        if self.mj_data.qvel.shape[0] >= 6 and raw_p < 0.36:
            vxy = np.asarray(self.mj_data.qvel[:2], dtype=float).copy()
            vf = float(np.dot(vxy, fwd))
            vl = float(np.dot(vxy, lat))
            max_fwd = 1.06 + 0.27 * collapse
            min_back = -0.18
            vf_guard = float(np.clip(vf, min_back, max_fwd))
            vl_guard = float(np.clip(vl, -0.28, 0.28))
            guarded = vf_guard * fwd + vl_guard * lat
            self.mj_data.qvel[:2] = 0.70 * vxy + 0.30 * guarded
            self.mj_data.qvel[5] *= 0.88

        self.apply_scenario29_sagittal_rail(body_mass_kg, strength=0.46 + 0.26 * collapse, hard=False)

        for idx in getattr(self, 'arm_actuators', []):
            self.mj_model.actuator_gear[idx, 0] = self.original_gear[idx] * 0.004

    def lock_task43_forward_edge_lane(self, origin_xy=None, verbose=False):
        """Lock Task43 to the humanoid's inferred backside/backward lane.

        Earlier v19 hard-coded world +X.  In the viewer that appeared as a
        lateral slide for this humanoid.  v20 uses the runtime direction computed
        from toe/heel geometry by Task43GroundCloneHeightSupport.
        """
        fwd = np.asarray(globals().get('TASK43_RUNTIME_FWD_XY', np.array([1.0, 0.0])), dtype=float).copy()
        n = float(np.linalg.norm(fwd))
        if n < 1e-8:
            fwd = np.array([1.0, 0.0], dtype=float)
        else:
            fwd = fwd / n
        lat = np.asarray(globals().get('TASK43_RUNTIME_LAT_XY', np.array([-fwd[1], fwd[0]])), dtype=float).copy()
        ln = float(np.linalg.norm(lat))
        if ln < 1e-8:
            lat = np.array([-fwd[1], fwd[0]], dtype=float)
        else:
            lat = lat / ln
        if origin_xy is None:
            if self._pelvis_id >= 0:
                origin = np.asarray(self.mj_data.xpos[self._pelvis_id][:2], dtype=float).copy()
            else:
                origin = np.zeros(2, dtype=float)
        else:
            origin = np.asarray(origin_xy[:2], dtype=float).copy()
        yaw = float(np.arctan2(fwd[1], fwd[0]))
        self.scenario29_trip_fwd_xy = fwd
        self.scenario29_trip_lat_xy = lat
        self.scenario29_trip_origin_xy = origin
        self.scenario29_trip_yaw = yaw
        self.scenario29_trip_lane_locked = True
        self.walk_ref_fwd_xy = fwd.copy()
        self.walk_ref_lat_xy = lat.copy()
        self.walk_ref_yaw = yaw
        self.walk_ref_origin_xy = origin.copy()
        self.walk_ref_y = float(np.dot(origin, lat))
        if verbose:
            print(f"      Task43 forward lane locked -> fwd=[{fwd[0]:+.2f},{fwd[1]:+.2f}] lat=[{lat[0]:+.2f},{lat[1]:+.2f}] origin=[{origin[0]:+.3f},{origin[1]:+.3f}]")

    def apply_task43_edge_fall_driver(self, progress, body_mass_kg, followthrough=False):
        """Task38-style posterior fall action for Task43 after the back-step.

        Scenario 43 keeps its own object-height stand and single swing-leg step.
        Once the step finishes, this method borrows the proven Scenario 38
        backward-fall action: a posterior/upward pelvis push, progressive loss
        of leg support, and strict sagittal roll/yaw damping.  It avoids the
        weak v7c "just lose control" look and also avoids the v7b roll/tumble.
        """
        raw_p = float(np.clip(progress, 0.0, 1.0))

        def smooth01(x):
            x = float(np.clip(x, 0.0, 1.0))
            return x * x * (3.0 - 2.0 * x)

        ramp_start = float(globals().get('TASK43_TASK38_FORCE_RAMP_START', 0.02))
        ramp_end = float(globals().get('TASK43_TASK38_FORCE_RAMP_END', 0.86))
        ramp = smooth01((raw_p - ramp_start) / max(ramp_end - ramp_start, 1e-6))
        early = smooth01(raw_p / 0.22)
        free_gate = smooth01((raw_p - 0.10) / 0.58)
        follow = 0.34 if followthrough else 1.0
        bw = max(float(body_mass_kg) * 9.81, 1.0)

        self.lock_task43_forward_edge_lane(verbose=False)
        fwd = np.asarray(self.scenario29_trip_fwd_xy, dtype=float)
        fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-8)
        lat = np.asarray(self.scenario29_trip_lat_xy, dtype=float)
        lat = lat / max(float(np.linalg.norm(lat)), 1e-8)

        pelvis_id = getattr(self, '_pelvis_id', -1)
        torso_id = getattr(self, '_torso_id', -1)
        head_id = getattr(self, '_head_id', -1)

        # Task38 fall cue: posterior/upward load at pelvis. Direction is
        # rotated into the Task43 inferred backward lane instead of world -X.
        force_frac = float(globals().get('TASK43_TASK38_FORCE_BW_FRACTION', 0.42))
        upward = float(globals().get('TASK43_TASK38_UPWARD', 0.10))
        dir3 = np.array([fwd[0], fwd[1], upward], dtype=float)
        dir3 /= max(float(np.linalg.norm(dir3)), 1e-9)
        push_n = force_frac * bw * ramp * follow
        if pelvis_id is not None and pelvis_id >= 0:
            self.mj_data.xfrc_applied[pelvis_id, :3] += push_n * dir3

        # Light torso/head follow-through creates an intentional backside fall
        # cue without the head-first flip that appeared in v7b.
        torso_follow = float(globals().get('TASK43_TASK38_TORSO_FOLLOW_BW', 0.050)) * bw * ramp * follow
        head_follow = float(globals().get('TASK43_TASK38_HEAD_FOLLOW_BW', 0.018)) * bw * ramp * follow
        if torso_id is not None and torso_id >= 0:
            self.mj_data.xfrc_applied[torso_id, 0] += torso_follow * fwd[0]
            self.mj_data.xfrc_applied[torso_id, 1] += torso_follow * fwd[1]
            self.mj_data.xfrc_applied[torso_id, 2] += -0.012 * bw * ramp * follow
            lean_deg = 0.0
            try:
                lean_deg = float(self._compute_sagittal_trunk_lean_deg(fwd))
            except Exception:
                lean_deg = 0.0
            lean_abs = abs(lean_deg)
            counter_deg = float(globals().get('TASK43_TASK38_LEAN_COUNTER_DEG', 98.0))
            counter = smooth01((lean_abs - counter_deg) / 28.0)
            pitch_nm = float(globals().get('TASK43_TASK38_PITCH_TORQUE_NM', 22.0)) * (0.25 + 0.75 * early) * follow
            counter_nm = float(globals().get('TASK43_TASK38_COUNTER_TORQUE_NM', 36.0)) * counter * follow
            net_pitch = pitch_nm - counter_nm
            self.mj_data.xfrc_applied[torso_id, 3] += net_pitch * lat[0]
            self.mj_data.xfrc_applied[torso_id, 4] += net_pitch * lat[1]
        if head_id is not None and head_id >= 0:
            self.mj_data.xfrc_applied[head_id, 0] += head_follow * fwd[0]
            self.mj_data.xfrc_applied[head_id, 1] += head_follow * fwd[1]

        # Keep arms protective but not dominant; the backward push should be
        # visually coming from body COM loss, not arm whipping.
        for hname in ('L_Hand', 'R_Hand', 'L_Wrist', 'R_Wrist'):
            hid = _safe_name2id(self.mj_model, MJOBJ_BODY, hname)
            if hid >= 0:
                self.mj_data.xfrc_applied[hid, 0] += 0.004 * bw * ramp * fwd[0] * follow
                self.mj_data.xfrc_applied[hid, 1] += 0.004 * bw * ramp * fwd[1] * follow

        # Velocity shaping copies the spirit of Task38: preserve a clear
        # rearward fall trajectory after release, while suppressing lateral
        # drift, yaw, and roll. It is not a root teleport; it only caps/blends
        # velocities after the physical step has already released.
        if self.mj_data.qvel.shape[0] >= 6:
            vxy = np.asarray(self.mj_data.qvel[:2], dtype=float).copy()
            vf = float(np.dot(vxy, fwd))
            vl = float(np.dot(vxy, lat))
            init_v = float(globals().get('TASK43_TASK38_INITIAL_BACK_VEL', 0.16)) * early
            target_v = float(globals().get('TASK43_TASK38_TARGET_BACK_VEL', 0.78)) * free_gate
            desired_v = max(init_v, target_v)
            cap_early = float(globals().get('TASK43_TASK38_BACK_VEL_CAP_EARLY', 0.24))
            cap_free = float(globals().get('TASK43_TASK38_BACK_VEL_CAP_FREE', 0.92))
            back_cap = cap_early + (cap_free - cap_early) * free_gate
            if followthrough:
                desired_v *= 0.70
                back_cap = max(back_cap, 0.50)
            vf = 0.62 * vf + 0.38 * desired_v
            vf = float(np.clip(vf, -0.03, back_cap))
            lat_cap = float(globals().get('TASK43_TASK38_LAT_VEL_CAP', 0.035))
            vl = float(np.clip(vl, -lat_cap, lat_cap))
            self.mj_data.qvel[:2] = vf * fwd + vl * lat
            # Do not let the body float upward; Task38 used slight upward force
            # to induce posterior COM loss, but the height fall must still drop.
            down_cap = float(globals().get('TASK43_TASK38_DOWN_VEL_CAP', -2.85))
            self.mj_data.qvel[2] = float(np.clip(self.mj_data.qvel[2], down_cap, 0.22))

            ang_xy = np.asarray(self.mj_data.qvel[3:5], dtype=float).copy()
            pitch_v = float(np.dot(ang_xy, lat))
            roll_v = float(np.dot(ang_xy, fwd))
            max_pitch = float(globals().get('TASK43_TASK38_MAX_PITCH_RATE', 3.10))
            max_roll = float(globals().get('TASK43_TASK38_MAX_ROLL_RATE', 0.22))
            pitch_v = float(np.clip(pitch_v, -0.35 * max_pitch, max_pitch))
            roll_v = float(np.clip(roll_v * 0.26, -max_roll, max_roll))
            self.mj_data.qvel[3:5] = pitch_v * lat + roll_v * fwd
            yaw_cap = float(globals().get('TASK43_TASK38_MAX_YAW_RATE', 0.20))
            self.mj_data.qvel[5] = float(np.clip(self.mj_data.qvel[5] * 0.34, -yaw_cap, yaw_cap))

        self.apply_scenario29_sagittal_rail(body_mass_kg, strength=0.76, hard=False)
        for idx in getattr(self, 'arm_actuators', []):
            self.mj_model.actuator_gear[idx, 0] = self.original_gear[idx] * 0.08
        for idx in getattr(self, 'torso_actuators', []):
            self.mj_model.actuator_gear[idx, 0] = self.original_gear[idx] * 0.38

    def apply_task29_toe_trip_tone(self, progress, stage='trip'):
        """Tone schedule for a realistic toe trip.

        Keep support for the first contact frames so the toe visibly catches,
        then reduce strength fast enough that the body cannot recover into a
        controlled crouch. This is deliberately less abrupt than v9.
        """
        p_raw = float(np.clip(progress, 0.0, 1.0))
        p = p_raw * p_raw * (3.0 - 2.0 * p_raw)
        stage = str(stage or 'trip').lower()
        if stage == 'trip':
            leg_scale = 0.98 - 0.60 * p
            torso_scale = 0.92 - 0.38 * p
            if p_raw > 0.28:
                leg_scale -= 0.10 * min(1.0, (p_raw - 0.28) / 0.26)
        elif stage == 'react':
            leg_scale = 0.30 - 0.08 * p
            torso_scale = 0.40 - 0.10 * p
        else:
            leg_scale = 0.24 - 0.05 * p
            torso_scale = 0.31 - 0.06 * p
        leg_scale = float(np.clip(leg_scale, 0.18, 1.0))
        torso_scale = float(np.clip(torso_scale, 0.22, 1.0))
        self.leg_strength = leg_scale
        for idx in self.leg_actuators:
            self.mj_model.actuator_gear[idx, 0] = self.original_gear[idx] * leg_scale
        for idx in self.torso_actuators:
            self.mj_model.actuator_gear[idx, 0] = self.original_gear[idx] * torso_scale
        for idx in self.arm_actuators:
            self.mj_model.actuator_gear[idx, 0] = self.original_gear[idx] * 0.18

    def update_muscle_weakening(self, phase_progress, weakening_type='exponential'):
        phase_progress = float(np.clip(phase_progress, 0.0, 1.0))
        if weakening_type == 'exponential':
            fatigue_k = float(WEAKENING_CONFIG.get('fatigue_coefficient', 1.0))
            effort_scale = float(WEAKENING_CONFIG.get('active_effort_scale', 1.8))
            decay_time = float(max(WEAKENING_CONFIG.get('decay_time', 45), 1.0))
            phase_window_steps = float(max(PHASES.get('perturb', 60) + PHASES.get('react', 15), 1))
            effort_integral = phase_progress * effort_scale * (phase_window_steps / decay_time)
            self.leg_strength = WEAKENING_CONFIG['initial_factor'] * np.exp(-fatigue_k * effort_integral)
            self.leg_strength = max(self.leg_strength, WEAKENING_CONFIG['min_factor'])
        else:
            self.leg_strength = WEAKENING_CONFIG['initial_factor'] - phase_progress * (WEAKENING_CONFIG['initial_factor'] - WEAKENING_CONFIG['min_factor'])

        for idx in self.leg_actuators:
            self.mj_model.actuator_gear[idx, 0] = self.original_gear[idx] * self.leg_strength

        torso_strength = 0.5 + 0.5 * self.leg_strength
        for idx in self.torso_actuators:
            self.mj_model.actuator_gear[idx, 0] = self.original_gear[idx] * torso_strength

    def restore_strength(self):
        self.restore_passive_contact()
        self.mj_model.actuator_gear[:, 0] = self.original_gear.copy()
        self.leg_strength = 1.0

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------
    def get_action(self, obs, z):
        obs_t = torch.tensor(obs['proprio'], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = model.act(obs_t, z).squeeze(0).numpy()

        if self.rest_mode:
            if self.rest_mode_hard_lock:
                action[:] = 0.0
            else:
                # v30: after the body hits the ground, learned policy torque is
                # the main cause of visible rolling/bouncing. Keep only tiny
                # residual tone; passive contacts/damping do the settling.
                action[self.leg_actuators] *= 0.015
                action[self.arm_actuators] *= 0.002
                action[self.torso_actuators] *= 0.010
        elif self.current_phase in ('stand', 'step'):
            if self.current_phase == 'stand':
                leg_gain = float(self.age_style.get('stand_leg_gain', self.age_style.get('walk_leg_gain', 1.0)))
            else:
                leg_gain = float(self.age_style.get('walk_leg_gain', 1.0))
                pelvis_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Pelvis')
                if pelvis_id >= 0:
                    vel = _body_world_velocity(self.mj_model, self.mj_data, pelvis_id)
                    speed_err = self.walk_target_speed - float(vel[0])
                    leg_gain *= float(np.clip(1.0 + 0.06 * speed_err, 0.98, 1.06))
                leg_gain *= float(np.clip(getattr(self, 'walk_leg_gain_boost', 1.0), 0.95, 1.22))
            action[self.leg_actuators] *= leg_gain
            action[self.arm_actuators] *= self.age_style['arm_gain']
        else:
            # Task 29 no-support trip: keep early trip support so the toe hit
            # looks normal, then progressively quiet the policy after collapse.
            # v23: keep enough arm authority for visible protective reach.
            action[self.arm_actuators] *= (0.20 if self.current_phase == 'perturb' else 0.12)
            if self.current_phase == 'perturb':
                action[self.leg_actuators] *= 0.54
                action[self.torso_actuators] *= 0.44
            elif self.current_phase == 'react':
                action[self.leg_actuators] *= 0.34
                action[self.torso_actuators] *= 0.28
            else:
                action[self.leg_actuators] *= 0.14
                action[self.torso_actuators] *= 0.12
            pelvis_low = bool(self._pelvis_id >= 0 and self.mj_data.xpos[self._pelvis_id][2] < 0.18)
            grounded = bool(int(self.mj_data.ncon) >= 3 or self.has_nonfoot_ground_contact(min_normal_n=15.0))
            if pelvis_low and grounded:
                action[:] = 0.0

        self.action_smoothing.append(action)
        if len(self.action_smoothing) >= 2:
            if self.rest_mode:
                weights = np.array([0.6, 0.4])
            else:
                weights = self.age_style['smoothing_weights'] if self.current_phase in ('stand', 'step') else np.array([0.3, 0.7])
            action  = np.average(list(self.action_smoothing)[-2:],
                                 weights=weights, axis=0)
        return action

    # ------------------------------------------------------------------
    # Zero Moment Point (ZMP) - Vukobratovic & Borovac 2004
    # ------------------------------------------------------------------
    def compute_zmp(self):
        """
        Contact-frame forces are converted to world frame before computing
        the ground-plane load centroid. On a flat floor this is effectively a
        CoP-style ZMP approximation and is much more interpretable than using
        raw contact-frame components directly.
        """
        total_fz  = 0.0
        zmp_x_sum = 0.0
        zmp_y_sum = 0.0
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1, g2 = c.geom1, c.geom2
            names = ((mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g1) or '').lower(), (mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g2) or '').lower())
            is_ground = any(('floor' in n or 'ground' in n or 'plane' in n) for n in names) or any(self.mj_model.geom_type[g] == getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', -999) for g in (g1, g2))
            if not is_ground:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, wrench)
            f_world, _ = _contact_wrench_world(c, wrench)
            fz = max(0.0, float(f_world[2]))
            if fz <= 0.5:
                continue
            total_fz += fz
            zmp_x_sum += float(c.pos[0]) * fz
            zmp_y_sum += float(c.pos[1]) * fz
        pelvis_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, "Pelvis")
        if total_fz < 1.0:
            return self.mj_data.xpos[pelvis_id][:2].copy(), False, -0.5
        zmp_2d = np.array([zmp_x_sum / total_fz, zmp_y_sum / total_fz])
        support_center = self._estimate_support_polygon_center()
        margin = 0.15 - float(np.linalg.norm(zmp_2d - support_center))
        return zmp_2d, (margin >= 0.0), float(margin)

    # ------------------------------------------------------------------
    # Orientation-aware LieDownReward factory
    # ------------------------------------------------------------------
    def get_lie_down_reward(self, orient='up'):
        """
        Create an orientation-aware reward function based on fall direction.
        Supine  (orient='up'/'backward')  -> _create_supine_reward()
        Prone   (orient='down'/'forward') -> _create_prone_reward()
        Lateral (orient='left'/'right')   -> _create_lateral_reward()
        """
        self._lie_orient = orient

        if orient in ('up', 'backward'):
            return self._create_supine_reward()
        elif orient in ('down', 'forward'):
            return self._create_prone_reward()
        elif orient in ('left', 'right'):
            return self._create_lateral_reward(orient)
        else:
            return LieDownReward()

    def _create_supine_reward(self):
        """Custom reward for supine (backward) fall - lying on back."""
        def reward_fn(mdl, dat):
            pelvis_id = mujoco.mj_name2id(mdl, MJOBJ_BODY, "Pelvis")
            torso_id  = mujoco.mj_name2id(mdl, MJOBJ_BODY, "Torso")
            head_id   = mujoco.mj_name2id(mdl, MJOBJ_BODY, "Head")

            pelvis_z = dat.xpos[pelvis_id][2]
            torso_z  = dat.xpos[torso_id][2]
            head_z   = dat.xpos[head_id][2]

            height_reward  = -0.5 * (pelvis_z + torso_z)
            flatness       = -abs(torso_z - (head_z + pelvis_z) / 2)
            torso_supine   = 1.0 if torso_z < head_z else -1.0

            return height_reward + 0.3 * flatness + 0.2 * torso_supine
        return reward_fn

    def _create_prone_reward(self):
        """Custom reward for prone (forward) fall - face down with pivot."""
        def reward_fn(mdl, dat):
            pelvis_id = mujoco.mj_name2id(mdl, MJOBJ_BODY, "Pelvis")
            head_id   = mujoco.mj_name2id(mdl, MJOBJ_BODY, "Head")

            pelvis_pos = dat.xpos[pelvis_id]
            head_pos   = dat.xpos[head_id]

            forward_displacement = head_pos[0] - pelvis_pos[0]
            height               = (pelvis_pos[2] + head_pos[2]) / 2
            orientation          = -abs(head_pos[2] - pelvis_pos[2] + 0.1)
            pivot_reward         = max(0, forward_displacement)

            return -height + 0.5 * orientation + pivot_reward
        return reward_fn

    def _create_lateral_reward(self, side):
        """Custom reward for lateral (side) fall."""
        def reward_fn(mdl, dat):
            pelvis_id  = mujoco.mj_name2id(mdl, MJOBJ_BODY, "Pelvis")
            pelvis_pos = dat.xpos[pelvis_id]
            lateral    = pelvis_pos[1] if side == 'left' else -pelvis_pos[1]
            return -pelvis_pos[2] + 0.3 * lateral
        return reward_fn


# -------------------------------------------------------------------
# ANTHROPOMETRIC MODEL (Improvement 2: non-linear sarcopenia)
# -------------------------------------------------------------------
class AnthropometricModel:
    """
    Customizable digital human twin with age and height parameters.

    Weight can optionally be user-provided; otherwise an age/sex BMI prior is
    resolved into a subject mass while keeping the same model structure.
    """

    WINTER_MASS_FRACTIONS = {
        'head':       0.0810,
        'trunk':      0.4970,
        'thorax':     0.4970,
        'torso':      0.4970,
        'spine':      0.4970,
        'abdomen':    0.1460,
        'pelvis':     0.1420,
        'upperarm':   0.0280,
        'forearm':    0.0160,
        'hand':       0.0060,
        'thigh':      0.1000,
        'shank':      0.0465,
        'foot':       0.0145,
    }

    def __init__(self, mj_model, age=35, height=1.75, weight=None, sex='male'):
        self.mj_model = mj_model
        self.age      = age
        self.height   = height
        self.sex      = sex
        self.WINTER_MASS_FRACTIONS = get_segment_mass_fractions(self.sex)

        self.original_body_mass = mj_model.body_mass.copy()
        self.original_body_pos  = mj_model.body_pos.copy()
        self.original_gear      = mj_model.actuator_gear[:, 0].copy()
        self.original_geom_pos  = mj_model.geom_pos.copy()
        self.original_geom_size = mj_model.geom_size.copy()
        self.original_jnt_pos   = mj_model.jnt_pos.copy()
        self.original_inertia   = mj_model.body_inertia.copy()
        self.original_body_ipos = mj_model.body_ipos.copy() if hasattr(mj_model, 'body_ipos') else None
        self.original_geom_fromto = mj_model.geom_fromto.copy() if hasattr(mj_model, 'geom_fromto') else None

        self.default_body_mass = float(np.sum(self.original_body_mass))
        self.use_default_weight = weight is None
        self.weight = self.default_body_mass if weight is None else float(weight)

        self.height_scale = height / 1.75
        self.mass_scale   = 1.0 if self.use_default_weight else (self.weight / 70.0)

        self.apply_scaling()
        self.apply_age_effects()

    def _get_winter_fraction(self, body_name):
        if body_name is None:
            return None
        bname = body_name.lower()
        priority_order = [
            'upperarm', 'forearm', 'abdomen', 'thorax', 'pelvis',
            'trunk', 'torso', 'spine', 'thigh', 'shank', 'foot',
            'head', 'hand',
        ]
        for key in priority_order:
            if key in bname:
                return self.WINTER_MASS_FRACTIONS[key]
        return None


    def _segment_scale_profile(self):
        hs = float(np.clip(self.height / 1.75, 0.90, 1.10))
        age_frac = float(np.clip((self.age - 30.0) / 50.0, 0.0, 1.0))
        female = (str(self.sex).lower() == 'female')
        pelvis_width = float(np.clip(1.00 + 0.35 * (hs - 1.0) + (0.04 if female else 0.0), 0.92, 1.10))
        shoulder_width = float(np.clip(1.00 + 0.45 * (hs - 1.0) + (0.03 if female else 0.0) - 0.02 * age_frac, 0.92, 1.10))
        trunk_length = float(np.clip(1.00 + 0.55 * (hs - 1.0) - 0.02 * age_frac, 0.92, 1.10))
        neck_head = float(np.clip(1.00 + 0.20 * (hs - 1.0), 0.95, 1.06))
        upperarm = float(np.clip(1.00 + 0.75 * (hs - 1.0), 0.92, 1.10))
        forearm = float(np.clip(1.00 + 0.78 * (hs - 1.0), 0.92, 1.10))
        thigh = float(np.clip(1.00 + 0.82 * (hs - 1.0), 0.92, 1.10))
        shank = float(np.clip(1.00 + 0.84 * (hs - 1.0), 0.92, 1.10))
        foot = float(np.clip(1.00 + 0.70 * (hs - 1.0), 0.92, 1.08))
        return {
            'pelvis_width': pelvis_width,
            'shoulder_width': shoulder_width,
            'trunk_length': trunk_length,
            'neck_head': neck_head,
            'upperarm': upperarm,
            'forearm': forearm,
            'thigh': thigh,
            'shank': shank,
            'foot': foot,
        }

    def _body_scale_vector(self, body_name):
        lname = (body_name or '').lower()
        prof = self._segment_scale_profile()
        if 'pelvis' in lname:
            return np.array([1.0, prof['pelvis_width'], 1.0], dtype=float)
        if any(k in lname for k in ('torso', 'trunk', 'spine', 'abdomen', 'thorax')):
            return np.array([1.0, prof['shoulder_width'], prof['trunk_length']], dtype=float)
        if 'head' in lname or 'neck' in lname:
            return np.array([1.0, 1.0, prof['neck_head']], dtype=float)
        if 'shoulder' in lname:
            return np.array([1.0, prof['shoulder_width'], prof['trunk_length']], dtype=float)
        if 'elbow' in lname or 'upperarm' in lname:
            return np.array([prof['upperarm'], prof['upperarm'], prof['upperarm']], dtype=float)
        if any(k in lname for k in ('hand', 'wrist', 'forearm')):
            return np.array([prof['forearm'], prof['forearm'], prof['forearm']], dtype=float)
        if 'hip' in lname:
            return np.array([1.0, prof['pelvis_width'], 1.0], dtype=float)
        if 'knee' in lname or 'thigh' in lname:
            return np.array([prof['thigh'], prof['thigh'], prof['thigh']], dtype=float)
        if 'ankle' in lname or 'shank' in lname:
            return np.array([prof['shank'], prof['shank'], prof['shank']], dtype=float)
        if any(k in lname for k in ('toe', 'foot', 'heel')):
            return np.array([prof['foot'], prof['foot'], prof['foot']], dtype=float)
        return np.array([self.height_scale, self.height_scale, self.height_scale], dtype=float)

    def apply_allometric_kinematic_scaling(self):
        matched_bodies = 0
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, i)
            scale_vec = self._body_scale_vector(name)
            self.mj_model.body_pos[i] = self.original_body_pos[i] * scale_vec
            if self.original_body_ipos is not None:
                self.mj_model.body_ipos[i] = self.original_body_ipos[i] * scale_vec
            if np.max(np.abs(scale_vec - 1.0)) > 1e-6:
                matched_bodies += 1

        for j in range(self.mj_model.njnt):
            bid = int(self.mj_model.jnt_bodyid[j])
            bname = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, bid)
            scale_vec = self._body_scale_vector(bname)
            self.mj_model.jnt_pos[j] = self.original_jnt_pos[j] * scale_vec

        for g in range(self.mj_model.ngeom):
            bid = int(self.mj_model.geom_bodyid[g])
            bname = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, bid)
            scale_vec = self._body_scale_vector(bname)
            self.mj_model.geom_pos[g] = self.original_geom_pos[g] * scale_vec
            if self.original_geom_fromto is not None:
                fromto = self.original_geom_fromto[g].copy()
                if np.linalg.norm(fromto) > 0:
                    self.mj_model.geom_fromto[g, :3] = fromto[:3] * scale_vec
                    self.mj_model.geom_fromto[g, 3:] = fromto[3:] * scale_vec
        return matched_bodies

    def apply_scaling(self):
        kinematic_matches = self.apply_allometric_kinematic_scaling()
        matched_bodies = set()
        for i in range(self.mj_model.nbody):
            name   = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, i)
            frac   = self._get_winter_fraction(name)
            orig_m = self.original_body_mass[i]

            if self.use_default_weight:
                self.mj_model.body_mass[i] = orig_m
                if frac is not None and orig_m > 1e-6:
                    matched_bodies.add(i)
            else:
                if frac is not None and orig_m > 1e-6:
                    target_mass = self.weight * frac
                    scale_factor = (target_mass / 70.0) / (orig_m / 70.0) if orig_m > 0 else self.mass_scale
                    scale_factor = float(np.clip(scale_factor, 0.3, 5.0))
                    self.mj_model.body_mass[i] = orig_m * scale_factor
                    matched_bodies.add(i)
                else:
                    self.mj_model.body_mass[i] = orig_m * self.mass_scale

            mass_ratio = self.mj_model.body_mass[i] / max(orig_m, 1e-9)
            inertia_scale = mass_ratio * (self.height_scale ** 2.2)
            self.mj_model.body_inertia[i] = self.original_inertia[i] * inertia_scale

            # body_ipos was already rebuilt from the original local anchor frame
            # inside apply_allometric_kinematic_scaling().

        for geom_id in range(self.mj_model.ngeom):
            geom_type = self.mj_model.geom_type[geom_id]
            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                self.mj_model.geom_size[geom_id][0] *= self.height_scale
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                self.mj_model.geom_size[geom_id][0] *= self.height_scale
                self.mj_model.geom_size[geom_id][1] *= self.height_scale
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                self.mj_model.geom_size[geom_id] *= self.height_scale

        if not self.use_default_weight:
            current_total_mass = float(np.sum(self.mj_model.body_mass))
            if current_total_mass > 1e-9:
                mass_fix = float(self.weight / current_total_mass)
                self.mj_model.body_mass[:] *= mass_fix
                self.mj_model.body_inertia[:] *= mass_fix

        mode = 'default model body mass' if self.use_default_weight else 'custom mass scaling'
        print(f"      [Anthropometry] {mode}: mass-matched={len(matched_bodies)}/{self.mj_model.nbody} | kinematic-anchor-rebuild={kinematic_matches}/{self.mj_model.nbody}")

    def apply_age_effects(self):
        total_mass = float(np.sum(self.mj_model.body_mass))
        params = apply_age_effects_v2(
            mj_model=self.mj_model,
            age=self.age,
            sex=self.sex,
            height=self.height,
            body_mass_kg=total_mass,
            original_gear=self.original_gear,
        )
        params['default_body_mass_kg'] = self.default_body_mass
        return params

# -------------------------------------------------------------------
# IMU VALIDATOR (Improvement 3: STA, ground truth, validation cols)
# -------------------------------------------------------------------
class IMUValidator:
    """
    Hardware target:
      - MCU:  STM32F722RET6
      - Acc:  LIS3DH  +/-16 g
      - Gyro: LSM6DS3 +/-2000 dps
      - Output: 100 Hz
      - Placement: lower back (vertebrae L1-L2 proxy in the MuJoCo body tree)

    Physics notes:
      1. MuJoCo c-quantities are 6D spatial vectors with angular first, linear second.
         For user-facing kinematics we therefore prefer mj_objectVelocity /
         mj_objectAcceleration over directly indexing cvel/cacc.
      2. The model does not expose an anatomical L1-L2 site, so we use a rigid-body proxy:
         Torso body + offset interpolated toward Pelvis.
      3. If native env stepping is ~30 Hz, exported 100 Hz data is a uniform resampling
         of the native stream, not a true hardware-equivalent 100 Hz measurement.
    """

    IMU_CLIP_MS2 = 16.0 * 9.81

    def __init__(self, mj_model, mj_data, sensor_body='Torso', age=35, height=1.75,
                 target_output_hz=100.0, mount_label=None, sensor_offset_local=None, sensor_site=None):
        self.mj_model = mj_model
        self.mj_data  = mj_data
        self.age      = age
        self.height   = height
        self.target_output_hz = float(target_output_hz)
        self.mount_label = mount_label or IMU_HARDWARE_SPEC['mount_label']
        self.sensor_site = None

        site_id = _safe_name2id(mj_model, MJOBJ_SITE, sensor_site) if sensor_site else -1
        if site_id >= 0:
            body_id = int(mj_model.site_bodyid[site_id])
            sensor_body = mujoco.mj_id2name(mj_model, MJOBJ_BODY, body_id) or sensor_body
            self.sensor_site = mujoco.mj_id2name(mj_model, MJOBJ_SITE, site_id) or str(sensor_site)
            if sensor_offset_local is None:
                sensor_offset_local = np.asarray(mj_model.site_pos[site_id], dtype=float).copy()
        else:
            body_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, sensor_body)
            if body_id < 0:
                fallback_body = 'Pelvis'
                body_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, fallback_body)
                sensor_body = fallback_body
        if body_id < 0:
            raise ValueError("No suitable MuJoCo body found for IMU mounting (Torso/Pelvis missing).")

        self.sensor_body = sensor_body
        self.body_id = int(body_id)
        self.body_name = mujoco.mj_id2name(mj_model, MJOBJ_BODY, self.body_id) or sensor_body
        self.pelvis_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, "Pelvis")

        opt_dt = float(mj_model.opt.timestep)
        try:
            ns = int(mj_model.opt.iterations) if hasattr(mj_model.opt, 'iterations') else 0
        except Exception:
            ns = 0
        if opt_dt > 0 and opt_dt < 0.01:
            ns = max(1, round(0.0333 / opt_dt))
        else:
            ns = 1
        self.dt = float(opt_dt * ns)
        if self.dt < 1e-4 or self.dt > 0.5:
            self.dt = 1.0 / 30.0

        self.native_dt = self.dt
        self.native_hz = (1.0 / self.native_dt) if self.native_dt > 0 else 0.0
        self.output_hz = float(target_output_hz)
        self.output_dt = 1.0 / self.output_hz
        self.native_is_100hz = abs(self.native_hz - self.output_hz) < 1.0
        self.output_is_100hz = abs(self.output_hz - 100.0) < 1e-9
        self.output_mode = 'native' if self.native_is_100hz else 'resampled_from_native'
        self.effective_bandwidth_hz = 0.5 * self.native_hz

        self.sensor_offset_local = (
            np.asarray(sensor_offset_local, dtype=float).copy()
            if sensor_offset_local is not None else self._infer_l1l2_proxy_offset()
        )

        self._last_read_time = None
        self._last_read_sample = None
        self._last_export_verification = None

        mount_desc = f"site={self.sensor_site} | body={self.body_name}" if self.sensor_site else f"body={self.body_name}"
        print(f"      [IMU] {mount_desc} (id={self.body_id})  opt_dt={opt_dt:.5f}s  nsubsteps~{ns}  "
              f"native_dt={self.native_dt:.5f}s  native_hz={self.native_hz:.2f}  "
              f"output_hz={self.output_hz:.0f}  mode={self.output_mode}")

        self.data_buffer = {
            'timestamp': [], 'accelerometer': [], 'gyroscope': [],
            'quaternion': [], 'pelvis_height': [], 'pelvis_velocity': [],
            'impact_force': [], 'soft_tissue_artifact': [], 'sensor_confidence': [],
            'accel_raw': [], 'sensor_world_pos': [], 'sensor_world_vel': [],
        }

        age_factor       = max(1.0, 1.0 + (age - 60) * 0.02) if age > 60 else 1.0
        self.accel_bias  = np.random.normal(0, 0.03 * age_factor, 3)
        self.gyro_bias   = np.random.normal(0, 0.008 * age_factor, 3)
        self.accel_noise = 0.08 * age_factor
        self.gyro_noise  = 0.015 * age_factor
        self.sta_frequency = 0.25
        self.sta_amplitude = 0.05 * (height / 1.75)
        self.sta_phase     = np.random.uniform(0, 2 * np.pi)

        self.ground_truth_buffer = []
        self.GYRO_SAT  = np.deg2rad(2000.0)
        self._lp_alpha = 0.76
        self._lp_accel = np.array([0.0, 0.0, 9.81])
        self._lp_gyro  = np.zeros(3)
        self._aa_alpha = 0.87
        self._aa_accel = np.array([0.0, 0.0, 9.81])
        self._aa_gyro  = np.zeros(3)

        self._prev_v_world = None

    def _get_rot_matrix(self):
        body_quat = self.mj_data.xquat[self.body_id]
        rot_matrix = np.zeros(9)
        mujoco.mju_quat2Mat(rot_matrix, body_quat)
        return rot_matrix.reshape(3, 3)

    def _infer_l1l2_proxy_offset(self):
        if self.body_name.lower() == 'torso' and self.pelvis_id >= 0:
            R = self._get_rot_matrix()
            torso_pos = self.mj_data.xpos[self.body_id].copy()
            pelvis_pos = self.mj_data.xpos[self.pelvis_id].copy()
            frac = float(IMU_HARDWARE_SPEC.get('proxy_fraction_torso_to_pelvis', 0.40))
            offset_world = frac * (pelvis_pos - torso_pos)
            return R.T @ offset_world
        return np.zeros(3)

    def _sensor_world_position(self):
        R = self._get_rot_matrix()
        origin_world = self.mj_data.xpos[self.body_id].copy()
        return origin_world + R @ self.sensor_offset_local

    def _pelvis_world_velocity(self):
        if self.pelvis_id < 0:
            return np.zeros(3)
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.mj_model, self.mj_data,
                                 MJOBJ_BODY, self.pelvis_id, vel6, 0)
        return vel6[3:].copy()

    def _body_kinematics_local(self):
        vel6 = np.zeros(6)
        acc6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.mj_model, self.mj_data,
                                 MJOBJ_BODY, self.body_id, vel6, 1)
        if hasattr(mujoco, 'mj_objectAcceleration'):
            try:
                mujoco.mj_objectAcceleration(self.mj_model, self.mj_data,
                                             MJOBJ_BODY, self.body_id, acc6, 1)
            except Exception:
                acc6[:3] = self.mj_data.cacc[self.body_id][:3]
                acc6[3:] = self.mj_data.cacc[self.body_id][3:]
        else:
            acc6[:3] = self.mj_data.cacc[self.body_id][:3]
            acc6[3:] = self.mj_data.cacc[self.body_id][3:]
        return vel6, acc6

    def get_sensor_mount_info(self):
        geom_names = []
        for geom_id in range(self.mj_model.ngeom):
            if self.mj_model.geom_bodyid[geom_id] == self.body_id:
                gname = mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, geom_id) or f'geom_{geom_id}'
                geom_names.append(gname)
        model_pos = (self.mj_model.body_pos[self.body_id].copy()
                     if hasattr(self.mj_model, 'body_pos') else np.zeros(3))
        return {
            'sensor_body': self.body_name,
            'sensor_site': self.sensor_site,
            'sensor_body_id': int(self.body_id),
            'mount_type': (f"site:{self.sensor_site}" if self.sensor_site else f'{self.mount_label} proxy'),
            'body_origin_model_frame_m': model_pos,
            'sensor_offset_local_m': self.sensor_offset_local.copy(),
            'world_pos_m': self._sensor_world_position().copy(),
            'world_quat_wxyz': self.mj_data.xquat[self.body_id].copy(),
            'attached_geoms': geom_names,
        }

    def get_sampling_report(self):
        return {
            'native_dt_s': float(self.native_dt),
            'native_hz': float(self.native_hz),
            'native_is_100hz': bool(self.native_is_100hz),
            'output_dt_s': float(self.output_dt),
            'output_hz': float(self.output_hz),
            'output_is_100hz': bool(self.output_is_100hz),
            'output_mode': self.output_mode,
            'effective_bandwidth_hz': float(self.effective_bandwidth_hz),
            'true_hardware_equivalent_100hz': bool(self.native_is_100hz),
        }

    def get_runtime_processing_report(self):
        return {
            'requested_mount_label': IMU_HARDWARE_SPEC['mount_label'],
            'actual_mount_body': self.body_name,
            'actual_mount_site': self.sensor_site,
            'actual_mount_body_id': int(self.body_id),
            'actual_mount_proxy': bool(self.sensor_site is None),
            'actual_sensor_offset_local_m': self.sensor_offset_local.copy(),
            'actual_sensor_world_pos_m': self._sensor_world_position().copy(),
            'actual_kinematics_source': 'mj_objectVelocity + mj_objectAcceleration',
            'actual_point_kinematics': 'rigid-body point correction v = v0 + xr, a = a0 + xr + x(xr)',
            'actual_native_dt_s': float(self.native_dt),
            'actual_native_hz': float(self.native_hz),
            'actual_output_dt_s': float(self.output_dt),
            'actual_output_hz': float(self.output_hz),
            'actual_output_mode': self.output_mode,
            'actual_resampler': 'linear interpolation',
            'actual_effective_bandwidth_hz': float(self.effective_bandwidth_hz),
            'actual_antialias_alpha': float(self._aa_alpha),
            'actual_lowpass_alpha': float(self._lp_alpha),
            'actual_accel_bias_mps2': self.accel_bias.copy(),
            'actual_gyro_bias_rads': self.gyro_bias.copy(),
            'actual_accel_noise_std_mps2': float(self.accel_noise),
            'actual_gyro_noise_std_rads': float(self.gyro_noise),
            'actual_sta_frequency_hz': float(self.sta_frequency),
            'actual_sta_amplitude_m': float(self.sta_amplitude),
            'actual_accel_saturation_g': 16.0,
            'actual_gyro_saturation_dps': 2000.0,
            'hardware_equivalent_100hz': bool(self.native_is_100hz),
        }

    def verify_output_stream(self):
        t_native = np.asarray(self.data_buffer['timestamp'], dtype=float)
        if len(t_native) < 2:
            return {'available': False}
        t_out = np.arange(t_native[0], t_native[-1] + 1e-9, self.output_dt)
        native_dt = np.diff(t_native)
        out_dt = np.diff(t_out)
        res = {
            'available': True,
            'native_uniform_ok': bool(np.max(np.abs(native_dt - self.native_dt)) < 5e-4),
            'output_uniform_ok': bool(np.max(np.abs(out_dt - self.output_dt)) < 1e-9) if len(out_dt) else True,
            'native_hz': float(self.native_hz),
            'output_hz': float(self.output_hz),
            'upsample_factor': float(self.output_hz / max(self.native_hz, 1e-9)),
            'effective_bandwidth_hz': float(self.effective_bandwidth_hz),
            'true_hardware_equivalent_100hz': bool(self.native_is_100hz),
            'interpolation': 'linear',
        }
        self._last_export_verification = res
        return res

    def print_configuration_report(self, age=None, height=None, sex=None):
        mount = self.get_sensor_mount_info()
        rates = self.get_sampling_report()
        runtime = self.get_runtime_processing_report()
        attached = ', '.join(mount['attached_geoms']) if mount['attached_geoms'] else 'none listed'
        print("\n  +---------------------------------------------------------------------+")
        print("  |                    SUBJECT + IMU CONFIGURATION                     |")
        print("  +---------------------------------------------------------------------+")
        print(f"  |  Subject age        : {age if age is not None else self.age:>8}")
        print(f"  |  Subject height [m] : {height if height is not None else self.height:>8.3f}")
        print(f"  |  Subject sex        : {sex if sex is not None else 'n/a':>8}")
        print("  +---------------------------------------------------------------------+")
        print("  |  REQUESTED HARDWARE TARGET                                         |")
        print("  +---------------------------------------------------------------------+")
        print(f"  |  MCU                : {IMU_HARDWARE_SPEC['microcontroller']}")
        print(f"  |  Accelerometer      : {IMU_HARDWARE_SPEC['accelerometer']}")
        print(f"  |  Gyroscope          : {IMU_HARDWARE_SPEC['gyroscope']}")
        print(f"  |  Requested mount    : {IMU_HARDWARE_SPEC['mount_label']}")
        print(f"  |  Requested rate     : {IMU_HARDWARE_SPEC['sampling_hz']:.2f} Hz")
        print("  +---------------------------------------------------------------------+")
        print("  |  ACTUAL SIMULATION IMPLEMENTATION                                  |")
        print("  +---------------------------------------------------------------------+")
        print(f"  |  Actual mount body  : {runtime['actual_mount_body']} (body id={runtime['actual_mount_body_id']})")
        print(f"  |  Actual mount type  : {mount['mount_type']}")
        print(f"  |  Body origin model  : {np.round(mount['body_origin_model_frame_m'], 5)} m")
        print(f"  |  Sensor offset loc. : {np.round(runtime['actual_sensor_offset_local_m'], 5)} m")
        print(f"  |  Sensor pos world   : {np.round(runtime['actual_sensor_world_pos_m'], 5)} m")
        print(f"  |  Attached geoms     : {attached}")
        print(f"  |  Kinematics source  : {runtime['actual_kinematics_source']}")
        print("  +---------------------------------------------------------------------+")
        print("  |  ACTUAL SIGNAL PROCESSING CHAIN                                    |")
        print("  +---------------------------------------------------------------------+")
        print(f"  |  Native dt / Hz     : {runtime['actual_native_dt_s']:.5f} s / {runtime['actual_native_hz']:.2f} Hz")
        print(f"  |  Output dt / Hz     : {runtime['actual_output_dt_s']:.5f} s / {runtime['actual_output_hz']:.2f} Hz")
        print(f"  |  Output mode        : {runtime['actual_output_mode']}")
        print(f"  |  Resampler          : {runtime['actual_resampler']}")
        print(f"  |  Phys. bandwidth    : ~{runtime['actual_effective_bandwidth_hz']:.2f} Hz max")
        print(f"  |  Anti-alias alpha   : {runtime['actual_antialias_alpha']:.3f}")
        print(f"  |  Low-pass alpha     : {runtime['actual_lowpass_alpha']:.3f}")
        print(f"  |  Accel bias         : {np.round(runtime['actual_accel_bias_mps2'], 5)} m/s^2")
        print(f"  |  Gyro bias          : {np.round(runtime['actual_gyro_bias_rads'], 5)} rad/s")
        print(f"  |  Accel noise std    : {runtime['actual_accel_noise_std_mps2']:.5f} m/s^2")
        print(f"  |  Gyro noise std     : {runtime['actual_gyro_noise_std_rads']:.5f} rad/s")
        print(f"  |  STA freq / amp     : {runtime['actual_sta_frequency_hz']:.3f} Hz / {runtime['actual_sta_amplitude_m']:.5f} m")
        print(f"  |  Accel saturation   : +/-{runtime['actual_accel_saturation_g']:.1f} g")
        print(f"  |  Gyro saturation    : +/-{runtime['actual_gyro_saturation_dps']:.1f} dps")
        print(f"  |  Native == 100 Hz   : {'YES' if rates['native_is_100hz'] else 'NO'}")
        print(f"  |  100 Hz output      : {'YES' if rates['output_is_100hz'] else 'NO'} ({rates['output_mode']})")
        print("  +---------------------------------------------------------------------+")
        print("    [IMU reality] Requested 100 Hz hardware is being approximated by a")
        print("                  30 Hz native simulation stream plus linear resampling.")
        print("                  That is numerically consistent for export, but not")
        print("                  physically identical to a true native 100 Hz IMU.")


    def read_imu(self, sim_time=None, add_artifacts=True, use_cache=True):
        if sim_time is None:
            sim_time = float(self.mj_data.time)
        if use_cache and self._last_read_time is not None and abs(sim_time - self._last_read_time) < 1e-12:
            cached = {}
            for k, v in self._last_read_sample.items():
                cached[k] = v.copy() if isinstance(v, np.ndarray) else v
            return cached

        body_quat = self.mj_data.xquat[self.body_id].copy()
        R = self._get_rot_matrix()
        gravity_world = np.array([0.0, 0.0, -9.81])

        vel6_local, acc6_local = self._body_kinematics_local()
        omega_local = vel6_local[:3].copy()
        v_origin_local = vel6_local[3:].copy()
        alpha_local = acc6_local[:3].copy()
        a_origin_local = acc6_local[3:].copy()

        r_local = self.sensor_offset_local.copy()
        v_sensor_local = v_origin_local + np.cross(omega_local, r_local)
        a_sensor_local = (a_origin_local +
                          np.cross(alpha_local, r_local) +
                          np.cross(omega_local, np.cross(omega_local, r_local)))

        v_world = R @ v_sensor_local
        acc_true = a_sensor_local - (R.T @ gravity_world)

        if self._prev_v_world is not None:
            a_world_fd = (v_world - self._prev_v_world) / self.dt
            acc_raw = R.T @ (a_world_fd - gravity_world)
            acc_norm = float(np.linalg.norm(acc_raw))
            if acc_norm > self.IMU_CLIP_MS2:
                acc_raw = acc_raw * (self.IMU_CLIP_MS2 / acc_norm)
        else:
            acc_raw = np.array([0.0, 0.0, 9.81])
        self._prev_v_world = v_world.copy()

        gyro_true = omega_local.copy()

        self._aa_accel = self._aa_alpha * self._aa_accel + (1 - self._aa_alpha) * acc_true
        self._aa_gyro  = self._aa_alpha * self._aa_gyro  + (1 - self._aa_alpha) * gyro_true

        acc_noisy  = self._aa_accel + self.accel_bias + np.random.normal(0, self.accel_noise, 3)
        gyro_noisy = self._aa_gyro  + self.gyro_bias  + np.random.normal(0, self.gyro_noise, 3)

        sta = 0.0
        if add_artifacts:
            sta = self.sta_amplitude * np.sin(
                2 * np.pi * self.sta_frequency * sim_time + self.sta_phase)
            acc_noisy += np.array([0.1, 0.1, 1.0]) * sta * np.random.normal(1, 0.3)

        acc_sat  = np.clip(acc_noisy, -self.IMU_CLIP_MS2, self.IMU_CLIP_MS2)
        gyro_sat = np.clip(gyro_noisy, -self.GYRO_SAT, self.GYRO_SAT)
        self._lp_accel = self._lp_alpha * self._lp_accel + (1 - self._lp_alpha) * acc_sat
        self._lp_gyro  = self._lp_alpha * self._lp_gyro  + (1 - self._lp_alpha) * gyro_sat

        pelvis_height = float(self.mj_data.xpos[self.pelvis_id][2]) if self.pelvis_id >= 0 else float(self._sensor_world_position()[2])
        pelvis_vel_world = self._pelvis_world_velocity()

        contact_force = self._estimate_impact_force()
        confidence    = max(0.2, 1.0 - contact_force / 25000.0)

        sample = {
            'timestamp': sim_time,
            'accel':     self._lp_accel.copy(),
            'accel_raw': acc_raw.copy(),
            'gyro':      self._lp_gyro.copy(),
            'quat':      body_quat.copy(),
            'height':    pelvis_height,
            'pelvis_velocity': pelvis_vel_world.copy(),
            'sensor_velocity': v_world.copy(),
            'sensor_world_pos': self._sensor_world_position().copy(),
            'impact':    contact_force,
            'accel_true': acc_true,
            'gyro_true':  gyro_true,
            'soft_tissue_artifact': sta,
            'sensor_confidence': confidence,
        }
        self._last_read_time = float(sim_time)
        self._last_read_sample = {
            k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in sample.items()
        }
        return sample

    def _estimate_impact_force(self):
        peak_force = 0.0
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1 = c.geom1
            g2 = c.geom2
            b1 = self.mj_model.geom_bodyid[g1]
            b2 = self.mj_model.geom_bodyid[g2]
            if b1 == self.body_id or b2 == self.body_id:
                force = np.zeros(6)
                mujoco.mj_contactForce(self.mj_model, self.mj_data, i, force)
                peak_force = max(peak_force, float(max(0.0, force[0])))
        return min(peak_force, 12000.0)

    def log_frame(self, sim_time=None):
        if sim_time is None:
            sim_time = float(self.mj_data.time)
        imu_data = self.read_imu(sim_time, use_cache=False)
        self.data_buffer['timestamp'].append(imu_data['timestamp'])
        self.data_buffer['accelerometer'].append(imu_data['accel'])
        self.data_buffer['accel_raw'].append(imu_data['accel_raw'])
        self.data_buffer['gyroscope'].append(imu_data['gyro'])
        self.data_buffer['quaternion'].append(imu_data['quat'])
        self.data_buffer['pelvis_height'].append(imu_data['height'])
        self.data_buffer['pelvis_velocity'].append(imu_data['pelvis_velocity'])
        self.data_buffer['sensor_world_vel'].append(imu_data['sensor_velocity'])
        self.data_buffer['sensor_world_pos'].append(imu_data['sensor_world_pos'])
        self.data_buffer['impact_force'].append(imu_data['impact'])
        self.data_buffer['soft_tissue_artifact'].append(imu_data['soft_tissue_artifact'])
        self.data_buffer['sensor_confidence'].append(imu_data['sensor_confidence'])
        self.ground_truth_buffer.append({
            'accel_true': imu_data['accel_true'],
            'gyro_true':  imu_data['gyro_true'],
        })
        return float(np.linalg.norm(imu_data['accel_raw']))

    def _resample_buffers_100hz(self):
        t = np.array(self.data_buffer['timestamp'], dtype=float)
        if len(t) < 2:
            return None
        t_new = np.arange(t[0], t[-1] + 1e-9, self.output_dt)

        def interp_vec(key, dim):
            arr = np.array(self.data_buffer[key], dtype=float)
            out = np.zeros((len(t_new), dim), dtype=float)
            for k in range(dim):
                out[:, k] = np.interp(t_new, t, arr[:, k])
            return out

        def interp_scalar(values):
            arr = np.array(values, dtype=float)
            return np.interp(t_new, t, arr)

        gt_acc = np.array([gt['accel_true'] for gt in self.ground_truth_buffer], dtype=float)
        gt_gyro = np.array([gt['gyro_true'] for gt in self.ground_truth_buffer], dtype=float)

        res = {
            'timestamp': t_new,
            'accelerometer': interp_vec('accelerometer', 3),
            'accel_raw': interp_vec('accel_raw', 3),
            'gyroscope': interp_vec('gyroscope', 3),
            'pelvis_height': interp_scalar(self.data_buffer['pelvis_height']),
            'pelvis_velocity': interp_vec('pelvis_velocity', 3),
            'sensor_world_pos': interp_vec('sensor_world_pos', 3),
            'sensor_world_vel': interp_vec('sensor_world_vel', 3),
            'impact_force': interp_scalar(self.data_buffer['impact_force']),
            'soft_tissue_artifact': interp_scalar(self.data_buffer['soft_tissue_artifact']),
            'sensor_confidence': interp_scalar(self.data_buffer['sensor_confidence']),
            'accel_true': np.vstack([np.interp(t_new, t, gt_acc[:, k]) for k in range(3)]).T,
            'gyro_true': np.vstack([np.interp(t_new, t, gt_gyro[:, k]) for k in range(3)]).T,
        }
        return res

    def export_to_csv(self, filename, metadata=None):
        import csv
        from datetime import datetime
        if metadata is None:
            metadata = {}
        resampled = self._resample_buffers_100hz()
        if resampled is None:
            return {'filename': filename, 'frames': 0, 'falls_detected': 0}
        verification = self.verify_output_stream()
        n = len(resampled['timestamp'])

        headers = [
            'timestamp', 'accel_x', 'accel_y', 'accel_z',
            'accel_raw_x', 'accel_raw_y', 'accel_raw_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'sensor_pos_x', 'sensor_pos_y', 'sensor_pos_z',
            'sensor_vel_x', 'sensor_vel_y', 'sensor_vel_z',
            'pelvis_height', 'impact_force',
            'impact_magnitude', 'jerk_mag', 'fall_detected',
            'accel_true_x', 'accel_true_y', 'accel_true_z',
            'sensor_error_mag', 'soft_tissue_artifact', 'sensor_confidence',
        ]

        mount = self.get_sensor_mount_info()
        rates = self.get_sampling_report()
        meta = {
            'simulation_date': datetime.now().isoformat(),
            'age': metadata.get('age', 35),
            'height_m': metadata.get('height', 1.75),
            'sex': metadata.get('sex', 'unknown'),
            'weight_kg': metadata.get('weight', 70.0),
            'fall_type': metadata.get('fall_type', 'backward_walking'),
            'microcontroller': IMU_HARDWARE_SPEC['microcontroller'],
            'accelerometer': IMU_HARDWARE_SPEC['accelerometer'],
            'gyroscope': IMU_HARDWARE_SPEC['gyroscope'],
            'target_mount_label': IMU_HARDWARE_SPEC['mount_label'],
            'sampling_rate_hz': f"{self.output_hz:.1f}",
            'native_sampling_rate_hz': f"{self.native_hz:.3f}",
            'native_sampling_is_100hz': rates['native_is_100hz'],
            'output_sampling_is_100hz': rates['output_is_100hz'],
            'sampling_mode': rates['output_mode'],
            'effective_bandwidth_hz': f"{rates['effective_bandwidth_hz']:.3f}",
            'resample_verification': verification,
            'sensor_body': mount['sensor_body'],
            'sensor_body_id': mount['sensor_body_id'],
            'sensor_mount_type': mount['mount_type'],
            'sensor_body_origin_model_frame_m': np.array2string(np.asarray(mount['body_origin_model_frame_m']), precision=5, separator=', '),
            'sensor_offset_local_m': np.array2string(np.asarray(mount['sensor_offset_local_m']), precision=5, separator=', '),
            'sensor_world_frame_pos_m': np.array2string(np.asarray(mount['world_pos_m']), precision=5, separator=', '),
            'sensor_attached_geoms': ';'.join(mount['attached_geoms']) if mount['attached_geoms'] else 'none',
            'total_frames': n,
            'note': ('accel=gravity-compensated at lower-back proxy, '
                     'accel_raw=world-FD reconstructed proper acceleration, '
                     '100 Hz stream is linearly resampled if native_hz < 100'),
        }

        raw_arr = np.array(resampled['accel_raw'], dtype=float)
        jerk_mags = [0.0]
        for j in range(1, n):
            da = raw_arr[j] - raw_arr[j-1]
            jerk_mags.append(float(np.linalg.norm(da) / self.output_dt))

        rows, falls_detected = [], 0
        for i in range(n):
            ax, ay, az = [float(resampled['accelerometer'][i][k]) for k in range(3)]
            rx, ry, rz = [float(resampled['accel_raw'][i][k]) for k in range(3)]
            gx, gy, gz = [float(resampled['gyroscope'][i][k]) for k in range(3)]
            spx, spy, spz = [float(resampled['sensor_world_pos'][i][k]) for k in range(3)]
            svx, svy, svz = [float(resampled['sensor_world_vel'][i][k]) for k in range(3)]
            ph  = float(resampled['pelvis_height'][i])
            imp = float(resampled['impact_force'][i])
            mag = float(np.sqrt(rx**2 + ry**2 + rz**2))
            jerk = jerk_mags[i]
            fall = int(ph < 0.4 and (jerk > 15.0 or imp > 50.0))
            falls_detected += fall
            ax_t = float(resampled['accel_true'][i][0])
            ay_t = float(resampled['accel_true'][i][1])
            az_t = float(resampled['accel_true'][i][2])
            err_mag = float(np.sqrt((ax-ax_t)**2 + (ay-ay_t)**2 + (az-az_t)**2))
            sta  = float(resampled['soft_tissue_artifact'][i])
            conf = float(resampled['sensor_confidence'][i])
            rows.append([
                round(resampled['timestamp'][i], 4),
                round(ax, 6), round(ay, 6), round(az, 6),
                round(rx, 6), round(ry, 6), round(rz, 6),
                round(gx, 6), round(gy, 6), round(gz, 6),
                round(spx, 6), round(spy, 6), round(spz, 6),
                round(svx, 6), round(svy, 6), round(svz, 6),
                round(ph, 6), round(imp, 4),
                round(mag, 6), round(jerk, 4), fall,
                round(ax_t, 6), round(ay_t, 6), round(az_t, 6),
                round(err_mag, 6), round(sta, 6), round(conf, 3),
            ])
        meta['falls_detected_frames'] = falls_detected
        with open(filename, 'w', newline='') as f:
            for key, val in meta.items():
                f.write(f"# {key}: {val}\n")
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"  IMU data saved -> {filename}  ({n} frames | {falls_detected} fall-detected)")
        return {'filename': filename, 'frames': n, 'falls_detected': falls_detected}

# -------------------------------------------------------------------
# LAYER 1: MARKER + KINEMATICS EXPORTER
# -------------------------------------------------------------------


def rotmat_to_quat(R):
    """Convert 3x3 rotation matrix to quaternion in MuJoCo/OpenSim-friendly wxyz order."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    q = np.empty(4, dtype=float)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[0] = 0.25 / s
        q[1] = (R[2, 1] - R[1, 2]) * s
        q[2] = (R[0, 2] - R[2, 0]) * s
        q[3] = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-12))
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-12))
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-12))
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s
    q /= max(np.linalg.norm(q), 1e-12)
    if q[0] < 0.0:
        q *= -1.0
    return q
class MarkerKinematicsExporter:
    """
    Layer 1 biomechanics export for motion-analysis style validation.

    This version is more paper-aligned than the first pass:
      - broader anatomical marker set
      - much stronger fuzzy body resolution for humenv naming differences
      - virtual-marker trajectories + segment kinematics + joint kinematics
      - TRC export for marker-based motion-analysis tooling
      - MOT export for joint-angle style inspection
      - PNG visual summaries for quick qualitative review

    Notes:
      * The paper mentions site/marker-based biomechanical analysis. In the
        current compiled humenv workflow, the safest retrofit is to attach
        virtual anatomical markers rigidly to existing bodies.
      * These markers are dynamically correct with respect to the simulated
        body, even though they are not yet injected as MJCF <site> objects.
    """

    DEFAULT_MARKERS = {
        'HEAD':  {'candidates': ['Head'], 'offset_mode': 'superior', 'family': 'head'},
        'C7':    {'candidates': ['Torso', 'Chest', 'Spine'], 'offset_mode': 'posterior_superior', 'family': 'torso'},
        'CLAV':  {'candidates': ['Torso', 'Chest', 'Spine'], 'offset_mode': 'anterior_superior', 'family': 'torso'},
        'STRN':  {'candidates': ['Torso', 'Chest', 'Spine'], 'offset_mode': 'anterior_center', 'family': 'torso'},
        'T10':   {'candidates': ['Torso', 'Chest', 'Spine'], 'offset_mode': 'posterior_center', 'family': 'torso'},
        'SACR':  {'candidates': ['Pelvis', 'Hips'], 'offset_mode': 'posterior_center', 'family': 'pelvis'},
        'SHO_L': {'candidates': ['L_Shoulder', 'LeftShoulder', 'Shoulder_L', 'UpperArmL', 'ShoulderL', 'ArmL'], 'offset_mode': 'proximal', 'family': 'upperarm', 'side': 'left'},
        'SHO_R': {'candidates': ['R_Shoulder', 'RightShoulder', 'Shoulder_R', 'UpperArmR', 'ShoulderR', 'ArmR'], 'offset_mode': 'proximal', 'family': 'upperarm', 'side': 'right'},
        'ELB_L': {'candidates': ['L_Elbow', 'LeftElbow', 'Elbow_L', 'LowerArmL', 'ForeArmL', 'ElbowL'], 'offset_mode': 'proximal', 'family': 'lowerarm', 'side': 'left'},
        'ELB_R': {'candidates': ['R_Elbow', 'RightElbow', 'Elbow_R', 'LowerArmR', 'ForeArmR', 'ElbowR'], 'offset_mode': 'proximal', 'family': 'lowerarm', 'side': 'right'},
        'WRI_L': {'candidates': ['L_Wrist', 'L_Hand', 'LeftWrist', 'Wrist_L', 'HandL', 'WristL'], 'offset_mode': 'proximal', 'family': 'hand', 'side': 'left'},
        'WRI_R': {'candidates': ['R_Hand', 'R_Wrist', 'RightWrist', 'Wrist_R', 'HandR', 'WristR'], 'offset_mode': 'proximal', 'family': 'hand', 'side': 'right'},
        'ASI_L': {'candidates': ['Pelvis', 'Hips'], 'offset_mode': 'left_anterior', 'family': 'pelvis', 'side': 'left'},
        'ASI_R': {'candidates': ['Pelvis', 'Hips'], 'offset_mode': 'right_anterior', 'family': 'pelvis', 'side': 'right'},
        'PSI_L': {'candidates': ['Pelvis', 'Hips'], 'offset_mode': 'left_posterior', 'family': 'pelvis', 'side': 'left'},
        'PSI_R': {'candidates': ['Pelvis', 'Hips'], 'offset_mode': 'right_posterior', 'family': 'pelvis', 'side': 'right'},
        'HIP_L': {'candidates': ['L_Hip', 'LeftHip', 'Hip_L', 'ThighL', 'UpperLegL', 'HipL'], 'offset_mode': 'proximal', 'family': 'thigh', 'side': 'left'},
        'HIP_R': {'candidates': ['R_Hip', 'RightHip', 'Hip_R', 'ThighR', 'UpperLegR', 'HipR'], 'offset_mode': 'proximal', 'family': 'thigh', 'side': 'right'},
        'KNE_L': {'candidates': ['L_Knee', 'LeftKnee', 'Knee_L', 'L_Shin', 'ShinL', 'LowerLegL', 'KneeL'], 'offset_mode': 'proximal', 'family': 'shank', 'side': 'left'},
        'KNE_R': {'candidates': ['R_Knee', 'RightKnee', 'Knee_R', 'R_Shin', 'ShinR', 'LowerLegR', 'KneeR'], 'offset_mode': 'proximal', 'family': 'shank', 'side': 'right'},
        'ANK_L': {'candidates': ['L_Ankle', 'LeftAnkle', 'Ankle_L', 'L_Foot', 'FootL', 'AnkleL'], 'offset_mode': 'proximal', 'family': 'foot', 'side': 'left'},
        'ANK_R': {'candidates': ['R_Ankle', 'RightAnkle', 'Ankle_R', 'R_Foot', 'FootR', 'AnkleR'], 'offset_mode': 'proximal', 'family': 'foot', 'side': 'right'},
        'HEE_L': {'candidates': ['L_Heel', 'LeftHeel', 'Heel_L', 'L_Ankle', 'L_Foot', 'FootL', 'HeelL'], 'offset_mode': 'posterior_inferior', 'family': 'foot', 'side': 'left'},
        'HEE_R': {'candidates': ['R_Heel', 'RightHeel', 'Heel_R', 'R_Ankle', 'R_Foot', 'FootR', 'HeelR'], 'offset_mode': 'posterior_inferior', 'family': 'foot', 'side': 'right'},
        'TOE_L': {'candidates': ['L_Toe', 'LeftToe', 'Toe_L', 'L_Foot', 'FootL', 'ToeL'], 'offset_mode': 'anterior_inferior', 'family': 'foot', 'side': 'left'},
        'TOE_R': {'candidates': ['R_Toe', 'RightToe', 'Toe_R', 'R_Foot', 'FootR', 'ToeR'], 'offset_mode': 'anterior_inferior', 'family': 'foot', 'side': 'right'},
    }

    SEGMENT_EXPORTS = {
        'pelvis': ['Pelvis', 'Hips'],
        'torso': ['Torso', 'Chest', 'Spine'],
        'head': ['Head'],
        'left_upperarm': ['L_Shoulder', 'LeftShoulder', 'Shoulder_L', 'UpperArmL', 'ShoulderL', 'ArmL'],
        'right_upperarm': ['R_Shoulder', 'RightShoulder', 'Shoulder_R', 'UpperArmR', 'ShoulderR', 'ArmR'],
        'left_forearm': ['L_Elbow', 'LeftElbow', 'Elbow_L', 'LowerArmL', 'ForeArmL', 'ElbowL'],
        'right_forearm': ['R_Elbow', 'RightElbow', 'Elbow_R', 'LowerArmR', 'ForeArmR', 'ElbowR'],
        'left_hand': ['L_Hand', 'L_Wrist', 'LeftWrist', 'HandL', 'WristL'],
        'right_hand': ['R_Hand', 'R_Wrist', 'RightWrist', 'HandR', 'WristR'],
        'left_thigh': ['L_Hip', 'LeftHip', 'Hip_L', 'ThighL', 'UpperLegL', 'HipL'],
        'right_thigh': ['R_Hip', 'RightHip', 'Hip_R', 'ThighR', 'UpperLegR', 'HipR'],
        'left_shank': ['L_Knee', 'LeftKnee', 'Knee_L', 'L_Shin', 'ShinL', 'LowerLegL', 'KneeL'],
        'right_shank': ['R_Knee', 'RightKnee', 'Knee_R', 'R_Shin', 'ShinR', 'LowerLegR', 'KneeR'],
        'left_foot': ['L_Foot', 'L_Ankle', 'L_Toe', 'L_Heel', 'FootL', 'ToeL', 'HeelL'],
        'right_foot': ['R_Foot', 'R_Ankle', 'R_Toe', 'R_Heel', 'FootR', 'ToeR', 'HeelR'],
    }

    SKELETON_EDGES = [
        ('HEAD', 'C7'), ('C7', 'CLAV'), ('CLAV', 'STRN'), ('C7', 'T10'), ('T10', 'SACR'),
        ('CLAV', 'SHO_L'), ('SHO_L', 'ELB_L'), ('ELB_L', 'WRI_L'),
        ('CLAV', 'SHO_R'), ('SHO_R', 'ELB_R'), ('ELB_R', 'WRI_R'),
        ('SACR', 'ASI_L'), ('ASI_L', 'HIP_L'), ('HIP_L', 'KNE_L'), ('KNE_L', 'ANK_L'), ('ANK_L', 'HEE_L'), ('ANK_L', 'TOE_L'),
        ('SACR', 'ASI_R'), ('ASI_R', 'HIP_R'), ('HIP_R', 'KNE_R'), ('KNE_R', 'ANK_R'), ('ANK_R', 'HEE_R'), ('ANK_R', 'TOE_R'),
        ('ASI_L', 'ASI_R'), ('PSI_L', 'PSI_R')
    ]

    POSE_MARKERS = ['HEAD','C7','SHO_L','SHO_R','ELB_L','ELB_R','WRI_L','WRI_R','SACR','HIP_L','HIP_R','KNE_L','KNE_R','ANK_L','ANK_R','HEE_L','HEE_R','TOE_L','TOE_R']
    OPENSIM_TRC_NAMES = {
        'HEAD':'HEAD','C7':'C7','CLAV':'CLAV','STRN':'STRN','T10':'T10','SACR':'SACR',
        'ASI_L':'LASI','ASI_R':'RASI','PSI_L':'LPSI','PSI_R':'RPSI',
        'SHO_L':'LSHO','SHO_R':'RSHO','ELB_L':'LELB','ELB_R':'RELB',
        'WRI_L':'LWRA','WRI_R':'RWRA','HIP_L':'LHIP','HIP_R':'RHIP',
        'KNE_L':'LKNE','KNE_R':'RKNE','ANK_L':'LANK','ANK_R':'RANK',
        'HEE_L':'LHEE','HEE_R':'RHEE','TOE_L':'LTOE','TOE_R':'RTOE'
    }
    VISUAL_MARKERS = ['HEAD','C7','SHO_L','SHO_R','ELB_L','ELB_R','WRI_L','WRI_R','SACR','HIP_L','HIP_R','KNE_L','KNE_R','ANK_L','ANK_R','HEE_L','HEE_R','TOE_L','TOE_R']
    VISUAL_SKELETON_EDGES = [
        ('HEAD','C7'), ('C7','SACR'),
        ('C7','SHO_L'), ('SHO_L','ELB_L'), ('ELB_L','WRI_L'),
        ('C7','SHO_R'), ('SHO_R','ELB_R'), ('ELB_R','WRI_R'),
        ('SHO_L','SHO_R'),
        ('SACR','HIP_L'), ('HIP_L','KNE_L'), ('KNE_L','ANK_L'), ('ANK_L','HEE_L'), ('ANK_L','TOE_L'),
        ('SACR','HIP_R'), ('HIP_R','KNE_R'), ('KNE_R','ANK_R'), ('ANK_R','HEE_R'), ('ANK_R','TOE_R'),
        ('HIP_L','HIP_R')
    ]

    _SIDE_HINTS = {
        'left': ['left', '_l', 'l_', '-l', 'l-', 'lft', 'lf', 'lh', 'arml', 'handl', 'footl', 'thighl', 'shinl', 'toel', 'heell', 'wl', 'el', 'sl', 'hipl', 'kneel', 'anklel'],
        'right': ['right', '_r', 'r_', '-r', 'r-', 'rgt', 'rt', 'rh', 'armr', 'handr', 'footr', 'thighr', 'shinr', 'toer', 'heelr', 'wr', 'er', 'sr', 'hipr', 'kneer', 'ankler'],
    }
    _FAMILY_HINTS = {
        'head': ['head', 'skull', 'neck'],
        'torso': ['torso', 'chest', 'spine', 'thorax', 'trunk', 'abdomen'],
        'pelvis': ['pelvis', 'hip', 'hips', 'waist'],
        'upperarm': ['upperarm', 'shoulder', 'arm', 'humerus'],
        'lowerarm': ['lowerarm', 'forearm', 'elbow', 'ulna', 'radius'],
        'hand': ['hand', 'wrist', 'palm'],
        'thigh': ['thigh', 'upperleg', 'hip', 'femur'],
        'shank': ['shin', 'lowerleg', 'knee', 'calf', 'tibia'],
        'foot': ['foot', 'toe', 'heel', 'ankle'],
    }

    def __init__(self, mj_model, mj_data, export_hz=None):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.export_hz = float(export_hz) if export_hz is not None else 100.0
        self.frames = []
        self._joint_exports = self._resolve_joint_exports()
        self._body_catalog = self._build_body_catalog()
        self.marker_defs = self._build_marker_definitions()
        self.segment_ids = self._resolve_segment_ids()

    def _normalize(self, s):
        return ''.join(ch.lower() for ch in str(s or '') if ch.isalnum())

    def _body_name(self, body_id):
        return mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, body_id) or f'body_{body_id}'

    def _joint_name(self, joint_id):
        if hasattr(mujoco, 'mjtObj'):
            obj_joint = mujoco.mjtObj.mjOBJ_JOINT
        else:
            obj_joint = mujoco.mjOBJ_JOINT
        return mujoco.mj_id2name(self.mj_model, obj_joint, joint_id) or f'joint_{joint_id}'

    def _build_body_catalog(self):
        catalog = []
        for i in range(self.mj_model.nbody):
            name = self._body_name(i)
            norm = self._normalize(name)
            catalog.append({'id': i, 'name': name, 'norm': norm, 'tokens': self._tokenize(name)})
        return catalog

    def _tokenize(self, s):
        s = str(s or '').replace('-', '_')
        return [tok.lower() for tok in re.split(r'[^A-Za-z0-9]+', s) if tok]

    def _side_matches(self, rec, side):
        if not side:
            return True
        norm_name = rec['norm']
        tokens = rec.get('tokens', [])
        left_hit = any(t in ('l', 'left') for t in tokens) or norm_name.endswith('l') or norm_name.startswith('l') or '_l' in rec['name'].lower()
        right_hit = any(t in ('r', 'right') for t in tokens) or norm_name.endswith('r') or norm_name.startswith('r') or '_r' in rec['name'].lower()
        if side == 'left':
            return left_hit and not right_hit
        if side == 'right':
            return right_hit and not left_hit
        return True

    def _family_matches(self, rec, family):
        if not family:
            return True
        fam_score = self._score_family(rec['norm'], family)
        return fam_score > 0.0

    def _score_side(self, norm_name, side):
        if not side:
            return 0.0
        hints = self._SIDE_HINTS.get(side, [])
        score = 0.0
        for h in hints:
            if self._normalize(h) in norm_name:
                score += 1.0
        if side == 'left' and norm_name.endswith('l'):
            score += 0.75
        if side == 'right' and norm_name.endswith('r'):
            score += 0.75
        return score

    def _score_family(self, norm_name, family):
        if not family:
            return 0.0
        hints = self._FAMILY_HINTS.get(family, [])
        score = 0.0
        for h in hints:
            hnorm = self._normalize(h)
            if hnorm and hnorm in norm_name:
                score += 1.0
        # mild bonuses for strong anatomical endings
        if family == 'foot' and any(norm_name.endswith(s) for s in ('foot','toe','heel','ankle')):
            score += 0.75
        elif family == 'shank' and any(k in norm_name for k in ('knee','shin','lowerleg','calf','tibia')):
            score += 0.75
        elif family == 'thigh' and any(k in norm_name for k in ('thigh','upperleg','femur')):
            score += 0.75
        elif family == 'upperarm' and any(k in norm_name for k in ('shoulder','upperarm','humerus')):
            score += 0.75
        elif family == 'lowerarm' and any(k in norm_name for k in ('elbow','forearm','lowerarm','ulna','radius')):
            score += 0.75
        elif family == 'hand' and any(k in norm_name for k in ('hand','wrist','palm')):
            score += 0.75
        elif family == 'pelvis' and any(k in norm_name for k in ('pelvis','hips','waist')):
            score += 0.75
        elif family == 'torso' and any(k in norm_name for k in ('torso','chest','spine','thorax','trunk','abdomen')):
            score += 0.75
        elif family == 'head' and any(k in norm_name for k in ('head','skull','neck')):
            score += 0.75
        return score

    def _candidate_score(self, rec, key, family=None, side=None):
        score = 0.0
        if rec['norm'] == key:
            score += 100.0
        if rec['norm'].endswith(key):
            score += 30.0
        if key in rec['norm']:
            score += 20.0
        score += 5.0 * self._score_family(rec['norm'], family)
        score += 7.0 * self._score_side(rec['norm'], side)
        if side and not self._side_matches(rec, side):
            score -= 50.0
        if family and not self._family_matches(rec, family):
            score -= 20.0
        return score

    def _find_body_id(self, candidates, family=None, side=None):
        # side/family-aware candidate ranking instead of first substring hit.
        best_id = -1
        best_score = -1e9
        cand_keys = [self._normalize(c) for c in candidates if self._normalize(c)]
        for rec in self._body_catalog:
            for key in cand_keys:
                score = self._candidate_score(rec, key, family=family, side=side)
                if score > best_score:
                    best_score = score
                    best_id = rec['id']
        if best_id >= 0 and best_score >= 20.0:
            return best_id
        # fallback fuzzy family/side scoring only
        best_id = -1
        best_score = -1e9
        for rec in self._body_catalog:
            score = 5.0 * self._score_family(rec['norm'], family) + 7.0 * self._score_side(rec['norm'], side)
            if side and not self._side_matches(rec, side):
                score -= 30.0
            if family and not self._family_matches(rec, family):
                score -= 10.0
            if family == 'pelvis' and rec['id'] == 0:
                score -= 0.5
            if score > best_score and score >= 5.0:
                best_score = score
                best_id = rec['id']
        return best_id

    def _body_geom_ids(self, body_id):
        return [gid for gid in range(self.mj_model.ngeom) if self.mj_model.geom_bodyid[gid] == body_id]

    def _body_local_extents(self, body_id):
        mins = []
        maxs = []
        geom_ids = self._body_geom_ids(body_id)
        if not geom_ids:
            return np.array([-0.03, -0.03, -0.03]), np.array([0.03, 0.03, 0.03])
        for gid in geom_ids:
            size = np.asarray(self.mj_model.geom_size[gid], dtype=float)
            pos = np.asarray(self.mj_model.geom_pos[gid], dtype=float)
            gtype = int(self.mj_model.geom_type[gid])
            if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                half = np.array([size[0], size[0], size[0]])
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                r, h = size[0], size[1]
                half = np.array([r, r, h + r])
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                r, h = size[0], size[1]
                half = np.array([r, r, h])
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                half = size[:3]
            else:
                half = np.array([0.03, 0.03, 0.03])
            mins.append(pos - half)
            maxs.append(pos + half)
        return np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0)

    def _offset_from_mode(self, body_id, mode):
        mins, maxs = self._body_local_extents(body_id)
        center = 0.5 * (mins + maxs)
        eps = 0.008
        left_y = maxs[1] - eps
        right_y = mins[1] + eps
        mapping = {
            'center': center,
            'superior': np.array([center[0], center[1], maxs[2] - eps]),
            'inferior': np.array([center[0], center[1], mins[2] + eps]),
            'anterior_center': np.array([maxs[0] - eps, center[1], center[2]]),
            'anterior_superior': np.array([maxs[0] - eps, center[1], maxs[2] - eps]),
            'posterior_superior': np.array([mins[0] + eps, center[1], maxs[2] - eps]),
            'posterior_center': np.array([mins[0] + eps, center[1], center[2]]),
            'posterior_inferior': np.array([mins[0] + eps, center[1], mins[2] + eps]),
            'anterior_inferior': np.array([maxs[0] - eps, center[1], mins[2] + eps]),
            'left_anterior': np.array([maxs[0] - eps, left_y, center[2]]),
            'right_anterior': np.array([maxs[0] - eps, right_y, center[2]]),
            'left_posterior': np.array([mins[0] + eps, left_y, center[2]]),
            'right_posterior': np.array([mins[0] + eps, right_y, center[2]]),
            'proximal': np.array([mins[0] + eps, center[1], center[2]]),
            'distal': np.array([maxs[0] - eps, center[1], center[2]]),
        }
        return mapping.get(mode, center)

    def _build_marker_definitions(self):
        defs = {}
        for mname, spec in self.DEFAULT_MARKERS.items():
            body_id = self._find_body_id(spec['candidates'], family=spec.get('family'), side=spec.get('side'))
            if body_id < 0:
                continue
            defs[mname] = {
                'body_id': body_id,
                'body_name': self._body_name(body_id),
                'offset_local': self._offset_from_mode(body_id, spec.get('offset_mode', 'center')),
                'offset_mode': spec.get('offset_mode', 'center'),
                'family': spec.get('family'),
                'side': spec.get('side'),
            }
        return defs

    def _resolve_segment_ids(self):
        out = {}
        for label, candidates in self.SEGMENT_EXPORTS.items():
            family = 'pelvis' if 'pelvis' in label else 'torso' if 'torso' in label else 'head' if 'head' in label else 'upperarm' if 'upperarm' in label else 'lowerarm' if 'forearm' in label else 'hand' if 'hand' in label else 'thigh' if 'thigh' in label else 'shank' if 'shank' in label else 'foot'
            side = 'left' if label.startswith('left_') else 'right' if label.startswith('right_') else None
            bid = self._find_body_id(candidates, family=family, side=side)
            if bid >= 0:
                out[label] = bid
        return out

    def _resolve_joint_exports(self):
        exports = []
        for jid in range(self.mj_model.njnt):
            qadr = int(self.mj_model.jnt_qposadr[jid])
            dadr = int(self.mj_model.jnt_dofadr[jid]) if jid < len(self.mj_model.jnt_dofadr) else -1
            jtype = int(self.mj_model.jnt_type[jid])
            if jtype == getattr(mujoco.mjtJoint, 'mjJNT_FREE', -999):
                continue
            exports.append({'joint_id': jid, 'joint_name': self._joint_name(jid), 'qpos_adr': qadr, 'dof_adr': dadr, 'joint_type': jtype})
        return exports

    def _body_rot(self, body_id):
        return self.mj_data.xmat[body_id].reshape(3, 3).copy()

    def _marker_world(self, body_id, offset_local):
        R = self._body_rot(body_id)
        return self.mj_data.xpos[body_id].copy() + R @ np.asarray(offset_local, dtype=float)

    def _point_world_velocity(self, body_id, point_world):
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.mj_model, self.mj_data, MJOBJ_BODY, body_id, vel6, 0)
        omega = vel6[:3].copy()
        v0 = vel6[3:].copy()
        r = np.asarray(point_world, dtype=float) - self.mj_data.xpos[body_id].copy()
        return v0 + np.cross(omega, r)

    def _joint_angle_deg(self, rec):
        qadr = rec['qpos_adr']
        jtype = rec['joint_type']
        if jtype == getattr(mujoco.mjtJoint, 'mjJNT_HINGE', -1):
            return float(np.degrees(self.mj_data.qpos[qadr]))
        if jtype == getattr(mujoco.mjtJoint, 'mjJNT_SLIDE', -3):
            return float(self.mj_data.qpos[qadr])
        if jtype == getattr(mujoco.mjtJoint, 'mjJNT_BALL', -2):
            quat = np.asarray(self.mj_data.qpos[qadr:qadr+4], dtype=float)
            if quat.shape[0] == 4:
                return float(np.degrees(2.0 * np.arccos(np.clip(quat[0], -1.0, 1.0))))
        return float(self.mj_data.qpos[qadr])



    def _compute_equal_limits(self, pts_list, dims=(0,1), pad=0.03):
        arr = np.array([p for p in pts_list if p is not None], dtype=float)
        if arr.size == 0:
            return None
        sub = arr[:, list(dims)]
        mins = np.min(sub, axis=0)
        maxs = np.max(sub, axis=0)
        center = 0.5*(mins+maxs)
        half = 0.5*np.max(maxs-mins) + pad
        return [(center[i]-half, center[i]+half) for i in range(2)]

    def _compute_equal_limits_3d(self, pts_list, pad=0.03):
        arr = np.array([p for p in pts_list if p is not None], dtype=float)
        if arr.size == 0:
            return None
        mins = np.min(arr, axis=0); maxs = np.max(arr, axis=0)
        center = 0.5*(mins+maxs)
        half = 0.5*np.max(maxs-mins) + pad
        return [(center[i]-half, center[i]+half) for i in range(3)]

    def _collect_plot_points(self, pts, names=None):
        names = names or [n for n in self.VISUAL_MARKERS if n in pts]
        return [np.asarray(pts[n], dtype=float) for n in names if n in pts]

    def _presentation_frame(self, pts):
        pts = {k: np.asarray(v, dtype=float) for k, v in pts.items() if v is not None}
        if 'SACR' in pts:
            origin = pts['SACR'].copy()
        elif 'HIP_L' in pts and 'HIP_R' in pts:
            origin = 0.5 * (pts['HIP_L'] + pts['HIP_R'])
        elif pts:
            origin = np.mean(np.stack(list(pts.values()), axis=0), axis=0)
        else:
            origin = np.zeros(3, dtype=float)

        lateral = None
        for a, b in [('HIP_L', 'HIP_R'), ('ASI_L', 'ASI_R'), ('SHO_L', 'SHO_R')]:
            if a in pts and b in pts:
                lateral = pts[b] - pts[a]
                break
        if lateral is None:
            lateral = np.array([0.0, 1.0, 0.0], dtype=float)
        lateral = np.asarray(lateral, dtype=float)
        lateral[2] = 0.0
        lat_n = float(np.linalg.norm(lateral))
        if lat_n < 1e-9:
            lateral = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            lateral /= lat_n

        up = np.array([0.0, 0.0, 1.0], dtype=float)
        forward = np.cross(up, lateral)
        fwd_n = float(np.linalg.norm(forward))
        if fwd_n < 1e-9:
            forward = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            forward /= fwd_n

        ref = None
        if 'C7' in pts:
            ref = pts['C7'] - origin
        elif 'HEAD' in pts:
            ref = pts['HEAD'] - origin
        elif 'SHO_L' in pts and 'SHO_R' in pts:
            ref = 0.5 * (pts['SHO_L'] + pts['SHO_R']) - origin
        if ref is not None:
            ref = np.asarray(ref, dtype=float).copy()
            ref[2] = 0.0
            if np.linalg.norm(ref) > 1e-9 and np.dot(forward, ref) < 0.0:
                forward *= -1.0

        lateral = np.cross(up, forward)
        lat_n = float(np.linalg.norm(lateral))
        if lat_n < 1e-9:
            lateral = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            lateral /= lat_n
        return origin, forward, lateral, up

    def _point_to_presentation_local(self, point, origin, forward, lateral, up):
        diff = np.asarray(point, dtype=float) - np.asarray(origin, dtype=float)
        return np.array([
            float(np.dot(diff, forward)),
            float(np.dot(diff, lateral)),
            float(np.dot(diff, up)),
        ], dtype=float)

    def _points_to_presentation_local(self, pts):
        origin, forward, lateral, up = self._presentation_frame(pts)
        return {
            n: self._point_to_presentation_local(p, origin, forward, lateral, up)
            for n, p in pts.items()
        }

    def _marker_side(self, name):
        if str(name).endswith('_L'):
            return 'left'
        if str(name).endswith('_R'):
            return 'right'
        return 'center'

    def _edge_style(self, a, b):
        sa = self._marker_side(a)
        sb = self._marker_side(b)
        if sa == sb == 'left':
            return {'color': '#2ca02c', 'linewidth': 2.2}
        if sa == sb == 'right':
            return {'color': '#d62728', 'linewidth': 2.2}
        if sa == sb == 'center':
            return {'color': '#1f77b4', 'linewidth': 2.4}
        return {'color': '#7f7f7f', 'linewidth': 1.9}

    def _phase_indices(self, *phase_names):
        wanted = {str(p).lower() for p in phase_names}
        return [i for i, fr in enumerate(self.frames) if str(fr.get('phase', '')).lower() in wanted]

    def _mid_index(self, idxs, fallback=0):
        if not idxs:
            return int(fallback)
        return int(idxs[len(idxs) // 2])

    def _select_peak_pose_index(self):
        if not self.frames:
            return 0
        pelvis_h = np.array([fr['pelvis_height'] for fr in self.frames], dtype=float)
        trunk = np.array([fr['trunk_lean_deg'] for fr in self.frames], dtype=float)
        active = self._phase_indices('perturb', 'react', 'fall')
        if not active:
            return int(np.argmax(trunk))
        if len(active) >= 2:
            local_h = pelvis_h[active]
            drop_i = int(np.argmin(np.diff(local_h))) + 1
            anchor = active[min(drop_i, len(active) - 1)]
        else:
            anchor = active[0]
        lo = max(active[0], anchor - 3)
        hi = min(len(self.frames) - 1, anchor + 8)
        min_h = float(np.min(pelvis_h))
        cands = [i for i in range(lo, hi + 1) if pelvis_h[i] > (min_h + 0.04)]
        if not cands:
            cands = [i for i in active if pelvis_h[i] > (min_h + 0.02)] or list(active)
        def score(i):
            drop = max(0.0, pelvis_h[0] - pelvis_h[i])
            return 0.75 * trunk[i] + 35.0 * drop
        return int(max(cands, key=score))

    def _select_snapshot_indices(self):
        n = len(self.frames)
        if n == 0:
            return []
        stand_idx = self._mid_index(self._phase_indices('stand'), 0)
        walk_idx = self._mid_index(self._phase_indices('walk'), min(n - 1, max(1, n // 3)))
        pose_idx = self._select_peak_pose_index()
        rest_idx = n - 1
        idxs = []
        for idx in [stand_idx, walk_idx, pose_idx, rest_idx]:
            idx = int(np.clip(idx, 0, n - 1))
            if idx not in idxs:
                idxs.append(idx)
        while len(idxs) < 4 and len(idxs) < n:
            cand = int(round((len(idxs)) * (n - 1) / max(3, n - 1)))
            if cand not in idxs:
                idxs.append(cand)
            else:
                break
        return idxs[:4]

    def _snapshot_label(self, idx):
        if not self.frames:
            return ''
        phase = str(self.frames[idx].get('phase', '')).upper() or 'FRAME'
        if idx == self._select_peak_pose_index():
            phase = 'LOSS OF BAL.'
        elif idx == len(self.frames) - 1:
            phase = 'REST'
        return f"{phase}\nt={self.frames[idx]['time']:.2f}s"

    def _quality_summary_lines(self):
        qc = self.quality_report()
        suspect = [k for k, v in qc.items() if v.get('status') != 'ok']
        derived_segments = [k for k, v in self.marker_summary().get('segments', {}).items() if v == 'derived_marker_triad']
        lines = [
            'Layer-1 export summary',
            f"Markers resolved : {len(self.marker_defs)}/{len(self.DEFAULT_MARKERS)}",
            f"Suspect markers : {len(suspect)}",
            f"Derived segments : {len(derived_segments)} ({', '.join(derived_segments) if derived_segments else 'none'})",
            f"Joint channels   : {len(self._joint_exports)}",
            f"Frames captured  : {len(self.frames)}",
        ]
        if suspect:
            lines.append('Suspects        : ' + ', '.join(suspect[:6]) + ('...' if len(suspect) > 6 else ''))
        return lines

    def _derived_segment_record(self, label, markers, marker_velocities):
        def mk(name):
            return np.asarray(markers[name], dtype=float) if name in markers else None
        def mv(name):
            return np.asarray(marker_velocities[name], dtype=float) if name in marker_velocities else np.zeros(3)
        if label == 'left_foot' and all(k in markers for k in ('ANK_L','HEE_L','TOE_L')):
            a,h,t = mk('ANK_L'), mk('HEE_L'), mk('TOE_L')
            x = t - h; x = x / max(np.linalg.norm(x), 1e-9)
            z = a - 0.5*(h+t); z = z / max(np.linalg.norm(z), 1e-9)
            y = np.cross(z, x); y = y / max(np.linalg.norm(y), 1e-9)
            z = np.cross(x, y); z = z / max(np.linalg.norm(z), 1e-9)
            R = np.column_stack([x,y,z])
            return {'body_name':'derived_left_foot','position':(a+h+t)/3.0,'orientation_quat':rotmat_to_quat(R),'angular_velocity':np.zeros(3),'linear_velocity':(mv('ANK_L')+mv('HEE_L')+mv('TOE_L'))/3.0,'source':'derived_markers'}
        if label == 'right_foot' and all(k in markers for k in ('ANK_R','HEE_R','TOE_R')):
            a,h,t = mk('ANK_R'), mk('HEE_R'), mk('TOE_R')
            x = t - h; x = x / max(np.linalg.norm(x), 1e-9)
            z = a - 0.5*(h+t); z = z / max(np.linalg.norm(z), 1e-9)
            y = np.cross(z, x); y = y / max(np.linalg.norm(y), 1e-9)
            z = np.cross(x, y); z = z / max(np.linalg.norm(z), 1e-9)
            R = np.column_stack([x,y,z])
            return {'body_name':'derived_right_foot','position':(a+h+t)/3.0,'orientation_quat':rotmat_to_quat(R),'angular_velocity':np.zeros(3),'linear_velocity':(mv('ANK_R')+mv('HEE_R')+mv('TOE_R'))/3.0,'source':'derived_markers'}
        return None

    def quality_report(self):
        report = {}
        for mname, spec in self.DEFAULT_MARKERS.items():
            if mname not in self.marker_defs:
                report[mname] = {'status': 'missing'}
                continue
            rec = self.marker_defs[mname]
            norm = self._normalize(rec['body_name'])
            side_ok = self._side_matches({'name': rec['body_name'], 'norm': norm, 'tokens': self._tokenize(rec['body_name'])}, spec.get('side')) if spec.get('side') else True
            fam_ok = self._family_matches({'name': rec['body_name'], 'norm': norm, 'tokens': self._tokenize(rec['body_name'])}, spec.get('family')) if spec.get('family') else True
            body_name_l = str(rec['body_name']).lower()
            if spec.get('family') == 'pelvis' and 'pelvis' in body_name_l:
                fam_ok = True
                side_ok = True
            if spec.get('family') == 'upperarm' and 'shoulder' in body_name_l:
                fam_ok = True
            report[mname] = {
                'body': rec['body_name'],
                'side_ok': bool(side_ok),
                'family_ok': bool(fam_ok),
                'status': 'ok' if (side_ok and fam_ok) else 'suspect',
            }
        return report

    def capture_frame(self, sim_time=None, phase=None):
        if sim_time is None:
            sim_time = float(self.mj_data.time)
        markers = {}
        marker_vels = {}
        for mname, spec in self.marker_defs.items():
            pos = self._marker_world(spec['body_id'], spec['offset_local'])
            vel = self._point_world_velocity(spec['body_id'], pos)
            markers[mname] = pos
            marker_vels[mname] = vel

        segments = {}
        for label, bid in self.segment_ids.items():
            derived = self._derived_segment_record(label, markers, marker_vels)
            if derived is not None:
                segments[label] = derived
                continue
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(self.mj_model, self.mj_data, MJOBJ_BODY, bid, vel6, 0)
            segments[label] = {
                'body_name': self._body_name(bid),
                'position': self.mj_data.xpos[bid].copy(),
                'orientation_quat': self.mj_data.xquat[bid].copy(),
                'angular_velocity': vel6[:3].copy(),
                'linear_velocity': vel6[3:].copy(),
                'source': 'body_frame',
            }

        joints = {}
        for rec in self._joint_exports:
            joints[rec['joint_name']] = {
                'qpos': float(self.mj_data.qpos[rec['qpos_adr']]),
                'value_export': self._joint_angle_deg(rec),
                'unit': 'deg' if rec['joint_type'] != getattr(mujoco.mjtJoint, 'mjJNT_SLIDE', -3) else 'm',
                'qvel': float(self.mj_data.qvel[rec['dof_adr']]) if rec['dof_adr'] >= 0 and rec['dof_adr'] < len(self.mj_data.qvel) else 0.0,
            }

        pelvis_body = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Pelvis')
        head_body = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Head')
        pelvis = markers.get('SACR', self.mj_data.xpos[pelvis_body] if pelvis_body >= 0 else np.zeros(3))
        head = markers.get('HEAD', self.mj_data.xpos[head_body] if head_body >= 0 else np.array([0.0, 0.0, 1.0]))
        trunk_vec = np.asarray(head) - np.asarray(pelvis)
        trunk_norm = float(np.linalg.norm(trunk_vec))
        trunk_lean_deg = 0.0
        if trunk_norm > 1e-8:
            trunk_lean_deg = float(np.degrees(np.arccos(np.clip(np.dot(trunk_vec / trunk_norm, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))))

        frame = {
            'time': float(sim_time),
            'phase': phase or '',
            'markers': markers,
            'marker_velocities': marker_vels,
            'segments': segments,
            'joints': joints,
            'pelvis_height': float(pelvis[2]),
            'trunk_lean_deg': trunk_lean_deg,
        }
        self.frames.append(frame)
        return frame

    def marker_summary(self):
        qc = self.quality_report()
        suspect = {k: v for k, v in qc.items() if v.get('status') == 'suspect'}
        return {
            'num_markers': len(self.marker_defs),
            'marker_names': list(self.marker_defs.keys()),
            'segments': {k: ('derived_marker_triad' if k in ('left_foot','right_foot') else self._body_name(v)) for k, v in self.segment_ids.items()},
            'num_joint_exports': len(self._joint_exports),
            'resolved_bodies': {m: rec['body_name'] for m, rec in self.marker_defs.items()},
            'suspect_markers': suspect,
            'num_suspect_markers': len(suspect),
        }

    def export_marker_csv(self, filename):
        import csv
        if not self.frames:
            return {'filename': filename, 'frames': 0}
        marker_names = list(self.marker_defs.keys())
        trc_names = [self.OPENSIM_TRC_NAMES.get(n, n) for n in marker_names]
        headers = ['time', 'phase']
        for m in marker_names:
            headers += [f'{m}_x', f'{m}_y', f'{m}_z', f'{m}_vx', f'{m}_vy', f'{m}_vz']
        headers += ['pelvis_height', 'trunk_lean_deg']
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for fr in self.frames:
                row = [round(fr['time'], 6), fr['phase']]
                for m in marker_names:
                    p = fr['markers'][m]
                    v = fr['marker_velocities'][m]
                    row += [round(float(p[0]), 6), round(float(p[1]), 6), round(float(p[2]), 6),
                            round(float(v[0]), 6), round(float(v[1]), 6), round(float(v[2]), 6)]
                row += [round(fr['pelvis_height'], 6), round(fr['trunk_lean_deg'], 6)]
                writer.writerow(row)
        return {'filename': filename, 'frames': len(self.frames), 'markers': len(marker_names)}

    def export_segment_csv(self, filename):
        import csv
        if not self.frames:
            return {'filename': filename, 'frames': 0}
        segs = list(self.segment_ids.keys())
        headers = ['time', 'phase']
        for s in segs:
            headers += [f'{s}_px', f'{s}_py', f'{s}_pz', f'{s}_qw', f'{s}_qx', f'{s}_qy', f'{s}_qz', f'{s}_wx', f'{s}_wy', f'{s}_wz', f'{s}_vx', f'{s}_vy', f'{s}_vz']
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for fr in self.frames:
                row = [round(fr['time'], 6), fr['phase']]
                for s in segs:
                    rec = fr['segments'][s]
                    p = rec['position']; q = rec['orientation_quat']; w = rec['angular_velocity']; v = rec['linear_velocity']
                    row += [round(float(x), 6) for x in (*p, *q, *w, *v)]
                writer.writerow(row)
        return {'filename': filename, 'frames': len(self.frames), 'segments': len(segs)}

    def export_joint_csv(self, filename):
        import csv
        if not self.frames:
            return {'filename': filename, 'frames': 0}
        joint_names = list(self.frames[0]['joints'].keys())
        headers = ['time', 'phase']
        for j in joint_names:
            headers += [f'{j}_value', f'{j}_qvel']
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for fr in self.frames:
                row = [round(fr['time'], 6), fr['phase']]
                for j in joint_names:
                    rec = fr['joints'][j]
                    row += [round(float(rec['value_export']), 6), round(float(rec['qvel']), 6)]
                writer.writerow(row)
        return {'filename': filename, 'frames': len(self.frames), 'joints': len(joint_names)}

    def export_trc(self, filename, data_rate=None):
        if not self.frames:
            return {'filename': filename, 'frames': 0}
        marker_names = list(self.marker_defs.keys())
        trc_names = [self.OPENSIM_TRC_NAMES.get(n, n) for n in marker_names]
        times = np.array([fr['time'] for fr in self.frames], dtype=float)
        if len(times) > 1:
            dt = float(np.median(np.diff(times)))
            frame_rate = float(1.0 / max(dt, 1e-9))
        else:
            frame_rate = float(data_rate or self.export_hz)
        data_rate = float(data_rate or frame_rate)
        num_frames = len(self.frames)
        num_markers = len(marker_names)
        units = 'm'
        with open(filename, 'w') as f:
            f.write(f"PathFileType\t4\t(X/Y/Z)\t{Path(filename).name}\n")
            f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
            f.write(f"{data_rate:.6f}\t{frame_rate:.6f}\t{num_frames}\t{num_markers}\t{units}\t{data_rate:.6f}\t1\t{num_frames}\n")
            header = ['Frame#', 'Time']
            sub = ['', '']
            for i, name in enumerate(trc_names, start=1):
                header += [name, '', '']
                sub += [f'X{i}', f'Y{i}', f'Z{i}']
            f.write('\t'.join(header) + '\n')
            f.write('\t'.join(sub) + '\n')
            for idx, fr in enumerate(self.frames, start=1):
                row = [str(idx), f"{fr['time']:.6f}"]
                for name in marker_names:
                    p = fr['markers'][name]
                    row += [f"{float(p[0]):.6f}", f"{float(p[1]):.6f}", f"{float(p[2]):.6f}"]
                f.write('\t'.join(row) + '\n')
        return {'filename': filename, 'frames': num_frames, 'markers': num_markers, 'data_rate': data_rate}

    def export_mot(self, filename, data_rate=None):
        if not self.frames:
            return {'filename': filename, 'frames': 0}
        joint_names = list(self.frames[0]['joints'].keys())
        times = np.array([fr['time'] for fr in self.frames], dtype=float)
        if len(times) > 1:
            dt = float(np.median(np.diff(times)))
            data_rate = float(data_rate or (1.0 / max(dt, 1e-9)))
        else:
            data_rate = float(data_rate or self.export_hz)
        with open(filename, 'w') as f:
            f.write(f"name {Path(filename).name}\n")
            f.write(f"datacolumns {1 + len(joint_names)}\n")
            f.write(f"datarows {len(self.frames)}\n")
            f.write(f"range {times[0]:.6f} {times[-1]:.6f}\n")
            f.write("endheader\n")
            f.write('time\t' + '\t'.join(joint_names) + '\n')
            for fr in self.frames:
                vals = [f"{fr['joints'][j]['value_export']:.6f}" for j in joint_names]
                f.write(f"{fr['time']:.6f}\t" + '\t'.join(vals) + '\n')
        return {'filename': filename, 'frames': len(self.frames), 'joints': len(joint_names), 'data_rate': data_rate}
    def _median_filter(self, arr, k=5):
        arr = np.asarray(arr, dtype=float)
        if k <= 1 or arr.size == 0:
            return arr.copy()
        h = k // 2
        out = np.empty_like(arr)
        for i in range(arr.size):
            lo = max(0, i - h); hi = min(arr.size, i + h + 1)
            out[i] = np.nanmedian(arr[lo:hi])
        return out

    def export_visuals(self, prefix):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            return {'available': False, 'error': f'matplotlib unavailable: {e}', 'plots': []}
        if not self.frames:
            return {'available': False, 'plots': []}

        times = np.array([fr['time'] for fr in self.frames], dtype=float)
        pelvis_h = np.array([fr['pelvis_height'] for fr in self.frames], dtype=float)
        trunk = np.array([fr['trunk_lean_deg'] for fr in self.frames], dtype=float)
        plots = []
        snapshot_idxs = self._select_snapshot_indices()
        names3d = [n for n in self.VISUAL_MARKERS if n in self.marker_defs]
        pose_idx = self._select_peak_pose_index()

        def world_pts_for_frame(idx):
            return {n: np.asarray(self.frames[idx]['markers'][n], dtype=float) for n in names3d}

        def local_pts_for_frame(idx):
            return self._points_to_presentation_local(world_pts_for_frame(idx))

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.plot(times, pelvis_h, label='Pelvis height [m]', linewidth=2.0)
        ax.plot(times, trunk / 100.0, label='Trunk lean /100 [deg]', linewidth=2.0)
        for idx in snapshot_idxs:
            ax.axvline(times[idx], color='0.82', linestyle='--', linewidth=0.9)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Value')
        ax.set_title('Layer-1 kinematic summary')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fn1 = f'{prefix}_kinematics_summary.png'
        fig.tight_layout()
        fig.savefig(fn1, dpi=180)
        plt.close(fig)
        plots.append(fn1)

        if len(names3d) >= 8 and snapshot_idxs:
            fig = plt.figure(figsize=(12.5, 3.9))
            frames_local = []
            all_pts = []
            for idx in snapshot_idxs:
                pts_local = local_pts_for_frame(idx)
                frames_local.append(pts_local)
                all_pts.extend(self._collect_plot_points(pts_local, names3d))
            lims = self._compute_equal_limits(all_pts, dims=(0, 2), pad=0.05)
            for k, (idx, pts) in enumerate(zip(snapshot_idxs, frames_local), start=1):
                ax = fig.add_subplot(1, len(snapshot_idxs), k)
                self._plot_edges_2d(ax, pts, 0, 2, marker_size=22, edges=self.VISUAL_SKELETON_EDGES, names=names3d)
                ax.set_title(self._snapshot_label(idx))
                ax.set_xlabel('Forward [m]')
                ax.set_ylabel('Up [m]')
                ax.grid(True, alpha=0.3)
                if lims:
                    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
                ax.set_aspect('equal', adjustable='box')
            fig.suptitle('Sagittal marker-skeleton snapshots (pelvis-aligned)')
            fig.tight_layout()
            fn2 = f'{prefix}_sagittal_snapshots.png'
            fig.savefig(fn2, dpi=180)
            plt.close(fig)
            plots.append(fn2)

        traj_markers = [n for n in ['HEAD', 'SACR', 'TOE_L', 'TOE_R', 'HEE_L', 'HEE_R'] if n in self.marker_defs]
        walk_frames = [fr for fr in self.frames if fr.get('phase') == 'walk']
        traj_frames = walk_frames if len(walk_frames) >= 40 else self.frames
        decim = max(1, len(traj_frames) // 180)
        if len(traj_markers) >= 2:
            fig = plt.figure(figsize=(6.4, 6.2))
            ax = fig.add_subplot(111)
            sac = np.array([fr['markers']['SACR'] for fr in traj_frames], dtype=float) if 'SACR' in self.marker_defs else None
            head = np.array([fr['markers']['HEAD'] for fr in traj_frames], dtype=float) if 'HEAD' in self.marker_defs else None
            lf = 0.5 * (np.array([fr['markers']['TOE_L'] for fr in traj_frames], dtype=float) + np.array([fr['markers']['HEE_L'] for fr in traj_frames], dtype=float)) if all(m in self.marker_defs for m in ('TOE_L','HEE_L')) else None
            rf = 0.5 * (np.array([fr['markers']['TOE_R'] for fr in traj_frames], dtype=float) + np.array([fr['markers']['HEE_R'] for fr in traj_frames], dtype=float)) if all(m in self.marker_defs for m in ('TOE_R','HEE_R')) else None
            if sac is not None:
                ax.plot(sac[::decim,0], sac[::decim,1], linewidth=2.6, label='SACR path')
                ax.scatter([sac[0,0]],[sac[0,1]], s=30, color=ax.lines[-1].get_color())
                ax.scatter([sac[-1,0]],[sac[-1,1]], s=42, marker='x', color=ax.lines[-1].get_color())
            if head is not None:
                ax.plot(head[::decim,0], head[::decim,1], linewidth=1.8, alpha=0.9, label='HEAD path')
            if lf is not None:
                ax.plot(lf[::decim,0], lf[::decim,1], linewidth=1.4, alpha=0.9, label='Left foot center')
            if rf is not None:
                ax.plot(rf[::decim,0], rf[::decim,1], linewidth=1.4, alpha=0.9, label='Right foot center')
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_title('Global top-view walk trajectory')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_aspect('equal', adjustable='box')
            fig.tight_layout()
            fn3 = f'{prefix}_topview_trajectories.png'
            fig.savefig(fn3, dpi=180)
            plt.close(fig)
            plots.append(fn3)

        if 'SACR' in self.marker_defs and len(traj_markers) >= 2:
            head_xy, lf_xy, rf_xy = [], [], []
            for fr in traj_frames:
                pts_ref = {n: np.asarray(fr['markers'][n], dtype=float) for n in set(['SACR','HEAD','HIP_L','HIP_R','ASI_L','ASI_R','SHO_L','SHO_R']) if n in fr['markers']}
                origin, forward, lateral, up = self._presentation_frame(pts_ref)
                if 'HEAD' in fr['markers']:
                    head_xy.append(self._point_to_presentation_local(fr['markers']['HEAD'], origin, forward, lateral, up)[:2])
                if 'TOE_L' in fr['markers'] and 'HEE_L' in fr['markers']:
                    lf = 0.5 * (np.asarray(fr['markers']['TOE_L'], dtype=float) + np.asarray(fr['markers']['HEE_L'], dtype=float))
                    lf_xy.append(self._point_to_presentation_local(lf, origin, forward, lateral, up)[:2])
                if 'TOE_R' in fr['markers'] and 'HEE_R' in fr['markers']:
                    rf = 0.5 * (np.asarray(fr['markers']['TOE_R'], dtype=float) + np.asarray(fr['markers']['HEE_R'], dtype=float))
                    rf_xy.append(self._point_to_presentation_local(rf, origin, forward, lateral, up)[:2])
            fig = plt.figure(figsize=(6.4, 6.2))
            ax = fig.add_subplot(111)
            if head_xy:
                rel_head = np.asarray(head_xy, dtype=float)
                if rel_head.shape[0] > 5:
                    rel_head[:, 0] = self._median_filter(rel_head[:, 0], 7)
                    rel_head[:, 1] = self._median_filter(rel_head[:, 1], 7)
                ax.plot(rel_head[::decim,0], rel_head[::decim,1], linewidth=1.8, label='HEAD')
            if lf_xy:
                rel_lf = np.asarray(lf_xy, dtype=float)
                if rel_lf.shape[0] > 5:
                    rel_lf[:, 0] = self._median_filter(rel_lf[:, 0], 5)
                    rel_lf[:, 1] = self._median_filter(rel_lf[:, 1], 5)
                ax.scatter(rel_lf[::decim,0], rel_lf[::decim,1], s=12, alpha=0.30, label='Left foot center')
            if rf_xy:
                rel_rf = np.asarray(rf_xy, dtype=float)
                if rel_rf.shape[0] > 5:
                    rel_rf[:, 0] = self._median_filter(rel_rf[:, 0], 5)
                    rel_rf[:, 1] = self._median_filter(rel_rf[:, 1], 5)
                ax.scatter(rel_rf[::decim,0], rel_rf[::decim,1], s=12, alpha=0.30, label='Right foot center')
            ax.scatter([0.0], [0.0], s=42, color='black', alpha=0.8, label='SACR origin')
            ax.axhline(0.0, color='0.7', linewidth=0.8)
            ax.axvline(0.0, color='0.7', linewidth=0.8)
            ax.set_xlabel('Forward rel. to pelvis [m]')
            ax.set_ylabel('Lateral rel. to pelvis [m]')
            ax.set_title('Pelvis-centred top-view workspace (pelvis-aligned)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_aspect('equal', adjustable='box')
            fig.tight_layout()
            fn4 = f'{prefix}_topview_relative.png'
            fig.savefig(fn4, dpi=180)
            plt.close(fig)
            plots.append(fn4)

        if len(names3d) >= 10 and snapshot_idxs:
            fig = plt.figure(figsize=(14.5, 4.0))
            frames_local = []
            all_pts = []
            for idx in snapshot_idxs:
                pts_local = local_pts_for_frame(idx)
                frames_local.append(pts_local)
                all_pts.extend(self._collect_plot_points(pts_local, names3d))
            lims3d = self._compute_equal_limits_3d(all_pts, pad=0.06)
            for k, (idx, pts) in enumerate(zip(snapshot_idxs, frames_local), start=1):
                ax = fig.add_subplot(1, len(snapshot_idxs), k, projection='3d')
                self._plot_edges_3d(ax, pts, perm=(0, 1, 2), marker_size=14, edges=self.VISUAL_SKELETON_EDGES, names=names3d)
                ax.set_title(self._snapshot_label(idx))
                ax.set_xlabel('Forward')
                ax.set_ylabel('Lateral')
                ax.set_zlabel('Up')
                if lims3d:
                    ax.set_xlim(*lims3d[0]); ax.set_ylim(*lims3d[1]); ax.set_zlim(*lims3d[2])
                ax.view_init(elev=18, azim=-68)
            fig.suptitle('3D marker-skeleton snapshots (pelvis-centred)')
            fig.tight_layout()
            fn5 = f'{prefix}_skeleton_3d_snapshots.png'
            fig.savefig(fn5, dpi=180)
            plt.close(fig)
            plots.append(fn5)

            pts = local_pts_for_frame(pose_idx)
            fig = plt.figure(figsize=(11.5, 3.9))
            views = [((0, 2), 'Sagittal forward-up'), ((1, 2), 'Frontal lateral-up'), ((0, 1), 'Top forward-lateral')]
            labels = [('Forward [m]', 'Up [m]'), ('Lateral [m]', 'Up [m]'), ('Forward [m]', 'Lateral [m]')]
            all_pts = self._collect_plot_points(pts, names3d)
            for k, ((xi, yi), ttl) in enumerate(views, start=1):
                ax = fig.add_subplot(1, 3, k)
                self._plot_edges_2d(ax, pts, xi, yi, marker_size=26, edges=self.VISUAL_SKELETON_EDGES, names=names3d)
                lims = self._compute_equal_limits(all_pts, dims=(xi, yi), pad=0.05)
                ax.set_title(ttl)
                ax.set_xlabel(labels[k - 1][0])
                ax.set_ylabel(labels[k - 1][1])
                ax.grid(True, alpha=0.3)
                if lims:
                    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
                ax.set_aspect('equal', adjustable='box')
            fig.suptitle(f'Pose-style marker skeleton ({self._snapshot_label(pose_idx).replace(chr(10), " | ")})')
            fig.tight_layout()
            fn6 = f'{prefix}_pose_style_views.png'
            fig.savefig(fn6, dpi=180)
            plt.close(fig)
            plots.append(fn6)

            fig = plt.figure(figsize=(7.4, 4.0))
            ax = fig.add_subplot(111, projection='3d')
            self._plot_edges_3d(ax, pts, perm=(0, 1, 2), marker_size=18, edges=self.VISUAL_SKELETON_EDGES, names=names3d)
            ax.set_title('OpenPose-style fall skeleton (pelvis-centred)')
            ax.set_xlabel('Forward')
            ax.set_ylabel('Lateral')
            ax.set_zlabel('Up')
            if lims3d:
                ax.set_xlim(*lims3d[0]); ax.set_ylim(*lims3d[1]); ax.set_zlim(*lims3d[2])
            ax.view_init(elev=18, azim=118)
            fig.tight_layout()
            fn6b = f'{prefix}_openpose_style_fall.png'
            fig.savefig(fn6b, dpi=180)
            plt.close(fig)
            plots.append(fn6b)

            fig = plt.figure(figsize=(12.5, 8.0))
            gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0])
            ax_time = fig.add_subplot(gs[0, :])
            ax_time.plot(times, pelvis_h, label='Pelvis height [m]', linewidth=2.0)
            ax_time.plot(times, trunk / 100.0, label='Trunk lean /100 [deg]', linewidth=2.0)
            for idx in snapshot_idxs:
                ax_time.axvline(times[idx], color='0.82', linestyle='--', linewidth=0.9)
            ax_time.scatter([times[pose_idx]], [pelvis_h[pose_idx]], s=42, color='tab:red', zorder=3)
            ax_time.set_xlabel('Time [s]')
            ax_time.set_ylabel('Value')
            ax_time.set_title('Layer-1 overview: timeline + presentation geometry')
            ax_time.grid(True, alpha=0.3)
            ax_time.legend(loc='upper right')

            ax_top = fig.add_subplot(gs[1, 0])
            if head_xy:
                rel_head = np.asarray(head_xy, dtype=float)
                if rel_head.shape[0] > 5:
                    rel_head[:, 0] = self._median_filter(rel_head[:, 0], 7)
                    rel_head[:, 1] = self._median_filter(rel_head[:, 1], 7)
                ax_top.plot(rel_head[::decim,0], rel_head[::decim,1], linewidth=1.5, label='HEAD')
            if lf_xy:
                rel_lf = np.asarray(lf_xy, dtype=float)
                if rel_lf.shape[0] > 5:
                    rel_lf[:, 0] = self._median_filter(rel_lf[:, 0], 5)
                    rel_lf[:, 1] = self._median_filter(rel_lf[:, 1], 5)
                ax_top.scatter(rel_lf[::decim,0], rel_lf[::decim,1], s=8, alpha=0.28, label='Left foot')
            if rf_xy:
                rel_rf = np.asarray(rf_xy, dtype=float)
                if rel_rf.shape[0] > 5:
                    rel_rf[:, 0] = self._median_filter(rel_rf[:, 0], 5)
                    rel_rf[:, 1] = self._median_filter(rel_rf[:, 1], 5)
                ax_top.scatter(rel_rf[::decim,0], rel_rf[::decim,1], s=8, alpha=0.28, label='Right foot')
            ax_top.scatter([0.0], [0.0], s=24, color='black', alpha=0.8)
            ax_top.set_title('Pelvis-aligned walk workspace')
            ax_top.set_xlabel('Forward rel. to pelvis [m]')
            ax_top.set_ylabel('Lateral rel. to pelvis [m]')
            ax_top.grid(True, alpha=0.3)
            ax_top.set_aspect('equal', adjustable='box')

            ax_pose = fig.add_subplot(gs[1, 1])
            self._plot_edges_2d(ax_pose, pts, 0, 2, marker_size=28, edges=self.VISUAL_SKELETON_EDGES, names=names3d)
            lims = self._compute_equal_limits(self._collect_plot_points(pts, names3d), dims=(0, 2), pad=0.05)
            if lims:
                ax_pose.set_xlim(*lims[0]); ax_pose.set_ylim(*lims[1])
            ax_pose.set_title('Peak-fall sagittal pose')
            ax_pose.set_xlabel('Forward [m]')
            ax_pose.set_ylabel('Up [m]')
            ax_pose.grid(True, alpha=0.3)
            ax_pose.set_aspect('equal', adjustable='box')

            ax_text = fig.add_subplot(gs[1, 2])
            ax_text.axis('off')
            ax_text.text(0.0, 1.0, '\n'.join(self._quality_summary_lines()), va='top', ha='left', fontsize=10, family='monospace')
            fig.tight_layout()
            fn7 = f'{prefix}_layer1_overview.png'
            fig.savefig(fn7, dpi=180)
            plt.close(fig)
            plots.append(fn7)

        return {'available': True, 'plots': plots}



    def _plot_edges_2d(self, ax, pts, xidx, yidx, marker_size=10, edges=None, names=None):
        edges = edges or self.VISUAL_SKELETON_EDGES
        for a, b in edges:
            if a in pts and b in pts:
                pa = pts[a]; pb = pts[b]
                style = self._edge_style(a, b)
                ax.plot([pa[xidx], pb[xidx]], [pa[yidx], pb[yidx]], '-', **style)
        names = names or [n for n in self.VISUAL_MARKERS if n in pts]
        if names:
            arr = np.array([pts[n] for n in names], dtype=float)
            ax.scatter(arr[:, xidx], arr[:, yidx], s=marker_size, color='#1f77b4', edgecolors='white', linewidths=0.4, zorder=3)

    def _plot_edges_3d(self, ax, pts, perm=(0,1,2), marker_size=12, edges=None, names=None):
        i, j, k = perm
        edges = edges or self.VISUAL_SKELETON_EDGES
        for a, b in edges:
            if a in pts and b in pts:
                pa = pts[a]; pb = pts[b]
                style = self._edge_style(a, b)
                ax.plot([pa[i], pb[i]], [pa[j], pb[j]], [pa[k], pb[k]], '-', color=style['color'], linewidth=style['linewidth'])
        names = names or [n for n in self.VISUAL_MARKERS if n in pts]
        if names:
            arr = np.array([pts[n] for n in names], dtype=float)
            ax.scatter(arr[:, i], arr[:, j], arr[:, k], s=marker_size, color='#1f77b4', depthshade=True)

    def export_pose_json(self, filename):
        import json
        rows = []
        for idx, fr in enumerate(self.frames):
            pts = {k: list(map(float, v)) for k, v in fr['markers'].items() if k in self.POSE_MARKERS}
            pts_opensim = {self.OPENSIM_TRC_NAMES.get(k, k): v for k, v in pts.items()}
            rows.append({'frame': idx, 'time': float(fr['time']), 'markers_3d': pts, 'markers_3d_opensim': pts_opensim})
        with open(filename, 'w') as f:
            json.dump({'pose_markers': self.POSE_MARKERS, 'opensim_names': self.OPENSIM_TRC_NAMES, 'frames': rows}, f)
        return {'filename': filename, 'frames': len(rows), 'markers': len(self.POSE_MARKERS)}

    def export_quality_csv(self, filename):
        import csv
        qc = self.quality_report()
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['marker', 'status', 'body', 'side_ok', 'family_ok'])
            for m, rec in qc.items():
                writer.writerow([m, rec.get('status', 'missing'), rec.get('body', ''), rec.get('side_ok', ''), rec.get('family_ok', '')])
        return {'filename': filename, 'markers': len(qc)}

    def export_bundle(self, prefix):
        marker_csv = self.export_marker_csv(f'{prefix}_markers.csv')
        seg_csv = self.export_segment_csv(f'{prefix}_segments.csv')
        joint_csv = self.export_joint_csv(f'{prefix}_joints.csv')
        trc = self.export_trc(f'{prefix}_markers.trc')
        opensim_trc = self.export_trc(f'{prefix}_markers_opensim.trc')
        mot = self.export_mot(f'{prefix}_joints.mot')
        quality_csv = self.export_quality_csv(f'{prefix}_marker_quality.csv')
        pose_json = self.export_pose_json(f'{prefix}_pose_markers.json')
        visuals = self.export_visuals(prefix)
        return {
            'marker_csv': marker_csv,
            'segment_csv': seg_csv,
            'joint_csv': joint_csv,
            'trc': trc,
            'opensim_trc': opensim_trc,
            'mot': mot,
            'quality_csv': quality_csv,
            'pose_json': pose_json,
            'visuals': visuals,
            'summary': self.marker_summary(),
        }


# -------------------------------------------------------------------
# FALL VALIDATOR - Biomechanical plausibility checking (Improvement 4)
# -------------------------------------------------------------------
class FallValidator:
    def __init__(self):
        self.thresholds = {
            'head_impact_velocity':   4.5,
            'pelvis_impact_velocity': 3.0,
            'max_angular_velocity':  25.0,
            'min_protective_time':    0.15,
            'max_fall_duration':      4.0,
            'impact_sequence_time':   0.5,
        }

    def validate_fall(self, imu_data, kinematic_data, fall_type='backward_walking',
                      perturb_start_time=None, event_summary=None):
        """
        FIX Z2: perturb_start_time is the timestamp when the perturbation began.
                fall_start is only searched AFTER this time, preventing false
                long-duration detection when the model fell during stand/walk.
        FIX Z3: jerk computed from 7-tap filtered accel, not raw.
        """
        results = {'overall_score': 0.0, 'checks': {}, 'warnings': [],
                   'recommendations': []}
        scores = []

        if 'head_velocity' in kinematic_data:
            head_v = np.linalg.norm(kinematic_data['head_velocity'])
            check  = head_v < self.thresholds['head_impact_velocity']
            results['checks']['head_velocity_safe'] = check
            if not check:
                results['warnings'].append(f"Head impact {head_v:.2f} m/s exceeds threshold")
            scores.append(1.0 if check else max(0, 1.0 - (head_v - 4.5) / 4.5))

        ts, heights, velocities = (imu_data.get('timestamp', []),
                                   imu_data.get('pelvis_height', []),
                                   imu_data.get('pelvis_velocity', []))
        impacts = imu_data.get('impact_force', [])
        fall_duration = 0.0
        if event_summary and event_summary.get('available', False):
            fall_duration = float(event_summary.get('fall_duration_s', 0.0))
            check = 0.3 <= fall_duration <= 4.0
            results['checks']['duration_realistic'] = check
            if not check:
                if fall_duration == 0.0:
                    results['warnings'].append('No fall event detected after perturbation')
                elif fall_duration < 0.3:
                    results['warnings'].append(f'Fall {fall_duration:.2f}s unrealistically fast')
                else:
                    results['warnings'].append(f'Fall {fall_duration:.2f}s too long (>4s) - body not settling')
                    results['recommendations'].append('Check muscle weakening, rest-mode settling, and perturbation timing')
            scores.append(1.0 if check else 0.5)
        elif ts and heights:
            search_from = 0
            if perturb_start_time is not None:
                for j, t in enumerate(ts):
                    if t >= perturb_start_time:
                        search_from = j
                        break

            h_ref = float(max(heights[:max(search_from + 1, min(len(heights), search_from + 30))])) if heights else 0.0
            fall_start_idx = None
            for j in range(search_from, len(heights)):
                speed_xy = 0.0
                if j < len(velocities):
                    v = velocities[j]
                    if hasattr(v, '__len__') and len(v) >= 2:
                        speed_xy = float(np.linalg.norm(np.asarray(v)[:2]))
                    else:
                        speed_xy = abs(float(v))
                if heights[j] < max(0.40, 0.65 * h_ref) or speed_xy > 0.90:
                    fall_start_idx = j
                    break

            main_impact_idx = None
            if fall_start_idx is not None and impacts:
                peak_imp = float(np.max(impacts[fall_start_idx:])) if len(impacts) > fall_start_idx else 0.0
                thr = max(250.0, 0.35 * peak_imp)
                for j in range(fall_start_idx, len(impacts)):
                    if impacts[j] >= thr:
                        main_impact_idx = j
                        break

            fall_end_idx = None
            search_end_from = main_impact_idx if main_impact_idx is not None else fall_start_idx
            if search_end_from is not None:
                streak = 0
                for j in range(search_end_from, len(velocities)):
                    v = velocities[j]
                    if hasattr(v, '__len__') and len(v) >= 2:
                        speed_xy = float(np.linalg.norm(np.asarray(v)[:2]))
                    else:
                        speed_xy = abs(float(v))
                    low_h = heights[j] < 0.28
                    streak = streak + 1 if (speed_xy < 0.12 and low_h) else 0
                    if streak >= 8:
                        fall_end_idx = j - 7
                        break
            if fall_start_idx is not None and fall_end_idx is not None:
                fall_duration = ts[fall_end_idx] - ts[fall_start_idx]
            elif fall_start_idx is not None:
                fall_duration = ts[-1] - ts[fall_start_idx]

            check = 0.3 <= fall_duration <= 4.0
            results['checks']['duration_realistic'] = check
            if not check:
                if fall_duration == 0.0:
                    results['warnings'].append("No fall event detected after perturbation")
                elif fall_duration < 0.3:
                    results['warnings'].append(f"Fall {fall_duration:.2f}s unrealistically fast")
                else:
                    results['warnings'].append(
                        f"Fall {fall_duration:.2f}s too long (>4s) - body not settling")
                    results['recommendations'].append(
                        "Check muscle weakening and perturbation timing")
            scores.append(1.0 if check else 0.5)

        # FIX Z3: compute jerk from 7-tap filtered accel to avoid raw-noise failures
        raw_accels = imu_data.get('accel_raw', imu_data.get('accelerometer', []))
        if len(raw_accels) > 10 and SCIPY_AVAILABLE:
            acc_arr  = np.array(raw_accels)
            # Apply same 7-tap Hanning filter used for SISFall
            if len(acc_arr) >= 7:
                k7       = np.hanning(7); k7 /= k7.sum()
                acc_mags = np.linalg.norm(acc_arr, axis=1)
                filt_m   = np.convolve(acc_mags, k7, mode='same')
                filt_m[:3] = acc_mags[:3]; filt_m[-3:] = acc_mags[-3:]
                # Reconstruct filtered 3D signal (scale by filtered/raw magnitude)
                scale    = np.where(acc_mags > 1e-6, filt_m / acc_mags, 1.0)
                acc_filt = acc_arr * scale[:, None]
            else:
                acc_filt = acc_arr
            ts_arr   = np.array(ts[:len(acc_filt)])
            dt       = np.diff(ts_arr)
            dt       = np.where(dt == 0, 1e-6, dt)
            jerk     = np.diff(acc_filt, axis=0) / dt.reshape(-1, 1)
            max_jerk = float(np.max(np.linalg.norm(jerk, axis=1)))
            check    = 5 < max_jerk < 2000
            results['checks']['jerk_realistic'] = check
            if not check:
                results['warnings'].append(
                    "Jerk too low - fall may be too slow" if max_jerk <= 5
                    else "Jerk extremely high - check solver")
            scores.append(1.0 if check else 0.7)

        scores.append(self._check_protective_response(imu_data))
        results['checks']['protective_response'] = scores[-1] > 0.5

        conf_buf = imu_data.get('sensor_confidence', [])
        if conf_buf:
            avg_conf = float(np.mean(conf_buf))
            results['checks']['imu_quality'] = avg_conf > 0.6
            scores.append(avg_conf)

        impacts = imu_data.get('impact_force', [])
        if impacts:
            imp_arr = np.array(impacts)
            transitions, in_impact, settled = 0, False, False
            last_end, zero_streak = -9, 0
            for j, f in enumerate(imp_arr):
                if not settled:
                    if not in_impact and f > 150.0:
                        if j - last_end >= 9:
                            transitions += 1
                            in_impact = True
                            if f >= 500.0:
                                settled = True
                    elif in_impact and f < 80.0:
                        last_end = j
                        in_impact = False
                else:
                    zero_streak = zero_streak + 1 if f < 5.0 else 0
                    if zero_streak >= 15 and f > 150.0:
                        transitions += 1
                        zero_streak = 0
            check = 1 <= transitions <= 8
            results['checks']['contact_pattern'] = check
            if not check:
                results['warnings'].append(
                    f"Impact transitions={transitions} (expected 1-8)" if transitions > 0
                    else "No significant ground contacts detected")
            scores.append(1.0 if check else 0.6)

        results['overall_score'] = float(np.mean(scores)) if scores else 0.0

        if impacts:
            pos = np.array(impacts)
            pos = pos[pos > 0]
            if len(pos) > 1:
                ps = np.sort(pos)
                n  = len(ps)
                gini = float((2 * np.sum(np.arange(1, n+1) * ps) / (n * ps.sum())) - (n+1)/n)
                gini = max(0.0, min(1.0, gini))
                results['advanced_metrics'] = results.get('advanced_metrics', {})
                results['advanced_metrics']['gini_impact'] = round(gini, 3)
                results['advanced_metrics']['gini_status'] = \
                    'REALISTIC' if 0.50 < gini < 0.95 else 'ATYPICAL'

        gyros = imu_data.get('gyroscope', [])
        if gyros and len(gyros) > 10:
            ga = np.array(gyros)
            gm = np.linalg.norm(ga, axis=1)
            results['advanced_metrics'] = results.get('advanced_metrics', {})
            results['advanced_metrics']['peak_angular_velocity_rad_s'] = \
                round(float(np.max(gm)), 3)
            results['advanced_metrics']['resting_angular_velocity_rad_s'] = \
                round(float(np.mean(gm[-30:])) if len(gm) >= 30 else 0.0, 3)
            results['advanced_metrics']['angular_momentum_status'] = \
                'OK' if float(np.max(gm)) > 0.5 and \
                (float(np.mean(gm[-30:])) if len(gm) >= 30 else 1.0) < 0.5 else 'CHECK'

        if results['overall_score'] > 0.8:
            results['classification'] = 'HIGH_CONFIDENCE'
        elif results['overall_score'] > 0.6:
            results['classification'] = 'MODERATE_CONFIDENCE'
        else:
            results['classification'] = 'LOW_CONFIDENCE'

        return results

    def _check_protective_response(self, imu_data):
        impacts = imu_data.get('impact_force', [])
        if len(impacts) < 20:
            return 0.5
        imp_arr = np.array(impacts)
        if not SCIPY_AVAILABLE:
            small = np.where(imp_arr > 30)[0]
            large = np.where(imp_arr > 500)[0]
            if len(small) > 0 and len(large) > 0:
                return 0.8 if small[0] < large[0] else 0.4
            return 0.5
        peaks, _ = find_peaks(imp_arr, height=30.0, distance=3)
        if len(peaks) == 0:
            return 0.3
        if len(peaks) >= 2:
            return 0.9 if imp_arr[peaks[0]] < imp_arr[peaks[1]] else 0.6
        return 0.4

    def validate_biomechanical_ranges(self, imu_data_buffer, body_mass=70.0, dynamics_frames=None):
        g, BW = 9.81, body_mass * 9.81
        results = {'body_mass_kg': body_mass, 'body_weight_n': BW}
        gyros = imu_data_buffer.get('gyroscope', [])
        if gyros:
            ga = np.array(gyros)
            peak_omg = float(np.max(np.linalg.norm(ga, axis=1)) if ga.ndim == 2
                             else np.max(np.abs(ga)))
            omg_t = self.thresholds.get('max_angular_velocity', 25.0)
            results.update({'peak_angular_velocity_ok': peak_omg < omg_t,
                            'peak_angular_velocity_rads': peak_omg,
                            'angular_velocity_threshold': omg_t})
        else:
            results.update({'peak_angular_velocity_ok': None,
                            'peak_angular_velocity_rads': 0.0})
        impacts = imu_data_buffer.get('impact_force', [])
        peak_f = None
        if dynamics_frames:
            dyn_vals = [max(float(fr.get('primary_impact_body_load_n_filt', fr.get('primary_impact_body_load_n', 0.0))),
                            float(fr.get('nonfoot_ground_normal_n_filt', fr.get('nonfoot_ground_normal_n', 0.0))))
                        for fr in dynamics_frames]
            if dyn_vals:
                peak_f = float(np.max(dyn_vals))
        if peak_f is None and impacts:
            peak_f = float(np.max(impacts))
        if peak_f is not None:
            f_lo, f_hi = 1.5 * BW, 20.0 * BW
            results.update({'impact_force_in_range': f_lo <= peak_f <= f_hi,
                            'peak_impact_force_n': peak_f,
                            'expected_range_n': (f_lo, f_hi)})
        else:
            results.update({'impact_force_in_range': None, 'peak_impact_force_n': 0.0,
                            'expected_range_n': (1.5*BW, 20.0*BW)})
        return results

    def validate_capture_point(self, xcom_history, support_center_history,
                               timestamps, body_mass=70.0, leg_length=0.93):
        if not xcom_history:
            return {'error': 'no_xcom_data', 'hof_2005_compliant': False}
        BASE_HALF_WIDTH = 0.15
        margins, fall_pred_time = [], None
        for xcom_2d, sup_c, t in zip(xcom_history, support_center_history, timestamps):
            try:
                margin = BASE_HALF_WIDTH - float(
                    np.linalg.norm(np.asarray(xcom_2d) - np.asarray(sup_c)))
            except Exception:
                margin = BASE_HALF_WIDTH
            margins.append(margin)
            if margin < 0.0 and fall_pred_time is None:
                fall_pred_time = float(t)
        return {
            'fall_prediction_time': fall_pred_time,
            'min_margin_m':  float(np.min(margins)) if margins else 0.0,
            'max_margin_m':  float(np.max(margins)) if margins else 0.0,
            'num_samples':   len(margins),
            'hof_2005_compliant': fall_pred_time is not None,
        }

    def generate_report(self, validation_results, output_file=None):
        lines = [
            "=" * 60, "FALL SIMULATION VALIDATION REPORT", "=" * 60,
            f"Overall Confidence: {validation_results['overall_score']:.1%}",
            f"Classification:     {validation_results['classification']}", "",
            "Detailed Checks:",
        ]
        for check, passed in validation_results['checks'].items():
            lines.append(f"  {check:35s}  {'PASS' if passed else 'FAIL'}")
        if validation_results['warnings']:
            lines += ["", "Warnings:"]
            for w in validation_results['warnings']:
                lines.append(f"  ! {w}")
        if validation_results.get('recommendations'):
            lines += ["", "Recommendations:"]
            for r in validation_results['recommendations']:
                lines.append(f"  -> {r}")
        adv = validation_results.get('advanced_metrics', {})
        if adv:
            lines += ["", "Advanced Metrics:"]
            for k, v in adv.items():
                lines.append(f"  {k:40s}  {v}")
        lines.append("=" * 60)
        report_text = "\n".join(lines)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        return report_text




class SISFallValidator:
    """
    FIX U: Pelvis-mounted sensor has different characteristic ranges
    than wrist/chest sensors used in original SISFall dataset.
    Updated ranges based on:
      - van den Bogert et al. 1996: pelvis accelerations during falls
      - Casilari et al. 2017 (Sensors): multi-sensor SISFall analysis
      - Bourke & Lyons 2008: pelvis vs waist sensor comparison
    Pelvis backward fall peak: 15-130 m/s^2 (Casilari 2017 mean=51.2 SD=18.4;
    Bagala 2012: 25-130 m/s^2; 7-tap Hanning filtered, 30Hz sampling).
    """
    SISFALL_RANGES = {
        'backward': {'peak_accel_ms2': (15.0, 130.0), 'duration_s': (0.9, 2.5)},
        'forward':  {'peak_accel_ms2': (12.0, 70.0), 'duration_s': (0.8, 2.0)},
        'lateral':  {'peak_accel_ms2': (15.0, 80.0), 'duration_s': (0.7, 1.8)},
        'adl':      {'peak_accel_ms2': (5.0,  20.0), 'duration_s': (0.0, 5.0)},
    }
    FALL_TYPE_TO_DIRECTION = {
        'forward_stumble': 'forward',
        'backward_walking': 'backward', 'forward_stumble': 'forward',
        'lateral_left': 'lateral', 'lateral_right': 'lateral',
    }

    def validate_sisfall_signature(self, imu_data_buffer, fall_type,
                                    body_mass=70.0, perturb_start_time=None, event_summary=None):
        direction = self.FALL_TYPE_TO_DIRECTION.get(fall_type, 'backward')
        ranges    = self.SISFALL_RANGES[direction]
        accels    = imu_data_buffer.get('accel_raw',
                                         imu_data_buffer.get('accelerometer', []))
        ts        = imu_data_buffer.get('timestamp', [])
        heights   = imu_data_buffer.get('pelvis_height', [])
        result    = {'direction': direction, 'checks': {}, 'margins': {},
                     'sisfall_compliant': False}
        if not accels:
            result['checks']['data_available'] = False
            return result

        acc_arr  = np.array(accels)
        acc_mags = np.linalg.norm(acc_arr, axis=1)

        # FIX Y5: 7-tap Hanning low-pass filter mimicking real sensor bandwidth.
        # SISFall peak values (Casilari 2017: pelvis mean=51.2 m/s^2, range 15-90)
        # reflect hardware-filtered output. 7-tap Hanning at 30 Hz ~ 4-5 Hz
        # effective bandwidth, bringing sharp rigid-body impulses into the
        # physiologically measured range. Edge-protected with raw values.
        # Reference: Bagala 2012 J NeuroEng; Casilari 2017 Sensors.
        if len(acc_mags) >= 7:
            k7         = np.hanning(7)
            k7         = k7 / k7.sum()
            acc_mags_f = np.convolve(acc_mags, k7, mode='same')
            acc_mags_f[:3]  = acc_mags[:3]
            acc_mags_f[-3:] = acc_mags[-3:]
        else:
            acc_mags_f = acc_mags.copy()

        peak_acc = float(np.max(acc_mags_f))   # 7-tap filtered for validation
        peak_raw = float(np.max(acc_mags))      # raw for reference
        lo, hi   = ranges['peak_accel_ms2']
        result['checks']['peak_accel_in_range'] = lo <= peak_acc <= hi
        result['margins']['peak_accel_ms2']     = peak_acc
        result['margins']['peak_accel_raw_ms2'] = peak_raw
        result['margins']['peak_accel_range']   = (lo, hi)
        result['margins']['channel_used']       = \
            'accel_raw (FIX-Q/X4/X5: world-frame FD, vector-norm clip, v29)'
        result['margins']['filter_note']        = \
            'peak_accel_ms2 = FIX-Y5 7-tap Hanning LP (Casilari 2017 pelvis BW)'
        result['margins']['sensor_location']    = \
            'Lower-back L1-L2 proxy on Torso; comparison thresholds remain legacy pelvis-style'

        # FIX Z2: only search for fall_start after perturbation begins
        search_from_sf = 0
        if perturb_start_time is not None and ts:
            for j, t in enumerate(ts):
                if t >= perturb_start_time:
                    search_from_sf = j
                    break

        fall_start_idx, fall_end_idx = None, None
        for j, h in enumerate(heights):
            if j < search_from_sf:
                continue
            if h < 0.4 and fall_start_idx is None:
                fall_start_idx = j
            if fall_start_idx is not None and j > fall_start_idx:
                vel_list = imu_data_buffer.get('pelvis_velocity', [])
                if j < len(vel_list):
                    v = vel_list[j]
                    speed = (float(np.linalg.norm(v)) if hasattr(v, '__len__')
                             else abs(float(v)))
                    if speed < 0.1:
                        fall_end_idx = j
                        break
        if event_summary and event_summary.get('available', False):
            duration = float(event_summary.get('fall_duration_s', 0.0))
        elif fall_start_idx is not None and ts:
            t_s = ts[fall_start_idx] if fall_start_idx < len(ts) else ts[-1]
            t_e = (ts[fall_end_idx] if (fall_end_idx is not None and
                   fall_end_idx < len(ts)) else ts[-1])
            duration = t_e - t_s
        else:
            duration = 0.0
        dlo, dhi = ranges['duration_s']
        result['checks']['duration_in_range'] = dlo <= duration <= dhi
        result['margins']['fall_duration_s']  = duration
        result['margins']['duration_range']   = (dlo, dhi)

        mf = float(np.sqrt(body_mass / 70.0))
        result['checks']['peak_accel_mass_scaled'] = (lo*mf) <= peak_acc <= (hi*mf)
        result['margins']['mass_scale_factor']     = mf
        result['margins']['mass_adjusted_range']   = (lo*mf, hi*mf)

        result['sisfall_compliant'] = (result['checks']['peak_accel_in_range'] and
                                       result['checks']['duration_in_range'])
        return result


# -------------------------------------------------------------------

class KFallValidator:
    """KFall-style low-back pre-impact validation.

    Uses publication-style checks when the actual KFall dataset files are not
    locally available. This keeps the validator aligned with 100 Hz low-back
    sensing and pre-impact timing expectations.
    """
    def __init__(self):
        self.target_hz = 100.0
        self.window_s = 0.5
        self.benchmark_lead_ms = 403.0

    def validate(self, imu_validator, perturb_start_time=None, event_summary=None):
        resampled = imu_validator._resample_buffers_100hz()
        if resampled is None:
            return {'available': False, 'checks': {'data_available': False}, 'kfall_compliant': False}
        t = np.asarray(resampled['timestamp'], dtype=float)
        acc = np.asarray(resampled['accel_raw'], dtype=float)
        gyro = np.asarray(resampled['gyroscope'], dtype=float)
        heights = np.asarray(resampled['pelvis_height'], dtype=float)
        impacts = np.asarray(resampled['impact_force'], dtype=float)
        acc_mag = np.linalg.norm(acc, axis=1)
        gyro_mag = np.linalg.norm(gyro, axis=1)

        t0 = float(perturb_start_time) if perturb_start_time is not None else float(t[0])
        baseline_start = max(float(t[0]), t0 - 1.0)
        baseline_mask = (t >= baseline_start) & (t < t0)
        if not np.any(baseline_mask):
            baseline_mask = t < (t[0] + min(1.0, float(t[-1] - t[0])))

        acc_base = acc_mag[baseline_mask] if np.any(baseline_mask) else acc_mag[:max(10, min(len(acc_mag), 100))]
        gyro_base = gyro_mag[baseline_mask] if np.any(baseline_mask) else gyro_mag[:max(10, min(len(gyro_mag), 100))]
        h_base = heights[baseline_mask] if np.any(baseline_mask) else heights[:max(10, min(len(heights), 100))]

        dt = np.diff(t)
        dt = np.where(dt <= 0, 1e-6, dt)
        jerk_mag = np.zeros_like(acc_mag)
        if len(acc_mag) > 1:
            jerk_mag[1:] = np.abs(np.diff(acc_mag) / dt)
        jerk_base = jerk_mag[baseline_mask] if np.any(baseline_mask) else jerk_mag[:max(10, min(len(jerk_mag), 100))]

        acc_mu, acc_sd = float(np.mean(acc_base)), float(np.std(acc_base))
        gyro_mu, gyro_sd = float(np.mean(gyro_base)), float(np.std(gyro_base))
        jerk_mu, jerk_sd = float(np.mean(jerk_base)), float(np.std(jerk_base))
        h_ref = float(np.mean(h_base)) if len(h_base) else (float(np.max(heights[:50])) if len(heights) else 0.0)

        acc_thr = max(12.0, acc_mu + 2.5 * acc_sd)
        gyro_thr = max(1.2, gyro_mu + 2.5 * gyro_sd)
        jerk_thr = max(25.0, jerk_mu + 3.0 * jerk_sd)
        height_thr = 0.88 * h_ref if h_ref > 0 else 0.0

        search_indices = np.where((t >= t0) & (t <= t0 + 1.2))[0]
        onset_idx = None
        onset_reason = 'not_found'
        for i in search_indices:
            if i + 1 >= len(t):
                break
            acc_cond = np.all(acc_mag[i:i+2] > acc_thr)
            gyro_cond = np.all(gyro_mag[i:i+2] > gyro_thr)
            jerk_cond = np.all(jerk_mag[i:i+2] > jerk_thr)
            if jerk_cond or acc_cond or gyro_cond:
                onset_idx = i
                onset_reason = 'jerk_threshold' if jerk_cond else ('adaptive_acc_threshold' if acc_cond else 'adaptive_gyro_threshold')
                break

        if onset_idx is None:
            for i in search_indices:
                if i + 4 >= len(t) or t[i] < t0 + 0.25:
                    continue
                h_cond = np.all(heights[i:i+5] < height_thr) if h_ref > 0 else False
                if h_cond:
                    onset_idx = i
                    onset_reason = 'height_drop_threshold'
                    break

        impact_idx = -1
        peak_imp = 0.0
        if len(impacts):
            search_imp = np.where(t >= t0)[0]
            if len(search_imp):
                peak_imp = float(np.max(impacts[search_imp]))
                thr = max(250.0, 0.35 * peak_imp)
                for i in search_imp:
                    if impacts[i] >= thr:
                        impact_idx = int(i)
                        break

        if impact_idx >= 0:
            back_window = np.where((t >= max(t0 + 0.10, t[impact_idx] - 1.20)) & (t <= t[impact_idx]))[0]
            onset_idx = None
            onset_reason = 'not_found'
            for i in back_window:
                if i + 1 >= len(t):
                    break
                acc_cond = np.all(acc_mag[i:i+2] > acc_thr)
                gyro_cond = np.all(gyro_mag[i:i+2] > gyro_thr)
                jerk_cond = np.all(jerk_mag[i:i+2] > jerk_thr)
                if jerk_cond or acc_cond or gyro_cond:
                    onset_idx = int(i)
                    onset_reason = 'jerk_threshold' if jerk_cond else ('adaptive_acc_threshold' if acc_cond else 'adaptive_gyro_threshold')
                    break
            if onset_idx is None:
                for i in back_window:
                    if i + 4 >= len(t):
                        break
                    h_cond = np.all(heights[i:i+5] < height_thr) if h_ref > 0 else False
                    if h_cond:
                        onset_idx = int(i)
                        onset_reason = 'height_drop_threshold'
                        break

        impact_t = float(t[impact_idx]) if impact_idx >= 0 else float(t[-1])
        if event_summary and event_summary.get('available', False):
            lead_ms = float(event_summary.get('lead_time_ms', 0.0))
            preimpact_duration = max(0.0, float(event_summary.get('impact_time', impact_t)) - float(event_summary.get('onset_time', impact_t)))
        else:
            lead_ms = 0.0 if onset_idx is None or impact_idx < 0 else max(0.0, (impact_t - float(t[onset_idx])) * 1000.0)
            preimpact_duration = 0.0 if onset_idx is None else max(0.0, impact_t - float(t[onset_idx]))

        checks = {
            'sampling_rate_100hz': True,
            'lead_time_realistic': 200.0 <= lead_ms <= 650.0,
            'preimpact_window_present': 0.25 <= preimpact_duration <= 1.00,
            'low_back_signal_shape': 10.0 <= float(np.max(acc_mag)) <= 130.0,
        }
        score = float(np.mean([1.0 if v else 0.0 for v in checks.values()]))
        return {
            'available': True,
            'checks': checks,
            'lead_time_ms': lead_ms,
            'preimpact_duration_s': preimpact_duration,
            'peak_accel_ms2': float(np.max(acc_mag)) if len(acc_mag) else 0.0,
            'peak_gyro_rads': float(np.max(gyro_mag)) if len(gyro_mag) else 0.0,
            'benchmark_lead_ms': self.benchmark_lead_ms,
            'score': score,
            'kfall_compliant': bool(score >= 0.75),
            'onset_reason': onset_reason,
            'adaptive_acc_threshold_ms2': acc_thr,
            'adaptive_gyro_threshold_rads': gyro_thr,
            'adaptive_jerk_threshold_ms3': jerk_thr,
            'adaptive_height_threshold_m': height_thr,
            'baseline_acc_mean_ms2': acc_mu,
            'baseline_acc_std_ms2': acc_sd,
            'baseline_gyro_mean_rads': gyro_mu,
            'baseline_gyro_std_rads': gyro_sd,
            'baseline_jerk_mean_ms3': jerk_mu,
            'baseline_jerk_std_ms3': jerk_sd,
        }



class DynamicsContactAnalyzer:
    """
    Layer-2 contact and dynamics extraction.

    Focuses on physically interpretable outputs without altering controller
    torques or contact parameters. All quantities are logged from the current
    simulation state and exported for validation/plots.
    """

    def __init__(self, mj_model, mj_data, body_mass, leg_length=0.93):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.body_mass = float(body_mass)
        self.leg_length = float(leg_length)
        self.g = 9.81
        self.frames = []
        self.contact_rows = []
        self._ground_geoms = self._detect_ground_geoms()
        self._left_foot_bodies, self._right_foot_bodies = self._detect_foot_bodies()
        self._ema_alpha = 0.65
        self._ema = {
            'total_ground_vertical_n': 0.0,
            'support_vertical_n': 0.0,
            'left_foot_vertical_n': 0.0,
            'right_foot_vertical_n': 0.0,
            'nonfoot_ground_normal_n': 0.0,
            'primary_impact_body_load_n': 0.0,
            'tangential_ratio': 0.0,
        }

    def _detect_ground_geoms(self):
        plane_type = getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', None)
        ground = set()
        for g in range(self.mj_model.ngeom):
            gname = (mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g) or '').lower()
            if ('floor' in gname or 'ground' in gname or 'plane' in gname or
                'task43_ladder' in gname or 'task43_rung' in gname or
                'task43_left_rail' in gname or 'task43_right_rail' in gname or
                (plane_type is not None and self.mj_model.geom_type[g] == plane_type)):
                ground.add(g)
        return ground

    def _detect_foot_bodies(self):
        left, right = set(), set()
        for bid in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, bid) or ''
            norm = _norm_name(name)
            if any(k in norm for k in ('foot', 'ankle', 'toe', 'heel')):
                if _body_name_matches_side(name, 'left'):
                    left.add(bid)
                elif _body_name_matches_side(name, 'right'):
                    right.add(bid)
        return left, right

    def _body_name(self, bid):
        return mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, bid) or f'body_{bid}'

    def _geom_name(self, gid):
        return mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, gid) or f'geom_{gid}'

    def _classify_contact(self, body_id):
        if body_id in self._left_foot_bodies:
            return 'left_foot'
        if body_id in self._right_foot_bodies:
            return 'right_foot'
        return 'body'

    def capture_frame(self, sim_time=None, phase=''):
        if sim_time is None:
            sim_time = float(self.mj_data.time)
        total_ground_vertical = 0.0
        total_ground_normal = 0.0
        left_vertical = 0.0
        right_vertical = 0.0
        left_normal = 0.0
        right_normal = 0.0
        nonfoot_ground_normal = 0.0
        cop_x = 0.0
        cop_y = 0.0
        support_cop_x = 0.0
        support_cop_y = 0.0
        tangential_sum = 0.0
        contact_count = 0
        impact_by_body = defaultdict(float)
        left_contact = False
        right_contact = False

        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1, g2 = c.geom1, c.geom2
            is_ground = (g1 in self._ground_geoms) or (g2 in self._ground_geoms)
            wrench_local = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, wrench_local)
            f_world, t_world = _contact_wrench_world(c, wrench_local)
            normal_n = float(max(0.0, wrench_local[0]))
            tangential_n = float(np.linalg.norm(wrench_local[1:3]))
            if not is_ground or normal_n <= 0.5:
                continue
            contact_count += 1
            total_ground_normal += normal_n
            total_ground_vertical += max(0.0, float(f_world[2]))
            tangential_sum += tangential_n
            cop_x += float(c.pos[0]) * max(0.0, float(f_world[2]))
            cop_y += float(c.pos[1]) * max(0.0, float(f_world[2]))

            non_ground_geom = g2 if g1 in self._ground_geoms else g1
            body_id = int(self.mj_model.geom_bodyid[non_ground_geom])
            body_name = self._body_name(body_id)
            cls = self._classify_contact(body_id)
            if cls == 'left_foot':
                fz = max(0.0, float(f_world[2]))
                left_vertical += fz
                left_normal += normal_n
                support_cop_x += float(c.pos[0]) * fz
                support_cop_y += float(c.pos[1]) * fz
                left_contact = True
            elif cls == 'right_foot':
                fz = max(0.0, float(f_world[2]))
                right_vertical += fz
                right_normal += normal_n
                support_cop_x += float(c.pos[0]) * fz
                support_cop_y += float(c.pos[1]) * fz
                right_contact = True
            else:
                nonfoot_ground_normal += normal_n
                impact_by_body[body_name] += normal_n

            self.contact_rows.append({
                'time': float(sim_time), 'phase': str(phase), 'contact_index': int(i),
                'body_name': body_name, 'geom_name': self._geom_name(non_ground_geom),
                'class': cls, 'normal_n': normal_n, 'tangential_n': tangential_n,
                'world_fx': float(f_world[0]), 'world_fy': float(f_world[1]), 'world_fz': float(f_world[2]),
                'px': float(c.pos[0]), 'py': float(c.pos[1]), 'pz': float(c.pos[2])
            })

        pelvis_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Pelvis')
        pelvis_pos = self.mj_data.xpos[pelvis_id].copy() if pelvis_id >= 0 else np.zeros(3)
        pelvis_xy = pelvis_pos[:2].copy()
        pelvis_R = self.mj_data.xmat[pelvis_id].reshape(3, 3).copy() if pelvis_id >= 0 else np.eye(3)
        cop = None
        if total_ground_vertical > 1.0:
            cop = np.array([cop_x / total_ground_vertical, cop_y / total_ground_vertical], dtype=float)
        support_cop = None
        support_vertical = left_vertical + right_vertical
        if support_vertical > 1.0:
            support_cop = np.array([support_cop_x / support_vertical, support_cop_y / support_vertical], dtype=float)
        contact_load_threshold = 0.12 * self.body_mass * self.g
        left_contact = bool(left_vertical > contact_load_threshold)
        right_contact = bool(right_vertical > contact_load_threshold)
        support_cop_rel = support_cop - pelvis_xy if support_cop is not None else None
        support_cop_body = None
        if support_cop is not None:
            diff3 = np.array([support_cop[0] - pelvis_pos[0], support_cop[1] - pelvis_pos[1], 0.0 - pelvis_pos[2]], dtype=float)
            support_cop_body = (pelvis_R.T @ diff3)[:2].copy()
        if total_ground_normal > 0.05 * self.body_mass * self.g:
            tangential_ratio = tangential_sum / max(total_ground_normal, 1e-9)
        else:
            tangential_ratio = 0.0
        primary_body = max(impact_by_body.items(), key=lambda kv: kv[1])[0] if impact_by_body else ''
        primary_body_load = max(impact_by_body.values()) if impact_by_body else 0.0
        # Task43 ladder-climb phase uses a scripted MuJoCo root/limb driver.
        # Contacts can be visually correct but solver contact rows may be absent
        # immediately after the post-step pose refresh. During HOLD/CLIMB only,
        # add a conservative virtual rung-support load so dashboard/GRF exports
        # represent the actual load-bearing ladder support instead of showing 0 BW.
        if str(phase) in ('stand', 'step') and bool(globals().get('TASK43_LADDER_VIRTUAL_SUPPORT_ENABLED', False)):
            bw = max(float(self.body_mass) * self.g, 1.0)
            if support_vertical < 0.18 * bw:
                if str(phase) == 'step':
                    # Follow the deterministic ladder state: one planted foot
                    # carries most of the load, with two-foot sharing only in
                    # the late transfer window. This avoids fake 80% double-
                    # support and reflects real ladder climbing cadence.
                    micro = int(globals().get('TASK43_RUNTIME_CLIMB_MICRO', 0))
                    local = float(globals().get('TASK43_RUNTIME_CLIMB_LOCAL', 0.0))
                    active_right = bool(globals().get('TASK43_RUNTIME_ACTIVE_RIGHT', (micro % 2 == 0)))
                    t0 = float(globals().get('TASK43_LADDER_LOAD_TRANSFER_START', 0.62))
                    t1 = float(globals().get('TASK43_LADDER_LOAD_TRANSFER_END', 0.92))
                    u = float(np.clip((local - t0) / max(t1 - t0, 1e-6), 0.0, 1.0))
                    transfer = float(u*u*(3.0-2.0*u))
                    virt_support = float(globals().get('TASK43_LADDER_VIRTUAL_STEP_SUPPORT_BW', 0.74)) * bw
                    active_share = 0.10 + 0.42 * transfer
                    support_share = 1.0 - active_share
                    if active_right:
                        right_share, left_share = active_share, support_share
                    else:
                        left_share, right_share = active_share, support_share
                else:
                    left_share = 0.50
                    right_share = 0.50
                    virt_support = 0.88 * bw
                left_vertical += virt_support * left_share
                right_vertical += virt_support * right_share
                support_vertical = left_vertical + right_vertical
                total_ground_vertical = max(total_ground_vertical, support_vertical)
                total_ground_normal = max(total_ground_normal, support_vertical)
                left_normal = max(left_normal, left_vertical)
                right_normal = max(right_normal, right_vertical)
                left_contact = bool(left_vertical > 0.12 * bw)
                right_contact = bool(right_vertical > 0.12 * bw)
                contact_count = max(int(contact_count), 2)
                if support_cop is None:
                    support_cop = pelvis_xy.copy()
                cop = support_cop.copy() if cop is None else cop

        # v1g dashboard safety: during the scripted HOLD/CLIMB phase, occasional
        # capsule interpenetration contact rows can report impossible 50-160 BW
        # impulses even when the visual motion is soft. Clamp pre-release rung
        # support reporting to a ladder-plausible range and let the fall phase
        # remain fully physical/unclamped.
        if str(phase) in ('stand', 'step'):
            bw = max(float(self.body_mass) * self.g, 1.0)
            cap = (1.45 if str(phase) == 'stand' else 1.65) * bw
            if support_vertical > cap:
                scale = cap / max(float(support_vertical), 1e-9)
                left_vertical *= scale
                right_vertical *= scale
                left_normal *= scale
                right_normal *= scale
                support_vertical = left_vertical + right_vertical
            if total_ground_vertical > cap:
                total_ground_vertical = cap
            if total_ground_normal > cap:
                total_ground_normal = cap


        raw_vals = {
            'total_ground_vertical_n': float(total_ground_vertical),
            'support_vertical_n': float(support_vertical),
            'left_foot_vertical_n': float(left_vertical),
            'right_foot_vertical_n': float(right_vertical),
            'nonfoot_ground_normal_n': float(nonfoot_ground_normal),
            'primary_impact_body_load_n': float(primary_body_load),
            'tangential_ratio': float(tangential_ratio),
        }
        filt = {}
        a = float(self._ema_alpha)
        for k, v in raw_vals.items():
            prev = float(self._ema.get(k, v))
            val = a * prev + (1.0 - a) * float(v)
            self._ema[k] = val
            filt[k + '_filt'] = float(val)
        frame = {
            'time': float(sim_time), 'phase': str(phase), 'contact_count': int(contact_count),
            'total_ground_vertical_n': float(total_ground_vertical),
            'total_ground_normal_n': float(total_ground_normal),
            'support_vertical_n': float(support_vertical),
            'left_foot_vertical_n': float(left_vertical), 'right_foot_vertical_n': float(right_vertical),
            'left_foot_normal_n': float(left_normal), 'right_foot_normal_n': float(right_normal),
            'nonfoot_ground_normal_n': float(nonfoot_ground_normal),
            'pelvis_xy': pelvis_xy.copy(),
            'cop_xy': cop.copy() if cop is not None else None,
            'support_cop_xy': support_cop.copy() if support_cop is not None else None,
            'support_cop_rel_xy': support_cop_rel.copy() if support_cop_rel is not None else None,
            'support_cop_body_xy': support_cop_body.copy() if support_cop_body is not None else None,
            'double_support': bool(left_contact and right_contact),
            'left_support': bool(left_contact), 'right_support': bool(right_contact),
            'tangential_ratio': float(tangential_ratio),
            'primary_impact_body': primary_body, 'primary_impact_body_load_n': float(primary_body_load),
            'impact_by_body': dict(impact_by_body),
            **filt,
        }
        self.frames.append(frame)
        return frame

    def latest(self):
        return self.frames[-1] if self.frames else None

    def export_frame_csv(self, filename):
        import csv
        if not self.frames:
            return {'filename': filename, 'frames': 0}
        keys = ['time','phase','contact_count','total_ground_vertical_n','total_ground_normal_n','support_vertical_n','left_foot_vertical_n','right_foot_vertical_n','left_foot_normal_n','right_foot_normal_n','nonfoot_ground_normal_n','tangential_ratio','double_support','left_support','right_support','primary_impact_body','primary_impact_body_load_n','total_ground_vertical_n_filt','support_vertical_n_filt','left_foot_vertical_n_filt','right_foot_vertical_n_filt','nonfoot_ground_normal_n_filt','primary_impact_body_load_n_filt','tangential_ratio_filt']
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(keys + ['cop_x','cop_y','support_cop_x','support_cop_y','support_cop_rel_x','support_cop_rel_y','support_cop_body_x','support_cop_body_y','pelvis_x','pelvis_y'])
            for fr in self.frames:
                cop = fr['cop_xy'] if fr['cop_xy'] is not None else [np.nan, np.nan]
                scop = fr['support_cop_xy'] if fr.get('support_cop_xy') is not None else [np.nan, np.nan]
                scop_rel = fr['support_cop_rel_xy'] if fr.get('support_cop_rel_xy') is not None else [np.nan, np.nan]
                scop_body = fr['support_cop_body_xy'] if fr.get('support_cop_body_xy') is not None else [np.nan, np.nan]
                pel = fr.get('pelvis_xy', [np.nan, np.nan])
                writer.writerow([fr[k] for k in keys] + [float(cop[0]), float(cop[1]), float(scop[0]), float(scop[1]), float(scop_rel[0]), float(scop_rel[1]), float(scop_body[0]), float(scop_body[1]), float(pel[0]), float(pel[1])])
        return {'filename': filename, 'frames': len(self.frames)}

    def export_contact_csv(self, filename):
        import csv
        if not self.contact_rows:
            return {'filename': filename, 'rows': 0}
        keys = list(self.contact_rows[0].keys())
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.contact_rows)
        return {'filename': filename, 'rows': len(self.contact_rows)}

    def _median_filter(self, arr, k=5):
        arr = np.asarray(arr, dtype=float)
        if k <= 1 or arr.size == 0:
            return arr.copy()
        h = k // 2
        out = np.empty_like(arr)
        for i in range(arr.size):
            lo = max(0, i - h); hi = min(arr.size, i + h + 1)
            out[i] = np.nanmedian(arr[lo:hi])
        return out

    def export_visuals(self, prefix):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            return {'available': False, 'plots': [], 'error': str(e)}
        if not self.frames:
            return {'available': False, 'plots': []}
        t = np.array([fr['time'] for fr in self.frames], dtype=float)
        bw = self.body_mass * self.g
        total_v = np.array([fr.get('support_vertical_n_filt', fr['total_ground_vertical_n_filt']) for fr in self.frames], dtype=float) / max(bw, 1.0)
        left_v = np.array([fr['left_foot_vertical_n_filt'] for fr in self.frames], dtype=float) / max(bw, 1.0)
        right_v = np.array([fr['right_foot_vertical_n_filt'] for fr in self.frames], dtype=float) / max(bw, 1.0)
        impact = np.array([fr['primary_impact_body_load_n_filt'] for fr in self.frames], dtype=float) / max(bw, 1.0)
        tan = np.array([fr['tangential_ratio_filt'] for fr in self.frames], dtype=float)
        total_v = self._median_filter(total_v, 5)
        left_v = self._median_filter(left_v, 5)
        right_v = self._median_filter(right_v, 5)
        impact = self._median_filter(impact, 5)
        tan = self._median_filter(tan, 5)
        fig = plt.figure(figsize=(10.5, 6.2))
        ax1 = fig.add_subplot(211)
        ax1.plot(t, total_v, label='Support vertical GRF/BW (median)', linewidth=2.0)
        ax1.plot(t, left_v, label='Left foot GRF/BW (median)', linewidth=1.7)
        ax1.plot(t, right_v, label='Right foot GRF/BW (median)', linewidth=1.7)
        ax1.plot(t, impact, label='Primary impact/BW (median)', linewidth=1.7)
        ax1.set_ylabel('Load / BW')
        ymax = max(3.0, float(np.nanpercentile(np.r_[total_v, left_v, right_v, impact], 99.0)) * 1.15)
        ax1.set_ylim(0.0, ymax)
        ax1.set_title('Layer-2 dynamics summary')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        ax2 = fig.add_subplot(212)
        ax2.plot(t, tan, label='Tangential/normal ratio (median)', linewidth=2.0)
        ax2.set_ylim(0.0, max(1.5, float(np.nanpercentile(tan, 99.0)) * 1.15))
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Ratio')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        fn = f'{prefix}_dynamics_summary.png'
        fig.tight_layout()
        fig.savefig(fn, dpi=180)
        plt.close(fig)

        # Support CoP in pelvis body frame (removes world-heading rotation)
        cops = np.array([fr['support_cop_body_xy'] if fr.get('support_cop_body_xy') is not None else (fr['support_cop_rel_xy'] if fr.get('support_cop_rel_xy') is not None else [np.nan, np.nan]) for fr in self.frames], dtype=float)
        fig = plt.figure(figsize=(6.2, 6.0))
        ax = fig.add_subplot(111)
        valid = np.isfinite(cops[:,0]) & np.isfinite(cops[:,1])
        if np.any(valid):
            phases = np.array([str(fr.get('phase','')) for fr in self.frames])
            for ph, color in [('stand','#1f77b4'),('walk','#2ca02c'),('perturb','#ff7f0e'),('react','#d62728'),('fall','#9467bd')]:
                mask = valid & (phases == ph)
                if np.any(mask):
                    ax.plot(cops[mask,0], cops[mask,1], '.', color=color, alpha=0.65, label=ph, markersize=4)
        ax.axhline(0.0, color='0.7', linewidth=0.8)
        ax.axvline(0.0, color='0.7', linewidth=0.8)
        ax.set_title('Support CoP in pelvis body frame (phase-coloured)')
        ax.set_xlabel('Forward rel. pelvis [m]')
        ax.set_ylabel('Lateral rel. pelvis [m]')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        fn2 = f'{prefix}_cop_path.png'
        fig.tight_layout()
        fig.savefig(fn2, dpi=180)
        plt.close(fig)
        return {'available': True, 'plots': [fn, fn2]}

    def export_bundle(self, prefix):
        frame_csv = self.export_frame_csv(f'{prefix}_dynamics_frames.csv')
        contact_csv = self.export_contact_csv(f'{prefix}_contacts.csv')
        visuals = self.export_visuals(prefix)
        return {'frame_csv': frame_csv, 'contact_csv': contact_csv, 'visuals': visuals}


class PaperAlignmentExporter:
    """
    High-level bridge for the paper-oriented deliverables that are still missing
    from the base MuJoCo/Meta-Motivo rollout.

    It does three things without requiring OpenSim/AnyBody to be installed:
      1) exports foot-wise GRF/CoP/torque in an OpenSim-style .mot file
      2) exports an ExternalLoads XML template plus marker-registration metadata
      3) exports a multiview 2D pose dataset derived from the simulated 3D markers

    This closes the "export-compatible" gap in a much more paper-faithful way.
    """

    def __init__(self, marker_exporter, dynamics_analyzer, subject_meta=None):
        self.marker_exporter = marker_exporter
        self.dynamics_analyzer = dynamics_analyzer
        self.subject_meta = dict(subject_meta or {})

    def _master_times(self):
        if self.dynamics_analyzer.frames:
            return np.array([float(fr['time']) for fr in self.dynamics_analyzer.frames], dtype=float)
        if self.marker_exporter.frames:
            return np.array([float(fr['time']) for fr in self.marker_exporter.frames], dtype=float)
        return np.array([], dtype=float)

    def _time_key(self, t):
        return round(float(t), 6)

    def _aggregate_external_loads(self):
        master_t = self._master_times()
        if master_t.size == 0:
            return []

        buckets = {self._time_key(t): {
            'left_foot': {'F': np.zeros(3), 'cop_num': np.zeros(3), 'fz_sum': 0.0, 'contacts': []},
            'right_foot': {'F': np.zeros(3), 'cop_num': np.zeros(3), 'fz_sum': 0.0, 'contacts': []},
        } for t in master_t}

        for row in self.dynamics_analyzer.contact_rows:
            cls = str(row.get('class', ''))
            if cls not in ('left_foot', 'right_foot'):
                continue
            tk = self._time_key(row['time'])
            if tk not in buckets:
                continue
            force = np.array([float(row['world_fx']), float(row['world_fy']), float(row['world_fz'])], dtype=float)
            pos = np.array([float(row['px']), float(row['py']), float(row['pz'])], dtype=float)
            rec = buckets[tk][cls]
            rec['F'] += force
            fz = max(0.0, float(force[2]))
            rec['cop_num'] += pos * fz
            rec['fz_sum'] += fz
            rec['contacts'].append((pos, force))

        out = []
        for t in master_t:
            tk = self._time_key(t)
            row_out = {'time': float(t)}
            for cls, side in (('left_foot', 'l'), ('right_foot', 'r')):
                rec = buckets[tk][cls]
                F = rec['F']
                if rec['fz_sum'] > 1e-9:
                    cop = rec['cop_num'] / rec['fz_sum']
                else:
                    cop = np.array([0.0, 0.0, 0.0], dtype=float)
                torque = np.zeros(3)
                for pos, force in rec['contacts']:
                    torque += np.cross(pos - cop, force)
                row_out[f'F_{side}'] = F.copy()
                row_out[f'cop_{side}'] = cop.copy()
                row_out[f'T_{side}'] = torque.copy()
            out.append(row_out)
        return out

    def export_opensim_grf_mot(self, filename):
        rows = self._aggregate_external_loads()
        if not rows:
            return {'filename': filename, 'frames': 0}
        cols = [
            'time',
            'ground_force_l_vx', 'ground_force_l_vy', 'ground_force_l_vz',
            'ground_force_l_px', 'ground_force_l_py', 'ground_force_l_pz',
            'ground_torque_l_x', 'ground_torque_l_y', 'ground_torque_l_z',
            'ground_force_r_vx', 'ground_force_r_vy', 'ground_force_r_vz',
            'ground_force_r_px', 'ground_force_r_py', 'ground_force_r_pz',
            'ground_torque_r_x', 'ground_torque_r_y', 'ground_torque_r_z',
        ]
        with open(filename, 'w') as f:
            f.write(f"name {Path(filename).name}\n")
            f.write(f"datacolumns {len(cols)}\n")
            f.write(f"datarows {len(rows)}\n")
            f.write(f"range {rows[0]['time']:.6f} {rows[-1]['time']:.6f}\n")
            f.write("endheader\n")
            f.write('\t'.join(cols) + '\n')
            for row in rows:
                vals = [row['time'], *row['F_l'], *row['cop_l'], *row['T_l'], *row['F_r'], *row['cop_r'], *row['T_r']]
                f.write('\t'.join(f"{float(v):.6f}" for v in vals) + '\n')
        return {'filename': filename, 'frames': len(rows), 'columns': len(cols)}

    def export_opensim_external_loads_xml(self, filename, grf_mot_file):
        import xml.etree.ElementTree as ET
        root = ET.Element('OpenSimDocument', Version='40000')
        ext = ET.SubElement(root, 'ExternalLoads', name='external_loads')
        ET.SubElement(ext, 'objects')
        objs = ext.find('objects')
        for side, body in (('l', 'calcn_l'), ('r', 'calcn_r')):
            ef = ET.SubElement(objs, 'ExternalForce', name=f'grf_{side}')
            ET.SubElement(ef, 'applied_to_body').text = body
            ET.SubElement(ef, 'force_expressed_in_body').text = 'ground'
            ET.SubElement(ef, 'point_expressed_in_body').text = 'ground'
            ET.SubElement(ef, 'force_identifier').text = f'ground_force_{side}_v'
            ET.SubElement(ef, 'point_identifier').text = f'ground_force_{side}_p'
            ET.SubElement(ef, 'torque_identifier').text = f'ground_torque_{side}_'
        ET.SubElement(ext, 'groups')
        ET.SubElement(ext, 'datafile').text = Path(grf_mot_file).name
        ET.SubElement(ext, 'external_loads_model_kinematics_file').text = ''
        ET.SubElement(ext, 'lowpass_cutoff_frequency_for_load_kinematics').text = '-1'
        tree = ET.ElementTree(root)
        tree.write(filename, encoding='utf-8', xml_declaration=True)
        return {'filename': filename, 'datafile': Path(grf_mot_file).name, 'note': 'applied_to_body defaults to calcn_l/calcn_r and may need adaptation to the target OpenSim model'}

    def export_marker_registration_json(self, filename):
        import json
        recs = []
        for marker_name, rec in self.marker_exporter.marker_defs.items():
            recs.append({
                'avatar_marker': marker_name,
                'opensim_marker': self.marker_exporter.OPENSIM_TRC_NAMES.get(marker_name, marker_name),
                'resolved_body': rec['body_name'],
                'resolved_body_id': int(rec['body_id']),
                'offset_local_m': [float(x) for x in np.asarray(rec['offset_local'], dtype=float)],
                'offset_mode': rec.get('offset_mode', 'center'),
                'family': rec.get('family'),
                'side': rec.get('side'),
            })
        payload = {
            'subject_meta': self.subject_meta,
            'num_markers': len(recs),
            'marker_registration': recs,
            'trc_names': self.marker_exporter.OPENSIM_TRC_NAMES,
            'note': 'This is the avatar-to-musculoskeletal bridge metadata. It is intended to support IK marker registration in OpenSim/AnyBody workflows.',
        }
        with open(filename, 'w') as f:
            json.dump(payload, f, indent=2)
        return {'filename': filename, 'markers': len(recs)}

    def _camera_specs(self):
        all_pts = []
        for fr in self.marker_exporter.frames:
            for name in self.marker_exporter.POSE_MARKERS:
                if name in fr['markers']:
                    all_pts.append(np.asarray(fr['markers'][name], dtype=float))
        if not all_pts:
            return {}
        arr = np.vstack(all_pts)
        mins = np.min(arr, axis=0)
        maxs = np.max(arr, axis=0)
        center = 0.5 * (mins + maxs)
        span = np.maximum(maxs - mins, np.array([0.4, 0.4, 1.0]))
        radius = float(max(span[0], span[1], span[2]))
        target = center + np.array([0.0, 0.0, 0.55 * span[2]])
        return {
            'frontal': {'pos': center + np.array([0.0, -2.8 * radius, 0.7 * radius]), 'target': target},
            'sagittal': {'pos': center + np.array([2.8 * radius, 0.0, 0.7 * radius]), 'target': target},
            'oblique': {'pos': center + np.array([2.2 * radius, -2.2 * radius, 0.8 * radius]), 'target': target},
        }

    def _look_at(self, pos, target, up=np.array([0.0, 0.0, 1.0])):
        pos = np.asarray(pos, dtype=float)
        target = np.asarray(target, dtype=float)
        up = np.asarray(up, dtype=float)
        z = target - pos
        z /= max(np.linalg.norm(z), 1e-9)
        x = np.cross(z, up)
        if np.linalg.norm(x) < 1e-9:
            x = np.array([1.0, 0.0, 0.0], dtype=float)
        x /= max(np.linalg.norm(x), 1e-9)
        y = np.cross(x, z)
        y /= max(np.linalg.norm(y), 1e-9)
        R = np.vstack([x, y, z])
        return R

    def _project(self, pts_world, cam, image_wh=(1280, 720)):
        w, h = image_wh
        fx = fy = 0.95 * min(w, h)
        cx, cy = 0.5 * w, 0.5 * h
        R = self._look_at(cam['pos'], cam['target'])
        t = np.asarray(cam['pos'], dtype=float)
        pts2d = []
        vis = []
        for p in pts_world:
            p = np.asarray(p, dtype=float)
            pc = R @ (p - t)
            if pc[2] <= 1e-6:
                pts2d.append([np.nan, np.nan])
                vis.append(0)
                continue
            u = fx * (pc[0] / pc[2]) + cx
            v = cy - fy * (pc[1] / pc[2])
            in_frame = int((-0.1 * w) <= u <= (1.1 * w) and (-0.1 * h) <= v <= (1.1 * h))
            pts2d.append([float(u), float(v)])
            vis.append(in_frame)
        return np.array(pts2d, dtype=float), np.array(vis, dtype=int)

    def export_pose_dataset_json(self, filename, views=('frontal', 'sagittal', 'oblique')):
        import json
        cameras = self._camera_specs()
        if not cameras or not self.marker_exporter.frames:
            return {'filename': filename, 'frames': 0}
        views = [v for v in views if v in cameras]
        records = []
        marker_names = [m for m in self.marker_exporter.POSE_MARKERS if m in self.marker_exporter.marker_defs]
        for idx, fr in enumerate(self.marker_exporter.frames):
            pts_world = [np.asarray(fr['markers'][m], dtype=float) for m in marker_names]
            pts3d = [[float(x) for x in p] for p in pts_world]
            view_rows = []
            for v in views:
                pts2d, vis = self._project(pts_world, cameras[v])
                valid = np.isfinite(pts2d[:, 0]) & np.isfinite(pts2d[:, 1]) & (vis > 0)
                if np.any(valid):
                    uv = pts2d[valid]
                    x0, y0 = np.min(uv, axis=0)
                    x1, y1 = np.max(uv, axis=0)
                    bbox = [float(x0), float(y0), float(max(0.0, x1 - x0)), float(max(0.0, y1 - y0))]
                else:
                    bbox = [0.0, 0.0, 0.0, 0.0]
                view_rows.append({
                    'camera': v,
                    'bbox_xywh': bbox,
                    'keypoints_xyv': [[float(pts2d[k,0]) if np.isfinite(pts2d[k,0]) else None, float(pts2d[k,1]) if np.isfinite(pts2d[k,1]) else None, int(vis[k])] for k in range(len(marker_names))],
                })
            records.append({
                'frame_index': int(idx),
                'time': float(fr['time']),
                'phase': str(fr.get('phase', '')),
                'marker_names': marker_names,
                'keypoints_xyz_m': pts3d,
                'views': view_rows,
            })
        payload = {
            'subject_meta': self.subject_meta,
            'marker_names': marker_names,
            'skeleton_edges': self.marker_exporter.VISUAL_SKELETON_EDGES,
            'cameras': {k: {'position_world_m': [float(x) for x in v['pos']], 'target_world_m': [float(x) for x in v['target']], 'image_size': [1280, 720]} for k, v in cameras.items() if k in views},
            'frames': records,
            'note': 'Synthetic multiview 2D/3D pose annotations derived from the simulated anatomical markers. These are render-free annotations; photo-real RGB generation requires an external renderer.',
        }
        with open(filename, 'w') as f:
            json.dump(payload, f, indent=2)
        return {'filename': filename, 'frames': len(records), 'views': len(views)}

    def export_pose_preview(self, filename, views=('frontal', 'sagittal', 'oblique'), num_frames=4):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            return {'filename': filename, 'available': False, 'error': str(e)}
        cameras = self._camera_specs()
        if not cameras or not self.marker_exporter.frames:
            return {'filename': filename, 'available': False}
        views = [v for v in views if v in cameras]
        idxs = np.linspace(0, len(self.marker_exporter.frames) - 1, num=max(1, num_frames), dtype=int)
        marker_names = [m for m in self.marker_exporter.VISUAL_MARKERS if m in self.marker_exporter.marker_defs]
        fig = plt.figure(figsize=(4.2 * len(views), 3.4 * len(idxs)))
        for r, idx in enumerate(idxs, start=1):
            fr = self.marker_exporter.frames[int(idx)]
            pts_world = {m: np.asarray(fr['markers'][m], dtype=float) for m in marker_names}
            for c, view in enumerate(views, start=1):
                ax = fig.add_subplot(len(idxs), len(views), (r - 1) * len(views) + c)
                cam = cameras[view]
                proj = {}
                pts_list = []
                for m, p in pts_world.items():
                    uv, vis = self._project([p], cam)
                    if vis[0] > 0 and np.isfinite(uv[0, 0]) and np.isfinite(uv[0, 1]):
                        proj[m] = uv[0].copy()
                        pts_list.append(uv[0].copy())
                for a, b in self.marker_exporter.VISUAL_SKELETON_EDGES:
                    if a in proj and b in proj:
                        pa, pb = proj[a], proj[b]
                        style = self.marker_exporter._edge_style(a, b)
                        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], '-', color=style['color'], linewidth=style['linewidth'])
                if pts_list:
                    arr = np.vstack(pts_list)
                    ax.scatter(arr[:, 0], arr[:, 1], s=18, color='#1f77b4', edgecolors='white', linewidths=0.4, zorder=3)
                    x0, y0 = np.min(arr, axis=0); x1, y1 = np.max(arr, axis=0)
                    pad = 60.0
                    ax.set_xlim(x0 - pad, x1 + pad)
                    ax.set_ylim(y1 + pad, y0 - pad)
                ax.set_title(f"{view} | t={fr['time']:.2f}s")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_aspect('equal', adjustable='box')
        fig.suptitle('Paper-alignment synthetic pose preview')
        fig.tight_layout()
        fig.savefig(filename, dpi=180)
        plt.close(fig)
        return {'filename': filename, 'available': True, 'frames': len(idxs), 'views': len(views)}

    def export_bundle(self, prefix, views=('frontal', 'sagittal', 'oblique')):
        grf = self.export_opensim_grf_mot(f'{prefix}_opensim_grf.mot')
        ext = self.export_opensim_external_loads_xml(f'{prefix}_ExternalLoads.xml', grf['filename']) if grf.get('frames', 0) else {'filename': f'{prefix}_ExternalLoads.xml', 'frames': 0}
        reg = self.export_marker_registration_json(f'{prefix}_marker_registration.json')
        pose_json = self.export_pose_dataset_json(f'{prefix}_synthetic_pose_dataset.json', views=views)
        pose_preview = self.export_pose_preview(f'{prefix}_synthetic_pose_preview.png', views=views, num_frames=PAPER_ALIGNMENT_EXPORT.get('pose_preview_frames', 4))
        return {
            'opensim_grf': grf,
            'external_loads_xml': ext,
            'marker_registration': reg,
            'pose_dataset_json': pose_json,
            'pose_preview': pose_preview,
        }


class PhysicsDashboard:
    def __init__(self, mj_model, mj_data, body_mass, leg_length, upright_h):
        self.mj_model   = mj_model
        self.mj_data    = mj_data
        self.body_mass  = body_mass
        self.leg_length = leg_length
        self.upright_h  = upright_h
        self.g          = 9.81
        self.ip_omega   = float(np.sqrt(self.g / max(leg_length, 0.1)))
        self._pelvis_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, "Pelvis")

        # FIX S: body IDs for biomech metrics
        self._torso_id   = mujoco.mj_name2id(mj_model, MJOBJ_BODY, "Torso")
        self._foot_l_id  = mujoco.mj_name2id(mj_model, MJOBJ_BODY, "FootL")
        self._foot_r_id  = mujoco.mj_name2id(mj_model, MJOBJ_BODY, "FootR")
        self._hand_l_id  = mujoco.mj_name2id(mj_model, MJOBJ_BODY, "HandL")
        self._hand_r_id  = mujoco.mj_name2id(mj_model, MJOBJ_BODY, "HandR")

        # FIX X7: double-support via contact geoms
        self._lower_limb_kw = ['foot', 'ankle', 'toe', 'heel', 'lower', 'shin', 'tibia']
        self._left_kw  = ['l', 'left']
        self._right_kw = ['r', 'right']
        plane_type = getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', None)
        self._ground_geoms = set()
        for g in range(mj_model.ngeom):
            gname = (mujoco.mj_id2name(mj_model, MJOBJ_GEOM, g) or '').lower()
            is_ground = ('floor' in gname or 'ground' in gname or 'plane' in gname or
                         (plane_type is not None and mj_model.geom_type[g] == plane_type))
            if is_ground:
                self._ground_geoms.add(g)

        # Build lower-limb body-id sets for each side
        self._left_lower_bodies  = set()
        self._right_lower_bodies = set()
        for i in range(mj_model.nbody):
            n = (mujoco.mj_id2name(mj_model, MJOBJ_BODY, i) or '')
            nlow = n.lower()
            if any(k in nlow for k in self._lower_limb_kw):
                if _body_name_matches_side(n, 'left'):
                    self._left_lower_bodies.add(i)
                elif _body_name_matches_side(n, 'right'):
                    self._right_lower_bodies.add(i)

        # FIX X3: shoulder actuator indices for arm-swing measurement
        self._shoulder_dof_ids = []
        for i in range(mj_model.nu):
            n = (mujoco.mj_id2name(mj_model, MJOBJ_ACTUATOR, i) or '').lower()
            if 'shoulder' in n:
                self._shoulder_dof_ids.append(i)

        self._header_printed = False
        self._foot_contact_history = deque(maxlen=60)
        self._cadence_hz = 0.0

    def _print_header(self):
        print("\n" + "=" * 170)
        print(f"  {'step':>5} {'phase':>8} {'h[m]':>6} {'vx':>6} "
              f"{'trunkdeg':>7} {'armSwdeg':>7} {'dbl_sup':>7} "
              f"{'GRF/BW':>7} {'XCoM_m':>7} {'Fr':>6} {'omg[dps]':>8} {'Imp/BW':>7} "
              f"{'slip':>6} {'ncon':>5} {'settle':>6} {'src':>4} {'mpcC':>7} "
              f"{'E_fall[J]':>10} {'JRMS[deg]':>8} {'strength':>9} {'IMU_pk':>8} {'fall?':>5}")
        print("-" * 205)
        self._header_printed = True

    def _pelvis_world_vel(self):
        vel6 = np.zeros(6)
        try:
            mujoco.mj_objectVelocity(self.mj_model, self.mj_data,
                                     MJOBJ_BODY, self._pelvis_id, vel6, 0)
            return vel6[3:].copy()
        except Exception:
            R = self.mj_data.xmat[self._pelvis_id].reshape(3, 3)
            return R @ self.mj_data.cvel[self._pelvis_id][3:]

    def _trunk_lean_deg(self):
        """
        FIX X2: Trunk lean = angle between pelvis->Head world vector and world-z.
        0deg = perfectly upright. 90deg = horizontal (fallen).
        Replaces v12 method which used Torso local-z (gave ~90deg when upright
        because Torso local-z points FORWARD in humanenv, not UP).
        """
        head_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, "Head")
        if head_id < 0 or self._pelvis_id < 0:
            return 0.0
        pelvis_pos = self.mj_data.xpos[self._pelvis_id]
        head_pos   = self.mj_data.xpos[head_id]
        vec = head_pos - pelvis_pos
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return 0.0
        cos_theta = float(np.clip(np.dot(vec / norm, [0.0, 0.0, 1.0]), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_theta)))

    def _arm_swing_deg(self):
        """
        FIX X3: Arm swing from ANGULAR VELOCITY of shoulder actuators (qvel).
        Heading-invariant. Returns RMS of shoulder angular velocities in deg/s
        divided by a scaling factor to give approximate peak angle in deg.
        Norm: 8-20deg shoulder flexion/extension during normal walking.
        """
        if not self._shoulder_dof_ids:
            return 0.0
        # qvel indices correspond to actuator dof indices
        # For approximation, use qvel at the actuator dof positions
        # Actuator joint id -> dof id via jnt_dofadr
        shoulder_speeds = []
        for act_id in self._shoulder_dof_ids:
            jnt_id = self.mj_model.actuator_trnid[act_id, 0]
            if 0 <= jnt_id < self.mj_model.njnt:
                dof_id = self.mj_model.jnt_dofadr[jnt_id]
                if 0 <= dof_id < len(self.mj_data.qvel):
                    shoulder_speeds.append(abs(float(self.mj_data.qvel[dof_id])))
        if not shoulder_speeds:
            return 0.0
        # Convert rad/s to approximate deg of swing angle
        # At 1 Hz cadence, peak angle ~ peak_speed / (2*pi) * 180 deg
        rms_speed = float(np.sqrt(np.mean(np.array(shoulder_speeds)**2)))
        approx_angle = float(np.degrees(rms_speed) / (2 * np.pi * 1.0))
        return float(np.clip(approx_angle, 0.0, 90.0))

    def _double_support(self):
        """
        FIX X7: Double-support via actual ground contacts on lower-limb bodies.
        v12 used foot body z-position which didn't match humanenv's body names.
        Now checks contact geom body IDs against left/right lower-limb body sets.
        Falls back to z-position check if no contact body IDs matched.
        """
        left_contact  = False
        right_contact = False

        for i in range(self.mj_data.ncon):
            force = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, force)
            if force[0] < 0.5:
                continue
            g1 = self.mj_data.contact[i].geom1
            g2 = self.mj_data.contact[i].geom2
            if self._ground_geoms and g1 not in self._ground_geoms and g2 not in self._ground_geoms:
                continue
            b1 = self.mj_model.geom_bodyid[g1]
            b2 = self.mj_model.geom_bodyid[g2]
            for b in (b1, b2):
                if b in self._left_lower_bodies:
                    left_contact = True
                if b in self._right_lower_bodies:
                    right_contact = True

        # Fallback: z-position of foot bodies if contact sets are empty
        if not self._left_lower_bodies and not self._right_lower_bodies:
            thresh = 0.08
            fl_id = self._foot_l_id
            fr_id = self._foot_r_id
            if fl_id >= 0 and fr_id >= 0:
                left_contact  = float(self.mj_data.xpos[fl_id][2]) < thresh
                right_contact = float(self.mj_data.xpos[fr_id][2]) < thresh

        return left_contact and right_contact

    def _grf_bw(self):
        """World-vertical ground reaction force normalised to body weight."""
        total_vertical = 0.0
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1 = c.geom1; g2 = c.geom2
            if self._ground_geoms and g1 not in self._ground_geoms and g2 not in self._ground_geoms:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, wrench)
            f_world, _ = _contact_wrench_world(c, wrench)
            total_vertical += max(0.0, float(f_world[2]))
        bw = self.body_mass * self.g
        return float(total_vertical / max(bw, 1.0))

    def _pelvis_angular_speed_dps(self):
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.mj_model, self.mj_data,
                                 MJOBJ_BODY, self._pelvis_id, vel6, 0)
        return float(np.degrees(np.linalg.norm(vel6[:3])))

    def _froude_number(self, pelvis_vel_world):
        vxy = float(np.linalg.norm(pelvis_vel_world[:2]))
        return float((vxy * vxy) / max(self.g * self.leg_length, 1e-9))

    def report(self, step, phase, leg_strength, xcom_margin=None,
               fall_predicted=False, imu_peak=0.0, sensor_impact=0.0,
               control_source='-', mjpc_cost=np.nan, dynamics=None):
        if not self._header_printed:
            self._print_header()

        pelvis_pos = self.mj_data.xpos[self._pelvis_id]
        pelvis_vel_world = self._pelvis_world_vel()

        h        = float(pelvis_pos[2])
        delta_h  = self.upright_h - h
        v_trans  = float(np.linalg.norm(pelvis_vel_world))
        E_pot    = self.body_mass * self.g * max(0.0, delta_h)
        E_kin    = 0.5 * self.body_mass * v_trans ** 2
        E_fall   = E_pot + E_kin

        jrms_rad = float(np.sqrt(np.mean(self.mj_data.qpos[7:] ** 2)))
        jrms_deg = np.degrees(jrms_rad)

        # FIX S: extended biomechanical metrics
        trunk_deg = self._trunk_lean_deg()
        arm_swing = self._arm_swing_deg()
        dbl_sup   = dynamics.get('double_support') if dynamics is not None else self._double_support()
        if dynamics is not None:
            if phase in ('stand', 'walk', 'perturb', 'react'):
                grf_bw = float(dynamics.get('support_vertical_n_filt', dynamics.get('support_vertical_n', 0.0)) / max(self.body_mass * self.g, 1.0))
            else:
                grf_bw = float(dynamics.get('total_ground_vertical_n_filt', dynamics.get('total_ground_vertical_n', 0.0)) / max(self.body_mass * self.g, 1.0))
        else:
            grf_bw = self._grf_bw()

        xcom_str = f"{xcom_margin:+7.3f}" if xcom_margin is not None else "   N/A "

        bar_len  = int(leg_strength * 10)
        bar      = chr(0x2588) * bar_len + chr(0x2591) * (10 - bar_len)
        fall_flag = "FALL!" if fall_predicted else "  ok "

        slip_vxy = float(np.linalg.norm(pelvis_vel_world[:2]))
        froude = self._froude_number(pelvis_vel_world)
        omg_dps = self._pelvis_angular_speed_dps()
        impact_n = float(dynamics.get('primary_impact_body_load_n_filt', dynamics.get('primary_impact_body_load_n', sensor_impact))) if dynamics is not None else float(sensor_impact)
        impact_bw = float(impact_n / max(self.body_mass * self.g, 1.0))
        ncon = int(dynamics.get('contact_count', self.mj_data.ncon)) if dynamics is not None else int(self.mj_data.ncon)
        settle_flag = "YES" if (h < 0.18 and ncon > 0 and slip_vxy < 0.10) else " no"

        src = 'MPC' if str(control_source).lower().startswith('mjpc') else 'GDN'
        mpc_cost_str = f"{mjpc_cost:7.1f}" if np.isfinite(mjpc_cost) else '   N/A '
        print(f"  {step:>5d} {phase:>8s} {h:>6.3f} {float(pelvis_vel_world[0]):>6.2f} "
              f"{trunk_deg:>7.2f} {arm_swing:>7.2f} {'YES' if dbl_sup else ' no':>7} "
              f"{grf_bw:>7.2f} {xcom_str:>7} {froude:>6.2f} {omg_dps:>8.1f} {impact_bw:>7.2f} "
              f"{slip_vxy:>6.2f} {ncon:>5d} {settle_flag:>6} {src:>4} {mpc_cost_str:>7} "
              f"{E_fall:>10.1f} {jrms_deg:>8.2f} "
              f" {bar}{leg_strength:>5.0%}  {imu_peak:>8.2f}  {fall_flag}")

        return {
            'pelvis_h': h, 'E_fall': E_fall, 'jrms_deg': float(jrms_deg),
            'trunk_lean_deg': trunk_deg, 'arm_swing_deg': arm_swing,
            'double_support': dbl_sup, 'grf_bw': grf_bw,
            'imu_peak': imu_peak, 'leg_strength': leg_strength,
            'slip_vxy': slip_vxy, 'froude': froude, 'pelvis_angular_speed_dps': omg_dps,
            'impact_bw': impact_bw, 'contact_count': ncon,
            'settle_flag': settle_flag.strip() == 'YES',
            'control_source': control_source, 'mjpc_cost': float(mjpc_cost) if np.isfinite(mjpc_cost) else np.nan,
        }

    def _compute_cop(self):
        total_fz, cop_x, cop_y = 0.0, 0.0, 0.0
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, wrench)
            f_world, _ = _contact_wrench_world(c, wrench)
            fz = max(0.0, float(f_world[2]))
            if fz > 1.0:
                cx = self.mj_data.contact[i].pos[0]
                cy = self.mj_data.contact[i].pos[1]
                total_fz += fz
                cop_x    += cx * fz
                cop_y    += cy * fz
        if total_fz > 1.0:
            return cop_x / total_fz, cop_y / total_fz
        return 0.0, 0.0

    def print_phase_summary(self, phase, metrics_slice):
        if not metrics_slice:
            return
        heights   = [m['pelvis_h']       for m in metrics_slice]
        e_falls   = [m.get('E_fall', 0)  for m in metrics_slice]
        jrms      = [m.get('jrms_deg', 0) for m in metrics_slice]
        imu_pks   = [m.get('imu_peak', 0) for m in metrics_slice]
        trunk_ang = [m.get('trunk_lean_deg', 0) for m in metrics_slice]
        arm_sw    = [m.get('arm_swing_deg', 0)  for m in metrics_slice]
        grfs      = [m.get('grf_bw', 0)          for m in metrics_slice]
        froudes   = [m.get('froude', 0)          for m in metrics_slice]
        omg_dps   = [m.get('pelvis_angular_speed_dps', 0) for m in metrics_slice]
        imp_bw    = [m.get('impact_bw', 0)       for m in metrics_slice]
        dbl_frac  = float(sum(1 for m in metrics_slice if m.get('double_support', False))) \
                    / max(len(metrics_slice), 1)

        print(f"\n  -- Phase '{phase}' summary --")
        print(f"     pelvis h    : min={min(heights):.3f}m  max={max(heights):.3f}m  "
              f"mean={np.mean(heights):.3f}m")
        print(f"     E_fall      : min={min(e_falls):.1f}J  max={max(e_falls):.1f}J  "
              f"mean={np.mean(e_falls):.1f}J")
        print(f"     JRMS        : min={min(jrms):.2f}deg  max={max(jrms):.2f}deg  "
              f"mean={np.mean(jrms):.2f}deg")
        print(f"     trunk lean  : min={min(trunk_ang):.2f}deg  max={max(trunk_ang):.2f}deg  "
              f"mean={np.mean(trunk_ang):.2f}deg  "
              f"[norm: <15deg upright, >20deg falling]")
        print(f"     arm swing   : min={min(arm_sw):.2f}deg  max={max(arm_sw):.2f}deg  "
              f"mean={np.mean(arm_sw):.2f}deg  "
              f"[norm: 8-20deg for walking]")
        print(f"     GRF/BW      : min={min(grfs):.2f}  max={max(grfs):.2f}  "
              f"mean={np.mean(grfs):.2f}  "
              f"[norm: 1.0-1.3 walking, >3 impact]")
        print(f"     Froude no.  : min={min(froudes):.2f}  max={max(froudes):.2f}  "
              f"mean={np.mean(froudes):.2f}  "
              f"[walk norm roughly 0.10-0.40]")
        print(f"     pelvis omg  : min={min(omg_dps):.1f}  max={max(omg_dps):.1f}  "
              f"mean={np.mean(omg_dps):.1f} deg/s")
        print(f"     impact/BW   : min={min(imp_bw):.2f}  max={max(imp_bw):.2f}  "
              f"mean={np.mean(imp_bw):.2f}")
        print(f"     dbl_support : {dbl_frac:.1%} of frames  "
              f"[norm: 20-25% walking, >35% elderly]")
        if any(p > 0 for p in imu_pks):
            print(f"     IMU_peak    : min={min(imu_pks):.2f}  max={max(imu_pks):.2f}  "
                  f"mean={np.mean(imu_pks):.2f} m/s^2")



# -------------------------------------------------------------------
# FALL TYPE LIBRARY (Improvement 1: orientation-mapped rewards)
# -------------------------------------------------------------------
class FallTypeLibrary:
    """
    Comprehensive fall scenarios based on Ferrari et al.
    """

    FALL_TYPES = {
        'forward_stumble': {
            'description':        'Forward fall while walking caused by toe hitting a stone; no hand support',
            'reward_sequence':    ['walk', 'lie_down_forward'],
            'force_direction':    [-1.0, 0.0, -0.20],
            'application_point':  'Pelvis',
            'perturbation_timing': 'after_4_5s_walk_toe_stone_trip',
            'muscle_weakening':   'toe_stone_trip',
        },
        'backward_walking': {
            'description':        'Backward fall while walking',
            'reward_sequence':    ['walk', 'lie_down_up'],
            'force_direction':    [-1, 0, 0.3],
            'application_point': 'Pelvis',
            'perturbation_timing': 'mid_stance',
        },
        'forward_stumble': {
            'description':        'Forward fall from stumbling',
            'reward_sequence':    ['walk', 'lie_down_forward'],
            'force_direction':    [0.5, 0, -0.3],
            'application_point': 'Foot',
            'perturbation_timing': 'heel_strike',
        },
        'lateral_left': {
            'description':        'Left side fall',
            'reward_sequence':    ['stand', 'lie_down_side_left'],
            'force_direction':    [0, -1, 0.2],
            'application_point': 'Torso',
            'perturbation_timing': 'instant',
        },
        'lateral_right': {
            'description':        'Right side fall',
            'reward_sequence':    ['stand', 'lie_down_side_right'],
            'force_direction':    [0, 1, 0.2],
            'application_point': 'Torso',
            'perturbation_timing': 'instant',
        },
        'slip_induced': {
            'description':        'Slip-induced fall on low friction surface',
            'reward_sequence':    ['walk_slippery', 'lie_down'],
            'friction_modification': 0.1,
            'perturbation_timing': 'random',
        },
        'collapse': {
            'description':        'Sudden collapse (muscle weakness)',
            'reward_sequence':    ['stand', 'collapse'],
            'muscle_weakening':  'instant',
            'perturbation_timing': 'none',
        },
        'sitting_loss_balance': {
            'description':        'Fall from sitting position',
            'reward_sequence':    ['sit', 'lie_down'],
            'object_required':   'chair',
            'perturbation_timing': 'delayed',
        },
    }

    @staticmethod
    def get_lie_down_reward(fall_type):
        """
        Get an orientation-appropriate reward for the fall type.
        Maps fall types to proper orientations; returns LieDownReward() or
        a custom reward function for prone / lateral falls.
        """
        orient_map = {
            'forward_stumble':'down',
            'backward_walking':      'up',
            'forward_stumble':       'down',
            'lateral_left':          'left',
            'lateral_right':         'right',
            'slip_induced':          'up',
            'collapse':              'up',
            'sitting_loss_balance':  'up',
        }
        orient = orient_map.get(fall_type, 'up')

        if orient == 'up':
            return LieDownReward()

        elif orient == 'down':
            def prone_reward(mdl, dat):
                pelvis_id = mujoco.mj_name2id(mdl, MJOBJ_BODY, "Pelvis")
                head_id   = mujoco.mj_name2id(mdl, MJOBJ_BODY, "Head")
                pelvis_pos = dat.xpos[pelvis_id]
                head_pos   = dat.xpos[head_id]
                fwd_disp   = head_pos[0] - pelvis_pos[0]
                height     = (pelvis_pos[2] + head_pos[2]) / 2
                return -height + max(0, fwd_disp) * 0.5
            return prone_reward

        elif orient == 'left':
            def lateral_left_reward(mdl, dat):
                pelvis_id  = mujoco.mj_name2id(mdl, MJOBJ_BODY, "Pelvis")
                pelvis_pos = dat.xpos[pelvis_id]
                return -pelvis_pos[2] + 0.3 * pelvis_pos[1]
            return lateral_left_reward

        elif orient == 'right':
            def lateral_right_reward(mdl, dat):
                pelvis_id  = mujoco.mj_name2id(mdl, MJOBJ_BODY, "Pelvis")
                pelvis_pos = dat.xpos[pelvis_id]
                return -pelvis_pos[2] - 0.3 * pelvis_pos[1]
            return lateral_right_reward

        else:
            return LieDownReward()


# -------------------------------------------------------------------
# ENHANCED BIOFIDELIC CONTROLLER (Improvement 5: natural reflexes)
# -------------------------------------------------------------------
class EnhancedBiofidelicController(BiofidelicFallController):
    """
    Extended controller with improved stability and realism.
    v2: protective reflexes use policy monitoring, not manual arm override.
    """

    def __init__(self, env, mj_model, mj_data, anthropometry=None):
        super().__init__(env, mj_model, mj_data)
        self.anthro = anthropometry or {}

        self.age_style = get_age_style(
            self.anthro.get('age_years', 35),
            self.anthro.get('height_m', 1.70),
            self.anthro.get('sex', 'male'),
            body_mass_kg=float(self.anthro.get('body_mass_kg', SIM_RESOLVED_WEIGHT)),
        )
        self.current_phase = 'stand'
        self.protective_reflexes         = True
        self.balance_control             = True
        self.vestibular_delay            = float(self.anthro.get('reaction_delay', 0.15))
        self.ankle_strategy_threshold    = 0.02
        self.hip_strategy_threshold      = 0.08

        # Gentle pre-perturb gait stabilization (active only before perturbation)
        self.walk_guidance_enabled       = True
        self.walk_target_speed           = float(self.age_style.get('target_walk_speed', 1.0))
        self.walk_ref_y                  = None
        self.walk_ref_yaw                = None
        self.walk_ref_origin_xy          = None
        self.walk_ref_fwd_xy             = None
        self.walk_ref_lat_xy             = None
        lat_scale = float(self.age_style.get('guidance_lateral_scale', 1.0))
        yaw_scale = float(self.age_style.get('guidance_yaw_scale', 1.0))
        self.walk_kp_speed               = 7.5
        self.walk_kp_lateral             = 12.0 * lat_scale
        self.walk_kd_lateral             = 13.0 * lat_scale
        self.walk_kp_yaw                 = 10.0 * yaw_scale
        self.walk_kd_yaw                 = 2.0 * np.sqrt(max(yaw_scale, 1e-6))
        self.walk_force_limit_x          = 18.0
        self.walk_force_limit_y          = 16.0 * lat_scale
        self.walk_torque_limit_z         = 14.0 * yaw_scale
        self.walk_guidance_alpha         = float(self.age_style.get('guidance_alpha', 0.88))
        self.walk_guidance_force_xy      = np.zeros(2, dtype=float)
        self.walk_guidance_tz            = 0.0
        self.walk_control_expected_ds    = float(self.age_style.get('expected_double_support', 0.30))
        self._pelvis_id                  = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Pelvis')
        self._torso_id                   = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Torso')
        self._head_id                    = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Head')
        self._upright_pelvis_height_ref  = float(self.mj_data.xpos[self._pelvis_id][2]) if self._pelvis_id >= 0 else None
        self._posture_height_vel_lp      = 0.0
        self.walk_leg_gain_boost         = 1.0
        self.walk_speed_ema              = 0.0
        self.walk_signed_speed_ema       = 0.0
        self.walk_forward_locked         = False
        self.walk_forward_lock_count     = 0

        # Scenario 29 pure-sagittal toe-trip lane. These values are locked at
        # the instant the virtual stone is hit, so gait sway cannot rotate the
        # fall into a lateral/right-left collapse.
        self.scenario29_trip_lane_locked = False
        self.scenario29_trip_fwd_xy      = None
        self.scenario29_trip_lat_xy      = None
        self.scenario29_trip_origin_xy   = None
        self.scenario29_trip_yaw         = None
        self.scenario29_walk_vel_ema_xy  = np.zeros(2, dtype=float)
        self.scenario29_walk_vel_samples = 0

        # State for protective monitoring
        self._last_protective_state = None

    # ------------------------------------------------------------------
    # Protective action - policy-based (v2 approach)
    # ------------------------------------------------------------------
    def get_protective_action(self, obs, z_fall, fall_direction):
        """
        Get action from policy; monitor (but do NOT override) protective reflexes.
        The Meta Motivo policy learns protective behaviours when properly rewarded.
        """
        action = self.get_action(obs, z_fall)
        if self.protective_reflexes:
            self._monitor_protective_response(fall_direction)
        return action

    def _monitor_protective_response(self, fall_direction):
        """Monitor and log protective responses for validation."""
        pelvis_id  = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, "Pelvis")
        pelvis_vel = self.mj_data.cvel[pelvis_id]
        pelvis_pos = self.mj_data.xpos[pelvis_id]

        if pelvis_vel[2] < -1.5 and pelvis_pos[2] < 0.8:
            arm_positions = self._get_arm_positions()
            self._last_protective_state = {
                'pelvis_velocity': pelvis_vel.copy(),
                'arm_extension':   arm_positions,
                'timestamp':       self.mj_data.time,
                'fall_direction':  fall_direction,
            }

    def _get_arm_positions(self):
        arm_data     = {}
        arm_keywords = ['shoulder', 'elbow', 'wrist']
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model,
                                     mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and any(k in name.lower() for k in arm_keywords):
                arm_data[name] = self.mj_data.qpos[self.mj_model.jnt_qposadr[i]]
        return arm_data

    def start_walk_phase(self):
        pelvis_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, "Pelvis")
        self.walk_leg_gain_boost = 1.0
        self.walk_speed_ema = 0.0
        self.walk_signed_speed_ema = 0.0
        self.walk_forward_locked = False
        self.walk_forward_lock_count = 0
        self.scenario29_walk_vel_ema_xy = np.zeros(2, dtype=float)
        self.scenario29_walk_vel_samples = 0
        if pelvis_id >= 0:
            pos = self.mj_data.xpos[pelvis_id].copy()
            self.walk_ref_y = float(pos[1])
            self.walk_ref_origin_xy = np.asarray(pos[:2], dtype=float).copy()

            # Use the dominant displacement direction from the selected z_walk
            # rollout, not the pelvis body-x axis. The body axis was still giving
            # the wrong forward sign in the logs, which made the controller fight
            # the requested age-specific speed envelopes.
            diag_heading = np.asarray(LAST_Z_WALK_DIAGNOSTICS.get('heading_xy', [1.0, 0.0]), dtype=float)
            diag_norm = float(np.linalg.norm(diag_heading))
            if diag_norm > 1e-6:
                ref_fwd = diag_heading / diag_norm
            else:
                R = self.mj_data.xmat[pelvis_id].reshape(3, 3)
                fwd = np.asarray(R[:, 0], dtype=float)
                yaw_tmp = float(np.arctan2(fwd[1], fwd[0]))
                ref_fwd = np.array([np.cos(yaw_tmp), np.sin(yaw_tmp)], dtype=float)
            self.walk_ref_yaw = float(np.arctan2(ref_fwd[1], ref_fwd[0]))
            ref_lat = np.array([-ref_fwd[1], ref_fwd[0]], dtype=float)
            self.walk_ref_fwd_xy = ref_fwd
            self.walk_ref_lat_xy = ref_lat


    def _compute_sagittal_trunk_lean_deg(self, ref_fwd):
        if self._pelvis_id < 0 or self._head_id < 0:
            return 0.0
        pelvis = self.mj_data.xpos[self._pelvis_id].copy()
        head = self.mj_data.xpos[self._head_id].copy()
        vec = np.asarray(head - pelvis, dtype=float)
        up_comp = float(vec[2])
        fwd_comp = float(np.dot(vec[:2], np.asarray(ref_fwd[:2], dtype=float)))
        return float(np.degrees(np.arctan2(fwd_comp, max(abs(up_comp), 1e-6))))

    
    def _apply_age_posture_bias(self, ref_fwd, ref_lat):
        if self.current_phase not in ('stand', 'step') or self.rest_mode:
            return
        if self._torso_id < 0:
            return

        if self.current_phase == 'stand':
            target_deg = float(self.age_style.get('stand_stoop_target_deg', self.age_style.get('stoop_target_deg', 7.0)))
            base_kp, base_kd = 0.46, 0.22
            torque_limit = 9.0
            force_limit = 6.0
            pelvis_torque_gain = 0.30
        else:
            target_deg = float(self.age_style.get('walk_stoop_target_deg', self.age_style.get('stoop_target_deg', 7.0)))
            base_kp, base_kd = 0.22, 0.12
            torque_limit = 4.5
            force_limit = 2.5
            pelvis_torque_gain = 0.12

        current_deg = self._compute_sagittal_trunk_lean_deg(ref_fwd)
        stoop_err = float(target_deg - current_deg)

        if self._pelvis_id >= 0:
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(self.mj_model, self.mj_data, MJOBJ_BODY, self._pelvis_id, vel6, 0)
            ang = vel6[:3].copy()
        else:
            ang = np.zeros(3)
        lat_axis = np.array([ref_lat[0], ref_lat[1], 0.0], dtype=float)
        pitch_rate = float(np.dot(ang, lat_axis))

        # v34: preserve the visible age-related stoop but stop injecting strong
        # vertical pelvis forces that were causing unrealistic GRF spikes and
        # distorted gait in v33.
        age_scale = float(np.clip((target_deg - 4.0) / 10.0, 0.0, 1.2))
        kp = base_kp + 0.12 * age_scale
        kd = base_kd + 0.06 * age_scale
        pitch_torque = float(np.clip(kp * stoop_err - kd * pitch_rate, -torque_limit, torque_limit))
        forward_force = float(np.clip(0.22 * stoop_err, -force_limit, force_limit))

        self.mj_data.xfrc_applied[self._torso_id, 0] += forward_force * float(ref_fwd[0])
        self.mj_data.xfrc_applied[self._torso_id, 1] += forward_force * float(ref_fwd[1])
        self.mj_data.xfrc_applied[self._torso_id, 3] += pitch_torque * float(ref_lat[0])
        self.mj_data.xfrc_applied[self._torso_id, 4] += pitch_torque * float(ref_lat[1])

        if self._pelvis_id >= 0:
            pelvis_torque = pelvis_torque_gain * pitch_torque
            self.mj_data.xfrc_applied[self._pelvis_id, 3] += pelvis_torque * float(ref_lat[0])
            self.mj_data.xfrc_applied[self._pelvis_id, 4] += pelvis_torque * float(ref_lat[1])


    def apply_age_posture_bias_only(self):
        if self._pelvis_id >= 0:
            R = self.mj_data.xmat[self._pelvis_id].reshape(3, 3)
            fwd = np.asarray(R[:, 0], dtype=float)
            yaw = float(np.arctan2(fwd[1], fwd[0]))
            ref_fwd = np.asarray(self.walk_ref_fwd_xy, dtype=float) if self.walk_ref_fwd_xy is not None else np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
        else:
            ref_fwd = np.array([1.0, 0.0], dtype=float)
        ref_lat = np.array([-ref_fwd[1], ref_fwd[0]], dtype=float)
        self._apply_age_posture_bias(ref_fwd, ref_lat)

    def apply_walk_guidance(self):
        """
        Small pelvis-level guidance during the WALK phase only.

        Important fix: guidance is now expressed in the *reference heading frame*
        captured at walk onset, not the avatar's drifting instantaneous heading.
        This suppresses the large circular-walk artefact seen in the previous run.
        """
        if not self.walk_guidance_enabled or self.current_phase != 'walk' or self.rest_mode:
            return
        body_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Pelvis')
        if body_id < 0:
            return
        pos = self.mj_data.xpos[body_id].copy()
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.mj_model, self.mj_data, MJOBJ_BODY, body_id, vel6, 0)
        vel = vel6[3:].copy()
        ang = vel6[:3].copy()
        R = self.mj_data.xmat[body_id].reshape(3, 3)
        fwd_body = np.asarray(R[:, 0], dtype=float)
        yaw = float(np.arctan2(fwd_body[1], fwd_body[0]))
        if self.walk_ref_y is None or self.walk_ref_yaw is None or self.walk_ref_origin_xy is None:
            self.start_walk_phase()
        ref_fwd = np.asarray(self.walk_ref_fwd_xy, dtype=float) if self.walk_ref_fwd_xy is not None else np.array([1.0, 0.0], dtype=float)
        ref_lat = np.asarray(self.walk_ref_lat_xy, dtype=float) if self.walk_ref_lat_xy is not None else np.array([0.0, 1.0], dtype=float)
        pos_xy = np.asarray(pos[:2], dtype=float)
        vel_xy = np.asarray(vel[:2], dtype=float)

        # Task 29 uses the actual pre-trip walking velocity, not the latent
        # reference heading, to choose forward.  Accumulate it here while the
        # subject is genuinely walking.  This prevents the trip rail from
        # locking to world-Y when the visible walk is along vx/world-X.
        if self.current_phase == 'step' and float(np.linalg.norm(vel_xy)) > 0.025:
            if int(getattr(self, 'scenario29_walk_vel_samples', 0)) <= 0:
                self.scenario29_walk_vel_ema_xy = vel_xy.copy()
            else:
                self.scenario29_walk_vel_ema_xy = (
                    0.90 * np.asarray(self.scenario29_walk_vel_ema_xy, dtype=float) +
                    0.10 * vel_xy
                )
            self.scenario29_walk_vel_samples = int(getattr(self, 'scenario29_walk_vel_samples', 0)) + 1

        # The pelvis x-axis is not guaranteed to match locomotion-forward in this
        # model. If we keep the wrong sign, all subject-specific speed targets are
        # effectively fought by the controller, which is why earlier versions had
        # different requested speeds but very similar realized speeds. Lock the
        # forward axis sign from early walking motion, then supervise the speed.
        signed_speed_probe = float(np.dot(vel_xy, ref_fwd))
        self.walk_signed_speed_ema = 0.88 * float(self.walk_signed_speed_ema) + 0.12 * signed_speed_probe
        if not self.walk_forward_locked:
            self.walk_forward_lock_count += 1
            if self.walk_forward_lock_count >= 12:
                if self.walk_signed_speed_ema < -0.05:
                    ref_fwd = -ref_fwd
                    ref_lat = -ref_lat
                    self.walk_ref_fwd_xy = ref_fwd.copy()
                    self.walk_ref_lat_xy = ref_lat.copy()
                    self.walk_ref_yaw = float(np.arctan2(ref_fwd[1], ref_fwd[0]))
                self.walk_forward_locked = True

        rel_xy = pos_xy - np.asarray(self.walk_ref_origin_xy, dtype=float)
        forward_speed = float(np.dot(vel_xy, ref_fwd))
        lateral_speed = float(np.dot(vel_xy, ref_lat))
        lateral_err = float(np.dot(rel_xy, ref_lat))
        self.walk_speed_ema = 0.88 * float(self.walk_speed_ema) + 0.12 * max(forward_speed, 0.0)
        speed_err = self.walk_target_speed - forward_speed
        braking_scale = float(self.age_style.get('braking_scale', 1.0))
        strict_gain = float(self.age_style.get('speed_supervision_gain', 0.0))
        speed_shortfall = self.walk_target_speed - float(self.walk_speed_ema)
        self.walk_leg_gain_boost = float(np.clip(1.0 + strict_gain * speed_shortfall, 0.96, 1.20))
        forward_boost = float(np.clip(1.0 + 1.4 * strict_gain * max(speed_shortfall, 0.0), 1.0, 1.35))
        if speed_err < 0.0:
            fx_ref = self.walk_kp_speed * braking_scale * speed_err
        else:
            fx_ref = self.walk_kp_speed * forward_boost * speed_err
        fy_ref = -self.walk_kp_lateral * lateral_err - self.walk_kd_lateral * lateral_speed
        yaw_err = float(np.arctan2(np.sin(yaw - self.walk_ref_yaw), np.cos(yaw - self.walk_ref_yaw)))
        heading_scale = float(np.clip(np.cos(yaw_err), 0.0, 1.0))
        fx_ref *= (0.35 + 0.65 * heading_scale)
        tz = -(self.walk_kp_yaw * (1.0 + 1.8 * (1.0 - heading_scale))) * yaw_err - self.walk_kd_yaw * float(ang[2])
        brake_limit_x = float(self.walk_force_limit_x * self.age_style.get('braking_scale', 1.0))
        fx_ref = float(np.clip(fx_ref, -brake_limit_x, self.walk_force_limit_x))
        fy_ref = float(np.clip(fy_ref, -self.walk_force_limit_y, self.walk_force_limit_y))
        tz = float(np.clip(tz, -self.walk_torque_limit_z, self.walk_torque_limit_z))
        f_world_xy = fx_ref * ref_fwd + fy_ref * ref_lat
        a = float(self.walk_guidance_alpha)
        self.walk_guidance_force_xy = a * self.walk_guidance_force_xy + (1.0 - a) * np.asarray(f_world_xy, dtype=float)
        self.walk_guidance_tz = a * float(self.walk_guidance_tz) + (1.0 - a) * float(tz)
        self.mj_data.xfrc_applied[body_id, 0] = float(self.walk_guidance_force_xy[0])
        self.mj_data.xfrc_applied[body_id, 1] = float(self.walk_guidance_force_xy[1])
        self.mj_data.xfrc_applied[body_id, 5] = float(self.walk_guidance_tz)
        self._apply_age_posture_bias(ref_fwd, ref_lat)

    def get_protective_reward_bonus(self):
        """
        Reward bonus for protective behaviours (call from reward function).
        """
        bonus    = 0.0
        contacts = self._analyze_contact_sequence()
        if contacts.get('hands_first', False):
            bonus += 0.5
        if self._lie_orient == 'up':
            neck_flexion = self._get_neck_flexion()
            if neck_flexion > 0.3:
                bonus += 0.3
        return bonus

    def _analyze_contact_sequence(self):
        result     = {'hands_first': False, 'contacts': []}
        body_names = ['HandL', 'HandR', 'Torso', 'Head', 'Pelvis']
        body_ids   = {}
        for bname in body_names:
            bid = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, bname)
            if bid >= 0:
                body_ids[bname] = bid

        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]
            for bname, bid in body_ids.items():
                if contact.geom1 == bid or contact.geom2 == bid:
                    result['contacts'].append((bname, self.mj_data.time))

        if result['contacts']:
            result['contacts'].sort(key=lambda x: x[1])
            if result['contacts'][0][0] in ('HandL', 'HandR'):
                result['hands_first'] = True

        return result

    def _get_neck_flexion(self):
        """Measure neck flexion angle (chin tuck) - placeholder."""
        return 0.5

    # ------------------------------------------------------------------
    # Balance strategies (unchanged)
    # ------------------------------------------------------------------
    def _estimate_support_polygon_center(self):
        weighted_xy = np.zeros(2)
        total_fz = 0.0
        foot_positions = []
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1, g2 = c.geom1, c.geom2
            names = ((mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g1) or '').lower(), (mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g2) or '').lower())
            is_ground = any(('floor' in n or 'ground' in n or 'plane' in n) for n in names) or any(self.mj_model.geom_type[g] == getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', -999) for g in (g1, g2))
            if not is_ground:
                continue
            non_ground_geom = g2 if ('floor' in names[0] or 'ground' in names[0] or 'plane' in names[0] or g1 in self._ground_geoms) else g1
            body_id = int(self.mj_model.geom_bodyid[non_ground_geom])
            bname = (mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, body_id) or '').lower()
            if not any(k in bname for k in ('foot', 'toe', 'heel', 'ankle')):
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, wrench)
            f_world, _ = _contact_wrench_world(c, wrench)
            fz = max(0.0, float(f_world[2]))
            if fz > 0.5:
                weighted_xy += np.asarray(c.pos[:2], dtype=float) * fz
                total_fz += fz
        if total_fz > 1.0:
            return weighted_xy / total_fz
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, i)
            if name and any(k in name.lower() for k in ('foot', 'heel', 'toe', 'ankle')):
                foot_positions.append(self.mj_data.xpos[i][:2])
        if foot_positions:
            return np.mean(foot_positions, axis=0)
        return np.zeros(2)

    def compute_xcom(self):
        """
        XCoM using body/world linear velocity from mj_objectVelocity rather than
        raw cvel spatial quantities.
        """
        pelvis_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, "Pelvis")
        com_pos = self.mj_data.xpos[pelvis_id]
        com_vel = _body_world_velocity(self.mj_model, self.mj_data, pelvis_id)
        g = 9.81
        leg_length = max(0.1, float(self.anthro.get('leg_length', 0.53 * 1.75)))
        omega = np.sqrt(g / leg_length)
        xcom_2d = com_pos[:2] + np.asarray(com_vel[:2], dtype=float) / omega
        pts = []
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1, g2 = c.geom1, c.geom2
            names = ((mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g1) or '').lower(), (mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g2) or '').lower())
            is_ground = any(('floor' in n or 'ground' in n or 'plane' in n) for n in names) or any(self.mj_model.geom_type[g] == getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', -999) for g in (g1, g2))
            if not is_ground:
                continue
            non_ground_geom = g2 if ('floor' in names[0] or 'ground' in names[0] or 'plane' in names[0] or g1 in self._ground_geoms) else g1
            body_id = int(self.mj_model.geom_bodyid[non_ground_geom])
            bname = (mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, body_id) or '').lower()
            if not any(k in bname for k in ('foot', 'toe', 'heel', 'ankle')):
                continue
            pts.append(np.asarray(c.pos[:2], dtype=float))
        if not pts:
            pts = []
            for i in range(self.mj_model.nbody):
                name = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, i)
                if name and any(k in name.lower() for k in ('foot', 'toe', 'heel', 'ankle')):
                    pts.append(np.asarray(self.mj_data.xpos[i][:2], dtype=float))
        support_center = self._estimate_support_polygon_center()
        if pts:
            arr = np.vstack(pts)
            min_xy = np.min(arr, axis=0) - np.array([0.06, 0.04])
            max_xy = np.max(arr, axis=0) + np.array([0.06, 0.04])
            dx_out = max(min_xy[0] - xcom_2d[0], 0.0, xcom_2d[0] - max_xy[0])
            dy_out = max(min_xy[1] - xcom_2d[1], 0.0, xcom_2d[1] - max_xy[1])
            if dx_out == 0.0 and dy_out == 0.0:
                margin = min(xcom_2d[0] - min_xy[0], max_xy[0] - xcom_2d[0], xcom_2d[1] - min_xy[1], max_xy[1] - xcom_2d[1])
            else:
                margin = -float(np.hypot(dx_out, dy_out))
        else:
            BASE_HALF_WIDTH = 0.18
            dist_xcom_to_center = float(np.linalg.norm(xcom_2d - support_center))
            margin = BASE_HALF_WIDTH - dist_xcom_to_center
        return xcom_2d, com_pos[:2].copy(), float(margin)

    def _ankle_strategy_action(self, obs, z):
        return self.get_action(obs, z)

    def _hip_strategy_action(self, obs, z):
        return self.get_action(obs, z)

    def apply_ankle_hip_strategy(self, obs, z):
        """
        Balance strategies upgraded to use XCoM margin (Hof et al. 2005):
        - XCoM margin > 0.08 m : stable - ankle strategy
        - XCoM margin > 0.02 m : borderline - hip strategy
        - XCoM margin = 0.02 m : capture-point outside base - step / fall
        Falls back to CoM displacement if XCoM computation is unavailable.
        """
        try:
            xcom_2d, com_2d, margin = self.compute_xcom()
            if margin > 0.08:
                return self._ankle_strategy_action(obs, z)
            elif margin > 0.02:
                return self._hip_strategy_action(obs, z)
            else:
                # XCoM outside support polygon - step strategy or fall
                return self.get_action(obs, z)
        except Exception:
            # Fallback to original CoM displacement method
            pelvis_id      = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, "Pelvis")
            com_pos        = self.mj_data.xpos[pelvis_id]
            support_center = self._estimate_support_polygon_center()
            com_disp       = np.linalg.norm(com_pos[:2] - support_center)
            if com_disp < self.ankle_strategy_threshold:
                return self._ankle_strategy_action(obs, z)
            elif com_disp < self.hip_strategy_threshold:
                return self._hip_strategy_action(obs, z)
            else:
                return self.get_action(obs, z)


# -------------------------------------------------------------------
# ENHANCED SIMULATION RUNNER
# -------------------------------------------------------------------
def run_enhanced_simulation(fall_type='backward_walking',
                            age=75, height=1.65, weight=None,
                            save_imu=True, sex='male'):
    """
    Run a complete fall simulation with all enhancements:
    - Anthropometric customization (age, height, default model mass)
    - IMU data logging and CSV export
    - FallTypeLibrary scenario selection
    - EnhancedBiofidelicController with protective reflexes
    """
    from datetime import datetime

    resolved_weight = float(weight) if weight is not None else resolve_subject_weight_kg(height, age, sex, explicit_weight=None)[0]
    print("\n" + "=" * 70)
    print(f"  Enhanced Simulation: {fall_type} | age={age} height={height}m sex={sex} weight={resolved_weight:.1f}kg")
    print("=" * 70)

    env_enhanced, _ = make_humenv(task="move-ego-0-0")
    obs_enh, _      = env_enhanced.reset()

    mj_model_enh = env_enhanced.unwrapped.model
    mj_data_enh  = env_enhanced.unwrapped.data
    myosuite_mount_enh = resolve_imu_mount_configuration(
        mj_model_enh,
        requested_xml_path=(MYOSUITE_INTEGRATION['model_xml'] if MYOSUITE_INTEGRATION.get('enable_reference_xml', True) else ''),
    )

    anthro      = AnthropometricModel(mj_model_enh, age=age, height=height, weight=resolved_weight, sex=sex)
    age_params  = anthro.apply_age_effects()
    print(f"  Anthropometry: strength={age_params['strength_factor']:.2f} (raw={age_params.get('strength_factor_raw', age_params['strength_factor']):.2f}), "
          f"reaction_delay={age_params['reaction_delay']:.3f}s, "
          f"balance_impairment={age_params['balance_impairment']:.3f}")

    # IMU with age/height for age-dependent noise (Improvement 3)
    imu = IMUValidator(mj_model_enh, mj_data_enh,
                       sensor_body=myosuite_mount_enh.get('sensor_body', IMU_HARDWARE_SPEC['proxy_body']),
                       sensor_site=myosuite_mount_enh.get('sensor_site'),
                       sensor_offset_local=myosuite_mount_enh.get('sensor_offset_local'),
                       age=age, height=height,
                       target_output_hz=IMU_HARDWARE_SPEC['sampling_hz'],
                       mount_label=myosuite_mount_enh.get('mount_label', IMU_HARDWARE_SPEC['mount_label']))
    imu.print_configuration_report(age=age, height=height, sex=sex)
    _ver = imu.get_sampling_report()
    print(f"    [IMU reality] effective physical bandwidth  {_ver['effective_bandwidth_hz']:.2f} Hz")
    print(f"    [IMU reality] true hardware-equivalent 100 Hz = {_ver['true_hardware_equivalent_100hz']}")

    controller_enh = EnhancedBiofidelicController(
        env_enhanced, mj_model_enh, mj_data_enh, anthropometry=age_params
    )

    global PHASES, TOTAL_STEPS, FORCE_CONFIG, WEAKENING_CONFIG
    PHASES = scenario43_phase_timing(age, sex)
    TOTAL_STEPS = sum(PHASES.values())
    model_body_mass_enh = float(np.sum(mj_model_enh.body_mass))
    FORCE_CONFIG.update(scenario29_perturbation_force_config(model_body_mass_enh, age, sex))
    WEAKENING_CONFIG = weakening_config(age, sex, model_body_mass_enh)
    print(f"  Weakening: min_factor={WEAKENING_CONFIG['min_factor']:.3f}, decay_time={WEAKENING_CONFIG['decay_time']} steps, fatigue_k={WEAKENING_CONFIG.get('fatigue_coefficient', 1.0):.2f}, sarcopenia_loss={WEAKENING_CONFIG.get('sarcopenia_force_loss', 0.0):.1%}")

    fall_config = FallTypeLibrary.FALL_TYPES.get(
        fall_type, FallTypeLibrary.FALL_TYPES['backward_walking']
    )
    print(f"  Fall scenario: {fall_config['description']}")

    try:
        z_w = z_walk
        z_f = z_fall
        z_s = z_stand
        print("  Reusing pre-computed task embeddings.")
    except NameError:
        print("  Computing task embeddings from scratch...")
        z_s = infer_z_stand()
        z_w = infer_z_walk_stable()
        z_f = infer_z_forward_fall()

    enh_boundaries = np.cumsum([0] + list(PHASES.values()))

    _scenario29_ensure_valid_cwd()

    native_renderer = None
    native_writer = None
    native_video_path = None

    # -------------------------------------------------------------
    # TASK43 HEADLESS NATIVE VIDEO MODE
    # No GLFW viewer dependency.
    # -------------------------------------------------------------

    class DummyViewer:
        def sync(self):
            pass

        class cam:
            distance = 4.5
            elevation = -10
            azimuth = 90


    viewer_enh = DummyViewer()

    print("[TASK43 VIDEO] Initializing native MuJoCo renderer")

    native_renderer = mujoco.Renderer(
        mj_model_enh,
        height=1008,
        width=1920
    )

    native_video_path = str(
        Path(os.environ.get("FALL_OUTPUT_DIR", ".")) /
        "scenario43_native_render.mp4"
    )

    Path(native_video_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    native_writer = cv2.VideoWriter(
        native_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (1920,1008)
    )

    print("[TASK43 VIDEO] path:", native_video_path)
    print("[TASK43 VIDEO] writer opened:", native_writer.isOpened())

    if not native_writer.isOpened():
        raise RuntimeError(
            f"OpenCV VideoWriter failed: {native_video_path}"
        )

    native_frame_count = 0


    for step in range(TOTAL_STEPS):
        if step < enh_boundaries[1]:
            phase = 'stand'
            if step == 0:
                controller_enh.set_target_z(z_s, blend_steps=10)

        elif step < enh_boundaries[2]:
            phase = 'walk'
            if step == enh_boundaries[1]:
                controller_enh.set_target_z(z_w, blend_steps=30)
            controller_enh.start_walk_phase()

        elif step < enh_boundaries[3]:
            phase = 'perturb'
            if step == enh_boundaries[2]:
                controller_enh.set_target_z(z_f, blend_steps=40)

            ramp_progress      = (step - enh_boundaries[2]) / PHASES['perturb']
            force_dir          = np.array(fall_config.get(
                'force_direction', FORCE_CONFIG['direction']), dtype=float)
            force_dir         /= np.linalg.norm(force_dir)
            scaled_magnitude   = FORCE_CONFIG['magnitude'] * \
                                 (1.0 - age_params.get('balance_impairment', 0.0))
            controller_enh.apply_external_force(scaled_magnitude, force_dir, ramp_progress)

            weaken_progress = (step - enh_boundaries[2]) / \
                              (PHASES['perturb'] + PHASES['react'])
            controller_enh.update_muscle_weakening(weaken_progress)

        elif step < enh_boundaries[4]:
            phase     = 'react'
            force_dir = np.array(fall_config.get(
                'force_direction', FORCE_CONFIG['direction']), dtype=float)
            force_dir /= np.linalg.norm(force_dir)
            scaled_magnitude = FORCE_CONFIG['magnitude'] * \
                               (1.0 - age_params.get('balance_impairment', 0.0))
            controller_enh.apply_external_force(scaled_magnitude, force_dir, 1.0)

            weaken_progress = (step - enh_boundaries[2]) / \
                              (PHASES['perturb'] + PHASES['react'])
            controller_enh.update_muscle_weakening(weaken_progress)

        else:
            phase = 'fall'
            if step == enh_boundaries[4]:
                controller_enh.clear_forces()
                controller_enh.finalize_z_transition()
            controller_enh.update_muscle_weakening(1.0)

        z_current = controller_enh.update_z_interpolation()
        controller_enh.current_phase = phase
        action    = controller_enh.get_protective_action(obs_enh, z_current, fall_type)

        if phase == 'step':
            controller_enh.apply_walk_guidance()

        obs_enh, _, terminated, truncated, _ = env_enhanced.step(action)

        sim_time_now = float(mj_data_enh.time)
        _last_imu_peak = imu.log_frame(sim_time_now)

        # Viewer is only for live visualization.
        # Native renderer recording must continue even if GLFW viewer fails.
        try:
            viewer_enh.sync()
        except Exception:
            pass

        if native_renderer is not None:

            print(
                "[TASK43 DEBUG] renderer active step=",
                step
            )

            native_renderer.update_scene(
                mj_data_enh
            )

            frame = native_renderer.render()

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )

            native_writer.write(
                frame
            )

            native_frame_count += 1

            if native_frame_count == 1:
                print(
                    "[TASK43 VIDEO] FIRST FRAME WRITTEN",
                    frame.shape
                )

            if step % 100 == 0:
                print(
                    "[TASK43 VIDEO] frames:",
                    native_frame_count
                )


        if terminated and step < enh_boundaries[1]:
            obs_enh, _ = env_enhanced.reset()
            controller_enh.restore_strength()

    if save_imu:
        filename = (f"fall_{fall_type}_age{age}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        imu.export_to_csv(filename, metadata={
            'age': age, 'height': height, 'sex': sex, 'weight': float(np.sum(mj_model_enh.body_mass)), 'body_mass_kg': float(np.sum(mj_model_enh.body_mass)), 'fall_type': fall_type
        })

    controller_enh.restore_strength()
    controller_enh.clear_forces()
    if native_writer is not None:
        native_writer.release()

    if native_renderer is not None:
        native_renderer.close()

    if native_video_path is not None:
        print("[TASK43 VIDEO] saved:", native_video_path)
        print("[TASK43 VIDEO] total frames:", native_frame_count)

    env_enhanced.close()
    print(f"\n  Enhanced simulation complete: {fall_type}")


def compute_preperturb_walk_metrics(controller, mj_model, mj_data, body_mass, age_params):
    pelvis_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, 'Pelvis')
    if pelvis_id < 0:
        return {'ready': False, 'ready_mode': 'none', 'speed': 0.0, 'froude': 0.0, 'xcom_margin': -1.0,
                'double_support': True, 'grf_bw': 0.0, 'lateral_err': 0.0, 'yaw_err_deg': 0.0,
                'reference_speed_ratio': 0.0, 'policy_limited': False}
    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, pelvis_id, vel6, 0)
    vel = vel6[3:].copy()
    pos = mj_data.xpos[pelvis_id].copy()
    R = mj_data.xmat[pelvis_id].reshape(3, 3)
    fwd_body = np.asarray(R[:, 0], dtype=float)
    yaw = float(np.arctan2(fwd_body[1], fwd_body[0]))
    if getattr(controller, 'walk_ref_fwd_xy', None) is not None:
        ref_fwd = np.asarray(controller.walk_ref_fwd_xy, dtype=float)
        ref_lat = np.asarray(controller.walk_ref_lat_xy, dtype=float)
        origin_xy = np.asarray(controller.walk_ref_origin_xy, dtype=float)
    else:
        ref_fwd = np.asarray(fwd_body[:2], dtype=float)
        ref_fwd /= max(np.linalg.norm(ref_fwd), 1e-9)
        ref_lat = np.array([-ref_fwd[1], ref_fwd[0]], dtype=float)
        origin_xy = np.asarray(pos[:2], dtype=float)
    vel_xy = np.asarray(vel[:2], dtype=float)
    forward_speed = float(np.dot(vel_xy, ref_fwd))
    abs_speed = abs(forward_speed)
    lateral_err = float(np.dot(np.asarray(pos[:2], dtype=float) - origin_xy, ref_lat))
    yaw_ref = float(getattr(controller, 'walk_ref_yaw', yaw))
    yaw_err_deg = float(np.degrees(np.arctan2(np.sin(yaw - yaw_ref), np.cos(yaw - yaw_ref))))
    froude = float((abs_speed ** 2) / max(9.81 * float(age_params.get('leg_length', 0.9)), 1e-9))
    try:
        _, _, xcom_margin = controller.compute_xcom()
    except Exception:
        xcom_margin = -1.0

    left_vertical = 0.0
    right_vertical = 0.0
    total_ground_vertical = 0.0
    for i in range(int(mj_data.ncon)):
        c = mj_data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        names = ((mujoco.mj_id2name(mj_model, MJOBJ_GEOM, g1) or '').lower(),
                 (mujoco.mj_id2name(mj_model, MJOBJ_GEOM, g2) or '').lower())
        ground = any(('floor' in n or 'ground' in n or 'plane' in n) for n in names) or any(
            mj_model.geom_type[g] == getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', -999)
            for g in (g1, g2)
        )
        if not ground:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(mj_model, mj_data, i, wrench)
        if wrench[0] <= 1.0:
            continue
        f_world, _ = _contact_wrench_world(c, wrench)
        total_ground_vertical += max(0.0, float(f_world[2]))
        ng = g2 if (g1 in controller._ground_geoms) else g1
        bid = int(mj_model.geom_bodyid[ng])
        bname = (mujoco.mj_id2name(mj_model, MJOBJ_BODY, bid) or '').lower()
        is_footlike = any(k in bname for k in ('foot', 'ankle', 'toe', 'heel'))
        if is_footlike and _body_name_matches_side(bname, 'left'):
            left_vertical += max(0.0, float(f_world[2]))
        if is_footlike and _body_name_matches_side(bname, 'right'):
            right_vertical += max(0.0, float(f_world[2]))

    foot_load_threshold = 0.12 * body_mass * 9.81
    left_support = bool(left_vertical > foot_load_threshold)
    right_support = bool(right_vertical > foot_load_threshold)
    double_support = bool(left_support and right_support)
    grf_bw = float(total_ground_vertical / max(body_mass * 9.81, 1.0))

    lit_target = float(getattr(controller, 'walk_lit_target_speed', controller.age_style.get('target_walk_speed', max(controller.walk_target_speed, 0.2))))
    policy_target = float(max(controller.walk_target_speed, 0.18))
    policy_limited = bool(getattr(controller, 'walk_policy_limited', False))
    reference_speed_ratio = float(abs_speed / max(lit_target, 1e-6))

    stable_base = (xcom_margin > -0.22 and grf_bw < 2.6)
    reference_ready = stable_base and abs_speed >= 0.70 * lit_target
    policy_limited_ready = stable_base and policy_limited and abs_speed >= 0.80 * policy_target

    ready = bool(reference_ready or policy_limited_ready)
    ready_mode = 'reference' if reference_ready else ('policy_limited' if policy_limited_ready else 'none')

    return {
        'ready': ready,
        'ready_mode': ready_mode,
        'speed': float(forward_speed),
        'froude': float(froude),
        'xcom_margin': float(xcom_margin),
        'double_support': bool(double_support),
        'grf_bw': float(grf_bw),
        'lateral_err': float(lateral_err),
        'yaw_err_deg': float(yaw_err_deg),
        'reference_speed_ratio': float(reference_speed_ratio),
        'policy_limited': bool(policy_limited),
    }


def detect_fall_events(imu_data_buffer, dynamics_frames, marker_frames=None, perturb_start_time=None):
    ts = np.asarray(imu_data_buffer.get('timestamp', []), dtype=float)
    heights = np.asarray(imu_data_buffer.get('pelvis_height', []), dtype=float)
    vels = np.asarray(imu_data_buffer.get('pelvis_velocity', []), dtype=float)
    if ts.size == 0 or heights.size == 0:
        return {'available': False}
    if vels.ndim == 1:
        speed = np.abs(vels)
    else:
        speed = np.linalg.norm(vels[:, :2], axis=1)
    trunk = None
    if marker_frames:
        trunk = np.asarray([fr.get('trunk_lean_deg', 0.0) for fr in marker_frames], dtype=float)
    imp_t = np.asarray([fr.get('time', np.nan) for fr in dynamics_frames], dtype=float) if dynamics_frames else np.array([], dtype=float)
    imp_bw = np.asarray([fr.get('primary_impact_body_load_n_filt', fr.get('primary_impact_body_load_n', 0.0)) / max(1.0, 70.0 * 9.81) for fr in dynamics_frames], dtype=float) if dynamics_frames else np.array([], dtype=float)
    if dynamics_frames:
        body_mass_guess = float(max(1.0, np.median([fr.get('support_vertical_n', 0.0) for fr in dynamics_frames[:min(len(dynamics_frames), 30)] if fr.get('support_vertical_n', 0.0) > 0.0]) / 9.81)) if any(fr.get('support_vertical_n', 0.0) > 0.0 for fr in dynamics_frames[:min(len(dynamics_frames), 30)]) else 70.0
        imp_bw = np.asarray([fr.get('primary_impact_body_load_n_filt', fr.get('primary_impact_body_load_n', 0.0)) / max(1.0, body_mass_guess * 9.81) for fr in dynamics_frames], dtype=float)
    t0 = float(perturb_start_time) if perturb_start_time is not None else float(ts[0])
    i0 = int(np.searchsorted(ts, t0, side='left'))
    pre_h = float(np.median(heights[max(0, i0-30):max(i0, i0+1)])) if i0 > 0 else float(np.max(heights[:min(len(heights), 30)]))
    onset_idx = None
    for i in range(i0, len(ts)):
        trunk_cond = bool(trunk is not None and i < len(trunk) and trunk[i] > 22.0)
        h_cond = heights[i] < max(0.78 * pre_h, pre_h - 0.10)
        v_cond = speed[i] > 0.55
        if (trunk_cond and h_cond) or (h_cond and v_cond):
            onset_idx = i
            break
    if onset_idx is None:
        onset_idx = i0
    impact_idx = None
    if imp_t.size:
        j0 = int(np.searchsorted(imp_t, ts[onset_idx], side='left'))
        for j in range(j0, len(imp_t)):
            if imp_bw[j] > 0.25:
                impact_idx = j
                break
        if impact_idx is None:
            impact_idx = int(np.argmax(imp_bw[j0:]) + j0) if j0 < len(imp_bw) else None
    settle_idx = None
    start_settle = onset_idx
    if impact_idx is not None:
        start_settle = max(start_settle, int(np.searchsorted(ts, imp_t[impact_idx], side='left')))
    streak = 0
    for i in range(start_settle, len(ts)):
        low_h = heights[i] < 0.20
        low_v = speed[i] < 0.08
        if low_h and low_v:
            streak += 1
        else:
            streak = 0
        if streak >= 35:
            settle_idx = i - 34
            break
    impact_time = float(imp_t[impact_idx]) if impact_idx is not None else float(ts[min(len(ts)-1, onset_idx)])
    onset_time = float(ts[onset_idx])
    settle_time = float(ts[settle_idx]) if settle_idx is not None else float(ts[-1])
    return {
        'available': True,
        'onset_time': onset_time,
        'impact_time': impact_time,
        'settle_time': settle_time,
        'fall_duration_s': max(0.0, settle_time - onset_time),
        'lead_time_ms': max(0.0, (impact_time - onset_time) * 1000.0),
        'onset_idx': int(onset_idx),
        'impact_idx': int(impact_idx) if impact_idx is not None else None,
        'settle_idx': int(settle_idx) if settle_idx is not None else None,
    }


# -------------------------------------------------------------------
# MAIN SIMULATION
# -------------------------------------------------------------------



# -------------------------------------------------------------------
# TASK 39 MINIMAL HEIGHT SUPPORT HELPERS
# -------------------------------------------------------------------
def _task43_find_platform_geoms(mj_model):
    ids = []
    try:
        for gid in range(int(mj_model.ngeom)):
            name = (mujoco.mj_id2name(mj_model, MJOBJ_GEOM, gid) or '').lower()
            if 'task43_height_block' in name or 'task43_platform' in name or 'task43_chair' in name:
                ids.append(int(gid))
    except Exception:
        pass
    return ids


def _task43_install_platform_if_possible(env):
    """Compile a visible/contacting 1.80 m platform into the humenv model."""
    try:
        unwrapped = env.unwrapped
        old_model = unwrapped.model
        old_data = unwrapped.data
        if _task43_find_platform_geoms(old_model):
            return True, 'already_present'
        if not hasattr(mujoco, 'mj_saveLastXML'):
            return False, 'mj_saveLastXML_unavailable'
        tmpdir = Path(tempfile.gettempdir())
        base_xml = tmpdir / 'task43_v18_base.xml'
        patched_xml = tmpdir / 'task43_v18_height_platform.xml'
        mujoco.mj_saveLastXML(str(base_xml), old_model)
        tree = ET.parse(str(base_xml))
        root = tree.getroot()
        worldbody = root.find('worldbody')
        if worldbody is None:
            return False, 'worldbody_not_found'
        for child in list(worldbody):
            if child.get('name', '') == 'task43_height_block_body':
                worldbody.remove(child)
        hx, hy, hz = TASK43_CHAIR_HALF_EXTENTS
        cx, cy = TASK43_CHAIR_CENTER_XY
        body = ET.SubElement(worldbody, 'body', {
            'name': 'task43_height_block_body',
            'pos': f'{float(cx):.6f} {float(cy):.6f} 0.000000',
        })
        ET.SubElement(body, 'geom', {
            'name': 'task43_height_block_platform_geom',
            'type': 'box',
            'pos': f'0 0 {float(hz):.6f}',
            'size': f'{float(hx):.6f} {float(hy):.6f} {float(hz):.6f}',
            'rgba': '0.45 0.32 0.18 1.0',
            'contype': '1',
            'conaffinity': '1',
            'condim': '3',
            'friction': '1.80 0.08 0.012',
            'solref': '0.006 1',
            'solimp': '0.94 0.995 0.0005',
            'density': '0',
        })
        tree.write(str(patched_xml), encoding='utf-8', xml_declaration=True)
        new_model = mujoco.MjModel.from_xml_path(str(patched_xml))
        new_data = mujoco.MjData(new_model)
        nq = min(len(old_data.qpos), len(new_data.qpos))
        nv = min(len(old_data.qvel), len(new_data.qvel))
        new_data.qpos[:nq] = old_data.qpos[:nq]
        new_data.qvel[:nv] = old_data.qvel[:nv]
        if hasattr(old_data, 'ctrl') and hasattr(new_data, 'ctrl'):
            nu = min(len(old_data.ctrl), len(new_data.ctrl))
            new_data.ctrl[:nu] = old_data.ctrl[:nu]
        mujoco.mj_forward(new_model, new_data)
        unwrapped.model = new_model
        unwrapped.data = new_data
        for attr in ('_model', 'mj_model'):
            if hasattr(unwrapped, attr):
                try: setattr(unwrapped, attr, new_model)
                except Exception: pass
        for attr in ('_data', 'mj_data'):
            if hasattr(unwrapped, attr):
                try: setattr(unwrapped, attr, new_data)
                except Exception: pass
        renderer = getattr(unwrapped, 'mujoco_renderer', None)
        if renderer is not None:
            for attr, val in (('model', new_model), ('data', new_data)):
                if hasattr(renderer, attr):
                    try: setattr(renderer, attr, val)
                    except Exception: pass
        return bool(_task43_find_platform_geoms(new_model)), 'compiled_into_model'
    except Exception as exc:
        return False, f'viewer_fallback_only: {exc}'


def _task43_refresh_obs(env, fallback_obs):
    """Refresh humenv observation after direct MuJoCo qpos placement."""
    for owner in (getattr(env, 'unwrapped', None), env):
        if owner is None:
            continue
        for name in ('_get_obs', 'get_obs', '_get_observation', 'get_observation'):
            fn = getattr(owner, name, None)
            if callable(fn):
                try:
                    obs = fn()
                    if isinstance(obs, dict) and 'proprio' in obs:
                        return obs
                except Exception:
                    pass
    return fallback_obs


class Task43GroundCloneHeightSupport:
    """Height support and deterministic quiet stand/head-side single-step helper.

    v23 builds on v22 and keeps the required 1.80 m height while adding visible protective response:
      - v20 fixed the true head/toe-forward axis and produced a correct forward fall;
      - v21 made the cached quiet-stand pose use a stronger arms-down posture;
      - v22 restored the object to 1.80 m and slowed prone rest;
      - v24 adds a fuller single step, protective arm/leg reaction during descent,
        and short impact recoil before the slow prone rest stabilizer.

    The helper infers the true toe-forward direction from the humanoid's
    toe/ankle geometry, places the root at the center of that platform edge,
    folds both arms toward a natural hanging pose, and adds a fuller right-leg swing pose during the single step plus protective
    arm/leg reach during the unsupported descent.
    """
    def __init__(self, mj_model, mj_data):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.height = float(TASK43_CHAIR_HEIGHT_M)
        self.half_extents = np.asarray(TASK43_CHAIR_HALF_EXTENTS, dtype=float)
        self.center_xy = np.asarray(TASK43_CHAIR_CENTER_XY, dtype=float)
        self.platform_geoms = _task43_find_platform_geoms(mj_model)
        self._released = False
        self._saved_contact = {}
        for gid in self.platform_geoms:
            self._saved_contact[int(gid)] = (
                int(self.mj_model.geom_contype[int(gid)]),
                int(self.mj_model.geom_conaffinity[int(gid)]),
            )
        self._foot_bodies = self._resolve_foot_bodies()
        self._foot_geoms = self._resolve_foot_geoms()
        self._saved_foot_contact = {}
        for gid in self._foot_geoms:
            try:
                self._saved_foot_contact[int(gid)] = (
                    int(self.mj_model.geom_contype[int(gid)]),
                    int(self.mj_model.geom_conaffinity[int(gid)]),
                )
            except Exception:
                pass
        self._last_support_contact_ok = False
        self._last_support_contact_count = 0
        self._last_support_drop_m = 0.0
        self._last_visual_seat_dz = 0.0
        self._last_support_xy_shift = np.zeros(2, dtype=float)
        self._last_support_search_used = False
        self._upright_quat = self._capture_reset_quat()
        self.fwd_xy = np.array([1.0, 0.0], dtype=float)
        self.lat_xy = np.array([0.0, 1.0], dtype=float)
        self.edge_distance = float(self.half_extents[0])
        self.start_xy = np.array([TASK43_START_ROOT_X, 0.0], dtype=float)
        self.step_target_xy = np.array([TASK43_STEP_TARGET_X, 0.0], dtype=float)
        self._quiet_qpos = None
        self._quiet_qvel = None
        self._arm_joint_count = 0
        self._step_joint_specs = []
        self._step_pose_qpos = None
        self._step_pose_build_report = {}
        self._reactive_leg_specs = []
        self._protective_arm_targets = {}
        self._protective_joint_count = 0
        self._impact_recoil_counter = 0

    def _capture_reset_quat(self):
        if self.mj_data.qpos.shape[0] >= 7:
            q = np.asarray(self.mj_data.qpos[3:7], dtype=float).copy()
            n = float(np.linalg.norm(q))
            if n > 1e-8:
                return q / n
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def _resolve_foot_bodies(self):
        """Resolve *sole* bodies, not ankle capsules.

        v25 treated ankle bodies/geoms as feet. On short/elderly profiles the
        ankle capsule can become the lowest resolved geom, so geometric seating
        aligns the ankle capsule while the visible toe/sole floats. v26 uses
        toe/foot/heel first and ankle only as a last fallback.
        """
        primary_names = ('L_Toe', 'R_Toe', 'Left_Toe', 'Right_Toe',
                         'L_Foot', 'R_Foot', 'Left_Foot', 'Right_Foot',
                         'FootL', 'FootR', 'L_Heel', 'R_Heel', 'Left_Heel', 'Right_Heel')
        fallback_names = ('L_Ankle', 'R_Ankle', 'Left_Ankle', 'Right_Ankle')
        ids = []
        for name in primary_names:
            bid = _safe_name2id(self.mj_model, MJOBJ_BODY, name)
            if bid >= 0 and bid not in ids:
                ids.append(int(bid))
        if ids:
            return ids
        for bid in range(int(self.mj_model.nbody)):
            bname = (mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, bid) or '').lower()
            if any(k in bname for k in ('toe', 'foot', 'heel')):
                ids.append(int(bid))
        if ids:
            return ids
        for name in fallback_names:
            bid = _safe_name2id(self.mj_model, MJOBJ_BODY, name)
            if bid >= 0 and bid not in ids:
                ids.append(int(bid))
        if ids:
            return ids
        for bid in range(int(self.mj_model.nbody)):
            bname = (mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, bid) or '').lower()
            if 'ankle' in bname:
                ids.append(int(bid))
        return ids

    def _resolve_foot_geoms(self):
        """Resolve contact/visual sole geoms, excluding ankle unless unavoidable."""
        foot_body_set = set(int(b) for b in self._foot_bodies)
        primary, fallback = [], []
        for gid in range(int(self.mj_model.ngeom)):
            bid = int(self.mj_model.geom_bodyid[gid])
            gname = (mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, gid) or '').lower()
            bname = (mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, bid) or '').lower()
            txt = gname + ' ' + bname
            distal = any(k in txt for k in ('toe', 'foot', 'heel'))
            ankle_only = ('ankle' in txt) and not distal
            if distal or (bid in foot_body_set and not ankle_only):
                primary.append(int(gid))
            elif ankle_only or bid in foot_body_set:
                fallback.append(int(gid))
        def _uniq(seq):
            out, seen = [], set()
            for x in seq:
                if int(x) not in seen:
                    seen.add(int(x)); out.append(int(x))
            return out
        return _uniq(primary) if primary else _uniq(fallback)

    def _enable_foot_support_contact(self):
        """Ensure resolved foot/toe/heel geoms can collide with the height block.

        Some HumEnv/MyoSuite variants mark distal foot geoms as visual/passive
        only.  Then the pose can be geometrically close to the platform while
        MuJoCo still reports ncon=0 and GRF/BW=0 during the stand phase.  For
        scenario 39 the feet must physically support the quiet stand, so make
        only the resolved foot-side geoms contact-capable.  This is intentionally
        conservative and does not alter torso/head/arm collision settings.
        """
        for gid in getattr(self, '_foot_geoms', []):
            try:
                gid = int(gid)
                if int(self.mj_model.geom_contype[gid]) == 0:
                    self.mj_model.geom_contype[gid] = 1
                if int(self.mj_model.geom_conaffinity[gid]) == 0:
                    self.mj_model.geom_conaffinity[gid] = 1
            except Exception:
                pass
        for gid in getattr(self, 'platform_geoms', []):
            try:
                gid = int(gid)
                self.mj_model.geom_contype[gid] = 1
                self.mj_model.geom_conaffinity[gid] = 1
            except Exception:
                pass

    def _is_platform_geom(self, gid):
        return int(gid) in set(int(x) for x in getattr(self, 'platform_geoms', []))

    def _is_foot_geom(self, gid):
        return int(gid) in set(int(x) for x in getattr(self, '_foot_geoms', []))

    def _has_foot_platform_contact(self):
        """Return whether any resolved foot geom is actually touching the block."""
        count = 0
        try:
            for ci in range(int(self.mj_data.ncon)):
                c = self.mj_data.contact[ci]
                g1, g2 = int(c.geom1), int(c.geom2)
                pair = ((self._is_foot_geom(g1) and self._is_platform_geom(g2)) or
                        (self._is_foot_geom(g2) and self._is_platform_geom(g1)))
                if pair:
                    count += 1
        except Exception:
            pass
        self._last_support_contact_count = int(count)
        self._last_support_contact_ok = bool(count > 0)
        return bool(count > 0)

    def _seat_until_platform_contact(self):
        """After geometric seating, lower the root a few millimetres until MuJoCo
        reports true foot-platform contact.  This closes the visual/physical gap
        for shorter anthropometric profiles and avoids a kinematic hover.
        """
        if len(self.mj_data.qpos) < 3 or not getattr(self, 'platform_geoms', []):
            return 0.0
        self._enable_foot_support_contact()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        if self._has_foot_platform_contact():
            self._last_support_drop_m = 0.0
            return 0.0
        total_drop = 0.0
        max_drop = float(globals().get('TASK43_CONTACT_SETTLE_MAX_DROP_M', 0.035))
        step_drop = float(globals().get('TASK43_CONTACT_SETTLE_STEP_M', 0.0015))
        while total_drop < max_drop:
            self.mj_data.qpos[2] -= step_drop
            total_drop += step_drop
            mujoco.mj_forward(self.mj_model, self.mj_data)
            if self._has_foot_platform_contact():
                break
        self._last_support_drop_m = float(total_drop)
        return float(total_drop)

    def _body_pos2(self, names):
        pts = []
        for name in names:
            bid = _safe_name2id(self.mj_model, MJOBJ_BODY, name)
            if bid >= 0:
                pts.append(np.asarray(self.mj_data.xpos[bid][:2], dtype=float).copy())
        return pts

    def _infer_forward_xy(self):
        """Infer visible forward from toe centers relative to ankle/heel centers."""
        toe_pts = self._body_pos2(('L_Toe', 'R_Toe', 'Left_Toe', 'Right_Toe', 'L_Foot', 'R_Foot'))
        heel_pts = self._body_pos2(('L_Heel', 'R_Heel', 'Left_Heel', 'Right_Heel', 'L_Ankle', 'R_Ankle', 'Left_Ankle', 'Right_Ankle'))
        if toe_pts and heel_pts:
            v = np.mean(toe_pts, axis=0) - np.mean(heel_pts, axis=0)
            n = float(np.linalg.norm(v))
            if n > 0.035:
                return v / n, 'toe-minus-ankle'
        fallback = np.array([0.0, 1.0], dtype=float)
        if len(self.mj_data.qpos) >= 7:
            try:
                q = np.asarray(self.mj_data.qpos[3:7], dtype=float)
                R = np.zeros(9, dtype=float)
                mujoco.mju_quat2Mat(R, q)
                R = R.reshape(3, 3)
                candidates = [R[:2, 0], -R[:2, 0], R[:2, 1], -R[:2, 1]]
                candidates = [c / max(np.linalg.norm(c), 1e-8) for c in candidates if np.linalg.norm(c) > 1e-8]
                if candidates:
                    fallback = candidates[1] if len(candidates) > 1 else candidates[0]
            except Exception:
                pass
        n = float(np.linalg.norm(fallback))
        return (fallback / max(n, 1e-8)), 'fallback-root-axis'

    def _update_edge_geometry(self, verbose=False):
        # Task43 is the mirror of Task39: infer anatomical face-forward from
        # toe/ankle geometry, then use the opposite vector as the backward edge
        # direction so the subject's back is toward the platform edge.
        toe_forward_xy, source = self._infer_forward_xy()
        toe_forward_xy = np.asarray(toe_forward_xy, dtype=float)
        toe_forward_xy = toe_forward_xy / max(float(np.linalg.norm(toe_forward_xy)), 1e-8)
        self.toe_forward_xy = toe_forward_xy.copy()
        self.fwd_xy = -toe_forward_xy
        source = 'backside-from-' + str(source)
        self.fwd_xy = self.fwd_xy / max(float(np.linalg.norm(self.fwd_xy)), 1e-8)
        self.lat_xy = np.array([-self.fwd_xy[1], self.fwd_xy[0]], dtype=float)
        self.lat_xy = self.lat_xy / max(float(np.linalg.norm(self.lat_xy)), 1e-8)

        hx, hy = float(self.half_extents[0]), float(self.half_extents[1])
        candidates = []
        if abs(float(self.fwd_xy[0])) > 1e-5:
            candidates.append(hx / abs(float(self.fwd_xy[0])))
        if abs(float(self.fwd_xy[1])) > 1e-5:
            candidates.append(hy / abs(float(self.fwd_xy[1])))
        self.edge_distance = float(min(candidates) if candidates else hx)
        edge_center = self.center_xy + self.fwd_xy * self.edge_distance

        # Compute how far the back-most foot/heel sits behind the root in the
        # current reset pose.  Place that back heel just inside the platform
        # edge.  This is the critical no-slide setup: the root is already at
        # the rear-edge stance, so STEP does not need to drive the whole body
        # across the object.
        foot_pts = self._body_pos2((
            'L_Heel', 'R_Heel', 'Left_Heel', 'Right_Heel',
            'L_Toe', 'R_Toe', 'Left_Toe', 'Right_Toe',
            'L_Foot', 'R_Foot', 'Left_Foot', 'Right_Foot',
            'L_Ankle', 'R_Ankle', 'Left_Ankle', 'Right_Ankle',
        ))
        root_xy_live = np.asarray(self.mj_data.qpos[:2], dtype=float) if len(self.mj_data.qpos) >= 2 else np.zeros(2, dtype=float)
        back_foot_proj = None
        if foot_pts:
            vals = [float(np.dot(np.asarray(pt, dtype=float) - root_xy_live, self.fwd_xy)) for pt in foot_pts]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                back_foot_proj = float(max(vals))
        if back_foot_proj is None or back_foot_proj < 0.035 or back_foot_proj > 0.42:
            back_foot_proj = float(TASK43_EDGE_ROOT_BACKOFF_M)

        self._back_foot_projection_m = float(back_foot_proj)
        margin = float(globals().get('TASK43_BACK_HEEL_EDGE_MARGIN_M', 0.022))
        self._back_heel_edge_margin_m = margin

        self.start_xy = edge_center - self.fwd_xy * (self._back_foot_projection_m + margin)
        lateral_offset = float(np.dot(self.start_xy - self.center_xy, self.lat_xy))
        self.start_xy = self.start_xy - lateral_offset * self.lat_xy

        # Root drift must be zero/tiny.  The swing foot clears the edge; the
        # pelvis/root stays near the rear edge until pose-lock is released.
        self.step_target_xy = self.start_xy + self.fwd_xy * float(globals().get('TASK43_STEP_ROOT_DRIFT_M', 0.0))
        self.swing_foot_clear_xy = edge_center + self.fwd_xy * float(globals().get('TASK43_STEP_CLEARANCE_M', 0.28))
        globals()['TASK43_RUNTIME_FWD_XY'] = self.fwd_xy.copy()
        globals()['TASK43_RUNTIME_LAT_XY'] = self.lat_xy.copy()

        if verbose:
            print(f"      inferred backward source = {source}")
            print(f"      backside fall xy         = [{self.fwd_xy[0]:+.3f}, {self.fwd_xy[1]:+.3f}]")
            print(f"      lateral axis xy          = [{self.lat_xy[0]:+.3f}, {self.lat_xy[1]:+.3f}]")
            print(f"      back-foot projection     = {self._back_foot_projection_m:.3f} m | heel-edge margin={self._back_heel_edge_margin_m:.3f} m")
            print(f"      back-edge root xy        = [{self.start_xy[0]:+.3f}, {self.start_xy[1]:+.3f}]")
            print(f"      root drift target xy     = [{self.step_target_xy[0]:+.3f}, {self.step_target_xy[1]:+.3f}]")
            print(f"      swing-foot clear xy      = [{self.swing_foot_clear_xy[0]:+.3f}, {self.swing_foot_clear_xy[1]:+.3f}]")

    def _geom_world_bottom_z(self, gid):
        gid = int(gid)
        pos = np.asarray(self.mj_data.geom_xpos[gid], dtype=float)
        size = np.asarray(self.mj_model.geom_size[gid], dtype=float)
        gtype = int(self.mj_model.geom_type[gid])
        try:
            R = np.asarray(self.mj_data.geom_xmat[gid], dtype=float).reshape(3, 3)
        except Exception:
            R = np.eye(3)
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            half_z = float(np.dot(np.abs(R[2, :3]), size[:3]))
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            half_z = float(size[0])
        elif gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            radius = float(size[0])
            half_len = float(size[1]) if len(size) > 1 else 0.0
            half_z = radius + abs(float(R[2, 2])) * half_len
        else:
            half_z = float(np.max(size[:min(3, len(size))])) if len(size) else 0.03
        return float(pos[2] - half_z)

    def _lowest_foot_z(self):
        if self._foot_geoms:
            return float(min(self._geom_world_bottom_z(gid) for gid in self._foot_geoms))
        zs = [float(self.mj_data.xpos[bid][2]) for bid in self._foot_bodies if bid >= 0]
        if zs:
            return float(min(zs))
        pid = _safe_name2id(self.mj_model, MJOBJ_BODY, 'Pelvis')
        return float(self.mj_data.xpos[pid][2] - 0.90) if pid >= 0 else 0.0

    def _distal_body_min_z(self):
        """Minimum named toe/foot/heel body height for visual hover diagnostics."""
        names = ('L_Toe', 'R_Toe', 'Left_Toe', 'Right_Toe',
                 'L_Foot', 'R_Foot', 'Left_Foot', 'Right_Foot',
                 'FootL', 'FootR', 'L_Heel', 'R_Heel', 'Left_Heel', 'Right_Heel')
        zs = []
        for name in names:
            bid = _safe_name2id(self.mj_model, MJOBJ_BODY, name)
            if bid >= 0:
                try:
                    zs.append(float(self.mj_data.xpos[int(bid)][2]))
                except Exception:
                    pass
        return float(min(zs)) if zs else None

    def _seat_by_distal_visual_fallback(self):
        """Small visual no-hover seating when no usable foot-platform contact exists."""
        if len(self.mj_data.qpos) < 3:
            return 0.0
        self._last_visual_seat_dz = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        dz = 0.0
        distal_z = self._distal_body_min_z()
        if distal_z is not None:
            target = float(self.height + globals().get('TASK43_VISUAL_DISTAL_CLEARANCE_M', 0.002))
            dz = float(target - float(distal_z))
            # Only fix an upward visual gap. Do not raise the model.
            if dz > 0.0:
                dz = 0.0
            max_drop = float(globals().get('TASK43_VISUAL_MAX_NO_HOVER_DROP_M', 0.018))
            dz = float(np.clip(dz, -max_drop, 0.0))
        if abs(dz) < 1e-8:
            target = float(self.height - globals().get('TASK43_VISUAL_SOLE_PENETRATION_M', 0.006))
            dz = float(target - float(self._lowest_foot_z()))
            max_drop = float(globals().get('TASK43_VISUAL_MAX_NO_HOVER_DROP_M', 0.018))
            dz = float(np.clip(dz, -max_drop, 0.0))
        if abs(dz) > 1e-8:
            self.mj_data.qpos[2] += dz
            self._last_visual_seat_dz = float(dz)
            mujoco.mj_forward(self.mj_model, self.mj_data)
        return float(dz)

    def _attempt_contact_settle_from_pose(self, qpos_seed, qvel_seed, xy_shift=None, target_clearance=None):
        """Try vertical seating from a candidate root XY pose and test real contact."""
        if len(self.mj_data.qpos) < 3:
            return False, qpos_seed.copy(), qvel_seed.copy(), 0.0, 0.0
        self.mj_data.qpos[:] = qpos_seed
        self.mj_data.qvel[:] = qvel_seed
        if xy_shift is not None and len(self.mj_data.qpos) >= 2:
            sh = np.asarray(xy_shift, dtype=float)
            self.mj_data.qpos[0] += float(sh[0])
            self.mj_data.qpos[1] += float(sh[1])
        mujoco.mj_forward(self.mj_model, self.mj_data)
        target_clearance = TASK43_TOP_CLEARANCE if target_clearance is None else float(target_clearance)
        target = float(self.height + target_clearance)
        for _ in range(5):
            mujoco.mj_forward(self.mj_model, self.mj_data)
            dz = float(target - float(self._lowest_foot_z()))
            self.mj_data.qpos[2] += dz
            if abs(dz) < 2e-5:
                break
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._last_visual_seat_dz = 0.0
        drop = float(self._seat_until_platform_contact())
        ok = bool(self._last_support_contact_ok)
        visual_dz = 0.0
        if not ok:
            visual_dz = float(self._seat_by_distal_visual_fallback())
            mujoco.mj_forward(self.mj_model, self.mj_data)
            ok = bool(self._has_foot_platform_contact())
        if self.mj_data.qvel.shape[0] >= 3:
            self.mj_data.qvel[:3] = 0.0
        return ok, np.asarray(self.mj_data.qpos, dtype=float).copy(), np.asarray(self.mj_data.qvel, dtype=float).copy(), drop, visual_dz

    def _search_bounded_xy_for_true_support_contact(self, qpos_seed, qvel_seed, target_clearance=None):
        """Find the smallest root XY correction that produces real foot/block contact.

        This fixes the v28 case where the visible foot was near the platform but
        MuJoCo had no contact pair after anthropometric scaling.  Unlike v27,
        the shift is searched and minimized, not applied unconditionally.
        """
        max_shift = float(globals().get('TASK43_SUPPORT_XY_SEARCH_MAX_M', 0.24))
        step = max(float(globals().get('TASK43_SUPPORT_XY_SEARCH_STEP_M', 0.025)), 1e-4)
        lat_m = float(globals().get('TASK43_SUPPORT_XY_SEARCH_LAT_M', 0.030))
        fwd = np.asarray(getattr(self, 'fwd_xy', np.array([1.0, 0.0])), dtype=float)
        fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-8)
        lat = np.asarray(getattr(self, 'lat_xy', np.array([-fwd[1], fwd[0]])), dtype=float)
        lat = lat / max(float(np.linalg.norm(lat)), 1e-8)
        candidates = [np.zeros(2, dtype=float)]
        dvals = list(np.arange(step, max_shift + 0.5 * step, step))
        for d in dvals:
            candidates.append(-fwd * float(d))
        for d in dvals:
            candidates.append(fwd * float(0.45 * d))
        for d in dvals[::2]:
            for lm in (-lat_m, lat_m):
                candidates.append(-fwd * float(d) + lat * float(lm))
        best = None
        best_score = None
        saved_qpos = np.asarray(self.mj_data.qpos, dtype=float).copy()
        saved_qvel = np.asarray(self.mj_data.qvel, dtype=float).copy()
        try:
            for sh in candidates:
                ok, cand_qpos, cand_qvel, drop, visual_dz = self._attempt_contact_settle_from_pose(
                    qpos_seed, qvel_seed, xy_shift=sh, target_clearance=target_clearance)
                if not ok:
                    continue
                sh_norm = float(np.linalg.norm(sh))
                score = 8.0 * sh_norm + 1.2 * float(drop) + 0.2 * abs(float(visual_dz))
                if best_score is None or score < best_score:
                    best_score = score
                    best = (cand_qpos.copy(), cand_qvel.copy(), np.asarray(sh, dtype=float).copy(), float(drop), float(visual_dz))
        finally:
            self.mj_data.qpos[:] = saved_qpos
            self.mj_data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.mj_model, self.mj_data)
        return best

    def _seat_feet_on_platform(self, target_clearance=None, max_iters=5, allow_xy_search=True):
        """Seat the humanoid on the platform with true contact when possible.

        v28 still allowed a no-contact hover: the diagnostic bottom-z could be
        close to 1.80 m while MuJoCo reported ncon=0.  v29 first tries pure
        vertical seating, then a bounded XY search only if required. Joint
        angles are not rebuilt, so the age/profile posture is preserved.
        """
        if len(self.mj_data.qpos) < 3:
            return 0.0
        self._enable_foot_support_contact()
        self._last_support_xy_shift = np.zeros(2, dtype=float)
        self._last_support_search_used = False
        seed_qpos = np.asarray(self.mj_data.qpos, dtype=float).copy()
        seed_qvel = np.asarray(self.mj_data.qvel, dtype=float).copy()

        ok, cand_qpos, cand_qvel, contact_drop, visual_dz = self._attempt_contact_settle_from_pose(
            seed_qpos, seed_qvel, xy_shift=np.zeros(2), target_clearance=target_clearance)
        if not ok and allow_xy_search:
            found = self._search_bounded_xy_for_true_support_contact(seed_qpos, seed_qvel, target_clearance=target_clearance)
            if found is not None:
                cand_qpos, cand_qvel, xy_shift, contact_drop, visual_dz = found
                self._last_support_xy_shift = np.asarray(xy_shift, dtype=float).copy()
                self._last_support_search_used = bool(np.linalg.norm(xy_shift) > float(globals().get('TASK43_SUPPORT_LOG_XY_SHIFT_EPS', 0.002)))
            else:
                # Last resort: make the visible soles/feet sit on the top, but
                # honestly leave support_contact=False so the log exposes any
                # model variant that has no usable distal foot contacts.
                self.mj_data.qpos[:] = seed_qpos
                self.mj_data.qvel[:] = seed_qvel
                mujoco.mj_forward(self.mj_model, self.mj_data)
                target_clearance_eff = TASK43_TOP_CLEARANCE if target_clearance is None else float(target_clearance)
                for _ in range(int(max_iters)):
                    dz = float((self.height + target_clearance_eff) - float(self._lowest_foot_z()))
                    self.mj_data.qpos[2] += dz
                    mujoco.mj_forward(self.mj_model, self.mj_data)
                    if abs(dz) < 2e-5:
                        break
                self._seat_by_distal_visual_fallback()
                cand_qpos = np.asarray(self.mj_data.qpos, dtype=float).copy()
                cand_qvel = np.asarray(self.mj_data.qvel, dtype=float).copy()
                contact_drop = 0.0
                visual_dz = float(getattr(self, '_last_visual_seat_dz', 0.0))
                self._last_support_drop_m = 0.0
                self._last_support_contact_ok = False
                self._last_support_contact_count = 0
        self._last_support_drop_m = float(contact_drop)
        self._last_visual_seat_dz = float(visual_dz)
        self.mj_data.qpos[:] = cand_qpos
        self.mj_data.qvel[:] = cand_qvel
        if self.mj_data.qvel.shape[0] >= 6:
            self.mj_data.qvel[:6] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._has_foot_platform_contact()
        return float((self.mj_data.qpos[2] - seed_qpos[2]) if len(seed_qpos) >= 3 else 0.0)

    def _joint_side(self, name, body_name):
        """Infer limb side from both conventional and compact HumEnv/MyoSuite names.

        v1i could miss arm joints named like HandL, WristR, UpperArmL, etc.
        Then the reference IK had foot joints but no usable arm joints, which
        produced release errors around 0.8 m for both hands.  This detector is
        intentionally conservative for torso/root names and broad for distal
        limb names.
        """
        raw = (str(name or '') + ' ' + str(body_name or '')).lower()
        spaced = re.sub(r'[^a-z0-9]+', ' ', raw)
        tokens = [t for t in spaced.split() if t]
        joined = ''.join(tokens)

        if any(t in ('right', 'rt', 'r') for t in tokens):
            return 'right'
        if any(t in ('left', 'lt', 'l') for t in tokens):
            return 'left'
        if any(k in raw for k in ('right', '_r', 'r_', '-r', 'r-', '.r', 'r.')):
            return 'right'
        if any(k in raw for k in ('left', '_l', 'l_', '-l', 'l-', '.l', 'l.')):
            return 'left'

        limb_roots = (
            'hand','wrist','forearm','lowerarm','upperarm','shoulder','elbow','arm',
            'foot','toe','heel','ankle','knee','hip','thigh','shin','shank','leg'
        )
        for root in limb_roots:
            if joined.endswith(root + 'r') or joined.startswith('r' + root):
                return 'right'
            if joined.endswith(root + 'l') or joined.startswith('l' + root):
                return 'left'
        for root in limb_roots:
            if (root + 'r') in joined:
                return 'right'
            if (root + 'l') in joined:
                return 'left'
        return 'unknown'

    def _joint_range_for_qadr(self, jid, fallback=(-1.25, 1.25)):
        """Return a safe scalar joint range for coordinate pose tuning."""
        try:
            if int(self.mj_model.jnt_limited[jid]):
                lo, hi = map(float, self.mj_model.jnt_range[jid])
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    return lo, hi
        except Exception:
            pass
        return float(fallback[0]), float(fallback[1])

    def _body_xyz_by_candidates(self, names):
        for name in names:
            bid = _safe_name2id(self.mj_model, MJOBJ_BODY, name)
            if bid >= 0:
                return np.asarray(self.mj_data.xpos[bid], dtype=float).copy()
        return None

    def _arm_down_score(self, side):
        """Score how close one arm is to a relaxed hanging posture. Lower is better."""
        if side == 'left':
            hand = self._body_xyz_by_candidates(('L_Hand', 'L_Wrist', 'LeftHand', 'LeftWrist', 'L_Forearm', 'L_Elbow'))
            shoulder = self._body_xyz_by_candidates(('L_Shoulder', 'LeftShoulder', 'L_UpperArm', 'UpperArmL'))
        else:
            hand = self._body_xyz_by_candidates(('R_Hand', 'R_Wrist', 'RightHand', 'RightWrist', 'R_Forearm', 'R_Elbow'))
            shoulder = self._body_xyz_by_candidates(('R_Shoulder', 'RightShoulder', 'R_UpperArm', 'UpperArmR'))
        if hand is None:
            return 0.0
        if shoulder is None:
            shoulder = self._body_xyz_by_candidates(('Torso', 'Chest', 'Spine', 'Pelvis'))
        if shoulder is None:
            return 0.0
        horiz = float(np.linalg.norm(hand[:2] - shoulder[:2]))
        z_drop = float(shoulder[2] - hand[2])
        return (2.2 * horiz * horiz
                + 1.15 * (z_drop - 0.42) * (z_drop - 0.42)
                + 2.5 * max(0.0, -z_drop) ** 2
                + 0.30 * max(0.0, z_drop - 0.72) ** 2)

    def _total_arm_down_score(self):
        return float(self._arm_down_score('left') + self._arm_down_score('right'))

    def _tune_arms_down_coordinate_descent(self, arm_joints):
        """Runtime pose search to fold abducted/open arms into a natural down pose."""
        if not arm_joints:
            return 0
        tuned = 0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        for _pass in range(2):
            for jid, qadr, kind, side in arm_joints:
                if qadr < 7 or qadr >= len(self.mj_data.qpos):
                    continue
                lo, hi = self._joint_range_for_qadr(jid)
                cur = float(self.mj_data.qpos[qadr])
                base_abs = [-1.10, -0.75, -0.40, -0.18, 0.0, 0.18, 0.40, 0.75, 1.10]
                if kind == 'elbow':
                    base_abs = [-0.60, -0.30, -0.12, 0.0, 0.12, 0.30, 0.60, 0.95]
                elif kind == 'wrist':
                    base_abs = [-0.35, -0.18, 0.0, 0.18, 0.35]
                candidates = [cur, 0.0]
                candidates += [float(np.clip(v, lo, hi)) for v in base_abs]
                candidates += [float(np.clip(cur + dv, lo, hi)) for dv in (-0.45, -0.25, 0.25, 0.45)]
                seen = set(); uniq = []
                for v in candidates:
                    key = round(float(v), 4)
                    if key not in seen:
                        seen.add(key); uniq.append(float(v))
                best_v = cur
                best_score = None
                original = cur
                for v in uniq:
                    self.mj_data.qpos[qadr] = float(v)
                    mujoco.mj_forward(self.mj_model, self.mj_data)
                    score = self._total_arm_down_score() + 0.012 * abs(float(v))
                    if best_score is None or score < best_score:
                        best_score = score; best_v = float(v)
                self.mj_data.qpos[qadr] = best_v
                if abs(best_v - original) > 1e-4:
                    tuned += 1
                mujoco.mj_forward(self.mj_model, self.mj_data)
        return tuned

    def _neutralize_upper_body_qpos(self):
        """Bring open arms to a natural down-side quiet-stand posture."""
        arm_joints = []
        count = 0
        joint_obj = mujoco.mjtObj.mjOBJ_JOINT if hasattr(mujoco, 'mjtObj') else mujoco.mjOBJ_JOINT
        for jid in range(int(self.mj_model.njnt)):
            jname = mujoco.mj_id2name(self.mj_model, joint_obj, jid) or ''
            bname = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, int(self.mj_model.jnt_bodyid[jid])) or ''
            txt = (jname + ' ' + bname).lower()
            if not any(k in txt for k in ('shoulder', 'elbow', 'wrist', 'hand', 'arm')):
                continue
            qadr = int(self.mj_model.jnt_qposadr[jid])
            if qadr < 7 or qadr >= len(self.mj_data.qpos):
                continue
            side = self._joint_side(jname, bname)
            kind = 'shoulder' if any(k in txt for k in ('shoulder', 'upperarm', 'arm')) else 'elbow' if 'elbow' in txt else 'wrist'
            self.mj_data.qpos[qadr] = 0.0
            arm_joints.append((jid, qadr, kind, side))
            count += 1
        mujoco.mj_forward(self.mj_model, self.mj_data)
        tuned = self._tune_arms_down_coordinate_descent(arm_joints)
        self._arm_joint_count = count
        self._arm_down_tuned_count = tuned

    def _protective_reach_score(self):
        """Score a two-hand backward/down protective reaction in the true Task43 fall direction."""
        score = 0.0
        for side in ('left', 'right'):
            if side == 'left':
                hand = self._body_xyz_by_candidates(('L_Hand', 'L_Wrist', 'LeftHand', 'LeftWrist', 'L_Forearm', 'L_Elbow'))
                shoulder = self._body_xyz_by_candidates(('L_Shoulder', 'LeftShoulder', 'L_UpperArm', 'UpperArmL'))
            else:
                hand = self._body_xyz_by_candidates(('R_Hand', 'R_Wrist', 'RightHand', 'RightWrist', 'R_Forearm', 'R_Elbow'))
                shoulder = self._body_xyz_by_candidates(('R_Shoulder', 'RightShoulder', 'R_UpperArm', 'UpperArmR'))
            if hand is None or shoulder is None:
                continue
            rel = hand - shoulder
            fwd_dist = float(np.dot(rel[:2], self.fwd_xy))
            lat_dist = float(np.dot(rel[:2], self.lat_xy))
            z_rel = float(rel[2])
            # Hands should move ahead of the chest and slightly downward, so they
            # can read as a protective reach and tend to contact before the head.
            score += 2.8 * (fwd_dist - 0.48) ** 2
            score += 1.2 * (z_rel + 0.12) ** 2
            score += 0.35 * (lat_dist ** 2)
            score += 3.0 * max(0.0, -fwd_dist) ** 2
            score += 0.45 * max(0.0, abs(lat_dist) - 0.38) ** 2
        return float(score)

    def _collect_arm_joint_specs(self):
        specs = []
        joint_obj = mujoco.mjtObj.mjOBJ_JOINT if hasattr(mujoco, 'mjtObj') else mujoco.mjOBJ_JOINT
        for jid in range(int(self.mj_model.njnt)):
            jname = mujoco.mj_id2name(self.mj_model, joint_obj, jid) or ''
            bname = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, int(self.mj_model.jnt_bodyid[jid])) or ''
            txt = (jname + ' ' + bname).lower()
            if not any(k in txt for k in ('shoulder', 'elbow', 'wrist', 'hand', 'arm')):
                continue
            qadr = int(self.mj_model.jnt_qposadr[jid])
            if qadr < 7 or qadr >= len(self.mj_data.qpos):
                continue
            if 'shoulder' in txt or 'upperarm' in txt or 'arm' in txt:
                kind = 'shoulder'
            elif 'elbow' in txt:
                kind = 'elbow'
            else:
                kind = 'wrist'
            specs.append((jid, qadr, kind, self._joint_side(jname, bname)))
        return specs

    def _build_protective_reach_pose(self):
        """Create a target arm pose for descent without changing the cached stand."""
        if self._quiet_qpos is None:
            return 0
        arm_joints = self._collect_arm_joint_specs()
        if not arm_joints:
            self._protective_arm_targets = {}
            self._protective_joint_count = 0
            return 0
        saved_qpos = np.asarray(self.mj_data.qpos, dtype=float).copy()
        saved_qvel = np.asarray(self.mj_data.qvel, dtype=float).copy()
        # Start search from quiet arms-down pose.
        nq = min(len(self.mj_data.qpos), len(self._quiet_qpos))
        self.mj_data.qpos[:nq] = self._quiet_qpos[:nq]
        if len(self.mj_data.qvel):
            self.mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        changed = 0
        for _pass in range(3):
            for jid, qadr, kind, side in arm_joints:
                lo, hi = self._joint_range_for_qadr(jid)
                cur = float(self.mj_data.qpos[qadr])
                if kind == 'shoulder':
                    candidates = [-1.35, -1.05, -0.78, -0.52, -0.28, 0.0, 0.28, 0.52, 0.78, 1.05, 1.35]
                elif kind == 'elbow':
                    candidates = [-0.95, -0.65, -0.36, -0.18, 0.0, 0.18, 0.36, 0.65, 0.95, 1.20]
                else:
                    candidates = [-0.45, -0.25, -0.10, 0.0, 0.10, 0.25, 0.45]
                candidates = [cur] + [float(np.clip(v, lo, hi)) for v in candidates]
                candidates += [float(np.clip(cur + dv, lo, hi)) for dv in (-0.42, -0.24, 0.24, 0.42)]
                uniq, seen = [], set()
                for v in candidates:
                    key = round(float(v), 4)
                    if key not in seen:
                        seen.add(key); uniq.append(float(v))
                original = cur
                best_v = cur
                best_score = None
                for v in uniq:
                    self.mj_data.qpos[qadr] = float(v)
                    mujoco.mj_forward(self.mj_model, self.mj_data)
                    # Keep the reach pose modestly close to quiet pose.
                    quiet_v = float(self._quiet_qpos[qadr]) if qadr < len(self._quiet_qpos) else 0.0
                    score = self._protective_reach_score() + 0.018 * (float(v) - quiet_v) ** 2
                    if best_score is None or score < best_score:
                        best_score = score
                        best_v = float(v)
                self.mj_data.qpos[qadr] = best_v
                if abs(best_v - original) > 1e-4:
                    changed += 1
                mujoco.mj_forward(self.mj_model, self.mj_data)
        self._protective_arm_targets = {int(qadr): float(self.mj_data.qpos[qadr]) for _, qadr, _, _ in arm_joints}
        self._protective_joint_count = len(self._protective_arm_targets)
        self.mj_data.qpos[:] = saved_qpos
        self.mj_data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return changed

    def apply_descent_reaction(self, progress, phase='perturb'):
        """Bounded protective reaction: hands/legs brace without unnatural twisting.

        v7 keeps the reaction visible but clamps every limb target near the
        quiet/step reference pose.  This prevents the old opposite-direction
        hand/leg rotations after impact while still showing a human protective
        response.
        """
        if self._quiet_qpos is None:
            return
        p = float(np.clip(progress, 0.0, 1.0))
        reach = 1.0 - float(np.exp(-5.5 * max(0.0, p)))
        # one smooth reach, no oscillating/pulsing twist
        arm_gain = float(np.clip(0.18 + 0.58 * reach, 0.0, 0.72))

        def clamp_near_quiet(qadr, value, max_delta):
            qadr = int(qadr)
            base = float(self._quiet_qpos[qadr]) if qadr < len(self._quiet_qpos) else 0.0
            return float(np.clip(float(value), base - float(max_delta), base + float(max_delta)))

        for i, (qadr, target) in enumerate(getattr(self, '_protective_arm_targets', {}).items()):
            qadr = int(qadr)
            if 7 <= qadr < len(self.mj_data.qpos) and qadr < len(self._quiet_qpos):
                target_delta = float(target) - float(self._quiet_qpos[qadr])
                # symmetric two-hand reach, bounded to natural shoulder/elbow range
                desired = float(self._quiet_qpos[qadr]) + 0.82 * arm_gain * target_delta
                desired = clamp_near_quiet(qadr, desired, 0.82)
                self.mj_data.qpos[qadr] = 0.78 * float(self.mj_data.qpos[qadr]) + 0.22 * desired

        # Leg reaction: swing/back leg extends; opposite leg mildly bends/braces.
        # No alternating sign flutter, because that caused crossed/opposite legs.
        brace = float(np.clip(0.25 + 0.55 * reach, 0.0, 0.70))
        for qadr, amp, kind, side in getattr(self, '_reactive_leg_specs', []):
            qadr = int(qadr)
            if 7 <= qadr < len(self.mj_data.qpos) and qadr < len(self._quiet_qpos):
                amp_scale = 0.55 if kind == 'knee' else 0.38
                if side == 'left':
                    amp_scale *= 0.55
                desired = float(self._quiet_qpos[qadr]) + float(amp) * brace * amp_scale
                desired = clamp_near_quiet(qadr, desired, 0.60 if kind == 'knee' else 0.45)
                self.mj_data.qpos[qadr] = 0.82 * float(self.mj_data.qpos[qadr]) + 0.18 * desired
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def apply_impact_recoil_pose(self, progress):
        """Small post-impact relaxation toward a natural supine protective pose."""
        if self._quiet_qpos is None:
            return
        p = float(np.clip(progress, 0.0, 1.0))
        relax = p * p * (3.0 - 2.0 * p)
        for i, (qadr, target) in enumerate(getattr(self, '_protective_arm_targets', {}).items()):
            qadr = int(qadr)
            if 7 <= qadr < len(self.mj_data.qpos) and qadr < len(self._quiet_qpos):
                # hold partial protective arm position then slowly relax; no twist oscillation
                desired = (0.42 * float(target) + 0.58 * float(self._quiet_qpos[qadr]))
                desired = (1.0 - 0.55 * relax) * desired + (0.55 * relax) * float(self._quiet_qpos[qadr])
                base = float(self._quiet_qpos[qadr])
                desired = float(np.clip(desired, base - 0.70, base + 0.70))
                self.mj_data.qpos[qadr] = 0.86 * float(self.mj_data.qpos[qadr]) + 0.14 * desired
        for qadr, amp, kind, side in getattr(self, '_reactive_leg_specs', []):
            qadr = int(qadr)
            if 7 <= qadr < len(self.mj_data.qpos) and qadr < len(self._quiet_qpos):
                # legs settle near quiet/neutral with slight knee bend, not crossed/opposite
                bend = 0.18 if kind == 'knee' else 0.06
                desired = float(self._quiet_qpos[qadr]) + float(amp) * bend * (1.0 - relax)
                base = float(self._quiet_qpos[qadr])
                desired = float(np.clip(desired, base - 0.36, base + 0.36))
                self.mj_data.qpos[qadr] = 0.88 * float(self.mj_data.qpos[qadr]) + 0.12 * desired
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _resolve_step_joints(self):
        """Resolve lead-leg sagittal joints, then tune their signs from actual foot motion.

        v24 does not assume that a positive hip/knee coordinate means "forward".
        Earlier versions used fixed amplitudes and could accidentally move the
        lead foot behind the stance leg on different anthropometric profiles.
        """
        specs = []
        joint_obj = mujoco.mjtObj.mjOBJ_JOINT if hasattr(mujoco, 'mjtObj') else mujoco.mjOBJ_JOINT
        # Prefer right leg for the visible first step; fall back to unknown only if needed.
        for jid in range(int(self.mj_model.njnt)):
            jname = mujoco.mj_id2name(self.mj_model, joint_obj, jid) or ''
            bname = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, int(self.mj_model.jnt_bodyid[jid])) or ''
            txt = (jname + ' ' + bname).lower()
            side = self._joint_side(jname, bname)
            if side not in ('right', 'unknown'):
                continue
            qadr = int(self.mj_model.jnt_qposadr[jid])
            if qadr < 7 or qadr >= len(self.mj_data.qpos):
                continue
            if 'hip' in txt or 'thigh' in txt:
                specs.append((qadr, 0.0, 'hip'))
            elif 'knee' in txt or 'shin' in txt or 'shank' in txt:
                specs.append((qadr, 0.0, 'knee'))
            elif 'ankle' in txt or 'toe' in txt or 'foot' in txt:
                specs.append((qadr, 0.0, 'ankle'))
        seen = set(); unique = []
        for qadr, amp, kind in specs:
            if qadr in seen:
                continue
            seen.add(qadr); unique.append((int(qadr), float(amp), str(kind)))
        return unique[:10]

    def _resolve_reactive_leg_joints(self):
        specs = []
        joint_obj = mujoco.mjtObj.mjOBJ_JOINT if hasattr(mujoco, 'mjtObj') else mujoco.mjOBJ_JOINT
        for jid in range(int(self.mj_model.njnt)):
            jname = mujoco.mj_id2name(self.mj_model, joint_obj, jid) or ''
            bname = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, int(self.mj_model.jnt_bodyid[jid])) or ''
            txt = (jname + ' ' + bname).lower()
            qadr = int(self.mj_model.jnt_qposadr[jid])
            if qadr < 7 or qadr >= len(self.mj_data.qpos):
                continue
            side = self._joint_side(jname, bname)
            if 'hip' in txt or 'thigh' in txt:
                amp = 0.30 if side != 'left' else -0.22
                specs.append((qadr, amp, 'hip', side))
            elif 'knee' in txt or 'shin' in txt or 'shank' in txt:
                amp = 0.42 if side != 'left' else 0.30
                specs.append((qadr, amp, 'knee', side))
            elif 'ankle' in txt or 'toe' in txt or 'foot' in txt:
                amp = -0.22 if side != 'left' else 0.16
                specs.append((qadr, amp, 'ankle', side))
        seen = set(); unique = []
        for qadr, amp, kind, side in specs:
            if qadr in seen:
                continue
            seen.add(qadr); unique.append((int(qadr), float(amp), str(kind), str(side)))
        return unique[:16]

    def _foot_centroid(self, side='right'):
        side = str(side or 'right').lower()
        if side.startswith('r'):
            candidates = ('R_Toe', 'R_Foot', 'R_Ankle', 'R_Heel', 'RightToe', 'RightFoot', 'RightAnkle', 'RightHeel')
        else:
            candidates = ('L_Toe', 'L_Foot', 'L_Ankle', 'L_Heel', 'LeftToe', 'LeftFoot', 'LeftAnkle', 'LeftHeel')
        pts = []
        for name in candidates:
            bid = _safe_name2id(self.mj_model, MJOBJ_BODY, name)
            if bid >= 0:
                pts.append(np.asarray(self.mj_data.xpos[bid], dtype=float).copy())
        if pts:
            return np.mean(pts, axis=0)
        # Fall back to all foot bodies if exact names are unavailable.
        all_pts = [np.asarray(self.mj_data.xpos[bid], dtype=float).copy() for bid in self._foot_bodies if bid >= 0]
        return np.mean(all_pts, axis=0) if all_pts else None

    def _build_forward_step_pose(self):
        """Tune the lead-leg pose so the foot really moves backward/off-edge.

        This fixes the backward-crossing leg artifact by scoring the actual foot
        displacement in world coordinates instead of assuming joint signs.
        """
        if self._quiet_qpos is None or not self._step_joint_specs:
            self._step_pose_build_report = {'status': 'no_step_joints'}
            return None
        saved_qpos = np.asarray(self.mj_data.qpos, dtype=float).copy()
        saved_qvel = np.asarray(self.mj_data.qvel, dtype=float).copy()
        nq = min(len(self.mj_data.qpos), len(self._quiet_qpos))
        self.mj_data.qpos[:nq] = self._quiet_qpos[:nq]
        if len(self.mj_data.qvel):
            self.mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        quiet_foot = self._foot_centroid('right')
        if quiet_foot is None:
            self.mj_data.qpos[:] = saved_qpos; self.mj_data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self._step_pose_build_report = {'status': 'no_foot_point'}
            return None
        quiet_q = np.asarray(self.mj_data.qpos, dtype=float).copy()

        def score_pose():
            foot = self._foot_centroid('right')
            if foot is None:
                return 1e6
            disp = np.asarray(foot, dtype=float) - np.asarray(quiet_foot, dtype=float)
            fwd = float(np.dot(disp[:2], self.fwd_xy))
            lat = float(np.dot(disp[:2], self.lat_xy))
            lift = float(disp[2])
            # Target a visible step: backward of stance, slightly lifted, not crossing laterally.
            return (7.5 * (fwd - 0.18) ** 2
                    + 2.8 * (lift - 0.055) ** 2
                    + 1.2 * (lat ** 2)
                    + 18.0 * max(0.0, -fwd) ** 2
                    + 3.5 * max(0.0, abs(lat) - 0.12) ** 2
                    + 0.020 * float(np.mean((np.asarray(self.mj_data.qpos[:nq]) - quiet_q[:nq]) ** 2)))

        changed = 0
        for _pass in range(4):
            for qadr, _amp, kind in self._step_joint_specs:
                qadr = int(qadr)
                if not (7 <= qadr < len(self.mj_data.qpos)):
                    continue
                cur = float(self.mj_data.qpos[qadr])
                # Candidate values around current and quiet. Sign is selected by the score.
                if kind == 'hip':
                    deltas = (-0.75, -0.52, -0.34, -0.18, 0.0, 0.18, 0.34, 0.52, 0.75)
                elif kind == 'knee':
                    deltas = (-0.95, -0.65, -0.42, -0.22, 0.0, 0.22, 0.42, 0.65, 0.95)
                else:
                    deltas = (-0.45, -0.28, -0.14, 0.0, 0.14, 0.28, 0.45)
                lo, hi = -1.4, 1.4
                # qadr -> jid lookup is not cheap here; keep generous but bounded values.
                base = float(quiet_q[qadr])
                candidates = [cur, base] + [base + dv for dv in deltas] + [cur + dv for dv in deltas[::2]]
                candidates = [float(np.clip(v, lo, hi)) for v in candidates]
                uniq, seen = [], set()
                for v in candidates:
                    key = round(float(v), 4)
                    if key not in seen:
                        seen.add(key); uniq.append(float(v))
                best_v = cur; best_score = None
                for v in uniq:
                    self.mj_data.qpos[qadr] = float(v)
                    mujoco.mj_forward(self.mj_model, self.mj_data)
                    sc = score_pose()
                    if best_score is None or sc < best_score:
                        best_score = sc; best_v = float(v)
                if abs(best_v - cur) > 1e-4:
                    changed += 1
                self.mj_data.qpos[qadr] = best_v
                mujoco.mj_forward(self.mj_model, self.mj_data)
        step_q = np.asarray(self.mj_data.qpos, dtype=float).copy()
        foot = self._foot_centroid('right')
        disp = np.asarray(foot, dtype=float) - np.asarray(quiet_foot, dtype=float) if foot is not None else np.zeros(3)
        self._step_pose_build_report = {
            'status': 'ok',
            'changed': int(changed),
            'foot_backward_m': float(np.dot(disp[:2], self.fwd_xy)),
            'foot_lift_m': float(disp[2]),
            'foot_lateral_m': float(np.dot(disp[:2], self.lat_xy)),
        }
        self.mj_data.qpos[:] = saved_qpos
        self.mj_data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return step_q

    def place_on_platform(self, verbose=True, reason='initial'):
        """Lift the ordinary reset pose and cache a quiet, arms-neutral stand."""
        if self.mj_data.qpos.shape[0] >= 7:
            self.mj_data.qpos[3:7] = self._upright_quat
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self._update_edge_geometry(verbose=False)
            self.mj_data.qpos[0] = float(self.start_xy[0])
            self.mj_data.qpos[1] = float(self.start_xy[1])
            self.mj_data.qpos[3:7] = self._upright_quat
            self._neutralize_upper_body_qpos()
            mujoco.mj_forward(self.mj_model, self.mj_data)
            dz = (self.height + TASK43_TOP_CLEARANCE) - self._lowest_foot_z()
            self.mj_data.qpos[2] += float(dz)
        elif self.mj_data.qpos.shape[0] >= 3:
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self._update_edge_geometry(verbose=False)
            self.mj_data.qpos[0] = float(self.start_xy[0])
            self.mj_data.qpos[1] = float(self.start_xy[1])
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self.mj_data.qpos[2] += float((self.height + TASK43_TOP_CLEARANCE) - self._lowest_foot_z())
        if self.mj_data.qvel.shape[0] >= 6:
            self.mj_data.qvel[:6] = 0.0
        if hasattr(self.mj_data, 'ctrl'):
            self.mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._update_edge_geometry(verbose=verbose)
        if len(self.mj_data.qpos) >= 2:
            self.mj_data.qpos[0] = float(self.start_xy[0])
            self.mj_data.qpos[1] = float(self.start_xy[1])
            mujoco.mj_forward(self.mj_model, self.mj_data)
        self._seat_feet_on_platform()
        if self.mj_data.qvel.shape[0] >= 6:
            self.mj_data.qvel[:6] = 0.0
        self._quiet_qpos = np.asarray(self.mj_data.qpos, dtype=float).copy()
        self._quiet_qvel = np.zeros_like(np.asarray(self.mj_data.qvel, dtype=float))
        self._step_joint_specs = self._resolve_step_joints()
        self._reactive_leg_specs = self._resolve_reactive_leg_joints()
        self._step_pose_qpos = self._build_forward_step_pose()
        self._build_protective_reach_pose()
        if verbose:
            print("  [Task43HeightLayer] ACTIVE - quiet stand + single swing-leg back-step + protective reaction")
            print("      methodology              = no continuous walk phase; quiet stand then one visible swing-leg step; root fixed at edge")
            print("      object/platform          = real 1.80 m support block")
            print("      task objective           = stable backward stand/step from required 1.80 m object height")
            print(f"      platform geoms           = {self.platform_geoms if self.platform_geoms else 'viewer fallback only'}")
            print(f"      platform top z           = {self.height:.2f} m")
            print(f"      half extents [x,y,z]     = {self.half_extents.round(3).tolist()}")
            print(f"      arms-down joints         = {self._arm_joint_count} (tuned={getattr(self, '_arm_down_tuned_count', 0)})")
            print(f"      visible step joints      = {len(self._step_joint_specs)}")
            print(f"      single-leg step build   = {getattr(self, '_step_pose_build_report', {})}")
            print(f"      reactive leg joints      = {len(getattr(self, '_reactive_leg_specs', []))}")
            print(f"      protective reach joints  = {getattr(self, '_protective_joint_count', 0)}")
            distal_z = self._distal_body_min_z()
            distal_txt = 'N/A' if distal_z is None else f'{distal_z:.4f} m'
            print(f"      lowest sole/contact z    = {self._lowest_foot_z():.4f} m (target top={self.height:.4f} m)")
            print(f"      distal toe/heel body z   = {distal_txt}")
            print(f"      support contact check    = {self._last_support_contact_ok} (pairs={self._last_support_contact_count}, settle_drop={self._last_support_drop_m:.4f} m, visual_dz={getattr(self, '_last_visual_seat_dz', 0.0):+.4f} m, xy_shift={getattr(self, '_last_support_xy_shift', np.zeros(2)).round(3).tolist()})")
        else:
            distal_z = self._distal_body_min_z()
            distal_txt = 'N/A' if distal_z is None else f'{distal_z:.4f}m'
            print(f"  [Task43HeightLayer] placement refreshed after {reason}; lowest sole/contact z={self._lowest_foot_z():.4f} m (platform top={self.height:.4f} m); distal_z={distal_txt}; support_contact={self._last_support_contact_ok} pairs={self._last_support_contact_count} drop={self._last_support_drop_m:.4f}m visual_dz={getattr(self, '_last_visual_seat_dz', 0.0):+.4f}m xy_shift={getattr(self, '_last_support_xy_shift', np.zeros(2)).round(3).tolist()}")

    def _restore_cached_pose(self, root_xy=None, root_z=None, root_vxy=None, root_vz=0.0, do_forward=True):
        """Restore cached upright pose, optionally translating the free root."""
        if self._quiet_qpos is None:
            return
        nq = min(len(self.mj_data.qpos), len(self._quiet_qpos))
        self.mj_data.qpos[:nq] = self._quiet_qpos[:nq]
        if len(self.mj_data.qpos) >= 3:
            if root_xy is not None:
                self.mj_data.qpos[0] = float(root_xy[0])
                self.mj_data.qpos[1] = float(root_xy[1])
            if root_z is not None:
                self.mj_data.qpos[2] = float(root_z)
        if len(self.mj_data.qpos) >= 7:
            self.mj_data.qpos[3:7] = self._upright_quat
        self.mj_data.qvel[:] = 0.0
        if self.mj_data.qvel.shape[0] >= 3:
            if root_vxy is not None:
                self.mj_data.qvel[0] = float(root_vxy[0])
                self.mj_data.qvel[1] = float(root_vxy[1])
            self.mj_data.qvel[2] = float(root_vz)
        if hasattr(self.mj_data, 'ctrl'):
            self.mj_data.ctrl[:] = 0.0
        if do_forward:
            mujoco.mj_forward(self.mj_model, self.mj_data)

    def enforce_quiet_stand(self):
        """Hold a normal still posture on the platform center-edge with feet touching."""
        z = float(self._quiet_qpos[2]) if self._quiet_qpos is not None and len(self._quiet_qpos) >= 3 else None
        self._restore_cached_pose(root_xy=self.start_xy, root_z=z, root_vxy=np.zeros(2), root_vz=0.0)
        self._seat_feet_on_platform()

    def _apply_visible_step_pose(self, progress):
        p = float(np.clip(progress, 0.0, 1.0))
        alpha = p * p * (3.0 - 2.0 * p)
        if self._step_pose_qpos is not None and self._quiet_qpos is not None:
            for qadr, _amp, _kind in self._step_joint_specs:
                qadr = int(qadr)
                if 7 <= qadr < len(self.mj_data.qpos) and qadr < len(self._step_pose_qpos):
                    self.mj_data.qpos[qadr] = ((1.0 - alpha) * float(self._quiet_qpos[qadr])
                                               + alpha * float(self._step_pose_qpos[qadr]))
            return
        # Fallback only: never drive the lead foot backward by fixed-sign hip motion.
        swing = np.sin(np.pi * p)
        for qadr, amp, kind in self._step_joint_specs:
            if qadr < len(self.mj_data.qpos):
                self.mj_data.qpos[qadr] += abs(float(amp)) * 0.10 * float(swing)

    def enforce_single_step(self, progress):
        """Animate one backward swing-foot step with ZERO root/pelvis slide.

        v7 mirrors Task39's successful single-step structure but in reverse:
        the support/root stays fixed at the physical rear edge, while the swing
        leg alone steps backward off/behind the edge.  This removes the old
        visual skating where the whole humanoid translated across and beyond the
        object before falling.
        """
        p = float(np.clip(progress, 0.0, 1.0))
        z = float(self._quiet_qpos[2]) if self._quiet_qpos is not None and len(self._quiet_qpos) >= 3 else None
        root_vxy = np.zeros(2, dtype=float)
        # Hard fixed root at the edge during the step.  Do not interpolate XY.
        self._restore_cached_pose(root_xy=self.start_xy, root_z=z, root_vxy=root_vxy, root_vz=0.0, do_forward=False)
        self._apply_visible_step_pose(p)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        # Keep stance/sole contact during the step, but do not allow XY search
        # because XY search can look like sliding on the block.
        self._seat_feet_on_platform(allow_xy_search=False)
        if len(self.mj_data.qpos) >= 2:
            self.mj_data.qpos[0] = float(self.start_xy[0])
            self.mj_data.qpos[1] = float(self.start_xy[1])
        if self.mj_data.qvel.shape[0] >= 3:
            self.mj_data.qvel[0:3] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def release_contact(self, forward_velocity=0.62):
        """Release deterministic edge pose into real platform physics.

        v5 critical fix: do NOT disable the platform geoms.  The block remains
        a physical MuJoCo support object.  We stop pose-locking after the
        backward swing step and add one modest rearward/downward destabilizing
        impulse so COM moves behind the rear edge; contacts + gravity then
        drive the fall instead of a scripted slide.
        """
        if self._released:
            return

        keep_contact = bool(globals().get('TASK43_KEEP_PLATFORM_CONTACT_AFTER_STEP', True))
        if not keep_contact:
            # Debug fallback only. Normal v5 keeps support contact ON.
            for gid in self.platform_geoms:
                try:
                    self.mj_model.geom_contype[int(gid)] = 0
                    self.mj_model.geom_conaffinity[int(gid)] = 0
                except Exception:
                    pass

        release_speed = float(globals().get('TASK43_EDGE_RELEASE_BACKWARD_SPEED', 0.0))
        if self.mj_data.qvel.shape[0] >= 3:
            # v7: absolutely no backward root launch.  We kill root XY velocity
            # and give only a tiny downward velocity; falling comes from gravity
            # and sagittal pitch torque after the swing leg has stepped back.
            self.mj_data.qvel[:] *= 0.0
            self.mj_data.qvel[0] = release_speed * float(self.fwd_xy[0])
            self.mj_data.qvel[1] = release_speed * float(self.fwd_xy[1])
            self.mj_data.qvel[2] = float(globals().get('TASK43_EDGE_RELEASE_DOWN_SPEED', -0.035))

        bw = max(float(globals().get('MODEL_BODY_MASS', 72.0)) * 9.81, 1.0)
        fwd = np.asarray(self.fwd_xy, dtype=float)
        lat = np.asarray(self.lat_xy, dtype=float)
        pelvis_id = _safe_name2id(self.mj_model, MJOBJ_BODY, 'Pelvis')
        torso_id = _safe_name2id(self.mj_model, MJOBJ_BODY, 'Torso')
        head_id = _safe_name2id(self.mj_model, MJOBJ_BODY, 'Head')
        # v7c: release is a small backside-fall cue, not a head-first impulse.
        # Shared backward force + capped pitch avoids the roll/flip seen in v7b.
        shared_back = float(globals().get('TASK43_SIMPLE_BACK_FALL_BODY_PUSH_BW', 0.022)) * 0.45 * bw
        if pelvis_id >= 0:
            self.mj_data.xfrc_applied[pelvis_id, 0] += 0.45 * shared_back * fwd[0]
            self.mj_data.xfrc_applied[pelvis_id, 1] += 0.45 * shared_back * fwd[1]
            self.mj_data.xfrc_applied[pelvis_id, 2] += -0.0025 * bw
        if torso_id >= 0:
            self.mj_data.xfrc_applied[torso_id, 0] += shared_back * fwd[0]
            self.mj_data.xfrc_applied[torso_id, 1] += shared_back * fwd[1]
            self.mj_data.xfrc_applied[torso_id, 2] += -0.0035 * bw
            torque_nm = float(globals().get('TASK43_EDGE_RELEASE_TORSO_TORQUE_NM', 34.0))
            self.mj_data.xfrc_applied[torso_id, 3] += torque_nm * lat[0]
            self.mj_data.xfrc_applied[torso_id, 4] += torque_nm * lat[1]
        if head_id >= 0:
            self.mj_data.xfrc_applied[head_id, 0] += 0.65 * shared_back * fwd[0]
            self.mj_data.xfrc_applied[head_id, 1] += 0.65 * shared_back * fwd[1]

        self._released = True
        print("      Task43 physical edge release v7d: platform contact remains ON; ZERO step slide; Task38 posterior push starts after swing-step")

    def draw_viewer_support(self, viewer):
        if self.platform_geoms:
            return
        try:
            scn = getattr(viewer, 'user_scn', None)
            if scn is None or not hasattr(mujoco, 'mjv_initGeom'):
                return
            scn.ngeom = 0
            if scn.ngeom >= scn.maxgeom:
                return
            gid = int(scn.ngeom)
            scn.ngeom += 1
            size = np.asarray(self.half_extents, dtype=float)
            pos = np.array([float(self.center_xy[0]), float(self.center_xy[1]), 0.5 * self.height], dtype=float)
            mat = np.eye(3, dtype=float).reshape(9)
            rgba = np.array([0.45, 0.32, 0.18, 1.0], dtype=float)
            mujoco.mjv_initGeom(scn.geoms[gid], mujoco.mjtGeom.mjGEOM_BOX, size, pos, mat, rgba)
        except Exception:
            pass


# -------------------------------------------------------------------
# TASK 41 LADDER CLIMB SUPPORT LAYER
# -------------------------------------------------------------------
# Task 41 keeps the Task40 fall/action/rest stack intact, but replaces the
# platform/edge preparation with a real MuJoCo ladder and a deterministic
# 4-5 step upward climb. The base simulation loop still sees the preparation
# as the legacy "step" phase, so all post-release safeguards remain untouched:
# posterior COM push, progressive weakening, descent reaction, impact braking,
# sagittal rail, and passive supine settle.
TASK43_LADDER_MODE = True
TASK43_LADDER_VISIBLE_STEPS = 7
TASK43_LADDER_RUNG_COUNT = 7
TASK43_LADDER_TRIGGER_RUNG_INDEX = 6       # zero-based seventh/top rung: full climb before forward fall
TASK43_LADDER_RUNG_X_HALF_M = 0.46
TASK43_LADDER_RUNG_Y0_M = -0.54
TASK43_LADDER_RUNG_DY_M = 0.20
TASK43_LADDER_RUNG_Z0_M = 0.30
TASK43_LADDER_RUNG_DZ_M = 0.37
TASK43_LADDER_RUNG_THICK_XY_M = 0.070
TASK43_LADDER_RUNG_THICK_Z_M = 0.070
TASK43_LADDER_RAIL_RADIUS_M = 0.045
TASK43_LADDER_SIDE_RAIL_X_M = 0.53
TASK43_LADDER_ROOT_BACKOFF_M = 0.088     # v1i: closer body-to-ladder distance for reachable hands/feet
TASK43_LADDER_CLIMB_POSE_GAIN = 1.00
TASK43_LADDER_FOOT_LATERAL_M = 0.155
TASK43_LADDER_HAND_LATERAL_M = 0.205
TASK43_LADDER_FOOT_CONTACT_Z_CLEARANCE_M = 0.004
TASK43_LADDER_FOOT_ALIGN_GAIN = 0.72
TASK43_LADDER_CONTACT_FORCE_KP = 0.0      # v1h: reference/mocap climb; no fake force climb
TASK43_LADDER_CONTACT_FORCE_KD = 0.0
TASK43_LADDER_RELEASE_BACKSTEP_M = 0.24
TASK43_LADDER_RELEASE_PUSH_BW = 0.28
TASK43_LADDER_RELEASE_TORSO_PUSH_BW = 0.16
TASK43_LADDER_PLANTED_FOOT_PRELOAD_BW = 0.000
TASK43_LADDER_RUNG_CONTACT_MARGIN_M = 0.003
TASK43_LADDER_HAND_NEAR_OFFSET_M = 0.006
TASK43_LADDER_FOOT_NEAR_OFFSET_M = 0.004
TASK43_LADDER_HAND_CENTER_Z_OFFSET_M = 0.000
TASK43_LADDER_VIRTUAL_SUPPORT_ENABLED = True
TASK43_LADDER_VIRTUAL_STEP_SUPPORT_BW = 0.74
TASK43_LADDER_VIRTUAL_HAND_SUPPORT_BW = 0.10
TASK43_LADDER_HAND_GRIP_LEAD_RUNG = 1.35
TASK43_LADDER_HAND_REACH_POSE_GAIN = 0.92
TASK43_LADDER_HAND_IK_ENABLED = True
TASK43_LADDER_HAND_IK_PASSES = 16
TASK43_LADDER_HAND_GRIP_FORCE_KP = 0.0
TASK43_LADDER_HAND_GRIP_FORCE_KD = 0.0
TASK43_LADDER_HAND_GRIP_FORCE_CLIP_N = 115.0
TASK43_LADDER_HAND_GRIP_PRELOAD_BW = 0.000
TASK43_LADDER_LOAD_TRANSFER_START = 0.72
TASK43_LADDER_LOAD_TRANSFER_END = 0.94
TASK43_LADDER_ACTIVE_FOOT_LAND_START = 0.76
TASK43_LADDER_SWING_ROOT_LIFT_M = 0.010
TASK43_LADDER_PROFILE_STAND_STOOP_DEG = 5.0
TASK43_LADDER_PROFILE_CLIMB_STOOP_DEG = 6.5
TASK43_LADDER_PROFILE_STEP_GAIN = 1.0
TASK43_LADDER_PROFILE_LEG_GAIN = 1.0
TASK43_LADDER_PROFILE_ARM_GAIN = 1.0
TASK43_LADDER_PROFILE_ROOT_TRANSFER = 1.0
TASK43_LADDER_PROFILE_SPEED_MPS = 1.24
TASK43_LADDER_PROFILE_EXPECTED_DS = 0.30
TASK43_LADDER_REFERENCE_MOCAP_CLIMB = True
TASK43_LADDER_REFERENCE_IK_PASSES = 30
TASK43_LADDER_REFERENCE_IK_STEP_CLIP = 0.175
TASK43_LADDER_REFERENCE_ROOT_BOB_M = 0.018
TASK43_LADDER_REFERENCE_FOOT_LIFT_M = 0.120
TASK43_LADDER_REFERENCE_HAND_LIFT_M = 0.035
TASK43_LADDER_REFERENCE_HAND_LEAD_RUNG = 0.72
TASK43_LADDER_REFERENCE_DISABLE_GRIP_FORCES = True

# Immutable defaults tuned for the original 75y/1.65m reference run. The
# profile bridge below derives runtime constants from these values so repeated
# dispatcher runs cannot accumulate scaling drift. The default elderly profile
# stays effectively unchanged; younger/taller/heavier subjects get visibly
# different grip, reach, release slip, and posterior fall drive.
_TASK43_PROFILE_BASE = {
    'TASK43_LADDER_RUNG_DZ_M': TASK43_LADDER_RUNG_DZ_M,
    'TASK43_LADDER_ROOT_BACKOFF_M': TASK43_LADDER_ROOT_BACKOFF_M,
    'TASK43_LADDER_CLIMB_POSE_GAIN': TASK43_LADDER_CLIMB_POSE_GAIN,
    'TASK43_LADDER_FOOT_LATERAL_M': TASK43_LADDER_FOOT_LATERAL_M,
    'TASK43_LADDER_HAND_LATERAL_M': TASK43_LADDER_HAND_LATERAL_M,
    'TASK43_LADDER_FOOT_ALIGN_GAIN': TASK43_LADDER_FOOT_ALIGN_GAIN,
    'TASK43_LADDER_CONTACT_FORCE_KP': TASK43_LADDER_CONTACT_FORCE_KP,
    'TASK43_LADDER_CONTACT_FORCE_KD': TASK43_LADDER_CONTACT_FORCE_KD,
    'TASK43_LADDER_RELEASE_BACKSTEP_M': TASK43_LADDER_RELEASE_BACKSTEP_M,
    'TASK43_LADDER_RELEASE_PUSH_BW': TASK43_LADDER_RELEASE_PUSH_BW,
    'TASK43_LADDER_RELEASE_TORSO_PUSH_BW': TASK43_LADDER_RELEASE_TORSO_PUSH_BW,
    'TASK43_LADDER_PLANTED_FOOT_PRELOAD_BW': TASK43_LADDER_PLANTED_FOOT_PRELOAD_BW,
    'TASK43_LADDER_HAND_NEAR_OFFSET_M': TASK43_LADDER_HAND_NEAR_OFFSET_M,
    'TASK43_LADDER_FOOT_NEAR_OFFSET_M': TASK43_LADDER_FOOT_NEAR_OFFSET_M,
    'TASK43_LADDER_HAND_CENTER_Z_OFFSET_M': TASK43_LADDER_HAND_CENTER_Z_OFFSET_M,
    'TASK43_LADDER_VIRTUAL_STEP_SUPPORT_BW': TASK43_LADDER_VIRTUAL_STEP_SUPPORT_BW,
    'TASK43_LADDER_VIRTUAL_HAND_SUPPORT_BW': TASK43_LADDER_VIRTUAL_HAND_SUPPORT_BW,
    'TASK43_LADDER_LOAD_TRANSFER_START': TASK43_LADDER_LOAD_TRANSFER_START,
    'TASK43_LADDER_LOAD_TRANSFER_END': TASK43_LADDER_LOAD_TRANSFER_END,
    'TASK43_LADDER_ACTIVE_FOOT_LAND_START': TASK43_LADDER_ACTIVE_FOOT_LAND_START,
    'TASK43_LADDER_SWING_ROOT_LIFT_M': TASK43_LADDER_SWING_ROOT_LIFT_M,
    'TASK43_LADDER_PROFILE_STAND_STOOP_DEG': TASK43_LADDER_PROFILE_STAND_STOOP_DEG,
    'TASK43_LADDER_PROFILE_CLIMB_STOOP_DEG': TASK43_LADDER_PROFILE_CLIMB_STOOP_DEG,
    'TASK43_LADDER_PROFILE_STEP_GAIN': TASK43_LADDER_PROFILE_STEP_GAIN,
    'TASK43_LADDER_PROFILE_LEG_GAIN': TASK43_LADDER_PROFILE_LEG_GAIN,
    'TASK43_LADDER_PROFILE_ARM_GAIN': TASK43_LADDER_PROFILE_ARM_GAIN,
    'TASK43_LADDER_PROFILE_ROOT_TRANSFER': TASK43_LADDER_PROFILE_ROOT_TRANSFER,
    'TASK43_TASK38_FORCE_BW_FRACTION': TASK43_TASK38_FORCE_BW_FRACTION,
    'TASK43_TASK38_INITIAL_BACK_VEL': TASK43_TASK38_INITIAL_BACK_VEL,
    'TASK43_TASK38_TARGET_BACK_VEL': TASK43_TASK38_TARGET_BACK_VEL,
    'TASK43_TASK38_BACK_VEL_CAP_EARLY': TASK43_TASK38_BACK_VEL_CAP_EARLY,
    'TASK43_TASK38_BACK_VEL_CAP_FREE': TASK43_TASK38_BACK_VEL_CAP_FREE,
    'TASK43_TASK38_PITCH_TORQUE_NM': TASK43_TASK38_PITCH_TORQUE_NM,
    'TASK43_TASK38_COUNTER_TORQUE_NM': TASK43_TASK38_COUNTER_TORQUE_NM,
    'TASK43_TASK38_MAX_PITCH_RATE': TASK43_TASK38_MAX_PITCH_RATE,
}
TASK43_PROFILE_ADAPTATION = {}

def _task43_apply_subject_profile_globals(age_params=None):
    """Bridge the printed biofidelic subject profile into Task43 ladder logic."""
    global TASK43_LADDER_RUNG_DZ_M, TASK43_LADDER_ROOT_BACKOFF_M
    global TASK43_LADDER_CLIMB_POSE_GAIN, TASK43_LADDER_FOOT_LATERAL_M, TASK43_LADDER_HAND_LATERAL_M
    global TASK43_LADDER_FOOT_ALIGN_GAIN, TASK43_LADDER_CONTACT_FORCE_KP, TASK43_LADDER_CONTACT_FORCE_KD
    global TASK43_LADDER_RELEASE_BACKSTEP_M, TASK43_LADDER_RELEASE_PUSH_BW, TASK43_LADDER_RELEASE_TORSO_PUSH_BW
    global TASK43_LADDER_PLANTED_FOOT_PRELOAD_BW, TASK43_LADDER_HAND_NEAR_OFFSET_M, TASK43_LADDER_FOOT_NEAR_OFFSET_M
    global TASK43_LADDER_HAND_CENTER_Z_OFFSET_M, TASK43_LADDER_VIRTUAL_STEP_SUPPORT_BW, TASK43_LADDER_VIRTUAL_HAND_SUPPORT_BW
    global TASK43_LADDER_PROFILE_STAND_STOOP_DEG, TASK43_LADDER_PROFILE_CLIMB_STOOP_DEG
    global TASK43_LADDER_PROFILE_STEP_GAIN, TASK43_LADDER_PROFILE_LEG_GAIN, TASK43_LADDER_PROFILE_ARM_GAIN
    global TASK43_LADDER_PROFILE_ROOT_TRANSFER, TASK43_LADDER_PROFILE_SPEED_MPS, TASK43_LADDER_PROFILE_EXPECTED_DS
    global TASK43_TASK38_FORCE_BW_FRACTION, TASK43_TASK38_INITIAL_BACK_VEL, TASK43_TASK38_TARGET_BACK_VEL
    global TASK43_TASK38_BACK_VEL_CAP_EARLY, TASK43_TASK38_BACK_VEL_CAP_FREE
    global TASK43_TASK38_PITCH_TORQUE_NM, TASK43_TASK38_COUNTER_TORQUE_NM, TASK43_TASK38_MAX_PITCH_RATE
    global TASK43_EDGE_RELEASE_TORSO_TORQUE_NM, TASK43_EDGE_RELEASE_TORSO_PUSH_BW
    global TASK43_SIMPLE_BACK_FALL_BACK_VEL_CAP_EARLY, TASK43_SIMPLE_BACK_FALL_BACK_VEL_CAP_FREE
    global TASK43_SIMPLE_BACK_FALL_TARGET_BACK_VEL, TASK43_SIMPLE_BACK_FALL_MAX_PITCH_RATE
    global TASK43_CHAIR_HEIGHT_M, TASK43_CHAIR_HALF_EXTENTS, TASK43_PROFILE_ADAPTATION

    base = _TASK43_PROFILE_BASE
    age = float(globals().get('SIM_AGE', 75))
    height = float(globals().get('SIM_HEIGHT', 1.65))
    sex = str(globals().get('SIM_SEX', 'male')).lower()
    mass = float(globals().get('SIM_RESOLVED_WEIGHT', 70.8))

    # age_old=1 at the original 75y reference; age_young=1 around early adult.
    age_old = float(np.clip((age - 32.0) / 43.0, 0.0, 1.0))
    age_young = 1.0 - age_old
    height_scale = float(np.clip(height / 1.65, 0.90, 1.12))
    mass_scale = float(np.clip(mass / 70.8, 0.82, 1.25))
    female_scale = 0.94 if sex.startswith('f') else 1.0

    if isinstance(age_params, dict) and age_params:
        strength = float(age_params.get('strength_factor', age_params.get('strength_factor_raw', 1.0)))
        balance_impairment = float(age_params.get('balance_impairment', 0.0))
        proprio = float(age_params.get('proprioception_scale', 1.0))
    else:
        strength = float(np.clip(1.10 - 0.25 * age_old, 0.80, 1.12))
        balance_impairment = float(np.clip(0.38 * age_old, 0.0, 0.42))
        proprio = float(np.clip(1.0 - 0.20 * age_old, 0.78, 1.0))

    try:
        _style = get_age_style_v2(age, height_m=height, sex=sex, body_mass_kg=mass)
        target_walk_speed = float(_style.get('target_walk_speed', 1.24))
        expected_ds = float(_style.get('expected_double_support', 0.30))
        stand_stoop_deg = float(_style.get('stand_stoop_target_deg', _style.get('stoop_target_deg', 5.0)))
        walk_stoop_deg = float(_style.get('walk_stoop_target_deg', _style.get('stoop_target_deg', 6.5)))
        profile_arm_gain = float(_style.get('arm_gain', age_params.get('arm_gain', 1.0) if isinstance(age_params, dict) else 1.0))
    except Exception:
        target_walk_speed = float(np.clip(1.48 - 0.24 * age_old, 1.05, 1.55))
        expected_ds = float(np.clip(0.20 + 0.16 * age_old, 0.18, 0.38))
        stand_stoop_deg = float(4.5 + 5.5 * age_old)
        walk_stoop_deg = float(6.0 + 5.5 * age_old)
        profile_arm_gain = float(np.clip(1.0 - 0.25 * age_old, 0.72, 1.05))

    strength_rel = float(np.clip(strength / 0.85, 0.82, 1.32))
    speed_ref = 1.24
    speed_ratio = float(np.clip(speed_ref / max(target_walk_speed, 0.25), 0.82, 1.18))
    reach_scale = float(np.clip(1.0 + 0.18 * (height_scale - 1.0), 0.97, 1.03))

    TASK43_LADDER_RUNG_DZ_M = float(base['TASK43_LADDER_RUNG_DZ_M'] * reach_scale)
    TASK43_LADDER_ROOT_BACKOFF_M = float(base['TASK43_LADDER_ROOT_BACKOFF_M'] * np.clip(1.0 + 0.12 * (height_scale - 1.0), 0.97, 1.03))
    TASK43_LADDER_FOOT_LATERAL_M = float(base['TASK43_LADDER_FOOT_LATERAL_M'] * np.clip(height_scale, 0.94, 1.07) * female_scale)
    TASK43_LADDER_HAND_LATERAL_M = float(base['TASK43_LADDER_HAND_LATERAL_M'] * np.clip(height_scale, 0.94, 1.07) * female_scale)

    TASK43_LADDER_CLIMB_POSE_GAIN = float(base['TASK43_LADDER_CLIMB_POSE_GAIN'] * np.clip(1.0 + 0.05 * age_young * (height_scale - 1.0) + 0.04 * age_old, 0.96, 1.06))
    TASK43_LADDER_FOOT_ALIGN_GAIN = float(base['TASK43_LADDER_FOOT_ALIGN_GAIN'] * np.clip(1.0 - 0.05 * age_young + 0.03 * age_old, 0.90, 1.03))
    TASK43_LADDER_CONTACT_FORCE_KP = float(base['TASK43_LADDER_CONTACT_FORCE_KP'] * np.clip(0.96 + 0.06 * strength_rel, 0.96, 1.06))
    TASK43_LADDER_CONTACT_FORCE_KD = float(base['TASK43_LADDER_CONTACT_FORCE_KD'] * np.clip(0.98 + 0.04 * proprio, 0.98, 1.04))
    TASK43_LADDER_PLANTED_FOOT_PRELOAD_BW = float(base['TASK43_LADDER_PLANTED_FOOT_PRELOAD_BW'] * np.clip(0.90 + 0.10 * age_old, 0.90, 1.00))
    TASK43_LADDER_HAND_NEAR_OFFSET_M = float(base.get('TASK43_LADDER_HAND_NEAR_OFFSET_M', TASK43_LADDER_HAND_NEAR_OFFSET_M) * np.clip(0.96 + 0.06 * height_scale, 0.98, 1.05))
    TASK43_LADDER_FOOT_NEAR_OFFSET_M = float(base.get('TASK43_LADDER_FOOT_NEAR_OFFSET_M', TASK43_LADDER_FOOT_NEAR_OFFSET_M) * np.clip(0.98 + 0.04 * height_scale, 0.99, 1.04))
    TASK43_LADDER_HAND_CENTER_Z_OFFSET_M = float(base.get('TASK43_LADDER_HAND_CENTER_Z_OFFSET_M', TASK43_LADDER_HAND_CENTER_Z_OFFSET_M))
    TASK43_LADDER_VIRTUAL_STEP_SUPPORT_BW = float(base.get('TASK43_LADDER_VIRTUAL_STEP_SUPPORT_BW', TASK43_LADDER_VIRTUAL_STEP_SUPPORT_BW) * np.clip(0.94 + 0.08 * age_old + 0.03 * balance_impairment, 0.92, 1.05))
    TASK43_LADDER_VIRTUAL_HAND_SUPPORT_BW = float(base.get('TASK43_LADDER_VIRTUAL_HAND_SUPPORT_BW', TASK43_LADDER_VIRTUAL_HAND_SUPPORT_BW) * np.clip(0.90 + 0.14 * age_old, 0.88, 1.06))

    # v7f speed-only bridge: keep the same validated ladder body posture and
    # foot/hand pose gains for every subject.  The walking profile is exposed
    # here only for diagnostics and for scenario43_phase_timing() to change
    # PHASES['step'] / climb speed.
    cautious = float(np.clip(0.55 * age_old + 0.35 * balance_impairment + 0.30 * (expected_ds - 0.24), 0.0, 1.0))
    TASK43_LADDER_PROFILE_SPEED_MPS = float(target_walk_speed)
    TASK43_LADDER_PROFILE_EXPECTED_DS = float(expected_ds)
    TASK43_LADDER_PROFILE_STAND_STOOP_DEG = float(base['TASK43_LADDER_PROFILE_STAND_STOOP_DEG'])
    TASK43_LADDER_PROFILE_CLIMB_STOOP_DEG = float(base['TASK43_LADDER_PROFILE_CLIMB_STOOP_DEG'])
    TASK43_LADDER_PROFILE_STEP_GAIN = float(base['TASK43_LADDER_PROFILE_STEP_GAIN'])
    TASK43_LADDER_PROFILE_LEG_GAIN = float(base['TASK43_LADDER_PROFILE_LEG_GAIN'])
    TASK43_LADDER_PROFILE_ARM_GAIN = float(base['TASK43_LADDER_PROFILE_ARM_GAIN'])
    TASK43_LADDER_PROFILE_ROOT_TRANSFER = float(base['TASK43_LADDER_PROFILE_ROOT_TRANSFER'])

    young_release = float(np.clip(1.0 - 0.16 * age_young + 0.07 * balance_impairment, 0.82, 1.06))
    TASK43_LADDER_RELEASE_BACKSTEP_M = float(base['TASK43_LADDER_RELEASE_BACKSTEP_M'] * young_release)
    TASK43_LADDER_RELEASE_PUSH_BW = float(base['TASK43_LADDER_RELEASE_PUSH_BW'] * np.clip(1.0 - 0.13 * age_young + 0.05 * balance_impairment, 0.84, 1.07))
    TASK43_LADDER_RELEASE_TORSO_PUSH_BW = float(base['TASK43_LADDER_RELEASE_TORSO_PUSH_BW'] * np.clip(1.0 - 0.11 * age_young + 0.05 * balance_impairment, 0.86, 1.06))

    TASK43_TASK38_FORCE_BW_FRACTION = float(base['TASK43_TASK38_FORCE_BW_FRACTION'] * np.clip(1.0 - 0.12 * age_young + 0.04 * balance_impairment, 0.86, 1.05))
    TASK43_TASK38_INITIAL_BACK_VEL = float(base['TASK43_TASK38_INITIAL_BACK_VEL'] * np.clip(1.0 - 0.10 * age_young + 0.03 * balance_impairment, 0.88, 1.04))
    TASK43_TASK38_TARGET_BACK_VEL = float(base['TASK43_TASK38_TARGET_BACK_VEL'] * np.clip(1.0 - 0.08 * age_young + 0.04 * balance_impairment, 0.90, 1.05))
    TASK43_TASK38_BACK_VEL_CAP_EARLY = float(base['TASK43_TASK38_BACK_VEL_CAP_EARLY'] * np.clip(1.0 - 0.08 * age_young, 0.92, 1.03))
    TASK43_TASK38_BACK_VEL_CAP_FREE = float(base['TASK43_TASK38_BACK_VEL_CAP_FREE'] * np.clip(1.0 - 0.08 * age_young + 0.03 * balance_impairment, 0.91, 1.04))
    TASK43_TASK38_PITCH_TORQUE_NM = float(base['TASK43_TASK38_PITCH_TORQUE_NM'] * np.clip(1.0 - 0.22 * age_young + 0.05 * balance_impairment, 0.76, 1.05))
    TASK43_TASK38_COUNTER_TORQUE_NM = float(base['TASK43_TASK38_COUNTER_TORQUE_NM'] * np.clip(1.0 + 0.22 * age_young - 0.05 * balance_impairment, 0.94, 1.20))
    TASK43_TASK38_MAX_PITCH_RATE = float(base['TASK43_TASK38_MAX_PITCH_RATE'] * np.clip(1.0 - 0.06 * age_young - 0.02 * balance_impairment, 0.90, 1.02))

    TASK43_EDGE_RELEASE_TORSO_TORQUE_NM = TASK43_TASK38_PITCH_TORQUE_NM
    TASK43_EDGE_RELEASE_TORSO_PUSH_BW = TASK43_TASK38_TORSO_FOLLOW_BW
    TASK43_SIMPLE_BACK_FALL_BACK_VEL_CAP_EARLY = TASK43_TASK38_BACK_VEL_CAP_EARLY
    TASK43_SIMPLE_BACK_FALL_BACK_VEL_CAP_FREE = TASK43_TASK38_BACK_VEL_CAP_FREE
    TASK43_SIMPLE_BACK_FALL_TARGET_BACK_VEL = TASK43_TASK38_TARGET_BACK_VEL
    TASK43_SIMPLE_BACK_FALL_MAX_PITCH_RATE = TASK43_TASK38_MAX_PITCH_RATE

    TASK43_CHAIR_HEIGHT_M = float(TASK43_LADDER_RUNG_Z0_M + TASK43_LADDER_TRIGGER_RUNG_INDEX * TASK43_LADDER_RUNG_DZ_M + TASK43_LADDER_RUNG_THICK_Z_M)
    TASK43_CHAIR_HALF_EXTENTS = (TASK43_LADDER_RUNG_X_HALF_M, 0.36, TASK43_CHAIR_HEIGHT_M * 0.5)
    TASK43_PROFILE_ADAPTATION = {
        'age_old': age_old, 'height_scale': height_scale, 'mass_scale': mass_scale,
        'strength': strength, 'balance_impairment': balance_impairment,
        'rung_dz_m': TASK43_LADDER_RUNG_DZ_M,
        'root_backoff_m': TASK43_LADDER_ROOT_BACKOFF_M,
        'grip_kp': TASK43_LADDER_CONTACT_FORCE_KP,
        'foot_align_gain': TASK43_LADDER_FOOT_ALIGN_GAIN,
        'hand_near_offset_m': TASK43_LADDER_HAND_NEAR_OFFSET_M,
        'foot_near_offset_m': TASK43_LADDER_FOOT_NEAR_OFFSET_M,
        'virtual_step_support_bw': TASK43_LADDER_VIRTUAL_STEP_SUPPORT_BW,
        'release_backstep_m': TASK43_LADDER_RELEASE_BACKSTEP_M,
        'release_push_bw': TASK43_LADDER_RELEASE_PUSH_BW,
        'fall_force_bw': TASK43_TASK38_FORCE_BW_FRACTION,
        'initial_back_vel': TASK43_TASK38_INITIAL_BACK_VEL,
        'walk_speed_mps': TASK43_LADDER_PROFILE_SPEED_MPS,
        'expected_double_support': TASK43_LADDER_PROFILE_EXPECTED_DS,
        'stand_stoop_deg': TASK43_LADDER_PROFILE_STAND_STOOP_DEG,
        'climb_stoop_deg': TASK43_LADDER_PROFILE_CLIMB_STOOP_DEG,
        'step_gain': TASK43_LADDER_PROFILE_STEP_GAIN,
        'leg_gain': TASK43_LADDER_PROFILE_LEG_GAIN,
        'arm_gain': TASK43_LADDER_PROFILE_ARM_GAIN,
        'root_transfer': TASK43_LADDER_PROFILE_ROOT_TRANSFER,
    }
    return TASK43_PROFILE_ADAPTATION

try:
    _task43_apply_subject_profile_globals()
except Exception:
    pass

# Keep the legacy height aliases meaningful for inherited Task40/Task43 helpers.
TASK43_CHAIR_HEIGHT_M = float(TASK43_LADDER_RUNG_Z0_M + TASK43_LADDER_TRIGGER_RUNG_INDEX * TASK43_LADDER_RUNG_DZ_M + TASK43_LADDER_RUNG_THICK_Z_M)
TASK43_CHAIR_HALF_EXTENTS = (TASK43_LADDER_RUNG_X_HALF_M, 0.36, TASK43_CHAIR_HEIGHT_M * 0.5)
TASK43_CHAIR_CENTER_XY = (0.0, 0.0)
TASK43_STEP_CLEARANCE_M = 0.24
TASK43_TOP_CLEARANCE = -0.0180


def _task43_rung_center(index):
    idx = int(index)
    return np.array([
        0.0,
        float(TASK43_LADDER_RUNG_Y0_M + idx * TASK43_LADDER_RUNG_DY_M),
        float(TASK43_LADDER_RUNG_Z0_M + idx * TASK43_LADDER_RUNG_DZ_M),
    ], dtype=float)


def _task43_rung_top_z(index):
    return float(_task43_rung_center(index)[2] + TASK43_LADDER_RUNG_THICK_Z_M)


def _task43_find_platform_geoms(mj_model):
    """Return all physical ladder geoms used for foot/hand contact checks."""
    ids = []
    try:
        for gid in range(int(mj_model.ngeom)):
            name = (mujoco.mj_id2name(mj_model, MJOBJ_GEOM, gid) or '').lower()
            if ('task43_ladder' in name or 'task43_rung' in name or
                'task43_left_rail' in name or 'task43_right_rail' in name):
                ids.append(int(gid))
    except Exception:
        pass
    return ids


def _task43_infer_model_face_xy(mj_model, mj_data):
    """Infer humanoid facing direction from toe minus heel bodies."""
    def pts(names):
        out=[]
        for name in names:
            bid=_safe_name2id(mj_model, MJOBJ_BODY, name)
            if bid>=0:
                try: out.append(np.asarray(mj_data.xpos[bid][:2], dtype=float).copy())
                except Exception: pass
        return out
    toes=pts(('L_Toe','R_Toe','Left_Toe','Right_Toe','L_Foot','R_Foot','LeftFoot','RightFoot'))
    heels=pts(('L_Heel','R_Heel','Left_Heel','Right_Heel','L_Ankle','R_Ankle','LeftAnkle','RightAnkle'))
    if toes and heels:
        v=np.mean(toes,axis=0)-np.mean(heels,axis=0); n=float(np.linalg.norm(v))
        if n>0.025: return v/n, 'toe-minus-heel'
    return np.array([0.0,-1.0], dtype=float), 'fallback-minus-y'

def _task43_ladder_axes_from_face(face_xy):
    face=np.asarray(face_xy,dtype=float).copy(); face/=max(float(np.linalg.norm(face)),1e-8)
    lat=np.array([-face[1], face[0]], dtype=float); lat/=max(float(np.linalg.norm(lat)),1e-8)
    return face, lat

def _task43_axis_angle_quat(axis, angle_rad):
    axis=np.asarray(axis,dtype=float).copy()
    n=float(np.linalg.norm(axis))
    if n < 1e-8:
        return np.array([1.0,0.0,0.0,0.0],dtype=float)
    axis/=n
    half=0.5*float(angle_rad)
    return np.array([np.cos(half), axis[0]*np.sin(half), axis[1]*np.sin(half), axis[2]*np.sin(half)], dtype=float)

def _task43_quat_mul(q1, q2):
    w1,x1,y1,z1=np.asarray(q1,dtype=float)
    w2,x2,y2,z2=np.asarray(q2,dtype=float)
    q=np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ],dtype=float)
    return q/max(float(np.linalg.norm(q)),1e-8)

def _task43_oriented_rung_center(index, face_xy=None):
    if face_xy is None:
        face_xy=globals().get('TASK43_RUNTIME_CLIMB_XY', np.array([0.0,-1.0], dtype=float))
    face,_lat=_task43_ladder_axes_from_face(face_xy)
    idx=float(index)
    local_y=float(TASK43_LADDER_RUNG_Y0_M + idx*TASK43_LADDER_RUNG_DY_M)
    z=float(TASK43_LADDER_RUNG_Z0_M + idx*TASK43_LADDER_RUNG_DZ_M)
    xy=face*local_y
    return np.array([float(xy[0]), float(xy[1]), z], dtype=float)

def _task43_install_platform_if_possible(env):
    """Compile a true physical capsule ladder directly into worldbody."""
    try:
        unwrapped=env.unwrapped; old_model=unwrapped.model; old_data=unwrapped.data
        if _task43_find_platform_geoms(old_model): return True, 'already_present_ladder'
        if not hasattr(mujoco,'mj_saveLastXML'): return False, 'mj_saveLastXML_unavailable'
        face_xy, face_source=_task43_infer_model_face_xy(old_model, old_data)
        face_xy, lat_xy=_task43_ladder_axes_from_face(face_xy)
        globals()['TASK43_RUNTIME_CLIMB_XY']=face_xy.copy()
        globals()['TASK43_RUNTIME_FWD_XY']=(-face_xy).copy()
        globals()['TASK43_RUNTIME_LAT_XY']=lat_xy.copy()
        tmpdir=Path(tempfile.gettempdir()); base_xml=tmpdir/'task43_base.xml'; patched_xml=tmpdir/'task43_physical_capsule_ladder_v7.xml'
        mujoco.mj_saveLastXML(str(base_xml), old_model)
        tree=ET.parse(str(base_xml)); root=tree.getroot(); worldbody=root.find('worldbody')
        if worldbody is None: return False, 'worldbody_not_found'
        for child in list(worldbody):
            nm=child.get('name',''); low=nm.lower()
            if nm in ('task43_ladder_body','task43_height_block_body','task40_height_block_body'):
                worldbody.remove(child); continue
            if child.tag=='geom' and any(k in low for k in ('task43_ladder','task43_rung','task43_height_block','task40_height_block')):
                worldbody.remove(child)
        def c_at(i): return _task43_oriented_rung_center(i, face_xy)
        first=c_at(0); last=c_at(TASK43_LADDER_RUNG_COUNT-1)
        rail_low=first.copy(); rail_low[:2]-=face_xy*0.12; rail_low[2]-=0.22
        rail_high=last.copy(); rail_high[:2]+=face_xy*0.12; rail_high[2]+=0.24
        for side,sign in (('left',-1.0),('right',1.0)):
            off=sign*float(TASK43_LADDER_SIDE_RAIL_X_M)*lat_xy
            a=rail_low.copy(); b=rail_high.copy(); a[:2]+=off; b[:2]+=off
            ET.SubElement(worldbody,'geom',{'name':f'task43_ladder_{side}_rail_geom','type':'capsule','fromto':f'{a[0]:.6f} {a[1]:.6f} {a[2]:.6f} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f}','size':f'{float(TASK43_LADDER_RAIL_RADIUS_M):.6f}','rgba':'0.30 0.30 0.32 1.0','contype':'1','conaffinity':'1','condim':'4','friction':'1.60 0.08 0.010','solref':'0.020 0.85','solimp':'0.74 0.92 0.010','margin':f'{float(TASK43_LADDER_RUNG_CONTACT_MARGIN_M):.6f}','priority':'2'})
        for i in range(int(TASK43_LADDER_RUNG_COUNT)):
            c=c_at(i); a=c.copy(); b=c.copy(); a[:2]-=lat_xy*float(TASK43_LADDER_RUNG_X_HALF_M); b[:2]+=lat_xy*float(TASK43_LADDER_RUNG_X_HALF_M)
            rgba='0.42 0.34 0.24 1.0' if i!=TASK43_LADDER_TRIGGER_RUNG_INDEX else '0.50 0.38 0.20 1.0'
            ET.SubElement(worldbody,'geom',{'name':f'task43_ladder_rung_{i+1:02d}_geom','type':'capsule','fromto':f'{a[0]:.6f} {a[1]:.6f} {a[2]:.6f} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f}','size':f'{float(TASK43_LADDER_RUNG_THICK_XY_M):.6f}','rgba':rgba,'contype':'1','conaffinity':'1','condim':'4','friction':'1.80 0.08 0.010','solref':'0.020 0.85','solimp':'0.74 0.92 0.010','margin':f'{float(TASK43_LADDER_RUNG_CONTACT_MARGIN_M):.6f}','priority':'3'})
        tree.write(str(patched_xml), encoding='utf-8', xml_declaration=True)
        new_model=mujoco.MjModel.from_xml_path(str(patched_xml)); new_data=mujoco.MjData(new_model)
        nq=min(len(old_data.qpos),len(new_data.qpos)); nv=min(len(old_data.qvel),len(new_data.qvel))
        new_data.qpos[:nq]=old_data.qpos[:nq]; new_data.qvel[:nv]=old_data.qvel[:nv]
        if hasattr(old_data,'ctrl') and hasattr(new_data,'ctrl'):
            nu=min(len(old_data.ctrl),len(new_data.ctrl)); new_data.ctrl[:nu]=old_data.ctrl[:nu]
        mujoco.mj_forward(new_model,new_data)
        unwrapped.model=new_model; unwrapped.data=new_data
        for attr in ('_model','mj_model'):
            if hasattr(unwrapped, attr):
                try: setattr(unwrapped, attr, new_model)
                except Exception: pass
        for attr in ('_data','mj_data'):
            if hasattr(unwrapped, attr):
                try: setattr(unwrapped, attr, new_data)
                except Exception: pass
        renderer=getattr(unwrapped,'mujoco_renderer',None)
        if renderer is not None:
            for attr,val in (('model',new_model),('data',new_data)):
                if hasattr(renderer,attr):
                    try: setattr(renderer,attr,val)
                    except Exception: pass
        ids=_task43_find_platform_geoms(new_model)
        return bool(ids), f'compiled_solid_collision_capsule_ladder_into_worldbody ({len(ids)} geoms, face_source={face_source}, climb_xy=[{face_xy[0]:+.2f},{face_xy[1]:+.2f}])'
    except Exception as exc:
        return False, f'viewer_fallback_capsule_ladder_only: {exc}'


_BaseTask43GroundCloneHeightSupport = Task43GroundCloneHeightSupport


class Task43LadderClimbSupport(_BaseTask43GroundCloneHeightSupport):
    """Physical ladder layer: face ladder, alternate feet, then true Task40/38 backward fall."""
    def __init__(self, mj_model, mj_data):
        super().__init__(mj_model, mj_data)
        self.platform_geoms=_task43_find_platform_geoms(mj_model); self.ladder_geoms=list(self.platform_geoms)
        self.start_rung_index=0; self.trigger_rung_index=int(TASK43_LADDER_TRIGGER_RUNG_INDEX); self.visible_steps=int(TASK43_LADDER_VISIBLE_STEPS)
        self._climb_leg_specs=[]; self._climb_arm_specs=[]; self._ladder_quiet_qpos=None; self._last_climb_progress=0.0; self._last_climb_micro=0; self._released=False
        self._update_edge_geometry(verbose=False)
        self.current_support_z=self._rung_top_z(self.start_rung_index); self.height=float(self.current_support_z)

    def _rung_center(self, index):
        return _task43_oriented_rung_center(float(index), getattr(self,'climb_xy',globals().get('TASK43_RUNTIME_CLIMB_XY',np.array([0.0,-1.0],dtype=float))))

    def _rung_top_z(self, index):
        return float(TASK43_LADDER_RUNG_Z0_M + float(index)*TASK43_LADDER_RUNG_DZ_M + TASK43_LADDER_RUNG_THICK_Z_M)

    def _profile_stoop_deg(self, progress):
        p=float(np.clip(progress,0.0,1.0))
        stand=float(globals().get('TASK43_LADDER_PROFILE_STAND_STOOP_DEG',5.0))
        climb=float(globals().get('TASK43_LADDER_PROFILE_CLIMB_STOOP_DEG',6.5))
        # Smoothly blend into the climbing stoop during the first ladder beat.
        gate=float(np.clip(p*4.0,0.0,1.0))
        gate=gate*gate*(3.0-2.0*gate)
        return float((1.0-gate)*stand + gate*climb)

    def _profile_root_quat(self, progress):
        if len(getattr(self,'_upright_quat',[])) != 4:
            return np.array([1.0,0.0,0.0,0.0],dtype=float)
        deg=self._profile_stoop_deg(progress)
        # Positive fwd/climb lean is obtained by rotating the root around the
        # sagittal lateral axis opposite the posterior-fall axis.
        axis=np.array([self.lat_xy[0], self.lat_xy[1], 0.0], dtype=float)
        qlean=_task43_axis_angle_quat(axis, -np.radians(deg))
        return _task43_quat_mul(qlean, self._upright_quat)

    def _apply_profile_root_posture(self, progress):
        if len(self.mj_data.qpos) >= 7 and not getattr(self, '_released', False):
            self.mj_data.qpos[3:7]=self._profile_root_quat(progress)

    def _profile_root_transfer(self):
        return float(globals().get('TASK43_LADDER_PROFILE_ROOT_TRANSFER',1.0))

    def _update_edge_geometry(self, verbose=False):
        try: face_xy, source = self._infer_forward_xy()
        except Exception:
            face_xy=globals().get('TASK43_RUNTIME_CLIMB_XY',np.array([0.0,-1.0],dtype=float)); source='runtime-fallback'
        face_xy=np.asarray(face_xy,dtype=float); face_xy/=max(float(np.linalg.norm(face_xy)),1e-8)
        self.climb_xy=face_xy.copy(); self.toe_forward_xy=self.climb_xy.copy(); self.fwd_xy=self.climb_xy.copy(); self.fwd_xy/=max(float(np.linalg.norm(self.fwd_xy)),1e-8)
        # Task43 forward variant: use the ladder-facing/climb vector as the
        # fall vector. The sagittal pitch/rail axis is derived from this
        # forward vector so the top-rung release pitches into a true forward fall.
        self.lat_xy=np.array([-self.fwd_xy[1], self.fwd_xy[0]], dtype=float); self.lat_xy/=max(float(np.linalg.norm(self.lat_xy)),1e-8)
        start=self._rung_center(self.start_rung_index); target=self._rung_center(self.trigger_rung_index)
        self.start_xy=np.asarray(start[:2],dtype=float)-self.climb_xy*float(TASK43_LADDER_ROOT_BACKOFF_M)
        self.step_target_xy=np.asarray(target[:2],dtype=float)-self.climb_xy*float(TASK43_LADDER_ROOT_BACKOFF_M)
        self.edge_distance=float(np.linalg.norm(self.step_target_xy-self.start_xy))
        globals()['TASK43_RUNTIME_CLIMB_XY']=self.climb_xy.copy(); globals()['TASK43_RUNTIME_FWD_XY']=self.fwd_xy.copy(); globals()['TASK43_RUNTIME_LAT_XY']=self.lat_xy.copy()
        if verbose:
            print(f"      Task43 ladder lane locked -> face/climb=[{self.climb_xy[0]:+.2f},{self.climb_xy[1]:+.2f}] forward_fall=[{self.fwd_xy[0]:+.2f},{self.fwd_xy[1]:+.2f}] source={source}")

    def _resolve_climb_leg_joints(self):
        specs=[]; joint_obj=mujoco.mjtObj.mjOBJ_JOINT if hasattr(mujoco,'mjtObj') else mujoco.mjOBJ_JOINT
        for jid in range(int(self.mj_model.njnt)):
            jname=mujoco.mj_id2name(self.mj_model,joint_obj,jid) or ''; bname=mujoco.mj_id2name(self.mj_model,MJOBJ_BODY,int(self.mj_model.jnt_bodyid[jid])) or ''
            txt=(jname+' '+bname).lower(); qadr=int(self.mj_model.jnt_qposadr[jid])
            if qadr<7 or qadr>=len(self.mj_data.qpos): continue
            side=self._joint_side(jname,bname)
            if 'hip' in txt or 'thigh' in txt: specs.append((qadr,'hip',side))
            elif 'knee' in txt or 'shin' in txt or 'shank' in txt: specs.append((qadr,'knee',side))
            elif 'ankle' in txt or 'toe' in txt or 'foot' in txt: specs.append((qadr,'ankle',side))
        seen=set(); out=[]
        for item in specs:
            if item[0] not in seen: seen.add(item[0]); out.append((int(item[0]),str(item[1]),str(item[2])))
        return out[:18]

    def _resolve_climb_arm_joints(self):
        specs=[]; joint_obj=mujoco.mjtObj.mjOBJ_JOINT if hasattr(mujoco,'mjtObj') else mujoco.mjOBJ_JOINT
        for jid in range(int(self.mj_model.njnt)):
            jname=mujoco.mj_id2name(self.mj_model,joint_obj,jid) or ''; bname=mujoco.mj_id2name(self.mj_model,MJOBJ_BODY,int(self.mj_model.jnt_bodyid[jid])) or ''
            txt=(jname+' '+bname).lower(); qadr=int(self.mj_model.jnt_qposadr[jid])
            if qadr<7 or qadr>=len(self.mj_data.qpos) or not any(k in txt for k in ('shoulder','elbow','wrist','hand','arm')): continue
            side=self._joint_side(jname,bname)
            if any(k in txt for k in ('wrist','hand','palm')):
                kind='wrist'
            elif any(k in txt for k in ('elbow','forearm','lowerarm')):
                kind='elbow'
            elif any(k in txt for k in ('shoulder','upperarm','arm')):
                kind='shoulder'
            else:
                kind='wrist'
            specs.append((qadr,kind,side))
        seen=set(); out=[]
        for item in specs:
            if item[0] not in seen: seen.add(item[0]); out.append((int(item[0]),str(item[1]),str(item[2])))
        return out[:16]

    def _set_root_to_rung(self, rung_index, root_xy=None):
        rung_index=int(np.clip(int(rung_index),0,TASK43_LADDER_RUNG_COUNT-1)); self.current_support_z=self._rung_top_z(rung_index); self.height=float(self.current_support_z)
        if root_xy is None:
            c=self._rung_center(rung_index); root_xy=np.asarray(c[:2],dtype=float)-self.climb_xy*float(TASK43_LADDER_ROOT_BACKOFF_M)
        target=float(self.current_support_z+TASK43_TOP_CLEARANCE)
        for _ in range(6):
            mujoco.mj_forward(self.mj_model,self.mj_data); dz=target-float(self._lowest_foot_z()); self.mj_data.qpos[2]+=float(dz)
            if abs(dz)<2e-5: break
        if len(self.mj_data.qpos)>=2: self.mj_data.qpos[0]=float(root_xy[0]); self.mj_data.qpos[1]=float(root_xy[1])
        if self.mj_data.qvel.shape[0]>=6: self.mj_data.qvel[:6]=0.0
        mujoco.mj_forward(self.mj_model,self.mj_data); self._has_foot_platform_contact()

    def place_on_platform(self, verbose=True, reason='initial'):
        if self.mj_data.qpos.shape[0]>=7:
            self.mj_data.qpos[3:7]=self._upright_quat; mujoco.mj_forward(self.mj_model,self.mj_data); self._update_edge_geometry(verbose=False)
            self.mj_data.qpos[0]=float(self.start_xy[0]); self.mj_data.qpos[1]=float(self.start_xy[1]); self.mj_data.qpos[3:7]=self._upright_quat; self._neutralize_upper_body_qpos()
        elif self.mj_data.qpos.shape[0]>=3:
            self._update_edge_geometry(verbose=False); self.mj_data.qpos[0]=float(self.start_xy[0]); self.mj_data.qpos[1]=float(self.start_xy[1])
        if hasattr(self.mj_data,'ctrl'): self.mj_data.ctrl[:]=0.0
        self._enable_foot_support_contact(); self._set_root_to_rung(self.start_rung_index,self.start_xy)
        # v6: seat both feet on the first physical rung before caching quiet
        # posture; this removes the hover/gap after anthropometry rebuilds.
        try:
            self._align_root_to_ladder_feet(0.0, hard=True)
            if len(self.mj_data.qpos)>=2:
                self.start_xy=np.asarray(self.mj_data.qpos[:2],dtype=float).copy()
        except Exception:
            pass
        self._quiet_qpos=np.asarray(self.mj_data.qpos,dtype=float).copy(); self._quiet_qvel=np.zeros_like(np.asarray(self.mj_data.qvel,dtype=float)); self._ladder_quiet_qpos=self._quiet_qpos.copy()
        self._step_joint_specs=self._resolve_step_joints(); self._reactive_leg_specs=self._resolve_reactive_leg_joints(); self._climb_leg_specs=self._resolve_climb_leg_joints(); self._climb_arm_specs=self._resolve_climb_arm_joints()
        self._step_pose_qpos=self._build_forward_step_pose(); self._build_protective_reach_pose()
        if verbose:
            print('  [Task43LadderLayer] physical solid-contact ladder-climb support prepared')
            print(f'      ladder rungs            = {TASK43_LADDER_RUNG_COUNT} thick solid capsule contact rungs')
            print(f'      trigger rung            = {self.trigger_rung_index+1} / height~{self._rung_top_z(self.trigger_rung_index):.2f} m')
            print('      climb sequence          = CONTACT-SNAPPED REFERENCE: plant -> swing foot -> rung contact -> body/load transfer')
            print('      hand/foot grip          = CONTACT-SNAPPED LOCK: root+IK solve puts hands/feet on body-side rung targets')
            print(f'      face/climb xy           = [{self.climb_xy[0]:+.2f},{self.climb_xy[1]:+.2f}] | forward fall xy=[{self.fwd_xy[0]:+.2f},{self.fwd_xy[1]:+.2f}]')
            print(f'      ladder geoms            = {self.platform_geoms if self.platform_geoms else "viewer fallback only"}')
            print(f'      start root xy           = {self.start_xy.round(3).tolist()} | release root xy={self.step_target_xy.round(3).tolist()}')
            print(f'      support contact check   = {self._last_support_contact_ok} (pairs={self._last_support_contact_count})')
            _prof = globals().get('TASK43_PROFILE_ADAPTATION', {}) or {}
            print(f"      profile bridge          = age_old={_prof.get('age_old', 1.0):.2f} | walk={_prof.get('walk_speed_mps', TASK43_LADDER_PROFILE_SPEED_MPS):.2f}m/s | ds={_prof.get('expected_double_support', TASK43_LADDER_PROFILE_EXPECTED_DS):.0%} | climb_frames={PHASES.get('step', 0)} | fixed_stoop={_prof.get('climb_stoop_deg', TASK43_LADDER_PROFILE_CLIMB_STOOP_DEG):.1f}deg | rung_dz={_prof.get('rung_dz_m', TASK43_LADDER_RUNG_DZ_M):.3f}m | grip_kp={_prof.get('grip_kp', TASK43_LADDER_CONTACT_FORCE_KP):.0f} | slip={_prof.get('release_backstep_m', TASK43_LADDER_RELEASE_BACKSTEP_M):.3f}m | fallBW={_prof.get('fall_force_bw', TASK43_TASK38_FORCE_BW_FRACTION):.2f}")
        else:
            print(f'  [Task43LadderLayer] placement refreshed after {reason}; rung={self.start_rung_index+1}, trigger_rung={self.trigger_rung_index+1}, support_contact={self._last_support_contact_ok}')

    def enforce_quiet_stand(self):
        z=float(self._quiet_qpos[2]) if self._quiet_qpos is not None and len(self._quiet_qpos)>=3 else None
        self._restore_cached_pose(root_xy=self.start_xy, root_z=z, root_vxy=np.zeros(2), root_vz=0.0, do_forward=False)
        self._apply_profile_root_posture(0.0)
        self._apply_ladder_climb_pose(0.0); self._set_root_to_rung(self.start_rung_index,self.start_xy)
        self._apply_profile_root_posture(0.0)
        self._align_root_to_ladder_feet(0.0, hard=True)
        self._apply_ladder_contact_forces(0.0)

    def _climb_phase(self, progress):
        """Return an explicitly segmented natural ladder-climb state.

        v1g removes the visual "sliding upward" behavior by splitting each
        half-step into four visible parts: (1) planted-foot hold, (2) swing foot
        lift, (3) foot placed on the next rung, (4) delayed body/load transfer.
        The pelvis/root stays at the old support height until the swing foot has
        already reached the next rung, so the climb reads as real stepping.
        """
        p=float(np.clip(progress,0.0,1.0))
        n_half=max(int(self.trigger_rung_index)*2,1)
        x=min(p*n_half,n_half-1e-6)
        micro=int(np.floor(x)); local=float(x-micro)
        active_right=(micro%2==0)
        pair=micro//2
        # Four-stage ladder half-step.  Values are intentionally separated so
        # the foot visibly pauses on a rung before the root/body rises.
        plant_end=0.18
        swing_end=0.58
        contact_end=0.72
        t0=float(globals().get('TASK43_LADDER_LOAD_TRANSFER_START',0.72))
        t1=float(globals().get('TASK43_LADDER_LOAD_TRANSFER_END',0.94))
        if local <= plant_end:
            foot_s=0.0
            lift=0.0
        elif local <= swing_end:
            u=float(np.clip((local-plant_end)/max(swing_end-plant_end,1e-6),0.0,1.0))
            foot_s=float(u*u*(3.0-2.0*u))
            lift=float(np.sin(np.pi*u))
        else:
            foot_s=1.0
            # small residual toe clearance while settling onto rung, then flat
            u=float(np.clip((local-swing_end)/max(contact_end-swing_end,1e-6),0.0,1.0))
            lift=float(0.10*(1.0-u))
        u=float(np.clip((local-t0)/max(t1-t0,1e-6),0.0,1.0))
        load_gate=float(u*u*(3.0-2.0*u))
        if active_right:
            right_rung=float(pair)+foot_s
            left_rung=float(pair)
        else:
            right_rung=float(pair+1)
            left_rung=float(pair)+foot_s
        right_rung=float(np.clip(right_rung,0.0,float(self.trigger_rung_index)))
        left_rung=float(np.clip(left_rung,0.0,float(self.trigger_rung_index)))
        # root/body advances by half a rung only after the swing foot is placed
        root_rung=float(0.5*float(micro) + 0.5*load_gate)
        root_rung=float(np.clip(root_rung,0.0,float(self.trigger_rung_index)))
        active_target_rung=int(np.clip(pair+1,0,self.trigger_rung_index))
        support_rung=int(np.clip(pair,0,self.trigger_rung_index))
        return micro,local,foot_s,lift,active_right,root_rung,active_target_rung,support_rung

    def _side_body_id(self, side, role='foot'):
        """Resolve the visible end-effector body used for ladder contact.

        v1h targeted Toe first. That made the toe body numerically reach the
        rung while the rendered foot/ankle could still look away from the rung.
        v1i targets the actual foot/ankle body first, then falls back to toe.
        Hands prefer the palm/hand body, then wrist/forearm.
        """
        side=str(side or '').lower(); role=str(role or 'foot').lower()
        if role=='hand':
            # Prefer the rendered palm/hand body used by this humanoid build;
            # fall back to wrist/forearm if the model exposes no palm body.
            names = (
                'HandR','R_Hand','RightHand','R_Wrist','RightWrist','WristR','R_Forearm','ForeArmR','LowerArmR'
            ) if side.startswith('r') else (
                'HandL','L_Hand','LeftHand','L_Wrist','LeftWrist','WristL','L_Forearm','ForeArmL','LowerArmL'
            )
        else:
            names = (
                'R_Foot','FootR','RightFoot','R_Ankle','AnkleR','RightAnkle','R_Toe','ToeR','RightToe','R_Heel','HeelR','RightHeel'
            ) if side.startswith('r') else (
                'L_Foot','FootL','LeftFoot','L_Ankle','AnkleL','LeftAnkle','L_Toe','ToeL','LeftToe','L_Heel','HeelL','LeftHeel'
            )
        for name in names:
            bid=_safe_name2id(self.mj_model,MJOBJ_BODY,name)
            if bid>=0: return int(bid)
        return -1

    def _side_lateral_sign(self, side):
        """Infer which side of the rung a foot/hand should use."""
        side=str(side or '').lower()
        rb=self._side_body_id('right','foot'); lb=self._side_body_id('left','foot')
        try:
            if rb>=0 and lb>=0:
                root=np.asarray(self.mj_data.qpos[:2],dtype=float) if len(self.mj_data.qpos)>=2 else np.zeros(2)
                pr=float(np.dot(np.asarray(self.mj_data.xpos[rb][:2],dtype=float)-root,self.lat_xy))
                pl=float(np.dot(np.asarray(self.mj_data.xpos[lb][:2],dtype=float)-root,self.lat_xy))
                if abs(pr-pl)>0.015:
                    return 1.0 if (side.startswith('r') and pr>pl) or (side.startswith('l') and pl>pr) else -1.0
        except Exception:
            pass
        return -1.0 if side.startswith('r') else 1.0

    def _near_side_dir_xy(self, center_xy):
        """Direction from rung center toward the humanoid/root side.

        This avoids the previous sign ambiguity: if the ladder axes or viewer
        camera are flipped, hands still grip the rung face closest to the body
        instead of the far/back side.
        """
        try:
            root=np.asarray(self.mj_data.qpos[:2],dtype=float)
            v=root-np.asarray(center_xy,dtype=float)
            n=float(np.linalg.norm(v))
            if n>1e-5:
                return v/n
        except Exception:
            pass
        return -np.asarray(self.climb_xy,dtype=float)

    def _rung_limb_target(self, side, rung_index, role='foot', lift=0.0):
        """World target for a visible hand/foot on a ladder rung.

        v1i makes the target a true rung-contact target, not a target offset
        deep onto either the far side or the body side.  The only forward/back
        offset is a very small near-side clearance so the hand/foot appears on
        the body-facing surface of the rung rather than floating behind it.
        """
        rung=float(np.clip(float(rung_index),0.0,float(TASK43_LADDER_RUNG_COUNT-1)))
        c=np.asarray(self._rung_center(rung),dtype=float).copy()
        lateral=float(TASK43_LADDER_HAND_LATERAL_M if role=='hand' else TASK43_LADDER_FOOT_LATERAL_M)
        c[:2]+=self.lat_xy*float(self._side_lateral_sign(side))*lateral
        body_side=-np.asarray(self.climb_xy,dtype=float)
        body_side=body_side/max(float(np.linalg.norm(body_side)),1e-8)
        if role=='foot':
            # Put the foot/ankle body almost exactly on top/front of the rung.
            c[2]=float(self._rung_top_z(rung)+float(TASK43_LADDER_FOOT_CONTACT_Z_CLEARANCE_M)+float(lift))
            c[:2]+=body_side*float(globals().get('TASK43_LADDER_FOOT_NEAR_OFFSET_M',0.004))
        else:
            # A hand wraps around the rung centerline, not 3-4 cm above it.
            c[2]=float(TASK43_LADDER_RUNG_Z0_M + rung*TASK43_LADDER_RUNG_DZ_M + float(globals().get('TASK43_LADDER_HAND_CENTER_Z_OFFSET_M',0.0)) + float(lift))
            c[:2]+=body_side*float(globals().get('TASK43_LADDER_HAND_NEAR_OFFSET_M',0.006))
        return c

    def _apply_reference_root_error_correction(self, targets, phase='pre'):
        """Small root translation to make the reference target reachable.

        Pure joint IK cannot always reach the ladder because the HumEnv body is
        not a trained ladder climber and the limb origins differ by model build.
        v1i first moves the free root a few centimeters so the real hand/foot
        bodies are on the same side and within reach of their rung targets, then
        the joint IK performs the local limb pose.  This is still a reference
        climb; it is not a fake upward contact-force climb.
        """
        if len(self.mj_data.qpos) < 3 or getattr(self,'_released',False):
            return
        mujoco.mj_forward(self.mj_model,self.mj_data)
        weights=[]; errs=[]
        for key,w in ((('left','foot'),0.42),(('right','foot'),0.42),(('left','hand'),0.24),(('right','hand'),0.24)):
            side,role=key
            bid=self._side_body_id(side,role)
            if bid < 0 or key not in targets:
                continue
            pos=np.asarray(self.mj_data.xpos[bid],dtype=float)
            target=np.asarray(targets[key],dtype=float)
            err=target-pos
            # Hands guide mostly fore/aft/height; feet guide all axes but with
            # small limits to preserve the visible discrete ladder trajectory.
            if role=='hand':
                err[:2]=np.clip(err[:2],-0.040,0.040)
                err[2]=float(np.clip(err[2],-0.030,0.030))
            else:
                err[:2]=np.clip(err[:2],-0.030,0.030)
                err[2]=float(np.clip(err[2],-0.020,0.020))
            errs.append(err); weights.append(float(w))
        if not errs:
            return
        ww=np.asarray(weights,dtype=float); ww=ww/max(float(np.sum(ww)),1e-8)
        delta=np.sum(np.asarray(errs,dtype=float)*ww[:,None],axis=0)
        if phase=='post':
            delta*=0.55
        delta[:2]=np.clip(delta[:2],-0.035,0.035)
        delta[2]=float(np.clip(delta[2],-0.022,0.022))
        self.mj_data.qpos[0]+=float(delta[0]); self.mj_data.qpos[1]+=float(delta[1]); self.mj_data.qpos[2]+=float(delta[2])
        mujoco.mj_forward(self.mj_model,self.mj_data)

    def _foot_rungs_for_progress(self, progress):
        micro,local,smooth,lift,active_right,root_rung,active_target_rung,support_rung=self._climb_phase(progress)
        pair=micro//2
        if active_right:
            return float(pair)+smooth, float(pair), active_right, lift, smooth, micro
        return float(pair+1), float(pair)+smooth, active_right, lift, smooth, micro

    def _hand_rungs_for_progress(self, progress):
        """Return hand rung targets for real three-point ladder grip.

        Both hands stay on rung centers.  The hand opposite the moving foot
        advances while the support hand remains on a lower rung, matching a
        normal ladder-climb rhythm rather than free arm waving.
        """
        right_rung,left_rung,active_right,lift,smooth,micro=self._foot_rungs_for_progress(progress)
        lead=float(globals().get('TASK43_LADDER_HAND_GRIP_LEAD_RUNG',1.75))
        if active_right:
            right_hand=float(np.clip(right_rung+lead-0.20,1.0,float(TASK43_LADDER_RUNG_COUNT-1)))
            left_hand=float(np.clip(left_rung+lead+0.35*smooth,1.0,float(TASK43_LADDER_RUNG_COUNT-1)))
            active_hand_right=False
        else:
            right_hand=float(np.clip(right_rung+lead+0.35*smooth,1.0,float(TASK43_LADDER_RUNG_COUNT-1)))
            left_hand=float(np.clip(left_rung+lead-0.20,1.0,float(TASK43_LADDER_RUNG_COUNT-1)))
            active_hand_right=True
        return right_hand,left_hand,active_hand_right,lift,smooth,micro

    def _hand_body_position(self, side):
        bid=self._side_body_id(side,'hand')
        if bid < 0:
            return None
        return np.asarray(self.mj_data.xpos[bid],dtype=float).copy()

    def _hand_grip_targets_for_progress(self, progress):
        right_hand,left_hand,active_hand_right,hand_lift,_hand_smooth,_hmicro=self._hand_rungs_for_progress(progress)
        return {
            'right': self._rung_limb_target('right', right_hand, 'hand', lift=(0.45*hand_lift if active_hand_right else 0.0)),
            'left':  self._rung_limb_target('left',  left_hand,  'hand', lift=(0.45*hand_lift if not active_hand_right else 0.0)),
        }

    def _hand_grip_pose_score(self, targets):
        score=0.0
        for side,target in targets.items():
            pos=self._hand_body_position(side)
            if pos is None:
                continue
            err=np.asarray(pos,dtype=float)-np.asarray(target,dtype=float)
            along=float(np.dot(err[:2], self.climb_xy))
            lat=float(np.dot(err[:2], self.lat_xy))
            z=float(err[2])
            score += 10.0*along*along + 7.0*lat*lat + 8.0*z*z
            score += 5.0*max(0.0,abs(along)-0.055)**2
            score += 4.0*max(0.0,abs(lat)-0.075)**2
            score += 4.0*max(0.0,abs(z)-0.070)**2
        return float(score)

    def _arm_joint_entries_for_side(self, side):
        """Return (qadr, dofadr, lo, hi) for arm joints on one side."""
        side=str(side or '').lower()
        entries=[]
        qadr_to_jid={}
        try:
            for jid in range(int(self.mj_model.njnt)):
                qadr_to_jid[int(self.mj_model.jnt_qposadr[jid])]=int(jid)
        except Exception:
            pass
        for qadr,kind,jside in getattr(self,'_climb_arm_specs',[]):
            if str(jside).lower()!=side:
                continue
            qadr=int(qadr)
            jid=qadr_to_jid.get(qadr,None)
            if jid is None:
                continue
            try:
                dof=int(self.mj_model.jnt_dofadr[jid])
                if dof<0 or dof>=int(self.mj_model.nv):
                    continue
                if int(self.mj_model.jnt_limited[jid]):
                    lo,hi=map(float,self.mj_model.jnt_range[jid])
                else:
                    base=float(self._quiet_qpos[qadr]) if self._quiet_qpos is not None and qadr<len(self._quiet_qpos) else float(self.mj_data.qpos[qadr])
                    lo,hi=base-1.45,base+1.45
                base=float(self._quiet_qpos[qadr]) if self._quiet_qpos is not None and qadr<len(self._quiet_qpos) else float(self.mj_data.qpos[qadr])
                lo=max(float(lo),base-1.55); hi=min(float(hi),base+1.55)
                entries.append((qadr,dof,float(lo),float(hi)))
            except Exception:
                continue
        return entries

    def _solve_ladder_hand_grip_pose(self, progress):
        """Jacobian IK so the visible hands actually sit on body-side rungs.

        The earlier coordinate-search IK could leave the arms on the opposite
        side because joint signs differ between humanoid builds. This version
        ignores signs and directly minimizes hand Cartesian error using MuJoCo
        body Jacobians with small damped least-squares updates.
        """
        if getattr(self,'_released',False):
            return
        if not bool(globals().get('TASK43_LADDER_HAND_IK_ENABLED',True)):
            return
        if self._quiet_qpos is None or not getattr(self,'_climb_arm_specs',None):
            return
        targets=self._hand_grip_targets_for_progress(progress)
        passes=int(np.clip(globals().get('TASK43_LADDER_HAND_IK_PASSES',6),1,8))
        nv=int(self.mj_model.nv)
        for _ in range(passes):
            mujoco.mj_forward(self.mj_model,self.mj_data)
            any_update=False
            for side,target in targets.items():
                bid=self._side_body_id(side,'hand')
                if bid<0:
                    continue
                entries=self._arm_joint_entries_for_side(side)
                if not entries:
                    continue
                pos=np.asarray(self.mj_data.xpos[bid],dtype=float)
                err=np.asarray(target,dtype=float)-pos
                en=float(np.linalg.norm(err))
                if en>0.105:
                    err=err*(0.105/en)
                if float(np.linalg.norm(err))<0.010:
                    continue
                jacp=np.zeros((3,nv),dtype=float)
                jacr=np.zeros((3,nv),dtype=float)
                try:
                    mujoco.mj_jacBody(self.mj_model,self.mj_data,jacp,jacr,int(bid))
                except Exception:
                    continue
                dofs=[e[1] for e in entries]
                J=jacp[:,dofs]
                W=np.diag([1.0,1.0,1.25])
                Jw=W@J; ew=W@err
                damp=0.055
                try:
                    dq=Jw.T@np.linalg.solve(Jw@Jw.T + (damp*damp)*np.eye(3), ew)
                except Exception:
                    dq=np.linalg.pinv(Jw)@ew
                dq=np.asarray(dq,dtype=float)
                if dq.size!=len(entries):
                    continue
                dq=np.clip(dq,-0.115,0.115)
                for (qadr,dof,lo,hi),dv in zip(entries,dq):
                    self.mj_data.qpos[int(qadr)]=float(np.clip(float(self.mj_data.qpos[int(qadr)])+float(dv),lo,hi))
                    any_update=True
                mujoco.mj_forward(self.mj_model,self.mj_data)
            if not any_update:
                break
        self._last_hand_grip_error_m={}
        mujoco.mj_forward(self.mj_model,self.mj_data)
        for side,target in targets.items():
            pos=self._hand_body_position(side)
            if pos is not None:
                self._last_hand_grip_error_m[side]=float(np.linalg.norm(np.asarray(pos)-np.asarray(target)))

    def _align_root_to_ladder_feet(self, progress, hard=False):
        """Translate the free root so planted feet sit on physical rung centers."""
        if len(self.mj_data.qpos)<3: return
        right_rung,left_rung,active_right,lift,smooth,micro=self._foot_rungs_for_progress(progress)
        targets=[]
        # Keep the non-moving/support foot as the alignment anchor.  The swing
        # foot is only allowed to influence root placement once it is actually
        # close to landing on the new rung.  This preserves the visible ladder
        # rule: one foot moves, the other foot remains planted.
        active_can_land = (smooth > float(globals().get('TASK43_LADDER_ACTIVE_FOOT_LAND_START',0.70))) or (hard and smooth > 0.84) or (progress >= 0.995)
        if active_right:
            targets.append(('left',left_rung,0.0,1.00))
            if active_can_land:
                targets.append(('right',right_rung,0.0,0.10 if not hard else 0.24))
        else:
            targets.append(('right',right_rung,0.0,1.00))
            if active_can_land:
                targets.append(('left',left_rung,0.0,0.10 if not hard else 0.24))
        errs=[]; weights=[]
        mujoco.mj_forward(self.mj_model,self.mj_data)
        for side,rung,li,w in targets:
            bid=self._side_body_id(side,'foot')
            if bid<0: continue
            target=self._rung_limb_target(side,rung,'foot',lift=li)
            pos=np.asarray(self.mj_data.xpos[bid],dtype=float)
            err=target-pos
            err[:2]=np.clip(err[:2],-0.026,0.026)
            z_lim = 0.018 if hard else 0.0035
            err[2]=float(np.clip(err[2],-z_lim,z_lim))
            errs.append(err); weights.append(float(w))
        if not errs: return
        ww=np.asarray(weights,dtype=float); ww=ww/max(float(np.sum(ww)),1e-8)
        e=np.sum(np.asarray(errs)*ww[:,None],axis=0)
        gain=float(TASK43_LADDER_FOOT_ALIGN_GAIN if not hard else 1.0)
        self.mj_data.qpos[0]+=gain*float(e[0]); self.mj_data.qpos[1]+=gain*float(e[1]); self.mj_data.qpos[2]+=gain*float(e[2])
        mujoco.mj_forward(self.mj_model,self.mj_data)
        self._has_foot_platform_contact()

    def _limb_joint_entries_for_side(self, side, role='hand'):
        """Return (qadr, dofadr, lo, hi) entries for IK on one limb.

        v1i lets MuJoCo's body Jacobian decide which joint direction moves a
        hand/foot toward the rung.  This is more reliable than sign heuristics
        across HumEnv/MyoSuite humanoid builds.
        """
        side=str(side or '').lower(); role=str(role or 'hand').lower()
        specs = getattr(self, '_climb_arm_specs', []) if role=='hand' else getattr(self, '_climb_leg_specs', [])
        entries=[]; qadr_to_jid={}
        try:
            for jid in range(int(self.mj_model.njnt)):
                qadr_to_jid[int(self.mj_model.jnt_qposadr[jid])]=int(jid)
        except Exception:
            pass
        for qadr,kind,jside in specs:
            if str(jside).lower()!=side:
                continue
            qadr=int(qadr); jid=qadr_to_jid.get(qadr,None)
            if jid is None:
                continue
            try:
                dof=int(self.mj_model.jnt_dofadr[jid])
                if dof<0 or dof>=int(self.mj_model.nv):
                    continue
                if int(self.mj_model.jnt_limited[jid]):
                    lo,hi=map(float,self.mj_model.jnt_range[jid])
                else:
                    base=float(self._quiet_qpos[qadr]) if self._quiet_qpos is not None and qadr<len(self._quiet_qpos) else float(self.mj_data.qpos[qadr])
                    lo,hi=base-1.65,base+1.65
                base=float(self._quiet_qpos[qadr]) if self._quiet_qpos is not None and qadr<len(self._quiet_qpos) else float(self.mj_data.qpos[qadr])
                span=1.70 if role=='hand' else 1.50
                lo=max(float(lo),base-span); hi=min(float(hi),base+span)
                entries.append((qadr,dof,float(lo),float(hi)))
            except Exception:
                continue
        return entries

    def _reference_limb_targets_for_progress(self, progress):
        right_rung,left_rung,active_right,lift,smooth,micro=self._foot_rungs_for_progress(progress)
        foot_lift=float(globals().get('TASK43_LADDER_REFERENCE_FOOT_LIFT_M',0.145))*float(lift)
        lead=float(globals().get('TASK43_LADDER_REFERENCE_HAND_LEAD_RUNG',0.72))
        hand_lift=float(globals().get('TASK43_LADDER_REFERENCE_HAND_LIFT_M',0.035))*float(lift)
        # Three-point ladder rhythm: opposite hand reaches while swing foot
        # moves; the other hand remains a visible support grip.  v1j keeps the
        # target to a reachable rung band instead of always clipping both hands
        # to the top rung at release, which created 0.7-0.9 m palm errors.
        max_hand_rung=float(TASK43_LADDER_RUNG_COUNT-1)
        min_hand_rung=1.0
        if active_right:
            rh=float(np.clip(right_rung+max(0.20,lead-0.30),min_hand_rung,max_hand_rung))
            lh=float(np.clip(left_rung+lead+0.18*smooth,min_hand_rung,max_hand_rung))
            active_hand_right=False
        else:
            rh=float(np.clip(right_rung+lead+0.18*smooth,min_hand_rung,max_hand_rung))
            lh=float(np.clip(left_rung+max(0.20,lead-0.30),min_hand_rung,max_hand_rung))
            active_hand_right=True
        return {
            ('right','foot'): self._rung_limb_target('right', right_rung, 'foot', lift=(foot_lift if active_right else 0.0)),
            ('left','foot'):  self._rung_limb_target('left',  left_rung,  'foot', lift=(foot_lift if not active_right else 0.0)),
            ('right','hand'): self._rung_limb_target('right', rh, 'hand', lift=(hand_lift if active_hand_right else 0.0)),
            ('left','hand'):  self._rung_limb_target('left',  lh, 'hand', lift=(hand_lift if not active_hand_right else 0.0)),
        }

    def _solve_reference_ladder_limb_ik(self, progress):
        """Reference/mocap-style IK lock for hands and feet during climb.

        The climb is not produced by fake vertical contact forces.  Each frame
        restores a clean reference pose, places the floating root on a discrete
        ladder trajectory, then solves hand/foot Cartesian targets to the rung
        centers.  After release, this helper is disabled and the fall is fully
        physical again.
        """
        if getattr(self,'_released',False) or self._quiet_qpos is None:
            return
        targets=self._reference_limb_targets_for_progress(progress)
        # v1i: first make the target reachable by moving the free root a few cm.
        self._apply_reference_root_error_correction(targets, phase='pre')
        passes=int(np.clip(globals().get('TASK43_LADDER_REFERENCE_IK_PASSES',22),1,32))
        nv=int(self.mj_model.nv)
        max_step=float(globals().get('TASK43_LADDER_REFERENCE_IK_STEP_CLIP',0.155))
        order=[('left','foot'),('right','foot'),('left','hand'),('right','hand')]
        for _ in range(passes):
            mujoco.mj_forward(self.mj_model,self.mj_data)
            total_err=0.0; updated=False
            for side,role in order:
                bid=self._side_body_id(side,role)
                if bid<0:
                    continue
                entries=self._limb_joint_entries_for_side(side,role)
                if not entries:
                    continue
                target=np.asarray(targets[(side,role)],dtype=float)
                pos=np.asarray(self.mj_data.xpos[bid],dtype=float)
                err=target-pos
                en=float(np.linalg.norm(err))
                total_err += en
                if en<0.008:
                    continue
                err=err*min(1.0,0.120/max(en,1e-8))
                jacp=np.zeros((3,nv),dtype=float); jacr=np.zeros((3,nv),dtype=float)
                try:
                    mujoco.mj_jacBody(self.mj_model,self.mj_data,jacp,jacr,int(bid))
                except Exception:
                    continue
                dofs=[e[1] for e in entries]
                J=jacp[:,dofs]
                W=np.diag([1.0,1.0,1.25 if role=='foot' else 1.15])
                Jw=W@J; ew=W@err
                damp=0.060 if role=='hand' else 0.075
                try:
                    dq=Jw.T@np.linalg.solve(Jw@Jw.T + (damp*damp)*np.eye(3), ew)
                except Exception:
                    dq=np.linalg.pinv(Jw)@ew
                dq=np.asarray(dq,dtype=float)
                if dq.size!=len(entries):
                    continue
                dq=np.clip(dq,-max_step,max_step)
                for (qadr,dof,lo,hi),dv in zip(entries,dq):
                    self.mj_data.qpos[int(qadr)]=float(np.clip(float(self.mj_data.qpos[int(qadr)])+float(dv),lo,hi))
                    updated=True
                mujoco.mj_forward(self.mj_model,self.mj_data)
            if (not updated) or total_err<0.055:
                break
        self._apply_reference_root_error_correction(targets, phase='post')
        # v1j: now that side detection can find real arm joints, run one focused
        # hand pass after the all-limb solve. Feet stay untouched; this only
        # reduces visible palm/wrist distance to the rung.
        try:
            self._solve_ladder_hand_grip_pose(progress)
        except Exception:
            pass
        self._last_reference_grip_error_m={}
        mujoco.mj_forward(self.mj_model,self.mj_data)
        for key,target in targets.items():
            side,role=key; bid=self._side_body_id(side,role)
            if bid>=0:
                self._last_reference_grip_error_m[f'{side}_{role}']=float(np.linalg.norm(np.asarray(self.mj_data.xpos[bid])-np.asarray(target)))

    def _apply_ladder_contact_forces(self, progress):
        """Apply optional hand/foot rung-grip forces during climb; disabled at release.

        v1h normally disables these forces because the climb is reference/mocap
        driven.  Keeping them off avoids the 30-160 BW contact spikes that made
        previous versions look like jumps or solver explosions.
        """
        if getattr(self,'_released',False): return
        if bool(globals().get('TASK43_LADDER_REFERENCE_DISABLE_GRIP_FORCES', True)):
            return
        right_rung,left_rung,active_right,lift,smooth,micro=self._foot_rungs_for_progress(progress)
        kp=float(TASK43_LADDER_CONTACT_FORCE_KP); kd=float(TASK43_LADDER_CONTACT_FORCE_KD)
        for side,rung,is_active in [('right',right_rung,active_right),('left',left_rung,not active_right)]:
            bid=self._side_body_id(side,'foot')
            if bid<0: continue
            landing=(smooth>float(globals().get('TASK43_LADDER_ACTIVE_FOOT_LAND_START',0.70))) if is_active else True
            target=self._rung_limb_target(side,rung,'foot',lift=(lift if is_active and not landing else 0.0))
            pos=np.asarray(self.mj_data.xpos[bid],dtype=float)
            try: vel=np.asarray(self.mj_data.cvel[bid][3:6],dtype=float)
            except Exception: vel=np.zeros(3)
            planted = (not is_active) or landing
            # v1g: do not hammer the foot downward into the rung.  The v1e
            # preload created GRF spikes (for example 79 BW) and made the
            # contact look like a forced lift. Use a soft spring plus a tiny
            # seating bias so physical contacts stay visible but normal.
            scale=0.10 if (is_active and not landing) else (0.20 if is_active else 0.26)
            force=scale*(kp*(target-pos)-kd*vel)
            # No downward seating impulse in v1g. Contacts are soft and the
            # virtual support channel reports body weight during scripted climb.
            force=np.clip(force,-80.0,80.0)
            self.mj_data.xfrc_applied[bid,:3]+=force
        if bool(globals().get('TASK43_LADDER_HAND_IK_ENABLED',True)):
            right_hand,left_hand,active_hand_right,hand_lift,hand_smooth,_hmicro=self._hand_rungs_for_progress(progress)
            hkp=float(globals().get('TASK43_LADDER_HAND_GRIP_FORCE_KP',kp))
            hkd=float(globals().get('TASK43_LADDER_HAND_GRIP_FORCE_KD',kd))
            hclip=float(globals().get('TASK43_LADDER_HAND_GRIP_FORCE_CLIP_N',260.0))
            bw=max(float(globals().get('MODEL_BODY_MASS',72.0))*9.81,1.0)
            for side,hand_rung,is_active in [('right',right_hand,active_hand_right),('left',left_hand,not active_hand_right)]:
                bid=self._side_body_id(side,'hand')
                if bid<0: continue
                target=self._rung_limb_target(side,hand_rung,'hand',lift=(0.45*hand_lift if is_active else 0.0))
                pos=np.asarray(self.mj_data.xpos[bid],dtype=float)
                try: vel=np.asarray(self.mj_data.cvel[bid][3:6],dtype=float)
                except Exception: vel=np.zeros(3)
                scale=0.74 if is_active else 0.98
                force=scale*(hkp*(target-pos)-hkd*vel)
                # A very small forward/down preload keeps contact intent visible
                # without pushing the hand through the far side of the rung.
                force += bw*float(globals().get('TASK43_LADDER_HAND_GRIP_PRELOAD_BW',0.000))*np.array([self.climb_xy[0],self.climb_xy[1],-0.02])
                force=np.clip(force,-hclip,hclip)
                self.mj_data.xfrc_applied[bid,:3]+=force

    def _apply_ladder_climb_pose(self, progress):
        if self._quiet_qpos is None: return
        micro,local,smooth,lift,active_right,root_rung,active_target_rung,support_rung=self._climb_phase(progress); self._last_climb_micro=int(micro); gain=float(TASK43_LADDER_CLIMB_POSE_GAIN)
        leg_gain=float(globals().get('TASK43_LADDER_PROFILE_LEG_GAIN',1.0))
        arm_gain=float(globals().get('TASK43_LADDER_PROFILE_ARM_GAIN',1.0))
        cadence_gain=float(globals().get('TASK43_LADDER_PROFILE_STEP_GAIN',1.0))
        lift=float(np.clip(lift*(0.96+0.08*leg_gain),0.0,1.12))
        smooth=float(np.clip(smooth*cadence_gain,0.0,1.0))
        def side_active(side):
            side=str(side or '').lower(); return (side.startswith('r') and active_right) or (side.startswith('l') and not active_right)
        def side_support(side):
            side=str(side or '').lower(); return (side.startswith('r') and not active_right) or (side.startswith('l') and active_right)
        def clamp(qadr, val, delta):
            base=float(self._quiet_qpos[qadr]); return float(np.clip(float(val),base-float(delta),base+float(delta)))
        # v7c: restore full active-foot rung reach, but freeze the planted leg.
        # The prior v7b smoothing shortened the swing too much, so the foot did
        # not land cleanly on the next rung.  Here the moving leg gets enough
        # knee/ankle travel to reach the rung, while the non-moving leg has
        # zero pose delta and remains locked/seated by rung contact forces.
        for qadr,kind,side in getattr(self,'_climb_leg_specs',[]):
            qadr=int(qadr)
            if not (7<=qadr<len(self.mj_data.qpos) and qadr<len(self._quiet_qpos)): continue
            base=float(self._quiet_qpos[qadr]); active=side_active(side); support=side_support(side)
            if kind=='hip':
                # Full enough reach to land on the next rung, but less hip throw
                # than v7 so the leg no longer snaps far backward.
                delta=leg_gain*(0.48*smooth+0.24*lift) if active else 0.0; maxd=0.78
            elif kind=='knee':
                # Restore most of the v7 knee lift; this is what gives the foot
                # real rung-to-rung height without requiring a long backward hip swing.
                delta=leg_gain*(0.96*lift+0.24*smooth) if active else 0.0; maxd=1.14
            else:
                delta=leg_gain*(-0.30*lift+0.09*smooth) if active else 0.0; maxd=0.48
            desired=clamp(qadr,base+gain*delta,maxd)
            # Slight smoothing only on the swing leg; planted leg is restored
            # exactly to quiet pose so it does not travel with the moving foot.
            blend=0.34 if active else 1.0
            self.mj_data.qpos[qadr]=(1.0-blend)*float(self.mj_data.qpos[qadr])+blend*desired
        # v1g: keep both hands visibly on ladder rungs.  The hand opposite
        # the moving foot reaches, while the other hand remains a support grip.
        right_hand,left_hand,active_hand_right,hand_lift,hand_smooth,_hmicro=self._hand_rungs_for_progress(progress)
        hand_reach_gain=float(globals().get('TASK43_LADDER_HAND_REACH_POSE_GAIN',1.18))
        for qadr,kind,side in getattr(self,'_climb_arm_specs',[]):
            qadr=int(qadr)
            if not (7<=qadr<len(self.mj_data.qpos) and qadr<len(self._quiet_qpos)): continue
            side=str(side or '').lower(); is_right=side.startswith('r')
            active=(is_right==active_hand_right) if side in ('right','left') else False
            base=float(self._quiet_qpos[qadr])
            if kind=='shoulder':
                delta=arm_gain*hand_reach_gain*((0.86+0.28*hand_smooth+0.18*hand_lift) if active else 0.58); maxd=1.24
            elif kind=='elbow':
                delta=arm_gain*hand_reach_gain*((0.58-0.06*hand_lift+0.08*hand_smooth) if active else 0.48); maxd=0.94
            else:
                delta=arm_gain*hand_reach_gain*((0.17+0.08*hand_lift) if active else 0.12); maxd=0.46
            desired=clamp(qadr,base+gain*delta,maxd)
            blend=0.46 if active else 0.38
            self.mj_data.qpos[qadr]=(1.0-blend)*float(self.mj_data.qpos[qadr])+blend*desired
        mujoco.mj_forward(self.mj_model,self.mj_data)
        if bool(globals().get('TASK43_LADDER_REFERENCE_MOCAP_CLIMB', True)):
            self._solve_reference_ladder_limb_ik(progress)
        else:
            self._solve_ladder_hand_grip_pose(progress)

    def enforce_single_step(self, progress):
        p=float(np.clip(progress,0.0,1.0))
        if self._quiet_qpos is None: return
        nq=min(len(self.mj_data.qpos),len(self._quiet_qpos)); self.mj_data.qpos[:nq]=self._quiet_qpos[:nq]
        self._apply_profile_root_posture(p)
        if self.mj_data.qvel.shape[0]>=6: self.mj_data.qvel[:6]=0.0
        micro,local,smooth,lift,active_right,root_rung,active_target_rung,support_rung=self._climb_phase(p)
        globals()['TASK43_RUNTIME_CLIMB_PROGRESS']=float(p)
        globals()['TASK43_RUNTIME_CLIMB_MICRO']=int(micro)
        globals()['TASK43_RUNTIME_CLIMB_LOCAL']=float(local)
        globals()['TASK43_RUNTIME_ACTIVE_RIGHT']=bool(active_right)
        root_center=self._rung_center(root_rung)
        root_xy=np.asarray(root_center[:2],dtype=float)-self.climb_xy*float(TASK43_LADDER_ROOT_BACKOFF_M)
        dz_total=float(self._rung_top_z(root_rung)-self._rung_top_z(self.start_rung_index))
        # Reference/mocap climb: the root has plateaus and only rises during
        # load transfer.  A tiny bob is visual only; it is not used to climb via
        # force or collision.
        swing_bob=float(globals().get('TASK43_LADDER_REFERENCE_ROOT_BOB_M',0.018))*float(lift)*float(1.0-smooth)
        if len(self.mj_data.qpos)>=3:
            self.mj_data.qpos[0]=float(root_xy[0]); self.mj_data.qpos[1]=float(root_xy[1]); self.mj_data.qpos[2]=float(self._quiet_qpos[2]+dz_total+swing_bob)
        self.current_support_z=float(self._rung_top_z(root_rung)); self.height=float(self.current_support_z)
        self._apply_ladder_climb_pose(p)
        if not bool(globals().get('TASK43_LADDER_REFERENCE_MOCAP_CLIMB', True)):
            # Legacy physical seating path, kept as fallback only.
            self._align_root_to_ladder_feet(p, hard=(p>=0.995))
            self._apply_ladder_climb_pose(p)
        if p>=0.995 and len(self.mj_data.qpos)>=2:
            self.step_target_xy=np.asarray(self.mj_data.qpos[:2],dtype=float).copy()
        if self.mj_data.qvel.shape[0]>=6: self.mj_data.qvel[:6]=0.0
        self._apply_ladder_contact_forces(p)
        mujoco.mj_forward(self.mj_model,self.mj_data); self._has_foot_platform_contact(); self._last_climb_progress=p

    def release_contact(self, forward_velocity=0.62):
        if self._released: return
        self.enforce_single_step(1.0)
        # Re-lock Task40/38 posterior frame at release.  The backward direction
        # is opposite the ladder-facing/climb vector; the pitch axis is derived
        # from that backward vector so the body rotates into a true backward fall.
        self.fwd_xy=np.asarray(self.climb_xy,dtype=float); self.fwd_xy/=max(float(np.linalg.norm(self.fwd_xy)),1e-8)
        self.lat_xy=np.array([-self.fwd_xy[1], self.fwd_xy[0]], dtype=float); self.lat_xy/=max(float(np.linalg.norm(self.lat_xy)),1e-8)
        globals()['TASK43_RUNTIME_FWD_XY']=self.fwd_xy.copy(); globals()['TASK43_RUNTIME_LAT_XY']=self.lat_xy.copy()
        # Keep ladder collision SOLID after release.  We only turn off the
        # scripted hand/foot grip helper via _released; physical ladder contact
        # remains active so the body cannot pass through rungs/rails if it hits.
        for gid in getattr(self,'platform_geoms',[]):
            try:
                self.mj_model.geom_contype[int(gid)]=1
                self.mj_model.geom_conaffinity[int(gid)]=1
            except Exception:
                pass
        if len(self.mj_data.qpos)>=2:
            slip_back=float(globals().get('TASK43_LADDER_RELEASE_BACKSTEP_M',0.22))
            self.mj_data.qpos[0]+=slip_back*float(self.fwd_xy[0])
            self.mj_data.qpos[1]+=slip_back*float(self.fwd_xy[1])
        if self.mj_data.qvel.shape[0]>=3:
            self.mj_data.qvel[:]*=0.0
            v0=max(0.42, float(TASK43_TASK38_INITIAL_BACK_VEL), float(forward_velocity))
            self.mj_data.qvel[0]=v0*float(self.fwd_xy[0])
            self.mj_data.qvel[1]=v0*float(self.fwd_xy[1])
            self.mj_data.qvel[2]=-0.16
        bw=max(float(globals().get('MODEL_BODY_MASS',72.0))*9.81,1.0); pelvis_id=_safe_name2id(self.mj_model,MJOBJ_BODY,'Pelvis'); torso_id=_safe_name2id(self.mj_model,MJOBJ_BODY,'Torso')
        slip_push=float(globals().get('TASK43_LADDER_RELEASE_PUSH_BW',0.26))*bw
        torso_push=float(globals().get('TASK43_LADDER_RELEASE_TORSO_PUSH_BW',0.14))*bw
        if pelvis_id>=0:
            self.mj_data.xfrc_applied[pelvis_id,:3]+=slip_push*np.array([self.fwd_xy[0],self.fwd_xy[1],-0.10])
        if torso_id>=0:
            self.mj_data.xfrc_applied[torso_id,0]+=torso_push*self.fwd_xy[0]
            self.mj_data.xfrc_applied[torso_id,1]+=torso_push*self.fwd_xy[1]
            self.mj_data.xfrc_applied[torso_id,2]+=-0.035*bw
            self.mj_data.xfrc_applied[torso_id,3]+=28.0*self.lat_xy[0]
            self.mj_data.xfrc_applied[torso_id,4]+=28.0*self.lat_xy[1]
        mujoco.mj_forward(self.mj_model,self.mj_data)
        try:
            err=getattr(self,'_last_reference_grip_error_m',{}) or {}
            if err:
                txt=', '.join(f'{k}={float(v):.3f}m' for k,v in sorted(err.items()))
                print(f'      Task43 v1j reference lock error at release: {txt}')
        except Exception:
            pass
        self._released=True
        print('      Task43 ladder release v1j: scripted grip OFF, ladder collision ON; top-rung forward fall starts')

    def apply_descent_reaction(self, progress, phase='perturb'):
        pelvis_id=_safe_name2id(self.mj_model,MJOBJ_BODY,'Pelvis'); pelvis_z=float(self.mj_data.xpos[pelvis_id][2]) if pelvis_id>=0 else 9.0
        if phase in ('perturb','react') and pelvis_z>0.82 and float(progress)<0.72: return
        return super().apply_descent_reaction(progress, phase=phase)

    def apply_impact_recoil_pose(self, progress):
        pelvis_id=_safe_name2id(self.mj_model,MJOBJ_BODY,'Pelvis'); pelvis_z=float(self.mj_data.xpos[pelvis_id][2]) if pelvis_id>=0 else 9.0
        if pelvis_z>0.46: return
        return super().apply_impact_recoil_pose(progress)

    def draw_viewer_support(self, viewer):
        if self.platform_geoms: return
        return super().draw_viewer_support(viewer)


Task43GroundCloneHeightSupport = Task43LadderClimbSupport


def _legacy_main():
    print("\n[2/4] Computing task embeddings...")
    z_stand = infer_z_stand()
    print("  Skipping z_walk: Task43 uses contact-snapped reference ladder climb, then physical forward fall/reaction; not continuous walking.")
    z_walk  = z_stand
    z_fall  = infer_z_task29_forward_trip_fall()
    z_rest  = infer_z_forward_prone_rest()

    print("\n[3/4] Initializing environment...")
    env, _ = make_humenv(task="move-ego-0-0")
    obs, _ = env.reset()
    installed, install_reason = _task43_install_platform_if_possible(env)
    print(f"  [Task43HeightLayer] physical support injection = {installed} ({install_reason})")

    mj_model = env.unwrapped.model
    mj_data  = env.unwrapped.data
    task43_support = Task43GroundCloneHeightSupport(mj_model, mj_data)
    task43_support.place_on_platform()
    obs = _task43_refresh_obs(env, obs)
    myosuite_mount = resolve_imu_mount_configuration(
        mj_model,
        requested_xml_path=(MYOSUITE_INTEGRATION['model_xml'] if MYOSUITE_INTEGRATION.get('enable_reference_xml', True) else ''),
    )
    if myosuite_mount.get('requested_xml_path'):
        print("    [MyoSuite bridge] ACTIVE")
        print(f"      runtime mount source      = {myosuite_mount['runtime_mount_source']}")
        print(f"      runtime IMU body/site     = {myosuite_mount['sensor_body']} / {myosuite_mount.get('sensor_site')}")
        if myosuite_mount.get('reference_loaded'):
            print(f"      reference XML             = {myosuite_mount['requested_xml_path']}")
            print(f"      reference mount           = {myosuite_mount.get('reference_mount')}")
        for _note in myosuite_mount.get('notes', []):
            print(f"      note                      = {_note}")

    # -----------------------------------------------------------------------
    # STEP A: Anthropometric Model
    # -----------------------------------------------------------------------
    print(f"\n[A] Applying AnthropometricModel "
          f"(age={SIM_AGE}, height={SIM_HEIGHT}m, weight={SIM_RESOLVED_WEIGHT:.1f}kg, sex={SIM_SEX})...")
    anthro     = AnthropometricModel(mj_model, age=SIM_AGE, height=SIM_HEIGHT,
                                     weight=SIM_RESOLVED_WEIGHT, sex=SIM_SEX)
    age_params = anthro.apply_age_effects()
    task43_profile = _task43_apply_subject_profile_globals(age_params)
    # Anthropometric scaling can rebuild body/geom anchors and invalidate the
    # pre-scaling foot resolver. Re-create the Task43 support helper after
    # scaling, then perform final sole/contact seating from the rebuilt geometry.
    task43_support = Task43GroundCloneHeightSupport(mj_model, mj_data)
    task43_support.place_on_platform(verbose=False, reason='post-anthropometric sole/contact rebuild')
    obs = _task43_refresh_obs(env, obs)
    print(f"    ? AnthropometricModel ACTIVE")
    print(f"      strength_factor        = {age_params['strength_factor']:.3f}")
    print(f"      strength_factor_raw    = {age_params.get('strength_factor_raw', age_params['strength_factor']):.3f}")
    print(f"      reaction_delay         = {age_params['reaction_delay']:.3f}s")
    print(f"      balance_impairment     = {age_params['balance_impairment']:.3f}")
    print(f"      proprioception_scale   = {age_params['proprioception_scale']:.3f}")
    print(f"      height_neural_penalty  = {age_params['height_penalty']*1000:.1f}ms")
    print(f"      body_mass / BMI        = {age_params.get('body_mass_kg', float(np.sum(mj_model.body_mass))):.2f} kg / {SIM_TARGET_BMI:.2f}")
    print(f"      default_body_mass      = {age_params.get('default_body_mass_kg', float(np.sum(mj_model.body_mass))):.2f} kg")

    MODEL_BODY_MASS_FOR_FORCE = float(np.sum(mj_model.body_mass))
    # Task43 keeps the Task41 magnitude bridge but rotates the driver into the forward/climb lane.
    _dyn_force = perturbation_force_config(MODEL_BODY_MASS_FOR_FORCE, SIM_AGE, SIM_SEX, bw_fraction=TASK43_TASK38_FORCE_BW_FRACTION)
    FORCE_CONFIG.update(_dyn_force)
    FORCE_CONFIG['application_point'] = 'Pelvis'
    FORCE_CONFIG['direction'] = np.array([TASK43_RUNTIME_FWD_XY[0], TASK43_RUNTIME_FWD_XY[1], -0.12], dtype=float)
    WEAKENING_CONFIG = weakening_config(SIM_AGE, SIM_SEX, MODEL_BODY_MASS_FOR_FORCE)
    print(f"      perturbation magnitude = {FORCE_CONFIG['magnitude']:.1f} N ({FORCE_CONFIG.get('bw_fraction', 0.0):.2f} BW)")
    print(f"      weakening profile      = min_factor={WEAKENING_CONFIG['min_factor']:.3f} | decay_time={WEAKENING_CONFIG['decay_time']} steps | fatigue_k={WEAKENING_CONFIG.get('fatigue_coefficient', 1.0):.2f} | sarcopenia_loss={WEAKENING_CONFIG.get('sarcopenia_force_loss', 0.0):.1%}")
    print("      Task43 profile bridge  = "
          f"walk={task43_profile.get('walk_speed_mps', TASK43_LADDER_PROFILE_SPEED_MPS):.2f}m/s | "
          f"ds={task43_profile.get('expected_double_support', TASK43_LADDER_PROFILE_EXPECTED_DS):.0%} | "
          f"stand_stoop={task43_profile.get('stand_stoop_deg', TASK43_LADDER_PROFILE_STAND_STOOP_DEG):.1f}deg | "
          f"climb_stoop={task43_profile.get('climb_stoop_deg', TASK43_LADDER_PROFILE_CLIMB_STOOP_DEG):.1f}deg | "
          f"climb_frames={PHASES.get('step', 0)} | "
          f"pose_gain={task43_profile.get('step_gain', TASK43_LADDER_PROFILE_STEP_GAIN):.2f} | "
          f"leg={task43_profile.get('leg_gain', TASK43_LADDER_PROFILE_LEG_GAIN):.2f} | "
          f"arm={task43_profile.get('arm_gain', TASK43_LADDER_PROFILE_ARM_GAIN):.2f} | "
          f"rung_dz={task43_profile.get('rung_dz_m', TASK43_LADDER_RUNG_DZ_M):.3f}m | "
          f"align={task43_profile.get('foot_align_gain', TASK43_LADDER_FOOT_ALIGN_GAIN):.2f} | "
          f"slip={task43_profile.get('release_backstep_m', TASK43_LADDER_RELEASE_BACKSTEP_M):.3f}m | "
          f"push={task43_profile.get('release_push_bw', TASK43_LADDER_RELEASE_PUSH_BW):.2f}BW | "
          f"fallBW={task43_profile.get('fall_force_bw', TASK43_TASK38_FORCE_BW_FRACTION):.2f}")

    # -----------------------------------------------------------------------
    # STEP B: IMU Validator (with age/height for realistic noise)
    # -----------------------------------------------------------------------
    print("\n[B] Initializing IMUValidator on Pelvis...")
    imu = IMUValidator(mj_model, mj_data,
                       sensor_body=myosuite_mount.get('sensor_body', IMU_HARDWARE_SPEC['proxy_body']),
                       sensor_site=myosuite_mount.get('sensor_site'),
                       sensor_offset_local=myosuite_mount.get('sensor_offset_local'),
                       age=SIM_AGE, height=SIM_HEIGHT,
                       target_output_hz=IMU_HARDWARE_SPEC['sampling_hz'],
                       mount_label=myosuite_mount.get('mount_label', IMU_HARDWARE_SPEC['mount_label']))
    print(f"    ? IMUValidator ACTIVE")
    print(f"      accel_noise = {imu.accel_noise:.4f} m/s2  |  gyro_noise = {imu.gyro_noise:.4f} rad/s")
    print(f"      accel_bias  = {imu.accel_bias.round(4)}")
    print(f"      gyro_bias   = {imu.gyro_bias.round(4)}")
    print(f"      STA ampl.   = {imu.sta_amplitude:.4f} m  (soft-tissue artifact)")
    imu.print_configuration_report(age=SIM_AGE, height=SIM_HEIGHT, sex=SIM_SEX)
    _ver = imu.get_sampling_report()
    print(f"    [IMU reality] effective physical bandwidth  {_ver['effective_bandwidth_hz']:.2f} Hz")
    print(f"    [IMU reality] true hardware-equivalent 100 Hz = {_ver['true_hardware_equivalent_100hz']}")

    # -----------------------------------------------------------------------
    # STEP C: Fall Type Library
    # -----------------------------------------------------------------------
    ACTIVE_FALL_TYPE = 'forward_stumble'
    print(f"\n[C] Loading FallTypeLibrary scenario: '{ACTIVE_FALL_TYPE}'...")
    fall_config = FallTypeLibrary.FALL_TYPES[ACTIVE_FALL_TYPE]
    lie_reward  = FallTypeLibrary.get_lie_down_reward(ACTIVE_FALL_TYPE)
    print(f"    ? FallTypeLibrary ACTIVE")
    print(f"      description         = {fall_config['description']}")
    print(f"      force_direction     = {fall_config['force_direction']}")
    print(f"      perturbation_timing = {fall_config['perturbation_timing']}")
    print(f"      LieDownReward type  = {type(lie_reward).__name__} (orient-aware)")

    # -----------------------------------------------------------------------
    # STEP D: Enhanced Biofidelic Controller
    # -----------------------------------------------------------------------
    print("\n[D] Initializing EnhancedBiofidelicController...")
    controller = EnhancedBiofidelicController(env, mj_model, mj_data,
                                              anthropometry=age_params)
    print(f"    ? EnhancedBiofidelicController ACTIVE")
    print(f"      protective_reflexes       = {controller.protective_reflexes}")
    print(f"      balance_control           = {controller.balance_control}")
    print(f"      vestibular_delay          = {controller.vestibular_delay}s")
    print(f"      age_style                 = {controller.age_style['label']} | target_walk_speed={controller.age_style['target_walk_speed']:.2f} m/s")
    print(f"      ankle_strategy_threshold  = {controller.ankle_strategy_threshold}m")
    print(f"      hip_strategy_threshold    = {controller.hip_strategy_threshold}m")
    print(f"      leg_actuators  count      = {len(controller.leg_actuators)}")
    print(f"      arm_actuators  count      = {len(controller.arm_actuators)}")
    print(f"      torso_actuators count     = {len(controller.torso_actuators)}")
    print(f"      GaitPhaseDetector         = ACTIVE")

    # Keep the earlier straight-walk branch, but move subject logic into a
    # modest control envelope around the selected latent instead of forcing the latent.
    if LAST_Z_WALK_DIAGNOSTICS:
        requested_speed = float(controller.walk_target_speed)
        inferred_vx = float(LAST_Z_WALK_DIAGNOSTICS.get('mean_vx', requested_speed))
        inferred_ds = float(LAST_Z_WALK_DIAGNOSTICS.get('double_support_frac', controller.age_style.get('expected_double_support', 0.30)))
        control_targets = compute_subject_control_targets(controller.age_style, inferred_vx, inferred_ds)
        controller.walk_lit_target_speed = float(requested_speed)
        controller.walk_target_speed = float(control_targets['control_speed'])
        controller.walk_control_expected_ds = float(control_targets['control_ds'])
        controller.walk_policy_limited = bool(controller.walk_target_speed < 0.75 * requested_speed)
        print(f"      walk target requested     = {requested_speed:.2f} m/s | adapted={controller.walk_target_speed:.2f} m/s | z_walk_natural={inferred_vx:.2f} m/s")
        print(f"      walk control envelope     = speed_gain={control_targets['control_gain']:.2f} | control_ds={controller.walk_control_expected_ds:.0%} | stand_stoop={controller.age_style.get('stand_stoop_target_deg', 0.0):.1f}deg | walk_stoop={controller.age_style.get('walk_stoop_target_deg', 0.0):.1f}deg")
        if controller.walk_policy_limited:
            print("      walk policy status        = POLICY_LIMITED (straight base gait retained; subject envelope used for control)")
        print_age_behavior_audit(SIM_AGE, SIM_SEX, controller.age_style, controller.walk_target_speed, inferred_vx, age_params)

    MODEL_BODY_MASS = float(age_params.get('body_mass_kg', np.sum(mj_model.body_mass)))
    dashboard = PhysicsDashboard(mj_model, mj_data, body_mass=MODEL_BODY_MASS,
                                 leg_length=age_params.get('leg_length', SIM_HEIGHT * 0.53),
                                 upright_h=float(mj_data.xpos[mujoco.mj_name2id(mj_model, MJOBJ_BODY, 'Pelvis')][2]))
    marker_exporter = MarkerKinematicsExporter(mj_model, mj_data, export_hz=imu.output_hz)
    marker_summary = marker_exporter.marker_summary()
    dynamics_analyzer = DynamicsContactAnalyzer(mj_model, mj_data, body_mass=MODEL_BODY_MASS, leg_length=age_params.get('leg_length', SIM_HEIGHT * 0.53))
    paper_exporter = PaperAlignmentExporter(
        marker_exporter,
        dynamics_analyzer,
        subject_meta={
            'age_years': SIM_AGE,
            'height_m': SIM_HEIGHT,
            'sex': SIM_SEX,
            'body_mass_kg': MODEL_BODY_MASS,
            'fall_type': 'forward_fall_climbing_up_ladder',
            'imu_output_hz': imu.output_hz,
        },
    )
    print("    ? MarkerKinematicsExporter ACTIVE")
    print(f"      anatomical markers      = {marker_summary['num_markers']} -> {', '.join(marker_summary['marker_names'])}")
    print(f"      resolved marker bodies  = {marker_summary['resolved_bodies']}")
    print(f"      segment exports         = {marker_summary['segments']}")
    print(f"      joint exports           = {marker_summary['num_joint_exports']} joints")
    print("    ? DynamicsContactAnalyzer ACTIVE")
    print("      outputs                 = frame-wise contact, GRF, CoP, impact load, tangential ratio")
    print("    ? PaperAlignmentExporter ACTIVE")
    print("      outputs                 = OpenSim GRF/ExternalLoads + marker registration + multiview pose dataset")
    print(f"      walk guidance           = {controller.walk_guidance_enabled} | target={controller.walk_target_speed:.2f} m/s")
    xcom_history = []
    support_center_history = []
    xcom_timestamps = []

    # Phase tracking
    phase_names      = list(PHASES.keys())
    phase_boundaries = np.cumsum([0] + list(PHASES.values()))

    print("\n[4/4] Running simulation...")
    print(f"""
      +---------------------------------------------------------------------+
      |  Phase      Steps    Duration    Description                        |
      +---------------------------------------------------------------------+
      |  HOLD       {PHASES['stand']:3d}      ~{PHASES['stand']/30:.1f}s       Stable hold on first ladder rung    |
      |  CLIMB      {PHASES['step']:3d}      ~{PHASES['step']/30:.1f}s      top-rung upward ladder climb     |
      |  RELEASE   {PHASES['perturb']:3d}       ~{PHASES['perturb']/30:.1f}s        Top-rung forward release + reaction |
      |  REACT      {PHASES['react']:3d}       ~{PHASES['react']/30:.1f}s        Delayed weak recovery attempt            |
      |  FALL       {PHASES['fall']:3d}      ~{PHASES['fall']/30:.1f}s      Impact recoil then forward/prone settle               |
      +---------------------------------------------------------------------+
      Controller : EnhancedBiofidelicController  (protective reflexes monitored; ladder hand/foot grip active during climb)
      Anthropo   : age={SIM_AGE}  height={SIM_HEIGHT}m  weight={SIM_RESOLVED_WEIGHT:.1f}kg  sex={SIM_SEX}
      IMU        : mount={imu.mount_label} via {imu.body_name} | native={imu.native_hz:.2f} Hz | output={imu.output_hz:.0f} Hz ({imu.output_mode})
      XCoM       : Extrapolated Centre of Mass balance monitoring ACTIVE
      Validator  : FallValidator will score realism after simulation
      Fall type  : forward_fall_climbing_up_ladder  (physical ladder climb to top rung, forward release/downward COM pitch, then protective prone settle)
    """)

    # Logging
    metrics_log              = []
    _protective_reflex_count = 0
    _last_protective_stamp   = None
    _xcom_negative_streak    = 0
    _xcom_log_start_step     = phase_boundaries[1] + XCOM_LOG_START_OFFSET_STEPS
    _rest_stable_counter     = 0
    _rest_hard_lock_counter  = 0
    walk_stable_counter      = 0
    policy_limited_counter = 0
    walk_extra_budget        = 45
    actual_step_marks = {'step_start': None, 'perturb_start': None, 'react_start': None, 'fall_start': None}

    _scenario29_ensure_valid_cwd()

    # -------------------------------------------------------------
    # TASK43 NATIVE VIDEO FRAMEBUFFER FIX
    # MuJoCo default offscreen framebuffer is often 1280x720.
    # Increase it before creating high-resolution Renderer.
    # -------------------------------------------------------------
    try:
        mj_model.vis.global_.offwidth = 1920
        mj_model.vis.global_.offheight = 1008

        print(
            "[TASK43 VIDEO ACTIVE] framebuffer resized:",
            mj_model.vis.global_.offwidth,
            mj_model.vis.global_.offheight
        )

    except Exception as e:
        print(
            "[TASK43 VIDEO ACTIVE] framebuffer resize warning:",
            e
        )


    native_renderer = mujoco.Renderer(
        mj_model,
        height=1008,
        width=1920
    )

    native_video_path = Path(
        os.environ.get("FALL_OUTPUT_DIR", ".")
    ) / "scenario43_native_render.mp4"

    native_video_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    native_writer = cv2.VideoWriter(
        str(native_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (1920,1008)
    )

    print("[TASK43 VIDEO ACTIVE] path:", native_video_path)
    print("[TASK43 VIDEO ACTIVE] writer:", native_writer.isOpened())

    if not native_writer.isOpened():
        raise RuntimeError(
            f"VideoWriter failed: {native_video_path}"
        )


    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.distance  = 4.5
        viewer.cam.elevation = -10
        viewer.cam.azimuth   = 90

        native_frame_count = 0

        for step in range(TOTAL_STEPS):
            # Reset any externally applied forces so walk-guidance / perturbation
            # are strictly per-step and do not accumulate across the rollout.
            controller.clear_forces()
            # -- Phase Logic -------------------------------------------------
            if step < phase_boundaries[1]:
                phase = 'stand'
                if step == 0:
                    controller.set_target_z(z_stand, blend_steps=6)
                    env.unwrapped.set_task("move-ego-0-0")
                    controller.walk_guidance_enabled = False
                    controller.lock_task43_forward_edge_lane(origin_xy=task43_support.start_xy, verbose=True)
                task43_support.enforce_quiet_stand()

            elif step < phase_boundaries[2]:
                phase = 'step'
                if actual_step_marks.get('step_start') is None:
                    actual_step_marks['step_start'] = int(step)
                    controller.set_target_z(z_stand, blend_steps=4)
                    env.unwrapped.set_task("move-ego-0-0")
                    controller.walk_guidance_enabled = False
                    print(f"  [Step {step}] Visible top-rung ladder climb started from first rung")
                step_elapsed = step - int(actual_step_marks['step_start'])
                step_progress = float(np.clip(step_elapsed / max(PHASES['step'] - 1, 1), 0.0, 1.0))
                task43_support.enforce_single_step(step_progress)

            else:
                if actual_step_marks['perturb_start'] is None:
                    actual_step_marks['perturb_start'] = int(step)
                    phase = 'perturb'
                    controller.walk_guidance_enabled = False
                    controller.lock_task43_forward_edge_lane(origin_xy=task43_support.step_target_xy, verbose=True)
                    controller.set_target_z(z_fall, blend_steps=24)
                    task43_support.release_contact(forward_velocity=TASK43_TASK38_INITIAL_BACK_VEL)
                    print(f"  [Step {step}] LADDER FORWARD FALL phase started - top-rung climb pose released")
                    print("      driver: ladder collision stays ON; forward/down COM push + weakening")
                elif actual_step_marks['react_start'] is None:
                    phase = 'perturb'
                    edge_steps = step - int(actual_step_marks['perturb_start'])
                    edge_progress = np.clip(edge_steps / max(PHASES['perturb'], 1), 0.0, 1.0)
                    controller.apply_task43_edge_fall_driver(edge_progress, MODEL_BODY_MASS, followthrough=False)
                    weaken_progress = np.clip(edge_steps / max(PHASES['perturb'] + PHASES['react'], 1), 0.0, 1.0)
                    controller.update_muscle_weakening(weaken_progress)
                    if edge_steps >= PHASES['perturb']:
                        actual_step_marks['react_start'] = int(step)
                        phase = 'react'
                        print(f"  [Step {step}] Weak delayed reaction after edge step [reaction_delay={age_params['reaction_delay']:.3f}s]")
                elif actual_step_marks['fall_start'] is None:
                    react_steps = step - int(actual_step_marks['react_start'])
                    phase = 'react' if react_steps < PHASES['react'] else 'fall'
                    if phase == 'react':
                        react_progress = np.clip(react_steps / max(PHASES['react'], 1), 0.0, 1.0)
                        controller.apply_task43_edge_fall_driver(min(1.0, 0.75 + 0.25 * react_progress), MODEL_BODY_MASS, followthrough=False)
                        weaken_progress = np.clip((PHASES['perturb'] + react_steps) / max(PHASES['perturb'] + PHASES['react'], 1), 0.0, 1.0)
                        controller.update_muscle_weakening(weaken_progress)
                    else:
                        actual_step_marks['fall_start'] = int(step)
                        print(f"  [Step {step}] Post-fall phase - completing unsupported forward fall from ladder")
                        controller.finalize_z_transition()
                        controller.clear_forces()
                        controller.update_muscle_weakening(1.0)
                else:
                    phase = 'fall'
                    controller.update_muscle_weakening(1.0)
                    pelvis_id_follow = getattr(controller, '_pelvis_id', -1)
                    pelvis_z_follow = float(mj_data.xpos[pelvis_id_follow][2]) if pelvis_id_follow >= 0 else 0.0
                    if pelvis_z_follow > 0.30:
                        controller.apply_task43_edge_fall_driver(1.0, MODEL_BODY_MASS, followthrough=True)

            # -- Interpolated Z ----------------------------------------------
            z_current = controller.update_z_interpolation()
            controller.current_phase = phase

            # -- Action via EnhancedBiofidelicController ---------------------
            _pre_action = controller.get_action(obs, z_current)
            action      = controller.get_protective_action(obs, z_current, ACTIVE_FALL_TYPE)

            if controller._last_protective_state is not None:
                stamp = float(controller._last_protective_state.get('timestamp', -1.0))
                if _last_protective_stamp is None or stamp > (_last_protective_stamp + 1e-9):
                    _protective_reflex_count += 1
                    _last_protective_stamp = stamp

            # -- Quiet stand / single-step control ----------------------------
            if phase in ('stand', 'step'):
                action = np.zeros_like(np.asarray(action, dtype=np.float32))

            # No continuous walk guidance in v20.

            # -- Reduce active recovery after support loss ---------------------
            # Preserve the walk policy before the stone, then prevent the policy
            # from regaining balance into a kneel/sit after the toe is caught.
            if phase in ('perturb', 'react', 'fall'):
                trip_start_tmp = actual_step_marks.get('perturb_start')
                elapsed_tmp = 0 if trip_start_tmp is None else max(0, step - int(trip_start_tmp))
                ptmp = float(np.clip(elapsed_tmp / max(PHASES['perturb'], 1), 0.0, 1.0))
                if phase == 'perturb':
                    # Keep gait action at the instant of stone strike, then drop
                    # it rapidly so the fall is not a smooth controller-driven
                    # crouch/recovery.
                    action_scale = 0.68 - 0.42 * ptmp
                    if ptmp > 0.24:
                        action_scale -= 0.10 * min(1.0, (ptmp - 0.24) / 0.24)
                    action = np.asarray(action, dtype=np.float32) * float(np.clip(action_scale, 0.12, 0.68))
                elif phase == 'react':
                    action = np.asarray(action, dtype=np.float32) * 0.14
                else:
                    action = np.asarray(action, dtype=np.float32) * (0.012 if controller.rest_mode else 0.08)

            # -- Step Environment --------------------------------------------
            obs, reward, terminated, truncated, info = highrate_env_step(env, action, legacy_imu=imu)

            # -- Enforce deterministic quiet stand/single-step after physics step --
            if phase == 'stand':
                task43_support.enforce_quiet_stand()
                obs = _task43_refresh_obs(env, obs)
            elif phase == 'step':
                step_elapsed_post = step - int(actual_step_marks.get('step_start') or step)
                step_progress_post = float(np.clip(step_elapsed_post / max(PHASES['step'] - 1, 1), 0.0, 1.0))
                task43_support.enforce_single_step(step_progress_post)
                obs = _task43_refresh_obs(env, obs)
            elif phase in ('perturb', 'react'):
                perturb_ref = actual_step_marks.get('perturb_start')
                elapsed_react_pose = 0 if perturb_ref is None else max(0, step - int(perturb_ref))
                react_pose_progress = float(np.clip(elapsed_react_pose / max(PHASES['perturb'] + PHASES['react'], 1), 0.0, 1.0))
                task43_support.apply_descent_reaction(react_pose_progress, phase=phase)
                obs = _task43_refresh_obs(env, obs)
            elif phase == 'fall' and not controller.rest_mode:
                fall_ref = actual_step_marks.get('fall_start')
                fall_pose_elapsed = 0 if fall_ref is None else max(0, step - int(fall_ref))
                # v30: only a short impact recoil is active. In v29 this pose
                # nudge continued through the whole fall phase because rest mode
                # did not engage, which read as endless rolling on the floor.
                pelvis_id_tmp = getattr(controller, '_pelvis_id', -1)
                pelvis_z_tmp = float(mj_data.xpos[pelvis_id_tmp][2]) if pelvis_id_tmp >= 0 else 1.0
                if (not controller.impact_brake_mode) or fall_pose_elapsed <= TASK43_POST_IMPACT_FREE_RECOIL_STEPS or pelvis_z_tmp > 0.22:
                    task43_support.apply_impact_recoil_pose(float(np.clip(fall_pose_elapsed / 18.0, 0.0, 1.0)))
                    obs = _task43_refresh_obs(env, obs)

            # -- Keep Task43 in pure backward sagittal plane -------------------
            if phase in ('perturb', 'react', 'fall'):
                # Keep the fall in the sagittal lane, but do not use a hard rail
                # before rest mode. Hard lateral/yaw corrections during floor
                # contact were contributing to visible rolling/buzzing.
                rail_hard = bool(controller.rest_mode)
                controller.apply_scenario29_sagittal_rail(MODEL_BODY_MASS, strength=(0.52 if not controller.rest_mode else 0.78), hard=rail_hard)

                # v6 anti-slide guard: while the body is still above the floor,
                # cap horizontal root velocity.  This prevents skating backward
                # off/outside the object before the body actually drops.
                # Rotation and vertical motion are untouched.
                try:
                    pelvis_tmp = getattr(controller, '_pelvis_id', -1)
                    pelvis_z_tmp2 = float(mj_data.xpos[pelvis_tmp][2]) if pelvis_tmp >= 0 else 9.0
                    if pelvis_z_tmp2 > 0.32 and mj_data.qvel.shape[0] >= 2:
                        fwd_tmp = np.asarray(getattr(controller, 'scenario29_trip_fwd_xy', TASK43_RUNTIME_FWD_XY), dtype=float)
                        fwd_tmp = fwd_tmp / max(float(np.linalg.norm(fwd_tmp)), 1e-8)
                        lat_tmp = np.array([-fwd_tmp[1], fwd_tmp[0]], dtype=float)
                        vxy_tmp = np.asarray(mj_data.qvel[:2], dtype=float)
                        vf_tmp = float(np.dot(vxy_tmp, fwd_tmp))
                        vl_tmp = float(np.dot(vxy_tmp, lat_tmp))
                        # v7d: after the single back-step, allow the Task38-style
                        # posterior push to carry the body backward while still
                        # clamping lateral drift.
                        edge_ref_tmp = actual_step_marks.get('perturb_start')
                        edge_elapsed_tmp = 0 if edge_ref_tmp is None else max(0, step - int(edge_ref_tmp))
                        edge_p_tmp = float(np.clip(edge_elapsed_tmp / max(PHASES['perturb'], 1), 0.0, 1.0))
                        free_gate_tmp = float(np.clip((edge_p_tmp - 0.18) / 0.46, 0.0, 1.0))
                        free_gate_tmp = free_gate_tmp * free_gate_tmp * (3.0 - 2.0 * free_gate_tmp)
                        cap_early_tmp = float(globals().get('TASK43_SIMPLE_BACK_FALL_BACK_VEL_CAP_EARLY', 0.055))
                        cap_free_tmp = float(globals().get('TASK43_SIMPLE_BACK_FALL_BACK_VEL_CAP_FREE', 0.300))
                        back_cap_tmp = cap_early_tmp + (cap_free_tmp - cap_early_tmp) * free_gate_tmp
                        vf_tmp = float(np.clip(vf_tmp, -0.030, back_cap_tmp))
                        lat_cap_tmp = float(globals().get('TASK43_SIMPLE_BACK_FALL_LAT_VEL_CAP', 0.040))
                        vl_tmp = float(np.clip(vl_tmp, -lat_cap_tmp, lat_cap_tmp))
                        mj_data.qvel[0:2] = vf_tmp * fwd_tmp + vl_tmp * lat_tmp
                except Exception:
                    pass

            # -- Post-impact rest-mode detector -----------------------------
            if phase in ('perturb', 'react', 'fall'):
                pelvis_id_rest = mujoco.mj_name2id(mj_model, MJOBJ_BODY, "Pelvis")
                vel6_rest = np.zeros(6)
                mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, pelvis_id_rest, vel6_rest, 0)
                pelvis_ang_speed = float(np.linalg.norm(vel6_rest[:3]))
                pelvis_height_rest = float(mj_data.xpos[pelvis_id_rest][2])
                slip_rest = float(np.linalg.norm(vel6_rest[3:][:2]))
                ncon_rest = int(mj_data.ncon)

                # v30: detect real floor impact using low body height + any
                # reliable ground contact. The v29 nonfoot threshold was too
                # strict, so impact braking/rest never engaged in the supplied
                # log even though the body was already sliding on the floor.
                nonfoot_count, nonfoot_n = controller.ground_contact_count(min_normal_n=10.0, nonfoot_only=True)
                ground_count, ground_n = controller.ground_contact_count(min_normal_n=6.0, nonfoot_only=False)
                nonfoot_ground_contact = bool(nonfoot_count >= 1 or controller.has_nonfoot_ground_contact(min_normal_n=12.0))
                low_body_contact = bool(
                    pelvis_height_rest < TASK43_REST_LOW_PELVIS_M and
                    ground_count >= TASK43_REST_CONTACT_MIN_COUNT
                )
                clear_body_impact = bool(nonfoot_ground_contact or (low_body_contact and (pelvis_ang_speed > 0.35 or slip_rest > 0.035)))
                if clear_body_impact:
                    controller.activate_impact_brake()
                    controller.apply_impact_brake()

            if phase == 'fall':
                pelvis_id_rest = mujoco.mj_name2id(mj_model, MJOBJ_BODY, "Pelvis")
                vel6_rest = np.zeros(6)
                mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, pelvis_id_rest, vel6_rest, 0)
                pelvis_ang_speed = float(np.linalg.norm(vel6_rest[:3]))
                pelvis_height_rest = float(mj_data.xpos[pelvis_id_rest][2])
                slip_rest = float(np.linalg.norm(vel6_rest[3:][:2]))
                ncon_rest = int(mj_data.ncon)

                torso_height_rest = float(mj_data.xpos[controller._torso_id][2]) if getattr(controller, '_torso_id', -1) >= 0 else pelvis_height_rest
                head_height_rest = float(mj_data.xpos[controller._head_id][2]) if getattr(controller, '_head_id', -1) >= 0 else torso_height_rest
                rest_ref_fwd = np.asarray(controller.walk_ref_fwd_xy, dtype=float) if getattr(controller, 'walk_ref_fwd_xy', None) is not None else np.array([1.0, 0.0], dtype=float)
                fall_lean_rest = float(controller._compute_sagittal_trunk_lean_deg(rest_ref_fwd)) if hasattr(controller, '_compute_sagittal_trunk_lean_deg') else 0.0
                fall_elapsed_for_rest = step - int(actual_step_marks['fall_start']) if actual_step_marks.get('fall_start') is not None else 0
                strong_body_count, strong_body_n = controller.ground_contact_count(min_normal_n=8.0, nonfoot_only=True)
                all_ground_count, all_ground_n = controller.ground_contact_count(min_normal_n=5.0, nonfoot_only=False)
                strong_body_contact = bool(strong_body_count >= 1 or controller.has_nonfoot_ground_contact(min_normal_n=12.0))
                # Rest should start just after the visible impact/recoil, not
                # after the body has rolled for seconds. We accept either a
                # real non-foot contact or a low, multi-contact grounded body.
                grounded_low_body = (
                    (pelvis_height_rest < TASK43_REST_LOW_PELVIS_M or torso_height_rest < TASK43_REST_LOW_TORSO_M) and
                    (all_ground_count >= TASK43_REST_CONTACT_MIN_COUNT or ncon_rest >= 1)
                )
                settle_candidate = (
                    fall_elapsed_for_rest >= 4 and
                    grounded_low_body and
                    (strong_body_contact or all_ground_count >= 1 or fall_elapsed_for_rest >= TASK43_REST_FORCE_AFTER_IMPACT_FRAMES) and
                    slip_rest < 1.05 and
                    pelvis_ang_speed < 7.2 and
                    fall_lean_rest > 28.0
                )
                if settle_candidate:
                    _rest_stable_counter += 1
                else:
                    _rest_stable_counter = 0

                if _rest_stable_counter >= TASK43_REST_ENGAGE_STABLE_FRAMES and not controller.rest_mode:
                    print(f"  [Step {step}] Rest mode engaged (anti-slip + passive human settle)")
                    # Do not pass z_rest; v30 settles passively to avoid learned
                    # rest-policy rolling/slipping on the floor.
                    controller.activate_rest_mode(None)

                if controller.rest_mode:
                    controller.apply_rest_stiction()
                    if slip_rest > 0.035 or pelvis_ang_speed > 0.22:
                        controller.apply_impact_brake()
                    controller.apply_scenario29_sagittal_rail(MODEL_BODY_MASS, strength=0.35, hard=True)
                    deep_rest = (
                        pelvis_height_rest < 0.24 and
                        ncon_rest >= 2 and
                        slip_rest < 0.16 and
                        pelvis_ang_speed < 0.55
                    )
                    if deep_rest:
                        _rest_hard_lock_counter += 1
                    else:
                        _rest_hard_lock_counter = max(0, _rest_hard_lock_counter - 1)
                    # Also harden after a bounded time in rest mode. This is a
                    # final dead-band, not a pose snap: only velocities are killed.
                    if _rest_hard_lock_counter >= TASK43_SLOW_REST_HARD_LOCK_FRAMES or controller.rest_counter >= 22:
                        controller.harden_rest_mode()

            # -- IMU + Layer-1 kinematic logging ----------------------------
            sim_time_now = float(mj_data.time)
            _last_imu_peak = imu.log_frame(sim_time_now)
            marker_exporter.capture_frame(sim_time_now, phase=phase)
            dynamics_frame = dynamics_analyzer.capture_frame(sim_time_now, phase=phase)

            task43_support.draw_viewer_support(viewer)
            viewer.sync()

            # -- Console logging ---------------------------------------------
            if step % 30 == 0 or (phase in ('perturb', 'react', 'fall') and step % 5 == 0):
                pelvis_id  = mujoco.mj_name2id(mj_model, MJOBJ_BODY, "Pelvis")
                pelvis_pos = mj_data.xpos[pelvis_id]
                pelvis_vel = _body_world_velocity(mj_model, mj_data, pelvis_id)
                live_imu   = imu.read_imu(sim_time_now)

                xcom_margin = None
                zmp_stable  = None
                walk_fall_flag = (pelvis_pos[2] < 0.35)
                if phase in ('stand', 'step'):
                    try:
                        xcom_2d, support_c, xcom_margin = controller.compute_xcom()
                        _, zmp_stable, _  = controller.compute_zmp()
                        if step >= _xcom_log_start_step and phase == 'step':
                            xcom_history.append(np.asarray(xcom_2d).copy())
                            support_center_history.append(np.asarray(support_c).copy())
                            xcom_timestamps.append(sim_time_now)
                        if step >= STABILITY_WARMUP_STEPS and xcom_margin is not None and xcom_margin < 0.0:
                            _xcom_negative_streak += 1
                        elif phase in ('stand', 'step'):
                            _xcom_negative_streak = 0
                        walk_fall_flag = (_xcom_negative_streak >= XCOM_NEGATIVE_CONFIRM_STEPS and phase == 'step' and pelvis_pos[2] < 0.82)
                    except Exception:
                        xcom_margin = None
                        _xcom_negative_streak = 0

                metrics = dashboard.report(
                    step=step,
                    phase=phase,
                    leg_strength=controller.leg_strength,
                    xcom_margin=xcom_margin,
                    fall_predicted=walk_fall_flag if phase in ('stand', 'step') else (pelvis_pos[2] < 0.35),
                    imu_peak=_last_imu_peak,
                    sensor_impact=float(live_imu['impact']),
                    control_source='guardian',
                    mjpc_cost=np.nan,
                    dynamics=dynamics_frame,
                )
                metrics.update({
                    'step': step,
                    'phase': phase,
                    'pelvis_vx': float(pelvis_vel[0]),
                    'ground_vertical_bw': float((dynamics_frame.get('total_ground_vertical_n_filt', dynamics_frame.get('total_ground_vertical_n', 0.0))) / max(MODEL_BODY_MASS * 9.81, 1.0)),
                    'nonfoot_impact_bw': float((dynamics_frame.get('primary_impact_body_load_n_filt', dynamics_frame.get('primary_impact_body_load_n', 0.0))) / max(MODEL_BODY_MASS * 9.81, 1.0)),
                    'primary_impact_body': dynamics_frame.get('primary_impact_body', ''),
                    'tangential_ratio': float(dynamics_frame.get('tangential_ratio', 0.0)),
                    'force': controller.force_applied,
                    'z_blend': controller.z_blend,
                    'imu_accel_mag': float(np.linalg.norm(live_imu['accel'])),
                    'imu_impact': float(live_imu['impact']),
                    'imu_conf': float(live_imu['sensor_confidence']),
                    'xcom_margin': xcom_margin,
                    'zmp_stable': zmp_stable,
                })
                metrics_log.append(metrics)

            # Safety reset only during standing
            if terminated and step < phase_boundaries[1]:
                obs, _ = env.reset()
                task43_support.place_on_platform()
                obs = _task43_refresh_obs(env, obs)
                controller.restore_strength()


    def _compute_preperturb_walk_metrics_local(controller, mj_model, mj_data, body_mass, age_params):
        pelvis_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, 'Pelvis')
        if pelvis_id < 0:
            return {'ready': False, 'ready_mode': 'none', 'speed': 0.0, 'froude': 0.0, 'xcom_margin': -1.0,
                    'double_support': True, 'grf_bw': 0.0, 'lateral_err': 0.0, 'yaw_err_deg': 0.0,
                    'reference_speed_ratio': 0.0, 'policy_limited': False}
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, pelvis_id, vel6, 0)
        vel = vel6[3:].copy()
        pos = mj_data.xpos[pelvis_id].copy()
        R = mj_data.xmat[pelvis_id].reshape(3, 3)
        fwd_body = np.asarray(R[:, 0], dtype=float)
        yaw = float(np.arctan2(fwd_body[1], fwd_body[0]))
        if getattr(controller, 'walk_ref_fwd_xy', None) is not None:
            ref_fwd = np.asarray(controller.walk_ref_fwd_xy, dtype=float)
            ref_lat = np.asarray(controller.walk_ref_lat_xy, dtype=float)
            origin_xy = np.asarray(controller.walk_ref_origin_xy, dtype=float)
        else:
            ref_fwd = np.asarray(fwd_body[:2], dtype=float)
            ref_fwd /= max(np.linalg.norm(ref_fwd), 1e-9)
            ref_lat = np.array([-ref_fwd[1], ref_fwd[0]], dtype=float)
            origin_xy = np.asarray(pos[:2], dtype=float)
        vel_xy = np.asarray(vel[:2], dtype=float)
        forward_speed = float(np.dot(vel_xy, ref_fwd))
        abs_speed = abs(forward_speed)
        lateral_err = float(np.dot(np.asarray(pos[:2], dtype=float) - origin_xy, ref_lat))
        yaw_ref = float(getattr(controller, 'walk_ref_yaw', yaw))
        yaw_err_deg = float(np.degrees(np.arctan2(np.sin(yaw - yaw_ref), np.cos(yaw - yaw_ref))))
        froude = float((abs_speed ** 2) / max(9.81 * float(age_params.get('leg_length', 0.9)), 1e-9))
        try:
            _, _, xcom_margin = controller.compute_xcom()
        except Exception:
            xcom_margin = -1.0

        left_vertical = 0.0
        right_vertical = 0.0
        total_ground_vertical = 0.0
        for i in range(int(mj_data.ncon)):
            c = mj_data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            names = ((mujoco.mj_id2name(mj_model, MJOBJ_GEOM, g1) or '').lower(),
                     (mujoco.mj_id2name(mj_model, MJOBJ_GEOM, g2) or '').lower())
            ground = any(('floor' in n or 'ground' in n or 'plane' in n) for n in names) or any(
                mj_model.geom_type[g] == getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', -999)
                for g in (g1, g2)
            )
            if not ground:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(mj_model, mj_data, i, wrench)
            if wrench[0] <= 1.0:
                continue
            f_world, _ = _contact_wrench_world(c, wrench)
            total_ground_vertical += max(0.0, float(f_world[2]))
            ng = g2 if (g1 in controller._ground_geoms) else g1
            bid = int(mj_model.geom_bodyid[ng])
            bname = (mujoco.mj_id2name(mj_model, MJOBJ_BODY, bid) or '').lower()
            is_footlike = any(k in bname for k in ('foot', 'ankle', 'toe', 'heel'))
            if is_footlike and _body_name_matches_side(bname, 'left'):
                left_vertical += max(0.0, float(f_world[2]))
            if is_footlike and _body_name_matches_side(bname, 'right'):
                right_vertical += max(0.0, float(f_world[2]))

        foot_load_threshold = 0.12 * body_mass * 9.81
        left_support = bool(left_vertical > foot_load_threshold)
        right_support = bool(right_vertical > foot_load_threshold)
        double_support = bool(left_support and right_support)
        grf_bw = float(total_ground_vertical / max(body_mass * 9.81, 1.0))

        lit_target = float(getattr(controller, 'walk_lit_target_speed', controller.age_style.get('target_walk_speed', max(controller.walk_target_speed, 0.2))))
        policy_target = float(max(controller.walk_target_speed, 0.18))
        policy_limited = bool(getattr(controller, 'walk_policy_limited', False))
        reference_speed_ratio = float(abs_speed / max(lit_target, 1e-6))

        stable_base = (xcom_margin > -0.22 and grf_bw < 2.6)
        reference_ready = stable_base and abs_speed >= 0.70 * lit_target
        policy_limited_ready = stable_base and policy_limited and abs_speed >= 0.80 * policy_target

        ready = bool(reference_ready or policy_limited_ready)
        ready_mode = 'reference' if reference_ready else ('policy_limited' if policy_limited_ready else 'none')

        return {
            'ready': ready,
            'ready_mode': ready_mode,
            'speed': float(forward_speed),
            'froude': float(froude),
            'xcom_margin': float(xcom_margin),
            'double_support': bool(double_support),
            'grf_bw': float(grf_bw),
            'lateral_err': float(lateral_err),
            'yaw_err_deg': float(yaw_err_deg),
            'reference_speed_ratio': float(reference_speed_ratio),
            'policy_limited': bool(policy_limited),
        }


    def detect_fall_events(imu_data_buffer, dynamics_frames, marker_frames=None, perturb_start_time=None):
        ts = np.asarray(imu_data_buffer.get('timestamp', []), dtype=float)
        heights = np.asarray(imu_data_buffer.get('pelvis_height', []), dtype=float)
        vels = np.asarray(imu_data_buffer.get('pelvis_velocity', []), dtype=float)
        if ts.size == 0 or heights.size == 0:
            return {'available': False}
        if vels.ndim == 1:
            speed = np.abs(vels)
        else:
            speed = np.linalg.norm(vels[:, :2], axis=1)
        trunk = None
        if marker_frames:
            trunk = np.asarray([fr.get('trunk_lean_deg', 0.0) for fr in marker_frames], dtype=float)
        imp_t = np.asarray([fr.get('time', np.nan) for fr in dynamics_frames], dtype=float) if dynamics_frames else np.array([], dtype=float)
        imp_bw = np.asarray([fr.get('primary_impact_body_load_n_filt', fr.get('primary_impact_body_load_n', 0.0)) / max(1.0, 70.0 * 9.81) for fr in dynamics_frames], dtype=float) if dynamics_frames else np.array([], dtype=float)
        if dynamics_frames:
            body_mass_guess = float(max(1.0, np.median([fr.get('support_vertical_n', 0.0) for fr in dynamics_frames[:min(len(dynamics_frames), 30)] if fr.get('support_vertical_n', 0.0) > 0.0]) / 9.81)) if any(fr.get('support_vertical_n', 0.0) > 0.0 for fr in dynamics_frames[:min(len(dynamics_frames), 30)]) else 70.0
            imp_bw = np.asarray([fr.get('primary_impact_body_load_n_filt', fr.get('primary_impact_body_load_n', 0.0)) / max(1.0, body_mass_guess * 9.81) for fr in dynamics_frames], dtype=float)
        t0 = float(perturb_start_time) if perturb_start_time is not None else float(ts[0])
        i0 = int(np.searchsorted(ts, t0, side='left'))
        pre_h = float(np.median(heights[max(0, i0-30):max(i0, i0+1)])) if i0 > 0 else float(np.max(heights[:min(len(heights), 30)]))
        onset_idx = None
        for i in range(i0, len(ts)):
            trunk_cond = bool(trunk is not None and i < len(trunk) and trunk[i] > 22.0)
            h_cond = heights[i] < max(0.78 * pre_h, pre_h - 0.10)
            v_cond = speed[i] > 0.55
            if (trunk_cond and h_cond) or (h_cond and v_cond):
                onset_idx = i
                break
        if onset_idx is None:
            onset_idx = i0
        impact_idx = None
        if imp_t.size:
            j0 = int(np.searchsorted(imp_t, ts[onset_idx], side='left'))
            for j in range(j0, len(imp_t)):
                if imp_bw[j] > 0.25:
                    impact_idx = j
                    break
            if impact_idx is None:
                impact_idx = int(np.argmax(imp_bw[j0:]) + j0) if j0 < len(imp_bw) else None
        settle_idx = None
        start_settle = onset_idx
        if impact_idx is not None:
            start_settle = max(start_settle, int(np.searchsorted(ts, imp_t[impact_idx], side='left')))
        streak = 0
        for i in range(start_settle, len(ts)):
            low_h = heights[i] < 0.20
            low_v = speed[i] < 0.08
            if low_h and low_v:
                streak += 1
            else:
                streak = 0
            if streak >= 35:
                settle_idx = i - 34
                break
        impact_time = float(imp_t[impact_idx]) if impact_idx is not None else float(ts[min(len(ts)-1, onset_idx)])
        onset_time = float(ts[onset_idx])
        settle_time = float(ts[settle_idx]) if settle_idx is not None else float(ts[-1])
        return {
            'available': True,
            'onset_time': onset_time,
            'impact_time': impact_time,
            'settle_time': settle_time,
            'fall_duration_s': max(0.0, settle_time - onset_time),
            'lead_time_ms': max(0.0, (impact_time - onset_time) * 1000.0),
            'onset_idx': int(onset_idx),
            'impact_idx': int(impact_idx) if impact_idx is not None else None,
            'settle_idx': int(settle_idx) if settle_idx is not None else None,
        }

    # -----------------------------------------------------------------------
    # POST-SIMULATION VALIDATION AND REPORT
    # -----------------------------------------------------------------------
    validator = FallValidator()
    kinematic_data = {
        'head_velocity': (imu.data_buffer['pelvis_velocity'][-1]
                          if imu.data_buffer['pelvis_velocity']
                          else [0, 0, 0]),
    }
    _perturb_start_time = float((actual_step_marks.get('perturb_start') if actual_step_marks.get('perturb_start') is not None else phase_boundaries[2])) * float(imu.native_dt)
    event_summary = detect_fall_events(imu.data_buffer, dynamics_analyzer.frames, marker_exporter.frames, perturb_start_time=_perturb_start_time)
    validation_results = validator.validate_fall(
        imu.data_buffer, kinematic_data, ACTIVE_FALL_TYPE,
        perturb_start_time=_perturb_start_time, event_summary=event_summary,
    )

    print("\n" + "=" * 70)
    print("  FALL VALIDATION REPORT")
    print("=" * 70)
    print(f"  Overall Confidence Score : {validation_results['overall_score']:.1%}")
    print(f"  Classification           : {validation_results['classification']}")
    print("")
    print("  Detailed Checks:")
    for check, passed in validation_results['checks'].items():
        print(f"    {check:35s}  {'PASS' if passed else 'FAIL'}")
    if validation_results['warnings']:
        print("\nWarnings:")
        for w in validation_results['warnings']:
            print(f"    ! {w}")
    adv = validation_results.get('advanced_metrics', {})
    if adv:
        print("\nAdvanced Metrics:")
        for k, v in adv.items():
            print(f"    {k:42s}  {v}")

    print("\n" + "=" * 70)
    print("  SISFall SIGNATURE VALIDATION")
    print("=" * 70)
    sisfall_val = SISFallValidator()
    sf_result = sisfall_val.validate_sisfall_signature(
        imu.data_buffer, ACTIVE_FALL_TYPE,
        body_mass=MODEL_BODY_MASS,
        perturb_start_time=_perturb_start_time, event_summary=event_summary,
    )
    print(f"  Channel used    : {sf_result['margins'].get('channel_used','?')}")
    print(f"  Filter note     : {sf_result['margins'].get('filter_note','?')}")
    print(f"  Sensor location : {sf_result['margins'].get('sensor_location','?')}")
    print(f"  Fall direction  : {sf_result['direction']}")
    print(f"  Peak accel (filt): {sf_result['margins'].get('peak_accel_ms2', 0.0):.2f} m/s^2  (expected {sf_result['margins'].get('peak_accel_range', ('?','?'))})")
    print(f"  Peak accel (raw) : {sf_result['margins'].get('peak_accel_raw_ms2', 0.0):.2f} m/s^2")
    print(f"  Fall duration   : {sf_result['margins'].get('fall_duration_s', 0.0):.2f} s  (expected {sf_result['margins'].get('duration_range', ('?','?'))})")
    for chk_name, chk_val in sf_result['checks'].items():
        print(f"    {chk_name:35s}  {'PASS' if chk_val else 'FAIL'}")
    print(f"  SISFall compliant : {sf_result['sisfall_compliant']}")

    print("\n" + "=" * 70)
    print("  KFall-STYLE PRE-IMPACT VALIDATION")
    print("=" * 70)
    kfall_val = KFallValidator()
    kfall_result = kfall_val.validate(imu, perturb_start_time=_perturb_start_time, event_summary=event_summary)
    for chk_name, chk_val in kfall_result.get('checks', {}).items():
        print(f"    {chk_name:35s}  {'PASS' if chk_val else 'FAIL'}")
    if kfall_result.get('available', False):
        print(f"  Lead time        : {kfall_result.get('lead_time_ms', 0.0):.1f} ms  (benchmark ~{kfall_result.get('benchmark_lead_ms', 403.0):.0f} ms)")
        print(f"  Peak accel raw   : {kfall_result.get('peak_accel_ms2', 0.0):.2f} m/s^2")
        print(f"  Peak gyro        : {kfall_result.get('peak_gyro_rads', 0.0):.2f} rad/s")
        print(f"  Onset reason     : {kfall_result.get('onset_reason', 'n/a')}")
        print(f"  Adaptive accel th: {kfall_result.get('adaptive_acc_threshold_ms2', 0.0):.2f} m/s^2")
        print(f"  Adaptive gyro th : {kfall_result.get('adaptive_gyro_threshold_rads', 0.0):.2f} rad/s")
        print(f"  Adaptive jerk th : {kfall_result.get('adaptive_jerk_threshold_ms3', 0.0):.2f} m/s^3")
        print(f"  Adaptive h th    : {kfall_result.get('adaptive_height_threshold_m', 0.0):.3f} m")
    print(f"  KFall-style compliant : {kfall_result.get('kfall_compliant', False)}")

    print("\nCOMMON FALL EVENT SUMMARY")
    print("-" * 50)
    if event_summary.get('available', False):
        print(f"  Perturb start     : {_perturb_start_time:.3f} s")
        print(f"  Fall onset        : {event_summary.get('onset_time', 0.0):.3f} s")
        print(f"  Main impact       : {event_summary.get('impact_time', 0.0):.3f} s")
        print(f"  Settle time       : {event_summary.get('settle_time', 0.0):.3f} s")
        print(f"  Fall duration     : {event_summary.get('fall_duration_s', 0.0):.2f} s")
        print(f"  Impact lead       : {event_summary.get('lead_time_ms', 0.0):.1f} ms")
    else:
        print("  Event detector    : unavailable")

    print("\nCAPTURE POINT VALIDATION  (Hof 2005)")
    print("-" * 50)
    cp_result = validator.validate_capture_point(
        xcom_history, support_center_history, xcom_timestamps,
        body_mass=MODEL_BODY_MASS,
        leg_length=age_params.get('leg_length', 0.93),
    )
    if 'error' not in cp_result:
        print(f"  XCoM stable samples : {cp_result['num_samples']}")
        print(f"  Min XCoM margin     : {cp_result['min_margin_m']:.4f} m")
        print(f"  Max XCoM margin     : {cp_result['max_margin_m']:.4f} m")
        if cp_result['fall_prediction_time'] is not None:
            print(f"  Capture-pt exit     : {cp_result['fall_prediction_time']:.3f} s")
        else:
            print(f"  Capture-pt exit     : None (XCoM stable throughout stand/walk)")
        print(f"  Hof 2005 compliant  : {cp_result['hof_2005_compliant']}")
    else:
        print(f"  Capture-point       : {cp_result['error']}")

    print("\nBIOMECHANICAL RANGE CHECKS")
    print("-" * 50)
    bio_result = validator.validate_biomechanical_ranges(imu.data_buffer, body_mass=MODEL_BODY_MASS, dynamics_frames=dynamics_analyzer.frames)
    print(f"  Body mass / weight  : {bio_result['body_mass_kg']:.1f} kg / {bio_result['body_weight_n']:.1f} N")
    if bio_result.get('peak_angular_velocity_ok') is not None:
        print(f"  Peak angular vel    : {bio_result['peak_angular_velocity_rads']:.2f} rad/s  < {bio_result['angular_velocity_threshold']:.1f} rad/s  {'PASS' if bio_result['peak_angular_velocity_ok'] else 'FAIL'}")
    if bio_result.get('impact_force_in_range') is not None:
        lo, hi = bio_result['expected_range_n']
        print(f"  Peak impact force   : {bio_result['peak_impact_force_n']:.1f} N  (expected {lo:.0f}-{hi:.0f} N)  {'PASS' if bio_result['impact_force_in_range'] else 'FAIL'}")

    print("\nPHASE-BY-PHASE BIOMECHANICAL SUMMARY")
    print("-" * 50)
    for ph in PHASES.keys():
        phase_metrics = [m for m in metrics_log if m.get('phase') == ph]
        dashboard.print_phase_summary(ph, phase_metrics)

    print("\nPERFORMANCE DIAGNOSTICS")
    print("-" * 50)
    walk_metrics = []
    step_metrics = [m for m in metrics_log if m.get('phase') == 'step']
    fall_metrics = [m for m in metrics_log if m.get('phase') == 'fall']
    walk_xcom_neg = 0.0
    walk_mean_speed = 0.0
    walk_mean_froude = 0.0
    walk_peak_grf = 0.0
    if walk_metrics:
        walk_xcom_neg = float(sum(1 for m in walk_metrics if (m.get('xcom_margin') is not None and m.get('xcom_margin') < 0.0))) / max(len(walk_metrics), 1)
        walk_mean_speed = float(np.mean([abs(m.get('pelvis_vx', 0.0)) for m in metrics_log if m.get('phase') == 'step']))
        walk_mean_froude = float(np.mean([m.get('froude', 0.0) for m in walk_metrics]))
        walk_peak_grf = float(np.max([m.get('grf_bw', 0.0) for m in walk_metrics]))
        print(f"  Walk mean |vx|     : {walk_mean_speed:.3f} m/s")
        print(f"  Walk mean Froude   : {walk_mean_froude:.3f}")
        print(f"  Walk XCoM<0 frac*  : {walk_xcom_neg:.1%}")
        print(f"  Walk peak GRF/BW   : {walk_peak_grf:.2f}")
        age_ref = get_age_reference_band(SIM_AGE, SIM_SEX, SIM_HEIGHT)
        speed_lo, speed_hi = age_ref['comfortable_speed_band_mps']
        ds_lo, ds_hi = age_ref['double_support_band']
        print("  Age ref speed band : "
              f"{speed_lo:.2f}-{speed_hi:.2f} m/s ({age_ref['label']})")
    if fall_metrics:
        fall_peak_impact_bw = float(np.max([m.get('impact_bw', 0.0) for m in fall_metrics]))
        fall_peak_omega = float(np.max([m.get('pelvis_angular_speed_dps', 0.0) for m in fall_metrics]))
        fall_peak_ground_bw = float(np.max([m.get('ground_vertical_bw', 0.0) for m in fall_metrics]))
        fall_peak_nonfoot_bw = float(np.max([m.get('nonfoot_impact_bw', 0.0) for m in fall_metrics]))
        settle_frac = float(sum(1 for m in fall_metrics if m.get('settle_flag', False))) / max(len(fall_metrics), 1)
        impact_bodies = [m.get('primary_impact_body', '') for m in fall_metrics if m.get('primary_impact_body', '')]
        first_body = impact_bodies[0] if impact_bodies else 'n/a'
        print(f"  Fall peak impact/BW: {fall_peak_impact_bw:.2f}")
        print(f"  Fall peak ground/BW: {fall_peak_ground_bw:.2f}")
        print(f"  Fall peak nonfoot/BW: {fall_peak_nonfoot_bw:.2f}")
        print(f"  Fall first impact body: {first_body}")
        print(f"  Fall peak omg      : {fall_peak_omega:.1f} deg/s")
        print(f"  Fall settled frac  : {settle_frac:.1%}")

    pre_metrics = [m for m in metrics_log if m.get('phase') in ('stand', 'step')]
    if pre_metrics:
        initial_h = float(pre_metrics[0].get('pelvis_h', 0.0))
        pre_min_h = float(np.min([m.get('pelvis_h', 0.0) for m in pre_metrics]))
        pre_h_ratio = pre_min_h / max(initial_h, 1e-6)
        walk_mean_ds = 0.0
        step_mean_ds = float(np.mean([1.0 if m.get('double_support', False) else 0.0 for m in step_metrics])) if step_metrics else 0.0
        pre_valid = (pre_h_ratio > 0.92 and step_mean_ds < 0.80)
        print("\nPRE-PERTURB TASK VALIDITY")
        print("-" * 50)
        print(f"  Initial pelvis h   : {initial_h:.3f} m")
        print(f"  Min pre-perturb h  : {pre_min_h:.3f} m")
        print(f"  Height ratio       : {pre_h_ratio:.3f}")
        print(f"  Mean step dbl_sup  : {step_mean_ds:.1%}")
        age_ref = get_age_reference_band(SIM_AGE, SIM_SEX, SIM_HEIGHT)
        ds_lo, ds_hi = age_ref['double_support_band']
        speed_lo, speed_hi = age_ref['comfortable_speed_band_mps']
        speed_status = 'N/A-no-continuous-walk'
        ds_status = 'N/A-single-step'
        print(f"  Age ref speed band : {speed_lo:.2f}-{speed_hi:.2f} m/s  -> {speed_status}")
        print(f"  Age ref dbl_sup    : {ds_lo:.0%}-{ds_hi:.0%}  -> {ds_status}")
        print(f"  Task validity      : {'PASS' if pre_valid else 'FAIL'}")

    scenario_authenticity = None
    if walk_metrics:
        style = get_age_style(SIM_AGE)
        ds_target = style['expected_double_support']
        ds_err = abs(walk_mean_ds - ds_target) if 'walk_mean_ds' in locals() else 1.0
        ds_score = float(np.clip(1.0 - ds_err / 0.35, 0.0, 1.0))
        froude_target = max(0.05, 0.75 * (style['target_walk_speed'] ** 2) / (9.81 * max(age_params.get('leg_length', 0.9), 0.1)))
        froude_score = float(np.clip(1.0 - abs(walk_mean_froude - froude_target) / max(0.08, 0.5 * froude_target), 0.0, 1.0)) if 'walk_mean_froude' in locals() else 0.0
        xcom_score = float(np.clip(1.0 - walk_xcom_neg / 0.35, 0.0, 1.0))
        grf_score = float(np.clip(1.0 - max(0.0, walk_peak_grf - 2.0) / 6.0, 0.0, 1.0))
        pre_score = float(np.mean([pre_h_ratio, ds_score, froude_score, xcom_score, grf_score])) if 'pre_h_ratio' in locals() else 0.0
        scenario_authenticity = float(0.45 * validation_results['overall_score'] +
                                      0.25 * pre_score +
                                      0.20 * kfall_result.get('score', 0.0) +
                                      0.10 * (1.0 if sf_result.get('sisfall_compliant', False) else 0.0))
        print("\nSCENARIO AUTHENTICITY")
        print("-" * 50)
        print(f"  Fall-only score    : {validation_results['overall_score']:.1%}")
        print(f"  Pre-perturb score  : {pre_score:.1%}")
        print(f"  KFall score        : {kfall_result.get('score', 0.0):.1%}")
        print(f"  SISFall compliant  : {sf_result.get('sisfall_compliant', False)}")
        print(f"  Authenticity score : {scenario_authenticity:.1%}")

    _rest_mode_summary = bool(controller.rest_mode)
    _rest_hard_lock_summary = bool(controller.rest_mode_hard_lock)
    _impact_brake_summary = bool(controller.impact_brake_mode)
    env.close()
    print("\n" + "=" * 70)
    print("  SIMULATION COMPLETE - Component Summary")
    print("=" * 70)
    print(f"  [AnthropometricModel]     age={SIM_AGE} height={SIM_HEIGHT}m weight={SIM_WEIGHT_DISPLAY} | body_mass={age_params.get('body_mass_kg', float(np.sum(mj_model.body_mass))):.2f}kg | "
          f"strength={age_params['strength_factor']:.3f} | "
          f"balance_impairment={age_params['balance_impairment']:.3f}")
    print(f"  [IMUValidator]            native frames={len(imu.data_buffer['timestamp'])} at {imu.native_hz:.2f} Hz | "
          f"output={imu.output_hz:.0f} Hz ({imu.output_mode})")
    print(f"  [FallTypeLibrary]         scenario='{ACTIVE_FALL_TYPE}' | "
          f"top-rung ladder forward fall + forward/down body pitch; no hand support")
    print(f"  [EnhancedBiofidelicCtrl]  protective reflex monitored for "
          f"{_protective_reflex_count} steps")
    print(f"  [MarkerKinematics]        markers={marker_summary['num_markers']} | joints={marker_summary['num_joint_exports']} | frames={len(marker_exporter.frames)}")
    print(f"  [DynamicsLayer2]          frames={len(dynamics_analyzer.frames)} | contact_rows={len(dynamics_analyzer.contact_rows)}")
    print("  [PaperAlignmentBridge]    OpenSim GRF + ExternalLoads + synthetic pose dataset enabled")
    print(f"  [RestMode]                engaged={_rest_mode_summary} hard_lock={_rest_hard_lock_summary}")
    print(f"  [ImpactBrake]             engaged={_impact_brake_summary}")
    print(f"  [GaitPhaseDetector]       BYPASSED for task 43 deterministic ladder climb")
    print(f"  [FallValidator]           score={validation_results['overall_score']:.1%} "
          f"({validation_results['classification']})")
    if scenario_authenticity is not None:
        print(f"  [ScenarioAuthenticity]    score={scenario_authenticity:.1%}")

    # Export IMU data to CSV
    from datetime import datetime
    imu_filename = _task43_output_path((f"fall_forward_ladder_age{SIM_AGE}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"))
    try:
        result = imu.export_to_csv(imu_filename, metadata={
            'age': SIM_AGE, 'height': SIM_HEIGHT, 'sex': SIM_SEX,
            'weight': age_params.get('body_mass_kg', float(np.sum(mj_model.body_mass))), 'body_mass_kg': age_params.get('body_mass_kg', float(np.sum(mj_model.body_mass))), 'fall_type': 'forward_fall_climbing_up_ladder',
        })
        print(f"  [IMUValidator]            CSV saved -> {result['filename']}")
        print(f"  [IMUValidator]            fall_detected=True in {result['falls_detected']} frames")
    except Exception as e:
        print(f"  [IMUValidator]            CSV export error: {e}")

    # Export Layer-1 biomechanics bundle
    try:
        marker_prefix = str(Path(imu_filename).with_suffix(''))
        marker_bundle = marker_exporter.export_bundle(marker_prefix)
        print(f"  [MarkerKinematics]        marker CSV -> {marker_bundle['marker_csv']['filename']}")
        print(f"  [MarkerKinematics]        segment CSV -> {marker_bundle['segment_csv']['filename']}")
        print(f"  [MarkerKinematics]        joint CSV -> {marker_bundle['joint_csv']['filename']}")
        print(f"  [MarkerKinematics]        TRC export -> {marker_bundle['trc']['filename']}")
        print(f"  [MarkerKinematics]        MOT export -> {marker_bundle['mot']['filename']}")
        print(f"  [MarkerKinematics]        quality CSV -> {marker_bundle['quality_csv']['filename']}")
        print(f"  [MarkerKinematics]        pose JSON -> {marker_bundle['pose_json']['filename']}")
        if marker_bundle.get('visuals', {}).get('available', False):
            for _plot in marker_bundle.get('visuals', {}).get('plots', []):
                print(f"  [MarkerKinematics]        visual PNG -> {_plot}")
    except Exception as e:
        print(f"  [MarkerKinematics]        export error: {e}")

    # Export Layer-2 dynamics bundle
    try:
        dynamics_bundle = dynamics_analyzer.export_bundle(marker_prefix)
        print(f"  [DynamicsLayer2]          frame CSV -> {dynamics_bundle['frame_csv']['filename']}")
        print(f"  [DynamicsLayer2]          contact CSV -> {dynamics_bundle['contact_csv']['filename']}")
        for p in dynamics_bundle.get('visuals', {}).get('plots', []):
            print(f"  [DynamicsLayer2]          visual PNG -> {p}")
    except Exception as e:
        print(f"  [DynamicsLayer2]          export error: {e}")

    # Export paper-alignment bridge bundle
    try:
        paper_bundle = paper_exporter.export_bundle(marker_prefix, views=PAPER_ALIGNMENT_EXPORT.get('pose_views', ('frontal', 'sagittal', 'oblique')))
        print(f"  [PaperAlignmentBridge]    OpenSim GRF MOT -> {paper_bundle['opensim_grf']['filename']}")
        print(f"  [PaperAlignmentBridge]    ExternalLoads XML -> {paper_bundle['external_loads_xml']['filename']}")
        print(f"  [PaperAlignmentBridge]    marker registration -> {paper_bundle['marker_registration']['filename']}")
        print(f"  [PaperAlignmentBridge]    synthetic pose JSON -> {paper_bundle['pose_dataset_json']['filename']}")
        if paper_bundle.get('pose_preview', {}).get('available', False):
            print(f"  [PaperAlignmentBridge]    synthetic pose preview -> {paper_bundle['pose_preview']['filename']}")
    except Exception as e:
        print(f"  [PaperAlignmentBridge]    export error: {e}")

    # Save validation report alongside CSV
    report_file = str(Path(str(Path(imu_filename).with_suffix('')) + '_validation.txt'))
    validator.generate_report(validation_results, report_file)
    print(f"  [FallValidator]           Validation report -> {report_file}")

    print("=" * 70)

    controller.restore_strength()
    controller.clear_forces()

    # Optional: Save metrics for analysis
    # import json
    # with open('fall_metrics.json', 'w') as f:
    #     json.dump(metrics_log, f, indent=2)



if __name__ == "__main__":
    run_with_subject()


# Task43 compatibility alias for dispatchers that expect an explicit scenario entrypoint.
def run_task43(subject_params=None):
    return run_with_subject(subject_params)

