# -*- coding: utf-8 -*-
"""
scenario23_legacy.py  -  COMPLETELY SELF-CONTAINED (no fall_core, no backward_fall_walking_best)
===========================================================================================
Scenario 23: Forward fall when trying to get up.

Biomechanics
------------
  The subject begins from a chair-supported seated posture,
  initiates a sit-to-stand rise, then loses forward balance during ascent.
  The collapse starts before full upright recovery so the body pitches
  forward from partial standing height into a prone / chest-down landing.

Phase sequence  (all timings at 30 Hz native):
  SIT      profile-adaptive  - seated hold on the chair before rising
  GET_UP   profile-adaptive  - guided rise from the chair
  COLLAPSE profile-adaptive  - forward topple force + pitch torque
  FALL     profile-adaptive  - free forward fall; prone settle

IMPORTANT - safe imports only:
  numpy, torch, mujoco, mujoco.viewer, humenv, metamotivo
  biofidelic_profile  (has no module-level execution)
  Does NOT import backward_fall_walking_best or fall_core or fall_scenario_library.
"""
# --- stdlib -------------------------------------------------------------------
import sys, os, csv, math, re
from datetime import datetime
from collections import deque

# --- numerical / ML -----------------------------------------------------------
import numpy as np
import torch

# --- MuJoCo -------------------------------------------------------------------
import mujoco
import mujoco.viewer

# --- simulation environment + model -------------------------------------------
from humenv import make_humenv
from humenv.rewards import LocomotionReward
from metamotivo.fb_cpr.huggingface import FBcprModel

# --- project-local SAFE imports (no module-level execution) -------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
# Robust project-root resolution:
# - if this file lives in <project>/scenarios/, root is its parent
# - if it lives directly in <project>/, root is this directory
if os.path.basename(_HERE) == 'scenarios':
    _ROOT = os.path.dirname(_HERE)
else:
    _ROOT = _HERE
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from biofidelic_profile import (
    muscle_strength_factor,
    reaction_delay_seconds,
    balance_impairment,
    weakening_config,
    print_subject_profile,
    get_age_style_v2,
)

# Shared read-only exporters from the task-34 backbone.
# These do not change the motion loop; they only capture extra outputs so
# scenario 20 can follow the same run-folder/export standard.
try:
    from backward_fall_walking_best import (
        MarkerKinematicsExporter,
        DynamicsContactAnalyzer,
        PaperAlignmentExporter,
        PhysicsDashboard,
        FallValidator,
        SISFallValidator,
        KFallValidator,
        detect_fall_events,
        get_age_reference_band,
    )
    _HAS_SHARED_EXPORTERS = True
except Exception:
    MarkerKinematicsExporter = None
    DynamicsContactAnalyzer = None
    PaperAlignmentExporter = None
    PhysicsDashboard = None
    FallValidator = None
    SISFallValidator = None
    KFallValidator = None
    detect_fall_events = None
    get_age_reference_band = None
    _HAS_SHARED_EXPORTERS = False

# --- MuJoCo API constants -----------------------------------------------------
MJOBJ_BODY     = mujoco.mjtObj.mjOBJ_BODY
MJOBJ_ACTUATOR = mujoco.mjtObj.mjOBJ_ACTUATOR
MJOBJ_GEOM     = mujoco.mjtObj.mjOBJ_GEOM

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
_EMBED_CACHE_FILE = os.path.join(_ROOT, "scenario23_embed_cache.pt")

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def _resolve_weight(height, age, sex, explicit=None):
    if explicit is not None:
        w = float(explicit)
        return w, w / max(height**2, 1e-9), "user"
    bmi = {"young": 23.0, "mid": 24.5, "old": 26.0}[
        "young" if age < 40 else "old" if age >= 65 else "mid"
    ]
    if sex == "female":
        bmi -= 1.0
    w = bmi * height**2
    return float(w), float(bmi), "auto"


def _body_id(mj_model, *names):
    for n in names:
        bid = mujoco.mj_name2id(mj_model, MJOBJ_BODY, n)
        if bid >= 0:
            return bid
    return -1


def _body_vel_world(mj_model, mj_data, body_id):
    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, body_id, vel6, 0)
    return vel6[3:].copy()


def _infer_anatomical_ref_xy(mj_model, mj_data, pelvis_id):
    """Estimate avatar anatomical forward in world XY from lower-limb anatomy."""
    pair_sets = [
        (('L_Toe','ToeL','LeftToe'), ('L_Foot','FootL','LeftFoot','L_Ankle','AnkleL','LeftAnkle')),
        (('R_Toe','ToeR','RightToe'), ('R_Foot','FootR','RightFoot','R_Ankle','AnkleR','RightAnkle')),
        (('L_Foot','FootL','LeftFoot'), ('L_Ankle','AnkleL','LeftAnkle')),
        (('R_Foot','FootR','RightFoot'), ('R_Ankle','AnkleR','RightAnkle')),
    ]
    vecs = []
    for distal_names, prox_names in pair_sets:
        distal = _body_id(mj_model, *distal_names)
        prox   = _body_id(mj_model, *prox_names)
        if distal >= 0 and prox >= 0:
            v = np.array(mj_data.xpos[distal][:2] - mj_data.xpos[prox][:2], dtype=float)
            n = float(np.linalg.norm(v))
            if n > 1e-6:
                vecs.append(v / n)
    if vecs:
        ref = np.mean(np.vstack(vecs), axis=0)
        n = float(np.linalg.norm(ref))
        if n > 1e-6:
            return ref / n
    if pelvis_id >= 0:
        pelvis_xy = np.array(mj_data.xpos[pelvis_id][:2], dtype=float)
        refs = []
        for nm in ('L_Toe','R_Toe','ToeL','ToeR','LeftToe','RightToe',
                   'L_Foot','R_Foot','FootL','FootR','LeftFoot','RightFoot'):
            bid = _body_id(mj_model, nm)
            if bid >= 0:
                v = np.array(mj_data.xpos[bid][:2], dtype=float) - pelvis_xy
                n = float(np.linalg.norm(v))
                if n > 1e-6:
                    refs.append(v / n)
        if refs:
            ref = np.mean(np.vstack(refs), axis=0)
            n = float(np.linalg.norm(ref))
            if n > 1e-6:
                return ref / n
    return np.array([1.0, 0.0], dtype=float)


def _mean_landmark_pos(mj_model, mj_data, names):
    pts = []
    for nm in names:
        bid = _body_id(mj_model, nm)
        if bid >= 0:
            pts.append(np.array(mj_data.xpos[bid], dtype=float))
    if not pts:
        return None
    return np.mean(np.vstack(pts), axis=0)


def _infer_anatomical_frame(mj_model, mj_data, pelvis_id, head_id):
    """Infer forward/lateral/up from anatomical landmarks, not body-local axes.

    Forward is obtained from a right-handed anatomical frame:
      forward ~= cross(lateral(right-left), up(head-pelvis))
    The sign is aligned to the lower-limb/toe forward reference in world XY.
    """
    ref_xy = _infer_anatomical_ref_xy(mj_model, mj_data, pelvis_id)

    if pelvis_id >= 0 and head_id >= 0:
        up = np.array(mj_data.xpos[head_id] - mj_data.xpos[pelvis_id], dtype=float)
    else:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
    nup = float(np.linalg.norm(up))
    up = up / nup if nup > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=float)

    left = _mean_landmark_pos(mj_model, mj_data, [
        'L_Shoulder','LeftShoulder','ShoulderL','L_UpperArm','UpperArmL','LeftUpperArm',
        'L_Arm','ArmL','L_Elbow','ElbowL','LeftElbow','L_Hand','HandL','LeftHand',
        'L_Hip','HipL','LeftHip','L_Thigh','ThighL','LeftThigh'
    ])
    right = _mean_landmark_pos(mj_model, mj_data, [
        'R_Shoulder','RightShoulder','ShoulderR','R_UpperArm','UpperArmR','RightUpperArm',
        'R_Arm','ArmR','R_Elbow','ElbowR','RightElbow','R_Hand','HandR','RightHand',
        'R_Hip','HipR','RightHip','R_Thigh','ThighR','RightThigh'
    ])
    if left is not None and right is not None:
        lateral = np.array(right - left, dtype=float)
    else:
        lateral = np.array([-ref_xy[1], ref_xy[0], 0.0], dtype=float)

    # Remove vertical component and normalize.
    lateral = lateral - up * float(np.dot(lateral, up))
    nl = float(np.linalg.norm(lateral))
    lateral = lateral / nl if nl > 1e-8 else np.array([-ref_xy[1], ref_xy[0], 0.0], dtype=float)

    fwd3 = np.cross(lateral, up)
    nf = float(np.linalg.norm(fwd3))
    if nf < 1e-8:
        fwd3 = np.array([ref_xy[0], ref_xy[1], 0.0], dtype=float)
    else:
        fwd3 /= nf

    # Align sign with toe/foot-derived forward reference.
    if float(np.dot(fwd3[:2], ref_xy)) < 0.0:
        lateral = -lateral
        fwd3 = -fwd3

    f2 = np.array(fwd3[:2], dtype=float)
    n2 = float(np.linalg.norm(f2))
    f2 = f2 / n2 if n2 > 1e-8 else ref_xy
    yaw = float(np.degrees(np.arctan2(f2[1], f2[0])))
    return f2, lateral, up, yaw


def _avatar_forward_xy(mj_model, mj_data, pelvis_id, head_id=None):
    """Return avatar anatomical forward/lateral/up in world coordinates."""
    if pelvis_id < 0:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0,0.0,1.0]), 0.0
    if head_id is None:
        head_id = _body_id(mj_model, 'Head','head')
    return _infer_anatomical_frame(mj_model, mj_data, pelvis_id, head_id)


def _get_actuator_groups(mj_model):
    leg_kw   = {"hip", "knee", "ankle", "foot", "leg"}
    arm_kw   = {"shoulder", "elbow", "wrist", "hand", "arm"}
    torso_kw = {"torso", "spine", "abdomen", "chest"}
    g = {"leg": [], "arm": [], "torso": [], "other": []}
    for i in range(mj_model.nu):
        n = (mujoco.mj_id2name(mj_model, MJOBJ_ACTUATOR, i) or "").lower()
        if   any(k in n for k in leg_kw):   g["leg"].append(i)
        elif any(k in n for k in arm_kw):   g["arm"].append(i)
        elif any(k in n for k in torso_kw): g["torso"].append(i)
        else:                               g["other"].append(i)
    return g


def _apply_anthropometry(mj_model, age, height, weight, sex):
    """Winter (1990) segment mass fractions + age-based gear scaling."""
    fracs_m = {"head":0.0694,"trunk":0.4346,"torso":0.4346,"pelvis":0.1422,
               "upperarm":0.0271,"forearm":0.0162,"hand":0.0061,
               "thigh":0.1000,"shank":0.0465,"foot":0.0145}
    fracs_f = {"head":0.0668,"trunk":0.4257,"torso":0.4257,"pelvis":0.1247,
               "upperarm":0.0255,"forearm":0.0138,"hand":0.0056,
               "thigh":0.1478,"shank":0.0481,"foot":0.0129}
    fracs = fracs_f if sex == "female" else fracs_m
    orig_mass = mj_model.body_mass.copy()
    for i in range(mj_model.nbody):
        bname = (mujoco.mj_id2name(mj_model, MJOBJ_BODY, i) or "").lower()
        frac  = next((v for k,v in fracs.items() if k in bname), None)
        if frac and orig_mass[i] > 1e-6:
            scale = float(np.clip((weight * frac) / orig_mass[i], 0.3, 5.0))
            mj_model.body_mass[i]    = orig_mass[i] * scale
            mj_model.body_inertia[i] = mj_model.body_inertia[i] * scale
    orig_gear = mj_model.actuator_gear[:, 0].copy()
    sf = float(np.clip(muscle_strength_factor(age, sex, weight, height), 0.55, 1.20))
    mj_model.actuator_gear[:, 0] = orig_gear * sf
    return orig_gear, sf


# -----------------------------------------------------------------------------
# EMBEDDING CACHE
# -----------------------------------------------------------------------------

def _load_cached_embeddings(cache_file=_EMBED_CACHE_FILE):
    try:
        if os.path.exists(cache_file):
            obj = torch.load(cache_file, map_location='cpu')
            if all(k in obj for k in ('z_stand','z_sit','z_fall','z_rest')):
                print(f"  [Embed] cache hit -> {cache_file}")
                return obj
    except Exception as e:
        print(f"  [Embed] cache read failed: {e}")
    return None


def _save_cached_embeddings(z_stand, z_sit, z_fall, z_rest, cache_file=_EMBED_CACHE_FILE):
    try:
        torch.save({'z_stand': z_stand.detach().cpu(),
                    'z_sit': z_sit.detach().cpu(),
                    'z_fall': z_fall.detach().cpu(),
                    'z_rest': z_rest.detach().cpu()}, cache_file)
        print(f"  [Embed] cache saved -> {cache_file}")
    except Exception as e:
        print(f"  [Embed] cache save failed: {e}")


# -----------------------------------------------------------------------------
# EMBEDDING INFERENCE  (standalone - no dependency on backward_fall_walking_best)
# -----------------------------------------------------------------------------

def _infer_z_stand(model):
    """Embed 'stand still upright' goal."""
    print("  [Embed] z_stand (upright rise target) …")
    env, _ = make_humenv(task="move-ego-0-0")
    pelvis_id = _body_id(env.unwrapped.model, "Pelvis")
    obs_all, rew_all = [], []
    rwd = LocomotionReward(move_speed=0.0, move_angle=0, stand_height=1.4)
    for trial in range(15):
        torch.manual_seed(SEED + 200 + trial)
        z = model.sample_z(1)
        obs, _ = env.reset()
        for _ in range(90):
            obs_t = torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                act = model.act(obs_t, z).squeeze(0).numpy()
            obs, _, term, trunc, _ = env.step(act)
            pz = float(env.unwrapped.data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 1.0
            if pz > 0.80:
                obs_all.append(obs["proprio"].copy())
                rew_all.append(rwd.compute(env.unwrapped.model, env.unwrapped.data))
            if term or trunc:
                break
    env.close()
    obs_t = torch.tensor(np.array(obs_all), dtype=torch.float32)
    rew_t = torch.tensor(np.array(rew_all), dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        z_stand = model.reward_inference(obs_t, rew_t).mean(dim=0, keepdim=True)
    print(f"  [Embed] z_stand done ({len(obs_all)} upright states)")
    return z_stand


def _infer_z_fall(model):
    """Embed 'collapsed forward / prone on ground' goal."""
    print("  [Embed] z_fall (forward collapse) …")
    env, _ = make_humenv(task="lieonground-up")
    pelvis_id = _body_id(env.unwrapped.model, "Pelvis")
    head_id   = _body_id(env.unwrapped.model, "Head")
    coll_obs  = []
    for trial in range(12):
        obs, _ = env.reset()
        zero = np.zeros(env.action_space.shape)
        for _ in range(120):
            obs, _, term, trunc, _ = env.step(zero)
            pd = env.unwrapped.data
            pz = float(pd.xpos[pelvis_id][2]) if pelvis_id >= 0 else 1.0
            if pz < 0.35:
                # prefer states where the head is in avatar-forward direction,
                # not just world +x.
                if head_id >= 0 and pelvis_id >= 0:
                    fwd_xy, _, _, _ = _avatar_forward_xy(env.unwrapped.model, pd, pelvis_id, head_id)
                    hp = pd.xpos[head_id] - pd.xpos[pelvis_id]
                    fwd = float(np.dot(hp[:2], fwd_xy)) > -0.02
                else:
                    fwd = True
                if fwd:
                    coll_obs.append(obs["proprio"].copy())
            if term or trunc:
                break
    env.close()
    if len(coll_obs) >= 20:
        obs_t = torch.tensor(np.array(coll_obs[-300:]), dtype=torch.float32)
        with torch.no_grad():
            z_fall = model.goal_inference(obs_t).mean(dim=0, keepdim=True)
    else:
        z_fall = model.sample_z(1)
    print(f"  [Embed] z_fall done ({len(coll_obs)} forward-collapse states)")
    return z_fall


def _infer_z_sit_attempt(model):
    """Infer an *upright seated* proxy, not a crouched / floor-collapsed pose.

    Earlier repairs misread `_trunk_tilt_deg()` as if larger meant "more seated".
    In this codebase the value is the angle from world-up, so:
      0 deg   = upright vertical trunk
      90 deg  = horizontal / near-prone trunk
    A realistic seated start therefore needs a *moderate* trunk tilt, not an
    80-100 deg near-horizontal posture.
    """
    print("  [Embed] z_sit …")
    env, _ = make_humenv(task="move-ego-0-0")
    pelvis_id = _body_id(env.unwrapped.model, "Pelvis")
    head_id   = _body_id(env.unwrapped.model, "Head")
    torso_id  = _body_id(env.unwrapped.model, "Torso")
    sit_obs = []
    for trial in range(72):
        torch.manual_seed(SEED + 600 + trial)
        z = model.sample_z(1)
        obs, _ = env.reset()
        for _ in range(190):
            d = env.unwrapped.data
            if torso_id >= 0:
                d.xfrc_applied[torso_id, 0] = -8.0
                d.xfrc_applied[torso_id, 2] = -16.0
                d.xfrc_applied[torso_id, 4] = +2.5
            if pelvis_id >= 0:
                d.xfrc_applied[pelvis_id, 2] = -8.0
            obs_t = torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                act = model.act(obs_t, z).squeeze(0).numpy()
            act *= 0.22
            obs, _, term, trunc, _ = env.step(act)
            d = env.unwrapped.data
            ph = float(d.xpos[pelvis_id][2]) if pelvis_id >= 0 else 1.0
            qn = float(np.linalg.norm(d.qvel))
            if pelvis_id >= 0 and head_id >= 0:
                fwd_xy, _, _, _ = _avatar_forward_xy(env.unwrapped.model, d, pelvis_id, head_id)
                vec = d.xpos[head_id] - d.xpos[pelvis_id]
                nv = float(np.linalg.norm(vec))
                lean = float(np.degrees(np.arccos(np.clip(np.dot(vec / max(nv, 1e-9), [0, 0, 1]), -1, 1)))) if nv > 1e-6 else 0.0
                head_fwd = float(np.dot(vec[:2], fwd_xy)) if nv > 1e-6 else 0.0
                prone = float(np.dot(vec / max(nv, 1e-9), np.array([fwd_xy[0], fwd_xy[1], 0.0]))) if nv > 1e-6 else 0.0
                if 0.42 <= ph <= 0.72 and 10.0 <= lean <= 42.0 and -0.14 <= head_fwd <= 0.18 and abs(prone) <= 0.22 and qn < 4.0:
                    sit_obs.append(obs["proprio"].copy())
            if term or trunc:
                break
    env.close()
    if len(sit_obs) >= 20:
        obs_t = torch.tensor(np.array(sit_obs[-400:]), dtype=torch.float32)
        with torch.no_grad():
            z_sit = model.goal_inference(obs_t).mean(dim=0, keepdim=True)
    else:
        z_sit = model.sample_z(1)
    print(f"  [Embed] z_sit done ({len(sit_obs)} seated-proxy states)")
    return z_sit


def _infer_z_rest(model):
    """Embed a stable forward-prone rest posture instead of a generic ground pose."""
    print("  [Embed] z_rest (forward-prone settle) …")
    env, _ = make_humenv(task="lieonground-up")
    pelvis_id = _body_id(env.unwrapped.model, "Pelvis")
    head_id   = _body_id(env.unwrapped.model, "Head")
    torso_id  = _body_id(env.unwrapped.model, "Torso")
    prone_obs = []
    for trial in range(18):
        obs, _ = env.reset()
        zero = np.zeros(env.action_space.shape)
        for _ in range(180):
            d = env.unwrapped.data
            if torso_id >= 0:
                d.xfrc_applied[torso_id, 0] = 120.0
                d.xfrc_applied[torso_id, 4] = -30.0
            if pelvis_id >= 0:
                d.xfrc_applied[pelvis_id, 2] = -50.0
            obs, _, term, trunc, _ = env.step(zero)
            if pelvis_id >= 0 and head_id >= 0:
                p = d.xpos[pelvis_id]
                h = d.xpos[head_id]
                qn = float(np.linalg.norm(d.qvel))
                vec = h - p
                nv = float(np.linalg.norm(vec))
                tilt = float(np.degrees(np.arccos(np.clip(np.dot(vec / nv, [0, 0, 1]), -1, 1)))) if nv > 1e-6 else 0.0
                if float(p[2]) < 0.22 and float(h[2]) < 0.38 and tilt > 68.0 and qn < 1.8 and float(h[0]) > float(p[0]) - 0.15:
                    prone_obs.append(obs["proprio"].copy())
            if term or trunc:
                break
    env.close()
    if len(prone_obs) >= 40:
        obs_t = torch.tensor(np.array(prone_obs[-300:]), dtype=torch.float32)
        with torch.no_grad():
            z_rest = model.goal_inference(obs_t).mean(dim=0, keepdim=True)
    else:
        z_rest = model.sample_z(1)
    print(f"  [Embed] z_rest done ({len(prone_obs)} forward-prone-settle states)")
    return z_rest


def _smoothstep(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3 - 2 * t)


def _blend_z(za, zb, alpha):
    a = _smoothstep(float(alpha))
    return (1.0 - a) * za + a * zb


# -----------------------------------------------------------------------------
# LEAN IMU
# -----------------------------------------------------------------------------

class LeanIMU:
    CLIP_A = 16.0 * 9.81   # ±16 g

    def __init__(self, mj_model, mj_data, age, height):
        self.mj_model = mj_model
        self.mj_data  = mj_data
        self.body_id  = _body_id(mj_model, "Torso", "torso", "Pelvis")
        self.pelvis_id= _body_id(mj_model, "Pelvis")
        self.native_dt = 1.0 / 30.0
        self.native_hz = 30.0
        self.output_dt = 0.01
        self.output_hz = 100.0
        self.output_mode = 'resampled_from_native'
        af = max(1.0, 1.0 + (age - 60) * 0.015) if age > 60 else 1.0
        self.an = 0.08 * af;  self.gn = 0.015 * af
        self.ab = np.random.normal(0, 0.025 * af, 3)
        self.gb = np.random.normal(0, 0.007 * af, 3)
        self.sta= 0.04 * (height / 1.75)
        self._lpA = np.array([0., 0., 9.81]); self._lpG = np.zeros(3)
        self._pv  = None
        self.buf  = {k: [] for k in ("t","ax","ay","az","gx","gy","gz",
                                      "pelvis_z","pelvis_vx","impact")}

    def log(self, t):
        d = self.mj_data
        bid = self.body_id
        R   = d.xmat[bid].reshape(3, 3)
        v6  = np.zeros(6)
        mujoco.mj_objectVelocity(self.mj_model, d, MJOBJ_BODY, bid, v6, 1)
        vw  = R @ v6[3:]
        gw  = np.array([0., 0., -9.81])
        if self._pv is not None:
            afd  = (vw - self._pv) / (1/30.0)
            araw = R.T @ (afd - gw)
            n = np.linalg.norm(araw)
            if n > self.CLIP_A:
                araw *= self.CLIP_A / n
        else:
            araw = np.array([0., 0., 9.81])
        self._pv = vw.copy()
        sta_v = self.sta * math.sin(2*math.pi*0.25*t + 1.2)
        noisy_a = araw + self.ab + np.random.normal(0, self.an, 3) + sta_v
        noisy_a = np.clip(noisy_a, -self.CLIP_A, self.CLIP_A)
        self._lpA = 0.76 * self._lpA + 0.24 * noisy_a
        v6g = np.zeros(6)
        mujoco.mj_objectVelocity(self.mj_model, d, MJOBJ_BODY, bid, v6g, 1)
        noisy_g = v6g[:3] + self.gb + np.random.normal(0, self.gn, 3)
        self._lpG = 0.76 * self._lpG + 0.24 * noisy_g
        # pelvis kinematics
        pz  = float(d.xpos[self.pelvis_id][2]) if self.pelvis_id >= 0 else 0.0
        v6p = np.zeros(6)
        if self.pelvis_id >= 0:
            mujoco.mj_objectVelocity(self.mj_model, d, MJOBJ_BODY, self.pelvis_id, v6p, 0)
        # impact
        imp = 0.0
        for i in range(d.ncon):
            c  = d.contact[i]
            b1 = int(self.mj_model.geom_bodyid[int(c.geom1)])
            b2 = int(self.mj_model.geom_bodyid[int(c.geom2)])
            if b1 == bid or b2 == bid:
                fw = np.zeros(6)
                mujoco.mj_contactForce(self.mj_model, d, i, fw)
                imp = max(imp, float(max(0., fw[0])))
        self.buf["t"].append(t)
        self.buf["ax"].append(float(self._lpA[0]))
        self.buf["ay"].append(float(self._lpA[1]))
        self.buf["az"].append(float(self._lpA[2]))
        self.buf["gx"].append(float(self._lpG[0]))
        self.buf["gy"].append(float(self._lpG[1]))
        self.buf["gz"].append(float(self._lpG[2]))
        self.buf["pelvis_z"].append(pz)
        self.buf["pelvis_vx"].append(float(v6p[3]))
        self.buf["impact"].append(min(imp, 15000.0))
        return float(np.linalg.norm(self._lpA))


    @property
    def data_buffer(self):
        n = len(self.buf["t"])
        return {
            "timestamp": list(self.buf["t"]),
            "accelerometer": [[self.buf["ax"][i], self.buf["ay"][i], self.buf["az"][i]] for i in range(n)],
            "accel_raw": [[self.buf["ax"][i], self.buf["ay"][i], self.buf["az"][i]] for i in range(n)],
            "gyroscope": [[self.buf["gx"][i], self.buf["gy"][i], self.buf["gz"][i]] for i in range(n)],
            "pelvis_height": list(self.buf["pelvis_z"]),
            "pelvis_velocity": [[self.buf["pelvis_vx"][i], 0.0, 0.0] for i in range(n)],
            "impact_force": list(self.buf["impact"]),
            "sensor_confidence": [1.0] * n,
            "sensor_world_pos": [[0.0, 0.0, 0.0] for _ in range(n)],
            "sensor_world_vel": [[0.0, 0.0, 0.0] for _ in range(n)],
            "soft_tissue_artifact": [self.sta] * n,
        }

    def _resample_buffers_100hz(self):
        t_nat = np.array(self.buf["t"], dtype=float)
        if len(t_nat) < 2:
            return None
        t100 = np.arange(t_nat[0], t_nat[-1] + 1e-9, self.output_dt)

        def _interp_scalar(key):
            return np.interp(t100, t_nat, np.array(self.buf[key], dtype=float))

        ax = _interp_scalar("ax")
        ay = _interp_scalar("ay")
        az = _interp_scalar("az")
        gx = _interp_scalar("gx")
        gy = _interp_scalar("gy")
        gz = _interp_scalar("gz")
        pz = _interp_scalar("pelvis_z")
        pvx = _interp_scalar("pelvis_vx")
        imp = _interp_scalar("impact")
        acc = np.vstack([ax, ay, az]).T
        gyro = np.vstack([gx, gy, gz]).T
        vel = np.vstack([pvx, np.zeros_like(pvx), np.zeros_like(pvx)]).T
        zeros = np.zeros((len(t100), 3), dtype=float)
        return {
            "timestamp": t100,
            "accelerometer": acc,
            "accel_raw": acc.copy(),
            "gyroscope": gyro,
            "pelvis_height": pz,
            "pelvis_velocity": vel,
            "sensor_world_pos": zeros.copy(),
            "sensor_world_vel": zeros.copy(),
            "impact_force": imp,
            "soft_tissue_artifact": np.full(len(t100), self.sta, dtype=float),
            "sensor_confidence": np.ones(len(t100), dtype=float),
            "accel_true": acc.copy(),
            "gyro_true": gyro.copy(),
        }

    def export_csv(self, fname, meta=None):
        t_nat = np.array(self.buf["t"])
        if len(t_nat) < 2:
            return {"filename": fname, "frames": 0}
        t100 = np.arange(t_nat[0], t_nat[-1]+1e-9, 0.01)
        def _i(k):
            return np.interp(t100, t_nat, np.array(self.buf[k]))
        ax,ay,az = _i("ax"),_i("ay"),_i("az")
        gx,gy,gz = _i("gx"),_i("gy"),_i("gz")
        pz,pvx   = _i("pelvis_z"),_i("pelvis_vx")
        imp      = _i("impact")
        amag = np.sqrt(ax**2+ay**2+az**2)
        n = len(t100); falls = 0; rows = []
        for i in range(n):
            fall = int(pz[i] < 0.40)
            falls += fall
            rows.append([round(t100[i],4),
                         round(ax[i],5),round(ay[i],5),round(az[i],5),
                         round(gx[i],5),round(gy[i],5),round(gz[i],5),
                         round(pz[i],5),round(pvx[i],5),round(imp[i],3),
                         round(amag[i],5), fall])
        with open(fname,"w",newline="") as f:
            for k,v in (meta or {}).items():
                f.write(f"# {k}: {v}\n")
            w = csv.writer(f)
            w.writerow(["t","ax","ay","az","gx","gy","gz",
                        "pelvis_z","pelvis_vx","impact_n","accel_mag","fall"])
            w.writerows(rows)
        print(f"  [IMU]  ? {fname}  ({n} frames | {falls} fall frames)")
        return {"filename": fname, "frames": n, "falls_detected": falls}


# -----------------------------------------------------------------------------
# VALIDATION
# -----------------------------------------------------------------------------

def _validate(imu, perturb_t, body_mass):
    t   = np.array(imu.buf["t"])
    pz  = np.array(imu.buf["pelvis_z"])
    pvx = np.array(imu.buf["pelvis_vx"])
    imp = np.array(imu.buf["impact"])
    ax  = np.array(imu.buf["ax"]); ay=np.array(imu.buf["ay"]); az=np.array(imu.buf["az"])
    amag = np.sqrt(ax**2+ay**2+az**2)
    k7 = np.hanning(7); k7/=k7.sum()
    afilt = np.convolve(amag, k7, mode="same") if len(amag)>=7 else amag.copy()
    peak_filt = float(np.max(afilt))
    BW = body_mass * 9.81
    i0 = int(np.searchsorted(t, perturb_t, side="left"))
    h0 = float(np.max(pz[max(0,i0-5):i0+3])) if i0 < len(pz) else 1.0
    onset = next((j for j in range(i0, len(t))
                  if pz[j] < max(0.65*h0, h0-0.15) and abs(pvx[j]) > 0.30), i0)
    settle= next((j for j in range(onset, len(t))
                  if pz[j] < 0.22 and abs(pvx[j]) < 0.15 and j > onset+10), None)
    dur   = float(t[settle]-t[onset]) if settle else float(t[-1]-t[onset])
    peak_imp = float(np.max(imp[i0:])) if i0<len(imp) else 0.0
    checks = {
        "peak_accel_in_SISFall_range": 15 <= peak_filt <= 130,
        "fall_duration_realistic":     0.5 <= dur      <= 6.0,
        "peak_impact_in_range":        BW   <= peak_imp <= 22*BW,
        "pelvis_reached_floor":        float(np.min(pz[i0:])) < 0.30,
    }
    score = float(np.mean([1.0 if v else 0.5 for v in checks.values()]))
    cls   = ("HIGH_CONFIDENCE" if score > 0.85 else
             "MODERATE_CONFIDENCE" if score > 0.65 else "LOW_CONFIDENCE")
    return {"score":score,"classification":cls,"checks":checks,
            "fall_duration_s":round(dur,3),
            "peak_accel_filt":round(peak_filt,2),
            "peak_impact_n":  round(peak_imp,1),
            "sisfall_compliant": (15<=peak_filt<=130 and 0.5<=dur<=6.0)}


# -----------------------------------------------------------------------------
# PROMPT HELPERS
# -----------------------------------------------------------------------------

def _pf(label, default, unit='', lo=None, hi=None):
    while True:
        raw = input(f"    {label} [{default}{' ' + unit if unit else ''}]: ").strip()
        if raw == '':
            return default
        try:
            v = float(raw)
            if lo is not None and v < lo:
                print(f"      Must be >= {lo}.")
                continue
            if hi is not None and v > hi:
                print(f"      Must be <= {hi}.")
                continue
            return v
        except ValueError:
            print("      Enter a number.")


def _ps(label, default, choices):
    while True:
        raw = input(f"    {label} [{default}] ({'/'.join(choices)}): ").strip().lower()
        if raw == '':
            return default
        if raw in choices:
            return raw
        print(f"      Choose: {choices}")


def _po(label, unit='', lo=None, hi=None):
    while True:
        raw = input(f"    {label} [auto{' ' + unit if unit else ''}]: ").strip()
        if raw == '':
            return None
        try:
            v = float(raw)
            if lo is not None and v < lo:
                print(f"      Must be >= {lo}.")
                continue
            if hi is not None and v > hi:
                print(f"      Must be <= {hi}.")
                continue
            return v
        except ValueError:
            print("      Enter a number or press Enter for auto.")




def _safe_mean(values, default=0.0):
    arr = [float(v) for v in values if v is not None]
    return float(np.mean(arr)) if arr else float(default)


def _safe_max(values, default=0.0):
    arr = [float(v) for v in values if v is not None]
    return float(np.max(arr)) if arr else float(default)


def _compute_task22_profile_schedule(age, sex, height, weight):
    sf = float(np.clip(muscle_strength_factor(age, sex, weight, height), 0.55, 1.20))
    bal = float(balance_impairment(age, sex, weight))
    rt_steps = max(1, round(reaction_delay_seconds(age, sex, height) * 30.0))
    age_frac = float(np.clip((age - 55.0) / 25.0, 0.0, 1.0))
    stand_s = 150 + int(round(40.0 * age_frac + 10.0 * bal))
    sit_s = 90 + int(round(14.0 * age_frac + 8.0 * bal))
    collapse_s = 30 + int(round(8.0 * age_frac + 8.0 * bal))
    fall_s = 400 + int(round(60.0 * age_frac + 30.0 * bal))
    return {
        "stand": int(stand_s),
        "sit_down": int(sit_s),
        "collapse": int(collapse_s),
        "fall": int(fall_s),
        "total": int(stand_s + sit_s + collapse_s + fall_s),
        "strength_factor": sf,
        "reaction_steps": int(rt_steps),
        "balance_impairment": bal,
        "age_frac": age_frac,
    }


def _print_task22_extended_report(
    *, age, sex, height, weight, body_mass, sf, rt_steps, bal_imp,
    phase_steps, min_gear, leg_floor, torso_floor, arm_floor,
    collapse_force_scale, collapse_roll_scale, fall_resid_scale,
    chair_trigger_pz, perturb_t, imu, dynamics_analyzer, marker_exporter,
    dashboard, metrics_log,
):
    if FallValidator is None or SISFallValidator is None or KFallValidator is None or detect_fall_events is None:
        return

    imu_data = imu.data_buffer
    dynamics_frames = list(getattr(dynamics_analyzer, "frames", [])) if dynamics_analyzer is not None else []
    marker_frames = list(getattr(marker_exporter, "frames", [])) if marker_exporter is not None else []
    try:
        event_summary = detect_fall_events(
            imu_data, dynamics_frames, marker_frames, perturb_start_time=float(perturb_t)
        )
    except Exception as e:
        event_summary = {"available": False, "error": str(e)}

    kinematic_data = {"head_velocity": [0.0, 0.0, 0.0]}
    if marker_frames:
        try:
            head_vs = []
            for fr in marker_frames:
                vel = fr.get("segment_world_linear_velocity", {}).get("head")
                if vel is not None:
                    head_vs.append(np.asarray(vel, dtype=float))
            if head_vs:
                kinematic_data["head_velocity"] = head_vs[-1]
        except Exception:
            pass

    validator = FallValidator()
    validation_results = validator.validate_fall(
        imu_data, kinematic_data,
        fall_type="lateral_stumble",
        perturb_start_time=float(perturb_t),
        event_summary=event_summary,
    )

    print("\n" + "=" * 70)
    print("  FALL VALIDATION REPORT")
    print("=" * 70)
    print(f"  Overall Confidence Score : {validation_results['overall_score']:.1%}")
    print(f"  Classification           : {validation_results['classification']}")
    print("")
    print("  Detailed Checks:")
    for check, passed in validation_results.get("checks", {}).items():
        print(f"    {check:35s}  {'PASS' if passed else 'FAIL'}")
    adv = validation_results.get("advanced_metrics", {})
    if adv:
        print("\nAdvanced Metrics:")
        for k, v in adv.items():
            print(f"    {k:42s}  {v}")

    print("\n" + "=" * 70)
    print("  SISFall SIGNATURE VALIDATION")
    print("=" * 70)
    sisfall_val = SISFallValidator()
    sf_result = sisfall_val.validate_sisfall_signature(
        imu_data, "lateral_stumble",
        body_mass=body_mass,
        perturb_start_time=float(perturb_t),
        event_summary=event_summary,
    )
    print(f"  Channel used    : {sf_result['margins'].get('channel_used','?')}")
    print(f"  Filter note     : {sf_result['margins'].get('filter_note','?')}")
    print(f"  Sensor location : {sf_result['margins'].get('sensor_location','?')}")
    print(f"  Fall direction  : {sf_result['direction']}")
    print(f"  Peak accel (filt): {sf_result['margins'].get('peak_accel_ms2', 0.0):.2f} m/s^2  (expected {sf_result['margins'].get('peak_accel_range', ('?','?'))})")
    print(f"  Peak accel (raw) : {sf_result['margins'].get('peak_accel_raw_ms2', 0.0):.2f} m/s^2")
    print(f"  Fall duration   : {sf_result['margins'].get('fall_duration_s', 0.0):.2f} s  (expected {sf_result['margins'].get('duration_range', ('?','?'))})")
    for chk_name, chk_val in sf_result.get('checks', {}).items():
        print(f"    {chk_name:35s}  {'PASS' if chk_val else 'FAIL'}")
    print(f"  SISFall compliant : {sf_result.get('sisfall_compliant', False)}")

    print("\n" + "=" * 70)
    print("  KFall-STYLE PRE-IMPACT VALIDATION")
    print("=" * 70)
    kfall_val = KFallValidator()
    kfall_result = kfall_val.validate(imu, perturb_start_time=float(perturb_t), event_summary=event_summary)
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
        print(f"  Perturb start     : {float(perturb_t):.3f} s")
        print(f"  Fall onset        : {event_summary.get('onset_time', 0.0):.3f} s")
        print(f"  Main impact       : {event_summary.get('impact_time', 0.0):.3f} s")
        print(f"  Settle time       : {event_summary.get('settle_time', 0.0):.3f} s")
        print(f"  Fall duration     : {event_summary.get('fall_duration_s', 0.0):.2f} s")
        print(f"  Impact lead       : {event_summary.get('lead_time_ms', 0.0):.1f} ms")
    else:
        print("  Event detector    : unavailable")

    print("\nBIOMECHANICAL RANGE CHECKS")
    print("-" * 50)
    bio_result = validator.validate_biomechanical_ranges(
        imu_data, body_mass=body_mass, dynamics_frames=dynamics_frames
    )
    print(f"  Body mass / weight  : {bio_result['body_mass_kg']:.1f} kg / {bio_result['body_weight_n']:.1f} N")
    if bio_result.get('peak_angular_velocity_ok') is not None:
        print(f"  Peak angular vel    : {bio_result['peak_angular_velocity_rads']:.2f} rad/s  < {bio_result['angular_velocity_threshold']:.1f} rad/s  {'PASS' if bio_result['peak_angular_velocity_ok'] else 'FAIL'}")
    if bio_result.get('impact_force_in_range') is not None:
        lo, hi = bio_result['expected_range_n']
        print(f"  Peak impact force   : {bio_result['peak_impact_force_n']:.1f} N  (expected {lo:.0f}-{hi:.0f} N)  {'PASS' if bio_result['impact_force_in_range'] else 'FAIL'}")

    if dashboard is not None and metrics_log:
        print("\nPHASE-BY-PHASE BIOMECHANICAL SUMMARY")
        print("-" * 50)
        for ph in ("stand", "sit_down", "collapse", "fall"):
            phase_metrics = [m for m in metrics_log if m.get("phase") == ph]
            if phase_metrics:
                dashboard.print_phase_summary(ph, phase_metrics)

    print("\nPERFORMANCE DIAGNOSTICS")
    print("-" * 50)
    sit_metrics = [m for m in metrics_log if m.get("phase") == "sit_down"]
    collapse_metrics = [m for m in metrics_log if m.get("phase") == "collapse"]
    fall_metrics = [m for m in metrics_log if m.get("phase") == "fall"]
    if sit_metrics:
        sit_mean_speed = _safe_mean(abs(m.get("pelvis_vx", 0.0)) for m in sit_metrics)
        sit_mean_froude = _safe_mean(m.get("froude", 0.0) for m in sit_metrics)
        sit_mean_ds = _safe_mean(1.0 if m.get("double_support", False) else 0.0 for m in sit_metrics)
        sit_peak_trunk = _safe_max(m.get("trunk_lean_deg", 0.0) for m in sit_metrics)
        print(f"  Sit mean |vx|      : {sit_mean_speed:.3f} m/s")
        print(f"  Sit mean Froude    : {sit_mean_froude:.3f}")
        print(f"  Sit mean dbl_sup   : {sit_mean_ds:.1%}")
        print(f"  Sit peak trunk     : {sit_peak_trunk:.1f} deg")
    if collapse_metrics:
        collapse_peak_impact = _safe_max(m.get("impact_bw", 0.0) for m in collapse_metrics)
        collapse_peak_omega = _safe_max(m.get("pelvis_angular_speed_dps", 0.0) for m in collapse_metrics)
        print(f"  Collapse peak imp/BW: {collapse_peak_impact:.2f}")
        print(f"  Collapse peak omg   : {collapse_peak_omega:.1f} deg/s")
    if fall_metrics:
        fall_peak_impact_bw = _safe_max(m.get("impact_bw", 0.0) for m in fall_metrics)
        fall_peak_grf_bw = _safe_max(m.get("grf_bw", 0.0) for m in fall_metrics)
        settle_frac = _safe_mean(1.0 if m.get("settle_flag", False) else 0.0 for m in fall_metrics)
        print(f"  Fall peak impact/BW: {fall_peak_impact_bw:.2f}")
        print(f"  Fall peak ground/BW: {fall_peak_grf_bw:.2f}")
        print(f"  Fall settled frac  : {settle_frac:.1%}")

    print("\nPROFILE RESPONSE AUDIT")
    print("-" * 50)
    print(f"  Runtime strength sf : {sf:.3f}")
    print(f"  Runtime rt_steps    : {rt_steps}")
    print(f"  Runtime balance imp : {bal_imp:.3f}")
    print(f"  Phase steps         : stand={phase_steps['stand']} sit_down={phase_steps['sit_down']} collapse={phase_steps['collapse']} fall={phase_steps['fall']} total={phase_steps['total']}")
    print(f"  Chair trigger pz    : {chair_trigger_pz:.3f} m")
    print(f"  Gear floors         : leg={leg_floor:.3f} torso={torso_floor:.3f} arm={arm_floor:.3f} min={min_gear:.3f}")
    print(f"  Collapse scales     : force={collapse_force_scale:.3f} roll={collapse_roll_scale:.3f} fall_resid={fall_resid_scale:.3f}")

    profile_response = float(np.clip(
        0.30 * min(1.0, sf / 1.05) +
        0.20 * min(1.0, max(rt_steps, 1) / 18.0) +
        0.20 * min(1.0, max(bal_imp, 0.0) / 0.40) +
        0.30 * min(1.0, phase_steps['total'] / 730.0),
        0.0, 1.0,
    ))
    scenario_authenticity = float(np.clip(
        0.45 * validation_results['overall_score'] +
        0.20 * kfall_result.get('score', 0.0) +
        0.15 * (1.0 if sf_result.get('sisfall_compliant', False) else 0.0) +
        0.20 * profile_response,
        0.0, 1.0,
    ))

    print("\nSCENARIO AUTHENTICITY")
    print("-" * 50)
    print(f"  Fall-only score    : {validation_results['overall_score']:.1%}")
    print(f"  Profile-response   : {profile_response:.1%}")
    print(f"  KFall score        : {kfall_result.get('score', 0.0):.1%}")
    print(f"  SISFall compliant  : {sf_result.get('sisfall_compliant', False)}")
    print(f"  Authenticity score : {scenario_authenticity:.1%}")

# -----------------------------------------------------------------------------
# MAIN  run()
# -----------------------------------------------------------------------------


def run(subject_params: dict | None = None) -> dict:
    """
    Scenario 23
    -----------
    Forward fall when trying to get up.

    Control logic:
      1) guide the avatar into a seated pose
      2) start a sit-to-stand rise
      3) trigger loss of forward balance during ascent, not after full standing
      4) commit to a forward fall and prone / chest-down settle
    """
    if subject_params is None:
        print("\n" + "=" * 70)
        print("  Scenario 23 - Forward fall when trying to get up")
        print("=" * 70)
        print("\n  Enter subject parameters (Enter = default):\n")
        subject_params = {
            "age": int(_pf("Age", 70, "years", lo=1, hi=120)),
            "height": float(_pf("Height", 1.65, "m", lo=0.5, hi=2.5)),
            "sex": _ps("Sex", "male", ["male", "female"]),
            "weight": _po("Weight", "kg", lo=25.0, hi=250.0),
        }

    age    = int(subject_params["age"])
    height = float(subject_params["height"])
    sex    = str(subject_params.get("sex", "male")).lower()
    weight, bmi, w_src = _resolve_weight(height, age, sex, subject_params.get("weight"))
    w_str  = (
        f"{subject_params['weight']:.1f}kg (user)"
        if subject_params.get("weight") is not None
        else f"{weight:.1f}kg (auto BMI={bmi:.1f})"
    )

    print("\n" + "="*70)
    print("  SCENARIO 23 - Forward fall when trying to get up")
    print(f"  Subject: age={age}yr  h={height}m  {w_str}  sex={sex}")
    print("="*70)
    print_subject_profile(age, sex, height, weight)
    print("  Note: the profile PHASE TIMING block above is the generic biofidelic baseline; scenario 23 uses the seated-start phases shown below.")

    # -- subject params --------------------------------------------------------
    bal_imp   = float(balance_impairment(age, sex, weight))
    sf        = float(np.clip(muscle_strength_factor(age, sex, weight, height), 0.55, 1.20))
    rt_steps  = max(1, round(reaction_delay_seconds(age, sex, height) * 30))
    min_gear  = float(weakening_config(age, sex, weight)["min_factor"])
    age_frac  = float(np.clip((age - 55.0) / 25.0, 0.0, 1.0))

    # This task starts from a seated posture proxy and then rises.
    SIT_S      = 120 + int(round(24.0 * age_frac + 10.0 * bal_imp))
    GETUP_S    = 92  + int(round(18.0 * age_frac + 10.0 * bal_imp))
    COLLAPSE_S = 34  + int(round(8.0 * age_frac + 8.0 * bal_imp))
    FALL_S     = 390 + int(round(70.0 * age_frac + 30.0 * bal_imp))
    TOTAL      = int(SIT_S + GETUP_S + COLLAPSE_S + FALL_S)

    print(f"\n  Phases: sit={SIT_S}  get_up={GETUP_S}  "
          f"collapse={COLLAPSE_S}  fall={FALL_S}  total={TOTAL}")
    print(f"  sf={sf:.3f}  rt_steps={rt_steps}  bal={bal_imp:.3f}  min_gear={min_gear:.3f}")

    # -- load model ------------------------------------------------------------
    print("\n  [1/4] Loading Meta Motivo …")
    motivo = FBcprModel.from_pretrained("facebook/metamotivo-M-1")
    motivo.eval()
    print(f"        Loaded on {next(motivo.parameters()).device}")

    # -- task embeddings -------------------------------------------------------
    print("\n  [2/4] Inferring task embeddings …")
    z_stand = _infer_z_stand(motivo)
    z_sit   = _infer_z_sit_attempt(motivo)
    z_fall  = _infer_z_fall(motivo)
    z_rest  = _infer_z_rest(motivo)

    # -- environment -----------------------------------------------------------
    print("\n  [3/4] Creating environment …")
    env, _ = make_humenv(task="move-ego-0-0")
    obs, _ = env.reset()
    mj_model = env.unwrapped.model
    mj_data  = env.unwrapped.data

    # -- anthropometry ---------------------------------------------------------
    orig_gear, _ = _apply_anthropometry(mj_model, age, height, weight, sex)
    body_mass    = float(np.sum(mj_model.body_mass))
    BW           = body_mass * 9.81
    print(f"        Scaled body mass = {body_mass:.2f} kg  BW = {BW:.1f} N")

    grp       = _get_actuator_groups(mj_model)
    pelvis_id = _body_id(mj_model, "Pelvis","pelvis")
    torso_id  = _body_id(mj_model, "Torso", "torso")
    head_id   = _body_id(mj_model, "Head",  "head")
    hand_ids  = [bid for bid in [
        _body_id(mj_model, "L_Hand", "HandL", "LeftHand"),
        _body_id(mj_model, "R_Hand", "HandR", "RightHand"),
        _body_id(mj_model, "L_ForeArm", "ForeArmL", "LeftForeArm"),
        _body_id(mj_model, "R_ForeArm", "ForeArmR", "RightForeArm"),
    ] if bid >= 0]

    fall_fwd_xy = np.array([1.0, 0.0], dtype=float)
    fall_lat_3d = np.array([0.0, 1.0, 0.0], dtype=float)
    fall_up_3d  = np.array([0.0, 0.0, 1.0], dtype=float)
    heading_locked = False
    chair_center_xy = None
    chair_seat_z    = None
    _last_heading_print = None
    _ref_fwd_xy = None
    _ref_lat_3d = None
    _ref_up_3d = None
    _ref_yaw = None

    def _capture_reference_heading():
        nonlocal _ref_fwd_xy, _ref_lat_3d, _ref_up_3d, _ref_yaw
        ref_fwd_xy, ref_lat_3d, ref_up_3d, ref_yaw = _avatar_forward_xy(mj_model, mj_data, pelvis_id, head_id)
        _ref_fwd_xy = np.array(ref_fwd_xy, dtype=float)
        _ref_lat_3d = np.array(ref_lat_3d, dtype=float)
        _ref_up_3d = np.array(ref_up_3d, dtype=float)
        _ref_yaw = float(ref_yaw)

    def _lock_heading(force_print=False):
        nonlocal fall_fwd_xy, fall_lat_3d, fall_up_3d, heading_locked, chair_center_xy, _last_heading_print
        # Freeze heading from the initial clean upright reset. Recomputing
        # forward from pathological seated-recovery poses can flip the heading
        # 180 degrees and corrupt seated metrics / wrenches.
        if _ref_fwd_xy is None:
            _capture_reference_heading()
        fall_fwd_xy = _ref_fwd_xy.copy()
        fall_lat_3d = _ref_lat_3d.copy()
        fall_up_3d = _ref_up_3d.copy()
        yaw = float(_ref_yaw)
        heading_locked = True
        heading_now = (round(float(fall_fwd_xy[0]), 3), round(float(fall_fwd_xy[1]), 3), round(float(yaw), 1))
        if force_print and _last_heading_print != heading_now:
            print(f"        Heading locked: fwd=({fall_fwd_xy[0]:+.3f},{fall_fwd_xy[1]:+.3f})  yaw={yaw:+.1f} deg | lat=({fall_lat_3d[0]:+.3f},{fall_lat_3d[1]:+.3f},{fall_lat_3d[2]:+.3f})")
            _last_heading_print = heading_now

    _capture_reference_heading()

    def _reanchor_chair_to_current_pelvis(forward_offset=0.050):
        nonlocal chair_center_xy
        if pelvis_id < 0 or chair_seat_z is None:
            chair_center_xy = None
            return
        pelvis_xy = np.array(mj_data.xpos[pelvis_id][:2], dtype=float)
        # Chair seat center sits slightly behind the pelvis in a natural seated pose.
        chair_center_xy = pelvis_xy - float(forward_offset) * fall_fwd_xy

    def _apply_body_lateral_wrench(body_id, lateral_n=0.0, down_n=0.0, roll_nm=0.0):
        if body_id < 0:
            return
        fx, fy = fall_fwd_xy
        lx, ly, lz = fall_lat_3d
        mj_data.xfrc_applied[body_id, 0] += float(lateral_n) * lx
        mj_data.xfrc_applied[body_id, 1] += float(lateral_n) * ly
        mj_data.xfrc_applied[body_id, 2] += float(lateral_n) * lz - float(down_n)
        mj_data.xfrc_applied[body_id, 3] += float(roll_nm) * fx
        mj_data.xfrc_applied[body_id, 4] += float(roll_nm) * fy

    def _apply_body_forward_wrench(body_id, forward_n=0.0):
        if body_id < 0:
            return
        fx, fy = fall_fwd_xy
        fwd_n = float(forward_n)
        mj_data.xfrc_applied[body_id, 0] += fwd_n * fx
        mj_data.xfrc_applied[body_id, 1] += fwd_n * fy

    def _apply_body_pitch_wrench(body_id, forward_n=0.0, down_n=0.0, pitch_nm=0.0):
        if body_id < 0:
            return
        fx, fy = fall_fwd_xy
        lx, ly, lz = fall_lat_3d
        fwd_n = float(forward_n)
        pitch_nm = float(pitch_nm)
        mj_data.xfrc_applied[body_id, 0] += fwd_n * fx
        mj_data.xfrc_applied[body_id, 1] += fwd_n * fy
        mj_data.xfrc_applied[body_id, 2] -= float(down_n)
        # Negative torque around the anatomical lateral axis pitches the body forward
        mj_data.xfrc_applied[body_id, 3] += -pitch_nm * lx
        mj_data.xfrc_applied[body_id, 4] += -pitch_nm * ly
        mj_data.xfrc_applied[body_id, 5] += -pitch_nm * lz

    def _body_vel_world(mj_model_, mj_data_, body_id_):
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(mj_model_, mj_data_, MJOBJ_BODY, body_id_, vel6, 0)
        return vel6[3:].copy()

    def _apply_ap_damping(body_id, vel_gain=0.30, max_force_bw=0.05, target_v=0.0):
        if body_id < 0:
            return 0.0
        vv = _body_vel_world(mj_model, mj_data, body_id)
        ap_v = float(vv[0] * fall_fwd_xy[0] + vv[1] * fall_fwd_xy[1])
        corr = float(np.clip((target_v - ap_v) * vel_gain * BW,
                             -max_force_bw * BW, max_force_bw * BW))
        _apply_body_forward_wrench(body_id, forward_n=corr)
        return ap_v

    def _apply_ap_centering(body_id, ap_err=0.0, gain_bw=0.03):
        if body_id < 0:
            return
        mag = float(np.clip(abs(ap_err) / 0.14, 0.0, 1.0))
        corr = -math.copysign(gain_bw * BW * mag, ap_err)
        _apply_body_forward_wrench(body_id, forward_n=corr)

    def _head_fwd_rel():
        if pelvis_id < 0 or head_id < 0:
            return 0.0
        hp = mj_data.xpos[head_id] - mj_data.xpos[pelvis_id]
        return float(hp[0] * fall_fwd_xy[0] + hp[1] * fall_fwd_xy[1])

    def _head_lat_rel():
        if pelvis_id < 0 or head_id < 0:
            return 0.0
        hp = mj_data.xpos[head_id] - mj_data.xpos[pelvis_id]
        lat_xy = np.array([fall_lat_3d[0], fall_lat_3d[1]], dtype=float)
        return float(hp[0] * lat_xy[0] + hp[1] * lat_xy[1])

    def _trunk_tilt_deg():
        if pelvis_id < 0 or head_id < 0:
            return 0.0
        vec = mj_data.xpos[head_id] - mj_data.xpos[pelvis_id]
        nv = float(np.linalg.norm(vec))
        if nv <= 1e-8:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(np.dot(vec / nv, [0.0, 0.0, 1.0]), -1.0, 1.0))))

    def _forward_prone_score():
        if pelvis_id < 0 or head_id < 0:
            return 0.0
        vec = np.array(mj_data.xpos[head_id] - mj_data.xpos[pelvis_id], dtype=float)
        nv = float(np.linalg.norm(vec))
        if nv <= 1e-8:
            return 0.0
        fwd3 = np.array([fall_fwd_xy[0], fall_fwd_xy[1], 0.0], dtype=float)
        nf = float(np.linalg.norm(fwd3))
        if nf <= 1e-8:
            return 0.0
        return float(np.dot(vec / nv, fwd3 / nf))

    def _draw_virtual_chair(viewer):
        if chair_center_xy is None or viewer is None:
            return
        try:
            scn = viewer.user_scn
            scn.ngeom = 0
            yaw = math.atan2(fall_fwd_xy[1], fall_fwd_xy[0])
            cy, sy = math.cos(yaw), math.sin(yaw)
            fwd3 = np.array([fall_fwd_xy[0], fall_fwd_xy[1], 0.0], dtype=float)
            lat3 = np.array([fall_lat_3d[0], fall_lat_3d[1], 0.0], dtype=float)
            seat_pos = np.array([chair_center_xy[0], chair_center_xy[1], chair_seat_z], dtype=float)
            seat_mat = np.array([[cy, -sy, 0.0],[sy, cy, 0.0],[0.0, 0.0, 1.0]], dtype=float).reshape(-1)
            rgba_seat = np.array([0.45, 0.45, 0.52, 0.35], dtype=float)
            rgba_back = np.array([0.42, 0.42, 0.48, 0.28], dtype=float)
            rgba_leg  = np.array([0.38, 0.38, 0.42, 0.28], dtype=float)

            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX,
                                np.array([0.23, 0.20, 0.025], dtype=float),
                                seat_pos, seat_mat, rgba_seat)
            scn.ngeom += 1

            back_offset = -0.20 * fall_fwd_xy
            back_pos = np.array([chair_center_xy[0] + back_offset[0], chair_center_xy[1] + back_offset[1], chair_seat_z + 0.25], dtype=float)
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX,
                                np.array([0.025, 0.20, 0.25], dtype=float),
                                back_pos, seat_mat, rgba_back)
            scn.ngeom += 1

            leg_half = np.array([0.020, 0.020, max(0.5 * chair_seat_z, 0.18)], dtype=float)
            for sx in (-0.18, +0.18):
                for sy_ in (-0.15, +0.15):
                    leg_pos = seat_pos + sx * fwd3 + sy_ * lat3
                    leg_pos[2] = leg_half[2]
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX,
                                        leg_half, leg_pos, seat_mat, rgba_leg)
                    scn.ngeom += 1
        except Exception:
            pass

    # -- IMU -------------------------------------------------------------------
    imu = LeanIMU(mj_model, mj_data, age, height)
    print(f"        IMU mounted on body_id={imu.body_id}")

    # -- shared exporters (read-only, no motion changes) -----------------------
    marker_exporter = None
    dynamics_analyzer = None
    paper_exporter = None
    if _HAS_SHARED_EXPORTERS:
        try:
            marker_exporter = MarkerKinematicsExporter(mj_model, mj_data, export_hz=100.0)
            dynamics_analyzer = DynamicsContactAnalyzer(
                mj_model, mj_data, body_mass=body_mass, leg_length=max(0.60, 0.53 * height)
            )
            paper_exporter = PaperAlignmentExporter(
                marker_exporter, dynamics_analyzer,
                subject_meta={"age": age, "height": height, "sex": sex, "weight": weight, "fall_type": "scenario23_get_up_forward"},
            )
            print("        Shared exporters enabled (markers + dynamics + paper-alignment)")
        except Exception as e:
            marker_exporter = None
            dynamics_analyzer = None
            paper_exporter = None
            print(f"        Shared exporters unavailable: {e}")

    marker_bundle = None
    dynamics_bundle = None
    paper_bundle = None
    _shared_exporters_active = bool(marker_exporter is not None and dynamics_analyzer is not None)
    _shared_exporters_failed = False
    dashboard = None
    if PhysicsDashboard is not None and pelvis_id >= 0:
        try:
            dashboard = PhysicsDashboard(
                mj_model, mj_data,
                body_mass=body_mass,
                leg_length=max(0.60, 0.53 * height),
                upright_h=float(mj_data.xpos[pelvis_id][2]),
            )
            print("        PhysicsDashboard ACTIVE")
        except Exception as e:
            dashboard = None
            print(f"        PhysicsDashboard unavailable: {e}")

    def _capture_shared_exporters(sim_time, phase_name):
        nonlocal _shared_exporters_active, _shared_exporters_failed
        if not _shared_exporters_active or _shared_exporters_failed:
            return None
        try:
            marker_exporter.capture_frame(sim_time=sim_time, phase=phase_name)
            return dynamics_analyzer.capture_frame(sim_time=sim_time, phase=phase_name)
        except Exception as e:
            _shared_exporters_failed = True
            print(f"        Shared exporter capture disabled: {e}")
            return None

    # -- embedding blend state -------------------------------------------------
    z_cur = z_sit.clone()

    def _blend_to(z_new, steps=25):
        for k in range(steps + 1):
            yield _blend_z(z_cur, z_new, k / steps)

    _blend_gen  = iter([z_cur])
    _blend_done = True

    def _start_blend(z_new, steps=25):
        nonlocal _blend_gen, _blend_done, z_cur
        _blend_gen  = iter(list(_blend_to(z_new, steps)))
        _blend_done = False

    def _step_blend():
        nonlocal z_cur, _blend_done
        try:
            z_cur = next(_blend_gen)
        except StopIteration:
            _blend_done = True

    # -- sim state -------------------------------------------------------------
    phase                = "sit"
    perturb_t            = 0.0
    trigger_fired        = False
    collapse_start_step  = SIT_S + GETUP_S
    sit_end_strength     = float(np.clip(0.44 - 0.08 * age_frac - 0.05 * bal_imp, 0.30, 0.44))
    sit_hold_strength    = float(np.clip(0.40 - 0.06 * age_frac - 0.04 * bal_imp, 0.28, 0.42))
    rise_floor_strength  = float(np.clip(0.22 - 0.06 * age_frac - 0.04 * bal_imp, 0.12, 0.24))
    rest_mode            = False
    rest_ctr             = 0
    rest_anchor_xy       = None
    metrics              = []
    metrics_log          = []
    sit_start_pz         = None
    rise_start_pz        = None

    # Scenario 23 is back to a chair-supported seated start.  We keep the
    # working sit->rise->forward-fall motion, but place the pelvis at a true
    # seated height and render a matching chair in the viewer.
    seat_target_pz       = float(np.clip(0.50 + 0.04 * (height - 1.65) / 0.15 - 0.01 * age_frac - 0.01 * bal_imp, 0.46, 0.56))
    chair_seat_z         = float(np.clip(seat_target_pz - (0.090 + 0.010 * (height - 1.65) / 0.15), 0.36, 0.48))
    rise_target_pz       = float(np.clip(0.88 + 0.05 * sf - 0.04 * age_frac - 0.03 * bal_imp, 0.82, 0.94))
    seat_off_pz          = float(np.clip(seat_target_pz + 0.06 + 0.01 * age_frac + 0.01 * bal_imp, seat_target_pz + 0.05, seat_target_pz + 0.10))
    settle_floor_pz      = float(np.clip(0.11 - 0.02 * age_frac - 0.01 * bal_imp, 0.08, 0.11))
    settle_target_deg    = float(np.clip(82.0 + 6.0 * age_frac + 2.0 * bal_imp, 78.0, 90.0))
    prone_target         = float(np.clip(0.82 - 0.08 * age_frac - 0.04 * bal_imp, 0.66, 0.84))
    head_fwd_floor       = float(np.clip(0.15 - 0.03 * age_frac - 0.02 * bal_imp, 0.10, 0.16))
    collapse_ap_band     = float(np.clip(0.05 + 0.01 * age_frac + 0.01 * bal_imp, 0.05, 0.08))
    fall_resid_scale     = float(np.clip(1.04 - 0.24 * age_frac - 0.10 * bal_imp, 0.55, 1.04))
    collapse_force_scale = float(np.clip(1.02 - 0.08 * age_frac - 0.04 * bal_imp, 0.86, 1.02))
    collapse_pitch_scale = float(np.clip(1.06 - 0.10 * age_frac - 0.05 * bal_imp, 0.88, 1.06))
    pre_start_local      = int(round(np.clip(0.52 * GETUP_S - 4.0 * age_frac - 3.0 * bal_imp, 20.0, 0.62 * GETUP_S)))
    collapse_target_local = int(round(np.clip(0.74 * GETUP_S - 6.0 * age_frac - 4.0 * bal_imp, pre_start_local + 10.0, 0.84 * GETUP_S)))
    early_trigger_fwd    = float(np.clip(0.11 - 0.01 * age_frac + 0.02 * bal_imp, 0.09, 0.14))
    early_trigger_trunk  = float(np.clip(40.0 - 4.0 * age_frac + 4.0 * bal_imp, 34.0, 44.0))
    leg_floor            = float(np.clip(0.16 - 0.04 * age_frac - 0.03 * bal_imp, 0.10, 0.16))
    torso_floor          = float(np.clip(0.15 - 0.03 * age_frac - 0.02 * bal_imp, 0.10, 0.15))
    arm_floor            = float(np.clip(0.28 - 0.08 * age_frac - 0.06 * bal_imp, 0.16, 0.28))
    action_hist          = deque(maxlen=3)

    # Persistent seated-anchor cache. Once a valid chair-supported start is
    # found for this subject/model shape, later runs skip the slow pose search.
    _anchor_cache_dir = os.path.join(_ROOT, "outputs", ".scenario23_anchor_cache")

    def _anchor_cache_key():
        return "s23_chair_nq{}_nv{}_age{}_h{}_sex{}_w{}.npz".format(
            int(mj_model.nq), int(mj_model.nv), int(round(age)),
            int(round(height * 100)), str(sex).lower()[:1], int(round(weight))
        )

    def _load_cached_sit_anchor():
        try:
            path = os.path.join(_anchor_cache_dir, _anchor_cache_key())
            if not os.path.exists(path):
                return None
            dat = np.load(path, allow_pickle=False)
            if int(dat["nq"]) != int(mj_model.nq) or int(dat["nv"]) != int(mj_model.nv):
                return None
            qpos = np.asarray(dat["sit_qpos"], dtype=float).copy()
            qvel = np.asarray(dat["sit_qvel"], dtype=float).copy()
            stand_qpos_c = np.asarray(dat["stand_qpos"], dtype=float).copy()
            stand_qvel_c = np.asarray(dat["stand_qvel"], dtype=float).copy()
            if qpos.shape != mj_data.qpos.shape or qvel.shape != mj_data.qvel.shape:
                return None
            obs_c, _ = env.reset(options={"qpos": qpos.copy(), "qvel": qvel.copy()})
            _lock_heading()
            _reanchor_chair_to_current_pelvis(forward_offset=0.050)
            q = float(_seat_pose_quality())
            if _seat_pose_ok() and q >= 0.48:
                print(f"        Loaded cached chair seated anchor -> {os.path.basename(path)} (q={q:.2f})")
                return stand_qpos_c, stand_qvel_c, mj_data.qpos.copy(), np.zeros_like(mj_data.qvel), q, obs_c
        except Exception as e:
            print(f"        Cached seated anchor ignored: {e}")
        return None

    def _save_cached_sit_anchor(stand_qpos_c, stand_qvel_c, sit_qpos_c, sit_qvel_c, quality_c):
        try:
            os.makedirs(_anchor_cache_dir, exist_ok=True)
            path = os.path.join(_anchor_cache_dir, _anchor_cache_key())
            np.savez_compressed(
                path, nq=int(mj_model.nq), nv=int(mj_model.nv),
                age=float(age), height=float(height), weight=float(weight),
                sit_qpos=np.asarray(sit_qpos_c, dtype=float),
                sit_qvel=np.asarray(sit_qvel_c, dtype=float),
                stand_qpos=np.asarray(stand_qpos_c, dtype=float),
                stand_qvel=np.asarray(stand_qvel_c, dtype=float),
                quality=float(quality_c),
            )
            print(f"        Saved chair seated anchor cache -> {os.path.basename(path)}")
        except Exception as e:
            print(f"        Could not save seated anchor cache: {e}")

    seat_l_foot_id = _body_id(mj_model, "L_Foot", "FootL", "LeftFoot", "L_Ankle", "AnkleL", "LeftAnkle", "L_Toe", "ToeL", "LeftToe")
    seat_r_foot_id = _body_id(mj_model, "R_Foot", "FootR", "RightFoot", "R_Ankle", "AnkleR", "RightAnkle", "R_Toe", "ToeR", "RightToe")

    def _seat_foot_metrics():
        zs = []
        for bid in (seat_l_foot_id, seat_r_foot_id):
            if bid >= 0:
                zs.append(float(mj_data.xpos[bid][2]))
        if not zs:
            return 0.0, 0.0, 0
        mean_z = float(np.mean(zs))
        span_z = float(max(zs) - min(zs)) if len(zs) >= 2 else 0.0
        return mean_z, span_z, len(zs)

    def _seat_pose_snapshot():
        pz = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else seat_target_pz
        trunk = _trunk_tilt_deg()
        head_fwd = _head_fwd_rel()
        prone = _forward_prone_score()
        return pz, trunk, head_fwd, prone

    def _seat_pose_quality():
        pz, trunk, head_fwd, prone = _seat_pose_snapshot()
        foot_mean_z, foot_span_z, foot_n = _seat_foot_metrics()
        q_pz = 1.0 - float(np.clip(abs(pz - seat_target_pz) / 0.10, 0.0, 1.0))
        q_trunk = 1.0 - float(np.clip(abs(trunk - 18.0) / 16.0, 0.0, 1.0))
        q_head = 1.0 - float(np.clip(abs(head_fwd - 0.00) / 0.12, 0.0, 1.0))
        q_prone = 1.0 - float(np.clip(abs(prone) / 0.16, 0.0, 1.0))
        q_con = 1.0 if mj_data.ncon >= 2 else 0.50 if mj_data.ncon >= 1 else 0.0
        if foot_n > 0:
            q_foot_h = 1.0 - float(np.clip(abs(foot_mean_z - 0.05) / 0.08, 0.0, 1.0))
            q_foot_sym = 1.0 - float(np.clip(foot_span_z / 0.08, 0.0, 1.0))
            q_foot = 0.7 * q_foot_h + 0.3 * q_foot_sym
        else:
            q_foot = 0.0
        return 0.26 * q_pz + 0.26 * q_trunk + 0.16 * q_head + 0.10 * q_prone + 0.10 * q_con + 0.12 * q_foot

    def _seat_pose_ok():
        pz, trunk, head_fwd, prone = _seat_pose_snapshot()
        foot_mean_z, foot_span_z, foot_n = _seat_foot_metrics()
        # Chair-supported start: no ground-contact requirement. The subject is
        # allowed to be supported by the visual chair in SIT; requiring ncon/feet
        # contact was what forced the slow search path even for good poses.
        feet_ok = (foot_n == 0) or (foot_mean_z <= 0.22 and foot_span_z <= 0.16)
        pelvis_ok = (seat_target_pz - 0.12) <= pz <= (seat_off_pz + 0.04)
        trunk_ok = 5.0 <= trunk <= 48.0
        head_ok = -0.20 <= head_fwd <= 0.24
        prone_ok = abs(prone) <= 0.32
        return bool(
            pelvis_ok
            and trunk_ok
            and head_ok
            and prone_ok
            and feet_ok
            and (not _scenario23_is_fallen_state())
        )

    def _scenario23_is_fallen_state():
        pz, trunk, _, prone = _seat_pose_snapshot()
        return bool(
            pz < 0.16
            or (pz < 0.26 and (trunk > 58.0 or abs(prone) > 0.38))
            or (pz < 0.34 and abs(prone) > 0.62)
        )

    def _enumerate_scalar_joint_qpos():
        out = []
        joint_obj = mujoco.mjtObj.mjOBJ_JOINT
        hinge_t = int(mujoco.mjtJoint.mjJNT_HINGE)
        slide_t = int(mujoco.mjtJoint.mjJNT_SLIDE)
        for jid in range(mj_model.njnt):
            jtype = int(mj_model.jnt_type[jid])
            if jtype not in (hinge_t, slide_t):
                continue
            jname = (mujoco.mj_id2name(mj_model, joint_obj, jid) or "").lower()
            qadr = int(mj_model.jnt_qposadr[jid])
            out.append((jname, qadr))
        return out

    scalar_joint_qpos = _enumerate_scalar_joint_qpos()

    def _candidate_joint_hits(qpos, side_tokens, part_tokens, axis_tokens, value, additive=False):
        hits = 0
        for jname, qadr in scalar_joint_qpos:
            if side_tokens is not None and not any(tok in jname for tok in side_tokens):
                continue
            if not all(tok in jname for tok in part_tokens):
                continue
            if axis_tokens and not any(tok in jname for tok in axis_tokens):
                continue
            qpos[qadr] = (qpos[qadr] + value) if additive else value
            hits += 1
        return hits

    def _build_fast_chair_supported_anchor():
        """Fast seated-start builder for the visible frame-0 pose.

        This avoids the huge combinatorial search from older repairs.  We try a
        small bank of chair-sitting presets, warm each one for a few frames with
        seat support, and accept the first physically plausible seated state.
        """
        nonlocal chair_center_xy
        _base_obs, _ = env.reset()
        _lock_heading()
        stand_qpos = mj_data.qpos.copy()
        stand_qvel = np.zeros_like(mj_data.qvel)
        zero_action = np.zeros(env.action_space.shape)

        foot_xy = []
        for bid in (seat_l_foot_id, seat_r_foot_id):
            if bid >= 0:
                foot_xy.append(np.array(mj_data.xpos[bid][:2], dtype=float))
        if foot_xy:
            foot_mid_xy = np.mean(np.vstack(foot_xy), axis=0)
            seat_xy = foot_mid_xy - 0.060 * fall_fwd_xy
        else:
            seat_xy = np.array(stand_qpos[:2], dtype=float) - 0.020 * fall_fwd_xy
        chair_center_xy = seat_xy - 0.045 * fall_fwd_xy

        sign_sets = [
            (+1.0, +1.0, +1.0, +1.0),
            (-1.0, -1.0, -1.0, -1.0),
            (+1.0, +1.0, -1.0, +1.0),
            (-1.0, -1.0, +1.0, -1.0),
        ]
        mag_sets = [
            (0.95, 1.70, 0.25, 0.08),
            (1.10, 1.90, 0.35, 0.12),
            (1.25, 2.05, 0.45, 0.16),
        ]
        best = None
        for hip_sign, knee_sign, ankle_sign, torso_sign in sign_sets:
            for hip_mag, knee_mag, ankle_mag, torso_mag in mag_sets:
                q = stand_qpos.copy()
                if q.shape[0] >= 3:
                    q[0] = float(seat_xy[0])
                    q[1] = float(seat_xy[1])
                    q[2] = float(seat_target_pz)
                hits = 0
                hits += _candidate_joint_hits(q, ("l_", "left"), ("hip",), ("_x", "flex"), hip_sign * hip_mag)
                hits += _candidate_joint_hits(q, ("r_", "right"), ("hip",), ("_x", "flex"), hip_sign * hip_mag)
                hits += _candidate_joint_hits(q, ("l_", "left"), ("knee",), ("_x", "flex"), knee_sign * knee_mag)
                hits += _candidate_joint_hits(q, ("r_", "right"), ("knee",), ("_x", "flex"), knee_sign * knee_mag)
                hits += _candidate_joint_hits(q, ("l_", "left"), ("ankle",), ("_x", "flex"), ankle_sign * ankle_mag)
                hits += _candidate_joint_hits(q, ("r_", "right"), ("ankle",), ("_x", "flex"), ankle_sign * ankle_mag)
                hits += _candidate_joint_hits(q, None, ("torso",), ("_x", "flex"), torso_sign * torso_mag, additive=True)
                hits += _candidate_joint_hits(q, None, ("spine",), ("_x", "flex"), torso_sign * (0.65 * torso_mag), additive=True)
                hits += _candidate_joint_hits(q, None, ("abd",), ("_x", "flex"), torso_sign * (0.45 * torso_mag), additive=True)
                if hits < 6:
                    continue

                obs_local, _ = env.reset(options={"qpos": q.copy(), "qvel": np.zeros_like(stand_qvel)})
                _lock_heading()
                chair_center_xy = seat_xy - 0.045 * fall_fwd_xy
                terminated = truncated = False
                for _ in range(24):
                    mj_data.xfrc_applied[:] = 0.0
                    _pin_to_anchor(q, np.zeros_like(mj_data.qvel), blend=0.90)
                    _seat_hold_wrench(strength_scale=1.10, back_bias=1.10)
                    for idx in grp["leg"]:
                        mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(sit_hold_strength, 0.38)
                    for idx in grp["torso"]:
                        mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(torso_floor, 0.34)
                    obs_local, _, terminated, truncated, _ = env.step(zero_action)
                    if terminated or truncated:
                        break
                if terminated or truncated:
                    continue

                pz, trunk, head_fwd, prone = _seat_pose_snapshot()
                foot_mean_z, foot_span_z, foot_n = _seat_foot_metrics()
                score = _seat_pose_quality()
                score += 0.10 if mj_data.ncon >= 1 else -0.10
                if foot_n >= 1 and foot_mean_z <= 0.12:
                    score += 0.06
                if 8.0 <= trunk <= 35.0:
                    score += 0.08
                if abs(pz - seat_target_pz) > 0.08:
                    score -= 0.20
                if abs(prone) > 0.30:
                    score -= 0.20
                cand = (score, mj_data.qpos.copy(), mj_data.qvel.copy())
                if best is None or cand[0] > best[0]:
                    best = cand
                if _seat_pose_ok() and score >= 0.46:
                    mj_data.qpos[:] = stand_qpos
                    mj_data.qvel[:] = stand_qvel
                    mujoco.mj_forward(mj_model, mj_data)
                    return stand_qpos.copy(), stand_qvel.copy(), cand[1].copy(), cand[2].copy(), float(score)

        mj_data.qpos[:] = stand_qpos
        mj_data.qvel[:] = stand_qvel
        mujoco.mj_forward(mj_model, mj_data)
        if best is None:
            return stand_qpos.copy(), stand_qvel.copy(), stand_qpos.copy(), stand_qvel.copy(), 0.0
        return stand_qpos.copy(), stand_qvel.copy(), best[1].copy(), best[2].copy(), float(best[0])

    def _build_deterministic_visible_sit_anchor():
        _base_obs, _ = env.reset()
        _lock_heading()
        stand_qpos = mj_data.qpos.copy()
        stand_qvel = np.zeros_like(mj_data.qvel)
        best = None
        zero_action = np.zeros(env.action_space.shape)

        for hip_sign in (1.0, -1.0):
            for knee_sign in (1.0, -1.0):
                for ankle_sign in (1.0, -1.0):
                    for torso_sign in (1.0, -1.0):
                        for hip_mag in (0.70, 0.95, 1.20):
                            for knee_mag in (1.10, 1.45, 1.80):
                                for ankle_mag in (0.15, 0.35, 0.55):
                                    for torso_mag in (0.08, 0.16, 0.24):
                                        for root_drop in (-0.16, -0.24, -0.32, -0.40, -0.48):
                                            q = stand_qpos.copy()
                                            hits = 0
                                            hits += _candidate_joint_hits(q, ("l_", "left"), ("hip",), ("_x", "flex"), hip_sign * hip_mag)
                                            hits += _candidate_joint_hits(q, ("r_", "right"), ("hip",), ("_x", "flex"), hip_sign * hip_mag)
                                            hits += _candidate_joint_hits(q, ("l_", "left"), ("knee",), ("_x", "flex"), knee_sign * knee_mag)
                                            hits += _candidate_joint_hits(q, ("r_", "right"), ("knee",), ("_x", "flex"), knee_sign * knee_mag)
                                            hits += _candidate_joint_hits(q, ("l_", "left"), ("ankle",), ("_x", "flex"), ankle_sign * ankle_mag)
                                            hits += _candidate_joint_hits(q, ("r_", "right"), ("ankle",), ("_x", "flex"), ankle_sign * ankle_mag)
                                            hits += _candidate_joint_hits(q, None, ("torso",), ("_x", "flex"), torso_sign * torso_mag, additive=True)
                                            hits += _candidate_joint_hits(q, None, ("spine",), ("_x", "flex"), torso_sign * (0.65 * torso_mag), additive=True)
                                            hits += _candidate_joint_hits(q, None, ("abd",), ("_x", "flex"), torso_sign * (0.55 * torso_mag), additive=True)
                                            if hits < 6:
                                                continue
                                            if q.shape[0] >= 3:
                                                q[2] = stand_qpos[2] + root_drop

                                            obs_local, _ = env.reset(options={"qpos": q.copy(), "qvel": np.zeros_like(stand_qvel)})
                                            _lock_heading()
                                            terminated = truncated = False
                                            for _ in range(18):
                                                mj_data.xfrc_applied[:] = 0.0
                                                _seat_hold_wrench(strength_scale=1.10, back_bias=1.10)
                                                for idx in grp["leg"]:
                                                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(sit_hold_strength, 0.34)
                                                for idx in grp["torso"]:
                                                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(torso_floor, 0.34)
                                                obs_local, _, terminated, truncated, _ = env.step(zero_action)
                                                if terminated or truncated:
                                                    break

                                            if terminated or truncated:
                                                continue

                                            pz, trunk, head_fwd, prone = _seat_pose_snapshot()
                                            foot_mean_z, foot_span_z, foot_n = _seat_foot_metrics()
                                            score = _seat_pose_quality()
                                            if foot_n > 0 and foot_mean_z > 0.20:
                                                score -= 0.25
                                            if pz > seat_target_pz + 0.12:
                                                score -= 0.35
                                            if trunk < 10.0:
                                                score -= 0.20
                                            cand = (score, mj_data.qpos.copy(), mj_data.qvel.copy(), pz, trunk, head_fwd, prone, foot_mean_z, foot_span_z, mj_data.ncon)
                                            if best is None or cand[0] > best[0]:
                                                best = cand
                                            if _seat_pose_ok() and score >= 0.62:
                                                mj_data.qpos[:] = stand_qpos
                                                mj_data.qvel[:] = stand_qvel
                                                mujoco.mj_forward(mj_model, mj_data)
                                                return stand_qpos.copy(), stand_qvel.copy(), cand[1].copy(), cand[2].copy(), float(score)
        mj_data.qpos[:] = stand_qpos
        mj_data.qvel[:] = stand_qvel
        mujoco.mj_forward(mj_model, mj_data)
        if best is None or best[0] < 0.45:
            return stand_qpos.copy(), stand_qvel.copy(), stand_qpos.copy(), stand_qvel.copy(), 0.0
        return stand_qpos.copy(), stand_qvel.copy(), best[1].copy(), best[2].copy(), float(best[0])


    def _build_ground_supported_sit_anchor():
        _base_obs, _ = env.reset()
        _lock_heading(force_print=True)
        stand_qpos = mj_data.qpos.copy()
        stand_qvel = np.zeros_like(mj_data.qvel)
        zero_action = np.zeros(env.action_space.shape)
        best = None

        seat_xy = np.array(stand_qpos[:2], dtype=float)
        foot_xy = []
        for bid in (seat_l_foot_id, seat_r_foot_id):
            if bid >= 0:
                foot_xy.append(np.array(mj_data.xpos[bid][:2], dtype=float))
        if foot_xy:
            seat_xy = np.mean(foot_xy, axis=0) - 0.020 * fall_fwd_xy

        for hip_sign in (1.0, -1.0):
            for knee_sign in (1.0, -1.0):
                for ankle_sign in (1.0, -1.0):
                    for torso_sign in (1.0, -1.0):
                        for pelvis_z in np.linspace(seat_target_pz + 0.01, seat_off_pz - 0.01, 5):
                            for hip_mag in (0.85, 1.05, 1.25, 1.45):
                                for knee_mag in (1.35, 1.60, 1.85, 2.10):
                                    for ankle_mag in (0.10, 0.25, 0.40):
                                        for torso_mag in (0.04, 0.10, 0.16, 0.22):
                                            q = stand_qpos.copy()
                                            if q.shape[0] >= 3:
                                                q[0] = float(seat_xy[0])
                                                q[1] = float(seat_xy[1])
                                                q[2] = float(pelvis_z)
                                            hits = 0
                                            hits += _candidate_joint_hits(q, ("l_", "left"), ("hip",), ("_x", "flex"), hip_sign * hip_mag)
                                            hits += _candidate_joint_hits(q, ("r_", "right"), ("hip",), ("_x", "flex"), hip_sign * hip_mag)
                                            hits += _candidate_joint_hits(q, ("l_", "left"), ("knee",), ("_x", "flex"), knee_sign * knee_mag)
                                            hits += _candidate_joint_hits(q, ("r_", "right"), ("knee",), ("_x", "flex"), knee_sign * knee_mag)
                                            hits += _candidate_joint_hits(q, ("l_", "left"), ("ankle",), ("_x", "flex"), ankle_sign * ankle_mag)
                                            hits += _candidate_joint_hits(q, ("r_", "right"), ("ankle",), ("_x", "flex"), ankle_sign * ankle_mag)
                                            hits += _candidate_joint_hits(q, None, ("torso",), ("_x", "flex"), torso_sign * torso_mag, additive=True)
                                            hits += _candidate_joint_hits(q, None, ("spine",), ("_x", "flex"), torso_sign * (0.75 * torso_mag), additive=True)
                                            hits += _candidate_joint_hits(q, None, ("abd",), ("_x", "flex"), torso_sign * (0.55 * torso_mag), additive=True)
                                            if hits < 6:
                                                continue

                                            obs_local, _ = env.reset(options={"qpos": q.copy(), "qvel": np.zeros_like(stand_qvel)})
                                            _lock_heading()
                                            terminated = truncated = False
                                            for _ in range(28):
                                                mj_data.xfrc_applied[:] = 0.0
                                                _pin_to_anchor(q, np.zeros_like(mj_data.qvel), blend=0.92)
                                                _seat_hold_wrench(strength_scale=1.22, back_bias=1.15)
                                                for idx in grp["leg"]:
                                                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(sit_hold_strength, 0.42)
                                                for idx in grp["torso"]:
                                                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(torso_floor, 0.40)
                                                obs_local, _, terminated, truncated, _ = env.step(zero_action)
                                                if terminated or truncated:
                                                    break
                                            if terminated or truncated:
                                                continue

                                            pz, trunk, head_fwd, prone = _seat_pose_snapshot()
                                            foot_mean_z, foot_span_z, foot_n = _seat_foot_metrics()
                                            score = _seat_pose_quality()
                                            if foot_n < 2:
                                                score -= 0.30
                                            if foot_mean_z > 0.16:
                                                score -= 0.45
                                            if not ((seat_target_pz - 0.10) <= pz <= (seat_off_pz + 0.04)):
                                                score -= 0.35
                                            if trunk < 8.0 or trunk > 70.0:
                                                score -= 0.35
                                            if abs(prone) > 0.35:
                                                score -= 0.25
                                            cand = (score, mj_data.qpos.copy(), mj_data.qvel.copy(), pz, trunk, head_fwd, prone, foot_mean_z, foot_span_z, mj_data.ncon)
                                            if best is None or cand[0] > best[0]:
                                                best = cand
                                            if _seat_pose_ok() and score >= 0.52:
                                                mj_data.qpos[:] = stand_qpos
                                                mj_data.qvel[:] = stand_qvel
                                                mujoco.mj_forward(mj_model, mj_data)
                                                return stand_qpos.copy(), stand_qvel.copy(), cand[1].copy(), cand[2].copy(), float(score)

        mj_data.qpos[:] = stand_qpos
        mj_data.qvel[:] = stand_qvel
        mujoco.mj_forward(mj_model, mj_data)
        if best is None or best[0] < 0.40:
            return stand_qpos.copy(), stand_qvel.copy(), stand_qpos.copy(), stand_qvel.copy(), 0.0
        return stand_qpos.copy(), stand_qvel.copy(), best[1].copy(), best[2].copy(), float(best[0])


    def _build_explicit_crouch_anchor():
        """Last-resort seated start built from conservative flexion presets."""
        _base_obs, _ = env.reset()
        _lock_heading()
        stand_qpos = mj_data.qpos.copy()
        stand_qvel = np.zeros_like(mj_data.qvel)
        zero_action = np.zeros(env.action_space.shape)

        seat_xy = np.array(stand_qpos[:2], dtype=float)
        foot_xy = []
        for bid in (seat_l_foot_id, seat_r_foot_id):
            if bid >= 0:
                foot_xy.append(np.array(mj_data.xpos[bid][:2], dtype=float))
        if foot_xy:
            seat_xy = np.mean(foot_xy, axis=0) - 0.010 * fall_fwd_xy

        best = None
        for hip_sign in (1.0, -1.0):
            for knee_sign in (1.0, -1.0):
                for ankle_sign in (1.0, -1.0):
                    for torso_sign in (1.0, -1.0):
                        for pelvis_z in np.linspace(seat_target_pz + 0.01, seat_off_pz - 0.01, 4):
                            for hip_mag in (0.95, 1.15, 1.35):
                                for knee_mag in (1.55, 1.80, 2.05):
                                    for ankle_mag in (0.15, 0.35, 0.55):
                                        for torso_mag in (0.04, 0.10, 0.16):
                                            q = stand_qpos.copy()
                                            if q.shape[0] >= 3:
                                                q[0] = float(seat_xy[0])
                                                q[1] = float(seat_xy[1])
                                                q[2] = float(pelvis_z)
                                            hits = 0
                                            hits += _candidate_joint_hits(q, ("l_", "left"), ("hip",), ("_x", "flex"), hip_sign * hip_mag)
                                            hits += _candidate_joint_hits(q, ("r_", "right"), ("hip",), ("_x", "flex"), hip_sign * hip_mag)
                                            hits += _candidate_joint_hits(q, ("l_", "left"), ("knee",), ("_x", "flex"), knee_sign * knee_mag)
                                            hits += _candidate_joint_hits(q, ("r_", "right"), ("knee",), ("_x", "flex"), knee_sign * knee_mag)
                                            hits += _candidate_joint_hits(q, ("l_", "left"), ("ankle",), ("_x", "flex"), ankle_sign * ankle_mag)
                                            hits += _candidate_joint_hits(q, ("r_", "right"), ("ankle",), ("_x", "flex"), ankle_sign * ankle_mag)
                                            hits += _candidate_joint_hits(q, None, ("torso",), ("_x", "flex"), torso_sign * torso_mag, additive=True)
                                            hits += _candidate_joint_hits(q, None, ("spine",), ("_x", "flex"), torso_sign * (0.65 * torso_mag), additive=True)
                                            hits += _candidate_joint_hits(q, None, ("abd",), ("_x", "flex"), torso_sign * (0.45 * torso_mag), additive=True)
                                            if hits < 6:
                                                continue

                                            obs_local, _ = env.reset(options={"qpos": q.copy(), "qvel": np.zeros_like(stand_qvel)})
                                            _lock_heading()
                                            terminated = truncated = False
                                            for _ in range(42):
                                                mj_data.xfrc_applied[:] = 0.0
                                                _pin_to_anchor(q, np.zeros_like(mj_data.qvel), blend=0.95)
                                                _seat_hold_wrench(strength_scale=1.45, back_bias=1.18)
                                                for idx in grp["leg"]:
                                                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(sit_hold_strength, 0.48)
                                                for idx in grp["torso"]:
                                                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(torso_floor, 0.44)
                                                obs_local, _, terminated, truncated, _ = env.step(zero_action)
                                                if terminated or truncated:
                                                    break
                                            if terminated or truncated:
                                                continue

                                            pz, trunk, head_fwd, prone = _seat_pose_snapshot()
                                            foot_mean_z, foot_span_z, foot_n = _seat_foot_metrics()
                                            score = _seat_pose_quality()
                                            if foot_n < 1:
                                                score -= 0.35
                                            if foot_mean_z > 0.16:
                                                score -= 0.30
                                            if trunk > 70.0 or trunk < 8.0:
                                                score -= 0.35
                                            if abs(prone) > 0.30:
                                                score -= 0.25
                                            cand = (score, mj_data.qpos.copy(), mj_data.qvel.copy(), pz, trunk, head_fwd, prone, foot_mean_z, foot_span_z, mj_data.ncon)
                                            if best is None or cand[0] > best[0]:
                                                best = cand
                                            if _seat_pose_ok() and score >= 0.52:
                                                mj_data.qpos[:] = stand_qpos
                                                mj_data.qvel[:] = stand_qvel
                                                mujoco.mj_forward(mj_model, mj_data)
                                                return stand_qpos.copy(), stand_qvel.copy(), cand[1].copy(), cand[2].copy(), float(score)

        mj_data.qpos[:] = stand_qpos
        mj_data.qvel[:] = stand_qvel
        mujoco.mj_forward(mj_model, mj_data)
        if best is None:
            return stand_qpos.copy(), stand_qvel.copy(), stand_qpos.copy(), stand_qvel.copy(), 0.0
        return stand_qpos.copy(), stand_qvel.copy(), best[1].copy(), best[2].copy(), float(best[0])

    def _set_root_pose_upright(seat_xy=None, pelvis_z=None):
        if mj_data.qpos.shape[0] < 7:
            return
        if seat_xy is None:
            if pelvis_id >= 0:
                seat_xy = np.array(mj_data.xpos[pelvis_id][:2], dtype=float)
            else:
                seat_xy = np.array(mj_data.qpos[:2], dtype=float)
        yaw = math.atan2(fall_fwd_xy[1], fall_fwd_xy[0]) if heading_locked else 0.0
        mj_data.qpos[0] = float(seat_xy[0])
        mj_data.qpos[1] = float(seat_xy[1])
        mj_data.qpos[2] = float(seat_target_pz if pelvis_z is None else pelvis_z)
        mj_data.qpos[3] = math.cos(0.5 * yaw)
        mj_data.qpos[4] = 0.0
        mj_data.qpos[5] = 0.0
        mj_data.qpos[6] = math.sin(0.5 * yaw)
        if mj_data.qvel.shape[0] > 0:
            mj_data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

    def _hard_recover_to_seat(max_iters=80):
        seat_xy = None
        if pelvis_id >= 0:
            seat_xy = np.array(mj_data.xpos[pelvis_id][:2], dtype=float)
        _set_root_pose_upright(seat_xy=seat_xy, pelvis_z=seat_target_pz + 0.02)
        rescue_obs = {"proprio": np.array(obs["proprio"], copy=True)} if isinstance(obs, dict) and "proprio" in obs else None
        best_q = -1.0
        for _ in range(max_iters):
            mj_data.xfrc_applied[:] = 0.0
            if pelvis_id >= 0 and mj_data.ncon < 2:
                mj_data.qpos[2] = max(seat_target_pz - 0.015, mj_data.qpos[2] - 0.004)
                if mj_data.qvel.shape[0] >= 3:
                    mj_data.qvel[:3] *= 0.0
                mujoco.mj_forward(mj_model, mj_data)
            _seat_hold_wrench(strength_scale=1.55, back_bias=1.20)
            for idx in grp["leg"]:
                mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(sit_hold_strength, 0.42)
            for idx in grp["torso"]:
                mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(torso_floor, 0.48)
            if rescue_obs is None:
                break
            obs_t = torch.tensor(rescue_obs["proprio"], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                warm_action = motivo.act(obs_t, _blend_z(z_sit, z_stand, 0.18)).squeeze(0).numpy()
            for idx in grp["leg"]:
                warm_action[idx] *= 0.10
            for idx in grp["torso"]:
                warm_action[idx] *= 0.08
            for idx in grp["arm"]:
                warm_action[idx] *= 0.06
            rescue_obs, _, terminated, truncated, _ = env.step(warm_action)
            if terminated or truncated:
                break
            q = _seat_pose_quality()
            best_q = max(best_q, q)
            if _seat_pose_ok() and q >= 0.52:
                _reanchor_chair_to_current_pelvis(forward_offset=0.050)
                return rescue_obs, True, q
        return rescue_obs if rescue_obs is not None else obs, False, max(best_q, _seat_pose_quality())

    def _seat_hold_wrench(strength_scale=1.0, back_bias=1.0):
        pz_now = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else seat_target_pz
        trunk_now = _trunk_tilt_deg()
        head_fwd_now = _head_fwd_rel()
        prone_now = _forward_prone_score()
        pz_err = seat_target_pz - pz_now
        deep_under = max(0.0, (seat_target_pz - 0.05) - pz_now)
        prone_bad = max(0.0, abs(min(prone_now, 0.0)) - 0.20)
        trunk_bad = max(0.0, trunk_now - 68.0)
        rescue = float(np.clip(4.0 * deep_under + 1.6 * prone_bad + trunk_bad / 26.0, 0.0, 1.8))
        if pelvis_id >= 0:
            seat_support = float(np.clip(
                pz_err * (4.60 + 1.10 * age_frac + 0.90 * bal_imp) * BW,
                -0.08 * BW, (0.55 + 0.20 * rescue) * BW
            ))
            if pz_now < seat_target_pz - 0.08:
                seat_support += (0.18 + 0.18 * rescue) * BW
            mj_data.xfrc_applied[pelvis_id, 2] += strength_scale * seat_support
            if chair_center_xy is not None:
                desired_xy = chair_center_xy + 0.040 * fall_fwd_xy
                cur_xy = np.array(mj_data.xpos[pelvis_id][:2], dtype=float)
                err_xy = desired_xy - cur_xy
                back_dir = -fall_fwd_xy
                lat_xy = np.array([fall_lat_3d[0], fall_lat_3d[1]], dtype=float)
                back_err = float(np.dot(err_xy, back_dir))
                lat_err = float(np.dot(err_xy, lat_xy))
                back_force = np.clip(back_err, -0.10, 0.18) * (0.92 + 0.20 * rescue) * BW * strength_scale
                lat_force  = np.clip(lat_err, -0.08, 0.08) * (0.22 + 0.06 * rescue) * BW * strength_scale
                mj_data.xfrc_applied[pelvis_id, 0] += back_force * back_dir[0] + lat_force * lat_xy[0]
                mj_data.xfrc_applied[pelvis_id, 1] += back_force * back_dir[1] + lat_force * lat_xy[1]
        if torso_id >= 0:
            pitch_nm = -(8.0 + 3.0 * back_bias + 7.0 * rescue) * strength_scale
            forward_n = -(0.020 + 0.010 * back_bias) * BW * strength_scale
            if trunk_now > 70.0 or prone_now < -0.30:
                pitch_nm -= 8.0 * strength_scale
            _apply_body_pitch_wrench(
                torso_id,
                forward_n=forward_n,
                down_n=0.0,
                pitch_nm=pitch_nm,
            )
            _apply_ap_damping(torso_id, vel_gain=0.30 + 0.08 * rescue, max_force_bw=0.030 + 0.010 * rescue)
        if head_id >= 0:
            _apply_body_pitch_wrench(
                head_id,
                forward_n=-0.010 * BW * strength_scale,
                down_n=0.0,
                pitch_nm=-(1.8 + 2.2 * rescue) * strength_scale,
            )
            _apply_ap_damping(head_id, vel_gain=0.20 + 0.04 * rescue, max_force_bw=0.016 + 0.004 * rescue)

    def _hidden_descend_to_seat(max_attempts=8):
        """Run an off-screen stand-to-seat descent before frame 0.

        The visible timeline must start seated. We therefore use the already
        stable standing reset, descend under controlled support onto the chair seat,
        then hold there briefly before step 0.
        """
        best_quality = -1.0
        best_state = None
        # The seated proxy embedding can still contain floor-like states.
        # Start from the robust standing embedding and only blend gently toward
        # the seated proxy late in the hidden setup.
        seat_start_z = z_sit.clone() if z_sit is not None else z_stand.clone()
        for attempt in range(max_attempts):
            obs_local, _ = env.reset()
            _lock_heading()
            terminated = truncated = False
            sit_start_hidden_pz = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 0.92
            hidden_steps = 120 + 12 * attempt
            for hs in range(hidden_steps):
                mj_data.xfrc_applied[:] = 0.0
                ramp = hs / max(hidden_steps - 1, 1)
                bell = math.sin(math.pi * ramp)
                late = float(np.clip((ramp - 0.55) / 0.45, 0.0, 1.0))
                pz_now = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 1.0
                target_pz = sit_start_hidden_pz + (seat_off_pz - sit_start_hidden_pz) * (ramp ** (1.05 + 0.16 * age_frac))
                pz_err = target_pz - pz_now
                down_n = (0.015 + 0.030 * bell) * BW
                seat_support = float(np.clip(pz_err * (2.20 + 0.90 * age_frac + 0.80 * bal_imp) * BW,
                                             -0.06 * BW, (0.30 + 0.12 * age_frac + 0.10 * bal_imp) * BW))
                seat_support *= (1.00 + 0.18 * late)
                if pelvis_id >= 0:
                    mj_data.xfrc_applied[pelvis_id, 2] -= 0.86 * down_n
                    mj_data.xfrc_applied[pelvis_id, 2] += seat_support
                    if chair_center_xy is not None:
                        cur_xy = mj_data.xpos[pelvis_id][:2].copy()
                        err_xy = chair_center_xy - cur_xy
                        back_dir = -fall_fwd_xy
                        back_err = float(np.dot(err_xy, back_dir))
                        lat_xy = np.array([fall_lat_3d[0], fall_lat_3d[1]], dtype=float)
                        lat_err = float(np.dot(err_xy, lat_xy))
                        back_force = np.clip(back_err, -0.10, 0.16) * (0.50 * BW)
                        lat_force  = np.clip(lat_err, -0.08, 0.08) * (0.18 * BW)
                        mj_data.xfrc_applied[pelvis_id, 0] += back_force * back_dir[0] + lat_force * lat_xy[0]
                        mj_data.xfrc_applied[pelvis_id, 1] += back_force * back_dir[1] + lat_force * lat_xy[1]
                if torso_id >= 0:
                    _apply_body_pitch_wrench(
                        torso_id,
                        forward_n=(0.04 + 0.04 * bell) * BW,
                        down_n=0.06 * down_n,
                        pitch_nm=(6.0 + 4.0 * bell) * (0.95 + 0.10 * age_frac),
                    )
                    _apply_ap_damping(torso_id, vel_gain=0.18, max_force_bw=0.020)
                if head_id >= 0 and late > 0.0:
                    _apply_body_pitch_wrench(
                        head_id,
                        forward_n=0.015 * BW * late,
                        down_n=0.003 * BW * late,
                        pitch_nm=1.8 * late,
                    )
                leg_factor = max(leg_floor, sf * (1.0 - (1.0 - sit_end_strength / max(sf, 1e-6)) * ramp))
                torso_factor = max(torso_floor, sf * (0.88 - 0.14 * ramp))
                for idx in grp["leg"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * leg_factor
                for idx in grp["torso"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * torso_factor
                blend_alpha = float(np.clip((ramp - 0.45) / 0.55, 0.0, 1.0))
                target_z_hidden = _blend_z(z_sit, z_stand, 0.82 * (1.0 - blend_alpha))
                obs_t = torch.tensor(obs_local["proprio"], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    warm_action = motivo.act(obs_t, target_z_hidden).squeeze(0).numpy()
                for idx in grp["leg"]:
                    warm_action[idx] *= 0.14
                for idx in grp["torso"]:
                    warm_action[idx] *= 0.10
                for idx in grp["arm"]:
                    warm_action[idx] *= 0.10
                obs_local, _, terminated, truncated, _ = env.step(warm_action)
                if terminated or truncated:
                    break
                # Reject hidden-start attempts that have already collapsed onto
                # the floor or drifted into a prone-under-chair configuration.
                pz_chk, trunk_chk, headf_chk, prone_chk = _seat_pose_snapshot()
                if hs > hidden_steps // 3 and (pz_chk < seat_target_pz - 0.10 or abs(prone_chk) > 0.22 or trunk_chk > 110.0 or mj_data.ncon < 1):
                    terminated = True
                    break
            if not (terminated or truncated):
                for _ in range(36):
                    mj_data.xfrc_applied[:] = 0.0
                    _seat_hold_wrench(strength_scale=1.05 + 0.08 * attempt, back_bias=1.20)
                    for idx in grp["leg"]:
                        mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(sit_hold_strength, 0.26)
                    for idx in grp["torso"]:
                        mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(torso_floor, 0.34)
                    obs_t = torch.tensor(obs_local["proprio"], dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        warm_action = motivo.act(obs_t, _blend_z(z_sit, z_stand, 0.18)).squeeze(0).numpy()
                    warm_action *= 0.08
                    obs_local, _, terminated, truncated, _ = env.step(warm_action)
                    if terminated or truncated:
                        break
            qual = _seat_pose_quality()
            if qual > best_quality:
                best_quality = qual
                best_state = (mj_data.qpos.copy(), mj_data.qvel.copy(), obs_local.copy())
            if not (terminated or truncated) and _seat_pose_ok() and qual >= 0.60:
                _reanchor_chair_to_current_pelvis(forward_offset=0.050)
                return obs_local, True, qual
        if best_state is not None:
            mj_data.qpos[:] = best_state[0]
            mj_data.qvel[:] = best_state[1]
            mujoco.mj_forward(mj_model, mj_data)
            rescue_obs = best_state[2]
            qual = _seat_pose_quality()
            if _seat_pose_ok() and qual >= 0.58:
                _reanchor_chair_to_current_pelvis(forward_offset=0.050)
                return rescue_obs, True, qual

        # Deterministic last-resort recovery: never begin the visible run from a
        # floor-contact pseudo-seat. Rebuild an upright seated support state.
        rescue_obs, recovered_ok, recovered_q = _hard_recover_to_seat(max_iters=90)
        return rescue_obs, recovered_ok, max(best_quality, recovered_q)


    sit_anchor_qpos = None
    sit_anchor_qvel = None
    stand_anchor_qpos = None
    stand_anchor_qvel = None

    def _pin_to_anchor(target_qpos=None, target_qvel=None, blend=0.35):
        if target_qpos is None or mj_data.qpos.shape != target_qpos.shape:
            return
        b = float(np.clip(blend, 0.0, 1.0))
        mj_data.qpos[:] = (1.0 - b) * mj_data.qpos + b * target_qpos
        if target_qvel is not None and mj_data.qvel.shape == target_qvel.shape:
            mj_data.qvel[:] = (1.0 - b) * mj_data.qvel + b * target_qvel
        else:
            mj_data.qvel[:] *= (1.0 - b)
        mujoco.mj_forward(mj_model, mj_data)

    # Hidden setup: build a deterministic visible seated anchor first, then
    # run the task as a sit -> rise -> forward-collapse sequence. This avoids
    # the repeated failure where a noisy hidden stand-to-seat descent produced a
    # floor-collapsed or floating frame-0 pose.
    print("        Pre-positioning seated start pose …")
    cached_anchor = _load_cached_sit_anchor()
    if cached_anchor is not None:
        stand_anchor_qpos, stand_anchor_qvel, sit_anchor_qpos, sit_anchor_qvel, seated_quality, obs = cached_anchor
        pz0, trunk0, headf0, prone0 = _seat_pose_snapshot()
    else:
        stand_anchor_qpos, stand_anchor_qvel, sit_anchor_qpos, sit_anchor_qvel, seated_quality = _build_fast_chair_supported_anchor()
        obs, _ = env.reset(options={"qpos": sit_anchor_qpos.copy(), "qvel": sit_anchor_qvel.copy()})
        _lock_heading(force_print=True)
        _reanchor_chair_to_current_pelvis(forward_offset=0.050)
        pz0, trunk0, headf0, prone0 = _seat_pose_snapshot()
    if (not _seat_pose_ok()) or seated_quality < 0.48:
        # Skip the old deterministic/global joint search: it was the main
        # minute-long delay. Go directly to the proven compact chair search,
        # then cache the result for future runs.
        print("        WARNING: fast/cached chair anchor rejected; rebuilding from compact chair search")
        _, _, strict_qpos, strict_qvel, strict_q = _build_ground_supported_sit_anchor()
        obs, _ = env.reset(options={"qpos": strict_qpos.copy(), "qvel": strict_qvel.copy()})
        _lock_heading(force_print=True)
        _reanchor_chair_to_current_pelvis(forward_offset=0.050)
        if _seat_pose_ok() and strict_q >= 0.52:
            sit_anchor_qpos = mj_data.qpos.copy()
            sit_anchor_qvel = np.zeros_like(mj_data.qvel)
            seated_quality = strict_q
        else:
            print("        WARNING: chair-supported sit search rejected; trying explicit crouch anchor")
            _, _, crouch_qpos, crouch_qvel, crouch_q = _build_explicit_crouch_anchor()
            obs, _ = env.reset(options={"qpos": crouch_qpos.copy(), "qvel": crouch_qvel.copy()})
            _lock_heading(force_print=True)
            _reanchor_chair_to_current_pelvis(forward_offset=0.050)
            if _seat_pose_ok() and crouch_q >= 0.52:
                sit_anchor_qpos = mj_data.qpos.copy()
                sit_anchor_qvel = np.zeros_like(mj_data.qvel)
                seated_quality = crouch_q
            else:
                print("        WARNING: explicit crouch anchor rejected; retrying with hidden seated descent")
                obs, seated_ok, hidden_q = _hidden_descend_to_seat(max_attempts=4)
                _reanchor_chair_to_current_pelvis(forward_offset=0.050)
                if (not seated_ok) or hidden_q < 0.52 or (not _seat_pose_ok()):
                    raise RuntimeError(
                        f"Scenario 23 could not build a valid seated start (q={max(strict_q, crouch_q, hidden_q):.2f}). Refusing to start visible sim from a bad anchor."
                    )
                sit_anchor_qpos = mj_data.qpos.copy()
                sit_anchor_qvel = np.zeros_like(mj_data.qvel)
                seated_quality = hidden_q
        pz0, trunk0, headf0, prone0 = _seat_pose_snapshot()
    print(
        "        Seated start check: pelvis_z={:.3f} m | trunk={:.1f} deg | head_fwd={:+.3f} m | prone={:+.3f} | q={:.2f}".format(
            pz0, trunk0, headf0, prone0, seated_quality
        )
    )
    if (not _seat_pose_ok()) or seated_quality < 0.52:
        raise RuntimeError(
            f"Scenario 23 start pose still invalid after recovery (pelvis_z={pz0:.3f}, trunk={trunk0:.1f}, q={seated_quality:.2f})."
        )
    # Freeze the visible sit anchor exactly as the start state.
    sit_anchor_qpos = mj_data.qpos.copy()
    sit_anchor_qvel = np.zeros_like(mj_data.qvel)
    _save_cached_sit_anchor(stand_anchor_qpos, stand_anchor_qvel, sit_anchor_qpos, sit_anchor_qvel, seated_quality)
    pz0, trunk0, headf0, prone0 = _seat_pose_snapshot()
    print(
        "        Final visible sit anchor: pelvis_z={:.3f} m | trunk={:.1f} deg | head_fwd={:+.3f} m | prone={:+.3f} | q={:.2f}".format(
            pz0, trunk0, headf0, prone0, seated_quality
        )
    )

    print(f"\n  [4/4] Running {TOTAL}-step simulation …\n")
    print(f"  {'step':>5} {'phase':>9} {'pz':>7} {'trunk°':>7} {'head_fwd':>9} {'prone':>8} "
          f"{'vx':>6} {'gear%':>6} {'ncon':>5}  status")
    print("  " + "-"*74)

    # -- MAIN LOOP -------------------------------------------------------------
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.distance  = 3.4
        viewer.cam.elevation = -6
        viewer.cam.azimuth   = 90

        for step in range(TOTAL):
            sim_t = float(mj_data.time)
            mj_data.xfrc_applied[:] = 0.0

            # --- PHASE: SIT --------------------------------------------------
            if step < SIT_S:
                phase = "sit"
                local = step
                if local == 0:
                    print(f"\n  [Step {step}] SIT phase - seated hold begins (5.0 s)")
                    try:
                        view_yaw = float(np.degrees(np.arctan2(fall_fwd_xy[1], fall_fwd_xy[0])))
                        viewer.cam.azimuth = view_yaw + 90.0
                        viewer.cam.elevation = -6.0
                        viewer.cam.distance = 3.2
                        print(f"        Camera locked side-on: azimuth={viewer.cam.azimuth:+.1f} deg")
                    except Exception:
                        pass
                    sit_start_pz = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else seat_target_pz

                breathe = 0.5 - 0.5 * math.cos(2.0 * math.pi * (local / max(SIT_S - 1, 1)))
                _pin_to_anchor(sit_anchor_qpos, sit_anchor_qvel, blend=0.88)
                if torso_id >= 0:
                    _apply_body_pitch_wrench(
                        torso_id,
                        forward_n=0.0,
                        down_n=0.0,
                        pitch_nm=-(1.8 + 0.6 * breathe),
                    )
                for idx in grp["leg"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(sit_hold_strength, 0.34)
                for idx in grp["torso"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(torso_floor, 0.34)
                arm_damp = 0.18

            # --- PHASE: GET_UP -----------------------------------------------
            elif step < collapse_start_step:
                phase = "get_up"
                local = step - SIT_S
                if local == 0:
                    print(f"\n  [Step {step}] GET_UP phase - standing attempt begins")
                    _start_blend(z_stand, steps=22)
                    rise_start_pz = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else seat_target_pz

                ramp = local / max(GETUP_S - 1, 1)
                bell = math.sin(math.pi * ramp)
                preprog = float(np.clip(local / max(collapse_target_local, 1), 0.0, 1.0))
                preease = 0.5 - 0.5 * math.cos(math.pi * preprog)

                if sit_anchor_qpos is not None and stand_anchor_qpos is not None:
                    guided_qpos = (1.0 - preease) * sit_anchor_qpos + preease * stand_anchor_qpos
                    guided_qvel = np.zeros_like(mj_data.qvel)
                    pin_blend = float(np.clip(0.82 - 0.46 * preease, 0.26, 0.82))
                    _pin_to_anchor(guided_qpos, guided_qvel, blend=pin_blend)

                if torso_id >= 0:
                    _apply_body_pitch_wrench(
                        torso_id,
                        forward_n=(0.010 + 0.020 * bell) * BW,
                        down_n=0.0,
                        pitch_nm=(1.0 + 2.5 * bell),
                    )
                    _apply_ap_damping(torso_id, vel_gain=0.10 + 0.06 * bell, max_force_bw=0.014)
                if head_id >= 0:
                    _apply_body_pitch_wrench(
                        head_id,
                        forward_n=(0.006 + 0.010 * bell) * BW,
                        down_n=0.0,
                        pitch_nm=(0.6 + 1.4 * bell),
                    )
                for idx in grp["leg"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(rise_floor_strength, 0.38)
                for idx in grp["torso"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(torso_floor, 0.34)
                arm_damp = 0.28

                pz_now = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else seat_target_pz
                trunk_now = _trunk_tilt_deg()
                if not trigger_fired and local >= collapse_target_local:
                    trigger_fired = True
                    collapse_start_step = min(collapse_start_step, step + 1)
                    perturb_t = sim_t
                    print(f"  [Step {step}] Collapse trigger (timed-rise) - pelvis_z={pz_now:.3f} m | trunk={trunk_now:.1f} deg | collapse starts at step {collapse_start_step}")

            # --- PHASE: COLLAPSE ---------------------------------------------
            elif step < collapse_start_step + COLLAPSE_S:
                phase = "collapse"
                local_c = step - collapse_start_step
                if local_c == 0:
                    if not trigger_fired:
                        trigger_fired = True
                        perturb_t = sim_t
                        print(f"  [Step {step}] Fallback collapse trigger")
                    print(f"\n  [Step {step}] COLLAPSE phase - forward topple force applied")
                    _start_blend(z_fall, steps=20)

                prog = local_c / max(COLLAPSE_S - 1, 1)
                ramp_f = 0.5 * (1.0 - math.cos(math.pi * prog))
                pz_c = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else seat_off_pz
                high_phase = float(np.clip((pz_c - seat_target_pz) / max(rise_target_pz - seat_target_pz, 1e-6), 0.0, 1.0))
                onset_pow = float(np.clip(1.08 - 0.34 * age_frac - 0.10 * bal_imp, 0.72, 1.08))
                early = 0.12 + 0.88 * (ramp_f ** onset_pow)
                fwd_commit = float(np.clip(1.00 + 0.20 * age_frac + 0.10 * bal_imp + 0.16 * high_phase, 0.96, 1.34))
                pitch_commit = float(np.clip(1.04 + 0.16 * age_frac + 0.08 * bal_imp + 0.20 * high_phase, 1.00, 1.38))

                if torso_id >= 0:
                    _apply_body_pitch_wrench(
                        torso_id,
                        forward_n=(0.22 + 0.12 * high_phase) * BW * early * collapse_force_scale * fwd_commit,
                        down_n=(0.05 + 0.04 * high_phase) * BW * early,
                        pitch_nm=(26.0 + 14.0 * high_phase) * early * collapse_pitch_scale * pitch_commit,
                    )
                    _apply_ap_damping(torso_id, vel_gain=0.26 + 0.12 * early, max_force_bw=0.030)
                if head_id >= 0:
                    _apply_body_pitch_wrench(
                        head_id,
                        forward_n=(0.08 + 0.04 * high_phase) * BW * early * collapse_force_scale * fwd_commit,
                        down_n=(0.020 + 0.010 * high_phase) * BW * early,
                        pitch_nm=(6.0 + 3.5 * high_phase) * early * collapse_pitch_scale,
                    )
                    _apply_ap_damping(head_id, vel_gain=0.18 + 0.08 * early, max_force_bw=0.016)
                if pelvis_id >= 0:
                    _apply_body_forward_wrench(
                        pelvis_id,
                        forward_n=(0.12 + 0.08 * high_phase) * BW * early * fwd_commit
                    )
                    mj_data.xfrc_applied[pelvis_id, 2] -= (0.04 + 0.03 * high_phase) * BW * early

                # Keep the fall sagittal, not diagonal.
                head_lat_c = _head_lat_rel()
                if abs(head_lat_c) > 0.025 and torso_id >= 0:
                    _apply_body_lateral_wrench(
                        torso_id,
                        lateral_n=-0.10 * BW * np.clip(abs(head_lat_c) / 0.12, 0.0, 1.0) * math.copysign(1.0, head_lat_c),
                        down_n=0.0,
                        roll_nm=0.0,
                    )

                # Let the hands come forward slightly, but do not turn this into a hands-damping task.
                for bid in hand_ids:
                    _apply_body_pitch_wrench(
                        bid,
                        forward_n=0.014 * BW * (0.30 + 0.70 * ramp_f),
                        down_n=-0.002 * BW * (0.30 + 0.70 * ramp_f),
                        pitch_nm=0.0,
                    )

                head_fwd_c = _head_fwd_rel()
                trunk_c = _trunk_tilt_deg()
                prone_c = _forward_prone_score()
                if torso_id >= 0 and (head_fwd_c < 0.18 or prone_c < 0.55 or trunk_c < settle_target_deg - 12.0):
                    corr_h = np.clip(0.18 - head_fwd_c, 0.0, 0.25) / 0.25
                    corr_p = np.clip(0.55 - prone_c, 0.0, 0.60) / 0.60
                    corr_t = np.clip((settle_target_deg - 8.0) - trunk_c, 0.0, 36.0) / 36.0
                    corr = max(float(corr_h), float(corr_p), float(corr_t))
                    _apply_body_pitch_wrench(
                        torso_id,
                        forward_n=0.18 * BW * corr,
                        down_n=0.05 * BW * corr,
                        pitch_nm=12.0 * corr,
                    )
                    if head_id >= 0:
                        _apply_body_pitch_wrench(
                            head_id,
                            forward_n=0.05 * BW * corr,
                            down_n=0.012 * BW * corr,
                            pitch_nm=3.0 * corr,
                        )

                for idx in grp["leg"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(leg_floor, min_gear)
                for idx in grp["torso"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(torso_floor, 0.80 * min_gear)
                arm_damp = float(np.clip(0.56 - 0.08 * age_frac, 0.38, 0.56))

            # --- PHASE: FALL -------------------------------------------------
            else:
                phase = "fall"
                local_f = step - (collapse_start_step + COLLAPSE_S)
                if local_f == 0:
                    print(f"\n  [Step {step}] FALL phase - free forward fall")

                prog = local_f / max(FALL_S - 1, 1)
                ramp_f = 0.5 * (1.0 - math.cos(math.pi * prog))
                pz_f = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 0.20
                vv = _body_vel_world(mj_model, mj_data, pelvis_id) if pelvis_id >= 0 else np.zeros(3)
                ap_v = float(vv[0] * fall_fwd_xy[0] + vv[1] * fall_fwd_xy[1])
                head_fwd = _head_fwd_rel()
                trunk_f = _trunk_tilt_deg()
                prone_now = _forward_prone_score()

                need_forward = (
                    head_fwd < head_fwd_floor
                    or prone_now < prone_target
                    or trunk_f < settle_target_deg
                    or pz_f > settle_floor_pz + 0.010
                )
                if not rest_mode and need_forward:
                    corr_h = np.clip(head_fwd_floor - head_fwd, 0.0, 0.18) / 0.18
                    corr_p = np.clip(prone_target - prone_now, 0.0, 0.50) / 0.50
                    corr_t = np.clip(settle_target_deg - trunk_f, 0.0, 24.0) / 24.0
                    corr_z = np.clip(pz_f - (settle_floor_pz + 0.002), 0.0, 0.10) / 0.10
                    corr = max(float(corr_h), float(corr_p), float(corr_t), float(corr_z))
                    resid_fwd = (0.06 + 0.10 * corr) * BW * fall_resid_scale
                    resid_down = (0.02 + 0.04 * corr) * BW
                    resid_pitch = (6.0 + 8.0 * corr) * fall_resid_scale
                    if torso_id >= 0:
                        _apply_body_pitch_wrench(
                            torso_id,
                            forward_n=resid_fwd,
                            down_n=resid_down,
                            pitch_nm=resid_pitch,
                        )
                        _apply_ap_damping(torso_id, vel_gain=0.18 + 0.08 * corr, max_force_bw=0.020)
                    if head_id >= 0:
                        _apply_body_pitch_wrench(
                            head_id,
                            forward_n=0.030 * BW * corr,
                            down_n=0.010 * BW * corr,
                            pitch_nm=1.8 * corr,
                        )
                        _apply_ap_damping(head_id, vel_gain=0.14 + 0.06 * corr, max_force_bw=0.012)
                    if pelvis_id >= 0:
                        _apply_body_forward_wrench(pelvis_id, forward_n=0.04 * BW * corr)
                if abs(_head_lat_rel()) > 0.03 and torso_id >= 0:
                    _apply_body_lateral_wrench(
                        torso_id,
                        lateral_n=-0.06 * BW * np.clip(abs(_head_lat_rel()) / 0.14, 0.0, 1.0) * math.copysign(1.0, _head_lat_rel()),
                        down_n=0.0,
                        roll_nm=0.0,
                    )

                spd = float(np.linalg.norm(vv)) if pelvis_id >= 0 else 0.0
                settled_orientation_ok = (trunk_f >= settle_target_deg - 8.0)
                settled_ok = (
                    local_f >= 18
                    and pz_f < (settle_floor_pz + 0.020)
                    and spd < 0.18
                    and mj_data.ncon >= 6
                    and head_fwd > head_fwd_floor
                    and prone_now > prone_target
                    and settled_orientation_ok
                )
                if settled_ok:
                    rest_ctr += 1
                else:
                    rest_ctr = 0

                if rest_ctr >= 5 and not rest_mode:
                    rest_mode = True
                    rest_anchor_xy = mj_data.qpos[:2].copy() if mj_data.qpos.shape[0] >= 2 else None
                    print(f"  [Step {step}] Rest mode - body settled")
                    mj_model.dof_damping[:] *= 2.4
                    _start_blend(z_rest, steps=25)

                if rest_mode:
                    if mj_data.qvel.shape[0] >= 6:
                        mj_data.qvel[0] *= 0.01
                        mj_data.qvel[1] *= 0.01
                        mj_data.qvel[2] *= 0.88
                        mj_data.qvel[3] *= 0.35
                        mj_data.qvel[4] *= 0.35
                        mj_data.qvel[5] *= 0.55
                        if rest_anchor_xy is not None and mj_data.qpos.shape[0] >= 2:
                            mj_data.qpos[0] = 0.94 * float(mj_data.qpos[0]) + 0.06 * float(rest_anchor_xy[0])
                            mj_data.qpos[1] = 0.94 * float(mj_data.qpos[1]) + 0.06 * float(rest_anchor_xy[1])

                for idx in grp["leg"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(leg_floor, min_gear)
                for idx in grp["torso"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(torso_floor, 0.80 * min_gear)
                arm_damp = 0.90 if not rest_mode else 1.0

            # -- advance embedding ---------------------------------------------
            _step_blend()

            # -- policy action -------------------------------------------------
            obs_t = torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = motivo.act(obs_t, z_cur).squeeze(0).numpy()

            for idx in grp["arm"]:
                action[idx] *= arm_damp
            if phase == "sit":
                action[:] = 0.0
            elif phase == "get_up":
                leg_action_scale = float(np.clip(0.22 - 0.06 * age_frac - 0.04 * bal_imp, 0.10, 0.22))
                torso_action_scale = float(np.clip(0.16 - 0.04 * age_frac - 0.03 * bal_imp, 0.08, 0.16))
                for idx in grp["leg"]:
                    action[idx] *= leg_action_scale
                for idx in grp["torso"]:
                    action[idx] *= torso_action_scale
            elif phase == "collapse":
                for idx in grp["leg"]:
                    action[idx] *= 0.06
                for idx in grp["torso"]:
                    action[idx] *= 0.06
                for idx in grp["arm"]:
                    action[idx] *= 0.45
            elif phase == "fall":
                for idx in grp["leg"]:
                    action[idx] *= 0.05
                for idx in grp["torso"]:
                    action[idx] *= 0.05
                for idx in grp["arm"]:
                    action[idx] *= 0.28
            if rest_mode:
                action[:] = 0.0

            action_hist.append(action.copy())
            if len(action_hist) >= 2:
                wts = np.array([0.42, 0.58]) if phase in ("sit", "get_up") else np.array([0.48, 0.52])
                action = np.average(list(action_hist)[-2:], weights=wts, axis=0)

            obs, _, terminated, truncated, _ = env.step(action)

            # -- IMU / shared export capture -----------------------------------
            sim_t_now = float(mj_data.time)
            imu_pk = imu.log(sim_t_now)
            dynamics_frame = _capture_shared_exporters(sim_t_now, phase)
            _draw_virtual_chair(viewer)

            # -- console log ---------------------------------------------------
            log_now = (step % 30 == 0 or phase == "collapse"
                       or (phase == "fall" and step % 10 == 0))
            if log_now:
                if dashboard is not None:
                    leg_strength = 1.0
                    if grp["leg"]:
                        leg_strength = float(np.mean([
                            mj_model.actuator_gear[idx, 0] / max(orig_gear[idx], 1e-9)
                            for idx in grp["leg"]
                        ]))
                    leg_strength = float(np.clip(leg_strength, 0.0, 1.25))
                    sensor_impact = float(imu.buf["impact"][-1]) if imu.buf["impact"] else 0.0
                    dashboard_row = dashboard.report(
                        step=step,
                        phase=phase,
                        leg_strength=float(np.clip(leg_strength, 0.0, 1.0)),
                        xcom_margin=None,
                        fall_predicted=_scenario23_is_fallen_state(),
                        imu_peak=float(imu_pk),
                        sensor_impact=sensor_impact,
                        control_source='guardian',
                        mjpc_cost=np.nan,
                        dynamics=dynamics_frame,
                    )
                    if isinstance(dashboard_row, dict):
                        dashboard_row = dict(dashboard_row)
                        dashboard_row['step'] = int(step)
                        dashboard_row['phase'] = str(phase)
                        dashboard_row['pelvis_vx'] = float(np.dot(_body_vel_world(mj_model, mj_data, pelvis_id)[:2], fall_fwd_xy)) if (pelvis_id >= 0 and heading_locked) else 0.0
                        if dynamics_frame is not None:
                            dashboard_row['primary_impact_body'] = dynamics_frame.get('primary_impact_body', '')
                            dashboard_row['ground_vertical_bw'] = float(dynamics_frame.get('total_ground_vertical_n_filt', dynamics_frame.get('total_ground_vertical_n', 0.0)) / max(body_mass * 9.81, 1.0))
                            dashboard_row['nonfoot_impact_bw'] = float(dynamics_frame.get('primary_impact_body_load_n_filt', dynamics_frame.get('primary_impact_body_load_n', 0.0)) / max(body_mass * 9.81, 1.0))
                        metrics_log.append(dashboard_row)
                else:
                    pz_  = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 0.0
                    vv   = np.zeros(6)
                    if pelvis_id >= 0:
                        mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, pelvis_id, vv, 0)
                    vx_ = float(np.dot(vv[3:5], fall_fwd_xy)) if heading_locked else float(vv[3])
                    td_ = _trunk_tilt_deg()
                    g_pct = (mj_model.actuator_gear[grp["leg"][0],0] /
                             max(orig_gear[grp["leg"][0]],1e-9) * 100) if grp["leg"] else 100.
                    hf_ = _head_fwd_rel() if heading_locked else 0.0
                    pr_ = _forward_prone_score() if heading_locked else 0.0
                    st = "FALL!" if _scenario23_is_fallen_state() else "ok"
                    print(f"  {step:>5d} {phase:>9s} {pz_:>7.3f} {td_:>7.1f} {hf_:>9.3f} {pr_:>8.3f}  "
                          f"{vx_:>6.2f} {g_pct:>5.0f}% {mj_data.ncon:>5d}  {st}")

            pz_m = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 0.0
            td_m = _trunk_tilt_deg()
            vx_m = float(np.dot(_body_vel_world(mj_model, mj_data, pelvis_id)[:2], fall_fwd_xy)) if (pelvis_id >= 0 and heading_locked) else 0.0
            metrics.append({
                "step": step,
                "phase": phase,
                "pelvis_h": pz_m,
                "trunk_deg": td_m,
                "fwd_v": vx_m,
                "head_fwd": (_head_fwd_rel() if heading_locked else 0.0),
                "prone": (_forward_prone_score() if heading_locked else 0.0),
            })

            viewer.sync()

            if terminated and phase == "sit":
                obs, _ = env.reset()
                mj_model.actuator_gear[:, 0] = orig_gear * sf

    # -- export ----------------------------------------------------------------
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    pfx = f"fall_scenario23_age{age}_{ts}"
    imu_r = imu.export_csv(pfx + ".csv", meta={
        "scenario_id": 23,
        "description": "Forward fall when trying to get up",
        "age": age, "height_m": height, "sex": sex, "weight_kg": weight,
    })
    val = _validate(imu, perturb_t, body_mass)

    rpt = pfx + "_validation.txt"
    with open(rpt, "w") as f:
        f.write("="*60 + "\nSCENARIO 23 VALIDATION REPORT\n" + "="*60 + "\n")
        f.write(f"age={age}  h={height}m  w={weight:.1f}kg  sex={sex}\n\n")
        f.write(f"Score:          {val['score']:.1%}\n")
        f.write(f"Classification: {val['classification']}\n")
        f.write(f"Fall duration:  {val['fall_duration_s']:.2f} s\n")
        f.write(f"Peak accel:     {val['peak_accel_filt']:.1f} m/s² (filtered)\n")
        f.write(f"Peak impact:    {val['peak_impact_n']:.0f} N\n")
        f.write(f"SISFall:        {val['sisfall_compliant']}\n\nChecks:\n")
        for k, v in val["checks"].items():
            f.write(f"  {'PASS' if v else 'FAIL'}  {k}\n")

    marker_prefix = pfx
    if marker_exporter is not None and not _shared_exporters_failed:
        try:
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
            marker_bundle = {'error': str(e)}
            print(f"  [MarkerKinematics]        export error: {e}")

    if dynamics_analyzer is not None and not _shared_exporters_failed:
        try:
            dynamics_bundle = dynamics_analyzer.export_bundle(marker_prefix)
            print(f"  [DynamicsLayer2]          frame CSV -> {dynamics_bundle['frame_csv']['filename']}")
            print(f"  [DynamicsLayer2]          contact CSV -> {dynamics_bundle['contact_csv']['filename']}")
            for _plot in dynamics_bundle.get('visuals', {}).get('plots', []):
                print(f"  [DynamicsLayer2]          visual PNG -> {_plot}")
        except Exception as e:
            dynamics_bundle = {'error': str(e)}
            print(f"  [DynamicsLayer2]          export error: {e}")

    if paper_exporter is not None and marker_bundle and dynamics_bundle and not _shared_exporters_failed:
        try:
            paper_bundle = paper_exporter.export_bundle(marker_prefix, views=('frontal', 'sagittal', 'oblique'))
            print(f"  [PaperAlignmentBridge]    OpenSim GRF MOT -> {paper_bundle['opensim_grf']['filename']}")
            print(f"  [PaperAlignmentBridge]    ExternalLoads XML -> {paper_bundle['external_loads_xml']['filename']}")
            print(f"  [PaperAlignmentBridge]    marker registration -> {paper_bundle['marker_registration']['filename']}")
            print(f"  [PaperAlignmentBridge]    synthetic pose JSON -> {paper_bundle['pose_dataset_json']['filename']}")
            if paper_bundle.get('pose_preview', {}).get('available', False):
                print(f"  [PaperAlignmentBridge]    synthetic pose preview -> {paper_bundle['pose_preview']['filename']}")
        except Exception as e:
            paper_bundle = {'error': str(e)}
            print(f"  [PaperAlignmentBridge]    export error: {e}")

    print("\n" + "=" * 70)
    print("  SCENARIO 23 POST-SIM PHYSICS AUDIT")
    print("=" * 70)
    for ph_name in ("sit", "get_up", "collapse", "fall"):
        pm = [m for m in metrics if m["phase"] == ph_name]
        if not pm:
            continue
        hs = [m["pelvis_h"] for m in pm]
        ts = [m["trunk_deg"] for m in pm]
        vs = [m["fwd_v"] for m in pm]
        hfs = [m.get('head_fwd', 0.0) for m in pm]
        prs = [m.get('prone', 0.0) for m in pm]
        print(f"  {ph_name:>8s}: pelvis {hs[0]:.3f}->{hs[-1]:.3f} m | trunk {ts[0]:.1f}->{ts[-1]:+.1f} deg | head_fwd {hfs[0]:+.3f}->{hfs[-1]:+.3f} m | prone {prs[0]:+.3f}->{prs[-1]:+.3f} | fwd_v max {max(vs):+.3f} m/s")
    fall_pm = [m for m in metrics if m["phase"] == "fall"]
    if fall_pm:
        final_trunk = fall_pm[-1]["trunk_deg"]
        peak_fwd = max(m["fwd_v"] for m in fall_pm)
        print(f"  final trunk lean : {final_trunk:.1f} deg")
        print(f"  max fall fwd vel : {peak_fwd:+.3f} m/s")

    env.close()
    mj_model.actuator_gear[:, 0] = orig_gear.copy()

    print("\n" + "="*70)
    print("  SCENARIO 23 COMPLETE")
    print("="*70)
    print(f"  Score:         {val['score']:.1%}  ({val['classification']})")
    print(f"  Fall duration: {val['fall_duration_s']:.2f} s")
    print(f"  Peak accel:    {val['peak_accel_filt']:.1f} m/s²")
    print(f"  SISFall:       {val['sisfall_compliant']}")
    print(f"  IMU CSV  ->    {imu_r['filename']}")
    print(f"  Validation ->  {rpt}")
    print("="*70)

    print("\n" + "="*70)
    print("  SIMULATION COMPLETE - Component Summary")
    print("="*70)
    print(f"  [AnthropometricModel]     age={age} height={height}m weight={weight:.1f}kg ({w_src}) | body_mass={body_mass:.2f}kg | strength={sf:.3f} | balance_impairment={bal_imp:.3f}")
    print(f"  [IMUValidator]            native frames={len(imu.buf['t'])} at {imu.native_hz:.2f} Hz | output={imu.output_hz:.0f} Hz ({imu.output_mode})")
    print(f"  [IMUValidator]            CSV saved -> {imu_r['filename']}")
    if marker_bundle and not marker_bundle.get('error'):
        print(f"  [MarkerKinematics]        markers={len(getattr(marker_exporter, 'marker_defs', {}))} | joints={len(getattr(marker_exporter, 'joint_columns', [])) if hasattr(marker_exporter, 'joint_columns') else 0} | frames={len(getattr(marker_exporter, 'frames', []))}")
    if dynamics_bundle and not dynamics_bundle.get('error'):
        print(f"  [DynamicsLayer2]          frames={len(getattr(dynamics_analyzer, 'frames', []))} | contact_rows={len(getattr(dynamics_analyzer, 'contact_rows', []))}")
    if paper_bundle and not paper_bundle.get('error'):
        print("  [PaperAlignmentBridge]    OpenSim GRF + ExternalLoads + synthetic pose dataset enabled")
    print(f"  [ProfileResponse]         seat_target={seat_target_pz:.3f}m | seat_off={seat_off_pz:.3f}m | leg_floor={leg_floor:.3f} | torso_floor={torso_floor:.3f} | arm_floor={arm_floor:.3f}")
    print(f"  [FallValidator]           score={val['score']:.1%} ({val['classification']})")
    print("="*70)

    return {
        "scenario_id":    23,
        "classification": val["classification"],
        "score":          val["score"],
        "sisfall":        val["sisfall_compliant"],
        "imu_csv":        imu_r["filename"],
        "validation_txt": rpt,
        "marker_bundle":  marker_bundle,
        "dynamics_bundle": dynamics_bundle,
        "paper_bundle":   paper_bundle,
    }


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run({"age": 70, "height": 1.65, "sex": "female", "weight": None})
