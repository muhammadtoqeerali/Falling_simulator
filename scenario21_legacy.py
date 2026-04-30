# -*- coding: utf-8 -*-
"""
scenario21_legacy.py  -  COMPLETELY SELF-CONTAINED (no fall_core, no backward_fall_walking_best)
===========================================================================================
Scenario 21: Backward fall when trying to sit down.

Biomechanics
------------
  The subject stands still for ~5 s, then attempts to lower onto a chair.
  During the controlled lowering phase the hips reach the seat region, but the trunk stays too upright and then drifts posteriorly. Extensor control fails and the subject tips backward into a supine landing (Pai & Rogers 1990; Schenkman et al. 1990).

Phase sequence  (all timings at 30 Hz native):
  STAND    150 steps  5.0 s  - stabilise upright with z_stand embedding
  SIT_DOWN  90 steps  3.0 s  - guided downward force lowers pelvis to ~0.50 m;
                               leg gears decay to 40%; blend toward z_fall
  COLLAPSE  30 steps  1.0 s  - backward topple force (0.60 BW ramped);
                               all muscles weaken toward min_gear
  FALL     400 steps 13.3 s  - free backward fall; supine landing; settle

IMPORTANT - safe imports only:
  numpy, torch, mujoco, mujoco.viewer, humenv, metamotivo
  biofidelic_profile  (has no module-level execution)
  Does NOT import backward_fall_walking_best or fall_core or fall_scenario_library.
"""
# --- stdlib -------------------------------------------------------------------
import sys, os, csv, math
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
_ROOT = os.path.dirname(_HERE)
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
_EMBED_CACHE_FILE = os.path.join(_ROOT, "scenario21_embed_cache.pt")

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
    print("  [Embed] z_stand …")
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
    print("  [Embed] z_fall (backward collapse) …")
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
    print(f"  [Embed] z_fall done ({len(coll_obs)} backward-collapse states)")
    return z_fall


def _infer_z_sit_attempt(model):
    """Embed the forward-lean postural shift used during sit-down onset."""
    print("  [Embed] z_sit …")
    env, _ = make_humenv(task="move-ego-0-0")
    pelvis_id = _body_id(env.unwrapped.model, "Pelvis")
    torso_id  = _body_id(env.unwrapped.model, "Torso")
    head_id   = _body_id(env.unwrapped.model, "Head")
    sit_obs = []
    for trial in range(24):
        torch.manual_seed(SEED + 600 + trial)
        z = model.sample_z(1)
        obs, _ = env.reset()
        for _ in range(100):
            if torso_id >= 0:
                env.unwrapped.data.xfrc_applied[torso_id, 0] = 18.0
                env.unwrapped.data.xfrc_applied[torso_id, 4] = -5.0
            obs_t = torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                act = model.act(obs_t, z).squeeze(0).numpy()
            obs, _, term, trunc, _ = env.step(act)
            d = env.unwrapped.data
            ph = float(d.xpos[pelvis_id][2]) if pelvis_id >= 0 else 1.0
            if ph > 0.82 and head_id >= 0 and pelvis_id >= 0:
                vec = d.xpos[head_id] - d.xpos[pelvis_id]
                nv = float(np.linalg.norm(vec))
                if nv > 1e-6:
                    lean = float(np.degrees(np.arccos(np.clip(np.dot(vec / nv, [0, 0, 1]), -1, 1))))
                    if 5.0 <= lean <= 30.0:
                        sit_obs.append(obs["proprio"].copy())
            if term or trunc:
                break
    env.close()
    if len(sit_obs) >= 40:
        obs_t = torch.tensor(np.array(sit_obs[-300:]), dtype=torch.float32)
        with torch.no_grad():
            z_sit = model.goal_inference(obs_t).mean(dim=0, keepdim=True)
    else:
        z_sit = model.sample_z(1)
    print(f"  [Embed] z_sit done ({len(sit_obs)} lean states)")
    return z_sit


def _infer_z_rest(model):
    """Embed a stable forward-prone rest posture instead of a generic ground pose."""
    print("  [Embed] z_rest (supine settle) …")
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
    print(f"  [Embed] z_rest done ({len(prone_obs)} supine-settle states)")
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


def _compute_task21_profile_schedule(age, sex, height, weight):
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


def _print_task21_extended_report(
    *, age, sex, height, weight, body_mass, sf, rt_steps, bal_imp,
    phase_steps, min_gear, leg_floor, torso_floor, arm_floor,
    collapse_force_scale, collapse_pitch_scale, fall_resid_scale,
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
        fall_type="backward_stumble",
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
        imu_data, "backward_stumble",
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
    print(f"  Collapse scales     : force={collapse_force_scale:.3f} pitch={collapse_pitch_scale:.3f} fall_resid={fall_resid_scale:.3f}")

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
    Entry-point called by fall_dispatcher or directly.
    subject_params: {age, height, sex, weight (optional)}
    """
    if subject_params is None:
        print("\n" + "=" * 70)
        print("  Scenario 21 - Backward fall when trying to sit down")
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
    sex    = str(subject_params.get("sex","male")).lower()
    weight, bmi, w_src = _resolve_weight(height, age, sex, subject_params.get("weight"))
    w_str  = f"{subject_params['weight']:.1f}kg (user)" if subject_params.get("weight") is not None else f"{weight:.1f}kg (auto BMI={bmi:.1f})"

    print("\n" + "="*70)
    print("  SCENARIO 21 - Backward fall when trying to sit down")
    print(f"  Subject: age={age}yr  h={height}m  {w_str}  sex={sex}")
    print("="*70)
    print_subject_profile(age, sex, height, weight)

    # -- subject params --------------------------------------------------------
    bal_imp   = float(balance_impairment(age, sex, weight))
    sf        = float(np.clip(muscle_strength_factor(age, sex, weight, height), 0.55, 1.20))
    rt_steps  = max(1, round(reaction_delay_seconds(age, sex, height) * 30))
    min_gear  = float(weakening_config(age, sex, weight)["min_factor"])
    profile_schedule = _compute_task21_profile_schedule(age, sex, height, weight)

    # -- timing constants ------------------------------------------------------
    STAND_S    = int(profile_schedule["stand"])
    SITDOWN_S  = int(profile_schedule["sit_down"])
    COLLAPSE_S = int(profile_schedule["collapse"])
    FALL_S     = int(profile_schedule["fall"])
    TOTAL      = int(profile_schedule["total"])

    print(f"\n  Phases: stand={STAND_S}  sit_down={SITDOWN_S}  "
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
    leg_trail_ids = [bid for bid in [
        _body_id(mj_model, "L_Ankle", "AnkleL", "LeftAnkle"),
        _body_id(mj_model, "R_Ankle", "AnkleR", "RightAnkle"),
        _body_id(mj_model, "L_Foot", "FootL", "LeftFoot"),
        _body_id(mj_model, "R_Foot", "FootR", "RightFoot"),
        _body_id(mj_model, "L_Shank", "ShankL", "LeftShank"),
        _body_id(mj_model, "R_Shank", "ShankR", "RightShank"),
    ] if bid >= 0]


    fall_fwd_xy = np.array([1.0, 0.0], dtype=float)
    fall_lat_3d = np.array([0.0, 1.0, 0.0], dtype=float)
    fall_up_3d = np.array([0.0, 0.0, 1.0], dtype=float)
    heading_locked = False
    fall_sign = -1.0  # backward along avatar sagittal axis

    def _lock_heading():
        nonlocal fall_fwd_xy, fall_lat_3d, fall_up_3d, heading_locked
        fall_fwd_xy, fall_lat_3d, fall_up_3d, yaw = _avatar_forward_xy(mj_model, mj_data, pelvis_id, head_id)
        heading_locked = True
        print(f"        Heading locked: fwd=({fall_fwd_xy[0]:+.3f},{fall_fwd_xy[1]:+.3f})  yaw={yaw:+.1f} deg | lat=({fall_lat_3d[0]:+.3f},{fall_lat_3d[1]:+.3f},{fall_lat_3d[2]:+.3f})")

    def _apply_body_sagittal_wrench(body_id, along_n=0.0, down_n=0.0, pitch_nm=0.0):
        if body_id < 0:
            return
        fx, fy = fall_fwd_xy
        lx, ly, lz = fall_lat_3d
        signed_along = float(along_n) * fall_sign
        signed_pitch = float(pitch_nm) * fall_sign
        mj_data.xfrc_applied[body_id, 0] += signed_along * fx
        mj_data.xfrc_applied[body_id, 1] += signed_along * fy
        mj_data.xfrc_applied[body_id, 2] -= float(down_n)
        mj_data.xfrc_applied[body_id, 3] += signed_pitch * lx
        mj_data.xfrc_applied[body_id, 4] += signed_pitch * ly
        mj_data.xfrc_applied[body_id, 5] += signed_pitch * lz

    def _head_axis_rel():
        if pelvis_id < 0 or head_id < 0:
            return 0.0
        hp = mj_data.xpos[head_id] - mj_data.xpos[pelvis_id]
        return float((hp[0] * fall_fwd_xy[0] + hp[1] * fall_fwd_xy[1]) * fall_sign)

    def _torso_lie_score():
        """+1 when the torso/front aligns with the desired resting orientation."""
        if pelvis_id < 0 or head_id < 0:
            return 0.0
        f2_now, lat_now, up_now, _ = _avatar_forward_xy(mj_model, mj_data, pelvis_id, head_id)
        front_now = np.cross(lat_now, up_now)
        nf = float(np.linalg.norm(front_now))
        if nf < 1e-8:
            return 0.0
        front_now /= nf
        if float(np.dot(front_now[:2], fall_fwd_xy)) < 0.0:
            front_now = -front_now
        return float(front_now[2])

    def _draw_virtual_chair(viewer):
        if chair_center_xy is None or viewer is None:
            return
        try:
            scn = viewer.user_scn
            scn.ngeom = 0
            yaw = math.atan2(fall_fwd_xy[1], fall_fwd_xy[0])
            cy, sy = math.cos(yaw), math.sin(yaw)
            seat_pos = np.array([chair_center_xy[0], chair_center_xy[1], chair_seat_z], dtype=float)
            seat_mat = np.array([[cy, -sy, 0.0],[sy, cy, 0.0],[0.0, 0.0, 1.0]], dtype=float).reshape(-1)
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX,
                                np.array([0.23, 0.20, 0.025], dtype=float),
                                seat_pos, seat_mat,
                                np.array([0.45, 0.30, 0.18, 0.55], dtype=float))
            scn.ngeom += 1
            back_offset = -0.20 * fall_fwd_xy
            back_pos = np.array([chair_center_xy[0] + back_offset[0], chair_center_xy[1] + back_offset[1], chair_seat_z + 0.25], dtype=float)
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX,
                                np.array([0.025, 0.20, 0.25], dtype=float),
                                back_pos, seat_mat,
                                np.array([0.40, 0.27, 0.15, 0.50], dtype=float))
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
                subject_meta={"age": age, "height": height, "sex": sex, "weight": weight, "fall_type": "scenario21_sit_down_backward"},
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
    z_cur = z_stand.clone()

    def _blend_to(z_new, steps=25):
        """Yield smoothly interpolated z from z_cur to z_new over `steps` steps."""
        for k in range(steps + 1):
            yield _blend_z(z_cur, z_new, k / steps)

    _blend_gen  = iter([z_stand])   # idle generator
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
    phase              = "stand"
    perturb_t          = 0.0
    trigger_fired      = False
    collapse_start_step = STAND_S + SITDOWN_S
    sit_end_strength   = 0.40
    rest_mode          = False
    rest_ctr           = 0
    rest_anchor_xy     = None
    sit_anchor_xy      = None
    chair_center_xy    = None
    chair_seat_z       = 0.50
    sitdown_start_pz   = None
    metrics            = []
    metrics_log        = []
    age_frac           = float(np.clip((age - 55.0) / 25.0, 0.0, 1.0))


    fall_resid_scale   = float(np.clip(1.00 - 0.45 * age_frac - 0.30 * bal_imp, 0.25, 1.00))
    collapse_pitch_scale = float(np.clip(1.00 - 0.35 * age_frac - 0.25 * bal_imp, 0.40, 1.00))
    collapse_force_scale = float(np.clip(1.00 - 0.18 * age_frac - 0.12 * bal_imp, 0.70, 1.00))
    settle_target_deg = float(np.clip(94.0 + 2.0 * age_frac + 2.0 * bal_imp, 90.0, 98.0))
    lie_target = float(np.clip(0.32 - 0.08 * age_frac - 0.06 * bal_imp, 0.18, 0.32))
    chair_trigger_pz = float(np.clip(0.54 - 0.02 * age_frac - 0.03 * bal_imp, 0.46, 0.54))
    sit_trigger_local = int(round(24 + 10 * age_frac + 4 * bal_imp))
    leg_floor = float(np.clip(0.16 - 0.04 * age_frac - 0.03 * bal_imp, 0.10, 0.16))
    torso_floor = float(np.clip(0.14 - 0.03 * age_frac - 0.03 * bal_imp, 0.09, 0.14))
    arm_floor = float(np.clip(0.26 - 0.08 * age_frac - 0.06 * bal_imp, 0.14, 0.26))
    action_hist        = deque(maxlen=3)

    print(f"\n  [4/4] Running {TOTAL}-step simulation …\n")
    print(f"  {'step':>5} {'phase':>9} {'pz':>7} {'trunk°':>7} {'head_b':>7} {'supine':>7} "
          f"{'vx':>6} {'gear%':>6} {'ncon':>5}  status")
    print("  " + "-"*70)

    # -- MAIN LOOP -------------------------------------------------------------
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.distance  = 3.6
        viewer.cam.elevation = -8
        viewer.cam.azimuth   = 90

        for step in range(TOTAL):
            sim_t = float(mj_data.time)
            mj_data.xfrc_applied[:] = 0.0     # clear external forces each step

            # --- PHASE: STAND ----------------------------------------------
            if step < STAND_S:
                phase = "stand"
                if step == 0:
                    _start_blend(z_stand, steps=15)
                # Keep leg gears at full strength for stable standing
                for idx in grp["leg"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * sf
                # Damp arm action - quiet arms while standing
                arm_damp = 0.20

            # --- PHASE: SIT_DOWN -------------------------------------------
            elif step < collapse_start_step:
                phase  = "sit_down"
                local  = step - STAND_S

                if local == 0:
                    print(f"\n  [Step {step}] SIT_DOWN phase - guided lowering begins")
                    _lock_heading()
                    try:
                        view_yaw = float(np.degrees(np.arctan2(fall_fwd_xy[1], fall_fwd_xy[0])))
                        viewer.cam.azimuth = view_yaw + 90.0
                        viewer.cam.elevation = -6.0
                        viewer.cam.distance = 3.3
                        print(f"        Camera locked side-on: azimuth={viewer.cam.azimuth:+.1f} deg")
                    except Exception:
                        pass
                    _start_blend(z_sit, steps=30)
                    sitdown_start_pz = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 0.90

                ramp = local / max(SITDOWN_S - 1, 1)
                bell = math.sin(math.pi * ramp)

                # Controlled sit-down should stay almost in place, with the hips
                # moving slightly backward toward the chair while the trunk leans
                # forward. Do NOT let the locomotion policy turn this into walking.
                sit_support_scale = float(np.clip(1.00 - 0.20 * age_frac - 0.14 * bal_imp, 0.62, 1.00))
                down_n = (0.014 + 0.030 * bell) * BW * sit_support_scale
                torso_back_gain = float(np.clip(0.55 + 0.35 * age_frac + 0.20 * bal_imp, 0.45, 1.05))
                torso_sag_n = 0.028 * BW * bell * torso_back_gain
                torso_pitch_nm = 4.2 * bell * torso_back_gain
                if torso_id >= 0:
                    _apply_body_sagittal_wrench(torso_id,
                                               along_n=torso_sag_n,
                                               pitch_nm=torso_pitch_nm,
                                               down_n=0.10 * down_n)
                if pelvis_id >= 0:
                    mj_data.xfrc_applied[pelvis_id, 2] -= 0.88 * down_n
                    # steer pelvis backward toward the chair center and suppress walk drift
                    if chair_center_xy is not None:
                        cur_xy = mj_data.xpos[pelvis_id][:2].copy()
                        err_xy = chair_center_xy - cur_xy
                        back_dir = -fall_fwd_xy
                        back_err = float(np.dot(err_xy, back_dir))
                        lat_xy = np.array([fall_lat_3d[0], fall_lat_3d[1]], dtype=float)
                        lat_n = float(np.linalg.norm(lat_xy))
                        lat_xy = lat_xy / max(lat_n, 1e-8)
                        lat_err = float(np.dot(err_xy, lat_xy))
                        back_force = np.clip(back_err, -0.18, 0.22) * (0.95 * BW)
                        lat_force  = np.clip(lat_err,  -0.12, 0.12) * (0.55 * BW)
                        mj_data.xfrc_applied[pelvis_id, 0] += back_force * back_dir[0] + lat_force * lat_xy[0]
                        mj_data.xfrc_applied[pelvis_id, 1] += back_force * back_dir[1] + lat_force * lat_xy[1]
                    # strong root translation damping to prevent walk-like drift
                    if mj_data.qvel.shape[0] >= 2:
                        mj_data.qvel[0] *= 0.30
                        mj_data.qvel[1] *= 0.30

                # -- Leg gear decays from 100% -> 55% (controlled eccentric sit) --
                leg_factor = max(leg_floor, sf * max(1.0 - 0.22 * ramp, 0.78))
                torso_factor = max(torso_floor, sf * max(1.0 - 0.18 * ramp, 0.70))
                for idx in grp["leg"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * leg_factor
                for idx in grp["torso"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * torso_factor

                arm_damp = float(np.clip(0.35 + 0.10 * sf - 0.15 * bal_imp, 0.20, 0.50))

                # Check trigger only once the avatar is actually low enough to be
                # near the seat height. This avoids an early walk-like collapse.
                pz_now = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 1.0
                if pelvis_id >= 0 and sitdown_start_pz is not None:
                    target_end_pz = chair_trigger_pz + 0.085 + 0.055 * age_frac + 0.040 * bal_imp
                    target_pz = sitdown_start_pz + (target_end_pz - sitdown_start_pz) * (ramp ** (1.12 + 0.18 * age_frac))
                    pz_err = target_pz - pz_now
                    support_gain = 1.85 + 1.30 * age_frac + 1.05 * bal_imp
                    support_cap = (0.26 + 0.18 * age_frac + 0.12 * bal_imp) * BW
                    late_guard = float(np.clip((local - 0.74 * SITDOWN_S) / max(0.26 * SITDOWN_S, 1.0), 0.0, 1.0))
                    seat_support = float(np.clip(pz_err * support_gain * BW, -0.06 * BW, support_cap)) * (1.0 - 0.35 * late_guard)
                    mj_data.xfrc_applied[pelvis_id, 2] += seat_support
                head_axis_now = _head_axis_rel()
                backward_ready = (head_axis_now < -0.01) or (local >= int(0.84 * SITDOWN_S))
                if not trigger_fired and pz_now < chair_trigger_pz and local >= sit_trigger_local and backward_ready:
                    trigger_fired = True
                    collapse_start_step = min(collapse_start_step, step + 1)
                    perturb_t = sim_t
                    print(f"  [Step {step}] Collapse trigger - pelvis_z={pz_now:.3f} m | collapse starts at step {collapse_start_step}")

            # --- PHASE: COLLAPSE -------------------------------------------
            elif step < collapse_start_step + COLLAPSE_S:
                phase   = "collapse"
                local_c = step - collapse_start_step

                if local_c == 0:
                    if not trigger_fired:
                        trigger_fired = True
                        perturb_t     = sim_t
                        print(f"  [Step {step}] Fallback collapse trigger")
                    print(f"\n  [Step {step}] COLLAPSE phase - backward topple force applied")
                    _start_blend(z_fall, steps=20)

                prog  = local_c / max(COLLAPSE_S - 1, 1)
                ramp_f = 0.5 * (1 - math.cos(math.pi * prog))  # 0?1
                if mj_data.qvel.shape[0] >= 2:
                    mj_data.qvel[0] *= 0.35
                    mj_data.qvel[1] *= 0.35



                # Failed sit-down backward: the pelvis reaches the seat region,
                # but the trunk remains too upright and then rotates posteriorly.
                early = 0.35 + 0.65 * ramp_f
                torso_back = 0.18 * BW * early * collapse_force_scale
                torso_down = 0.10 * BW * early
                torso_pitch = 24.0 * early * collapse_pitch_scale
                if torso_id >= 0:
                    _apply_body_sagittal_wrench(torso_id,
                                               along_n=torso_back,
                                               down_n=torso_down,
                                               pitch_nm=torso_pitch)
                if head_id >= 0:
                    _apply_body_sagittal_wrench(head_id,
                                               along_n=0.10 * BW * early * collapse_force_scale,
                                               down_n=0.05 * BW * early,
                                               pitch_nm=12.0 * early * collapse_pitch_scale)
                if pelvis_id >= 0:
                    mj_data.xfrc_applied[pelvis_id, 2] -= 0.04 * BW * early
                    mj_data.xfrc_applied[pelvis_id, 0] += 0.06 * BW * early * fall_fwd_xy[0]
                    mj_data.xfrc_applied[pelvis_id, 1] += 0.06 * BW * early * fall_fwd_xy[1]
                    if chair_center_xy is not None and local_c < 14:
                        cur_xy = mj_data.xpos[pelvis_id][:2].copy()
                        rel_xy = cur_xy - chair_center_xy
                        seat_back = float(np.dot(rel_xy, -fall_fwd_xy))
                        if seat_back > -0.18:
                            mj_data.xfrc_applied[pelvis_id, 0] -= 0.08 * BW * early * fall_fwd_xy[0]
                            mj_data.xfrc_applied[pelvis_id, 1] -= 0.08 * BW * early * fall_fwd_xy[1]
                if hand_ids:
                    reach_gain = 0.45 + 0.55 * ramp_f
                    for bid in hand_ids:
                        _apply_body_sagittal_wrench(bid,
                                                   along_n=0.020 * BW * reach_gain,
                                                   down_n=-0.004 * BW * reach_gain,
                                                   pitch_nm=0.0)
                if leg_trail_ids and (_head_axis_rel() > 0.06 or prog > 0.25):
                    trail_gain = 0.35 + 0.65 * ramp_f
                    for bid in leg_trail_ids:
                        _apply_body_sagittal_wrench(bid,
                                                   along_n=-0.020 * BW * trail_gain,
                                                   down_n=0.006 * BW * trail_gain,
                                                   pitch_nm=0.0)

                # If the head is still not travelling behind the pelvis, add
                # extra upper-body backward pitch immediately.
                head_axis_c = _head_axis_rel()
                lie_c = _torso_lie_score()
                if torso_id >= 0 and (head_axis_c < 0.06 or lie_c < lie_target * 0.35):
                    corr_h = np.clip(0.06 - head_axis_c, 0.0, 0.18) / 0.18
                    corr_l = np.clip(lie_target * 0.35 - lie_c, 0.0, 0.35) / 0.35
                    corr = max(float(corr_h), float(corr_l))
                    _apply_body_sagittal_wrench(torso_id,
                                               along_n=0.12 * BW * corr,
                                               down_n=0.04 * BW * corr,
                                               pitch_nm=18.0 * corr)
                    if pelvis_id >= 0:
                        mj_data.xfrc_applied[pelvis_id, 0] += 0.05 * BW * corr * fall_fwd_xy[0]
                        mj_data.xfrc_applied[pelvis_id, 1] += 0.05 * BW * corr * fall_fwd_xy[1]

                # -- All muscles weaken toward min_gear -----------------------
                gear_f = max(leg_floor, sf * (sit_end_strength - prog * (sit_end_strength - leg_floor)))
                torso_f = max(torso_floor, sf * (0.46 - prog * (0.46 - torso_floor)))
                arm_f = max(arm_floor, sf * arm_floor)
                for idx in grp["leg"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * gear_f
                for idx in grp["torso"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * torso_f
                for idx in grp["arm"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * arm_f
                arm_damp = float(np.clip(0.72 + 0.10 * sf - 0.10 * bal_imp, 0.55, 0.90))

            # --- PHASE: FALL -----------------------------------------------
            else:
                phase = "fall"
                local_f = step - (collapse_start_step + COLLAPSE_S)

                if local_f == 0:
                    print(f"\n  [Step {step}] FALL phase - free fall")

                # Keep a short decaying backward-supine residual so the avatar
                # completes the posterior rotation instead of rebounding forward.
                if local_f < 34 and torso_id >= 0:
                    decay = 0.90 ** local_f
                    resid_back = 0.04 * BW * decay * fall_resid_scale
                    resid_pitch = 12.0 * decay * fall_resid_scale
                    _apply_body_sagittal_wrench(torso_id,
                                               along_n=resid_back,
                                               pitch_nm=resid_pitch)
                    if pelvis_id >= 0:
                        mj_data.xfrc_applied[pelvis_id, 0] -= 0.02 * BW * decay * fall_resid_scale * fall_fwd_xy[0]
                        mj_data.xfrc_applied[pelvis_id, 1] -= 0.02 * BW * decay * fall_resid_scale * fall_fwd_xy[1]

                # Keep some residual leg and arm control so the body does not fold unnaturally.
                mj_model.actuator_gear[:, 0] = orig_gear * max(min_gear, sf * min_gear)
                for idx in grp["leg"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(min_gear, sf * leg_floor)
                for idx in grp["torso"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(min_gear, sf * torso_floor)
                for idx in grp["arm"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(min_gear, sf * arm_floor)

                pz_f   = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 1.0
                v6_p   = np.zeros(6)
                if pelvis_id >= 0:
                    mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, pelvis_id, v6_p, 0)
                spd    = float(np.linalg.norm(v6_p[3:5]))

                trunk_f = 0.0
                if pelvis_id >= 0 and head_id >= 0:
                    vec_f = mj_data.xpos[head_id] - mj_data.xpos[pelvis_id]
                    nv_f = float(np.linalg.norm(vec_f))
                    if nv_f > 1e-8:
                        trunk_f = float(np.degrees(np.arccos(np.clip(np.dot(vec_f / nv_f, [0.0, 0.0, 1.0]), -1.0, 1.0))))
                head_axis = _head_axis_rel()

                lie_now = _torso_lie_score()
                # Keep driving rotation until the body is not just horizontal but
                # also anatomically prone (torso forward axis pointing downward).
                if not rest_mode and torso_id >= 0 and local_f < 100:
                    need_backward = (head_axis < 0.10) or (trunk_f < settle_target_deg - 2.0) or (lie_now < lie_target)
                    if need_backward:
                        gain = max(0.0, 1.0 - local_f / 100.0)
                        soft = float(np.clip(1.00 - 0.35 * age_frac - 0.20 * bal_imp, 0.55, 1.00))
                        extra_back = 0.07 * BW * gain * soft
                        extra_pitch = 14.0 * gain * soft
                        _apply_body_sagittal_wrench(torso_id, along_n=extra_back, down_n=0.02*BW*gain*soft, pitch_nm=extra_pitch)
                        if pelvis_id >= 0:
                            mj_data.xfrc_applied[pelvis_id, 0] += 0.03 * BW * gain * soft * fall_fwd_xy[0]
                            mj_data.xfrc_applied[pelvis_id, 1] += 0.03 * BW * gain * soft * fall_fwd_xy[1]
                if local_f < 24:
                    if hand_ids:
                        rg = max(0.0, 1.0 - local_f / 24.0)
                        for bid in hand_ids:
                            _apply_body_sagittal_wrench(bid,
                                                       along_n=0.012 * BW * rg,
                                                       down_n=-0.003 * BW * rg,
                                                       pitch_nm=0.0)
                    if leg_trail_ids and trunk_f > 45.0:
                        tg = max(0.0, 1.0 - local_f / 24.0)
                        for bid in leg_trail_ids:
                            _apply_body_sagittal_wrench(bid,
                                                       along_n=-0.012 * BW * tg,
                                                       down_n=0.005 * BW * tg,
                                                       pitch_nm=0.0)

                # Over-rotation control for older / weaker profiles
                if trunk_f > settle_target_deg + 6.0 and mj_data.qvel.shape[0] >= 6:
                    mj_data.qvel[3] *= 0.65
                    mj_data.qvel[4] *= 0.65
                    mj_data.qvel[5] *= 0.82

                settled_ok = (
                    pz_f < 0.23 and spd < 0.16 and mj_data.ncon >= 6
                    and head_axis > 0.10 and lie_now > lie_target
                    and (settle_target_deg - 8.0) <= trunk_f <= (settle_target_deg + 10.0)
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
                arm_damp = 1.0

            # -- advance embedding ---------------------------------------------
            _step_blend()

            # -- policy action -------------------------------------------------
            obs_t = torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = motivo.act(obs_t, z_cur).squeeze(0).numpy()

            # suppress arm swing while standing; partial in collapse
            for idx in grp["arm"]:
                action[idx] *= arm_damp
            if phase == "sit_down":
                # This is a near-stationary stand-to-sit attempt, not walking.
                # Strongly reduce policy drive so external seat/hip mechanics dominate.
                for idx in grp["leg"]:
                    action[idx] *= 0.40
                for idx in grp["torso"]:
                    action[idx] *= 0.32
                for idx in grp["arm"]:
                    action[idx] *= 0.35
            elif phase == "collapse":
                # Collapse is dominated by failed extensor control and external
                # rotational mechanics. Keep policy action very small so the
                # controller does not hold the trunk upright in young subjects.
                for idx in grp["leg"]:
                    action[idx] *= 0.18
                for idx in grp["torso"]:
                    action[idx] *= 0.12
                for idx in grp["arm"]:
                    action[idx] *= 0.85
            elif phase == "fall":
                for idx in grp["leg"]:
                    action[idx] *= 0.12
                for idx in grp["torso"]:
                    action[idx] *= 0.18
                for idx in grp["arm"]:
                    action[idx] *= 0.45
            # in rest mode zero everything
            if rest_mode:
                action[:] = 0.0

            # temporal smoothing (reduces jitter)
            action_hist.append(action.copy())
            if len(action_hist) >= 2:
                wts = np.array([0.30, 0.70]) if phase == "stand" else np.array([0.45, 0.55])
                action = np.average(list(action_hist)[-2:], weights=wts, axis=0)

            # -- step env -----------------------------------------------------
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
                        fall_predicted=bool((float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 1.0) < 0.35),
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
                    if heading_locked:
                        vx_ = float(np.dot(vv[3:5], fall_fwd_xy))
                    else:
                        vx_ = float(vv[3])
                    td_  = 0.0
                    if pelvis_id >= 0 and head_id >= 0:
                        vec = mj_data.xpos[head_id] - mj_data.xpos[pelvis_id]
                        n_  = np.linalg.norm(vec)
                        if n_ > 1e-8:
                            td_ = float(np.degrees(np.arccos(
                                np.clip(np.dot(vec/n_, [0.,0.,1.]), -1.,1.))))
                    g_pct = (mj_model.actuator_gear[grp["leg"][0],0] /
                             max(orig_gear[grp["leg"][0]],1e-9) * 100) if grp["leg"] else 100.
                    hf_ = _head_axis_rel() if heading_locked else 0.0
                    pr_ = _torso_lie_score() if heading_locked else 0.0
                    st   = "FALL!" if pz_ < 0.35 else "ok"
                    print(f"  {step:>5d} {phase:>9s} {pz_:>7.3f} {td_:>7.1f} {hf_:>7.3f} {pr_:>7.3f}  "
                          f"{vx_:>6.2f} {g_pct:>5.0f}% {mj_data.ncon:>5d}  {st}")

            pz_m = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 0.0
            vv_m = np.zeros(6)
            if pelvis_id >= 0:
                mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, pelvis_id, vv_m, 0)
            vx_m = float(np.dot(vv_m[3:5], fall_fwd_xy)) if heading_locked else float(vv_m[3])
            td_m = 0.0
            if pelvis_id >= 0 and head_id >= 0:
                vec_m = mj_data.xpos[head_id] - mj_data.xpos[pelvis_id]
                n_m = float(np.linalg.norm(vec_m))
                if n_m > 1e-8:
                    td_m = float(np.degrees(np.arccos(np.clip(np.dot(vec_m / n_m, [0.0, 0.0, 1.0]), -1.0, 1.0))))
            metrics.append({"step": step, "phase": phase, "pelvis_h": pz_m, "trunk_deg": td_m, "fwd_v": vx_m, "head_axis": (_head_axis_rel() if heading_locked else 0.0), "lie": (_torso_lie_score() if heading_locked else 0.0)})

            viewer.sync()

            # reset only during stand if env signals terminal
            if terminated and phase == "stand":
                obs, _ = env.reset()
                mj_model.actuator_gear[:, 0] = orig_gear * sf

    # -- export ----------------------------------------------------------------
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    pfx = f"fall_scenario21_age{age}_{ts}"
    imu_r = imu.export_csv(pfx + ".csv", meta={
        "scenario_id": 21,
        "description": "Backward fall when trying to sit down",
        "age": age, "height_m": height, "sex": sex, "weight_kg": weight,
    })
    val = _validate(imu, perturb_t, body_mass)

    # -- validation report -----------------------------------------------------
    rpt = pfx + "_validation.txt"
    with open(rpt, "w") as f:
        f.write("="*60 + "\nSCENARIO 21 VALIDATION REPORT\n" + "="*60 + "\n")
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
    print("  SCENARIO 21 POST-SIM PHYSICS AUDIT")
    print("=" * 70)
    for ph_name in ("stand", "sit_down", "collapse", "fall"):
        pm = [m for m in metrics if m["phase"] == ph_name]
        if not pm:
            continue
        hs = [m["pelvis_h"] for m in pm]
        ts = [m["trunk_deg"] for m in pm]
        vs = [m["fwd_v"] for m in pm]
        hfs = [m.get('head_axis', m.get('head_fwd', 0.0)) for m in pm]
        prs = [m.get('lie', m.get('prone', 0.0)) for m in pm]
        print(f"  {ph_name:>8s}: pelvis {hs[0]:.3f}->{hs[-1]:.3f} m | trunk {ts[0]:.1f}->{ts[-1]:+.1f} deg | head_axis {hfs[0]:+.3f}->{hfs[-1]:+.3f} m | lie {prs[0]:+.3f}->{prs[-1]:+.3f} | fwd_v max {max(vs):+.3f} m/s")
    fall_pm = [m for m in metrics if m["phase"] == "fall"]
    if fall_pm:
        final_trunk = fall_pm[-1]["trunk_deg"]
        peak_fwd = max(m["fwd_v"] for m in fall_pm)
        print(f"  final trunk lean : {final_trunk:.1f} deg")
        print(f"  max fall fwd vel : {peak_fwd:+.3f} m/s")

    env.close()
    mj_model.actuator_gear[:, 0] = orig_gear.copy()

    print("\n" + "="*70)
    print("  SCENARIO 21 COMPLETE")
    print("="*70)
    print(f"  Score:         {val['score']:.1%}  ({val['classification']})")
    print(f"  Fall duration: {val['fall_duration_s']:.2f} s")
    print(f"  Peak accel:    {val['peak_accel_filt']:.1f} m/s²")
    print(f"  SISFall:       {val['sisfall_compliant']}")
    print(f"  IMU CSV  ?     {imu_r['filename']}")
    print(f"  Validation ?   {rpt}")
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
    print(f"  [ProfileResponse]         chair_trigger={chair_trigger_pz:.3f}m | leg_floor={leg_floor:.3f} | torso_floor={torso_floor:.3f} | arm_floor={arm_floor:.3f}")
    print(f"  [FallValidator]           score={val['score']:.1%} ({val['classification']})")
    print("="*70)

    return {
        "scenario_id":    21,
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