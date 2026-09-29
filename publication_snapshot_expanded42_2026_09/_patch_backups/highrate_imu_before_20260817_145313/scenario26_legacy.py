# -*- coding: utf-8 -*-
"""
scenario26_legacy.py  -  chair-supported seated lateral fainting runtime v1
===========================================================================================
Scenario 26: Lateral fall while sitting, caused by fainting.

Biomechanics
------------
  The subject stands still for ~5 s, then attempts to lower onto a chair.
  During the controlled lowering phase the CoM passes anterior to the base of
  support.  Quadriceps eccentric control fails; the subject tips forward at
  ~45-70° trunk lean and cannot recover (Cress et al. 2000; Schultz 1992).

Phase sequence  (all timings at 30 Hz native):
  STAND    150 steps  5.0 s  - stabilise upright with z_stand embedding
  SIT_DOWN  90 steps  3.0 s  - guided downward force lowers pelvis to ~0.50 m;
                               leg gears decay to 40%; blend toward z_fall
  COLLAPSE  30 steps  1.0 s  - lateral topple force (0.65 BW ramped);
                               all muscles weaken toward min_gear
  FALL     400 steps 13.3 s  - free lateral fall; side landing; settle

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

# Shared read-only exporters/dashboard from the task-34 backbone. This is the
# same exporter family used by scenarios 23/24, so task 25 now produces the same
# dashboard, marker, dynamics and paper-alignment outputs when available.
try:
    from backward_fall_walking_best import (
        MarkerKinematicsExporter,
        DynamicsContactAnalyzer,
        PaperAlignmentExporter,
        PhysicsDashboard,
    )
    _HAS_SHARED_EXPORTERS = True
except Exception:
    MarkerKinematicsExporter = None
    DynamicsContactAnalyzer = None
    PaperAlignmentExporter = None
    PhysicsDashboard = None
    _HAS_SHARED_EXPORTERS = False

# --- MuJoCo API constants -----------------------------------------------------
MJOBJ_BODY     = mujoco.mjtObj.mjOBJ_BODY
MJOBJ_ACTUATOR = mujoco.mjtObj.mjOBJ_ACTUATOR
MJOBJ_GEOM     = mujoco.mjtObj.mjOBJ_GEOM

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
_EMBED_CACHE_FILE = os.path.join(_ROOT, "scenario26_embed_cache.pt")

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
    sf = float(np.clip(muscle_strength_factor(age, sex, weight, height), 0.85, 1.15))
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
    print("  [Embed] z_fall (lateral collapse) …")
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
    print(f"  [Embed] z_fall done ({len(coll_obs)} lateral-collapse states)")
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
    """Embed a stable side-lying rest posture instead of a generic ground pose."""
    print("  [Embed] z_rest (side-lying settle) …")
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
    print(f"  [Embed] z_rest done ({len(prone_obs)} prone-settle states)")
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
        self.buf["pelvis_vx"].append(float(np.linalg.norm(v6p[3:5])))
        self.buf["impact"].append(min(imp, 15000.0))
        return float(np.linalg.norm(self._lpA))

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
        print("  Scenario 26 - Lateral fall while sitting, caused by fainting")
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
    print("  SCENARIO 26 - Lateral fall while sitting, caused by fainting [v1-from-s25-v8-lateral]")
    print(f"  Subject: age={age}yr  h={height}m  {w_str}  sex={sex}")
    print("="*70)
    print_subject_profile(age, sex, height, weight)

    # -- timing constants ------------------------------------------------------
    # Scenario 26 starts VISIBLY seated on the chair.  The hidden pre-pass below
    # builds a stable chair-supported sitting anchor before frame 0.
    SIT_HOLD_S    =  90   # 3.0 s seated hold / early faint prodrome
    TRY_RISE_S    =  60   # 2.0 s small attempted rise, not full standing
    FAINT_S       =  90   # 3.0 s seated/partial-rise fainting struggle
    COLLAPSE_S    =  60   # 2.0 s smooth forward faint collapse
    FALL_S        = 390   # fall + settle
    TOTAL         = SIT_HOLD_S + TRY_RISE_S + FAINT_S + COLLAPSE_S + FALL_S

    # -- subject params --------------------------------------------------------
    rt_steps  = max(1, round(reaction_delay_seconds(age, sex, height) * 30))
    bal_imp   = balance_impairment(age, sex, weight)
    sf        = float(np.clip(muscle_strength_factor(age, sex, weight, height), 0.85, 1.15))
    min_gear  = float(weakening_config(age, sex, weight)["min_factor"])
    print(f"\n  Phases: sit_hold={SIT_HOLD_S}  try_rise={TRY_RISE_S}  faint={FAINT_S}  "
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
    _ref_fwd_xy = None
    _ref_lat_3d = None
    _ref_up_3d = None
    _ref_yaw = None
    side_sign = +1.0  # +1 = subject-right lateral fall; set -1 for subject-left

    def _fall_side_xy():
        lat2 = np.array([fall_lat_3d[0], fall_lat_3d[1]], dtype=float)
        n = float(np.linalg.norm(lat2))
        if n < 1e-8:
            lat2 = np.array([-fall_fwd_xy[1], fall_fwd_xy[0]], dtype=float)
            n = float(np.linalg.norm(lat2))
        return float(side_sign) * lat2 / max(n, 1e-8)

    def _capture_reference_heading():
        """Freeze clean upright anatomical heading from reset."""
        nonlocal _ref_fwd_xy, _ref_lat_3d, _ref_up_3d, _ref_yaw
        fwd_xy, lat_3d, up_3d, yaw = _avatar_forward_xy(mj_model, mj_data, pelvis_id, head_id)
        _ref_fwd_xy = np.array(fwd_xy, dtype=float)
        _ref_lat_3d = np.array(lat_3d, dtype=float)
        _ref_up_3d = np.array(up_3d, dtype=float)
        _ref_yaw = float(yaw)

    def _lock_heading(verbose=False):
        nonlocal fall_fwd_xy, fall_lat_3d, fall_up_3d, heading_locked
        if _ref_fwd_xy is None:
            _capture_reference_heading()
        fall_fwd_xy = _ref_fwd_xy.copy()
        fall_lat_3d = _ref_lat_3d.copy()
        fall_up_3d = _ref_up_3d.copy()
        yaw = float(_ref_yaw)
        heading_locked = True
        if verbose:
            print(f"        Heading locked: fwd=({fall_fwd_xy[0]:+.3f},{fall_fwd_xy[1]:+.3f})  yaw={yaw:+.1f} deg | lat=({fall_lat_3d[0]:+.3f},{fall_lat_3d[1]:+.3f},{fall_lat_3d[2]:+.3f})")

    def _reanchor_chair_to_current_pelvis(forward_offset=0.050):
        """Place the visible chair seat slightly behind the current pelvis."""
        nonlocal chair_center_xy
        if pelvis_id < 0:
            chair_center_xy = None
            return
        pelvis_xy = np.array(mj_data.xpos[pelvis_id][:2], dtype=float)
        chair_center_xy = pelvis_xy - float(forward_offset) * fall_fwd_xy

    def _apply_body_forward_wrench(body_id, fwd_n=0.0, down_n=0.0, pitch_nm=0.0):
        if body_id < 0:
            return
        fx, fy = fall_fwd_xy
        lx, ly, lz = fall_lat_3d
        mj_data.xfrc_applied[body_id, 0] += float(fwd_n) * fx
        mj_data.xfrc_applied[body_id, 1] += float(fwd_n) * fy
        mj_data.xfrc_applied[body_id, 2] -= float(down_n)
        # world torque around avatar-lateral axis -> forward pitch
        # Positive torque around the avatar's +lateral axis pitches the trunk/head
        # forward in a right-handed world frame. The previous sign was reversed,
        # which made young/strong subjects rotate head-backward during collapse
        # even while the translational force was forward.
        mj_data.xfrc_applied[body_id, 3] += float(pitch_nm) * lx
        mj_data.xfrc_applied[body_id, 4] += float(pitch_nm) * ly
        mj_data.xfrc_applied[body_id, 5] += float(pitch_nm) * lz

    def _apply_body_lateral_wrench(body_id, lateral_n=0.0, down_n=0.0, roll_nm=0.0):
        """Force/torque toward the selected lateral fall side."""
        if body_id < 0:
            return
        fx, fy = fall_fwd_xy
        side2 = _fall_side_xy()
        mj_data.xfrc_applied[body_id, 0] += float(lateral_n) * side2[0]
        mj_data.xfrc_applied[body_id, 1] += float(lateral_n) * side2[1]
        mj_data.xfrc_applied[body_id, 2] -= float(down_n)
        mj_data.xfrc_applied[body_id, 3] += float(roll_nm) * fx
        mj_data.xfrc_applied[body_id, 4] += float(roll_nm) * fy

    def _head_forward_rel():
        if pelvis_id < 0 or head_id < 0:
            return 0.0
        hp = mj_data.xpos[head_id] - mj_data.xpos[pelvis_id]
        return float(hp[0] * fall_fwd_xy[0] + hp[1] * fall_fwd_xy[1])

    def _head_lateral_rel():
        if pelvis_id < 0 or head_id < 0:
            return 0.0
        hp = mj_data.xpos[head_id] - mj_data.xpos[pelvis_id]
        side2 = _fall_side_xy()
        return float(hp[0] * side2[0] + hp[1] * side2[1])

    def _torso_side_lie_score():
        """+1 ~= selected body side points down toward floor."""
        if pelvis_id < 0 or head_id < 0:
            return 0.0
        _, lat_now, _, _ = _avatar_forward_xy(mj_model, mj_data, pelvis_id, head_id)
        return float(-float(side_sign) * lat_now[2])

    def _torso_prone_score():
        """+1 ~= chest/front points down (prone), -1 ~= chest/front points up."""
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
        return float(-front_now[2])

    def _draw_virtual_chair(viewer):
        """Draw the same visible chair style used in scenarios 23/24.

        Viewer-only chair geoms do not change physics.  The seated controller
        keeps the pelvis supported; the chair is redrawn every frame so it is
        visible during stand, sit-down, faint, collapse, and fall.
        """
        if chair_center_xy is None or viewer is None or chair_seat_z is None:
            return
        try:
            scn = viewer.user_scn
            scn.ngeom = 0
            yaw = math.atan2(fall_fwd_xy[1], fall_fwd_xy[0])
            cy, sy = math.cos(yaw), math.sin(yaw)
            fwd3 = np.array([fall_fwd_xy[0], fall_fwd_xy[1], 0.0], dtype=float)
            lat3 = np.array([fall_lat_3d[0], fall_lat_3d[1], 0.0], dtype=float)
            seat_pos = np.array([chair_center_xy[0], chair_center_xy[1], chair_seat_z], dtype=float)
            seat_mat = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float).reshape(-1)
            rgba_seat = np.array([0.34, 0.34, 0.40, 1.00], dtype=float)
            rgba_back = np.array([0.30, 0.30, 0.36, 1.00], dtype=float)
            rgba_leg = np.array([0.25, 0.25, 0.30, 1.00], dtype=float)

            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX,
                                np.array([chair_half_fwd, chair_half_lat, 0.025], dtype=float),
                                seat_pos, seat_mat, rgba_seat)
            scn.ngeom += 1

            back_offset = -(chair_half_fwd - 0.025) * fall_fwd_xy
            back_pos = np.array([chair_center_xy[0] + back_offset[0],
                                 chair_center_xy[1] + back_offset[1],
                                 chair_seat_z + 0.25], dtype=float)
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX,
                                np.array([0.025, 0.20, 0.25], dtype=float),
                                back_pos, seat_mat, rgba_back)
            scn.ngeom += 1

            leg_half = np.array([0.020, 0.020, max(0.5 * chair_seat_z, 0.18)], dtype=float)
            for sx in (-0.78 * chair_half_fwd, +0.78 * chair_half_fwd):
                for sy_ in (-0.75 * chair_half_lat, +0.75 * chair_half_lat):
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


    def _chair_contact_proxy(mode="seat", gain=1.0):
        """Chair contact proxy with bounded support forces.

        The visible chair is still drawn as viewer geoms because the humenv model
        is already compiled when this scenario is created.  This proxy supplies
        the missing seat/backrest physics with *bounded* one-way forces: support
        during sitting, smooth release during fainting, and no trapping during
        collapse/fall.  The cap prevents the unrealistic 100x-400x BW contact
        bursts that happen when a stiff invisible spring fights the floor solver.
        """
        if chair_center_xy is None or chair_seat_z is None or pelvis_id < 0:
            return
        fwd2 = fall_fwd_xy.astype(float)
        lat2 = np.array([fall_lat_3d[0], fall_lat_3d[1]], dtype=float)
        latn = float(np.linalg.norm(lat2))
        if latn < 1e-8:
            lat2 = np.array([-fwd2[1], fwd2[0]], dtype=float)
        else:
            lat2 /= latn

        pxy = mj_data.xpos[pelvis_id][:2].copy()
        rel = pxy - chair_center_xy
        rf = float(np.dot(rel, fwd2))
        rl = float(np.dot(rel, lat2))
        inside_seat = (abs(rf) <= chair_half_fwd + 0.075 and abs(rl) <= chair_half_lat + 0.065)
        target_pz = float(seat_target_pz)
        g = float(np.clip(gain, 0.0, 1.0))

        if inside_seat and mode not in ("collapse", "fall", "release") and g > 0.0:
            pz = float(mj_data.xpos[pelvis_id][2])
            z_err = target_pz - pz
            if z_err > 0.0:
                vup = float(mj_data.qvel[2]) if mj_data.qvel.shape[0] > 2 else 0.0
                spring = float(np.clip(z_err, 0.0, 0.070)) * (18.0 * BW) * g
                damper = max(0.0, -np.clip(vup, -1.1, 0.0)) * (0.75 * BW) * g
                support = min(spring + damper, support_force_cap_bw * BW * g)
                mj_data.xfrc_applied[pelvis_id, 2] += support

            # Contact damping fades out before collapse.  This gives a stable
            # seated hold without the hips looking glued to the chair.
            if mj_data.qvel.shape[0] > 2:
                if mode == "preseat":
                    mj_data.qvel[2] *= 0.40
                    mj_data.qvel[0] *= 0.36
                    mj_data.qvel[1] *= 0.36
                elif mode == "sit":
                    mj_data.qvel[2] *= 0.52
                    mj_data.qvel[0] *= 0.34
                    mj_data.qvel[1] *= 0.34
                elif mode == "faint" and g > 0.16:
                    fade = float(np.clip(g / 0.40, 0.0, 1.0))
                    mj_data.qvel[2] *= (0.78 - 0.18 * fade)
                    mj_data.qvel[0] *= (0.88 - 0.16 * fade)
                    mj_data.qvel[1] *= (0.88 - 0.16 * fade)

        # Backrest only protects the seated phases. It is disabled during
        # collapse/fall so it cannot pull the avatar into/through the chair.
        back_plane = -(chair_half_fwd + 0.020)
        if mode not in ("collapse", "fall", "release") and g > 0.05 and abs(rl) <= 0.30 and rf < back_plane:
            pen = back_plane - rf
            push = min(float(np.clip(pen, 0.0, 0.15)) * (1.65 * BW) * g, 0.38 * BW * g)
            mj_data.xfrc_applied[pelvis_id, 0] += push * fwd2[0]
            mj_data.xfrc_applied[pelvis_id, 1] += push * fwd2[1]
            if torso_id >= 0:
                mj_data.xfrc_applied[torso_id, 0] += 0.16 * push * fwd2[0]
                mj_data.xfrc_applied[torso_id, 1] += 0.16 * push * fwd2[1]

    # -- shared exporters/dashboard (read-only, no motion changes) ------------
    marker_exporter = None
    dynamics_analyzer = None
    paper_exporter = None
    marker_bundle = None
    dynamics_bundle = None
    paper_bundle = None
    dashboard = None
    metrics_log = []
    _shared_exporters_failed = False
    if _HAS_SHARED_EXPORTERS:
        try:
            marker_exporter = MarkerKinematicsExporter(mj_model, mj_data, export_hz=100.0)
            dynamics_analyzer = DynamicsContactAnalyzer(
                mj_model, mj_data, body_mass=body_mass, leg_length=max(0.60, 0.53 * height)
            )
            paper_exporter = PaperAlignmentExporter(
                marker_exporter, dynamics_analyzer,
                subject_meta={"age": age, "height": height, "sex": sex, "weight": weight, "fall_type": "scenario26_sitting_faint_lateral"},
            )
            print("        Shared exporters enabled (markers + dynamics + paper-alignment)")
        except Exception as e:
            marker_exporter = None
            dynamics_analyzer = None
            paper_exporter = None
            _shared_exporters_failed = True
            print(f"        Shared exporters unavailable: {e}")
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
        nonlocal _shared_exporters_failed
        if marker_exporter is None or dynamics_analyzer is None or _shared_exporters_failed:
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
    phase              = "sit_hold"
    perturb_t          = 0.0
    trigger_fired      = False
    collapse_start_step = SIT_HOLD_S + TRY_RISE_S + FAINT_S
    sit_end_strength   = 0.40
    rest_mode          = False
    rest_ctr           = 0
    rest_anchor_xy     = None
    sit_anchor_xy      = None
    sit_anchor_qpos    = None
    sit_anchor_qvel    = None
    partial_rise_qpos  = None
    chair_center_xy    = None
    chair_seat_z       = None
    metrics            = []
    age_frac           = float(np.clip((age - 55.0) / 25.0, 0.0, 1.0))
    young_frac         = float(np.clip((50.0 - age) / 25.0, 0.0, 1.0))
    bmi_excess         = float(np.clip((bmi - 25.0) / 8.0, 0.0, 1.0))
    high_mass_guard    = float(np.clip(0.55 * young_frac + 0.45 * bmi_excess, 0.0, 1.0))

    # Separate pelvis/root target height from visual chair-board height.
    # The MuJoCo root/pelvis marker is inside the hip complex, not on the visible
    # buttock surface. Therefore the rendered chair board must sit noticeably
    # below the pelvis marker. v4 raised the board too close to the pelvis and
    # then lifted the pelvis during collapse; that caused the floating / arcing
    # collapse. v5 keeps the board lower and lets the body release naturally.
    seat_target_pz     = float(np.clip(0.50 + 0.04 * (height - 1.65) / 0.15 - 0.010 * age_frac - 0.008 * bal_imp, 0.46, 0.56))
    chair_clearance    = float(np.clip(0.132 + 0.014 * bmi_excess + 0.010 * young_frac, 0.125, 0.155))
    chair_seat_z       = float(np.clip(seat_target_pz - chair_clearance, 0.33, 0.43))
    chair_half_fwd     = float(np.clip(0.215 + 0.035 * (height - 1.65) + 0.018 * bmi_excess, 0.205, 0.265))
    chair_half_lat     = float(np.clip(0.190 + 0.020 * (height - 1.65) + 0.018 * bmi_excess, 0.180, 0.235))

    # Subject-aware normal seated posture. Young/strong subjects should not
    # start half-collapsed; older/balance-impaired subjects are allowed a little
    # more stoop. The lateral fainting lean is introduced later in FAINT.
    seated_trunk_target = float(np.clip(14.5 + 5.0 * age_frac + 2.2 * bal_imp - 0.8 * high_mass_guard, 12.0, 22.0))
    seated_head_target  = float(np.clip(0.075 + 0.075 * age_frac + 0.040 * bal_imp - 0.010 * young_frac, 0.055, 0.185))
    seated_trunk_margin = float(np.clip(6.0 + 5.0 * age_frac + 2.5 * bal_imp - 1.0 * high_mass_guard, 4.5, 13.0))
    seated_prone_limit  = float(np.clip(0.44 + 0.15 * age_frac + 0.10 * bal_imp, 0.40, 0.72))
    visible_anchor_blend = float(np.clip(0.012 + 0.020 * high_mass_guard + 0.006 * (1.0 - age_frac), 0.010, 0.038))
    support_force_cap_bw = float(np.clip(1.90 + 0.40 * high_mass_guard, 1.70, 2.35))

    # v8: make the attempted stand-up visibly clear. Previous versions raised
    # the root numerically, but the lift was too sinusoidal/brief and the trunk
    # stayed in a seated/falling lean. This subject-aware lift creates a small
    # but readable chair-unweighting phase before fainting.
    partial_rise_lift = float(np.clip(
        0.086 + 0.020 * young_frac - 0.010 * age_frac - 0.010 * bal_imp - 0.006 * high_mass_guard,
        0.066,
        0.108,
    ))
    partial_rise_hold = float(np.clip(0.82 - 0.10 * age_frac - 0.08 * bal_imp, 0.68, 0.86))

    fall_resid_scale   = float(np.clip(1.00 - 0.55 * age_frac - 0.35 * bal_imp, 0.22, 1.00))
    collapse_pitch_scale = float(np.clip(1.00 - 0.45 * age_frac - 0.30 * bal_imp, 0.35, 1.00))
    collapse_force_scale = float(np.clip(1.00 - 0.20 * age_frac - 0.15 * bal_imp, 0.65, 1.00))
    settle_target_deg = float(np.clip(86.0 - 4.0 * age_frac - 6.0 * bal_imp, 76.0, 88.0))
    side_lie_target = float(np.clip(0.48 - 0.10 * age_frac - 0.08 * bal_imp, 0.26, 0.48))
    chair_trigger_pz = float(np.clip(0.56 - 0.03 * age_frac - 0.03 * bal_imp, 0.48, 0.56))
    sit_trigger_local = int(round(20 + 8 * age_frac))
    leg_floor = float(np.clip(0.18 - 0.05 * age_frac - 0.04 * bal_imp, 0.12, 0.18))
    torso_floor = float(np.clip(0.16 - 0.04 * age_frac - 0.03 * bal_imp, 0.10, 0.16))
    arm_floor = float(np.clip(0.34 - 0.12 * age_frac - 0.10 * bal_imp, 0.18, 0.34))
    action_hist        = deque(maxlen=5)

    print(f"\n  [4/4] Running {TOTAL}-step simulation …")

    # Build the chair before the viewer loop so it is visible from frame 0 and
    # the contact proxy is active during the whole simulation.
    if pelvis_id >= 0:
        _capture_reference_heading()
        _lock_heading()
        chair_center_xy = mj_data.xpos[pelvis_id][:2].copy() - 0.22 * fall_fwd_xy
        # Chair visual/contact proxy initialized from frame 0; keep dashboard clean.

    # --- robust seated-anchor builder ----------------------------------------
    # Scenario 26 needs a true chair-safe seated start.  Do not rely on a hidden
    # force-only descent: that was the source of the pelvis-under-chair bug.
    seat_l_foot_id = _body_id(mj_model, "L_Foot", "FootL", "LeftFoot", "L_Ankle", "AnkleL", "LeftAnkle", "L_Toe", "ToeL", "LeftToe")
    seat_r_foot_id = _body_id(mj_model, "R_Foot", "FootR", "RightFoot", "R_Ankle", "AnkleR", "RightAnkle", "R_Toe", "ToeR", "RightToe")

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

    def _root_seat_servo(target_qpos=None, target_z=None, blend_xy=0.035, blend_z=0.030, damp_xy=0.55, damp_z=0.70):
        """Keep the pelvis/root over the chair without freezing the whole body."""
        if target_qpos is None or mj_data.qpos.shape != target_qpos.shape or mj_data.qpos.shape[0] < 3:
            return
        bx = float(np.clip(blend_xy, 0.0, 0.25))
        bz = float(np.clip(blend_z, 0.0, 0.25))
        mj_data.qpos[0] = (1.0 - bx) * float(mj_data.qpos[0]) + bx * float(target_qpos[0])
        mj_data.qpos[1] = (1.0 - bx) * float(mj_data.qpos[1]) + bx * float(target_qpos[1])
        zt = float(target_z) if target_z is not None else float(target_qpos[2])
        mj_data.qpos[2] = (1.0 - bz) * float(mj_data.qpos[2]) + bz * zt
        if mj_data.qvel.shape[0] >= 3:
            mj_data.qvel[0] *= float(np.clip(damp_xy, 0.0, 1.0))
            mj_data.qvel[1] *= float(np.clip(damp_xy, 0.0, 1.0))
            mj_data.qvel[2] *= float(np.clip(damp_z, 0.0, 1.0))
        mujoco.mj_forward(mj_model, mj_data)

    def _visible_chair_seat_guard(target_z=None, min_clearance=0.085, strength=1.0, xy_strength=0.22):
        """Last-resort visible chair guard for sitting phases.

        The chair is drawn with viewer geoms, so it cannot physically collide
        with the humanoid. This small guarded root correction acts only before
        real collapse, keeping the pelvis/root above the rendered chair board
        and over the chair footprint. It prevents visible chair penetration
        without freezing the later fall.
        """
        if pelvis_id < 0 or sit_anchor_qpos is None or mj_data.qpos.shape[0] < 3:
            return
        g = float(np.clip(strength, 0.0, 1.0))
        if g <= 0.0:
            return
        z_ref = float(target_z) if target_z is not None else float(seat_target_pz)
        safe_z = max(float(chair_seat_z) + float(min_clearance), z_ref)
        pz_now = float(mj_data.xpos[pelvis_id][2])
        changed = False
        if pz_now < safe_z:
            # Only a small correction per frame. This is enough to keep the
            # seated root above the chair, but avoids a hard visible pop.
            dz = min((safe_z - pz_now), 0.018 * g)
            mj_data.qpos[2] += dz
            if mj_data.qvel.shape[0] >= 3:
                mj_data.qvel[2] = max(0.0, float(mj_data.qvel[2])) * 0.12
            changed = True
        if chair_center_xy is not None:
            # Keep the pelvis slightly in front of the chair center/backrest,
            # instead of allowing it to sink backward into the chair footprint.
            desired_xy = np.array(chair_center_xy, dtype=float) + 0.065 * fall_fwd_xy
            err = desired_xy - np.array(mj_data.xpos[pelvis_id][:2], dtype=float)
            xy_gain = float(np.clip(xy_strength, 0.0, 1.0)) * g
            if xy_gain > 0.0:
                mj_data.qpos[0] += np.clip(err[0], -0.030, 0.030) * xy_gain
                mj_data.qpos[1] += np.clip(err[1], -0.030, 0.030) * xy_gain
                if mj_data.qvel.shape[0] >= 2:
                    mj_data.qvel[0] *= 0.25
                    mj_data.qvel[1] *= 0.25
                changed = True
        if changed:
            mujoco.mj_forward(mj_model, mj_data)

    def _chair_exit_guard(target_z=None, forward_min=0.10, min_clearance=0.095, strength=0.70, xy_strength=0.55):
        """Keep the hips above/just in front of the visual chair until they clear it.

        This is deliberately active only during late faint and early collapse.
        It prevents the common visual failure where the pelvis drops through the
        viewer-only chair before the upper body has actually fallen forward.
        """
        if pelvis_id < 0 or mj_data.qpos.shape[0] < 3 or chair_center_xy is None:
            return
        g = float(np.clip(strength, 0.0, 1.0))
        if g <= 0.0:
            return
        z_ref = float(target_z) if target_z is not None else float(seat_target_pz)
        safe_z = max(float(chair_seat_z) + float(min_clearance), z_ref)
        pz_now = float(mj_data.xpos[pelvis_id][2])
        changed = False
        if pz_now < safe_z:
            dz = min((safe_z - pz_now), 0.022 * g)
            mj_data.qpos[2] += dz
            if mj_data.qvel.shape[0] >= 3:
                mj_data.qvel[2] = max(0.0, float(mj_data.qvel[2])) * 0.10
            changed = True

        rel = np.array(mj_data.xpos[pelvis_id][:2], dtype=float) - np.array(chair_center_xy, dtype=float)
        rf = float(np.dot(rel, fall_fwd_xy))
        if rf < float(forward_min):
            push = min(float(forward_min) - rf, 0.045)
            xy_gain = float(np.clip(xy_strength, 0.0, 1.0)) * g
            mj_data.qpos[0] += push * xy_gain * fall_fwd_xy[0]
            mj_data.qpos[1] += push * xy_gain * fall_fwd_xy[1]
            if mj_data.qvel.shape[0] >= 2:
                mj_data.qvel[0] *= 0.22
                mj_data.qvel[1] *= 0.22
            changed = True
        if changed:
            mujoco.mj_forward(mj_model, mj_data)

    def _seat_edge_friction(edge_target=None, max_fwd_speed=0.11, strength=0.65, active_height_margin=0.075):
        """Reduce artificial sliding along the viewer-only chair front edge.

        This does not hold the avatar upright.  It only damps root translation
        while the pelvis is still near/above the chair seat.  The head/trunk can
        continue to rotate forward, then the hips release downward after the
        committed-fall lean gate is reached.
        """
        if pelvis_id < 0 or chair_center_xy is None or mj_data.qvel.shape[0] < 2:
            return 0.0
        fwd2 = np.array(fall_fwd_xy, dtype=float)
        lat2 = np.array([fall_lat_3d[0], fall_lat_3d[1]], dtype=float)
        nlat = float(np.linalg.norm(lat2))
        if nlat < 1e-8:
            lat2 = np.array([-fwd2[1], fwd2[0]], dtype=float)
        else:
            lat2 /= nlat
        pxy = np.array(mj_data.xpos[pelvis_id][:2], dtype=float)
        rel = pxy - np.array(chair_center_xy, dtype=float)
        rf = float(np.dot(rel, fwd2))
        rl = float(np.dot(rel, lat2))
        pz = float(mj_data.xpos[pelvis_id][2])
        if abs(rl) > chair_half_lat + 0.090 or pz < chair_seat_z + float(active_height_margin):
            return rf
        edge = float(edge_target) if edge_target is not None else float(0.62 * chair_half_fwd)
        g = float(np.clip(strength, 0.0, 1.0))
        vxy = np.array(mj_data.qvel[:2], dtype=float)
        vf = float(np.dot(vxy, fwd2))
        vmax = float(max_fwd_speed)
        if vf > vmax:
            mj_data.qvel[0] -= (vf - vmax) * g * fwd2[0]
            mj_data.qvel[1] -= (vf - vmax) * g * fwd2[1]
        if rf > edge:
            excess = min(rf - edge, 0.070)
            mj_data.qpos[0] -= excess * 0.18 * g * fwd2[0]
            mj_data.qpos[1] -= excess * 0.18 * g * fwd2[1]
            mj_data.xfrc_applied[pelvis_id, 0] -= (0.055 * BW * g + 0.25 * BW * excess) * fwd2[0]
            mj_data.xfrc_applied[pelvis_id, 1] -= (0.055 * BW * g + 0.25 * BW * excess) * fwd2[1]
            if mj_data.qvel.shape[0] >= 2:
                mj_data.qvel[0] *= (0.70 + 0.22 * (1.0 - g))
                mj_data.qvel[1] *= (0.70 + 0.22 * (1.0 - g))
            mujoco.mj_forward(mj_model, mj_data)
        return rf

    def _chair_side_exit_guard(target_z=None, side_min=0.10, forward_target=0.09, min_clearance=0.095, strength=0.70, xy_strength=0.55):
        """Keep hips above the visual chair until they have cleared the side edge."""
        if pelvis_id < 0 or mj_data.qpos.shape[0] < 3 or chair_center_xy is None:
            return
        g = float(np.clip(strength, 0.0, 1.0))
        if g <= 0.0:
            return
        side2 = _fall_side_xy()
        z_ref = float(target_z) if target_z is not None else float(seat_target_pz)
        safe_z = max(float(chair_seat_z) + float(min_clearance), z_ref)
        changed = False
        pz_now = float(mj_data.xpos[pelvis_id][2])
        if pz_now < safe_z:
            dz = min((safe_z - pz_now), 0.022 * g)
            mj_data.qpos[2] += dz
            if mj_data.qvel.shape[0] >= 3:
                mj_data.qvel[2] = max(0.0, float(mj_data.qvel[2])) * 0.10
            changed = True
        rel = np.array(mj_data.xpos[pelvis_id][:2], dtype=float) - np.array(chair_center_xy, dtype=float)
        rs = float(np.dot(rel, side2))
        rf = float(np.dot(rel, fall_fwd_xy))
        if rs < float(side_min):
            push = min(float(side_min) - rs, 0.045)
            xy_gain = float(np.clip(xy_strength, 0.0, 1.0)) * g
            mj_data.qpos[0] += push * xy_gain * side2[0]
            mj_data.qpos[1] += push * xy_gain * side2[1]
            changed = True
        ap_err = rf - float(forward_target)
        if abs(ap_err) > 0.018:
            corr = float(np.clip(-ap_err, -0.035, 0.035)) * float(np.clip(xy_strength, 0.0, 1.0)) * g
            mj_data.qpos[0] += corr * fall_fwd_xy[0]
            mj_data.qpos[1] += corr * fall_fwd_xy[1]
            changed = True
        if changed:
            if mj_data.qvel.shape[0] >= 2:
                mj_data.qvel[0] *= 0.24
                mj_data.qvel[1] *= 0.24
            mujoco.mj_forward(mj_model, mj_data)

    def _seat_side_edge_friction(edge_target=None, max_side_speed=0.10, strength=0.65, active_height_margin=0.075):
        """Reduce artificial side sliding over the viewer-only chair edge."""
        if pelvis_id < 0 or chair_center_xy is None or mj_data.qvel.shape[0] < 2:
            return 0.0
        side2 = _fall_side_xy()
        rel = np.array(mj_data.xpos[pelvis_id][:2], dtype=float) - np.array(chair_center_xy, dtype=float)
        rs = float(np.dot(rel, side2))
        rf = float(np.dot(rel, fall_fwd_xy))
        pz = float(mj_data.xpos[pelvis_id][2])
        if abs(rf) > chair_half_fwd + 0.095 or pz < chair_seat_z + float(active_height_margin):
            return rs
        edge = float(edge_target) if edge_target is not None else float(0.62 * chair_half_lat)
        g = float(np.clip(strength, 0.0, 1.0))
        vxy = np.array(mj_data.qvel[:2], dtype=float)
        vs = float(np.dot(vxy, side2))
        vmax = float(max_side_speed)
        if vs > vmax:
            mj_data.qvel[0] -= (vs - vmax) * g * side2[0]
            mj_data.qvel[1] -= (vs - vmax) * g * side2[1]
        if rs > edge:
            excess = min(rs - edge, 0.070)
            mj_data.qpos[0] -= excess * 0.18 * g * side2[0]
            mj_data.qpos[1] -= excess * 0.18 * g * side2[1]
            mj_data.xfrc_applied[pelvis_id, 0] -= (0.055 * BW * g + 0.25 * BW * excess) * side2[0]
            mj_data.xfrc_applied[pelvis_id, 1] -= (0.055 * BW * g + 0.25 * BW * excess) * side2[1]
            if mj_data.qvel.shape[0] >= 2:
                mj_data.qvel[0] *= (0.70 + 0.22 * (1.0 - g))
                mj_data.qvel[1] *= (0.70 + 0.22 * (1.0 - g))
            mujoco.mj_forward(mj_model, mj_data)
        return rs

    def _trunk_tilt_deg():
        if pelvis_id < 0 or head_id < 0:
            return 0.0
        vec = mj_data.xpos[head_id] - mj_data.xpos[pelvis_id]
        nv = float(np.linalg.norm(vec))
        if nv <= 1e-8:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(np.dot(vec / nv, [0.0, 0.0, 1.0]), -1.0, 1.0))))

    def _living_trunk_servo(target_deg=14.0, target_head_fwd=0.06, gain=1.0, max_torque=180.0):
        """Living seated posture support that fades out during fainting."""
        if pelvis_id < 0 or head_id < 0:
            return
        trunk = _trunk_tilt_deg()
        head_fwd = _head_forward_rel()
        ctrl = float(np.clip(0.80 * ((trunk - float(target_deg)) / 55.0) + 0.45 * ((head_fwd - float(target_head_fwd)) / 0.42), -1.0, 1.0))
        if abs(ctrl) < 0.015:
            return
        fwd_corr = -0.060 * BW * ctrl * gain
        lift_corr = -0.012 * BW * max(ctrl, 0.0) * gain
        pitch_corr = -float(max_torque) * ctrl * gain
        if torso_id >= 0:
            _apply_body_forward_wrench(torso_id, fwd_n=fwd_corr, down_n=lift_corr, pitch_nm=pitch_corr)
        if head_id >= 0:
            _apply_body_forward_wrench(head_id, fwd_n=0.42 * fwd_corr, down_n=0.65 * lift_corr, pitch_nm=0.35 * pitch_corr)
        if mj_data.qvel.shape[0] >= 6 and ctrl > 0.0:
            damp = float(np.clip(1.0 - 0.18 * gain, 0.58, 0.96))
            mj_data.qvel[3] *= damp
            mj_data.qvel[4] *= damp
            mj_data.qvel[5] *= max(damp, 0.78)

    def _anti_overlean_guard(max_deg=52.0, gain=1.0, label=""):
        """Keep the faint phase as a sideward prodrome instead of a completed fall."""
        if pelvis_id < 0 or head_id < 0:
            return
        trunk = _trunk_tilt_deg()
        excess = float(np.clip((trunk - float(max_deg)) / 38.0, 0.0, 1.0))
        if excess <= 0.0:
            return
        g = float(np.clip(gain, 0.0, 1.5))
        if torso_id >= 0:
            _apply_body_lateral_wrench(torso_id, lateral_n=-0.110 * BW * excess * g, down_n=-0.010 * BW * excess * g, roll_nm=-40.0 * excess * g)
        if head_id >= 0:
            _apply_body_lateral_wrench(head_id, lateral_n=-0.055 * BW * excess * g, down_n=-0.005 * BW * excess * g, roll_nm=-18.0 * excess * g)
        if mj_data.qvel.shape[0] >= 6:
            damp = float(np.clip(1.0 - 0.36 * excess * g, 0.42, 0.92))
            mj_data.qvel[3] *= damp
            mj_data.qvel[4] *= damp
            mj_data.qvel[5] *= max(damp, 0.70)

    def _seat_foot_metrics():
        zs = []
        for bid in (seat_l_foot_id, seat_r_foot_id):
            if bid >= 0:
                zs.append(float(mj_data.xpos[bid][2]))
        if not zs:
            return 0.0, 0.0, 0
        return float(np.mean(zs)), float(max(zs) - min(zs)) if len(zs) >= 2 else 0.0, len(zs)

    def _seat_pose_snapshot():
        pz = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else seat_target_pz
        trunk = _trunk_tilt_deg()
        head_fwd = _head_forward_rel()
        prone = _torso_prone_score()
        return pz, trunk, head_fwd, prone

    def _seat_pose_quality():
        pz, trunk, head_fwd, prone = _seat_pose_snapshot()
        foot_mean_z, foot_span_z, foot_n = _seat_foot_metrics()
        q_pz = 1.0 - float(np.clip(abs(pz - seat_target_pz) / 0.10, 0.0, 1.0))
        # Scenario 26 starts in a normal seated posture.  A nearly upright trunk
        # (about 7-22 deg) is valid; the forward lean appears during faint/collapse.
        # The previous validator rejected a good run at trunk=8.8 deg.
        q_trunk = 1.0 - float(np.clip(abs(trunk - seated_trunk_target) / 20.0, 0.0, 1.0))
        q_head = 1.0 - float(np.clip(abs(head_fwd - seated_head_target) / 0.20, 0.0, 1.0))
        q_prone = 1.0 - float(np.clip(abs(prone) / seated_prone_limit, 0.0, 1.0))
        q_con = 1.0 if mj_data.ncon >= 1 else 0.45
        if foot_n > 0:
            q_foot_h = 1.0 - float(np.clip(abs(foot_mean_z - 0.05) / 0.16, 0.0, 1.0))
            q_foot_sym = 1.0 - float(np.clip(foot_span_z / 0.16, 0.0, 1.0))
            q_foot = 0.7 * q_foot_h + 0.3 * q_foot_sym
        else:
            q_foot = 0.0
        return 0.30 * q_pz + 0.22 * q_trunk + 0.14 * q_head + 0.08 * q_prone + 0.10 * q_con + 0.16 * q_foot

    def _seat_pose_ok():
        pz, trunk, head_fwd, prone = _seat_pose_snapshot()
        foot_mean_z, foot_span_z, foot_n = _seat_foot_metrics()

        # Hard safety checks only.  Do not reject a valid upright seated start
        # because of foot contact/prone-score noise from the humanoid model.
        pelvis_ok = (seat_target_pz - 0.11) <= pz <= (seat_target_pz + 0.13)
        above_board = pz >= (chair_seat_z + 0.055)
        trunk_hi = float(np.clip(seated_trunk_target + seated_trunk_margin + 5.5, 24.0, 34.0))
        trunk_ok = 3.0 <= trunk <= trunk_hi
        head_ok = -0.30 <= head_fwd <= min(0.31, seated_head_target + 0.18)
        not_floor = pz > 0.36

        # Feet are useful quality hints, but some Meta-Motivo/MuJoCo builds report
        # ankle/toe contact a little differently.  Only reject impossible feet.
        feet_ok = (foot_n == 0) or (foot_mean_z <= 0.34 and foot_span_z <= 0.28)
        prone_ok = abs(prone) <= seated_prone_limit

        return bool(pelvis_ok and above_board and trunk_ok and head_ok and prone_ok and feet_ok and not_floor)

    def _seat_pose_hard_safe():
        """Minimal non-negotiable guard: avatar must be above the visible chair, not inside/floor-collapsed."""
        pz, trunk, head_fwd, prone = _seat_pose_snapshot()
        return bool(
            pz >= (chair_seat_z + 0.055)
            and pz > 0.36
            and pz <= (seat_target_pz + 0.15)
            and 0.0 <= trunk <= float(np.clip(seated_trunk_target + seated_trunk_margin + 12.0, 30.0, 40.0))
            and -0.35 <= head_fwd <= 0.38
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

    def _build_fast_s25_chair_anchor():
        nonlocal chair_center_xy
        base_obs, _ = env.reset()
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
        chair_center_xy = seat_xy - 0.075 * fall_fwd_xy

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
            (1.35, 2.15, 0.55, 0.20),
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
                chair_center_xy = seat_xy - 0.075 * fall_fwd_xy
                terminated = truncated = False
                for _warm in range(30):
                    mj_data.xfrc_applied[:] = 0.0
                    _pin_to_anchor(q, np.zeros_like(mj_data.qvel), blend=0.92)
                    _chair_contact_proxy(mode="preseat", gain=1.05)
                    if pelvis_id >= 0 and float(mj_data.xpos[pelvis_id][2]) < seat_target_pz - 0.025:
                        mj_data.xfrc_applied[pelvis_id, 2] += 0.45 * BW
                    for idx in grp["leg"]:
                        mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(0.72 * sf, 0.50)
                    for idx in grp["torso"]:
                        mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(0.62 * sf, 0.40)
                    obs_local, _, terminated, truncated, _ = env.step(zero_action)
                    if terminated or truncated:
                        break
                if terminated or truncated:
                    continue

                score = float(_seat_pose_quality())
                pz, trunk, head_fwd, prone = _seat_pose_snapshot()
                if pz < chair_seat_z + 0.045:
                    score -= 0.40
                trunk_pref_hi = seated_trunk_target + seated_trunk_margin
                trunk_hard_hi = float(np.clip(trunk_pref_hi + 5.5, 24.0, 34.0))
                if seated_trunk_target - 3.0 <= trunk <= trunk_pref_hi:
                    score += 0.12
                else:
                    score -= 0.020 * max(0.0, trunk - (trunk_pref_hi + 2.0))
                if trunk > trunk_hard_hi:
                    score -= 0.35 + 0.025 * (trunk - trunk_hard_hi)
                if abs(head_fwd - seated_head_target) <= 0.12:
                    score += 0.05
                if abs(prone) <= min(0.32, seated_prone_limit):
                    score += 0.04
                cand = (score, mj_data.qpos.copy(), np.zeros_like(mj_data.qvel), obs_local.copy())
                if best is None or cand[0] > best[0]:
                    best = cand
                if _seat_pose_ok() and score >= 0.50:
                    mj_data.qpos[:] = stand_qpos
                    mj_data.qvel[:] = stand_qvel
                    mujoco.mj_forward(mj_model, mj_data)
                    return stand_qpos.copy(), stand_qvel.copy(), cand[1].copy(), cand[2].copy(), float(score), cand[3]

        mj_data.qpos[:] = stand_qpos
        mj_data.qvel[:] = stand_qvel
        mujoco.mj_forward(mj_model, mj_data)
        if best is None:
            raise RuntimeError("Scenario 26 could not construct any seated anchor candidate")
        return stand_qpos.copy(), stand_qvel.copy(), best[1].copy(), best[2].copy(), float(best[0]), best[3]

    # Hidden pre-positioning: build a stable seated anchor BEFORE the viewer opens,
    # so frame 0 is already seated above the chair instead of inside it.
    if pelvis_id >= 0 and chair_center_xy is not None:
        print("        Pre-positioning seated fainting start pose ...")
        stand_anchor_qpos, stand_anchor_qvel, sit_anchor_qpos, sit_anchor_qvel, seated_quality, obs = _build_fast_s25_chair_anchor()
        obs, _ = env.reset(options={"qpos": sit_anchor_qpos.copy(), "qvel": sit_anchor_qvel.copy()})

        # Hidden living-seat stabilization: pelvis stays on the chair, but the
        # whole pose is not frozen. This makes frame 0 look like an awake seated
        # subject before tone loss starts.
        z_live = z_stand.clone()
        for _live in range(105):
            mj_data.xfrc_applied[:] = 0.0
            _root_seat_servo(sit_anchor_qpos, target_z=seat_target_pz, blend_xy=0.050 + 0.020 * high_mass_guard, blend_z=0.046 + 0.016 * high_mass_guard, damp_xy=0.24, damp_z=0.52)
            _chair_contact_proxy(mode="preseat", gain=0.72)
            _living_trunk_servo(target_deg=seated_trunk_target, target_head_fwd=seated_head_target, gain=1.25 + 0.45 * high_mass_guard, max_torque=230.0 + 80.0 * high_mass_guard)
            for idx in grp["leg"]:
                mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(0.92 * sf, 0.72)
            for idx in grp["torso"]:
                mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(1.05 * sf, 0.84)
            for idx in grp["arm"]:
                mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(0.90 * sf, 0.65)
            obs_t = torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a_live = motivo.act(obs_t, z_live).squeeze(0).numpy()
            for idx in grp["leg"]:
                a_live[idx] *= 0.12
            for idx in grp["torso"]:
                a_live[idx] *= 0.20
            for idx in grp["arm"]:
                a_live[idx] *= 0.16
            obs, _, terminated, truncated, _ = env.step(a_live)
            if terminated or truncated:
                obs, _ = env.reset(options={"qpos": sit_anchor_qpos.copy(), "qvel": sit_anchor_qvel.copy()})
        # Roll back and re-settle if hidden policy drift produced an awkward
        # young/heavy seated posture.  This protects the good older-subject run
        # while fixing the 32-year/high-weight case that leaned into the chair.
        pz_live, trunk_live, head_live, prone_live = _seat_pose_snapshot()
        live_bad = (
            trunk_live > seated_trunk_target + seated_trunk_margin
            or head_live > seated_head_target + 0.16
            or abs(prone_live) > seated_prone_limit
            or pz_live < chair_seat_z + 0.070
        )
        if live_bad:
            obs, _ = env.reset(options={"qpos": sit_anchor_qpos.copy(), "qvel": np.zeros_like(mj_data.qvel)})
            for _settle in range(60):
                mj_data.xfrc_applied[:] = 0.0
                _pin_to_anchor(sit_anchor_qpos, np.zeros_like(mj_data.qvel), blend=0.18 + 0.10 * high_mass_guard)
                _root_seat_servo(sit_anchor_qpos, target_z=seat_target_pz, blend_xy=0.040 + 0.018 * high_mass_guard, blend_z=0.036 + 0.014 * high_mass_guard, damp_xy=0.22, damp_z=0.50)
                _chair_contact_proxy(mode="preseat", gain=0.62)
                _living_trunk_servo(target_deg=seated_trunk_target, target_head_fwd=seated_head_target, gain=1.55 + 0.35 * high_mass_guard, max_torque=290.0)
                obs, _, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape))
                if terminated or truncated:
                    obs, _ = env.reset(options={"qpos": sit_anchor_qpos.copy(), "qvel": np.zeros_like(mj_data.qvel)})
                    break

        sit_anchor_qpos = mj_data.qpos.copy()
        sit_anchor_qvel = np.zeros_like(mj_data.qvel)
        _lock_heading(verbose=True)
        _reanchor_chair_to_current_pelvis(forward_offset=0.075 + 0.020 * high_mass_guard)
        pz0, td0, hf0, pr0 = _seat_pose_snapshot()
        seat_ok = _seat_pose_ok()
        hard_ok = _seat_pose_hard_safe()
        if (not hard_ok) or seated_quality < 0.35:
            raise RuntimeError(
                f"Scenario 26 invalid seated start: pelvis_z={pz0:.3f}, chair_board_z={chair_seat_z:.3f}, trunk={td0:.1f}, head_fwd={hf0:+.3f}, prone={pr0:+.3f}, q={seated_quality:.2f}"
            )
        print(
            "        Seated start check: pelvis_z={:.3f} m | target_z={:.3f} m | chair_board_z={:.3f} m | trunk={:.1f} deg (target {:.1f}) | head_fwd={:+.3f} m (target {:+.3f}) | prone={:+.3f} | q={:.2f} | ok={}".format(
                pz0, seat_target_pz, chair_seat_z, td0, seated_trunk_target, hf0, seated_head_target, pr0, seated_quality, "YES" if seat_ok else "SOFT"
            )
        )
        partial_rise_qpos = sit_anchor_qpos.copy()
        if partial_rise_qpos.shape[0] >= 3:
            # Visible partial stand: only 6-11 cm above sitting, but held long
            # enough that the viewer sees a real try-to-rise before fainting.
            partial_rise_qpos[2] = min(seat_target_pz + partial_rise_lift, seat_target_pz + 0.115)
        z_cur = z_stand.clone()
        _blend_gen = iter([z_cur])
        _blend_done = True
        mj_data.time = 0.0

    # -- MAIN LOOP -------------------------------------------------------------
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.distance  = 3.6
        viewer.cam.elevation = -8
        viewer.cam.azimuth   = 90

        for step in range(TOTAL):
            sim_t = float(mj_data.time)
            mj_data.xfrc_applied[:] = 0.0     # clear external forces each step

            # --- PHASE: SIT HOLD / FAINT PRODROME ------------------------------
            if step < SIT_HOLD_S:
                phase = "sit_hold"
                if step == 0:
                    print(f"\n  [Step {step}] SIT_HOLD phase - visible simulation starts already seated on chair")
                    try:
                        view_yaw = float(np.degrees(np.arctan2(fall_fwd_xy[1], fall_fwd_xy[0])))
                        viewer.cam.azimuth = view_yaw + 180.0
                        viewer.cam.elevation = -6.0
                        viewer.cam.distance = 3.3
                        print(f"        Camera locked lateral-view: azimuth={viewer.cam.azimuth:+.1f} deg")
                    except Exception:
                        pass
                    _start_blend(z_stand, steps=20)
                prog_sit = step / max(SIT_HOLD_S - 1, 1)
                prodrome = _smoothstep((prog_sit - 0.72) / 0.28)
                micro = math.sin(2.0 * math.pi * 1.2 * prog_sit)
                if sit_anchor_qpos is not None:
                    _root_seat_servo(sit_anchor_qpos, target_z=seat_target_pz, blend_xy=0.038 + 0.014 * high_mass_guard, blend_z=0.046 + 0.014 * high_mass_guard, damp_xy=0.24, damp_z=0.56)
                    _visible_chair_seat_guard(target_z=seat_target_pz - 0.012, min_clearance=0.088, strength=0.92, xy_strength=0.18)
                    if step < 76:
                        _pin_to_anchor(sit_anchor_qpos, sit_anchor_qvel, blend=visible_anchor_blend * (1.25 - 0.55 * prodrome))
                _chair_contact_proxy(mode="sit", gain=0.78)
                _living_trunk_servo(target_deg=seated_trunk_target + 1.6 * prodrome, target_head_fwd=seated_head_target + 0.018 * prodrome, gain=1.15 + 0.55 * high_mass_guard, max_torque=235.0 + 70.0 * high_mass_guard)
                if pelvis_id >= 0:
                    if mj_data.qvel.shape[0] >= 2:
                        mj_data.qvel[0] *= 0.12
                        mj_data.qvel[1] *= 0.12
                    if chair_center_xy is not None:
                        err_xy = chair_center_xy - mj_data.xpos[pelvis_id][:2].copy()
                        mj_data.xfrc_applied[pelvis_id, 0] += np.clip(err_xy[0], -0.05, 0.05) * 0.28 * BW
                        mj_data.xfrc_applied[pelvis_id, 1] += np.clip(err_xy[1], -0.05, 0.05) * 0.20 * BW
                if torso_id >= 0:
                    # subtle early faint symptoms: small forward nod, not a collapse yet
                    _apply_body_forward_wrench(torso_id,
                                               fwd_n=(0.0015 + 0.0045*prodrome) * BW,
                                               down_n=0.0012 * BW * prodrome,
                                               pitch_nm=0.12 * prodrome + 0.08*micro*prodrome)
                if head_id >= 0:
                    _apply_body_forward_wrench(head_id,
                                               fwd_n=(0.0012 + 0.0035*prodrome) * BW,
                                               down_n=0.0008 * BW * prodrome,
                                               pitch_nm=0.10 * prodrome + 0.05*micro*prodrome)
                for idx in grp["leg"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(0.78 * sf, 0.58)
                for idx in grp["torso"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(0.98 * sf, 0.78)
                arm_damp = 0.34

            # --- PHASE: SMALL TRY-TO-STAND / STRUGGLE ------------------------
            elif step < SIT_HOLD_S + TRY_RISE_S:
                phase  = "try_rise"
                local  = step - SIT_HOLD_S
                if local == 0:
                    print(f"\n  [Step {step}] TRY_RISE phase - visible weak partial stand-up from chair")
                    _start_blend(z_stand, steps=35)
                ramp = local / max(TRY_RISE_S - 1, 1)
                # v8 visible partial-rise profile: lift quickly, hold briefly,
                # then keep most of the lift instead of dropping back into the
                # chair before FAINT. This makes the "standing up little" phase
                # readable in the viewer for young, middle and older subjects.
                lift_up = _smoothstep(ramp / 0.34)
                late_release = _smoothstep((ramp - 0.72) / 0.28)
                rise_hold = lift_up * (1.0 - (1.0 - partial_rise_hold) * late_release)
                # Stand-like torso briefly straightens during the rise; it does
                # not lean forward until the faint prodrome starts.
                standish_trunk = seated_trunk_target - (3.5 + 2.0 * young_frac) * lift_up + 2.2 * late_release
                standish_head = seated_head_target - 0.010 * lift_up + 0.014 * late_release
                if sit_anchor_qpos is not None and partial_rise_qpos is not None:
                    rise_z = float(sit_anchor_qpos[2] + partial_rise_lift * rise_hold)
                    _root_seat_servo(
                        sit_anchor_qpos,
                        target_z=rise_z,
                        blend_xy=0.018 + 0.010 * (1.0 - lift_up),
                        blend_z=0.058 + 0.020 * lift_up,
                        damp_xy=0.26,
                        damp_z=0.38,
                    )
                    _visible_chair_seat_guard(
                        target_z=rise_z - 0.006,
                        min_clearance=0.090,
                        strength=0.70 * (1.0 - 0.18 * late_release),
                        xy_strength=0.10,
                    )
                # Unweight the chair during the clear rise/hold; still keep a
                # small support proxy so feet/seat do not explode numerically.
                _chair_contact_proxy(mode="sit", gain=0.42 - 0.16 * lift_up + 0.05 * late_release)
                _living_trunk_servo(
                    target_deg=standish_trunk,
                    target_head_fwd=standish_head,
                    gain=1.08 + 0.30 * lift_up + 0.25 * high_mass_guard,
                    max_torque=235.0 + 55.0 * high_mass_guard,
                )
                if pelvis_id >= 0:
                    lift = (0.032 + (0.058 - 0.010 * high_mass_guard) * rise_hold) * BW
                    mj_data.xfrc_applied[pelvis_id, 2] += lift
                    if torso_id >= 0:
                        mj_data.xfrc_applied[torso_id, 2] += 0.010 * BW * rise_hold
                    if chair_center_xy is not None:
                        # Keep the hips just at the front half of the chair while
                        # rising; do not slide toward the seat edge yet. The true
                        # side transfer happens during FAINT/COLLAPSE.
                        front_shift = 0.076 + 0.030 * rise_hold
                        desired_xy = chair_center_xy + front_shift * fall_fwd_xy
                        err_xy = desired_xy - mj_data.xpos[pelvis_id][:2].copy()
                        mj_data.xfrc_applied[pelvis_id, 0] += np.clip(err_xy[0], -0.050, 0.050) * 0.22 * BW
                        mj_data.xfrc_applied[pelvis_id, 1] += np.clip(err_xy[1], -0.050, 0.050) * 0.18 * BW
                    if mj_data.qvel.shape[0] >= 3:
                        mj_data.qvel[0] *= 0.10
                        mj_data.qvel[1] *= 0.10
                        # Do not kill vertical motion completely; otherwise the
                        # viewer sees the old seated-fall instead of a lift.
                        mj_data.qvel[2] *= 0.42
                if torso_id >= 0:
                    # Rise effort is mostly upward/straightening. Only a very
                    # small forward struggle is kept near the end.
                    _apply_body_forward_wrench(
                        torso_id,
                        fwd_n=(0.001 + 0.006 * late_release) * BW,
                        down_n=-0.004 * BW * lift_up + 0.002 * BW * late_release,
                        pitch_nm=-0.10 * lift_up + 0.25 * late_release,
                    )
                leg_tone = max(0.78, 0.99 - 0.08 * late_release)
                torso_tone = max(0.78, 1.02 - 0.10 * late_release)
                for idx in grp["leg"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * leg_tone
                for idx in grp["torso"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * torso_tone
                arm_damp = 0.30

            # --- PHASE: FAINT WHILE SEATED / PARTIAL RISE --------------------
            elif step < collapse_start_step:
                phase = "faint"
                local_faint = step - (SIT_HOLD_S + TRY_RISE_S)
                if local_faint == 0:
                    print(f"\n  [Step {step}] FAINT phase - tone loss from partial rise, sideward head/trunk drift")
                    _start_blend(z_sit, steps=45)
                prog_faint = local_faint / max(FAINT_S - 1, 1)
                faint_ease = 0.5 * (1.0 - math.cos(math.pi * prog_faint))
                sway = math.sin(2.0 * math.pi * (0.9 + 0.35 * age_frac) * prog_faint)
                if pelvis_id >= 0:
                    # Keep the hips visibly seated throughout the faint prodrome.
                    # Only the head/trunk should drift forward here; the hips are
                    # released at the COLLAPSE boundary. This prevents the common
                    # bug where the pelvis sinks below the viewer-only chair just
                    # before the fall begins.
                    support_hold = 1.0 - 0.12 * _smoothstep((faint_ease - 0.78) / 0.22)
                    # v8: start FAINT from the raised partial-stand height. The
                    # knees buckle gradually, but the pelvis must not instantly
                    # fall back into the chair at the phase boundary.
                    faint_hold = 1.0 - 0.45 * _smoothstep((faint_ease - 0.30) / 0.70)
                    faint_seat_z = seat_target_pz + partial_rise_lift * faint_hold - 0.004 * _smoothstep((faint_ease - 0.84) / 0.16)
                    if sit_anchor_qpos is not None:
                        _root_seat_servo(
                            sit_anchor_qpos,
                            target_z=faint_seat_z,
                            blend_xy=0.014 * support_hold,
                            blend_z=0.028 * support_hold,
                            damp_xy=0.58,
                            damp_z=0.68,
                        )
                    _chair_contact_proxy(mode="faint", gain=0.45 * support_hold)
                    _visible_chair_seat_guard(
                        target_z=faint_seat_z,
                        min_clearance=0.088,
                        strength=0.84 * support_hold,
                        xy_strength=0.11,
                    )
                    # Keep faint as a *prodrome*, not a completed collapse.
                    # v5 dropped the support too early, so trunk reached 120+ deg
                    # before the collapse phase. Keep a weak living correction all
                    # the way to the boundary.
                    live_gain = 0.22 + 0.78 * (1.0 - _smoothstep((faint_ease - 0.35) / 0.65))
                    _living_trunk_servo(
                        target_deg=seated_trunk_target + 1.5 + 18.0 * faint_ease,
                        target_head_fwd=seated_head_target + 0.018 + 0.090 * faint_ease,
                        gain=0.72 * live_gain,
                        max_torque=185.0,
                    )
                    if mj_data.qvel.shape[0] >= 3:
                        mj_data.qvel[0] *= 0.54 + 0.30 * faint_ease
                        mj_data.qvel[1] *= 0.54 + 0.30 * faint_ease
                        mj_data.qvel[2] *= 0.52 + 0.22 * faint_ease
                    # Do not apply downward pelvis unloading during faint. The
                    # collapse release below creates the actual drop.
                    if chair_center_xy is not None and faint_ease < 0.94:
                        # Shift hips toward/front of the chair before tone loss,
                        # so collapse happens outside the seat footprint rather
                        # than underneath the board.
                        # Transfer forward only after the partial rise is
                        # visible. Early FAINT should look like standing-up then
                        # losing tone, not like sliding off the chair.
                        side_transfer = _smoothstep((faint_ease - 0.22) / 0.60)
                        side2 = _fall_side_xy()
                        desired_xy = chair_center_xy + 0.088 * fall_fwd_xy + (0.020 + 0.052 * side_transfer) * side2
                        err_xy = desired_xy - mj_data.xpos[pelvis_id][:2].copy()
                        guide = 1.0 - 0.40 * _smoothstep((faint_ease - 0.76) / 0.24)
                        mj_data.xfrc_applied[pelvis_id, 0] += np.clip(err_xy[0], -0.050, 0.050) * 0.090 * BW * guide
                        mj_data.xfrc_applied[pelvis_id, 1] += np.clip(err_xy[1], -0.050, 0.050) * 0.078 * BW * guide
                if torso_id >= 0:
                    # Slow fainting: the upper body drifts sideward while the pelvis
                    # remains chair-supported until the lateral collapse transition.
                    _apply_body_lateral_wrench(torso_id,
                                               lateral_n=(0.004 + 0.018 * faint_ease) * BW,
                                               down_n=(0.001 + 0.0015 * faint_ease) * BW,
                                               roll_nm=(0.15 + 1.10 * faint_ease + 0.12 * sway))
                if head_id >= 0:
                    _apply_body_lateral_wrench(head_id,
                                               lateral_n=(0.006 + 0.014 * faint_ease) * BW,
                                               down_n=(0.001 + 0.0015 * faint_ease) * BW,
                                               roll_nm=(0.25 + 0.95 * faint_ease))
                # Keep enough tone during FAINT so the body does not collapse
                # through the chair before the release phase.
                faint_trunk_limit = float(np.clip(46.0 + 7.0 * age_frac + 4.0 * bal_imp, 46.0, 58.0))
                _anti_overlean_guard(max_deg=faint_trunk_limit, gain=1.05)
                leg_tone = max(0.38, 0.84 - 0.38 * faint_ease)
                torso_tone = max(0.34, 0.86 - 0.44 * faint_ease)
                for idx in grp["leg"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * leg_tone
                for idx in grp["torso"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * torso_tone
                arm_damp = 0.50
                if local_faint == FAINT_S - 1 and not trigger_fired:
                    trigger_fired = True
                    perturb_t = sim_t
                    print(f"  [Step {step}] Faint collapse trigger - seated tone lost | collapse starts at step {collapse_start_step}")

            # --- PHASE: COLLAPSE -------------------------------------------
            elif step < collapse_start_step + COLLAPSE_S:
                phase   = "collapse"
                local_c = step - collapse_start_step

                if local_c == 0:
                    if not trigger_fired:
                        trigger_fired = True
                        perturb_t     = sim_t
                        print(f"  [Step {step}] Fallback collapse trigger")
                    print(f"\n  [Step {step}] COLLAPSE phase - lateral fainting topple after partial rise")
                    _start_blend(z_fall, steps=20)

                prog  = local_c / max(COLLAPSE_S - 1, 1)
                ramp_f = 0.5 * (1 - math.cos(math.pi * prog))  # 0..1

                # Chair-side clearance gate. The pelvis is allowed to drop only
                # after it has moved toward the selected side edge of the visual
                # chair. This keeps task 26 lateral, not a forward slide.
                rs_clear = 1.0
                clear_gate = 1.0
                side2 = _fall_side_xy()
                side_clear_min = float(np.clip(0.50 * chair_half_lat + 0.035, 0.115, 0.165))
                edge_stop = float(np.clip(side_clear_min + 0.030, 0.145, 0.205))
                trunk_gate = 0.0
                if pelvis_id >= 0 and chair_center_xy is not None:
                    rel_xy0 = mj_data.xpos[pelvis_id][:2].copy() - chair_center_xy
                    rs_clear = float(np.dot(rel_xy0, side2))
                    rf_now = float(np.dot(rel_xy0, fall_fwd_xy))
                    clear_gate = _smoothstep((rs_clear - (side_clear_min - 0.018)) / 0.075)
                    if local_c < 34:
                        _seat_side_edge_friction(
                            edge_target=edge_stop,
                            max_side_speed=0.070 + 0.030 * ramp_f,
                            strength=0.82 * (1.0 - _smoothstep(local_c / 34.0)),
                            active_height_margin=0.078,
                        )
                    if local_c < 26 and clear_gate < 0.98:
                        hold = (1.0 - clear_gate) * (1.0 - _smoothstep(local_c / 26.0))
                        _chair_side_exit_guard(
                            target_z=chair_seat_z + 0.104,
                            side_min=side_clear_min,
                            forward_target=0.088,
                            min_clearance=0.100,
                            strength=0.52 * hold,
                            xy_strength=0.72,
                        )
                        mj_data.xfrc_applied[pelvis_id, 0] += 0.045 * BW * hold * side2[0]
                        mj_data.xfrc_applied[pelvis_id, 1] += 0.045 * BW * hold * side2[1]
                        if mj_data.qvel.shape[0] >= 3:
                            mj_data.qvel[2] *= 0.76 + 0.18 * clear_gate
                    # AP centering: suppress forward/back chair slipping during lateral roll.
                    ap_err = rf_now - 0.092
                    if abs(ap_err) > 0.018:
                        ap_corr = float(np.clip(-ap_err / 0.11, -1.0, 1.0)) * 0.025 * BW * (1.0 - 0.25 * ramp_f)
                        mj_data.xfrc_applied[pelvis_id, 0] += ap_corr * fall_fwd_xy[0]
                        mj_data.xfrc_applied[pelvis_id, 1] += ap_corr * fall_fwd_xy[1]
                trunk_gate = _smoothstep((_trunk_tilt_deg() - 62.0) / 20.0)
                if mj_data.qvel.shape[0] >= 2:
                    damp_xy = 0.74 + 0.14 * trunk_gate
                    mj_data.qvel[0] *= damp_xy
                    mj_data.qvel[1] *= damp_xy
                if local_c < 4:
                    _chair_contact_proxy(mode="release", gain=0.05)

                # Fainting must be visible: head rolls sideward first, then trunk
                # follows. Keep this gradual because FAINT already introduced
                # the prodrome lean.
                if head_id >= 0:
                    _apply_body_lateral_wrench(head_id,
                                               lateral_n=(0.030 + 0.065*ramp_f) * BW,
                                               down_n=(0.014 + 0.030*ramp_f) * BW,
                                               roll_nm=(2.8 + 7.5*ramp_f))

                # Rotational couple: lateral/down upper body plus delayed pelvis
                # unloading. Downward pelvis force is gated by side clearance.
                early = 0.16 + 0.84 * ramp_f
                torso_lat = 0.108 * BW * early * collapse_force_scale
                torso_down = 0.040 * BW * early
                torso_roll = 12.5 * early * collapse_pitch_scale
                side2 = _fall_side_xy()
                if torso_id >= 0:
                    _apply_body_lateral_wrench(torso_id,
                                               lateral_n=torso_lat,
                                               down_n=torso_down,
                                               roll_nm=torso_roll)
                if head_id >= 0:
                    _apply_body_lateral_wrench(head_id,
                                               lateral_n=0.082 * BW * early * collapse_force_scale,
                                               down_n=0.038 * BW * early,
                                               roll_nm=9.5 * early * collapse_pitch_scale)
                if pelvis_id >= 0:
                    down_gate = _smoothstep(local_c / 24.0) * clear_gate * trunk_gate
                    brake_gate = 0.35 + 0.65 * clear_gate
                    mj_data.xfrc_applied[pelvis_id, 2] -= 0.026 * BW * early * down_gate
                    mj_data.xfrc_applied[pelvis_id, 0] += 0.014 * BW * early * brake_gate * side2[0]
                    mj_data.xfrc_applied[pelvis_id, 1] += 0.014 * BW * early * brake_gate * side2[1]
                    if chair_center_xy is not None and local_c < 24:
                        need = float(np.clip((side_clear_min - rs_clear) / 0.16, 0.0, 1.0))
                        if need > 0.0:
                            mj_data.xfrc_applied[pelvis_id, 0] += 0.040 * BW * need * side2[0]
                            mj_data.xfrc_applied[pelvis_id, 1] += 0.040 * BW * need * side2[1]
                if hand_ids:
                    reach_gain = 0.55 + 0.45 * ramp_f
                    for bid in hand_ids:
                        _apply_body_lateral_wrench(bid,
                                                   lateral_n=0.030 * BW * reach_gain,
                                                   down_n=-0.006 * BW * reach_gain,
                                                   roll_nm=0.0)
                if leg_trail_ids and (_head_lateral_rel() > 0.12 or prog > 0.25):
                    trail_gain = 0.35 + 0.65 * ramp_f
                    for bid in leg_trail_ids:
                        _apply_body_lateral_wrench(bid,
                                                   lateral_n=-0.020 * BW * trail_gain,
                                                   down_n=0.008 * BW * trail_gain,
                                                   roll_nm=0.0)

                # If the head is still not sideward enough during collapse, add
                # extra upper-body lateral roll immediately instead of waiting
                # until the later fall phase.
                head_l_c = _head_lateral_rel()
                side_c = _torso_side_lie_score()
                if torso_id >= 0 and (head_l_c < 0.08 or side_c < side_lie_target * 0.35):
                    corr_h = np.clip(0.08 - head_l_c, 0.0, 0.20) / 0.20
                    corr_s = np.clip(side_lie_target * 0.35 - side_c, 0.0, 0.40) / 0.40
                    corr = max(float(corr_h), float(corr_s))
                    _apply_body_lateral_wrench(torso_id,
                                               lateral_n=0.095 * BW * corr,
                                               down_n=0.035 * BW * corr,
                                               roll_nm=16.0 * corr)
                    if pelvis_id >= 0:
                        side2 = _fall_side_xy()
                        mj_data.xfrc_applied[pelvis_id, 0] -= 0.028 * BW * corr * side2[0]
                        mj_data.xfrc_applied[pelvis_id, 1] -= 0.028 * BW * corr * side2[1]

                # -- All muscles weaken toward min_gear -----------------------
                gear_f = sit_end_strength - prog * (sit_end_strength - leg_floor)
                torso_f = 0.46 - prog * (0.46 - torso_floor)
                for idx in grp["leg"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(gear_f, leg_floor)
                for idx in grp["torso"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(torso_f, torso_floor)
                for idx in grp["arm"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * arm_floor
                arm_damp = 0.80

            # --- PHASE: FALL -----------------------------------------------
            else:
                phase = "fall"
                local_f = step - (collapse_start_step + COLLAPSE_S)

                if local_f == 0:
                    print(f"\n  [Step {step}] FALL phase - free fall")

                # Keep a short decaying side-lying residual so the avatar
                # completes lateral rotation instead of rebounding upright.
                if local_f < 34 and torso_id >= 0:
                    decay = 0.90 ** local_f
                    resid_lat = 0.050 * BW * decay * fall_resid_scale
                    resid_roll = 16.0 * decay * fall_resid_scale
                    _apply_body_lateral_wrench(torso_id,
                                               lateral_n=resid_lat,
                                               roll_nm=resid_roll)
                    if pelvis_id >= 0:
                        side2 = _fall_side_xy()
                        mj_data.xfrc_applied[pelvis_id, 0] += 0.022 * BW * decay * fall_resid_scale * side2[0]
                        mj_data.xfrc_applied[pelvis_id, 1] += 0.022 * BW * decay * fall_resid_scale * side2[1]

                # Keep some residual leg and arm control so the body does not fold unnaturally.
                mj_model.actuator_gear[:, 0] = orig_gear * min_gear
                for idx in grp["leg"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * leg_floor
                for idx in grp["torso"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * torso_floor
                for idx in grp["arm"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * arm_floor

                pz_f   = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 1.0
                v6_p   = np.zeros(6)
                if pelvis_id >= 0:
                    mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, pelvis_id, v6_p, 0)
                spd    = float(np.linalg.norm(v6_p[3:5]))
                # Once contact with the ground begins, damp root translation so
                # the body settles instead of skating forward after impact.
                if pelvis_id >= 0 and pz_f < 0.22 and mj_data.qvel.shape[0] >= 2:
                    slide_damp = 0.82 if local_f < 70 else 0.70
                    mj_data.qvel[0] *= slide_damp
                    mj_data.qvel[1] *= slide_damp

                trunk_f = 0.0
                if pelvis_id >= 0 and head_id >= 0:
                    vec_f = mj_data.xpos[head_id] - mj_data.xpos[pelvis_id]
                    nv_f = float(np.linalg.norm(vec_f))
                    if nv_f > 1e-8:
                        trunk_f = float(np.degrees(np.arccos(np.clip(np.dot(vec_f / nv_f, [0.0, 0.0, 1.0]), -1.0, 1.0))))
                head_lat = _head_lateral_rel()

                side_now = _torso_side_lie_score()
                # Keep driving rotation until the body is side-lying, not just horizontal.
                if not rest_mode and torso_id >= 0 and local_f < 100:
                    need_side = (head_lat < 0.16) or (trunk_f < settle_target_deg - 2.0) or (side_now < side_lie_target)
                    if need_side:
                        gain = max(0.0, 1.0 - local_f / 100.0)
                        soft = float(np.clip(1.00 - 0.35 * age_frac - 0.20 * bal_imp, 0.55, 1.00))
                        extra_lat = 0.10 * BW * gain * soft
                        extra_roll = 22.0 * gain * soft
                        _apply_body_lateral_wrench(torso_id, lateral_n=extra_lat, down_n=0.03*BW*gain*soft, roll_nm=extra_roll)
                        if pelvis_id >= 0:
                            side2 = _fall_side_xy()
                            mj_data.xfrc_applied[pelvis_id, 0] -= 0.035 * BW * gain * soft * side2[0]
                            mj_data.xfrc_applied[pelvis_id, 1] -= 0.035 * BW * gain * soft * side2[1]
                if local_f < 24:
                    if hand_ids:
                        rg = max(0.0, 1.0 - local_f / 24.0)
                        for bid in hand_ids:
                            _apply_body_lateral_wrench(bid,
                                                       lateral_n=0.022 * BW * rg,
                                                       down_n=-0.004 * BW * rg,
                                                       roll_nm=0.0)
                    if leg_trail_ids and trunk_f > 45.0:
                        tg = max(0.0, 1.0 - local_f / 24.0)
                        for bid in leg_trail_ids:
                            _apply_body_lateral_wrench(bid,
                                                       lateral_n=-0.014 * BW * tg,
                                                       down_n=0.006 * BW * tg,
                                                       roll_nm=0.0)

                # Over-rotation control for older / weaker profiles
                if trunk_f > settle_target_deg + 6.0 and mj_data.qvel.shape[0] >= 6:
                    mj_data.qvel[3] *= 0.65
                    mj_data.qvel[4] *= 0.65
                    mj_data.qvel[5] *= 0.82

                min_rest_frames = int(round(28 + 10 * age_frac + 5 * bal_imp))
                settled_ok = (
                    local_f >= min_rest_frames and pz_f < 0.23 and spd < 0.16 and mj_data.ncon >= 6
                    and head_lat > 0.16 and side_now > side_lie_target
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
            if phase == "sit_hold":
                for idx in grp["leg"]:
                    action[idx] *= 0.04
                for idx in grp["torso"]:
                    action[idx] *= 0.08
                for idx in grp["arm"]:
                    action[idx] *= 0.08
            elif phase == "try_rise":
                # v8: stronger stand embedding during the lift/hold so knees/hips
                # visibly unweight from the chair, but still below full standing.
                for idx in grp["leg"]:
                    action[idx] *= 0.48
                for idx in grp["torso"]:
                    action[idx] *= 0.42
                for idx in grp["arm"]:
                    action[idx] *= 0.30
            elif phase == "faint":
                # Fainting prodrome: fade from alive control to weak collapse.
                local_f_for_action = max(0, step - (SIT_HOLD_S + TRY_RISE_S))
                faint_a = 0.5 * (1.0 - math.cos(math.pi * (local_f_for_action / max(FAINT_S - 1, 1))))
                for idx in grp["leg"]:
                    action[idx] *= (0.28 - 0.14 * faint_a)
                for idx in grp["torso"]:
                    action[idx] *= (0.30 - 0.18 * faint_a)
                for idx in grp["arm"]:
                    action[idx] *= (0.36 - 0.10 * faint_a)
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
                recent = list(action_hist)
                if phase == "sit_hold":
                    wts = np.linspace(0.25, 1.0, len(recent))
                    action = np.average(recent, weights=wts, axis=0)
                else:
                    wts = np.array([0.45, 0.55])
                    action = np.average(recent[-2:], weights=wts, axis=0)

            # -- step env -----------------------------------------------------
            obs, _, terminated, truncated, _ = env.step(action)
            if phase == "sit_hold":
                _visible_chair_seat_guard(target_z=seat_target_pz - 0.014, min_clearance=0.088, strength=0.95, xy_strength=0.18)
            elif phase == "try_rise":
                try_prog = (step - SIT_HOLD_S) / max(TRY_RISE_S - 1, 1)
                try_lift_up = _smoothstep(try_prog / 0.34)
                try_late_release = _smoothstep((try_prog - 0.72) / 0.28)
                try_hold = try_lift_up * (1.0 - (1.0 - partial_rise_hold) * try_late_release)
                try_z = seat_target_pz + partial_rise_lift * try_hold - 0.006
                _visible_chair_seat_guard(target_z=try_z, min_clearance=0.090, strength=0.66 * (1.0 - 0.18 * try_late_release), xy_strength=0.10)
            elif phase == "faint":
                faint_prog_post = (step - (SIT_HOLD_S + TRY_RISE_S)) / max(FAINT_S - 1, 1)
                faint_ease_post = 0.5 * (1.0 - math.cos(math.pi * faint_prog_post))
                faint_hold_post = 1.0 - 0.45 * _smoothstep((faint_ease_post - 0.30) / 0.70)
                faint_seat_z_post = seat_target_pz + partial_rise_lift * faint_hold_post - 0.004 * _smoothstep((faint_ease_post - 0.84) / 0.16)
                _visible_chair_seat_guard(
                    target_z=faint_seat_z_post,
                    min_clearance=0.088,
                    strength=0.88,
                    xy_strength=0.10,
                )
            elif phase == "collapse":
                # Post-step safety: for the first few collapse frames only,
                # prevent the pelvis from integrating below the viewer-only
                # chair before it has cleared the front edge. This is a short,
                # weak guard; after clearance the body is fully released.
                local_c_post = step - collapse_start_step
                if local_c_post < 24 and pelvis_id >= 0 and chair_center_xy is not None:
                    rel_post = mj_data.xpos[pelvis_id][:2].copy() - chair_center_xy
                    side2_post = _fall_side_xy()
                    rs_post = float(np.dot(rel_post, side2_post))
                    side_min_post = float(np.clip(0.58 * chair_half_lat + 0.045, 0.135, 0.185))
                    clear_post = _smoothstep((rs_post - (side_min_post - 0.020)) / 0.085)
                    if clear_post < 0.96:
                        hold_post = (1.0 - clear_post) * (1.0 - _smoothstep(local_c_post / 24.0))
                        _chair_side_exit_guard(
                            target_z=chair_seat_z + 0.104,
                            side_min=side_min_post,
                            forward_target=0.092,
                            min_clearance=0.100,
                            strength=0.34 * hold_post,
                            xy_strength=0.80,
                        )

            # -- IMU log -------------------------------------------------------
            imu_pk = imu.log(sim_t)
            dynamics_frame = _capture_shared_exporters(sim_t, phase)
            _draw_virtual_chair(viewer)

            # -- console log: standard PhysicsDashboard table -----------------
            log_now = (step % 30 == 0 or phase == "collapse"
                       or (phase == "fall" and step % 10 == 0))
            if log_now:
                pz_ = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 0.0
                leg_strength_live = (mj_model.actuator_gear[grp["leg"][0], 0] /
                                     max(orig_gear[grp["leg"][0]], 1e-9)) if grp["leg"] else 1.0
                fall_pred = bool(pz_ < 0.35 or phase in ("collapse", "fall"))
                if dashboard is not None:
                    dash_m = dashboard.report(
                        step=step,
                        phase=phase,
                        leg_strength=float(np.clip(leg_strength_live, 0.0, 1.2)),
                        xcom_margin=None,
                        fall_predicted=fall_pred,
                        imu_peak=float(imu_pk),
                        sensor_impact=float(imu_pk),
                        control_source='guardian',
                        mjpc_cost=np.nan,
                        dynamics=dynamics_frame,
                    )
                    metrics_log.append({**dash_m, 'step': step, 'phase': phase})
                else:
                    print(f"  {step:>5d} {phase:>8s} h={pz_:>6.3f} strength={leg_strength_live:>5.0%} ncon={mj_data.ncon:>3d} {'FALL!' if fall_pred else 'ok'}")

            pz_m = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 0.0
            vv_m = np.zeros(6)
            if pelvis_id >= 0:
                mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, pelvis_id, vv_m, 0)
            vx_m = float(np.dot(vv_m[3:5], _fall_side_xy())) if heading_locked else float(np.linalg.norm(vv_m[3:5]))
            td_m = 0.0
            if pelvis_id >= 0 and head_id >= 0:
                vec_m = mj_data.xpos[head_id] - mj_data.xpos[pelvis_id]
                n_m = float(np.linalg.norm(vec_m))
                if n_m > 1e-8:
                    td_m = float(np.degrees(np.arccos(np.clip(np.dot(vec_m / n_m, [0.0, 0.0, 1.0]), -1.0, 1.0))))
            metrics.append({"step": step, "phase": phase, "pelvis_h": pz_m, "trunk_deg": td_m, "lat_v": vx_m, "head_lat": (_head_lateral_rel() if heading_locked else 0.0), "side_lie": (_torso_side_lie_score() if heading_locked else 0.0), "head_fwd": (_head_forward_rel() if heading_locked else 0.0)})

            viewer.sync()

            # If the env reports terminal during the protected seated hold, do
            # not reset to default standing; restore the validated chair anchor.
            if terminated and phase == "sit_hold":
                if sit_anchor_qpos is not None:
                    obs, _ = env.reset(options={"qpos": sit_anchor_qpos.copy(), "qvel": sit_anchor_qvel.copy()})
                    _lock_heading()
                    _reanchor_chair_to_current_pelvis(forward_offset=0.075 + 0.020 * high_mass_guard)
                else:
                    obs, _ = env.reset()
                mj_model.actuator_gear[:, 0] = orig_gear * sf

    # -- export ----------------------------------------------------------------
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    pfx = f"fall_scenario26_age{age}_{ts}"
    imu_r = imu.export_csv(pfx + ".csv", meta={
        "scenario_id": 26,
        "description": "Lateral fall while sitting, caused by fainting",
        "age": age, "height_m": height, "sex": sex, "weight_kg": weight,
    })
    val = _validate(imu, perturb_t, body_mass)

    # -- validation report -----------------------------------------------------
    rpt = pfx + "_validation.txt"
    with open(rpt, "w") as f:
        f.write("="*60 + "\nSCENARIO 26 VALIDATION REPORT\n" + "="*60 + "\n")
        f.write(f"age={age}  h={height}m  w={weight:.1f}kg  sex={sex}\n\n")
        f.write(f"Score:          {val['score']:.1%}\n")
        f.write(f"Classification: {val['classification']}\n")
        f.write(f"Fall duration:  {val['fall_duration_s']:.2f} s\n")
        f.write(f"Peak accel:     {val['peak_accel_filt']:.1f} m/s² (filtered)\n")
        f.write(f"Peak impact:    {val['peak_impact_n']:.0f} N\n")
        f.write(f"SISFall:        {val['sisfall_compliant']}\n\nChecks:\n")
        for k, v in val["checks"].items():
            f.write(f"  {'PASS' if v else 'FAIL'}  {k}\n")

    print("\n" + "=" * 70)
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

    if dashboard is not None and metrics_log:
        print("\nPHASE-BY-PHASE BIOMECHANICAL SUMMARY")
        print("-" * 50)
        for _ph in ("sit_hold", "try_rise", "faint", "collapse", "fall"):
            dashboard.print_phase_summary(_ph, [m for m in metrics_log if m.get('phase') == _ph])

    print("  SCENARIO 26 POST-SIM PHYSICS AUDIT")
    print("=" * 70)
    for ph_name in ("sit_hold", "try_rise", "faint", "collapse", "fall"):
        pm = [m for m in metrics if m["phase"] == ph_name]
        if not pm:
            continue
        hs = [m["pelvis_h"] for m in pm]
        ts = [m["trunk_deg"] for m in pm]
        vs = [m["lat_v"] for m in pm]
        hls = [m['head_lat'] for m in pm]
        sls = [m['side_lie'] for m in pm]
        hfs = [m.get('head_fwd', 0.0) for m in pm]
        print(f"  {ph_name:>8s}: pelvis {hs[0]:.3f}->{hs[-1]:.3f} m | trunk {ts[0]:.1f}->{ts[-1]:+.1f} deg | head_lat {hls[0]:+.3f}->{hls[-1]:+.3f} m | side {sls[0]:+.3f}->{sls[-1]:+.3f} | AP_head {hfs[0]:+.3f}->{hfs[-1]:+.3f} | lat_v max {max(vs):+.3f} m/s")
    fall_pm = [m for m in metrics if m["phase"] == "fall"]
    if fall_pm:
        final_trunk = fall_pm[-1]["trunk_deg"]
        peak_lat = max(m["lat_v"] for m in fall_pm)
        print(f"  final trunk lean : {final_trunk:.1f} deg")
        print(f"  max fall lateral vel : {peak_lat:+.3f} m/s")

    env.close()
    mj_model.actuator_gear[:, 0] = orig_gear.copy()

    print("\n" + "="*70)
    print("  SCENARIO 26 COMPLETE")
    print("="*70)
    print(f"  Score:         {val['score']:.1%}  ({val['classification']})")
    print(f"  Fall duration: {val['fall_duration_s']:.2f} s")
    print(f"  Peak accel:    {val['peak_accel_filt']:.1f} m/s²")
    print(f"  SISFall:       {val['sisfall_compliant']}")
    print(f"  IMU CSV  ->    {imu_r['filename']}")
    print(f"  Validation ->  {rpt}")
    print("="*70)

    return {
        "scenario_id":    25,
        "classification": val["classification"],
        "score":          val["score"],
        "sisfall":        val["sisfall_compliant"],
        "imu_csv":        imu_r["filename"],
        "validation_txt": rpt,
    }


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run({"age": 70, "height": 1.65, "sex": "female", "weight": None})
