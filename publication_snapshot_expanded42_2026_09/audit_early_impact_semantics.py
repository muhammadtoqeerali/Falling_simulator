from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")
CAMPAIGN = ROOT / "outputs/_highrate_overnight/campaign_highrate_truth_v1"
MANIFEST = CAMPAIGN / "highrate_event_manifest_FINAL.csv"
OUT = CAMPAIGN / "early_impact_semantics_audit.csv"

EARLY = [20,21,22,23,24,25,26,27,250]

def finite(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except Exception:
        return None

def norm(x):
    p = Path(str(x))
    return p if p.is_absolute() else ROOT / p

def stem_from_truth(p):
    ending = "_highrate_truth.csv"
    if not p.name.endswith(ending):
        raise RuntimeError(p)
    return p.name[:-len(ending)]

def find_col(df, names):
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.lower() in lower:
            return lower[n.lower()]
    return None

m = pd.read_csv(MANIFEST)
m = m[m["scenario_id"].isin(EARLY)].copy()

rows = []

for _, r in m.iterrows():
    task = int(r["scenario_id"])
    age = int(r["sim_age_profile"])
    onset = float(r["fall_onset_time_s"])
    recovered = float(r["main_impact_time_s"])

    current = norm(r["current_truth_csv"])
    legacy = norm(r["legacy_csv"])
    stem = stem_from_truth(current)

    dyn_path = current.with_name(stem + "_dynamics_frames.csv")
    dyn = pd.read_csv(dyn_path)

    tc = find_col(dyn, ["time","timestamp","time_s"])
    fc = find_col(dyn, [
        "primary_impact_body_load_n_filt",
        "primary_impact_body_load_n",
    ])
    sc = find_col(dyn, ["support_vertical_n"])

    t = pd.to_numeric(dyn[tc], errors="coerce").to_numpy(float)
    force = pd.to_numeric(dyn[fc], errors="coerce").fillna(0).to_numpy(float)

    weight = finite(r.get("weight_kg")) or 70.0
    mass = weight

    if sc:
        support = pd.to_numeric(dyn[sc], errors="coerce").fillna(0).to_numpy(float)
        early_support = support[:min(30, len(support))]
        positive = early_support[early_support > 0]
        if len(positive):
            mass = max(1.0, float(np.median(positive) / 9.81))
    else:
        support = np.full(len(t), np.nan)

    bw = mass * 9.81
    load_bw = force / bw

    j0 = int(np.searchsorted(t, onset, side="left"))
    j0 = min(j0, len(t)-1)

    # Main diagnostic window: first 3 s after onset.
    mask = (
        np.isfinite(t)
        & (t >= onset)
        & (t <= onset + 3.0)
    )
    idx = np.where(mask)[0]

    if len(idx):
        peak_j = int(idx[np.nanargmax(load_bw[idx])])
        peak_t = float(t[peak_j])
        peak_bw = float(load_bw[peak_j])
    else:
        peak_t = np.nan
        peak_bw = np.nan

    # First 0.20 s: does the threshold already hold continuously?
    immediate = (
        np.isfinite(t)
        & (t >= onset)
        & (t <= onset + 0.20)
    )
    immediate_idx = np.where(immediate)[0]

    frac_immediate_gt025 = (
        float(np.mean(load_bw[immediate_idx] > 0.25))
        if len(immediate_idx)
        else np.nan
    )

    # Strongest positive body-load transient in first 3 s.
    dload = np.gradient(force, t)
    if len(idx):
        rise_j = int(idx[np.nanargmax(dload[idx])])
        rise_t = float(t[rise_j])
    else:
        rise_t = np.nan

    # Legacy impact_n / accel_mag are QC comparators only.
    legacy_impact_peak_t = np.nan
    legacy_impact_peak = np.nan
    legacy_accel_peak_t = np.nan
    legacy_accel_peak = np.nan

    if legacy.exists():
        leg = pd.read_csv(legacy, comment="#")
        ltc = find_col(leg, ["t","timestamp","time","Time_s"])

        if ltc:
            lt = pd.to_numeric(leg[ltc], errors="coerce").to_numpy(float)
            lmask = (
                np.isfinite(lt)
                & (lt >= onset)
                & (lt <= onset + 3.0)
            )
            li = np.where(lmask)[0]

            if len(li) and "impact_n" in leg.columns:
                x = pd.to_numeric(
                    leg["impact_n"], errors="coerce"
                ).fillna(0).to_numpy(float)
                j = int(li[np.nanargmax(x[li])])
                legacy_impact_peak_t = float(lt[j])
                legacy_impact_peak = float(x[j])

            if len(li) and "accel_mag" in leg.columns:
                x = pd.to_numeric(
                    leg["accel_mag"], errors="coerce"
                ).to_numpy(float)
                j = int(li[np.nanargmax(x[li])])
                legacy_accel_peak_t = float(lt[j])
                legacy_accel_peak = float(x[j])

    rows.append({
        "scenario_id": task,
        "sim_age_profile": age,
        "onset_s": onset,
        "recovered_impact_s": recovered,
        "recovered_delay_s": recovered - onset,
        "load_at_onset_BW": float(load_bw[j0]),
        "support_at_onset_BW": (
            float(support[j0] / bw)
            if np.isfinite(support[j0])
            else np.nan
        ),
        "fraction_first_0p2s_load_gt_0p25BW": frac_immediate_gt025,
        "peak_body_load_time_s": peak_t,
        "peak_body_load_delay_s": peak_t - onset if np.isfinite(peak_t) else np.nan,
        "peak_body_load_BW": peak_bw,
        "strongest_load_rise_time_s": rise_t,
        "strongest_load_rise_delay_s": rise_t - onset if np.isfinite(rise_t) else np.nan,
        "legacy_impact_peak_time_s": legacy_impact_peak_t,
        "legacy_impact_peak_delay_s": (
            legacy_impact_peak_t - onset
            if np.isfinite(legacy_impact_peak_t)
            else np.nan
        ),
        "legacy_impact_peak_N": legacy_impact_peak,
        "legacy_accel_peak_time_s": legacy_accel_peak_t,
        "legacy_accel_peak_delay_s": (
            legacy_accel_peak_t - onset
            if np.isfinite(legacy_accel_peak_t)
            else np.nan
        ),
        "legacy_accel_peak": legacy_accel_peak,
    })

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

print("=" * 105)
print("EARLY-SCENARIO IMPACT SEMANTICS AUDIT")
print("=" * 105)
print("records:", len(out))

cols = [
    "recovered_delay_s",
    "load_at_onset_BW",
    "fraction_first_0p2s_load_gt_0p25BW",
    "peak_body_load_delay_s",
    "strongest_load_rise_delay_s",
    "legacy_impact_peak_delay_s",
    "legacy_accel_peak_delay_s",
]

for c in cols:
    print("\n", c)
    print(
        out.groupby("scenario_id")[c]
        .agg(["count","median","min","max"])
        .to_string()
    )

print("\nRepresentative age-25 records:")
print(
    out[out["sim_age_profile"] == 25][
        [
            "scenario_id",
            "onset_s",
            "recovered_delay_s",
            "load_at_onset_BW",
            "fraction_first_0p2s_load_gt_0p25BW",
            "peak_body_load_delay_s",
            "strongest_load_rise_delay_s",
            "legacy_impact_peak_delay_s",
            "legacy_accel_peak_delay_s",
        ]
    ].to_string(index=False)
)

print("\nAudit CSV:", OUT)
print("EARLY_IMPACT_SEMANTICS_AUDIT_DONE")
