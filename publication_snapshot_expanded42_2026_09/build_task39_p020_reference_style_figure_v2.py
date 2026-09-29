#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")
RUN = PROJECT / (
    "outputs/_highrate_overnight/campaign_highrate_truth_v3_rne_corrected396/"
    "runs/P020/task_39/scenario39_age49_h1p63_sex_female_w73p0_20260826_144048"
)
MAIN = RUN / "fall_scenario39_age49_20260826_144048.csv"
HR = RUN / "fall_scenario39_age49_20260826_144048_highrate_truth.csv"

OUT = PROJECT / "outputs/task39_p020_reference_style_figure_v2"

PHASES = [
    ("SETUP",         4.633333333333462),
    ("FALL ONSET",    5.633333333333462),
    ("MAX DESCENT",   6.0133),
    ("FIRST CONTACT", 6.1433),
    ("PEAK IMPACT",   6.166666666666878),
    ("POST-IMPACT",   6.416666666666878),
    ("REST",          7.4833),
]
SHORT = ["Setup", "Fall\nonset", "Max\ndescent", "First\ncontact",
         "Peak\nimpact", "Post-\nimpact", "Rest"]

def read_csv(p):
    return pd.read_csv(p, comment="#", low_memory=False)

def nearest(df, t):
    return df.loc[(df.timestamp - t).abs().idxmin()]

def magnitude(df, cols):
    a = np.column_stack([df[c].to_numpy(float) for c in cols])
    return np.sqrt((a*a).sum(axis=1))

def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.35, alpha=0.28)
    ax.tick_params(direction="out", length=3, width=0.7)

def draw_schematic_person(ax, tilt_deg, ground_y, center_y, scale=1.0):
    """
    Schematic only: not a joint reconstruction.
    The figure is intentionally human-readable while the exact quantitative
    data are shown in the adjacent panels.
    """
    th = np.deg2rad(tilt_deg)
    c, s = np.cos(th), np.sin(th)

    def rot(x, y):
        return c*x - s*y, s*x + c*y

    # local body geometry, intentionally generic
    pts = {
        "pelvis": (0.0, 0.0),
        "neck":   (0.0, 0.62),
        "head":   (0.0, 0.82),
        "lsho":   (-0.18, 0.56),
        "rsho":   ( 0.18, 0.56),
        "lelb":   (-0.33, 0.34),
        "relb":   ( 0.35, 0.32),
        "lwri":   (-0.44, 0.12),
        "rwri":   ( 0.48, 0.08),
        "lhip":   (-0.10,-0.05),
        "rhip":   ( 0.10,-0.05),
        "lkne":   (-0.14,-0.40),
        "rkne":   ( 0.18,-0.42),
        "lank":   (-0.08,-0.76),
        "rank":   ( 0.27,-0.74),
    }
    # Slight phase-dependent protective reach while falling.
    if tilt_deg > 25:
        pts["lelb"] = (-0.05, 0.28)
        pts["relb"] = ( 0.34, 0.18)
        pts["lwri"] = ( 0.15, 0.08)
        pts["rwri"] = ( 0.55,-0.02)
    if tilt_deg > 65:
        pts["lkne"] = (-0.22,-0.28)
        pts["rkne"] = ( 0.12,-0.36)
        pts["lank"] = (-0.42,-0.42)
        pts["rank"] = ( 0.30,-0.48)

    world = {}
    for k,(x,y) in pts.items():
        xx, yy = rot(x*scale, y*scale)
        world[k] = (xx, yy + center_y)

    bones = [
        ("pelvis","neck"),
        ("lsho","rsho"),
        ("neck","lsho"),("neck","rsho"),
        ("lsho","lelb"),("lelb","lwri"),
        ("rsho","relb"),("relb","rwri"),
        ("lhip","rhip"),
        ("pelvis","lhip"),("pelvis","rhip"),
        ("lhip","lkne"),("lkne","lank"),
        ("rhip","rkne"),("rkne","rank"),
    ]
    for a,b in bones:
        ax.plot([world[a][0],world[b][0]],[world[a][1],world[b][1]],
                lw=2.0, color="0.22", solid_capstyle="round")
    for k,(x,y) in world.items():
        if k == "head":
            continue
        ax.scatter(x,y,s=10,color="0.15",zorder=3)
    hx,hy=world["head"]
    ax.add_patch(plt.Circle((hx,hy),0.10*scale,fill=False,lw=1.8,color="0.20"))
    ax.axhline(ground_y,lw=0.9,color="0.45")
    ax.set_xlim(-0.75,0.8)
    ax.set_ylim(ground_y-0.06,max(1.05,center_y+0.9))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

def phase_color(i):
    cmap = plt.get_cmap("tab10")
    return cmap(i % 10)

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    df = read_csv(MAIN)
    hr = read_csv(HR)

    req = {"timestamp","pelvis_height","sensor_pos_x","sensor_pos_y","sensor_pos_z",
           "sensor_vel_x","sensor_vel_y","sensor_vel_z",
           "impact_force","impact_magnitude","jerk_mag"}
    miss = sorted(req-set(df.columns))
    if miss:
        raise SystemExit(f"Missing main columns: {miss}")

    for c in ["accel_true_x","accel_true_y","accel_true_z",
              "gyro_true_x","gyro_true_y","gyro_true_z"]:
        if c not in hr.columns:
            raise SystemExit(f"Missing high-rate truth column: {c}")

    phase_rows=[]
    setup = nearest(df,PHASES[0][1])
    setup_h=float(setup.pelvis_height)
    for idx,(name,t) in enumerate(PHASES,1):
        r=nearest(df,t); rh=nearest(hr,t)
        phase_rows.append({
            "index":idx,"phase":name,"target_time_s":t,
            "sample_time_s":float(r.timestamp),
            "pelvis_height_m":float(r.pelvis_height),
            "pelvis_drop_m":setup_h-float(r.pelvis_height),
            "sensor_x_m":float(r.sensor_pos_x),
            "sensor_y_m":float(r.sensor_pos_y),
            "sensor_z_m":float(r.sensor_pos_z),
            "impact_force_N":float(r.impact_force),
            "impact_magnitude":float(r.impact_magnitude),
            "jerk_mag":float(r.jerk_mag),
            "accel_mag_mps2":float(np.sqrt(rh.accel_true_x**2+rh.accel_true_y**2+rh.accel_true_z**2)),
            "gyro_mag_rps":float(np.sqrt(rh.gyro_true_x**2+rh.gyro_true_y**2+rh.gyro_true_z**2)),
        })
    ps=pd.DataFrame(phase_rows)
    ps.to_csv(OUT/"phase_values_exact.csv",index=False)

    # Derived exact series.
    t=df.timestamp.to_numpy(float)
    th=hr.timestamp.to_numpy(float)
    accmag=magnitude(hr,["accel_true_x","accel_true_y","accel_true_z"])
    gyromag=magnitude(hr,["gyro_true_x","gyro_true_y","gyro_true_z"])

    t0=PHASES[0][1]-0.25
    t1=PHASES[-1][1]+0.35
    mm=(t>=t0)&(t<=t1)
    mh=(th>=t0)&(th<=t1)

    plt.rcParams.update({
        "font.family":"DejaVu Sans",
        "font.size":7.8,
        "axes.titlesize":8.7,
        "axes.labelsize":8.0,
        "xtick.labelsize":6.8,
        "ytick.labelsize":6.8,
        "legend.fontsize":6.6,
        "pdf.fonttype":42,
        "ps.fonttype":42,
    })

    fig=plt.figure(figsize=(7.15,7.75))
    gs=GridSpec(4,2,figure=fig,
                height_ratios=[1.18,1.25,1.00,0.93],
                hspace=0.58,wspace=0.34)

    # ------------------------------------------------------------------
    # A. Reference-style seven-phase schematic strip
    # ------------------------------------------------------------------
    top=gs[0,:].subgridspec(1,7,wspace=0.05)
    drops=ps.pelvis_drop_m.to_numpy()
    dnorm=(drops-drops.min())/(max(drops.max()-drops.min(),1e-9))
    # Visual tilt is schematic, monotonically tied to canonical phase progression.
    tilts=[0,12,42,68,82,88,90]
    min_h=float(ps.pelvis_height_m.min())
    max_h=float(ps.pelvis_height_m.max())
    for i in range(7):
        ax=fig.add_subplot(top[0,i])
        # normalize pelvis height to visual panel without pretending exact body geometry
        hnorm=(ps.iloc[i].pelvis_height_m-min_h)/max(max_h-min_h,1e-9)
        center_y=0.30+0.56*hnorm
        draw_schematic_person(ax,tilts[i],0.0,center_y,0.54)
        ax.text(0.5,1.045,f"{i+1}",transform=ax.transAxes,ha="center",va="bottom",
                fontsize=7.0,fontweight="bold",color=phase_color(i))
        ax.text(0.5,-0.07,SHORT[i],transform=ax.transAxes,ha="center",va="top",
                fontsize=6.5,fontweight="bold")
        ax.text(0.5,-0.23,f"{ps.iloc[i].target_time_s:.2f} s",
                transform=ax.transAxes,ha="center",va="top",fontsize=5.9,color="0.30")
    fig.text(0.075,0.968,"(a) Seven-phase Task-39 fall sequence",
             ha="left",va="top",fontsize=8.8,fontweight="bold")
    fig.text(0.075,0.945,
             "Body glyphs are schematic only; phase timing and all quantitative panels use the exact corrected396 P020 data.",
             ha="left",va="top",fontsize=6.5,color="0.35")

    # ------------------------------------------------------------------
    # B. 3D lower-back sensor trajectory (reference OpenPose-like grammar)
    # ------------------------------------------------------------------
    ax=fig.add_subplot(gs[1,0],projection="3d")
    sl=df.loc[mm]
    ax.plot(sl.sensor_pos_x,sl.sensor_pos_y,sl.sensor_pos_z,lw=1.45,color="0.15")
    for i,row in ps.iterrows():
        ax.scatter(row.sensor_x_m,row.sensor_y_m,row.sensor_z_m,
                   s=26,color=phase_color(i),depthshade=False)
        ax.text(row.sensor_x_m,row.sensor_y_m,row.sensor_z_m,
                f" {i+1}",fontsize=6.2)
    ax.set_xlabel("X [m]",labelpad=1)
    ax.set_ylabel("Y [m]",labelpad=1)
    ax.set_zlabel("Z [m]",labelpad=1)
    ax.tick_params(pad=0,labelsize=5.8)
    ax.view_init(elev=23,azim=-58)
    ax.set_title("(b) Canonical lower-back 3-D trajectory",loc="left",fontweight="bold",pad=4)
    ax.grid(True,linewidth=0.3,alpha=0.35)

    # ------------------------------------------------------------------
    # C. Pelvis height and impact
    # ------------------------------------------------------------------
    ax=fig.add_subplot(gs[1,1])
    ax.plot(t[mm],df.loc[mm,"pelvis_height"],lw=1.35,label="Pelvis height [m]")
    axr=ax.twinx()
    axr.plot(t[mm],df.loc[mm,"impact_force"],lw=1.05,ls="--",label="Impact force [N]")
    for i,(_,tt) in enumerate(PHASES):
        ax.axvline(tt,lw=0.65,color=phase_color(i),alpha=0.7)
    ax.set_xlim(t0,t1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pelvis height [m]")
    axr.set_ylabel("Impact force [N]")
    clean(ax); axr.spines["top"].set_visible(False)
    ax.set_title("(c) Descent and ground-impact response",loc="left",fontweight="bold")
    lines=ax.get_lines()[:1]+axr.get_lines()[:1]
    ax.legend(lines,[l.get_label() for l in lines],frameon=False,loc="upper right")

    # ------------------------------------------------------------------
    # D. Corrected acceleration magnitude
    # ------------------------------------------------------------------
    ax=fig.add_subplot(gs[2,0])
    ax.plot(th[mh],accmag[mh],lw=1.25,label="|a|")
    for i,(_,tt) in enumerate(PHASES):
        ax.axvline(tt,lw=0.65,color=phase_color(i),alpha=0.7)
    ax.set_xlim(t0,t1)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Acceleration [m/s²]")
    clean(ax)
    ax.set_title("(d) Corrected lower-back acceleration",loc="left",fontweight="bold")
    ax.legend(frameon=False)

    # E. Corrected angular velocity
    ax=fig.add_subplot(gs[2,1])
    ax.plot(th[mh],gyromag[mh],lw=1.25,label="|ω|")
    for i,(_,tt) in enumerate(PHASES):
        ax.axvline(tt,lw=0.65,color=phase_color(i),alpha=0.7)
    ax.set_xlim(t0,t1)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Angular velocity [rad/s]")
    clean(ax)
    ax.set_title("(e) Corrected lower-back angular velocity",loc="left",fontweight="bold")
    ax.legend(frameon=False)

    # ------------------------------------------------------------------
    # F. Exact phase values in compact academic table
    # ------------------------------------------------------------------
    ax=fig.add_subplot(gs[3,:]); ax.axis("off")
    table_data=[]
    for _,r in ps.iterrows():
        table_data.append([
            str(int(r["index"])),
            r["phase"].replace(" ","\n") if r["phase"]=="FIRST CONTACT" else r["phase"],
            f'{r["target_time_s"]:.2f}',
            f'{r["pelvis_height_m"]:.3f}',
            f'{r["impact_force_N"]:.0f}',
            f'{r["accel_mag_mps2"]:.1f}',
            f'{r["gyro_mag_rps"]:.2f}',
        ])
    collabels=["#","Phase","t [s]","Pelvis h [m]","Impact F [N]","|a| [m/s²]","|ω| [rad/s]"]
    tab=ax.table(cellText=table_data,colLabels=collabels,loc="center",
                 cellLoc="center",colLoc="center",
                 colWidths=[0.045,0.20,0.09,0.13,0.13,0.13,0.13])
    tab.auto_set_font_size(False); tab.set_fontsize(6.3); tab.scale(1,1.18)
    for (r,c),cell in tab.get_celld().items():
        cell.set_linewidth(0.35)
        cell.set_edgecolor("0.72")
        if r==0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("0.94")
    ax.set_title("(f) Exact synchronized phase values",loc="left",fontweight="bold",pad=3)

    fig.subplots_adjust(left=0.075,right=0.96,top=0.91,bottom=0.055)

    base=OUT/"fig_task39_p020_reference_style_v2"
    fig.savefig(base.with_suffix(".png"),dpi=600,bbox_inches="tight",facecolor="white")
    fig.savefig(base.with_suffix(".pdf"),bbox_inches="tight",facecolor="white")
    fig.savefig(base.with_suffix(".svg"),bbox_inches="tight",facecolor="white")
    plt.close(fig)

    caption=(
        "Task-39 forward fall from height for representative profile P020 from the final corrected396 campaign. "
        "(a) Seven-phase visual storyboard at the locked canonical phase times; human glyphs are schematic and "
        "are included only to make the event progression visually interpretable, not as reconstructed joint kinematics. "
        "(b) Exact canonical three-dimensional lower-back virtual-sensor trajectory. "
        "(c) Canonical pelvis-height descent and impact-force response. "
        "(d,e) Corrected physics-truth lower-back acceleration and angular-velocity magnitudes. "
        "(f) Exact synchronized phase values. No replay-derived qpos, anatomical skeleton, contact identity, "
        "or MuJoCo phase render is used because the historical-source replay failed the strict equivalence gate."
    )
    (OUT/"FIGURE_CAPTION.txt").write_text(caption+"\n",encoding="utf-8")

    prov={
        "status":"PAPER_SAFE_CANONICAL_PLUS_EXPLICIT_SCHEMATIC",
        "canonical_main":str(MAIN),
        "canonical_highrate_truth":str(HR),
        "phase_times_s":dict(PHASES),
        "schematic_pose_panel":{
            "exact_joint_reconstruction":False,
            "purpose":"visual explanation only",
            "scientific_values_in_panel":"phase times and pelvis-height-based vertical placement",
        },
        "replay_pose_used":False,
        "replay_skeleton_used":False,
        "all_quantitative_curves_from_corrected396":True,
    }
    (OUT/"FIGURE_PROVENANCE.json").write_text(json.dumps(prov,indent=2),encoding="utf-8")

    print("="*92)
    print("TASK39 P020 REFERENCE-STYLE FIGURE V2: COMPLETE")
    print("="*92)
    print("Status: PAPER_SAFE_CANONICAL_PLUS_EXPLICIT_SCHEMATIC")
    print("Exact replay skeleton used: NO")
    print("PNG:",base.with_suffix(".png"))
    print("PDF:",base.with_suffix(".pdf"))
    print("SVG:",base.with_suffix(".svg"))
    print("Caption:",OUT/"FIGURE_CAPTION.txt")
    print("Phase CSV:",OUT/"phase_values_exact.csv")
    print("="*92)

if __name__=="__main__":
    main()
