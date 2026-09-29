#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, shutil
from datetime import datetime
from pathlib import Path

TEXT_EXTENSIONS={".py",".sh",".txt",".md",".json",".yaml",".yml",".toml",".ini",".cfg",".csv"}
PAPER_TASK_IDS=[20,21,22,23,24,28,29,30,31,32,33,34,37,38,39,40,41,42]

SIMULATOR_FILES=[
"fall_core.py","high_rate_imu_sidecar.py","highrate_runtime_bridge.py",
"biofidelic_profile.py","fall_scenario_library.py","legacy_runtime_common.py",
"fall_dispatcher.py","fall_task_base.py","backward_fall_walking_best.py",
"humanoid_basic.py","myosuite_age_profile.py","imu_pipeline_labels.py","label_imu_csv.py"
]

DATASET_FILES=[
"run_rne_corrected_396_campaign.py","finalize_highrate_event_manifest.py",
"build_phase2_corrected_event_dataset.py","build_phase2_corrected_split_manifests.py",
"build_phase2_corrected_augmentation_manifests.py",
"build_phase2_protechto_augmentation_manifests.py",
"build_phase2_lightweight_paper_aligned_inputs.py",
"make_protechto_compatible_simulated_dataset.py",
"prepare_protechto_exact_synthetic_cache.py",
"build_real_validation_inventory_v2.py","build_highrate_univr_dataset.py",
"analyze_real_vs_sim_univr.py"
]

CNN_FILES=[
"preflight_phase2_lightweight_paper_final.py",
"run_phase2_lightweight_paper_final_fold.py",
"aggregate_phase2_lightweight_paper_final.py",
"preflight_protechto_clean_real_only.py",
"preflight_protechto_exact_full_campaign.py",
"prepare_protechto_execution_mirror.py",
"run_protechto_exact_full_campaign.py",
"collect_protechto_exact_results.py",
"run_phase2_protechto_final_condition_fold.py",
"evaluate_protechto_final_condition_fold_events.py",
"run_phase2_protechto_parity_condition_fold.py",
"evaluate_protechto_parity_fold0_events.py",
"run_phase2_validation_fold0.py",
"protechto_requirements_clean.txt"
]

VALIDATION_FILES=[
"validate_simulator_against_real_v2.py","run_v2_publication_variant.py",
"generate_publication_results.py","compare_v2_truth_sampling_rich221.py",
"compare_v2_acquisition_variants.py",
"validate_gyro_inertial_frame_reconstruction.py","diagnose_gyro_frame_semantics.py"
]

AUDIT_FILES=[
"audit_phase2_publication_gate_stage1.py","compact_canonical_validation_audit.py",
"build_paper_evidence_archive.py","build_compact_paper_writing_archive.py",
"curate_paper_evidence_v3.py","build_final_canonical_publication_figures.py",
"collect_cnn_dataset_split_audit_stage1.py","locate_dataset_split_evidence.py",
"build_publication_snapshot_v1.py"
]

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def copy_one(root,src_rel,dst,manifest,missing):
    src=root/src_rel
    if not src.is_file():
        missing.append(src_rel); return
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
    st=src.stat()
    manifest.append({
        "source_path":src_rel,
        "snapshot_path":str(dst.relative_to(root/"publication_snapshot_2026_09")),
        "size_bytes":st.st_size,
        "modified_local":datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "sha256":sha256(src),
    })

def copy_tree_code(root,src_dir,dst_dir,manifest,missing):
    if not src_dir.is_dir():
        missing.append(str(src_dir.relative_to(root))); return
    for src in sorted(src_dir.rglob("*")):
        if not src.is_file() or src.suffix.lower() not in TEXT_EXTENSIONS: continue
        if any(x in src.parts for x in {"__pycache__",".git",".venv"}): continue
        dst=dst_dir/src.relative_to(src_dir)
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        st=src.stat()
        manifest.append({
            "source_path":str(src.relative_to(root)),
            "snapshot_path":str(dst.relative_to(root/"publication_snapshot_2026_09")),
            "size_bytes":st.st_size,
            "modified_local":datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "sha256":sha256(src),
        })

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path.cwd())
    a=ap.parse_args()
    root=a.root.resolve()
    snap=root/"publication_snapshot_2026_09"
    snap.mkdir(parents=True,exist_ok=True)
    manifest=[]; missing=[]

    for name in SIMULATOR_FILES:
        copy_one(root,name,snap/"simulator"/name,manifest,missing)

    for sid in PAPER_TASK_IDS:
        copy_one(root,f"scenario{sid}_legacy.py",snap/"scenarios"/"legacy"/f"scenario{sid}_legacy.py",manifest,missing)
        copy_one(root,f"scenarios/scenario_{sid}.py",snap/"scenarios"/"wrappers"/f"scenario_{sid}.py",manifest,missing)

    for name in DATASET_FILES:
        copy_one(root,name,snap/"dataset_generation"/name,manifest,missing)
    for name in CNN_FILES:
        copy_one(root,name,snap/"cnn_evaluation"/"campaign"/name,manifest,missing)
    for name in VALIDATION_FILES:
        copy_one(root,name,snap/"validation"/name,manifest,missing)
    for name in AUDIT_FILES:
        copy_one(root,name,snap/"audit_tools"/name,manifest,missing)

    base=root/"outputs"/"paper_evidence_archive_v1"/"01_METHOD_PROTECHTO"
    copy_tree_code(root,base/"canonical_source",snap/"cnn_evaluation"/"protechto_canonical",manifest,missing)
    copy_tree_code(root,base/"campaign_scripts",snap/"cnn_evaluation"/"protechto_campaign_archive",manifest,missing)
    copy_one(root,
        "outputs/paper_evidence_archive_v1/01_METHOD_PROTECHTO/execution_mirror/train.py",
        snap/"cnn_evaluation"/"protechto_execution_mirror"/"train.py",
        manifest,missing)

    manifest.sort(key=lambda r:r["snapshot_path"])
    with (snap/"MANIFEST.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["source_path","snapshot_path","size_bytes","modified_local","sha256"])
        w.writeheader(); w.writerows(manifest)

    (snap/"MISSING_FILES.txt").write_text("\n".join(sorted(set(missing)))+("\n" if missing else ""),encoding="utf-8")

    readme=f"""# Publication implementation snapshot

This is a compact code-only snapshot for auditing the implementation behind the current paper.

Paper task IDs:
{", ".join(map(str,PAPER_TASK_IDS))}

Included:
- current simulator/runtime code
- the 18 paper scenario implementations and wrappers
- corrected 396-fall campaign/event/split builders
- late-stage lightweight and Protechto CNN evaluation chains
- curated canonical Protechto source preserved in the existing paper evidence archive
- validation and audit scripts

Not included:
- raw KFall or UniVrFall data
- raw simulator sensor CSVs
- checkpoints
- videos
- large result folders

MANIFEST.csv records source path, snapshot path, size, modification time, and SHA-256.

The GitHub audit should establish which exact late-stage evaluation chain generated the manuscript Table VI / Figs. 6-7, after which a tiny local extractor can compute fold-level dataset counts, duplicate simulated draws, and leakage checks.
"""
    (snap/"README.md").write_text(readme,encoding="utf-8")

    total=sum(r["size_bytes"] for r in manifest)
    print("Snapshot:",snap)
    print("Copied files:",len(manifest))
    print("Missing requested paths:",len(set(missing)))
    print(f"Total source size: {total/1024/1024:.2f} MiB")
    print('Next: git add publication_snapshot_2026_09')

if __name__=="__main__":
    main()
