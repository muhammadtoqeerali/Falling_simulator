Final lightweight-paper campaign bundle.

Copy all four executable scripts into:
/mnt/hdd16T/ToqeerHomeBackup/mujoco_project

Run preflight first:
./.venv/bin/python -u preflight_phase2_lightweight_paper_final.py

Only if it prints FINAL CAMPAIGN PRELAUNCH GATE: PASS, launch:
CAMPAIGN_GPU=1 bash run_phase2_lightweight_paper_final_campaign.sh

The launcher is resume-safe: any fold with a COMPLETE marker is skipped.
Final output root:
outputs/phase2_lightweight_paper_final_campaign_v1

This runner consumes the already pre-impact-150-aligned physical and synthetic inputs. It does not apply the 150-ms correction again.
