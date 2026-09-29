#!/usr/bin/env python3
"""
TASK39 CORRECTED396 PROVENANCE ADJUDICATION V4
===============================================

Static/read-only follow-up to V3.

Why V4 exists
-------------
V3 correctly found SOURCE_OR_RUNTIME_VERSION_MISMATCH, but its generic ranking
allowed diagnostic scripts to become "source candidates". In the observed V3
run, diagnose_task39_replay_mismatch_v1.py ranked first, which is a provenance
false positive: diagnostic code can mention both canonical and replay strings
without ever being executable campaign source.

V4 therefore:
  * never imports or executes scenario39/MuJoCo/HumEnv/Meta Motivo;
  * excludes diagnostic/replay/audit/extractor/patch/test code from runtime candidates;
  * restricts simulator-source candidates to scenario39-family source/backups/history;
  * inspects pre-campaign Git history and filesystem backups;
  * traces pre-campaign campaign/runner references to Task 39;
  * compares canonical export-name/version/timing signatures;
  * preserves the paper gate as BLOCK.

No simulator rerun is performed.
"""
from __future__ import annotations

import ast
import csv
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT = Path(os.environ.get(
    "MUJOCO_PROJECT_ROOT",
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
)).resolve()

V3 = PROJECT / "outputs/task39_corrected396_provenance_v3"
CANON_ROOT = PROJECT / "outputs/_highrate_overnight/campaign_highrate_truth_v3_rne_corrected396"
P020_ROOT = CANON_ROOT / "runs/P020/task_39/scenario39_age49_h1p63_sex_female_w73p0_20260826_144048"
CANON_MAIN = P020_ROOT / "fall_scenario39_age49_20260826_144048.csv"
CANON_MANIFEST = P020_ROOT / "run_manifest.json"

OUT = PROJECT / "outputs/task39_corrected396_provenance_adjudication_v4"
SRC = OUT / "01_TRUE_SOURCE_CANDIDATES"
GIT = OUT / "02_PRECAMPAIGN_GIT"
CALL = OUT / "03_CAMPAIGN_CALL_CHAIN"
SIG = OUT / "04_SIGNATURE_COMPARISON"
DEC = OUT / "05_DECISION"

CAMPAIGN_END = dt.datetime(2026, 8, 27, 23, 59, 59)
CAMPAIGN_GIT_BEFORE = "2026-08-28 00:00:00"
EXPECTED_CANON_DURATION_S = 18.56
NATIVE_DT_HINT = 1.0 / 30.0
EXPECTED_NATIVE_INTERVALS = EXPECTED_CANON_DURATION_S / NATIVE_DT_HINT
FAILED_REPLAY_HASH = "873171f5879da95233e9094cd655f214b1538ac0031fd3ee8e64299f8eef1fb8"

EXCLUDE_TOKENS = {
    "diagnos", "diagnostic", "replay", "audit", "extract", "collector", "patch",
    "test", "bundle", "schema", "probe", "evidence", "figure", "provenance"
}

SCENARIO_NAME_RE = re.compile(
    r"^scenario[_-]?39(?:$|[_\-.].*)",
    re.I
)

TEXTLIKE_EXTS = {
    ".py", ".txt", ".bak", ".old", ".orig", ".backup", ".save", ".tmp", ".copy",
    ".py~", ".disabled"
}

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

def write_csv(p: Path, rows: list[dict[str, Any]], fields: list[str] | None = None):
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    if fields is None:
        fields, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    fields.append(k)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def read_csv_dicts(p: Path) -> list[dict[str, str]]:
    if not p.exists():
        return []
    with p.open("r", newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))

def safe_text(p: Path, max_bytes=8_000_000) -> str:
    try:
        if p.stat().st_size > max_bytes:
            return ""
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

def run_git(args: list[str], timeout=30) -> str:
    try:
        cp = subprocess.run(
            ["git", "-C", str(PROJECT)] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return cp.stdout
    except Exception as e:
        return f"ERROR: {e}"

def is_excluded_name(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in EXCLUDE_TOKENS)

def is_scenario39_family_name(name: str) -> bool:
    low = name.lower()
    stemish = low
    # Accept source backups such as scenario39_legacy.py.bak.
    return bool(SCENARIO_NAME_RE.match(stemish)) and not is_excluded_name(low)

def iter_source_tree() -> Iterable[Path]:
    prune = {".venv", ".git", "__pycache__", "node_modules", "outputs", ".cache"}
    for root, dirs, files in os.walk(PROJECT):
        dirs[:] = [d for d in dirs if d not in prune]
        for fn in files:
            yield Path(root) / fn

def collect_scenario_family_files() -> list[Path]:
    out = set()
    for p in iter_source_tree():
        if not is_scenario39_family_name(p.name):
            continue
        try:
            if p.stat().st_size <= 8_000_000:
                out.add(p.resolve())
        except Exception:
            pass
    return sorted(out)

def source_static_signatures(txt: str) -> dict[str, Any]:
    low = txt.lower()
    sig: dict[str, Any] = {
        "has_run_with_subject": bool(re.search(r"\bdef\s+run_with_subject\s*\(", txt)),
        "has_task39_height_layer": "Task39HeightLayer" in txt,
        "has_bio_controller": "EnhancedBiofidelicController" in txt,
        "has_marker_exporter": "MarkerKinematicsExporter" in txt,
        "has_contact_analyzer": "DynamicsContactAnalyzer" in txt,
        "has_paper_alignment": "PaperAlignmentExporter" in txt,
        "has_rne_postconstraint": "mj_rnePostConstraint" in txt,
        "canonical_filename_literal": "fall_scenario39" in low,
        "failed_replay_filename_literal": "fall_forward_height" in low,
        "scenario39_banner": "scenario 39" in low,
        "forward_fall_phrase": "forward fall" in low,
        "seed_42_literal": bool(re.search(r"\bseed\s*[:=]\s*42\b", low)),
    }

    nums = []
    step_assignments = []
    version_strings = []
    try:
        tree = ast.parse(txt)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                for t in targets:
                    if not isinstance(t, ast.Name):
                        continue
                    nm = t.id
                    try:
                        lit = ast.literal_eval(value)
                    except Exception:
                        continue
                    if isinstance(lit, (int, float)):
                        if "step" in nm.lower() or "frame" in nm.lower():
                            step_assignments.append({
                                "name": nm, "value": lit, "line": getattr(node, "lineno", None)
                            })
                            nums.append(float(lit))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value
                sl = s.lower()
                if ("scenario 39" in sl or "v30" in sl or "postfall" in sl or
                    "forward fall" in sl):
                    version_strings.append(s[:500])
    except Exception as e:
        sig["ast_error"] = str(e)

    sig["step_assignments"] = step_assignments
    sig["version_strings"] = sorted(set(version_strings))
    sig["numeric_step_values"] = sorted(set(nums))

    # Static timing hints only; never treated as replay equivalence.
    expected_frames_rounded = round(EXPECTED_NATIVE_INTERVALS)
    sig["contains_native_frame_hint_557"] = any(abs(n - 557) <= 2 for n in nums)
    sig["contains_failed_replay_total_432"] = any(abs(n - 432) <= 1 for n in nums)
    sig["expected_native_intervals_at_30hz"] = EXPECTED_NATIVE_INTERVALS
    sig["expected_native_frames_near"] = expected_frames_rounded
    return sig

def key_excerpts(txt: str, radius=3) -> str:
    terms = [
        "Scenario 39", "fall_scenario39", "fall_forward_height", "run_with_subject",
        "stand_steps", "walk_steps", "step_steps", "perturb", "react", "fall_steps",
        "total_steps", "seed", "Task39HeightLayer", "EnhancedBiofidelicController",
        "MarkerKinematicsExporter", "DynamicsContactAnalyzer", "PaperAlignmentExporter",
        "mj_rnePostConstraint", "v30-postfall", "simulation_steps", "duration"
    ]
    lines = txt.splitlines()
    hit = []
    used = set()
    for i, line in enumerate(lines):
        if any(t.lower() in line.lower() for t in terms):
            for j in range(max(0, i-radius), min(len(lines), i+radius+1)):
                if j not in used:
                    used.add(j)
                    hit.append((j+1, lines[j]))
        if len(hit) > 500:
            break
    return "\n".join(f"{n:6d}: {line}" for n, line in hit)

def git_precampaign_versions(rel: Path) -> list[dict[str, Any]]:
    log = run_git([
        "log", "--all", f"--before={CAMPAIGN_GIT_BEFORE}",
        "--date=iso", "--format=%H|%ad|%an|%s", "--", str(rel)
    ])
    rows = []
    for line in log.splitlines():
        if not re.match(r"^[0-9a-f]{40}\|", line):
            continue
        commit, date, author, subject = (line.split("|", 3) + ["","","",""])[:4]
        blob = run_git(["show", f"{commit}:{rel}"], timeout=20)
        if blob.startswith("fatal:") or blob.startswith("ERROR:"):
            continue
        b = blob.encode("utf-8", errors="replace")
        sig = source_static_signatures(blob)
        rows.append({
            "commit": commit,
            "date": date,
            "author": author,
            "subject": subject,
            "relative_path": str(rel),
            "blob_sha256": sha256_bytes(b),
            "canonical_filename_literal": sig["canonical_filename_literal"],
            "failed_replay_filename_literal": sig["failed_replay_filename_literal"],
            "has_run_with_subject": sig["has_run_with_subject"],
            "has_rne_postconstraint": sig["has_rne_postconstraint"],
            "contains_native_frame_hint_557": sig["contains_native_frame_hint_557"],
            "contains_failed_replay_total_432": sig["contains_failed_replay_total_432"],
            "version_strings": json.dumps(sig["version_strings"]),
            "step_assignments": json.dumps(sig["step_assignments"]),
            "_blob": blob,
        })
    return rows

def all_historical_scenario_paths() -> list[str]:
    out = run_git([
        "log", "--all", f"--before={CAMPAIGN_GIT_BEFORE}",
        "--name-only", "--pretty=format:"
    ], timeout=45)
    paths = set()
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        name = Path(s).name
        if is_scenario39_family_name(name):
            paths.add(s)
    return sorted(paths)

def load_v3_campaign_rows() -> list[dict[str, str]]:
    return read_csv_dicts(V3 / "03_CAMPAIGN_CHAIN/campaign_chain_candidates.csv")

def plausible_campaign_controller(row: dict[str, str]) -> bool:
    p = Path(row.get("path", ""))
    name = p.name.lower()
    if is_excluded_name(name):
        return False
    # Do not allow scenario source itself to count as its own caller.
    if is_scenario39_family_name(name):
        return False
    # Prefer actual code/launchers/config/logs, not generated result manifests.
    return p.suffix.lower() in {".py", ".sh", ".bash", ".zsh", ".txt", ".log", ".json", ".yaml", ".yml"}

def precampaign_file(p: Path) -> bool:
    try:
        return dt.datetime.fromtimestamp(p.stat().st_mtime) <= CAMPAIGN_END
    except Exception:
        return False

def direct_reference_hits(p: Path, candidate_names: set[str], candidate_paths: set[str]) -> list[dict[str, Any]]:
    txt = safe_text(p)
    if not txt:
        return []
    rows = []
    lines = txt.splitlines()
    for i, line in enumerate(lines, 1):
        low = line.lower()
        refs = []
        for n in candidate_names:
            if n.lower() in low:
                refs.append(n)
        for rp in candidate_paths:
            if rp.lower() in low:
                refs.append(rp)
        if refs or "scenario39" in low or "scenario_39" in low:
            # Require Task39/scenario context; preserve line for review.
            if any(k in low for k in ["scenario39", "scenario_39", "task_39", "task39", " 39", "[39", "'39", '"39']):
                rows.append({
                    "controller_path": str(p),
                    "controller_mtime_iso": dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                    "controller_precampaign_mtime": precampaign_file(p),
                    "line": i,
                    "references": ";".join(sorted(set(refs))),
                    "text": line[:2000],
                })
    return rows[:300]

def canonical_preamble() -> list[str]:
    out = []
    if not CANON_MAIN.exists():
        return out
    with CANON_MAIN.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                out.append(line.rstrip())
            else:
                break
    return out

def main():
    for d in [OUT, SRC, GIT, CALL, SIG, DEC]:
        d.mkdir(parents=True, exist_ok=True)

    print("="*94)
    print("TASK39 CORRECTED396 PROVENANCE ADJUDICATION V4 — NO SIMULATOR RERUN")
    print("="*94)
    print("Paper gate on entry: BLOCK")
    print("V3 ranking false-positive class explicitly excluded: diagnostic/replay/audit/etc.")

    # ------------------------------------------------------------------
    # A. Filesystem scenario39-family candidates only
    # ------------------------------------------------------------------
    fs_files = collect_scenario_family_files()
    fs_rows = []
    excerpts = []
    candidate_names = set()
    candidate_relpaths = set()

    for p in fs_files:
        txt = safe_text(p)
        if not txt:
            continue
        try:
            rel = p.relative_to(PROJECT)
        except Exception:
            continue
        candidate_names.add(p.name)
        candidate_relpaths.add(str(rel))
        st = p.stat()
        sig = source_static_signatures(txt)
        fs_rows.append({
            "path": str(p),
            "relative_path": str(rel),
            "sha256": sha256_file(p),
            "mtime_iso": dt.datetime.fromtimestamp(st.st_mtime).isoformat(),
            "mtime_before_campaign_end": precampaign_file(p),
            "same_hash_as_failed_replay_source": sha256_file(p) == FAILED_REPLAY_HASH,
            **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
               for k, v in sig.items()},
        })
        excerpts.append(
            f"\n{'='*110}\nFILESYSTEM SOURCE: {p}\n"
            f"SHA256: {sha256_file(p)}\n"
            f"MTIME: {dt.datetime.fromtimestamp(st.st_mtime).isoformat()}\n"
            f"{'='*110}\n{key_excerpts(txt)}\n"
        )

    write_csv(SRC / "filesystem_scenario39_candidates.csv", fs_rows)
    (SRC / "filesystem_source_excerpts.txt").write_text("".join(excerpts), encoding="utf-8")

    # ------------------------------------------------------------------
    # B. Git paths/versions that existed before campaign end
    # ------------------------------------------------------------------
    git_paths = set(all_historical_scenario_paths())
    git_paths.update(candidate_relpaths)
    git_rows = []
    latest_pre_by_path = {}
    hist_excerpts = []

    for srel in sorted(git_paths):
        rel = Path(srel)
        versions = git_precampaign_versions(rel)
        if not versions:
            continue
        latest = versions[0]  # git log newest first
        latest_pre_by_path[srel] = latest
        for idx, r in enumerate(versions[:20]):
            blob = r.pop("_blob")
            r["version_rank_newest_first"] = idx + 1
            git_rows.append(r)
            # Preserve newest pre-campaign blob for inspection without executing it.
            if idx == 0:
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", srel)
                bp = GIT / "latest_precampaign_blobs" / f"{safe_name}__{r['commit'][:12]}.py.txt"
                bp.parent.mkdir(parents=True, exist_ok=True)
                bp.write_text(blob, encoding="utf-8")
                hist_excerpts.append(
                    f"\n{'='*110}\nGIT PRE-CAMPAIGN SOURCE: {srel}\n"
                    f"COMMIT: {r['commit']}\nDATE: {r['date']}\nSHA256: {r['blob_sha256']}\n"
                    f"{'='*110}\n{key_excerpts(blob)}\n"
                )

    write_csv(GIT / "scenario39_precampaign_git_versions.csv", git_rows)
    (GIT / "latest_precampaign_source_excerpts.txt").write_text("".join(hist_excerpts), encoding="utf-8")

    # ------------------------------------------------------------------
    # C. Campaign caller chain — filter out diagnostics, replays, source itself
    # ------------------------------------------------------------------
    v3_rows = load_v3_campaign_rows()
    controller_rows = []
    seen_ctrl = set()
    for r in v3_rows:
        if not plausible_campaign_controller(r):
            continue
        p = Path(r.get("path", ""))
        if not p.exists() or p in seen_ctrl:
            continue
        seen_ctrl.add(p)
        controller_rows.extend(direct_reference_hits(p, candidate_names, candidate_relpaths))

    # Also scan likely root launchers even if V3 did not retain them.
    for p in iter_source_tree():
        if p in seen_ctrl or not p.is_file():
            continue
        lowname = p.name.lower()
        if is_excluded_name(lowname):
            continue
        if p.suffix.lower() not in {".py", ".sh", ".bash", ".zsh", ".txt", ".yaml", ".yml", ".json"}:
            continue
        if not any(k in lowname for k in ["campaign", "run", "launch", "batch", "overnight", "highrate", "high_rate", "all"]):
            continue
        hits = direct_reference_hits(p, candidate_names, candidate_relpaths)
        if hits:
            controller_rows.extend(hits)
            seen_ctrl.add(p)

    write_csv(CALL / "campaign_task39_direct_references.csv", controller_rows)

    # ------------------------------------------------------------------
    # D. Candidate adjudication ranking
    # ------------------------------------------------------------------
    candidates = []

    def add_candidate(kind, path, sha, sig, date_or_mtime, historical_commit=None):
        score = 0
        reasons = []
        if sig.get("canonical_filename_literal"):
            score += 25; reasons.append("contains canonical fall_scenario39 filename signature")
        if sig.get("failed_replay_filename_literal"):
            score -= 12; reasons.append("contains failed-replay fall_forward_height signature")
        if sig.get("has_run_with_subject"):
            score += 10; reasons.append("defines run_with_subject")
        if sig.get("has_task39_height_layer"):
            score += 8; reasons.append("contains Task39HeightLayer")
        if sig.get("has_bio_controller"):
            score += 8; reasons.append("contains EnhancedBiofidelicController")
        if sig.get("has_rne_postconstraint"):
            score += 8; reasons.append("contains mj_rnePostConstraint")
        if sig.get("contains_native_frame_hint_557"):
            score += 15; reasons.append("contains ~557-frame static timing hint compatible with 18.56 s at ~30 Hz")
        if sig.get("contains_failed_replay_total_432"):
            score -= 10; reasons.append("contains 432-frame hint associated with failed 14.36 s replay")
        if sha == FAILED_REPLAY_HASH:
            score -= 20; reasons.append("same hash as already-failed replay source")

        name = Path(path).name
        rel = str(path)
        direct = []
        for rr in controller_rows:
            refs = rr.get("references", "")
            txt = rr.get("text", "")
            if name in refs or rel in refs or name.lower() in txt.lower():
                direct.append(rr)
        pre_direct = [x for x in direct if x.get("controller_precampaign_mtime") is True]
        if pre_direct:
            score += 35
            reasons.append("direct Task-39 reference from controller with pre-campaign filesystem mtime")
        elif direct:
            score += 12
            reasons.append("direct Task-39 controller reference found; timestamp not pre-campaign")

        if kind == "GIT_PRECAMPAIGN":
            score += 20
            reasons.append("version is preserved in Git before corrected396 campaign")

        candidates.append({
            "score": score,
            "kind": kind,
            "path": path,
            "sha256": sha,
            "date_or_mtime": date_or_mtime,
            "historical_commit": historical_commit or "",
            "same_hash_as_failed_replay_source": sha == FAILED_REPLAY_HASH,
            "canonical_filename_literal": sig.get("canonical_filename_literal"),
            "failed_replay_filename_literal": sig.get("failed_replay_filename_literal"),
            "has_run_with_subject": sig.get("has_run_with_subject"),
            "has_rne_postconstraint": sig.get("has_rne_postconstraint"),
            "contains_native_frame_hint_557": sig.get("contains_native_frame_hint_557"),
            "contains_failed_replay_total_432": sig.get("contains_failed_replay_total_432"),
            "direct_controller_references": len(direct),
            "precampaign_controller_references": len(pre_direct),
            "version_strings": json.dumps(sig.get("version_strings", [])),
            "step_assignments": json.dumps(sig.get("step_assignments", [])),
            "reasons": " | ".join(reasons),
        })

    for r in fs_rows:
        sig = {}
        for key in [
            "canonical_filename_literal", "failed_replay_filename_literal",
            "has_run_with_subject", "has_task39_height_layer", "has_bio_controller",
            "has_rne_postconstraint", "contains_native_frame_hint_557",
            "contains_failed_replay_total_432", "version_strings", "step_assignments"
        ]:
            v = r.get(key)
            if key in {"version_strings", "step_assignments"} and isinstance(v, str):
                try: v = json.loads(v)
                except Exception: pass
            sig[key] = v
        add_candidate("FILESYSTEM", r["relative_path"], r["sha256"], sig, r["mtime_iso"])

    # Use newest pre-campaign version of each historical path as the historically relevant candidate.
    for srel, r in latest_pre_by_path.items():
        # Re-read blob text to get full static signature reliably.
        blob = run_git(["show", f"{r['commit']}:{srel}"], timeout=20)
        if blob.startswith("fatal:") or blob.startswith("ERROR:"):
            continue
        sig = source_static_signatures(blob)
        add_candidate("GIT_PRECAMPAIGN", srel, r["blob_sha256"], sig, r["date"], r["commit"])

    candidates.sort(key=lambda x: (-x["score"], x["kind"], x["path"]))
    write_csv(DEC / "adjudicated_source_ranking.csv", candidates)

    # ------------------------------------------------------------------
    # E. Canonical signature report + proof grade
    # ------------------------------------------------------------------
    preamble = canonical_preamble()
    write_json(SIG / "canonical_signature.json", {
        "canonical_main": str(CANON_MAIN),
        "canonical_main_exists": CANON_MAIN.exists(),
        "canonical_main_sha256": sha256_file(CANON_MAIN) if CANON_MAIN.exists() else None,
        "canonical_duration_s": EXPECTED_CANON_DURATION_S,
        "native_dt_hint_s": NATIVE_DT_HINT,
        "native_intervals_if_30hz": EXPECTED_NATIVE_INTERVALS,
        "canonical_filename": CANON_MAIN.name,
        "canonical_comment_preamble": preamble,
        "failed_replay_source_hash": FAILED_REPLAY_HASH,
    })

    top = candidates[0] if candidates else None
    proof_grade = "UNRESOLVED"
    replay_authorized = False

    if top:
        # A-grade requires a preserved pre-campaign Git source plus a direct pre-campaign
        # controller reference. Even A-grade still requires review before replay.
        if (top["kind"] == "GIT_PRECAMPAIGN"
                and top["precampaign_controller_references"] > 0
                and top["canonical_filename_literal"]
                and not top["same_hash_as_failed_replay_source"]):
            proof_grade = "A_STRONG_HISTORICAL_RUNTIME_CANDIDATE_REVIEW_REQUIRED"
        elif (top["kind"] == "GIT_PRECAMPAIGN"
              and top["canonical_filename_literal"]
              and not top["same_hash_as_failed_replay_source"]):
            proof_grade = "B_HISTORICAL_SOURCE_CANDIDATE_NO_DIRECT_CALLER_PROOF"
        elif (top["canonical_filename_literal"]
              and not top["same_hash_as_failed_replay_source"]):
            proof_grade = "C_FILESYSTEM_SOURCE_CANDIDATE_REVIEW_REQUIRED"
        else:
            proof_grade = "UNRESOLVED"

    decision = {
        "v2_root_cause_class": "SOURCE_OR_RUNTIME_VERSION_MISMATCH",
        "v3_false_positive_observed": "diagnose_task39_replay_mismatch_v1.py was incorrectly rankable as source",
        "v4_filter_rule": "Only scenario39-family source/backups/Git versions can be runtime candidates; diagnostics/replay/audit/etc are excluded.",
        "paper_gate": "BLOCK_REMAINS",
        "simulator_rerun_performed": False,
        "replay_authorized_now": replay_authorized,
        "proof_grade": proof_grade,
        "top_candidate": top,
        "filesystem_scenario39_candidates": len(fs_rows),
        "historical_scenario39_paths": len(git_paths),
        "precampaign_git_versions": len(git_rows),
        "campaign_task39_reference_lines": len(controller_rows),
        "next_rule": (
            "Do not rerun yet. Review the top adjudicated candidate, its pre-campaign Git blob if any, "
            "and the exact controller reference lines. Only after the candidate is accepted as the "
            "corrected396 Task-39 runtime source should a single instrumented P020 replay be prepared."
        )
    }
    write_json(DEC / "PROVENANCE_ADJUDICATION.json", decision)

    summary = [
        "="*94,
        "TASK39 CORRECTED396 PROVENANCE ADJUDICATION V4",
        "="*94,
        "Root cause class: SOURCE_OR_RUNTIME_VERSION_MISMATCH",
        "Paper gate: BLOCK_REMAINS",
        "Simulator rerun performed: NO",
        "Replay authorized now: NO",
        f"Filesystem scenario39-family candidates: {len(fs_rows)}",
        f"Historical scenario39 paths in pre-campaign Git: {len(git_paths)}",
        f"Pre-campaign scenario39 Git versions: {len(git_rows)}",
        f"Campaign/runner Task-39 reference lines: {len(controller_rows)}",
        f"Proof grade: {proof_grade}",
    ]
    if top:
        summary += [
            f"Top adjudicated candidate kind: {top['kind']}",
            f"Top adjudicated candidate: {top['path']}",
            f"Top candidate SHA256: {top['sha256']}",
            f"Top candidate commit: {top['historical_commit'] or 'N/A'}",
            f"Top candidate score: {top['score']}",
            f"Same hash as failed replay source: {top['same_hash_as_failed_replay_source']}",
            f"Canonical filename signature: {top['canonical_filename_literal']}",
            f"Failed-replay filename signature: {top['failed_replay_filename_literal']}",
            f"Pre-campaign controller references: {top['precampaign_controller_references']}",
            f"Top candidate reasons: {top['reasons']}",
        ]
    summary += [
        "",
        "REVIEW THESE FIRST:",
        "  05_DECISION/PROVENANCE_ADJUDICATION.json",
        "  05_DECISION/adjudicated_source_ranking.csv",
        "  03_CAMPAIGN_CALL_CHAIN/campaign_task39_direct_references.csv",
        "  02_PRECAMPAIGN_GIT/scenario39_precampaign_git_versions.csv",
        "  02_PRECAMPAIGN_GIT/latest_precampaign_source_excerpts.txt",
        "  01_TRUE_SOURCE_CANDIDATES/filesystem_scenario39_candidates.csv",
        "  01_TRUE_SOURCE_CANDIDATES/filesystem_source_excerpts.txt",
        "  04_SIGNATURE_COMPARISON/canonical_signature.json",
        "",
        "DO NOT rerun Task-39 yet."
    ]
    (OUT / "DIAGNOSTIC_SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    # Archive result.
    import zipfile
    zip_path = PROJECT / "outputs/task39_corrected396_provenance_adjudication_v4.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT.parent))
    digest = sha256_file(zip_path)
    sha_path = Path(str(zip_path) + ".sha256")
    sha_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")

    print("-"*94)
    for line in summary[3:]:
        print(line)
    print("-"*94)
    print("ZIP:", zip_path)
    print("ZIP SHA256:", digest)
    print("SHA256 sidecar:", sha_path)
    print("="*94)

if __name__ == "__main__":
    main()
