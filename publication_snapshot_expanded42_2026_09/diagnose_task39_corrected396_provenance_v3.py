#!/usr/bin/env python3
"""
TASK39 CORRECTED396 PROVENANCE RECONCILIATION V3
READ-ONLY with respect to simulator/source/canonical data.

Purpose
-------
Resolve which Task-39 source/runtime/campaign path most likely produced the
final corrected396 P020 trajectory, WITHOUT importing or executing scenario39,
MuJoCo, HumEnv, Meta Motivo, or any simulator code.

The script only:
  * reads source, manifests, CSV metadata, logs, histories, and git metadata;
  * computes hashes and static signatures;
  * writes diagnostic reports under outputs/task39_corrected396_provenance_v3.

It does NOT rerun a simulation and does NOT modify project source files.
"""
from __future__ import annotations

import ast
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT = Path(os.environ.get(
    "MUJOCO_PROJECT_ROOT",
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
)).resolve()

CANON_ROOT = PROJECT / "outputs/_highrate_overnight/campaign_highrate_truth_v3_rne_corrected396"
P020_ROOT = CANON_ROOT / "runs/P020/task_39/scenario39_age49_h1p63_sex_female_w73p0_20260826_144048"
REPLAY_ROOT = PROJECT / "outputs/task39_p020_exact_pose_replay_v2_headless"
V2_ROOT = PROJECT / "outputs/task39_replay_mismatch_diagnostic_v2"

OUT = PROJECT / "outputs/task39_corrected396_provenance_v3"
EVID = OUT / "01_EVIDENCE"
SRC = OUT / "02_SOURCE_CANDIDATES"
CAMP = OUT / "03_CAMPAIGN_CHAIN"
MANI = OUT / "04_MANIFEST_CENSUS"
HIST = OUT / "05_HISTORY_GIT"
RANK = OUT / "06_RANKING"

CANON_MAIN = P020_ROOT / "fall_scenario39_age49_20260826_144048.csv"
CANON_HR = P020_ROOT / "fall_scenario39_age49_20260826_144048_highrate_truth.csv"
CANON_MANIFEST = P020_ROOT / "run_manifest.json"
CANON_VALIDATION = P020_ROOT / "fall_scenario39_age49_20260826_144048_validation.txt"

CURRENT_REPLAY_SOURCE_HASH = "873171f5879da95233e9094cd655f214b1538ac0031fd3ee8e64299f8eef1fb8"
CAMPAIGN_DATE = dt.datetime(2026, 8, 26, 23, 59, 59)

KEY_TERMS = [
    "campaign_highrate_truth_v3_rne_corrected396",
    "campaign_highrate_truth_v3_rne",
    "corrected396",
    "rne_corrected",
    "_highrate_overnight",
    "scenario39",
    "task_39",
    "high_rate_imu_sidecar",
    "mj_rnePostConstraint",
    "run_with_subject",
    "fall_scenario39",
    "fall_forward_height",
]

TEXT_EXTS = {
    ".py", ".sh", ".bash", ".zsh", ".txt", ".log", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".md", ".csv", ".out", ".err"
}

SKIP_DIRS = {
    ".venv", "__pycache__", ".git/objects", "node_modules",
    "task39_p020_exact_pose_replay_v2_headless",
    "task39_replay_mismatch_diagnostic_v2",
    "task39_corrected396_provenance_v3",
}

MAX_SCAN_BYTES = 4_000_000
MAX_CONTEXT_HITS_PER_FILE = 40


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fields is None:
        fields = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def safe_text(p: Path) -> str:
    try:
        if p.stat().st_size > MAX_SCAN_BYTES:
            return ""
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_json(v, key))
    elif isinstance(obj, list):
        if len(obj) <= 20 and all(not isinstance(x, (dict, list)) for x in obj):
            out[prefix] = obj
        else:
            out[prefix] = f"<list:{len(obj)}>"
    else:
        out[prefix] = obj
    return out


def comment_preamble(p: Path, max_lines=200) -> list[str]:
    out = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            if line.startswith("#"):
                out.append(line.rstrip("\r\n"))
            else:
                break
    return out


def first_csv_header(p: Path) -> list[str]:
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            return next(csv.reader([line]))
    return []


def run_git(args: list[str]) -> str:
    try:
        cp = subprocess.run(
            ["git", "-C", str(PROJECT)] + args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=25, check=False
        )
        return cp.stdout
    except Exception as e:
        return f"ERROR: {e}\n"


def iter_project_files_without_outputs() -> Iterable[Path]:
    """Efficiently walk source tree while pruning large/generated trees."""
    prune_names = {".venv", ".git", "__pycache__", "node_modules", "outputs", ".cache"}
    for root, dirs, files in os.walk(PROJECT):
        dirs[:] = [d for d in dirs if d not in prune_names]
        rp = Path(root)
        for fn in files:
            yield rp / fn


def iter_targeted_output_files() -> Iterable[Path]:
    """Inspect only launcher/log/provenance layers in outputs; never descend into campaign runs."""
    roots = [
        PROJECT / "outputs/_highrate_overnight",
        PROJECT / "outputs",
    ]
    seen = set()
    for base in roots:
        if not base.exists():
            continue
        for root, dirs, files in os.walk(base):
            r = Path(root)
            try:
                relparts = r.relative_to(PROJECT / "outputs").parts
            except Exception:
                relparts = ()
            # Prevent duplicate/deep scans and never traverse per-run generated data.
            dirs[:] = [d for d in dirs if d not in {"runs", "__pycache__", "task39_corrected396_provenance_v3"}]
            if len(relparts) > 4:
                dirs[:] = []
            for fn in files:
                p = r / fn
                try:
                    key = p.resolve()
                except Exception:
                    key = p
                if key not in seen:
                    seen.add(key)
                    yield p


def source_candidates() -> list[Path]:
    name_re = re.compile(r"(?:scenario[_]?39|task[_]?39)", re.I)
    found = set()
    for p in iter_project_files_without_outputs():
        if p.suffix.lower() != ".py" or not name_re.search(p.name):
            continue
        try:
            if p.stat().st_size <= MAX_SCAN_BYTES:
                found.add(p.resolve())
        except Exception:
            pass

    # Include exact replay-copied source as comparison evidence only.
    copied = REPLAY_ROOT / "04_AUDIT/scenario39_legacy_SOURCE_USED.py"
    if copied.exists():
        found.add(copied.resolve())
    return sorted(found)


def ast_static_info(p: Path, txt: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "functions": [],
        "classes": [],
        "imports": [],
        "step_assignments": [],
        "seed_assignments": [],
        "version_strings": [],
    }
    try:
        tree = ast.parse(txt, filename=str(p))
    except Exception as e:
        out["ast_error"] = str(e)
        return out

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out["functions"].append(node.name)
        elif isinstance(node, ast.ClassDef):
            out["classes"].append(node.name)
        elif isinstance(node, ast.Import):
            out["imports"].extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            out["imports"].append(node.module or "")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            value = None
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            else:
                targets = [node.target]
                value = node.value
            names = []
            for t in targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
            if value is not None:
                for name in names:
                    lname = name.lower()
                    try:
                        lit = ast.literal_eval(value)
                    except Exception:
                        lit = None
                    if "step" in lname and isinstance(lit, (int, float)):
                        out["step_assignments"].append({"name": name, "value": lit, "line": getattr(node, "lineno", None)})
                    if "seed" in lname and isinstance(lit, (int, float, str)):
                        out["seed_assignments"].append({"name": name, "value": lit, "line": getattr(node, "lineno", None)})
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            low = s.lower()
            if ("scenario 39" in low or "v30" in low or "forward fall" in low or
                "postfall" in low or "height" in low and len(s) < 250):
                out["version_strings"].append(s[:500])

    for k in ("functions", "classes", "imports", "version_strings"):
        out[k] = sorted(set(out[k]))
    return out


def text_signatures(txt: str) -> dict[str, Any]:
    low = txt.lower()
    sig = {}
    for term in KEY_TERMS:
        sig[term] = term.lower() in low
    # Output naming signatures are especially discriminative here.
    sig["literal_fall_scenario39"] = "fall_scenario39" in low
    sig["literal_fall_forward_height"] = "fall_forward_height" in low
    sig["has_run_with_subject"] = re.search(r"\bdef\s+run_with_subject\s*\(", txt) is not None
    sig["has_marker_exporter"] = "MarkerKinematicsExporter" in txt
    sig["has_contact_analyzer"] = "DynamicsContactAnalyzer" in txt
    sig["has_paper_alignment"] = "PaperAlignmentExporter" in txt
    sig["has_height_layer"] = "Task39HeightLayer" in txt
    sig["has_bio_controller"] = "EnhancedBiofidelicController" in txt
    sig["has_rne_call"] = "mj_rnePostConstraint" in txt
    sig["mentions_30hz"] = bool(re.search(r"\b30(?:\.0+)?\s*(?:hz|fps)\b", low))
    sig["mentions_100hz"] = bool(re.search(r"\b100(?:\.0+)?\s*(?:hz|fps)\b", low))
    return sig


def relevant_source_lines(txt: str, max_lines=350) -> str:
    terms = [
        "fall_scenario39", "fall_forward_height", "run_with_subject",
        "stand_steps", "step_steps", "walk_steps", "perturb", "react", "fall_steps",
        "total_steps", "seed", "Task39HeightLayer", "EnhancedBiofidelicController",
        "MarkerKinematicsExporter", "DynamicsContactAnalyzer", "PaperAlignmentExporter",
        "mj_rnePostConstraint", "high_rate_imu_sidecar", "output", "Scenario 39",
        "v30-postfall", "platform", "height"
    ]
    lines = txt.splitlines()
    selected = []
    seen = set()
    for i, line in enumerate(lines, 1):
        if any(t.lower() in line.lower() for t in terms):
            for j in range(max(1, i-2), min(len(lines), i+2)+1):
                if j not in seen:
                    seen.add(j)
                    selected.append((j, lines[j-1]))
                    if len(selected) >= max_lines:
                        break
        if len(selected) >= max_lines:
            break
    return "\n".join(f"{n:6d}: {line}" for n, line in selected)


def likely_campaign_file(p: Path) -> bool:
    name = p.name.lower()
    rel = str(p).lower()
    if p.suffix.lower() not in TEXT_EXTS:
        return False
    if any(x in rel for x in ["/.venv/", "/__pycache__/", "/.git/objects/"]):
        return False
    # Avoid scanning all result CSVs; manifests are handled separately.
    if str(CANON_ROOT).lower() in rel:
        return p.name in {"run_manifest.json"} or "manifest" in name or p.suffix.lower() in {".log", ".txt", ".json"}
    return any(k in name for k in [
        "campaign", "overnight", "highrate", "high_rate", "scenario39",
        "task39", "task_39", "run_all", "runner", "batch", "launch", "resume"
    ])


def scan_campaign_chain() -> list[dict[str, Any]]:
    rows = []
    candidates = list(iter_project_files_without_outputs()) + list(iter_targeted_output_files())
    seen = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in seen:
            continue
        seen.add(rp)
        if not p.is_file() or not likely_campaign_file(p):
            continue
        txt = safe_text(p)
        if not txt:
            continue
        low = txt.lower()
        hits = [t for t in KEY_TERMS if t.lower() in low]
        if not hits:
            continue
        st = p.stat()
        try:
            srel = str(p.relative_to(PROJECT))
        except Exception:
            srel = str(p)
        rows.append({
            "path": str(p),
            "relative_path": srel,
            "sha256": sha256_file(p),
            "size": st.st_size,
            "mtime_iso": dt.datetime.fromtimestamp(st.st_mtime).isoformat(),
            "terms": ";".join(hits),
            "mentions_exact_corrected_root": "campaign_highrate_truth_v3_rne_corrected396" in low,
            "mentions_scenario39": "scenario39" in low,
            "mentions_run_with_subject": "run_with_subject" in low,
            "mentions_rne": "mj_rnepostconstraint" in low,
            "mentions_fall_scenario39": "fall_scenario39" in low,
            "mentions_fall_forward_height": "fall_forward_height" in low,
        })
    return rows


def context_excerpt(p: Path, terms: Iterable[str], radius=4) -> str:
    txt = safe_text(p)
    if not txt:
        return ""
    lines = txt.splitlines()
    inds = []
    for i, line in enumerate(lines):
        if any(t.lower() in line.lower() for t in terms):
            inds.append(i)
    out = []
    used = set()
    for i in inds[:MAX_CONTEXT_HITS_PER_FILE]:
        a, b = max(0, i-radius), min(len(lines), i+radius+1)
        for j in range(a, b):
            if j not in used:
                used.add(j)
                out.append(f"{j+1:6d}: {lines[j]}")
        out.append("------")
    return "\n".join(out)


def manifest_census():
    manifests = sorted(CANON_ROOT.glob("runs/P*/task_*/**/run_manifest.json"))
    key_counts = Counter()
    key_values = defaultdict(Counter)
    sourceish_rows = []
    task39_rows = []
    for p in manifests:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        flat = flatten_json(obj)
        for k, v in flat.items():
            key_counts[k] += 1
            sv = json.dumps(v, sort_keys=True, default=str) if not isinstance(v, str) else v
            if len(sv) < 500:
                key_values[k][sv] += 1
            lk = k.lower()
            if any(token in lk for token in ["source", "script", "version", "seed", "commit", "git", "command", "python", "device", "model", "scenario", "task"]):
                sourceish_rows.append({
                    "manifest": str(p),
                    "key": k,
                    "value": sv[:1000],
                })
        if "/task_39/" in str(p):
            task39_rows.append({
                "manifest": str(p),
                "sha256": sha256_file(p),
                **{k: v for k, v in flat.items()
                   if any(t in k.lower() for t in ["source", "script", "version", "seed", "commit", "git", "command", "python", "device", "model", "scenario", "task", "duration"])}
            })
    summary = []
    for k, cnt in key_counts.most_common():
        vals = key_values[k].most_common(8)
        summary.append({
            "key": k, "count": cnt,
            "top_values": " | ".join(f"{v} ({n})" for v, n in vals)
        })
    return manifests, summary, sourceish_rows, task39_rows


def parse_zsh_history(p: Path) -> list[dict[str, Any]]:
    rows = []
    pat = re.compile(r"^:\s+(\d+):\d+;(.*)$")
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for idx, raw in enumerate(f, 1):
                line = raw.rstrip("\r\n")
                m = pat.match(line)
                ts = None
                cmd = line
                if m:
                    try:
                        ts = dt.datetime.fromtimestamp(int(m.group(1))).isoformat()
                    except Exception:
                        ts = None
                    cmd = m.group(2)
                low = cmd.lower()
                if any(t.lower() in low for t in KEY_TERMS):
                    rows.append({"history": str(p), "line": idx, "timestamp": ts, "command": cmd})
    except Exception:
        pass
    return rows


def histories() -> list[Path]:
    cands = [
        Path.home()/".zsh_history",
        Path.home()/".bash_history",
        PROJECT/".zsh_history",
        PROJECT/"nohup.out",
    ]
    # Also likely log files near project root.
    for pat in ["*.log", "*.out", "*.err", "*history*", "*nohup*"]:
        cands.extend(PROJECT.glob(pat))
    out = []
    seen = set()
    for p in cands:
        if p.exists() and p.is_file():
            rp = p.resolve()
            if rp not in seen and rp.stat().st_size <= 100_000_000:
                seen.add(rp)
                out.append(rp)
    return out


def source_git_history(p: Path) -> str:
    try:
        rel = p.resolve().relative_to(PROJECT)
    except Exception:
        return ""
    return run_git(["log", "--all", "--follow", "--date=iso", "--format=%H|%ad|%an|%s", "--", str(rel)])


def main():
    if not PROJECT.exists():
        raise SystemExit(f"ERROR: project root not found: {PROJECT}")
    for d in [OUT, EVID, SRC, CAMP, MANI, HIST, RANK]:
        d.mkdir(parents=True, exist_ok=True)

    print("="*86)
    print("TASK39 CORRECTED396 PROVENANCE RECONCILIATION V3 — STATIC/READ-ONLY")
    print("="*86)
    print("Project:", PROJECT)
    print("Canonical root:", CANON_ROOT)
    print("NO simulator/scenario imports. NO MuJoCo execution.")

    # 0. Carry forward the decisive V2 reports if the directory still exists.
    v2_copy = EVID / "V2_CARRY_FORWARD"
    v2_copy.mkdir(parents=True, exist_ok=True)
    for rel in [
        "ROOT_CAUSE_CLASSIFICATION.json",
        "DIAGNOSTIC_SUMMARY.txt",
        "00_SIGNAL_DIAG/identity.json",
        "00_SIGNAL_DIAG/event_timing.json",
        "00_SIGNAL_DIAG/initial_final_coordinates.json",
        "00_SIGNAL_DIAG/channel_mismatch_metrics.csv",
        "00_SIGNAL_DIAG/time_shift_search.json",
        "02_SOURCE_VERSIONS/scenario39_source_versions.csv",
        "02_SOURCE_VERSIONS/scenario39_source_excerpts.txt",
        "03_CAMPAIGN_PROVENANCE/campaign_code_hits.csv",
        "03_CAMPAIGN_PROVENANCE/campaign_code_excerpts.txt",
    ]:
        srcp = V2_ROOT / rel
        if srcp.exists() and srcp.is_file():
            dstp = v2_copy / rel
            dstp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(srcp, dstp)

    # 1. Canonical identity / metadata
    identity = {
        "project": str(PROJECT),
        "canonical_root": str(CANON_ROOT),
        "p020_root": str(P020_ROOT),
        "canonical_main": str(CANON_MAIN),
        "canonical_main_exists": CANON_MAIN.exists(),
        "canonical_main_sha256": sha256_file(CANON_MAIN) if CANON_MAIN.exists() else None,
        "canonical_main_header": first_csv_header(CANON_MAIN),
        "canonical_main_comments": comment_preamble(CANON_MAIN),
        "canonical_highrate": str(CANON_HR),
        "canonical_highrate_exists": CANON_HR.exists(),
        "canonical_highrate_sha256": sha256_file(CANON_HR) if CANON_HR.exists() else None,
        "canonical_highrate_header": first_csv_header(CANON_HR),
        "canonical_highrate_comments": comment_preamble(CANON_HR),
        "canonical_manifest": str(CANON_MANIFEST),
        "canonical_manifest_sha256": sha256_file(CANON_MANIFEST) if CANON_MANIFEST.exists() else None,
        "current_replay_source_hash_from_prior_audit": CURRENT_REPLAY_SOURCE_HASH,
        "v2_diagnostic_root_exists": V2_ROOT.exists(),
    }
    if CANON_MANIFEST.exists():
        try:
            identity["canonical_manifest_flat"] = flatten_json(json.loads(CANON_MANIFEST.read_text(encoding="utf-8")))
        except Exception as e:
            identity["canonical_manifest_error"] = str(e)
    if CANON_VALIDATION.exists():
        identity["canonical_validation_text"] = safe_text(CANON_VALIDATION)
    write_json(EVID/"canonical_p020_identity.json", identity)

    # 2. Source candidates
    src_rows = []
    src_excerpts = []
    candidates = source_candidates()
    for p in candidates:
        txt = safe_text(p)
        st = p.stat()
        h = sha256_file(p)
        static = ast_static_info(p, txt)
        sig = text_signatures(txt)
        try:
            rel = str(p.relative_to(PROJECT))
        except Exception:
            rel = str(p)
        is_replay_copy = str(REPLAY_ROOT) in str(p)
        mtime = dt.datetime.fromtimestamp(st.st_mtime)
        row = {
            "path": str(p),
            "relative_path": rel,
            "sha256": h,
            "size": st.st_size,
            "mtime_iso": mtime.isoformat(),
            "mtime_before_or_on_campaign_date": mtime <= CAMPAIGN_DATE,
            "is_current_replay_source_hash": h == CURRENT_REPLAY_SOURCE_HASH,
            "is_replay_audit_copy": is_replay_copy,
            "functions": ";".join(static.get("functions", [])),
            "classes": ";".join(static.get("classes", [])),
            "step_assignments": json.dumps(static.get("step_assignments", [])),
            "seed_assignments": json.dumps(static.get("seed_assignments", [])),
            "version_strings": json.dumps(static.get("version_strings", [])),
            **sig,
        }
        src_rows.append(row)
        src_excerpts.append(
            f"\n{'='*100}\nSOURCE: {p}\nSHA256: {h}\n"
            f"MTIME: {mtime.isoformat()}\nCURRENT_REPLAY_HASH: {h == CURRENT_REPLAY_SOURCE_HASH}\n"
            f"{'='*100}\n{relevant_source_lines(txt)}\n"
        )
        gh = source_git_history(p)
        if gh:
            (HIST / f"git_history_{hashlib.sha256(str(p).encode()).hexdigest()[:12]}.txt").write_text(
                f"SOURCE: {p}\n\n{gh}", encoding="utf-8"
            )
    write_csv(SRC/"scenario39_source_candidates.csv", src_rows)
    (SRC/"scenario39_source_key_excerpts.txt").write_text("".join(src_excerpts), encoding="utf-8")

    # 3. Campaign chain
    camp_rows = scan_campaign_chain()
    write_csv(CAMP/"campaign_chain_candidates.csv", camp_rows)
    exact = [r for r in camp_rows if r.get("mentions_exact_corrected_root")]
    excerpts = []
    for r in sorted(exact, key=lambda x: x["relative_path"]):
        p = Path(r["path"])
        excerpts.append(
            f"\n{'='*100}\nFILE: {p}\nSHA256: {r['sha256']}\n{'='*100}\n"
            + context_excerpt(p, KEY_TERMS)
            + "\n"
        )
    (CAMP/"exact_corrected396_contexts.txt").write_text("".join(excerpts), encoding="utf-8")

    # 4. Run manifest census across 396
    manifests, summary, sourceish, task39 = manifest_census()
    write_csv(MANI/"manifest_key_census.csv", summary)
    write_csv(MANI/"manifest_source_runtime_fields.csv", sourceish)
    write_csv(MANI/"task39_manifest_runtime_fields.csv", task39)
    write_json(MANI/"manifest_count.json", {
        "manifests_found": len(manifests),
        "task39_manifests_found": len(task39),
        "expected_total": 396,
        "expected_task39": 22,
    })

    # 5. Shell/log history + git metadata
    history_rows = []
    for hp in histories():
        history_rows.extend(parse_zsh_history(hp))
        if hp.suffix.lower() in {".log", ".out", ".err", ".txt"} or hp.name == "nohup.out":
            txt = safe_text(hp)
            for idx, line in enumerate(txt.splitlines(), 1):
                low = line.lower()
                if any(t.lower() in low for t in KEY_TERMS):
                    history_rows.append({
                        "history": str(hp), "line": idx, "timestamp": None, "command": line
                    })
    # Deduplicate
    uniq = []
    seen = set()
    for r in history_rows:
        key = (r.get("history"), r.get("line"), r.get("command"))
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    write_csv(HIST/"campaign_history_hits.csv", uniq)
    (HIST/"git_status.txt").write_text(run_git(["status", "--short", "--branch"]), encoding="utf-8")
    (HIST/"git_log_august_2026.txt").write_text(
        run_git(["log", "--all", "--since=2026-08-01", "--until=2026-09-02",
                 "--date=iso", "--format=%H|%ad|%an|%s", "--name-status"]),
        encoding="utf-8"
    )
    (HIST/"git_reflog_august_2026.txt").write_text(
        run_git(["reflog", "--all", "--date=iso", "--since=2026-08-01", "--until=2026-09-02"]),
        encoding="utf-8"
    )

    # 6. Evidence ranking. This ranks candidates for REVIEW; it is not an equivalence gate.
    ref_texts = []
    for r in exact:
        ref_texts.append(safe_text(Path(r["path"])))
    history_blob = "\n".join(r.get("command", "") for r in uniq)
    corrected_context_blob = "\n".join(ref_texts) + "\n" + history_blob

    rank_rows = []
    for r in src_rows:
        if r.get("is_replay_audit_copy"):
            continue
        score = 0
        reasons = []
        rel = r["relative_path"]
        name = Path(r["path"]).name
        lowctx = corrected_context_blob.lower()

        if name.lower() in lowctx:
            score += 35
            reasons.append("candidate filename appears in corrected396 campaign/history evidence")
        if rel.lower() in lowctx:
            score += 45
            reasons.append("candidate relative path appears in corrected396 campaign/history evidence")
        if r.get("literal_fall_scenario39"):
            score += 18
            reasons.append("contains canonical fall_scenario39 export signature")
        if r.get("literal_fall_forward_height"):
            score -= 8
            reasons.append("contains replay-style fall_forward_height export signature")
        if r.get("has_run_with_subject"):
            score += 8
            reasons.append("defines run_with_subject")
        if r.get("has_rne_call"):
            score += 8
            reasons.append("contains mj_rnePostConstraint")
        if r.get("mtime_before_or_on_campaign_date"):
            score += 5
            reasons.append("filesystem mtime is not later than corrected396 campaign date")
        if r.get("is_current_replay_source_hash"):
            score -= 10
            reasons.append("same hash as already-failed standalone replay source")

        rank_rows.append({
            "score": score,
            "path": r["path"],
            "sha256": r["sha256"],
            "is_current_replay_source_hash": r["is_current_replay_source_hash"],
            "canonical_export_signature": r["literal_fall_scenario39"],
            "replay_export_signature": r["literal_fall_forward_height"],
            "reasons": " | ".join(reasons),
        })
    rank_rows.sort(key=lambda x: (-x["score"], x["path"]))
    write_csv(RANK/"source_candidate_ranking.csv", rank_rows)

    # 7. Decision summary
    top = rank_rows[0] if rank_rows else None
    direct_ref_hits = []
    if top:
        top_name = Path(top["path"]).name.lower()
        top_rel = str(Path(top["path"]).relative_to(PROJECT)).lower() if Path(top["path"]).is_relative_to(PROJECT) else ""
        for r in exact:
            txt = safe_text(Path(r["path"])).lower()
            if top_name in txt or (top_rel and top_rel in txt):
                direct_ref_hits.append(r["path"])
        for r in uniq:
            cmd = r.get("command", "").lower()
            if top_name in cmd or (top_rel and top_rel in cmd):
                direct_ref_hits.append(r.get("history"))

    proof_level = "UNRESOLVED"
    if top and direct_ref_hits:
        proof_level = "DIRECT_REFERENCE_FOUND_REVIEW_REQUIRED"
    elif top and top["score"] >= 45:
        proof_level = "STRONG_STATIC_CANDIDATE_REVIEW_REQUIRED"
    elif top:
        proof_level = "CANDIDATES_RANKED_BUT_NOT_PROVEN"

    decision = {
        "root_cause_class_from_v2": "SOURCE_OR_RUNTIME_VERSION_MISMATCH",
        "paper_gate": "BLOCK_REMAINS",
        "simulator_rerun_performed": False,
        "proof_level": proof_level,
        "top_candidate": top,
        "direct_reference_evidence_files": sorted(set(x for x in direct_ref_hits if x)),
        "source_candidates_found": len([r for r in src_rows if not r.get("is_replay_audit_copy")]),
        "campaign_chain_files_with_hits": len(camp_rows),
        "campaign_chain_files_with_exact_corrected_root": len(exact),
        "manifest_count": len(manifests),
        "task39_manifest_count": len(task39),
        "history_hits": len(uniq),
        "next_rule": (
            "DO NOT rerun Task-39 yet. Review the top candidate against direct campaign/history "
            "evidence and static timing/export signatures. Only after the original corrected396 "
            "entrypoint/source path is proven should one instrumented P020 replay be prepared."
        )
    }
    write_json(OUT/"ROOT_PROVENANCE_DECISION.json", decision)

    summary_lines = [
        "="*86,
        "TASK39 CORRECTED396 PROVENANCE RECONCILIATION V3",
        "="*86,
        f"Root cause class: SOURCE_OR_RUNTIME_VERSION_MISMATCH",
        f"Paper gate: BLOCK_REMAINS",
        f"Simulator rerun performed: NO",
        f"Scenario/Task39 source candidates: {decision['source_candidates_found']}",
        f"Campaign chain files with relevant hits: {len(camp_rows)}",
        f"Files explicitly mentioning corrected396 root: {len(exact)}",
        f"Corrected396 manifests found: {len(manifests)} (expected 396)",
        f"Task-39 manifests found: {len(task39)} (expected 22)",
        f"Shell/log history hits: {len(uniq)}",
        f"Provenance proof level: {proof_level}",
    ]
    if top:
        summary_lines += [
            f"Top static candidate: {top['path']}",
            f"Top candidate SHA256: {top['sha256']}",
            f"Top candidate score: {top['score']}",
            f"Top candidate reasons: {top['reasons']}",
        ]
    summary_lines += [
        "",
        "REVIEW THESE FIRST:",
        "  ROOT_PROVENANCE_DECISION.json",
        "  06_RANKING/source_candidate_ranking.csv",
        "  03_CAMPAIGN_CHAIN/exact_corrected396_contexts.txt",
        "  05_HISTORY_GIT/campaign_history_hits.csv",
        "  04_MANIFEST_CENSUS/task39_manifest_runtime_fields.csv",
        "  02_SOURCE_CANDIDATES/scenario39_source_key_excerpts.txt",
        "",
        "DO NOT rerun Task-39 until this provenance evidence is reviewed.",
    ]
    (OUT/"DIAGNOSTIC_SUMMARY.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # 8. Make output ZIP + hash
    zip_path = PROJECT / "outputs/task39_corrected396_provenance_v3.zip"
    if zip_path.exists():
        zip_path.unlink()
    import zipfile
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT.parent))
    digest = sha256_file(zip_path)
    sha_path = Path(str(zip_path) + ".sha256")
    sha_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")

    print("-"*86)
    for line in summary_lines[3:]:
        print(line)
    print("-"*86)
    print("ZIP:", zip_path)
    print("ZIP SHA256:", digest)
    print("SHA256 sidecar:", sha_path)
    print("="*86)


if __name__ == "__main__":
    main()
