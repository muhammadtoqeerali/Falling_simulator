#!/usr/bin/env python3
"""
TASK39 CORRECTED396 RUNTIME-RESIDUE FORENSICS V6
================================================

Read-only/static follow-up to V5.

V5 established:
  * source/runtime mismatch remains;
  * pre-campaign scenario39_legacy.py differs from failed replay source;
  * two filesystem-era controller references existed;
  * Git could not reconstruct those controller versions;
  * historical scenario source itself does not contain mj_rnePostConstraint.

V6 therefore searches the historical *runtime residue* that source/Git-only
analysis can miss:
  1) project launchers/configs/logs/backups around 2026-08-26;
  2) Python .pyc bytecode metadata/constants, decoded WITHOUT execution;
  3) git reflog/stashes/dangling commits;
  4) common editor local-history locations for scenario39/campaign files;
  5) corrected396 parent-level provenance files (never campaign run data);
  6) Task-39 canonical duration distribution as a timing fingerprint.

NO scenario/MuJoCo/HumEnv/Meta Motivo import.
NO bytecode execution.
NO simulator rerun.
NO source modification.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import marshal
import os
import re
import subprocess
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT = Path(os.environ.get(
    "MUJOCO_PROJECT_ROOT",
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
)).resolve()
HOME = Path.home()

CANON_ROOT = PROJECT / "outputs/_highrate_overnight/campaign_highrate_truth_v3_rne_corrected396"
V4 = PROJECT / "outputs/task39_corrected396_provenance_adjudication_v4"
V5 = PROJECT / "outputs/task39_corrected396_callchain_trace_v5"

OUT = PROJECT / "outputs/task39_corrected396_runtime_residue_forensics_v6"
RES = OUT / "01_RUNTIME_RESIDUE"
PYC = OUT / "02_BYTECODE"
GIT = OUT / "03_GIT_RECOVERY"
EDH = OUT / "04_EDITOR_HISTORY"
TIM = OUT / "05_TIMING_FINGERPRINT"
DEC = OUT / "06_DECISION"

CAMPAIGN_START = dt.datetime(2026, 8, 24, 0, 0, 0)
CAMPAIGN_END = dt.datetime(2026, 8, 28, 23, 59, 59)
FAILED_REPLAY_HASH = "873171f5879da95233e9094cd655f214b1538ac0031fd3ee8e64299f8eef1fb8"
V4_SOURCE_HASH = "ae5180d9afdc1a76b3b7bcc82e8d3b6f4fb65c005688734f96bdfd583d81f399"
V4_SOURCE_COMMIT = "db9a7b67163edcf58d64d66afd4ed3271b34899b"

TERMS = [
    "scenario39_legacy",
    "scenario39",
    "task_39",
    "task39",
    "fall_scenario39",
    "fall_forward_height",
    "campaign_highrate_truth_v3_rne_corrected396",
    "campaign_highrate_truth_v3_rne",
    "corrected396",
    "rne_corrected",
    "_highrate_overnight",
    "high_rate_imu_sidecar",
    "mj_rnePostConstraint",
    "mj_objectAcceleration",
    "run_with_subject",
]

LIKELY_NAME_TOKENS = [
    "campaign", "highrate", "high_rate", "overnight", "runner", "run_all", "batch",
    "launch", "resume", "scenario39", "task39", "task_39", "sidecar", "rne"
]

TEXT_EXTS = {
    ".py", ".sh", ".bash", ".zsh", ".txt", ".log", ".json", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".md", ".out", ".err", ".bak", ".old", ".orig",
    ".backup", ".save", ".copy"
}

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def write_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

def write_csv(p: Path, rows: list[dict[str, Any]], fields=None):
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

def safe_text(p: Path, max_bytes=12_000_000) -> str:
    try:
        if p.stat().st_size > max_bytes:
            return ""
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

def is_campaign_window(p: Path) -> bool:
    try:
        mt = dt.datetime.fromtimestamp(p.stat().st_mtime)
        return CAMPAIGN_START <= mt <= CAMPAIGN_END
    except Exception:
        return False

def run_git(args: list[str], timeout=50) -> str:
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

def term_hits(txt: str) -> list[str]:
    low = txt.lower()
    return [t for t in TERMS if t.lower() in low]

def context(txt: str, terms: list[str] = TERMS, radius=5, max_hits=60) -> str:
    lines = txt.splitlines()
    out, used = [], set()
    n_hits = 0
    for i, line in enumerate(lines):
        if any(t.lower() in line.lower() for t in terms):
            n_hits += 1
            if n_hits > max_hits:
                break
            for j in range(max(0, i-radius), min(len(lines), i+radius+1)):
                if j not in used:
                    used.add(j)
                    out.append((j+1, lines[j]))
            out.append((None, "------"))
    return "\n".join("------" if n is None else f"{n:6d}: {s}" for n, s in out)

def iter_project_runtime_candidates() -> Iterable[Path]:
    """
    Walk project source tree, pruning huge/generated trees. Inspect only parent
    layers of outputs, never the 396 run directories.
    """
    prune_names = {".venv", ".git", "node_modules", ".cache"}
    for root, dirs, files in os.walk(PROJECT):
        r = Path(root)
        try:
            rel = r.relative_to(PROJECT)
        except Exception:
            rel = Path()
        parts = rel.parts

        # Completely prune per-run generated campaign data.
        if "runs" in parts:
            dirs[:] = []
            continue

        # Keep __pycache__ because V6 specifically needs historical bytecode.
        dirs[:] = [d for d in dirs if d not in prune_names]

        # Other output diagnostics are noisy; only parent-level files under outputs.
        if parts and parts[0] == "outputs":
            if "task39_corrected396_runtime_residue_forensics_v6" in parts:
                dirs[:] = []
                continue
            if len(parts) > 4:
                dirs[:] = []

        for fn in files:
            yield r / fn

def likely_runtime_file(p: Path) -> bool:
    name = p.name.lower()
    if p.suffix.lower() == ".pyc":
        return any(k in name for k in ["scenario39", "campaign", "highrate", "runner", "sidecar", "run"])
    if p.suffix.lower() not in TEXT_EXTS:
        return False
    return any(k in name for k in LIKELY_NAME_TOKENS) or is_campaign_window(p)

def scan_runtime_residue():
    rows = []
    excerpts = []
    for p in iter_project_runtime_candidates():
        if not p.is_file() or p.suffix.lower() == ".pyc":
            continue
        if not likely_runtime_file(p):
            continue
        txt = safe_text(p)
        if not txt:
            continue
        hits = term_hits(txt)
        if not hits:
            continue
        try:
            rel = str(p.relative_to(PROJECT))
        except Exception:
            rel = str(p)
        st = p.stat()
        mt = dt.datetime.fromtimestamp(st.st_mtime)
        row = {
            "path": str(p),
            "relative_path": rel,
            "sha256": sha256_file(p),
            "size": st.st_size,
            "mtime_iso": mt.isoformat(),
            "mtime_in_campaign_window": CAMPAIGN_START <= mt <= CAMPAIGN_END,
            "git_tracked_now": bool(run_git(["ls-files", "--error-unmatch", rel]).strip() and
                                    not run_git(["ls-files", "--error-unmatch", rel]).startswith("error:")),
            "hits": ";".join(hits),
            "mentions_exact_corrected396": "campaign_highrate_truth_v3_rne_corrected396" in txt,
            "mentions_scenario39_legacy": "scenario39_legacy" in txt.lower(),
            "mentions_fall_scenario39": "fall_scenario39" in txt.lower(),
            "mentions_fall_forward_height": "fall_forward_height" in txt.lower(),
            "mentions_rne": "mj_rnepostconstraint" in txt.lower(),
            "mentions_object_accel": "mj_objectacceleration" in txt.lower(),
            "mentions_sidecar": "high_rate_imu_sidecar" in txt.lower(),
            "mentions_run_with_subject": "run_with_subject" in txt.lower(),
        }
        rows.append(row)
        if row["mtime_in_campaign_window"] or row["mentions_exact_corrected396"] or row["mentions_scenario39_legacy"]:
            excerpts.append(
                f"\n{'='*112}\nFILE: {p}\nSHA256: {row['sha256']}\n"
                f"MTIME: {row['mtime_iso']}\nHITS: {row['hits']}\n{'='*112}\n"
                + context(txt) + "\n"
            )
    rows.sort(key=lambda r: (
        not r["mtime_in_campaign_window"],
        not r["mentions_exact_corrected396"],
        r["relative_path"]
    ))
    write_csv(RES / "runtime_residue_candidates.csv", rows)
    (RES / "runtime_residue_excerpts.txt").write_text("".join(excerpts), encoding="utf-8")
    return rows

# ------------------------- safe PYC decoding -------------------------

def walk_code(co: types.CodeType):
    yield co
    for c in co.co_consts:
        if isinstance(c, types.CodeType):
            yield from walk_code(c)

def pyc_payload(p: Path) -> bytes:
    b = p.read_bytes()
    # Modern CPython pyc header is 16 bytes (PEP 552). Try 16 first, then 12.
    for off in (16, 12):
        try:
            obj = marshal.loads(b[off:])
            if isinstance(obj, types.CodeType):
                return b[off:]
        except Exception:
            pass
    raise ValueError("could not decode pyc code object with 16/12-byte header")

def decode_pyc(p: Path) -> dict[str, Any]:
    payload = pyc_payload(p)
    co = marshal.loads(payload)
    strings = []
    names = set()
    filenames = set()
    funcs = []
    for sub in walk_code(co):
        filenames.add(sub.co_filename)
        names.update(sub.co_names)
        funcs.append(sub.co_name)
        for c in sub.co_consts:
            if isinstance(c, str):
                if any(t.lower() in c.lower() for t in TERMS):
                    strings.append(c[:4000])
    joined = "\n".join(strings) + "\n" + "\n".join(sorted(names))
    return {
        "co_filenames": sorted(filenames),
        "co_names": sorted(names),
        "code_names": sorted(set(funcs)),
        "relevant_string_constants": strings,
        "hits": term_hits(joined),
    }

def scan_pyc():
    rows = []
    detail = []
    for p in iter_project_runtime_candidates():
        if not p.is_file() or p.suffix.lower() != ".pyc":
            continue
        name = p.name.lower()
        if not any(k in name for k in ["scenario39", "campaign", "highrate", "runner", "sidecar", "run"]):
            continue
        try:
            info = decode_pyc(p)
        except Exception as e:
            continue
        if not info["hits"] and not any("scenario39" in x.lower() for x in info["co_filenames"]):
            continue
        st = p.stat()
        mt = dt.datetime.fromtimestamp(st.st_mtime)
        try:
            rel = str(p.relative_to(PROJECT))
        except Exception:
            rel = str(p)
        row = {
            "path": str(p),
            "relative_path": rel,
            "sha256": sha256_file(p),
            "mtime_iso": mt.isoformat(),
            "mtime_in_campaign_window": CAMPAIGN_START <= mt <= CAMPAIGN_END,
            "hits": ";".join(info["hits"]),
            "co_filenames": json.dumps(info["co_filenames"]),
            "code_names": json.dumps(info["code_names"]),
            "mentions_scenario39_legacy": (
                "scenario39_legacy" in json.dumps(info).lower()
            ),
            "mentions_fall_scenario39": (
                "fall_scenario39" in json.dumps(info).lower()
            ),
            "mentions_fall_forward_height": (
                "fall_forward_height" in json.dumps(info).lower()
            ),
            "mentions_rne": (
                "mj_rnepostconstraint" in json.dumps(info).lower()
            ),
            "mentions_sidecar": (
                "high_rate_imu_sidecar" in json.dumps(info).lower()
            ),
            "relevant_string_constants": json.dumps(info["relevant_string_constants"]),
        }
        rows.append(row)
        detail.append(
            f"\n{'='*112}\nPYC: {p}\nSHA256: {row['sha256']}\nMTIME: {row['mtime_iso']}\n"
            f"CO_FILENAMES: {row['co_filenames']}\nHITS: {row['hits']}\n{'='*112}\n"
            + "\n".join(info["relevant_string_constants"]) + "\n"
        )
    rows.sort(key=lambda r: (not r["mtime_in_campaign_window"], r["relative_path"]))
    write_csv(PYC / "pyc_runtime_signatures.csv", rows)
    (PYC / "pyc_relevant_constants.txt").write_text("".join(detail), encoding="utf-8")
    return rows

# ------------------------- Git recovery -------------------------

def git_recovery():
    reflog = run_git(["reflog", "--all", "--date=iso"])
    stash = run_git(["stash", "list", "--date=iso"])
    fsck = run_git(["fsck", "--full", "--no-reflogs", "--unreachable"], timeout=90)

    (GIT / "git_reflog_all.txt").write_text(reflog, encoding="utf-8")
    (GIT / "git_stash_list.txt").write_text(stash, encoding="utf-8")
    (GIT / "git_fsck_unreachable.txt").write_text(fsck, encoding="utf-8")

    candidate_commits = set()
    for txt in (reflog, stash, fsck):
        for m in re.finditer(r"\b[0-9a-f]{40}\b", txt):
            candidate_commits.add(m.group(0))
        for m in re.finditer(r"\b[0-9a-f]{7,39}\b", txt):
            # short reflog hashes can be resolved later
            if len(m.group(0)) >= 7:
                candidate_commits.add(m.group(0))

    rows = []
    excerpts = []
    checked = 0
    for rev in sorted(candidate_commits):
        if checked > 800:
            break
        typ = run_git(["cat-file", "-t", rev], timeout=8).strip()
        if typ != "commit":
            continue
        checked += 1
        meta = run_git(["show", "-s", "--date=iso", "--format=%H|%ad|%an|%s", rev], timeout=8).strip()
        tree_names = run_git(["ls-tree", "-r", "--name-only", rev], timeout=15)
        if not any("scenario39" in line.lower() or
                   any(k in line.lower() for k in ["campaign", "highrate", "runner", "sidecar"])
                   for line in tree_names.splitlines()):
            continue
        for rel in tree_names.splitlines():
            low = rel.lower()
            if not (("scenario39" in low) or any(k in low for k in ["campaign", "highrate", "runner", "sidecar"])):
                continue
            blob = run_git(["show", f"{rev}:{rel}"], timeout=15)
            if blob.startswith("fatal:") or blob.startswith("ERROR:"):
                continue
            hits = term_hits(blob)
            if not hits:
                continue
            rows.append({
                "rev": rev,
                "meta": meta,
                "relative_path": rel,
                "blob_sha256": hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest(),
                "hits": ";".join(hits),
                "mentions_scenario39_legacy": "scenario39_legacy" in blob.lower(),
                "mentions_fall_scenario39": "fall_scenario39" in blob.lower(),
                "mentions_fall_forward_height": "fall_forward_height" in blob.lower(),
                "mentions_rne": "mj_rnepostconstraint" in blob.lower(),
                "mentions_sidecar": "high_rate_imu_sidecar" in blob.lower(),
            })
            excerpts.append(
                f"\n{'='*112}\nREV: {rev}\nMETA: {meta}\nPATH: {rel}\n"
                f"HITS: {';'.join(hits)}\n{'='*112}\n{context(blob)}\n"
            )
    write_csv(GIT / "recovered_git_runtime_candidates.csv", rows)
    (GIT / "recovered_git_runtime_excerpts.txt").write_text("".join(excerpts), encoding="utf-8")
    return rows

# ------------------------- Editor/local history -------------------------

def editor_history_roots() -> list[Path]:
    candidates = [
        HOME / ".config/Code/User/History",
        HOME / ".config/VSCodium/User/History",
        HOME / ".local/share/code-server/User/History",
        HOME / ".vscode-server/data/User/History",
        PROJECT / ".history",
        PROJECT / ".local-history",
    ]
    return [p for p in candidates if p.exists() and p.is_dir()]

def scan_editor_history():
    rows = []
    excerpts = []
    for root in editor_history_roots():
        count = 0
        for dirpath, dirs, files in os.walk(root):
            # Bound the search: history can be large.
            if count > 25000:
                break
            for fn in files:
                count += 1
                p = Path(dirpath) / fn
                try:
                    if p.stat().st_size > 4_000_000:
                        continue
                except Exception:
                    continue
                txt = safe_text(p, 4_000_000)
                if not txt:
                    continue
                hits = term_hits(txt)
                if not hits:
                    continue
                st = p.stat()
                mt = dt.datetime.fromtimestamp(st.st_mtime)
                row = {
                    "history_root": str(root),
                    "path": str(p),
                    "sha256": sha256_file(p),
                    "mtime_iso": mt.isoformat(),
                    "mtime_in_campaign_window": CAMPAIGN_START <= mt <= CAMPAIGN_END,
                    "hits": ";".join(hits),
                    "mentions_scenario39_legacy": "scenario39_legacy" in txt.lower(),
                    "mentions_fall_scenario39": "fall_scenario39" in txt.lower(),
                    "mentions_fall_forward_height": "fall_forward_height" in txt.lower(),
                    "mentions_rne": "mj_rnepostconstraint" in txt.lower(),
                    "mentions_sidecar": "high_rate_imu_sidecar" in txt.lower(),
                }
                rows.append(row)
                excerpts.append(
                    f"\n{'='*112}\nEDITOR HISTORY: {p}\nMTIME: {row['mtime_iso']}\n"
                    f"HITS: {row['hits']}\n{'='*112}\n{context(txt)}\n"
                )
    rows.sort(key=lambda r: (not r["mtime_in_campaign_window"], r["path"]))
    write_csv(EDH / "editor_history_runtime_candidates.csv", rows)
    (EDH / "editor_history_runtime_excerpts.txt").write_text("".join(excerpts), encoding="utf-8")
    write_json(EDH / "editor_history_roots.json", [str(p) for p in editor_history_roots()])
    return rows

# ------------------------- Timing fingerprint -------------------------

def csv_duration(p: Path):
    first = None
    last = None
    n = 0
    try:
        with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(line for line in f if not line.startswith("#"))
            header = next(reader)
            if "timestamp" not in header:
                return None
            ti = header.index("timestamp")
            for row in reader:
                if len(row) <= ti:
                    continue
                try:
                    t = float(row[ti])
                except Exception:
                    continue
                if first is None:
                    first = t
                last = t
                n += 1
    except Exception:
        return None
    if first is None or last is None:
        return None
    return n, first, last, last-first

def timing_fingerprint():
    rows = []
    for p in sorted(CANON_ROOT.glob("runs/P*/task_39/*/fall_scenario39_*.csv")):
        if p.name.endswith("_highrate_truth.csv"):
            continue
        d = csv_duration(p)
        if d is None:
            continue
        n, first, last, dur = d
        profile = next((x for x in p.parts if re.fullmatch(r"P\d{3}", x)), "")
        rows.append({
            "profile": profile,
            "path": str(p),
            "rows": n,
            "first_timestamp": first,
            "last_timestamp": last,
            "duration_s": dur,
            "intervals_at_30hz_equivalent": dur * 30.0,
        })
    write_csv(TIM / "task39_canonical_duration_fingerprint.csv", rows)
    if rows:
        ds = sorted(r["duration_s"] for r in rows)
        summary = {
            "n": len(ds),
            "min_s": min(ds),
            "max_s": max(ds),
            "median_s": ds[len(ds)//2],
            "unique_rounded_0p01s": dict(Counter(round(x,2) for x in ds)),
            "p020_duration_s": next((r["duration_s"] for r in rows if r["profile"]=="P020"), None),
        }
    else:
        summary = {"n": 0}
    write_json(TIM / "task39_duration_summary.json", summary)
    return rows, summary

def score_evidence(runtime_rows, pyc_rows, git_rows, editor_rows):
    evidence = []

    def add(src, row, base):
        score = base
        reasons = []
        if row.get("mtime_in_campaign_window"):
            score += 25; reasons.append("mtime in Aug-24..28 campaign window")
        if row.get("mentions_exact_corrected396"):
            score += 25; reasons.append("mentions exact corrected396 root")
        if row.get("mentions_scenario39_legacy"):
            score += 18; reasons.append("mentions scenario39_legacy")
        if row.get("mentions_fall_scenario39"):
            score += 15; reasons.append("contains canonical fall_scenario39 signature")
        if row.get("mentions_rne"):
            score += 15; reasons.append("contains mj_rnePostConstraint")
        if row.get("mentions_sidecar"):
            score += 12; reasons.append("contains high_rate_imu_sidecar")
        if row.get("mentions_fall_forward_height"):
            score += 4; reasons.append("contains raw fall_forward_height signature")
        evidence.append({
            "score": score,
            "source_class": src,
            "path_or_rev": row.get("relative_path") or row.get("path") or row.get("rev"),
            "sha256": row.get("sha256") or row.get("blob_sha256") or "",
            "reasons": " | ".join(reasons),
            "raw": json.dumps(row, default=str)[:12000],
        })

    for r in runtime_rows: add("FILESYSTEM_TEXT", r, 10)
    for r in pyc_rows: add("PYC_BYTECODE", r, 20)
    for r in git_rows: add("GIT_RECOVERY", r, 25)
    for r in editor_rows: add("EDITOR_HISTORY", r, 18)

    evidence.sort(key=lambda x: (-x["score"], x["source_class"], x["path_or_rev"] or ""))
    write_csv(DEC / "runtime_residue_evidence_ranking.csv", evidence)
    return evidence

def main():
    for d in [OUT, RES, PYC, GIT, EDH, TIM, DEC]:
        d.mkdir(parents=True, exist_ok=True)

    print("="*100)
    print("TASK39 CORRECTED396 RUNTIME-RESIDUE FORENSICS V6 — NO SIMULATOR RERUN")
    print("="*100)
    print("Mode: static/read-only")
    print("Bytecode policy: decode constants/metadata only; NEVER execute code objects")

    runtime_rows = scan_runtime_residue()
    print(f"Filesystem runtime/provenance candidates: {len(runtime_rows)}")

    pyc_rows = scan_pyc()
    print(f"Relevant PYC bytecode artifacts: {len(pyc_rows)}")

    git_rows = git_recovery()
    print(f"Recovered Git runtime candidates: {len(git_rows)}")

    editor_rows = scan_editor_history()
    print(f"Editor/local-history candidates: {len(editor_rows)}")

    timing_rows, timing_summary = timing_fingerprint()
    print(f"Canonical Task-39 duration rows: {len(timing_rows)}")

    ranking = score_evidence(runtime_rows, pyc_rows, git_rows, editor_rows)
    top = ranking[0] if ranking else None

    # Conservative classification. V6 never authorizes a replay itself.
    exact_chain_signal = False
    if top:
        raw = top.get("raw","").lower()
        exact_chain_signal = (
            "scenario39_legacy" in raw and
            ("mj_rnepostconstraint" in raw or "high_rate_imu_sidecar" in raw) and
            ("fall_scenario39" in raw or "corrected396" in raw)
        )

    proof = (
        "STRONG_RUNTIME_RESIDUE_FOUND_REVIEW_REQUIRED"
        if exact_chain_signal else
        "RUNTIME_RESIDUE_FOUND_REVIEW_REQUIRED"
        if ranking else
        "UNRESOLVED"
    )

    decision = {
        "root_cause_class": "SOURCE_OR_RUNTIME_VERSION_MISMATCH",
        "paper_gate": "BLOCK_REMAINS",
        "simulator_rerun_performed": False,
        "bytecode_executed": False,
        "replay_authorized_now": False,
        "v4_source_commit": V4_SOURCE_COMMIT,
        "v4_source_hash": V4_SOURCE_HASH,
        "failed_replay_source_hash": FAILED_REPLAY_HASH,
        "filesystem_candidates": len(runtime_rows),
        "pyc_candidates": len(pyc_rows),
        "git_recovery_candidates": len(git_rows),
        "editor_history_candidates": len(editor_rows),
        "task39_duration_summary": timing_summary,
        "proof_grade": proof,
        "top_runtime_residue_evidence": top,
        "next_rule": (
            "Do not rerun yet. Review the top runtime-residue candidate and its excerpt. "
            "If it establishes the Task39 source + high-rate RNE wrapper/sidecar + canonical "
            "output packaging path used on Aug 26, prepare exactly one historical-runtime P020 "
            "replay with unchanged instrumentation and unchanged strict equivalence thresholds."
        ),
    }
    write_json(DEC / "RUNTIME_RESIDUE_DECISION.json", decision)

    report = [
        "="*100,
        "TASK39 CORRECTED396 RUNTIME-RESIDUE FORENSICS V6",
        "="*100,
        "Root cause class: SOURCE_OR_RUNTIME_VERSION_MISMATCH",
        "Paper gate: BLOCK_REMAINS",
        "Simulator rerun performed: NO",
        "Bytecode executed: NO",
        "Replay authorized now: NO",
        f"Filesystem runtime/provenance candidates: {len(runtime_rows)}",
        f"Relevant PYC bytecode artifacts: {len(pyc_rows)}",
        f"Recovered Git runtime candidates: {len(git_rows)}",
        f"Editor/local-history candidates: {len(editor_rows)}",
        f"Canonical Task-39 duration rows: {len(timing_rows)}",
        f"Proof grade: {proof}",
    ]
    if top:
        report += [
            f"Top evidence class: {top['source_class']}",
            f"Top evidence: {top['path_or_rev']}",
            f"Top evidence score: {top['score']}",
            f"Top evidence reasons: {top['reasons']}",
        ]
    report += [
        "",
        "REVIEW THESE FIRST:",
        "  06_DECISION/RUNTIME_RESIDUE_DECISION.json",
        "  06_DECISION/runtime_residue_evidence_ranking.csv",
        "  01_RUNTIME_RESIDUE/runtime_residue_excerpts.txt",
        "  02_BYTECODE/pyc_runtime_signatures.csv",
        "  02_BYTECODE/pyc_relevant_constants.txt",
        "  03_GIT_RECOVERY/recovered_git_runtime_candidates.csv",
        "  03_GIT_RECOVERY/recovered_git_runtime_excerpts.txt",
        "  04_EDITOR_HISTORY/editor_history_runtime_candidates.csv",
        "  05_TIMING_FINGERPRINT/task39_duration_summary.json",
        "  05_TIMING_FINGERPRINT/task39_canonical_duration_fingerprint.csv",
        "",
        "DO NOT rerun Task-39 yet."
    ]
    (OUT / "DIAGNOSTIC_SUMMARY.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    import zipfile
    zip_path = PROJECT / "outputs/task39_corrected396_runtime_residue_forensics_v6.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT.parent))
    digest = sha256_file(zip_path)
    sha_path = Path(str(zip_path) + ".sha256")
    sha_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")

    print("-"*100)
    for line in report[3:]:
        print(line)
    print("-"*100)
    print("ZIP:", zip_path)
    print("ZIP SHA256:", digest)
    print("SHA256 sidecar:", sha_path)
    print("="*100)

if __name__ == "__main__":
    main()
