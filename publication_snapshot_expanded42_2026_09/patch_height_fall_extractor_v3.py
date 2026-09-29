#!/usr/bin/env python3
from pathlib import Path
import re
import shutil
import py_compile

TARGET = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/prepare_height_fall_paper_evidence.py")

NEW_FUNCTION = """
def parse_task_profile_from_path(path: Path):
    \"\"\"
    Resolve task/profile IDs from filename and nearby path components.

    This intentionally avoids the old generic `p...` match against the full
    absolute path, which incorrectly parsed `mujoco_project` as profile
    `roject`.

    Canonical profile IDs are normalized to P###.
    \"\"\"
    nearby_parts = [path.name] + list(reversed(path.parts[-6:-1]))

    task = None
    profile = None

    profile_patterns = [
        re.compile(r"(?<![A-Za-z0-9])P[_\\- .]*0*(\\d{1,4})(?![A-Za-z0-9])", re.I),
        re.compile(r"(?<![A-Za-z0-9])(?:profile|prof)[_\\- .]*(?:P[_\\- .]*)?0*(\\d{1,4})(?![A-Za-z0-9])", re.I),
        re.compile(r"(?<![A-Za-z0-9])(?:subject|subj)[_\\- .]*(?:P[_\\- .]*)?0*(\\d{1,4})(?![A-Za-z0-9])", re.I),
    ]

    for s in nearby_parts:
        for pat in profile_patterns:
            m = pat.search(str(s))
            if m:
                profile = f"P{int(m.group(1)):03d}"
                break
        if profile is not None:
            break

    task_patterns = [
        re.compile(r"(?<![A-Za-z0-9])task[_\\- .]*0*(\\d{1,3})(?!\\d)", re.I),
        re.compile(r"(?<![A-Za-z0-9])scenario[_\\- .]*0*(\\d{1,3})(?!\\d)", re.I),
        re.compile(r"(?<![A-Za-z0-9])T[_\\- .]*0*(\\d{1,3})(?!\\d)", re.I),
    ]

    for s in nearby_parts:
        for pat in task_patterns:
            m = pat.search(str(s))
            if m:
                task = int(m.group(1))
                break
        if task is not None:
            break

    return task, profile
"""

def self_test_function():
    ns = {"Path": Path, "re": re}
    exec(NEW_FUNCTION, ns)
    fn = ns["parse_task_profile_from_path"]

    tests = [
        (Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs/task_39/P014/trial.csv"), 39, "P014"),
        (Path("/x/profile-P015/task39_data.csv"), 39, "P015"),
        (Path("/x/P016/T39_truth.csv"), 39, "P016"),
        (Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs/task_39/trial.csv"), 39, None),
    ]
    for p, exp_t, exp_p in tests:
        got_t, got_p = fn(p)
        if got_t != exp_t or got_p != exp_p:
            raise RuntimeError(
                f"SELF-TEST FAILED for {p}: got {(got_t, got_p)}, expected {(exp_t, exp_p)}"
            )

    t, p = fn(Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs/task_39/trial.csv"))
    if p == "roject":
        raise RuntimeError("REGRESSION: parser still produced fake profile 'roject'")

def patch_target():
    if not TARGET.exists():
        raise SystemExit(f"ERROR: target extractor not found: {TARGET}")

    self_test_function()
    print("PARSER SELF-TEST: PASS")

    text = TARGET.read_text(encoding="utf-8", errors="replace")

    start = text.find("def parse_task_profile_from_path")
    end = text.find("\ndef header_task_profile", start)
    if start < 0 or end < 0:
        raise SystemExit(
            "ERROR: could not locate parse_task_profile_from_path() and header_task_profile() boundaries."
        )

    backup = TARGET.with_name(TARGET.name + ".before_profile_parser_v3")
    if not backup.exists():
        shutil.copy2(TARGET, backup)

    patched = text[:start] + NEW_FUNCTION.strip() + "\n\n" + text[end + 1:]
    TARGET.write_text(patched, encoding="utf-8")

    py_compile.compile(str(TARGET), doraise=True)

    print("PATCH: PASS")
    print("TARGET COMPILE: PASS")
    print("Target :", TARGET)
    print("Backup :", backup)
    print()
    print("The parser now requires explicit profile tokens such as P014/profile-P014")
    print("and cannot interpret the word 'mujoco_project' as profile 'roject'.")

if __name__ == "__main__":
    patch_target()
