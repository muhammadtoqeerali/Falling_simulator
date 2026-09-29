from __future__ import annotations

import argparse
import re
from pathlib import Path


IMPORT_LINE = (
    "from highrate_runtime_bridge "
    "import highrate_env_step"
)

OLD_CALL = "env.step(action)"

NEW_CALL = (
    "highrate_env_step("
    "env, action, legacy_imu=imu)"
)


FILES = (
    [Path("fall_core.py"),
     Path("backward_fall_walking_best.py")]
    + sorted(
        Path(".").glob(
            "scenario*_legacy.py"
        )
    )
)


def patch_text(path: Path, text: str):

    notes = []

    old_count = len(
        re.findall(
            r"\benv\.step\s*"
            r"\(\s*action\s*\)",
            text,
        )
    )

    new_count = text.count(
        NEW_CALL
    )

    if old_count == 0 and new_count == 1:
        notes.append(
            "already patched step"
        )

    elif old_count != 1:
        raise RuntimeError(
            f"{path}: expected exactly "
            "1 main env.step(action), "
            f"found {old_count}"
        )

    else:
        text = re.sub(
            r"\benv\.step\s*"
            r"\(\s*action\s*\)",
            NEW_CALL,
            text,
            count=1,
        )

        notes.append(
            "replace main env.step(action)"
        )

    if IMPORT_LINE in text:
        notes.append(
            "import already present"
        )

    else:
        # All audited runtime files import numpy.
        # Insert immediately after numpy import;
        # this avoids interfering with module
        # docstrings or __future__ imports.
        lines = text.splitlines(
            keepends=True
        )

        insertion = None

        for i, line in enumerate(lines):
            if re.match(
                r"^\s*import\s+numpy\s+as\s+np\s*$",
                line.rstrip("\r\n"),
            ):
                insertion = i + 1
                break

        if insertion is None:
            raise RuntimeError(
                f"{path}: could not find "
                "'import numpy as np'"
            )

        newline = (
            "\r\n"
            if lines[insertion - 1].endswith(
                "\r\n"
            )
            else "\n"
        )

        lines.insert(
            insertion,
            IMPORT_LINE + newline,
        )

        text = "".join(lines)

        notes.append(
            "add highrate bridge import"
        )

    # Final invariants.
    if text.count(NEW_CALL) != 1:
        raise RuntimeError(
            f"{path}: patched main-call "
            "count is not exactly 1"
        )

    if text.count(IMPORT_LINE) != 1:
        raise RuntimeError(
            f"{path}: bridge import "
            "count is not exactly 1"
        )

    return text, notes


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Actually write files. "
            "Without this flag the "
            "script is read-only."
        ),
    )

    args = ap.parse_args()

    mode = (
        "APPLY"
        if args.apply
        else "DRY RUN - NO FILES WRITTEN"
    )

    print("=" * 100)
    print(
        "HIGH-RATE IMU "
        f"RUNTIME PATCH: {mode}"
    )
    print("=" * 100)

    candidates = []

    for path in FILES:

        if not path.exists():
            continue

        original = path.read_text(
            encoding="utf-8",
            errors="strict",
        )

        # Only runtime files containing the
        # audited exact main anchor participate.
        has_old = bool(
            re.search(
                r"\benv\.step\s*"
                r"\(\s*action\s*\)",
                original,
            )
        )

        has_new = (
            NEW_CALL in original
        )

        if not has_old and not has_new:
            continue

        patched, notes = patch_text(
            path,
            original,
        )

        changed = (
            patched != original
        )

        candidates.append(
            (
                path,
                original,
                patched,
                notes,
                changed,
            )
        )

        print(
            f"\n{path}"
        )

        print(
            "  changed:",
            changed,
        )

        for note in notes:
            print(
                "  -",
                note,
            )

        # Show the exact resulting main line.
        for lineno, line in enumerate(
            patched.splitlines(),
            1,
        ):
            if NEW_CALL in line:
                print(
                    f"  resulting step "
                    f"line {lineno}: "
                    f"{line.strip()}"
                )

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        "runtime files:",
        len(candidates),
    )

    print(
        "files needing change:",
        sum(
            1
            for x in candidates
            if x[4]
        ),
    )

    if len(candidates) != 27:
        raise SystemExit(
            "REFUSING: expected 27 "
            "audited runtimes, found "
            f"{len(candidates)}"
        )

    if not args.apply:
        print(
            "\nDRY_RUN_OK"
        )
        print(
            "No files were modified."
        )
        return

    # Only reach here with explicit --apply.
    for (
        path,
        original,
        patched,
        notes,
        changed,
    ) in candidates:

        if changed:
            path.write_text(
                patched,
                encoding="utf-8",
            )

    print(
        "\nPATCH_APPLIED_OK"
    )


if __name__ == "__main__":
    main()
