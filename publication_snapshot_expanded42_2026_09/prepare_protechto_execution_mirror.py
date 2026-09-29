from pathlib import Path
import shutil
import hashlib

SRC = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master"
)

DST = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/"
    "outputs/protechto_clean_execution_mirror_v1"
)

EXPECTED_TRAIN_SHA = (
    "e5b71a5c5a9688cbe9a655c939541808"
    "a9d4b357fd2a00b0c86d1843c97108b2"
)

print("=" * 100)
print("PROTECHTO CLEAN EXECUTION MIRROR")
print("=" * 100)

raw = (SRC / "train.py").read_bytes()

sha = hashlib.sha256(raw).hexdigest()

print("Canonical train.py SHA256:")
print(sha)

if sha != EXPECTED_TRAIN_SHA:
    raise RuntimeError(
        "Canonical train.py changed since preflight."
    )

# Identify every non-ASCII / potentially problematic source byte.
bad = []

for lineno, line in enumerate(
    raw.splitlines(),
    start=1,
):
    try:
        line.decode("utf-8")
    except UnicodeDecodeError:
        bad.append(
            (lineno, line)
        )

print()
print("Non-UTF8 lines in train.py:")

for lineno, line in bad:
    print(
        f"line {lineno}:",
        line.decode(
            "cp1252",
            errors="replace",
        ),
    )

if not bad:
    print("NONE")

# This mirror is allowed to repair encoding ONLY when the
# offending byte occurs after a Python comment marker.
for lineno, line in bad:

    hash_pos = line.find(b"#")

    if hash_pos < 0:
        raise RuntimeError(
            f"Non-UTF8 byte occurs outside a comment "
            f"at train.py line {lineno}"
        )

    for i, b in enumerate(line):
        if b >= 128 and i < hash_pos:
            raise RuntimeError(
                f"Non-ASCII byte before comment marker "
                f"at train.py line {lineno}"
            )

print()
print(
    "ENCODING SAFETY GATE: "
    "all problematic bytes are comment-only"
)

if DST.exists():
    shutil.rmtree(DST)

DST.mkdir(
    parents=True,
    exist_ok=True,
)

# Copy source only. Do NOT copy data/checkpoints/results/env.
for name in [
    "converter",
    "dataloaders",
    "losses",
    "models",
    "preprocessing",
    "simulation",
]:
    src = SRC / name

    if src.exists():
        shutil.copytree(
            src,
            DST / name,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
            ),
        )

for p in SRC.iterdir():

    if p.is_file() and (
        p.suffix == ".py"
        or p.name in {
            "config.json",
            "requirements.txt",
        }
    ):
        shutil.copy2(
            p,
            DST / p.name,
        )

# Repair ONLY non-UTF8 comment bytes in mirror train.py.
train = DST / "train.py"
raw = train.read_bytes()

fixed_lines = []

for line in raw.splitlines(
    keepends=True
):

    try:
        line.decode("utf-8")
        fixed_lines.append(line)
        continue

    except UnicodeDecodeError:
        pass

    hash_pos = line.find(b"#")

    if hash_pos < 0:
        raise RuntimeError(
            "Unsafe non-UTF8 line during mirror creation"
        )

    before = line[:hash_pos]
    comment = line[hash_pos:]

    # cp1252 0x92 = right single quotation mark.
    comment = comment.replace(
        b"\x92",
        b"'",
    )

    fixed_lines.append(
        before + comment
    )

train.write_bytes(
    b"".join(fixed_lines)
)

# Verify mirror is now valid UTF-8.
train.read_text(
    encoding="utf-8"
)

print()
print("Canonical source:")
print(SRC)

print()
print("Execution mirror:")
print(DST)

print()
print(
    "Canonical source was NOT modified."
)

print(
    "Only comment encoding was normalized "
    "in the execution mirror."
)

print()
print(
    "PROTECHTO EXECUTION MIRROR PREPARATION: PASS"
)
