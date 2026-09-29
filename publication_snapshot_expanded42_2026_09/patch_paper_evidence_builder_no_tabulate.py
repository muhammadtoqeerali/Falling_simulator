#!/usr/bin/env python3
from pathlib import Path

p = Path("build_paper_evidence_archive.py")
if not p.exists():
    raise SystemExit(f"Missing {p.resolve()}")

s = p.read_text(encoding="utf-8")

helper = r"""
def dataframe_to_markdown(df, floatfmt=".2f"):
    # Small dependency-free markdown table formatter.
    cols = [str(c) for c in df.columns]
    rows = []

    for _, r in df.iterrows():
        vals = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                if v != v:
                    vals.append("N/A")
                else:
                    if floatfmt.startswith(".") and floatfmt.endswith("f"):
                        vals.append(format(v, floatfmt))
                    else:
                        vals.append(str(v))
            else:
                vals.append(str(v))
        rows.append(vals)

    def esc(x):
        return str(x).replace("|", r"\|").replace("\n", " ")

    lines = []
    lines.append("| " + " | ".join(esc(x) for x in cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for row in rows:
        lines.append("| " + " | ".join(esc(x) for x in row) + " |")
    return "\n".join(lines)
"""

if "def dataframe_to_markdown(" not in s:
    marker = "def write_text(path: Path, text: str) -> None:\n"
    idx = s.find(marker)
    if idx < 0:
        raise SystemExit("Could not find write_text() insertion point")
    s = s[:idx] + helper + "\n\n" + s[idx:]

replacements = {
    'seg_paper.to_markdown(index=False, floatfmt=".2f")':
        'dataframe_to_markdown(seg_paper, floatfmt=".2f")',
    'event_paper.to_markdown(index=False, floatfmt=".2f")':
        'dataframe_to_markdown(event_paper, floatfmt=".2f")',
    'sim_diag.to_markdown(index=False, floatfmt=".2f")':
        'dataframe_to_markdown(sim_diag, floatfmt=".2f")',
}

for old, new in replacements.items():
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("TABULATE-FREE PATCH: PASS")
