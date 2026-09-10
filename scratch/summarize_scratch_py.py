import os
import glob
from pathlib import Path

repo_root = Path(r"C:\Dev\repos\Public repos\ieee-aiot")
py_files = sorted(glob.glob(str(repo_root / "scratch" / "*.py")))

print(f"Found {len(py_files)} python files in scratch/:")
for f in py_files:
    fn = Path(f).name
    with open(f, "r", encoding="utf-8", errors="ignore") as stream:
        lines = stream.readlines()
    first_few = [l.strip() for l in lines[:10] if l.strip() and not l.strip().startswith("#")]
    print(f"--- {fn} ({len(lines)} lines) ---")
    if lines and lines[0].strip().startswith('"""') or lines[0].strip().startswith("'''"):
        # print docstring
        doc = []
        for l in lines[:15]:
            doc.append(l.strip())
            if l.strip().endswith('"""') and len(doc) > 1:
                break
        print("  Doc: " + " ".join(doc))
    else:
        print("  First non-comment code: " + (first_few[0] if first_few else "Empty"))
