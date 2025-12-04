# autofix_punkt_tab.py
# Run this from your project root (where `venv` folder is).
# It will:
#  - look under venv/nltk_data/tokenizers/punkt_tab
#  - create .bak backups for each file it touches
#  - keep only lines that have at least one TAB and a non-empty RHS after the first TAB

from pathlib import Path
import shutil
import sys

project_root = Path.cwd()
punkt_dir = project_root / "venv" / "nltk_data" / "tokenizers" / "punkt_tab"

if not punkt_dir.exists():
    print("ERROR: punkt_tab directory not found at:", punkt_dir)
    print("Make sure you run this from the project root (the folder that contains 'venv').")
    sys.exit(1)

print("Scanning:", punkt_dir)

files = sorted(punkt_dir.glob("**/*.*"))
txts = [p for p in files if p.suffix.lower() in (".txt", ".tab")]

if not txts:
    print("No .txt or .tab files found under punkt_tab.")
    sys.exit(0)

summary = []
for p in txts:
    text = p.read_text(encoding="utf-8", errors="replace").splitlines()
    kept = []
    removed_count = 0
    for line in text:
        # keep lines that contain a tab and have at least one non-space char after the first tab
        if "\t" not in line:
            removed_count += 1
            continue
        parts = line.split("\t")
        if len(parts) < 2 or parts[1].strip() == "":
            removed_count += 1
            continue
        # keep original line (preserve full line)
        kept.append(line)
    if removed_count > 0:
        backup = p.with_suffix(p.suffix + ".bak")
        shutil.copy2(p, backup)
        p.write_text("\n".join(kept) + ("\n" if kept and not kept[-1].endswith("\n") else ""), encoding="utf-8")
        summary.append((p, len(text), len(kept), removed_count, backup))
        print(f"Fixed {p.name}: lines={len(text)} kept={len(kept)} removed={removed_count} backup={backup.name}")
    else:
        print(f"No change: {p.name} (looks ok)")

print("\nDone. Summary:")
for p, total, kept, removed, backup in summary:
    print(f" - {p.name}: total={total} kept={kept} removed={removed} backup={backup.name}")

print("\nIf errors persist, run: python train.py from the project root.")
