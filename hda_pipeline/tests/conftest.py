import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "data" / "pilot" / "val.jsonl").exists():
    subprocess.run([sys.executable, "-m", "hda.generator", "--config", str(ROOT / "configs" / "pilot.json"),
                    "--out", str(ROOT / "data" / "pilot")], cwd=ROOT, check=True)
