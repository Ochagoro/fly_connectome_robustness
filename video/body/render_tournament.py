"""Render one still per tournament entry (run with this folder's venv)."""
import json, sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_fly import FlyRenderer

ROOT = Path(__file__).resolve().parents[2]
entries = json.load(open(ROOT / "results/video_tournament.json"))
fr = FlyRenderer(520, 364, "side_head", seed=4)
for e in entries:
    ext = max(0.0, min(1.0, e["score"])) if not e["runaway"] else 0.0
    img = fr.frame(ext, 1.0 if e["runaway"] else 0.0)
    Image.fromarray(img).save(ROOT / f"video/build/tour_{e['key']}.png")
    print(e["key"], ext)
