"""Pick representative error conditions for the video's 'error tournament' and write their scores."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
m = pd.read_csv(ROOT / "results/error_metrics.csv")
pick = [("intact", 0.0, "誤りなし"), ("false_pos", 0.8, "偽のシナプスを 80% 追加"),
        ("deletion", 0.2, "シナプスを 20% 見逃す"), ("deletion", 0.4, "シナプスを 40% 見逃す"),
        ("sign_flip", 0.02, "2% の細胞でアクセル⇄ブレーキ"), ("sign_flip", 0.2, "20% の細胞でアクセル⇄ブレーキ"),
        ("nt_resample", 0.25, "伝達物質を現実的な率（6%）で誤判定"), ("weight_noise", 1.0, "シナプスの強さが大きくバラつく")]
out = []
for k, (model, lv, label) in enumerate(pick):
    g = m[(m.model == model) & ((m.level - lv).abs() < 1e-9)]
    out.append(dict(key=f"e{k}", label=label, model=model, level=lv,
                    score=float(g.sugar_rel.mean()), runaway=bool(g.runaway.mean() > 0.5),
                    run_k=int(round(g.runaway.sum())), run_n=int(len(g))))
json.dump(out, open(ROOT / "results/video_tournament.json", "w"), ensure_ascii=False, indent=1)
for e in out:
    print(e)
