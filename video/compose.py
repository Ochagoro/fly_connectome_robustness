"""Compose the research video (1280x720, 30 fps, Japanese captions).

Segments: title, round 1 (as published: left vs right hemisphere), two causes,
round 2 (corrected right hemisphere), error tournament, take-home message.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from brain_render import BrainCanvas  # noqa: E402

W, H, FPS = 1280, 720, 30
BUILD = HERE / "build"
BG = (14, 19, 30)
PANEL = (24, 31, 46)
INK = (240, 242, 246)
INK2 = (168, 176, 190)
BLUE = (57, 135, 229)
ORANGE = (235, 104, 52)
AQUA = (27, 175, 122)
RED = (230, 103, 103)
FONT_R = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
_fonts = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FONT_B if bold else FONT_R, size, index=0)
    return _fonts[k]


def text(d, xy, s, size=24, bold=False, fill=INK, anchor="la"):
    d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def blank():
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def fade(frames, n=8):
    out = []
    for i, f in enumerate(frames):
        a = min(1.0, (i + 1) / n, (len(frames) - i) / n)
        out.append((np.asarray(f).astype(np.float32) * a + np.array(BG) * (1 - a)).astype(np.uint8))
    return out


# --------------------------------------------------------------- brain canvas
_canvas = {}


def canvas(w, h):
    if (w, h) not in _canvas:
        xy = np.load(HERE / "brain_xy.npy")
        _canvas[(w, h)] = BrainCanvas(xy, w, h)
    return _canvas[(w, h)]


class Glow:
    def __init__(self, name):
        d = np.load(BUILD / f"{name}_glow.npz")
        self.idx, self.val, self.ptr = d["idx"], d["val"].astype(np.float32), d["ptr"]
        self.rate, self.act, self.cum = d["rate"], d["act"], d["cum"]
        self.ms_per_frame = float(d["ms_per_frame"])
        self.n = len(self.ptr) - 1

    def intensity(self, f, n_neurons=138639):
        x = np.zeros(n_neurons, np.float32)
        lo, hi = self.ptr[f], self.ptr[f + 1]
        x[self.idx[lo:hi]] = self.val[lo:hi]
        return x


def body_frames(name):
    return [fr for fr in imageio.get_reader(BUILD / f"{name}_body.mp4")]


# ------------------------------------------------------------------ segments
def title_card(sec=4.0):
    frames = []
    for i in range(int(sec * FPS)):
        im, d = blank()
        text(d, (W // 2, 250), "左の脳と右の脳は、同じ答えを出すか？", 46, True, anchor="mm")
        text(d, (W // 2, 320), "ハエの全脳コネクトーム・シミュレーションで「鏡像テスト」", 28, False, INK2, anchor="mm")
        text(d, (W // 2, 430), "FlyWire 配線図（ニューロン 138,639 個・シナプス 5,449 万個）", 22, False, INK2, anchor="mm")
        text(d, (W // 2, 465), "スパイキングニューラルネットワーク（Shiu ら 2024 のモデル）", 22, False, INK2, anchor="mm")
        frames.append(np.asarray(im))
    return fade(frames)


def explain_card(lines, sec=5.0, accent=INK):
    frames = []
    for i in range(int(sec * FPS)):
        im, d = blank()
        y = 230
        for j, (s, size, bold, col) in enumerate(lines):
            text(d, (W // 2, y), s, size, bold, col, anchor="mm")
            y += int(size * 1.9)
        frames.append(np.asarray(im))
    return fade(frames)


def meter(d, x, y, w, h, frac, col, label, value):
    d.rounded_rectangle([x, y, x + w, y + h], 6, fill=(34, 42, 60))
    hh = int(h * max(0.0, min(1.0, frac)))
    if hh > 0:
        d.rounded_rectangle([x, y + h - hh, x + w, y + h], 6, fill=col)
    text(d, (x + w // 2, y - 12), label, 15, False, INK2, anchor="ms")
    text(d, (x + w // 2, y + h + 22), value, 18, True, INK, anchor="ms")


def battle(left_name, right_name, header, sub, right_label, right_col=ORANGE, verdict=None):
    gl, gr = Glow(left_name), Glow(right_name)
    bl, br = body_frames(left_name), body_frames(right_name)
    n = min(gl.n, gr.n, len(bl), len(br))
    cv = canvas(600, 268)
    frames = []
    for f in range(n):
        im, d = blank()
        t_ms = (f + 1) * gl.ms_per_frame
        text(d, (40, 34), header, 30, True)
        text(d, (40, 72), sub, 18, False, INK2)
        text(d, (W - 40, 40), f"t = {t_ms:5.0f} ms", 26, True, anchor="ra")
        text(d, (W - 40, 74), f"{1000 / gl.ms_per_frame / FPS:.0f} 倍スロー", 16, False, INK2, anchor="ra")
        for k, (g, b, lab, col, x0) in enumerate([(gl, bl, "左の砂糖センサーを刺激", BLUE, 20),
                                                    (gr, br, right_label, right_col, 660)]):
            d.rounded_rectangle([x0, 100, x0 + 600, 700], 12, fill=PANEL)
            d.rectangle([x0, 100, x0 + 600, 106], fill=col)
            text(d, (x0 + 20, 118), lab, 22, True, col)
            brain = cv.frame(np.log1p(g.intensity(f)) * 0.7)
            im.paste(Image.fromarray(brain), (x0, 156))
            body = Image.fromarray(b[f]).resize((386, 270))
            im.paste(body, (x0 + 10, 428))
            act = g.act[f]
            meter(d, x0 + 425, 538, 64, 112, g.rate[f] / 100, col, "MN9", f"{g.rate[f]:.0f} Hz")
            meter(d, x0 + 515, 538, 64, 112, np.log10(max(act, 1)) / 4, RED if act > 1500 else INK2,
                  "活動数", f"{int(act):,}")
            text(d, (x0 + 412, 446), "飲む指令（累計）", 15, False, INK2)
            text(d, (x0 + 412, 466), f"{int(g.cum[f])} 回", 22, True, INK)
            if act > 1500:
                text(d, (x0 + 300, 300), "暴走！", 46, True, RED, anchor="mm")
        if verdict and f > n - 60:
            d.rounded_rectangle([340, 640, 940, 700], 10, fill=(10, 12, 18))
            text(d, (W // 2, 670), verdict, 24, True, INK, anchor="mm")
        frames.append(np.asarray(im))
    return fade(frames, 6)


def cause_synapses(sec=7.0):
    L, R = 265, 125
    frames = []
    n = int(sec * FPS)
    for i in range(n):
        im, d = blank()
        text(d, (60, 60), "原因① 右側の砂糖センサーは、配線図上のシナプスが左の半分", 30, True)
        text(d, (60, 105), "砂糖を感じるニューロン1個あたりの出力シナプス数（FlyWire）", 20, False, INK2)
        p = min(1.0, i / (FPS * 1.5))
        for k, (v, col, lab, n_) in enumerate([(L, BLUE, "左", 33), (R, ORANGE, "右", 24)]):
            y = 230 + k * 150
            text(d, (100, y + 30), lab, 34, True, col, anchor="mm")
            d.rounded_rectangle([150, y, 150 + int(3.2 * v * p), y + 60], 8, fill=col)
            text(d, (170 + int(3.2 * v * p), y + 30), f"{v} 個 × {n_} ニューロン", 24, True, INK, anchor="lm")
        if i > FPS * 2.5:
            text(d, (W // 2, 590), "左の砂糖センサーから同じだけシナプスを抜くと、左でも口吻は伸びなくなる", 24, True, INK, anchor="mm")
            text(d, (W // 2, 635), "→ 右の失敗は「脳の違い」ではなく「配線図の取りこぼし」で説明できる", 22, False, INK2, anchor="mm")
        frames.append(np.asarray(im))
    return fade(frames)


def cause_serotonin(sec=9.0):
    cv = canvas(760, 330)
    ann = pd.read_csv(ROOT / "data/neuron_annotations.tsv", sep="\t", usecols=["root_id", "cell_class", "top_nt"])
    comp = pd.read_csv(ROOT / "data/shiu_model/Completeness_783.csv", index_col=0)
    ann = ann.set_index("root_id").reindex(comp.index)
    gate = ((ann.cell_class == "ALLN") & (ann.top_nt == "serotonin")).to_numpy()
    entry = int(np.flatnonzero(comp.index.to_numpy() == 720575940629884119)[0])
    ex, ey = int(cv.px[entry]) + 260, int(cv.py[entry]) + 132
    frames = []
    n = int(sec * FPS)
    for i in range(n):
        im, d = blank()
        text(d, (60, 56), "原因② 暴走には「入口」と「増幅器」がそろう必要があった", 30, True)
        text(d, (60, 100), "右の砂糖センサー 24 個を 1 個ずつ抜いて調べた結果", 20, False, INK2)
        inten = np.zeros(len(gate), np.float32)
        pulse = 0.6 + 0.4 * np.sin(i / 6)
        if i > FPS * 2.5:
            inten[gate] = 2.5 * pulse
        im.paste(Image.fromarray(cv.frame(inten)), (260, 132))
        d = ImageDraw.Draw(im)
        r = 9 + 3 * np.sin(i / 5)
        d.ellipse([ex - r, ey - r, ex + r, ey + r], outline=AQUA, width=3)
        text(d, (ex + 16, ey + 4), "入口", 20, True, AQUA, anchor="lm")
        text(d, (70, 520), "入口：右の砂糖センサーの 1 個だけが、嗅覚の中枢（触角葉）へ 22 個のシナプスを出す（左は最大 3 個）", 21, True, AQUA)
        if i > FPS * 2.5:
            text(d, (70, 562), "増幅器：触角葉の 40 個を AI が「セロトニン」と判定（確信度の中央値 0.35）→ モデルではアクセル扱い", 21, True, (255, 196, 64))
            text(d, (70, 596), "　　　　本来はブレーキ役の細胞のはず。触角葉に入るセロトニン細胞は片側 1 個（CSD ニューロン）だけ", 18, False, INK2)
        if i > FPS * 5:
            text(d, (W // 2, 660), "どちらか一方を取り除くと、右の暴走は消える（約 8,500 個 → 60〜240 個）", 25, True, INK, anchor="mm")
        frames.append(np.asarray(im))
    return fade(frames)


def tournament(sec=11.0):
    p = ROOT / "results/video_tournament.json"
    if not p.exists():
        return []
    entries = json.load(open(p))
    frames = []
    n = int(sec * FPS)
    bodies = {e["key"]: Image.open(BUILD / f"tour_{e['key']}.png") for e in entries}
    for i in range(n):
        im, d = blank()
        text(d, (60, 50), "誤りトーナメント：配線図にわざと誤りを入れても、ハエは砂糖を飲めるか？", 28, True)
        text(d, (60, 92), "左の砂糖センサーを刺激したときの MN9（口吻）発火率。誤りのない配線図を 100 とする", 18, False, INK2)
        prog = min(1.0, i / (FPS * 3.0))
        for k, e in enumerate(entries):
            cx, cy = 60 + (k % 4) * 300, 128 + (k // 4) * 262
            d.rounded_rectangle([cx, cy, cx + 280, cy + 250], 12, fill=PANEL)
            im.paste(bodies[e["key"]].resize((240, 168)), (cx + 20, cy + 8))
            text(d, (cx + 14, cy + 186), e["label"], 15, True)
            v = e["score"] * prog
            col = AQUA if e["score"] >= 0.5 else ORANGE
            if e.get("run_k", 0) > 0 and prog >= 1:
                col = RED if e.get("runaway") else col
            d.rounded_rectangle([cx + 14, cy + 210, cx + 266, cy + 226], 6, fill=(34, 42, 60))
            d.rounded_rectangle([cx + 14, cy + 210, cx + 14 + max(2, int(252 * min(v, 1.0))), cy + 226], 6, fill=col)
            tag = f"{v * 100:.0f}"
            if e.get("run_k", 0) > 0 and prog >= 1:
                tag += f"　暴走 {e['run_k']}/{e['run_n']} 回"
            text(d, (cx + 266, cy + 246), tag, 15, True, col, anchor="rs")
        if i > FPS * 4:
            d.rounded_rectangle([140, 660, 1140, 708], 10, fill=(10, 12, 18))
            text(d, (W // 2, 684), "偽のシナプスには強いが、見逃しには弱い。符号の取り違えや強さのばらつきは暴走の引き金に", 20, True, anchor="mm")
        frames.append(np.asarray(im))
    return fade(frames)


def ending(sec=6.0):
    return explain_card([
        ("わかったこと", 32, True, INK),
        ("同じ脳の左右で、シミュレーションの答えが食い違った（131 種類の感覚でも同様）", 24, False, INK),
        ("原因はおそらく脳ではなく配線図：シナプスの取りこぼし・1 個の異常な配線・伝達物質の誤判定", 23, False, INK),
        ("シナプスを 14〜24% 見逃すだけで、多くの予測は効果が半分に", 24, False, INK),
        ("配線図のシミュレーションは、左右で答え合わせをしてから信じよう", 26, True, AQUA),
    ], sec)


def main():
    segs = [title_card(),
            explain_card([("第 1 ラウンド", 34, True, INK),
                          ("砂糖センサーを刺激して、口吻を伸ばす神経（MN9）が動くか", 24, False, INK2),
                          ("左右の脳で、まったく同じ実験をする", 24, False, INK2)], 3.5),
            battle("sugar_left_intact_0_15s", "sugar_right_intact_0_15s", "第 1 ラウンド：公開された配線図のまま",
                   "同じ砂糖刺激（100 Hz）を、左右どちらかの味覚ニューロンに入れる", "右の砂糖センサーを刺激",
                   verdict="左は飲む。右は脳全体が暴走して飲めない"),
            cause_synapses(), cause_serotonin(),
            battle("sugar_left_intact_0_12s", "sugar_right_rescued_0_12s", "第 2 ラウンド：2 つの違いを補正した配線図",
                   "右側：少ないシナプスを左と同じ量に補い、触角葉の 40 個の符号を直す", "右（補正後）を刺激", AQUA,
                   verdict="補正すると、右の脳も砂糖を飲めるようになった"),
            tournament(), ending()]
    out = HERE / "fly_mirror_test_ja.mp4"
    w = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=8,
                           ffmpeg_params=["-pix_fmt", "yuv420p"])
    n = 0
    for s in segs:
        for f in s:
            w.append_data(f)
            n += 1
    w.close()
    print("wrote", out, n / FPS, "s")


if __name__ == "__main__":
    main()
