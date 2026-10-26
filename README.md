# fly_connectome_robustness

ハエの全脳配線図（FlyWire v783）から作ったスパイキングニューラルネットワーク（Shiu ら 2024, Nature）の予測が、配線図の誤りにどれだけ左右されるかを調べる研究です。

同じ脳の左半球と右半球を「答え合わせの相手」として使う**鏡像テスト**を行いました。公開済みの 6 つの予測は、細胞タイプ単位で刺激すると右半球ではすべて崩れ、脳全体が暴走しました。原因として、左右の非対称性（おそらく配線図の再構成の誤り）を 4 つ特定しました。

1. 右の味覚ニューロンのシナプスが少ない
2. 読み出しニューロン aBN1 の入力が少ない
3. 右の砂糖ニューロン 1 個だけが触角葉へ異常な配線を持つ（暴走の入口）
4. 触角葉の 40 個が「セロトニン」と判定され、興奮性として扱われている（暴走の増幅器）

さらに、7 種類の誤りを人工的に入れて予測の壊れ方を測り、131 種類の感覚ニューロンで鏡像テストを行いました。

2026 年 10 月 27 日（1 日）で実施。

## 成果物

| 成果物 | ファイル |
|---|---|
| ① 論文（Nature 投稿形式、英語） | [paper/paper_en.pdf](paper/paper_en.pdf)（原稿: [paper/paper_en_src.html](paper/paper_en_src.html)、`paper/build_paper.py` で引用番号を付けて paper_en.html を生成） |
| ① 論文（日本語版） | [paper/paper_ja.pdf](paper/paper_ja.pdf) |
| ② 発表資料（10 分＋Appendix、デジタル庁デザインシステム準拠） | [slides/slides.pdf](slides/slides.pdf) |
| ③ 予備知識なしで読める解説 | [explainer/explainer.pdf](explainer/explainer.pdf)（元: [explainer/explainer.html](explainer/explainer.html)。Web 版は [explainer/web/index.html](explainer/web/index.html)） |
| ④ 研究動画（日本語字幕） | [video/fly_mirror_test_ja.mp4](video/fly_mirror_test_ja.mp4) |
| 発表原稿（10 分、想定問答つき） | [slides/script.md](slides/script.md) |
| 図（Nature 形式、PDF/SVG/PNG） | [figures/](figures/) |
| 数値のまとめ | [notes/results_summary.md](notes/results_summary.md) |

## フォルダ構成

```
data/          FlyWire v783/v630 の結合表（Shiu らのリポジトリ）と細胞注釈
src/           シミュレータ・実験・図のコード
  sim.py            GPU 版の全脳 LIF シミュレータ（Brian 2 版と r = 1.000 で一致）
  neurons.py        刺激と読み出しのニューロン集団（細胞タイプ×半球）
  perturb.py        配線図の誤りモデルと補正
  experiments.py    すべての実験（mirror, version, targeted, rescue, screen, errors, reliability）
  analysis_extra.py 査読後に追加した解析（Shiu らの元のニューロン組、1 個抜き、再集計、左右の偏り）
  figures.py        論文の図
  style.py          Nature の図の体裁
validation_brian2/  元の Brian 2 実装との比較
results/       実験結果（CSV、スパイク記録）
figures/       論文の図
paper/         論文（HTML → PDF）
slides/        発表資料
explainer/     解説資料
video/         研究動画（body/ はハエの体の 3D レンダラー、別の Python 環境）
notes/         文献メモと数値のまとめ
```

## 再現のしかた

入力データ（公開データ、計 約 400 MB）はリポジトリに含めていません。最初に次の 2 つで取得してください。

```bash
git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git data/shiu_model
```

```bash
curl -L -o data/neuron_annotations.tsv https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv
```

全タイプ比較の生の発火率（results/screen/*.npy、各 約 70 MB）も容量の都合で含めていません。`experiments.py screen reliability` で作り直せます。図と論文に使った集計済みの結果（CSV・parquet）は含めています。

```bash
uv sync
cd src
../.venv/bin/python experiments.py mirror version targeted rescue screen errors reliability
../.venv/bin/python analysis_extra.py shiu_sets leave_one_out screen_recount lateral
../.venv/bin/python figures.py fig1 fig2 fig3 fig4 ed1 ed2 ed3 ed4
```

GPU（CUDA）が必要です。RTX 5070 Ti で全実験が約 1.5 時間です。PDF は `src/topdf.sh 入力.html 出力.pdf`（Google Chrome のヘッドレス印刷）で作ります。

## 使ったデータとコード

- FlyWire v783 の結合表と Shiu らのモデル: https://github.com/philshiu/Drosophila_brain_model
- FlyWire の細胞注釈: https://github.com/flyconnectome/flywire_annotations
- ハエの体: NeuroMechFly v2（flygym）
- デジタル庁デザインシステムのトークン（MIT License）: https://github.com/digital-go-jp/design-tokens
