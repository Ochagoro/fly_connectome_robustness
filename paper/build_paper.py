"""Number citations in order of first appearance and build paper_en.html from paper_en_src.html.

Citations are written as [[key1,key2]] in the source. "ref. [[key]]" becomes "ref. N";
every other citation becomes a superscript with ranges compressed (1–3).
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

REFS = {
    "dork": "Dorkenwald, S. et al. Neuronal wiring diagram of an adult brain. <i>Nature</i> <b>634</b>, 124–138 (2024).",
    "schlegel": "Schlegel, P. et al. Whole-brain annotation and multi-connectome cell typing of <i>Drosophila</i>. <i>Nature</i> <b>634</b>, 139–152 (2024).",
    "shiu": "Shiu, P. K. et al. A <i>Drosophila</i> computational brain model reveals sensorimotor processing. <i>Nature</i> <b>634</b>, 210–219 (2024).",
    "eckstein": "Eckstein, N. et al. Neurotransmitter classification from electron microscopy images at synaptic sites in <i>Drosophila melanogaster</i>. <i>Cell</i> <b>187</b>, 2574–2594.e23 (2024).",
    "buhmann": "Buhmann, J. et al. Automatic detection of synaptic partners in a whole-brain <i>Drosophila</i> electron microscopy data set. <i>Nat. Methods</i> <b>18</b>, 771–774 (2021).",
    "lapp": "Lappalainen, J. K. et al. Connectome-constrained networks predict neural activity across the fly visual system. <i>Nature</i> <b>634</b>, 1132–1140 (2024).",
    "pospisil": "Pospisil, D. A. et al. The fly connectome reveals a path to the effectome. <i>Nature</i> <b>634</b>, 201–209 (2024).",
    "scheffer": "Scheffer, L. K. et al. A connectome and analysis of the adult <i>Drosophila</i> central brain. <i>eLife</i> <b>9</b>, e57443 (2020).",
    "pugliese": "Pugliese, S. M. et al. Connectome simulations identify a central pattern generator circuit for fly walking. Preprint at bioRxiv https://doi.org/10.1101/2025.09.12.675944 (2026).",
    "beiran": "Beiran, M. &amp; Litwin-Kumar, A. Prediction of neural activity in connectome-constrained recurrent networks. <i>Nat. Neurosci.</i> <b>28</b>, 2561–2574 (2025).",
    "pedigo": "Pedigo, B. D. et al. Generative network modeling reveals quantitative definitions of bilateral symmetry exhibited by a whole insect brain connectome. <i>eLife</i> <b>12</b>, e83739 (2023).",
    "smizell": "Schneider-Mizell, C. M. et al. Quantitative neuroanatomy for connectomics in <i>Drosophila</i>. <i>eLife</i> <b>5</b>, e12059 (2016).",
    "yu": "Yu, S.-C. et al. New synapse detection in the whole-brain connectome of <i>Drosophila</i>. Preprint at bioRxiv https://doi.org/10.1101/2025.07.11.664377 (2025).",
    "bargmann": "Bargmann, C. I. &amp; Marder, E. From the connectome to brain function. <i>Nat. Methods</i> <b>10</b>, 483–490 (2013).",
    "randi": "Randi, F., Sharma, A. K., Dvali, S. &amp; Leifer, A. M. Neural signal propagation atlas of <i>Caenorhabditis elegans</i>. <i>Nature</i> <b>623</b>, 406–414 (2023).",
    "bates": "Bates, A. S. et al. Distributed control circuits across a brain-and-cord connectome. <i>Nature</i> <b>656</b>, 957–970 (2026).",
    "berg": "Berg, S. et al. Sexual dimorphism in the complete <i>Drosophila</i> male central nervous system connectome. <i>Cell</i> <b>189</b>, 5504–5526.e15 (2026).",
    "roy": "Roy, B. et al. Metamorphosis of an identified serotonergic neuron in the <i>Drosophila</i> olfactory system. <i>Neural Dev.</i> <b>2</b>, 20 (2007).",
    "zhang": "Zhang, X. &amp; Gaudry, Q. Functional integration of a serotonergic neuron in the <i>Drosophila</i> antennal lobe. <i>eLife</i> <b>5</b>, e16836 (2016).",
    "nmf": "Wang-Chen, S. et al. NeuroMechFly v2: simulating embodied sensorimotor control in adult <i>Drosophila</i>. <i>Nat. Methods</i> <b>21</b>, 2353–2362 (2024).",
    "brian": "Stimberg, M., Brette, R. &amp; Goodman, D. F. M. Brian 2, an intuitive and efficient neural simulator. <i>eLife</i> <b>8</b>, e47314 (2019).",
    "gordon": "Gordon, M. D. &amp; Scott, K. Motor control in a <i>Drosophila</i> taste circuit. <i>Neuron</i> <b>61</b>, 373–384 (2009).",
    "hampel": "Hampel, S., Franconville, R., Simpson, J. H. &amp; Seeds, A. M. A neural command circuit for grooming movement control. <i>eLife</i> <b>4</b>, e08758 (2015).",
    "repo": "Shiu, P. K. Drosophila_brain_model. GitHub https://github.com/philshiu/Drosophila_brain_model (2024).",
    "engert": "Engert, S., Sterne, G. R., Bock, D. D. &amp; Scott, K. <i>Drosophila</i> gustatory projections are segregated by taste modality and connectivity. <i>eLife</i> <b>11</b>, e78110 (2022).",
    "chen": "Chen, W.-Q. &amp; Xi, W. Whole-brain connectomics of <i>Drosophila</i> reveals a robust, distributed architecture for the suppression of feeding during escape. Preprint at bioRxiv https://doi.org/10.64898/2025.12.14.694122 (2025).",
    "chou": "Chou, Y.-H. et al. Diversity and wiring variability of olfactory local interneurons in the <i>Drosophila</i> antennal lobe. <i>Nat. Neurosci.</i> <b>13</b>, 439–449 (2010).",
}


def compress(nums):
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return ",".join(out)


def build(src="paper_en_src.html", dst="paper_en.html"):
    s = (HERE / src).read_text()
    order = {}
    for m in re.finditer(r"\[\[([a-z0-9_,]+)\]\]", s):
        for k in m.group(1).split(","):
            if k not in REFS:
                raise KeyError(k)
            order.setdefault(k, len(order) + 1)

    def repl(m):
        nums = [order[k] for k in m.group(2).split(",")]
        if m.group(1):
            return m.group(1) + compress(nums)
        return f"<sup>{compress(nums)}</sup>"

    s = re.sub(r"(ref\. |refs\. )?\[\[([a-z0-9_,]+)\]\]", repl, s)
    refs = "<ol class=\"refs\">\n" + "\n".join(f"<li>{REFS[k]}</li>" for k in sorted(order, key=order.get)) + "\n</ol>"
    s = s.replace("{{REFERENCES}}", refs)

    def words(html):
        t = re.sub(r"<figure>.*?</figure>", " ", html, flags=re.S)
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"&[a-z]+;", " ", t)
        return len(t.split())
    a0 = s.index('<p class="summary">')
    summ = words(s[a0:s.index("</p>", a0)])
    main = words(s[s.index("<h2>Main</h2>"):s.index("<h2>Methods</h2>")])
    meth = words(s[s.index("<h2>Methods</h2>"):s.index("<h2>References</h2>")])
    s = s.replace("{{WORDCOUNT}}", f"Article (submission format) · Summary {summ} words · Main text {main:,} words · "
                  f"Methods {meth:,} words · 4 figures · 4 Extended Data figures · {len(order)} references")
    (HERE / dst).write_text(s)
    unused = [k for k in REFS if k not in order]
    print("summary", summ, "main", main, "methods", meth, "refs", len(order), "unused", unused)


if __name__ == "__main__":
    build()
