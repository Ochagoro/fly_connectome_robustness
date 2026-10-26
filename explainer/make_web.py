"""Turn the print explainer (explainer.html) into a responsive, theme-aware web page (web/index.html)."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = (HERE / "explainer.html").read_text()

head_style = s[s.index("<style>") + len("<style>"):s.index("</style>")]
body = s[s.index("<body>") + len("<body>"):s.index("</body>")]

# 1) tokens: light palette on :root, dark palette in both dark blocks
tokens_light = head_style[head_style.index(":root {"):head_style.index("}", head_style.index(":root {")) + 1]
light_extra = """
  --bg: #ffffff;
  --surface: #ffffff;
  --fig-bg: #ffffff;
  --fig-border: #e6e6e6;
  --on-key: #ffffff;
"""
tokens_light = tokens_light.replace("--font:", light_extra.strip() + "\n  --font:")
dark = """
  --key-50: #111c3d;
  --key-100: #16244d;
  --key-200: #2a3c75;
  --key-900: #9db7f9;
  --key-1000: #c5d7fb;
  --text: #eceef2;
  --gray-50: #1c1f26;
  --gray-100: #262a33;
  --gray-200: #3a3f4b;
  --gray-300: #4d5260;
  --gray-536: #a3a8b3;
  --gray-600: #b3b8c2;
  --gray-700: #c4c8d0;
  --gray-800: #dfe2e8;
  --left: #9db7f9;
  --left-bg: #111c3d;
  --right: #ffa66d;
  --right-bg: #3a1e0c;
  --exc: #ff9696;
  --inh: #79e2f2;
  --ok: #71c598;
  --ok-bg: #11301f;
  --ng: #ff9696;
  --ng-bg: #3a1414;
  --warn-bg: #2e2708;
  --warn: #ffd43d;
  --bg: #12151b;
  --surface: #12151b;
  --fig-bg: #f4f5f7;
  --fig-border: #3a3f4b;
  --on-key: #0b1230;
  color-scheme: dark;
"""
theme_css = (tokens_light + "\n@media (prefers-color-scheme: dark) { :root:not([data-theme=\"light\"]) {" + dark + "} }\n"
             + ":root[data-theme=\"dark\"] {" + dark + "}\n")

rest = head_style.replace(tokens_light.replace(light_extra.strip() + "\n  --font:", "--font:"), "")
# drop print-only rules
rest = re.sub(r"@page \{.*?\n\}\n", "", rest, flags=re.S)
rest = rest.replace("@media print {\n  main { max-width: none; padding: 0; }\n}\n", "")
rest = rest.replace("html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }\n", "")
# literal colours -> tokens
rest = rest.replace("background: #ffffff;\n  color: var(--text);", "background: var(--bg);\n  color: var(--text);")
rest = rest.replace(".box.caution { background: #ffffff;", ".box.caution { background: var(--surface);")
rest = rest.replace("border-left: 2px solid #ffffff;", "border-left: 2px solid var(--bg);")
rest = rest.replace(".faq .q .mark { background: var(--key-900); color: #ffffff; }", ".faq .q .mark { background: var(--key-900); color: var(--on-key); }")
rest = rest.replace("color: #ffffff;\n  background: var(--warn);", "color: var(--bg);\n  background: var(--warn);")
# side gutter with safe areas
rest = rest.replace("  padding: 40px 16px 64px;\n}", "  padding-block: 40px 64px;\n  padding-left: max(16px, env(safe-area-inset-left, 0px));\n  padding-right: max(16px, env(safe-area-inset-right, 0px));\n}")

web_css = """
/* web additions: figures sit on a light card in both themes because their drawings use fixed print colours */
figure svg { background: var(--fig-bg); border: 1px solid var(--fig-border); border-radius: 8px; padding: 6px; }
h1, h2, h3 { text-wrap: balance; }
.tbl-wrap { overflow-x: auto; }
.tbl-wrap table { min-width: 520px; }
a { color: var(--key-900); }
:focus-visible { outline: 2px solid var(--key-900); outline-offset: 2px; }
.fig-side > *, .stats > *, dl.glossary div > * { min-width: 0; }
@media (max-width: 640px) {
  h1 { font-size: 24px; }
  h2 { font-size: 21px; }
  .stats { grid-template-columns: 1fr; }
  .toc { columns: 1; }
  .fig-side { grid-template-columns: 1fr; }
  dl.glossary div { grid-template-columns: 1fr; gap: 2px; }
}
"""

head = ('<title>同じ脳の、左と右</title>\n'
        '<meta name="claude:viewport-fit-sides" content="cover">\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap">\n')
out = head + "<style>\n" + theme_css + rest + web_css + "</style>\n" + body
# make every table scroll inside its own wrapper
out = re.sub(r"(?<!<div class=\"tbl-wrap\">\n)(\s*)<table", r"\1<table", out)
(HERE / "web").mkdir(exist_ok=True)
(HERE / "web/index.html").write_text(out)
print("written", len(out), "bytes; tables:", out.count("<table"), "in tbl-wrap:", out.count('class="tbl-wrap"'))
