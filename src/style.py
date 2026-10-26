"""Figure style following Nature's artwork guidelines.

- Arial/Helvetica (Liberation Sans is metric-compatible), 5-7 pt text, 8 pt bold
  lower-case panel letters; single column 89 mm, double column 183 mm.
- Line widths 0.5-1 pt, no chart junk, units in brackets-free "Rate (Hz)" form.
- Colours: validated categorical slots (blue, orange, aqua) for at most three
  series per panel; one-hue blue ramp for magnitude; blue-grey-red for signed values.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4
SINGLE, ONE_HALF, DOUBLE = 89 * MM, 120 * MM, 183 * MM

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW, MAGENTA, GREEN, VIOLET, RED = "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#d9d8d4"
LEFT, RIGHT, FIXED = BLUE, ORANGE, AQUA
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.5,
        "axes.titlesize": 7,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": True,
        "figure.dpi": 150,
    })


def panel_label(ax, letter, x=-0.28, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left", color=INK)


def fmt_int(n: int) -> str:
    """Nature style: thousands separated by commas."""
    return f"{n:,}" if abs(n) >= 1000 else str(n)
