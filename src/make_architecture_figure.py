"""
Figure 1 - system architecture, five layers.

Layer-and-rail layout: every inter-layer link runs along a horizontal rail
rather than a direct diagonal. Diagonals between thirteen boxes produce
unreadable crossings; a rail keeps the reading order strictly top-down.

The two lower layers (ASSURANCE, DELIVERY) are what separate this system from
a prediction pipeline, so they are drawn at the same weight as the rest.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "figure.dpi": 600,
})

COL2 = 7.16
RAIL = "#33404d"

# palette, one per layer
C_DATA = ("#eef2f6", RAIL)
C_EST = ("#dce7f0", "#1f4e79")
C_DEC = ("#cfe0d5", "#256b46")
C_ASR = ("#f7ead9", "#8a5a1a")
C_DEL = ("#e8e2ef", "#5b4b8a")

fig, ax = plt.subplots(figsize=(COL2, 2.62))
ax.set_xlim(0, 100)
ax.set_ylim(0, 70)
ax.axis("off")

BH = 7.2          # box height
XS = (7, 37, 67)  # column left edges
BW = 26


def box(x, y, text, pal, fs=6.8, bold=False, w=BW, h=BH):
    fc, ec = pal
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
        linewidth=0.8, facecolor=fc, edgecolor=ec, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=4, linespacing=1.45,
            fontweight="bold" if bold else "normal")


def vline(x, y1, y2, head=False):
    ax.add_patch(FancyArrowPatch(
        (x, y1), (x, y2), arrowstyle="-|>" if head else "-",
        mutation_scale=7, linewidth=0.7, color=RAIL,
        shrinkA=0, shrinkB=0, zorder=1))


def hline(x1, x2, y):
    ax.plot([x1, x2], [y, y], color=RAIL, lw=0.7, zorder=1)


def arrow_r(x1, x2, y):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                 mutation_scale=7, lw=0.7, color=RAIL, zorder=2))


def layer(y, txt):
    ax.text(1.4, y, txt, fontsize=5.6, color="#7a8894",
            fontweight="bold", va="center", ha="center", rotation=90)


def rail(y_from, y_to, cols=(20, 50, 80), src=(20, 50, 80)):
    """Drop from src columns to a shared rail, then head down into cols."""
    mid = (y_from + y_to) / 2
    for x in src:
        vline(x, y_from, mid)
    hline(min(min(cols), min(src)), max(max(cols), max(src)), mid)
    for x in cols:
        vline(x, mid, y_to, head=True)


# ------------------------------------------------------------------- DATA
Y1 = 58.5
layer(Y1 + BH / 2, "DATA")
box(XS[0], Y1, "dunnhumby weekly panel\n524,926 store-week-UPC rows",
    C_DATA, bold=True)
box(XS[1], Y1, "Schema contract and loader\ncleaning, cost derivation", C_DATA)
box(XS[2], Y1, "Temporal split\ntrain / validation / test", C_DATA)
arrow_r(33, 37, Y1 + BH / 2)
arrow_r(63, 67, Y1 + BH / 2)

# ------------------------------------------------------------- ESTIMATION
Y2 = 45.5
layer(Y2 + BH / 2, "ESTIMATE")
rail(Y1, Y2 + BH, src=(80,))
box(XS[0], Y2, "Price elasticity\nHausman 2SLS + shrinkage", C_EST)
box(XS[1], Y2, "Promotion uplift\ntwo-way fixed-effects DiD", C_EST)
box(XS[2], Y2, "Demand forecast\ngradient boosting + smearing", C_EST)

# --------------------------------------------------------------- DECISION
Y3 = 32.5
layer(Y3 + BH / 2, "DECISION")
rail(Y2, Y3 + BH)
box(XS[0], Y3, "Price optimisation\nLerner rule + guardrails", C_DEC)
box(XS[1], Y3, "Markdown schedule\nbackward induction (DP)", C_DEC)
box(XS[2], Y3, "Replenishment\nforecast-driven order-up-to", C_DEC)

# -------------------------------------------------------------- ASSURANCE
Y4 = 19.5
layer(Y4 + BH / 2, "ASSURE")
rail(Y3, Y4 + BH)
box(XS[0], Y4, "Explanation\nSHAP + exact price formula", C_ASR)
box(XS[1], Y4, "Guardrail and\nfairness audit", C_ASR)
box(XS[2], Y4, "Evidence-graded ledger\nmodelled vs. observed", C_ASR, bold=True)

# --------------------------------------------------------------- DELIVERY
Y5 = 6.8
layer(Y5 + BH / 2, "DELIVER")
for x in (20, 50, 80):
    vline(x, Y4, 15.8)
hline(20, 80, 15.8)
vline(50, 15.8, Y5 + BH, head=True)
box(15, Y5, "Dashboard: recommendations recomputed live from the "
    "stated cost and\nguardrail assumptions, each shown with its evidence grade",
    C_DEL, fs=6.9, w=70, bold=True)

# ----------------------------------------------------- validation envelope
ax.add_patch(FancyBboxPatch(
    (4.2, 2.6), 92.5, 66.0, boxstyle="round,pad=0.2,rounding_size=1.5",
    linewidth=0.7, facecolor="none", edgecolor="#b0bac4",
    linestyle=(0, (4, 3)), zorder=0))
ax.text(50, 4.0, "The identical pipeline is re-run on a schema-matched synthetic "
                 "replica whose elasticity and promotion lift are known by construction",
        ha="center", va="center", fontsize=6.5, style="italic", color="#6b7783")

out = sys.argv[1] if len(sys.argv) > 1 else "fig1_architecture"
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight", pad_inches=0.02)
print(f"wrote {out}.pdf / .png")
