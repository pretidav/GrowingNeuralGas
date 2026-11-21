"""Picture of what happens to the cells when one unit moves by delta.

    python -m example.shift [--out example/shift.png]

The old cell of the unit is dashed, the new cells are solid. The thin strips between the old and the new
borders contain the signals that change owner. Their width is proportional to delta (it is exaggerated here).
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from GNG.style import ACCENT, INK, PAPER, new_figure
from example.voronoi import C, UNITS, cell

DELTA = np.array([0.08, 0.055])  # the shift of the unit (exaggerated)


def owners(units, n=900):
    g = (np.arange(n) + 0.5) / n
    X = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    return ((X[:, None, :] - units[None, :, :]) ** 2).sum(-1).argmin(1).reshape(n, n)


def main(out):
    moved = UNITS.copy()
    moved[C] = UNITS[C] + DELTA

    fig, ax = new_figure(2)
    ax.grid(False)

    # strips: the signals that change owner
    old, new = owners(UNITS), owners(moved)
    overlay = np.zeros(old.shape + (4,))
    overlay[(old != C) & (new == C)] = to_rgba(ACCENT, 0.75)  # pass to the unit
    overlay[(old == C) & (new != C)] = to_rgba(INK, 0.45)  # leave the unit
    ax.imshow(overlay, extent=(0, 1, 0, 1), origin="lower", interpolation="nearest", zorder=1)

    for i in range(len(moved)):
        P = cell(i, moved)
        ax.plot(*np.vstack([P, P[:1]]).T, color=INK, lw=2, zorder=3)
    P = cell(C, UNITS)
    ax.plot(*np.vstack([P, P[:1]]).T, color=INK, lw=2, ls=(0, (3, 3)), zorder=4)

    others = np.arange(len(UNITS)) != C
    ax.scatter(*UNITS[others].T, marker="s", s=80, facecolors=PAPER, edgecolors=ACCENT, linewidths=2.5, zorder=5)
    ax.scatter(*UNITS[C], marker="s", s=80, facecolors=PAPER, edgecolors=INK, linewidths=2, zorder=5)
    ax.scatter(*moved[C], marker="s", s=80, color=ACCENT, zorder=6)
    ax.annotate("", xy=moved[C], xytext=UNITS[C],
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8, mutation_scale=10, shrinkA=8, shrinkB=8), zorder=6)

    kw = dict(fontsize=17, fontweight="bold", zorder=7)
    ax.annotate(r"$w_c$", UNITS[C], xytext=(-14, -38), textcoords="offset points", **kw)
    ax.annotate(r"$w_c + \delta$", moved[C], xytext=(-34, 20), textcoords="offset points", **kw)
    ax.annotate(r"$\delta$", (UNITS[C] + DELTA / 2), xytext=(8, -26), textcoords="offset points", **kw)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [
        Patch(facecolor=to_rgba(ACCENT, 0.75), edgecolor=INK, linewidth=1.5, label="PASS TO c"),
        Patch(facecolor=to_rgba(INK, 0.45), edgecolor=INK, linewidth=1.5, label="LEAVE c"),
        plt.Line2D([], [], color=INK, lw=2, ls=(0, (3, 3)), label="OLD BORDER"),
        plt.Line2D([], [], color=INK, lw=2, label="NEW BORDERS"),
    ]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=4, fancybox=False,
                    framealpha=1, edgecolor=INK, facecolor=PAPER, borderpad=0.7, fontsize=8)
    leg.get_frame().set_linewidth(2)
    fig.text(0.02, 0.97, "[ ONE UNIT MOVES BY DELTA ]  THE BORDERS MOVE, AND SOME SIGNALS CHANGE OWNER",
             ha="left", va="top", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150, facecolor=PAPER)
    plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="example/shift.png")
    main(parser.parse_args().out)
