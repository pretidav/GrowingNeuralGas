"""Picture of the Voronoi cells of a few units, used to explain V_c, s_1 and s_2.

    python -m example.voronoi [--out example/voronoi.png]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from GNG.style import ACCENT, INK, PAPER, new_figure

# hand-picked unit positions in the unit square
UNITS = np.array(
    [
        [0.16, 0.20], [0.42, 0.10], [0.80, 0.22],
        [0.12, 0.62], [0.45, 0.50], [0.76, 0.60],
        [0.30, 0.88], [0.68, 0.90], [0.94, 0.88],
    ]
)
C = 4  # the highlighted unit
XI = np.array([0.57, 0.57])  # the signal


def cell(i, units=UNITS):
    """Voronoi cell of unit i inside [0,1]^2: the square clipped by the bisector with every other unit."""
    poly = [np.array(p) for p in ((0, 0), (1, 0), (1, 1), (0, 1))]
    for j, wj in enumerate(units):
        if j == i:
            continue
        n = wj - units[i]  # keep the points p with p.n <= (|wj|^2 - |wi|^2) / 2
        b = (wj @ wj - units[i] @ units[i]) / 2
        out = []
        for k in range(len(poly)):
            p, q = poly[k], poly[(k + 1) % len(poly)]
            fp, fq = p @ n - b, q @ n - b
            if fp <= 0:
                out.append(p)
            if fp * fq < 0:
                out.append(p + (q - p) * fp / (fp - fq))
        poly = out
    return np.array(poly)


def main(out):
    fig, ax = new_figure(2)
    ax.grid(False)
    for i in range(len(UNITS)):
        P = cell(i)
        if i == C:
            ax.fill(*P.T, color=ACCENT, alpha=0.22, lw=0, zorder=1)
        ax.plot(*np.vstack([P, P[:1]]).T, color=INK, lw=2, zorder=2)

    d = ((UNITS - XI) ** 2).sum(1)
    s1, s2 = np.argsort(d)[:2]
    assert s1 == C, "the signal must lie in the highlighted cell"

    ax.plot(*np.array([XI, UNITS[s1]]).T, color=INK, lw=2, zorder=3)
    ax.plot(*np.array([XI, UNITS[s2]]).T, color=INK, lw=2, ls=(0, (3, 3)), zorder=3)
    ax.scatter(*UNITS.T, marker="s", s=80, facecolors=PAPER, edgecolors=ACCENT, linewidths=2.5, zorder=4)
    ax.scatter(*XI, marker="s", s=90, color=INK, zorder=5)

    kw = dict(fontsize=17, fontweight="bold", zorder=6)
    ax.annotate(r"$\xi$", XI, xytext=(8, 6), textcoords="offset points", **kw)
    ax.annotate(r"$w_{s_1}$", UNITS[s1], xytext=(-34, -20), textcoords="offset points", **kw)
    ax.annotate(r"$w_{s_2}$", UNITS[s2], xytext=(8, 10), textcoords="offset points", **kw)
    ax.text(0.33, 0.37, r"$V_{s_1}$", color=INK, ha="center", **kw)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.text(0.02, 0.97, "[ VORONOI CELLS ]  WINNER s1 = SOLID, RUNNER-UP s2 = DASHED",
             ha="left", va="top", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150, facecolor=PAPER)
    plt.close(fig)
    print("winner", s1, "runner-up", s2, "->", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="example/voronoi.png")
    main(parser.parse_args().out)
