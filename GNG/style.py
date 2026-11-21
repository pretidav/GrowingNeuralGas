"""Plot style shared by the gas and the example distributions.

Retro / digital-brutalist: monospace font, hard black box, square markers,
dotted grid, one accent colour. Same look as the figures of the blog.
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PAPER, INK, ACCENT = "#f2efe6", "#000000", "#ff3b00"

RC = {
    "font.family": ["Space Mono", "Monaco", "Courier New", "monospace"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Space Mono",
    "mathtext.it": "Space Mono:italic",
    "mathtext.bf": "Space Mono:bold",
}


def is_3d(ax):
    return hasattr(ax, "get_zlim")


def style_axes(ax):
    """Closed black box, inward ticks, dotted grid, paper background."""
    ax.set_facecolor(PAPER)
    ax.grid(True, which="major", color=INK, lw=0.5, ls=(0, (1, 3)))
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", length=6, width=2, color=INK, labelsize=8)
    ax.locator_params(nbins=5)
    if is_3d(ax):
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor(PAPER)
            axis.pane.set_edgecolor(INK)
            axis.pane.set_alpha(1)
            try:  # private API, but the only way to style the 3d grid
                axis._axinfo["grid"].update(color=INK, linewidth=0.5, linestyle=":")
            except (AttributeError, KeyError):
                pass
    else:
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(3)
            sp.set_color(INK)
            sp.set_capstyle("projecting")


def new_figure(dim=2):
    """Figure with one styled axes, 3d if dim >= 3."""
    plt.rcParams.update(RC)
    fig = plt.figure(figsize=(8, 5), facecolor=PAPER)
    ax = fig.add_subplot(111, projection="3d" if dim >= 3 else None)
    style_axes(ax)
    return fig, ax


def scatter_cloud(ax, v):
    """Draw the data points (rows of v are the coordinates) as small ink squares."""
    step = max(1, v.shape[1] // 2000)  # plot at most ~2000 points
    v = v[:, ::step]
    kw = dict(s=7, marker="s", color=INK, alpha=0.18, linewidths=0, zorder=1)
    if is_3d(ax):
        kw["depthshade"] = False
    return ax.scatter(*v, **kw)


def legend_below(ax):
    handles = [
        Line2D([], [], marker="s", ls="none", ms=5, color=INK, alpha=0.4, label="DATA"),
        Line2D([], [], marker="s", ls="none", ms=8, mfc=PAPER, mec=ACCENT, mew=2, label="UNIT"),
        Line2D([], [], color=INK, lw=1.6, label="EDGE"),
        Line2D([], [], marker="s", ls="none", ms=8, mfc=INK, mec=INK, label="LAST SIGNAL"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04 if is_3d(ax) else -0.17),
        ncol=4,
        fancybox=False,
        framealpha=1,
        edgecolor=INK,
        facecolor=PAPER,
        borderpad=0.7,
        fontsize=8,
    )
    leg.get_frame().set_linewidth(2)
    return leg
