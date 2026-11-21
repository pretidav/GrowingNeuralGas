import os

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from tqdm import tqdm

from .style import ACCENT, INK, PAPER, legend_below, new_figure


class GraphNeuralGas:
    """Growing Neural Gas of Fritzke, "A Growing Neural Gas Network Learns
    Topologies", NIPS 7 (1994). The comments "step k" refer to the numbered
    steps of the algorithm in the paper.

    Parameter names: growing_rate = lambda, eps_b, eps_n, alpha, a_max, beta = d.

    The stopping criterion uses ``quantization_error``, an estimate of the mean
    squared quantization error per dimension, in the units of the data.
    """

    def __init__(
        self,
        distribution,
        plot_rate=10,
        figpath="./",
        error_tolerance=1e-3,
        max_number=500,
        min_number=100,
        eps_b=0.2,
        eps_n=0.1,
        a_max=50,
        growing_rate=10,
        gas_d=3,
        alpha=0.5,
        beta=0.995,
    ):
        if not 0 < beta < 1:
            raise ValueError("beta (the error decay d of the paper) must be in (0, 1)")
        self.max_number = max_number
        self.min_number = min_number
        self.error_tolerance = error_tolerance
        self.plot_rate = plot_rate
        self.figpath = figpath
        self.distribution = distribution
        self.eps_b = eps_b
        self.eps_n = eps_n
        self.alpha = alpha
        self.beta = beta
        self.a_max = a_max
        self.growing_rate = growing_rate
        self.gas_d = gas_d
        self.gng = ig.Graph()
        self.init()

    # ------------------------------------------------------------------ setup
    def init(self):
        """Step 0: two units at random positions, joined by an edge."""
        for _ in range(2):
            self.gng.add_vertex(
                weight=np.random.uniform(low=-1, high=1, size=self.gas_d), error=0.0
            )
        self.gng.add_edge(0, 1, age=0)

    def embed(self, x):
        """Zero-pad a signal to the dimension of the gas."""
        x = list(x)
        while len(x) < self.gas_d:
            x.append(0)
        return np.array(x, dtype=float)

    # ------------------------------------------------------------------ steps
    def get_nn(self, x):
        """Step 2: winner s1 and runner-up s2 (squared Euclidean distance)."""
        d = np.array(
            [np.dot(weight - x, weight - x) for weight in self.gng.vs["weight"]]
        ).argsort()
        return self.gng.vs[int(d[0])], self.gng.vs[int(d[1])]

    def increment_age(self, s1):
        """Step 3: age of all edges emanating from s1."""
        for edge_id in self.gng.incident(s1.index):
            self.gng.es[edge_id]["age"] += 1

    def increment_error(self, x, s1):
        """Step 4: add the *squared* distance to the error of the winner."""
        diff = s1["weight"] - x
        self.gng.vs[s1.index]["error"] += float(np.dot(diff, diff))

    def update_units(self, x, s1):
        """Step 5: move s1 by eps_b and its direct neighbours by eps_n towards x."""
        s1["weight"] += self.eps_b * (x - s1["weight"])
        for neuron in self.gng.vs[self.gng.neighbors(s1.index)]:
            neuron["weight"] += self.eps_n * (x - neuron["weight"])

    def reset_age(self, s1, s2):
        """Step 6: refresh the edge s1-s2, creating it if it does not exist."""
        edge_id = self.gng.get_eid(s1.index, s2.index, directed=False, error=False)
        if edge_id == -1:
            self.gng.add_edge(s1.index, s2.index, age=0)
        else:
            self.gng.es[edge_id]["age"] = 0

    def remove_units(self):
        """Step 7: remove old edges, then the units left without edges.

        The ids are collected first and deleted in one call: deleting while
        iterating renumbers the graph and skips entries.
        """
        old = [i for i, age in enumerate(self.gng.es["age"]) if age > self.a_max]
        if old:
            self.gng.delete_edges(old)
        dead = [i for i, deg in enumerate(self.gng.degree()) if deg == 0]
        if dead:
            self.gng.delete_vertices(dead)

    def step(self):
        """Steps 1 to 7 for one input signal."""
        xi = self.embed(self.distribution.sample())
        s1, s2 = self.get_nn(x=xi)
        self.increment_age(s1=s1)
        self.increment_error(s1=s1, x=xi)
        self.update_units(s1=s1, x=xi)
        self.reset_age(s1=s1, s2=s2)
        self.remove_units()
        return xi

    # ----------------------------------------------------- step 8: insertion
    def find_q(self):
        """Unit with the largest accumulated error."""
        return self.gng.vs[int(np.argmax(self.gng.vs["error"]))]

    def find_f(self, q):
        """Neighbour of q with the largest error."""
        return max(self.gng.vs[self.gng.neighbors(q.index)], key=lambda v: v["error"])

    def add_neuron(self, q, f):
        """New unit r halfway between q and f."""
        self.gng.add_vertex(weight=0.5 * (q["weight"] + f["weight"]), error=0.0)
        return self.gng.vs[len(self.gng.vs) - 1]

    def update_edges(self, r, f, q):
        """Replace the edge q-f by q-r and r-f, then redistribute the errors."""
        self.gng.delete_edges(self.gng.get_eid(q.index, f.index))
        self.gng.add_edge(q.index, r.index, age=0)
        self.gng.add_edge(r.index, f.index, age=0)
        q["error"] *= self.alpha
        f["error"] *= self.alpha
        r["error"] = q["error"]

    def insert_unit(self):
        q = self.find_q()
        f = self.find_f(q)
        r = self.add_neuron(q, f)
        self.update_edges(r, f, q)

    # ------------------------------------------------------- step 9: decay
    def update_errors(self):
        """Step 9: decay all error variables by beta."""
        self.gng.vs["error"] = [e * self.beta for e in self.gng.vs["error"]]

    # ------------------------------------------------------------ diagnostics
    def get_tot_error(self):
        return float(np.sum(self.gng.vs["error"]))

    def quantization_error(self):
        """Estimate of the mean squared quantization error per dimension.

        With frozen units the expected error variable settles at
        E_c = d/(1-d) D_c, where D_c is the contribution of unit c to the
        quantization error, so sum_c D_c ~ (1-d)/d * sum_c E_c.
        """
        return (1 - self.beta) / self.beta * self.get_tot_error() / self.gas_d

    # ---------------------------------------------------------------- plotting
    def plot(self, step=None, signal=None):
        dim = 3 if self.gas_d >= 3 else 2
        fig, ax = new_figure(dim)
        self.distribution.plot(ax)

        W = np.array(self.gng.vs["weight"])[:, :dim]
        segments = [(W[i], W[j]) for i, j in self.gng.get_edgelist()]
        scale = min(1.0, 60 / len(W))  # smaller markers and lines for large networks
        if dim == 3:
            ax.add_collection3d(
                Line3DCollection(segments, colors=INK, linewidths=max(0.6, 1.6 * scale), zorder=2)
            )
        else:
            ax.add_collection(
                LineCollection(segments, colors=INK, linewidths=max(0.6, 1.6 * scale), zorder=3.5)
            )
        ax.scatter(
            *W.T, marker="s", s=max(4, 36 * scale**2), facecolors=PAPER, edgecolors=ACCENT,
            linewidths=max(1.0, 2 * scale), zorder=3,
            **({"depthshade": False} if dim == 3 else {}),
        )
        if signal is not None:
            ax.scatter(
                *np.asarray(signal)[:dim], marker="s", s=64 * max(scale, 0.5) ** 2, color=INK, zorder=4,
                **({"depthshade": False} if dim == 3 else {}),
            )
        if dim == 2:
            ax.set_aspect("equal", adjustable="box")
        else:  # a flat direction (e.g. a circle in the xy plane) would get absurd tick labels
            setters = (ax.set_xlim, ax.set_ylim, ax.set_zlim)
            lims = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
            widest = max(b - a for a, b in lims)
            for set_lim, (a, b) in zip(setters, lims):
                if b - a < 1e-3 * widest:
                    set_lim((a + b - widest) / 2, (a + b + widest) / 2)
        for label, setter in zip("XYZ"[:dim], ("set_xlabel", "set_ylabel", "set_zlabel")):
            getattr(ax, setter)(label, fontweight="bold", labelpad=6)

        title = "[ GROWING NEURAL GAS ]"
        if step is not None:
            title += "  STEP {}  UNITS {}  ERR {:.3g}".format(
                step, len(self.gng.vs), self.quantization_error()
            )
        legend_below(ax)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.text(0.02, 0.97, title, ha="left", va="top", fontweight="bold", fontsize=10)
        return fig

    def save_plot(self, n, signal):
        os.makedirs(self.figpath, exist_ok=True)
        fig = self.plot(step=n, signal=signal)
        fig.savefig(
            os.path.join(self.figpath, "epoch_{}.png".format(n)), dpi=150, facecolor=PAPER
        )
        plt.close(fig)

    # --------------------------------------------------------------------- run
    def evolve(self, steps):
        for n in tqdm(range(1, steps + 1)):
            x = self.step()
            if n % self.growing_rate == 0:  # step 8
                self.insert_unit()
            self.update_errors()  # step 9
            tot_units = len(self.gng.vs)
            err = self.quantization_error()
            if n % self.plot_rate == 0:
                print("epoch: {} n: {} error/dim: {}".format(n, tot_units, err))
                self.save_plot(n, x)
            if tot_units >= self.max_number or (
                err <= self.error_tolerance and tot_units > self.min_number
            ):
                print("n: {} units: {} error/dim: {}".format(n, tot_units, err))
                break
