import math

import matplotlib.pyplot as plt
import numpy as np

from GNG.style import new_figure, scatter_cloud


class Distribution:
    """A finite point cloud ``v`` (one row per coordinate) that is sampled uniformly."""

    d = 2  # dimension of the plot

    def sample(self):
        i = np.random.randint(low=0, high=self.n)
        return self.v[:, i]

    def plot(self, ax=None):
        if ax is None:
            _, ax = new_figure(self.d)
        scatter_cloud(ax, self.v)
        return ax


class Box(Distribution):
    def __init__(self, dims, n=200):
        self.dims = dims
        self.d = len(dims)
        self.n = n
        self.v = np.array([np.random.uniform(low=0.0, high=d, size=n) for d in dims])


class Circle(Distribution):
    def __init__(self, n=2, d=2):
        self.n = n
        self.d = d
        r = np.random.uniform(0, 2 * math.pi, n)
        self.v = np.array([np.cos(r), np.sin(r)])


class TiltedCircle(Distribution):
    d = 3

    def __init__(self, n=200):
        self.n = n
        r = np.random.uniform(0, 2 * math.pi, n)
        self.v = np.array([np.cos(r), np.sin(r), np.cos(3 * r)])


class HypPar(Distribution):
    d = 3

    def __init__(self, n=200):
        self.n = n
        x, y = np.random.uniform(3, 13, n), np.random.uniform(3, 13, n)
        self.v = np.array([x, y, x * x + y * y])


if __name__ == "__main__":
    for C in (
        Circle(n=1000, d=2),
        Circle(n=1000, d=3),
        TiltedCircle(n=1000),
        Box(dims=[10, 5]),
        Box(dims=[10, 5, 2]),
        HypPar(n=1000),
    ):
        C.plot()
        plt.show()
