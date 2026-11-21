"""Regenerates the pictures of the examples.

    python -m example.generate                 # all of them, into example/<name>/
    python -m example.generate Circle Box2d    # only some
    python -m example.generate --out /tmp/gng  # elsewhere

Every picture is saved as epoch_<step>.png every 500 signals. The parameters are the ones of the paper
(lambda=100, eps_b=0.2, eps_n=0.006, alpha=0.5, a_max=50, d=0.995). The point clouds are large (100000
points), so that the network sees a distribution that is close to continuous.
"""
import argparse
import os

import numpy as np

from GNG.neuralgas import GraphNeuralGas
from example.distributions import Box, Circle, HypPar, TiltedCircle

N = 100000
# name: (distribution, dimension of the gas, number of signals)
EXAMPLES = {
    "Circle": (lambda: Circle(n=N, d=2), 2, 13000),
    "Circle3d": (lambda: Circle(n=N, d=3), 3, 8500),
    "TiltedCircle": (lambda: TiltedCircle(n=N), 3, 19000),
    "Box2d": (lambda: Box(dims=[2, 4], n=N), 2, 30500),
    "Box3d": (lambda: Box(dims=[2, 4, 2], n=N), 3, 22000),
    "HypPar3d": (lambda: HypPar(n=N), 3, 27500),
}


def run(name, out, seed=0):
    make, gas_d, steps = EXAMPLES[name]
    np.random.seed(seed)
    NG = GraphNeuralGas(
        distribution=make(),
        growing_rate=100,
        eps_b=0.2,
        eps_n=0.006,
        a_max=50,
        beta=0.995,
        alpha=0.5,
        plot_rate=500,
        figpath=os.path.join(out, name),
        gas_d=gas_d,
        max_number=10**9,  # no early stop: the number of signals is fixed
        min_number=10**9,
        error_tolerance=-1,
    )
    NG.evolve(steps=steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", choices=list(EXAMPLES))
    parser.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()
    for name in args.names or EXAMPLES:
        run(name, args.out)
