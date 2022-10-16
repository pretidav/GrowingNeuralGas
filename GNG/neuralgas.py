import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import igraph as ig
import os

class GraphNeuralGas:
    def __init__(
        self,
        distribution,
        plot_rate=10,
        figpath='./',
        error_tolerance=1e-3,
        max_number=500,
        min_number=100,
        eps_b=0.2,
        eps_n=0.1,
        eps_g=0.2,
        a_max=50,
        growing_rate=10,
        gas_d=3,
        alpha=0.5,
        beta=0.995,
    ):
        self.max_number = max_number
        self.min_number = min_number
        self.error_tolerance = error_tolerance
        self.plot_rate = plot_rate
        self.figpath = figpath
        self.distribution = distribution
        self.eps_b = eps_b
        self.eps_n = eps_n
        self.eps_g = eps_g
        self.alpha = alpha
        self.beta = beta
        self.a_max = a_max
        self.growing_rate = growing_rate
        self.gas_d = gas_d
        self.gng = ig.Graph()
        self.init()

    def init(self):
        for _ in range(2):
            self.gng.add_vertex(
                weight=np.random.uniform(low=-1, high=1, size=self.gas_d), error=0
            )
        self.gng.add_edge(0, 1, age=0)

    def plot(self):
        ax = self.distribution.plot()
        for k in self.gng.vs:
            ax.scatter(*np.array(k["weight"]), color="red")
            for l in self.gng.vs[self.gng.neighbors(k.index)]:
                cc = [[k["weight"][j], l["weight"][j]] for j in range(self.gas_d)]
                ax.plot(*np.array(cc), color="red")
        return ax

    def embed(self, x):
        while len(x) < self.gas_d:
            x.append(0)
        return np.array(x)

    def get_nn(self, x):
        d = np.array(
            [np.dot(weight - x, weight - x) for weight in self.gng.vs["weight"]]
        ).argsort()
        s1, s2 = self.gng.vs[d[0]], self.gng.vs[d[1]]
        return s1, s2

    def increment_age(self, s1):
        for edge_id in self.gng.incident(s1.index):
            self.gng.es[edge_id]["age"] += 1

    def increment_error(self, x, s1):
        self.gng.vs[s1.index]["error"] += np.linalg.norm(s1["weight"] - x)

    def update_units(self, x, s1, s2):
        s1["weight"] += self.eps_b * (x - s1["weight"])
        for neuron in self.gng.vs[self.gng.neighbors(s1.index)]:
            neuron["weight"] += self.eps_n * (x - s2["weight"])

    def reset_age(self, s1, s2):
        EDGE_FLAG = self.gng.get_eid(s1.index, s2.index, directed=False, error=False)
        if EDGE_FLAG == -1:  # FLAG for no edge detected
            self.gng.add_edge(s1.index, s2.index, age=0)
        else:
            self.gng.es[EDGE_FLAG]["age"] = 0

    def remove_units(self):
        for edge in self.gng.es:
            if edge["age"] > self.a_max:
                self.gng.delete_edges(edge.index)
        for neuron in self.gng.vs:
            if len(self.gng.incident(neuron)) == 0:
                self.gng.delete_vertices(neuron)

    def step(self):
        xi = self.embed(x=list(self.distribution.sample()))
        s1, s2 = self.get_nn(x=xi)
        self.increment_age(s1=s1)
        self.increment_error(s1=s1, x=xi)
        self.update_units(s1=s1, s2=s2, x=xi)
        self.reset_age(s1=s1, s2=s2)
        self.remove_units()
        return xi

    def update_edges(self, r, f, q):
        self.gng.delete_edges(self.gng.get_eid(q.index, f.index))
        self.gng.add_edge(q.index, r.index, age=0)
        self.gng.add_edge(r.index, f.index, age=0)
        q["error"] *= self.alpha
        f["error"] *= self.alpha
        r["error"] = q["error"]

    def update_errors(self):
        for neuron in self.gng.vs:
            neuron["error"] *= self.beta

    def find_f(self, q):
        error = np.array(
            [
                (neuron["error"], neuron.index)
                for neuron in self.gng.vs[self.gng.neighbors(q.index)]
            ]
        )
        error = np.sort(error, axis=0)
        f = self.gng.vs[int(error[-1, 1])]
        return f, f["weight"]

    def find_q(self):
        error_index = np.array([error for error in self.gng.vs["error"]]).argsort()
        q = self.gng.vs[error_index[-1]]
        return q, q["weight"]

    def add_neuron(self, q, f):
        self.gng.add_vertex(weight=0.5 * (q["weight"] + f["weight"]), error=0)
        r = self.gng.vs[len(self.gng.vs) - 1]
        return r, r["weight"]

    def get_tot_error(self):
        return np.sum([n["error"] for n in self.gng.vs])

    def evolve(self, steps):
        for n in tqdm(range(1, steps + 1)):
            x = self.step()
            if n % self.growing_rate == 0:
                q, _ = self.find_q()
                f, _ = self.find_f(q)
                r, _ = self.add_neuron(q, f)
                self.update_edges(r, f, q)
            tot_units = len(self.gng.vs)
            tot_error = self.get_tot_error() / tot_units / self.gas_d
            if n % self.plot_rate == 0:
                print(
                    "epoch: {} n: {} error/dim/units: {}".format(
                        n, tot_units, tot_error
                    )
                )
                ax = self.plot()
                ax.scatter(*x, color="green")
                plt.title("epoch: {} n: {} error/dim/units: {}".format(n, tot_units, round(tot_error,4)))
                plt.savefig(self.figpath+'/epoch_'+str(n)+'.png')
                plt.clf() 
            if (
                (tot_units >= self.max_number)
                or (tot_error <= self.error_tolerance)
                and (tot_units > self.min_number)
            ):
                print("n: {} error: {}".format(n, tot_error))
                break
