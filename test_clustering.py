from GNG.neuralgas import GraphNeuralGas
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt 

class Distribution(): 
  def __init__(self,sk_data,n=200): 
    self.v, self.y = sk_data(n_samples=n, noise=0.05)
    self.n = len(self.v)
    self.d = np.array(self.v).shape[-1]

  def sample(self): 
    i = np.random.randint(low=0,high=self.n-1)
    return self.v[i]

  def plot(self): 
    if self.d==3: 
      ax = plt.axes(projection ="3d");
    else : 
      ax = plt.axes()
    color_map = {1:'blue',0:'green'}   
    ax.scatter(self.v[:,0],self.v[:,1], color = list(map(color_map.get,self.y)), alpha=0.2);
    return ax 

NG = GraphNeuralGas(
    distribution=Distribution(sk_data=datasets.make_moons,n=500),
    growing_rate=100,
    eps_b=0.2,
    eps_n=0.006,
    a_max=50,
    beta=0.995,
    alpha=0.5,
    plot_rate=100,
    figpath='./example/Moons',
    gas_d=2,
    max_number=500,
    min_number=50,
    error_tolerance=1,
)
NG.evolve(steps=10000)

