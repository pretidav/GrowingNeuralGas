import numpy as np
import math
import matplotlib.pyplot as plt

class Box(): 
  def __init__(self,dims,n=200): 
    self.dims = dims 
    self.d = len(dims)
    self.v = []
    self.n = n
    for d in self.dims:
      self.v.append(np.random.uniform(low=0.0,high=d,size=self.n))
    self.v = np.array(self.v)

  def sample(self): 
    i = np.random.randint(low=0,high=self.n-1)
    return self.v[:,i]

  def plot(self,n=200): 
    if self.d==3: 
      ax = plt.axes(projection ="3d");
    else : 
      ax = plt.axes()
    ax.scatter(*self.v, color = "blue", alpha=0.05);
    return ax 

class Circle(): 
  def __init__(self,n=2,d=2): 
    self.n = n
    self.d = d
    self.v = []
    r = np.random.uniform(0,2*math.pi,self.n)
    self.v = np.array([np.cos(r),np.sin(r)])

  def sample(self): 
    i = np.random.randint(low=0,high=self.n)
    return self.v[:,i]

  def plot(self): 
    if self.d==3:
      ax = plt.axes(projection ="3d");
    else : 
      ax = plt.axes()
    ax.scatter(*self.v, color = "blue",alpha=0.02);
    return ax 

class TiltedCircle(): 
  def __init__(self,n=200): 
    self.n = n
    self.v = []
    r = np.random.uniform(0,2*math.pi,self.n)
    self.v = np.array([np.cos(r),np.sin(r),np.cos(3*r)])

  def sample(self): 
    i = np.random.randint(low=0,high=self.n)
    return self.v[:,i]

  def plot(self): 
    ax = plt.axes(projection ="3d");
    ax.scatter(*self.v, color = "blue", alpha=0.02);
    return ax 

class HypPar(): 
  def __init__(self,n=200): 
    self.n = n
    self.v = []
    x,y = np.random.uniform(3,13,n),np.random.uniform(3,13,n)
    self.v = np.array([x,y,x*x+y*y])

  def sample(self): 
    i = np.random.randint(low=0,high=self.n)
    return self.v[:,i]

  def plot(self): 
    ax = plt.axes(projection ="3d");
    ax.scatter(*self.v, color = "blue",alpha=0.07);
    return ax 

if __name__=='__main__': 
  C = Circle(n=1000,d=2)
  C.plot()
  plt.show()

  C = Circle(n=1000,d=3)
  C.plot()
  plt.show()

  C = TiltedCircle(n=1000)
  C.plot()
  plt.show()

  C = Box(dims=[10,5])
  C.plot() 
  plt.show() 

  C = Box(dims=[10,5,2])
  C.plot() 
  plt.show() 

  C = HypPar(n=1000)
  C.plot()
  plt.show()