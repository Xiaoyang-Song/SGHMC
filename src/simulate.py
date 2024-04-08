import os
import sys
sys.path.append('src/')
from hmc import *


def function(x): return -2*x**2 + x**4

def posterior(x): return np.exp(-function(x))

def grad(x): return -4*x + 4*x**3

def grad_hat(x): return grad(x) + np.random.normal(loc=0, scale=2, size=x.shape)

def gt(x):
    p = posterior(x)
    return p/sum(p)

if __name__ == '__main__':
    pass