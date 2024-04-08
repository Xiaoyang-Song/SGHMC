import numpy as np
from matplotlib import pyplot as plt

def naive_hmc(f, grad, M, eps, L, x0, correction):
    rho0 = np.random.normal(0, M**0.5, 1)
    (x, rho) = (x0, rho0)

    rho = rho - 0.5 * eps * grad(x)
    for _ in range(L):
        x = x + eps * (1 / M) * rho
        rho = rho - eps * grad(x)
    rho = rho - 0.5 * eps * grad(x)

    if not correction:
        return x
    else:
        H0 = -f(x0)  -0.5 * rho0**2/M
        Ht = -f(x)  -0.5 * rho**2/M
        p = min(1, np.exp(Ht - H0))
        u = np.random.uniform(0,1)
        if p > u:
            return x
        else:
            return x0

def sghmc(f, grad, M, eps, L, x0, C, B):
    rho0 = np.random.normal(0, M**0.5, 1)
    (x, rho) = (x0, rho0)
    for _ in range(L):
        x = x + eps * (1 / M) * rho
        rho = rho - eps * grad(x) - eps * C * (1 / M) * rho + np.random.normal(0, 1) * (2 * (C-B) * eps)**0.5
    return x
if __name__ == '__main__':
    print("HMC")