import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import Counter
from collections import defaultdict
from tqdm import tqdm
import copy
from copy import deepcopy as deep

# Setup device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)
    
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        return self.fc2(out)
    
class BNN():
    def __init__(self, n_max_samples=100):
        super().__init__()
        # NN model
        self.model = MLP().to(DEVICE)
        # Sampled weights from posterior
        self.weights = []
        self.n_max_samples = n_max_samples

    def save_weight_samples(self):
        if len(self.weights) >= self.n_max_samples:
            self.weights.pop(0)
        # Save model weights
        self.weights.append(deep(self.model.state_dict()))

    def predict(self, X, Y):
        self.model.eval()
        out = self.model(X)
        loss = F.cross_entropy(out, Y, reduction='sum')
        probs = torch.softmax(out, dim=1).detach().numpy()
        preds = out.max(dim=1)[1]
        err = preds.ne(Y).sum()

        return loss.data, err, probs
    
    # TODO: predict using multiple samples



class SGHMC_OPT(torch.optim.Optimizer):
    def __init__(self, parameters, lr=1e-3, C=0.05, alpha0=0.01, beta0=0):
        super().__init__(parameters, dict())
        # Hyperparameters
        self.alpha0 = alpha0
        self.beta0 = beta0
        # SGHMC discretization step size
        self.lr = lr
        self.C = C
        # Each parameter group needs to have running dictionary to store momentum

    def step(self, resample_r=False, resample_prior=False):
        for w_group in self.param_groups:
            for w in w_group['params']:
                if w.grad is None:
                    continue
                if w not in self.state.keys():
                    # Sampling momentum from normal distribution
                    self.state[w]['r'] = torch.normal(mean=torch.zeros_like(w), std=torch.ones_like(w))

                # Resample priors if needed
                if resample_prior:
                    pass # Note that this is NOT needed for reproducing SGHMC paper results

                # Resample momentum if needed
                if resample_r:
                    self.state[w]['r'] = torch.normal(mean=torch.zeros_like(w), std=torch.ones_like(w))

                # Start SGHMC update
                r = self.state[w]['r']
                # B_hat = 0 # This is the simplest choice as mentioned in the SGHMC paper

                # delta_w = self.lr * r # We use identity matrix for the mass
                # noise = torch.normal(mean=torch.zeros_like(w), std=torch.ones_like(w) * (2*self.lr * (self.C - B_hat))**0.5)
                # delta_r = -self.lr * w.grad.data - self.lr * self.C * r + noise

                # Connect with SGD with momentum (section 3.3)
                delta_w = r # We use identity matrix for the mass
                noise = torch.normal(mean=torch.zeros_like(w), std=torch.ones_like(w) * (2 * self.lr * (self.alpha0 - self.beta0))**0.5)
                delta_r = -self.lr * w.grad.data - self.alpha0 * r + noise

                # Save updated parameters
                self.state[w]['r'] += delta_r
                w.data += delta_w


if __name__ == '__main__':
    print("Model Backbone")
    model = BNN()
    print(model)