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
    def __init__(self, model=MLP().to(DEVICE), n_max_samples=100):
        super().__init__()
        # NN backbone model
        self.model = model
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
        probs = torch.softmax(out, dim=1).detach().cpu().numpy()
        preds = out.max(dim=1)[1]
        err = preds.ne(Y).sum()

        return loss.data, err, probs

    def predict_multiple(self, X, Y, n=5, base_model=MLP().to(DEVICE)):
        model = base_model
        losses, probs, errs = [], [], []
        for i in range(n):
            model.load_state_dict(self.weights[-i-1]) # Extract last few samples
            out = self.model(X)
            loss = F.cross_entropy(out, Y, reduction='sum')
            prob = torch.softmax(out, dim=1).detach().cpu().numpy()
            pred = out.max(dim=1)[1]
            err = pred.ne(Y).sum()

            # Append relevant statistics
            losses.append(loss)
            probs.append(prob)
            errs.append(err)
        
        return np.mean(losses), np.mean(errs), losses, errs, probs



class SGHMC_OPT(torch.optim.Optimizer):
    def __init__(self, parameters, lr=1e-3, C=0.01, alpha0=0.01, beta_hat=0):
        super().__init__(parameters, dict())
        # Hyperparameters used when connecting with SGD momentum
        self.alpha0 = alpha0
        self.beta_hat = beta_hat  
        # Gamma prior info  
        self.a, self.b = 1, 1 
        self.lr = lr # SGHMC discretization step size
        self.C = C # User-specified friction

    def step(self, resample_r=False, resample_prior=False):
        for w_group in self.param_groups:
            for w in w_group['params']:
                if w.grad is None:
                    continue
                if w not in self.state.keys():
                    # Sampling momentum from normal distribution
                    self.state[w]['r'] = torch.normal(mean=torch.zeros_like(w), std=torch.ones_like(w))
                    self.state[w]['decay'] = np.random.gamma(shape=self.a, scale=1 / (self.b), size=None)

                # Resample priors if needed
                if resample_prior:
                    # Gibbs sampling according to Appendix in SGHMC paper
                    alpha = self.a + w.data.nelement() / 2
                    beta = self.b + (w.data ** 2).sum().item() / 2
                    self.state[w]['decay'] = np.random.gamma(shape=alpha, scale=1/beta, size=None)
                
                # Decay if needed
                if self.state[w]['decay'] != 0:
                    w.grad.data += w.data * self.state[w]['decay']

                # Resample momentum if needed
                if resample_r:
                    self.state[w]['r'] = torch.normal(mean=torch.zeros_like(w), std=torch.ones_like(w))

                # Start SGHMC update
                r = self.state[w]['r']
                B_hat = 0 # This is the simplest choice as mentioned in the SGHMC paper (sec 3.3)

                # Algorithm 2 reimplementation
                delta_w = self.lr * r # We use identity matrix for the mass
                noise = torch.normal(mean=torch.zeros_like(w), std=torch.ones_like(w) * (2*self.lr * (self.C - B_hat))**0.5)
                delta_r = -self.lr * w.grad.data - self.lr * self.C * r + noise

                # Variant of Algo 2: when connecting with SGD with momentum (section 3.3)
                # delta_w = r 
                # noise = torch.normal(mean=torch.zeros_like(w), std=torch.ones_like(w) * (2 * self.lr * (self.alpha0 - self.beta_hat))**0.5)
                # delta_r = -self.lr * w.grad.data - self.alpha0 * r + noise

                # Update potential (target w) and momentum
                self.state[w]['r'] += delta_r
                w.data += delta_w


if __name__ == '__main__':
    print("Model Backbone")
    model = BNN()
    print(model)