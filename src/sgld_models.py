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

# NN Backbone Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)
    
    def forward(self, x):
        out = self.fc1(x)
        out = F.sigmoid(out)
        return self.fc2(out)

# Bayesian Neural Network
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

# SGLD
class SGLD_OPT(torch.optim.Optimizer):
    def __init__(self, parameters, lr=1e-3, alpha=0.99, eps=1e-8, addnoise=True):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, addnoise=addnoise)
        super(SGLD_OPT, self).__init__(parameters, defaults)
    
    def __setstate__(self, state):
        super(SGLD_OPT, self).__setstate__(state)
    
    def step(self):
        for w_group in self.param_groups:
            for p in w_group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = w_group['alpha']
                state['step'] += 1

                square_avg.mul_(alpha).addcmul_(1 - alpha, d_p, d_p)
                avg = square_avg.sqrt().add_(w_group['eps'])
                
                if w_group['addnoise']:
                    langevin_noise = p.data.new(p.data.size()).normal_(mean=0, std=1) / np.sqrt(w_group['lr'])
                    p.data.add_(-w_group['lr'],
                                0.5 * d_p.div_(avg) + langevin_noise / torch.sqrt(avg))
                else:
                    p.data.addcdiv_(-w_group['lr'], 0.5 * d_p, avg)