from model import *
import time
import torch.utils.data
from torchvision import transforms, datasets
import argparse
import matplotlib

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

bsz_tri, bsz_val = 500, 500
triset = datasets.MNIST(root='Data', train=True, download=True, transform=transform_train)
valset = datasets.MNIST(root='Data', train=False, download=True, transform=transform_test)
n_tri, n_val = len(triset), len(valset)
print(n_tri, n_val)
trildr = torch.utils.data.DataLoader(triset, batch_size=bsz_tri, shuffle=True, pin_memory=False)
valldr = torch.utils.data.DataLoader(valset, batch_size=bsz_val, shuffle=False, pin_memory=False)

bnn = BNN()

lr=1e-3
n_epochs = 100
burn_in = 50
n_resample_r = 50
n_resample_prior = 1e8 # basically we do not resample; prior is fixed according to the original paper

optimizer = SGHMC_OPT(bnn.model.parameters(), lr=lr)

epoch, its = 0, 0
tri_l, tri_err, val_l, val_err = np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs)
for i in tqdm(range(n_epochs)):
    bnn.model.train()
    n_samples = 0
    for x, y in trildr:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = x.view(x.shape[0], -1)
        optimizer.zero_grad()
        out = bnn.model(x)
        loss = F.cross_entropy(out, y, reduction='mean')
        loss = loss * n_tri # Estimate the entire data loss
        loss.backward()

        optimizer.step(resample_r=(its % n_resample_r == 0), resample_prior=(its % n_resample_prior == 0))

        # Log training statistics
        preds = out.max(dim=1)[1]
        err = preds.ne(y).sum()

        its += 1
        tri_l[i] += loss * len(x) / n_tri # We compute the entire batch loss (not whole dataset)
        tri_err[i] += err
        n_samples += len(x)

    tri_l[i] /= n_samples
    tri_err[i] /= n_samples
    print(f"Epoch {i}: {tri_err[i]}")
    # Save BNN weights after burn-in stage
    if i >= burn_in:
        model.save_weight_samples()

    # Test
    n_samples = 0
    for x, y in trildr:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = x.view(x.shape[0], -1)
        loss, err, _ = bnn.predict(x, y)

        val_l[i] += loss * len(x) # We compute the entire batch loss (not mean)
        val_err[i] += err
        n_samples += len(x)

    val_l[i] /= n_samples
    val_err[i] /= n_samples
    print(f"Epoch {i}: {val_err[i]}")
