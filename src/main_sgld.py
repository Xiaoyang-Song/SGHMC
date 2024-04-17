from sgld_models import *
import time
import torch.utils.data
from torchvision import transforms, datasets

# MNIST
transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
triset = datasets.MNIST(root='Data', train=True, download=True, transform=transform_train)
valset = datasets.MNIST(root='Data', train=False, download=True, transform=transform_test)
n_tri, n_val = len(triset), len(valset)
bsz_tri, bsz_val = 500, 500 # Fixed in SGHMC paper
trildr = torch.utils.data.DataLoader(triset, batch_size=bsz_tri, shuffle=True, pin_memory=False)
valldr = torch.utils.data.DataLoader(valset, batch_size=bsz_val, shuffle=False, pin_memory=False)

# FashionMNIST (alternative dataset)
# triset = datasets.FashionMNIST("Data", download=True, train=True, transform=transforms.Compose([transforms.ToTensor()]))
# valset = datasets.FashionMNIST("Data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
# n_tri, n_val = len(triset), len(valset)
# bsz_tri, bsz_val = 500, 500 # Fixed in paper
# trildr = torch.utils.data.DataLoader(triset, batch_size=bsz_tri, shuffle=True, pin_memory=False)
# valldr = torch.utils.data.DataLoader(valset, batch_size=bsz_val, shuffle=False, pin_memory=False)

# Settings
bnn = BNN()
lr = 2e-4
n_epochs = 800
burn_in = 50

# Use sampling-based method SGLD to train BNN
optimizer = SGLD_OPT(bnn.model.parameters(), lr=lr)
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

        optimizer.step()

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
        bnn.save_weight_samples()

    # Evaluating on test set 
    with torch.no_grad():
        n_samples = 0
        for x, y in trildr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.view(x.shape[0], -1)
            # Classic prediction 
            # Actually this is used in original paper when comparing results,
            # Though the training follows bayesian route
            loss, err, _ = bnn.predict(x, y)

            # Customized Fully-Bayesian approach after burn-in stage (can set m=1, 2, 5, 10 samples, will increase runtime)
            # Note that before burn-in, there should not be significant differences
            m = 5
            if i >= burn_in + m:
                loss, err, losses, errs, _ = bnn.predict_multiple(x, y, m)

            val_l[i] += loss * len(x) # We compute the entire batch loss (not whole dataset)
            val_err[i] += err
            n_samples += len(x)

        val_l[i] /= n_samples
        val_err[i] /= n_samples
        print(f"Epoch {i}: {val_err[i]}")

# Save models
np.save("MNIST_tri_err.npy", tri_err)
np.save("MNIST_val_err.npy", val_err)
torch.save(bnn.weights, "MNIST_weights.pt")
torch.save(bnn, "MNIST_bnn.pt")

# np.save("FashionMNIST_tri_err.npy", tri_err)
# np.save("FashionMNIST_val_err.npy", val_err)
# torch.save(bnn.weights, "FashionMNIST_weights.pt")
# torch.save(bnn, "FashionMNIST_bnn.pt")