from model import *
import time
import torch.utils.data
from torchvision import transforms, datasets
import argparse
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=800)
parser.add_argument('--dset', type=str, default='mnist')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--n_burnin', type=int, default=50)
parser.add_argument('--n_resample_r', type=int, default=50)
parser.add_argument('--n_resample_prior', type=int, default=100)
args = parser.parse_args()

if args.dset == 'mnist':
    # MNIST
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
    triset = datasets.MNIST(root='Data', train=True, download=True, transform=transform_train)
    valset = datasets.MNIST(root='Data', train=False, download=True, transform=transform_test)
elif args.dset == 'fashion-mnist':
    # FashionMNIST
    triset = datasets.FashionMNIST("Data", download=True, train=True, transform=transforms.Compose([transforms.ToTensor()]))
    valset = datasets.FashionMNIST("Data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
else:
    assert False, 'Unrecognized argument dset'

n_tri, n_val = len(triset), len(valset)
bsz_tri, bsz_val = 500, 500 # Fixed in SGHMC paper
trildr = torch.utils.data.DataLoader(triset, batch_size=bsz_tri, shuffle=True, pin_memory=False)
valldr = torch.utils.data.DataLoader(valset, batch_size=bsz_val, shuffle=False, pin_memory=False)


bnn = BNN()
# lr=0.2 * 1e-5
lr = args.lr
n_epochs = args.epochs
burn_in = args.n_burnin
n_resample_r = args.n_resample_r
n_resample_prior = args.n_resample_prior

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
            # Note that before burn-in, there should not be significant differences (SGHMC does not specify this)
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
ckpt_dir = os.path.join('ckpt', args.dset, 'sghmc')
os.makedirs(ckpt_dir, exist_ok=True)
torch.save(tri_err, os.path.join(ckpt_dir, 'tri_err.pt'))
torch.save(val_err, os.path.join(ckpt_dir, 'val_err.pt'))
torch.save(bnn.weights, os.path.join(ckpt_dir, 'weights.pt'))
torch.save(bnn, os.path.join(ckpt_dir, 'bnn.pt'))

