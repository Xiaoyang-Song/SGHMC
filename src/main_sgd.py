from model import *
import time
import torch.utils.data
from torchvision import transforms, datasets
import argparse
import matplotlib

# Example commands:
# python src/main_sgd.py
# python src/main_sgd.py --dset='fashion-mnist' --opt='sgd-m' --lr=1e-3

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=800)
parser.add_argument('--opt', type=str, default='sgd')
parser.add_argument('--dset', type=str, default='mnist')
parser.add_argument('--lr', type=float, default=1e-5)
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
# print(n_tri, n_val)

bsz_tri, bsz_val = 500, 500
trildr = torch.utils.data.DataLoader(triset, batch_size=bsz_tri, shuffle=True, pin_memory=False)
valldr = torch.utils.data.DataLoader(valset, batch_size=bsz_val, shuffle=False, pin_memory=False)


model = MLP() # Need to specify model as well for CIFAR10 experiments
n_epochs = args.epochs

if args.opt == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
elif args.opt == 'sgd-m':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99) # Fixed value according to appendix
else:
    assert False, 'Unrecognized argument opt'

epoch, its = 0, 0
tri_l, tri_err, val_l, val_err = np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs)
for i in tqdm(range(n_epochs)):
    model.train()
    n_samples = 0
    for x, y in trildr:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = x.view(x.shape[0], -1)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y, reduction='mean')
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

    # Evaluating on test set 
    with torch.no_grad():
        n_samples = 0
        for x, y in trildr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.view(x.shape[0], -1)
            out = model(x)
            loss = F.cross_entropy(out, y, reduction='sum')

            # Log testing statistics
            preds = out.max(dim=1)[1]
            err = preds.ne(y).sum()

            val_l[i] += loss * len(x) # We compute the entire batch loss (not whole dataset)
            val_err[i] += err
            n_samples += len(x)

        val_l[i] /= n_samples
        val_err[i] /= n_samples
        print(f"Epoch {i}: {val_err[i]}")


# Save models
ckpt_dir = os.path.join('ckpt', args.dset, args.opt)
os.makedirs(ckpt_dir, exist_ok=True)
torch.save(tri_err, os.path.join(ckpt_dir, 'tri_err.pt'))
torch.save(val_err, os.path.join(ckpt_dir, 'val_err.pt'))
torch.save(model.state_dict(), os.path.join(ckpt_dir, 'weights.pt'))