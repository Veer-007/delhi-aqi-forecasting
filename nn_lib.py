"""Small PyTorch helpers shared by the RNN / LSTM / quantile-LSTM scripts."""
import copy
import numpy as np
import torch
import torch.nn as nn
from utils import *

torch.set_num_threads(1)


class Net(nn.Module):
    """Vanilla RNN or LSTM -> last hidden state -> linear head."""

    def __init__(self, kind, hidden, layers, out_dim=1):
        super().__init__()
        cell = nn.LSTM if kind == "lstm" else nn.RNN
        self.rnn = cell(1, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1])


def make_windows(z, lag, start, end):
    """Row i uses z[i-lag : i] (the `lag` PAST days) to predict z[i]. Windows for the validation /
    test blocks reach back into the preceding block - these are past observations, so no leakage."""
    idx = np.arange(start, end)
    X = np.stack([z[i - lag:i] for i in idx])[:, :, None]
    Y = z[idx][:, None]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


def pinball(tau):
    """Pinball (quantile) loss: mean(max(tau*e, (tau-1)*e)), e = y - q_hat."""
    def f(pred, y):
        e = y - pred
        return torch.mean(torch.maximum(tau * e, (tau - 1) * e))
    return f


def mse(pred, y):
    return torch.mean((pred - y) ** 2)


def fit(kind, cfg, seed, loss_fn, z, n_tr, n_va, max_epochs=60, patience=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    lag = cfg["lag"]
    Xtr, Ytr = make_windows(z, lag, lag, n_tr)
    Xva, Yva = make_windows(z, lag, n_tr, n_tr + n_va)
    net = Net(kind, cfg["hidden"], cfg["layers"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    best, best_state, bad, hist = 1e9, None, 0, []
    g = torch.Generator().manual_seed(seed)
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), cfg["batch"]):
            b = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            loss = loss_fn(net(Xtr[b]), Ytr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            trl = loss_fn(net(Xtr), Ytr).item()
            val = loss_fn(net(Xva), Yva).item()
        hist.append((trl, val))
        if val < best - 1e-5:
            best, best_state, bad = val, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, hist


def predict(net, z, lag, start, end, mu, sd):
    X, _ = make_windows(z, lag, start, end)
    net.eval()
    with torch.no_grad():
        return net(X).numpy() * sd + mu
