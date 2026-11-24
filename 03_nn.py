"""Tasks 10-11: RNN and LSTM, tuned on the validation block (random search), evaluated on test.

Random search over: lag, batch size, learning rate, number of layers, number of neurons.
The SAME sampled configurations are tried for the RNN and the LSTM (fair comparison).
The best configuration is then re-trained with 5 seeds; mean +/- sd of the test metrics is reported.
"""
import time, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from nn_lib import *

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SEEDS = [0, 1, 2, 3, 4]
y, tr, va, te = get_split()
n_tr, n_va, n = len(tr), len(va), len(y)
mu, sd = float(tr.mean()), float(tr.std())
z = ((y - mu) / sd).values
t0 = time.time()

space = dict(lag=[7, 14, 21, 30], batch=[32, 64, 128], lr=[1e-3, 3e-3, 1e-2], layers=[1, 2], hidden=[16, 32, 64])
rng = np.random.default_rng(42)
configs, seen = [], set()
while len(configs) < N_TRIALS:
    c = {k: (float(rng.choice(v)) if k == "lr" else int(rng.choice(v))) for k, v in space.items()}
    key = tuple(c.values())
    if key not in seen:
        seen.add(key)
        configs.append(c)

out, allpred = {}, {}
for kind in ["rnn", "lstm"]:
    rows = []
    for i, c in enumerate(configs):
        t1 = time.time()
        net, hist = fit(kind, c, seed=0, loss_fn=mse, z=z, n_tr=n_tr, n_va=n_va)
        pv = predict(net, z, c["lag"], n_tr, n_tr + n_va, mu, sd).ravel()
        mv = metrics(va, pv)
        rows.append({**c, "epochs": len(hist), "val_RMSE": mv["RMSE"], "val_MAE": mv["MAE"], "secs": time.time() - t1})
        print(f"[{time.time() - t0:6.0f}s] {kind} trial {i + 1}/{N_TRIALS} {c} val_RMSE={mv['RMSE']:.2f} ({len(hist)} ep)", flush=True)
    tbl = pd.DataFrame(rows).sort_values("val_RMSE")
    tbl.to_csv(f"{RES}/nn_search_{kind}.csv", index=False)
    best = {k: tbl.iloc[0][k] for k in space}
    best = {k: (float(v) if k == "lr" else int(v)) for k, v in best.items()}
    # --- final: 5 seeds with the best configuration
    per_seed, hist0 = [], None
    for s in SEEDS:
        net, hist = fit(kind, best, seed=s, loss_fn=mse, z=z, n_tr=n_tr, n_va=n_va)
        if s == 0:
            hist0 = hist
        pv = predict(net, z, best["lag"], n_tr, n_tr + n_va, mu, sd).ravel()
        pt = predict(net, z, best["lag"], n_tr + n_va, n, mu, sd).ravel()
        allpred[f"{kind}_s{s}"] = np.r_[pv, pt]
        per_seed.append({"seed": s, "val": metrics(va, pv), "test": metrics(te, pt), "epochs": len(hist)})
        print(f"[{time.time() - t0:6.0f}s] {kind} final seed {s}: test RMSE={per_seed[-1]['test']['RMSE']:.2f}", flush=True)
    keys = ["RMSE", "MAE", "MAPE_%", "R2"]
    mean = {k: float(np.mean([p["test"][k] for p in per_seed])) for k in keys}
    std = {k: float(np.std([p["test"][k] for p in per_seed], ddof=1)) for k in keys}
    ens = np.mean([allpred[f"{kind}_s{s}"] for s in SEEDS], axis=0)
    out[kind] = {"best_config": best, "per_seed": per_seed, "test_mean": mean, "test_sd": std,
                 "val_mean_RMSE": float(np.mean([p["val"]["RMSE"] for p in per_seed])),
                 "ensemble_test": metrics(te, ens[n_va:]), "ensemble_val": metrics(va, ens[:n_va]),
                 "search_best5": tbl.head(5).round(3).to_dict("records"),
                 "search_range_val_RMSE": [float(tbl.val_RMSE.min()), float(tbl.val_RMSE.max())],
                 "hist_seed0": hist0}
    # plot (seed 0 - fixed in advance, not cherry-picked)
    out[kind]["plot_metrics_seed0"] = pred_plot(te.index, te.values, allpred[f"{kind}_s0"][n_va:],
                                                f"{kind.upper()} (seed 0, lag={best['lag']}, h={best['hidden']}, L={best['layers']})",
                                                f"fig12_{kind}.png")
    save_json(out, "nn.json")
    pd.DataFrame(allpred, index=y.index[n_tr:n_tr + len(next(iter(allpred.values())))]).assign(
        actual=y.iloc[n_tr:].values[:len(next(iter(allpred.values())))]).to_csv(f"{RES}/pred_nn.csv")

# ---- tuning figure + training curves
fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
for kind, c in [("rnn", "#2e6f95"), ("lstm", "#d1495b")]:
    t = pd.read_csv(f"{RES}/nn_search_{kind}.csv")
    ax[0].scatter(t["lag"] + (0.4 if kind == "lstm" else -0.4), t.val_RMSE, color=c, label=kind.upper(), s=18, alpha=.8)
    ax[1].scatter(t["hidden"] * (1.04 if kind == "lstm" else .96), t.val_RMSE, color=c, s=18, alpha=.8)
ax[0].set_xlabel("lag (days)"); ax[0].set_ylabel("validation RMSE"); ax[0].legend(); ax[0].set_title("Tuning trials: lag")
ax[1].set_xscale("log", base=2); ax[1].set_xlabel("neurons"); ax[1].set_title("Tuning trials: neurons")
for kind, c in [("rnn", "#2e6f95"), ("lstm", "#d1495b")]:
    h = np.array(out[kind]["hist_seed0"])
    ax[2].plot(h[:, 0], color=c, ls="--", label=f"{kind.upper()} train"); ax[2].plot(h[:, 1], color=c, label=f"{kind.upper()} val")
ax[2].set_xlabel("epoch"); ax[2].set_ylabel("MSE (standardised)"); ax[2].legend(fontsize=7); ax[2].set_title("Training curves (best config, seed 0)")
fig.tight_layout(); fig.savefig(f"{FIG}/fig13_nn_tuning.png"); plt.close(fig)
print("DONE", flush=True)
