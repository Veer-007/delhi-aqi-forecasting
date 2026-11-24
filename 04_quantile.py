"""Task 13: probabilistic forecasts with a PAIR of LSTMs trained by back-propagating the pinball loss.

  lower model: tau = 0.025      upper model: tau = 0.975      -> nominal 95 % prediction interval

Evaluation: PICP (share of test values inside the interval) and MPIW (mean interval width).
Extras used in the decision chapter:
  * split-conformal (CQR) calibration of the interval using the validation block;
  * extra single-quantile LSTMs (tau = 0.5 ... 0.9) used by the cost-based decision rule.
"""
import time, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from nn_lib import *

SEEDS = [0, 1, 2, 3, 4]
y, tr, va, te = get_split()
n_tr, n_va, n = len(tr), len(va), len(y)
mu, sd = float(tr.mean()), float(tr.std())
z = ((y - mu) / sd).values
cfg = load_json("nn.json")["lstm"]["best_config"]       # architecture found in Task 11
print("LSTM config re-used for the quantile models:", cfg, flush=True)
t0 = time.time()


def train_quantile(tau, seed):
    z_tau = (tau)   # pinball operates in standardised units; quantile of standardised target
    net, hist = fit("lstm", cfg, seed, pinball(z_tau), z, n_tr, n_va)
    pv = predict(net, z, cfg["lag"], n_tr, n_tr + n_va, mu, sd).ravel()
    pt = predict(net, z, cfg["lag"], n_tr + n_va, n, mu, sd).ravel()
    return np.r_[pv, pt], hist


def pinball_np(y_, q, tau):
    e = np.asarray(y_) - np.asarray(q)
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))


def interval_stats(lo, hi, yy):
    inside = (yy >= lo) & (yy <= hi)
    w = hi - lo
    winkler = w + (2 / 0.05) * (np.maximum(lo - yy, 0) + np.maximum(yy - hi, 0))
    return {"PICP": float(inside.mean() * 100), "MPIW": float(w.mean()),
            "NMPIW_%": float(w.mean() / (y.max() - y.min()) * 100), "Winkler": float(winkler.mean()),
            "below_%": float((yy < lo).mean() * 100), "above_%": float((yy > hi).mean() * 100)}


preds = {}
out = {"cfg": cfg, "per_seed": []}
for s in SEEDS:
    lo, h_lo = train_quantile(0.025, s)
    hi, h_hi = train_quantile(0.975, s)
    crossing = int((lo > hi).sum())
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)           # guard against quantile crossing
    preds[f"q0.025_s{s}"], preds[f"q0.975_s{s}"] = lo, hi
    # --- conformalised quantile regression (CQR) using the validation block only
    yv = va.values
    score = np.maximum(lo[:n_va] - yv, yv - hi[:n_va])
    k = int(np.ceil((n_va + 1) * 0.95))
    qhat = float(np.sort(score)[min(k, n_va) - 1])
    lo_c, hi_c = lo - qhat, hi + qhat
    preds[f"cqr_lo_s{s}"], preds[f"cqr_hi_s{s}"] = lo_c, hi_c
    row = {"seed": s, "crossing_days": crossing, "cqr_qhat": qhat,
           "val": interval_stats(lo[:n_va], hi[:n_va], yv),
           "test": interval_stats(lo[n_va:], hi[n_va:], te.values),
           "test_cqr": interval_stats(lo_c[n_va:], hi_c[n_va:], te.values),
           "pinball_test_lo": pinball_np(te.values, lo[n_va:], .025), "pinball_test_hi": pinball_np(te.values, hi[n_va:], .975),
           "epochs": [len(h_lo), len(h_hi)]}
    out["per_seed"].append(row)
    print(f"[{time.time() - t0:5.0f}s] seed {s}: PICP={row['test']['PICP']:.1f}% MPIW={row['test']['MPIW']:.0f} | "
          f"CQR PICP={row['test_cqr']['PICP']:.1f}% MPIW={row['test_cqr']['MPIW']:.0f} (qhat={qhat:.1f})", flush=True)
    if s == 0:
        out["hist_lo"], out["hist_hi"] = h_lo, h_hi


def agg(key):
    ks = out["per_seed"][0][key].keys()
    return {k: [float(np.mean([r[key][k] for r in out["per_seed"]])), float(np.std([r[key][k] for r in out["per_seed"]], ddof=1))] for k in ks}


out["test_mean_sd"], out["test_cqr_mean_sd"], out["val_mean_sd"] = agg("test"), agg("test_cqr"), agg("val")

# ---- extra single quantiles for the cost-based decision rule (seed 0 and 1)
extra_taus = [0.5, 2 / 3, 0.75, 0.8, 0.9]
for tau in extra_taus:
    for s in [0, 1, 2]:
        p, _ = train_quantile(tau, s)
        preds[f"q{tau:.3f}_s{s}"] = p
    print(f"[{time.time() - t0:5.0f}s] extra quantile {tau:.3f} done", flush=True)
out["extra_taus"] = extra_taus

# ---- where does the interval fail / how informative is its width?
lo0, hi0 = preds["q0.025_s0"][n_va:], preds["q0.975_s0"][n_va:]
yt = te.values
width = hi0 - lo0
absmid = np.abs(yt - (lo0 + hi0) / 2)
out["width_vs_abs_error_spearman"] = float(stats.spearmanr(width, absmid)[0])
winter = te.index.month.isin([11, 12, 1])
out["PICP_winter_Nov_Jan"] = float(((yt >= lo0) & (yt <= hi0))[winter].mean() * 100)
out["PICP_other"] = float(((yt >= lo0) & (yt <= hi0))[~winter].mean() * 100)
out["MPIW_winter"], out["MPIW_other"] = float(width[winter].mean()), float(width[~winter].mean())
terc = pd.qcut(width, 3, labels=["narrow", "medium", "wide"])
out["width_tercile"] = {str(t): {"mean_width": float(width[terc == t].mean()),
                                 "mean_abs_err_to_midpoint": float(absmid[terc == t].mean()),
                                 "PICP": float(((yt >= lo0) & (yt <= hi0))[terc == t].mean() * 100)} for t in ["narrow", "medium", "wide"]}
save_json(out, "quantile.json")
pd.DataFrame(preds, index=y.index[n_tr:]).assign(actual=y.iloc[n_tr:].values).to_csv(f"{RES}/pred_quantile.csv")

# ---- figures
def band_plot(lo, hi, title, fname, seg=None):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    idx = te.index
    ax.fill_between(idx, lo, hi, color="#2e6f95", alpha=.28, label="95% prediction interval")
    ax.plot(idx, yt, color="#222", lw=.9, label="Actual")
    out_ = (yt < lo) | (yt > hi)
    ax.scatter(idx[out_], yt[out_], s=14, color="#d1495b", zorder=5, label=f"Outside interval ({out_.sum()} days)")
    ax.axhline(THRESHOLD, color="grey", ls="--", lw=.8)
    ax.set_ylabel("AQI"); ax.set_title(title, fontsize=9); ax.legend(loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(f"{FIG}/{fname}"); plt.close(fig)


r0 = out["per_seed"][0]
band_plot(lo0, hi0, f"Quantile-LSTM pair (2.5%/97.5%), seed 0:  PICP = {r0['test']['PICP']:.1f}%,  MPIW = {r0['test']['MPIW']:.0f} AQI points", "fig14_interval.png")
band_plot(preds["cqr_lo_s0"][n_va:], preds["cqr_hi_s0"][n_va:],
          f"Same pair after conformal (CQR) calibration on validation: PICP = {r0['test_cqr']['PICP']:.1f}%,  MPIW = {r0['test_cqr']['MPIW']:.0f}", "fig15_interval_cqr.png")
# zoom into last winter
zi = (te.index >= "2025-10-15") & (te.index <= "2026-02-28")
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.fill_between(te.index[zi], lo0[zi], hi0[zi], color="#2e6f95", alpha=.28, label="95% interval")
ax.plot(te.index[zi], yt[zi], color="#222", lw=1.1, label="Actual")
ax.axhline(THRESHOLD, color="grey", ls="--", lw=.8); ax.set_title("Zoom: winter 2025-26 (the season that matters for decisions)", fontsize=9)
ax.legend(fontsize=8); ax.set_ylabel("AQI"); fig.tight_layout(); fig.savefig(f"{FIG}/fig16_interval_zoom.png"); plt.close(fig)
print("DONE", flush=True)
