"""Tasks 1-3: data description, decomposition, autocorrelation, chronological split."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from utils import *

df = load_series()
y, tr, va, te = get_split()
out = {}

# ---------- Task 1/3: overview + split ----------
out["n_days"] = len(y)
out["range"] = [str(y.index[0].date()), str(y.index[-1].date())]
out["interpolated_days"] = int(df["interpolated"].sum())
out["split"] = {k: {"n": len(s), "start": str(s.index[0].date()), "end": str(s.index[-1].date()),
                    "mean": float(s.mean()), "std": float(s.std()),
                    "share_gt300": float((s > THRESHOLD).mean()), "days_gt300": int((s > THRESHOLD).sum())}
                for k, s in [("train", tr), ("val", va), ("test", te)]}
out["describe"] = y.describe().to_dict()

fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(y.index, y, lw=.6, color="#333")
for s, c, l in [(tr, "#cfe8cf", "Train"), (va, "#ffe7a8", "Validation"), (te, "#f4c2c2", "Test")]:
    ax.axvspan(s.index[0], s.index[-1], color=c, alpha=.5, label=l)
for t, ls in [(200, ":"), (300, "--"), (400, "-.")]:
    ax.axhline(t, color="grey", lw=.7, ls=ls)
ax.text(y.index[5], 303, "300: 'Very Poor' begins", fontsize=7, color="grey")
ax.set_ylabel("Delhi daily AQI (CPCB)"); ax.set_title("Delhi daily AQI, May 2015 – Oct 2026, with chronological split")
ax.legend(ncol=3, loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/fig01_series_split.png"); plt.close(fig)

# yearly profile
yr = y.groupby(y.index.year).agg(["mean", "max", lambda s: (s > THRESHOLD).mean() * 100, "count"])
yr.columns = ["mean", "max", "pct_days_gt300", "n"]
out["yearly"] = yr.round(1).to_dict("index")
mo = y.groupby(y.index.month).agg(["mean", lambda s: (s > THRESHOLD).mean() * 100])
mo.columns = ["mean", "pct_days_gt300"]
out["monthly"] = mo.round(1).to_dict("index")

# ---------- Task 2: decomposition ----------
stl = STL(y, period=365, robust=True, seasonal=31).fit()
trend, seas, resid = stl.trend, stl.seasonal, stl.resid
out["stl"] = {"var_share_trend": float(trend.var() / y.var()), "var_share_seasonal": float(seas.var() / y.var()),
              "var_share_resid": float(resid.var() / y.var()),
              "seasonal_peak_day": str(seas.groupby(seas.index.dayofyear).mean().idxmax()),
              "seasonal_amplitude": float(seas.groupby(seas.index.dayofyear).mean().max() - seas.groupby(seas.index.dayofyear).mean().min()),
              "trend_first": float(trend.iloc[:365].mean()), "trend_last": float(trend.iloc[-365:].mean())}
fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
for a, s, t, c in zip(ax, [y, trend, seas, resid], ["Observed", "Trend (STL)", "Annual seasonality (STL, period 365)", "Remainder"],
                      ["#333", "#2e6f95", "#d1495b", "#6a994e"]):
    a.plot(s.index, s, lw=.7, color=c); a.set_ylabel(t, fontsize=7.5)
fig.suptitle("STL decomposition of Delhi daily AQI", y=.995)
fig.tight_layout(); fig.savefig(f"{FIG}/fig02_stl.png"); plt.close(fig)

# seasonal profile by calendar month + weekday effect
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
dd = pd.DataFrame({"aqi": y, "m": y.index.month})
ax[0].boxplot([dd[dd.m == m].aqi for m in range(1, 13)], tick_labels=list("JFMAMJJASOND"), showfliers=False)
ax[0].set_title("AQI by calendar month (box = IQR)"); ax[0].set_ylabel("AQI")
rem_dow = resid.groupby(resid.index.dayofweek).mean()
ax[1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rem_dow.values, color="#6a994e")
ax[1].set_title("Mean STL remainder by weekday (AQI units)"); ax[1].axhline(0, color="k", lw=.6)
fig.tight_layout(); fig.savefig(f"{FIG}/fig03_month_weekday.png"); plt.close(fig)
groups = [resid[resid.index.dayofweek == d].values for d in range(7)]
kw = stats.kruskal(*groups)
out["weekday"] = {"mean_remainder": rem_dow.round(2).tolist(), "kruskal_p": float(kw.pvalue),
                  "range": float(rem_dow.max() - rem_dow.min())}

# white-noise check of remainder
lb = acorr_ljungbox(resid, lags=[1, 7, 14, 30], return_df=True)
out["ljungbox_remainder"] = {"lags": [1, 7, 14, 30], "pvalue": lb["lb_pvalue"].tolist(), "stat": lb["lb_stat"].tolist()}
r_acf = acf(resid, nlags=30)
out["remainder_acf"] = {str(k): float(r_acf[k]) for k in [1, 2, 3, 7, 14, 30]}
out["remainder_std"] = float(resid.std())

# ---------- Task 2: ACF / PACF (training data only) ----------
nl = 60
a = acf(tr, nlags=400, fft=True)
p = pacf(tr, nlags=nl, method="ywm")
ci = 1.96 / np.sqrt(len(tr))
out["acf_train"] = {str(k): float(a[k]) for k in [1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 30, 60, 90, 180, 365]}
out["pacf_train"] = {str(k): float(p[k]) for k in range(1, 11)}
out["ci95"] = float(ci)
out["first_lag_acf_below_0.5"] = int(np.argmax(a < 0.5))
out["first_lag_acf_below_ci"] = int(np.argmax(np.abs(a) < ci))
sig_pacf = [k for k in range(1, nl + 1) if abs(p[k]) > ci]
out["pacf_significant_lags"] = sig_pacf

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for a_, vals, lag, t in [(ax[0], a[:nl + 1], nl, "ACF (lags 0–60)"), (ax[1], p, nl, "PACF (lags 0–60)")]:
    a_.vlines(range(lag + 1), 0, vals[:lag + 1], color="#2e6f95", lw=1.2)
    a_.axhline(ci, color="r", ls="--", lw=.7); a_.axhline(-ci, color="r", ls="--", lw=.7)
    a_.set_title(t); a_.set_xlabel("Lag (days)")
ax[2].plot(range(401), a, color="#d1495b", lw=1); ax[2].axhline(ci, color="r", ls="--", lw=.7); ax[2].axhline(-ci, color="r", ls="--", lw=.7)
ax[2].axvline(365, color="grey", ls=":"); ax[2].set_title("ACF up to lag 400 (annual cycle at 365)"); ax[2].set_xlabel("Lag (days)")
fig.tight_layout(); fig.savefig(f"{FIG}/fig04_acf_pacf.png"); plt.close(fig)

# lag plots
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
for a_, k in zip(ax, [1, 7, 30]):
    a_.scatter(tr.values[:-k], tr.values[k:], s=4, alpha=.35, color="#2e6f95")
    a_.set_title(f"lag-{k} scatter, r = {np.corrcoef(tr.values[:-k], tr.values[k:])[0, 1]:.2f}")
    a_.set_xlabel("AQI(t−k)"); a_.set_ylabel("AQI(t)")
fig.tight_layout(); fig.savefig(f"{FIG}/fig05_lagplots.png"); plt.close(fig)

# naive persistence benchmark (needed throughout the report)
naive = y.shift(1).loc[te.index]
out["persistence_test"] = metrics(te, naive)
out["persistence_val"] = metrics(va, y.shift(1).loc[va.index])
# day-to-day change distribution
d1 = y.diff().dropna()
out["daily_change"] = {"std": float(d1.std()), "p95_abs": float(d1.abs().quantile(.95)), "max_abs": float(d1.abs().max())}
save_json(out, "eda.json")
print(json_dump := __import__("json").dumps(out, indent=1, default=float)[:6000])
