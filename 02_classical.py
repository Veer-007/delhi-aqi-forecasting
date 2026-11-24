"""Tasks 4-9: MA(q), AR(p), unit-root tests, differencing, ARIMA, seasonal models.

Protocol (identical for every model, so results are comparable):
  * parameters are estimated on the TRAIN block only;
  * orders / hyper-parameters are chosen by one-step-ahead RMSE on the VALIDATION block;
  * test predictions are one-step-ahead: the fitted parameters are frozen, but the
    filter is fed the *actual* observations up to day t-1 to predict day t.
"""
import warnings, time, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from utils import *

warnings.filterwarnings("ignore")
y, tr, va, te = get_split()
y.index.freq = "D"
n_tr, n_va, n = len(tr), len(va), len(y)
val_slice, test_slice = slice(n_tr, n_tr + n_va), slice(n_tr + n_va, n)
out, preds = {}, {}
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def fourier(index, K, period=365.25):
    t = np.arange(len(index)) + 0.0
    cols = {}
    for k in range(1, K + 1):
        cols[f"s{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"c{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=index)


def arima_one_step(order, seasonal_order=(0, 0, 0, 0), exog=None, trend=None):
    """fit on train, return one-step-ahead predictions for the full series + fitted result."""
    if trend is None:
        trend = "c" if (order[1] == 0 and seasonal_order[1] == 0) else "n"
    if exog is None:
        mod = SARIMAX(y.iloc[:n_tr], order=order, seasonal_order=seasonal_order, trend=trend)
        res = mod.fit(disp=False, maxiter=200)
        full = res.apply(y, refit=False)
    else:
        mod = SARIMAX(y.iloc[:n_tr], exog=exog.iloc[:n_tr], order=order, seasonal_order=seasonal_order, trend=trend)
        res = mod.fit(disp=False, maxiter=200)
        full = res.apply(y, exog=exog, refit=False)
    pm = full.get_prediction(start=n_tr, end=n - 1, dynamic=False).predicted_mean
    return pm, res


def evaluate(name, pm, label=None):
    """pm = predictions for val+test (index = dates). Returns val & test metrics, stores preds."""
    pv, pt = pm.iloc[:n_va], pm.iloc[n_va:]
    preds[name] = pm
    return {"val": metrics(va, pv), "test": metrics(te, pt)}


# ======================================================================================
# Task 6/7: stationarity (on the training block, as it drives model choice)
# ======================================================================================
def tests(s, label):
    a = adfuller(s, autolag="AIC", regression="c")
    a2 = adfuller(s, autolag="AIC", regression="ct")
    k = kpss(s, regression="c", nlags="auto")
    k2 = kpss(s, regression="ct", nlags="auto")
    return {"series": label, "ADF_c_stat": a[0], "ADF_c_p": a[1], "ADF_c_lags": a[2],
            "ADF_ct_stat": a2[0], "ADF_ct_p": a2[1], "KPSS_c_stat": k[0], "KPSS_c_p": k[1],
            "KPSS_ct_stat": k2[0], "KPSS_ct_p": k2[1], "n": len(s)}


rows = [tests(tr, "AQI (levels)")]
d1 = tr.diff().dropna()
rows.append(tests(d1, "First difference (d=1)"))
d365 = (tr - tr.shift(365)).dropna()
rows.append(tests(d365, "Seasonal difference (lag 365)"))
# deterministic de-seasonalising with 3 Fourier harmonics fitted on TRAIN
F = fourier(y.index, 3)
beta = np.linalg.lstsq(np.c_[np.ones(n_tr), F.iloc[:n_tr].values], tr.values, rcond=None)[0]
seas_fit = pd.Series(np.c_[np.ones(n), F.values] @ beta, index=y.index)
deseas = (y - seas_fit)
rows.append(tests(deseas.iloc[:n_tr], "Fourier-deseasonalised (K=3)"))
stat_tbl = pd.DataFrame(rows)
stat_tbl.to_csv(f"{RES}/stationarity_tests.csv", index=False)
out["stationarity"] = stat_tbl.round(4).to_dict("records")
out["fourier_seasonal_R2_train"] = float(1 - np.var(tr.values - seas_fit.iloc[:n_tr].values) / np.var(tr.values))
log("stationarity done")
print(stat_tbl.round(3).to_string())

fig, ax = plt.subplots(2, 2, figsize=(11, 6), sharex=False)
for a, s, t in [(ax[0, 0], tr, "AQI levels (train)"), (ax[0, 1], d1, "First difference"),
                (ax[1, 0], d365, "Seasonal difference (lag 365)"), (ax[1, 1], deseas.iloc[:n_tr], "Fourier-deseasonalised")]:
    a.plot(s.index, s.values, lw=.5, color="#aaa")
    a.plot(s.index, s.rolling(180).mean(), color="#d1495b", lw=1.2, label="180-day mean")
    a.plot(s.index, s.rolling(180).std(), color="#2e6f95", lw=1.2, label="180-day std")
    a.set_title(t); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/fig06_stationarity.png"); plt.close(fig)

d_choice = 0  # decided after looking at the table; ARIMA grid below tests d=0 and d=1 anyway

# ======================================================================================
# Task 5: AR(p)  (fast OLS)
# ======================================================================================
def ar_one_step(p):
    m = AutoReg(tr.values, lags=p, trend="c").fit()
    c, phi = m.params[0], m.params[1:]
    vals = y.values
    pred = np.array([c + phi @ vals[t - p:t][::-1] for t in range(n_tr, n)])
    return pd.Series(pred, index=y.index[n_tr:]), m


ar_rows = []
for p in [1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 30]:
    pm, m = ar_one_step(p)
    mv = metrics(va, pm.iloc[:n_va])
    ar_rows.append({"p": p, "AIC": m.aic, "BIC": m.bic, "val_RMSE": mv["RMSE"], "val_MAE": mv["MAE"]})
ar_tbl = pd.DataFrame(ar_rows)
ar_tbl.to_csv(f"{RES}/ar_search.csv", index=False)
p_best = int(ar_tbl.loc[ar_tbl.val_RMSE.idxmin(), "p"])
pm, m = ar_one_step(p_best)
out["AR"] = {"p": p_best, "search": ar_tbl.round(3).to_dict("records"),
             "p_by_AIC": int(ar_tbl.loc[ar_tbl.AIC.idxmin(), "p"]), "p_by_BIC": int(ar_tbl.loc[ar_tbl.BIC.idxmin(), "p"]),
             "params": [float(v) for v in m.params], **evaluate("AR", pm)}
log("AR done p=", p_best)
print(ar_tbl.round(2).to_string())

# ======================================================================================
# Task 4: MA(q)  = ARIMA(0,0,q) with constant
# ======================================================================================
ma_rows = []
best = (None, 1e9, None)
for q in [1, 2, 3, 5, 7, 10, 14, 21, 30]:
    pm_q, res = arima_one_step((0, 0, q))
    mv = metrics(va, pm_q.iloc[:n_va])
    ma_rows.append({"q": q, "AIC": res.aic, "BIC": res.bic, "val_RMSE": mv["RMSE"], "val_MAE": mv["MAE"]})
    if mv["RMSE"] < best[1]:
        best = (q, mv["RMSE"], pm_q)
    log("MA q=", q, round(mv["RMSE"], 2))
ma_tbl = pd.DataFrame(ma_rows)
ma_tbl.to_csv(f"{RES}/ma_search.csv", index=False)
q_best = best[0]
out["MA"] = {"q": q_best, "search": ma_tbl.round(3).to_dict("records"),
             "q_by_AIC": int(ma_tbl.loc[ma_tbl.AIC.idxmin(), "q"]), **evaluate("MA", best[2])}

# ======================================================================================
# Task 8: ARIMA(p,d,q) grid on validation
# ======================================================================================
ar_grid = []
best_a = (None, 1e9, None)
for d in [0, 1]:
    for p in [0, 1, 2, 3, 5]:
        for q in [0, 1, 2, 3]:
            if p == 0 and q == 0:
                continue
            try:
                pm_a, res = arima_one_step((p, d, q))
            except Exception as e:                    # noqa
                continue
            mv = metrics(va, pm_a.iloc[:n_va])
            ar_grid.append({"p": p, "d": d, "q": q, "AIC": res.aic, "BIC": res.bic, "val_RMSE": mv["RMSE"], "val_MAE": mv["MAE"]})
            if mv["RMSE"] < best_a[1]:
                best_a = ((p, d, q), mv["RMSE"], pm_a)
    log("ARIMA grid d=", d, "best so far", best_a[0], round(best_a[1], 2))
arima_tbl = pd.DataFrame(ar_grid)
arima_tbl.to_csv(f"{RES}/arima_search.csv", index=False)
out["ARIMA"] = {"order": list(best_a[0]), "order_by_AIC": arima_tbl.loc[arima_tbl.AIC.idxmin(), ["p", "d", "q"]].astype(int).tolist(),
                "top5_val": arima_tbl.sort_values("val_RMSE").head(5).round(3).to_dict("records"), **evaluate("ARIMA", best_a[2])}
# residual diagnostics of chosen ARIMA on test
res_arima = te - preds["ARIMA"].iloc[n_va:]
lb = acorr_ljungbox(res_arima, lags=[1, 7, 14], return_df=True)
out["ARIMA"]["ljungbox_test_residuals_p"] = lb["lb_pvalue"].tolist()
log("ARIMA done", best_a[0])

# ======================================================================================
# Task 9: seasonal models
# ======================================================================================
seas_rows = []
best_s = (None, 1e9, None)
cands = [("weekly SARIMA(1,0,1)(1,0,1,7)", dict(order=(1, 0, 1), seasonal_order=(1, 0, 1, 7), K=0)),
         ("ARIMA(2,0,1) + Fourier K=2 (annual)", dict(order=(2, 0, 1), seasonal_order=(0, 0, 0, 0), K=2)),
         ("ARIMA(2,0,1) + Fourier K=4 (annual)", dict(order=(2, 0, 1), seasonal_order=(0, 0, 0, 0), K=4)),
         ("ARIMA(1,0,1) + Fourier K=3 + weekly (1,0,1,7)", dict(order=(1, 0, 1), seasonal_order=(1, 0, 1, 7), K=3))]
spreds = {}
for name, c in cands:
    ex = fourier(y.index, c["K"]) if c["K"] > 0 else None
    try:
        pm_s, res = arima_one_step(c["order"], c["seasonal_order"], exog=ex, trend="c" if ex is None else "n")
    except Exception as e:                           # noqa
        log("failed", name, e); continue
    mv = metrics(va, pm_s.iloc[:n_va])
    seas_rows.append({"model": name, "AIC": res.aic, "val_RMSE": mv["RMSE"], "val_MAE": mv["MAE"]})
    spreds[name] = pm_s
    if mv["RMSE"] < best_s[1]:
        best_s = (name, mv["RMSE"], pm_s)
    log("seasonal", name, round(mv["RMSE"], 2))
seas_tbl = pd.DataFrame(seas_rows)
seas_tbl.to_csv(f"{RES}/seasonal_search.csv", index=False)
out["SEASONAL"] = {"search": seas_tbl.round(3).to_dict("records"), "chosen": best_s[0], **evaluate("SARIMA", best_s[2])}
for nm, pm_s in spreds.items():
    out["SEASONAL"].setdefault("all_test", {})[nm] = metrics(te, pm_s.iloc[n_va:])

# persistence baseline for reference
pers = y.shift(1).iloc[n_tr:]
out["PERSISTENCE"] = evaluate("Persistence", pers)

# ======================================================================================
# figures + saved predictions
# ======================================================================================
tbl = {}
for key, okey, title, fn in [("MA", "MA", f"MA({q_best})", "fig07_ma.png"), ("AR", "AR", f"AR({p_best})", "fig08_ar.png"),
                       ("ARIMA", "ARIMA", f"ARIMA{tuple(best_a[0])}", "fig10_arima.png"),
                       ("SARIMA", "SEASONAL", "Seasonal: " + str(best_s[0]), "fig11_sarima.png")]:
    pm = preds[key].iloc[n_va:]
    out[okey]["test"] = pred_plot(te.index, te.values, pm.values, title, fn)

pd.DataFrame({k: v for k, v in preds.items()}).assign(actual=y.iloc[n_tr:]).to_csv(f"{RES}/pred_classical.csv")

# model-selection plots
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
ax[0].plot(ma_tbl.q, ma_tbl.val_RMSE, "o-"); ax[0].set_title("MA(q): validation RMSE"); ax[0].set_xlabel("q")
ax[1].plot(ar_tbl.p, ar_tbl.val_RMSE, "o-", color="#d1495b"); ax[1].set_title("AR(p): validation RMSE"); ax[1].set_xlabel("p")
piv = arima_tbl[arima_tbl.d == best_a[0][1]].pivot(index="p", columns="q", values="val_RMSE")
im = ax[2].imshow(piv.values, cmap="viridis_r", aspect="auto")
ax[2].set_xticks(range(len(piv.columns))); ax[2].set_xticklabels(piv.columns); ax[2].set_yticks(range(len(piv.index))); ax[2].set_yticklabels(piv.index)
ax[2].set_title(f"ARIMA(p,{best_a[0][1]},q): validation RMSE"); ax[2].set_xlabel("q"); ax[2].set_ylabel("p"); fig.colorbar(im, ax=ax[2])
fig.tight_layout(); fig.savefig(f"{FIG}/fig09_order_selection.png"); plt.close(fig)

save_json(out, "classical.json")
log("ALL DONE")
print({k: (out[k]["test"] if "test" in out[k] else None) for k in ["MA", "AR", "ARIMA", "SEASONAL", "PERSISTENCE"]})
