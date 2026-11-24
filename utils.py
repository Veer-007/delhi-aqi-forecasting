"""Shared helpers: data loading, chronological split, metrics, plotting."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "delhi_aqi_daily.csv")
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

TRAIN_FRAC, VAL_FRAC = 0.65, 0.15          # remaining 0.20 = test
THRESHOLD = 300                            # CPCB 'Very Poor' category starts above 300

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})


def load_series():
    return pd.read_csv(DATA, parse_dates=["date"]).set_index("date")


def split_indices(n):
    n_tr = int(round(n * TRAIN_FRAC))
    n_va = int(round(n * VAL_FRAC))
    return n_tr, n_tr + n_va            # train=[0,n_tr) val=[n_tr,n_tr+n_va) test=rest


def get_split():
    df = load_series()
    y = df["aqi"].astype(float)
    a, b = split_indices(len(y))
    return y, y.iloc[:a], y.iloc[a:b], y.iloc[b:]


def metrics(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    e = y - yhat
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    mape = float(np.mean(np.abs(e / y)) * 100)
    r2 = float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))
    return {"RMSE": rmse, "MAE": mae, "MAPE_%": mape, "R2": r2}


def save_json(obj, name):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def pred_plot(idx, y, yhat, title, fname):
    """Two-panel figure: actual vs predicted over time + scatter. Returns metrics."""
    m = metrics(y, yhat)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [3, 1]})
    ax[0].plot(idx, y, color="#333", lw=1, label="Actual")
    ax[0].plot(idx, yhat, color="#d1495b", lw=1, label="Predicted (1-step ahead)")
    ax[0].set_title(f"{title}   RMSE={m['RMSE']:.1f}  MAE={m['MAE']:.1f}  "
                    f"MAPE={m['MAPE_%']:.1f}%  R²={m['R2']:.3f}", fontsize=8.5)
    ax[0].set_ylabel("AQI")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[1].scatter(y, yhat, s=5, alpha=.5, color="#2e6f95")
    lim = [min(np.min(y), np.min(yhat)), max(np.max(y), np.max(yhat))]
    ax[1].plot(lim, lim, "k--", lw=.8)
    ax[1].set_xlabel("Actual")
    ax[1].set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)
    return m
