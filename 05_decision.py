"""Tasks 12 & 14: from forecasts to actions, value in rupees, and risk.

DECISION (see report, chapter 6)
  Every evening the institution decides whether to run its 'pollution protocol' for the NEXT day
  (indoor activities, purifiers on high, masks for staff on outdoor duty ...).
  A day is 'bad' when the CPCB AQI > 300 (Very Poor / Severe).

  cost of acting on a day                C_ACT   (paid whether or not the day turns out bad)
  cost of a bad day with no protocol     C_MISS  (illness absence, lost productivity, ...)

  >>> C_ACT and C_MISS are ILLUSTRATIVE ASSUMPTIONS. Replace them with your own figures. <<<

Newsvendor logic: act iff P(bad) > C_ACT / C_MISS  <=>  the (1 - C_ACT/C_MISS)-quantile of the
forecast distribution exceeds 300.  A quantile LSTM trained with pinball loss at that tau therefore
gives the cost-optimal rule directly.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from utils import *

C_ACT, C_MISS = 15_000.0, 60_000.0          # INR per day  (ratio 4)
RATIO = C_MISS / C_ACT
TAU_STAR = 1 - C_ACT / C_MISS               # 0.75
y, tr, va, te = get_split()
n_tr, n_va = len(tr), len(va)
nt, nv = len(te), len(va)
cl = pd.read_csv(f"{RES}/pred_classical.csv", index_col=0, parse_dates=True)
nn = pd.read_csv(f"{RES}/pred_nn.csv", index_col=0, parse_dates=True)
q = pd.read_csv(f"{RES}/pred_quantile.csv", index_col=0, parse_dates=True)
yv, yt = va.values, te.values
bad_v, bad_t = yv > THRESHOLD, yt > THRESHOLD
yrs = nt / 365.0


def cost_days(act, bad, c_act=C_ACT, c_miss=C_MISS):
    return c_act * act + c_miss * (bad & ~act)


def tv(frame, col):  # split a val+test column
    return frame[col].values[:nv], frame[col].values[nv:]


def per_seed(frame, cols, rule, bad=bad_t, c_act=C_ACT, c_miss=C_MISS):
    days = np.array([cost_days(rule(frame[c].values[nv:]), bad, c_act, c_miss) for c in cols])
    acts = np.array([rule(frame[c].values[nv:]) for c in cols])
    return days, acts            # shape (seeds, nt)


rows, daycost, acts_store = {}, {}, {}


def register(name, days, acts):
    days, acts = np.atleast_2d(days), np.atleast_2d(acts)
    tot = days.sum(1)
    rows[name] = {"cost_total": float(tot.mean()), "cost_total_sd": float(tot.std(ddof=1)) if len(tot) > 1 else 0.0,
                  "days_acted": float(acts.sum(1).mean()),
                  "missed_bad_days": float(((~acts) & bad_t).sum(1).mean()),
                  "false_alarms": float((acts & ~bad_t).sum(1).mean()),
                  "caught_bad_days": float((acts & bad_t).sum(1).mean())}
    daycost[name] = days.mean(0)
    acts_store[name] = acts


never = np.zeros(nt, bool)
always = np.ones(nt, bool)
pers = (y.shift(1).iloc[n_tr + n_va:].values > THRESHOLD)
register("Never act", cost_days(never, bad_t), never)
register("Always act", cost_days(always, bad_t), always)
register("Persistence (act if today > 300)", cost_days(pers, bad_t), pers)
ar_act = cl["AR"].values[nv:] > THRESHOLD
register("AR(7) point rule", cost_days(ar_act, bad_t), ar_act)
sar_act = cl["SARIMA"].values[nv:] > THRESHOLD
register("Best classical (ARIMA+Fourier) point rule", cost_days(sar_act, bad_t), sar_act)
lcols = [f"lstm_s{s}" for s in range(5)]
d, a = per_seed(nn, lcols, lambda p: p > THRESHOLD)
register("LSTM point rule (act if forecast > 300)", d, a)

# point forecast with decision threshold tuned on validation (fair competitor to the quantile rule)
def tuned_threshold(frame, cols, c_act, c_miss):
    best = None
    for th in range(120, 361, 5):
        c = np.mean([cost_days(frame[col].values[:nv] > th, bad_v, c_act, c_miss).sum() for col in cols])
        if best is None or c < best[1]:
            best = (th, c)
    return best[0]


th_star = tuned_threshold(nn, lcols, C_ACT, C_MISS)
d, a = per_seed(nn, lcols, lambda p: p > th_star)
register(f"LSTM point, threshold tuned on validation ({th_star})", d, a)
ucols = [f"q0.975_s{s}" for s in range(5)]
d, a = per_seed(q, ucols, lambda p: p > THRESHOLD)
register("LSTM upper bound rule (97.5% > 300)", d, a)
cqr_cols = [f"cqr_hi_s{s}" for s in range(5)]
d, a = per_seed(q, cqr_cols, lambda p: p > THRESHOLD)
register("LSTM calibrated upper bound (CQR) > 300", d, a)
tcols = [f"q{TAU_STAR:.3f}_s{s}" for s in range(3)]
d, a = per_seed(q, tcols, lambda p: p > THRESHOLD)
register(f"LSTM cost-optimal quantile rule (tau={TAU_STAR:.2f})", d, a)
register("Oracle (perfect forecast)", cost_days(bad_t, bad_t), bad_t)

tbl = pd.DataFrame(rows).T
best_naive = min(tbl.loc["Never act", "cost_total"], tbl.loc["Always act", "cost_total"])
tbl["per_year_INR"] = tbl.cost_total / yrs
tbl["per_year_INR_sd"] = tbl.cost_total_sd / yrs
tbl["saving_vs_best_static_per_year"] = (best_naive - tbl.cost_total) / yrs
tbl["saving_vs_persistence_per_year"] = (tbl.loc["Persistence (act if today > 300)", "cost_total"] - tbl.cost_total) / yrs
orc = tbl.loc["Oracle (perfect forecast)", "cost_total"]
tbl["value_captured_%"] = (best_naive - tbl.cost_total) / (best_naive - orc) * 100
tbl.to_csv(f"{RES}/decision_table.csv")
print(tbl.round(0).to_string())

# ---- block bootstrap (14-day blocks) of the saving vs persistence and vs best static
rng = np.random.default_rng(7)
B, L = 4000, 14
nb = int(np.ceil(nt / L))
starts_all = np.arange(0, nt - L + 1)


def boot(a, b):
    out = []
    for _ in range(B):
        st = rng.choice(starts_all, nb)
        idx = np.concatenate([np.arange(s, s + L) for s in st])[:nt]
        out.append((a[idx].sum() - b[idx].sum()) / yrs)
    return np.percentile(out, [2.5, 50, 97.5]).tolist(), float(np.mean(np.array(out) > 0))


k_tau = f"LSTM cost-optimal quantile rule (tau={TAU_STAR:.2f})"
k_pt = "LSTM point rule (act if forecast > 300)"
k_per = "Persistence (act if today > 300)"
k_up = "LSTM upper bound rule (97.5% > 300)"
k_cls = "Best classical (ARIMA+Fourier) point rule"
static_best_days = daycost["Never act"] if tbl.loc["Never act", "cost_total"] <= tbl.loc["Always act", "cost_total"] else daycost["Always act"]
boot_res = {}
for nm, (a_, b_) in {"tau_rule_vs_persistence": (daycost[k_per], daycost[k_tau]),
                     "tau_rule_vs_best_static": (static_best_days, daycost[k_tau]),
                     "lstm_point_vs_persistence": (daycost[k_per], daycost[k_pt]),
                     "tau_rule_vs_classical_point": (daycost[k_cls], daycost[k_tau]),
                     "tau_rule_vs_lstm_point": (daycost[k_pt], daycost[k_tau])}.items():
    ci, pg = boot(a_, b_)
    boot_res[nm] = {"CI95": ci, "p_saving_positive": pg}
print(boot_res)

# ---- sensitivity to the cost ratio
sens = []
for r in [1.5, 2, 3, 4, 5, 7, 10, 15]:
    cm = C_ACT * r
    row = {"ratio": r}
    row["never"] = cost_days(never, bad_t, C_ACT, cm).sum() / yrs
    row["always"] = cost_days(always, bad_t, C_ACT, cm).sum() / yrs
    row["persistence"] = cost_days(pers, bad_t, C_ACT, cm).sum() / yrs
    row["lstm_point"] = np.mean([cost_days(nn[c].values[nv:] > THRESHOLD, bad_t, C_ACT, cm).sum() for c in lcols]) / yrs
    th = tuned_threshold(nn, lcols, C_ACT, cm)
    row["lstm_point_tuned"] = np.mean([cost_days(nn[c].values[nv:] > th, bad_t, C_ACT, cm).sum() for c in lcols]) / yrs
    row["lstm_upper97.5"] = np.mean([cost_days(q[c].values[nv:] > THRESHOLD, bad_t, C_ACT, cm).sum() for c in ucols]) / yrs
    tau = 1 - 1 / r
    col = f"q{tau:.3f}_s0"
    if col in q.columns:
        row["lstm_quantile"] = np.mean([cost_days(q[f"q{tau:.3f}_s{s}"].values[nv:] > THRESHOLD, bad_t, C_ACT, cm).sum() for s in range(3)]) / yrs
    row["oracle"] = cost_days(bad_t, bad_t, C_ACT, cm).sum() / yrs
    sens.append(row)
sens = pd.DataFrame(sens)
sens.to_csv(f"{RES}/sensitivity.csv", index=False)
print(sens.round(0).to_string())

# ---- risk: zones from the 95% interval, error quantiles, monthly cost distribution
lo0, hi0 = q["q0.025_s0"].values[nv:], q["q0.975_s0"].values[nv:]
clo, chi = q["cqr_lo_s0"].values[nv:], q["cqr_hi_s0"].values[nv:]
zones = {}
for nm, (l_, h_) in {"raw 95% interval": (lo0, hi0), "CQR-calibrated interval": (clo, chi)}.items():
    red, green = l_ > THRESHOLD, h_ < THRESHOLD
    amber = ~red & ~green
    zones[nm] = {z_: {"days": int(m.sum()), "share_%": float(m.mean() * 100), "bad_rate_%": float(bad_t[m].mean() * 100) if m.sum() else None,
                      "bad_days": int(bad_t[m].sum())} for z_, m in [("red (lower>300)", red), ("amber (straddles 300)", amber), ("green (upper<300)", green)]}
print(zones)

lstm_s0 = nn["lstm_s0"].values[nv:]
err = np.abs(yt - lstm_s0)
errq = {f"p{p}": float(np.percentile(err, p)) for p in [50, 75, 90, 95, 99]}
errq["max"] = float(err.max())
big = pd.Series(yt - lstm_s0, index=te.index)
top_err = big.abs().sort_values(ascending=False).head(10)
top_list = [{"date": str(i.date()), "actual": float(y.loc[i]), "forecast": float(lstm_s0[te.index.get_loc(i)]),
             "yesterday": float(y.shift(1).loc[i])} for i in top_err.index]

months = te.index.to_period("M")
mcost = pd.DataFrame({k: pd.Series(daycost[k], index=te.index).groupby(months).sum() for k in
                      ["Never act", k_per, k_pt, k_up, k_tau, "Oracle (perfect forecast)"]})
mstats = pd.DataFrame({"mean": mcost.mean(), "p95": mcost.quantile(.95), "worst_month": mcost.max()})
mstats.to_csv(f"{RES}/monthly_cost_stats.csv")
print(mstats.round(0))

# false-alarm / miss detail for headline rule
act_t = acts_store[k_tau][0] if acts_store[k_tau].ndim > 1 else acts_store[k_tau]
tp, fp = int((act_t & bad_t).sum()), int((act_t & ~bad_t).sum())
fn, tn = int((~act_t & bad_t).sum()), int((~act_t & ~bad_t).sum())
conf = {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "recall_%": tp / (tp + fn) * 100, "precision_%": tp / (tp + fp) * 100}
act_p = pers
conf_p = {"TP": int((act_p & bad_t).sum()), "FP": int((act_p & ~bad_t).sum()), "FN": int((~act_p & bad_t).sum()), "TN": int((~act_p & ~bad_t).sum())}
conf_p["recall_%"] = conf_p["TP"] / (conf_p["TP"] + conf_p["FN"]) * 100
conf_p["precision_%"] = conf_p["TP"] / (conf_p["TP"] + conf_p["FP"]) * 100

# the 'transition' days: persistence is wrong by construction when the state flips
flip = (bad_t != (y.shift(1).iloc[n_tr + n_va:].values > THRESHOLD))
trans = {"days_where_state_flips": int(flip.sum()),
         "tau_rule_correct_on_flips_%": float((act_t == bad_t)[flip].mean() * 100),
         "persistence_correct_on_flips_%": 0.0,
         "tau_rule_correct_on_non_flips_%": float((act_t == bad_t)[~flip].mean() * 100),
         "persistence_correct_on_non_flips_%": float((pers == bad_t)[~flip].mean() * 100)}

save_json({"assumptions": {"C_ACT": C_ACT, "C_MISS": C_MISS, "ratio": RATIO, "tau_star": TAU_STAR, "threshold": THRESHOLD,
                           "test_days": nt, "test_years": yrs, "bad_days_test": int(bad_t.sum()), "tuned_threshold": th_star},
           "table": tbl.round(1).reset_index().rename(columns={"index": "policy"}).to_dict("records"),
           "bootstrap": boot_res, "zones": zones, "abs_error_quantiles": errq, "top_errors": top_list,
           "monthly_cost": mstats.round(0).reset_index().rename(columns={"index": "policy"}).to_dict("records"),
           "confusion_tau_rule_seed0": conf, "confusion_persistence": conf_p, "flip_analysis": trans,
           "best_static": "Never act" if tbl.loc["Never act", "cost_total"] <= tbl.loc["Always act", "cost_total"] else "Always act"},
          "decision.json")

# ---------------- figures ----------------
fig, ax = plt.subplots(figsize=(10, 4))
order = ["Never act", "Always act", k_per, k_cls, k_pt, k_up, k_tau, "Oracle (perfect forecast)"]
lab = ["Never\nact", "Always\nact", "Persistence\n(reactive)", "Best\nclassical", "LSTM\npoint", "LSTM upper\n97.5%", f"LSTM quantile\nτ={TAU_STAR:.2f}", "Oracle"]
vals = [tbl.loc[o, "per_year_INR"] / 1e5 for o in order]
errs = [tbl.loc[o, "per_year_INR_sd"] / 1e5 for o in order]
cols = ["#aaa", "#aaa", "#8d99ae", "#8d99ae", "#2e6f95", "#2e6f95", "#d1495b", "#6a994e"]
ax.bar(range(len(order)), vals, yerr=errs, color=cols, capsize=3)
for i, v in enumerate(vals):
    ax.text(i, v + 0.15, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks(range(len(order))); ax.set_xticklabels(lab, fontsize=8)
ax.set_ylabel("Expected cost per year (₹ lakh)"); ax.set_title(f"Cost of each decision rule on the test period (C_act=₹{C_ACT:,.0f}, C_miss=₹{C_MISS:,.0f})", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/fig17_policy_costs.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
for col, c, l, ls in [("never", "#aaa", "Never act", "-"), ("always", "#bbb", "Always act", "--"), ("persistence", "#8d99ae", "Persistence rule", "-"),
                      ("lstm_point", "#2e6f95", "LSTM point (>300)", "-"), ("lstm_point_tuned", "#6c9fc0", "LSTM point, tuned threshold", "--"),
                      ("lstm_quantile", "#d1495b", "LSTM quantile rule (tau = 1-1/r)", "-"), ("oracle", "#6a994e", "Oracle", ":")]:
    s_ = sens.dropna(subset=[col])
    ax.plot(s_.ratio, s_[col] / 1e5, ls, color=c, marker="o", ms=3, label=l)
ax.set_xlabel("Cost ratio  r = C_miss / C_act"); ax.set_ylabel("Cost per year (₹ lakh), C_act fixed")
ax.set_title("Sensitivity: how the ranking of rules depends on the cost ratio", fontsize=9); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/fig18_sensitivity.png"); plt.close(fig)

# decision timeline for last winter
zi = (te.index >= "2025-10-15") & (te.index <= "2026-02-28")
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(te.index[zi], yt[zi], color="#222", lw=1.2, label="Actual AQI")
ax.plot(te.index[zi], q[f"q{TAU_STAR:.3f}_s0"].values[nv:][zi], color="#d1495b", lw=1.1, label=f"LSTM {TAU_STAR:.0%}-quantile forecast")
ax.plot(te.index[zi], lstm_s0[zi], color="#2e6f95", lw=1, alpha=.8, label="LSTM point forecast")
a0 = acts_store[k_tau][0]
ax.fill_between(te.index[zi], 0, 520, where=a0[zi], color="#d1495b", alpha=.10, label="Protocol ON (decided the evening before)")
ax.axhline(THRESHOLD, color="grey", ls="--", lw=.8); ax.set_ylim(0, 520)
ax.set_ylabel("AQI"); ax.set_title("Forecast → action, winter 2025-26", fontsize=9); ax.legend(fontsize=7, ncol=2, loc="lower left")
fig.tight_layout(); fig.savefig(f"{FIG}/fig19_decision_timeline.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
w = hi0 - lo0
ax[0].scatter(w, np.abs(yt - (lo0 + hi0) / 2), s=6, alpha=.4, color="#2e6f95"); ax[0].set_xlabel("Interval width (AQI points)")
ax[0].set_ylabel("|actual − interval midpoint|"); ax[0].set_title("Does a wide interval signal a harder day?", fontsize=9)
zn = zones["raw 95% interval"]
ax[1].bar([k.split(" ")[0] for k in zn], [zn[k]["bad_rate_%"] or 0 for k in zn], color=["#d1495b", "#f2a541", "#6a994e"])
for i, k in enumerate(zn):
    ax[1].text(i, (zn[k]["bad_rate_%"] or 0) + 2, f"{zn[k]['days']} days", ha="center", fontsize=8)
ax[1].set_ylabel("% of days with AQI > 300"); ax[1].set_title("Observed bad-day rate by interval zone (test)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/fig20_risk_zones.png"); plt.close(fig)
print("DONE")
