import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt
#final version
CSV_PATH = r"D:/My MQF/sem 3/risk/risk 2/risk project/Master_Loan_Summary.csv"

GRADE_GROUPS = {"AB": ["A", "B"], "C": ["C"], "D": ["D"], "EG": ["E", "F", "G"]}

def parse_dates(df):
    for c in ["origination_date", "last_payment_date", "next_payment_due_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def grade_group(g):
    if pd.isna(g):
        return "NA"
    g = str(g).strip().upper()
    for k, vals in GRADE_GROUPS.items():
        if g in vals:
            return k
    return "OTHER"

def delinq_bucket(x):
    if pd.isna(x):
        return "NA"
    d = float(x)
    if d <= 0:
        return "0"
    if d <= 30:
        return "1_30"
    if d <= 90:
        return "31_90"
    return "90p"

def make_liquidity_vars(df):
    df = df.copy()
    num_cols = ["principal_paid", "interest_paid", "late_fees_paid", "installment", "days_past_due"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["actual_paid"] = df[["principal_paid", "interest_paid", "late_fees_paid"]].sum(axis=1, min_count=1)
    df["scheduled"] = df["installment"]
    df["pay_ratio"] = df["actual_paid"] / df["scheduled"]
    df.loc[~np.isfinite(df["pay_ratio"]), "pay_ratio"] = np.nan
    df["pay_ratio"] = df["pay_ratio"].clip(0.0, 2.0)
    df["shortfall"] = (1.0 - df["pay_ratio"]).clip(lower=0.0)
    df["shortfall_usd"] = df["shortfall"] * df["scheduled"]

    df["grade_group"] = df["grade"].apply(grade_group) if "grade" in df.columns else "NA"
    df["term_m"] = pd.to_numeric(df["term"], errors="coerce")
    df["term_group"] = np.where(df["term_m"] >= 48, "60", "36")
    df["delinq_bucket"] = df["days_past_due"].apply(delinq_bucket) if "days_past_due" in df.columns else "NA"
    df["orig_cohort"] = df["origination_date"].dt.to_period("Q").astype(str) if "origination_date" in df.columns else "NA"

    if "next_payment_due_date" in df.columns:
        df["month"] = df["next_payment_due_date"].dt.to_period("M").dt.to_timestamp()
    elif "last_payment_date" in df.columns:
        df["month"] = df["last_payment_date"].dt.to_period("M").dt.to_timestamp()
    else:
        df["month"] = pd.NaT

    df["segment"] = df["grade_group"].astype(str) + "_" + df["term_group"].astype(str)
    return df

def build_segment_series(df, value_col="shortfall_usd", min_obs_per_month=50):
    d = df.dropna(subset=["month", value_col, "segment"]).copy()
    g = d.groupby(["month", "segment"])
    agg = g.agg(n=("loan_number", "count"), val=(value_col, "sum")).reset_index()
    agg = agg[agg["n"] >= min_obs_per_month].copy()
    wide = agg.pivot(index="month", columns="segment", values="val").sort_index()
    wide = wide.dropna(axis=0, how="any")
    return wide

def rank_to_uniform(x):
    x = np.asarray(x)
    r = stats.rankdata(x, method="average")
    n = len(x)
    u = (r - 0.5) / n
    return np.clip(u, 1e-10, 1 - 1e-10)

def pseudo_obs(X):
    X = np.asarray(X)
    return np.column_stack([rank_to_uniform(X[:, j]) for j in range(X.shape[1])])

def log_c_frank(U, theta):
    U = np.asarray(U)
    if abs(theta) < 1e-12:
        return np.zeros(U.shape[0])
    u, v = U[:, 0], U[:, 1]
    th = float(theta)
    e_th = np.exp(-th)
    eu = np.exp(-th * u)
    ev = np.exp(-th * v)
    num = th * (1.0 - e_th) * eu * ev
    den = ((1.0 - e_th) + (eu - 1.0) * (ev - 1.0)) ** 2
    return np.log(np.maximum(num, 1e-300)) - np.log(np.maximum(den, 1e-300))

def log_c_clayton(U, theta):
    U = np.asarray(U)
    th = float(theta)
    u, v = U[:, 0], U[:, 1]
    t = u ** (-th) + v ** (-th) - 1.0
    return np.log1p(th) + (-1.0 - th) * (np.log(u) + np.log(v)) + (-2.0 - 1.0 / th) * np.log(t)

def log_c_gumbel(U, theta):
    U = np.asarray(U)
    th = float(theta)
    u, v = U[:, 0], U[:, 1]
    x = (-np.log(u)) ** th
    y = (-np.log(v)) ** th
    S = x + y
    A = S ** (1.0 / th)
    C = np.exp(-A)
    lnu = np.log(u)
    lnv = np.log(v)
    t1 = (-lnu) ** (th - 1.0)
    t2 = (-lnv) ** (th - 1.0)
    S_pow = S ** (1.0 / th - 2.0)
    term = (A + th - 1.0) * t1 * t2 * S_pow / (u * v)
    return np.log(np.maximum(C, 1e-300)) + np.log(np.maximum(term, 1e-300))

def inv_tau_clayton(tau):
    tau = np.clip(tau, 1e-10, 0.999999)
    return 2.0 * tau / (1.0 - tau)

def inv_tau_gumbel(tau):
    tau = np.clip(tau, 1e-10, 0.999999)
    return 1.0 / (1.0 - tau)

def tau_frank(theta):
    if abs(theta) < 1e-10:
        return 0.0
    th = float(theta)
    from scipy.integrate import quad
    def integrand(t):
        return t / np.expm1(t)
    val, _ = quad(integrand, 0.0, th, limit=200)
    D1 = val / th
    return 1.0 - 4.0 / th + 4.0 * D1 / th

def inv_tau_frank(tau):
    if abs(tau) < 1e-6:
        return 0.0
    def f(th):
        return tau_frank(th) - tau
    a, b = (-50.0, -1e-4) if tau < 0 else (1e-4, 50.0)
    try:
        return float(brentq(f, a, b, maxiter=200))
    except Exception:
        return 0.0

def fit_frank_mle(U):
    U = np.asarray(U)
    tau, _ = stats.kendalltau(U[:, 0], U[:, 1])
    tau = 0.0 if not np.isfinite(tau) else float(tau)
    th0 = inv_tau_frank(tau)

    def nll(th):
        th = float(th[0])
        ll = log_c_frank(U, th)
        return -np.sum(ll)

    if abs(th0) < 1e-6:
        ll0 = float(np.sum(log_c_frank(U, 0.0)))
        return 0.0, ll0

    res = minimize(nll, x0=np.array([th0]), bounds=[(-50.0, 50.0)], method="L-BFGS-B")
    if not res.success or not np.isfinite(res.fun):
        ll0 = float(np.sum(log_c_frank(U, 0.0)))
        return 0.0, ll0
    return float(res.x[0]), float(-res.fun)

def tail_dependence(clayton_theta=None, gumbel_theta=None):
    lamL = None
    lamU = None
    if clayton_theta is not None and clayton_theta > 0:
        lamL = 2.0 ** (-1.0 / clayton_theta)
    if gumbel_theta is not None and gumbel_theta >= 1:
        lamU = 2.0 - 2.0 ** (1.0 / gumbel_theta)
    return lamL, lamU

def fit_bivariate_all(U_pair):
    n = U_pair.shape[0]
    tau, _ = stats.kendalltau(U_pair[:, 0], U_pair[:, 1])
    tau = 0.0 if not np.isfinite(tau) else float(tau)

    th_c = inv_tau_clayton(max(tau, 0.0))
    th_g = max(inv_tau_gumbel(max(tau, 0.0)), 1.0)
    th_f, ll_f = fit_frank_mle(U_pair)

    ll_c = float(np.sum(log_c_clayton(U_pair, th_c))) if th_c > 0 else float(np.sum(log_c_clayton(U_pair, 1e-6)))
    ll_g = float(np.sum(log_c_gumbel(U_pair, th_g))) if th_g >= 1 else float(np.sum(log_c_gumbel(U_pair, 1.0)))

    aic_f = 2 * 1 - 2 * ll_f
    aic_c = 2 * 1 - 2 * ll_c
    aic_g = 2 * 1 - 2 * ll_g
    best = min([("frank", aic_f), ("clayton", aic_c), ("gumbel", aic_g)], key=lambda x: x[1])[0]

    lamL, lamU = tail_dependence(th_c, th_g)

    return {
        "tau": tau,
        "theta_frank": th_f, "ll_frank": ll_f, "aic_frank": aic_f,
        "theta_clayton": th_c, "ll_clayton": ll_c, "aic_clayton": aic_c,
        "theta_gumbel": th_g, "ll_gumbel": ll_g, "aic_gumbel": aic_g,
        "lambda_L_clayton": lamL, "lambda_U_gumbel": lamU,
        "best": best,
        "n": n
    }

def avg_pairwise_tau(U):
    d = U.shape[1]
    taus = []
    for i in range(d):
        for j in range(i + 1, d):
            t, _ = stats.kendalltau(U[:, i], U[:, j])
            if np.isfinite(t):
                taus.append(float(t))
    return float(np.mean(taus)) if taus else 0.0

def simulate_gumbel_exchangeable(n, d, theta, seed=0):
    rng = np.random.default_rng(seed)
    alpha = 1.0 / float(theta)
    scale = (np.cos(np.pi * alpha / 2.0)) ** (1.0 / alpha)
    W = stats.levy_stable.rvs(alpha, 1.0, loc=0.0, scale=scale, size=n, random_state=rng)
    W = np.maximum(W, 1e-12)
    E = rng.exponential(1.0, size=(n, d))
    U = np.exp(- (E / W[:, None]) ** alpha)
    return np.clip(U, 1e-10, 1 - 1e-10)

def simulate_independent(n, d, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.uniform(size=(n, d)), 1e-10, 1 - 1e-10)

def inv_empirical_from_history(x_hist, u):
    x = np.asarray(x_hist)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.full_like(u, np.nan, dtype=float)
    xs = np.sort(x)
    q = u * (len(xs) - 1)
    lo = np.floor(q).astype(int)
    hi = np.ceil(q).astype(int)
    w = q - lo
    return xs[lo] * (1 - w) + xs[hi] * w

def compute_cfar(wide, n_sims=200000, q=0.99, seed=0):
    X = wide.to_numpy()
    U_hist = pseudo_obs(X)
    tau_bar = avg_pairwise_tau(U_hist)
    theta_g = inv_tau_gumbel(max(tau_bar, 0.0))
    d = X.shape[1]

    Ug = simulate_gumbel_exchangeable(n_sims, d, theta_g, seed=seed)
    Ui = simulate_independent(n_sims, d, seed=seed + 1)

    Xg = np.column_stack([inv_empirical_from_history(X[:, j], Ug[:, j]) for j in range(d)])
    Xi = np.column_stack([inv_empirical_from_history(X[:, j], Ui[:, j]) for j in range(d)])

    Tg = np.nansum(Xg, axis=1)
    Ti = np.nansum(Xi, axis=1)

    cfar_g = float(np.quantile(Tg, q))
    cfar_i = float(np.quantile(Ti, q))
    es_g = float(np.mean(Tg[Tg >= cfar_g]))
    es_i = float(np.mean(Ti[Ti >= cfar_i]))

    return {
        "tau_avg": tau_bar,
        "theta_gumbel": theta_g,
        "q": q,
        "CFaR_indep": cfar_i,
        "CFaR_gumbel": cfar_g,
        "ES_indep": es_i,
        "ES_gumbel": es_g,
        "uplift_CFaR_pct": (cfar_g / cfar_i - 1.0) * 100.0,
        "uplift_ES_pct": (es_g / es_i - 1.0) * 100.0,
        "T_indep": Ti,
        "T_gumbel": Tg
    }
def plot_cfar_es_vs_q(wide, q_grid=None, n_sims=200000, seed=7):
    if q_grid is None:
        q_grid = [0.95, 0.975, 0.99, 0.995]

    rows = []
    for q in q_grid:
        r = compute_cfar(wide, n_sims=n_sims, q=q, seed=seed)
        rows.append({
            "q": q,
            "CFaR_indep": r["CFaR_indep"],
            "CFaR_gumbel": r["CFaR_gumbel"],
            "ES_indep": r["ES_indep"],
            "ES_gumbel": r["ES_gumbel"],
            "uplift_CFaR_pct": r["uplift_CFaR_pct"],
            "uplift_ES_pct": r["uplift_ES_pct"],
            "tau_avg": r["tau_avg"],
            "theta_gumbel": r["theta_gumbel"],
        })

    dfq = pd.DataFrame(rows).sort_values("q")

    # Plot 1: levels
    plt.figure(figsize=(10, 5))
    plt.plot(dfq["q"], dfq["CFaR_indep"], marker="o", label="CFaR indep")
    plt.plot(dfq["q"], dfq["CFaR_gumbel"], marker="o", label="CFaR gumbel")
    plt.plot(dfq["q"], dfq["ES_indep"], marker="o", label="ES indep")
    plt.plot(dfq["q"], dfq["ES_gumbel"], marker="o", label="ES gumbel")
    plt.title("Liquidity tail risk vs confidence level q")
    plt.xlabel("q")
    plt.ylabel("USD")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: uplift curves
    plt.figure(figsize=(10, 4))
    plt.plot(dfq["q"], dfq["uplift_CFaR_pct"], marker="o", label="CFaR uplift (%)")
    plt.plot(dfq["q"], dfq["uplift_ES_pct"], marker="o", label="ES uplift (%)")
    plt.axhline(0.0)
    plt.title("Dependence uplift vs confidence level q")
    plt.xlabel("q")
    plt.ylabel("uplift (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nTail-risk vs q table:")
    print(dfq.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    return dfq

def hill_tail_index(x, k=None):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if len(x) < 50:
        return np.nan, np.nan
    xs = np.sort(x)
    if k is None:
        k = max(20, int(0.05 * len(xs)))
        k = min(k, len(xs) - 5)
    tail = xs[-k:]
    xk = tail[0]
    if xk <= 0:
        return np.nan, np.nan
    gamma = np.mean(np.log(tail) - np.log(xk))
    alpha = 1.0 / gamma if gamma > 0 else np.nan
    return alpha, k

def mean_excess(x, grid_q=(0.7, 0.99), m=40):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return np.array([]), np.array([])
    qs = np.linspace(grid_q[0], grid_q[1], m)
    us = np.quantile(x, qs)
    es = np.array([np.mean(x[x > u] - u) if np.any(x > u) else np.nan for u in us])
    return us, es

def plot_heavy_tail_diagnostics(series, title_prefix):
    x = np.asarray(series)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if len(x) < 20:
        return

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist(x, bins=40)
    ax1.set_title(f"{title_prefix} histogram")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(np.log1p(x), bins=40)
    ax2.set_title(f"{title_prefix} log(1+x) histogram")

    ax3 = fig.add_subplot(2, 2, 3)
    stats.probplot(np.log1p(x), dist="norm", plot=ax3)
    ax3.set_title(f"{title_prefix} QQ: log(1+x) vs Normal")

    ax4 = fig.add_subplot(2, 2, 4)
    u, e = mean_excess(x)
    if len(u) > 0:
        ax4.plot(u, e)
    ax4.set_title(f"{title_prefix} mean excess plot")
    ax4.set_xlabel("threshold u")
    ax4.set_ylabel("E[X-u | X>u]")

    plt.tight_layout()
    plt.show()

    krt = stats.kurtosis(x, fisher=False, nan_policy="omit")
    skw = stats.skew(x, nan_policy="omit")
    alpha, k = hill_tail_index(x)
    print(f"{title_prefix}: n={len(x)}  skew={skw:.3f}  kurtosis={krt:.3f}  Hill alpha~{alpha:.3f} (k={k})")

def plot_time_series(wide):
    plt.figure(figsize=(12, 6))
    for c in wide.columns:
        plt.plot(wide.index, wide[c], label=c)
    plt.title("Monthly total shortfall (USD) by segment")
    plt.xlabel("month")
    plt.ylabel("shortfall_usd (sum)")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

def plot_total_aic(out_df):
    aic_tot = out_df[["aic_frank", "aic_clayton", "aic_gumbel"]].sum()
    plt.figure(figsize=(8, 4))
    plt.bar(["frank", "clayton", "gumbel"], [aic_tot["aic_frank"], aic_tot["aic_clayton"], aic_tot["aic_gumbel"]])
    plt.title("Total AIC across all segment pairs (lower is better)")
    plt.ylabel("AIC total")
    plt.tight_layout()
    plt.show()

def plot_tail_dependence(out_df):
    lamU = out_df["lambda_U_gumbel"].to_numpy()
    lamL = out_df["lambda_L_clayton"].to_numpy()
    plt.figure(figsize=(10, 4))
    plt.hist(lamU[np.isfinite(lamU)], bins=20)
    plt.title("Implied upper-tail dependence (Gumbel) across pairs")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.hist(lamL[np.isfinite(lamL)], bins=20)
    plt.title("Implied lower-tail dependence (Clayton) across pairs")
    plt.tight_layout()
    plt.show()

def plot_cfar_hist(Ti, Tg, q=0.99):
    qi = np.quantile(Ti, q)
    qg = np.quantile(Tg, q)
    plt.figure(figsize=(12, 5))
    plt.hist(Ti, bins=60, alpha=0.6, label="independence")
    plt.hist(Tg, bins=60, alpha=0.6, label="gumbel")
    plt.axvline(qi)
    plt.axvline(qg)
    plt.title(f"Total monthly shortfall simulation; vertical lines = q={q}")
    plt.xlabel("total shortfall (USD)")
    plt.ylabel("frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv(CSV_PATH, low_memory=False, dtype={"listing_title": "string"})
    df = parse_dates(df)
    df = make_liquidity_vars(df)

    wide = build_segment_series(df, value_col="shortfall_usd", min_obs_per_month=50)
    dfq = plot_cfar_es_vs_q(wide, q_grid=[0.95, 0.975, 0.99, 0.995], n_sims=200000, seed=7)

    print("Segments:", list(wide.columns))
    print("Monthly points used:", len(wide))

    plot_time_series(wide)

    total_shortfall = wide.sum(axis=1).to_numpy()
    plot_heavy_tail_diagnostics(total_shortfall, "Total monthly shortfall")

    for c in wide.columns[:3]:
        plot_heavy_tail_diagnostics(wide[c].to_numpy(), f"Segment {c} monthly shortfall")

    U = pseudo_obs(wide.to_numpy())
    segs = list(wide.columns)

    rows = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            fit = fit_bivariate_all(U[:, [i, j]])
            rows.append({
                "seg_i": segs[i],
                "seg_j": segs[j],
                **fit
            })
    out = pd.DataFrame(rows)

    print("\nBest-family counts (AIC):")
    print(out["best"].value_counts().to_string())

    print("\nTotal AIC (lower is better):")
    print(out[["aic_frank", "aic_clayton", "aic_gumbel"]].sum().to_string())

    plot_total_aic(out)
    plot_tail_dependence(out)

    res99 = compute_cfar(wide, n_sims=200000, q=0.99, seed=7)
    print("\nExchangeable Gumbel fit (from avg Kendall tau):")
    print(f"tau_avg={res99['tau_avg']:.6f}  theta_gumbel={res99['theta_gumbel']:.6f}")
    print("\nLiquidity risk metrics on total monthly shortfall:")
    print(f"CFaR(99%) indep={res99['CFaR_indep']:.3f}  gumbel={res99['CFaR_gumbel']:.3f}  uplift%={res99['uplift_CFaR_pct']:.2f}")
    print(f"ES(99%)   indep={res99['ES_indep']:.3f}  gumbel={res99['ES_gumbel']:.3f}  uplift%={res99['uplift_ES_pct']:.2f}")

    plot_cfar_hist(res99["T_indep"], res99["T_gumbel"], q=0.99)

if __name__ == "__main__":
    main()
