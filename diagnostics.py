"""
diagnostics.py
==============
Validation and error assessment for the fitted model.

Contents
--------
1. Residual analysis          -- which states the model misses, and whether
                                 the misses form a pattern
2. Leave-one-out validation   -- honest out-of-sample error, since fitting and
                                 evaluating on the same 50 states inflates
                                 apparent accuracy
3. Error propagation          -- Monte Carlo propagation of census uncertainty
                                 into prediction uncertainty
4. Objective sensitivity      -- how far the fitted elasticities move when the
                                 definition of error changes
5. Figures                    -- best-fit curve and Predicted-Observed plot

Reference: Altac (2024), Ch. 1 (Errors), Ch. 7 (Least Squares)
BSIT 400 -- CLO 1, CLO 6
"""

import numpy as np

from regression_engine import PowerLawRegression


# ---------------------------------------------------------------------------
# 1. Residual analysis
# ---------------------------------------------------------------------------

def residual_table(states, pop, gdp, ports, model, top=8):
    """
    Percentage residuals, sorted, with the extremes at each end.

    A positive value means the model predicted too few ports for that state.
    Systematic structure in these extremes indicates an omitted variable
    rather than random noise.
    """
    y = np.asarray(ports, dtype=float)
    pred = model.predict(pop, gdp)
    pct = (y - pred) / y * 100.0
    order = np.argsort(-pct)

    rows = []
    for i in order:
        rows.append({"state": states[i], "observed": y[i],
                     "predicted": pred[i], "pct": pct[i]})
    return rows, {"under": [rows[i] for i in range(top)],
                  "over": [rows[-(i + 1)] for i in range(top)][::-1]}


# ---------------------------------------------------------------------------
# 2. Leave-one-out cross-validation
# ---------------------------------------------------------------------------

def leave_one_out(pop, gdp, ports, objective="log"):
    """
    Refit the model 50 times, each time holding out one state, and predict the
    held-out state from a model that never saw it.

    This is the honest error estimate. In-sample error measures how well the
    model describes the data it was fitted to, which is always optimistic. The
    gap between the two quantifies overfitting. With three parameters and 50
    observations the gap should be small; a large gap would indicate the model
    is memorising rather than generalising.
    """
    pop = np.asarray(pop, dtype=float)
    gdp = np.asarray(gdp, dtype=float)
    y = np.asarray(ports, dtype=float)
    n = len(y)

    errors = np.zeros(n)
    thetas = np.zeros((n, 3))

    for k in range(n):
        keep = np.ones(n, dtype=bool)
        keep[k] = False

        m = PowerLawRegression(objective=objective)
        m.fit(pop[keep], gdp[keep], y[keep])

        pred_k = float(m.predict(pop[k], gdp[k]))
        errors[k] = abs((y[k] - pred_k) / y[k]) * 100.0
        thetas[k] = m.theta

    return {
        "errors": errors,
        "median_pct": float(np.median(errors)),
        "mape": float(errors.mean()),
        "theta_spread": thetas.std(axis=0),
        "thetas": thetas,
    }


# ---------------------------------------------------------------------------
# 3. Error propagation
# ---------------------------------------------------------------------------

def propagate_census_error(pop, gdp, ports, rel_error=0.01, trials=300,
                           seed=400, objective="log"):
    """
    Monte Carlo propagation of population uncertainty into the fitted model.

    Census population figures are estimates carrying sampling error. Each
    trial perturbs every population value by a uniform relative error, refits
    the model from scratch, and records the resulting elasticities and
    predictions. The spread across trials is the uncertainty the data error
    induces in the output.

    This is distinct from round-off error, which arises from finite-precision
    arithmetic and is addressed by column scaling. Data error cannot be
    removed by any algorithm and must be reported as an uncertainty band.
    """
    pop = np.asarray(pop, dtype=float)
    gdp = np.asarray(gdp, dtype=float)
    y = np.asarray(ports, dtype=float)
    rng = np.random.default_rng(seed)

    base = PowerLawRegression(objective=objective).fit(pop, gdp, y)
    base_pred = base.predict(pop, gdp)

    thetas, pred_rel = [], []
    for _ in range(trials):
        noise = 1.0 + rng.uniform(-rel_error, rel_error, size=pop.shape)
        m = PowerLawRegression(objective=objective)
        try:
            m.fit(pop * noise, gdp, y)
        except Exception:
            continue
        thetas.append(m.theta)
        pred_rel.append(np.abs((m.predict(pop, gdp) - base_pred) / base_pred))

    thetas = np.array(thetas)
    pred_rel = np.array(pred_rel)

    return {
        "input_error_pct": rel_error * 100,
        "theta_mean": thetas.mean(axis=0),
        "theta_std": thetas.std(axis=0),
        "b_range": (thetas[:, 1].min(), thetas[:, 1].max()),
        "c_range": (thetas[:, 2].min(), thetas[:, 2].max()),
        "pred_median_pct": float(np.median(pred_rel) * 100),
        "pred_p95_pct": float(np.percentile(pred_rel, 95) * 100),
        "amplification": float(np.percentile(pred_rel, 95) / rel_error),
        "trials": len(thetas),
    }


# ---------------------------------------------------------------------------
# 4. Objective sensitivity
# ---------------------------------------------------------------------------

def objective_sensitivity(pop, gdp, ports):
    """
    Fit under both objectives and compare.

    The purpose is to establish which conclusions are robust to the choice of
    error definition and which are artefacts of it. A parameter that moves
    substantially between objectives is not identified by the data, and
    reporting a single value for it would overstate what has been learned.
    """
    out = {}
    for objective in ("log", "relative", "absolute"):
        m = PowerLawRegression(objective=objective).fit(pop, gdp, ports)
        s = m.score(pop, gdp, ports)
        b = m.bias(pop, gdp, ports)
        out[objective] = {"theta": m.theta, "iters": len(m.history),
                          "signed_pct": b["median_signed_pct"],
                          "under": b["under_predicted"], **s}
    return out


# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------

def make_figures(states, pop, gdp, ports, model, outdir="figures"):
    """
    Produce the two figures required by the project brief.

    Figure 1 -- best-fit curve: observed ports against population on
                logarithmic axes, with the fitted model evaluated at the
                median GDP per capita. Log axes are used because both
                variables span orders of magnitude; on linear axes 45 of the
                50 states would be compressed into one corner.

    Figure 2 -- Predicted-Observed plot: predictions against observations with
                the 1:1 line. Points above the line are under-predicted. The
                +/-30% band shows the achieved median accuracy.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    y = np.asarray(ports, dtype=float)
    pred = model.predict(pop, gdp)
    paths = []

    # -- Figure 1: fitted curve ------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    gdp_med = float(np.median(gdp))
    grid = np.linspace(min(pop) * 0.9, max(pop) * 1.1, 300)

    ax.scatter(pop, y, s=42, alpha=0.75, edgecolor="white", linewidth=0.6,
               color="#1F6FB2", label="observed states", zorder=3)
    ax.plot(grid, model.predict(grid, np.full_like(grid, gdp_med)),
            color="#C0392B", linewidth=2.2, zorder=2,
            label=f"fitted model at median GDP (${gdp_med:,.0f})")

    for name in ("California", "Texas", "Vermont", "Alaska", "Wyoming"):
        if name in states:
            i = states.index(name)
            ax.annotate(name, (pop[i], y[i]), fontsize=8,
                        xytext=(5, 4), textcoords="offset points",
                        color="#333333")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("State population (log scale)")
    ax.set_ylabel("Public EV charging ports (log scale)")
    a, b, c = model.theta
    ax.set_title(f"Fitted power law:  ports = {np.exp(a):.2e}"
                 f" x pop^{b:.3f} x gdp^{c:.3f}", fontsize=11)
    ax.grid(True, which="both", alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    p1 = os.path.join(outdir, "fig1_best_fit_curve.png")
    fig.savefig(p1, dpi=170)
    plt.close(fig)
    paths.append(p1)

    # -- Figure 2: predicted vs observed ---------------------------------
    fig, ax = plt.subplots(figsize=(7, 6.5))
    lim = [min(y.min(), pred.min()) * 0.6, max(y.max(), pred.max()) * 1.7]

    ax.plot(lim, lim, color="#333333", linewidth=1.4, label="perfect prediction")
    ax.plot(lim, [v * 1.3 for v in lim], "--", color="#888888",
            linewidth=1.0, label="+/- 30%")
    ax.plot(lim, [v * 0.7 for v in lim], "--", color="#888888", linewidth=1.0)

    resid_pct = (y - pred) / y * 100
    sc = ax.scatter(y, pred, c=resid_pct, cmap="coolwarm_r", s=54,
                    edgecolor="white", linewidth=0.6, zorder=3,
                    vmin=-90, vmax=90)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("residual (%)  positive = under-predicted", fontsize=9)

    for name in ("California", "Texas", "Vermont", "Alaska", "Louisiana"):
        if name in states:
            i = states.index(name)
            ax.annotate(name, (y[i], pred[i]), fontsize=8,
                        xytext=(6, -9), textcoords="offset points",
                        color="#333333")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Observed ports")
    ax.set_ylabel("Predicted ports")
    s = model.score(pop, gdp, y)
    ax.set_title(f"Predicted-Observed plot\nmedian error "
                 f"{s['median_pct']:.1f}%   MAPE {s['mape']:.1f}%",
                 fontsize=11)
    ax.grid(True, which="both", alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    p2 = os.path.join(outdir, "fig2_predicted_observed.png")
    fig.savefig(p2, dpi=170)
    plt.close(fig)
    paths.append(p2)

    return paths
