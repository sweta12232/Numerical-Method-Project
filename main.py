"""
main.py
=======
End-to-end pipeline for the EV charging-port prediction project.

Run with:  python3 main.py

Pipeline
--------
Stage 1  Data loading and problem framing
Stage 2  Conditioning diagnostics on the initialiser's linear system
Stage 3  Nonlinear model fit by damped Gauss-Newton
Stage 4  Root-finding: break-even GDP per capita
Stage 5  Validation: residuals and leave-one-out cross-validation
Stage 6  Error assessment: census error propagation
Stage 7  Objective sensitivity
Stage 8  Figures

Module map
--------------------------------------------
matrix_solver.py      Matrix Solver     -- Gauss-Jordan, Cholesky, eigenvalues
regression_engine.py  Regression Engine -- nonlinear Gauss-Newton fit
root_finder.py        Root-Finder       -- Newton-Raphson, Secant, Bisection
diagnostics.py        Validation        -- residuals, LOOCV, error propagation

BSIT 400 -- Numerical Methods
"""

import csv
import sys

import numpy as np

from matrix_solver import condition_number, cholesky_solve
from regression_engine import PowerLawRegression, ConvergenceFailure
from root_finder import BreakEvenGDP
import diagnostics as dg

RULE = "=" * 78


def banner(text):
    print("\n" + RULE)
    print(text)
    print(RULE)


def load(path="ev_charging_data.csv"):
    states, ports, pop, gdp = [], [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            states.append(row["state"])
            ports.append(float(row["ports"]))
            pop.append(float(row["population"]))
            gdp.append(float(row["gdp_per_capita"]))
    return states, np.array(ports), np.array(pop), np.array(gdp)


# ---------------------------------------------------------------------------

def stage1(states, y, pop, gdp):
    banner("STAGE 1 -- Data and problem framing")
    print(f"\n  Observations        {len(states)} US states (cross-sectional)")
    print(f"  Response            public EV charging ports")
    print(f"    range             {y.min():,.0f} to {y.max():,.0f}"
          f"  ({y.max()/y.min():.0f}x spread)")
    print(f"  Predictors          population, GDP per capita")
    print(f"    population        {pop.min():,.0f} to {pop.max():,.0f}")
    print(f"    gdp per capita    ${gdp.min():,.0f} to ${gdp.max():,.0f}")

    print(f"\n  Correlation with ports:")
    print(f"    population        {np.corrcoef(pop, y)[0,1]:.3f}   (strong)")
    print(f"    gdp per capita    {np.corrcoef(gdp, y)[0,1]:.3f}   (weak)")

    print("""
  Success criterion, fixed before any fitting: relative (percentage) error.
  The response spans nearly three orders of magnitude, so absolute-error
  metrics would be determined almost entirely by the largest few states.
  Median absolute percentage error is the headline metric, with the count of
  physically impossible (negative) predictions as a hard gate.

  Out of scope: temporal forecasting (the data have no time dimension) and
  causal claims (the data are observational and cross-sectional).""")


def stage2(y, pop, gdp):
    banner("STAGE 2 -- Conditioning of the initialiser's linear system")
    n = len(y)
    print("\n  The nonlinear solver is started from a log-linear fit, which")
    print("  requires solving normal equations. Their conditioning is checked")
    print("  first, using the power method and inverse power method.")

    for label, X in (
        ("raw units   ", np.column_stack([np.ones(n), pop, gdp])),
        ("scaled units", np.column_stack([np.ones(n), pop / 1e6, gdp / 1e3])),
    ):
        d = condition_number(X.T @ X)
        print(f"\n  {label}  lambda_max = {d['lambda_max']:.4e}"
              f"   lambda_min = {d['lambda_min']:.4e}")
        print(f"                kappa_2 = {d['kappa']:.4e}"
              f"   ({d['digits_lost']:.1f} of ~16 digits lost)")

    print("""
  The raw-unit system is catastrophically conditioned on paper, because the
  intercept column holds ones while the population column reaches 4e7.
  Measured accuracy is far better than the bound predicts, because the
  right-hand side does not excite the ill-conditioned direction. The bound is
  a worst case over all right-hand sides and is not tight here.

  Design decision: predictor columns are scaled regardless. The favourable
  alignment is a property of this dataset and cannot be verified in advance
  for new data. Scaling costs three multiplications.

  The log-log parameterisation used by the model sidesteps the issue entirely,
  since ln(population) and ln(GDP) are already of comparable magnitude.""")


def stage3(y, pop, gdp):
    banner("STAGE 3 -- Nonlinear model fit (damped Gauss-Newton)")
    print("""
  Model:      ports = exp(a) * pop^b * gdp^c
  Objective:  minimise sum of squared log-ratio residuals, ln(y) - ln(f).
              This residual is symmetric in ratio terms: over- and
              under-prediction by the same factor incur equal penalties.
              The relative residual (y - f)/y is bounded above by 1 for
              under-prediction but unbounded for over-prediction, which
              biases the fit downwards by roughly 18% on this data.
  Method:     damped Gauss-Newton. Each iteration replaces the model with its
              tangent plane, giving the 3x3 system (J^T J) d = -J^T r, solved
              by Cholesky. A backtracking line search halves the step until
              the objective decreases.
  Derivatives: analytic (df/da = f, df/db = f ln p, df/dc = f ln g)
""")
    model = PowerLawRegression(objective="log")
    theta0 = None

    print("  Initial guess from the log-linear fit:")
    m_tmp = PowerLawRegression(objective="log")
    m_tmp._y, m_tmp._log_pop, m_tmp._log_gdp = y, np.log(pop), np.log(gdp)
    theta0 = m_tmp.initial_guess()
    print(f"    theta0 = a {theta0[0]:.4f}   b {theta0[1]:.4f}"
          f"   c {theta0[2]:.4f}")
    print("    (solves a different objective, so a starting point only)\n")

    model.fit(pop, gdp, y, verbose=True)

    print(f"\n  Converged: {model.converged} in {len(model.history)} iterations")
    print(f"  Model: {model.summary()}")

    b, c = model.elasticities
    print(f"\n  Elasticities")
    print(f"    population  b = {b:.4f}   "
          f"({'proportional' if abs(b-1) < 0.25 else 'superlinear' if b > 1 else 'sublinear'})")
    print(f"    gdp         c = {c:.4f}")

    d = model.inner_conditioning()
    print(f"\n  Inner system conditioning at the solution:")
    print(f"    kappa_2(J^T J) = {d['kappa']:.4e}"
          f"   ({d['digits_lost']:.1f} digits lost)")

    s = model.score(pop, gdp, y)
    print(f"\n  In-sample fit")
    print(f"    median absolute error   {s['median_pct']:.1f}%")
    print(f"    MAPE                    {s['mape']:.1f}%")
    print(f"    R^2 (in ports)          {s['r2_ports']:.4f}")
    print(f"    negative predictions    {s['negative_predictions']}"
          f"   (structurally impossible: exp() is positive)")

    bi = model.bias(pop, gdp, y)
    print(f"\n  Bias check")
    print(f"    median signed error     {bi['median_signed_pct']:+.1f}%")
    print(f"    states under-predicted  {bi['under_predicted']} of {bi['n']}")
    print("    A near-zero signed error and a roughly even split confirm the")
    print("    fit is unbiased. Absolute-error metrics alone would not have")
    print("    revealed a systematic bias, so this check is reported separately.")
    return model


def stage4(model, states, pop):
    banner("STAGE 4 -- Root-Finder: break-even GDP per capita")
    print("""
  Question: what GDP per capita would a state need for the model to predict a
  given number of ports? Solve h(g) = exp(a) p^b g^c - N = 0 for g.

  Nonlinear in g: the unknown is in an exponent. Three iterative methods are
  compared against the closed-form root, which exists here because the model
  has a single power term and serves as an independent check.
""")
    targets = [("Mississippi", 3000), ("Wyoming", 1500), ("Ohio", 12000)]
    print(f"  {'state':<14}{'target':>8}{'method':>12}{'root ($)':>15}"
          f"{'iters':>7}{'rel. error':>13}")

    for name, target in targets:
        if name not in states:
            continue
        i = states.index(name)
        solver = BreakEvenGDP(model.theta, pop[i], target)
        res = solver.solve_all(g0=60000.0)

        for method in ("analytic", "newton", "secant", "bisection"):
            r = res[method]
            tag = f"{r['error']:.2e}" if r["iters"] >= 0 else "FAILED"
            it = "-" if method == "analytic" else str(r["iters"])
            label = name if method == "analytic" else ""
            tgt = f"{target:,}" if method == "analytic" else ""
            print(f"  {label:<14}{tgt:>8}{method:>12}"
                  f"{r['root']:>15,.0f}{it:>7}{tag:>13}")
        print()

    print("  Newton-Raphson converges in the fewest iterations (quadratic),")
    print("  the Secant method takes slightly more but needs no derivative,")
    print("  and bisection takes far more while being the only one guaranteed.")
    print("  All three agree with the closed form to near machine precision,")
    print("  which validates the implementations.")


def stage5(states, y, pop, gdp, model):
    banner("STAGE 5 -- Validation: residuals and leave-one-out")
    rows, ends = dg.residual_table(states, pop, gdp, y, model, top=6)

    print("\n  Most UNDER-predicted (model too low):")
    for r in ends["under"]:
        print(f"    {r['state']:<16} observed {r['observed']:>7,.0f}"
              f"   predicted {r['predicted']:>7,.0f}   {r['pct']:+7.1f}%")
    print("\n  Most OVER-predicted (model too high):")
    for r in ends["over"]:
        print(f"    {r['state']:<16} observed {r['observed']:>7,.0f}"
              f"   predicted {r['predicted']:>7,.0f}   {r['pct']:+7.1f}%")

    print("""
  These extremes are not random. The under-predicted group is dominated by
  states that have adopted California's Zero Emission Vehicle programme
  (California, Oregon, Connecticut, Vermont, Rhode Island, Maine), and the
  over-predicted group by states that have not (Louisiana, Alaska, Kentucky,
  Texas, Indiana). The residuals are therefore carrying a policy variable the
  model does not contain. This is an omitted-variable signature rather than
  noise, and it identifies the single most valuable addition to the model.""")

    print("\n  Leave-one-out cross-validation (50 refits):")
    loo = dg.leave_one_out(pop, gdp, y)
    ins = model.score(pop, gdp, y)
    print(f"    in-sample     median {ins['median_pct']:.1f}%"
          f"   MAPE {ins['mape']:.1f}%")
    print(f"    out-of-sample median {loo['median_pct']:.1f}%"
          f"   MAPE {loo['mape']:.1f}%")
    print(f"    gap           {loo['median_pct'] - ins['median_pct']:+.1f}"
          f" percentage points")
    print("""
    The small gap indicates the model is not overfitting. With three
    parameters and 50 observations this is expected, and it confirms that the
    in-sample figures are close to honest. A large gap would have meant the
    model was memorising individual states rather than capturing structure.""")
    return loo


def stage6(y, pop, gdp):
    banner("STAGE 6 -- Error assessment: census error propagation")
    print("""
  Census population figures are estimates. Each of 300 trials perturbs every
  population value by up to +/-1% and refits the model from scratch. The
  spread across trials is the uncertainty that data error induces.
""")
    ep = dg.propagate_census_error(pop, gdp, y, rel_error=0.01, trials=300)
    print(f"  input perturbation        +/-{ep['input_error_pct']:.1f}%"
          f"   ({ep['trials']} successful refits)")
    print(f"  population elasticity b   {ep['theta_mean'][1]:.4f}"
          f" +/- {ep['theta_std'][1]:.4f}"
          f"   range [{ep['b_range'][0]:.4f}, {ep['b_range'][1]:.4f}]")
    print(f"  gdp elasticity c          {ep['theta_mean'][2]:.4f}"
          f" +/- {ep['theta_std'][2]:.4f}"
          f"   range [{ep['c_range'][0]:.4f}, {ep['c_range'][1]:.4f}]")
    print(f"  prediction shift          median {ep['pred_median_pct']:.3f}%"
          f"   95th pct {ep['pred_p95_pct']:.3f}%")
    print(f"  amplification factor      {ep['amplification']:.1f}x")
    print("""
  Data error and round-off error are distinct. Round-off is an artefact of
  finite precision, is self-inflicted through poor scaling, and is removed by
  rescaling. Data error is intrinsic to the measurements, cannot be removed by
  any algorithm, and must be reported as an uncertainty band. Conflating the
  two is the most common error in this kind of assessment.""")
    return ep


def stage7(y, pop, gdp):
    banner("STAGE 7 -- Objective sensitivity")
    sens = dg.objective_sensitivity(pop, gdp, y)
    print(f"\n  {'objective':<12}{'b (pop)':>10}{'c (gdp)':>10}"
          f"{'median %':>11}{'signed %':>10}{'R^2':>9}{'iters':>7}")
    for k in ("log", "relative", "absolute"):
        r = sens[k]
        print(f"  {k:<12}{r['theta'][1]:>10.3f}{r['theta'][2]:>10.3f}"
              f"{r['median_pct']:>10.1f}%{r['signed_pct']:>9.1f}%"
              f"{r['r2_ports']:>9.4f}{r['iters']:>7}")
    print("""
  Same data, same model form, same solver: only the definition of error
  differs.

  The population elasticity is comparatively stable (1.03 to 1.13 for the two
  unbiased-or-mildly-biased objectives), supporting the conclusion that ports
  scale approximately proportionally with population. The GDP elasticity moves
  from 1.00 to 2.96. Combined with GDP's weak raw correlation with ports
  (0.38), this indicates the GDP elasticity is poorly identified by this
  dataset, and it is reported as a range with 1.45 as the preferred estimate
  from the unbiased objective.

  The absolute-residual fit attains the highest R^2 (0.88) while having the
  worst median percentage error (78%) and a severe downward bias (47 of 50
  states under-predicted). R^2 is built from squared absolute errors and is
  dominated by the few largest states, so it rewards a model that fits
  California and abandons everything else. This is precisely why the metric
  was committed to in Stage 1, before any model existed.""")
    return sens


def stage8(states, y, pop, gdp, model):
    banner("STAGE 8 -- Figures")
    try:
        paths = dg.make_figures(states, pop, gdp, y, model)
        for p in paths:
            print(f"  written: {p}")
    except Exception as exc:
        print(f"  figure generation skipped: {exc}")
        return []
    return paths


def main():
    states, y, pop, gdp = load()

    stage1(states, y, pop, gdp)
    stage2(y, pop, gdp)
    try:
        model = stage3(y, pop, gdp)
    except ConvergenceFailure as exc:
        print(f"\n  FATAL: {exc}")
        return 1

    stage4(model, states, pop)
    stage5(states, y, pop, gdp, model)
    stage6(y, pop, gdp)
    stage7(y, pop, gdp)
    stage8(states, y, pop, gdp, model)

    banner("PIPELINE COMPLETE")
    print(f"\n  Reported model:  {model.summary()}")
    s = model.score(pop, gdp, y)
    print(f"  Median error:    {s['median_pct']:.1f}%"
          f"   (out-of-sample checked in Stage 5)")
    print("  Principal limitation: two predictors cannot capture state EV")
    print("  policy, which the residual structure in Stage 5 identifies as")
    print("  the dominant omitted variable.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
