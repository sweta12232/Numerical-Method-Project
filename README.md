# EV Charging Port Prediction — BSIT 400 Numerical Methods

Predicts US state public EV charging-port counts from population and GDP per
capita using a nonlinear power-law model fitted by damped Gauss-Newton.

All numerical algorithms are implemented from first principles. `numpy` is used
only as an array container — no `numpy.linalg`, no `scipy`.

## Quick start

```bash
python3 main.py              # full pipeline, all 8 stages
python3 matrix_solver.py     # self-test of the linear algebra routines
```

Requires `numpy` and `matplotlib`.

## Module map

| File | Role in the project brief | Contents |
|---|---|---|
| `matrix_solver.py` | **Matrix Solver** | Gauss-Jordan with partial pivoting, Cholesky decomposition, forward/back substitution, power method, inverse power method, condition number |
| `regression_engine.py` | **Regression Engine** | `PowerLawRegression` — damped Gauss-Newton fit of `ports = A·pop^b·gdp^c` under three selectable objectives |
| `root_finder.py` | **Root-Finder** | Newton-Raphson, Secant, Bisection, and the `BreakEvenGDP` application |
| `diagnostics.py` | Validation | Residual analysis, leave-one-out CV, Monte Carlo error propagation, figures |
| `main.py` | Driver | Runs stages 1–8 end to end |
| `step1_conditioning.py` | Experiment | Conditioning of the normal equations, raw vs scaled units |
| `step1b_error_analysis.py` | Experiment | Tests whether the κ bound is attained (float32 vs float64) |
| `REPORT.md` | Deliverable | Technical simulation report |

## Dependency structure

```
main.py
 ├── regression_engine.py ──┐
 ├── root_finder.py         │
 └── diagnostics.py ────────┤
                            └── matrix_solver.py
```

Note that `matrix_solver.condition_number` calls `inverse_power_method`, which
calls `cholesky_decompose`. The eigenvalue routines are built on the linear
solver rather than being independent of it.

## Result

```
ports = 2.6221e-11 · population^1.0285 · gdp_per_capita^1.4530
```

| Metric | Value |
|---|---|
| Median absolute error, in-sample | 26.1% |
| Median absolute error, leave-one-out | 27.3% |
| Median signed error (bias) | −3.8% |
| Negative predictions | 0 of 50 |
| Gauss-Newton iterations | 3 |
| κ₂ of inner system at solution | 1.64 × 10⁶ |

**Elasticities.** Population 1.029 (robust across objectives — ports scale
approximately proportionally with population). GDP 1.453 (**not identified** —
ranges 1.00 to 2.96 depending on the objective function; report the range, not
the point estimate).

## Methods used, and why

| Method | Chapter | Purpose |
|---|---|---|
| Cholesky decomposition | 2 | Solves the symmetric positive-definite Gauss-Newton inner system; half the cost of LU, and its failure signals lost conditioning |
| Gauss-Jordan + partial pivoting | 2 | Comparison baseline; solves without warning where Cholesky halts |
| Power / inverse power method | 11 | Computes κ₂ = λ_max/λ_min without a library call; distinguishes numerical from structural failure |
| Damped Gauss-Newton | 4, 7 | Fits the power law directly; the line search guarantees monotone descent |
| Newton-Raphson, Secant, Bisection | 4 | Break-even GDP threshold; three convergence orders compared against the closed form |
| Monte Carlo propagation | 1 | Census error → prediction uncertainty |

## Methods deliberately excluded

| Method | Reason |
|---|---|
| Lagrange interpolation | Degree-49 polynomial through 50 scattered points oscillates violently (Runge's phenomenon) and yields negative counts. Regression smooths noise; interpolation reproduces it. |
| Cubic splines | Require an ordered independent variable. These are scattered points in a 2-D predictor plane with no natural ordering. |
| Gauss-Seidel | Direct methods win outright on a 3×3 system. |
| Gaussian quadrature | No integral arises in the problem. |



## Verification

`matrix_solver.py` self-tests against a 3×3 system with exact solution
[1, 2, 3], a matrix whose Cholesky factor is a known result, and a diagonal
matrix with known spectrum {4, 2, 1}. All root-finders are checked against the
closed-form break-even GDP to ~10⁻¹⁶ relative error.
