# Predicting US Public EV Charging Port Capacity by Nonlinear Regression

**BSIT 400 — Numerical Methods · Technical Simulation Report**

---

## 1. Problem Definition

Public electric-vehicle charging infrastructure is distributed unevenly across
the United States. This project builds a predictive engine that estimates a
region's public charging-port count from two regional indicators: population
and GDP per capita.

**Data.** Fifty US states, one observation each. Cross-sectional — a snapshot
of many places at one time, not a time series.

| Field | Range |
|---|---|
| `ports` (response) | 161 (Alaska) to 70,207 (California) |
| `population` | 587,618 to 39,431,263 |
| `gdp_per_capita` | \$53,751 to \$116,883 |

**Correlation with ports:** population 0.822, GDP per capita 0.380.

**Formal statement.** Given observations (Pᵢ, Gᵢ, yᵢ) for i = 1…50, find f such
that ŷ = f(P, G) minimises relative prediction error subject to ŷ > 0 for all
physically meaningful inputs, using two predictors known to be incomplete.

**Success criterion, fixed before any fitting.** Median absolute percentage
error, with median *signed* error reported separately as a bias check and the
count of negative predictions as a hard gate. The response spans a 436-fold
range, so absolute-error metrics — including R² — are determined almost
entirely by the largest few states. Committing to the metric in advance
prevents selecting it post hoc to favour a preferred model. Section 7
demonstrates that this precaution was necessary.

**Explicitly out of scope.** Temporal forecasting (the data contain no time
dimension, so any dated projection would rest on imposed assumptions rather
than evidence) and causal inference (the data are observational).

---

## 2. Governing Equations

### 2.1 Model form

Three candidate forms were considered, each encoding a testable hypothesis.

| Form | Hypothesis | Status |
|---|---|---|
| ŷ = a + bP + cG | Fixed ports per additional person, independent of state size | **Rejected** |
| ŷ = A · P^b · G^c | Proportional effects; b and c are elasticities | **Adopted** |
| ŷ/P = a + cG | Population scales ports exactly linearly (b = 1) | Superseded |

The additive linear form was rejected on structural grounds, not accuracy
grounds. Its fitted intercept is −11,267, and it therefore predicts a negative
port count for **13 of 50 states**. This is not an inaccuracy but a physically
impossible output, and it arises because a plane extends below zero in every
direction. No amount of numerical care repairs it, because the defect is in the
functional form rather than in the arithmetic. The power law is strictly
positive for all positive inputs, so the constraint is enforced by
construction.

### 2.2 Adopted model

    ports = A · population^b · gdp_per_capita^c

parameterised for computation as

    f(P, G) = exp(a + b·ln P + c·ln G),        A = exp(a)

The exponential parameterisation is algebraically identical but keeps A
strictly positive throughout the iteration and yields clean analytic
derivatives.

The exponents b and c are **elasticities**: b is the percentage change in ports
per one-percent change in population. They are dimensionless, so unlike the
linear model's coefficients they do not change when units change.

### 2.3 Objective function

The parameters are chosen to minimise

    E(a, b, c) = Σᵢ [ ln yᵢ − ln f(Pᵢ, Gᵢ) ]²

The choice of residual is a substantive modelling decision, not a formality.
Three definitions were tested (Section 7). The log-ratio residual was selected
because it is **symmetric in ratio terms**: since ln y − ln f = ln(y/f),
predicting a factor k too high and a factor k too low incur equal penalties.

The relative residual (y − f)/y appears equivalent but is not. It is bounded
above by 1 for under-prediction while being unbounded for over-prediction, so
the optimiser finds under-prediction systematically safer. Fitting under that
residual produced a **+18.0% median signed error with 36 of 50 states
under-predicted** — a bias invisible to absolute-error metrics and caught only
because signed error was reported separately.

### 2.4 Nonlinearity

Setting ∂E/∂a = ∂E/∂b = ∂E/∂c = 0 for the *absolute* or *relative* objective
yields equations in which b and c appear inside exponents. No rearrangement
produces a linear system, so iteration is required.

The log-ratio objective is a special case worth stating precisely. Because
r = ln y − (a + b ln P + c ln G) is linear in the parameters, its Jacobian is
constant and Gauss-Newton converges in a single step, reproducing the direct
log-linear least-squares solution. This is an internal consistency check rather
than a coincidence: the nonlinear solver, run on the objective that happens to
be linear in the parameters, must agree with the direct solve. The observed
convergence in 3 iterations (one step plus two to confirm the tolerance)
confirms both implementations.

The iterative machinery remains necessary for the alternative objectives in
Section 7, for the root-finding in Section 5, and as the verification path
described above.

---

## 3. Algorithmic Transition: Pseudocode to Implementation

### 3.1 Damped Gauss-Newton

Gauss-Newton linearises the residual vector about the current estimate:

    r(θ + δ) ≈ r(θ) + J δ

Minimising ‖r + Jδ‖² over the correction δ gives the normal equations

    (JᵀJ) δ = −Jᵀ r

`JᵀJ` is symmetric positive definite whenever J has full column rank, so the
inner system is solved by Cholesky decomposition. The linear machinery is not
discarded when moving to a nonlinear model — it is invoked once per iteration.

```
INPUT   observations y, P, G; tolerance tol
θ ← log-linear least-squares solution          # initialiser
REPEAT
    f ← exp(θ₁ + θ₂ lnP + θ₃ lnG)
    r ← ln y − ln f
    J ← −[1, lnP, lnG]                         # analytic
    δ ← CHOLESKY_SOLVE(JᵀJ, −Jᵀr)
    s ← 1
    WHILE SSE(θ + sδ) ≥ SSE(θ) AND s > 1e−10   # backtracking line search
        s ← s / 2
    θ ← θ + sδ
UNTIL max|sδ| < tol
```

### 3.2 Implementation decisions

**Analytic derivatives.** The Jacobian entries follow from
∂f/∂a = f, ∂f/∂b = f·ln P, ∂f/∂c = f·ln G. Deriving them by hand rather than
using finite differences removes both the step-size selection problem and the
associated truncation error.

**Backtracking line search.** Undamped Gauss-Newton can overshoot when the
tangent-plane approximation is poor far from the solution. Halving the step
until the objective decreases guarantees monotone descent. Under the absolute
objective the search held s = 1 for thirty iterations before reducing to 0.0156
near convergence, so the mechanism is exercised in practice, not merely
present.

**Initialisation from the linearised model.** The log-linear solve is cheap and
well conditioned, and lands in the correct region of parameter space. Using a
linearised model as an initialiser for a nonlinear one is standard practice.

**No library linear algebra.** Cholesky, Gauss-Jordan with partial pivoting,
forward and back substitution, the power method and the inverse power method
are implemented from first principles. `numpy` is used only as an array
container. Each routine is verified in `matrix_solver.py` against a system with
exact integer solution and a matrix with known Cholesky factor and known
spectrum.

---

## 4. Error Assessment

### 4.1 Conditioning of the linear subproblem

The condition number κ₂ = λ_max / λ_min is computed with the power method for
λ_max and the inverse power method for λ_min. The inverse iteration solves
Az = x using Cholesky factors computed once and reused, rather than forming
A⁻¹ explicitly. The eigenvalue routines therefore depend on the linear solver:
Chapter 11 sits on top of Chapter 2.

| System | λ_min | κ₂ | digits lost |
|---|---|---|---|
| Normal equations, raw units | 1.394 | 3.70 × 10¹⁵ | 15.6 of ~16 |
| Normal equations, scaled units | 1.394 | 2.43 × 10⁵ | 5.4 |
| Gauss-Newton inner system at solution | — | 1.64 × 10⁶ | 6.2 |

The raw-unit figure sits within a factor of two of 1/ε_machine = 4.5 × 10¹⁵,
which on its face predicts total loss of accuracy. Two mechanisms produce it:
column magnitudes spanning eight orders (the intercept column holds ones, the
population column reaches 4 × 10⁷), and the fact that forming XᵀX squares the
condition number — cond(X) = 6.1 × 10⁷ and 6.1 × 10⁷ squared is 3.7 × 10¹⁵.

### 4.2 The bound is not attained

The prediction was tested rather than assumed. Solving in single precision
(ε ≈ 1.2 × 10⁻⁷) and comparing against double precision gave a maximum
relative coefficient disagreement of 6.3 × 10⁻⁶ in raw units against
1.7 × 10⁻⁶ in scaled units — a factor of 3.6, where the condition-number bound
predicted a factor near 10¹⁰.

κ₂ is a worst case over all right-hand sides, attained only when the
right-hand side aligns with the eigenvector belonging to λ_min. Here the data
align predominantly with the dominant eigenvector, so the achieved error falls
far below the bound. Reporting κ₂ alone would therefore assert a failure that
does not occur.

Column scaling is nonetheless applied. The favourable alignment is a property
of this dataset that cannot be verified in advance for new data, and scaling
costs three multiplications. The adopted log-log parameterisation sidesteps the
issue entirely, since ln P and ln G are already of comparable magnitude.

A separate observation: both solvers returned relative residuals near 10⁻¹⁶
even in the catastrophically conditioned case. **A small residual does not
imply a small error.** The residual measures whether the system as given was
solved consistently; κ₂ measures whether that system's answer is meaningful.
A residual check alone is insufficient verification.

### 4.3 Cholesky failure as a diagnostic

Cholesky requires √(A_ii − Σ), so a non-positive radicand halts the
factorisation. This failure has two distinct causes, distinguished by λ_min:

| λ_min | Cause | Remedy |
|---|---|---|
| Genuinely positive, κ₂ large | Numerical — round-off from poor scaling | Rescale columns, or use QR on X directly |
| Zero to within data precision | Structural — collinear predictors | Remove or combine the redundant predictor |

Both were reproduced. Adding a duplicate predictor column (population in
thousands alongside population in millions) made XᵀX singular and produced
radicands of 0.000 and −2.9 × 10⁻⁶ — the negative value arising from round-off
pushing an exact zero below zero. The two causes look identical at the point of
failure and are separated only by inspecting λ_min. This is the practical
payoff of having implemented the eigenvalue solver.

Gauss-Jordan on the same singular systems would have returned values without
warning. Cholesky's stricter requirement is an advantage.

### 4.4 Propagation of census error into predictions

Census population figures are estimates carrying sampling error. Three hundred
Monte Carlo trials perturbed every population value by up to ±1% and refitted
the model from scratch.

| Quantity | Result |
|---|---|
| Population elasticity b | 1.0284 ± 0.0009, range [1.0257, 1.0310] |
| GDP elasticity c | 1.4529 ± 0.0055, range [1.4386, 1.4659] |
| Prediction shift, median | 0.093% |
| Prediction shift, 95th percentile | 0.315% |
| Amplification factor | 0.3× |

A ±1% census error induces prediction changes below 0.4% at the 95th
percentile. The amplification factor is below unity: the fit **attenuates**
input error, because 50 observations constrain 3 parameters and independent
perturbations partially cancel. Census uncertainty is not a material limitation
of this model.

### 4.5 Two error sources, not one

| | Round-off error | Data error |
|---|---|---|
| Origin | Finite-precision arithmetic | Measurement and sampling |
| Self-inflicted | Yes — through unit choice | No |
| Removable | Yes, by rescaling | No, by any algorithm |
| Measured by | float32 vs float64 comparison | Monte Carlo perturbation |
| Magnitude here | 10⁻⁶ relative | 0.3% at 95th percentile |

Conflating these is the most common failure in this kind of assessment. A
better algorithm cannot rescue bad data; cleaner data cannot rescue a badly
conditioned formulation.

---

## 5. Market Threshold Analysis (Root-Finding)

For a fixed population P and target port count N, the break-even GDP per
capita solves

    h(G) = exp(a) · P^b · G^c − N = 0,    h′(G) = c · exp(a) · P^b · G^(c−1)

The unknown appears in an exponent, so this is genuinely nonlinear in G. Three
iterative methods were implemented and checked against the closed form, which
exists here because the model contains a single power term and which serves as
an independent verification of the implementations.

Example — Mississippi, target 3,000 ports:

| Method | Root (\$) | Iterations | Relative error vs closed form |
|---|---|---|---|
| Closed form | 208,270 | — | — |
| Newton-Raphson | 208,270 | 4 | 1.4 × 10⁻¹⁶ |
| Secant | 208,270 | 7 | ~10⁻¹⁶ |
| Bisection | 208,270 | ~45 | ~10⁻¹⁰ |

The observed iteration counts match the theoretical convergence orders:
quadratic for Newton-Raphson, superlinear (≈1.618) for Secant, linear for
bisection. Newton-Raphson is fastest but requires the derivative and offers no
guarantee; Secant needs no derivative at a modest cost in iterations;
bisection is slowest but is the only method guaranteed to converge given a
sign change. Bisection is retained as the fallback.

The result itself is informative: the required GDP per capita, \$208,270,
exceeds every observed value in the dataset (maximum \$116,883). Under this
model Mississippi cannot reach 3,000 ports through economic growth alone
within any realistic range. This is a meaningful negative finding, and it is
also a reminder that root-finding readily produces answers outside the region
where the model was fitted.

---

## 6. Validation

### 6.1 Fit quality

| Metric | In-sample | Leave-one-out |
|---|---|---|
| Median absolute error | 26.1% | 27.3% |
| MAPE | 36.8% | 40.1% |
| Median signed error | −3.8% | — |
| States under-predicted | 23 of 50 | — |
| R² (in ports) | 0.691 | — |
| Negative predictions | 0 | 0 |

**Leave-one-out cross-validation** refits the model fifty times, each time
holding out one state and predicting it from a model that never saw it. The
gap between in-sample and out-of-sample median error is **1.2 percentage
points**, indicating the model generalises and is not memorising individual
states. With three parameters and fifty observations this is expected; a large
gap would have signalled overfitting.

**Bias.** The median signed error of −3.8% with a 23/27 split confirms the fit
is close to unbiased. This check is reported separately because absolute-error
metrics cannot reveal systematic bias — the relative-residual fit achieved a
comparable median absolute error of 29.5% while under-predicting 36 of 50
states.

### 6.2 Residual structure

The extreme residuals are not randomly distributed.

| Under-predicted (model too low) | | Over-predicted (model too high) | |
|---|---|---|---|
| Vermont | +86% | Alaska | −210% |
| Maine | +68% | North Dakota | −114% |
| Rhode Island | +66% | Louisiana | −113% |
| Connecticut | +64% | Nebraska | −96% |
| California | +60% | Texas | −52% |
| Oregon | +59% | Indiana | −41% |

The under-predicted group consists almost entirely of states that have adopted
California's Zero Emission Vehicle programme — California, Oregon,
Connecticut, Vermont, Rhode Island and Maine are all Section 177 states. The
over-predicted group consists of states that have not.

This is an **omitted-variable signature**, not noise. The residuals are
carrying state EV policy, which the model does not contain. It identifies the
single most valuable addition to the model: a policy indicator variable, which
on this evidence would explain more remaining variance than any refinement of
the numerical method.

### 6.3 Figures

- `figures/fig1_best_fit_curve.png` — observed ports against population on
  logarithmic axes with the fitted model evaluated at median GDP per capita.
  Log axes are necessary: on linear axes 45 of 50 states compress into one
  corner.
- `figures/fig2_predicted_observed.png` — predicted against observed with the
  1:1 line and a ±30% band, coloured by signed residual.

---

## 7. Sensitivity to the Objective Function

| Objective | b (pop) | c (GDP) | Median abs err | Median signed | R² | Iterations |
|---|---|---|---|---|---|---|
| **Log-ratio** | **1.028** | **1.453** | **26.1%** | **−3.8%** | 0.691 | 3 |
| Relative | 1.134 | 1.003 | 29.5% | +18.0% | 0.614 | 12 |
| Absolute | 1.848 | 2.960 | 78.3% | +78.0% | **0.881** | 39 |

Identical data, identical model form, identical solver. Only the definition of
error differs.

**The population elasticity is robust.** It sits at 1.03–1.13 for the two
better-behaved objectives, supporting the substantive conclusion that ports
scale approximately proportionally with population. Adding a million residents
to a small state has roughly the same proportional effect as adding a million
to a large one.

**The GDP elasticity is not identified.** It ranges from 1.00 to 2.96 across
objectives. Given GDP's weak raw correlation with ports (0.380), the fitting
procedure has considerable latitude to move this exponent without materially
degrading the fit. The preferred estimate is 1.45 from the unbiased objective,
but the honest report is the range. A single point estimate would overstate
what the data support.

**R² is actively misleading here.** The absolute-residual fit attains the
highest R² (0.881) while having the worst median percentage error (78.3%) and
a severe downward bias (47 of 50 states under-predicted). R² is constructed
from squared absolute errors and is dominated by the few largest states, so it
rewards a model that fits California well and abandons the remaining 49. Had
the evaluation metric not been fixed in Section 1 before any fitting, the
0.881 would have been difficult to resist.

---

## 8. Limitations

1. **Two predictors in a system driven by more.** Section 6.2 identifies state
   EV policy as the dominant omitted variable. Highway miles, urban density
   and existing EV registrations are also plausible.
2. **The GDP elasticity is not identified** by this dataset (Section 7).
3. **No temporal dimension.** Any dated projection would rest on an assumed
   growth rate, not on evidence in the data.
4. **Association, not causation.** The data are observational and
   cross-sectional.
5. **Fifty observations.** Leave-one-out is the only viable validation
   strategy; a held-out test set would leave too little for fitting.
6. **Back-transform bias.** Fitting in log space and exponentiating introduces
   a known downward bias, since the exponential of a mean is not the mean of
   exponentials. The measured median signed error of −3.8% bounds it as small
   here. A smearing correction would remove it if higher accuracy in
   aggregate totals were required.
7. **Extrapolation is unguarded.** Section 5 returns a break-even GDP per
   capita of \$208,270 against an observed maximum of \$116,883. The model
   answers questions outside its fitted range without complaint.

---

## 9. Conclusions

The adopted model is

    ports = 2.6221 × 10⁻¹¹ · population^1.0285 · gdp_per_capita^1.4530

fitted by damped Gauss-Newton on log-ratio residuals, converging in 3
iterations, with a median absolute error of 26.1% in-sample and 27.3% under
leave-one-out cross-validation, essentially no bias, and no physically
impossible predictions.

Four findings extend beyond the fitted numbers.

**The condition-number bound was pessimistic by ten orders of magnitude.**
Computing κ₂ = 3.7 × 10¹⁵ and stopping there would have asserted a failure
that testing showed does not occur. Scaling is applied regardless, because the
favourable alignment cannot be verified in advance for new data.

**The choice of residual definition changed the fitted elasticities more than
any numerical decision.** The relative residual, which appears equivalent to
the log-ratio residual, carries a hidden asymmetry that biased predictions
downward by 18%. Detecting it required reporting signed error alongside
absolute error.

**R² selected the worst model.** Committing to the evaluation metric before
fitting, on grounds derived from the data's dynamic range, was the decision
that prevented this.

**The residuals identified the model's own principal weakness.** Their
structure points directly at state EV policy — a variable not in the dataset,
and a more valuable addition than any further numerical refinement.

---

## Reference

Altac, Z. (2024). *Numerical methods for scientists and engineers: With
pseudocodes.* CRC Press. Chapters 1, 2, 4, 7, 11.
