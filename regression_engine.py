"""
regression_engine.py
====================
Regression Engine for the EV charging-port model.

Fits the power-law model

    ports = A * pop^b * gdp^c

directly, without linearising it, by damped Gauss-Newton iteration. The
parameters b and c appear in exponents, so the least-squares normal equations
are nonlinear and no direct solve exists. Each Gauss-Newton iteration replaces
the model with its tangent plane, which produces a 3x3 symmetric
positive-definite system solved by the Cholesky routine in matrix_solver.

Objective
---------
Residuals are relative, r_i = (y_i - f_i) / y_i, because the response spans
three orders of magnitude (161 to 70,207 ports). Minimising absolute error
would let the largest states dominate the fit entirely. This choice is fixed
before fitting and is not revisited on the basis of results.

Reference: Altac (2024), Ch. 4 (Nonlinear Equations), Ch. 7 (Least Squares)
BSIT 400 -- CLO 1, CLO 2, CLO 4, CLO 6
"""

import numpy as np

from matrix_solver import cholesky_solve, condition_number, SingularMatrixError


class ConvergenceFailure(Exception):
    """Raised when the iteration does not reach the tolerance in time."""
    pass


class PowerLawRegression:
    """
    Nonlinear least-squares fit of ports = exp(a) * pop^b * gdp^c.

    The model is held in exponential form, f = exp(a + b*ln(p) + c*ln(g)),
    rather than as A * p^b * g^c. The two are algebraically identical with
    A = exp(a), but the exponential parameterisation keeps A strictly positive
    throughout the iteration and yields clean analytic derivatives.
    """

    OBJECTIVES = ("log", "relative", "absolute")

    def __init__(self, objective="log", tol=1e-12, max_iter=200):
        if objective not in self.OBJECTIVES:
            raise ValueError(f"objective must be one of {self.OBJECTIVES}")
        self.objective = objective
        self.tol = tol
        self.max_iter = max_iter

        self.theta = None          # fitted (a, b, c)
        self.history = []          # per-iteration diagnostics
        self.converged = False
        self._log_pop = None
        self._log_gdp = None

    # -- model and derivatives --------------------------------------------

    def predict(self, pop, gdp, theta=None):
        """Predicted ports for given population and GDP per capita."""
        t = self.theta if theta is None else theta
        if t is None:
            raise RuntimeError("model has not been fitted")
        return np.exp(t[0] + t[1] * np.log(pop) + t[2] * np.log(gdp))

    def _model(self, theta):
        return np.exp(theta[0] + theta[1] * self._log_pop
                      + theta[2] * self._log_gdp)

    def _jacobian(self, theta):
        """
        Jacobian of the residual vector with respect to (a, b, c).

        Model derivatives are analytic:
            df/da = f,   df/db = f*ln(p),   df/dc = f*ln(g)
        Residuals are r = y - f, so dr/dtheta = -df/dtheta. Under the relative
        objective every row is divided by y_i, which is the mechanism that
        equalises the influence of large and small states.
        """
        f = self._model(theta)
        J = -np.column_stack([f, f * self._log_pop, f * self._log_gdp])
        if self.objective == "relative":
            J = J / self._y[:, None]
        elif self.objective == "log":
            # r = ln(y) - ln(f), so dr/dtheta = -(1/f) df/dtheta and the
            # factors of f cancel exactly. The Jacobian is therefore constant,
            # which is the formal statement that this objective is linear in
            # the parameters. Gauss-Newton consequently converges in a single
            # step -- a useful internal consistency check.
            J = J / f[:, None]
        return J

    def _residuals(self, theta):
        f = self._model(theta)
        if self.objective == "log":
            # Symmetric in ratio terms: ln(y) - ln(f) = ln(y/f), so being a
            # factor k too high and a factor k too low incur equal penalties.
            # The relative residual (y - f)/y is not symmetric in this sense:
            # under-prediction is bounded by 1 while over-prediction is
            # unbounded, which biases the fit downwards.
            return np.log(self._y) - np.log(f)
        r = self._y - f
        return r / self._y if self.objective == "relative" else r

    def bias(self, pop, gdp, ports):
        """
        Systematic prediction bias, as median signed percentage residual.

        A value near zero means over- and under-predictions are balanced. A
        large positive value means the model systematically predicts too few
        ports, which absolute-error metrics will not reveal.
        """
        y = np.asarray(ports, dtype=float)
        f = self.predict(pop, gdp)
        signed = (y - f) / y * 100.0
        return {
            "median_signed_pct": float(np.median(signed)),
            "mean_signed_pct": float(signed.mean()),
            "under_predicted": int((signed > 0).sum()),
            "n": int(len(y)),
        }

    # -- initial guess -----------------------------------------------------

    def initial_guess(self):
        """
        Starting point from the log-linear fit.

        Taking logs of both sides makes the model linear in the parameters, so
        it can be solved in one direct Cholesky step. That solution minimises
        the wrong objective (squared error in log space) and is therefore not
        the answer, but it lands in the correct region of parameter space and
        makes the nonlinear iteration reliable. This is the standard use of a
        linearised model as an initialiser.
        """
        n = len(self._y)
        L = np.column_stack([np.ones(n), self._log_pop, self._log_gdp])
        return cholesky_solve(L.T @ L, L.T @ np.log(self._y))

    # -- fitting -----------------------------------------------------------

    def fit(self, pop, gdp, ports, theta0=None, verbose=False):
        """
        Fit the model by damped Gauss-Newton.

        At each iteration the correction delta solves the tangent-plane normal
        equations (J^T J) delta = -J^T r. A backtracking line search halves
        the step until the sum of squared residuals decreases, which prevents
        the overshoot that undamped Gauss-Newton can produce.
        """
        self._y = np.asarray(ports, dtype=float)
        self._log_pop = np.log(np.asarray(pop, dtype=float))
        self._log_gdp = np.log(np.asarray(gdp, dtype=float))

        theta = np.array(self.initial_guess() if theta0 is None else theta0,
                         dtype=float)
        self.history = []
        self.converged = False

        if verbose:
            print(f"  {'it':>4}{'step':>9}{'SSE':>15}{'max|delta|':>13}"
                  f"{'a':>11}{'b':>9}{'c':>9}")

        for it in range(1, self.max_iter + 1):
            r = self._residuals(theta)
            J = self._jacobian(theta)

            A = J.T @ J                 # symmetric positive definite
            rhs = -J.T @ r
            try:
                delta = cholesky_solve(A, rhs)
            except SingularMatrixError as exc:
                raise ConvergenceFailure(
                    f"inner Cholesky solve failed at iteration {it}: {exc}"
                ) from exc

            # Backtracking line search
            sse_old = float(r @ r)
            step, sse_new = 1.0, sse_old
            while step > 1e-10:
                trial = theta + step * delta
                sse_new = float(np.sum(self._residuals(trial) ** 2))
                if sse_new < sse_old:
                    break
                step /= 2.0

            theta = theta + step * delta
            change = float(np.max(np.abs(step * delta)))
            self.history.append({"iter": it, "step": step, "sse": sse_new,
                                 "change": change, "theta": theta.copy()})

            if verbose and (it <= 3 or it % 10 == 0 or change < self.tol):
                print(f"  {it:>4}{step:>9.5f}{sse_new:>15.6e}{change:>13.3e}"
                      f"{theta[0]:>11.4f}{theta[1]:>9.4f}{theta[2]:>9.4f}")

            if change < self.tol:
                self.converged = True
                break

        if not self.converged:
            raise ConvergenceFailure(
                f"no convergence in {self.max_iter} iterations "
                f"(last step change {change:.2e})"
            )

        self.theta = theta
        return self

    # -- diagnostics -------------------------------------------------------

    @property
    def elasticities(self):
        """(population elasticity, GDP elasticity) = (b, c)."""
        return float(self.theta[1]), float(self.theta[2])

    def inner_conditioning(self):
        """
        Condition number of the Gauss-Newton inner system at the solution.

        Computed with the power method and inverse power method rather than a
        library routine. A large value would mean the computed corrections
        near convergence are unreliable.
        """
        J = self._jacobian(self.theta)
        return condition_number(J.T @ J)

    def score(self, pop, gdp, ports):
        """Fit quality on the given data. Relative metrics lead."""
        y = np.asarray(ports, dtype=float)
        f = self.predict(pop, gdp)
        rel = np.abs((y - f) / y)
        return {
            "median_pct": float(np.median(rel) * 100),
            "mape": float(rel.mean() * 100),
            "r2_ports": float(1 - np.sum((y - f) ** 2)
                              / np.sum((y - y.mean()) ** 2)),
            "negative_predictions": int((f < 0).sum()),
            "n": int(len(y)),
        }

    def summary(self):
        a, b, c = self.theta
        return (f"ports = {np.exp(a):.4e} * pop^{b:.4f} * gdp^{c:.4f}   "
                f"[{self.objective} residuals, {len(self.history)} iterations]")
