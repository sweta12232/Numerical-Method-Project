"""
root_finder.py
==============
Root-Finder module: market threshold analysis.

Question answered
-----------------
Given the fitted power-law model and a state's population, what GDP per
capita would be required for the model to predict a target number of
charging ports? Formally, solve for g in

    h(g) = exp(a) * p^b * g^c - N_target = 0

with p, a, b, c fixed. This is genuinely nonlinear in g: the unknown sits in
an exponent, so no rearrangement produces a linear system. An iterative
root-finder is required.

Note on the analytic solution
-----------------------------
This particular equation happens to admit a closed form,
g = (N_target / (exp(a) * p^b))^(1/c), because there is a single power term.
That is a fortunate property of this model rather than the general case, and
it is used here as an independent check on the iterative results. Reporting
both is stronger than reporting either alone: the analytic value verifies the
implementation, and the iterative methods demonstrate the machinery that
would still work if the model contained, for example, an additive saturation
term with no closed-form inverse.

Methods implemented
-------------------
Newton-Raphson  -- quadratic convergence, requires the derivative
Secant          -- superlinear convergence, derivative-free
Bisection       -- linear convergence, but guaranteed given a sign change

Reference: Altac (2024), Ch. 4 (Nonlinear Equations)
BSIT 400 -- CLO 1, CLO 4
"""

import numpy as np


class RootFindingFailure(Exception):
    """Raised when a root-finder does not converge."""
    pass


# ---------------------------------------------------------------------------
# Newton-Raphson
# ---------------------------------------------------------------------------

def newton_raphson(f, df, x0, tol=1e-10, max_iter=100, record=None):
    """
    Solve f(x) = 0 by Newton-Raphson iteration.

        x_{k+1} = x_k - f(x_k) / f'(x_k)

    Geometrically, the curve is replaced by its tangent at x_k and the
    tangent's root becomes the next estimate. Convergence is quadratic near a
    simple root: the number of correct digits roughly doubles each step.

    Failure modes: a near-zero derivative sends the step to infinity, and a
    poor starting point can diverge or cycle. Both are guarded below.
    """
    x = float(x0)
    for k in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-300:
            raise RootFindingFailure(
                f"derivative vanished at iteration {k} (x = {x:.6e})"
            )

        step = fx / dfx
        x_new = x - step

        if record is not None:
            record.append({"iter": k, "x": x_new, "f": f(x_new),
                           "step": abs(step)})

        if abs(step) < tol * max(abs(x_new), 1.0):
            return x_new, k
        x = x_new

    raise RootFindingFailure(f"Newton-Raphson did not converge in {max_iter} iterations")


# ---------------------------------------------------------------------------
# Secant
# ---------------------------------------------------------------------------

def secant(f, x0, x1, tol=1e-10, max_iter=100, record=None):
    """
    Solve f(x) = 0 by the Secant method.

    Replaces the analytic derivative with the slope of the line through the
    two most recent points:

        x_{k+1} = x_k - f(x_k) * (x_k - x_{k-1}) / (f(x_k) - f(x_{k-1}))

    Convergence is superlinear (order ~1.618) rather than quadratic, but no
    derivative is needed. This matters when the model is only available as a
    black box or when the derivative is expensive to obtain.
    """
    x_prev, x_cur = float(x0), float(x1)
    f_prev, f_cur = f(x_prev), f(x_cur)

    for k in range(1, max_iter + 1):
        denom = f_cur - f_prev
        if abs(denom) < 1e-300:
            raise RootFindingFailure(
                f"secant slope vanished at iteration {k}"
            )

        step = f_cur * (x_cur - x_prev) / denom
        x_new = x_cur - step

        if record is not None:
            record.append({"iter": k, "x": x_new, "f": f(x_new),
                           "step": abs(step)})

        if abs(step) < tol * max(abs(x_new), 1.0):
            return x_new, k

        x_prev, f_prev = x_cur, f_cur
        x_cur, f_cur = x_new, f(x_new)

    raise RootFindingFailure(f"Secant did not converge in {max_iter} iterations")


# ---------------------------------------------------------------------------
# Bisection
# ---------------------------------------------------------------------------

def bisection(f, lo, hi, tol=1e-10, max_iter=200, record=None):
    """
    Solve f(x) = 0 by bisection on a bracket where f changes sign.

    Halves the interval each step, so the error is bounded by
    (hi - lo) / 2^k. Convergence is only linear, but it is guaranteed for any
    continuous function with a sign change, which neither Newton-Raphson nor
    the Secant method can promise. Used here as the safe fallback.
    """
    lo, hi = float(lo), float(hi)
    f_lo, f_hi = f(lo), f(hi)

    if f_lo * f_hi > 0:
        raise RootFindingFailure(
            f"no sign change on [{lo:.4g}, {hi:.4g}]: "
            f"f(lo) = {f_lo:.4g}, f(hi) = {f_hi:.4g}"
        )

    for k in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)

        if record is not None:
            record.append({"iter": k, "x": mid, "f": f_mid,
                           "step": 0.5 * (hi - lo)})

        if 0.5 * (hi - lo) < tol * max(abs(mid), 1.0):
            return mid, k

        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    raise RootFindingFailure(f"Bisection did not converge in {max_iter} iterations")


# ---------------------------------------------------------------------------
# Application: break-even GDP per capita
# ---------------------------------------------------------------------------

class BreakEvenGDP:
    """
    Wraps the fitted model into a root-finding problem in GDP per capita.

    For a fixed population p and target port count N, defines

        h(g)  = exp(a) * p^b * g^c - N
        h'(g) = c * exp(a) * p^b * g^(c-1)

    and solves h(g) = 0 by three independent methods.
    """

    def __init__(self, theta, population, target_ports):
        self.a, self.b, self.c = (float(v) for v in theta)
        self.p = float(population)
        self.N = float(target_ports)
        self._k = np.exp(self.a) * self.p ** self.b   # constant factor

    def h(self, g):
        return self._k * g ** self.c - self.N

    def dh(self, g):
        return self.c * self._k * g ** (self.c - 1.0)

    def analytic(self):
        """Closed-form root, available because the model has a single term."""
        return (self.N / self._k) ** (1.0 / self.c)

    def solve_all(self, g0=60000.0, bracket=(1e3, 1e7)):
        """
        Solve by all three methods and return results with iteration counts.

        The analytic value is included as the reference against which each
        iterative result is checked.
        """
        exact = self.analytic()
        out = {"analytic": {"root": exact, "iters": 0, "error": 0.0,
                            "trace": []}}

        for name, run in (
            ("newton", lambda tr: newton_raphson(self.h, self.dh, g0, record=tr)),
            ("secant", lambda tr: secant(self.h, g0, g0 * 1.5, record=tr)),
            ("bisection", lambda tr: bisection(self.h, bracket[0], bracket[1],
                                               record=tr)),
        ):
            trace = []
            try:
                root, iters = run(trace)
                out[name] = {"root": root, "iters": iters,
                             "error": abs(root - exact) / exact,
                             "trace": trace}
            except RootFindingFailure as exc:
                out[name] = {"root": float("nan"), "iters": -1,
                             "error": float("nan"), "trace": trace,
                             "failure": str(exc)}
        return out
