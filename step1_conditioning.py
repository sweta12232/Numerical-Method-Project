"""
step1_conditioning.py
=====================
Conditioning analysis of the normal equations for the EV charging-port model.

This experiment answers one question: can the least-squares normal equations
for this dataset be solved reliably in double precision, and what does column
scaling do to the answer?

BSIT 400 -- CLO 1 (error propagation), CLO 2 (direct methods), CLO 3 (eigenvalues)
"""

import csv
import numpy as np

from matrix_solver import (
    cholesky_solve,
    gauss_jordan,
    condition_number,
    residual_norm,
    SingularMatrixError,
)

EPS_MACHINE = np.finfo(float).eps


def load_data(path="ev_charging_data.csv"):
    """Read the state-level dataset into parallel arrays."""
    states, ports, pop, gdp = [], [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            states.append(row["state"])
            ports.append(float(row["ports"]))
            pop.append(float(row["population"]))
            gdp.append(float(row["gdp_per_capita"]))
    return states, np.array(ports), np.array(pop), np.array(gdp)


def build_design_matrix(pop, gdp, scaled):
    """
    Assemble the design matrix X for  ports = a + b*pop + c*gdp.

    scaled=False -> raw units: population in persons, GDP in dollars
    scaled=True  -> population in millions, GDP in thousands of dollars

    Scaling changes nothing about the underlying model. It only changes the
    numerical representation, which is exactly the point of the experiment.
    """
    n = len(pop)
    if scaled:
        return np.column_stack([np.ones(n), pop / 1e6, gdp / 1e3])
    return np.column_stack([np.ones(n), pop, gdp])


def report(label, X, y):
    """Run the full conditioning and solution report for one scaling choice."""
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    # Form the normal equations  (X^T X) beta = X^T y
    A = X.T @ X
    b = X.T @ y

    print("\nNormal-equations matrix X^T X:")
    for row in A:
        print("   ", "  ".join(f"{v:14.4e}" for v in row))

    # Conditioning via our own power / inverse power iterations
    try:
        diag = condition_number(A)
    except SingularMatrixError as exc:
        print(f"\n  Cholesky failed inside the eigenvalue solver: {exc}")
        print("  Conditioning cannot be measured -- the matrix is numerically singular.")
        return None

    print(f"\n  lambda_max          = {diag['lambda_max']:.6e}"
          f"   ({diag['iterations_max']} power iterations)")
    print(f"  lambda_min          = {diag['lambda_min']:.6e}"
          f"   ({diag['iterations_min']} inverse iterations)")
    print(f"  kappa_2(X^T X)      = {diag['kappa']:.4e}")
    print(f"  1 / eps_machine     = {1.0 / EPS_MACHINE:.4e}")
    print(f"  decimal digits lost = {diag['digits_lost']:.1f} of ~16 available")

    if diag["kappa"] > 1.0 / EPS_MACHINE:
        verdict = "NO correct digits survive -- solution is meaningless"
    elif diag["digits_lost"] > 10:
        verdict = "fewer than 6 correct digits remain -- unusable for inference"
    elif diag["digits_lost"] > 5:
        verdict = "roughly 10 correct digits remain -- acceptable"
    else:
        verdict = "well conditioned"
    print(f"  verdict             = {verdict}")

    # Error amplification: what does a 1% census error become?
    amplified = diag["kappa"] * 0.01
    print(f"\n  A 1% error in the input data can be amplified to a relative")
    print(f"  error of up to {amplified:.3e} in the fitted coefficients.")

    # Solve the system two ways
    print("\n  Solving the normal equations:")
    for name, solver in (("Cholesky", cholesky_solve), ("Gauss-Jordan", gauss_jordan)):
        try:
            beta = solver(A, b)
            res = residual_norm(A, beta, b)
            rel = res / np.sqrt(b @ b)
            print(f"    {name:<13} beta = ["
                  + ", ".join(f"{v:12.5f}" for v in beta)
                  + f"]   relative residual = {rel:.2e}")
        except SingularMatrixError as exc:
            print(f"    {name:<13} FAILED: {exc}")

    return diag


def main():
    states, y, pop, gdp = load_data()
    print(f"Loaded {len(states)} states.")
    print(f"  ports            : {y.min():>12,.0f} to {y.max():>12,.0f}")
    print(f"  population       : {pop.min():>12,.0f} to {pop.max():>12,.0f}")
    print(f"  gdp per capita   : {gdp.min():>12,.0f} to {gdp.max():>12,.0f}")
    print(f"\n  Dynamic range across columns spans {np.log10(pop.max() / 1.0):.0f}"
          " orders of magnitude, since the intercept column is all ones")
    print("  while the population column reaches ~4e7. This mismatch is the")
    print("  source of the conditioning problem measured below.")

    raw = report("EXPERIMENT A -- raw units (persons, dollars)",
                 build_design_matrix(pop, gdp, scaled=False), y)

    scl = report("EXPERIMENT B -- scaled units (millions of persons, thousands of dollars)",
                 build_design_matrix(pop, gdp, scaled=True), y)

    if raw and scl:
        print("\n" + "=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        factor = raw["kappa"] / scl["kappa"]
        print(f"\n  Column scaling reduced kappa_2 by a factor of {factor:.3e}")
        print(f"  ({raw['digits_lost']:.1f} digits lost  ->  {scl['digits_lost']:.1f} digits lost).")
        print("\n  The mathematical model is identical in both experiments. Only the")
        print("  units changed. Scaling is therefore not cosmetic: it is a")
        print("  prerequisite for the normal-equations approach to produce any")
        print("  trustworthy coefficients at all on this dataset.")
        print("\n  Design decision: the Regression Engine will scale all predictor")
        print("  columns to unit order of magnitude before forming X^T X.")


if __name__ == "__main__":
    main()
