"""
matrix_solver.py
================
Direct solvers, eigenvalue iterations, and conditioning diagnostics.
"""

import numpy as np


class SingularMatrixError(Exception):
    """Raised when a matrix is singular or has lost positive-definiteness."""
    pass


# ---------------------------------------------------------------------------
# Direct method 1: Gauss-Jordan elimination with partial pivoting
# ---------------------------------------------------------------------------

def gauss_jordan(A, b):
    """
    Solve A x = b by Gauss-Jordan elimination with partial pivoting.

    Reduces the augmented matrix [A|b] to reduced row-echelon form, so the
    solution is read directly from the final column with no back-substitution.
    Partial pivoting swaps in the row with the largest pivot magnitude at each
    step, which bounds the growth of round-off error.

    Operation count: ~n^3 multiplications (Altac, 2024, Ch. 2).

    Parameters
    ----------
    A : (n, n) array_like -- coefficient matrix
    b : (n,)   array_like -- right-hand side

    Returns
    -------
    x : (n,) ndarray -- solution vector

    Raises
    ------
    SingularMatrixError -- if no non-zero pivot can be found
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]

    # Build the augmented matrix [A | b]
    aug = np.hstack([A, b.reshape(-1, 1)])

    for col in range(n):
        # --- Partial pivoting: find the row with the largest |pivot| ---
        pivot_row = col + int(np.argmax(np.abs(aug[col:, col])))
        if abs(aug[pivot_row, col]) < 1e-14:
            raise SingularMatrixError(
                f"No usable pivot in column {col}; matrix is singular."
            )
        if pivot_row != col:
            aug[[col, pivot_row]] = aug[[pivot_row, col]]

        # --- Normalise the pivot row so the pivot becomes 1 ---
        aug[col] = aug[col] / aug[col, col]

        # --- Eliminate this column from every other row ---
        for row in range(n):
            if row != col:
                aug[row] = aug[row] - aug[row, col] * aug[col]

    return aug[:, -1]


# ---------------------------------------------------------------------------
# Direct method 2: Cholesky decomposition
# ---------------------------------------------------------------------------

def cholesky_decompose(A):
    """
    Factor a symmetric positive-definite matrix as A = L L^T.

    Cholesky requires roughly half the operations of LU (~n^3/6 vs ~n^3/3)
    because it exploits symmetry, and it needs no pivoting when A is genuinely
    positive definite. Critically, the algorithm requires a square root of a
    quantity that stays positive only while A remains positive definite -- so a
    negative radicand is a direct numerical warning that conditioning has been
    lost (Altac, 2024, Ch. 2).

    Parameters
    ----------
    A : (n, n) array_like -- symmetric positive-definite matrix

    Returns
    -------
    L : (n, n) ndarray -- lower-triangular factor

    Raises
    ------
    SingularMatrixError -- if a negative or zero radicand is encountered
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))

            if i == j:
                radicand = A[i, i] - s
                if radicand <= 0.0:
                    raise SingularMatrixError(
                        f"Cholesky failed at pivot {i}: radicand = {radicand:.3e}. "
                        "The matrix is not numerically positive definite."
                    )
                L[i, j] = np.sqrt(radicand)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]

    return L


def forward_substitution(L, b):
    """Solve L y = b for lower-triangular L (Altac, 2024, Ch. 2)."""
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float)
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i, k] * y[k] for k in range(i))) / L[i, i]
    return y


def back_substitution(U, y):
    """Solve U x = y for upper-triangular U (Altac, 2024, Ch. 2)."""
    U = np.asarray(U, dtype=float)
    y = np.asarray(y, dtype=float)
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, k] * x[k] for k in range(i + 1, n))) / U[i, i]
    return x


def cholesky_solve(A, b):
    """
    Solve A x = b for symmetric positive-definite A using Cholesky.

    Two triangular solves follow the factorisation:
        A x = b  ->  L L^T x = b  ->  L y = b, then L^T x = y
    """
    L = cholesky_decompose(A)
    y = forward_substitution(L, b)
    x = back_substitution(L.T, y)
    return x


# ---------------------------------------------------------------------------
# Eigenvalue iterations (Ch. 11)
# ---------------------------------------------------------------------------

def power_method(A, tol=1e-12, max_iter=1000):
    """
    Find the dominant eigenvalue of A by power iteration.

    Repeatedly applying A to a vector amplifies the component along the
    eigenvector of largest |eigenvalue|. Normalising each iterate prevents
    overflow. The Rayleigh quotient x^T A x / x^T x gives the eigenvalue
    estimate (Altac, 2024, Ch. 11).

    Convergence is linear with ratio |lambda_2 / lambda_1|.

    Returns
    -------
    lam        : float -- dominant eigenvalue
    x          : ndarray -- corresponding unit eigenvector
    iterations : int -- iterations used
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    x = np.ones(n) / np.sqrt(n)   # deterministic start, so results reproduce
    lam_old = 0.0

    for it in range(1, max_iter + 1):
        Ax = A @ x
        norm = np.sqrt(Ax @ Ax)
        if norm < 1e-300:
            raise SingularMatrixError("Power method collapsed to the zero vector.")
        x = Ax / norm

        lam = x @ (A @ x)         # Rayleigh quotient
        if abs(lam - lam_old) < tol * max(abs(lam), 1.0):
            return lam, x, it
        lam_old = lam

    return lam_old, x, max_iter


def inverse_power_method(A, tol=1e-12, max_iter=1000):
    """
    Find the smallest-magnitude eigenvalue of A by inverse power iteration.

    The eigenvalues of A^-1 are the reciprocals of those of A, so the dominant
    eigenvalue of A^-1 corresponds to the smallest of A. Rather than inverting
    A explicitly, each iteration solves A z = x using the Cholesky factors,
    which are computed once and reused (Altac, 2024, Ch. 11).

    Returns
    -------
    lam        : float -- smallest eigenvalue
    x          : ndarray -- corresponding unit eigenvector
    iterations : int -- iterations used
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    L = cholesky_decompose(A)     # factor once, reuse every iteration
    x = np.ones(n) / np.sqrt(n)
    lam_old = 0.0

    for it in range(1, max_iter + 1):
        y = forward_substitution(L, x)
        z = back_substitution(L.T, y)     # z = A^-1 x

        norm = np.sqrt(z @ z)
        x = z / norm

        lam = x @ (A @ x)         # Rayleigh quotient on A, not A^-1
        if abs(lam - lam_old) < tol * max(abs(lam), 1.0):
            return lam, x, it
        lam_old = lam

    return lam_old, x, max_iter


# ---------------------------------------------------------------------------
# Conditioning diagnostics (CLO 1)
# ---------------------------------------------------------------------------

def condition_number(A):
    """
    Spectral condition number of a symmetric positive-definite matrix.

        kappa_2(A) = lambda_max / lambda_min

    computed from the power method and the inverse power method rather than
    from any library routine.

    Interpretation: kappa_2 bounds error amplification. A relative
    perturbation of size eps in the data can produce a relative error of up to
    kappa_2 * eps in the solution. Once kappa_2 approaches 1/eps_machine
    (~4.5e15 in IEEE double precision) no correct digits survive.

    Returns
    -------
    dict with keys: lambda_max, lambda_min, kappa, digits_lost
    """
    lam_max, _, it_max = power_method(A)
    lam_min, _, it_min = inverse_power_method(A)

    kappa = lam_max / lam_min
    digits_lost = np.log10(kappa) if kappa > 0 else float("inf")

    return {
        "lambda_max": lam_max,
        "lambda_min": lam_min,
        "kappa": kappa,
        "digits_lost": digits_lost,
        "iterations_max": it_max,
        "iterations_min": it_min,
    }


def residual_norm(A, x, b):
    """Euclidean norm ||A x - b||_2, used to verify a computed solution."""
    r = np.asarray(A, dtype=float) @ np.asarray(x, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(r @ r))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 66)
    print("matrix_solver.py -- verification against hand-checkable problems")
    print("=" * 66)

    # A symmetric positive-definite system with an exact integer solution
    A = np.array([[4.0, 12.0, -16.0],
                  [12.0, 37.0, -43.0],
                  [-16.0, -43.0, 98.0]])
    x_true = np.array([1.0, 2.0, 3.0])
    b = A @ x_true

    x_gj = gauss_jordan(A, b)
    x_ch = cholesky_solve(A, b)

    print("\nTest system: 3x3 symmetric positive definite, exact x = [1, 2, 3]")
    print(f"  Gauss-Jordan  x = {x_gj}   residual = {residual_norm(A, x_gj, b):.2e}")
    print(f"  Cholesky      x = {x_ch}   residual = {residual_norm(A, x_ch, b):.2e}")

    # Cholesky factor of this matrix is the textbook example L = [[2,0,0],[6,1,0],[-8,5,3]]
    L = cholesky_decompose(A)
    print("\nCholesky factor L (expected rows [2,0,0], [6,1,0], [-8,5,3]):")
    for row in L:
        print("   ", np.array2string(row, precision=4, suppress_small=True))
    print(f"  Reconstruction error ||L L^T - A|| = {np.abs(L @ L.T - A).max():.2e}")

    # Eigenvalues of a matrix with known spectrum {1, 2, 4}
    D = np.diag([4.0, 2.0, 1.0])
    lam_max, _, i1 = power_method(D)
    lam_min, _, i2 = inverse_power_method(D)
    print(f"\nDiagonal matrix with spectrum {{4, 2, 1}}:")
    print(f"  power method         lambda_max = {lam_max:.10f}  ({i1} iterations)")
    print(f"  inverse power method lambda_min = {lam_min:.10f}  ({i2} iterations)")
    print(f"  kappa = {lam_max / lam_min:.4f}  (expected 4.0000)")

    print("\nAll checks passed." if abs(lam_max - 4) < 1e-8 else "\nCHECK FAILED")
