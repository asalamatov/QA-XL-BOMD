"""Linear XL-BOMD recurrences under the linear SCF surrogate, and the
time-grid matrices that encode them.

With the surrogate D(P) = (1 - gamma) D* + gamma P, every XL-BOMD update
used here becomes a constant-coefficient linear recurrence

    P_{n+1} = sum_{j=0}^{m-1} a_j P_{n-j} + c,

where c is a fixed matrix proportional to D*.  Because the coefficients are
scalars, the stacked system over all time steps factors as
M = M_time (x) I_{d^2}, so all solver properties (singular values, condition
number, QLSA polynomial error) are those of the (T+1) x (T+1) matrix M_time.

Two schemes are provided:

* ``paper``  : the two-step dissipative update of the manuscript, Eq. (xlbomd-recurrence),
               which is Eq. (24) of Niklasson et al., JCP 130, 214109 (2009):
               P_{n+1} = 2P_n - P_{n-1} + (2-alpha)(D_n-P_n) - alpha(D_{n-1}-P_{n-1})
* ``pyseqm`` : the scheme PySEQM's XL_BOMD class actually integrates
               (Niklasson et al., JCP 130, 214109 (2009), with PySEQM's
               c = 0.95 mixing), for history length k.
"""
from __future__ import annotations

import numpy as np

# Niklasson et al. (2009) coefficients exactly as tabulated in
# PySEQM seqm/MolecularDynamics.py (commit 9163fad): kappa, alpha, c0..ck
NIKLASSON_2009 = {
    3: [1.69, 150e-3, -2.0, 3.0, 0.0, -1.0],
    4: [1.75, 57e-3, -3.0, 6.0, -2.0, -2.0, 1.0],
    5: [1.82, 18e-3, -6.0, 14.0, -8.0, -3.0, 4.0, -1.0],
    6: [1.84, 5.5e-3, -14.0, 36.0, -27.0, -2.0, 12.0, -6.0, 1.0],
    7: [1.86, 1.6e-3, -36.0, 99.0, -88.0, 11.0, 32.0, -25.0, 8.0, -1.0],
}
PYSEQM_MIX = 0.95  # c in PySEQM's _propagate_P


def paper_coefficients(alpha: float, gamma: float):
    """Return (a, d_coef) with P_{n+1} = a[0] P_n + a[1] P_{n-1} + d_coef D*.

    Substituting D = (1-gamma) D* + gamma P into Niklasson's Eq. (24) gives
    a0 = 2 - (2-alpha)(1-gamma), a1 = -1 + alpha(1-gamma), d_coef = 2(1-alpha)(1-gamma).
    The fixed point is P = D*; the roots lie strictly inside the unit circle
    for 0 < alpha < 1 and -1 < gamma < 1.
    """
    a0 = 2.0 - (2.0 - alpha) * (1.0 - gamma)
    a1 = -1.0 + alpha * (1.0 - gamma)
    d_coef = 2.0 * (1.0 - alpha) * (1.0 - gamma)
    return np.array([a0, a1]), d_coef


def pyseqm_coefficients(k: int, gamma: float, mix: float = PYSEQM_MIX):
    """Coefficients of PySEQM's XL-BOMD update with D = (1-gamma)D* + gamma P.

    PySEQM:  P_{n+1} = kappa*(mix*D_n + (1-mix)*P_n) + sum_j coeff_j P_{n-j},
             coeff_0 = 2 - kappa + alpha*c0, coeff_1 = alpha*c1 - 1,
             coeff_j = alpha*c_j (j >= 2).
    """
    kappa, alpha, *c = NIKLASSON_2009[k]
    coeff = alpha * np.array(c, dtype=float)
    coeff[0] += 2.0 - kappa
    coeff[1] -= 1.0
    a = coeff.copy()
    a[0] += kappa * (mix * gamma + (1.0 - mix))
    d_coef = kappa * mix * (1.0 - gamma)
    return a, d_coef


def coefficients(scheme: str, gamma: float, alpha: float = 0.0, k: int = 6):
    if scheme == "paper":
        return paper_coefficients(alpha, gamma)
    if scheme == "pyseqm":
        return pyseqm_coefficients(k, gamma)
    raise ValueError(scheme)


def characteristic_roots(a: np.ndarray) -> np.ndarray:
    """Roots of lambda^m - a0 lambda^{m-1} - ... - a_{m-1} = 0."""
    return np.roots(np.concatenate([[1.0], -np.asarray(a)]))


def time_matrix(a: np.ndarray, T: int) -> np.ndarray:
    """(T+1)x(T+1) lower-triangular banded matrix M_time.

    The first m rows are identity rows that fix the initial history
    P_0..P_{m-1}; each later row n encodes
    P_n - sum_j a_j P_{n-1-j} = c.
    """
    m = len(a)
    n = T + 1
    if n <= m:
        raise ValueError("T must exceed the recurrence order")
    M = np.eye(n)
    for row in range(m, n):
        for j, aj in enumerate(a):
            M[row, row - 1 - j] = -aj
    return M


def rhs_time_basis(m: int, T: int) -> np.ndarray:
    """Time profiles of the right-hand side.

    b = sum_{i<m} e_i (x) vec(P_i) + 1_{>=m} (x) vec(c), returned as an
    array of shape (m+1, T+1): rows 0..m-1 are e_i, row m is 1_{>=m}.
    """
    B = np.zeros((m + 1, T + 1))
    for i in range(m):
        B[i, i] = 1.0
    B[m, m:] = 1.0
    return B


def classical_sequential(a, d_coef, history, Dstar, T):
    """Step-by-step propagation; history = [P_0, ..., P_{m-1}] (d x d each)."""
    m = len(a)
    P = [np.array(h, dtype=float) for h in history]
    c = d_coef * Dstar
    for n in range(m - 1, T):
        nxt = c.copy()
        for j, aj in enumerate(a):
            nxt += aj * P[n - j]
        P.append(nxt)
    return np.stack(P)  # (T+1, d, d)
