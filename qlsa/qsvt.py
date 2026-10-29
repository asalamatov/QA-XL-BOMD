"""Matrix-level emulation of a QSVT quantum linear system solver.

QSVT applied to a block-encoding of A = M / alpha with an odd polynomial p
produces, after post-selection, a state proportional to V p(Sigma) U^T |b>,
where A = U Sigma V^T.  Choosing p ~ 1/x on [1/kappa, 1] gives a state
proportional to A^{-1}|b>.  Because a QSVT circuit with exact phase factors
implements p exactly, applying p to the singular values reproduces the
algorithm's output state, including its polynomial-approximation error.
What this does NOT model: gate-level noise, phase-factor finding error,
the success probability / amplitude amplification cost, and readout
(tomography or shadows) sampling error.

Polynomial: Childs, Kothari & Somma, SIAM J. Comput. 46, 1920 (2017),
Lemmas 17-19: f_b(x) = (1 - (1 - x^2)^b) / x with b = ceil(kappa^2 ln(kappa/eps)),
truncated to its first j0+1 odd Chebyshev terms,
j0 = ceil(sqrt(b ln(4b/eps))).  Its degree, 2*j0 + 1, is the number of
queries to the block-encoding.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.stats import binom

MAX_DEGREE = 4_000_001


@dataclass
class QSVTResult:
    x_state: np.ndarray        # normalized output state, same shape as rhs
    degree: int                # polynomial degree = block-encoding queries
    kappa_eff: float           # alpha / sigma_min, the kappa QSVT must handle
    alpha: float               # block-encoding normalization
    feasible: bool


def cks_chebyshev_coeffs(kappa: float, eps: float):
    b = int(np.ceil(kappa**2 * np.log(kappa / eps)))
    j0 = int(np.ceil(np.sqrt(b * np.log(4.0 * b / eps))))
    degree = 2 * j0 + 1
    if degree > MAX_DEGREE:
        return None, degree
    j = np.arange(j0 + 1)
    # coefficient of T_{2j+1}: 4 (-1)^j Pr[X >= b + j + 1], X ~ Bin(2b, 1/2)
    tail = binom.sf(b + j, 2 * b, 0.5)
    coef = np.zeros(degree + 1)
    coef[2 * j + 1] = 4.0 * ((-1.0) ** j) * tail
    return coef, degree


def block_encoding_alpha(M: np.ndarray, mode: str = "sparse") -> float:
    """Normalization alpha of the block-encoding U_M of M/alpha.

    'sparse': the standard sparse-access construction, alpha = s * max|M_ij|
              with s the row sparsity padded to a power of two (as in the
              manuscript's Lemma, where s = 4).
    'ideal' : alpha = ||M||_2, the best possible normalization.
    """
    if mode == "ideal":
        return float(np.linalg.norm(M, 2))
    s = int(max((M != 0).sum(axis=1).max(), (M != 0).sum(axis=0).max()))
    s_pad = 1 << (s - 1).bit_length()
    return float(s_pad * np.abs(M).max())


def qsvt_solve(M: np.ndarray, rhs: np.ndarray, eps: float,
               alpha_mode: str = "sparse") -> QSVTResult:
    """Emulate QSVT inversion of M on rhs (shape (n,) or (n, k) for M (x) I_k)."""
    alpha = block_encoding_alpha(M, alpha_mode)
    A = M / alpha
    U, s, Vt = np.linalg.svd(A)
    kappa = 1.0 / s.min()
    coef, degree = cks_chebyshev_coeffs(max(kappa, 1.0 + 1e-12), eps)
    if coef is None:
        return QSVTResult(np.full_like(rhs, np.nan, dtype=float), degree, kappa, alpha, False)
    p = C.chebval(s, coef)
    y = Vt.T @ (p[:, None] * (U.T @ rhs.reshape(rhs.shape[0], -1)))
    y = y.reshape(rhs.shape)
    return QSVTResult(y / np.linalg.norm(y), degree, kappa, alpha, True)
