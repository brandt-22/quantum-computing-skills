"""QSVT linear-system solver for emulated (ideal, noiseless) qubits.

Pieces:
  inverse_poly(kappa)   odd Chebyshev polynomial P(x) ~ 1/(2*kappa*x) on [1/kappa, 1] with |P| <= 1 on [-1, 1]
  qsvt_angles(cheb)     QSVT phase angles (pyqsp Laurent solver, converted QSP->QSVT convention by PennyLane)
  solve(A, b, ...)      block-encode A (dense unitary dilation), run QSVT on |b>, return the normalised solution direction

Honest scope: this emulates the *algorithm* (block encoding + polynomial phase sequence) on a statevector.
The real part of the QSVT block is taken by combining the +phi and -phi sequences (equivalent to an LCU with one extra
ancilla, post-selected); success probability and degree are reported.
"""
import time
import numpy as np


def inverse_poly(kappa, c=7.0, rel_tol=1e-2, dmax=1501):
    """Odd Chebyshev coefficients of g(x) = (1 - exp(-c (kappa x)^2)) / (2 kappa x), the smallest odd degree meeting rel_tol on [1/kappa, 1].

    g is entire, so its Chebyshev series converges super-geometrically; g(x) -> 1/(2 kappa x) for kappa x >> 1, max|g| < 1."""
    M = 8 * dmax
    j = np.arange(M)
    x = np.cos(np.pi * (j + 0.5) / M)
    g = (1.0 - np.exp(-c * (kappa * x) ** 2)) / (2.0 * kappa * x)
    k = np.arange(dmax + 1)
    # Chebyshev-Gauss quadrature: a_k = (2/M) sum_j g(x_j) cos(k theta_j)   (k >= 1)
    theta = np.pi * (j + 0.5) / M
    a = np.array([(2.0 / M) * np.dot(g, np.cos(kk * theta)) for kk in k])
    a[0::2] = 0.0  # odd function
    xs = np.linspace(1.0 / kappa, 1.0, 400)
    target = 1.0 / (2.0 * kappa * xs)
    for d in range(1, dmax + 1, 2):
        P = np.polynomial.chebyshev.chebval(xs, a[: d + 1])
        if np.max(np.abs(P - target) / target) < rel_tol:
            return a[: d + 1], d
    raise RuntimeError("no polynomial found below dmax")


def qsvt_angles(cheb):
    """QSVT phase angles for an odd real Chebyshev polynomial (|P| <= 1) using pyqsp, converted for PennyLane's PCPhase convention."""
    import pennylane as qml
    from pyqsp.angle_sequence import QuantumSignalProcessingPhases
    poly = np.polynomial.chebyshev.Chebyshev(cheb)
    phi = QuantumSignalProcessingPhases(poly, method="laurent", measurement="x", signal_operator="Wx")
    return np.array(qml.transform_angles(np.array(phi), "QSP", "QSVT"))


def block_encode_unitary(A, alpha):
    """Dense unitary dilation U = [[A/a, sqrt(I - A A^T/a^2)], [sqrt(I - A^T A/a^2), -A^T/a]] for real A, via SVD (what qml.BlockEncode builds)."""
    N = A.shape[0]
    Asc = A / alpha
    W, s, Vt = np.linalg.svd(Asc)
    C = np.sqrt(np.clip(1.0 - s ** 2, 0.0, None))
    left = (W * C) @ W.T          # sqrt(I - A A^T)
    right = (Vt.T * C) @ Vt       # sqrt(I - A^T A)
    return np.block([[Asc, left], [right, -Asc.T]]), s
