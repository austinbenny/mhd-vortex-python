"""Unit tests for the HLLD Riemann solver."""

from __future__ import annotations

import numpy as np

from vortex.equations import (
    IBX,
    IBY,
    IBZ,
    IEN,
    IMX,
    IMY,
    IMZ,
    IRHO,
    NVAR,
    flux_x,
    flux_y,
    prim_to_cons,
)
from vortex.riemann import hlld_x, hlld_y


def _zero_state(shape=(1, 1)):
    return np.zeros((NVAR, *shape))


def test_hlld_constant_state_matches_analytic():
    Q = _zero_state((2, 2))
    Q[IRHO] = 1.0
    Q[IMX] = 0.1
    Q[IMY] = -0.2
    Q[IMZ] = 0.05
    Q[IBX] = 0.3
    Q[IBY] = -0.1
    Q[IBZ] = 0.2
    Q[IEN] = 0.8
    F_hlld = hlld_x(Q, Q, ch=1.0)
    U = prim_to_cons(Q)
    F_ana = flux_x(U, ch=1.0)
    assert np.allclose(F_hlld, F_ana, atol=1e-13)

    G_hlld = hlld_y(Q, Q, ch=1.0)
    G_ana = flux_y(U, ch=1.0)
    assert np.allclose(G_hlld, G_ana, atol=1e-13)


def test_hlld_euler_limit_sod():
    """B=0 Sod tube should produce a positive mass flux at the interface."""
    QL = _zero_state()
    QR = _zero_state()
    QL[IRHO] = 1.0
    QL[IEN] = 1.0
    QR[IRHO] = 0.125
    QR[IEN] = 0.1
    F = hlld_x(QL, QR, ch=1.0)
    assert F[IRHO, 0, 0] > 0.35
    assert F[IRHO, 0, 0] < 0.50


def test_hlld_mirror_symmetry():
    """Flipping the Riemann states (QL<->QR with u-> -u, Bx-> -Bx) flips the mass flux."""
    rng = np.random.default_rng(123)
    shape = (1, 1)
    QL = _zero_state(shape)
    QR = _zero_state(shape)
    QL[IRHO] = 1.0
    QR[IRHO] = 0.5
    QL[IMX] = 0.2
    QR[IMX] = -0.3
    QL[IMY] = 0.0
    QR[IMY] = 0.0
    QL[IBX] = 0.1
    QR[IBX] = 0.1
    QL[IBY] = 0.05
    QR[IBY] = -0.05
    QL[IEN] = 1.0
    QR[IEN] = 0.6
    F = hlld_x(QL, QR, ch=1.0)

    QLp = QR.copy()
    QRp = QL.copy()
    QLp[IMX] *= -1.0
    QRp[IMX] *= -1.0
    QLp[IBX] *= -1.0
    QRp[IBX] *= -1.0
    Fp = hlld_x(QLp, QRp, ch=1.0)
    # Mirror symmetry: mass flux flips sign, pressure flux unchanged.
    assert np.allclose(Fp[IRHO], -F[IRHO], atol=1e-12)
    _ = rng  # reserved for future randomized checks
