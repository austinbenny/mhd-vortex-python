"""Unit tests for the equation-of-state and flux helpers."""

from __future__ import annotations

import numpy as np

from vortex.equations import (
    GAMMA,
    IBX,
    IBY,
    IBZ,
    IEN,
    IMX,
    IMY,
    IMZ,
    IPSI,
    IRHO,
    NVAR,
    cons_to_prim,
    flux_x,
    flux_y,
    prim_to_cons,
)


def _random_prim(shape, rng):
    Q = np.zeros((NVAR, *shape))
    Q[IRHO] = rng.uniform(0.5, 2.0, size=shape)
    Q[IMX] = rng.uniform(-0.5, 0.5, size=shape)
    Q[IMY] = rng.uniform(-0.5, 0.5, size=shape)
    Q[IMZ] = rng.uniform(-0.5, 0.5, size=shape)
    Q[IBX] = rng.uniform(-0.3, 0.3, size=shape)
    Q[IBY] = rng.uniform(-0.3, 0.3, size=shape)
    Q[IBZ] = rng.uniform(-0.3, 0.3, size=shape)
    Q[IEN] = rng.uniform(0.5, 1.5, size=shape)
    Q[IPSI] = rng.uniform(-0.01, 0.01, size=shape)
    return Q


def test_prim_cons_roundtrip():
    rng = np.random.default_rng(42)
    Q = _random_prim((4, 5), rng)
    U = prim_to_cons(Q)
    Q_back = cons_to_prim(U)
    assert np.allclose(Q, Q_back, atol=1e-14)


def test_flux_x_euler_limit():
    """With B=0 and psi=0, the MHD flux reduces to the Euler flux."""
    rng = np.random.default_rng(0)
    Q = _random_prim((3, 3), rng)
    Q[IBX] = 0.0
    Q[IBY] = 0.0
    Q[IBZ] = 0.0
    Q[IPSI] = 0.0
    U = prim_to_cons(Q)
    F = flux_x(U, ch=1.0)

    rho, u, v, w, p = Q[IRHO], Q[IMX], Q[IMY], Q[IMZ], Q[IEN]
    E = U[IEN]
    assert np.allclose(F[IRHO], rho * u)
    assert np.allclose(F[IMX], rho * u * u + p)
    assert np.allclose(F[IMY], rho * u * v)
    assert np.allclose(F[IMZ], rho * u * w)
    assert np.allclose(F[IEN], (E + p) * u)


def test_flux_y_matches_flux_x_with_swap():
    """Swapping x<->y components on Q should swap the x/y components of the flux."""
    rng = np.random.default_rng(1)
    Q = _random_prim((3, 3), rng)
    U = prim_to_cons(Q)

    Gy = flux_y(U, ch=0.7, gamma=GAMMA)

    # Manually swap u<->v and Bx<->By.
    Q_swap = Q.copy()
    Q_swap[IMX], Q_swap[IMY] = Q[IMY].copy(), Q[IMX].copy()
    Q_swap[IBX], Q_swap[IBY] = Q[IBY].copy(), Q[IBX].copy()
    U_swap = prim_to_cons(Q_swap)
    Fx_swap = flux_x(U_swap, ch=0.7, gamma=GAMMA)

    # Unswap components for comparison.
    Gy_expected = Fx_swap.copy()
    Gy_expected[IMX], Gy_expected[IMY] = Fx_swap[IMY].copy(), Fx_swap[IMX].copy()
    Gy_expected[IBX], Gy_expected[IBY] = Fx_swap[IBY].copy(), Fx_swap[IBX].copy()

    assert np.allclose(Gy, Gy_expected, atol=1e-14)
