"""Ideal MHD equation of state, fluxes, and variable conversions.

State layout (axis 0 of all arrays):

    0: rho         mass density
    1: rho*u       x-momentum
    2: rho*v       y-momentum
    3: rho*w       z-momentum
    4: Bx          magnetic field x
    5: By          magnetic field y
    6: Bz          magnetic field z
    7: E           total energy density
    8: psi         GLM divergence-cleaning scalar

Primitive layout mirrors conserved with ``[rho, u, v, w, Bx, By, Bz, p, psi]``.

Conventions follow Miyoshi and Kusano (2005), Sec. 2 and the GLM coupling of
Dedner et al. (2002), Sec. 3. The fluxes returned here are the *analytic*
conservative fluxes used directly in the Riemann solver.
"""

from __future__ import annotations

import numpy as np

GAMMA: float = 5.0 / 3.0

IRHO, IMX, IMY, IMZ, IBX, IBY, IBZ, IEN, IPSI = range(9)
NVAR: int = 9


def cons_to_prim(U: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Convert conserved state to primitive."""
    rho = U[IRHO]
    inv_rho = 1.0 / rho
    u = U[IMX] * inv_rho
    v = U[IMY] * inv_rho
    w = U[IMZ] * inv_rho
    bx, by, bz = U[IBX], U[IBY], U[IBZ]
    kin = 0.5 * rho * (u * u + v * v + w * w)
    mag = 0.5 * (bx * bx + by * by + bz * bz)
    p = (gamma - 1.0) * (U[IEN] - kin - mag)
    Q = np.empty_like(U)
    Q[IRHO] = rho
    Q[IMX] = u
    Q[IMY] = v
    Q[IMZ] = w
    Q[IBX] = bx
    Q[IBY] = by
    Q[IBZ] = bz
    Q[IEN] = p
    Q[IPSI] = U[IPSI]
    return Q


def prim_to_cons(Q: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Convert primitive state to conserved."""
    rho = Q[IRHO]
    u, v, w = Q[IMX], Q[IMY], Q[IMZ]
    bx, by, bz = Q[IBX], Q[IBY], Q[IBZ]
    p = Q[IEN]
    kin = 0.5 * rho * (u * u + v * v + w * w)
    mag = 0.5 * (bx * bx + by * by + bz * bz)
    U = np.empty_like(Q)
    U[IRHO] = rho
    U[IMX] = rho * u
    U[IMY] = rho * v
    U[IMZ] = rho * w
    U[IBX] = bx
    U[IBY] = by
    U[IBZ] = bz
    U[IEN] = p / (gamma - 1.0) + kin + mag
    U[IPSI] = Q[IPSI]
    return U


def pressure(U: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    rho = U[IRHO]
    kin = 0.5 * (U[IMX] ** 2 + U[IMY] ** 2 + U[IMZ] ** 2) / rho
    mag = 0.5 * (U[IBX] ** 2 + U[IBY] ** 2 + U[IBZ] ** 2)
    return (gamma - 1.0) * (U[IEN] - kin - mag)


def fast_speed(
    rho: np.ndarray,
    p: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    bn: np.ndarray,
    gamma: float = GAMMA,
) -> np.ndarray:
    """Fast magnetosonic speed in the direction whose normal B-component is ``bn``.

    Uses the standard expression (Miyoshi & Kusano 2005 eq. 3):

        c_f^2 = 0.5 * ( a^2 + b^2/rho + sqrt((a^2 + b^2/rho)^2 - 4 a^2 bn^2/rho) )

    where ``a^2 = gamma p / rho`` and ``b^2 = Bx^2 + By^2 + Bz^2``.
    """
    a2 = gamma * np.maximum(p, 1e-20) / rho
    b2 = (bx * bx + by * by + bz * bz) / rho
    bn2 = bn * bn / rho
    s = a2 + b2
    disc = np.maximum(s * s - 4.0 * a2 * bn2, 0.0)
    return np.sqrt(0.5 * (s + np.sqrt(disc)))


def flux_x(U: np.ndarray, ch: float, gamma: float = GAMMA) -> np.ndarray:
    """Analytic x-direction flux including GLM psi coupling."""
    rho = U[IRHO]
    inv_rho = 1.0 / rho
    u, v, w = U[IMX] * inv_rho, U[IMY] * inv_rho, U[IMZ] * inv_rho
    bx, by, bz = U[IBX], U[IBY], U[IBZ]
    psi = U[IPSI]
    p = (gamma - 1.0) * (
        U[IEN]
        - 0.5 * rho * (u * u + v * v + w * w)
        - 0.5 * (bx * bx + by * by + bz * bz)
    )
    ptot = p + 0.5 * (bx * bx + by * by + bz * bz)
    vdotb = u * bx + v * by + w * bz

    F = np.empty_like(U)
    F[IRHO] = rho * u
    F[IMX] = rho * u * u + ptot - bx * bx
    F[IMY] = rho * u * v - bx * by
    F[IMZ] = rho * u * w - bx * bz
    F[IBX] = psi
    F[IBY] = u * by - v * bx
    F[IBZ] = u * bz - w * bx
    F[IEN] = (U[IEN] + ptot) * u - bx * vdotb
    F[IPSI] = ch * ch * bx
    return F


def flux_y(U: np.ndarray, ch: float, gamma: float = GAMMA) -> np.ndarray:
    """Analytic y-direction flux including GLM psi coupling."""
    rho = U[IRHO]
    inv_rho = 1.0 / rho
    u, v, w = U[IMX] * inv_rho, U[IMY] * inv_rho, U[IMZ] * inv_rho
    bx, by, bz = U[IBX], U[IBY], U[IBZ]
    psi = U[IPSI]
    p = (gamma - 1.0) * (
        U[IEN]
        - 0.5 * rho * (u * u + v * v + w * w)
        - 0.5 * (bx * bx + by * by + bz * bz)
    )
    ptot = p + 0.5 * (bx * bx + by * by + bz * bz)
    vdotb = u * bx + v * by + w * bz

    G = np.empty_like(U)
    G[IRHO] = rho * v
    G[IMX] = rho * v * u - by * bx
    G[IMY] = rho * v * v + ptot - by * by
    G[IMZ] = rho * v * w - by * bz
    G[IBX] = v * bx - u * by
    G[IBY] = psi
    G[IBZ] = v * bz - w * by
    G[IEN] = (U[IEN] + ptot) * v - by * vdotb
    G[IPSI] = ch * ch * by
    return G


def swap_xy(U: np.ndarray) -> np.ndarray:
    """Return a view-like copy of ``U`` with x/y components swapped.

    After this swap, the y-direction problem has the same shape as an
    x-direction problem, so the x-only HLLD routine can be reused.
    """
    S = U.copy()
    S[IMX], S[IMY] = U[IMY].copy(), U[IMX].copy()
    S[IBX], S[IBY] = U[IBY].copy(), U[IBX].copy()
    return S
