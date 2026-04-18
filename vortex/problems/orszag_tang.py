"""Orszag-Tang vortex initial conditions.

Canonical 2D MHD benchmark. Periodic domain ``[0,1]^2`` with

    rho = 25 / (36 pi)     -> normalized to 1 after scaling of p
    p = 5 / (12 pi)        -> normalized to 1/gamma
    u = -sin(2 pi y)
    v =  sin(2 pi x)
    w = 0
    Bx = -sin(2 pi y) / gamma^{1/2}    ... conventions vary
    By =  sin(4 pi x) / gamma^{1/2}
    Bz = 0

We use the non-dimensional form from Londrillo & Del Zanna (2000) and the
Athena++ test page:

    rho = 1
    p   = 1/gamma
    V   = (-sin 2 pi y,  sin 2 pi x, 0)
    B   = ((1/sqrt(4 pi)) -> rescaled) (-sin 2 pi y, sin 4 pi x, 0)

Both conventions produce the same qualitative density and pressure contours at
t=0.5 once the scaling of B is consistent with p. The solver uses the
"gamma-normalized" form with B_0 = 1/gamma so that beta_0 = 2 p_0 / B_0^2 =
2 gamma.
"""

from __future__ import annotations

import numpy as np

from vortex.equations import GAMMA, NVAR, prim_to_cons
from vortex.mesh import Mesh


def initial_conditions(mesh: Mesh, gamma: float = GAMMA) -> np.ndarray:
    """Return the initial conserved-state array for Orszag-Tang."""
    xc, yc = mesh.cell_centers()
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    B0 = 1.0 / gamma
    rho = np.ones_like(X)
    p = np.full_like(X, 1.0 / gamma)
    u = -np.sin(2.0 * np.pi * Y)
    v = np.sin(2.0 * np.pi * X)
    w = np.zeros_like(X)
    bx = -B0 * np.sin(2.0 * np.pi * Y)
    by = B0 * np.sin(4.0 * np.pi * X)
    bz = np.zeros_like(X)
    psi = np.zeros_like(X)

    Q = np.stack([rho, u, v, w, bx, by, bz, p, psi], axis=0)
    assert Q.shape[0] == NVAR
    return prim_to_cons(Q, gamma)
