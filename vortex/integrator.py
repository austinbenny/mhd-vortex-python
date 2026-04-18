"""Right-hand side evaluator and SSP-RK2 time stepper.

The RHS returns ``-(dF/dx + dG/dy)`` for the interior of the domain. The outer
``run`` loop in ``vortex.solver`` handles ghost fills and GLM source-term
updates around the RK stages.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from vortex.boundary import apply as apply_bc
from vortex.equations import IBX, IBY, IMX, IMY, IRHO, cons_to_prim
from vortex.mesh import Mesh
from vortex.reconstruction import reconstruct_x, reconstruct_y
from vortex.riemann import hlld_x, hlld_y

RHSFn = Callable[[np.ndarray, float], np.ndarray]


def compute_dt(U: np.ndarray, mesh: Mesh, cfl: float, ch: float, gamma: float) -> float:
    """CFL-limited step using the maximum fast magnetosonic speed.

    ``dt = cfl * min(dx, dy) / max(|u|+c_f, |v|+c_f, ch)``
    """
    g = mesh.nghost
    interior = (slice(None), slice(g, g + mesh.nx), slice(g, g + mesh.ny))
    Ui = U[interior]
    rho = Ui[IRHO]
    u = Ui[IMX] / rho
    v = Ui[IMY] / rho
    bx, by, bz = Ui[IBX], Ui[IBY], Ui[6]
    inv_rho = 1.0 / rho
    a2 = gamma * _pressure(Ui, gamma) * inv_rho
    b2 = (bx * bx + by * by + bz * bz) * inv_rho
    bn2_x = bx * bx * inv_rho
    bn2_y = by * by * inv_rho
    cf_x = np.sqrt(0.5 * (a2 + b2 + np.sqrt((a2 + b2) ** 2 - 4.0 * a2 * bn2_x)))
    cf_y = np.sqrt(0.5 * (a2 + b2 + np.sqrt((a2 + b2) ** 2 - 4.0 * a2 * bn2_y)))
    max_sx = float(np.max(np.abs(u) + cf_x))
    max_sy = float(np.max(np.abs(v) + cf_y))
    sig = max(max_sx, max_sy, ch)
    return cfl * min(mesh.dx, mesh.dy) / sig


def _pressure(U: np.ndarray, gamma: float) -> np.ndarray:
    rho = U[IRHO]
    kin = 0.5 * (U[IMX] ** 2 + U[IMY] ** 2 + U[5] ** 2) / rho
    mag = 0.5 * (U[IBX] ** 2 + U[IBY] ** 2 + U[6] ** 2)
    return np.maximum((gamma - 1.0) * (U[7] - kin - mag), 1e-20)


def rhs(U: np.ndarray, mesh: Mesh, ch: float, limiter: str, gamma: float) -> np.ndarray:
    """Compute L(U) = -(dF/dx + dG/dy) on the interior cells.

    The returned array has interior shape ``(NVAR, nx, ny)``.
    """
    apply_bc(U, mesh)
    Q = cons_to_prim(U, gamma)

    QLx, QRx = reconstruct_x(Q, limiter=limiter)  # (NVAR, nx+1, ny)
    QLy, QRy = reconstruct_y(Q, limiter=limiter)  # (NVAR, nx, ny+1)

    Fx = hlld_x(QLx, QRx, ch=ch, gamma=gamma)  # (NVAR, nx+1, ny)
    Fy = hlld_y(QLy, QRy, ch=ch, gamma=gamma)  # (NVAR, nx, ny+1)

    dx, dy = mesh.dx, mesh.dy
    L = -(Fx[:, 1:, :] - Fx[:, :-1, :]) / dx - (Fy[:, :, 1:] - Fy[:, :, :-1]) / dy
    return L


def ssp_rk2_step(
    U: np.ndarray,
    mesh: Mesh,
    dt: float,
    ch: float,
    limiter: str,
    gamma: float,
) -> None:
    """Two-stage SSP-RK2 (Shu-Osher) in-place update of the interior.

    ``U^* = U^n + dt L(U^n)``
    ``U^{n+1} = 0.5 (U^n + U^* + dt L(U^*))``
    """
    g = mesh.nghost
    ix = slice(g, g + mesh.nx)
    iy = slice(g, g + mesh.ny)

    U0 = U.copy()
    L1 = rhs(U, mesh, ch, limiter, gamma)
    U[:, ix, iy] = U0[:, ix, iy] + dt * L1

    L2 = rhs(U, mesh, ch, limiter, gamma)
    U[:, ix, iy] = 0.5 * (U0[:, ix, iy] + U[:, ix, iy] + dt * L2)
