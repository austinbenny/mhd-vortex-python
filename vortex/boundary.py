"""Ghost-cell fills for the four supported boundary kinds.

The conserved-variable layout is ``[rho, rho u, rho v, rho w, Bx, By, Bz, E,
psi]``. For reflective walls, the normal momentum and normal magnetic field
components flip sign; tangential components and psi are copied.
"""

from __future__ import annotations

import numpy as np

from vortex.mesh import BoundarySpec, Mesh

IDX_RHO_U = 1
IDX_RHO_V = 2
IDX_BX = 4
IDX_BY = 5


def apply(U: np.ndarray, mesh: Mesh) -> None:
    """In-place fill of ghost cells according to ``mesh.bc``."""
    _fill_x(U, mesh, mesh.bc)
    _fill_y(U, mesh, mesh.bc)


def _fill_x(U: np.ndarray, mesh: Mesh, bc: BoundarySpec) -> None:
    g, nx = mesh.nghost, mesh.nx
    if bc.x_low == "periodic" and bc.x_high == "periodic":
        U[:, :g, :] = U[:, nx : nx + g, :]
        U[:, nx + g : nx + 2 * g, :] = U[:, g : 2 * g, :]
        return
    _fill_x_low(U, mesh, bc.x_low)
    _fill_x_high(U, mesh, bc.x_high)


def _fill_y(U: np.ndarray, mesh: Mesh, bc: BoundarySpec) -> None:
    g, ny = mesh.nghost, mesh.ny
    if bc.y_low == "periodic" and bc.y_high == "periodic":
        U[:, :, :g] = U[:, :, ny : ny + g]
        U[:, :, ny + g : ny + 2 * g] = U[:, :, g : 2 * g]
        return
    _fill_y_low(U, mesh, bc.y_low)
    _fill_y_high(U, mesh, bc.y_high)


def _fill_x_low(U: np.ndarray, mesh: Mesh, kind: str) -> None:
    g = mesh.nghost
    if kind == "outflow":
        for k in range(g):
            U[:, k, :] = U[:, g, :]
    elif kind == "reflective":
        for k in range(g):
            U[:, k, :] = U[:, 2 * g - 1 - k, :]
            U[IDX_RHO_U, k, :] *= -1.0
            U[IDX_BX, k, :] *= -1.0
    else:
        raise ValueError(f"unsupported x_low bc: {kind}")


def _fill_x_high(U: np.ndarray, mesh: Mesh, kind: str) -> None:
    g, nx = mesh.nghost, mesh.nx
    if kind == "outflow":
        for k in range(g):
            U[:, nx + g + k, :] = U[:, nx + g - 1, :]
    elif kind == "reflective":
        for k in range(g):
            U[:, nx + g + k, :] = U[:, nx + g - 1 - k, :]
            U[IDX_RHO_U, nx + g + k, :] *= -1.0
            U[IDX_BX, nx + g + k, :] *= -1.0
    else:
        raise ValueError(f"unsupported x_high bc: {kind}")


def _fill_y_low(U: np.ndarray, mesh: Mesh, kind: str) -> None:
    g = mesh.nghost
    if kind == "outflow":
        for k in range(g):
            U[:, :, k] = U[:, :, g]
    elif kind == "reflective":
        for k in range(g):
            U[:, :, k] = U[:, :, 2 * g - 1 - k]
            U[IDX_RHO_V, :, k] *= -1.0
            U[IDX_BY, :, k] *= -1.0
    else:
        raise ValueError(f"unsupported y_low bc: {kind}")


def _fill_y_high(U: np.ndarray, mesh: Mesh, kind: str) -> None:
    g, ny = mesh.nghost, mesh.ny
    if kind == "outflow":
        for k in range(g):
            U[:, :, ny + g + k] = U[:, :, ny + g - 1]
    elif kind == "reflective":
        for k in range(g):
            U[:, :, ny + g + k] = U[:, :, ny + g - 1 - k]
            U[IDX_RHO_V, :, ny + g + k] *= -1.0
            U[IDX_BY, :, ny + g + k] *= -1.0
    else:
        raise ValueError(f"unsupported y_high bc: {kind}")
