"""Diagnostic quantities: div(B), conserved-variable totals, kinetic / magnetic
energies.
"""

from __future__ import annotations

import numpy as np

from vortex.equations import IBX, IBY, IEN, IMX, IMY, IMZ, IRHO
from vortex.mesh import Mesh


def div_b(U: np.ndarray, mesh: Mesh) -> np.ndarray:
    """Centered-difference div(B) at interior cell centers.

    Ghost cells are assumed to already be filled, so we can access neighbours
    of every interior cell.
    """
    g = mesh.nghost
    nx, ny = mesh.nx, mesh.ny
    bx_plus = U[IBX, g + 1 : g + nx + 1, g : g + ny]
    bx_minus = U[IBX, g - 1 : g + nx - 1, g : g + ny]
    by_plus = U[IBY, g : g + nx, g + 1 : g + ny + 1]
    by_minus = U[IBY, g : g + nx, g - 1 : g + ny - 1]
    return (bx_plus - bx_minus) / (2.0 * mesh.dx) + (by_plus - by_minus) / (
        2.0 * mesh.dy
    )


def conserved_totals(U: np.ndarray, mesh: Mesh) -> dict[str, float]:
    """Area-weighted sums of the primary conserved quantities on the interior."""
    g = mesh.nghost
    Ui = U[:, g : g + mesh.nx, g : g + mesh.ny]
    dA = mesh.dx * mesh.dy
    return {
        "mass": float(Ui[IRHO].sum() * dA),
        "mom_x": float(Ui[IMX].sum() * dA),
        "mom_y": float(Ui[IMY].sum() * dA),
        "mom_z": float(Ui[IMZ].sum() * dA),
        "energy": float(Ui[IEN].sum() * dA),
    }
