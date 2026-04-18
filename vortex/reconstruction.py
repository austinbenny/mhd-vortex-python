"""Piecewise-linear MUSCL reconstruction of primitive variables.

For each interior cell we compute a limited slope from left/right differences
and evaluate the primitive state at the two adjacent interfaces. Reconstruction
runs on primitives rather than conserved variables so that positive-definite
quantities (rho, p) stay positive when the limiter pins slopes to zero near
extrema.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Limiter = Literal["minmod", "mc", "vanleer"]


def _limited_slope(dL: np.ndarray, dR: np.ndarray, kind: Limiter) -> np.ndarray:
    if kind == "minmod":
        return np.where(
            dL * dR <= 0.0,
            0.0,
            np.where(np.abs(dL) < np.abs(dR), dL, dR),
        )
    if kind == "mc":
        same = dL * dR > 0.0
        dC = 0.5 * (dL + dR)
        mag = np.minimum(np.minimum(2.0 * np.abs(dL), 2.0 * np.abs(dR)), np.abs(dC))
        return np.where(same, np.sign(dC) * mag, 0.0)
    if kind == "vanleer":
        denom = dL + dR
        safe = np.where(np.abs(denom) > 1e-30, denom, 1e-30)
        return np.where(dL * dR > 0.0, 2.0 * dL * dR / safe, 0.0)
    raise ValueError(f"unknown limiter: {kind}")


def reconstruct_x(
    Q: np.ndarray, limiter: Limiter = "minmod"
) -> tuple[np.ndarray, np.ndarray]:
    """Return (QL, QR) primitive states on the x-faces of the interior cells.

    ``Q`` has shape ``(nvar, nx+2g, ny+2g)`` with ``g`` ghost layers. The
    returned ``QL`` and ``QR`` are the left- and right-biased primitive states
    at every *interior* x-face, shape ``(nvar, nx+1, ny_interior)``.
    """
    dL = Q[:, 1:-1, :] - Q[:, :-2, :]
    dR = Q[:, 2:, :] - Q[:, 1:-1, :]
    slope = _limited_slope(dL, dR, limiter)

    # Primitive states evaluated at x-faces of each cell (left and right face).
    qL_face = Q[:, 1:-1, :] - 0.5 * slope
    qR_face = Q[:, 1:-1, :] + 0.5 * slope

    # Interior x-faces span nx+1 faces; each sees the +face of the left cell
    # and the -face of the right cell.
    QL = qR_face[:, :-1, :]
    QR = qL_face[:, 1:, :]
    # Trim y-ghost rows to interior.
    QL = QL[:, :, 2:-2]
    QR = QR[:, :, 2:-2]
    return QL, QR


def reconstruct_y(
    Q: np.ndarray, limiter: Limiter = "minmod"
) -> tuple[np.ndarray, np.ndarray]:
    """Return (QL, QR) primitive states on the y-faces of the interior cells."""
    dL = Q[:, :, 1:-1] - Q[:, :, :-2]
    dR = Q[:, :, 2:] - Q[:, :, 1:-1]
    slope = _limited_slope(dL, dR, limiter)

    qL_face = Q[:, :, 1:-1] - 0.5 * slope
    qR_face = Q[:, :, 1:-1] + 0.5 * slope

    QL = qR_face[:, :, :-1]
    QR = qL_face[:, :, 1:]
    QL = QL[:, 2:-2, :]
    QR = QR[:, 2:-2, :]
    return QL, QR
