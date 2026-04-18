"""Unit tests for ghost-cell fills."""

from __future__ import annotations

import numpy as np

from vortex.boundary import apply as apply_bc
from vortex.mesh import BoundarySpec, Mesh


def test_periodic_matches_donor_cells():
    mesh = Mesh(0.0, 1.0, 0.0, 1.0, 8, 6, bc=BoundarySpec())
    U = mesh.allocate_state()
    # Fill interior with a known pattern.
    g = mesh.nghost
    rng = np.random.default_rng(7)
    U[:, g : g + mesh.nx, g : g + mesh.ny] = rng.standard_normal(
        (U.shape[0], mesh.nx, mesh.ny)
    )
    apply_bc(U, mesh)

    # x-low ghost k should equal interior (nx-g+k) ... i.e. periodic wrap.
    assert np.allclose(U[:, :g, :], U[:, mesh.nx : mesh.nx + g, :])
    assert np.allclose(U[:, mesh.nx + g : mesh.nx + 2 * g, :], U[:, g : 2 * g, :])
    assert np.allclose(U[:, :, :g], U[:, :, mesh.ny : mesh.ny + g])
    assert np.allclose(U[:, :, mesh.ny + g : mesh.ny + 2 * g], U[:, :, g : 2 * g])


def test_outflow_zero_gradient():
    bc = BoundarySpec(
        x_low="outflow", x_high="outflow", y_low="outflow", y_high="outflow"
    )
    mesh = Mesh(0.0, 1.0, 0.0, 1.0, 4, 4, bc=bc)
    U = mesh.allocate_state()
    g = mesh.nghost
    U[:, g : g + mesh.nx, g : g + mesh.ny] = 3.14
    apply_bc(U, mesh)
    # All ghosts should equal the nearest interior value.
    assert np.all(U[:, :g, g : g + mesh.ny] == 3.14)
    assert np.all(U[:, g + mesh.nx :, g : g + mesh.ny] == 3.14)
    assert np.all(U[:, g : g + mesh.nx, :g] == 3.14)
    assert np.all(U[:, g : g + mesh.nx, g + mesh.ny :] == 3.14)
