"""GLM divergence-cleaning source update (Dedner et al. 2002).

The hyperbolic part of GLM (``psi`` and ``Bx`` coupled transport) is baked into
the analytic fluxes used by the Riemann solver. What remains here is the
parabolic damping ``d psi/dt = -(ch/cr) psi`` integrated analytically over one
RK stage of length ``dt``.
"""

from __future__ import annotations

import numpy as np

from vortex.equations import IPSI


def damp_psi(U: np.ndarray, dt: float, ch: float, cr: float) -> None:
    """Apply ``psi *= exp(-dt * ch / cr)`` in place.

    ``cr`` is the non-dimensional damping parameter from Dedner et al. (2002);
    a value around 0.18 is recommended.
    """
    if cr <= 0.0 or ch <= 0.0:
        return
    U[IPSI] *= np.exp(-dt * ch / cr)
