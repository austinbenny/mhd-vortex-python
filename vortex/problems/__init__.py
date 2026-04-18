"""Registry of initial-condition problems."""

from __future__ import annotations

from typing import Callable

import numpy as np

from vortex.mesh import Mesh
from vortex.problems.orszag_tang import initial_conditions as _ot_ic

ProblemFn = Callable[[Mesh, float], np.ndarray]

_PROBLEMS: dict[str, ProblemFn] = {
    "orszag_tang": _ot_ic,
}


def get(name: str) -> ProblemFn:
    if name not in _PROBLEMS:
        raise KeyError(f"unknown problem: {name!r}. known: {sorted(_PROBLEMS)}")
    return _PROBLEMS[name]
