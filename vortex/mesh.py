"""Structured 2D Cartesian mesh loaded from TOML.

The mesh stores cell-centered coordinates and a boundary specification. State
arrays are allocated with ``NGHOST`` ghost layers on each side of each
direction; ghost fills are performed by ``vortex.boundary``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

NGHOST: int = 2
NVAR: int = 9

BCKind = Literal["periodic", "outflow", "reflective"]


@dataclass(frozen=True)
class BoundarySpec:
    x_low: BCKind = "periodic"
    x_high: BCKind = "periodic"
    y_low: BCKind = "periodic"
    y_high: BCKind = "periodic"


@dataclass(frozen=True)
class RunConfig:
    problem: str
    tfinal: float
    cfl: float
    limiter: str
    glm_cr: float
    snapshot_times: tuple[float, ...]
    gamma: float
    log_every: int
    run_name: str


@dataclass(frozen=True)
class Mesh:
    xlo: float
    xhi: float
    ylo: float
    yhi: float
    nx: int
    ny: int
    nghost: int = NGHOST
    bc: BoundarySpec = field(default_factory=BoundarySpec)

    @property
    def dx(self) -> float:
        return (self.xhi - self.xlo) / self.nx

    @property
    def dy(self) -> float:
        return (self.yhi - self.ylo) / self.ny

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nx + 2 * self.nghost, self.ny + 2 * self.nghost)

    @property
    def interior(self) -> tuple[slice, slice]:
        g = self.nghost
        return (slice(g, g + self.nx), slice(g, g + self.ny))

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        g = self.nghost
        i = np.arange(-g, self.nx + g)
        j = np.arange(-g, self.ny + g)
        xc = self.xlo + (i + 0.5) * self.dx
        yc = self.ylo + (j + 0.5) * self.dy
        return xc, yc

    def interior_cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        xc, yc = self.cell_centers()
        g = self.nghost
        return xc[g : g + self.nx], yc[g : g + self.ny]

    def allocate_state(self, nvar: int = NVAR) -> np.ndarray:
        return np.zeros((nvar, *self.shape), dtype=np.float64)


def load_config(path: str | Path) -> tuple[Mesh, RunConfig]:
    """Load a mesh + run configuration from a TOML file."""
    path = Path(path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    mesh_cfg = data["mesh"]
    bc_cfg = data.get("boundary", {})
    bc = BoundarySpec(
        x_low=bc_cfg.get("x_low", "periodic"),
        x_high=bc_cfg.get("x_high", "periodic"),
        y_low=bc_cfg.get("y_low", "periodic"),
        y_high=bc_cfg.get("y_high", "periodic"),
    )
    mesh = Mesh(
        xlo=float(mesh_cfg["xlo"]),
        xhi=float(mesh_cfg["xhi"]),
        ylo=float(mesh_cfg["ylo"]),
        yhi=float(mesh_cfg["yhi"]),
        nx=int(mesh_cfg["nx"]),
        ny=int(mesh_cfg["ny"]),
        bc=bc,
    )

    run_cfg = data["run"]
    glm_cfg = data.get("glm", {})
    cfg = RunConfig(
        problem=str(run_cfg["problem"]),
        tfinal=float(run_cfg["tfinal"]),
        cfl=float(run_cfg.get("cfl", 0.4)),
        limiter=str(run_cfg.get("limiter", "minmod")),
        glm_cr=float(glm_cfg.get("cr", 0.18)),
        snapshot_times=tuple(run_cfg.get("snapshot_times", [])),
        gamma=float(run_cfg.get("gamma", 5.0 / 3.0)),
        log_every=int(run_cfg.get("log_every", 25)),
        run_name=str(run_cfg.get("run_name", path.stem)),
    )
    return mesh, cfg
