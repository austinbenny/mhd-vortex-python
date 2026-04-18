"""Snapshot save/load and run-directory logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from vortex.mesh import Mesh


def ensure_run_dir(base: str | Path, run_name: str) -> Path:
    """Create ``<base>/<run_name>/`` and return the path."""
    run_dir = Path(base) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(run_dir: Path, name: str = "vortex") -> logging.Logger:
    """Configure a logger that writes to ``run_dir/run.log`` (truncated) and stdout."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_path = run_dir / "run.log"
    if log_path.exists():
        log_path.unlink()

    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    return logger


def save_snapshot(run_dir: Path, tag: str, t: float, U: np.ndarray, mesh: Mesh) -> Path:
    """Write an .npz snapshot. Ghost cells are stripped before saving."""
    g = mesh.nghost
    Ui = U[:, g : g + mesh.nx, g : g + mesh.ny]
    path = run_dir / f"snap_{tag}.npz"
    np.savez_compressed(
        path,
        U=Ui,
        t=np.asarray(t),
        xlo=np.asarray(mesh.xlo),
        xhi=np.asarray(mesh.xhi),
        ylo=np.asarray(mesh.ylo),
        yhi=np.asarray(mesh.yhi),
        nx=np.asarray(mesh.nx),
        ny=np.asarray(mesh.ny),
    )
    return path


def load_snapshot(path: str | Path) -> dict:
    """Load a snapshot .npz into a plain dict."""
    with np.load(Path(path)) as f:
        return {k: f[k] for k in f.files}
