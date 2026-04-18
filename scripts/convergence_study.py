"""Run 32/64/128 cases and compute L1/L2 errors in density against a 256 ref.

Usage
-----
    uv run python -m scripts.convergence_study

Writes:
    data/final/convergence/errors.csv
    data/final/convergence/figs/convergence.pdf
"""

from __future__ import annotations

import csv
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

from vortex.equations import IRHO
from vortex.io import load_snapshot
from vortex.mesh import load_config
from vortex.solver import run as run_solver

CASES = [
    ("data/raw/orszag_tang_32.yaml", "orszag_tang_32"),
    ("data/raw/orszag_tang_64.yaml", "orszag_tang_64"),
    ("data/raw/orszag_tang_128.yaml", "orszag_tang_128"),
]
REF_CFG = ("data/raw/orszag_tang_256.yaml", "orszag_tang_256")

DATA_ROOT = Path("data/final")
OUT_DIR = DATA_ROOT / "convergence" / "figs"
CSV_OUT = DATA_ROOT / "convergence" / "errors.csv"


def _ensure_run(cfg_path: str, run_name: str) -> Path:
    run_dir = DATA_ROOT / run_name
    snap = run_dir / "snap_final.npz"
    if not snap.exists():
        mesh, cfg = load_config(cfg_path)
        run_solver(mesh, cfg, data_root=str(DATA_ROOT))
    return run_dir


def _area_average(field: np.ndarray, factor: int) -> np.ndarray:
    """Coarsen a 2D field by integer factor via block-mean."""
    nx, ny = field.shape
    assert nx % factor == 0 and ny % factor == 0
    return field.reshape(nx // factor, factor, ny // factor, factor).mean(axis=(1, 3))


@click.command(help=__doc__)
def main() -> None:
    ref_dir = _ensure_run(*REF_CFG)
    ref = load_snapshot(ref_dir / "snap_final.npz")
    rho_ref = ref["U"][IRHO]
    nx_ref = rho_ref.shape[0]

    rows = []
    for cfg_path, run_name in CASES:
        run_dir = _ensure_run(cfg_path, run_name)
        snap = load_snapshot(run_dir / "snap_final.npz")
        rho = snap["U"][IRHO]
        nx = rho.shape[0]
        factor = nx_ref // nx
        rho_ref_c = _area_average(rho_ref, factor)
        err = rho - rho_ref_c
        dA = 1.0 / (nx * nx)
        l1 = float(np.sum(np.abs(err)) * dA)
        l2 = float(np.sqrt(np.sum(err * err) * dA))
        rows.append((nx, l1, l2))
        click.echo(f"nx={nx:4d}  L1={l1:.4e}  L2={l2:.4e}")

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["nx", "L1_rho", "L2_rho"])
        writer.writerows(rows)

    nxs = np.asarray([r[0] for r in rows], dtype=float)
    l1s = np.asarray([r[1] for r in rows])
    l2s = np.asarray([r[2] for r in rows])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.loglog(nxs, l1s, marker="o", label="L1")
    ax.loglog(nxs, l2s, marker="s", label="L2")
    # Reference slopes anchored at the coarsest L1 point.
    ref_pt = (nxs[0], l1s[0])
    ax.loglog(nxs, ref_pt[1] * (ref_pt[0] / nxs), "k--", lw=0.8, label="1st order")
    ax.loglog(nxs, ref_pt[1] * (ref_pt[0] / nxs) ** 2, "k:", lw=0.8, label="2nd order")
    ax.set_xlabel("nx")
    ax.set_ylabel(r"error in $\rho$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = OUT_DIR / "convergence.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"wrote {CSV_OUT} and {out_path}")


if __name__ == "__main__":
    main()
