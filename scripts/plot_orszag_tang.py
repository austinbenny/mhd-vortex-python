"""Generate density, pressure, magnetic-pressure, and div(B) plots from a
snapshot directory.

Figures go into ``<run_dir>/figs/`` by default, one per snapshot file found.
The div(B) time history is produced from ``run.log``.

Usage
-----
    uv run python -m scripts.plot_orszag_tang data/final/orszag_tang_128
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

from vortex.equations import GAMMA, IBX, IBY, IBZ, IEN, IMX, IMY, IRHO
from vortex.io import load_snapshot


def _primitives(U: np.ndarray, gamma: float = GAMMA):
    rho = U[IRHO]
    u = U[IMX] / rho
    v = U[IMY] / rho
    bx, by, bz = U[IBX], U[IBY], U[IBZ]
    # Kinetic + magnetic energy densities to back out thermal pressure.
    kin = 0.5 * rho * (u * u + v * v + (U[3] / rho) ** 2)
    mag = 0.5 * (bx * bx + by * by + bz * bz)
    p = np.maximum((gamma - 1.0) * (U[IEN] - kin - mag), 1e-12)
    return rho, p, bx, by, bz


def _plot_contour(field, extent, title, path, cmap="viridis", levels=30):
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(
        field.T,
        origin="lower",
        extent=extent,
        cmap=cmap,
        aspect="equal",
    )
    ax.contour(
        np.linspace(extent[0], extent[1], field.shape[0]),
        np.linspace(extent[2], extent[3], field.shape[1]),
        field.T,
        levels=levels,
        colors="k",
        linewidths=0.35,
        alpha=0.6,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _divb_centered(U: np.ndarray, dx: float, dy: float) -> np.ndarray:
    bx_plus = np.roll(U[IBX], -1, axis=0)
    bx_minus = np.roll(U[IBX], 1, axis=0)
    by_plus = np.roll(U[IBY], -1, axis=1)
    by_minus = np.roll(U[IBY], 1, axis=1)
    return (bx_plus - bx_minus) / (2.0 * dx) + (by_plus - by_minus) / (2.0 * dy)


def _snap_tag(path: Path) -> str:
    """Strip the ``snap_`` prefix and extension to form a file-name suffix."""
    stem = path.stem
    return stem.removeprefix("snap_") or "snap"


def plot_snapshot(snap_path: Path, out_dir: Path) -> None:
    snap = load_snapshot(snap_path)
    U = snap["U"]
    t = float(snap["t"])
    nx, ny = int(snap["nx"]), int(snap["ny"])
    extent = (
        float(snap["xlo"]),
        float(snap["xhi"]),
        float(snap["ylo"]),
        float(snap["yhi"]),
    )
    dx = (extent[1] - extent[0]) / nx
    dy = (extent[3] - extent[2]) / ny

    rho, p, bx, by, bz = _primitives(U)
    pmag = 0.5 * (bx * bx + by * by + bz * bz)
    divb = _divb_centered(U, dx, dy)

    tag = _snap_tag(snap_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_contour(rho, extent, f"density at t={t:.3f}", out_dir / f"rho_{tag}.pdf")
    _plot_contour(
        p, extent, f"thermal pressure at t={t:.3f}", out_dir / f"p_{tag}.pdf", cmap="magma"
    )
    _plot_contour(
        pmag,
        extent,
        f"magnetic pressure at t={t:.3f}",
        out_dir / f"pmag_{tag}.pdf",
        cmap="inferno",
    )
    _plot_contour(
        divb, extent, f"div(B) at t={t:.3f}", out_dir / f"divb_{tag}.pdf", cmap="RdBu_r"
    )


def _parse_log(log_path: Path):
    # The log writer pads ``step=%6d`` so ``=`` is separated by whitespace;
    # a regex is more robust than token splitting.
    pat = re.compile(r"step=\s*(\d+)\s+t=([0-9.e+-]+).*?max\|divB\|=([0-9.eE+-]+)")
    steps, times, divb = [], [], []
    for line in log_path.read_text().splitlines():
        m = pat.search(line)
        if m is None:
            continue
        steps.append(int(m.group(1)))
        times.append(float(m.group(2)))
        divb.append(float(m.group(3)))
    return np.asarray(steps), np.asarray(times), np.asarray(divb)


def plot_divb_history(run_dir: Path, out_dir: Path) -> None:
    _, times, divb = _parse_log(run_dir / "run.log")
    if len(times) == 0:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.semilogy(times, divb, marker="o", ms=3, lw=1.0)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\max|\nabla\cdot B|$")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "divb_history.pdf", bbox_inches="tight")
    plt.close(fig)


@click.command(help=__doc__)
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--out",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory (default: <run_dir>/figs).",
)
@click.option(
    "--snap",
    default=None,
    help="Specific snapshot filename. If omitted, all snap_*.npz are plotted.",
)
def main(run_dir: str, out: str | None, snap: str | None) -> None:
    run_path = Path(run_dir)
    out_dir = Path(out) if out else run_path / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if snap:
        plot_snapshot(run_path / snap, out_dir)
    else:
        for snap_path in sorted(run_path.glob("snap_*.npz")):
            plot_snapshot(snap_path, out_dir)

    plot_divb_history(run_path, out_dir)
    click.echo(f"wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
