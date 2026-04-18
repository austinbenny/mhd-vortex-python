"""Generate density, pressure, magnetic-pressure, and div(B) plots from a
snapshot directory.

Usage
-----
    uv run python -m scripts.plot_orszag_tang data/final/orszag_tang_128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vortex.equations import GAMMA, IBX, IBY, IBZ, IEN, IMX, IMY, IMZ, IRHO
from vortex.io import load_snapshot

OUT_DIR_DEFAULT = Path("docs/source/figs")


def _primitives(U: np.ndarray, gamma: float = GAMMA):
    rho = U[IRHO]
    u = U[IMX] / rho
    v = U[IMY] / rho
    w = U[IMZ] / rho
    bx, by, bz = U[IBX], U[IBY], U[IBZ]
    kin = 0.5 * rho * (u * u + v * v + w * w)
    mag = 0.5 * (bx * bx + by * by + bz * bz)
    p = np.maximum((gamma - 1.0) * (U[IEN] - kin - mag), 1e-12)
    return rho, u, v, p, bx, by, bz


def _plot_contour(field, extent, title, path, levels=30, cmap="viridis"):
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


def make_plots(run_dir: Path, snap_name: str, out_dir: Path, tag_prefix: str = "ot"):
    snap = load_snapshot(run_dir / snap_name)
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

    rho, u, v, p, bx, by, bz = _primitives(U)
    pmag = 0.5 * (bx * bx + by * by + bz * bz)
    divb = _divb_centered(U, dx, dy)

    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_contour(
        rho,
        extent,
        f"density at t={t:.3f}",
        out_dir / f"{tag_prefix}_rho.pdf",
        cmap="viridis",
    )
    _plot_contour(
        p,
        extent,
        f"thermal pressure at t={t:.3f}",
        out_dir / f"{tag_prefix}_p.pdf",
        cmap="magma",
    )
    _plot_contour(
        pmag,
        extent,
        f"magnetic pressure at t={t:.3f}",
        out_dir / f"{tag_prefix}_pmag.pdf",
        cmap="inferno",
    )
    _plot_contour(
        divb,
        extent,
        f"div(B) at t={t:.3f}",
        out_dir / f"{tag_prefix}_divb.pdf",
        cmap="RdBu_r",
    )


def _parse_log(log_path: Path):
    """Return (steps, times, divb_max) from run.log.

    The log writer uses printf-style padding (``step=%6d``) so the ``=`` is
    separated from the value by whitespace for most fields. We use a regex
    pass to be robust to that.
    """
    import re

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


def plot_divb_history(run_dir: Path, out_dir: Path, tag_prefix: str = "ot"):
    steps, times, divb = _parse_log(run_dir / "run.log")
    if len(times) == 0:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.semilogy(times, divb, marker="o", ms=3, lw=1.0)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\max|\nabla\cdot B|$")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag_prefix}_divb_history.pdf", bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Directory containing snap_*.npz and run.log")
    parser.add_argument(
        "--out", default=str(OUT_DIR_DEFAULT), help="Output directory for figures"
    )
    parser.add_argument("--snap", default="snap_final.npz", help="Snapshot filename")
    parser.add_argument("--tag", default="ot", help="Filename prefix for outputs")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out)
    make_plots(run_dir, args.snap, out_dir, args.tag)
    plot_divb_history(run_dir, out_dir, args.tag)
    print(f"wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
