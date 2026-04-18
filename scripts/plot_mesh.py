"""Render a mesh-preview figure: the structured grid plus the initial velocity
and magnetic-field vectors at a coarse sampling.

Usage
-----
    uv run python -m scripts.plot_mesh meshes/orszag_tang_128.yaml \
        --out data/final/mesh_preview/figs/mesh_preview.pdf
"""

from __future__ import annotations

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

from vortex import problems
from vortex.equations import IBX, IBY, IMX, IMY, IRHO
from vortex.mesh import load_config


# Downsample factors so the vector overlay is legible regardless of nx.
ARROW_TARGET_COUNT = 16


def _draw_grid(ax, mesh, max_lines: int = 48) -> None:
    # Cap the number of gridlines drawn so 128+-cell meshes stay readable.
    stride_x = max(1, mesh.nx // max_lines)
    stride_y = max(1, mesh.ny // max_lines)
    xs = np.linspace(mesh.xlo, mesh.xhi, mesh.nx + 1)[::stride_x]
    ys = np.linspace(mesh.ylo, mesh.yhi, mesh.ny + 1)[::stride_y]
    for x in xs:
        ax.axvline(x, color="0.7", lw=0.3)
    for y in ys:
        ax.axhline(y, color="0.7", lw=0.3)


def _quiver_downsample(X, Y, U, V, target=ARROW_TARGET_COUNT) -> tuple[np.ndarray, ...]:
    nx, ny = X.shape
    sx = max(1, nx // target)
    sy = max(1, ny // target)
    return X[::sx, ::sy], Y[::sx, ::sy], U[::sx, ::sy], V[::sx, ::sy]


@click.command(help=__doc__)
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--out",
    type=click.Path(dir_okay=False),
    required=True,
    help="Output PDF path.",
)
def main(config: str, out: str) -> None:
    mesh, cfg = load_config(config)
    ic_fn = problems.get(cfg.problem)
    U = ic_fn(mesh, cfg.gamma)

    # Strip ghosts for plotting.
    g = mesh.nghost
    Ui = U[:, g : g + mesh.nx, g : g + mesh.ny]
    xi, yi = mesh.interior_cell_centers()
    X, Y = np.meshgrid(xi, yi, indexing="ij")

    rho = Ui[IRHO]
    u = Ui[IMX] / rho
    v = Ui[IMY] / rho
    bx = Ui[IBX]
    by = Ui[IBY]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), constrained_layout=True)

    # Left: grid + velocity vectors.
    ax = axes[0]
    _draw_grid(ax, mesh)
    Xs, Ys, us, vs = _quiver_downsample(X, Y, u, v)
    ax.quiver(Xs, Ys, us, vs, color="C0", scale=20, width=0.003)
    ax.set_title(
        f"mesh {mesh.nx}x{mesh.ny}: cells + initial velocity  (problem={cfg.problem})"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(mesh.xlo, mesh.xhi)
    ax.set_ylim(mesh.ylo, mesh.yhi)

    # Right: grid + magnetic field vectors.
    ax = axes[1]
    _draw_grid(ax, mesh)
    Xs, Ys, bxs, bys = _quiver_downsample(X, Y, bx, by)
    ax.quiver(Xs, Ys, bxs, bys, color="C3", scale=10, width=0.003)
    ax.set_title(f"cells + initial magnetic field  ($\\Delta x$ = {mesh.dx:.4f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(mesh.xlo, mesh.xhi)
    ax.set_ylim(mesh.ylo, mesh.yhi)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"wrote {out_path}")


if __name__ == "__main__":
    main()
