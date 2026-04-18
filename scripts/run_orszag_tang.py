"""CLI entry point: run an Orszag-Tang simulation from a YAML config.

Usage
-----
    uv run python -m scripts.run_orszag_tang meshes/orszag_tang_128.yaml
"""

from __future__ import annotations

import click

from vortex.mesh import load_config
from vortex.solver import run


@click.command(help=__doc__)
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--data-root",
    type=click.Path(file_okay=False),
    default="data/final",
    show_default=True,
    help="Directory to write run artifacts under.",
)
def main(config: str, data_root: str) -> None:
    mesh, cfg = load_config(config)
    summary = run(mesh, cfg, data_root=data_root)
    click.echo(f"run directory: {summary['run_dir']}")
    click.echo(f"steps: {summary['steps']}  elapsed: {summary['elapsed_s']:.2f}s")


if __name__ == "__main__":
    main()
