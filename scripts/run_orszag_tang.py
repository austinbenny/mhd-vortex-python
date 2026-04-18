"""CLI entry point: run an Orszag-Tang simulation from a TOML config.

Usage
-----
    uv run python -m scripts.run_orszag_tang meshes/orszag_tang_128.toml
"""

from __future__ import annotations

import argparse
import sys

from vortex.mesh import load_config
from vortex.solver import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a mesh/run TOML file")
    parser.add_argument(
        "--data-root",
        default="data/final",
        help="Directory to write run artifacts under (default: data/final)",
    )
    args = parser.parse_args(argv)

    mesh, cfg = load_config(args.config)
    summary = run(mesh, cfg, data_root=args.data_root)
    print(f"run directory: {summary['run_dir']}")
    print(f"steps: {summary['steps']}  elapsed: {summary['elapsed_s']:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
