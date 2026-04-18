# 2D Ideal MHD Solver

A from-scratch finite-volume solver for the 2D ideal magnetohydrodynamics
equations, written in vectorized NumPy, validated on the Orszag-Tang vortex.

The scheme is Godunov-type FVM on a uniform Cartesian grid, with MUSCL
piecewise-linear reconstruction on primitive variables, an HLLD approximate
Riemann solver following Miyoshi & Kusano (2005), hyperbolic GLM divergence
cleaning following Dedner et al. (2002), and SSP-RK2 time integration. A
$128^2$ Orszag-Tang run to $t=0.5$ finishes in about seven seconds on a
laptop and reproduces the canonical density/pressure topology reported in
Londrillo & Del Zanna (2000) and the Athena test suite.

## What to look at

- [`presentation/slides.pdf`](presentation/slides.pdf) — the project talk.
  Highlights the method, the Orszag-Tang results at $t = 0.1, 0.3, 0.5$, the
  convergence study, and divergence-error control. Open this first if you
  want the short version.
- [`docs/source/index.tex`](docs/source/index.tex) — the long-form write-up.
  Build it with `make docs`, which drops the PDF at `docs/build/index.pdf`.
  Covers the math, the scheme, validation, convergence, and limitations.
- [`data/raw/orszag_tang_128.yaml`](data/raw/orszag_tang_128.yaml) — the
  YAML config the production run uses.
- `data/final/orszag_tang_128/` — after `make solve`, this holds snapshots
  (`snap_*.npz`), the run log, and the post-processed figures.

## Reproducing the results

```
uv sync
make all
```

Targets you can also run individually:

| target | does |
|---|---|
| `make mesh` | render a mesh-preview figure with the initial velocity and B fields |
| `make solve` | run the solver on `MESH=data/raw/orszag_tang_128.yaml` |
| `make post` | post-process snapshots into density/pressure/divB figures |
| `make convergence` | run 32/64/128 against a 256 reference, compute $L^1/L^2$ errors |
| `make docs` | build the LaTeX report |
| `make slides` | build the Beamer presentation |
| `make test` | run the unit test suite |

Pass `MESH=data/raw/orszag_tang_64.yaml RUN_NAME=orszag_tang_64` to any of
the run-specific targets to use a different resolution.

## Repo layout

| path | contents |
|---|---|
| [`vortex/`](vortex/) | solver package: [`mesh`](vortex/mesh.py), [`equations`](vortex/equations.py), [`reconstruction`](vortex/reconstruction.py), [`riemann`](vortex/riemann.py), [`glm`](vortex/glm.py), [`integrator`](vortex/integrator.py), [`solver`](vortex/solver.py) |
| [`scripts/`](scripts/) | click CLIs: [`run_orszag_tang`](scripts/run_orszag_tang.py), [`plot_orszag_tang`](scripts/plot_orszag_tang.py), [`convergence_study`](scripts/convergence_study.py), [`plot_mesh`](scripts/plot_mesh.py) |
| [`data/raw/`](data/raw/) | YAML mesh + problem configs |
| `data/final/` | run outputs (snapshots, log, figures); gitignored |
| [`tests/`](tests/) | pytest suite: [`equations`](tests/test_equations.py), [`riemann`](tests/test_riemann.py), [`boundary`](tests/test_boundary.py) |
| [`docs/`](docs/source/) | LaTeX report source and build dir |
| [`presentation/`](presentation/) | Beamer slides |
| [`references/`](references/) | source PDFs of the primary references |

## Key references

- Miyoshi, T. and Kusano, K. "A multi-state HLL approximate Riemann solver for ideal magnetohydrodynamics." *J. Comput. Phys.* 208 (2005). [[local pdf]](references/miyoshi-kusano-2005-hlld-riemann-solver-ideal-mhd.pdf)
- Dedner, A. et al. "Hyperbolic divergence cleaning for the MHD equations." *J. Comput. Phys.* 175 (2002). [[local pdf]](references/dedner-et-al-2002-hyperbolic-divergence-cleaning-mhd.pdf)
- Londrillo, P. and Del Zanna, L. "High-order upwind schemes for multidimensional MHD." *ApJ* 530 (2000). [[local pdf]](references/londrillo-del-zanna-2000-high-order-upwind-multidimensional-mhd.pdf)
- Toro, E. F. *Riemann Solvers and Numerical Methods for Fluid Dynamics*. Springer, 2009.
- [Athena Orszag-Tang test](https://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/)
