# 2D Ideal MHD Solver Using the Finite Volume Method

## 1. Introduction and Problem Motivation

Plasma flows are governed by the equations of magnetohydrodynamics (MHD), which couple
fluid dynamics with electromagnetism. Unlike ordinary fluid mechanics, the presence of
magnetic fields introduces new wave families (fast/slow magnetosonic, Alfven), the Lorentz
force, and a divergence-free constraint on the magnetic field that has no counterpart in
the Euler or Navier-Stokes equations.

This project builds a 2D finite volume method (FVM) solver for the ideal MHD equations
from scratch. The solver targets canonical MHD test problems and demonstrates
correctness through quantitative comparison with published reference solutions and
established open-source codes.

### Context and applications

Ideal MHD governs solar wind dynamics, tokamak plasma confinement, astrophysical jets,
and magnetospheric physics. The ideal approximation (zero resistivity, viscosity) is
valid when the magnetic Reynolds number is large, which holds in most space and
astrophysical plasmas.

Key references:
> - Chen Ch. 1-3 for plasma fundamentals (Debye length, single-particle motion in EM
>   fields, reduction of two-fluid to single-fluid MHD)
> - Freidberg Ch. 2-3 as a rigorous fallback reference; not a first read

## 2. Mathematical Model

The ideal MHD system in 2D conservative form: 8 coupled hyperbolic conservation laws.

### Conserved variables (state vector U)

- Mass density (rho)
- Momentum (rho\*u, rho\*v, rho\*w) -- all 3 components even in 2D (w is out-of-plane)
- Magnetic field (Bx, By, Bz) -- all 3 components even in 2D (Bz is out-of-plane)
- Total energy E = p/(gamma-1) + 0.5\*rho\*(u^2+v^2+w^2) + 0.5*(Bx^2+By^2+Bz^2)

### Flux structure

dU/dt + dF(U)/dx + dG(U)/dy = 0, where F and G include both hydrodynamic and magnetic
flux contributions. The total pressure is p_total = p + B^2/2, and the flux contains
Maxwell stress terms (-Bi*Bj).

### Closure

Ideal gas equation of state with gamma = 5/3.

### Constraint

div(B) = 0 must hold at all times. This is not an evolution equation but a constraint
inherited from Maxwell's equations (no magnetic monopoles). It is satisfied by the
initial data and preserved analytically, but not automatically by discrete FVM updates.

### Wave structure

The 1D Riemann problem admits 7 waves -- two fast magnetosonic, two Alfven, two slow
magnetosonic, and one contact/entropy wave. The system is not strictly hyperbolic (fast
and Alfven speeds can coincide when B is aligned with the propagation direction).

Key references:
> - Chen Sec. 4.19 and 4.22 for Alfven and magnetosonic waves from the fluid picture
> - Miyoshi & Kusano (2005) Sec. 2 for the conservative form and flux expressions
> - Powell et al. (1999) Sec. 2 for the eight-wave eigenstructure

## 3. Numerical Model

### 3.1 Finite Volume Discretization

Godunov-type FVM on a 2D structured Cartesian grid. Cell-averaged conserved variables
are updated via the integral form of the conservation law. The numerical flux at each
cell interface is computed by solving an approximate Riemann problem.

Key references:
> - Versteeg & Malalasekera Ch. 5, 11 for FVM fundamentals on structured grids
> - Moukalled et al. Ch. 4-8 for practical FVM implementation with code examples

### 3.2 Riemann Solver: HLLD

The HLLD (Harten-Lax-van Leer Discontinuities) approximate Riemann solver resolves the
full MHD wave structure using 4 intermediate states (between the two fast waves). It
exactly resolves isolated discontinuities and reduces to HLLC when B = 0.

Key references:
> - Miyoshi & Kusano (2005) -- complete HLLD algorithm derivation
> - Athena++ source `src/hydro/rsolvers/mhd/hlld.cpp` for a clean implementation

### 3.3 Spatial Reconstruction: MUSCL

Second-order accuracy via piecewise-linear reconstruction of primitive variables at cell
interfaces (MUSCL scheme). Slope limiters (minmod, MC, van Leer) enforce monotonicity
and suppress spurious oscillations near discontinuities.

Key references:
> - Versteeg & Malalasekera Ch. 5.9 for slope limiters
> - Toro (2009) Ch. 13-14 for MUSCL-Hancock in multiple dimensions

### 3.4 Time Integration

Second-order TVD Runge-Kutta (SSP-RK2) or the MUSCL-Hancock predictor-corrector
approach. CFL condition based on the fast magnetosonic speed.

### 3.5 Divergence Cleaning: GLM

The generalized Lagrange multiplier (GLM) method of Dedner et al. augments the MHD
system with a 9th equation for a scalar field psi that couples to div(B). Divergence
errors are transported to domain boundaries at a cleaning speed ch and damped at a rate
controlled by a parameter cr. This modifies only the induction and psi equations -- the
Riemann solver and reconstruction require minimal changes.

Key references:
> - Dedner et al. (2002) -- GLM derivation and recommended parameter choices
> - Dedner et al. Sec. 4 for coupling with Godunov-type FVM

### 3.6 Boundary Conditions

Periodic (Orszag-Tang), outflow/zero-gradient (rotor), and reflective where needed.
Implemented as ghost cell fills before reconstruction.

## 4. Validation

The validation is structured in three stages, each building on the previous one.

### Stage 1: Brio-Wu Shock Tube (1D, run as 2D with trivial y-direction)

This proves the Riemann solver works in isolation before moving to 2D.

- Tests all 7 MHD wave families: fast/slow shocks, rarefactions, compound waves, contact
- Compare density/pressure/By profiles against the exact solution (Brio & Wu 1988)
- Also compare against Athena++ output at matching resolution

> *Reference solution:* Brio, M. & Wu, C.C. "An upwind differencing scheme for the
> equations of ideal MHD." *J. Comput. Phys.* 75, 400-422, 1988.

### Stage 2: Orszag-Tang Vortex (2D) -- primary validation

This proves the full 2D solver works. It is the simplest 2D MHD benchmark: all
sinusoidal initial conditions, uniform density/pressure, periodic BCs on all sides.
No geometry to discretize, no density jumps, no outflow BCs -- if something is wrong,
it is the solver, not the setup.

- Domain: [0,1]^2, periodic on all sides
- IC: rho=1, p=1/gamma, V=(-sin(2*pi*y), sin(2*pi*x), 0), B=(1/gamma)*(-sin(2*pi*y), sin(4*pi*x), 0)
- Compare density and pressure contours at t=0.5 against published results (Londrillo &
  Del Zanna 2000, Athena++ test suite)
- Convergence study: run at 128^2, 256^2, 512^2, compute L1/L2 error norms against a
  fine-grid reference (1024^2), plot on log-log axes to verify second-order convergence

> *Reference solution:* Londrillo, P. & Del Zanna, L. "High-order upwind schemes for
> multidimensional MHD." *ApJ* 530, 508, 2000. Also reproduced in Athena++ test suite.

### Stage 3: MHD Rotor Problem (2D) -- stretch goal

This demonstrates robustness under harsher conditions: 10:1 density ratio, narrow taper
zone, outflow BCs. Only attempted after Orszag-Tang is fully working.

- Domain: [0,1]^2, non-reflecting BCs on all sides
- IC: dense spinning disk (rho=10, r<=0.1) in uniform Bx=5/sqrt(4*pi), gamma=1.4
- Compare Mach number and magnetic pressure contours against Balsara & Spicer (1999)
- Monitor max(|div(B)|) over time to demonstrate GLM cleaning effectiveness

> *Reference solution:* Balsara, D.S. & Spicer, D. "A staggered mesh algorithm using
> high order Godunov fluxes to ensure solenoidal magnetic fields in MHD simulations."
> *J. Comput. Phys.* 149, 270-292, 1999.

## 5. Discussion of Results

Plan to address:

- *Accuracy:* Do the shock positions, wave speeds, and jump conditions match reference
  solutions? Where are the largest errors (compound waves, current sheets)?
- *Convergence:* Does the solver achieve second-order convergence in smooth regions?
  What happens to the convergence rate at shocks (expected to drop to first-order)?
- *Divergence control:* How does max(|div(B)|) behave over time with GLM cleaning?
  What are the effects of the cleaning speed ch and damping parameter cr?
- *Dissipation and resolution:* How do different slope limiters (minmod vs. MC)
  affect the sharpness of contact discontinuities and current sheets?
- *Comparison with established codes:* Overlay plots with Athena++ or PLUTO output.
  Identify and explain any discrepancies.
- *Limitations:* Where does the solver fail or degrade (very low-beta plasma, strong
  grid-aligned B-fields, non-strict hyperbolicity degeneracies)?

## 6. Conclusions

Summarize what was built, what was demonstrated, and what the limitations are:

- Restate the problem and approach (2D ideal MHD, Godunov FVM, HLLD, GLM)
- Key findings from the validation (which benchmarks passed, convergence rates achieved)
- Main limitations (ideal only, Cartesian only, Python performance ceiling)
- Natural extensions: resistive MHD, AMR, constrained transport, 3D

## References

### Plasma Physics

| Reference | Role |
|-----------|------|
| Chen, F.F. *Introduction to Plasma Physics and Controlled Fusion*. Springer, 2016. | Primary physics text: plasma fundamentals, single-particle motion in EM fields, reduction of two-fluid to single-fluid MHD |
| Freidberg, J.P. *Ideal MHD*. Cambridge University Press, 2014. | Rigorous fallback reference for MHD derivation; fusion-focused and too advanced as a first read |
| Goldston, R.J. & Rutherford, P.H. *Introduction to Plasma Physics*. IoP Publishing, 1995. | Advanced plasma physics, transport (optional) |

### Computational Fluid Dynamics

| Reference | Role |
|-----------|------|
| Versteeg, H.K. & Malalasekera, W. *An Introduction to CFD: The Finite Volume Method*. 2nd ed., 2007. | FVM fundamentals |
| Moukalled, F. et al. *The Finite Volume Method in Computational Fluid Dynamics*. | Practical FVM implementation |
| Toro, E.F. *Riemann Solvers and Numerical Methods for Fluid Dynamics*. 3rd ed., 2009. | Riemann solvers, MUSCL, higher-order FVM |
| ASME V&V 20-2021. | Verification and validation methodology |

### MHD Numerics

| Reference | Role |
|-----------|------|
| Miyoshi, T. & Kusano, K. *J. Comput. Phys.* 208, 315-344, 2005. | HLLD Riemann solver |
| Dedner, A. et al. *J. Comput. Phys.* 175, 645-673, 2002. | GLM divergence cleaning |
| Powell, K.G. et al. *J. Comput. Phys.* 154, 284-309, 1999. | Eight-wave formulation |

### Benchmark / Validation Sources

| Reference | Role |
|-----------|------|
| Brio, M. & Wu, C.C. *J. Comput. Phys.* 75, 400-422, 1988. | Brio-Wu shock tube |
| Londrillo, P. & Del Zanna, L. *ApJ* 530, 508, 2000. | Orszag-Tang reference |
| Balsara, D.S. & Spicer, D. *J. Comput. Phys.* 149, 270-292, 1999. | MHD rotor problem |

### Additional Resources

- UT Austin ASE 382Q course outline (`ase382q.10-outline.pdf`)
- Farside plasma physics notes: http://farside.ph.utexas.edu/teaching/plasma/Plasma/index.html

### Open-Source Codes (for cross-reference and validation)

| Code | Notes |
|------|-------|
| [Athena++](https://github.com/PrincetonUniversity/athena) | Astrophysical MHD, Godunov FVM, HLLD, constrained transport |
| [PLUTO](http://plutocode.ph.unito.it/) | Astrophysical gas dynamics and MHD, modular Riemann solvers |
