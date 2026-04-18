# MHD Reading Guide

A step-by-step reading plan for building up to a 2D ideal MHD finite volume solver.
Each phase builds on the previous one. Within each phase, read in the listed order.

References marked with `[local]` are in the `references/` directory.


## Phase 1: Plasma Physics Foundations

*Goal:* Understand why MHD exists, what assumptions it makes, and what the governing
equations look like before touching any numerics.

### 1.1 Plasma fundamentals

- Chen, *Introduction to Plasma Physics and Controlled Fusion*, Ch. 1
  - What a plasma is: Debye shielding, plasma parameter, criteria for plasma behavior
  - Read for physical intuition, not derivation detail
- Chen, Ch. 2
  - Single-particle motion in E and B fields: gyromotion, ExB drift, grad-B drift
  - This chapter is the single best cure for "I don't have intuition for magnetism".
    Work through the drift derivations to build physical feel for how B pushes
    charged particles
- Chen, Ch. 3
  - Plasmas as fluids: the two-fluid picture and how it reduces to a single-fluid
    MHD-like description
  - This is the chapter that answers "how does the Lorentz force enter the momentum
    equation". Read it carefully

### 1.2 Deriving the ideal MHD equations

- Chen, Ch. 4 (especially Sec. 4.19 on hydromagnetic waves) and Ch. 3 review
  - Bridge from two-fluid to single-fluid MHD
  - Alfven wave derivation from the fluid picture
- Powell et al. (1999), Sec. 2 (pp. 284-288) `[local]`
  - Clean statement of the 8-equation conservative form in 3D
  - Covers: mass, Faraday's law, momentum, energy, and the equation of state
  - This is the compact reference card for the equations your solver actually updates
- Freidberg, *Ideal MHD*, Ch. 2-3 (as reference only)
  - Use as a fallback if Chen's derivation feels too hand-wavy. Freidberg gives a
    more rigorous treatment of the single-fluid limit, but you do not need to read
    it cover-to-cover

### 1.3 MHD wave structure

- Chen, Sec. 4.19 and 4.22 (hydromagnetic and magnetosonic waves)
  - Alfven wave, fast and slow magnetosonic modes derived from the fluid picture
- Brio and Wu (1988), Sec. II (pp. 400-402) `[local]`
  - Compact presentation of the 7x7 eigenvalue structure
  - Proves the MHD equations are nonconvex (compound waves can exist)
  - This is short; read the whole section
- Miyoshi and Kusano (2005), Sec. 2 (pp. 317-318) `[local]`
  - Conservative form, eigenvalues, wave speeds (c_a, c_f, c_s)
  - Discontinuity types: fast/slow shocks, Alfven (rotational), contact, tangential
  - Good summary to read right after Brio-Wu Sec. II


## Phase 2: Finite Volume Method Foundations

*Goal:* Understand Godunov-type FVM on structured grids before adding MHD-specific
complications. If you already know FVM from a CFD course, skim this phase.

### 2.1 FVM fundamentals

- Versteeg and Malalasekera, *An Introduction to CFD: The Finite Volume Method*, Ch. 5, 11
  - Control volume formulation, integral conservation laws, structured grids
  - Numerical flux, conservation form, consistency
- Moukalled et al., *The Finite Volume Method in Computational Fluid Dynamics*, Ch. 4-8
  - Practical implementation: data structures, flux evaluation, boundary conditions
  - Code examples that translate directly to implementation

### 2.2 Riemann solvers and upwind methods

- Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, Ch. 4, 6, 10
  - Ch. 4: The Riemann problem for the Euler equations (builds intuition for MHD)
  - Ch. 6: HLL family of solvers -- the foundation that HLLD extends
  - Ch. 10: Godunov's method and the role of the Riemann solver in FVM

### 2.3 MUSCL reconstruction and slope limiters

- Toro, Ch. 13-14
  - MUSCL-Hancock scheme for second-order accuracy
  - Slope limiters (minmod, MC, van Leer) and why they matter at discontinuities
- Versteeg and Malalasekera, Ch. 5.9
  - Practical limiter descriptions with implementation guidance

### 2.4 Time integration

- Toro, Ch. 6 (CFL condition)
  - CFL condition: dt <= CFL * dx / max_wave_speed
  - For MHD the relevant speed is the fast magnetosonic speed
- SSP-RK2 (Shu-Osher): standard reference, widely documented in Toro and elsewhere
  - Two-stage scheme: U* = U^n + dt*L(U^n), U^{n+1} = 0.5*(U^n + U* + dt*L(U*))


## Phase 3: MHD Riemann Solvers

*Goal:* Understand how the Riemann problem is solved for MHD, culminating in the HLLD
solver you will implement.

### 3.1 Roe-type solver for MHD (historical context)

- Brio and Wu (1988), Sec. III-V (pp. 402-415) `[local]`
  - Roe linearization applied to MHD: averaged states, eigenvalues, eigenvectors
  - Numerical experiments on the coplanar Riemann problem
  - Demonstrates compound waves (shock + attached rarefaction) -- unique to MHD
  - Read Sec. III (Roe construction) carefully; Sec. IV-V for results and context

### 3.2 HLL family leading to HLLD

- Miyoshi and Kusano (2005), Sec. 3-4 (pp. 318-322) `[local]`
  - Sec. 3: HLL solver review -- single intermediate state, two bounding wave speeds
  - Sec. 4: HLLC for Euler equations -- adding a contact wave
  - This sets up exactly why HLLD is needed: MHD has more intermediate states

### 3.3 The HLLD solver

- Miyoshi and Kusano (2005), Sec. 5 (pp. 322-330) `[local]`
  - *This is the core algorithm you will implement.*
  - Sec. 5.1: Construction of the four intermediate states (U*_L, U**_L, U**_R, U*_R)
  - Sec. 5.2: Proof that HLLD exactly resolves isolated discontinuities
  - Sec. 5.3: Positivity preserving property
  - Work through every equation. Implement directly from this section.
- Miyoshi and Kusano (2005), Sec. 6 (pp. 330-340) `[local]`
  - Numerical tests: Brio-Wu, Dai-Woodward, Orszag-Tang
  - Compare your results against these figures
- Athena++ source code: `src/hydro/rsolvers/mhd/hlld.cpp`
  - Reference implementation to cross-check your code


## Phase 4: The Divergence-Free Constraint

*Goal:* Understand why div(B) = 0 is hard to maintain numerically, what goes wrong when
you violate it, and the GLM cleaning approach you will implement.

### 4.1 Why div(B) matters

- Balsara and Spicer (1999), Sec. 1 (pp. 270-272) `[local]`
  - Physical consequences of violating div(B) = 0: wrong field topology, unphysical
    plasma transport orthogonal to B, loss of momentum and energy conservation
  - Survey of approaches: constrained transport, Hodge projection, Powell source terms
- Powell et al. (1999), Sec. 2.8 (pp. 289-292) `[local]`
  - The "eight-wave" formulation: retaining div(B) source terms to symmetrize the system
  - Why this helps but does not fully solve the problem

### 4.2 Overview of divergence control methods

- Dedner et al. (2002), Sec. 1-2 (pp. 645-649) `[local]`
  - Survey of approaches: projection (elliptic), constrained transport, Powell source terms
  - Sec. 2: Constrained formulations via a generalized Lagrange multiplier psi
  - Three choices for the operator D(psi): elliptic, parabolic, hyperbolic
  - The hyperbolic choice is the GLM method -- divergence errors become waves
- Londrillo and Del Zanna (2000), Sec. 1 (pp. 508-509) `[local]`
  - Additional perspective on how different MHD codes handle div(B)
  - Relationship between upwind fluxes and the divergence-free property

### 4.3 GLM divergence cleaning (the method you will implement)

- Dedner et al. (2002), Sec. 3-4 (pp. 649-660) `[local]`
  - Sec. 3: Eigensystem of the GLM-MHD system (9 equations instead of 8)
  - Sec. 4: Coupling GLM with Godunov-type FVM
  - Parameter choices: cleaning speed c_h (typically max fast magnetosonic speed),
    damping ratio c_r (controls exponential decay of psi)
  - Implementation: only the induction equation and the psi equation change
- Dedner et al. (2002), Sec. 5 (pp. 660-670) `[local]`
  - Numerical experiments comparing GLM against Powell source terms and no cleaning
  - Demonstrates that GLM produces smaller divergence errors


## Phase 5: Validation and Benchmarks

*Goal:* Understand the test problems you will use to verify your solver, and what the
correct answers look like.

### 5.1 Brio-Wu shock tube (1D validation)

- Brio and Wu (1988), Sec. V (pp. 410-420) `[local]`
  - Initial conditions, exact solution structure
  - All 7 MHD wave families appear: fast/slow shocks, rarefactions, compound waves, contact
  - Compare: density, pressure, By profiles
  - Run as a quasi-1D problem (2D grid with trivial y-direction)

### 5.2 Orszag-Tang vortex (2D validation, primary benchmark)

- Londrillo and Del Zanna (2000), Sec. 5 (pp. 517-522) `[local]`
  - Reference density and pressure contours at t = 0.5
  - Initial conditions: rho=1, p=1/gamma, sinusoidal velocity and B-field
  - Periodic BCs on [0,1]^2
  - This is the simplest 2D MHD benchmark -- if something is wrong, it is the solver
- Miyoshi and Kusano (2005), Sec. 6 `[local]`
  - HLLD results on Orszag-Tang for direct comparison

### 5.3 MHD rotor problem (2D, stretch goal)

- Balsara and Spicer (1999), Sec. 3 (pp. 280-290) `[local]`
  - Dense spinning disk (rho=10, r<=0.1) in uniform Bx field
  - 10:1 density ratio, outflow BCs
  - Compare: Mach number contours, magnetic pressure contours
  - Use to demonstrate GLM cleaning effectiveness via max(|div(B)|) over time


## Suggested Reading Order (Condensed)

For someone starting from scratch, the minimum viable reading path:

1. Chen Ch. 1-3 (plasma fundamentals, single-particle motion in EM fields, fluid
   description)
2. Chen Sec. 4.19 + Brio-Wu (1988) Sec. II (MHD waves and eigenstructure)
3. Powell et al. (1999) Sec. 2 (conservative form, compact equation reference)
4. Toro Ch. 6, 10 (Godunov FVM, HLL)
5. Toro Ch. 13-14 (MUSCL reconstruction)
6. Miyoshi-Kusano (2005) Sec. 2-5 (HLL -> HLLD derivation, the core algorithm)
7. Dedner et al. (2002) Sec. 2-4 (GLM divergence cleaning)
8. Brio-Wu (1988) Sec. V (first test problem)
9. Londrillo-Del Zanna (2000) Sec. 5 (Orszag-Tang reference)
10. Balsara-Spicer (1999) Sec. 3 (rotor reference, if time permits)

Steps 1-3 give you the physics. Steps 4-5 give you FVM. Steps 6-7 give you the
MHD-specific numerics. Steps 8-10 give you validation targets.
