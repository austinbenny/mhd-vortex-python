# Prep Notes: 2D Ideal MHD Solver on the Orszag-Tang Vortex

These notes are a deeper companion to the report at
`docs/source/index.tex` (rendered to `docs/build/index.pdf`) and the
slides at `presentation/slides.pdf`. The report is the polished
write-up; this document is the long-form study guide. It walks through
the physics, the equations, the numerical scheme, the code, and the
validation results in enough detail that a reader can rebuild the
mental model from scratch and answer "why does this line of code look
the way it does".

Symbols are written in keyboard-friendly form (`rho`, `gamma`, `B_x`,
`psi`, `c_h`, `c_f`, `S_M`) rather than Greek glyphs. Cross-references
to the implementation use the form `vortex/foo.py:NN`.


## 1. Orientation

### 1.1 What was built

A from-scratch finite-volume solver for the 2D ideal magnetohydrodynamic
(MHD) equations, written entirely in vectorized NumPy. The solver is:

- Godunov-type, cell-centered, on a uniform Cartesian grid.
- Second-order in space via piecewise-linear MUSCL reconstruction of
  *primitive* variables with a TVD slope limiter (minmod default).
- Second-order in time via Shu-Osher SSP-RK2.
- The interface flux is the HLLD approximate Riemann solver of Miyoshi
  and Kusano (2005).
- The divergence constraint `div(B) = 0` is handled by the hyperbolic
  Generalized Lagrange Multiplier (GLM) cleaning of Dedner et al.
  (2002).

The code is roughly 1000 lines of Python in `vortex/`. The
production benchmark is the 2D Orszag-Tang vortex on a 128 x 128 grid,
run to t = 0.5; this finishes in about seven seconds on a laptop.

### 1.2 Why these methods

- HLLD is the right Riemann solver for ideal MHD because it captures
  the four intermediate states between the two outer fast waves --- the
  two rotational Alfven discontinuities and the contact / entropy wave
  in between --- without solving the full 7x7 eigensystem. It exactly
  resolves an isolated discontinuity, falls back to HLLC when the
  magnetic field vanishes, and remains positivity-preserving.
- MUSCL on primitives is chosen so that the limiter, when it pins a
  slope to zero at a local extremum, leaves the cell mean of `rho` and
  `p` untouched and therefore positive. Limiting *conserved* variables
  there can drive a positive primitive negative.
- GLM cleaning is the simplest divergence-control method that does not
  require a staggered grid. It adds one scalar field `psi`, modifies
  only the induction equation and the `psi` equation, and otherwise
  leaves the existing eight-equation MHD machinery untouched.
- SSP-RK2 is the standard pairing: second-order, total-variation-
  diminishing, two stages, the same CFL constant as forward Euler.

### 1.3 Where to look

| If you want                                | Open                                                   |
|--------------------------------------------|--------------------------------------------------------|
| The polished write-up                      | `docs/build/index.pdf` (source `docs/source/index.tex`)|
| The talk-length summary                    | `presentation/slides.pdf`                              |
| The deep-dive (this file)                  | `references/prep-notes.md`                             |
| The proposal-style scoping doc             | `references/2D-IDEAL-MHD-FVM-SOLVER.md`                |
| The reading list                           | `references/MHD-READING-GUIDE.md`                      |
| The actual solver                          | `vortex/`                                              |
| Test-problem initial conditions            | `vortex/problems/orszag_tang.py`                       |
| Production config (128^2)                  | `data/raw/orszag_tang_128.yaml`                        |


## 2. Plasma Physics Setup

The point of this section is to motivate every term in the equations
the solver actually evolves. If the equations already look natural,
skim to Section 3.

### 2.1 What is a plasma

A plasma is a gas of charged particles (typically electrons and ions)
that, on the scales of interest, behaves as a single quasi-neutral
conducting fluid. Two length and time scales matter:

- The Debye length `lambda_D = sqrt(eps_0 k_B T / (n e^2))`. Charge
  imbalances are screened on this scale; outside a Debye sphere the
  plasma looks neutral.
- The plasma frequency `omega_p = sqrt(n e^2 / (eps_0 m_e))`. This is
  the characteristic oscillation rate of electrons relative to the
  much heavier ions.

For MHD to apply we want length scales much larger than `lambda_D` and
time scales much longer than `1 / omega_p`, so that the plasma can be
treated as a continuum, *and* the magnetic Reynolds number
`Rm = U L / eta` should be large so that resistive diffusion of the
field is negligible. In the solar wind, tokamak interiors, and
astrophysical jets all of these are easily satisfied.

### 2.2 From two-fluid to single-fluid MHD

The most fundamental fluid description carries two species (electrons,
ions) with their own continuity, momentum, and energy equations,
coupled by the Lorentz force `q (E + v x B)`. Adding the two momentum
equations weighted by mass and using quasi-neutrality eliminates the
electric field at leading order and gives a single bulk velocity
`v = (m_i n_i v_i + m_e n_e v_e) / (m_i n_i + m_e n_e)`. The remaining
"current density" `J = sum q_s n_s v_s` couples to the field through
Ampere's law, and the Lorentz force on the bulk fluid becomes
`J x B`. The Hall, electron-pressure, and finite-electron-mass terms
in Ohm's law are all dropped: we keep only the ideal Ohm's law

```
E = -v x B.
```

That single line is what makes the model "ideal". It says the field is
frozen into the flow: a fluid element drags its threading field lines
along with it, with no slip.

The result is a closed eight-equation set in three spatial dimensions:
mass continuity, three momentum equations (with the magnetic Lorentz
contribution baked into a Maxwell stress tensor), Faraday's law for
the three components of `B`, and total energy. The closure is an
ideal-gas equation of state with `gamma = 5/3`.

### 2.3 Forces and pressures

The Lorentz force per unit volume `J x B` can be rearranged using
Ampere's law as

```
J x B = (1 / mu_0) (B . grad) B  -  grad(|B|^2 / (2 mu_0)).
```

The first piece is *magnetic tension* along the field lines (a curved
field line wants to straighten, just like a stretched rubber band).
The second piece is *magnetic pressure*: a gradient of `|B|^2 / 2`
acting like an isotropic pressure. We use Heaviside-Lorentz units in
the code (`mu_0 = 1`), so the Lorentz force inside the conservation
flux looks like

```
divergence of   p_T * I  -  B otimes B,    p_T = p + |B|^2 / 2.
```

`p_T` is the *total pressure*: the gas pressure plus the magnetic
pressure. This is what shows up in every flux expression in
`vortex/equations.py:flux_x`/`flux_y`.

### 2.4 Why all three components in 2D

We are solving a problem with no z-dependence: every field is a
function of `(x, y, t)`. But Alfven waves carry the in-plane momentum
of `B_z` and `v_z`, and the rotational discontinuities at the heart of
HLLD couple the in-plane components to the out-of-plane components
through `B_x` (the normal component at an x-face). Drop the
out-of-plane components and you lose the wave structure HLLD is built
to resolve. The state vector therefore carries `(rho, rho u, rho v,
rho w, B_x, B_y, B_z, E)`, with `w` and `B_z` evolving as in the 3D
case, just with no z-derivatives.

### 2.5 Plasma beta

The plasma beta is the ratio of gas pressure to magnetic pressure:

```
beta = 2 p / |B|^2.
```

`beta >> 1` is "high-beta": the gas dominates and MHD looks almost
like ordinary compressible flow. `beta << 1` is "low-beta": the
magnetic field dominates and small numerical errors in `B` translate
into large errors in the dynamics. Orszag-Tang has `beta_0 = 2 gamma`
at t = 0, so it is moderate-beta everywhere; once shocks form, local
beta can drop to O(1) along the current sheet, and that is where the
solver is exercised hardest. Keeping `rho` and `p` positive in low-beta
cells is what motivates the slope limiter, the GLM cleaning, and the
choice to reconstruct primitives.


## 3. The Governing Equations

### 3.1 The conservative form

In two dimensions:

```
dU/dt + dF(U)/dx + dG(U)/dy = 0,
```

with the eight-component state vector

```
U = (rho, rho u, rho v, rho w, B_x, B_y, B_z, E)^T.
```

GLM augments this with a ninth scalar `psi`; in the code the layout is

```
U = (rho, rho u, rho v, rho w, B_x, B_y, B_z, E, psi)^T,
```

with the index map at `vortex/equations.py:28` (`IRHO`, `IMX`, `IMY`,
`IMZ`, `IBX`, `IBY`, `IBZ`, `IEN`, `IPSI`).

The total energy density is

```
E = p / (gamma - 1)  +  0.5 rho (u^2 + v^2 + w^2)  +  0.5 (B_x^2 + B_y^2 + B_z^2).
```

The closure is the ideal-gas equation of state with `gamma = 5/3`,
which inverts to

```
p = (gamma - 1) (E - 0.5 rho |v|^2 - 0.5 |B|^2).
```

`prim_to_cons` and `cons_to_prim` at `vortex/equations.py:32-74`
implement these two relations directly.

### 3.2 The fluxes

The x-direction flux (without GLM) is

```
F(U) = ( rho u,
         rho u^2 + p_T - B_x^2,
         rho u v - B_x B_y,
         rho u w - B_x B_z,
         0,
         u B_y - v B_x,
         u B_z - w B_x,
         (E + p_T) u  -  B_x (u B_x + v B_y + w B_z) )^T.
```

The fifth row is zero because the divergence theorem applied to
Faraday's law for `B_x` only sees x-derivatives of `B_x`, and `B_x`
itself is what we are differentiating. With GLM coupling the fifth
row becomes `psi` and the new ninth row becomes `c_h^2 B_x`; both
modifications appear at `vortex/equations.py:129,133`. The y-direction
flux `G(U)` follows by swapping `(u, B_x) <-> (v, B_y)` everywhere.

### 3.3 The seven-wave structure

Linearizing the system about a constant state and looking for plane
waves `exp(i (k x - omega t))` gives, in 1D, seven wave families:

- Two *fast magnetosonic* waves at `+/- c_f`. Compressive; carry
  density, pressure, and tangential field perturbations all in phase.
  Always the outermost waves of the fan.
- Two *Alfven* waves at `+/- c_a`, where `c_a = |B_n| / sqrt(rho)` and
  `B_n` is the field component along the wave normal. Incompressible
  and rotational: they tilt the tangential `B` and `v` components
  without changing density or pressure.
- Two *slow magnetosonic* waves at `+/- c_s`. Compressive; carry the
  remaining mode.
- One *entropy* / contact wave at the bulk velocity `u`. Carries a
  density jump at constant pressure and constant `B`.

The fast and slow speeds are

```
c_{f,s}^2 = 0.5 ( a^2 + b^2 / rho  +/-  sqrt((a^2 + b^2 / rho)^2 - 4 a^2 b_n^2 / rho) ),
```

with the sound speed `a^2 = gamma p / rho` and `b^2 = |B|^2`. The
implementation at `vortex/equations.py:84-106` uses exactly this
expression with a `max(., 0)` guard on the discriminant for cells
where round-off would give a tiny negative argument under the square
root.

The system is hyperbolic (real eigenvalues, complete set of
eigenvectors) but *not strictly* hyperbolic: when `B_n = 0` the Alfven
and slow speeds coincide, and when `b^2 / rho = 2 a^2` the Alfven and
fast speeds coincide. These degeneracies are exactly where naive
schemes break down. HLLD handles them by having tangential
field/velocity formulas that pass through the degenerate cases without
dividing by zero --- the explicit guard in `vortex/riemann.py:139-149`
is what catches `B_n^2` becoming negligible compared to
`rho (S - u)^2`.

### 3.4 The divergence constraint

Faraday's law in conservative form is `dB/dt + curl(E) = 0`. Taking
the divergence and using the ideal Ohm's law `E = -v x B` gives
`d(div B)/dt = 0`. So `div B = 0` is preserved exactly by the
*continuous* equations: if it holds at t = 0, it holds for all time.

Discretely it does not. A piecewise-constant Godunov update treats the
six magnetic flux contributions independently, and a centered finite-
difference of the resulting cell-centered `B` field generally produces
a nonzero divergence. Letting that drift go unbounded leads to:

- A spurious Lorentz force component along `B`, which has no physical
  origin and breaks momentum conservation.
- Energy errors at the level of `(div B)^2 / rho`.
- Eventual blow-up in low-beta cells.

The major remedies are:

- *Constrained transport* (Evans-Hawley, Balsara-Spicer, Londrillo-Del
  Zanna): store `B` on staggered cell *faces*, evolve via the curl of
  a corner-centered electric field, and the discrete divergence is
  exactly zero by construction. Best in principle; requires staggered
  storage and careful interpolation back to cell centers for the
  Riemann solver.
- *Powell source terms*: keep the eight-wave formulation symmetrized,
  retain a `div B` source on the right-hand side, and rely on
  upwinding to limit error. Cheap; not strictly conservative.
- *Hyperbolic GLM*: introduce a ninth scalar `psi` whose evolution
  couples to the *normal* magnetic field as a damped wave equation,
  so that local divergence errors propagate away at finite speed
  `c_h` and decay exponentially with rate `c_h / c_r`. Cheap; preserves
  the unstaggered cell-centered layout.

This solver uses GLM. Section 4.4 below details the implementation.


## 4. Numerical Scheme: a Guided Tour of the Code

### 4.1 The mesh and ghost cells

The mesh is a uniform Cartesian grid `[xlo, xhi] x [ylo, yhi]` with
`nx` cells in `x` and `ny` in `y`. The dataclass `Mesh` at
`vortex/mesh.py:44-86` carries the geometry and the boundary
specification; cell centers are at `xlo + (i + 0.5) dx` for
`i = 0, ..., nx-1`.

Two *ghost* layers are allocated on every side of every direction. The
shape of the state array is therefore `(NVAR, nx + 4, ny + 4)`, and
the interior slice `g : g + nx, g : g + ny` (with `g = 2` from
`NGHOST` at `vortex/mesh.py:17`) is what physically corresponds to the
domain. Two layers, not one, because the MUSCL stencil uses
`{U_{i-1}, U_i, U_{i+1}}` to compute the slope in cell `i`, and the
flux at the right face of cell `i` reads `U_{i+1}` --- so the
right-most interior cell `i = nx - 1` needs `U_{nx}` and `U_{nx+1}`
populated, and the slope at that cell needs `U_{nx-2}, U_{nx-1},
U_{nx}` so we need both layers.

The ghost fills are dispatched in `vortex/boundary.py:apply` and the
periodic case is the obvious wrap

```
U[:, :g, :]            = U[:, nx : nx+g, :]
U[:, nx+g : nx+2g, :]  = U[:, g : 2g,   :]
```

at `vortex/boundary.py:28-31`. Outflow copies the nearest interior
column / row into every ghost layer. Reflective mirrors the interior
across the wall and *flips the sign* of the normal momentum and normal
magnetic-field component (the tangential components and `psi` are
copied as-is; see `vortex/boundary.py:51-55` and `:65-69`).

### 4.2 MUSCL reconstruction

Given the cell averages `Q_{i-1}, Q_i, Q_{i+1}` of a primitive variable
`Q`, the slope in cell `i` is computed as

```
delta_L = Q_i - Q_{i-1},
delta_R = Q_{i+1} - Q_i,
slope   = limited(delta_L, delta_R),
```

and the left- and right-biased face values follow from a Taylor
expansion to first order:

```
Q_L^{face i+1/2} = Q_i + 0.5 * slope_i,
Q_R^{face i-1/2} = Q_i - 0.5 * slope_i.
```

(`Q_L`/`Q_R` here mean "what cell `i` provides to the face on its
left / right". The Riemann solver consumes the pair
`(Q_R^{face from cell i+1/2 left side}, Q_L^{face from cell i-1/2 right side})`
at every face.)

Three TVD limiters are implemented at
`vortex/reconstruction.py:19-35`:

- *minmod*: `0` if the two slopes have opposite signs, otherwise the
  one with smaller absolute value. The most diffusive choice; the
  default in this solver because it never overshoots.
- *MC* (monotonized central): `sign(d_C) * min(2|d_L|, 2|d_R|, |d_C|)`
  with `d_C = 0.5 (d_L + d_R)`. Sharper than minmod; fine for smooth
  problems but can excite oscillations near low-beta cells with strong
  `B` gradients.
- *van Leer*: `2 d_L d_R / (d_L + d_R)` when the signs agree, else
  `0`. Smooth; intermediate steepness.

Reconstruction is on *primitive* variables (`rho`, `u`, `v`, `w`,
`B_x`, `B_y`, `B_z`, `p`, `psi`), not on conserved variables. This
matters for two reasons:

- When the limiter pins a slope to zero at an extremum, the cell
  average is unchanged. If that cell average is `rho > 0` and `p > 0`,
  the reconstructed face values are also positive. Reconstructing
  conserved variables and then converting can flip primitive sign at
  shocks because the conversion is nonlinear in `E`.
- The Riemann solver needs primitives anyway, so we save one
  conserved-to-primitive conversion per face per RK stage.

The reconstruction routines `reconstruct_x` /  `reconstruct_y` at
`vortex/reconstruction.py:38-80` return arrays of shape
`(NVAR, nx + 1, ny)` and `(NVAR, nx, ny + 1)` respectively --- one
left-state and one right-state per *face*, with the y-ghosts trimmed
back to the interior because we only need x-faces over the interior y
range.

### 4.3 The HLLD Riemann solver

This is the heart of the scheme. The implementation is at
`vortex/riemann.py:40-241`. It follows Miyoshi and Kusano (2005)
Section 5 equations 38-67. We walk through it in the order the code
executes.

#### 4.3.1 Setup

Take primitive states `Q_L, Q_R` on either side of an x-face. The
ratio of specific heats is `gamma`, and `c_h` is the GLM cleaning
speed (see Section 4.4).

#### 4.3.2 GLM correction to `B_n` and `psi`

Before doing anything else, resolve the `(B_x, psi)` two-by-two
Riemann problem analytically. The GLM `(B_x, psi)` subsystem is

```
dB_x/dt + dpsi/dx     = 0,
dpsi/dt + c_h^2 dB_x/dx = 0,
```

a linear damped-wave system with eigenvalues `+/- c_h` and
eigenvectors `(1, +/- c_h)`. Its exact Riemann solution at the face
is

```
B_x^*  = 0.5 (B_{x,L} + B_{x,R})  -  0.5 (psi_R - psi_L) / c_h,
psi^*  = 0.5 (psi_L  + psi_R)     -  0.5 c_h (B_{x,R} - B_{x,L}).
```

Implementation: `vortex/riemann.py:62-66`. We then *overwrite* the
normal field on both sides with `B_x^*` (line 70-72), so the rest of
the HLLD calculation sees a single-valued `B_n` --- this is the trick
that makes the eight-wave system look like a seven-wave one in the
interior.

#### 4.3.3 Outer wave speeds (Davis-Einfeldt)

The outermost waves of the fan are the two fast magnetosonic shocks /
rarefactions. Bound them by

```
S_L = min(u_L - c_{f,L},  u_R - c_{f,R}),
S_R = max(u_L + c_{f,L},  u_R + c_{f,R}).
```

This is Davis (1988) / Einfeldt (1988) and is what HLLD inherits from
its HLL ancestor. Implementation: `vortex/riemann.py:96-97`. The fast
speeds `c_{f,L}` and `c_{f,R}` are computed face-by-face from the
already-overwritten normal field (`vortex/riemann.py:92-93`,
`vortex/riemann.py:244-257`).

If `S_L >= 0`, the entire wave fan moves rightward and the upwind
state is just `Q_L`; the flux is `F(U_L)`. If `S_R <= 0`, the upwind
state is `Q_R` and the flux is `F(U_R)`. These two cases short-circuit
the rest of the calculation and are picked out by the region masks at
`vortex/riemann.py:213-217`.

#### 4.3.4 Entropy (contact) wave speed `S_M`

Inside the fan we need a third wave speed: the contact / entropy speed
`S_M`. From the Rankine-Hugoniot jump conditions for mass and
x-momentum across the L-to-* interface (`S_L`) and *-to-R interface
(`S_R`), demanding continuity of `rho * (u - S)` and total pressure
`p_T` across the contact gives

```
S_M = ( (S_R - u_R) rho_R u_R  -  (S_L - u_L) rho_L u_L  -  p_T,R + p_T,L )
      / ( (S_R - u_R) rho_R  -  (S_L - u_L) rho_L ).
```

Implementation: `vortex/riemann.py:104-107`. The `_TINY` floor on the
denominator handles the rare degenerate case where both bounding
states are nearly stationary in the face frame.

#### 4.3.5 Star-region total pressure

Continuity of total pressure across the contact wave forces a single
star-region `p_T^*` (this is *the* extra constraint that makes the
five-wave fan well-posed):

```
p_T^*  =  p_T,L + rho_L (S_L - u_L) (S_M - u_L)
       =  p_T,R + rho_R (S_R - u_R) (S_M - u_R).
```

The code uses the L-side form at `vortex/riemann.py:110`.

#### 4.3.6 Densities in the star region

Mass conservation across the L-to-*L and R-to-*R fast waves gives

```
rho_L^*  =  rho_L (S_L - u_L) / (S_L - S_M),
rho_R^*  =  rho_R (S_R - u_R) / (S_R - S_M).
```

`vortex/riemann.py:113-114`.

#### 4.3.7 Tangential velocity and field in the star region

The Rankine-Hugoniot conditions for the tangential momentum and
tangential induction equation across the fast wave produce, after a
few lines of algebra,

```
v_L^*    =  v_L  -  B_n B_{y,L}  *  (S_M - u_L) / D_L,
B_{y,L}^* = B_{y,L} * ( rho_L (S_L - u_L)^2  -  B_n^2 ) / D_L,

D_L  =  rho_L (S_L - u_L) (S_L - S_M)  -  B_n^2,
```

with the same expressions for the z-tangential components and the same
form on the R side with subscripts swapped. Implementation:
`vortex/riemann.py:117-149`. The three subtleties:

- The denominator `D_L` (or `D_R`) can vanish when `B_n^2` approaches
  `rho_L (S_L - u_L)^2`. That is the Alfven-sing-fast degeneracy. The
  code carries `_TINY` floors and a degeneracy mask
  (`deg_mask_L` / `deg_mask_R` at lines 140-149) that, when the
  denominator is tiny, replaces the formula with the trivial passthrough
  `v^* = v`, `B_y^* = B_y`. This is the Miyoshi-Kusano "limiting case"
  of Section 5.2.
- The expressions for `v^*` and `B_y^*` *both* use the same
  `D` denominator; computing `1/D` once and reusing it is a small
  performance win.
- The tangential `z` components (`w`, `B_z`) follow exactly the same
  formula with `(v, B_y)` replaced by `(w, B_z)`. The code handles all
  three components at once.

#### 4.3.8 Star-region energies

From energy conservation across the fast wave,

```
E_L^*  =  ( (S_L - u_L) E_L  -  p_T,L u_L  +  p_T^* S_M
            +  B_n ( v_L . B_L  -  v_L^* . B_L^* ) )  /  (S_L - S_M),
```

and analogously for `E_R^*`. The code at `vortex/riemann.py:157-162`
matches this expression line for line, with `vdotbL`, `vdotbR`,
`vdotbL_s`, `vdotbR_s` precomputed at lines 152-155.

#### 4.3.9 The Alfven (rotational) waves and double-star states

Between the contact wave and each fast wave there is one further
inner wave: the rotational Alfven discontinuity. Its speed is

```
S_L^*  =  S_M  -  |B_n| / sqrt(rho_L^*),
S_R^*  =  S_M  +  |B_n| / sqrt(rho_R^*).
```

`vortex/riemann.py:167-168`. Across these waves the density and
normal velocity stay constant (it is a rotational, not compressive,
wave), but the tangential velocity and tangential field rotate. The
double-star states are obtained by demanding continuity of total
pressure and using the Rankine-Hugoniot conditions on the rotational
wave:

```
v_dbl  =  ( sqrt(rho_L^*) v_L^* + sqrt(rho_R^*) v_R^*  +  (B_{y,R}^* - B_{y,L}^*) sign(B_n) )
          /  ( sqrt(rho_L^*) + sqrt(rho_R^*) ),

B_y_dbl = ( sqrt(rho_L^*) B_{y,R}^*  +  sqrt(rho_R^*) B_{y,L}^*
            +  sqrt(rho_L^*) sqrt(rho_R^*) (v_R^* - v_L^*) sign(B_n) )
          /  ( sqrt(rho_L^*) + sqrt(rho_R^*) ),

E_L^**  =  E_L^*  -  sqrt(rho_L^*) (v_L^* . B_L^*  -  v_dbl . B_dbl) sign(B_n),
E_R^**  =  E_R^*  +  sqrt(rho_R^*) (v_R^* . B_R^*  -  v_dbl . B_dbl) sign(B_n).
```

The same form holds for `w_dbl` and `B_z_dbl`. Both sides see the
*same* `v_dbl`, `B_y_dbl`, `B_z_dbl` --- those are continuous across
the contact wave by construction --- but `E^**` and `rho^**` differ.
Implementation: `vortex/riemann.py:170-193`.

#### 4.3.10 Region selection and assembly

With six possible upwind states (`L`, `L^*`, `L^**`, `R^**`, `R^*`,
`R`) the final flux at the face is picked by which region the face
lives in:

| Region    | Flux               | Condition                |
|-----------|--------------------|--------------------------|
| L         | `F_L`              | `S_L >= 0`               |
| L^*       | `F_L + S_L (U_L^* - U_L)`               | `S_L < 0 <= S_L^*` |
| L^**      | `F_L^* + S_L^* (U_L^** - U_L^*)`        | `S_L^* < 0 <= S_M` |
| R^**      | `F_R^* + S_R^* (U_R^** - U_R^*)`        | `S_M < 0 <= S_R^*` |
| R^*       | `F_R + S_R (U_R^* - U_R)`               | `S_R^* < 0 <= S_R` |
| R         | `F_R`              | `S_R <= 0`               |

The code builds the four candidate fluxes (`F_L^*`, `F_R^*`,
`F_L^**`, `F_R^**`) at `vortex/riemann.py:206-209`, the boolean masks
at lines 213-217, and the final selection by nested `np.where` at
lines 219-236.

#### 4.3.11 GLM corrections to the assembled flux

The HLLD machine above produced an eight-equation flux. Two rows of
that flux (the `B_x` row and the `psi` row) are *wrong* for GLM,
because the wave structure of the GLM subsystem is different. Replace
them with the analytic Riemann solution from Section 4.3.2:

```
F[B_x] = psi^*,
F[psi] = c_h^2 * B_x^*.
```

`vortex/riemann.py:239-240`. After this, the eight-equation MHD flux
plus the two-row GLM flux is consistent.

#### 4.3.12 The y-direction

The y-direction Riemann solver at `vortex/riemann.py:304-327` is
implemented by *swapping x and y components* on the input primitive
states (`u <-> v`, `B_x <-> B_y`), calling `hlld_x` on the swapped
states, and unswapping the resulting flux. This avoids a duplicate
algebra sheet for `hlld_y` at the cost of a few extra array copies.
The component-swap helper is `vortex/riemann.py:316-327`.

#### 4.3.13 Sanity checks

Three properties of HLLD are unit-tested at
`tests/test_riemann.py`:

- *Constant-state consistency*: with `Q_L == Q_R`, the HLLD flux must
  equal the analytic flux `F(U)` (no Riemann fan to resolve). This is
  `test_hlld_constant_state_matches_analytic`.
- *Euler limit*: with `B == 0` and Sod-tube ICs, the mass flux at the
  face must be O(0.4) and positive. This is
  `test_hlld_euler_limit_sod`.
- *Mirror symmetry*: flipping `(L, R)` along with `(u, B_n)` flips the
  sign of the mass flux. This is `test_hlld_mirror_symmetry`.

### 4.4 GLM divergence cleaning

GLM augments the eight-equation system with a ninth scalar `psi` whose
evolution couples *only* to the normal magnetic field. Dedner et al.
(2002) showed that three choices of operator on `psi` are possible
(elliptic, parabolic, hyperbolic) and that the *hyperbolic* choice,

```
dB/dt   + curl(E)            =  -grad(psi),
dpsi/dt + c_h^2 div(B)       =  -(c_h / c_r) psi,
```

is the cheapest in an unstaggered FVM context: it adds one scalar, one
extra equation, only modifies the induction and `psi` equations, and
all the new structure lives at the Riemann-solver level (Section
4.3.2 above). The underlying mathematics is that any local divergence
error becomes a wave packet that travels outward at finite speed
`c_h`, while the parabolic damping with rate `c_h / c_r`
exponentially decays the wave in place.

#### 4.4.1 The hyperbolic part: baked into the flux

The GLM additions to the flux are exactly two rows: `F[B_x] = psi`,
`F[psi] = c_h^2 B_x` (and `G[B_y] = psi`, `G[psi] = c_h^2 B_y`).
These are visible in `vortex/equations.py:flux_x` lines 129, 133 and
in `flux_y` lines 158, 161. The Riemann solver then overwrites them
with the analytic 2x2 Riemann solution at
`vortex/riemann.py:239-240`. Nothing else in the eight-equation MHD
flux is changed.

#### 4.4.2 The parabolic part: damping

The remaining piece is the source term `-(c_h / c_r) psi` on the
right-hand side of the `psi` equation. For a fixed `c_h` and `c_r`
this is a linear ODE in `psi`:

```
dpsi/dt = -(c_h / c_r) psi   ==>   psi(t + dt) = psi(t) exp(-c_h dt / c_r).
```

The implementation `damp_psi` at `vortex/glm.py:16-25` integrates this
analytically over each RK stage; no operator splitting error to leading
order. This is robust to `c_h * dt / c_r` of order unity, which is what
matters for stability.

#### 4.4.3 Choice of `c_h` and `c_r`

`c_h` must bound the largest signal speed of the MHD subsystem so that
the GLM waves move at least as fast as the physical waves; otherwise a
divergence error generated at a fast shock would not have time to
propagate away before the shock moves on. The code uses

```
c_h = max over the interior of  ( max(|u|, |v|) + c_f ),
```

evaluated once per step at `vortex/solver.py:24-39` (`_fast_ch`). This
gives the tightest CFL-respecting choice.

`c_r` is a non-dimensional damping ratio. Dedner et al. recommend
`c_r ~ 0.18`; smaller values damp `psi` harder per time step but make
the parabolic stiffness worse, and larger values let divergence errors
linger. The default is `c_r = 0.18` at
`vortex/mesh.py:118` (`load_config`) and the YAML at
`data/raw/orszag_tang_128.yaml:27`.

### 4.5 Time integration

The semi-discrete update is `dU/dt = L(U)` with `L(U) = -(dF/dx +
dG/dy)`. Time-stepping uses the Shu-Osher SSP-RK2 scheme:

```
U^*    = U^n  +  dt L(U^n),
U^{n+1} = 0.5 ( U^n + U^*  +  dt L(U^*) ).
```

Two stages, second-order accurate, total-variation-diminishing under
the same CFL constant as forward Euler. Implementation:
`vortex/integrator.py:74-96`. After each RK stage the loop in
`vortex/solver.py:run` calls `damp_psi` to apply the parabolic GLM
source.

#### 4.5.1 The CFL condition

The largest signal in the discrete system is

```
sigma = max ( |u| + c_f^x,  |v| + c_f^y,  c_h ),
```

with `c_f^x` the fast magnetosonic speed in the x-direction (i.e.
using `B_x` as the normal field) and `c_f^y` analogously. The CFL
step is

```
dt = CFL * min(dx, dy) / sigma,
```

with `CFL = 0.3` for the production runs. Why `c_h` is in the max:
because the GLM eigenvalues are `+/- c_h` and HLLD already uses the
fast magnetosonic speed for the eight-equation MHD subsystem; the
combined signal speed is the larger of the two. Implementation:
`vortex/integrator.py:23-45`.

#### 4.5.2 Boundary fills inside the RHS

`rhs` at `vortex/integrator.py:55-71` calls `apply_bc` *first*. This
matters for SSP-RK2 because the second stage evaluates `L(U^*)` with
`U^*` the result of the first stage --- which has its own ghost-cell
state that is *stale* relative to the new interior. Refilling ghosts
inside `rhs` keeps the second-stage flux consistent with the first.

### 4.6 The run loop

`vortex.solver.run` at `vortex/solver.py:42-123` ties everything
together:

1. Load the initial conditions for the named problem
   (`problems.get(cfg.problem)`).
2. Apply the boundary fills.
3. Snapshot the initial state to disk.
4. Loop until `t >= tfinal`:
   a. Compute `c_h` from the current state (`_fast_ch`).
   b. Compute `dt` from CFL with that `c_h`.
   c. Clip `dt` so we land exactly on the next requested snapshot
      time or `tfinal`.
   d. Run one SSP-RK2 step (`ssp_rk2_step`), then one parabolic GLM
      damping (`damp_psi`).
   e. Every `log_every` steps, log `t`, `dt`, `c_h`,
      `max|div B|`, mass and energy drift, and wall time.
   f. If `t` has crossed any requested snapshot time, save a
      compressed `.npz` of the interior state.
5. Save a `final` snapshot.

The conserved-totals diagnostics `mass`, `mom_x/y/z`, and `energy` come
from `vortex/diagnostics.py:30-41`; the `max|div B|` from
`vortex/diagnostics.py:13-27`.


## 5. Boundary Conditions

The four supported kinds (`vortex/boundary.py:BCKind`) are:

- *Periodic*: ghost layer = the opposite end of the interior. Used for
  every side of Orszag-Tang.
- *Outflow* (zero-gradient): every ghost cell takes the value of the
  nearest interior cell. Suitable when waves should leave the domain
  without reflection. Implemented at `vortex/boundary.py:48-50` and
  `:62-64`.
- *Reflective*: ghost cells mirror the interior across the boundary,
  with the *normal* momentum component (`rho u` at an x-wall, `rho v`
  at a y-wall) and the *normal* magnetic-field component (`B_x` at an
  x-wall, `B_y` at a y-wall) sign-flipped. Tangential `B`, tangential
  velocity, density, energy, and `psi` are copied. Source:
  `vortex/boundary.py:51-55`, `:65-69`, `:80-83`, `:93-96`.

Why `psi` is *copied* rather than flipped at a reflective wall: `psi`
is a Lagrange multiplier dual to `div B`, not a physical vector
component. Reflecting the normal `B` already enforces `B_n = 0` at the
wall in the limit of a thin ghost layer; flipping `psi` would create a
spurious source of divergence cleaning at the wall.

The unit tests `tests/test_boundary.py` check periodic and outflow
fills with controlled state arrays. Reflective is not currently
exercised in tests because Orszag-Tang does not use it.


## 6. Diagnostics

### 6.1 Cell-centered `div B`

The simplest diagnostic for divergence error is centered finite
differences on cell-centered `B`:

```
(div B)_{i,j} = ( B_{x, i+1,j} - B_{x, i-1,j} ) / (2 dx)
              + ( B_{y, i,j+1} - B_{y, i,j-1} ) / (2 dy).
```

Implementation: `vortex/diagnostics.py:13-27`. *This is not the
quantity GLM actually damps.* GLM cleans a different discrete operator
that lives in the `psi` evolution; the cell-centered diagnostic above
is one consistent measure but tends to be *pessimistic* (i.e., larger
than the GLM-natural one) because it does not match the upwind
stencil used inside the Riemann solver. Section 8.5 below shows what
that means in practice.

### 6.2 Conservation drift

`conserved_totals` at `vortex/diagnostics.py:30-41` returns
`area * sum` over the interior of `rho`, `rho u`, `rho v`, `rho w`,
and `E`. The `run` loop logs the drift relative to the initial
snapshot every `log_every` steps. For a finite-volume solver with
periodic boundaries these should drift only at floating-point
rounding (~1e-16 per step), and they do.


## 7. Test Problem: the Orszag-Tang Vortex

### 7.1 The setup

The Orszag-Tang vortex is the canonical 2D MHD benchmark. The domain
is `[0, 1]^2` with periodic boundaries on every side. Initial state is
*uniform* `rho` and `p`, with sinusoidal velocity and field:

```
rho      =  1,
p        =  1 / gamma,
v(x,y,0) =  (-sin(2 pi y),   sin(2 pi x),   0),
B(x,y,0) =  B_0 (-sin(2 pi y),   sin(4 pi x),   0),
```

with `B_0 = 1 / gamma`. Two normalization conventions appear in the
literature: `B_0 = 1 / gamma` (used here, gives `beta_0 = 2 gamma`)
and `B_0 = 1 / sqrt(4 pi)` (Athena and Athena++ default). The
qualitative density and pressure topology at `t = 0.5` is the same
under both conventions; only the absolute amplitude of the magnetic
field differs. Implementation: `vortex/problems/orszag_tang.py:36-54`.

### 7.2 Why this problem

Three properties make it the right first 2D benchmark:

- All-sinusoidal initial data: no geometric features to discretize, no
  density jumps to capture.
- Periodic on every side: no outflow / inflow boundary conditions to
  debug.
- Smooth at `t = 0` but develops sharp shocks and a thin central
  current sheet by `t = 0.5`. Every wave family is exercised.

If something is wrong, it is the solver, not the setup.

### 7.3 What the solution looks like

By `t = 0.5`:

- A *central bow-tie current sheet* runs roughly along `y = 0.5`
  (with two cusps, hence "bow tie").
- *Four oblique shocks* bound four low-density bays, one per
  quadrant, in the corners of the domain.
- *Density ridges* compress symmetrically near the top and bottom.
- *Magnetic pressure* peaks at four spots along the four arms of the
  current sheet --- this is *flux pile-up* just upstream of the
  reconnection line.

This pattern is the de-facto reference; see Londrillo and Del Zanna
(2000) Section 5, the Athena Orszag-Tang test page, and the Athena++
test suite. The solver reproduces it at 128^2 with minmod (Section 8).

### 7.4 Grid configurations

Four resolutions are configured in YAML:

- `data/raw/orszag_tang_32.yaml`
- `data/raw/orszag_tang_64.yaml`
- `data/raw/orszag_tang_128.yaml` (production)
- `data/raw/orszag_tang_256.yaml` (convergence reference)

All use `CFL = 0.3`, `gamma = 5/3`, `limiter = minmod`,
`glm.cr = 0.18`, and snapshot times `[0.1, 0.2, 0.3, 0.4, 0.5]`.


## 8. Validation Results

Numbers in this section come from the report
(`docs/source/index.tex`) and the `run.log` files written under
`data/final/<run>/`.

### 8.1 Density topology at t = 0.5

The 128^2 minmod run reproduces the canonical topology (central
bow-tie, four oblique shocks, density ridges). Quantitatively the
density extrema are roughly `rho_min ~ 0.47` and `rho_max ~ 2.16`.
The Athena++ MC-limiter reference at the same resolution sits closer
to `rho_min ~ 0.1` and `rho_max ~ 3.5`. The gap is dominated by the
limiter: minmod is the most diffusive of the common TVD limiters and
clips slopes at extrema more aggressively than MC or van Leer.

### 8.2 Time evolution

Snapshots at `t = 0.1, 0.3, 0.5` show:

- `t = 0.1`: still essentially smooth; the vortices are just rotating.
- `t = 0.3`: diagonal shocks have formed and are starting to drive gas
  toward the central bow-tie.
- `t = 0.5`: bow-tie is fully developed; magnetic pressure peaks line
  up along the four arms.

This is the same progression reproduced in every Orszag-Tang reference
run.

### 8.3 Convergence

Density error against the 256^2 reference at `t = 0.5`, with the
reference area-averaged onto each coarse grid:

| nx  | L^1 error | L^2 error |
|-----|-----------|-----------|
| 32  | 1.255e-1  | 1.733e-1  |
| 64  | 7.271e-2  | 1.074e-1  |
| 128 | 2.873e-2  | 4.258e-2  |

Observed L^1 convergence rates:

- 32 -> 64: `log_2(0.126 / 0.073) ~ 0.79`.
- 64 -> 128: `log_2(0.073 / 0.029) ~ 1.33`.

The rate is between 1 and 2. This is the expected behavior for any
TVD-limited Godunov scheme on a shock-dominated flow: the underlying
MUSCL machine is formally second-order in smooth regions, but at
shocks and current sheets the limiter clips slopes and the local
order drops to one. The global rate is therefore some weighted average
of 1 and 2; on Orszag-Tang at `t = 0.5` essentially every cell has
nontrivial gradients, so the rate sits closer to 1 than to 2 at
coarse resolutions and approaches 2 as the smooth fraction grows.

The convergence study script is at `scripts/convergence_study.py:54`.

### 8.4 Conservation

Mass, all three momenta, and total energy are conserved to floating-
point precision at every logged step (drift ~1e-16 per step in the
run log). This is the expected behavior of a conservative
finite-volume update with periodic boundaries; the conservation check
is a basic consistency test on the flux assembly and the boundary
fills, and serves as a sanity check after every change.

### 8.5 Divergence control

`max|div B|` (the cell-centered centered-difference diagnostic) on the
128^2 run grows steeply during shock formation (up to about `t = 0.15`
when the diagonal shocks first crystallize), then plateaus in the
range `1e0` to `1e1` through the bulk of the simulation, with a
late-time uptick near `t = 0.5` as the central current sheet tightens.

Two things are worth noting:

- *Plateau, not decay.* GLM bounds the divergence error rather than
  damping it to zero. This is the correct behavior of the hyperbolic
  cleaning: divergence is converted into wave packets that propagate
  away at `c_h`, and the parabolic source decays them at rate
  `c_h / c_r`. In a periodic domain the waves never leave; equilibrium
  is set by the balance between source (shocks generating divergence)
  and decay.
- *Pessimistic diagnostic.* The cell-centered centered-difference
  measure is *not* the operator GLM actually damps. The GLM-natural
  measure (the discrete `div B` consistent with the upwind flux
  stencil) is consistently smaller. Above-unity absolute values in the
  cell-centered diagnostic do not correspond to visible corruption of
  the density or pressure pattern.

The divergence history is plotted in
`data/final/orszag_tang_128/figs/divb_history.pdf`.


## 9. Discussion

### 9.1 Limiter choice and accuracy

The minmod limiter at `vortex/reconstruction.py:20-25` returns

```
0,                       if delta_L * delta_R <= 0
the smaller-magnitude one, otherwise.
```

This is the most-diffusive TVD limiter: it always picks the smallest
slope consistent with monotonicity, so it never overshoots, but at the
cost of smearing sharp features. MC at `:26-30` allows up to twice the
one-sided slope when the central difference agrees in sign, and
recovers sharper extrema. Van Leer at `:31-34` is intermediate.

For Orszag-Tang the practical effect is the density-extrema gap
mentioned in 8.1: minmod clips the extrema toward the cell mean and
the visible bow-tie is slightly broader than the Athena++ MC
reference. Switching `limiter: minmod` to `limiter: mc` in the YAML
config closes most of that gap, at the cost of occasional positivity
issues in the lowest-beta cells along the current sheet. The default
is conservative on purpose.

### 9.2 GLM vs constrained transport vs Powell

| Method                | Cost          | Strictness     | Storage      | Notes                                                                |
|-----------------------|---------------|----------------|--------------|----------------------------------------------------------------------|
| Constrained transport | one extra E_z | exact: O(eps_machine) | staggered B   | Best long-time fidelity; needs face-centered B and corner-centered E |
| Powell source terms   | trivial       | bounded        | unstaggered  | Loses strict conservation; symmetrizes the system                     |
| GLM (this solver)     | one extra var | bounded        | unstaggered  | Cheapest unstaggered option; what we use                              |

The pathological case for GLM is a long-running simulation with
recurring strong-shock events in a closed domain: the divergence-error
plateau can creep upward over many sound-crossing times. CT does not
have that problem. For a single-vortex run to `t = 0.5` (a few
sound-crossing times) GLM is more than adequate.

### 9.3 Where the solver is expected to degrade

- *Very low-plasma-beta cells*. In strong-field, weak-pressure regions
  the magnetic pressure dominates and small numerical errors in `B`
  feed back as large errors in the dynamics. The minmod limiter is the
  first line of defence; positivity-preserving variants of MUSCL or
  WENO would do better.
- *Strong grid-aligned `B`*. When `B` is closely aligned with the
  x-axis (so `B_y -> 0`) the Alfven and slow-wave speeds in the
  y-direction become degenerate. The HLLD code has guards
  (`vortex/riemann.py:140-149`) but nearby cells can still be noisy.
- *Strict-hyperbolicity degeneracies*. When `B_n = 0` exactly (e.g.
  along certain symmetry lines) the Alfven speed vanishes, the star
  region collapses, and the formulas for `v^*` / `B_y^*` divide by
  small numbers. The `_TINY` floors handle this without raising
  exceptions, but the local accuracy is reduced.

### 9.4 Why the convergence rate sits where it does

The rule of thumb for a TVD scheme on a shock-dominated 2D problem is
that the global L^1 / L^2 rate is between the smooth-region rate (2 for
MUSCL) and the shock-region rate (1 for any TVD scheme that clips
slopes at extrema). Toro (2009, Ch. 13) gives this argument
explicitly. The observed rate of ~0.79 at coarse resolutions and ~1.33
at fine resolutions matches that pattern: the coarse runs see almost
nothing but shock-dominated cells, and the fine runs start to resolve
the smooth background and pull the rate toward 2.

Three diagnostics that would tighten the rate further:

- Repeat the study with the MC limiter --- expect `rate -> 1.5` or so.
- Repeat in smooth regions only (a cropped L^1 norm excluding the
  current sheet).
- Use an SSP-RK3 time integrator and a third-order spatial
  reconstruction (PPM or WENO) --- expect a true second-order rate in
  smooth regions, but the global rate is still capped at 1 wherever
  TVD limiting kicks in.


## 10. Repo Layout Cheat Sheet

This table is for navigation, complementary to the `Repo layout`
table in the README.

### `vortex/` --- the solver package

| File                              | One-line role                                                                              |
|-----------------------------------|--------------------------------------------------------------------------------------------|
| `mesh.py`                         | `Mesh` and `RunConfig` dataclasses, YAML loader.                                          |
| `equations.py`                    | State-vector layout, primitive <-> conserved, analytic fluxes, fast magnetosonic speed.    |
| `reconstruction.py`               | MUSCL slopes (minmod / MC / van Leer) and `reconstruct_x` / `reconstruct_y`.               |
| `riemann.py`                      | HLLD x-direction Riemann solver, GLM `B_n` / `psi` correction, y-direction by axis swap.   |
| `glm.py`                          | Parabolic damping `psi *= exp(-c_h dt / c_r)`.                                             |
| `boundary.py`                     | Ghost-cell fills (periodic / outflow / reflective).                                        |
| `integrator.py`                   | RHS evaluator and SSP-RK2 stepper, CFL.                                                    |
| `diagnostics.py`                  | Centered-difference `div(B)` and conserved-totals area-weighted sums.                      |
| `io.py`                           | Snapshot save/load (`.npz`) and run-directory logger.                                      |
| `solver.py`                       | Top-level `run` function: ties the modules into a time-stepping loop.                      |
| `problems/orszag_tang.py`         | Orszag-Tang initial conditions in primitive form.                                          |

### `tests/`

| File                              | Tests                                                                                       |
|-----------------------------------|--------------------------------------------------------------------------------------------|
| `test_equations.py`               | `prim_to_cons` / `cons_to_prim` round-trip, Euler limit of `flux_x`, x/y flux symmetry.     |
| `test_riemann.py`                 | HLLD constant-state consistency, Euler-Sod limit, mirror symmetry under `(L, R)` flip.      |
| `test_boundary.py`                | Periodic and outflow ghost fills.                                                          |

### `data/raw/`

| File                              | Purpose                                                                                    |
|-----------------------------------|--------------------------------------------------------------------------------------------|
| `orszag_tang_32.yaml`             | Coarse grid for the convergence study.                                                     |
| `orszag_tang_64.yaml`             | Mid grid for the convergence study.                                                        |
| `orszag_tang_128.yaml`            | Production resolution; the run figured in the report.                                      |
| `orszag_tang_256.yaml`            | Reference grid for the convergence study.                                                  |


## 11. Annotated Reference List

PDFs in `references/` are linked to the part of the project they
ground.

| Reference                                                                                               | Where it grounds the project                                                                 |
|---------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| Miyoshi, T. and Kusano, K. *J. Comput. Phys.* 208 (2005). [`miyoshi-kusano-2005-...pdf`]                | The HLLD algorithm in `vortex/riemann.py`. Sections 2 and 5 are what the code mirrors.       |
| Dedner, A. et al. *J. Comput. Phys.* 175 (2002). [`dedner-et-al-2002-...pdf`]                           | GLM divergence cleaning. Sections 3-4 cover the eigensystem and the FVM coupling.            |
| Powell, K. G. et al. *J. Comput. Phys.* 154 (1999). [`powell-et-al-1999-...pdf`]                        | The eight-wave conservative form. Section 2 is the compact equation reference.               |
| Brio, M. and Wu, C. C. *J. Comput. Phys.* 75 (1988). [`brio-wu-1988-...pdf`]                            | The 1D MHD shock tube and the 7x7 eigenstructure. Useful for understanding the MHD waves.    |
| Londrillo, P. and Del Zanna, L. *ApJ* 530 (2000). [`londrillo-del-zanna-2000-...pdf`]                   | Reference Orszag-Tang topology at `t = 0.5`. Also: discussion of div(B) in upwind FVM codes. |
| Balsara, D. S. and Spicer, D. *J. Comput. Phys.* 149 (1999). [`balsara-spicer-1999-...pdf`]             | Constrained transport. Background reading for why we picked GLM instead.                     |
| Toro, E. F. *Riemann Solvers and Numerical Methods for Fluid Dynamics*. Springer, 2009.                 | Godunov FVM, MUSCL, slope limiters, CFL, convergence-rate arguments.                         |
| Chen, F. F. *Introduction to Plasma Physics and Controlled Fusion*. Springer, 2016.                     | The plasma-physics setup in Section 2.                                                        |
| Athena Orszag-Tang test page (referenced in `references/athena++-mhd-code.webloc`)                      | Visual reference for what the density and pressure should look like at `t = 0.5`.            |


## 12. Open Questions and Natural Extensions

In rough order of effort:

- *MC or van Leer limiter for the production run.* Already implemented;
  one YAML edit to switch.
- *Constrained transport.* Replace GLM with a face-centered `B` /
  corner-centered `E_z` update. Bigger change: requires a new mesh
  layout, new flux assembly, and a new boundary module.
- *PPM or WENO reconstruction.* Drops the limiter, gains formal
  third-order accuracy in smooth regions, but does not change the
  shock-dominated rate.
- *SSP-RK3 time integration.* Three stages, third-order, same CFL
  constant as SSP-RK2; small change to `vortex/integrator.py`.
- *Resistive MHD.* Add a parabolic source `nabla x (eta nabla x B)` to
  the induction equation. Sub-cycled or super-time-stepped to avoid a
  parabolic CFL bottleneck.
- *3D.* The state vector and fluxes are already 3D-ready (we carry all
  three components of `v` and `B`); the change is to the mesh,
  reconstruction, and Riemann-solver glue, which all need a third
  axis.
- *AMR.* Block-structured AMR (a la PARAMESH or AMReX) on top of the
  existing solver. Substantial rewrite of the time loop and the
  boundary fills.
- *Numba- or JAX-compiled core.* The current vectorized NumPy is the
  practical performance ceiling at ~512^2 on a laptop. A JIT-compiled
  inner loop should give a 5--20x speedup with a few decorator
  changes; JAX gives the same plus GPU acceleration at the cost of a
  non-trivial port of the in-place updates.
