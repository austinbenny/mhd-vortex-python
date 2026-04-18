"""Top-level run orchestrator.

Ties together mesh loading, problem ICs, ghost-cell fills, SSP-RK2 stepping,
HLLD fluxes, GLM damping, logging, and snapshot IO.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from vortex import problems
from vortex.boundary import apply as apply_bc
from vortex.diagnostics import conserved_totals, div_b
from vortex.equations import IBX, IBY, IBZ, IMX, IMY, IMZ, IRHO
from vortex.glm import damp_psi
from vortex.integrator import compute_dt, ssp_rk2_step
from vortex.io import ensure_run_dir, save_snapshot, setup_logger
from vortex.mesh import Mesh, RunConfig


def _fast_ch(U: np.ndarray, mesh: Mesh, gamma: float) -> float:
    """Global max fast magnetosonic speed + flow speed over the interior."""
    g = mesh.nghost
    Ui = U[:, g : g + mesh.nx, g : g + mesh.ny]
    rho = Ui[IRHO]
    u = Ui[IMX] / rho
    v = Ui[IMY] / rho
    bx, by, bz = Ui[IBX], Ui[IBY], Ui[IBZ]
    kin = 0.5 * (Ui[IMX] ** 2 + Ui[IMY] ** 2 + Ui[IMZ] ** 2) / rho
    mag = 0.5 * (bx * bx + by * by + bz * bz)
    p = np.maximum((gamma - 1.0) * (Ui[7] - kin - mag), 1e-20)
    a2 = gamma * p / rho
    b2 = (bx * bx + by * by + bz * bz) / rho
    bn2 = np.maximum(bx * bx, by * by) / rho
    cf = np.sqrt(0.5 * (a2 + b2 + np.sqrt((a2 + b2) ** 2 - 4.0 * a2 * bn2)))
    return float(np.max(np.maximum(np.abs(u), np.abs(v)) + cf))


def run(mesh: Mesh, cfg: RunConfig, data_root: str | Path = "data/final") -> dict:
    """Execute a simulation from t=0 to ``cfg.tfinal``.

    Returns a summary dict with the final state path, elapsed wall time, and
    total step count.
    """
    run_dir = ensure_run_dir(data_root, cfg.run_name)
    logger = setup_logger(run_dir)

    ic = problems.get(cfg.problem)
    U = ic(mesh, cfg.gamma)
    apply_bc(U, mesh)

    t = 0.0
    step = 0
    totals0 = conserved_totals(U, mesh)
    t_wall0 = time.perf_counter()

    logger.info(
        "start problem=%s nx=%d ny=%d tfinal=%.4f cfl=%.3f limiter=%s gamma=%.4f cr=%.3f",
        cfg.problem,
        mesh.nx,
        mesh.ny,
        cfg.tfinal,
        cfg.cfl,
        cfg.limiter,
        cfg.gamma,
        cfg.glm_cr,
    )

    snapshot_times = list(cfg.snapshot_times)
    save_snapshot(run_dir, "t0", t, U, mesh)

    while t < cfg.tfinal - 1e-14:
        ch = _fast_ch(U, mesh, cfg.gamma)
        dt = compute_dt(U, mesh, cfg.cfl, ch, cfg.gamma)

        # Land exactly on the next checkpoint or tfinal.
        next_target = cfg.tfinal
        if snapshot_times:
            next_target = min(next_target, snapshot_times[0])
        dt = min(dt, next_target - t)

        ssp_rk2_step(U, mesh, dt, ch, cfg.limiter, cfg.gamma)
        damp_psi(U, dt, ch, cfg.glm_cr)

        t += dt
        step += 1

        if step % cfg.log_every == 0 or t >= cfg.tfinal - 1e-14:
            apply_bc(U, mesh)
            divb = div_b(U, mesh)
            totals = conserved_totals(U, mesh)
            drift = {k: totals[k] - totals0[k] for k in totals}
            logger.info(
                "step=%6d t=%.5f dt=%.3e ch=%.3f max|divB|=%.3e "
                "dmass=%+.2e denergy=%+.2e wall=%.1fs",
                step,
                t,
                dt,
                ch,
                float(np.max(np.abs(divb))),
                drift["mass"],
                drift["energy"],
                time.perf_counter() - t_wall0,
            )

        while snapshot_times and t >= snapshot_times[0] - 1e-12:
            ts = snapshot_times.pop(0)
            tag = f"t{ts:.3f}".replace(".", "p")
            save_snapshot(run_dir, tag, t, U, mesh)
            logger.info("snapshot saved at t=%.4f -> %s", t, tag)

    save_snapshot(run_dir, "final", t, U, mesh)
    elapsed = time.perf_counter() - t_wall0
    logger.info("done steps=%d elapsed=%.2fs final_t=%.5f", step, elapsed, t)
    return {
        "run_dir": str(run_dir),
        "steps": step,
        "elapsed_s": elapsed,
        "final_t": t,
    }
