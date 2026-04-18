"""HLLD approximate Riemann solver for 2D ideal MHD with GLM coupling.

Implementation follows Miyoshi and Kusano (2005), Sec. 5, equations 38-67.
Inputs are left and right *primitive* states at a single face orientation, plus
the GLM cleaning speed ``ch``. The solver evaluates the five-wave pattern
(S_L, S_L*, S_M, S_R*, S_R) and returns the HLLD numerical flux in the
conservative 9-variable layout used elsewhere in the package.

The psi/Bx coupling from Dedner et al. (2002) is handled up front by
extracting a single-valued normal magnetic field from the psi/Bx Riemann
problem, then evaluating the MHD flux with that normal field. The psi and Bx
fluxes are then corrected to the Dedner values.

Only the x-direction solver is implemented; y-direction is obtained by a
component swap performed by the caller, then unswapping the returned flux.
"""

from __future__ import annotations

import numpy as np

from vortex.equations import (
    GAMMA,
    IBX,
    IBY,
    IBZ,
    IEN,
    IMX,
    IMY,
    IMZ,
    IPSI,
    IRHO,
    NVAR,
    prim_to_cons,
)

_TINY = 1e-12


def hlld_x(
    QL: np.ndarray,
    QR: np.ndarray,
    ch: float,
    gamma: float = GAMMA,
) -> np.ndarray:
    """HLLD flux at x-faces given primitive left/right states.

    Parameters
    ----------
    QL, QR : arrays of shape ``(NVAR, nfaces_x, nfaces_y)``
        Primitive states on the left and right of each face.
    ch : float
        GLM cleaning speed used for the psi/Bx Riemann coupling.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    F : array of shape ``(NVAR, nfaces_x, nfaces_y)``
        Conservative flux at each face.
    """
    # GLM: resolve Bx and psi from their 1D Riemann problem.
    bxL, bxR = QL[IBX], QR[IBX]
    psiL, psiR = QL[IPSI], QR[IPSI]
    bx_star = 0.5 * (bxL + bxR) - 0.5 * (psiR - psiL) / ch
    psi_star = 0.5 * (psiL + psiR) - 0.5 * ch * (bxR - bxL)

    # Overwrite the normal field seen by both sides with bx_star for the MHD
    # wave fan computation.
    QL = QL.copy()
    QR = QR.copy()
    QL[IBX] = bx_star
    QR[IBX] = bx_star

    # Unpack primitive states.
    rhoL, uL, vL, wL = QL[IRHO], QL[IMX], QL[IMY], QL[IMZ]
    byL, bzL = QL[IBY], QL[IBZ]
    pL = QL[IEN]

    rhoR, uR, vR, wR = QR[IRHO], QR[IMX], QR[IMY], QR[IMZ]
    byR, bzR = QR[IBY], QR[IBZ]
    pR = QR[IEN]

    bxN = bx_star  # normal (x) field; single-valued across the fan

    # Total pressure and fast speeds.
    b2L = bxN * bxN + byL * byL + bzL * bzL
    b2R = bxN * bxN + byR * byR + bzR * bzR
    ptL = pL + 0.5 * b2L
    ptR = pR + 0.5 * b2R

    cfL = _fast_speed_x(rhoL, pL, bxN, byL, bzL, gamma)
    cfR = _fast_speed_x(rhoR, pR, bxN, byR, bzR, gamma)

    # Outer wave speeds (Davis/Einfeldt bounds).
    SL = np.minimum(uL - cfL, uR - cfR)
    SR = np.maximum(uL + cfL, uR + cfR)

    # Conservative states (for total energy etc.).
    UL = prim_to_cons(QL, gamma)
    UR = prim_to_cons(QR, gamma)
    EL, ER = UL[IEN], UR[IEN]

    # Entropy wave speed S_M (eq. 38).
    num = (SR - uR) * rhoR * uR - (SL - uL) * rhoL * uL - ptR + ptL
    den = (SR - uR) * rhoR - (SL - uL) * rhoL
    SM = num / np.where(np.abs(den) > _TINY, den, np.sign(den) * _TINY + _TINY)

    # Total pressure in star region (eq. 41).
    pt_star = ptL + rhoL * (SL - uL) * (SM - uL)

    # Densities in the * region (eq. 43).
    rhoL_s = rhoL * (SL - uL) / (SL - SM)
    rhoR_s = rhoR * (SR - uR) / (SR - SM)

    # Tangential velocity/field in the * region (eqs. 44-47).
    denomL = rhoL * (SL - uL) * (SL - SM) - bxN * bxN
    denomR = rhoR * (SR - uR) * (SR - SM) - bxN * bxN
    safeL = np.where(np.abs(denomL) > _TINY, denomL, np.sign(denomL) * _TINY + _TINY)
    safeR = np.where(np.abs(denomR) > _TINY, denomR, np.sign(denomR) * _TINY + _TINY)

    fac_vL = (SM - uL) / safeL
    fac_vR = (SM - uR) / safeR

    vL_s = vL - bxN * byL * fac_vL
    wL_s = wL - bxN * bzL * fac_vL
    vR_s = vR - bxN * byR * fac_vR
    wR_s = wR - bxN * bzR * fac_vR

    fac_bL = (rhoL * (SL - uL) ** 2 - bxN * bxN) / safeL
    fac_bR = (rhoR * (SR - uR) ** 2 - bxN * bxN) / safeR

    byL_s = byL * fac_bL
    bzL_s = bzL * fac_bL
    byR_s = byR * fac_bR
    bzR_s = bzR * fac_bR

    # Degenerate case: when Bx^2 is negligible relative to rho(S-u)^2 we just
    # pass through the original tangential components (Miyoshi-Kusano Sec. 5.2).
    deg_mask_L = np.abs(denomL) <= _TINY
    deg_mask_R = np.abs(denomR) <= _TINY
    vL_s = np.where(deg_mask_L, vL, vL_s)
    wL_s = np.where(deg_mask_L, wL, wL_s)
    byL_s = np.where(deg_mask_L, byL, byL_s)
    bzL_s = np.where(deg_mask_L, bzL, bzL_s)
    vR_s = np.where(deg_mask_R, vR, vR_s)
    wR_s = np.where(deg_mask_R, wR, wR_s)
    byR_s = np.where(deg_mask_R, byR, byR_s)
    bzR_s = np.where(deg_mask_R, bzR, bzR_s)

    # Energy in the * region (eq. 48).
    vdotbL = uL * bxN + vL * byL + wL * bzL
    vdotbR = uR * bxN + vR * byR + wR * bzR
    vdotbL_s = SM * bxN + vL_s * byL_s + wL_s * bzL_s
    vdotbR_s = SM * bxN + vR_s * byR_s + wR_s * bzR_s

    EL_s = ((SL - uL) * EL - ptL * uL + pt_star * SM + bxN * (vdotbL - vdotbL_s)) / (
        SL - SM
    )
    ER_s = ((SR - uR) * ER - ptR * uR + pt_star * SM + bxN * (vdotbR - vdotbR_s)) / (
        SR - SM
    )

    # ** region (Alfven wave speeds and double-star states, eqs. 51-59).
    sqrt_rhoL_s = np.sqrt(np.maximum(rhoL_s, _TINY))
    sqrt_rhoR_s = np.sqrt(np.maximum(rhoR_s, _TINY))
    SL_s = SM - np.abs(bxN) / sqrt_rhoL_s
    SR_s = SM + np.abs(bxN) / sqrt_rhoR_s

    denom_dbl = sqrt_rhoL_s + sqrt_rhoR_s
    denom_dbl = np.where(denom_dbl > _TINY, denom_dbl, _TINY)
    sign_bn = np.where(bxN >= 0.0, 1.0, -1.0)

    v_dbl = (
        sqrt_rhoL_s * vL_s + sqrt_rhoR_s * vR_s + (byR_s - byL_s) * sign_bn
    ) / denom_dbl
    w_dbl = (
        sqrt_rhoL_s * wL_s + sqrt_rhoR_s * wR_s + (bzR_s - bzL_s) * sign_bn
    ) / denom_dbl
    by_dbl = (
        sqrt_rhoL_s * byR_s
        + sqrt_rhoR_s * byL_s
        + sqrt_rhoL_s * sqrt_rhoR_s * (vR_s - vL_s) * sign_bn
    ) / denom_dbl
    bz_dbl = (
        sqrt_rhoL_s * bzR_s
        + sqrt_rhoR_s * bzL_s
        + sqrt_rhoL_s * sqrt_rhoR_s * (wR_s - wL_s) * sign_bn
    ) / denom_dbl

    vdotb_dbl = SM * bxN + v_dbl * by_dbl + w_dbl * bz_dbl
    EL_ss = EL_s - sqrt_rhoL_s * (vdotbL_s - vdotb_dbl) * sign_bn
    ER_ss = ER_s + sqrt_rhoR_s * (vdotbR_s - vdotb_dbl) * sign_bn

    # Assemble conservative star and double-star states.
    UL_s = _assemble(rhoL_s, SM, vL_s, wL_s, bxN, byL_s, bzL_s, EL_s, psi_star)
    UR_s = _assemble(rhoR_s, SM, vR_s, wR_s, bxN, byR_s, bzR_s, ER_s, psi_star)
    UL_ss = _assemble(rhoL_s, SM, v_dbl, w_dbl, bxN, by_dbl, bz_dbl, EL_ss, psi_star)
    UR_ss = _assemble(rhoR_s, SM, v_dbl, w_dbl, bxN, by_dbl, bz_dbl, ER_ss, psi_star)

    # Fluxes on the outer states (MHD flux in conservative form, GLM plugged
    # in afterwards for the Bx/psi components).
    FL = _mhd_flux_x(UL, ptL, gamma)
    FR = _mhd_flux_x(UR, ptR, gamma)

    FL_s = FL + SL * (UL_s - UL)
    FR_s = FR + SR * (UR_s - UR)
    FL_ss = FL_s + SL_s * (UL_ss - UL_s)
    FR_ss = FR_s + SR_s * (UR_ss - UR_s)

    F = np.empty_like(UL)
    # Region masks.
    m0 = SL >= 0.0
    m4 = SR <= 0.0
    m1 = (SL < 0.0) & (SL_s >= 0.0)
    m3 = (SR > 0.0) & (SR_s <= 0.0)
    m2 = (~m0) & (~m1) & (~m3) & (~m4)

    for k in range(NVAR):
        F[k] = np.where(
            m0,
            FL[k],
            np.where(
                m4,
                FR[k],
                np.where(
                    m1,
                    FL_s[k],
                    np.where(
                        m3,
                        FR_s[k],
                        np.where(m2 & (SM >= 0.0), FL_ss[k], FR_ss[k]),
                    ),
                ),
            ),
        )

    # GLM corrections: the analytic psi/Bx fluxes override the MHD values.
    F[IBX] = psi_star
    F[IPSI] = ch * ch * bx_star
    return F


def _fast_speed_x(
    rho: np.ndarray,
    p: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    gamma: float,
) -> np.ndarray:
    a2 = gamma * np.maximum(p, 1e-20) / rho
    b2 = (bx * bx + by * by + bz * bz) / rho
    bn2 = bx * bx / rho
    s = a2 + b2
    disc = np.maximum(s * s - 4.0 * a2 * bn2, 0.0)
    return np.sqrt(0.5 * (s + np.sqrt(disc)))


def _assemble(
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    E: np.ndarray,
    psi: np.ndarray,
) -> np.ndarray:
    U = np.empty((NVAR, *rho.shape), dtype=rho.dtype)
    U[IRHO] = rho
    U[IMX] = rho * u
    U[IMY] = rho * v
    U[IMZ] = rho * w
    U[IBX] = bx
    U[IBY] = by
    U[IBZ] = bz
    U[IEN] = E
    U[IPSI] = psi
    return U


def _mhd_flux_x(U: np.ndarray, ptot: np.ndarray, gamma: float) -> np.ndarray:
    rho = U[IRHO]
    u = U[IMX] / rho
    v = U[IMY] / rho
    w = U[IMZ] / rho
    bx, by, bz = U[IBX], U[IBY], U[IBZ]
    vdotb = u * bx + v * by + w * bz
    F = np.empty_like(U)
    F[IRHO] = rho * u
    F[IMX] = rho * u * u + ptot - bx * bx
    F[IMY] = rho * u * v - bx * by
    F[IMZ] = rho * u * w - bx * bz
    F[IBX] = 0.0  # overridden to psi downstream
    F[IBY] = u * by - v * bx
    F[IBZ] = u * bz - w * bx
    F[IEN] = (U[IEN] + ptot) * u - bx * vdotb
    F[IPSI] = 0.0  # overridden to ch^2 * bx downstream
    return F


def hlld_y(
    QL: np.ndarray, QR: np.ndarray, ch: float, gamma: float = GAMMA
) -> np.ndarray:
    """HLLD flux at y-faces by swapping x<->y components, calling hlld_x, then
    unswapping.
    """
    QLs = _swap_xy_prim(QL)
    QRs = _swap_xy_prim(QR)
    Fs = hlld_x(QLs, QRs, ch, gamma)
    return _swap_xy_cons_flux(Fs)


def _swap_xy_prim(Q: np.ndarray) -> np.ndarray:
    S = Q.copy()
    S[IMX], S[IMY] = Q[IMY].copy(), Q[IMX].copy()
    S[IBX], S[IBY] = Q[IBY].copy(), Q[IBX].copy()
    return S


def _swap_xy_cons_flux(F: np.ndarray) -> np.ndarray:
    S = F.copy()
    S[IMX], S[IMY] = F[IMY].copy(), F[IMX].copy()
    S[IBX], S[IBY] = F[IBY].copy(), F[IBX].copy()
    return S
