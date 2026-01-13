#!/usr/bin/env python3
"""
Phase lines (F–M and M–SG) for the “dreaming” Hopfield model in
Agliari/Barra et al. (2019 preprint), replica-symmetric (RS) theory.

It computes, for each fixed t:
  - Tc(alpha): “outer” line where the retrieval solution (m>0) disappears under
              continuation in T (interpreted as M–SG boundary in the paper).
  - TR(alpha): “inner” line where RS free energies cross:
              A_retr(alpha,T,t) = A_SG(alpha,T,t)  (interpreted as F–M boundary).

Numerical stability features:
  - Gauss–Hermite quadrature for Gaussian averages.
  - sech^2 computed as 1 - tanh^2 (no cosh overflow).
  - logcosh computed stably.
  - Damped fixed-point iteration + adaptive damping.
  - Free-energy term with 1/t cancellation evaluated in a finite form.

WARNING (physics): RS may be wrong at low T / high alpha; use these curves as RS predictions.
"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np


# --------------------------
# Gaussian quadrature (Dx)
# --------------------------

@dataclass(frozen=True)
class GaussHermite:
    x: np.ndarray      # nodes for standard normal measure Dx
    w: np.ndarray      # weights for standard normal measure Dx

    @staticmethod
    def make(n: int = 80) -> "GaussHermite":
        # hermgauss integrates ∫ e^{-y^2} f(y) dy
        y, wy = np.polynomial.hermite.hermgauss(n)
        x = np.sqrt(2.0) * y
        w = wy / np.sqrt(np.pi)  # so that Σ w f(x) ≈ ∫ Dx f(x)
        return GaussHermite(x=x, w=w)


# --------------------------
# Stable elementary functions
# --------------------------

_LOG2 = math.log(2.0)

def logcosh_stable(u: np.ndarray) -> np.ndarray:
    """log(cosh(u)) without overflow."""
    a = np.abs(u)
    # log cosh u = |u| + log(1 + exp(-2|u|)) - log 2
    return a + np.log1p(np.exp(-2.0 * a)) - _LOG2

def expect(gh: GaussHermite, fvals: np.ndarray) -> float:
    """Compute ∫ Dx f(x) from vector of f(x_nodes)."""
    return float(np.sum(gh.w * fvals))


# --------------------------
# RS fixed-point maps
# --------------------------

@dataclass
class RSState:
    m: float
    q: float
    Q: float
    p: float
    Delta: float

    def as_array(self) -> np.ndarray:
        return np.array([self.m, self.q, self.Q, self.p, self.Delta], dtype=float)

    @staticmethod
    def from_array(a: np.ndarray) -> "RSState":
        return RSState(m=float(a[0]), q=float(a[1]), Q=float(a[2]), p=float(a[3]), Delta=float(a[4]))


def rs_map_t_gt_0(
    st: RSState, *, alpha: float, beta: float, t: float, gh: GaussHermite,
    branch: str, denom_min: float
) -> Optional[RSState]:
    """
    RS self-consistency (Eq. 2.10) mapping for t>0.

    branch:
      - "retr": retrieval (m>0)
      - "sg"  : spin-glass branch enforced with m=0
    """
    m, q, Q = st.m, st.q, st.Q

    denom = 1.0 - beta * (1.0 + t) * (Q - q)  # D ≡ 1 - β(1+t)(Q-q)
    if not np.isfinite(denom) or denom <= denom_min:
        return None

    Delta = 1.0 + (alpha * t) / denom                          # (2.10c)
    if not np.isfinite(Delta) or Delta <= 0.0:
        return None

    p = q * (1.0 + t) ** 2 / (denom ** 2)                      # (2.10b)
    if not np.isfinite(p) or p < 0.0:
        return None

    # Gaussian averages with u = (β/Δ)(m + sqrt(α p) x)
    s = math.sqrt(max(alpha * p, 0.0))
    u = (beta / Delta) * (m + s * gh.x)

    th = np.tanh(u)
    sech2 = 1.0 - th * th  # stable sech^2

    I_tanh = expect(gh, th)
    I_sech2 = expect(gh, sech2)

    # (2.10a)
    m_new = ((1.0 + t) / (Delta + t)) * I_tanh
    if branch == "sg":
        m_new = 0.0
    else:
        # pick the +m branch
        m_new = abs(m_new)

    # (2.10d)
    q_new = Q + t / (beta * (1.0 + t) * Delta) - I_sech2 / (Delta ** 2)

    # (2.10e): Q Δ^2 = RHS
    RHS = (
        1.0
        - (t * Delta) / (beta * (1.0 + t))
        + (alpha * p * t * t) / ((1.0 + t) ** 2)
        - (m * m) * t * (t + 2.0 * Delta) / ((1.0 + t) ** 2)
        - (2.0 * alpha * beta * p * t) / ((1.0 + t) * Delta) * I_sech2
    )
    Q_new = RHS / (Delta ** 2)

    if not (np.isfinite(m_new) and np.isfinite(q_new) and np.isfinite(Q_new)):
        return None

    return RSState(m=m_new, q=float(q_new), Q=float(Q_new), p=float(p), Delta=float(Delta))


def rs_map_t_eq_0(
    m: float, q: float, *, alpha: float, beta: float, gh: GaussHermite,
    branch: str, denom_min: float
) -> Optional[Tuple[float, float, float]]:
    """
    Hopfield RS (paper Eqs. 1.8-1.9), computed with explicit ξ-average (ξ=±1).
    Returns (m_new, q_new, r) where r := q / [1 - β(1-q)]^2 (often used in Hopfield literature).
    """
    denom = 1.0 - beta * (1.0 - q)  # 1 - β(1-q)
    if not np.isfinite(denom) or denom <= denom_min:
        return None

    # noise amplitude per Eqs (1.8)-(1.9):  x * sqrt(α q) / denom
    amp = math.sqrt(max(alpha * q, 0.0)) / denom

    u_p = beta * ( m + amp * gh.x)   # ξ = +1
    u_m = beta * (-m + amp * gh.x)   # ξ = -1

    th_p = np.tanh(u_p)
    th_m = np.tanh(u_m)

    # ξ-average:
    # < ξ tanh(β(m ξ + ...)) >_ξ = 0.5*(tanh(u_p) - tanh(u_m))
    m_new = 0.5 * expect(gh, th_p - th_m)
    # < tanh^2(...) >_ξ = 0.5*(tanh^2(u_p) + tanh^2(u_m))
    q_new = 0.5 * expect(gh, th_p * th_p + th_m * th_m)

    if branch == "sg":
        m_new = 0.0
    else:
        m_new = abs(m_new)

    if not (np.isfinite(m_new) and np.isfinite(q_new)):
        return None

    q_new = float(np.clip(q_new, 0.0, 1.0))
    r = q_new / (denom ** 2)
    return float(m_new), float(q_new), float(r)






# --------------------------
# RS free energies (Eq. 2.9 + stable rearrangements)
# --------------------------

def free_energy_t_gt_0(st: RSState, *, alpha: float, beta: float, t: float, gh: GaussHermite, denom_min: float) -> Optional[float]:
    m, q, Q, p, Delta = st.m, st.q, st.Q, st.p, st.Delta

    denom = 1.0 - beta * (1.0 + t) * (Q - q)
    if not np.isfinite(denom) or denom <= denom_min:
        return None
    if not (np.isfinite(Delta) and Delta > 0.0 and np.isfinite(p) and p >= 0.0):
        return None

    # E[log cosh( (β/Δ)(m + sqrt(αp)x) )]
    s = math.sqrt(max(alpha * p, 0.0))
    u = (beta / Delta) * (m + s * gh.x)
    I_logcosh = expect(gh, logcosh_stable(u))

    # Term with 1/t cancellation, written finite:
    # (1+t)(Δ-1)/(2t) Q + (1+t)(1-Δ)/(2tΔ)
    # = (1+t)/2 * ((Δ-1)/t) * (Q - 1/Δ)
    dm = Delta - 1.0
    cancel_term = 0.5 * (1.0 + t) * (dm / t) * (Q - 1.0 / Delta)

    A = (
        (m * m) / (2.0 * (1.0 + t)) * (1.0 + t / Delta)
        + cancel_term
        + 0.5 * alpha * beta * p * (Q - q)
        + (alpha / (2.0 * beta)) * math.log(denom)
        - 0.5 * alpha * q * (1.0 + t) / denom
        + (0.5 / beta) * math.log(Delta)
        + 0.5 * alpha * p * t / ((1.0 + t) * Delta)
        - (1.0 / beta) * I_logcosh
        - (_LOG2 / beta)
    )
    if not np.isfinite(A):
        return None
    return float(A)


def free_energy_t_eq_0(
    m: float, q: float, r_unused: float, *, alpha: float, beta: float,
    gh: GaussHermite, denom_min: float
) -> Optional[float]:
    """
    Hopfield RS free energy: paper Eq. (1.7), with explicit ξ-average.
    Note: any additive constant cancels in ΔA, but we keep -log2/beta for consistency.
    """
    denom = 1.0 - beta * (1.0 - q)
    if not np.isfinite(denom) or denom <= denom_min:
        return None

    amp = math.sqrt(max(alpha * q, 0.0)) / denom
    u_p = beta * ( m + amp * gh.x)
    u_m = beta * (-m + amp * gh.x)

    I_logcosh = expect(gh, 0.5 * (logcosh_stable(u_p) + logcosh_stable(u_m)))

    A = (
        0.5 * m * m
        - (1.0 / beta) * I_logcosh
        + (alpha / (2.0 * beta)) * math.log(denom)
        - 0.5 * alpha * q / denom
        + 0.5 * alpha * q / (denom ** 2)
        - (_LOG2 / beta)
    )
    if not np.isfinite(A):
        return None
    return float(A)


# --------------------------
# Solver: damped fixed point with adaptive damping
# --------------------------

@dataclass
class SolveResult:
    ok: bool
    state: Optional[RSState] = None
    m: Optional[float] = None
    q: Optional[float] = None
    p: Optional[float] = None


def solve_rs(
    *, alpha: float, T: float, t: float, branch: str, gh: GaussHermite,
    x0: Optional[RSState] = None, m0: float = 0.9,
    max_iter: int = 400, tol: float = 1e-10,
    denom_min: float = 1e-10
) -> SolveResult:
    beta = 1.0 / T

    if t == 0.0:
        # Hopfield reduced system with Q=1, Delta=1
        m = 0.0 if branch == "sg" else m0
        q = 0.2 if branch == "sg" else 0.95
        omega = 0.4
        prev_err = np.inf

        for _ in range(max_iter):
            out = rs_map_t_eq_0(m, q, alpha=alpha, beta=beta, gh=gh, branch=branch, denom_min=denom_min)
            if out is None:
                omega *= 0.5
                if omega < 1e-4:
                    return SolveResult(ok=False)
                continue
            m_new, q_new, p_new = out
            dm = m_new - m
            dq = q_new - q
            err = max(abs(dm), abs(dq))

            # adaptive damping
            if err > prev_err * 1.05:
                omega *= 0.5
            else:
                omega = min(0.8, omega * 1.05)

            m = (1.0 - omega) * m + omega * m_new
            q = (1.0 - omega) * q + omega * q_new

            # mild physical clamps
            q = float(np.clip(q, 0.0, 1.0))
            if branch != "sg":
                m = float(max(m, 0.0))

            if err < tol:
                return SolveResult(ok=True, m=m, q=q, p=p_new)
            prev_err = err

        return SolveResult(ok=False)

    # t > 0: full 5D system
    if x0 is None:
        q0 = 0.2 if branch == "sg" else 0.95
        x = RSState(
            m=0.0 if branch == "sg" else m0,
            q=q0,
            Q=q0,          # crucial: start with Q≈q so D>0
            p=0.0,         # overwritten by the map anyway
            Delta=1.0 + alpha * t,
        )
    else:
        x = RSState.from_array(x0.as_array().copy())
        if branch == "sg":
            x.m = 0.0
        else:
            x.m = abs(x.m)

    omega = 0.35
    prev_err = np.inf

    for _ in range(max_iter):
        y = rs_map_t_gt_0(x, alpha=alpha, beta=beta, t=t, gh=gh, branch=branch, denom_min=denom_min)
        if y is None:
            omega *= 0.5
            if omega < 1e-4:
                return SolveResult(ok=False)
            continue

        dx = y.as_array() - x.as_array()
        err = float(np.max(np.abs(dx)))

        if err > prev_err * 1.05:
            omega *= 0.5
        else:
            omega = min(0.85, omega * 1.05)

        x_arr = (1.0 - omega) * x.as_array() + omega * y.as_array()
        x = RSState.from_array(x_arr)

        # mild clamps (avoid wild excursions without “projecting” too hard)
        x.q = float(np.clip(x.q, 0.0, 1.2))
        x.Q = float(np.clip(x.Q, -0.2, 2.0))
        x.p = float(max(x.p, 0.0))
        x.Delta = float(max(x.Delta, 1e-8))
        if branch == "sg":
            x.m = 0.0
        else:
            x.m = float(max(abs(x.m), 0.0))

        if err < tol:
            return SolveResult(ok=True, state=x)
        prev_err = err

    return SolveResult(ok=False)


# --------------------------
# Phase line extraction
# --------------------------

def compute_lines_for_t(
    *, t: float, alpha_grid: np.ndarray, T_grid: np.ndarray, gh: GaussHermite,
    m_min: float = 1e-3, denom_min: float = 1e-10,
    refine_bisect: int = 18
) -> Dict[str, np.ndarray]:
    """
    Returns dict with arrays:
      alpha, Tc, TR  (NaNs where not found)
    """
    Tc = np.full_like(alpha_grid, np.nan, dtype=float)
    TR = np.full_like(alpha_grid, np.nan, dtype=float)

    # warm starts across alpha (use last alpha's low-T solution as seed)
    seed_retr: Optional[RSState] = None
    seed_sg: Optional[RSState] = None

    for ia, alpha in enumerate(alpha_grid):
        # --- sweep SG along T
        sg_solutions: Dict[float, object] = {}
        x0_sg = seed_sg
        for T in T_grid:
            res = solve_rs(alpha=alpha, T=T, t=t, branch="sg", gh=gh, x0=x0_sg, denom_min=denom_min)
            if not res.ok:
                sg_solutions[T] = None
                x0_sg = None
            else:
                if t == 0.0:
                    sg_solutions[T] = (res.m, res.q, 1.0, res.p, 1.0)
                else:
                    sg_solutions[T] = res.state
                    x0_sg = res.state

        # --- sweep retrieval along T (continuation from low T upward)
        retr_solutions: Dict[float, object] = {}
        x0_retr = seed_retr
        for T in T_grid:
            res = solve_rs(alpha=alpha, T=T, t=t, branch="retr", gh=gh, x0=x0_retr, denom_min=denom_min)
            if not res.ok:
                retr_solutions[T] = None
                x0_retr = None
            else:
                if t == 0.0:
                    retr_solutions[T] = (res.m, res.q, 1.0, res.p, 1.0)
                else:
                    retr_solutions[T] = res.state
                    x0_retr = res.state

        # store seeds for next alpha (use lowest-T successful states)
        if t > 0.0:
            for T in T_grid[:5]:
                if retr_solutions[T] is not None:
                    seed_retr = retr_solutions[T]
                    break
            for T in T_grid[:5]:
                if sg_solutions[T] is not None:
                    seed_sg = sg_solutions[T]
                    break

        # --- Tc(alpha): last T where retrieval exists with m>m_min under continuation
        good_Ts = []
        for T in T_grid:
            sol = retr_solutions[T]
            if sol is None:
                break
            if t == 0.0:
                m = sol[0]
            else:
                m = sol.m
            if m is None or (m <= m_min):
                break
            good_Ts.append(T)

        if len(good_Ts) > 0:
            Tc[ia] = good_Ts[-1]

        # --- TR(alpha): free-energy crossing where both branches exist (below Tc)
        # Compute ΔA(T)=A_retr-A_sg on points where both exist.
        Ts_common = []
        dA = []
        for T in T_grid:
            if not np.isfinite(Tc[ia]) or T > Tc[ia] + 1e-15:
                break
            s_retr = retr_solutions[T]
            s_sg = sg_solutions[T]
            if s_retr is None or s_sg is None:
                continue

            beta = 1.0 / T
            if t == 0.0:
                mR, qR, _, pR, _ = s_retr
                mS, qS, _, pS, _ = s_sg
                AR = free_energy_t_eq_0(mR, qR, pR, alpha=alpha, beta=beta, gh=gh, denom_min=denom_min)
                AS = free_energy_t_eq_0(mS, qS, pS, alpha=alpha, beta=beta, gh=gh, denom_min=denom_min)
            else:
                AR = free_energy_t_gt_0(s_retr, alpha=alpha, beta=beta, t=t, gh=gh, denom_min=denom_min)
                AS = free_energy_t_gt_0(s_sg,   alpha=alpha, beta=beta, t=t, gh=gh, denom_min=denom_min)

            if AR is None or AS is None:
                continue
            Ts_common.append(T)
            dA.append(AR - AS)

        if len(Ts_common) >= 2:
            Ts_common = np.array(Ts_common, float)
            dA = np.array(dA, float)

            # look for a sign change (retrieval lower -> SG lower typically means dA crosses 0)
            idx = None
            for k in range(len(dA) - 1):
                if dA[k] == 0.0:
                    idx = k
                    break
                if dA[k] * dA[k + 1] < 0.0:
                    idx = k
                    break

            if idx is not None:
                T_lo, T_hi = float(Ts_common[idx]), float(Ts_common[min(idx + 1, len(Ts_common) - 1)])
                if dA[idx] == 0.0:
                    TR[ia] = T_lo
                else:
                    # bisection refinement: re-solve both branches at midpoints
                    d_lo = float(dA[idx])
                    d_hi = float(dA[idx + 1])

                    # seeds from endpoints
                    if t == 0.0:
                        # for t=0 we just use endpoint m,q as initial guesses via m0 in solve_rs
                        pass
                    else:
                        seedR_lo: RSState = retr_solutions[T_lo]
                        seedR_hi: RSState = retr_solutions[T_hi]
                        seedS_lo: RSState = sg_solutions[T_lo]
                        seedS_hi: RSState = sg_solutions[T_hi]

                    for _ in range(refine_bisect):
                        T_mid = 0.5 * (T_lo + T_hi)
                        beta_mid = 1.0 / T_mid

                        if t == 0.0:
                            # warm start: use endpoint m as m0, q as implicit via internal init
                            # (kept simple; robust enough with damping)
                            resR = solve_rs(alpha=alpha, T=T_mid, t=0.0, branch="retr", gh=gh, m0=0.9, denom_min=denom_min)
                            resS = solve_rs(alpha=alpha, T=T_mid, t=0.0, branch="sg",   gh=gh, m0=0.0, denom_min=denom_min)
                            if not (resR.ok and resS.ok):
                                break
                            AR = free_energy_t_eq_0(resR.m, resR.q, resR.p, alpha=alpha, beta=beta_mid, gh=gh, denom_min=denom_min)
                            AS = free_energy_t_eq_0(resS.m, resS.q, resS.p, alpha=alpha, beta=beta_mid, gh=gh, denom_min=denom_min)
                        else:
                            # choose closer endpoint as initial guess
                            x0R = seedR_lo if (T_mid - T_lo) < (T_hi - T_mid) else seedR_hi
                            x0S = seedS_lo if (T_mid - T_lo) < (T_hi - T_mid) else seedS_hi

                            resR = solve_rs(alpha=alpha, T=T_mid, t=t, branch="retr", gh=gh, x0=x0R, denom_min=denom_min)
                            resS = solve_rs(alpha=alpha, T=T_mid, t=t, branch="sg",   gh=gh, x0=x0S, denom_min=denom_min)
                            if not (resR.ok and resS.ok):
                                break
                            AR = free_energy_t_gt_0(resR.state, alpha=alpha, beta=beta_mid, t=t, gh=gh, denom_min=denom_min)
                            AS = free_energy_t_gt_0(resS.state, alpha=alpha, beta=beta_mid, t=t, gh=gh, denom_min=denom_min)

                        if AR is None or AS is None:
                            break

                        d_mid = float(AR - AS)
                        if d_lo * d_mid <= 0.0:
                            T_hi, d_hi = T_mid, d_mid
                            if t > 0.0:
                                seedR_hi = resR.state
                                seedS_hi = resS.state
                        else:
                            T_lo, d_lo = T_mid, d_mid
                            if t > 0.0:
                                seedR_lo = resR.state
                                seedS_lo = resS.state

                    TR[ia] = 0.5 * (T_lo + T_hi)

    return {"alpha": alpha_grid, "Tc": Tc, "TR": TR}


# --------------------------
# CLI / main
# --------------------------

def default_grids_for_t(t: float) -> Tuple[np.ndarray, np.ndarray]:
    # Nonuniform T grid (dense at low T, still covers up to ~1.1)
    T_grid = np.concatenate([
        np.linspace(0.02, 0.20, 40, endpoint=False),
        np.linspace(0.20, 1.10, 70),
    ])

    if t == 0.0:
        alpha_max = 0.14
        alpha_grid = np.concatenate([
            np.linspace(0.0, 0.10, 60, endpoint=False),
            np.linspace(0.10, alpha_max, 80),
        ])
    elif t <= 0.15:
        alpha_max = 0.25
        alpha_grid = np.concatenate([
            np.linspace(0.0, 0.15, 70, endpoint=False),
            np.linspace(0.15, alpha_max, 70),
        ])
    else:
        alpha_max = 0.60
        alpha_grid = np.concatenate([
            np.linspace(0.0, 0.30, 80, endpoint=False),
            np.linspace(0.30, alpha_max, 80),
        ])
    return alpha_grid, T_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, nargs="+", default=[0.0, 0.1, 0.5], help="t values")
    ap.add_argument("--gh", type=int, default=80, help="Gauss–Hermite points")
    ap.add_argument("--out", type=str, default="phase_lines", help="output prefix")
    ap.add_argument("--plot", action="store_true", help="plot lines (requires matplotlib)")
    args = ap.parse_args()

    gh = GaussHermite.make(args.gh)

    all_results = {}
    for t in args.t:
        alpha_grid, T_grid = default_grids_for_t(t)
        res = compute_lines_for_t(t=t, alpha_grid=alpha_grid, T_grid=T_grid, gh=gh)
        all_results[t] = res

        fn = f"{args.out}_t{t:.3f}.csv"
        data = np.column_stack([res["alpha"], res["TR"], res["Tc"]])
        header = "alpha,TR,Tc"
        np.savetxt(fn, data, delimiter=",", header=header, comments="")
        print(f"[ok] wrote {fn}")

    if args.plot:
        import matplotlib.pyplot as plt
        for t, res in all_results.items():
            a, TR, Tc = res["alpha"], res["TR"], res["Tc"]
            plt.figure()
            plt.plot(a, Tc, label="Tc (M–SG)")
            plt.plot(a, TR, label="TR (F–M)")
            plt.ylim(0.0, 1.12)
            plt.xlim(0.0, float(np.nanmax(a)))
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$T$")
            plt.title(f"RS phase lines, t={t}")
            plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
