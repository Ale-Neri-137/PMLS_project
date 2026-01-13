# ptflow.py  (drop-in replacement)
#
# Goals vs your original:
# 1) Make "macro" analysis (stride = period) sane by default (phase-averaged offsets).
# 2) Make "flow/current" computations theoretically consistent even when P is nonreversible
#    (use backward committor q^- via time-reversed kernel, not 1-q).
# 3) Add a "micro" (stride=1) phase-aware analysis via a lifted chain on (node r, phase φ),
#    so edge-level plots on your *primitive* swap edges are meaningful.
# 4) Add sanity checks: stationarity, detailed balance residuals, one-way fraction, CK test.
#
# Minimal API changes:
# - analyze_one_perm2 / analyze_results_list / analyze_I_stack accept:
#       level="macro" (default) or level="micro"
#       period=3 (default)
#       average_offsets=True (default, affects macro when stride>1)
# - Existing plot_* still work for macro outputs.
# - New micro plot helpers are prefixed plot_micro_* and expect the dict returned by level="micro".

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Iterable, Union, Literal


# ============================================================
# Meta + endpoints
# ============================================================

@dataclass(frozen=True)
class PTMeta:
    beta: np.ndarray      # (R,)
    t_grid: np.ndarray    # (B,)
    k_start: np.ndarray   # (B+1,)
    b_of_r: np.ndarray    # (R,)
    t_of_r: np.ndarray    # (R,)
    R: int
    B: int
    # Optional primitive vertical edge list (E,2) in node indices r.
    vertical_edges: Optional[np.ndarray] = None

def meta_from_sys(sys) -> PTMeta:
    beta   = np.asarray(sys.beta, dtype=np.float64)
    t_grid = np.asarray(getattr(sys, "t_grid", [0.0]), dtype=np.float64)
    k_start = np.asarray(getattr(sys, "k_start", [0, beta.size]), dtype=np.int64)
    b_of_r = np.asarray(getattr(sys, "b_of_r", np.zeros(beta.size, dtype=np.int64)), dtype=np.int64)

    R = int(beta.size)
    B = int(k_start.size - 1)
    t_of_r = t_grid[b_of_r] if t_grid.size > 0 else np.zeros(R, dtype=np.float64)

    # Try to discover a vertical edge list from sys if you have one.
    vE = None
    for name in ("vertical_edges", "edge_list", "edges_vertical", "pt_vertical_edges"):
        if hasattr(sys, name):
            X = getattr(sys, name)
            if X is None:
                continue
            X = np.asarray(X, dtype=np.int64)
            if X.ndim == 2 and X.shape[1] == 2:
                vE = X
                break

    return PTMeta(beta=beta, t_grid=t_grid, k_start=k_start, b_of_r=b_of_r,
                  t_of_r=t_of_r, R=R, B=B, vertical_edges=vE)

def meta_1d(beta: np.ndarray) -> PTMeta:
    beta = np.asarray(beta, dtype=np.float64)
    R = int(beta.size)
    t_grid = np.asarray([0.0], dtype=np.float64)
    k_start = np.asarray([0, R], dtype=np.int64)
    b_of_r = np.zeros(R, dtype=np.int64)
    t_of_r = np.zeros(R, dtype=np.float64)
    return PTMeta(beta=beta, t_grid=t_grid, k_start=k_start, b_of_r=b_of_r,
                  t_of_r=t_of_r, R=R, B=1, vertical_edges=None)

@dataclass(frozen=True)
class Ends:
    hot_idx: np.ndarray
    cold_idx: np.ndarray
    b0: int
    row0_boxes: np.ndarray

def ends_row0(meta: PTMeta, *, b0: int = 0, hot_k: int = 0, cold_top: int = 1) -> Ends:
    r0, r1 = int(meta.k_start[b0]), int(meta.k_start[b0 + 1])
    K0 = r1 - r0
    if not (0 <= hot_k < K0):
        raise ValueError("hot_k out of range")
    cold_top = int(max(1, cold_top))
    cold_ks = np.arange(max(0, K0 - cold_top), K0, dtype=np.int64)

    hot_idx = np.asarray([r0 + hot_k], dtype=np.int64)
    cold_idx = (r0 + cold_ks).astype(np.int64)
    if np.intersect1d(hot_idx, cold_idx).size:
        raise ValueError("hot and cold sets overlap (adjust hot_k/cold_top).")

    row0 = np.arange(r0, r1, dtype=np.int64)
    return Ends(hot_idx=hot_idx, cold_idx=cold_idx, b0=b0, row0_boxes=row0)

def ends_linear(R: int, *, hot_k: int = 0, cold_top: int = 1) -> Ends:
    R = int(R)
    cold_top = int(max(1, cold_top))
    hot_idx = np.asarray([hot_k], dtype=np.int64)
    cold_idx = np.arange(max(0, R - cold_top), R, dtype=np.int64)
    if np.intersect1d(hot_idx, cold_idx).size:
        raise ValueError("hot and cold sets overlap (adjust hot_k/cold_top).")
    row0 = np.arange(0, R, dtype=np.int64)
    return Ends(hot_idx=hot_idx, cold_idx=cold_idx, b0=0, row0_boxes=row0)

def node_to_bk(meta: PTMeta, r: int) -> Tuple[int, int]:
    b = int(meta.b_of_r[r])
    k = int(r - meta.k_start[b])
    return b, k


# ============================================================
# Permutation access
# ============================================================

def get_perm_ts(res) -> np.ndarray:
    """
    Returns permutation series with shape (2, T, R): box -> label.
    """
    for name in ("I_ts", "Ψ_ts", "Psi_ts", "psi_ts"):
        if hasattr(res, name):
            X = getattr(res, name)
            if X is None:
                continue
            X = np.asarray(X)
            if X.ndim == 3 and X.shape[0] == 2:
                return X.astype(np.int64, copy=False)
    raise AttributeError("No (2,T,R) permutation series found (I_ts/Ψ_ts/...).")

def _burn_idx(T: int, burn_in: Union[int, float]) -> int:
    if isinstance(burn_in, float):
        if not (0 <= burn_in < 1):
            raise ValueError("burn_in float must be in [0,1).")
        return int(np.floor(burn_in * T))
    return int(burn_in)


# ============================================================
# Transition kernel estimation
# ============================================================

def transition_counts_from_perm(
    I_TR: np.ndarray,
    *,
    burn_in: Union[int, float] = 0.1,
    stride: int = 1,
    offset: int = 0,
    label_subsample: Optional[np.ndarray] = None,
    chunk: int = 4000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    I_TR: (T, R) box->label permutations at each tick.

    Returns:
      C: (R,R) transition counts on node index (box) for label walks
      visits: (R,) counts of node occupancy at the 'from' time

    Important:
      - stride=1 gives micro-step transitions (time-inhomogeneous if schedule is periodic)
      - stride=period with offset fixed gives stroboscopic macro transitions
      - averaging offsets 0..stride-1 is usually the right thing for a macro kernel.
    """
    I = np.asarray(I_TR, dtype=np.int64)
    T, R = I.shape
    stride = int(max(1, stride))
    offset = int(offset % stride)

    t0 = max(0, min(_burn_idx(T, burn_in), T - 1))
    t0 = t0 + offset
    t_end = T - stride
    if t0 >= t_end:
        return np.zeros((R, R), dtype=np.int64), np.zeros(R, dtype=np.int64)

    labels = np.arange(R, dtype=np.int64) if label_subsample is None else np.asarray(label_subsample, dtype=np.int64)

    C = np.zeros((R, R), dtype=np.int64)
    visits = np.zeros(R, dtype=np.int64)

    idxs = np.arange(t0, t_end, stride, dtype=np.int64)
    n = idxs.size
    for s in range(0, n, int(max(1, chunk))):
        e = min(n, s + int(max(1, chunk)))
        sel = idxs[s:e]
        IA = I[sel]           # (M,R)
        IB = I[sel + stride]  # (M,R)

        # inverse permutations: posA[m, label] = box
        posA = np.argsort(IA, axis=1)  # (M,R)
        posB = np.argsort(IB, axis=1)

        u = posA[:, labels].reshape(-1)
        v = posB[:, labels].reshape(-1)

        np.add.at(visits, u, 1)
        np.add.at(C, (u, v), 1)

    return C, visits

def _P_pi_from_counts(C: np.ndarray, visits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    C = np.asarray(C, dtype=np.int64)
    visits = np.asarray(visits, dtype=np.int64)
    R = C.shape[0]

    pi = visits.astype(np.float64)
    s = pi.sum()
    pi = (pi / s) if s > 0 else np.full(R, 1.0 / float(R), dtype=np.float64)

    row = C.sum(axis=1, keepdims=True).astype(np.float64)
    P = C.astype(np.float64) / np.maximum(1.0, row)
    z = (row[:, 0] == 0)
    if np.any(z):
        P[z, :] = 0.0
        P[z, z] = 1.0
    return P, pi

def estimate_P_pi_from_perm2(
    I2: np.ndarray,
    *,
    burn_in: Union[int, float] = 0.1,
    stride: int = 1,
    average_offsets: bool = True,
    label_subsample: Optional[np.ndarray] = None,
    chunk: int = 4000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    I2: (2,T,R) box->label perms for 2 chains.

    Returns:
      P: (R,R) transition matrix on nodes
      pi: (R,) stationary estimate from visits
      C: (R,R) counts

    Note:
      For stride>1, average_offsets=True pools offsets 0..stride-1 to avoid phase-strobing artifacts.
    """
    I2 = np.asarray(I2, dtype=np.int64)
    assert I2.ndim == 3 and I2.shape[0] == 2
    _, T, R = I2.shape
    stride = int(max(1, stride))
    average_offsets = bool(average_offsets)

    C = np.zeros((R, R), dtype=np.int64)
    visits = np.zeros(R, dtype=np.int64)

    offsets = range(stride) if (average_offsets and stride > 1) else (0,)
    for off in offsets:
        C0, v0 = transition_counts_from_perm(I2[0], burn_in=burn_in, stride=stride, offset=off,
                                             label_subsample=label_subsample, chunk=chunk)
        C1, v1 = transition_counts_from_perm(I2[1], burn_in=burn_in, stride=stride, offset=off,
                                             label_subsample=label_subsample, chunk=chunk)
        C += (C0 + C1)
        visits += (v0 + v1)

    P, pi = _P_pi_from_counts(C, visits)
    return P, pi, C


# ============================================================
# Lifted (phase-aware) micro-step chain: state = (phi, node)
# ============================================================

def estimate_lifted_P_pi_from_perm2(
    I2: np.ndarray,
    *,
    period: int = 3,
    burn_in: Union[int, float] = 0.1,
    label_subsample: Optional[np.ndarray] = None,
    chunk: int = 4000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a time-homogeneous Markov chain on lifted states (phi, r), phi in {0..period-1}.
    Uses micro-step transitions t -> t+1 and assigns phi = t mod period.

    Returns:
      Pℓ : (period*R, period*R)
      piℓ: (period*R,)
      Cℓ : counts
    """
    I2 = np.asarray(I2, dtype=np.int64)
    assert I2.ndim == 3 and I2.shape[0] == 2
    _, T, R = I2.shape
    period = int(max(1, period))

    t0 = max(0, min(_burn_idx(T, burn_in), T - 2))
    t_end = T - 1
    if t0 >= t_end:
        L = period * R
        C = np.zeros((L, L), dtype=np.int64)
        visits = np.zeros(L, dtype=np.int64)
        P, pi = _P_pi_from_counts(C, visits)
        return P, pi, C

    labels = np.arange(R, dtype=np.int64) if label_subsample is None else np.asarray(label_subsample, dtype=np.int64)

    L = period * R
    C = np.zeros((L, L), dtype=np.int64)
    visits = np.zeros(L, dtype=np.int64)

    # iterate times in chunks, for both chains
    idxs = np.arange(t0, t_end, 1, dtype=np.int64)
    n = idxs.size

    for chain in (0, 1):
        I = I2[chain]
        for s in range(0, n, int(max(1, chunk))):
            e = min(n, s + int(max(1, chunk)))
            sel = idxs[s:e]
            IA = I[sel]
            IB = I[sel + 1]
            posA = np.argsort(IA, axis=1)  # (M,R)
            posB = np.argsort(IB, axis=1)

            u = posA[:, labels].reshape(-1)
            v = posB[:, labels].reshape(-1)

            # phases for each row in this chunk
            ph = (sel % period).astype(np.int64)          # (M,)
            ph_next = ((ph + 1) % period).astype(np.int64)

            # expand phases to match flattened labels
            M = sel.size
            ph_rep = np.repeat(ph, labels.size)
            phn_rep = np.repeat(ph_next, labels.size)

            from_idx = ph_rep * R + u
            to_idx   = phn_rep * R + v

            np.add.at(visits, from_idx, 1)
            np.add.at(C, (from_idx, to_idx), 1)

    P, pi = _P_pi_from_counts(C, visits)
    return P, pi, C

def lift_indices_for_set(R: int, period: int, nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.int64)
    period = int(period)
    out = []
    for ph in range(period):
        out.append(ph * R + nodes)
    return np.concatenate(out).astype(np.int64)


# ============================================================
# TPT: committors + reactive flux/current (nonreversible-safe)
# ============================================================

def forward_committor_to_cold(P: np.ndarray, hot_idx: np.ndarray, cold_idx: np.ndarray) -> np.ndarray:
    """
    q+[r] = Prob(hit cold before hot | start r).
    Boundary: q+=0 on hot, q+=1 on cold.
    Solve (I - P_ff) q_f = P_fC * 1.
    """
    P = np.asarray(P, dtype=np.float64)
    R = P.shape[0]
    hot = np.zeros(R, dtype=bool);  hot[np.asarray(hot_idx, dtype=np.int64)] = True
    cold = np.zeros(R, dtype=bool); cold[np.asarray(cold_idx, dtype=np.int64)] = True
    free = ~(hot | cold)

    q = np.zeros(R, dtype=np.float64)
    q[hot] = 0.0
    q[cold] = 1.0

    free_idx = np.flatnonzero(free)
    if free_idx.size == 0:
        return q

    P_ff = P[np.ix_(free, free)]
    rhs = P[np.ix_(free, cold)].sum(axis=1)
    A = np.eye(free_idx.size, dtype=np.float64) - P_ff

    try:
        q_free = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        q_free = np.linalg.lstsq(A, rhs, rcond=None)[0]

    q[free_idx] = q_free
    return np.clip(q, 0.0, 1.0)

def time_reversed_kernel(P: np.ndarray, pi: np.ndarray, *, eps: float = 1e-300) -> np.ndarray:
    """
    P*_{uv} = (pi_v P_{vu}) / pi_u  (on support where pi_u>0)
    """
    P = np.asarray(P, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    R = P.shape[0]
    Pstar = np.zeros_like(P)
    for u in range(R):
        if pi[u] <= eps:
            Pstar[u, u] = 1.0
            continue
        Pstar[u, :] = (pi * P[:, u]) / pi[u]
    # numerical row-normalize
    row = Pstar.sum(axis=1, keepdims=True)
    z = (row[:, 0] == 0)
    if np.any(z):
        Pstar[z, :] = 0.0
        Pstar[z, z] = 1.0
    else:
        Pstar = Pstar / row
    return Pstar

def backward_committor_to_hot(P: np.ndarray, pi: np.ndarray, hot_idx: np.ndarray, cold_idx: np.ndarray) -> np.ndarray:
    """
    q-[r] = Prob(last hit hot more recently than cold | at r),
    computed as forward committor on time-reversed chain:
      boundary: q-=1 on hot, q-=0 on cold
      solve on P*.
    """
    Pstar = time_reversed_kernel(P, pi)
    # forward committor to hot before cold on P*
    # i.e. boundary 1 on hot, 0 on cold
    P = np.asarray(Pstar, dtype=np.float64)
    R = P.shape[0]
    hot = np.zeros(R, dtype=bool);  hot[np.asarray(hot_idx, dtype=np.int64)] = True
    cold = np.zeros(R, dtype=bool); cold[np.asarray(cold_idx, dtype=np.int64)] = True
    free = ~(hot | cold)

    q = np.zeros(R, dtype=np.float64)
    q[hot] = 1.0
    q[cold] = 0.0

    free_idx = np.flatnonzero(free)
    if free_idx.size == 0:
        return q

    P_ff = P[np.ix_(free, free)]
    rhs = P[np.ix_(free, hot)].sum(axis=1)  # target=hot with value 1
    A = np.eye(free_idx.size, dtype=np.float64) - P_ff

    try:
        q_free = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        q_free = np.linalg.lstsq(A, rhs, rcond=None)[0]

    q[free_idx] = q_free
    return np.clip(q, 0.0, 1.0)

def reactive_flux_and_current(
    P: np.ndarray, pi: np.ndarray,
    q_plus: np.ndarray, q_minus: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    General (nonreversible) TPT:
      f_uv = pi_u P_uv q^-_u q^+_v
      J_uv = f_uv - f_vu
    Reaction rate k_AB is sum_{u in A} sum_v f_uv if q^-|A=1, q^+|A=0, q^+|B=1, q^-|B=0.
    """
    P = np.asarray(P, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    qp = np.asarray(q_plus, dtype=np.float64)
    qm = np.asarray(q_minus, dtype=np.float64)

    f = (pi[:, None] * P) * (qm[:, None]) * (qp[None, :])
    J = f - f.T
    # k_AB = sum_{u in A} sum_v f_uv, but caller can compute using hot_idx;
    # here we return total reactive flux out of q^+=0 states if desired; keep scalar as sum over all u,v:
    # (for well-posed boundaries this equals k_AB)
    k = float(f.sum())
    return k, f, J

def tpt_on_sets(
    P: np.ndarray, pi: np.ndarray,
    hot_idx: np.ndarray, cold_idx: np.ndarray
) -> Dict[str, Any]:
    q_plus = forward_committor_to_cold(P, hot_idx, cold_idx)
    q_minus = backward_committor_to_hot(P, pi, hot_idx, cold_idx)
    k, f, J = reactive_flux_and_current(P, pi, q_plus, q_minus)
    # reaction rate out of hot (more standard scalar)
    hot = np.asarray(hot_idx, dtype=np.int64)
    k_hot = float(f[hot, :].sum())
    return dict(q_plus=q_plus, q_minus=q_minus, k=k_hot, f=f, J=J)


# ============================================================
# Diagnostics: stationarity, DB, one-way, CK test
# ============================================================

def stationarity_tv(P: np.ndarray, pi: np.ndarray) -> float:
    """0.5 * ||piP - pi||_1 (TV distance between pi and piP)."""
    P = np.asarray(P, float)
    pi = np.asarray(pi, float)
    piP = pi @ P
    return float(0.5 * np.sum(np.abs(piP - pi)))

def detailed_balance_report(P, pi, C=None, *, eps=1e-300,
                           quantiles=(0.5, 0.9, 0.99),
                           thresholds=(1e-2, 5e-2, 1e-1, 2e-1)) -> Dict[str, Any]:
    """
    Edge-restricted detailed balance residual:
      rel_uv = |pi_u P_uv - pi_v P_vu| / (pi_u P_uv + pi_v P_vu + eps) in [0,1]
    """
    P = np.asarray(P, float)
    pi = np.asarray(pi, float)
    flow = pi[:, None] * P
    delta = flow - flow.T
    sym = flow + flow.T

    if C is not None:
        C = np.asarray(C)
        obs = (C + C.T) > 0
    else:
        obs = sym > 0
    np.fill_diagonal(obs, False)
    if not np.any(obs):
        return dict(n_edges=0, rel_quantiles={}, frac_rel_gt={}, stat_tv=stationarity_tv(P, pi))

    u, v = np.where(obs)
    rel = np.abs(delta[u, v]) / np.maximum(sym[u, v], eps)

    rel_q = {q: float(np.quantile(rel, q)) for q in quantiles}
    frac_gt = {thr: float(np.mean(rel > thr)) for thr in thresholds}

    return dict(
        n_edges=int(rel.size),
        rel_quantiles=rel_q,
        frac_rel_gt=frac_gt,
        stat_tv=float(stationarity_tv(P, pi))
    )

def one_way_fraction(P: np.ndarray, pi: np.ndarray, C: Optional[np.ndarray] = None) -> float:
    """
    Fraction of observed *directed* edges where one direction has zero flow.
    """
    P = np.asarray(P, float)
    pi = np.asarray(pi, float)
    flow = pi[:, None] * P
    if C is not None:
        C = np.asarray(C)
        obs = (C + C.T) > 0
    else:
        obs = flow > 0
    np.fill_diagonal(obs, False)
    u, v = np.where(obs)
    if u.size == 0:
        return np.nan
    return float(np.mean((flow[u, v] == 0.0) | (flow[v, u] == 0.0)))

def ck_test_markovianity(I2: np.ndarray, *, burn_in=0.1, label_subsample=None, chunk=4000) -> Dict[str, float]:
    """
    Simple Chapman–Kolmogorov sanity check on the permutation-derived kernel:
    compare empirical P2 (stride=2) vs (P1@P1) in L1 over rows.
    This is only a heuristic (permutation chain is only approximately Markov).
    """
    P1, pi1, C1 = estimate_P_pi_from_perm2(I2, burn_in=burn_in, stride=1,
                                           average_offsets=False,
                                           label_subsample=label_subsample, chunk=chunk)
    P2, pi2, C2 = estimate_P_pi_from_perm2(I2, burn_in=burn_in, stride=2,
                                           average_offsets=True,
                                           label_subsample=label_subsample, chunk=chunk)
    P1sq = P1 @ P1
    row_l1 = np.sum(np.abs(P2 - P1sq), axis=1)
    return dict(
        ck_row_l1_median=float(np.median(row_l1)),
        ck_row_l1_max=float(np.max(row_l1)),
        stat_tv_P1=float(stationarity_tv(P1, pi1)),
        stat_tv_P2=float(stationarity_tv(P2, pi2)),
    )


# ============================================================
# Spectral gap proxy
# ============================================================

def spectral_gap(P: np.ndarray, pi: np.ndarray) -> Tuple[float, float]:
    """
    Proxy via reversible symmetrization on support of pi>0:
      S = D^{1/2} P D^{-1/2}, Ssym=(S+S^T)/2, eigvals(Ssym).
    """
    P = np.asarray(P, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    mask = pi > 0
    if mask.sum() < 2:
        return np.nan, np.nan

    Pm = P[np.ix_(mask, mask)]
    pim = pi[mask]
    sqrtpi = np.sqrt(pim)
    S = (sqrtpi[:, None] * Pm) / np.maximum(1e-300, sqrtpi[None, :])
    Ssym = 0.5 * (S + S.T)

    w = np.linalg.eigvalsh(Ssym)
    w = np.sort(w)
    lam2 = float(w[-2])
    gap = float(1.0 - lam2)
    return lam2, gap


# ============================================================
# Primitive edge lists (horizontal even/odd + vertical)
# ============================================================

def horizontal_edges_even(meta: PTMeta) -> np.ndarray:
    """Edges (u,v) for even-phase horizontal swaps: (k,k+1) with k even within each row."""
    edges = []
    for b in range(meta.B):
        r0, r1 = int(meta.k_start[b]), int(meta.k_start[b + 1])
        Kb = r1 - r0
        for k in range(0, Kb - 1, 2):
            edges.append((r0 + k, r0 + k + 1))
    return np.asarray(edges, dtype=np.int64) if edges else np.empty((0, 2), dtype=np.int64)

def horizontal_edges_odd(meta: PTMeta) -> np.ndarray:
    """Edges (u,v) for odd-phase horizontal swaps: (k,k+1) with k odd within each row."""
    edges = []
    for b in range(meta.B):
        r0, r1 = int(meta.k_start[b]), int(meta.k_start[b + 1])
        Kb = r1 - r0
        for k in range(1, Kb - 1, 2):
            edges.append((r0 + k, r0 + k + 1))
    return np.asarray(edges, dtype=np.int64) if edges else np.empty((0, 2), dtype=np.int64)

def canonical_undirected_edges(edge_list: np.ndarray) -> np.ndarray:
    E = np.asarray(edge_list, dtype=np.int64)
    if E.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    u = np.minimum(E[:, 0], E[:, 1])
    v = np.maximum(E[:, 0], E[:, 1])
    Ev = np.unique(np.stack([u, v], axis=1), axis=0)
    return Ev


# ============================================================
# Grid padding + plotting helpers (macro)
# ============================================================

def pad_by_rows(meta: PTMeta, x: np.ndarray, *, fill=np.nan) -> np.ndarray:
    x = np.asarray(x)
    Kmax = 0
    Ks = []
    for b in range(meta.B):
        r0, r1 = int(meta.k_start[b]), int(meta.k_start[b + 1])
        Ks.append(r1 - r0)
        Kmax = max(Kmax, r1 - r0)

    out = np.full((meta.B, Kmax), fill, dtype=np.float64)
    for b in range(meta.B):
        r0, r1 = int(meta.k_start[b]), int(meta.k_start[b + 1])
        out[b, : (r1 - r0)] = x[r0:r1]
    return out

def horizontal_current_matrix(meta: PTMeta, J: np.ndarray) -> np.ndarray:
    """
    Macro helper: returns (B, Kmax-1) entries J[r, r+1] within each row.
    Meaningful ONLY as "direct macro-transition current between adjacent boxes",
    not as "swap-edge current" unless you're in micro/phase-aware mode.
    """
    Kmax = 0
    for b in range(meta.B):
        r0, r1 = int(meta.k_start[b]), int(meta.k_start[b + 1])
        Kmax = max(Kmax, r1 - r0)
    out = np.full((meta.B, max(1, Kmax - 1)), np.nan, dtype=np.float64)
    for b in range(meta.B):
        r0, r1 = int(meta.k_start[b]), int(meta.k_start[b + 1])
        Kb = r1 - r0
        if Kb >= 2:
            for k in range(Kb - 1):
                out[b, k] = J[r0 + k, r0 + k + 1]
    return out

def plot_committor(meta: PTMeta, q: np.ndarray, *, title: str = "Committor q+ (to cold)") -> None:
    Q = pad_by_rows(meta, q)
    M = np.ma.masked_invalid(Q)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    im = ax.imshow(M, aspect="auto", interpolation="nearest")
    ax.set_xlabel("k (within row)")
    ax.set_ylabel("b (row index)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="q+")
    plt.tight_layout(); plt.show()

def plot_horizontal_current(meta: PTMeta, J: np.ndarray, *, absval: bool = True,
                           title: str = "Horizontal net current (macro adjacency)") -> None:
    H = horizontal_current_matrix(meta, J)
    H = np.abs(H) if absval else H
    M = np.ma.masked_invalid(H)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    im = ax.imshow(M, aspect="auto", interpolation="nearest")
    ax.set_xlabel("k edge (between k and k+1)")
    ax.set_ylabel("b (row index)")
    ax.set_title(title + (" |J|" if absval else " (signed)"))
    plt.colorbar(im, ax=ax, label="|J|" if absval else "J")
    plt.tight_layout(); plt.show()


# ============================================================
# Micro (lifted) plots on primitive edges
# ============================================================

def _lift_idx(period: int, R: int, ph: int, r: int) -> int:
    return int(ph) * int(R) + int(r)

def micro_edge_flux_on_undirected(
    f_lifted: np.ndarray, period: int, R: int, ph: int, edges_uv: np.ndarray
) -> np.ndarray:
    """
    For each undirected primitive edge (u,v), return reactive flux through that edge during phase ph:
      flux(u<->v) = f((ph,u)->(ph+1,v)) + f((ph,v)->(ph+1,u))
    """
    edges = canonical_undirected_edges(edges_uv)
    if edges.size == 0:
        return np.empty((0,), dtype=np.float64)
    phn = (ph + 1) % period
    u = edges[:, 0]; v = edges[:, 1]
    iu = ph * R + u
    iv = ph * R + v
    ju = phn * R + u
    jv = phn * R + v
    return f_lifted[iu, jv] + f_lifted[iv, ju]

def micro_edge_current_on_undirected(
    J_lifted: np.ndarray, period: int, R: int, ph: int, edges_uv: np.ndarray
) -> np.ndarray:
    """
    Same as above but with net current J on those forward lifted edges:
      cur(u<->v) = |J((ph,u),(ph+1,v))| + |J((ph,v),(ph+1,u))|
    Useful when you just want a "where is reactive transport happening" picture.
    """
    edges = canonical_undirected_edges(edges_uv)
    if edges.size == 0:
        return np.empty((0,), dtype=np.float64)
    phn = (ph + 1) % period
    u = edges[:, 0]; v = edges[:, 1]
    iu = ph * R + u
    iv = ph * R + v
    jv = phn * R + v
    ju = phn * R + u
    return np.abs(J_lifted[iu, jv]) + np.abs(J_lifted[iv, ju])

def micro_horizontal_heatmap(meta: PTMeta, vals_on_edges: np.ndarray, edges_uv: np.ndarray, *, fill=np.nan) -> np.ndarray:
    """
    Map per-edge values (aligned with canonical_undirected_edges(edges_uv)) into (B,Kmax-1) heatmap.
    Only horizontal edges are supported (v=u+1 within row).
    """
    edges = canonical_undirected_edges(edges_uv)
    if edges.size == 0:
        return np.full((meta.B, 1), np.nan, dtype=np.float64)

    # build dict u->val for edges (u,u+1)
    m = {}
    for (u, v), val in zip(edges, vals_on_edges):
        m[(int(u), int(v))] = float(val)

    Kmax = 0
    for b in range(meta.B):
        r0, r1 = int(meta.k_start[b]), int(meta.k_start[b + 1])
        Kmax = max(Kmax, r1 - r0)
    out = np.full((meta.B, max(1, Kmax - 1)), fill, dtype=np.float64)

    for b in range(meta.B):
        r0, r1 = int(meta.k_start[b]), int(meta.k_start[b + 1])
        Kb = r1 - r0
        for k in range(Kb - 1):
            u = r0 + k
            v = r0 + k + 1
            key = (u, v) if u < v else (v, u)
            if key in m:
                out[b, k] = m[key]
    return out

def plot_micro_horizontal(meta: PTMeta, f_lifted: np.ndarray, *, period: int = 3,
                          phase: Literal["even", "odd"] = "even",
                          kind: Literal["flux", "current"] = "flux",
                          title: Optional[str] = None) -> None:
    """
    Meaningful primitive-edge plot:
      - phase="even" uses horizontal even edges at ph=0
      - phase="odd"  uses horizontal odd  edges at ph=1
    kind="flux" uses reactive flux f, kind="current" uses |J| proxy (still based on TPT).
    """
    R = meta.R
    if phase == "even":
        ph = 0
        edges = horizontal_edges_even(meta)
        default_title = "micro: horizontal EVEN phase (primitive edges)"
    else:
        ph = 1
        edges = horizontal_edges_odd(meta)
        default_title = "micro: horizontal ODD phase (primitive edges)"

    if kind == "flux":
        vals = micro_edge_flux_on_undirected(f_lifted, period, R, ph, edges)
        lab = "reactive flux"
    else:
        # if you want current, derive from f via J=f-f^T outside and pass f only -> compute J once in analysis
        raise ValueError("kind='current' requires J_lifted; use plot_micro_horizontal_current().")

    H = micro_horizontal_heatmap(meta, vals, edges)
    M = np.ma.masked_invalid(H)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    im = ax.imshow(M, aspect="auto", interpolation="nearest")
    ax.set_xlabel("k edge (between k and k+1)")
    ax.set_ylabel("b (row index)")
    ax.set_title(title or (default_title + f" — {lab}"))
    plt.colorbar(im, ax=ax, label=lab)
    plt.tight_layout(); plt.show()

def plot_micro_horizontal_current(meta: PTMeta, J_lifted: np.ndarray, *, period: int = 3,
                                  phase: Literal["even", "odd"] = "even",
                                  title: Optional[str] = None) -> None:
    R = meta.R
    if phase == "even":
        ph = 0
        edges = horizontal_edges_even(meta)
        default_title = "micro: horizontal EVEN phase (primitive edges) — |J|"
    else:
        ph = 1
        edges = horizontal_edges_odd(meta)
        default_title = "micro: horizontal ODD phase (primitive edges) — |J|"

    vals = micro_edge_current_on_undirected(J_lifted, period, R, ph, edges)
    H = micro_horizontal_heatmap(meta, vals, edges)
    M = np.ma.masked_invalid(H)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    im = ax.imshow(M, aspect="auto", interpolation="nearest")
    ax.set_xlabel("k edge (between k and k+1)")
    ax.set_ylabel("b (row index)")
    ax.set_title(title or default_title)
    plt.colorbar(im, ax=ax, label="|J| proxy")
    plt.tight_layout(); plt.show()

def plot_micro_vertical_barcode(meta: PTMeta, f_lifted: np.ndarray, *, period: int = 3,
                                phase: int = 2, top: Optional[int] = 50,
                                vertical_edges: Optional[np.ndarray] = None,
                                title: str = "micro: vertical phase — reactive flux on vertical edges") -> None:
    """
    Vertical edges are meaningful ONLY if you provide the primitive vertical edge list.
    By default uses meta.vertical_edges if present.
    """
    R = meta.R
    vE = meta.vertical_edges if vertical_edges is None else np.asarray(vertical_edges, np.int64)
    if vE is None or np.asarray(vE).size == 0:
        print("No vertical_edges provided/found; cannot do a meaningful vertical-edge plot.")
        return
    edges = canonical_undirected_edges(vE)
    vals = micro_edge_flux_on_undirected(f_lifted, period, R, phase, edges)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    if top is not None:
        vals = vals[: int(top)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(vals, marker="o", linestyle="none")
    ax.set_xlabel("vertical edge rank")
    ax.set_ylabel("reactive flux")
    ax.set_title(title)
    plt.tight_layout(); plt.show()


# ============================================================
# High-level analysis
# ============================================================

def analyze_one_perm2(
    I2: np.ndarray,
    meta: PTMeta,
    ends: Ends,
    *,
    level: Literal["macro", "micro"] = "macro",
    period: int = 3,
    burn_in: Union[int, float] = 0.1,
    stride: int = 3,
    average_offsets: bool = True,
    label_subsample: Optional[np.ndarray] = None,
    chunk: int = 4000
) -> Dict[str, Any]:
    """
    level="macro":
      - estimates P on nodes r using stride (typically = period)
      - phase-averages offsets by default (average_offsets=True) to avoid strobing artifacts
      - computes q+/q- and reactive f/J on the *macro graph*
      - edge plots are meaningful only as macro transitions

    level="micro":
      - builds lifted chain on (phi,r) using stride=1
      - computes q+/q- and reactive f/J on lifted chain
      - provides micro-edge plotting via plot_micro_* on primitive edge lists
    """
    if level == "macro":
        P, pi, C = estimate_P_pi_from_perm2(
            I2, burn_in=burn_in, stride=stride, average_offsets=average_offsets,
            label_subsample=label_subsample, chunk=chunk
        )
        tpt = tpt_on_sets(P, pi, ends.hot_idx, ends.cold_idx)
        lam2, gap = spectral_gap(P, pi)
        db = detailed_balance_report(P, pi, C=C)
        ow = one_way_fraction(P, pi, C=C)
        return dict(
            level="macro",
            P=P, pi=pi, C=C,
            q_plus=tpt["q_plus"], q_minus=tpt["q_minus"],
            k=tpt["k"], f=tpt["f"], J=tpt["J"],
            lambda2=lam2, gap=gap,
            diag=dict(db=db, one_way=ow),
        )

    # micro (lifted)
    Pℓ, piℓ, Cℓ = estimate_lifted_P_pi_from_perm2(
        I2, period=period, burn_in=burn_in, label_subsample=label_subsample, chunk=chunk
    )
    R = meta.R
    hotℓ = lift_indices_for_set(R, period, ends.hot_idx)
    coldℓ = lift_indices_for_set(R, period, ends.cold_idx)

    tpt = tpt_on_sets(Pℓ, piℓ, hotℓ, coldℓ)
    lam2, gap = spectral_gap(Pℓ, piℓ)
    db = detailed_balance_report(Pℓ, piℓ, C=Cℓ)
    ow = one_way_fraction(Pℓ, piℓ, C=Cℓ)
    return dict(
        level="micro",
        period=int(period),
        P_lifted=Pℓ, pi_lifted=piℓ, C_lifted=Cℓ,
        q_plus_lifted=tpt["q_plus"], q_minus_lifted=tpt["q_minus"],
        k=tpt["k"], f_lifted=tpt["f"], J_lifted=tpt["J"],
        lambda2=lam2, gap=gap,
        diag=dict(db=db, one_way=ow),
    )

# ============================================================
# NEW: Hybrid Endpoint Logic
# ============================================================

def ends_hybrid(meta: PTMeta, *, b0: int = 0, hot_k: int = 0, cold_top: int = 1) -> Ends:
    """
    Hybrid Ends:
      - HOT: The entire column at index `hot_k` (across ALL rows b).
             This captures mixing from any replica at the hot boundary.
      - COLD: Strictly the physical target (Row `b0`, last `cold_top` nodes).
    """
    # 1. Identify all nodes in column `hot_k`
    # k_start[b] is the node index of k=0 in row b.
    # So k=hot_k is just k_start[b] + hot_k.
    
    # meta.k_start has size B+1. We iterate 0..B-1.
    starts = meta.k_start[:-1] 
    hot_idx = starts + int(hot_k)
    
    # Safety check: ensure hot_k fits in every row? 
    # Assuming rows are long enough for hot_k. If not, this index is invalid.
    # (We assume regular grids for now, or user sets hot_k=0).

    # 2. Identify cold nodes strictly in the target row (b0)
    r0, r1 = int(meta.k_start[b0]), int(meta.k_start[b0 + 1])
    K0 = r1 - r0
    cold_top = int(max(1, cold_top))
    
    cold_ks = np.arange(max(0, K0 - cold_top), K0, dtype=np.int64)
    cold_idx = (r0 + cold_ks).astype(np.int64)

    # 3. Validation
    if np.intersect1d(hot_idx, cold_idx).size > 0:
        raise ValueError("hot and cold sets overlap (adjust hot_k/cold_top).")

    row0 = np.arange(r0, r1, dtype=np.int64)
    return Ends(hot_idx=hot_idx, cold_idx=cold_idx, b0=b0, row0_boxes=row0)

from typing import Optional
from typing import Literal

def get_flexible_ends(
    meta: PTMeta, 
    *, 
    mode: Literal["C0_H0", "C0_C4", "C0_Hall", "C0_Dual"] = "C0_Hall", # <--- NEW MODE
    b_cold: int = 0,       
    b_aux: int = 4,        
    k_top: int = 1,        
    invert: bool = True    # MUST be True for this test (Source=Cold Trap)
) -> Ends:
    """
    Final Universal Ends Generator.
    
    New Mode:
      - "C0_Dual": Source = Physical Cold (C0).
                   Sink   = UNION of Thermal Hot (H_all) AND Smooth Row (C4/All Smooth).
                   (The "Fork in the Road" Test).
    """
    
    # 1. Define C0 (The Trap / Source)
    r0_c, r1_c = int(meta.k_start[b_cold]), int(meta.k_start[b_cold + 1])
    c0_nodes = np.arange(r1_c - k_top, r1_c, dtype=np.int64)
    
    # 2. Define the Target Set based on Mode
    hot_nodes = np.array([], dtype=np.int64)
    
    # ... (Previous modes C0_H0, C0_C4, C0_Hall logic remains same) ...
    if mode == "C0_H0":
        hot_nodes = np.array([r0_c], dtype=np.int64)
    elif mode == "C0_C4":
        r0_aux, r1_aux = int(meta.k_start[b_aux]), int(meta.k_start[b_aux + 1])
        hot_nodes = np.arange(r1_aux - k_top, r1_aux, dtype=np.int64)
    elif mode == "C0_Hall":
        hot_nodes = meta.k_start[:-1].astype(np.int64)
        
    # --- THE NEW DUAL MODE ---
    elif mode == "C0_Dual":
        # Target A: All Thermal Hot Nodes (k=0 everywhere)
        target_A = meta.k_start[:-1].astype(np.int64)
        
        # Target B: The Entire Smooth Row (or just the tip, but entire row is safer for "capture")
        if not (0 <= b_aux < meta.B):
            raise ValueError(f"b_aux={b_aux} is out of bounds.")
        r_start = int(meta.k_start[b_aux])
        r_end   = int(meta.k_start[b_aux + 1])
        target_B = np.arange(r_start, r_end, dtype=np.int64)
        
        # Combine them
        hot_nodes = np.unique(np.concatenate([target_A, target_B]))
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 3. Safety: Remove Overlaps
    if np.intersect1d(c0_nodes, hot_nodes).size > 0:
        hot_nodes = np.setdiff1d(hot_nodes, c0_nodes)
        if hot_nodes.size == 0:
            raise ValueError(f"Mode {mode} resulted in identical Source and Sink sets!")

    # 4. Directionality
    # For the Branching Test, we ALWAYS want Source=Trap (Invert=True)
    if not invert and mode == "C0_Dual":
        print("Warning: C0_Dual usually implies Trap->Exit. You set invert=False (Exit->Trap).")

    if invert:
        src = c0_nodes
        tgt = hot_nodes
    else:
        src = hot_nodes
        tgt = c0_nodes

    return Ends(hot_idx=src, cold_idx=tgt, b0=b_cold, row0_boxes=np.arange(r0_c, r1_c))
# ============================================================
# UPDATED: Analysis Functions
# ============================================================


from typing import Iterable, Any, Optional, Literal, Union, Dict

def analyze_results_list(
    results: Iterable[Any],
    sys,
    *,
    # --- TPT Endpoint Configuration ---
    mode: Literal["C0_H0", "C0_C4", "C0_Hall", "C0_Dual"] = "C0_Hall",
    b0: int = 0,             # Maps to b_cold (Physical Row)
    b_aux: int = 4,          # Maps to b_aux (Smooth/Aux Row)
    cold_top: int = 1,       # Maps to k_top (Size of sink/source tip)
    invert: bool = False,    # False = Rain (Hot->Cold), True = Fountain (Cold->Hot)
    
    # --- Analysis Parameters ---
    level: Literal["macro", "micro"] = "macro",
    period: int = 3,
    burn_in: Union[int, float] = 0.1,
    stride: int = 3,
    average_offsets: bool = True,
    label_subsample: Optional[np.ndarray] = None,
    chunk: int = 4000,
) -> Dict[str, Any]:
    
    meta = meta_from_sys(sys)
    
    # Call the final "Swiss Army Knife" ends generator
    ends = get_flexible_ends(
        meta, 
        mode=mode, 
        b_cold=b0,        # The physical row index
        b_aux=b_aux,      # The smooth row index (used if mode is C0_C4)
        k_top=cold_top,   # Size of the cold tip
        invert=invert
    )

    out = dict(rid=[], k=[], lambda2=[], gap=[])
    per = {}

    for res in results:
        I2 = get_perm_ts(res)
        a = analyze_one_perm2(
            I2, meta, ends,
            level=level, period=period, burn_in=burn_in,
            stride=stride, average_offsets=average_offsets,
            label_subsample=label_subsample, chunk=chunk
        )
        rid = int(getattr(res, "rid", len(out["rid"])))
        out["rid"].append(rid)
        out["k"].append(a["k"])
        out["lambda2"].append(a["lambda2"])
        out["gap"].append(a["gap"])
        per[rid] = a

    for k in ("rid", "k", "lambda2", "gap"):
        out[k] = np.asarray(out[k], dtype=np.float64 if k != "rid" else np.int64)
    
    out["meta"] = meta
    out["ends"] = ends
    out["per_rid"] = per
    out["level"] = level
    out["period"] = int(period)
    out["stride"] = int(stride)
    out["average_offsets"] = bool(average_offsets)
    
    # Store config for reproducibility
    out["ends_config"] = {
        "mode": mode,
        "b0": b0,
        "b_aux": b_aux,
        "invert": invert
    }
    
    return out

def analyze_I_stack(
    I_stack: np.ndarray,
    beta: np.ndarray,
    *,
    hot_k: int = 0,
    cold_top: int = 1,
    level: Literal["macro", "micro"] = "macro",
    period: int = 3,
    burn_in: Union[int, float] = 0.1,
    stride: int = 3,
    average_offsets: bool = True,
    label_subsample: Optional[np.ndarray] = None,
    chunk: int = 4000,
) -> Dict[str, Any]:
    I_stack = np.asarray(I_stack, dtype=np.int64)
    Nd, two, T, R = I_stack.shape
    assert two == 2
    meta = meta_1d(beta)
    ends = ends_linear(R, hot_k=hot_k, cold_top=cold_top)

    out = dict(rid=np.arange(Nd, dtype=np.int64), k=[], lambda2=[], gap=[])
    per = {}

    for d in range(Nd):
        a = analyze_one_perm2(
            I_stack[d], meta, ends,
            level=level, period=period, burn_in=burn_in,
            stride=stride, average_offsets=average_offsets,
            label_subsample=label_subsample, chunk=chunk
        )
        out["k"].append(a["k"])
        out["lambda2"].append(a["lambda2"])
        out["gap"].append(a["gap"])
        per[int(d)] = a

    for k in ("k", "lambda2", "gap"):
        out[k] = np.asarray(out[k], dtype=np.float64)
    out["meta"] = meta
    out["ends"] = ends
    out["per_rid"] = per
    out["level"] = level
    out["period"] = int(period)
    out["stride"] = int(stride)
    out["average_offsets"] = bool(average_offsets)
    return out

# ============================================================
# ADD-ON PLOTS (meaningful in macro + micro/lifted settings)
# Paste below the rest of ptflow.py
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple

# ------------------------------
# Small utilities
# ------------------------------

def _as_undirected_unique(edges_uv: np.ndarray) -> np.ndarray:
    E = np.asarray(edges_uv, dtype=np.int64)
    if E.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    u = np.minimum(E[:, 0], E[:, 1])
    v = np.maximum(E[:, 0], E[:, 1])
    Ev = np.unique(np.stack([u, v], axis=1), axis=0)
    return Ev

def observed_undirected_edges_from_counts(C: np.ndarray) -> np.ndarray:
    """Undirected edges u<v where C_uv + C_vu > 0 and u!=v."""
    C = np.asarray(C)
    obs = (C + C.T) > 0
    np.fill_diagonal(obs, False)
    u, v = np.where(np.triu(obs, 1))
    return np.stack([u, v], axis=1).astype(np.int64)

def edges_touching_set(edges_uv: np.ndarray, node_set: np.ndarray, R: int) -> np.ndarray:
    """Mask of edges that touch any node in node_set. node_set is in [0,R)."""
    node_set = np.asarray(node_set, dtype=np.int64)
    E = np.asarray(edges_uv, dtype=np.int64)
    if E.size == 0 or node_set.size == 0:
        return np.zeros(E.shape[0], dtype=bool)
    S = np.zeros(R, dtype=bool)
    S[node_set] = True
    return S[E[:, 0]] | S[E[:, 1]]

def qbar_on_edges(q: np.ndarray, edges_uv: np.ndarray) -> np.ndarray:
    u = edges_uv[:, 0]
    v = edges_uv[:, 1]
    return 0.5 * (q[u] + q[v])

def _lift_edge_vals_absJ(J_lifted: np.ndarray, *, R: int, period: int, ph: int, edges_uv: np.ndarray) -> np.ndarray:
    """
    For undirected primitive edges (u,v), return |J| proxy on lifted forward edges in phase ph:
      val = |J((ph,u)->(ph+1,v))| + |J((ph,v)->(ph+1,u))|
    """
    E = _as_undirected_unique(edges_uv)
    if E.size == 0:
        return np.empty((0,), dtype=np.float64)
    phn = (ph + 1) % period
    u = E[:, 0]
    v = E[:, 1]
    iu = ph * R + u
    iv = ph * R + v
    jv = phn * R + v
    ju = phn * R + u
    return np.abs(J_lifted[iu, jv]) + np.abs(J_lifted[iv, ju])

def _lift_edge_vals_flux(f_lifted: np.ndarray, *, R: int, period: int, ph: int, edges_uv: np.ndarray) -> np.ndarray:
    """
    For undirected primitive edges (u,v), return reactive flux in phase ph:
      val = f((ph,u)->(ph+1,v)) + f((ph,v)->(ph+1,u))
    """
    E = _as_undirected_unique(edges_uv)
    if E.size == 0:
        return np.empty((0,), dtype=np.float64)
    phn = (ph + 1) % period
    u = E[:, 0]
    v = E[:, 1]
    iu = ph * R + u
    iv = ph * R + v
    jv = phn * R + v
    ju = phn * R + u
    return f_lifted[iu, jv] + f_lifted[iv, ju]

# ------------------------------
# Vertical-flux-fraction helpers
# ------------------------------

def vertical_flux_fraction_macro(meta, C: np.ndarray, J: np.ndarray) -> float:
    """
    Macro meaning:
      fraction of |J| carried by macro-edges that change row (b_of_r differs).
    This is meaningful on the *macro transition graph* implied by your chosen stride/averaging.
    """
    R = int(meta.R)
    C = np.asarray(C)
    J = np.asarray(J, float)
    obs = (C + C.T) > 0
    np.fill_diagonal(obs, False)
    if not np.any(obs):
        return np.nan
    b = np.asarray(meta.b_of_r, dtype=np.int64)
    bb = (b[:, None] != b[None, :])
    num = np.abs(J)[obs & bb].sum()
    den = np.abs(J)[obs].sum()
    return float(num / den) if den > 0 else np.nan

def vertical_flux_fraction_micro(meta, f_lifted: np.ndarray, *,
                                period: int = 3,
                                vertical_edges: Optional[np.ndarray] = None,
                                ph_vertical: int = 2) -> float:
    """
    Micro meaning:
      fraction of reactive flux carried by primitive vertical edges during the vertical phase.
    Requires a primitive vertical edge list (u,v) in node indices r.
    """
    R = int(meta.R)
    if vertical_edges is None:
        vertical_edges = getattr(meta, "vertical_edges", None)
    if vertical_edges is None:
        return np.nan
    vE = np.asarray(vertical_edges, dtype=np.int64)
    if vE.size == 0:
        return np.nan

    # define primitive horizontal edges in their phases if available
    try:
        he = horizontal_edges_even(meta)
        ho = horizontal_edges_odd(meta)
    except Exception:
        he = np.empty((0, 2), np.int64)
        ho = np.empty((0, 2), np.int64)

    flux_even = _lift_edge_vals_flux(f_lifted, R=R, period=period, ph=0, edges_uv=he).sum()
    flux_odd  = _lift_edge_vals_flux(f_lifted, R=R, period=period, ph=1, edges_uv=ho).sum()
    flux_ver  = _lift_edge_vals_flux(f_lifted, R=R, period=period, ph=ph_vertical, edges_uv=vE).sum()

    den = flux_even + flux_odd + flux_ver
    return float(flux_ver / den) if den > 0 else np.nan

def compute_vff(out: Dict[str, Any], *, mode: str = "macro",
                period: int = 3,
                vertical_edges: Optional[np.ndarray] = None,
                ph_vertical: int = 2) -> np.ndarray:
    """
    Compute vff array aligned with out["rid"] from out["per_rid"].
    mode:
      - "macro": uses vertical_flux_fraction_macro(meta,C,J)
      - "micro": uses vertical_flux_fraction_micro(meta,f_lifted,...)
    """
    meta = out["meta"]
    rids = list(out["per_rid"].keys())
    # keep in the same order as out["rid"] if present
    if "rid" in out:
        rids = [int(x) for x in out["rid"]]
    vff = np.full(len(rids), np.nan, dtype=np.float64)
    for i, rid in enumerate(rids):
        A = out["per_rid"][int(rid)]
        if mode == "macro":
            vff[i] = vertical_flux_fraction_macro(meta, A["C"], A["J"])
        elif mode == "micro":
            vff[i] = vertical_flux_fraction_micro(meta, A["f_lifted"], period=period,
                                                  vertical_edges=vertical_edges, ph_vertical=ph_vertical)
        else:
            raise ValueError("mode must be 'macro' or 'micro'")
    return vff

# ------------------------------
# 1) speedup summary plots
# ------------------------------

def plot_speedup_hist(F1: np.ndarray, F2: np.ndarray, *, bins=30, title="Speedup S = F2/F1") -> None:
    F1 = np.asarray(F1, dtype=np.float64)
    F2 = np.asarray(F2, dtype=np.float64)
    m = np.isfinite(F1) & np.isfinite(F2) & (F1 > 0) & (F2 > 0)
    if not np.any(m):
        print("No finite positive values.")
        return
    S = F2[m] / F1[m]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(np.log10(S), bins=bins)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("log10 speedup")
    ax.set_ylabel("count")
    ax.set_title(title)
    plt.tight_layout(); plt.show()

def plot_speedup_cdf(F1: np.ndarray, F2: np.ndarray, *, title="Speedup CDF") -> None:
    F1 = np.asarray(F1, dtype=np.float64)
    F2 = np.asarray(F2, dtype=np.float64)
    m = np.isfinite(F1) & np.isfinite(F2) & (F1 > 0) & (F2 > 0)
    if not np.any(m):
        print("No finite positive values.")
        return
    S = np.sort(F2[m] / F1[m])
    y = np.linspace(0, 1, S.size, endpoint=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.semilogx(S, y)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("speedup S")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    plt.tight_layout(); plt.show()

def plot_vff_vs_speedup(vff2: np.ndarray, F1: np.ndarray, F2: np.ndarray, *, title="Vertical flux fraction vs speedup") -> None:
    vff2 = np.asarray(vff2, dtype=np.float64)
    F1 = np.asarray(F1, dtype=np.float64)
    F2 = np.asarray(F2, dtype=np.float64)
    m = np.isfinite(vff2) & np.isfinite(F1) & np.isfinite(F2) & (F1 > 0) & (F2 > 0)
    if not np.any(m):
        print("No finite data.")
        return
    S = F2[m] / F1[m]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(vff2[m], S, "o")
    ax.set_xlabel("vertical flux fraction")
    ax.set_ylabel("speedup S")
    ax.set_yscale("log")
    ax.set_title(title)
    plt.tight_layout(); plt.show()

# ------------------------------
# 2) distributions / ranked compare
# ------------------------------

def plot_distribution_compare(x1: np.ndarray, x2: np.ndarray, *,
                              label1="1D", label2="2D",
                              logx=False, bins=30, title="Distribution compare") -> None:
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    x1 = x1[np.isfinite(x1)]
    x2 = x2[np.isfinite(x2)]
    if x1.size == 0 or x2.size == 0:
        print("Empty data.")
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    if logx:
        x1p = np.log10(np.maximum(x1, 1e-300))
        x2p = np.log10(np.maximum(x2, 1e-300))
        ax.hist(x1p, bins=bins, alpha=0.6, label=label1)
        ax.hist(x2p, bins=bins-10, alpha=0.6, label=label2)
        ax.set_xlabel("log10(value)")
    else:
        ax.hist(x1, bins=bins, alpha=0.6, label=label1)
        ax.hist(x2, bins=bins, alpha=0.6, label=label2)
        ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.legend()
    ax.set_title(title)
    plt.tight_layout(); plt.show()

def plot_ranked_compare(y1: np.ndarray, y2: np.ndarray, *,
                        title="Ranked compare (sorted by y1)", ylabel="value") -> None:
    y1 = np.asarray(y1, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)
    m = np.isfinite(y1) & np.isfinite(y2)
    if not np.any(m):
        print("No finite pairs.")
        return
    order = np.argsort(y1[m])  # easiest->hardest or reverse depending on meaning
    a = y1[m][order]
    b = y2[m][order]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(a, label="y1")
    ax.plot(b, label="y2")
    ax.set_xlabel("rank (sorted by y1)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout(); plt.show()

# ------------------------------
# 3) bottleneck concentration: Pareto of |J|
# ------------------------------

def plot_pareto_edge_current(meta, C: np.ndarray, J: np.ndarray, *, title="Pareto of |J| over observed edges") -> None:
    edges = observed_undirected_edges_from_counts(C)
    if edges.size == 0:
        print("No observed edges.")
        return
    u = edges[:, 0]; v = edges[:, 1]
    vals = np.abs(np.asarray(J, float)[u, v])
    vals = np.sort(vals)[::-1]
    s = vals.sum()
    if s <= 0:
        print("Zero current.")
        return
    cdf = np.cumsum(vals) / s
    x = np.arange(1, vals.size + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(x, cdf)
    ax.set_xlabel("# edges (sorted by |J|)")
    ax.set_ylabel("cumulative fraction of total |J|")
    ax.set_title(title)
    plt.tight_layout(); plt.show()

# ------------------------------
# 4) vertical diagnostics (dispatch: macro vs lifted)
# ------------------------------

def plot_vertical_barcode(meta, J: np.ndarray, vertical_edges: np.ndarray, *,
                          top: Optional[int] = 40,
                          period: int = 3,
                          phase: int = 2,
                          title: str = "Vertical edges barcode") -> None:
    """
    If J is (R,R): treats as macro current; plots |J[u,v]| on provided edges.
    If J is (period*R, period*R): treats as lifted current; plots |J| proxy on lifted forward edges in given phase.
    """
    R = int(meta.R)
    E = _as_undirected_unique(vertical_edges)
    if E.size == 0:
        print("No edges provided.")
        return
    J = np.asarray(J, float)
    if J.shape[0] == R:
        u, v = E[:, 0], E[:, 1]
        vals = np.abs(J[u, v])
    else:
        vals = _lift_edge_vals_absJ(J, R=R, period=int(period), ph=int(phase), edges_uv=E)

    vals = np.asarray(vals, float)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    if top is not None:
        vals = vals[: int(top)]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(vals, marker="o", linestyle="none")
    ax.set_xlabel("edge rank")
    ax.set_ylabel("|J| (macro) or |J| proxy (lifted)")
    ax.set_title(title)
    plt.tight_layout(); plt.show()

def plot_vertical_rowpair_heatmap(meta, J: np.ndarray, vertical_edges: np.ndarray, *,
                                  period: int = 3,
                                  phase: int = 2,
                                  title="Sum |J| between row pairs") -> None:
    R = int(meta.R)
    B = int(meta.B)
    E = _as_undirected_unique(vertical_edges)
    if E.size == 0:
        print("No edges provided.")
        return
    J = np.asarray(J, float)

    if J.shape[0] == R:
        u, v = E[:, 0], E[:, 1]
        vals = np.abs(J[u, v])
    else:
        vals = _lift_edge_vals_absJ(J, R=R, period=int(period), ph=int(phase), edges_uv=E)

    M = np.zeros((B, B), dtype=np.float64)
    for (u, v), val in zip(E, vals):
        bu = int(meta.b_of_r[u]); bv = int(meta.b_of_r[v])
        M[bu, bv] += float(val)
        M[bv, bu] += float(val)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(M, interpolation="nearest")
    ax.set_xlabel("row b")
    ax.set_ylabel("row b")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="sum |J| / proxy")
    plt.tight_layout(); plt.show()

# ------------------------------
# 5) endpoint-bias checks + J vs qbar
# ------------------------------

def plot_top_edges_excluding_endpoints(meta, C: np.ndarray, J: np.ndarray,
                                       hot_idx: np.ndarray, cold_idx: np.ndarray,
                                       *, top=25, title="Top |J| edges excluding endpoints") -> None:
    """
    Macro-only meaningful: uses observed edges from C on the macro graph.
    (For lifted micro, use edge-list/phase-restricted plots instead.)
    """
    R = int(meta.R)
    edges = observed_undirected_edges_from_counts(C)
    if edges.size == 0:
        print("No observed edges.")
        return

    end = np.unique(np.concatenate([np.asarray(hot_idx, np.int64), np.asarray(cold_idx, np.int64)]))
    touch = edges_touching_set(edges, end, R=R)
    edges2 = edges[~touch]
    if edges2.size == 0:
        print("All observed edges touch endpoints.")
        return

    u, v = edges2[:, 0], edges2[:, 1]
    vals = np.abs(np.asarray(J, float)[u, v])
    order = np.argsort(vals)[::-1]
    edges2 = edges2[order]
    vals = vals[order]
    m = min(int(top), vals.size)

    labels = []
    colors = []
    for u, v in edges2[:m]:
        bu, ku = node_to_bk(meta, int(u))
        bv, kv = node_to_bk(meta, int(v))
        labels.append(f"({bu},{ku})–({bv},{kv})")
        colors.append("C1" if bu != bv else "C0")

    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.bar(np.arange(m), vals[:m], color=colors)
    ax.set_xticks(np.arange(m))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("|J|")
    ax.set_title(title + " (orange=row-change, blue=within-row)")
    plt.tight_layout(); plt.show()

def plot_J_vs_qbar(meta, C: np.ndarray, J: np.ndarray, q: np.ndarray, *,
                   vertical_edges: Optional[np.ndarray] = None,
                   title="|J| vs mid-edge committor q̄") -> None:
    """
    Macro-only meaningful: q should be q_plus on nodes.
    """
    edges = observed_undirected_edges_from_counts(C)
    if edges.size == 0:
        print("No observed edges.")
        return
    u, v = edges[:, 0], edges[:, 1]
    vals = np.abs(np.asarray(J, float)[u, v])
    qb = qbar_on_edges(np.asarray(q, float), edges)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(qb, vals, "o", markersize=3, alpha=0.6)
    ax.set_xlabel("q̄ = (q_u + q_v)/2")
    ax.set_ylabel("|J|")
    ax.set_title(title)
    plt.tight_layout(); plt.show()

    if vertical_edges is not None:
        E = _as_undirected_unique(vertical_edges)
        if E.size == 0:
            return
        u, v = E[:, 0], E[:, 1]
        valsV = np.abs(np.asarray(J, float)[u, v])
        qbV = qbar_on_edges(np.asarray(q, float), E)

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(qbV, valsV, "o", markersize=3, alpha=0.7)
        ax.set_xlabel("q̄ on provided edges")
        ax.set_ylabel("|J|")
        ax.set_title(title + " (restricted edge set)")
        plt.tight_layout(); plt.show()
# ============================================================
# NEW VISUALIZATION MODULE (Meaningful 2D Flows)
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def _get_phase_indices(period: int = 3):
    """
    Returns mapping of physical operator to transition index t->t+1.
    Based on standard schedule:
       1. Apply Even Swaps -> Record I[0]
       2. Apply Odd Swaps  -> Record I[1]
       3. Apply Vert Swaps -> Record I[2]
       4. Apply Even Swaps -> Record I[3]...
    
    Transition I[0]->I[1] captures ODD operator.
    Transition I[1]->I[2] captures VERT operator.
    Transition I[2]->I[3] captures EVEN operator.
    """
    # This mapping assumes the recording order in your snippet
    return dict(odd=0, vert=1, even=2)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_2d_current_streamlines(
    meta: PTMeta, 
    J_macro: np.ndarray, 
    *, 
    title: str = "Macro Net Current Streamlines (Color = Magnitude)",
    min_flux: float = 1e-10,
    cmap_name: str = 'magma'
):
    """
    Visualizes the "Net Current" J as a 2D vector field.
    ROBUST VERSION: Projects the flux J[u,v] onto the vector (v - u).
    This correctly handles stride > 1 (long jumps), diagonals, and all boundaries.
    """
    B = meta.B
    # 1. Determine Grid Dimensions & Map Nodes to Coordinates
    max_k = 0
    node_coords = {} # map node_idx -> (k, b)
    
    for b in range(B):
        r_start = int(meta.k_start[b])
        r_end = int(meta.k_start[b+1])
        len_k = r_end - r_start
        max_k = max(max_k, len_k)
        
        for k_idx in range(len_k):
            u = r_start + k_idx
            node_coords[u] = (k_idx, b)

    # 2. Initialize Vector Grid
    X = np.arange(max_k)
    Y = np.arange(B)
    
    U = np.zeros((B, max_k)) 
    V = np.zeros((B, max_k))
    MAG = np.zeros((B, max_k))
    
    # 3. Calculate Net Current Vector at each node u
    #    Formula: vec_u = sum_v ( J[u,v] * (pos_v - pos_u) )
    
    # Iterate over all source nodes u
    for u, (kx_u, ky_u) in node_coords.items():
        
        vec_x = 0.0
        vec_y = 0.0
        
        # We only need to iterate over v where J[u, v] is non-zero.
        # Since J is likely (R, R), we can just iterate the row u.
        # If J is sparse, this is fast. If dense but small (R~100), also fast.
        
        # Get all v indices (columns)
        # Optimization: only check v where J[u,v] != 0
        targets = np.where(np.abs(J_macro[u]) > 0)[0]
        
        for v in targets:
            flux = J_macro[u, v]
            
            # Get coordinates of target v
            if v not in node_coords: 
                continue # Should not happen if meta matches J
            
            kx_v, ky_v = node_coords[v]
            
            # Displacement vector
            dx = kx_v - kx_u
            dy = ky_v - ky_u
            dist = np.sqrt(dx*dx + dy*dy)
            weight_x = dx / dist
            weight_y = dy / dist
            # Add weighted contribution
            vec_x += flux * weight_x
            vec_y += flux * weight_y
            
        U[ky_u, kx_u] = vec_x
        V[ky_u, kx_u] = vec_y
        MAG[ky_u, kx_u] = np.sqrt(vec_x**2 + vec_y**2)

    # 4. Normalization for Fixed Length Arrows
    mask = MAG > min_flux
    
    U_norm = np.copy(U)
    V_norm = np.copy(V)
    
    # Normalize valid vectors to unit length
    U_norm[mask] /= MAG[mask]
    V_norm[mask] /= MAG[mask]
    
    # Hide insignificant
    U_norm[~mask] = np.nan
    V_norm[~mask] = np.nan
    
    # 5. Plotting
    fig, ax = plt.subplots(figsize=(10, 5 + 0.5*B))
    
    # LogNorm for colors to see both the "Trickle" and the "Flood"
    norm = mcolors.LogNorm(vmin=max(min_flux, MAG.min()), vmax=MAG.max())
    
    q = ax.quiver(X, Y, U_norm, V_norm, MAG, 
                  cmap=cmap_name, norm=norm,
                  pivot='mid', 
                  scale=30, scale_units='width',
                  headwidth=4, headlength=5, width=0.005)
    
    cbar = plt.colorbar(q, ax=ax)
    cbar.set_label('Net Flux Magnitude (Log Scale)')
    
    ax.set_xticks(X)
    ax.set_yticks(Y)
    ax.set_xlabel("Temperature Index k")
    ax.set_ylabel("Replica Row b")
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.scatter(np.tile(X, B), np.repeat(Y, len(X)), c='gray', s=2, alpha=0.2)
    
    plt.tight_layout()
    plt.show()

def plot_micro_vertical_flux(meta: PTMeta, f_lifted: np.ndarray, *, 
                             period: int = 3,
                             title: str = "Micro: Vertical Flux (Active Phase)"):
    """
    Plots the specific reactive flux carried by vertical edges 
    during the phase where vertical swaps actually happen.
    """
    # Identify the vertical phase
    # Based on our analysis: Phase 1 (index 1->2) is Vertical.
    phases = _get_phase_indices(period)
    ph_vert = phases['vert']
    
    R = meta.R
    vE = meta.vertical_edges
    
    if vE is None or vE.size == 0:
        print("No vertical edges defined in meta.")
        return

    # Calculate flux on these edges in this phase
    # flux = f((ph, u) -> (ph+1, v)) + f((ph, v) -> (ph+1, u))
    # Here u,v are node indices.
    
    edges = canonical_undirected_edges(vE)
    vals = micro_edge_flux_on_undirected(f_lifted, period, R, ph_vert, edges)
    
    # Map to a heatmap of (Row b -> Row b+1)
    # We assume edges connect adjacent rows.
    B = meta.B
    if B < 2: 
        return
        
    # We will plot "Total Flux between Row b and b+1" vs "Approximate Column k"
    # Since edges might be irregular, let's just plot the raw edges values 
    # but colored by their position.
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # X axis: k index of the 'upper' node (smaller b)
    # Y axis: flux value
    
    for i, (u, v) in enumerate(edges):
        val = vals[i]
        if val <= 0: continue
        
        # Determine which is 'top' (row b) and 'bottom' (row b+1)
        b_u, k_u = node_to_bk(meta, u)
        b_v, k_v = node_to_bk(meta, v)
        
        if b_u == b_v: continue # not vertical
        
        # x-coordinate = average k
        x = (k_u + k_v) / 2.0
        # color = which row gap
        gap_idx = min(b_u, b_v)
        
        ax.plot(x, val, 'o', color=f"C{gap_idx}", label=f"Row {gap_idx}->{gap_idx+1}" if i==0 else "")
        ax.vlines(x, 0, val, color=f"C{gap_idx}", alpha=0.3)

    ax.set_xlabel("k index (approx)")
    ax.set_ylabel("Reactive Flux")
    ax.set_title(title)
    # Deduplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.show()

def compare_rates(k_1d, k_2d):
    """Simple text report on speedup."""
    m1, m2 = np.nanmean(k_1d), np.nanmean(k_2d)
    print(f"--- Rate Comparison ---")
    print(f"Mean Rate 1D: {m1:.2e}")
    print(f"Mean Rate 2D: {m2:.2e}")
    if m1 > 0:
        print(f"Speedup Factor: {m2/m1:.2f}x")
    else:
        print(f"Speedup Factor: Undefined (1D rate is 0)")


