import numpy as np
from typing import Dict, Any, Optional, Union, Literal, Iterable,Tuple
from dataclasses import dataclass

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











def time_reversed_kernel(P: np.ndarray, pi: np.ndarray, *, eps: float = 1e-300) -> np.ndarray:
    """
    P*_{u,v} = (pi[v] * P[v,u]) / pi[u]
    FIXED: row-normalize nonzero rows even if some rows are zero.
    """
    P = np.asarray(P, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    R = P.shape[0]
    Pstar = np.zeros_like(P)

    # fill
    for u in range(R):
        if pi[u] <= eps:
            continue
        Pstar[u, :] = (pi * P[:, u]) / pi[u]

    # row normalize with proper handling of zero rows
    row = Pstar.sum(axis=1, keepdims=True)
    z = (row[:, 0] <= 0)
    if np.any(z):
        Pstar[z, :] = 0.0
        Pstar[z, z] = 1.0
        row = Pstar.sum(axis=1, keepdims=True)

    nz = ~z
    Pstar[nz, :] = Pstar[nz, :] / row[nz, :]
    return Pstar


def forward_committor(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    q+[i] = Prob(hit B before A | start i), with boundary q=0 on A, q=1 on B.
    """
    P = np.asarray(P, dtype=np.float64)
    R = P.shape[0]
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)

    maskA = np.zeros(R, bool); maskA[A] = True
    maskB = np.zeros(R, bool); maskB[B] = True
    free = ~(maskA | maskB)

    q = np.zeros(R, np.float64)
    q[maskA] = 0.0
    q[maskB] = 1.0

    idx = np.flatnonzero(free)
    if idx.size == 0:
        return q

    Pff = P[np.ix_(free, free)]
    rhs = P[np.ix_(free, maskB)].sum(axis=1)   # since q=1 on B
    M = np.eye(idx.size) - Pff
    try:
        qf = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        qf = np.linalg.lstsq(M, rhs, rcond=None)[0]

    q[idx] = qf
    return np.clip(q, 0.0, 1.0)


def backward_committor(P: np.ndarray, pi: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    q-[i] for general (possibly nonreversible) chain, computed as forward committor
    on time-reversed (P*), with boundary q-=1 on A and q-=0 on B.
    """
    Pstar = time_reversed_kernel(P, pi)
    # forward committor on P* to hit A before B:
    # boundary 1 on A, 0 on B
    Pstar = np.asarray(Pstar, dtype=np.float64)
    R = Pstar.shape[0]
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)

    maskA = np.zeros(R, bool); maskA[A] = True
    maskB = np.zeros(R, bool); maskB[B] = True
    free = ~(maskA | maskB)

    q = np.zeros(R, np.float64)
    q[maskA] = 1.0
    q[maskB] = 0.0

    idx = np.flatnonzero(free)
    if idx.size == 0:
        return q

    Pff = Pstar[np.ix_(free, free)]
    rhs = Pstar[np.ix_(free, maskA)].sum(axis=1)   # since q=1 on A
    M = np.eye(idx.size) - Pff
    try:
        qf = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        qf = np.linalg.lstsq(M, rhs, rcond=None)[0]

    q[idx] = qf
    return np.clip(q, 0.0, 1.0)


def tpt_rate_flux(P: np.ndarray, pi: np.ndarray, A: np.ndarray, B: np.ndarray) -> Dict[str, Any]:
    """
    Nonreversible-safe TPT:
      f_uv = pi_u P_uv q^-_u q^+_v
      k_AB = sum_{u in A} sum_v f_uv
    """
    P = np.asarray(P, np.float64)
    pi = np.asarray(pi, np.float64)

    qp = forward_committor(P, A, B)         # to hit B before A
    qm = backward_committor(P, pi, A, B)

    f = (pi[:, None] * P) * (qm[:, None]) * (qp[None, :])
    J = f - f.T

    A = np.asarray(A, np.int64)
    kAB = float(f[A, :].sum())
    return dict(q_plus=qp, q_minus=qm, k=kAB, f=f, J=J)


def mfpt_to_set(P: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    m[i] = E_i[tau_B], expected hitting time of B.
    Boundary: m=0 on B. Solve on F = not-B:
      m_F = 1 + P_FF m_F
      (I - P_FF) m_F = 1
    """
    P = np.asarray(P, np.float64)
    R = P.shape[0]
    B = np.asarray(B, np.int64)
    maskB = np.zeros(R, bool); maskB[B] = True
    F = ~maskB

    m = np.zeros(R, np.float64)
    idx = np.flatnonzero(F)
    if idx.size == 0:
        return m

    Pff = P[np.ix_(F, F)]
    rhs = np.ones(idx.size, np.float64)
    M = np.eye(idx.size) - Pff
    try:
        mf = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        mf = np.linalg.lstsq(M, rhs, rcond=None)[0]
    m[idx] = mf
    m[maskB] = 0.0
    return m


def mean_mfpt_A_to_B(P: np.ndarray, pi: np.ndarray, A: np.ndarray, B: np.ndarray,
                     *, start: Literal["piA"] = "piA") -> float:
    """
    Average MFPT from A to B using a chosen start distribution on A.
    Default start="piA": stationary restricted to A.
    """
    P = np.asarray(P, np.float64)
    pi = np.asarray(pi, np.float64)
    A = np.asarray(A, np.int64)
    B = np.asarray(B, np.int64)

    m = mfpt_to_set(P, B)

    if start == "piA":
        w = pi[A].copy()
        s = w.sum()
        if s <= 0:
            w[:] = 1.0 / max(1, w.size)
        else:
            w /= s
        return float((w * m[A]).sum())

    raise ValueError("unknown start distribution")


def estimate_theory_kernel_from_perm2(
    I2: np.ndarray,
    *,
    stride: int,
    period: int = 3,
    burn_in: Union[int, float] = 0.1,
    average_offsets: bool = True,
    label_subsample: Optional[np.ndarray] = None,
    chunk: int = 4000,
) -> Dict[str, Any]:
    """
    Unified kernel estimator.
      - stride=1: lifted micro chain on (phi,r) to make schedule homogeneous.
      - stride>1: macro chain on nodes with that stride (optionally average offsets).

    Returns dict with P, pi, C, dt, lifted flags, and (if lifted) R_nodes & period.
    """
    stride = int(stride)
    if stride == 1:
        P, pi, C = estimate_lifted_P_pi_from_perm2(
            I2, period=period, burn_in=burn_in, label_subsample=label_subsample, chunk=chunk
        )
        # dt in underlying ticks per kernel step:
        return dict(P=P, pi=pi, C=C, dt=1, lifted=True, period=int(period))
    else:
        P, pi, C = estimate_P_pi_from_perm2(
            I2, burn_in=burn_in, stride=stride, average_offsets=average_offsets,
            label_subsample=label_subsample, chunk=chunk
        )
        return dict(P=P, pi=pi, C=C, dt=stride, lifted=False, period=None)


def analyze_one_perm2_theory(
    I2: np.ndarray,
    *,
    meta,         # PTMeta
    ends,         # Ends
    stride: int = 3,
    period: int = 3,
    burn_in: Union[int, float] = 0.1,
    average_offsets: bool = True,
    label_subsample: Optional[np.ndarray] = None,
    chunk: int = 4000,
) -> Dict[str, Any]:
    """
    Drop-in: returns k and MFPT in both "per kernel-step" and "per underlying tick" units.
    Also returns committors/flux/current if you need later.
    """
    ker = estimate_theory_kernel_from_perm2(
        I2, stride=stride, period=period, burn_in=burn_in,
        average_offsets=average_offsets, label_subsample=label_subsample, chunk=chunk
    )
    P, pi, dt = ker["P"], ker["pi"], int(ker["dt"])

    # endpoints (lift if needed)
    if ker["lifted"]:
        R_nodes = int(meta.R)
        hot = lift_indices_for_set(R_nodes, period, ends.hot_idx)
        cold = lift_indices_for_set(R_nodes, period, ends.cold_idx)
    else:
        hot = np.asarray(ends.hot_idx, np.int64)
        cold = np.asarray(ends.cold_idx, np.int64)

    # TPT rates both directions
    tpt_hc = tpt_rate_flux(P, pi, hot, cold)
    tpt_ch = tpt_rate_flux(P, pi, cold, hot)

    # MFPT means both directions (same P)
    tau_hc = mean_mfpt_A_to_B(P, pi, hot, cold, start="piA")
    tau_ch = mean_mfpt_A_to_B(P, pi, cold, hot, start="piA")

    # convert units
    out = dict(
        dt=dt, lifted=ker["lifted"], period=ker["period"],
        P=P, pi=pi, C=ker["C"],
        hot_idx=hot, cold_idx=cold,

        # rates per kernel-step
        k_hc=tpt_hc["k"],
        k_ch=tpt_ch["k"],

        # rates per underlying tick (for comparing to raw time series)
        k_hc_per_tick=tpt_hc["k"] / dt,
        k_ch_per_tick=tpt_ch["k"] / dt,

        # mean MFPT in kernel-steps
        tau_hc=tau_hc,
        tau_ch=tau_ch,

        # mean MFPT in underlying ticks
        tau_hc_ticks=tau_hc * dt,
        tau_ch_ticks=tau_ch * dt,

        # keep committors/currents if you want later
        q_plus_hc=tpt_hc["q_plus"], q_minus_hc=tpt_hc["q_minus"],
        J_hc=tpt_hc["J"],
    )
    return out


def analyze_I_stack_theory(
    I_ts_stack: np.ndarray,
    sys,
    *,
    ends,                          # Ends (same object you already build)
    rids: Optional[np.ndarray] = None,
    stride: int = 3,
    period: int = 3,
    burn_in: Union[int, float] = 0.1,
    average_offsets: bool = True,
    label_subsample: Optional[np.ndarray] = None,
    chunk: int = 4000,
) -> Dict[str, Any]:
    """
    Theory analysis, but driven by I_ts_stack instead of heavy results objects.

    I_ts_stack: (Rdis, 2, T, N)  where N must equal meta.R = len(sys.beta).
    """
    meta = meta_from_sys(sys)

    I = np.asarray(I_ts_stack)
    if I.ndim != 4 or I.shape[1] != 2:
        raise ValueError("I_ts_stack must have shape (Rdis, 2, T, N).")
    Rdis, two, T, N = I.shape

    if meta.R != N:
        raise ValueError(f"sys.beta length meta.R={meta.R} must match stack N={N}.")

    if rids is None:
        rids = np.arange(Rdis, dtype=np.int64)
    else:
        rids = np.asarray(rids, dtype=np.int64)
        if rids.shape[0] != Rdis:
            raise ValueError("rids must have length Rdis.")

    out = dict(
        rid=[],
        dt=[],
        k_hc=[], k_ch=[], k_hc_per_tick=[], k_ch_per_tick=[],
        tau_hc=[], tau_ch=[], tau_hc_ticks=[], tau_ch_ticks=[],
        per_rid={},
        meta=meta,
        ends=ends,
        stride=int(stride),
        period=int(period),
        burn_in=burn_in,
        average_offsets=bool(average_offsets),
    )

    for i in range(Rdis):
        rid = int(rids[i])
        I2 = I[i].astype(np.int64, copy=False)  # (2,T,N)

        a = analyze_one_perm2_theory(
            I2,
            meta=meta,
            ends=ends,
            stride=stride,
            period=period,
            burn_in=burn_in,
            average_offsets=average_offsets,
            label_subsample=label_subsample,
            chunk=chunk,
        )

        out["rid"].append(rid)
        out["dt"].append(a["dt"])
        for k in ("k_hc","k_ch","k_hc_per_tick","k_ch_per_tick",
                  "tau_hc","tau_ch","tau_hc_ticks","tau_ch_ticks"):
            out[k].append(float(a[k]))
        out["per_rid"][rid] = a

    out["rid"] = np.asarray(out["rid"], np.int64)
    out["dt"]  = np.asarray(out["dt"],  np.int64)
    for k in ("k_hc","k_ch","k_hc_per_tick","k_ch_per_tick",
              "tau_hc","tau_ch","tau_hc_ticks","tau_ch_ticks"):
        out[k] = np.asarray(out[k], np.float64)

    return out



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

def time_reversed_kernel(P: np.ndarray, pi: np.ndarray, *, eps: float = 1e-300) -> np.ndarray:
    """
    P*_{u,v} = (pi_v P_{v,u}) / pi_u
    FIX: normalize nonzero rows even if some rows are zero.
    """
    P = np.asarray(P, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    R = P.shape[0]
    Pstar = np.zeros_like(P)

    for u in range(R):
        if pi[u] <= eps:
            continue
        Pstar[u, :] = (pi * P[:, u]) / pi[u]

    row = Pstar.sum(axis=1, keepdims=True)
    z = (row[:, 0] <= 0)
    if np.any(z):
        Pstar[z, :] = 0.0
        Pstar[z, z] = 1.0
        row = Pstar.sum(axis=1, keepdims=True)

    nz = ~z
    Pstar[nz, :] = Pstar[nz, :] / row[nz, :]
    return Pstar


def backward_committor_to_hot(P: np.ndarray, pi: np.ndarray,
                              hot_idx: np.ndarray, cold_idx: np.ndarray) -> np.ndarray:
    """
    q-(x) = Prob(last hit hot more recently than cold | at x),
    computed as forward committor on time-reversed chain P* with boundary:
      q-=1 on hot, q-=0 on cold.
    """
    Pstar = time_reversed_kernel(P, pi)
    Pstar = np.asarray(Pstar, dtype=np.float64)
    R = Pstar.shape[0]

    hot = np.zeros(R, dtype=bool);  hot[np.asarray(hot_idx, dtype=np.int64)] = True
    cold = np.zeros(R, dtype=bool); cold[np.asarray(cold_idx, dtype=np.int64)] = True
    free = ~(hot | cold)

    q = np.zeros(R, dtype=np.float64)
    q[hot] = 1.0
    q[cold] = 0.0

    free_idx = np.flatnonzero(free)
    if free_idx.size == 0:
        return q

    P_ff = Pstar[np.ix_(free, free)]
    rhs  = Pstar[np.ix_(free, hot)].sum(axis=1)   # target=hot with value 1
    A = np.eye(free_idx.size) - P_ff

    try:
        q_free = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        q_free = np.linalg.lstsq(A, rhs, rcond=None)[0]

    q[free_idx] = q_free
    return np.clip(q, 0.0, 1.0)

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
def reactive_flux_current_nonrev(P: np.ndarray, pi: np.ndarray,
                                 hot_idx: np.ndarray, cold_idx: np.ndarray) -> Dict[str, Any]:
    """
    Nonreversible TPT objects for A=hot, B=cold:
      q+ solves forward committor to cold before hot
      q- solves backward committor via time-reversed kernel
      f_uv = pi_u P_uv q-_u q+_v
      J = f - f^T
      k_AB = sum_{u in A} sum_v f_uv  (reactive flux out of A)
      rho_AB = sum_x pi_x q-_x q+_x    (reactive density)
      tpath ~ rho/k                    (mean reactive path duration, optional)
    """
    hot_idx  = np.asarray(hot_idx,  dtype=np.int64)
    cold_idx = np.asarray(cold_idx, dtype=np.int64)

    q_plus  = forward_committor_to_cold(P, hot_idx, cold_idx)
    q_minus = backward_committor_to_hot(P, pi, hot_idx, cold_idx)

    f = (pi[:, None] * P) * (q_minus[:, None]) * (q_plus[None, :])
    J = f - f.T

    k = float(f[hot_idx, :].sum())
    rho = float(np.sum(pi * q_minus * q_plus))
    tpath = rho / max(k, 1e-300)

    return dict(q_plus=q_plus, q_minus=q_minus, f=f, J=J, k=k, rho=rho, tpath=tpath)


def analyze_I_stack_tpt_macro(
    I_ts_stack: np.ndarray,
    sys,
    *,
    ends,                              # Ends object (hot_idx=cold? etc as you wish)
    rids: Optional[np.ndarray] = None,
    burn_in: Union[int, float] = 0.0,
    stride: int = 3,
    average_offsets: bool = True,
    label_subsample: Optional[np.ndarray] = None,
    chunk: int = 4000,
    store_f: bool = False,             # J is usually enough for plotting
) -> Dict[str, Any]:
    """
    Stack-driven replacement for analyze_results_list(... level="macro") focused on TPT current.

    Returns:
      out["per_rid"][rid]["J"]  (and q+/q-, k, etc.)
    """
    meta = meta_from_sys(sys)
    I = np.asarray(I_ts_stack)

    if I.ndim != 4 or I.shape[1] != 2:
        raise ValueError("I_ts_stack must have shape (Rdis, 2, T, N).")
    Rdis, _, _, N = I.shape
    if int(meta.R) != int(N):
        raise ValueError(f"sys.beta length meta.R={meta.R} must match stack N={N}.")

    if rids is None:
        rids = np.arange(Rdis, dtype=np.int64)
    else:
        rids = np.asarray(rids, dtype=np.int64)
        if rids.shape[0] != Rdis:
            raise ValueError("rids must have length Rdis.")

    out = dict(
        rid=rids.copy(),
        k=np.full(Rdis, np.nan, dtype=np.float64),
        rho=np.full(Rdis, np.nan, dtype=np.float64),
        tpath=np.full(Rdis, np.nan, dtype=np.float64),
        per_rid={},
        meta=meta,
        ends=ends,
        burn_in=burn_in,
        stride=int(stride),
        average_offsets=bool(average_offsets),
        level="macro",
    )

    for i in range(Rdis):
        rid = int(rids[i])
        I2 = I[i].astype(np.int64, copy=False)  # (2,T,N)

        P, pi, C = estimate_P_pi_from_perm2(
            I2,
            burn_in=burn_in,
            stride=stride,
            average_offsets=average_offsets,
            label_subsample=label_subsample,
            chunk=chunk
        )

        tpt = reactive_flux_current_nonrev(P, pi, ends.hot_idx, ends.cold_idx)

        rec = dict(
            P=P, pi=pi, C=C,
            q_plus=tpt["q_plus"], q_minus=tpt["q_minus"],
            J=tpt["J"],
            k=tpt["k"],
            rho=tpt["rho"],
            tpath=tpt["tpath"],
        )
        if store_f:
            rec["f"] = tpt["f"]

        out["k"][i] = tpt["k"]
        out["rho"][i] = tpt["rho"]
        out["tpath"][i] = tpt["tpath"]
        out["per_rid"][rid] = rec

    return out
