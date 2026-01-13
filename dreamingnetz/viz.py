import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# -----------------------
# ladder helpers (fix t)
# -----------------------
def _resolve_b(meta, t_sel):
    tg = np.asarray(meta.t_grid, float)
    if isinstance(t_sel, (int, np.integer)):
        if t_sel < 0 or t_sel >= tg.size:
            raise IndexError(f"t_sel index {t_sel} out of range [0,{tg.size})")
        return int(t_sel)
    return int(np.argmin(np.abs(tg - float(t_sel))))

def _ladder_indices(meta, t_sel, *, sort_by_T=True):
    b = _resolve_b(meta, t_sel)
    r0, r1 = int(meta.k_start[b]), int(meta.k_start[b + 1])
    r = np.arange(r0, r1, dtype=int)
    beta = np.asarray(meta.beta[r], float)
    T = 1.0 / beta
    if sort_by_T:
        ordT = np.argsort(T)
        r, T, beta = r[ordT], T[ordT], beta[ordT]
    return b, r, T, beta

def _edges_from_centers(x):
    x = np.asarray(x, float)
    if x.size == 1:
        dx = 1.0
        return np.array([x[0] - 0.5 * dx, x[0] + 0.5 * dx])
    mid = 0.5 * (x[1:] + x[:-1])
    e0 = x[0] - (mid[0] - x[0])
    eN = x[-1] + (x[-1] - mid[-1])
    return np.concatenate([[e0], mid, [eN]])

# -----------------------
# 1) |m_mu| heatmap (per rid)
# -----------------------
def plot_mabs_heatmap(
    f, folder, *, rid, t_sel,
    burn=0, thin=1,
    chain_avg="mean",   # "mean" or "median"
    time_avg="mean",    # "mean" or "median"
    ax=None, title=None,
):
    """
    Heatmap: x = Temperature (uneven pixels via pcolormesh), y = pattern index (mu_to_store),
    color = time-avg & chain-avg of |m_mu| for fixed t.

    Requires that you stored many/all mu's in m_sel; y will be meta.mu_to_store.
    """
    meta = f(folder, "meta", rid=rid)
    b, r, T, _beta = _ladder_indices(meta, t_sel, sort_by_T=True)

    m = f(folder, "m", rid=rid, mmap=True)  # (2, Tsamp, R, P_sel) typically
    if m.ndim != 4 or m.shape[0] != 2 or m.shape[2] != meta.R:
        raise ValueError(f"Unexpected m shape {m.shape}, expected (2,Tsamp,R,P_sel)")

    # slice fixed t ladder
    ms = m[:, burn::thin, r, :]          # (2, T', Kb, P_sel)
    ms = np.abs(ms)

    # average over chains
    if chain_avg == "mean":
        ms = ms.mean(axis=0)             # (T', Kb, P_sel)
    elif chain_avg == "median":
        ms = np.median(ms, axis=0)
    else:
        raise ValueError(chain_avg)

    # average over time
    if time_avg == "mean":
        A = ms.mean(axis=0)              # (Kb, P_sel)
    elif time_avg == "median":
        A = np.median(ms, axis=0)
    else:
        raise ValueError(time_avg)

    # y-axis = stored mu labels; sort them for nicer y
    mu = np.asarray(meta.mu_to_store, int)
    ord_mu = np.argsort(mu)
    mu = mu[ord_mu]
    A = A[:, ord_mu].T                   # (P_sel, Kb)

    x_edges = _edges_from_centers(T)
    y_edges = _edges_from_centers(mu.astype(float))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    pcm = ax.pcolormesh(x_edges, y_edges, A, shading="auto")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("pattern index μ (stored)")
    ax.set_title(title or f"|m_μ| heatmap (rid={rid}, t={meta.t_grid[b]:g})")
    plt.colorbar(pcm, ax=ax, label=r"$\langle |m_\mu| \rangle_{\mathrm{chains,time}}$")
    plt.xlim(0.1,2)
    return ax

# -----------------------
# 2) Parisi P(q) stacked histograms (per rid)
# -----------------------
def plot_Pq_stacked(
    f, folder, *, rid, t_sel,
    burn=0, thin=1,
    bins=120, q_range=(-1, 1),
    normalize_each=True,
    ax=None, title=None,
):
    """
    Ridge-style stacked histograms: one per temperature (fixed t ladder).
    q is taken from q01 timeseries at each slot r (i.e. each beta in the ladder).
    """
    meta = f(folder, "meta", rid=rid)
    b, r, T, _beta = _ladder_indices(meta, t_sel, sort_by_T=True)

    q = f(folder, "q01", rid=rid, mmap=True)  # (Tsamp, R)
    if q.ndim != 2 or q.shape[1] != meta.R:
        raise ValueError(f"Unexpected q01 shape {q.shape}, expected (Tsamp,R)")

    qs = q[burn::thin, :][:, r]  # (T', Kb)

    edges = np.linspace(q_range[0], q_range[1], bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, max(12, 0.18 * r.size)))

    # vertical offsets: bottom is low T (since T sorted increasing, bottom will be small T)
    for i in range(r.size):
        h, _ = np.histogram(qs[:, i], bins=edges, density=True)
        if normalize_each and h.max() > 0:
            h = h / h.max()
        y0 = i
        ax.plot(centers, y0 + h, lw=1.0)
        ax.fill_between(centers, y0, y0 + h, alpha=0.25)

    ax.set_yticks(np.arange(r.size))
    ax.set_yticklabels([f"T={Ti:.3g}" for Ti in T])
    ax.set_xlabel("q")
    ax.set_ylabel("temperature index (fixed t)")
    ax.set_title(title or f"Stacked P(q) (rid={rid}, t={meta.t_grid[b]:g})")
    ax.set_xlim(q_range)
    ax.grid(alpha=0.2)
    return ax

# -----------------------
# 3) Binder analysis for q (SG–P hint)
# -----------------------
def compute_q_binder_curve(
    f, folder, *, rid, t_sel,
    burn=0, thin=1, eps=1e-16,
):
    """
    Per disorder (rid): U4(q)(T) on the fixed-t ladder.
    Returns (T_sorted, U4_sorted).
    """
    meta = f(folder, "meta", rid=rid)
    _b, r, T, _beta = _ladder_indices(meta, t_sel, sort_by_T=True)

    q = f(folder, "q01", rid=rid, mmap=True)  # (Tsamp, R)
    qs = q[burn::thin, :][:, r]               # (T', Kb)

    q2 = np.mean(qs * qs, axis=0)
    q4 = np.mean((qs * qs) ** 2, axis=0)
    U4 = 1.0 - q4 / (3.0 * np.maximum(q2 * q2, eps))
    return T, U4

def analyze_q_binder(
    f, folder, *, rids, t_sel,
    burn=0, thin=1, eps=1e-16,
):
    """
    Robust aggregation across disorders:
      - compute U4 per disorder (thermal moments first)
      - aggregate across disorders using both mean±SEM and median+quantiles

    NOTE: For SG–P with a single N, Binder is suggestive, not definitive (no size crossing).
    """
    Ts = None
    curves = []
    used = []
    for rid in rids:
        try:
            T, U4 = compute_q_binder_curve(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin, eps=eps)
        except Exception as e:
            # skip incomplete/corrupt disorders
            continue
        if Ts is None:
            Ts = T
        else:
            if T.shape != Ts.shape or not np.allclose(T, Ts, rtol=0, atol=0):
                raise RuntimeError("Temperature grids differ across disorders (unexpected for fixed run).")
        curves.append(U4)
        used.append(rid)

    if not curves:
        raise RuntimeError("No valid disorders for binder analysis.")

    U = np.vstack(curves)  # (nR, Kb)
    nR = U.shape[0]

    mean = U.mean(axis=0)
    sem = U.std(axis=0, ddof=1) / np.sqrt(nR) if nR > 1 else np.zeros_like(mean)

    med = np.median(U, axis=0)
    lo = np.quantile(U, 0.16, axis=0)
    hi = np.quantile(U, 0.84, axis=0)

    return dict(T=Ts, U4_per_r=U, used_rids=used, mean=mean, sem=sem, median=med, q16=lo, q84=hi)

def plot_q_binder(res, *, ax=None, alpha=0.15, title=None, show_individual=True):
    T = res["T"]
    U = res["U4_per_r"]
    mean, sem = res["mean"], res["sem"]
    med, lo, hi = res["median"], res["q16"], res["q84"]
    nR = U.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    if show_individual:
        for i in range(nR):
            ax.plot(T, U[i], lw=1.0, alpha=alpha)

    ax.plot(T, mean, lw=2.5, label="mean")
    ax.fill_between(T, mean - sem, mean + sem, alpha=0.25, label="±SEM")

    ax.plot(T, med, lw=2.5, ls="--", label="median")
    ax.fill_between(T, lo, hi, alpha=0.20, label="16–84%")

    ax.set_xlabel("Temperature T")
    ax.set_ylabel(r"$U_4(q)=1-\langle q^4\rangle/(3\langle q^2\rangle^2)$")
    ax.set_title(title or f"Binder U4(q) (n_disorders={nR})")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import PowerNorm, Normalize

def _build_b_of_r(meta):
    b_of_r = np.empty(int(meta.R), dtype=np.int64)
    for b in range(int(meta.B)):
        r0, r1 = int(meta.k_start[b]), int(meta.k_start[b + 1])
        b_of_r[r0:r1] = b
    return b_of_r

def _y_from_beta(beta, y_axis):
    beta = np.asarray(beta, dtype=np.float64)
    if y_axis == "beta":
        return beta
    if y_axis == "T":
        if np.any(beta <= 0):
            raise ValueError("beta<=0 present; cannot plot T=1/beta.")
        return 1.0 / beta
    raise ValueError("y_axis must be 'beta' or 'T'")

def plot_pt_graph_avg_acceptance(
    f, folder, *,
    rids,
    min_attempts=1,
    add_horizontal=True,
    add_vertical=True,
    add_nodes=True,
    y_axis="beta",          # "beta" or "T"
    log_y=True,
    invert_y_for_T=False,
    cmap="viridis",
    gamma=0.5,              # PowerNorm gamma (smaller -> boosts low acceptances)
    title=None,
):
    """
    Plot 2D PT graph in (t, beta) or (t, T) with edge colors = acceptance averaged over disorders.

    Inputs:
      f: your loader
      folder: run root containing rXXX/
      rids: iterable of disorder ids to average over
    """
    rids = list(rids)
    if len(rids) == 0:
        raise ValueError("rids is empty.")

    # --- meta (from first rid) ---
    meta0 = f(folder, "meta", rid=rids[0])
    b_of_r = _build_b_of_r(meta0)
    t_of_r = np.asarray(meta0.t_grid, dtype=np.float64)[b_of_r]
    y_of_r = _y_from_beta(np.asarray(meta0.beta, dtype=np.float64), y_axis=y_axis)

    edge_list = np.asarray(meta0.edge_list, dtype=np.int64)
    E = int(edge_list.shape[0])

    # --- aggregate acceptances ---
    # horizontal: accepted counts (2, R-1); attempts ~ n_samples per chain per interface
    sum_h_acc = np.zeros((2, meta0.R - 1), dtype=np.float64)
    sum_h_att = np.zeros((2, meta0.R - 1), dtype=np.float64)

    # vertical: accepted/attempted (2, E)
    sum_v_acc = np.zeros((2, E), dtype=np.float64)
    sum_v_att = np.zeros((2, E), dtype=np.float64)

    h_mask_valid = None

    for rid in rids:
        meta = f(folder, "meta", rid=rid)
        acc  = f(folder, "acc",  rid=rid)

        if h_mask_valid is None:
            h_mask_valid = np.asarray(acc.h_mask_valid, dtype=bool)
        else:
            if not np.array_equal(h_mask_valid, np.asarray(acc.h_mask_valid, dtype=bool)):
                raise RuntimeError("h_mask_valid differs across disorders (unexpected).")

        # horizontal
        h_acc = np.asarray(acc.h_accepted, dtype=np.float64)  # (2, R-1)
        sum_h_acc += h_acc
        n = float(meta.n_samples)
        sum_h_att += n  # broadcast to (2, R-1)

        # vertical
        if E > 0:
            v_acc = np.asarray(acc.v_accepted, dtype=np.float64)
            v_att = np.asarray(acc.v_attempted, dtype=np.float64)
            sum_v_acc += v_acc
            sum_v_att += v_att

    # attempt-weighted over chains too
    h_den = sum_h_att.sum(axis=0)                       # (R-1,)
    h_num = sum_h_acc.sum(axis=0)                       # (R-1,)
    a_h = np.full(meta0.R - 1, np.nan, dtype=np.float64)
    m_h = (h_den >= float(min_attempts)) & h_mask_valid
    a_h[m_h] = h_num[m_h] / (10*h_den[m_h])

    if E > 0:
        v_den = sum_v_att.sum(axis=0)                   # (E,)
        v_num = sum_v_acc.sum(axis=0)                   # (E,)
        a_v = np.full(E, np.nan, dtype=np.float64)
        m_v = v_den >= float(min_attempts)
        a_v[m_v] = v_num[m_v] / v_den[m_v]
    else:
        a_v = np.empty((0,), dtype=np.float64)

    # --- build segments ---
    segs_h, col_h = [], []
    if add_horizontal:
        for b in range(int(meta0.B)):
            r0, r1 = int(meta0.k_start[b]), int(meta0.k_start[b + 1])
            if (r1 - r0) < 2:
                continue
            t = float(meta0.t_grid[b])
            yb = _y_from_beta(meta0.beta[r0:r1], y_axis=y_axis)
            for k in range((r1 - r0) - 1):
                iface = (r0 + k)  # interface index in 0..R-2
                if np.isfinite(a_h[iface]):
                    segs_h.append([[t, yb[k]], [t, yb[k + 1]]])
                    col_h.append(a_h[iface])

    segs_h = np.asarray(segs_h, dtype=np.float64) if segs_h else np.empty((0, 2, 2))
    col_h  = np.asarray(col_h,  dtype=np.float64) if col_h  else np.empty((0,))

    segs_v, col_v = [], []
    if add_vertical and E > 0:
        r1 = edge_list[:, 0].astype(np.int64, copy=False)
        r2 = edge_list[:, 1].astype(np.int64, copy=False)
        b1 = b_of_r[r1]
        b2 = b_of_r[r2]
        t1 = np.asarray(meta0.t_grid, float)[b1]
        t2 = np.asarray(meta0.t_grid, float)[b2]
        y1 = _y_from_beta(np.asarray(meta0.beta, float)[r1], y_axis=y_axis)
        y2 = _y_from_beta(np.asarray(meta0.beta, float)[r2], y_axis=y_axis)

        for e in range(E):
            if np.isfinite(a_v[e]):
                segs_v.append([[t1[e], y1[e]], [t2[e], y2[e]]])
                col_v.append(a_v[e])

    segs_v = np.asarray(segs_v, dtype=np.float64) if segs_v else np.empty((0, 2, 2))
    col_v  = np.asarray(col_v,  dtype=np.float64) if col_v  else np.empty((0,))

    # --- plotting ---
    norm = PowerNorm(gamma=float(gamma), vmin=0.0, vmax=1.0, clip=True)
    fig, ax = plt.subplots(figsize=(8.0, 11.0))

    if segs_h.shape[0] > 0:
        lc_h = LineCollection(segs_h, array=col_h, cmap=cmap, norm=norm)
        lc_h.set_linewidth(3.0)
        ax.add_collection(lc_h)

    if segs_v.shape[0] > 0:
        lc_v = LineCollection(segs_v, array=col_v, cmap=cmap, norm=norm)
        lc_v.set_linewidth(3.0)
        ax.add_collection(lc_v)

    if add_nodes:
        ax.scatter(t_of_r, y_of_r, s=18, zorder=10, color="black")

    ax.autoscale()
    ax.set_xlabel("t")
    ax.set_ylabel("beta" if y_axis == "beta" else "T = 1/beta")

    if log_y:
        ax.set_yscale("log")
    if y_axis == "T" and invert_y_for_T:
        ax.invert_yaxis()

    ax.set_title(title or f"2D PT graph: avg acceptance over disorders (n={len(rids)})")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label("acceptance (avg over disorders; vertical attempt-weighted; horizontal ~ per-sample)")

    plt.tight_layout()
    plt.show()
    return ax

# ============================
# Magnetization analysis tools
# Append to viz.py
# ============================

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# ladder helpers (fix t)
# -----------------------
def _resolve_b(meta, t_sel):
    tg = np.asarray(meta.t_grid, float)
    if isinstance(t_sel, (int, np.integer)):
        if t_sel < 0 or t_sel >= tg.size:
            raise IndexError(f"t_sel index {t_sel} out of range [0,{tg.size})")
        return int(t_sel)
    return int(np.argmin(np.abs(tg - float(t_sel))))

def _ladder_indices(meta, t_sel, *, sort_by_T=True):
    b = _resolve_b(meta, t_sel)
    r0, r1 = int(meta.k_start[b]), int(meta.k_start[b + 1])
    r = np.arange(r0, r1, dtype=int)
    beta = np.asarray(meta.beta[r], float)
    T = 1.0 / beta
    if sort_by_T:
        ordT = np.argsort(T)
        r, T, beta = r[ordT], T[ordT], beta[ordT]
    return b, r, T, beta

def _edges_from_centers(x):
    x = np.asarray(x, float)
    if x.size == 1:
        dx = 1.0
        return np.array([x[0] - 0.5 * dx, x[0] + 0.5 * dx])
    mid = 0.5 * (x[1:] + x[:-1])
    e0 = x[0] - (mid[0] - x[0])
    eN = x[-1] + (x[-1] - mid[-1])
    return np.concatenate([[e0], mid, [eN]])

# -----------------------
# data extraction helpers
# -----------------------
def _load_m_ladder(f, folder, *, rid, t_sel, burn=0, thin=1):
    """
    Returns:
      meta, b, T_sorted, r_sorted,
      m : (2, S, Kb, P_sel) float
    """
    meta = f(folder, "meta", rid=rid)
    b, r, T, _beta = _ladder_indices(meta, t_sel, sort_by_T=True)
    m = f(folder, "m", rid=rid, mmap=True)  # expected (2, Tsamp, R, P_sel)
    if m.ndim != 4 or m.shape[0] != 2 or m.shape[2] != meta.R:
        raise ValueError(f"Unexpected m shape {m.shape}, expected (2,Tsamp,R,P_sel)")
    ms = m[:, burn::thin, r, :]  # (2, S, Kb, P_sel)
    if ms.shape[1] < 2:
        raise ValueError("Too few samples after burn/thin for magnetization analysis.")
    return meta, b, T, r, ms

def _load_q_ladder(f, folder, *, rid, t_sel, burn=0, thin=1):
    meta = f(folder, "meta", rid=rid)
    b, r, T, _beta = _ladder_indices(meta, t_sel, sort_by_T=True)
    q = f(folder, "q01", rid=rid, mmap=True)  # (Tsamp, R)
    if q.ndim != 2 or q.shape[1] != meta.R:
        raise ValueError(f"Unexpected q01 shape {q.shape}, expected (Tsamp,R)")
    qs = q[burn::thin, :][:, r]  # (S, Kb)
    if qs.shape[0] < 2:
        raise ValueError("Too few samples after burn/thin for q analysis.")
    return meta, b, T, r, qs

def _load_E_ladder(f, folder, *, rid, t_sel, burn=0, thin=1, chain_avg=True):
    meta = f(folder, "meta", rid=rid)
    b, r, T, _beta = _ladder_indices(meta, t_sel, sort_by_T=True)
    E = f(folder, "E", rid=rid, mmap=True)  # (2, Tsamp, R)
    if E.ndim != 3 or E.shape[0] != 2 or E.shape[2] != meta.R:
        raise ValueError(f"Unexpected E shape {E.shape}, expected (2,Tsamp,R)")
    Es = E[:, burn::thin, r]  # (2, S, Kb)
    if Es.shape[1] < 2:
        raise ValueError("Too few samples after burn/thin for energy analysis.")
    if chain_avg:
        return meta, b, T, r, Es.mean(axis=0)  # (S,Kb)
    return meta, b, T, r, Es  # (2,S,Kb)

# -----------------------
# derived magnetization stats
# -----------------------
def _m_stats(ms, *, eps=1e-16):
    """
    ms: (2, S, Kb, P_sel)
    Returns dict with arrays shaped (2,S,Kb) or (S,Kb) where noted.
    """
    abs_m = np.abs(ms)
    # per-chain, per-sample
    m_max = abs_m.max(axis=-1)                           # (2,S,Kb)
    m_sq = ms * ms
    m_norm2 = m_sq.sum(axis=-1)                          # (2,S,Kb)
    m_norm = np.sqrt(np.maximum(m_norm2, 0.0))           # (2,S,Kb)
    ratio = m_max / np.maximum(m_norm, eps)              # (2,S,Kb)

    # winner index & sign (per chain)
    win_pos = abs_m.argmax(axis=-1).astype(np.int64)      # (2,S,Kb) position in stored mu axis
    # sign of the *raw* m at the winning component
    # gather: m[c,s,k, win_pos[c,s,k]]
    c, s, k = np.indices(win_pos.shape)
    win_val = ms[c, s, k, win_pos]                        # (2,S,Kb)
    win_sign = np.sign(win_val).astype(np.int8)           # (2,S,Kb)

    # participation ratio / N_eff
    # w_mu = m^2 / sum m^2 ; N_eff = 1/sum w^2
    denom = np.maximum(m_norm2[..., None], eps)           # (2,S,Kb,1)
    w = m_sq / denom                                      # (2,S,Kb,P_sel)
    Neff = 1.0 / np.maximum((w * w).sum(axis=-1), eps)    # (2,S,Kb)

    # cosine similarity between chains (per sample, per T)
    # uses raw m (not abs); if you want sign-invariant compare, take abs later.
    m0 = ms[0]                                            # (S,Kb,P_sel)
    m1 = ms[1]
    dot = (m0 * m1).sum(axis=-1)                          # (S,Kb)
    n0 = np.sqrt(np.maximum((m0 * m0).sum(axis=-1), eps))
    n1 = np.sqrt(np.maximum((m1 * m1).sum(axis=-1), eps))
    cos = dot / np.maximum(n0 * n1, eps)                  # (S,Kb)

    return dict(
        abs_m=abs_m,
        m_max=m_max,
        m_norm=m_norm,
        m_norm2=m_norm2,
        ratio=ratio,
        win_pos=win_pos,
        win_sign=win_sign,
        Neff=Neff,
        cos=cos,
    )

def _time_reduce(x, stat="mean"):
    if stat == "mean":
        return np.mean(x, axis=0)
    if stat == "median":
        return np.median(x, axis=0)
    raise ValueError(stat)

def _entropy_from_winner(win_mu_labels, P_sel, *, eps=1e-16):
    """
    win_mu_labels: 1D array of mu labels (not positions).
    """
    # map labels to 0..P_sel-1 by sorting unique stored labels is external; here assume labels are in meta.mu_to_store
    # so we can just bincount after remapping; caller provides already positions or mapped indices.
    counts = np.bincount(win_mu_labels, minlength=P_sel).astype(np.float64)
    p = counts / np.maximum(counts.sum(), 1.0)
    m = p > 0
    H = -np.sum(p[m] * np.log(p[m] + eps))
    Hn = H / np.log(max(P_sel, 2))  # normalized entropy in [0,1]
    return H, Hn


# -----------------------
# 2) winner entropy + histogram
# -----------------------
def plot_winner_entropy_vs_T(
    f, folder, *, rid, t_sel,
    burn=0, thin=1,
    ax=None, title=None,
):
    """
    Winner index entropy (over time and both chains) for each temperature.
    Uses winner positions in stored mu-axis; if mu_to_store is subset, entropy is over that subset.
    """
    meta, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
    P_sel = int(meta.mu_to_store.size)
    st = _m_stats(ms)

    H = np.zeros(T.shape[0], dtype=np.float64)
    Hn = np.zeros_like(H)
    for k in range(T.shape[0]):
        # concatenate both chains over time: positions 0..P_sel-1
        w = np.concatenate([st["win_pos"][0, :, k], st["win_pos"][1, :, k]], axis=0)
        hk, hnk = _entropy_from_winner(w, P_sel)
        H[k], Hn[k] = hk, hnk

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(T, Hn, marker="o")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("normalized winner entropy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title or f"Winner entropy vs T (rid={rid}, t={meta.t_grid[b]:g})")
    ax.grid(alpha=0.25)
    return ax

def plot_winner_hist_at_T(
    f, folder, *, rid, t_sel, T_target,
    burn=0, thin=1,
    ax=None, title=None,
):
    """
    Histogram of winner μ* at a single temperature on the fixed-t ladder (nearest T).
    """
    meta, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
    st = _m_stats(ms)
    k = int(np.argmin(np.abs(T - float(T_target))))
    w = np.concatenate([st["win_pos"][0, :, k], st["win_pos"][1, :, k]], axis=0)
    counts = np.bincount(w, minlength=int(meta.mu_to_store.size))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(np.arange(counts.size), counts)
    ax.set_xlabel("stored mu position (0..P_sel-1)")
    ax.set_ylabel("count")
    ax.set_title(title or f"Winner histogram at T≈{T[k]:.3g} (rid={rid}, t={meta.t_grid[b]:g})")
    ax.grid(alpha=0.2)
    return ax

# -----------------------
# 3) time×T heatmap of m_max (proxy for “flow blobs”)
# -----------------------
def plot_mmax_time_T_heatmap(
    f, folder, *, rid, t_sel,
    burn=0, thin=1,
    chain_avg="mean",  # "mean" or "median"
    ax=None, title=None,
):
    """
    Heatmap: x=Temperature (uneven pixels), y=time index, color=chain-avg m_max at each (time,T).
    """
    meta, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
    st = _m_stats(ms)
    mmax = st["m_max"]  # (2,S,Kb)

    if chain_avg == "mean":
        A = mmax.mean(axis=0)     # (S,Kb)
    elif chain_avg == "median":
        A = np.median(mmax, axis=0)
    else:
        raise ValueError(chain_avg)

    x_edges = _edges_from_centers(T)
    y_edges = np.arange(A.shape[0] + 1, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    pcm = ax.pcolormesh(x_edges, y_edges, A, shading="auto")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("time index (post burn/thin)")
    ax.set_title(title or f"m_max time×T heatmap (rid={rid}, t={meta.t_grid[b]:g})")
    plt.colorbar(pcm, ax=ax, label=r"$m_{\max}$ (chain-avg)")
    return ax



# -----------------------
# 5) stacked histograms of m_max
# -----------------------
def plot_mmax_stacked_hist(
    f, folder, *, rid, t_sel,
    burn=0, thin=1,
    bins=60, m_range=(0.0, 1.0),
    normalize_each=True,
    ax=None, title=None,T_range=(0,0)
):
    """
    Ridge-style stacked histograms of m_max, one per temperature (fixed t ladder).
    Uses chain-averaged m_max samples per time point.
    """
    meta, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
    st = _m_stats(ms)

    mmax = st["m_max"].mean(axis=0)  # (S,Kb)
    edges = np.linspace(m_range[0], m_range[1], bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, max(4, 0.18 * T.size)))

    for k in range(T_range[0],T.size-T_range[1]):
        h, _ = np.histogram(mmax[:, k], bins=edges, density=True)
        if normalize_each and h.max() > 0:
            h = h / h.max()
        y0 = k
        ax.plot(centers, y0 + h, lw=1.0)
        ax.fill_between(centers, y0, y0 + h, alpha=0.25)

    ax.set_yticks(np.arange(T.size))
    ax.set_yticklabels([f"T={Ti:.3g}" for Ti in T])
    ax.set_xlabel(r"$m_{\max}$")
    ax.set_ylabel("temperature index (fixed t)")
    ax.set_title(title or f"Stacked P(m_max) (rid={rid}, t={meta.t_grid[b]:g})")
    ax.set_xlim(m_range)
    ax.grid(alpha=0.2)
    return ax

# -----------------------
# 6) joint scatter/hexbin views
# -----------------------
def plot_joint_mmax_q(
    f, folder, *, rid, t_sel, T_target,gridsize,
    burn=0, thin=1,
    max_points=30000,
    ax=None, title=None,
):
    """
    Joint view at one temperature: (q, m_max).
    """
    meta_m, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
    meta_q, b2, T2, r2, qs = _load_q_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
    if not np.allclose(T, T2):
        raise RuntimeError("Internal mismatch: T grids differ between m and q loaders.")

    st = _m_stats(ms)
    mmax = st["m_max"].mean(axis=0)   # (S,Kb)

    k = int(np.argmin(np.abs(T - float(T_target))))
    x = qs[:, k]
    y = mmax[:, k]

    # subsample
    n = x.shape[0]
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        x, y = x[idx], y[idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.hexbin(x, y, gridsize=gridsize, mincnt=1,extent=(-1, 1, 0, 1))
    ax.set_xlabel("q")
    ax.set_ylabel(r"$m_{\max}$ (chain-avg)")
    ax.set_title(title or f"(q, m_max) at T≈{T[k]:.3g} (rid={rid}, t={meta_m.t_grid[b]:g})")
    return ax

def plot_joint_mmax_E(
    f, folder, *, rid, t_sel, T_target,gridsize,
    burn=0, thin=1,
    max_points=30000,
    ax=None, title=None,
):
    """
    Joint view at one temperature: (E, m_max).
    Uses chain-averaged energy per time point.
    """
    meta_m, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
    meta_E, b2, T2, r2, Es = _load_E_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin, chain_avg=True)
    if not np.allclose(T, T2):
        raise RuntimeError("Internal mismatch: T grids differ between m and E loaders.")

    st = _m_stats(ms)
    mmax = st["m_max"].mean(axis=0)   # (S,Kb)

    k = int(np.argmin(np.abs(T - float(T_target))))
    x = Es[:, k]
    y = mmax[:, k]

    n = x.shape[0]
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        x, y = x[idx], y[idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.hexbin(x, y, gridsize=gridsize, mincnt=1)
    ax.set_xlabel("E (chain-avg)")
    ax.set_ylabel(r"$m_{\max}$ (chain-avg)")
    ax.set_title(title or f"(E, m_max) at T≈{T[k]:.3g} (rid={rid}, t={meta_m.t_grid[b]:g})")
    return ax

# -----------------------
# 7) robust bands across disorders (for m_max)
# -----------------------
def analyze_mmax_across_disorders(
    f, folder, *, rids, t_sel,
    burn=0, thin=1,
):
    """
    For each rid: compute per-T median and 16/84% of chain-avg m_max over time.
    Then aggregate across disorders: median-of-medians + (16,84)% across disorders,
    plus mean±SEM for reference.

    Returns dict with:
      T, per_rid_median, per_rid_q16, per_rid_q84,
      median, q16, q84, mean, sem, used_rids
    """
    rids = list(rids)
    per_med = []
    per_q16 = []
    per_q84 = []
    used = []
    Tref = None

    for rid in rids:
        try:
            meta, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
            st = _m_stats(ms)
            mmax = st["m_max"].mean(axis=0)  # (S,Kb)
            med = np.mean(mmax, axis=0)
            q16 = np.quantile(mmax, 0.16, axis=0)
            q84 = np.quantile(mmax, 0.84, axis=0)
        except Exception:
            continue

        if Tref is None:
            Tref = T
        else:
            if T.shape != Tref.shape or not np.allclose(T, Tref, rtol=0, atol=0):
                raise RuntimeError("Temperature grids differ across disorders (unexpected for a fixed run).")

        per_med.append(med)
        per_q16.append(q16)
        per_q84.append(q84)
        used.append(rid)

    if not per_med:
        raise RuntimeError("No valid disorders for m_max aggregation.")

    per_med = np.vstack(per_med)
    per_q16 = np.vstack(per_q16)
    per_q84 = np.vstack(per_q84)
    nR = per_med.shape[0]

    mean = per_med.mean(axis=0)
    sem = per_med.std(axis=0, ddof=1) / np.sqrt(nR) if nR > 1 else np.zeros_like(mean)

    med = np.median(per_med, axis=0)
    lo = np.quantile(per_med, 0.16, axis=0)
    hi = np.quantile(per_med, 0.84, axis=0)

    return dict(
        T=Tref, used_rids=used,
        per_rid_median=per_med, per_rid_q16=per_q16, per_rid_q84=per_q84,
        median=med, q16=lo, q84=hi, mean=mean, sem=sem,
    )

def plot_mmax_across_disorders(res,t_sel, *, ax=None, alpha=0.25, title=None, show_individual=True):
    T = res["T"]
    M = res["per_rid_median"]
    nR = M.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    if show_individual:
        for i in range(nR):
            ax.plot(T, M[i], lw=1.0, alpha=alpha)

    ax.plot(T, res["mean"], lw=2.5, label="mean of per-rid medians")
    ax.fill_between(T, res["mean"] - res["sem"], res["mean"] + res["sem"], alpha=0.20, label="±SEM")

    ax.plot(T, res["median"], lw=2.5, ls="--", label="median of per-rid medians")
    ax.fill_between(T, res["q16"], res["q84"], alpha=0.15, label="16–84% across disorders")

    ax.set_xlabel("Temperature T")
    ax.set_ylabel(r"$m_{\max}$ (chain-avg, time-median per rid)")
    ax.set_title(title or f"m_max across disorders (n={nR}), t = {t_sel}")
    ax.grid(alpha=0.25)
    ax.legend()
    return ax

# -----------------------
# 8) simple autocorrelation / tau_int proxy for m_max
# -----------------------
def _acf_fft(x):
    x = np.asarray(x, float)
    x = x - np.mean(x)
    n = x.size
    nfft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(fx * np.conj(fx), n=nfft)[:n]
    ac /= np.arange(n, 0, -1)
    return ac / max(ac[0], 1e-30)

def tau_int_initial_positive(x, *, max_lag=2000):
    """
    Lightweight τ_int estimator: 1 + 2 * sum_{t=1}^{M} ρ(t), stop at first negative ρ or max_lag.
    This is *not* as robust as your Geyer/BM machinery, but useful as a quick magnetization diagnostic.
    """
    x = np.asarray(x, float)
    if x.size < 8:
        return np.nan
    acf = _acf_fft(x)
    M = min(int(max_lag), acf.size - 1)
    s = 0.0
    for t in range(1, M + 1):
        if acf[t] <= 0:
            break
        s += acf[t]
    return 1.0 + 2.0 * s



# ==========================================================
# Drop-in replacements / additions for magnetization viz
# Paste BELOW the previous block in viz.py (overrides names)
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt


def plot_m_summary_vs_T(
    f, folder, *, rid, t_sel,
    burn=0, thin=1,
    time_stat="mean",
    ax=None, title=None,
):
    """
    NEW layout:
      - 2 subplots (stacked):
          top: <N_eff>(T)
          bottom: all others (<m_max>, <||m||>, <ratio>, <cos>, P(agree))
    """
    meta, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
    st = _m_stats(ms)

    mmax  = _time_reduce(st["m_max"].mean(axis=0), stat=time_stat)       # (Kb,)
    mnorm = _time_reduce(st["m_norm"].mean(axis=0), stat=time_stat)
    ratio = _time_reduce(st["ratio"].mean(axis=0), stat=time_stat)
    neff  = _time_reduce(st["Neff"].mean(axis=0), stat=time_stat)
    cos   = _time_reduce(st["cos"], stat=time_stat)

    agree = (st["win_pos"][0] == st["win_pos"][1])                       # (S,Kb)
    agree_p = _time_reduce(agree.astype(np.float64), stat=time_stat)

    if ax is None:
        fig, (ax1, ax2,ax3) = plt.subplots(
            3, 1, figsize=(7, 6), sharex=True,
            gridspec_kw=dict(height_ratios=[1, 2, 1])
        )
    else:
        # allow passing (ax1, ax2)
        ax1, ax2,ax3 = ax

    ax1.plot(T, neff, marker="o", label=r"$\langle N_{\mathrm{eff}}\rangle$")
    ax1.set_ylabel(r"$\langle N_{\mathrm{eff}}\rangle$")

    ax1.grid(alpha=0.25)
    ax1.legend()

    ax2.plot(T, mmax,  marker="o", label=r"$\langle m_{\max}\rangle$")
    ax2.plot(T, mnorm, marker="o", label=r"$\langle \|m\|\rangle$")
    ax2.plot(T, ratio, marker="o", label=r"$\langle m_{\max}/\|m\|\rangle$")
    ax2.plot(T, agree_p, marker="o", label=r"$\Pr(\mu^*_0=\mu^*_1)$")

    ax2.grid(alpha=0.25)
    ax2.legend()

    ax3.plot(T, cos,   marker="o", label=r"$\langle \cos\theta\rangle$")
    ax3.set_ylabel(r"$\langle \cos\theta\rangle$")

    ax3.set_yscale('log')
    ax3.set_xlabel("Temperature T")

    ax3.grid(alpha=0.25)
    ax3.legend()
    if title is None:
        title = f"Magnetization summary vs T (rid={rid}, t={meta.t_grid[b]:g})"
    ax1.set_title(title)

    return (ax1, ax2)


def plot_switching_rates_vs_T(
    f, folder, *, rid, t_sel,
    burn=0, thin=1,
    ax=None, title=None,
):
    """
    NEW layout:
      - 2 subplots (stacked):
          top: winner switching rate
          bottom: winner sign-flip rate
    """
    meta, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
    st = _m_stats(ms)

    win = st["win_pos"]    # (2,S,Kb)
    sgn = st["win_sign"]   # (2,S,Kb)

    def rate(x):
        return np.mean(x[1:] != x[:-1], axis=0)

    sw = 0.5 * (rate(win[0]) + rate(win[1]))  # (Kb,)
    sf = 0.5 * (rate(sgn[0]) + rate(sgn[1]))

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    else:
        ax1, ax2 = ax

    ax1.plot(T, sw, marker="o")
    ax1.set_ylabel("winner switch rate")
    ax1.grid(alpha=0.25)

    ax2.plot(T, sf, marker="o")
    ax2.set_xlabel("Temperature T")
    ax2.set_ylabel("winner sign-flip rate")
    ax2.grid(alpha=0.25)

    if title is None:
        title = f"Pattern/sign switching vs T (rid={rid}, t={meta.t_grid[b]:g})"
    ax1.set_title(title)

    return (ax1, ax2)


# -----------------------
# Extra “control” helpers for m_max at fixed temperature index k
# -----------------------
def plot_mmax_timeseries_at_k(
    f, folder, *, rid, t_sel, k,
    burn=0, thin=1,
    chain_avg="mean",     # "mean" or "median"
    ax=None, title=None,
):
    """
    Plot m_max vs time at a specific temperature index k (in the fixed-t ladder, T sorted ascending).
    """
    meta, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
    st = _m_stats(ms)
    mmax = st["m_max"]  # (2,S,Kb)

    k = int(k)
    if k < 0 or k >= T.size:
        raise IndexError(f"k={k} out of range [0,{T.size})")

    if chain_avg == "mean":
        y = mmax[:, :, k].mean(axis=0)   # (S,)
    elif chain_avg == "median":
        y = np.median(mmax[:, :, k], axis=0)
    else:
        raise ValueError(chain_avg)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.arange(y.size), y)
    ax.set_xlabel("time index (post burn/thin)")
    ax.set_ylabel(r"$m_{\max}$")
    ax.set_title(title or f"m_max timeseries (rid={rid}, t={meta.t_grid[b]:g}, k={k}, T={T[k]:.3g})")
    ax.grid(alpha=0.25)
    return ax


def plot_mmax_hist_at_k(
    f, folder, *, rid, t_sel, k,
    burn=0, thin=1,
    bins=60, m_range=(0.0, 1.0),
    chain_avg="mean",
    density=True,
    ax=None, title=None,
):
    """
    Histogram of m_max at a specific temperature index k.
    """
    meta, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
    st = _m_stats(ms)
    mmax = st["m_max"]  # (2,S,Kb)

    k = int(k)
    if k < 0 or k >= T.size:
        raise IndexError(f"k={k} out of range [0,{T.size})")

    if chain_avg == "mean":
        y = mmax[:, :, k].mean(axis=0)  # (S,)
    elif chain_avg == "median":
        y = np.median(mmax[:, :, k], axis=0)
    else:
        raise ValueError(chain_avg)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(y, bins=bins, range=m_range, density=density)
    ax.set_xlabel(r"$m_{\max}$")
    ax.set_ylabel("density" if density else "count")
    ax.set_title(title or f"P(m_max) at (rid={rid}, t={meta.t_grid[b]:g}, k={k}, T={T[k]:.3g})")
    ax.grid(alpha=0.25)
    return ax


def magnetization_report(
    f, folder, *,
    rid,
    t_sel,
    burn=0, thin=1,
    rids_for_aggregate=None,
    T_targets=None,
):
    """
    Same as before, but:
      - NO tau plot (removed)
      - uses updated summary/switching plot layouts
    """
    meta = f(folder, "meta", rid=rid)
    b = _resolve_b(meta, t_sel)
    t_val = float(meta.t_grid[b])

    # pick default T_targets if not given
    meta, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
    if T_targets is None:
        ks = [0, int((T.size - 1) * 0.5), T.size - 1]
        T_targets = [float(T[k]) for k in ks]

    # per-rid suite
    plot_m_summary_vs_T(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
    plot_winner_entropy_vs_T(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
    plot_switching_rates_vs_T(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
    plot_mmax_time_T_heatmap(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
    plot_mmax_stacked_hist(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)

    for Tt in T_targets:
        plot_joint_mmax_q(f, folder, rid=rid, t_sel=t_sel, T_target=Tt, burn=burn, thin=thin,gridsize = 40)
        plot_joint_mmax_E(f, folder, rid=rid, t_sel=t_sel, T_target=Tt, burn=burn, thin=thin,gridsize = 40)

    if rids_for_aggregate is not None:
        res = analyze_mmax_across_disorders(f, folder, rids=rids_for_aggregate, t_sel=t_sel, burn=burn, thin=thin)
        plot_mmax_across_disorders(res, title=f"m_max across disorders (t={t_val:g})")
        return res

    return None


# ==========================================================
# Disorder-aggregated magnetization plots
# Append to viz.py
# Requires the helpers already defined earlier in viz.py:
#   _load_m_ladder, _load_q_ladder, _load_E_ladder, _m_stats,
#   _resolve_b, _edges_from_centers
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt


# -----------------------
# generic aggregators
# -----------------------
def _agg_1d_across_rids(Y):
    """
    Y: (nR, K) array
    returns dict with mean/sem and robust median bands (q16/q84).
    """
    Y = np.asarray(Y, float)
    nR = Y.shape[0]
    mean = np.nanmean(Y, axis=0)
    sem = (np.nanstd(Y, axis=0, ddof=1) / np.sqrt(nR)) if nR > 1 else np.zeros_like(mean)
    med = np.nanmedian(Y, axis=0)
    q16 = np.nanquantile(Y, 0.16, axis=0)
    q84 = np.nanquantile(Y, 0.84, axis=0)
    return dict(Y=Y, nR=nR, mean=mean, sem=sem, median=med, q16=q16, q84=q84)

def _check_T_consistent(Tref, T):
    if T.shape != Tref.shape or not np.allclose(T, Tref, rtol=0, atol=0):
        raise RuntimeError("Temperature grids differ across disorders (unexpected for a fixed run).")

def _time_reduce_axis0(x, stat="median"):
    if stat == "median":
        return np.median(x, axis=0)
    if stat == "mean":
        return np.mean(x, axis=0)
    raise ValueError(stat)


# ==========================================================
# 1) plot_m_summary_vs_T aggregated
# ==========================================================
def analyze_m_summary_across_disorders(
    f, folder, *, rids, t_sel, burn=0, thin=1, time_stat="median"
):
    """
    Per rid -> per-T curves:
      neff, mmax, mnorm, ratio, cos, agree_p
    Then aggregate across rids robustly.
    """
    rids = list(rids)
    Tref = None

    neff_list = []
    mmax_list = []
    mnorm_list = []
    ratio_list = []
    cos_list  = []
    agree_list = []
    used = []

    for rid in rids:
        try:
            meta, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
            st = _m_stats(ms)

            # chain-avg time series (S,K)
            mmax_ts  = st["m_max"].mean(axis=0)
            mnorm_ts = st["m_norm"].mean(axis=0)
            ratio_ts = st["ratio"].mean(axis=0)
            neff_ts  = st["Neff"].mean(axis=0)
            cos_ts   = st["cos"]  # already (S,K)

            agree_ts = (st["win_pos"][0] == st["win_pos"][1]).astype(np.float64)  # (S,K)

            mmax  = _time_reduce_axis0(mmax_ts,  stat=time_stat)
            mnorm = _time_reduce_axis0(mnorm_ts, stat=time_stat)
            ratio = _time_reduce_axis0(ratio_ts, stat=time_stat)
            neff  = _time_reduce_axis0(neff_ts,  stat=time_stat)
            cos   = _time_reduce_axis0(cos_ts,   stat=time_stat)
            agree = _time_reduce_axis0(agree_ts, stat=time_stat)

        except Exception:
            continue

        if Tref is None:
            Tref = T
        else:
            _check_T_consistent(Tref, T)

        neff_list.append(neff)
        mmax_list.append(mmax)
        mnorm_list.append(mnorm)
        ratio_list.append(ratio)
        cos_list.append(cos)
        agree_list.append(agree)
        used.append(rid)

    if not used:
        raise RuntimeError("No valid disorders for summary aggregation.")

    return dict(
        T=Tref, used_rids=used,
        neff=_agg_1d_across_rids(np.vstack(neff_list)),
        mmax=_agg_1d_across_rids(np.vstack(mmax_list)),
        mnorm=_agg_1d_across_rids(np.vstack(mnorm_list)),
        ratio=_agg_1d_across_rids(np.vstack(ratio_list)),
        cos=_agg_1d_across_rids(np.vstack(cos_list)),
        agree=_agg_1d_across_rids(np.vstack(agree_list)),
        t_value=float(f(folder, "meta", rid=used[0]).t_grid[_resolve_b(f(folder, "meta", rid=used[0]), t_sel)]),
    )

def plot_m_summary_vs_T_across_disorders(
    res, *, alpha=0.20, show_individual=False, title=None
):
    """
    Same layout as your per-rid version:
      top: N_eff
      bottom: mmax, mnorm, ratio, cos, agree
    Using median + (16–84)% band; optional faint individual curves.
    """
    T = res["T"]
    neff = res["neff"]
    mmax = res["mmax"]
    mnorm = res["mnorm"]
    ratio = res["ratio"]
    cos = res["cos"]
    agree = res["agree"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True, gridspec_kw=dict(height_ratios=[1, 2])
    )

    if show_individual:
        for y in neff["Y"]:
            ax1.plot(T, y, lw=1.0, alpha=alpha)
    ax1.plot(T, neff["median"], lw=2.5)
    ax1.fill_between(T, neff["q16"], neff["q84"], alpha=0.20)
    ax1.set_ylabel(r"$\langle N_{\mathrm{eff}}\rangle$")
    ax1.grid(alpha=0.25)

    def _plot_band(ax, agg, label):
        if show_individual:
            for y in agg["Y"]:
                ax.plot(T, y, lw=1.0, alpha=alpha)
        ax.plot(T, agg["median"], lw=2.0, label=label)
        ax.fill_between(T, agg["q16"], agg["q84"], alpha=0.12)

    _plot_band(ax2, mmax,  r"$\langle m_{\max}\rangle$")
    _plot_band(ax2, mnorm, r"$\langle \|m\|\rangle$")
    _plot_band(ax2, ratio, r"$\langle m_{\max}/\|m\|\rangle$")
    _plot_band(ax2, cos,   r"$\langle \cos\theta\rangle$")
    _plot_band(ax2, agree, r"$\Pr(\mu^*_0=\mu^*_1)$")

    ax2.set_xlabel("Temperature T")
    ax2.grid(alpha=0.25)
    ax2.legend()

    ax1.set_title(title or f"Magnetization summary across disorders (n={neff['nR']})")
    plt.tight_layout()
    return (ax1, ax2)


# ==========================================================
# 2) winner entropy aggregated
# ==========================================================
def analyze_winner_entropy_across_disorders(
    f, folder, *, rids, t_sel, burn=0, thin=1, alpha_smooth=0.5
):
    """
    Per rid per T:
      normalized entropy of winner index distribution over time, pooling both chains.
    Uses Dirichlet smoothing with alpha_smooth to reduce finite-sample bias.
    """
    rids = list(rids)
    Tref = None
    ent_list = []
    used = []

    for rid in rids:
        try:
            meta, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
            P_sel = int(meta.mu_to_store.size)
            st = _m_stats(ms)
            win = st["win_pos"]  # (2,S,K)

            Hn = np.zeros(T.size, float)
            for k in range(T.size):
                w = np.concatenate([win[0, :, k], win[1, :, k]], axis=0)
                counts = np.bincount(w, minlength=P_sel).astype(float)
                # smoothing
                counts = counts + float(alpha_smooth)
                p = counts / counts.sum()
                m = p > 0
                H = -np.sum(p[m] * np.log(p[m]))
                Hn[k] = H / np.log(max(P_sel, 2))
        except Exception:
            continue

        if Tref is None:
            Tref = T
        else:
            _check_T_consistent(Tref, T)

        ent_list.append(Hn)
        used.append(rid)

    if not used:
        raise RuntimeError("No valid disorders for winner-entropy aggregation.")

    return dict(T=Tref, used_rids=used, ent=_agg_1d_across_rids(np.vstack(ent_list)))

def plot_winner_entropy_vs_T_across_disorders(res, *, alpha=0.20, show_individual=False, title=None):
    T = res["T"]
    ent = res["ent"]
    fig, ax = plt.subplots(figsize=(7, 4))

    if show_individual:
        for y in ent["Y"]:
            ax.plot(T, y, lw=1.0, alpha=alpha)

    ax.plot(T, ent["median"], lw=2.5)
    ax.fill_between(T, ent["q16"], ent["q84"], alpha=0.20)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("normalized winner entropy")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.set_title(title or f"Winner entropy across disorders (n={ent['nR']})")
    return ax


# ==========================================================
# 3) switching rates aggregated
# ==========================================================
def analyze_switching_rates_across_disorders(
    f, folder, *, rids, t_sel, burn=0, thin=1
):
    """
    Per rid per T:
      - winner switching rate
      - winner sign-flip rate
    (per-step rates, averaged over chains)
    """
    rids = list(rids)
    Tref = None
    sw_list, sf_list = [], []
    used = []

    for rid in rids:
        try:
            meta, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
            st = _m_stats(ms)
            win = st["win_pos"]   # (2,S,K)
            sgn = st["win_sign"]  # (2,S,K)

            def rate(x):  # x: (S,K)
                return np.mean(x[1:] != x[:-1], axis=0)

            sw = 0.5 * (rate(win[0]) + rate(win[1]))
            sf = 0.5 * (rate(sgn[0]) + rate(sgn[1]))

        except Exception:
            continue

        if Tref is None:
            Tref = T
        else:
            _check_T_consistent(Tref, T)

        sw_list.append(sw)
        sf_list.append(sf)
        used.append(rid)

    if not used:
        raise RuntimeError("No valid disorders for switching-rate aggregation.")

    return dict(
        T=Tref, used_rids=used,
        sw=_agg_1d_across_rids(np.vstack(sw_list)),
        sf=_agg_1d_across_rids(np.vstack(sf_list)),
    )

def plot_switching_rates_vs_T_across_disorders(res, *, alpha=0.20, show_individual=False, title=None):
    T = res["T"]
    sw = res["sw"]
    sf = res["sf"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    if show_individual:
        for y in sw["Y"]:
            ax1.plot(T, y, lw=1.0, alpha=alpha)
    ax1.plot(T, sw["median"], lw=2.5)
    ax1.fill_between(T, sw["q16"], sw["q84"], alpha=0.20)
    ax1.set_ylabel("winner switch rate")
    ax1.grid(alpha=0.25)

    if show_individual:
        for y in sf["Y"]:
            ax2.plot(T, y, lw=1.0, alpha=alpha)
    ax2.plot(T, sf["median"], lw=2.5)
    ax2.fill_between(T, sf["q16"], sf["q84"], alpha=0.20)
    ax2.set_xlabel("Temperature T")
    ax2.set_ylabel("winner sign-flip rate")
    ax2.grid(alpha=0.25)

    ax1.set_title(title or f"Switching rates across disorders (n={sw['nR']})")
    plt.tight_layout()
    return (ax1, ax2)


# ==========================================================
# 4) stacked histograms P(m_max) aggregated
# ==========================================================
def analyze_mmax_hist_across_disorders(
    f, folder, *, rids, t_sel, burn=0, thin=1,
    bins=60, m_range=(0.0, 1.0), density=True, chain_avg="mean"
):
    """
    Per rid per T: histogram of chain-avg m_max over time on common bin edges.
    Aggregate binwise across disorders via median + quantile band.
    Returns (T, bin_edges, centers, med, q16, q84).
    """
    rids = list(rids)
    edges = np.linspace(m_range[0], m_range[1], int(bins) + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])

    Tref = None
    H_list = []
    used = []

    for rid in rids:
        try:
            meta, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
            st = _m_stats(ms)
            mmax = st["m_max"]  # (2,S,K)

            if chain_avg == "mean":
                y = mmax.mean(axis=0)  # (S,K)
            elif chain_avg == "median":
                y = np.median(mmax, axis=0)
            else:
                raise ValueError(chain_avg)

            Hk = []
            for k in range(T.size):
                h, _ = np.histogram(y[:, k], bins=edges, density=density)
                Hk.append(h)
            Hk = np.stack(Hk, axis=0)  # (K, bins)
        except Exception:
            continue

        if Tref is None:
            Tref = T
        else:
            _check_T_consistent(Tref, T)

        H_list.append(Hk)
        used.append(rid)

    if not used:
        raise RuntimeError("No valid disorders for m_max histogram aggregation.")

    H = np.stack(H_list, axis=0)  # (nR, K, bins)
    med = np.nanmedian(H, axis=0)
    q16 = np.nanquantile(H, 0.16, axis=0)
    q84 = np.nanquantile(H, 0.84, axis=0)

    return dict(T=Tref, edges=edges, centers=centers, med=med, q16=q16, q84=q84, nR=H.shape[0])

def plot_mmax_stacked_hist_across_disorders(
    res, *, normalize_each=True, show_band=True, title=None
):
    """
    Ridge plot of aggregated histograms:
      - median ridge per T
      - optional band using q16/q84
    """
    T = res["T"]
    x = res["centers"]
    med, lo, hi = res["med"], res["q16"], res["q84"]  # (K,bins)

    fig, ax = plt.subplots(figsize=(7, max(12, 0.18 * T.size)))

    for k in range(3,T.size-3):
        m = med[k].copy()
        l = lo[k].copy()
        h = hi[k].copy()

        if normalize_each:
            s = max(m.max(), 1e-30)
            m /= s; l /= s; h /= s

        y0 = k
        ax.plot(x, y0 + m, lw=1.2)
        ax.fill_between(x, y0, y0 + m, alpha=0.22)
        if show_band:
            ax.fill_between(x, y0 + l, y0 + h, alpha=0.12)

    ax.set_yticks(np.arange(T.size))
    ax.set_yticklabels([f"T={Ti:.3g}" for Ti in T])
    ax.set_xlabel(r"$m_{\max}$")
    ax.set_ylabel("temperature index (fixed t)")
    ax.grid(alpha=0.2)
    ax.set_title(title or f"Stacked P(m_max) across disorders (n={res['nR']})")
    plt.tight_layout()
    return ax


# ==========================================================
# 5/6) joint (q,m_max) and (E,m_max) aggregated via 2D hist median
# ==========================================================
def _analyze_joint_2d_hist_across_disorders(
    f, folder, *, rids, t_sel, T_target, burn=0, thin=1,
    gridsize=60, x_range=(-1, 1), y_range=(0, 1),
    per_rid_max_points=5000, rng_seed=0,
    kind="q"  # "q" or "E"
):
    """
    For each rid, at the nearest temperature to T_target (within fixed t):
      build 2D histogram of (x, y) where y=m_max(chain-avg) and x is q or E.
    Aggregate binwise across disorders via median.
    """
    rids = list(rids)
    rng = np.random.default_rng(int(rng_seed))

    x_edges = np.linspace(x_range[0], x_range[1], int(gridsize) + 1)
    y_edges = np.linspace(y_range[0], y_range[1], int(gridsize) + 1)

    H_list = []
    used = []
    Tref = None
    kref = None

    for rid in rids:
        try:
            meta_m, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
            st = _m_stats(ms)
            mmax = st["m_max"].mean(axis=0)  # (S,K)

            k = int(np.argmin(np.abs(T - float(T_target))))
            if kind == "q":
                meta_q, b2, T2, r2, qs = _load_q_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
                _check_T_consistent(T, T2)
                x = qs[:, k]
            elif kind == "E":
                meta_E, b2, T2, r2, Es = _load_E_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin, chain_avg=True)
                _check_T_consistent(T, T2)
                x = Es[:, k]
            else:
                raise ValueError(kind)

            y = mmax[:, k]

            # equalize disorder weights by subsampling per rid
            n = x.shape[0]
            if n > per_rid_max_points:
                idx = rng.choice(n, size=int(per_rid_max_points), replace=False)
                x = x[idx]; y = y[idx]

            H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], density=False)  # (gx, gy)

        except Exception:
            continue

        if Tref is None:
            Tref = T
            kref = k
        else:
            _check_T_consistent(Tref, T)
            # we expect same k if T grids identical; enforce for sanity
            if k != kref:
                raise RuntimeError("Nearest-k to T_target differs across disorders; check T grids / T_target.")

        H_list.append(H)
        used.append(rid)

    if not used:
        raise RuntimeError("No valid disorders for joint 2D histogram aggregation.")

    Hs = np.stack(H_list, axis=0)  # (nR, gx, gy)
    med = np.nansum(Hs, axis=0)
    q16 = np.nanquantile(Hs, 0.16, axis=0)
    q84 = np.nanquantile(Hs, 0.84, axis=0)

    return dict(
        T=Tref, k=kref, T_at=float(Tref[kref]),
        x_edges=x_edges, y_edges=y_edges,
        med=med, q16=q16, q84=q84,
        nR=Hs.shape[0], used_rids=used,
        kind=kind,
    )

def plot_joint_mmax_q_across_disorders(
    f, folder, *, rids, t_sel, T_target,
    burn=0, thin=1,
    gridsize=60, q_range=(-1, 1), m_range=(0, 1),
    per_rid_max_points=5000,
    show_band=False,
    title=None,
):
    res = _analyze_joint_2d_hist_across_disorders(
        f, folder, rids=rids, t_sel=t_sel, T_target=T_target, burn=burn, thin=thin,
        gridsize=gridsize, x_range=q_range, y_range=m_range,
        per_rid_max_points=per_rid_max_points, kind="q"
    )
    X, Y = np.meshgrid(res["x_edges"], res["y_edges"], indexing="ij")

    fig, ax = plt.subplots(figsize=(6, 4))
    pcm = ax.pcolormesh(X, Y, res["med"], shading="auto")
    plt.colorbar(pcm, ax=ax, label="median density across disorders")

    ax.set_xlabel("q")
    ax.set_ylabel(r"$m_{\max}$ (chain-avg)")
    ax.set_title(title or f"(q, m_max) across disorders (n={res['nR']}), T≈{res['T_at']:.3g}")

    if show_band:
        # optional diagnostic: show relative width as overlay is messy; keep off by default
        pass

    plt.tight_layout()
    return ax

def plot_joint_mmax_E_across_disorders(
    f, folder, *, rids, t_sel, T_target,
    burn=0, thin=1,
    gridsize=60, E_range=None, m_range=(0, 1),
    per_rid_max_points=5000,
    title=None,
):
    # if E_range not provided, pick from a quick pilot on first available rid
    if E_range is None:
        for rid0 in list(rids):
            try:
                meta_E, b, T, r, Es = _load_E_ladder(f, folder, rid=rid0, t_sel=t_sel, burn=burn, thin=thin, chain_avg=True)
                k = int(np.argmin(np.abs(T - float(T_target))))
                x = Es[:, k]
                lo, hi = np.quantile(x, [0.01, 0.99])
                pad = 0.05 * (hi - lo + 1e-12)
                E_range = (float(lo - pad), float(hi + pad))
                break
            except Exception:
                continue
        if E_range is None:
            raise RuntimeError("Could not infer E_range; provide it explicitly.")

    res = _analyze_joint_2d_hist_across_disorders(
        f, folder, rids=rids, t_sel=t_sel, T_target=T_target, burn=burn, thin=thin,
        gridsize=gridsize, x_range=E_range, y_range=m_range,
        per_rid_max_points=per_rid_max_points, kind="E"
    )
    X, Y = np.meshgrid(res["x_edges"], res["y_edges"], indexing="ij")

    fig, ax = plt.subplots(figsize=(6, 4))
    pcm = ax.pcolormesh(X, Y, res["med"], shading="auto")
    plt.colorbar(pcm, ax=ax, label="median density across disorders")

    ax.set_xlabel("E (chain-avg)")
    ax.set_ylabel(r"$m_{\max}$ (chain-avg)")
    ax.set_title(title or f"(E, m_max) across disorders (n={res['nR']}), T≈{res['T_at']:.3g}")
    plt.tight_layout()
    return ax


import numpy as np
import matplotlib.pyplot as plt

def plot_joint_mmax_q_across_disorders_hexbin(
    f, folder, *, rids, t_sel,
    k=None, T_target=None,
    gridsize=60,
    q_range=(-1, 1), m_range=(0, 1),
    burn=0, thin=1,
    per_rid_max_points=None,     # e.g. 30000; None = no subsampling
    mincnt=1,                    # threshold: bins with <mincnt are not drawn -> white background
    ax=None, title=None,
    seed=0,
):
    """
    Aggregate (q, m_max) by SUMMING hexbin counts across disorders.

    - For each rid: extract x=q[:,k], y=mmax[:,k] at fixed t ladder.
    - Optionally subsample per rid to equalize contributions.
    - Concatenate all points; one final ax.hexbin() -> summed counts per hex cell.

    Choose either:
      - k (index in the *sorted-by-T* ladder used by _load_* helpers), OR
      - T_target (nearest T chosen inside each rid; identical ladders -> same k).
    """
    rids = list(rids)
    if (k is None) == (T_target is None):
        raise ValueError("Provide exactly one of k or T_target.")

    rng = np.random.default_rng(seed)

    X_all = []
    Y_all = []
    Tref = None
    kref = None
    used = []

    for rid in rids:
        try:
            meta_m, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
            meta_q, b2, T2, r2, qs = _load_q_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
            if not np.allclose(T, T2):
                raise RuntimeError("T grids differ between m and q loaders.")

            st = _m_stats(ms)
            mmax = st["m_max"].mean(axis=0)  # (S,K)

            if Tref is None:
                Tref = T
            else:
                if not np.allclose(T, Tref):
                    raise RuntimeError("T ladder differs across disorders (unexpected).")

            if k is None:
                kk = int(np.argmin(np.abs(T - float(T_target))))
            else:
                kk = int(k)
                if kk < 0 or kk >= T.size:
                    raise IndexError(f"k={kk} out of range [0,{T.size})")

            if kref is None:
                kref = kk
            else:
                if kk != kref:
                    # should not happen if ladders identical and you used T_target consistently
                    raise RuntimeError("Different k selected across disorders; use k explicitly.")

            x = qs[:, kk]
            y = mmax[:, kk]

            if per_rid_max_points is not None and x.shape[0] > per_rid_max_points:
                idx = rng.choice(x.shape[0], size=int(per_rid_max_points), replace=False)
                x = x[idx]
                y = y[idx]

            X_all.append(x)
            Y_all.append(y)
            used.append(rid)

        except Exception:
            continue

    if not used:
        raise RuntimeError("No valid disorders found for aggregation.")

    x = np.concatenate(X_all, axis=0)
    y = np.concatenate(Y_all, axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    extent = (q_range[0], q_range[1], m_range[0], m_range[1])
    ax.hexbin(x, y, gridsize=gridsize, extent=extent,mincnt=mincnt,norm=PowerNorm(gamma=0.5))

    ax.set_xlabel("q")
    ax.set_ylabel(r"$m_{\max}$ (chain-avg)")
    Tk = float(Tref[kref])
    ax.set_title(title or f"(q, m_max) summed across disorders (n={len(used)}), T≈{Tk:.3g}")

    plt.tight_layout()
    #cb.set_label("hexbin count (summed across disorders)")
    return ax

import numpy as np
import matplotlib.pyplot as plt


def analyze_mmax_hist_across_disorders_Twindow(
    f, folder, *, rids, t_sel,
    T_min=None, T_max=None,
    burn=0, thin=1,
    bins=60, m_range=(0.0, 1.0),
    density=True,
    chain_avg="mean",
    agg="median",          # "median" | "mean" | "sum_counts"
):
    """
    Like analyze_mmax_hist_across_disorders, but only for temperatures in [T_min, T_max].

    Returns dict with:
      T_sel, k_sel, edges, centers, H_agg, (optionally q16/q84), nR, agg, density
    """
    rids = list(rids)
    edges = np.linspace(m_range[0], m_range[1], int(bins) + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])

    Tref = None
    k_sel = None
    H_list = []
    used = []

    for rid in rids:
        try:
            meta, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
            st = _m_stats(ms)
            mmax = st["m_max"]  # (2,S,K)

            if chain_avg == "mean":
                y = mmax.mean(axis=0)  # (S,K)
            elif chain_avg == "median":
                y = np.median(mmax, axis=0)
            else:
                raise ValueError(chain_avg)

            if Tref is None:
                Tref = T
                # choose k indices once from the common ladder
                mask = np.ones(Tref.size, dtype=bool)
                if T_min is not None:
                    mask &= (Tref >= float(T_min))
                if T_max is not None:
                    mask &= (Tref <= float(T_max))
                k_sel = np.where(mask)[0]
                if k_sel.size == 0:
                    raise RuntimeError(f"No temperatures in window [{T_min},{T_max}] (available T in [{Tref.min()},{Tref.max()}]).")
            else:
                _check_T_consistent(Tref, T)

            Hk = []
            for k in k_sel:
                # IMPORTANT: for sum_counts we need raw counts (density=False)
                h, _ = np.histogram(y[:, k], bins=edges, density=(density and agg != "sum_counts"))
                Hk.append(h)
            Hk = np.stack(Hk, axis=0)  # (K_sel, bins)

        except Exception:
            continue

        H_list.append(Hk)
        used.append(rid)

    if not used:
        raise RuntimeError("No valid disorders for m_max histogram aggregation.")

    H = np.stack(H_list, axis=0)  # (nR, K_sel, bins)

    out = dict(
        T=Tref[k_sel],
        k_sel=k_sel,
        edges=edges,
        centers=centers,
        nR=H.shape[0],
        used_rids=used,
        agg=agg,
        density=density,
        T_window=(T_min, T_max),
    )

    if agg == "median":
        out["H"] = np.nanmedian(H, axis=0)
        out["q16"] = np.nanquantile(H, 0.16, axis=0)
        out["q84"] = np.nanquantile(H, 0.84, axis=0)
    elif agg == "mean":
        out["H"] = np.nanmean(H, axis=0)
        out["q16"] = np.nanquantile(H, 0.16, axis=0)
        out["q84"] = np.nanquantile(H, 0.84, axis=0)
    elif agg == "sum_counts":
        # Here each rid histogram was computed with density=False regardless of `density`
        Hsum = np.nansum(H, axis=0)  # (K_sel, bins)
        out["H"] = Hsum
        # variability band doesn’t really match a sum; omit by default
    else:
        raise ValueError("agg must be 'median', 'mean', or 'sum_counts'")

    return out


def plot_mmax_stacked_hist_across_disorders_Twindow(
    res, *,
    normalize_each=True,
    show_band=True,
    title=None,
):
    """
    Ridge plot for the selected temperature window.
    res from analyze_mmax_hist_across_disorders_Twindow.
    """
    T = res["T"]
    x = res["centers"]
    H = res["H"]  # (K_sel, bins)

    has_band = ("q16" in res) and ("q84" in res) and show_band
    lo = res.get("q16", None)
    hi = res.get("q84", None)

    fig, ax = plt.subplots(figsize=(7, max(6, 0.30 * T.size)))

    for i in range(T.size):
        m = H[i].copy()
        if normalize_each:
            s = max(m.max(), 1e-30)
            m /= s
        y0 = i
        ax.plot(x, y0 + m, lw=1.2)
        ax.fill_between(x, y0, y0 + m, alpha=0.22)

        if has_band:
            l = lo[i].copy()
            h = hi[i].copy()
            if normalize_each:
                s = max(H[i].max(), 1e-30)
                l /= s; h /= s
            ax.fill_between(x, y0 + l, y0 + h, alpha=0.12)

    ax.set_yticks(np.arange(T.size))
    ax.set_yticklabels([f"T={Ti:.3g}" for Ti in T])
    ax.set_xlabel(r"$m_{\max}$")
    ax.set_ylabel("selected temperatures (fixed t)")
    ax.grid(alpha=0.2)

    agg = res.get("agg", "?")
    nR = res.get("nR", 0)
    Tw = res.get("T_window", (None, None))
    ax.set_title(title or f"Stacked P(m_max) across disorders (agg={agg}, n={nR}, T∈[{Tw[0]},{Tw[1]}])")
    plt.tight_layout()
    return ax
import numpy as np
import matplotlib.pyplot as plt


def analyze_mmax_hist_single_rid_Twindow(
    f, folder, *, rid, t_sel,
    T_min=None, T_max=None,
    burn=0, thin=1,
    bins=60, m_range=(0.0, 1.0),
    density=True,
    chain_avg="mean",
):
    """
    Single rid: compute per-temperature histogram of chain-avg m_max over time,
    but only for temperatures in [T_min, T_max] (within fixed t_sel ladder).
    """
    edges = np.linspace(m_range[0], m_range[1], int(bins) + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])

    meta, b, T, r, ms = _load_m_ladder(f, folder, rid=rid, t_sel=t_sel, burn=burn, thin=thin)
    st = _m_stats(ms)
    mmax = st["m_max"]  # (2,S,K)

    if chain_avg == "mean":
        y = mmax.mean(axis=0)  # (S,K)
    elif chain_avg == "median":
        y = np.median(mmax, axis=0)
    else:
        raise ValueError(chain_avg)

    mask = np.ones(T.size, dtype=bool)
    if T_min is not None:
        mask &= (T >= float(T_min))
    if T_max is not None:
        mask &= (T <= float(T_max))
    k_sel = np.where(mask)[0]
    if k_sel.size == 0:
        raise RuntimeError(f"No temperatures in window [{T_min},{T_max}] for t={meta.t_grid[b]:g}.")

    Hk = []
    for k in k_sel:
        h, _ = np.histogram(y[:, k], bins=edges, density=density)
        Hk.append(h)
    Hk = np.stack(Hk, axis=0)  # (K_sel, bins)

    return dict(
        rid=int(rid),
        t_value=float(meta.t_grid[b]),
        T=T[k_sel],
        k_sel=k_sel,
        edges=edges,
        centers=centers,
        H=Hk,
        density=bool(density),
        chain_avg=str(chain_avg),
        T_window=(T_min, T_max),
    )


def plot_mmax_stacked_hist_single_rid_Twindow(
    res, *, normalize_each=True, title=None, ax=None
):
    """
    Ridge plot for a single rid and a selected temperature window.
    """
    T = res["T"]
    x = res["centers"]
    H = res["H"]  # (K_sel, bins)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, max(6, 0.30 * T.size)))

    for i in range(T.size):
        m = H[i].copy()
        if normalize_each:
            s = max(m.max(), 1e-30)
            m /= s
        y0 = i
        ax.plot(x, y0 + m, lw=1.2)
        ax.fill_between(x, y0, y0 + m, alpha=0.22)

    ax.set_yticks(np.arange(T.size))
    ax.set_yticklabels([f"T={Ti:.3g}" for Ti in T])
    ax.set_xlabel(r"$m_{\max}$")
    ax.set_ylabel("selected temperatures")
    ax.grid(alpha=0.2)

    if title is None:
        Tw = res["T_window"]
        title = f"rid={res['rid']} | t={res['t_value']:.3g} | T∈[{Tw[0]},{Tw[1]}]"
    ax.set_title(title)
    return ax


def plot_mmax_stacked_hist_compare_two_t_two_windows_single_rid(
    f, folder, *, rid,
    t_sel_a, T_min_a=None, T_max_a=None,
    t_sel_b=None, T_min_b=None, T_max_b=None,
    burn=0, thin=1,
    bins=60, m_range=(0.0, 1.0),
    density=True, chain_avg="mean",
    normalize_each=True,
    title=None,
):
    """
    Two-panel comparison for a single rid:
      panel A: t_sel_a with [T_min_a, T_max_a]
      panel B: t_sel_b with [T_min_b, T_max_b]
    """
    if t_sel_b is None:
        raise ValueError("Provide t_sel_b.")

    resA = analyze_mmax_hist_single_rid_Twindow(
        f, folder, rid=rid, t_sel=t_sel_a, T_min=T_min_a, T_max=T_max_a,
        burn=burn, thin=thin, bins=bins, m_range=m_range, density=density, chain_avg=chain_avg
    )
    resB = analyze_mmax_hist_single_rid_Twindow(
        f, folder, rid=rid, t_sel=t_sel_b, T_min=T_min_b, T_max=T_max_b,
        burn=burn, thin=thin, bins=bins, m_range=m_range, density=density, chain_avg=chain_avg
    )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14,   2* max(resA["T"].size, resB["T"].size) ),
        sharex=True
    )
    plot_mmax_stacked_hist_single_rid_Twindow(resA, normalize_each=normalize_each, ax=ax1)
    plot_mmax_stacked_hist_single_rid_Twindow(resB, normalize_each=normalize_each, ax=ax2)

    if title is not None:
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    return (ax1, ax2)
