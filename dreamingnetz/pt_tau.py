
from __future__ import annotations

"""
pt_tau.py — τ_int estimators + helpers for 1D/2D PT comparisons.

Drop-in targets (used in your notebooks):
  row_slice, take_row, take_t0, betas_by_row, temps_by_row, cold_node_index
  tau_q01_t0, plot_tau_overlay, plot_tau_overlay_compare
  tau_m_mu_t0, tau_m_all_mu_streamed_t0, tau_m_norm2_streamed_t0, tau_m_maxabs_streamed_t0
  compare_taus_from_roots, plot_compare_tau
  save_compare, load_compare
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union

import gzip
import pickle
import numpy as np


# ============================================================
# A) Ladder / slicing helpers
# ============================================================

def row_slice(k_start: np.ndarray, b: int = 0) -> tuple[int, int]:
    ks = np.asarray(k_start, dtype=np.int64).ravel()
    if ks.ndim != 1 or ks.size < 2:
        raise ValueError("k_start must be 1D with length B+1.")
    B = ks.size - 1
    if b < 0 or b >= B:
        raise ValueError(f"b out of range: {b} not in [0,{B}).")
    r0, r1 = int(ks[b]), int(ks[b + 1])
    if r1 < r0:
        raise ValueError("k_start must be nondecreasing.")
    return r0, r1


def take_row(Y: np.ndarray, k_start: np.ndarray, b: int = 0, *, axis: int = -1) -> np.ndarray:
    r0, r1 = row_slice(k_start, b)
    sl = [slice(None)] * Y.ndim
    sl[axis] = slice(r0, r1)
    return Y[tuple(sl)]


def take_t0(Y: np.ndarray, k_start: np.ndarray, *, axis: int = -1) -> np.ndarray:
    return take_row(Y, k_start, b=0, axis=axis)


def betas_by_row(beta: np.ndarray, k_start: np.ndarray, b: int = 0) -> np.ndarray:
    beta = np.asarray(beta, dtype=np.float64).ravel()
    r0, r1 = row_slice(k_start, b)
    if r1 > beta.size:
        raise ValueError("beta shorter than k_start expects.")
    return beta[r0:r1]


def temps_by_row(beta: np.ndarray, k_start: np.ndarray, b: int = 0) -> np.ndarray:
    br = betas_by_row(beta, k_start, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / br


def cold_node_index(beta_row: np.ndarray) -> int:
    beta_row = np.asarray(beta_row, dtype=np.float64).ravel()
    if beta_row.size == 0:
        raise ValueError("beta_row empty.")
    return int(np.argmax(beta_row))


# ============================================================
# B) τ_int for one series
# ============================================================

@dataclass(frozen=True)
class Tau1D:
    tau_geyer: float
    tau_bm: float
    neff_geyer: float
    neff_bm: float
    ok_geyer: bool
    ok_bm: bool
    split_z: float
    n_used: int
    burn_used: int
    bm_block: int


def _split_mean_z(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if n < 8:
        return np.nan
    a = y[: n // 2]
    b = y[n // 2 :]
    va = float(np.var(a, ddof=1)) if a.size > 1 else np.nan
    vb = float(np.var(b, ddof=1)) if b.size > 1 else np.nan
    if not np.isfinite(va) or not np.isfinite(vb) or va <= 0.0 or vb <= 0.0:
        return np.nan
    se = np.sqrt(va / a.size + vb / b.size)
    return float((np.mean(a) - np.mean(b)) / se) if se > 0 else np.nan


def _autocov_fft(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    y = y - np.mean(y)
    nfft = 1 << ((2 * n - 1).bit_length())
    f = np.fft.rfft(y, n=nfft)
    ac = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n]
    denom = np.arange(n, 0, -1, dtype=np.float64)  # n-k
    return ac / denom


def _pava_decreasing(x: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators for a nonincreasing sequence."""
    x = np.asarray(x, dtype=np.float64).ravel()
    v = x.copy()
    w = np.ones_like(v)
    m = v.size
    i = 0
    while i < m - 1:
        if v[i] < v[i + 1]:  # violation of nonincreasing
            # merge i and i+1
            new_w = w[i] + w[i + 1]
            new_v = (w[i] * v[i] + w[i + 1] * v[i + 1]) / new_w
            v[i] = new_v
            w[i] = new_w
            v = np.delete(v, i + 1)
            w = np.delete(w, i + 1)
            m -= 1
            if i > 0:
                i -= 1
        else:
            i += 1
    # expand piecewise-constant levels back to length of original x
    # by replaying merges isn't worth it; we only need sum(v_k) with weights.
    # BUT for IMS we need the pooled sequence values per k (same length as v),
    # and we will sum the pooled levels with their weights.
    # We'll return the "levels" and let caller use weights via np.unique? not.
    # Here we return a length-m vector of levels and encode weights via repetition:
    out = np.repeat(v, w.astype(int))
    # if weights weren't integer (they are), this matches original length.
    return out


def tau_geyer_ims(y: np.ndarray) -> float:
    """
    Geyer IMS τ estimator (returns τ >= 1 or nan).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if n < 4:
        return np.nan

    gamma = _autocov_fft(y)
    g0 = float(gamma[0])
    if not np.isfinite(g0) or g0 <= 0:
        return np.nan

    max_k = (n - 1) // 2
    G = gamma[: 2 * max_k + 2].reshape(-1, 2).sum(axis=1)  # G_k
    # truncate at first nonpositive
    m = 0
    for k in range(G.size):
        if G[k] <= 0.0:
            break
        m = k + 1
    if m == 0:
        return np.nan

    Gm = _pava_decreasing(G[:m])
    tau = -1.0 + (2.0 / g0) * float(np.sum(Gm))
    if not np.isfinite(tau) or tau < 1.0:
        tau = 1.0
    return float(tau)


def _bm_candidates(y: np.ndarray, *, min_batches: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return (valid_bs, tau_hats, gamma0) for batch means candidates.

    valid_bs, tau_hats are 1D arrays over block sizes b=1,2,4,... with a >= min_batches.
    gamma0 is sample variance of y (ddof=1). If gamma0 <= 0 or non-finite, returns empty arrays.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(y.size)
    gamma0 = float(np.var(y, ddof=1))
    if not np.isfinite(gamma0) or gamma0 <= 0.0 or n < 8:
        return (np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64), gamma0)

    bs = []
    b = 1
    while b <= n // int(min_batches):
        bs.append(int(b))
        b *= 2

    valid_bs: list[int] = []
    tau_hats: list[float] = []
    for b in bs:
        a = n // b
        if a < int(min_batches):
            continue
        n_use = a * b
        bm = y[:n_use].reshape(a, b).mean(axis=1)
        s2 = float(np.var(bm, ddof=1))
        if not np.isfinite(s2) or s2 <= 0.0:
            continue
        tau = (b * s2) / gamma0
        if np.isfinite(tau) and tau >= 1.0:
            valid_bs.append(int(b))
            tau_hats.append(float(tau))

    return (
        np.asarray(valid_bs, dtype=np.int64),
        np.asarray(tau_hats, dtype=np.float64),
        gamma0,
    )

def tau_batch_means_plateau(
    y: np.ndarray,
    *,
    min_batches: int = 30,
    rel_tol: float = 0.10,
    consec: int = 2,
) -> tuple[float, int]:
    """
    Batch means over block sizes b=1,2,4,... Choose first plateau.

    Returns (tau, b_chosen) or (nan, -1) if no plateau was found.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(y.size)
    if n < 8:
        return (np.nan, -1)

    valid_bs, tau_hats, _ = _bm_candidates(y, min_batches=int(min_batches))
    if tau_hats.size < int(consec) + 1:
        return (np.nan, -1)

    rel = np.abs(np.diff(tau_hats)) / np.maximum(tau_hats[:-1], 1e-12)
    for i in range(rel.size - (int(consec) - 1)):
        if np.all(rel[i : i + int(consec)] <= float(rel_tol)):
            j = i + int(consec)
            return (float(tau_hats[j]), int(valid_bs[j]))
    return (np.nan, -1)


def tau_int_1d(
    y: np.ndarray,
    *,
    burn: float | int = 0.0,
    min_n: int = 200,
    bm_min_batches: int = 30,
    bm_rel_tol: float = 0.10,
    bm_consec: int = 2,
    bm_fail: Literal["nan", "last", "max", "n", "cap"] = "nan",
    bm_fail_cap: float | None = None,
    geyer_fail: Literal["nan", "n", "cap"] = "nan",
    geyer_fail_cap: float | None = None,
) -> Tau1D:
    """
    Compute τ_int for a single 1D time series.

    Notes on failure handling:
      - Geyer IMS failures default to NaN; set geyer_fail="n" to fill with n_used (conservative),
        or geyer_fail="cap" with geyer_fail_cap.
      - BM plateau failures default to NaN; set bm_fail="last" to fill with the *largest-b* τ̂(b)
        (a lower bound when τ̂ keeps increasing). This avoids NaNs while keeping ok_bm=False.

    bm_block encoding:
      >0  : plateau found at that block size
      <0  : no plateau; filled from candidates, with abs(bm_block)=block size used (last/max)
      -2  : no valid blocks (e.g., variance zero)
      -3  : series too short for BM candidates
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n0 = int(y.size)

    if isinstance(burn, float):
        if burn < 0 or burn >= 1:
            raise ValueError("burn as float must be in [0,1).")
        b = int(np.floor(burn * n0))
    else:
        b = int(burn)
        if b < 0:
            raise ValueError("burn as int must be >= 0.")

    y2 = y[b:]
    n = int(y2.size)

    split_z = _split_mean_z(y2)

    # ---- Geyer IMS ----
    tau_g = tau_geyer_ims(y2)
    if np.isfinite(tau_g) and tau_g < 1.0:
        tau_g = 1.0
    elif not np.isfinite(tau_g) and geyer_fail != "nan":
        if geyer_fail == "n":
            tau_g = float(max(n, 1))
        elif geyer_fail == "cap":
            cap = float(geyer_fail_cap) if geyer_fail_cap is not None else float(max(n, 1))
            tau_g = cap
        if tau_g < 1.0:
            tau_g = 1.0

    # ---- Batch means plateau ----
    tau_b = np.nan
    b_chosen = -3  # too short default
    ok_plateau = False

    if n >= 8:
        valid_bs, tau_hats, gamma0 = _bm_candidates(y2, min_batches=int(bm_min_batches))
        if tau_hats.size == 0:
            # no valid blocks (often gamma0 <= 0)
            b_chosen = -2
            if bm_fail != "nan":
                # very conservative: ESS ~ 1 => tau ~ n
                if bm_fail in ("n", "last", "max"):
                    tau_b = float(max(n, 1))
                elif bm_fail == "cap":
                    cap = float(bm_fail_cap) if bm_fail_cap is not None else float(max(n, 1))
                    tau_b = cap
        else:
            # try to certify a plateau
            if tau_hats.size >= int(bm_consec) + 1:
                rel = np.abs(np.diff(tau_hats)) / np.maximum(tau_hats[:-1], 1e-12)
                for i in range(rel.size - (int(bm_consec) - 1)):
                    if np.all(rel[i : i + int(bm_consec)] <= float(bm_rel_tol)):
                        j = i + int(bm_consec)
                        tau_b = float(tau_hats[j])
                        b_chosen = int(valid_bs[j])
                        ok_plateau = True
                        break

            if not ok_plateau:
                if bm_fail == "nan":
                    tau_b = np.nan
                    b_chosen = -int(valid_bs[-1])  # record largest candidate block size
                else:
                    if bm_fail == "last":
                        tau_b = float(tau_hats[-1])
                        b_chosen = -int(valid_bs[-1])
                    elif bm_fail == "max":
                        j = int(np.nanargmax(tau_hats))
                        tau_b = float(tau_hats[j])
                        b_chosen = -int(valid_bs[j])
                    elif bm_fail == "n":
                        tau_b = float(max(n, 1))
                        b_chosen = -int(valid_bs[-1])
                    elif bm_fail == "cap":
                        cap = float(bm_fail_cap) if bm_fail_cap is not None else float(max(n, 1))
                        tau_b = cap
                        b_chosen = -int(valid_bs[-1])
                    else:
                        raise ValueError(f"unknown bm_fail={bm_fail}")

    if np.isfinite(tau_b) and tau_b < 1.0:
        tau_b = 1.0

    neff_g = n / (2.0 * tau_g) if np.isfinite(tau_g) and tau_g > 0 else np.nan
    neff_b = n / (2.0 * tau_b) if np.isfinite(tau_b) and tau_b > 0 else np.nan

    ok_geyer = bool(np.isfinite(tau_g) and n >= int(min_n) and abs(split_z) < 5.0)
    ok_bm = bool(np.isfinite(tau_b) and n >= int(min_n) and ok_plateau and abs(split_z) < 5.0)

    return Tau1D(
        tau_geyer=float(tau_g) if np.isfinite(tau_g) else np.nan,
        tau_bm=float(tau_b) if np.isfinite(tau_b) else np.nan,
        neff_geyer=float(neff_g) if np.isfinite(neff_g) else np.nan,
        neff_bm=float(neff_b) if np.isfinite(neff_b) else np.nan,
        ok_geyer=ok_geyer,
        ok_bm=ok_bm,
        split_z=float(split_z) if np.isfinite(split_z) else np.nan,
        n_used=n,
        burn_used=b,
        bm_block=int(b_chosen),
    )


# ============================================================
# C) τ_int on a grid of series (time last)
# ============================================================

@dataclass(frozen=True)
class TauGrid:
    grid_shape: tuple[int, ...]
    tau_geyer: np.ndarray
    tau_bm: np.ndarray
    neff_geyer: np.ndarray
    neff_bm: np.ndarray
    ok_geyer: np.ndarray
    ok_bm: np.ndarray
    split_z: np.ndarray


def tau_int_grid(
    Y: np.ndarray,
    *,
    burn: float | int = 0.0,
    min_n: int = 200,
    bm_min_batches: int = 30,
    bm_rel_tol: float = 0.10,
    bm_consec: int = 2,
    bm_fail: Literal["nan", "last", "max", "n", "cap"] = "nan",
    bm_fail_cap: float | None = None,
    geyer_fail: Literal["nan", "n", "cap"] = "nan",
    geyer_fail_cap: float | None = None,
) -> TauGrid:
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim < 2:
        raise ValueError("Y must have shape (...,T).")
    T = int(Y.shape[-1])
    grid_shape = tuple(int(s) for s in Y.shape[:-1])

    tau_g = np.full(grid_shape, np.nan, dtype=np.float64)
    tau_b = np.full(grid_shape, np.nan, dtype=np.float64)
    neff_g = np.full(grid_shape, np.nan, dtype=np.float64)
    neff_b = np.full(grid_shape, np.nan, dtype=np.float64)
    ok_g = np.zeros(grid_shape, dtype=bool)
    ok_bm = np.zeros(grid_shape, dtype=bool)
    split = np.full(grid_shape, np.nan, dtype=np.float64)

    flat = Y.reshape((-1, T))
    for i in range(flat.shape[0]):
        t = tau_int_1d(
            flat[i],
            burn=burn,
            min_n=min_n,
            bm_min_batches=bm_min_batches,
            bm_rel_tol=bm_rel_tol,
            bm_consec=bm_consec,
            bm_fail=bm_fail,
            bm_fail_cap=bm_fail_cap,
            geyer_fail=geyer_fail,
            geyer_fail_cap=geyer_fail_cap,
        )
        tau_g.flat[i] = t.tau_geyer
        tau_b.flat[i] = t.tau_bm
        neff_g.flat[i] = t.neff_geyer
        neff_b.flat[i] = t.neff_bm
        ok_g.flat[i] = t.ok_geyer
        ok_bm.flat[i] = t.ok_bm
        split.flat[i] = t.split_z

    return TauGrid(
        grid_shape=grid_shape,
        tau_geyer=tau_g,
        tau_bm=tau_b,
        neff_geyer=neff_g,
        neff_bm=neff_b,
        ok_geyer=ok_g,
        ok_bm=ok_bm,
        split_z=split,
    )


def _reduce_chain(grid: TauGrid, chain_reduce: Literal["none", "mean"]) -> TauGrid:
    if chain_reduce == "none":
        return grid
    if chain_reduce != "mean":
        raise ValueError("chain_reduce must be 'none' or 'mean'.")
    if len(grid.grid_shape) < 2 or grid.grid_shape[1] != 2:
        raise ValueError("reduce_chain(mean) expects axis=1 size=2.")
    return TauGrid(
        grid_shape=(grid.grid_shape[0],) + grid.grid_shape[2:],
        tau_geyer=np.mean(grid.tau_geyer, axis=1),
        tau_bm=np.mean(grid.tau_bm, axis=1),
        neff_geyer=np.mean(grid.neff_geyer, axis=1),
        neff_bm=np.mean(grid.neff_bm, axis=1),
        ok_geyer=np.all(grid.ok_geyer, axis=1),
        ok_bm=np.all(grid.ok_bm, axis=1),
        split_z=np.mean(grid.split_z, axis=1),
    )


# ============================================================
# D) Observable wrappers (q01, m_mu, streaming scalars)
# ============================================================

def tau_q01_row(
    q01_all: np.ndarray,
    *,
    k_start: np.ndarray,
    b: int = 0,
    burn: float | int = 0.0,
    min_n: int = 200,
    bm_min_batches: int = 30,
    bm_rel_tol: float = 0.10,
    bm_consec: int = 2,
    bm_fail: Literal["nan", "last", "max", "n", "cap"] = "nan",
    bm_fail_cap: float | None = None,
    geyer_fail: Literal["nan", "n", "cap"] = "nan",
    geyer_fail_cap: float | None = None,
) -> TauGrid:
    q01_all = np.asarray(q01_all, dtype=np.float64)
    if q01_all.ndim != 3:
        raise ValueError("q01_all must have shape (n_rid,T,R).")
    r0, r1 = row_slice(k_start, b)
    Y = np.transpose(q01_all[:, :, r0:r1], (0, 2, 1))  # (n_rid,Kb,T)
    return tau_int_grid(
        Y,
        burn=burn,
        min_n=min_n,
        bm_min_batches=bm_min_batches,
        bm_rel_tol=bm_rel_tol,
        bm_consec=bm_consec,
        bm_fail=bm_fail,
        bm_fail_cap=bm_fail_cap,
        geyer_fail=geyer_fail,
        geyer_fail_cap=geyer_fail_cap,
    )


def tau_q01_t0(
    q01_all: np.ndarray,
    *,
    k_start: np.ndarray,
    burn: float | int = 0.0,
    min_n: int = 200,
    bm_min_batches: int = 30,
    bm_rel_tol: float = 0.10,
    bm_consec: int = 2,
    bm_fail: Literal["nan", "last", "max", "n", "cap"] = "nan",
    bm_fail_cap: float | None = None,
    geyer_fail: Literal["nan", "n", "cap"] = "nan",
    geyer_fail_cap: float | None = None,
) -> TauGrid:
    return tau_q01_row(
        q01_all,
        k_start=k_start,
        b=0,
        burn=burn,
        min_n=min_n,
        bm_min_batches=bm_min_batches,
        bm_rel_tol=bm_rel_tol,
        bm_consec=bm_consec,
        bm_fail=bm_fail,
        bm_fail_cap=bm_fail_cap,
        geyer_fail=geyer_fail,
        geyer_fail_cap=geyer_fail_cap,
    )


def tau_m_mu_row(
    m_mu_all: np.ndarray,
    *,
    k_start: np.ndarray,
    b: int = 0,
    burn: float | int = 0.0,
    min_n: int = 200,
    bm_min_batches: int = 30,
    bm_rel_tol: float = 0.10,
    bm_consec: int = 2,
    bm_fail: Literal["nan", "last", "max", "n", "cap"] = "nan",
    bm_fail_cap: float | None = None,
    geyer_fail: Literal["nan", "n", "cap"] = "nan",
    geyer_fail_cap: float | None = None,
    chain_reduce: Literal["none", "mean"] = "mean",
) -> TauGrid:
    m_mu_all = np.asarray(m_mu_all, dtype=np.float64)
    if m_mu_all.ndim != 4 or m_mu_all.shape[1] != 2:
        raise ValueError("m_mu_all must have shape (n_rid,2,T,R).")
    r0, r1 = row_slice(k_start, b)
    Y = np.transpose(m_mu_all[:, :, :, r0:r1], (0, 1, 3, 2))  # (n_rid,2,Kb,T)
    out = tau_int_grid(
        Y,
        burn=burn,
        min_n=min_n,
        bm_min_batches=bm_min_batches,
        bm_rel_tol=bm_rel_tol,
        bm_consec=bm_consec,
        bm_fail=bm_fail,
        bm_fail_cap=bm_fail_cap,
        geyer_fail=geyer_fail,
        geyer_fail_cap=geyer_fail_cap,
    )
    return _reduce_chain(out, chain_reduce)


def tau_m_mu_t0(
    m_mu_all: np.ndarray,
    *,
    k_start: np.ndarray,
    burn: float | int = 0.0,
    min_n: int = 200,
    bm_min_batches: int = 30,
    bm_rel_tol: float = 0.10,
    bm_consec: int = 2,
    bm_fail: Literal["nan", "last", "max", "n", "cap"] = "nan",
    bm_fail_cap: float | None = None,
    geyer_fail: Literal["nan", "n", "cap"] = "nan",
    geyer_fail_cap: float | None = None,
    chain_reduce: Literal["none", "mean"] = "mean",
) -> TauGrid:
    return tau_m_mu_row(
        m_mu_all,
        k_start=k_start,
        b=0,
        burn=burn,
        min_n=min_n,
        bm_min_batches=bm_min_batches,
        bm_rel_tol=bm_rel_tol,
        bm_consec=bm_consec,
        bm_fail=bm_fail,
        bm_fail_cap=bm_fail_cap,
        geyer_fail=geyer_fail,
        geyer_fail_cap=geyer_fail_cap,
        chain_reduce=chain_reduce,
    )


@dataclass(frozen=True)
class TauMuGrid:
    grid_shape: tuple[int, ...]  # (n_rid,2?,Kb,P_mu) or (n_rid,Kb,P_mu)
    tau_geyer: np.ndarray
    tau_bm: np.ndarray
    neff_geyer: np.ndarray
    neff_bm: np.ndarray
    ok_geyer: np.ndarray
    ok_bm: np.ndarray
    split_z: np.ndarray
    mu_list: np.ndarray
    beta_row: np.ndarray
    k_start_row: tuple[int, int]


def _ensure_int_list(mu_list, P: int) -> np.ndarray:
    if mu_list is None:
        return np.arange(P, dtype=np.int64)
    mu = np.asarray(mu_list, dtype=np.int64).ravel()
    if mu.size == 0:
        raise ValueError("mu_list is empty.")
    if mu.min() < 0 or mu.max() >= P:
        raise ValueError(f"mu_list contains out-of-range values for P={P}.")
    return mu


def _slice_row_m_mu(m_mu_all: np.ndarray, k_start: np.ndarray, b: int) -> np.ndarray:
    r0, r1 = row_slice(k_start, b)
    return m_mu_all[:, :, :, r0:r1]  # (n_rid,2,T,Kb)


def load_common_meta(f_loader: Callable, run_root):
    metas = f_loader(run_root, "meta")
    if isinstance(metas, list):
        if not metas:
            raise ValueError("No metas returned.")
        return metas[0]
    return metas


def tau_m_all_mu_streamed_t0(
    f_loader: Callable,
    run_root,
    *,
    b: int = 0,
    mu_list=None,
    burn: float | int = 0.0,
    min_n: int = 200,
    bm_min_batches: int = 30,
    bm_rel_tol: float = 0.10,
    bm_consec: int = 2,
    bm_fail: Literal["nan", "last", "max", "n", "cap"] = "nan",
    bm_fail_cap: float | None = None,
    geyer_fail: Literal["nan", "n", "cap"] = "nan",
    geyer_fail_cap: float | None = None,
    chain_reduce: Literal["none", "mean"] = "none",
    progress: bool = True,
) -> TauMuGrid:
    meta = load_common_meta(f_loader, run_root)
    k_start = np.asarray(meta.k_start, dtype=np.int64)
    beta_row = betas_by_row(np.asarray(meta.beta), k_start, b)
    Kb = int(beta_row.size)

    P_all = int(np.asarray(meta.mu_to_store).size)
    mu = _ensure_int_list(mu_list, P_all)
    P_mu = int(mu.size)

    m0 = np.asarray(f_loader(run_root, "m", mu=int(mu[0])), dtype=np.float64)
    if m0.ndim != 4 or m0.shape[1] != 2:
        raise ValueError("f_loader(run_root,'m',mu=..) must return (n_rid,2,T,R).")
    n_rid, _, T, _R = m0.shape

    if chain_reduce == "none":
        grid_shape = (n_rid, 2, Kb, P_mu)
    elif chain_reduce == "mean":
        grid_shape = (n_rid, Kb, P_mu)
    else:
        raise ValueError("chain_reduce must be 'none' or 'mean'.")

    tau_g = np.full(grid_shape, np.nan, dtype=np.float64)
    tau_b = np.full(grid_shape, np.nan, dtype=np.float64)
    neff_g = np.full(grid_shape, np.nan, dtype=np.float64)
    neff_b = np.full(grid_shape, np.nan, dtype=np.float64)
    ok_g = np.zeros(grid_shape, dtype=bool)
    ok_bm = np.zeros(grid_shape, dtype=bool)
    split = np.full(grid_shape, np.nan, dtype=np.float64)

    for j, muj in enumerate(mu):
        if progress:
            print(f"[tau_mu] mu={int(muj)} ({j+1}/{P_mu})", flush=True)
        m_mu_all = np.asarray(f_loader(run_root, "m", mu=int(muj)), dtype=np.float64)
        m_row = _slice_row_m_mu(m_mu_all, k_start, b=b)   # (n_rid,2,T,Kb)
        Y = np.transpose(m_row, (0, 1, 3, 2))             # (n_rid,2,Kb,T)
        tg = tau_int_grid(
            Y,
            burn=burn, min_n=min_n,
            bm_min_batches=bm_min_batches, bm_rel_tol=bm_rel_tol, bm_consec=bm_consec,
            bm_fail=bm_fail,
            bm_fail_cap=bm_fail_cap,
            geyer_fail=geyer_fail,
            geyer_fail_cap=geyer_fail_cap,
        )
        if chain_reduce == "none":
            tau_g[:, :, :, j] = tg.tau_geyer
            tau_b[:, :, :, j] = tg.tau_bm
            neff_g[:, :, :, j] = tg.neff_geyer
            neff_b[:, :, :, j] = tg.neff_bm
            ok_g[:, :, :, j] = tg.ok_geyer
            ok_bm[:, :, :, j] = tg.ok_bm
            split[:, :, :, j] = tg.split_z
        else:
            tg2 = _reduce_chain(tg, "mean")               # (n_rid,Kb)
            tau_g[:, :, j] = tg2.tau_geyer
            tau_b[:, :, j] = tg2.tau_bm
            neff_g[:, :, j] = tg2.neff_geyer
            neff_b[:, :, j] = tg2.neff_bm
            ok_g[:, :, j] = tg2.ok_geyer
            ok_bm[:, :, j] = tg2.ok_bm
            split[:, :, j] = tg2.split_z

    r0, r1 = row_slice(k_start, b)
    return TauMuGrid(
        grid_shape=grid_shape,
        tau_geyer=tau_g, tau_bm=tau_b,
        neff_geyer=neff_g, neff_bm=neff_b,
        ok_geyer=ok_g, ok_bm=ok_bm,
        split_z=split,
        mu_list=mu,
        beta_row=np.asarray(beta_row, dtype=np.float64),
        k_start_row=(int(r0), int(r1)),
    )


def tau_m_norm2_streamed_t0(
    f_loader: Callable,
    run_root,
    *,
    b: int = 0,
    mu_list=None,
    burn: float | int = 0.0,
    min_n: int = 200,
    bm_min_batches: int = 30,
    bm_rel_tol: float = 0.10,
    bm_consec: int = 2,
    bm_fail: Literal["nan", "last", "max", "n", "cap"] = "nan",
    bm_fail_cap: float | None = None,
    geyer_fail: Literal["nan", "n", "cap"] = "nan",
    geyer_fail_cap: float | None = None,
    chain_reduce: Literal["none", "mean"] = "none",
    progress: bool = True,
) -> TauGrid:
    meta = load_common_meta(f_loader, run_root)
    k_start = np.asarray(meta.k_start, dtype=np.int64)
    beta_row = betas_by_row(np.asarray(meta.beta), k_start, b)
    Kb = int(beta_row.size)

    P_all = int(np.asarray(meta.mu_to_store).size)
    mu = _ensure_int_list(mu_list, P_all)
    m0 = np.asarray(f_loader(run_root, "m", mu=int(mu[0])), dtype=np.float64)
    n_rid, _, T, _R = m0.shape

    S = np.zeros((n_rid, 2, T, Kb), dtype=np.float64)
    for j, muj in enumerate(mu):
        if progress:
            print(f"[tau_norm2] mu={int(muj)} ({j+1}/{mu.size})", flush=True)
        m_mu_all = np.asarray(f_loader(run_root, "m", mu=int(muj)), dtype=np.float64)
        m_row = _slice_row_m_mu(m_mu_all, k_start, b=b)
        S += m_row * m_row

    Y = np.transpose(S, (0, 1, 3, 2))
    out = tau_int_grid(
        Y,
        burn=burn, min_n=min_n,
        bm_min_batches=bm_min_batches, bm_rel_tol=bm_rel_tol, bm_consec=bm_consec,
            bm_fail=bm_fail,
            bm_fail_cap=bm_fail_cap,
            geyer_fail=geyer_fail,
            geyer_fail_cap=geyer_fail_cap,
    )
    return _reduce_chain(out, chain_reduce)


def tau_m_maxabs_streamed_t0(
    f_loader: Callable,
    run_root,
    *,
    b: int = 0,
    mu_list=None,
    burn: float | int = 0.0,
    min_n: int = 200,
    bm_min_batches: int = 30,
    bm_rel_tol: float = 0.10,
    bm_consec: int = 2,
    bm_fail: Literal["nan", "last", "max", "n", "cap"] = "nan",
    bm_fail_cap: float | None = None,
    geyer_fail: Literal["nan", "n", "cap"] = "nan",
    geyer_fail_cap: float | None = None,
    chain_reduce: Literal["none", "mean"] = "none",
    progress: bool = True,
) -> TauGrid:
    meta = load_common_meta(f_loader, run_root)
    k_start = np.asarray(meta.k_start, dtype=np.int64)
    beta_row = betas_by_row(np.asarray(meta.beta), k_start, b)
    Kb = int(beta_row.size)

    P_all = int(np.asarray(meta.mu_to_store).size)
    mu = _ensure_int_list(mu_list, P_all)
    m0 = np.asarray(f_loader(run_root, "m", mu=int(mu[0])), dtype=np.float64)
    n_rid, _, T, _R = m0.shape

    M = np.zeros((n_rid, 2, T, Kb), dtype=np.float64)
    for j, muj in enumerate(mu):
        if progress:
            print(f"[tau_maxabs] mu={int(muj)} ({j+1}/{mu.size})", flush=True)
        m_mu_all = np.asarray(f_loader(run_root, "m", mu=int(muj)), dtype=np.float64)
        m_row = _slice_row_m_mu(m_mu_all, k_start, b=b)
        M = np.maximum(M, np.abs(m_row))

    Y = np.transpose(M, (0, 1, 3, 2))
    out = tau_int_grid(
        Y,
        burn=burn, min_n=min_n,
        bm_min_batches=bm_min_batches, bm_rel_tol=bm_rel_tol, bm_consec=bm_consec,
            bm_fail=bm_fail,
            bm_fail_cap=bm_fail_cap,
            geyer_fail=geyer_fail,
            geyer_fail_cap=geyer_fail_cap,
    )
    return _reduce_chain(out, chain_reduce)


# ============================================================
# E) Plotting (matplotlib)
# ============================================================

def plot_tau_overlay(
    x: np.ndarray,
    tau: np.ndarray,
    *,
    ax=None,
    log_x: bool = True,
    yscale: Literal["linear", "log"] = "log",
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    chain: Literal["mean", "both", 0, 1] = "mean",
    alpha: float = 0.25,
    lw: float = 1.0,
    show_summary: bool = True,
    q: float = 0.10,
    ylabel: str = r"$\tau_{\mathrm{int}}$  ",
    xlabel: str = "T",
    title: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=np.float64).ravel()
    tau = np.asarray(tau, dtype=np.float64)

    if tau.ndim == 2:
        Y = tau
    elif tau.ndim == 3 and tau.shape[1] == 2:
        if chain == "mean":
            Y = np.nanmean(tau, axis=1)
        elif chain in (0, 1):
            Y = tau[:, int(chain), :]
        elif chain == "both":
            Y = np.concatenate([tau[:, 0, :], tau[:, 1, :]], axis=0)
        else:
            raise ValueError("chain must be 'mean', 'both', 0, or 1.")
    else:
        raise ValueError("tau must have shape (n_rid,K) or (n_rid,2,K).")

    if Y.shape[1] != x.size:
        raise ValueError(f"x has len {x.size} but tau has K={Y.shape[1]}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
    else:
        fig = ax.figure

    for y in Y:
        ax.plot(x, y, alpha=alpha, linewidth=lw)

    if show_summary:
        med = np.nanmedian(Y, axis=0)
        lo = np.nanquantile(Y, q, axis=0)
        hi = np.nanquantile(Y, 1.0 - q, axis=0)
        ax.plot(x, med, linewidth=2.0, label="median")
        ax.fill_between(x, lo, hi, alpha=0.12)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_x:
        ax.set_xscale("log")
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title is not None:
        ax.set_title(title)
    if show_summary:
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_tau_overlay_compare(
    x: np.ndarray,
    tau_left: np.ndarray,
    tau_right: np.ndarray,
    *,
    labels: Tuple[str, str] = ("1D", "2D"),
    ax=None,
    log_x: bool = True,
    yscale: Literal["linear", "log"] = "log",
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    chain: Literal["mean", "both", 0, 1] = "mean",
    alpha: float = 0.15,
    lw: float = 1.0,
    show_summary: bool = True,
    q: float = 0.10,
    ylabel: str = r"$\tau_{\mathrm{int}}$  ",
    xlabel: str = "T",
    suptitle: Optional[str] = None,
    sharey: bool = True,
):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=sharey)
    else:
        axes = np.asarray(ax).ravel()
        if axes.size != 2:
            raise ValueError("ax must be None or an iterable of 2 Axes.")
        fig = axes[0].figure

    plot_tau_overlay(
        x, tau_left, ax=axes[0], log_x=log_x, yscale=yscale, ylim=ylim, xlim=xlim,
        chain=chain, alpha=alpha, lw=lw, show_summary=show_summary, q=q,
        ylabel=ylabel, xlabel=xlabel, title=labels[0],
    )
    plot_tau_overlay(
        x, tau_right, ax=axes[1], log_x=log_x, yscale=yscale, ylim=ylim, xlim=xlim,
        chain=chain, alpha=alpha, lw=lw, show_summary=show_summary, q=q,
        ylabel=ylabel, xlabel=xlabel, title=labels[1],
    )
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, axes


# ============================================================
# F) 1D vs 2D bundles + caching
# ============================================================

@dataclass(frozen=True)
class RunTauSet:
    q01: TauGrid
    m_mu0: Optional[TauGrid] = None
    m_norm2: Optional[TauGrid] = None
    m_maxabs: Optional[TauGrid] = None
    m_all_mu: Optional[TauMuGrid] = None


@dataclass(frozen=True)
class CompareTauResult:
    b: int
    k_start: np.ndarray
    beta_row: np.ndarray
    T_row: np.ndarray
    labels: Tuple[str, str]
    left_root: str
    right_root: str
    left: RunTauSet
    right: RunTauSet


def compare_taus_from_roots(
    f_loader: Callable,
    run_root_left: str,
    run_root_right: str,
    *,
    labels: Tuple[str, str] = ("1D", "2D"),
    b: int = 0,
    burn: float | int = 0.0,
    min_n: int = 200,
    bm_min_batches: int = 20,
    bm_rel_tol: float = 0.12,
    bm_consec: int = 2,
    bm_fail: Literal["nan", "last", "max", "n", "cap"] = "nan",
    bm_fail_cap: float | None = None,
    geyer_fail: Literal["nan", "n", "cap"] = "nan",
    geyer_fail_cap: float | None = None,
    compute_m_mu0: bool = True,
    mu0: int = 0,
    chain_reduce: Literal["none", "mean"] = "mean",
    compute_norm2: bool = True,
    compute_maxabs: bool = True,
    compute_all_mu: bool = False,
    mu_list=None,
    progress: bool = True,
) -> CompareTauResult:
    metaL = load_common_meta(f_loader, run_root_left)
    metaR = load_common_meta(f_loader, run_root_right)

    if not np.allclose(np.asarray(metaL.beta), np.asarray(metaR.beta)):
        raise RuntimeError("beta differs between roots (not comparable).")
    if not np.array_equal(np.asarray(metaL.k_start), np.asarray(metaR.k_start)):
        raise RuntimeError("k_start differs between roots (not comparable).")
    if np.asarray(metaL.mu_to_store).size != np.asarray(metaR.mu_to_store).size:
        raise RuntimeError("mu_to_store size differs between roots (not comparable).")

    k_start = np.asarray(metaL.k_start, dtype=np.int64)
    beta_row = betas_by_row(np.asarray(metaL.beta), k_start, b)
    T_row = temps_by_row(np.asarray(metaL.beta), k_start, b)

    qL = f_loader(run_root_left, "q01")
    qR = f_loader(run_root_right, "q01")
    tau_qL = tau_q01_row(
        qL, k_start=k_start, b=b,
        burn=burn, min_n=min_n,
        bm_min_batches=bm_min_batches, bm_rel_tol=bm_rel_tol, bm_consec=bm_consec
    )
    tau_qR = tau_q01_row(
        qR, k_start=k_start, b=b,
        burn=burn, min_n=min_n,
        bm_min_batches=bm_min_batches, bm_rel_tol=bm_rel_tol, bm_consec=bm_consec
    )

    tau_m0L = tau_m0R = None
    if compute_m_mu0:
        mL = f_loader(run_root_left, "m", mu=int(mu0))
        mR = f_loader(run_root_right, "m", mu=int(mu0))
        tau_m0L = tau_m_mu_row(
            mL, k_start=k_start, b=b,
            burn=burn, min_n=min_n,
            bm_min_batches=bm_min_batches, bm_rel_tol=bm_rel_tol, bm_consec=bm_consec,
        bm_fail=bm_fail,
        bm_fail_cap=bm_fail_cap,
        geyer_fail=geyer_fail,
        geyer_fail_cap=geyer_fail_cap,
            chain_reduce=chain_reduce,
        )
        tau_m0R = tau_m_mu_row(
            mR, k_start=k_start, b=b,
            burn=burn, min_n=min_n,
            bm_min_batches=bm_min_batches, bm_rel_tol=bm_rel_tol, bm_consec=bm_consec,
        bm_fail=bm_fail,
        bm_fail_cap=bm_fail_cap,
        geyer_fail=geyer_fail,
        geyer_fail_cap=geyer_fail_cap,
            chain_reduce=chain_reduce,
        )

    tau_norm2L = tau_norm2R = None
    if compute_norm2:
        tau_norm2L = tau_m_norm2_streamed_t0(
            f_loader, run_root_left, b=b,
            mu_list=mu_list,
            burn=burn, min_n=min_n,
            bm_min_batches=bm_min_batches, bm_rel_tol=bm_rel_tol, bm_consec=bm_consec,
        bm_fail=bm_fail,
        bm_fail_cap=bm_fail_cap,
        geyer_fail=geyer_fail,
        geyer_fail_cap=geyer_fail_cap,
            chain_reduce=chain_reduce, progress=progress,
        )
        tau_norm2R = tau_m_norm2_streamed_t0(
            f_loader, run_root_right, b=b,
            mu_list=mu_list,
            burn=burn, min_n=min_n,
            bm_min_batches=bm_min_batches, bm_rel_tol=bm_rel_tol, bm_consec=bm_consec,
        bm_fail=bm_fail,
        bm_fail_cap=bm_fail_cap,
        geyer_fail=geyer_fail,
        geyer_fail_cap=geyer_fail_cap,
            chain_reduce=chain_reduce, progress=progress,
        )

    tau_maxabsL = tau_maxabsR = None
    if compute_maxabs:
        tau_maxabsL = tau_m_maxabs_streamed_t0(
            f_loader, run_root_left, b=b,
            mu_list=mu_list,
            burn=burn, min_n=min_n,
            bm_min_batches=bm_min_batches, bm_rel_tol=bm_rel_tol, bm_consec=bm_consec,
        bm_fail=bm_fail,
        bm_fail_cap=bm_fail_cap,
        geyer_fail=geyer_fail,
        geyer_fail_cap=geyer_fail_cap,
            chain_reduce=chain_reduce, progress=progress,
        )
        tau_maxabsR = tau_m_maxabs_streamed_t0(
            f_loader, run_root_right, b=b,
            mu_list=mu_list,
            burn=burn, min_n=min_n,
            bm_min_batches=bm_min_batches, bm_rel_tol=bm_rel_tol, bm_consec=bm_consec,
        bm_fail=bm_fail,
        bm_fail_cap=bm_fail_cap,
        geyer_fail=geyer_fail,
        geyer_fail_cap=geyer_fail_cap,
            chain_reduce=chain_reduce, progress=progress,
        )

    tau_allmuL = tau_allmuR = None
    if compute_all_mu:
        tau_allmuL = tau_m_all_mu_streamed_t0(
            f_loader, run_root_left, b=b,
            mu_list=mu_list,
            burn=burn, min_n=min_n,
            bm_min_batches=bm_min_batches, bm_rel_tol=bm_rel_tol, bm_consec=bm_consec,
        bm_fail=bm_fail,
        bm_fail_cap=bm_fail_cap,
        geyer_fail=geyer_fail,
        geyer_fail_cap=geyer_fail_cap,
            chain_reduce=chain_reduce, progress=progress,
        )
        tau_allmuR = tau_m_all_mu_streamed_t0(
            f_loader, run_root_right, b=b,
            mu_list=mu_list,
            burn=burn, min_n=min_n,
            bm_min_batches=bm_min_batches, bm_rel_tol=bm_rel_tol, bm_consec=bm_consec,
        bm_fail=bm_fail,
        bm_fail_cap=bm_fail_cap,
        geyer_fail=geyer_fail,
        geyer_fail_cap=geyer_fail_cap,
            chain_reduce=chain_reduce, progress=progress,
        )

    left = RunTauSet(q01=tau_qL, m_mu0=tau_m0L, m_norm2=tau_norm2L, m_maxabs=tau_maxabsL, m_all_mu=tau_allmuL)
    right = RunTauSet(q01=tau_qR, m_mu0=tau_m0R, m_norm2=tau_norm2R, m_maxabs=tau_maxabsR, m_all_mu=tau_allmuR)

    return CompareTauResult(
        b=int(b),
        k_start=k_start,
        beta_row=np.asarray(beta_row, dtype=np.float64),
        T_row=np.asarray(T_row, dtype=np.float64),
        labels=labels,
        left_root=str(run_root_left),
        right_root=str(run_root_right),
        left=left,
        right=right,
    )


def plot_compare_tau(
    comp: CompareTauResult,
    *,
    obs: Literal["q01", "m_mu0", "m_norm2", "m_maxabs"] = "q01",
    estimator: Literal["geyer", "bm"] = "geyer",
    x_axis: Literal["beta", "T"] = "T",
    log_x: bool = True,
    yscale: Literal["linear", "log"] = "log",
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    chain: Literal["mean", "both", 0, 1] = "mean",
    alpha: float = 0.15,
    lw: float = 1.0,
    show_summary: bool = True,
    q: float = 0.10,
    ylabel: str = r"$\tau_{\mathrm{int}}$  ",
    suptitle: Optional[str] = None,
):
    x = comp.beta_row if x_axis == "beta" else comp.T_row

    def pick(grid: TauGrid) -> np.ndarray:
        return grid.tau_geyer if estimator == "geyer" else grid.tau_bm

    Lset = getattr(comp.left, obs)
    Rset = getattr(comp.right, obs)
    if Lset is None or Rset is None:
        raise ValueError(f"obs='{obs}' not computed in compare_taus_from_roots().")

    return plot_tau_overlay_compare(
        x, pick(Lset), pick(Rset),
        labels=comp.labels,
        log_x=log_x, yscale=yscale, ylim=ylim, xlim=xlim,
        chain=chain, alpha=alpha, lw=lw, show_summary=show_summary, q=q,
        ylabel=ylabel, xlabel=x_axis, suptitle=suptitle,
    )




# ============================================================
# G) Small helpers for "is 2D better?"
# ============================================================

@dataclass(frozen=True)
class RidSummary:
    median: np.ndarray
    qlo: np.ndarray
    qhi: np.ndarray
    q: Tuple[float, float]
    frac_finite: np.ndarray


def summarize_over_rids(Y: np.ndarray, *, q: float = 0.10, axis: int = 0) -> RidSummary:
    """Robust summaries across rid; ignores NaNs."""
    Y = np.asarray(Y, dtype=np.float64)
    loq, hiq = float(q), float(1.0 - q)
    med = np.nanmedian(Y, axis=axis)
    qlo = np.nanquantile(Y, loq, axis=axis)
    qhi = np.nanquantile(Y, hiq, axis=axis)
    frac = np.mean(np.isfinite(Y), axis=axis)
    return RidSummary(median=med, qlo=qlo, qhi=qhi, q=(loq, hiq), frac_finite=frac)


def tau_array(grid: TauGrid, estimator: Literal["geyer", "bm"] = "geyer") -> np.ndarray:
    return grid.tau_geyer if estimator == "geyer" else grid.tau_bm


def ok_array(grid: TauGrid, estimator: Literal["geyer", "bm"] = "geyer") -> np.ndarray:
    return grid.ok_geyer if estimator == "geyer" else grid.ok_bm


def paired_log10_ratio(left: np.ndarray, right: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    log10(left/right). Positive means right is smaller (better) when these are taus.
    """
    L = np.asarray(left, dtype=np.float64)
    R = np.asarray(right, dtype=np.float64)
    if L.shape != R.shape:
        raise ValueError("left and right must have same shape.")
    return np.log10(L / np.maximum(R, eps))


def frac_improved(left: np.ndarray, right: np.ndarray, *, eps: float = 1e-12, axis: int = 0) -> np.ndarray:
    """Fraction of rid where right < left."""
    L = np.asarray(left, dtype=np.float64)
    R = np.asarray(right, dtype=np.float64)
    return np.nanmean(R < np.maximum(L, eps), axis=axis)


def save_compare(comp: CompareTauResult, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(comp, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_compare(path: Union[str, Path]) -> CompareTauResult:
    with gzip.open(Path(path), "rb") as f:
        return pickle.load(f)
from typing import Literal, Optional, Tuple, Sequence, Dict
import numpy as np
import matplotlib.pyplot as plt

def plot_compare_tau_grid_2x2(
    comp,
    *,
    obs_list: Sequence[Literal["q01", "m_mu0", "m_norm2", "m_maxabs"]] = ("q01", "m_mu0"),
    estimator: Literal["geyer", "bm"] = "geyer",
    x_axis: Literal["beta", "T"] = "T",
    log_x: bool = True,
    yscale: Literal["linear", "log"] = "log",
    xlim: Optional[Tuple[float, float]] = None,
    ylim_by_obs: Optional[Dict[str, Tuple[float, float]]] = None,
    chain: Literal["mean", "both", 0, 1] = "mean",
    alpha: float = 0.15,
    lw: float = 1.0,
    show_summary: bool = True,
    q: float = 0.10,
    ylabel: str = r"$\tau_{\mathrm{int}}$",
    suptitle: Optional[str] = None,
    sharey: bool = True,
):
    """
    2×2 convenience wrapper when len(obs_list)==2.
    More generally, makes an (n_obs × 2) grid: columns are (left,right).
    """

    # x-axis
    x = comp.beta_row if x_axis == "beta" else comp.T_row

    # pick estimator grid
    def pick(grid) -> np.ndarray:
        return grid.tau_geyer if estimator == "geyer" else grid.tau_bm

    labels = getattr(comp, "labels", ("left", "right"))

    # validate
    if len(obs_list) != 2:
        raise ValueError(f"plot_compare_tau_grid_2x2 expects exactly 2 observables, got {len(obs_list)}.")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharex=True, sharey=sharey)
    axes = np.asarray(axes)

    # quantile band label (symmetric)
    q_lo = int(round(100 * q))
    q_hi = int(round(100 * (1 - q)))
    band_txt = f"band: {q_lo}–{q_hi}%"

    for i, obs in enumerate(obs_list):
        Lset = getattr(comp.left, obs)
        Rset = getattr(comp.right, obs)
        if Lset is None or Rset is None:
            raise ValueError(f"obs='{obs}' not computed in compare_taus_from_roots().")

        # left panel
        plot_tau_overlay(
            x, pick(Lset), ax=axes[i, 0],
            log_x=log_x, yscale=yscale, xlim=xlim,
            chain=chain, alpha=alpha, lw=lw, show_summary=show_summary, q=q,
            ylabel=ylabel, xlabel=x_axis,
            title=f"{labels[0]} - {obs}",
        )
        # right panel
        plot_tau_overlay(
            x, pick(Rset), ax=axes[i, 1],
            log_x=log_x, yscale=yscale, xlim=xlim,
            chain=chain, alpha=alpha, lw=lw, show_summary=show_summary, q=q,
            ylabel=ylabel, xlabel=x_axis,
            title=f"{labels[1]} - {obs}",
        )

        # optional per-obs y-limits
        if ylim_by_obs and obs in ylim_by_obs:
            axes[i, 0].set_ylim(*ylim_by_obs[obs])
            axes[i, 1].set_ylim(*ylim_by_obs[obs])

        # annotate band info
        for j in (0, 1):
            axes[i, j].text(
                0.02, 0.98, band_txt,
                transform=axes[i, j].transAxes,
                ha="left", va="top",
                fontsize=9,
                alpha=0.8,
            )

            # de-duplicate legend entries (kills repeated "median")
            leg = axes[i, j].get_legend()
            if leg is not None:
                handles, labels_ = axes[i, j].get_legend_handles_labels()
                seen = set()
                H2, L2 = [], []
                for h, lab in zip(handles, labels_):
                    if lab not in seen:
                        seen.add(lab)
                        H2.append(h); L2.append(lab)
                axes[i, j].legend(H2, L2, frameon=True, fontsize=9)

    if suptitle:
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    return fig, axes
