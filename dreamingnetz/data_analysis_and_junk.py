# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Imports

# %%
import numpy as np
from numba import njit, types
from numba import  int8, float64, int64, uint64, intp, void, boolean, int32

import os, math, json, shutil, pathlib, tempfile
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import Literal, Optional
from multiprocessing import get_context

from datetime import datetime
import time
from typing import cast
from scipy.special import erfcinv

import hashlib
import re

from dataclasses import replace

import matplotlib.pyplot as plt
import os
import glob
from matplotlib import cm  # for colormap



# %% [markdown]
# # Data Analysis

# %% [markdown]
# ## Magnetizations

# %%
def _crossing_temperature(T, w, p_cross=0.5):
    """
    Return T* where w(T) crosses p_cross using linear interpolation.
    Assumes T is sorted increasing. Works whether w decreases or increases.
    Returns np.nan if no crossing.
    """
    w = np.asarray(w, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    s = w - p_cross
    # find indices where sign changes or exactly hits zero
    idx = np.where(s == 0.0)[0]
    if idx.size:
        return float(T[idx[0]])

    sign = np.sign(s)
    changes = np.where(sign[:-1] * sign[1:] < 0)[0]
    if changes.size == 0:
        return np.nan

    i = changes[0]
    # linear interpolation between (T[i],w[i]) and (T[i+1],w[i+1])
    T0, T1 = T[i], T[i+1]
    w0, w1 = w[i], w[i+1]
    if w1 == w0:
        return float(0.5 * (T0 + T1))
    return float(T0 + (p_cross - w0) * (T1 - T0) / (w1 - w0))


def analyze_retrieval_weight(
    run_root,
    m0=0.80,
    rho0=0.40,
    p_cross=0.50,
    eps=1e-12,
    make_plots=True,
):
    """
    Computes retrieval weight w_r(T) per disorder:
      I = 1[ m_max > m0 AND (m_2nd/m_max) < rho0 ]
      w_r(T) = <I>_{replica,time}
    Then averages over disorder and extracts per-disorder crossing temperatures.

    Requires:
      r*/sysconfig.npz with beta or β
      r*/timeseries/*.m_sel.npy of shape (R, Tchunk, K, P)

    Returns dict with:
      T, w_per_disorder (nR,K), w_mean, w_sem,
      Tstar_per_disorder, Tstar_median, Tstar_IQR
    """
    rdirs_all = sorted(glob.glob(os.path.join(run_root, "r*")))
    if not rdirs_all:
        print(f"No r* dirs found in {run_root}")
        return None

    # ---- load temperature grid ----
    syscfg = np.load(os.path.join(rdirs_all[0], "sysconfig.npz"))
    beta_key = "β" if "β" in syscfg.files else ("beta" if "beta" in syscfg.files else None)
    if beta_key is None:
        raise KeyError("sysconfig.npz has no 'beta' or 'β'")
    betas = np.asarray(syscfg[beta_key], dtype=np.float64)
    T_axis = 1.0 / betas
    K = betas.size

    # sort by T for reporting/plotting
    order = np.argsort(T_axis)
    T_sorted = T_axis[order]

    w_list = []
    used_rdirs = []

    for rdir in rdirs_all:
        ts_dir = os.path.join(rdir, "timeseries")
        m_files = sorted(glob.glob(os.path.join(ts_dir, "*.m_sel.npy")))
        if not m_files:
            continue

        sum_I = np.zeros(K, dtype=np.float64)
        cnt   = np.zeros(K, dtype=np.int64)

        ok = True
        for fpath in m_files:
            m = np.load(fpath)  # (R,Tchunk,K,P)
            if m.ndim != 4:
                ok = False
                break
            R, Tchunk, K0, P = m.shape
            if K0 != K:
                raise ValueError(f"{fpath}: K={K0} but sysconfig K={K}")

            # |m|
            np.abs(m, out=m)

            if P >= 2:
                # top-2 along pattern axis without full sort:
                # last two entries are the two largest (unsorted)
                top2 = np.partition(m, P - 2, axis=-1)[..., -2:]   # (R,Tchunk,K,2)
                m_max = np.max(top2, axis=-1)                      # (R,Tchunk,K)
                m_2nd = np.min(top2, axis=-1)                      # (R,Tchunk,K)
            else:
                m_max = m[..., 0]
                m_2nd = np.zeros_like(m_max)

            rho = m_2nd / np.maximum(m_max, eps)
            I = (m_max > m0) & (rho < rho0)

            sum_I += I.sum(axis=(0, 1))
            cnt   += R * Tchunk

            del m, m_max, m_2nd, rho, I

        if (not ok) or np.any(cnt == 0):
            continue

        w_r = (sum_I / cnt)[order]  # sort by T
        w_list.append(w_r)
        used_rdirs.append(rdir)

    if not w_list:
        print("No complete realizations found.")
        return None

    w_per_disorder = np.asarray(w_list, dtype=np.float64)  # (nR,K)
    nR = w_per_disorder.shape[0]

    w_mean = w_per_disorder.mean(axis=0)
    w_sem  = (w_per_disorder.std(axis=0, ddof=1) / np.sqrt(nR)) if nR > 1 else np.zeros_like(w_mean)

    # per-disorder crossing temperatures
    Tstar = np.array([_crossing_temperature(T_sorted, w_per_disorder[r], p_cross) for r in range(nR)], dtype=float)
    Tstar_finite = Tstar[np.isfinite(Tstar)]
    if Tstar_finite.size:
        Tstar_median = float(np.median(Tstar_finite))
        q25, q75 = np.percentile(Tstar_finite, [25, 75])
        Tstar_IQR = (float(q25), float(q75))
    else:
        Tstar_median = np.nan
        Tstar_IQR = (np.nan, np.nan)

    out = {
        "T": T_sorted,
        "w_per_disorder": w_per_disorder,
        "w_mean": w_mean,
        "w_sem": w_sem,
        "Tstar_per_disorder": Tstar,
        "Tstar_median": Tstar_median,
        "Tstar_IQR": Tstar_IQR,
        "rdirs": used_rdirs,
        "params": {"m0": m0, "rho0": rho0, "p_cross": p_cross},
    }

    if make_plots:
        # ---- Plot 1: all disorder curves + mean ± SEM ----
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        ax = axes[0]
        for r in range(nR):
            ax.plot(T_sorted, w_per_disorder[r], alpha=0.25, lw=1.0)
        ax.plot(T_sorted, w_mean, lw=2.5, label="mean")
        ax.fill_between(T_sorted, w_mean - w_sem, w_mean + w_sem, alpha=0.25, label="±SEM")
        ax.axhline(p_cross, ls="--", alpha=0.5)
        ax.set_ylabel(r"$w(T)=\langle I\rangle_{t,\mathrm{rep}}$")
        ax.set_title(rf"Retrieval weight: $I=\mathbf{{1}}[m_{{\max}}>{m0} \wedge (m_{{2nd}}/m_{{\max}})<{rho0}]$  (n_disorders={nR})")
        ax.grid(alpha=0.3)
        ax.legend()

        # ---- Plot 2: heatmap w_r(T) ----
        ax = axes[1]
        im = ax.imshow(
            w_per_disorder,
            aspect="auto",
            origin="lower",
            extent=[T_sorted[0], T_sorted[-1], 0, nR],
        )
        ax.set_ylabel("disorder index")
        ax.set_title("Heatmap of $w_r(T)$")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        # ---- Plot 3: histogram of T* ----
        ax = axes[2]
        if Tstar_finite.size:
            ax.hist(Tstar_finite, bins=min(12, max(4, Tstar_finite.size)), density=False)
            ax.axvline(Tstar_median, ls="--", alpha=0.7, label=f"median={Tstar_median:.3g}")
            ax.legend()
            ax.set_title(rf"Crossing temperatures $T_r^*$ where $w_r(T)={p_cross}$ (IQR={Tstar_IQR[0]:.3g}–{Tstar_IQR[1]:.3g})")
        else:
            ax.text(0.5, 0.5, "No crossings found", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Temperature T")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    return out



# %% [markdown]
# ## E-A Overlap



# %%
def pt_diagnostics_report(
    I_ts_stack,
    betas=None,
    acc_stack=None,
    *,
    q_lo=0.20,                  # disorder-guard quantile for acceptance summaries
    burn_in=0,                  # int steps, or float fraction in (0,1)
    chains=(0, 1),              # which independent chains to analyze (if present)
    plot_disorders=10,           # how many representative disorders to plot as lines
    trace_walkers=6,            # how many walkers to show in k_w(t) traces
    trace_disorder="worst",     # "worst" | "median" | int disorder index
    trace_chain=0,
    save_prefix=None,           # e.g. "diag_" -> saves diag_*.png
    show=True,
):
    """
    Drop-in PT diagnostics from I_ts (slot->walker time series) and optional acc_stack.

    Expected shapes:
      - I_ts_stack: (R, B, T, K) or (B, T, K) or (T, K)
      - acc_stack : (R, K-1) or (K-1,) (optional)

    Returns a dict of computed metrics (arrays are numpy).
    Produces plots (matplotlib) unless show=False.
    """


    # ---------- normalize input shapes ----------
    I = np.asarray(I_ts_stack)
    if I.ndim == 4:
        R, B, T, K = I.shape
    elif I.ndim == 3:
        R = 1
        B, T, K = I.shape
        I = I[None, ...]
    elif I.ndim == 2:
        R = 1
        B = 1
        T, K = I.shape
        I = I[None, None, ...]
    else:
        raise ValueError("I_ts_stack must have shape (R,B,T,K), (B,T,K), or (T,K).")

    if betas is None:
        beta = None
    else:
        beta = np.asarray(betas, float)
        if beta.shape[0] != K:
            raise ValueError(f"betas length {beta.shape[0]} does not match K={K}.")

    # burn-in
    if isinstance(burn_in, float):
        if not (0.0 <= burn_in < 1.0):
            raise ValueError("burn_in as float must be in [0,1).")
        t0 = int(np.floor(burn_in * T))
    else:
        t0 = int(burn_in)
    t0 = max(0, min(t0, T - 1))
    Teff = T - t0

    temperatures = np.sort(1/np.asarray(betas))


    # chains to analyze
    chains = tuple(int(c) for c in chains)
    for c in chains:
        if not (0 <= c < B):
            raise ValueError(f"chain index {c} out of range for B={B}.")

    hot_k, cold_k = 0, K - 1

    # ---------- helpers ----------
    def _roundtrips_from_ends(w_hot, w_cold, K):
        """
        Hot-based round trips: hot->cold->hot for each walker.
        Returns (rt_times list, n_rt, rt_rate).
        """
        t_hot_start = np.full(K, -1, dtype=np.int64)
        seen_cold   = np.zeros(K, dtype=np.bool_)
        rts = []

        for t in range(w_hot.shape[0]):
            wh = int(w_hot[t])
            wc = int(w_cold[t])

            # cold visit marks success-in-between if a hot-start exists
            if t_hot_start[wc] != -1:
                seen_cold[wc] = True

            # hot visit closes a round trip if cold was seen
            if t_hot_start[wh] != -1 and seen_cold[wh]:
                rts.append(t - t_hot_start[wh])

            # (re)start from hot
            t_hot_start[wh] = t
            seen_cold[wh] = False

        n_rt = len(rts)
        rt_rate = n_rt / max(1, w_hot.shape[0])
        return rts, n_rt, rt_rate

    def _passages_from_ends(w_hot, w_cold, K):
        """
        One-way end-to-end passage times:
          - hot->cold and cold->hot, using "last end visited" timestamps per walker.
        Returns two lists: tau_hc, tau_ch.
        """
        last_end = np.zeros(K, dtype=np.int8)     # 0 none, +1 hot, -1 cold
        last_t   = np.full(K, -1, dtype=np.int64)

        tau_hc = []
        tau_ch = []

        for t in range(w_hot.shape[0]):
            wh = int(w_hot[t])
            wc = int(w_cold[t])

            # cold hit
            if last_end[wc] == +1:
                tau_hc.append(t - last_t[wc])
            last_end[wc] = -1
            last_t[wc] = t

            # hot hit
            if last_end[wh] == -1:
                tau_ch.append(t - last_t[wh])
            last_end[wh] = +1
            last_t[wh] = t

        return tau_hc, tau_ch

    def _flow_profile_f(I_bt):
        """
        Directional flow profile f(k) from one chain:
        I_bt: (Teff, K) slot->walker.
        Implements:
          label walker by last end visited: +1 (hot), -1 (cold), 0 unknown.
          f(k) = P(label=+1 | at slot k), estimated over time, excluding unknown labels.
        Returns:
          f: (K,) with NaNs possible if denom=0 at a slot (unlikely)
          denom: (K,) number of labeled samples accumulated at each slot
        """
        labels = np.zeros(K, dtype=np.int8)  # per walker: 0 unknown, +1 hot, -1 cold
        num = np.zeros(K, dtype=np.float64)
        den = np.zeros(K, dtype=np.float64)

        for t in range(I_bt.shape[0]):
            wh = int(I_bt[t, hot_k])
            wc = int(I_bt[t, cold_k])
            labels[wh] = +1
            labels[wc] = -1

            lab_slots = labels[I_bt[t]]  # (K,)
            known = (lab_slots != 0)
            den[known] += 1.0
            num[known] += (lab_slots[known] == +1)

        f = np.full(K, np.nan, dtype=np.float64)
        mask = den > 0
        f[mask] = num[mask] / den[mask]
        return f, den

    def _invert_perm(I_t):
        """Given I_t (K,) slot->walker, return k_of_w (K,) walker->slot."""
        k_of_w = np.empty(K, dtype=np.int16)
        k_of_w[I_t.astype(np.int64)] = np.arange(K, dtype=np.int16)
        return k_of_w

    def _backtracking_rho1(I_bt, walkers=None):
        """
        Backtracking correlation rho1 = Corr(delta k(t), delta k(t+1)) for selected walkers,
        where k_w(t) is walker position (slot index).
        Returns:
          rho_w: (len(walkers),) rho1 per walker (NaN if too few moves)
          rho_med: median rho1 ignoring NaNs
        """
        Te = I_bt.shape[0]
        if walkers is None:
            walkers = np.arange(K, dtype=np.int64)
        else:
            walkers = np.asarray(walkers, dtype=np.int64)

        # build k_w(t) for selected walkers only, streaming
        k_prev = _invert_perm(I_bt[0])[walkers].astype(np.int16)
        dk_prev = None
        xs = [[] for _ in range(walkers.size)]
        ys = [[] for _ in range(walkers.size)]

        for t in range(1, Te):
            k_now = _invert_perm(I_bt[t])[walkers].astype(np.int16)
            dk = (k_now - k_prev).astype(np.int16)
            if dk_prev is not None:
                for i in range(walkers.size):
                    xs[i].append(int(dk_prev[i]))
                    ys[i].append(int(dk[i]))
            dk_prev = dk
            k_prev = k_now

        rho = np.full(walkers.size, np.nan, dtype=np.float64)
        for i in range(walkers.size):
            x = np.array(xs[i], dtype=np.float64)
            y = np.array(ys[i], dtype=np.float64)
            if x.size >= 10 and np.std(x) > 0 and np.std(y) > 0:
                rho[i] = np.corrcoef(x, y)[0, 1]
        rho_med = np.nanmedian(rho)
        return rho, rho_med

    # ---------- compute per-disorder, per-chain metrics ----------
    rt_count = np.zeros((R, B), dtype=np.int64)
    rt_rate  = np.zeros((R, B), dtype=np.float64)
    rt_med   = np.full((R, B), np.nan, dtype=np.float64)

    tau_hc_med = np.full((R, B), np.nan, dtype=np.float64)
    tau_ch_med = np.full((R, B), np.nan, dtype=np.float64)

    f_prof = np.full((R, B, K), np.nan, dtype=np.float64)
    f_slope_max = np.full((R, B), np.nan, dtype=np.float64)

    rho1_med = np.full((R, B), np.nan, dtype=np.float64)

    for r in range(R):
        for b in chains:
            I_bt = I[r, b, t0:, :].astype(np.int64)  # (Teff,K)
            w_hot  = I_bt[:, hot_k]
            w_cold = I_bt[:, cold_k]

            rts, nrt, rate = _roundtrips_from_ends(w_hot, w_cold, K)
            rt_count[r, b] = nrt
            rt_rate[r, b]  = rate
            if rts:
                rt_med[r, b] = float(np.median(np.array(rts, dtype=np.int64)))

            tau_hc, tau_ch = _passages_from_ends(w_hot, w_cold, K)
            if tau_hc:
                tau_hc_med[r, b] = float(np.median(np.array(tau_hc, dtype=np.int64)))
            if tau_ch:
                tau_ch_med[r, b] = float(np.median(np.array(tau_ch, dtype=np.int64)))

            f, _den = _flow_profile_f(I_bt)
            f_prof[r, b, :] = f
            if np.all(np.isnan(f)):
                f_slope_max[r, b] = np.nan
            else:
                df = np.diff(f)
                f_slope_max[r, b] = np.nanmax(np.abs(df))

            # backtracking correlation on a small subset of walkers (cheap + informative)
            walkers = np.linspace(0, K - 1, min(trace_walkers, K), dtype=np.int64)
            _rho, rho_med_val = _backtracking_rho1(I_bt, walkers=walkers)
            rho1_med[r, b] = rho_med_val

    # ---------- acceptance aggregation (optional) ----------
    acc_summary = None
    if acc_stack is not None:
        A = np.asarray(acc_stack, float)
        if A.ndim == 1:
            if A.shape[0] != K - 1:
                raise ValueError("acc_stack length must be K-1.")
            acc_med = A.copy()
            acc_lo  = A.copy()
            acc_hi  = A.copy()
        elif A.ndim == 2:
            if A.shape[1] != K - 1:
                raise ValueError("acc_stack must have shape (R,K-1).")
            acc_med = np.quantile(A, 0.50, axis=0)
            acc_lo  = np.quantile(A, q_lo, axis=0)
            acc_hi  = np.quantile(A, 1.0 - q_lo, axis=0)
        else:
            raise ValueError("acc_stack must be shape (K-1,) or (R,K-1).")

        acc_summary = dict(acc_med=acc_med, acc_lo=acc_lo, acc_hi=acc_hi)

    # ---------- choose representative disorders ----------
    # Use median over analyzed chains of rt_rate to rank disorders
    rt_rate_used = np.nanmedian(rt_rate[:, list(chains)], axis=1)
    worst_r = int(np.nanargmin(rt_rate_used))
    best_r  = int(np.nanargmax(rt_rate_used))
    med_r   = int(np.argsort(rt_rate_used)[len(rt_rate_used)//2]) if R > 1 else 0

    if trace_disorder == "worst":
        trace_r = worst_r
    elif trace_disorder == "median":
        trace_r = med_r
    elif isinstance(trace_disorder, int):
        trace_r = int(trace_disorder)
    else:
        raise ValueError("trace_disorder must be 'worst', 'median', or an int index.")

    reps = [worst_r, med_r, best_r]
    reps = reps[:min(plot_disorders, len(reps))]

    # ---------- plots ----------
    figs = []

    # 1) Acceptance per interface (median/low/high quantiles across disorders)
    if acc_summary is not None:
        fig = plt.figure(figsize=(7, 3.8))
        x = np.arange(K - 1)
        plt.axhline(0.2, linestyle="--")
        plt.plot(x,A.min(axis=0), marker="o", linestyle=":", label="acc min")
        plt.plot(x, acc_summary["acc_med"], marker="o", linestyle="-", label="acc median")
        plt.plot(x, acc_summary["acc_lo"],  marker="o", linestyle="--", label=f"acc q={q_lo:.2f}")
        plt.plot(x, acc_summary["acc_hi"],  marker="o", linestyle="--", label=f"acc q={1-q_lo:.2f}")
        plt.xlabel("interface k (between β_k and β_{k+1})")
        plt.ylabel("acceptance")
        plt.title("Swap acceptance summaries across disorder")
        plt.legend()
        plt.tight_layout()
        figs.append(("acceptance", fig))

    # 2) Round-trip rate per disorder (scatter)
    fig = plt.figure(figsize=(7, 3.8))

    rate1 = 100 / max(1, Teff)
    rate2 = 2/ max(1, Teff)
    plt.axhline(rate1, linestyle="--",label=f"{100} round trips")
    plt.axhline(rate2, linestyle="--",label=f"{2} round trips")
    x = np.arange(R)
    for b in chains:
        plt.plot(x, rt_rate[:, b], marker="o", linestyle="none", label=f"chain {b}")
    plt.xlabel("disorder index r")
    plt.ylabel("round trips per step")
    plt.title(f"Round-trip rate (burn-in t0={t0}, Teff={Teff})")
    #plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    figs.append(("rt_rate", fig))

    # 3) Round-trip time median per disorder
    fig = plt.figure(figsize=(7, 3.8))
    for b in chains:
        plt.plot(x, rt_med[:, b], marker="o", linestyle="none", label=f"chain {b}")
    plt.xlabel("disorder index r")
    plt.ylabel("median RT time (steps)")
    plt.title("Median hot→cold→hot round-trip time")
    #plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    figs.append(("rt_median_time", fig))

    order = np.argsort(rt_rate_used)          # worst -> best
    mid = len(order) // 2

    groups = {
        "worst 3": order[:3],
        "median ±1": order[max(0, mid-1):min(len(order), mid+2)],
        "best 3": order[-3:],
    }

    x = np.arange(K)  # slot index; or use beta / (1/beta) if you want later

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (title, idxs) in zip(axes, groups.items()):
        for r in idxs:
            for b in chains:
                ax.plot(x, f_prof[r, b], marker="o", linestyle="-", label=f"r={r}, b={b}")
        ax.set_title(title)
        ax.set_xlabel("slot k (0 hot → K-1 cold)")
    axes[0].set_ylabel("f(k)")
    axes[0].legend(ncol=1, fontsize=8)
    plt.tight_layout()
    figs.append(("flow_groups", fig))


    # 5) Heatmap of f(k) across disorders (one chain)
    if R > 1:
        fig = plt.figure(figsize=(7, 4.2))
        M = f_prof[:, trace_chain, :]
        plt.imshow(M, aspect="auto", interpolation="nearest")
        plt.colorbar(label="f(k)")
        plt.xlabel("temperature slot k")
        plt.ylabel("disorder index r")
        plt.title(f"f(k) heatmap across disorders (chain {trace_chain})")
        plt.tight_layout()
        figs.append(("flow_heatmap", fig))

    # 6) Walker traces k_w(t) for a few walkers in a chosen disorder
    # This is where k_ts matters: for visualization/debug of "bouncing"/plateaus.
    fig = plt.figure(figsize=(7, 4.2))
    I_bt = I[trace_r, trace_chain, t0:, :].astype(np.int64)
    Te = I_bt.shape[0]
    walkers = np.linspace(0, K - 1, min(trace_walkers, K), dtype=np.int64)

    # build k_w(t) for selected walkers
    k_tr = np.empty((Te, walkers.size), dtype=np.int16)
    for t in range(Te):
        k_of_w = _invert_perm(I_bt[t])
        k_tr[t, :] = k_of_w[walkers]

    t_axis = np.arange(Te)
    for i, w in enumerate(walkers):
        plt.plot(t_axis, k_tr[:, i], linestyle="-", label=f"w={int(w)}")
    plt.xlabel("time step (post burn-in)")
    plt.ylabel("temperature slot k_w(t)")
    plt.title(f"Walker temperature trajectories (r={trace_r}, chain={trace_chain})")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    figs.append(("walker_traces", fig))

    # 7) Backtracking (rho1) per disorder
    fig = plt.figure(figsize=(7, 3.8))
    for b in chains:
        plt.plot(np.arange(R), rho1_med[:, b], marker="o", linestyle="none", label=f"chain {b}")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("disorder index r")
    plt.ylabel("median ρ₁ over sampled walkers")
    plt.title("Backtracking indicator: Corr(Δk(t), Δk(t+1))")
    plt.legend()
    plt.tight_layout()
    figs.append(("backtracking", fig))

    # ---------- save / show ----------
    if save_prefix is not None:
        for name, fig in figs:
            fig.savefig(f"{save_prefix}{name}.png", dpi=150)

    if show:
        plt.show()
    else:
        # avoid GUI side effects; caller can manage figures
        pass

    # ---------- return metrics ----------
    out = dict(
        shape=dict(R=R, B=B, T=T, K=K, burn_in=t0, Teff=Teff),
        betas=beta,
        acc_summary=acc_summary,
        rt_count=rt_count,
        rt_rate=rt_rate,
        rt_med=rt_med,
        tau_hc_med=tau_hc_med,
        tau_ch_med=tau_ch_med,
        f_profile=f_prof,
        f_slope_max=f_slope_max,
        rho1_med=rho1_med,
        representative=dict(worst=worst_r, median=med_r, best=best_r, trace=trace_r),
    )
    return out


# %%
import numpy as np
from dataclasses import replace

def refine_ladder(
    sys_template,                  # SysConfig without β fixed
    beta_proposed: np.ndarray,      # 1D array (K,)
    trial,                          # TrialConfig
    R_workers: int, R_total: int,   # pool knobs (disorders)
    A_low=0.20,
    eps=0.01,                      # hysteresis margin
    max_insert=4,
    K_max=64, max_passes=2, verbose=True, start_method="fork"
):
    """
    Refine a proposed ladder by inserting at the WORST interface across disorders:
        acc_min[k] = min_r acc_stack[r,k]
    until acc_min[k] >= A_low - 2*eps for all k, or until limits are reached.

    Returns: (betas, acc_stack, I_ts_stack) from the last evaluation.
    """

    betas = np.asarray(beta_proposed, float).copy()
    inserts = 0

    last_acc_stack = None
    last_I_ts_stack = None

    thr = A_low - 2.0 * eps

    for p in range(1, max_passes + 1):
        sys = replace(sys_template, K=betas.size, β=betas)

        results = pool_orchestrator_stats(
            sys, trial,
            R_workers=R_workers, R_total=R_total,
            start_method=start_method
        )

        acc_stack  = np.stack([r.acc_edge for r in results], axis=0)  # (R_total, K-1)
        I_ts_stack = np.stack([r.I_ts     for r in results], axis=0)  # (R_total, B, T, K)

        last_acc_stack = acc_stack
        last_I_ts_stack = I_ts_stack

        acc_min = acc_stack.min(axis=0)            # (K-1,)
        k_worst = int(np.argmin(acc_min))
        a_worst = float(acc_min[k_worst])

        if verbose:
            print(f"\nrefine pass {p:2d} | K={betas.size:2d} | "
                  f"worst acc_min={a_worst:.4f} at iface {k_worst} | thr={thr:.4f}")

        # success: no single “doomed” interface remains
        if a_worst >= thr:
            if verbose: print("→ refine done (all acc_min above threshold)")
            return betas, acc_stack, I_ts_stack

        # cannot insert further
        if betas.size >= K_max:
            if verbose: print("→ refine stopped (reached K_max)")
            return betas, acc_stack, I_ts_stack

        if inserts >= max_insert:
            if verbose: print("→ refine stopped (reached max_insert)")
            return betas, acc_stack, I_ts_stack

        # insert midpoint at the *current* worst interface
        betas = np.insert(betas, k_worst + 1, 0.5 * (betas[k_worst] + betas[k_worst + 1]))
        inserts += 1
        if verbose: print(f"＋ insert after iface {k_worst}  -> K={betas.size}")

    if verbose: print("→ refine stopped (reached max_passes)")
    return betas, last_acc_stack, last_I_ts_stack


# %%
def reshape_run_n(
    sys_template,
    trial,
    betas: np.ndarray,
    acc_stack: np.ndarray,
    I_ts_stack: np.ndarray,
    *,
    n_rounds: int,
    R_workers: int,
    R_total: int,
    A_star: float,
    q_lo: float = 0.20,
    gamma: float = 0.5,
    clip=(0.75, 1.35),
    start_method: str = "fork",
    verbose: bool = True,
):
    """
    Fixed-K loop:
      (1) reshape betas using current acc_stack
      (2) run pilot on the reshaped ladder to get new acc_stack and I_ts_stack
    repeated n_rounds times.

    Requires these to exist in your namespace:
      - reshape_betas_from_acceptance(betas, acc_stack, A_star, q_lo=..., gamma=..., clip=...)
      - pool_orchestrator_stats(sys_now, trial, R_workers, R_total, start_method=...)
      - SysConfig has fields K and β (beta array) compatible with replace()

    Returns: (betas, acc_stack, I_ts_stack) from the *last* pilot run.
    """
    betas = np.asarray(betas, float)

    # Basic shape sanity for acc_stack
    acc_stack = np.asarray(acc_stack, float)
    if acc_stack.ndim != 2:
        raise ValueError("acc_stack must have shape (R, K-1).")
    R, K_minus_1 = acc_stack.shape
    if betas.size != K_minus_1 + 1:
        raise ValueError(f"betas length {betas.size} must equal acc_stack.shape[1]+1 = {K_minus_1+1}.")
    if R != R_total:
        # not fatal, but usually you want them consistent
        if verbose:
            print(f"⚠ acc_stack has R={R} but R_total={R_total}; will use R_total for new pilots.")

    for it in range(int(n_rounds)):
        # --- reshape using current acc_stack ---
        betas = reshape_betas_from_acceptance(
            betas, acc_stack, A_star,
            q_lo=q_lo,
            gamma=gamma,
            clip=clip
        )

        # --- rerun pilot ---
        sys_now = replace(sys_template, K=betas.size, β=betas)
        results = pool_orchestrator_stats(sys_now, trial, R_workers, R_total, start_method=start_method)

        acc_stack  = np.stack([r.acc_edge for r in results], axis=0)  # (R_total, K-1)
        I_ts_stack = np.stack([r.I_ts     for r in results], axis=0)  # (R_total, 2, T, K) in your code

        if verbose:
            acc_lo = np.quantile(acc_stack, q_lo, axis=0)
            acc_md = np.quantile(acc_stack, 0.50, axis=0)
            print(f"[reshape_run_n] iter {it+1}/{n_rounds} | K={betas.size} | "
                  f"acc_lo min={acc_lo.min():.3f} | acc_med min={acc_md.min():.3f} | acc_med max={acc_md.max():.3f}")

    return betas, acc_stack, I_ts_stack



def analyze_run_full(run_root):
    """
    Streaming version: same interface, but avoids loading all snapshots at once.

    Returns
    -------
    result : dict with keys
        "T"          : (K,) sorted temperatures
        "m_max"      : <max_μ |m_μ|>
        "m_2nd"      : <2nd-largest |m_μ|>
        "ratio_m"    : <m_2nd/m_max>
        "k_eff"      : <effective # of condensed patterns>
        "S_m"        : <Σ_μ m_μ^2>
        "binder_m"   : Binder cumulant of m_max
        "qEA"        : <|q|>
        "chi_SG"     : N <q^2>
        "binder_q"   : Binder cumulant of q
    """
    rdirs = sorted(glob.glob(os.path.join(run_root, "r*")))
    if not rdirs:
        print(f"No r* dirs found in {run_root}")
        return None

    # --- load basic system info from first realization ---
    syscfg = np.load(os.path.join(rdirs[0], "sysconfig.npz"))
    if "β" in syscfg.files:
        betas = syscfg["β"]
    else:
        betas = syscfg["beta"]
    T_axis = 1.0 / betas
    K = betas.size

    if "N" in syscfg.files:
        N = int(syscfg["N"])
    else:
        raise KeyError("sysconfig.npz has no 'N' entry; needed for chi_SG")

    # =========================================================
    #  accumulators
    # =========================================================
    eps = 1e-16

    # m-based accumulators per temperature index k
    sum_m_max   = np.zeros(K, dtype=np.float64)
    sum_m_2nd   = np.zeros(K, dtype=np.float64)
    sum_ratio   = np.zeros(K, dtype=np.float64)
    sum_k_eff   = np.zeros(K, dtype=np.float64)
    sum_Sm      = np.zeros(K, dtype=np.float64)
    sum_m2_max  = np.zeros(K, dtype=np.float64)
    sum_m4_max  = np.zeros(K, dtype=np.float64)
    count_m     = np.zeros(K, dtype=np.int64)

    # q-based accumulators
    sum_q_abs   = np.zeros(K, dtype=np.float64)
    sum_q2      = np.zeros(K, dtype=np.float64)
    sum_q4      = np.zeros(K, dtype=np.float64)
    count_q     = np.zeros(K, dtype=np.int64)

    found_m = False
    found_q = False
    P_ref = None

    # =========================================================
    #  streaming over realizations and chunks
    # =========================================================
    for rdir in rdirs:
        ts_dir = os.path.join(rdir, "timeseries")

        # ----- m_sel chunks -----
        m_files = sorted(glob.glob(os.path.join(ts_dir, "*.m_sel.npy")))
        for fpath in m_files:
            found_m = True
            m_arr = np.load(fpath)  # (R, T_chunk, K, P)

            if m_arr.ndim != 4:
                raise ValueError(f"{fpath} has shape {m_arr.shape}, expected 4D (R, T_chunk, K, P)")

            R, T_chunk, K_m, P = m_arr.shape
            if K_m != K:
                raise ValueError(
                    f"Inconsistent K in {rdir}: m_sel has K={K_m}, expected {K}"
                )

            if P_ref is None:
                P_ref = P
            elif P != P_ref:
                raise ValueError(f"Inconsistent P across runs: saw {P_ref} and {P}")

            # compute per-snapshot quantities for this chunk
            m_abs = np.abs(m_arr)                          # (R, T_chunk, K, P)
            sorted_m = np.sort(m_abs, axis=-1)[..., ::-1]  # descending in μ

            m_max = sorted_m[..., 0]                       # (R, T_chunk, K)
            if P > 1:
                m_2nd = sorted_m[..., 1]
            else:
                m_2nd = np.zeros_like(m_max)

            # effective number of condensed patterns
            m2_vec = np.sum(m_abs**2, axis=-1)             # (R, T_chunk, K)
            m4_vec = np.sum(m_abs**4, axis=-1)
            k_eff  = (m2_vec**2) / np.maximum(m4_vec, eps)

            # total pattern strength (using signed m)
            S_m_vec = np.sum(m_arr**2, axis=-1)            # (R, T_chunk, K)

            # ratio of 2nd to max overlap
            ratio   = m_2nd / np.maximum(m_max, eps)

            # flatten snapshot axis: (R, T_chunk) -> snapshots
            m_max_flat   = m_max.reshape(-1, K)
            m_2nd_flat   = m_2nd.reshape(-1, K)
            ratio_flat   = ratio.reshape(-1, K)
            k_eff_flat   = k_eff.reshape(-1, K)
            S_m_flat     = S_m_vec.reshape(-1, K)

            # update accumulators
            sum_m_max   += m_max_flat.sum(axis=0)
            sum_m_2nd   += m_2nd_flat.sum(axis=0)
            sum_ratio   += ratio_flat.sum(axis=0)
            sum_k_eff   += k_eff_flat.sum(axis=0)
            sum_Sm      += S_m_flat.sum(axis=0)
            sum_m2_max  += (m_max_flat**2).sum(axis=0)
            sum_m4_max  += (m_max_flat**4).sum(axis=0)
            count_m     += m_max_flat.shape[0]

            # free large arrays
            del m_arr, m_abs, sorted_m, m_max, m_2nd, ratio, m2_vec, m4_vec, k_eff, S_m_vec
            del m_max_flat, m_2nd_flat, ratio_flat, k_eff_flat, S_m_flat

        # ----- q01 chunks -----
        q_files = sorted(glob.glob(os.path.join(ts_dir, "*.q01.npy")))
        for fpath in q_files:
            found_q = True
            q_arr = np.load(fpath)   # (T_chunk, K) or (R, T_chunk, K)? assume (T_chunk, K) as in original

            if q_arr.ndim == 2:
                # (T_chunk, K)
                T_chunk, K_q = q_arr.shape
                if K_q != K:
                    raise ValueError(
                        f"Inconsistent K in {rdir}: q01 has K={K_q}, expected {K}"
                    )
                q_flat = q_arr.reshape(-1, K)
            elif q_arr.ndim == 3:
                # be robust in case it's (R, T_chunk, K)
                R_q, T_chunk, K_q = q_arr.shape
                if K_q != K:
                    raise ValueError(
                        f"Inconsistent K in {rdir}: q01 has K={K_q}, expected {K}"
                    )
                q_flat = q_arr.reshape(-1, K)
            else:
                raise ValueError(f"{fpath} has unexpected shape {q_arr.shape}")

            q_abs  = np.abs(q_flat)
            q2     = q_flat**2

            sum_q_abs += q_abs.sum(axis=0)
            sum_q2    += q2.sum(axis=0)
            sum_q4    += (q2**2).sum(axis=0)
            count_q   += q_flat.shape[0]

            del q_arr, q_flat, q_abs, q2

    # ---------------------------------------------------------
    #  sanity checks: did we actually find any data?
    # ---------------------------------------------------------
    if not found_m:
        print(f"No m_sel data found under {run_root}")
        return None
    if not found_q:
        print(f"No q01 data found under {run_root}")
        return None

    if np.any(count_m == 0):
        raise RuntimeError("Some temperatures have zero m-snapshots; check your data layout.")
    if np.any(count_q == 0):
        raise RuntimeError("Some temperatures have zero q-snapshots; check your data layout.")

    # =========================================================
    #  convert accumulators to observables
    # =========================================================
    # m-based
    m_max_T =   sum_m_max  / count_m
    m_2nd_T =   sum_m_2nd  / count_m
    ratio_T =   sum_ratio  / count_m
    k_eff_T =   sum_k_eff  / count_m
    S_m_T   =   sum_Sm     / count_m

    m2_max  =   sum_m2_max / count_m
    m4_max  =   sum_m4_max / count_m
    binder_m = 1.0 - m4_max / (3.0 * np.maximum(m2_max**2, eps))

    # q-based
    qEA_T  = sum_q_abs / count_q          # <|q|>
    q2_T   = sum_q2    / count_q          # <q^2>
    q4_T   = sum_q4    / count_q          # <q^4>

    chi_SG_T = N * q2_T                    # spin-glass susceptibility
    binder_q = 1.0 - q4_T / (3.0 * np.maximum(q2_T**2, eps))

    # =========================================================
    #  sort everything by T (betas were low->high, so T high->low)
    # =========================================================
    order = np.argsort(T_axis)
    T_sorted   = T_axis[order]

    m_max_T    = m_max_T[order]
    m_2nd_T    = m_2nd_T[order]
    ratio_T    = ratio_T[order]
    k_eff_T    = k_eff_T[order]
    S_m_T      = S_m_T[order]
    binder_m   = binder_m[order]

    qEA_T      = qEA_T[order]
    chi_SG_T   = chi_SG_T[order]
    binder_q   = binder_q[order]

    # =========================================================
    #  plotting: 8 vertical panels, shared x
    # =========================================================
    nrows = 8
    fig, axes = plt.subplots(nrows, 1, figsize=(8, 2.0 * nrows), sharex=True)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes

    # 1) largest & 2nd-largest |m|
    ax1.plot(T_sorted, m_max_T, label=r"$\langle m_{\max}\rangle$")
    ax1.plot(T_sorted, m_2nd_T, label=r"$\langle m_{\text{2nd}}\rangle$")
    ax1.set_ylabel(r"$\langle |m|\rangle$")
    ax1.set_title("Largest & 2nd-largest |m|")
    ax1.legend()

    # 2) ratio
    ax2.plot(T_sorted, ratio_T)
    ax2.set_ylabel(r"$\langle m_{\text{2nd}}/m_{\max}\rangle$")
    ax2.set_title("Relative size of 2nd overlap")

    # 3) k_eff
    ax3.plot(T_sorted, k_eff_T)
    ax3.set_ylabel(r"$\langle k_{\rm eff}\rangle$")
    ax3.set_title("Effective # of condensed patterns")

    # 4) S_m
    ax4.plot(T_sorted, S_m_T)
    ax4.set_ylabel(r"$\langle \sum_\mu m_\mu^2\rangle$")
    ax4.set_title(r"Total pattern strength $S_m$")

    # 5) Binder of m_max
    ax5.plot(T_sorted, binder_m)
    ax5.set_ylabel(r"$U_4^{(m_{\max})}$")
    ax5.set_title("Binder cumulant of retrieval strength")

    # 6) q_EA
    ax6.plot(T_sorted, qEA_T)
    ax6.set_ylabel(r"$\langle |q|\rangle$")
    ax6.set_title(r"EA overlap $q_{\rm EA}(T)$")

    # 7) χ_SG
    ax7.plot(T_sorted, chi_SG_T)
    ax7.set_ylabel(r"$\chi_{\rm SG}$")
    ax7.set_title(r"Spin-glass susceptibility $N\langle q^2\rangle$")

    # 8) Binder of q
    ax8.plot(T_sorted, binder_q)
    ax8.set_ylabel(r"$U_4^{(q)}$")
    ax8.set_xlabel("Temperature T")
    ax8.set_title("Binder cumulant of q")

    plt.tight_layout()
    plt.show()

    return {
        "T":        T_sorted,
        "m_max":    m_max_T,
        "m_2nd":    m_2nd_T,
        "ratio_m":  ratio_T,
        "k_eff":    k_eff_T,
        "S_m":      S_m_T,
        "binder_m": binder_m,
        "qEA":      qEA_T,
        "chi_SG":   chi_SG_T,
        "binder_q": binder_q,
    }




# %%
def _analyze_run_core(run_root):
    """
    Streaming analysis for a single Hopfield PT run.
    Returns a dict with the same observables as the old analyze_run_full,
    but does NOT plot.
    """
    rdirs = sorted(glob.glob(os.path.join(run_root, "r*")))
    if not rdirs:
        print(f"No r* dirs found in {run_root}")
        return None

    # --- load basic system info from first realization ---
    syscfg = np.load(os.path.join(rdirs[0], "sysconfig.npz"))
    if "β" in syscfg.files:
        betas = syscfg["β"]
    else:
        betas = syscfg["beta"]
    T_axis = 1.0 / betas
    K = betas.size

    if "N" in syscfg.files:
        N = int(syscfg["N"])
    else:
        raise KeyError("sysconfig.npz has no 'N' entry; needed for chi_SG")

    # =========================================================
    #  accumulators
    # =========================================================
    eps = 1e-16

    # m-based accumulators per temperature index k
    sum_m_max   = np.zeros(K, dtype=np.float64)
    sum_m_2nd   = np.zeros(K, dtype=np.float64)
    sum_ratio   = np.zeros(K, dtype=np.float64)
    sum_k_eff   = np.zeros(K, dtype=np.float64)
    sum_Sm      = np.zeros(K, dtype=np.float64)
    sum_m2_max  = np.zeros(K, dtype=np.float64)
    sum_m4_max  = np.zeros(K, dtype=np.float64)
    count_m     = np.zeros(K, dtype=np.int64)

    # q-based accumulators
    sum_q_abs   = np.zeros(K, dtype=np.float64)
    sum_q2      = np.zeros(K, dtype=np.float64)
    sum_q4      = np.zeros(K, dtype=np.float64)
    count_q     = np.zeros(K, dtype=np.int64)

    found_m = False
    found_q = False
    P_ref = None

    # =========================================================
    #  streaming over realizations and chunks
    # =========================================================
    for rdir in rdirs:
        ts_dir = os.path.join(rdir, "timeseries")

        # ----- m_sel chunks -----
        m_files = sorted(glob.glob(os.path.join(ts_dir, "*.m_sel.npy")))
        for fpath in m_files:
            found_m = True
            m_arr = np.load(fpath)  # (R, T_chunk, K, P)

            if m_arr.ndim != 4:
                raise ValueError(f"{fpath} has shape {m_arr.shape}, expected 4D (R, T_chunk, K, P)")

            R, T_chunk, K_m, P = m_arr.shape
            if K_m != K:
                raise ValueError(
                    f"Inconsistent K in {rdir}: m_sel has K={K_m}, expected {K}"
                )

            if P_ref is None:
                P_ref = P
            elif P != P_ref:
                raise ValueError(f"Inconsistent P across runs: saw {P_ref} and {P}")

            # compute per-snapshot quantities for this chunk
            m_abs = np.abs(m_arr)                          # (R, T_chunk, K, P)
            sorted_m = np.sort(m_abs, axis=-1)[..., ::-1]  # descending in μ

            m_max = sorted_m[..., 0]                       # (R, T_chunk, K)
            if P > 1:
                m_2nd = sorted_m[..., 1]
            else:
                m_2nd = np.zeros_like(m_max)

            # effective number of condensed patterns
            m2_vec = np.sum(m_abs**2, axis=-1)             # (R, T_chunk, K)
            m4_vec = np.sum(m_abs**4, axis=-1)
            k_eff  = (m2_vec**2) / np.maximum(m4_vec, eps)

            # total pattern strength (using signed m)
            S_m_vec = np.sum(m_arr**2, axis=-1)            # (R, T_chunk, K)

            # ratio of 2nd to max overlap
            ratio   = m_2nd / np.maximum(m_max, eps)

            # flatten snapshot axis: (R, T_chunk) -> snapshots
            m_max_flat   = m_max.reshape(-1, K)
            m_2nd_flat   = m_2nd.reshape(-1, K)
            ratio_flat   = ratio.reshape(-1, K)
            k_eff_flat   = k_eff.reshape(-1, K)
            S_m_flat     = S_m_vec.reshape(-1, K)

            # update accumulators
            sum_m_max   += m_max_flat.sum(axis=0)
            sum_m_2nd   += m_2nd_flat.sum(axis=0)
            sum_ratio   += ratio_flat.sum(axis=0)
            sum_k_eff   += k_eff_flat.sum(axis=0)
            sum_Sm      += S_m_flat.sum(axis=0)
            sum_m2_max  += (m_max_flat**2).sum(axis=0)
            sum_m4_max  += (m_max_flat**4).sum(axis=0)
            count_m     += m_max_flat.shape[0]

            # free large arrays
            del m_arr, m_abs, sorted_m, m_max, m_2nd, ratio, m2_vec, m4_vec, k_eff, S_m_vec
            del m_max_flat, m_2nd_flat, ratio_flat, k_eff_flat, S_m_flat

        # ----- q01 chunks -----
        q_files = sorted(glob.glob(os.path.join(ts_dir, "*.q01.npy")))
        for fpath in q_files:
            found_q = True
            q_arr = np.load(fpath)   # (T_chunk, K) or (R, T_chunk, K)? assume (T_chunk, K) as in original

            if q_arr.ndim == 2:
                # (T_chunk, K)
                T_chunk, K_q = q_arr.shape
                if K_q != K:
                    raise ValueError(
                        f"Inconsistent K in {rdir}: q01 has K={K_q}, expected {K}"
                    )
                q_flat = q_arr.reshape(-1, K)
            elif q_arr.ndim == 3:
                # be robust in case it's (R, T_chunk, K)
                R_q, T_chunk, K_q = q_arr.shape
                if K_q != K:
                    raise ValueError(
                        f"Inconsistent K in {rdir}: q01 has K={K_q}, expected {K}"
                    )
                q_flat = q_arr.reshape(-1, K)
            else:
                raise ValueError(f"{fpath} has unexpected shape {q_arr.shape}")

            q_abs  = np.abs(q_flat)
            q2     = q_flat**2

            sum_q_abs += q_abs.sum(axis=0)
            sum_q2    += q2.sum(axis=0)
            sum_q4    += (q2**2).sum(axis=0)
            count_q   += q_flat.shape[0]

            del q_arr, q_flat, q_abs, q2

    # ---------------------------------------------------------
    #  sanity checks: did we actually find any data?
    # ---------------------------------------------------------
    if not found_m:
        print(f"No m_sel data found under {run_root}")
        return None
    if not found_q:
        print(f"No q01 data found under {run_root}")
        return None

    if np.any(count_m == 0):
        raise RuntimeError("Some temperatures have zero m-snapshots; check your data layout.")
    if np.any(count_q == 0):
        raise RuntimeError("Some temperatures have zero q-snapshots; check your data layout.")

    # =========================================================
    #  convert accumulators to observables
    # =========================================================
    # m-based
    m_max_T =   sum_m_max  / count_m
    m_2nd_T =   sum_m_2nd  / count_m
    ratio_T =   sum_ratio  / count_m
    k_eff_T =   sum_k_eff  / count_m
    S_m_T   =   sum_Sm     / count_m

    m2_max  =   sum_m2_max / count_m
    m4_max  =   sum_m4_max / count_m
    binder_m = 1.0 - m4_max / (3.0 * np.maximum(m2_max**2, eps))

    # q-based
    qEA_T  = sum_q_abs / count_q          # <|q|>
    q2_T   = sum_q2    / count_q          # <q^2>
    q4_T   = sum_q4    / count_q          # <q^4>

    chi_SG_T = N * q2_T                    # spin-glass susceptibility
    binder_q = 1.0 - q4_T / (3.0 * np.maximum(q2_T**2, eps))

    # =========================================================
    #  sort everything by T (betas were low->high, so T high->low)
    # =========================================================
    order = np.argsort(T_axis)
    T_sorted   = T_axis[order]

    m_max_T    = m_max_T[order]
    m_2nd_T    = m_2nd_T[order]
    ratio_T    = ratio_T[order]
    k_eff_T    = k_eff_T[order]
    S_m_T      = S_m_T[order]
    binder_m   = binder_m[order]

    qEA_T      = qEA_T[order]
    chi_SG_T   = chi_SG_T[order]
    binder_q   = binder_q[order]

    return {
        "T":        T_sorted,
        "m_max":    m_max_T,
        "m_2nd":    m_2nd_T,
        "ratio_m":  ratio_T,
        "k_eff":    k_eff_T,
        "S_m":      S_m_T,
        "binder_m": binder_m,
        "qEA":      qEA_T,
        "chi_SG":   chi_SG_T,
        "binder_q": binder_q,
    }

def analyze_run_compare(run_root1, run_root2):
    """
    Analyze two PT runs and plot their observables on the same set of subplots,
    even if their temperature grids differ (as long as each run is internally consistent).

    Parameters
    ----------
    run_root1, run_root2 : str
        Directories containing r*/sysconfig.npz and r*/timeseries/*.m_sel.npy / *.q01.npy

    Returns
    -------
    out : dict
        {"run1": result1, "run2": result2}
        where each result is the dict returned by _analyze_run_core.
    """
    name1 = "α = 0.070"
    name2 = "α = 0.035"

    res1 = _analyze_run_core(run_root1)
    res2 = _analyze_run_core(run_root2)

    if res1 is None or res2 is None:
        print("One or both runs returned no data; skipping plots.")
        return {"run1": res1, "run2": res2}

    T1 = res1["T"]
    T2 = res2["T"]

    # Optional sanity: warn if T ranges don't match
    if not (np.isclose(T1.min(), T2.min()) and np.isclose(T1.max(), T2.max())):
        print(
            "Warning: Temperature ranges differ between runs:\n"
            f"  {name1}: [{T1.min():.4g}, {T1.max():.4g}]\n"
            f"  {name2}: [{T2.min():.4g}, {T2.max():.4g}]"
        )

    # =========================================================
    #  plotting: 8 vertical panels, shared x
    # =========================================================
    nrows = 8
    fig, axes = plt.subplots(nrows, 1, figsize=(8, 2.0 * nrows), sharex=True)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes

    # 1) largest & 2nd-largest |m|
    ax1.plot(T1, res1["m_max"], label=fr"{name1} $\langle m_{{\max}}\rangle$")
    ax1.plot(T1, res1["m_2nd"], label=fr"{name1} $\langle m_{{\text{{2nd}}}}\rangle$")
    ax1.plot(T2, res2["m_max"], "--", label=fr"{name2} $\langle m_{{\max}}\rangle$")
    ax1.plot(T2, res2["m_2nd"], "--", label=fr"{name2} $\langle m_{{\text{{2nd}}}}\rangle$")
    ax1.set_ylabel(r"$\langle |m|\rangle$")
    ax1.set_title(f"Largest & 2nd-largest |m| ({name1} vs {name2})")
    ax1.legend(fontsize=8)

    # 2) ratio
    ax2.plot(T1, res1["ratio_m"], label=name1)
    ax2.plot(T2, res2["ratio_m"], "--", label=name2)
    ax2.set_ylabel(r"$\langle m_{\text{2nd}}/m_{\max}\rangle$")
    ax2.set_title(f"Relative size of 2nd overlap ({name1} vs {name2})")

    # 3) k_eff
    ax3.plot(T1, res1["k_eff"], label=name1)
    ax3.plot(T2, res2["k_eff"], "--", label=name2)
    ax3.set_ylabel(r"$\langle k_{\rm eff}\rangle$")
    ax3.set_title(f"Effective # of condensed patterns ({name1} vs {name2})")

    # 4) S_m
    ax4.plot(T1, res1["S_m"], label=name1)
    ax4.plot(T2, res2["S_m"], "--", label=name2)
    ax4.set_ylabel(r"$\langle \sum_\mu m_\mu^2\rangle$")
    ax4.set_title(rf"Total pattern strength $S_m$ ({name1} vs {name2})")

    # 5) Binder of m_max
    ax5.plot(T1, res1["binder_m"], label=name1)
    ax5.plot(T2, res2["binder_m"], "--", label=name2)
    ax5.set_ylabel(r"$U_4^{(m_{\max})}$")
    ax5.set_title(f"Binder cumulant of retrieval strength ({name1} vs {name2})")

    # 6) q_EA
    ax6.plot(T1, res1["qEA"], label=name1)
    ax6.plot(T2, res2["qEA"], "--", label=name2)
    ax6.set_ylabel(r"$\langle |q|\rangle$")
    ax6.set_title(rf"EA overlap $q_{{\rm EA}}(T)$ ({name1} vs {name2})")

    # 7) χ_SG
    ax7.plot(T1, res1["chi_SG"], label=name1)
    ax7.plot(T2, res2["chi_SG"], "--", label=name2)
    ax7.set_ylabel(r"$\chi_{\rm SG}$")
    ax7.set_title(rf"Spin-glass susceptibility $N\langle q^2\rangle$ ({name1} vs {name2})")

    # 8) Binder of q
    ax8.plot(T1, res1["binder_q"], label=name1)
    ax8.plot(T2, res2["binder_q"], "--", label=name2)
    ax8.set_ylabel(r"$U_4^{(q)}$")
    ax8.set_xlabel("Temperature T")
    ax8.set_title(f"Binder cumulant of q ({name1} vs {name2})")

    # one legend for run labels on the last panel
    ax8.legend(fontsize=8)
    #plt.xlim(0.5,0.7)
    plt.tight_layout()
    plt.show()

    return {"run1": res1, "run2": res2}



# %%
out = analyze_run_compare("runs/hope2_pool", "runs/hope_pool")


# %%
def analyze_binder_per_disorder(run_root, eps=1e-16):
    """
    Compute Binder cumulants U4(m_max) and U4(q) PER disorder realization r*.

    For each rdir:
      - reads timeseries/*.m_sel.npy (R,Tchunk,K,P), computes m_max = max_mu |m_mu|
      - reads timeseries/*.q01.npy   ((Tchunk,K) or (R,Tchunk,K))
      - accumulates <x^2>, <x^4> per temperature
      - returns Binder curves for that rdir

    Then aggregates across rdirs:
      mean and SEM across disorders.

    Returns dict:
      {
        "T": T_sorted,
        "binder_mmax_per_r": (nR, K),
        "binder_q_per_r":    (nR, K),
        "binder_mmax_mean":  (K,),
        "binder_mmax_sem":   (K,),
        "binder_q_mean":     (K,),
        "binder_q_sem":      (K,),
        "rdirs": list_of_rdirs_used
      }
    """
    rdirs_all = sorted(glob.glob(os.path.join(run_root, "r*")))
    if not rdirs_all:
        print(f"No r* dirs found in {run_root}")
        return None

    # --- load T axis from first realization ---
    syscfg = np.load(os.path.join(rdirs_all[0], "sysconfig.npz"))
    beta_key = "β" if "β" in syscfg.files else ("beta" if "beta" in syscfg.files else None)
    if beta_key is None:
        raise KeyError("sysconfig.npz has no 'beta' or 'β'")
    betas = np.asarray(syscfg[beta_key], dtype=np.float64)
    T_axis = 1.0 / betas
    K = betas.size
    order = np.argsort(T_axis)
    T_sorted = T_axis[order]

    binder_mmax_list = []
    binder_q_list = []
    used_rdirs = []

    for rdir in rdirs_all:
        ts_dir = os.path.join(rdir, "timeseries")
        m_files = sorted(glob.glob(os.path.join(ts_dir, "*.m_sel.npy")))
        q_files = sorted(glob.glob(os.path.join(ts_dir, "*.q01.npy")))
        if (not m_files) or (not q_files):
            # skip incomplete realizations
            continue

        # per-rdir accumulators
        sum_m2 = np.zeros(K, dtype=np.float64)
        sum_m4 = np.zeros(K, dtype=np.float64)
        cnt_m  = np.zeros(K, dtype=np.int64)

        sum_q2 = np.zeros(K, dtype=np.float64)
        sum_q4 = np.zeros(K, dtype=np.float64)
        cnt_q  = np.zeros(K, dtype=np.int64)

        # ---- m_max moments ----
        for fpath in m_files:
            m = np.load(fpath)  # (R,Tchunk,K,P)
            if m.ndim != 4:
                raise ValueError(f"{fpath} has shape {m.shape}, expected (R,Tchunk,K,P)")
            R, Tchunk, K0, P = m.shape
            if K0 != K:
                raise ValueError(f"{fpath}: K={K0} but sysconfig K={K}")

            np.abs(m, out=m)          # in-place |m|
            mmax = m.max(axis=-1)     # (R,Tchunk,K)

            m2 = mmax * mmax
            sum_m2 += m2.sum(axis=(0, 1))
            sum_m4 += (m2 * m2).sum(axis=(0, 1))
            cnt_m  += R * Tchunk

            del m, mmax, m2

        # ---- q moments ----
        for fpath in q_files:
            q = np.load(fpath)
            if q.ndim == 2:
                # (Tchunk,K)
                Tchunk, K0 = q.shape
                if K0 != K:
                    raise ValueError(f"{fpath}: K={K0} but sysconfig K={K}")
                q2 = q * q
                sum_q2 += q2.sum(axis=0)
                sum_q4 += (q2 * q2).sum(axis=0)
                cnt_q  += Tchunk
                del q, q2

            elif q.ndim == 3:
                # (R,Tchunk,K)
                Rq, Tchunk, K0 = q.shape
                if K0 != K:
                    raise ValueError(f"{fpath}: K={K0} but sysconfig K={K}")
                q2 = q * q
                sum_q2 += q2.sum(axis=(0, 1))
                sum_q4 += (q2 * q2).sum(axis=(0, 1))
                cnt_q  += Rq * Tchunk
                del q, q2

            else:
                raise ValueError(f"{fpath} has unexpected shape {q.shape}")

        # require data at all temps (otherwise comparison gets messy)
        if np.any(cnt_m == 0) or np.any(cnt_q == 0):
            continue

        m2 = sum_m2 / cnt_m
        m4 = sum_m4 / cnt_m
        binder_mmax = 1.0 - m4 / (3.0 * np.maximum(m2 * m2, eps))

        q2 = sum_q2 / cnt_q
        q4 = sum_q4 / cnt_q
        binder_q = 1.0 - q4 / (3.0 * np.maximum(q2 * q2, eps))

        binder_mmax_list.append(binder_mmax[order])
        binder_q_list.append(binder_q[order])
        used_rdirs.append(rdir)

    if not binder_mmax_list:
        print("No complete realizations found (need both m_sel and q01 in each rdir).")
        return None

    binder_mmax_per_r = np.asarray(binder_mmax_list, dtype=np.float64)  # (nR,K)
    binder_q_per_r    = np.asarray(binder_q_list, dtype=np.float64)     # (nR,K)
    nR = binder_mmax_per_r.shape[0]

    # mean + SEM across disorders
    m_mean = binder_mmax_per_r.mean(axis=0)
    q_mean = binder_q_per_r.mean(axis=0)

    if nR > 1:
        m_sem = binder_mmax_per_r.std(axis=0, ddof=1) / np.sqrt(nR)
        q_sem = binder_q_per_r.std(axis=0, ddof=1) / np.sqrt(nR)
    else:
        m_sem = np.zeros_like(m_mean)
        q_sem = np.zeros_like(q_mean)

    return {
        "T": T_sorted,
        "binder_mmax_per_r": binder_mmax_per_r,
        "binder_q_per_r": binder_q_per_r,
        "binder_mmax_mean": m_mean,
        "binder_mmax_sem": m_sem,
        "binder_q_mean": q_mean,
        "binder_q_sem": q_sem,
        "rdirs": used_rdirs,
    }



# %%
def plot_binder_per_disorder(res, title_prefix="Binder per disorder", alpha=0.60):
    """
    Plot per-disorder Binder curves (faint) plus mean±SEM (bold) for both:
      - U4(m_max)
      - U4(q)

    Parameters
    ----------
    res : dict
        Output of analyze_binder_per_disorder(...)
    title_prefix : str
        Prefix for subplot titles
    alpha : float
        Transparency for individual disorder curves
    """
    if res is None:
        print("res is None")
        return

    T = res["T"]
    bm = res["binder_mmax_per_r"]   # (nR,K)
    bq = res["binder_q_per_r"]      # (nR,K)

    bm_mean = res["binder_mmax_mean"]
    bm_sem  = res["binder_mmax_sem"]
    bq_mean = res["binder_q_mean"]
    bq_sem  = res["binder_q_sem"]

    nR = bm.shape[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # --- U4(m_max) ---
    for r in range(nR):
        ax1.plot(T, bm[r], lw=1.0, alpha=alpha)
    ax1.plot(T, bm_mean, lw=2.5, label="mean")
    ax1.fill_between(T, bm_mean - bm_sem, bm_mean + bm_sem, alpha=0.25, label="±SEM")
    ax1.set_ylabel(r"$U_4(m_{\max})$")
    ax1.set_title(f"{title_prefix}: $U_4(m_{{\\max}})$  (n_disorders={nR})")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # --- U4(q) ---
    for r in range(nR):
        ax2.plot(T, bq[r], lw=1.0, alpha=alpha)
    ax2.plot(T, bq_mean, lw=2.5, label="mean")
    ax2.fill_between(T, bq_mean - bq_sem, bq_mean + bq_sem, alpha=0.25, label="±SEM")
    ax2.set_ylabel(r"$U_4(q)$")
    ax2.set_xlabel("Temperature T")
    ax2.set_title(f"{title_prefix}: $U_4(q)$")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()



# %%
def _crossing_temperature(T, w, p_cross=0.5):
    """
    Return T* where w(T) crosses p_cross using linear interpolation.
    Assumes T is sorted increasing. Works whether w decreases or increases.
    Returns np.nan if no crossing.
    """
    w = np.asarray(w, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    s = w - p_cross
    # find indices where sign changes or exactly hits zero
    idx = np.where(s == 0.0)[0]
    if idx.size:
        return float(T[idx[0]])

    sign = np.sign(s)
    changes = np.where(sign[:-1] * sign[1:] < 0)[0]
    if changes.size == 0:
        return np.nan

    i = changes[0]
    # linear interpolation between (T[i],w[i]) and (T[i+1],w[i+1])
    T0, T1 = T[i], T[i+1]
    w0, w1 = w[i], w[i+1]
    if w1 == w0:
        return float(0.5 * (T0 + T1))
    return float(T0 + (p_cross - w0) * (T1 - T0) / (w1 - w0))


def analyze_retrieval_weight(
    run_root,
    m0=0.80,
    rho0=0.40,
    p_cross=0.50,
    eps=1e-12,
    make_plots=True,
):
    """
    Computes retrieval weight w_r(T) per disorder:
      I = 1[ m_max > m0 AND (m_2nd/m_max) < rho0 ]
      w_r(T) = <I>_{replica,time}
    Then averages over disorder and extracts per-disorder crossing temperatures.

    Requires:
      r*/sysconfig.npz with beta or β
      r*/timeseries/*.m_sel.npy of shape (R, Tchunk, K, P)

    Returns dict with:
      T, w_per_disorder (nR,K), w_mean, w_sem,
      Tstar_per_disorder, Tstar_median, Tstar_IQR
    """
    rdirs_all = sorted(glob.glob(os.path.join(run_root, "r*")))
    if not rdirs_all:
        print(f"No r* dirs found in {run_root}")
        return None

    # ---- load temperature grid ----
    syscfg = np.load(os.path.join(rdirs_all[0], "sysconfig.npz"))
    beta_key = "β" if "β" in syscfg.files else ("beta" if "beta" in syscfg.files else None)
    if beta_key is None:
        raise KeyError("sysconfig.npz has no 'beta' or 'β'")
    betas = np.asarray(syscfg[beta_key], dtype=np.float64)
    T_axis = 1.0 / betas
    K = betas.size

    # sort by T for reporting/plotting
    order = np.argsort(T_axis)
    T_sorted = T_axis[order]

    w_list = []
    used_rdirs = []

    for rdir in rdirs_all:
        ts_dir = os.path.join(rdir, "timeseries")
        m_files = sorted(glob.glob(os.path.join(ts_dir, "*.m_sel.npy")))
        if not m_files:
            continue

        sum_I = np.zeros(K, dtype=np.float64)
        cnt   = np.zeros(K, dtype=np.int64)

        ok = True
        for fpath in m_files:
            m = np.load(fpath)  # (R,Tchunk,K,P)
            if m.ndim != 4:
                ok = False
                break
            R, Tchunk, K0, P = m.shape
            if K0 != K:
                raise ValueError(f"{fpath}: K={K0} but sysconfig K={K}")

            # |m|
            np.abs(m, out=m)

            if P >= 2:
                # top-2 along pattern axis without full sort:
                # last two entries are the two largest (unsorted)
                top2 = np.partition(m, P - 2, axis=-1)[..., -2:]   # (R,Tchunk,K,2)
                m_max = np.max(top2, axis=-1)                      # (R,Tchunk,K)
                m_2nd = np.min(top2, axis=-1)                      # (R,Tchunk,K)
            else:
                m_max = m[..., 0]
                m_2nd = np.zeros_like(m_max)

            rho = m_2nd / np.maximum(m_max, eps)
            I = (m_max > m0) & (rho < rho0)

            sum_I += I.sum(axis=(0, 1))
            cnt   += R * Tchunk

            del m, m_max, m_2nd, rho, I

        if (not ok) or np.any(cnt == 0):
            continue

        w_r = (sum_I / cnt)[order]  # sort by T
        w_list.append(w_r)
        used_rdirs.append(rdir)

    if not w_list:
        print("No complete realizations found.")
        return None

    w_per_disorder = np.asarray(w_list, dtype=np.float64)  # (nR,K)
    nR = w_per_disorder.shape[0]

    w_mean = w_per_disorder.mean(axis=0)
    w_sem  = (w_per_disorder.std(axis=0, ddof=1) / np.sqrt(nR)) if nR > 1 else np.zeros_like(w_mean)

    # per-disorder crossing temperatures
    Tstar = np.array([_crossing_temperature(T_sorted, w_per_disorder[r], p_cross) for r in range(nR)], dtype=float)
    Tstar_finite = Tstar[np.isfinite(Tstar)]
    if Tstar_finite.size:
        Tstar_median = float(np.median(Tstar_finite))
        q25, q75 = np.percentile(Tstar_finite, [25, 75])
        Tstar_IQR = (float(q25), float(q75))
    else:
        Tstar_median = np.nan
        Tstar_IQR = (np.nan, np.nan)

    out = {
        "T": T_sorted,
        "w_per_disorder": w_per_disorder,
        "w_mean": w_mean,
        "w_sem": w_sem,
        "Tstar_per_disorder": Tstar,
        "Tstar_median": Tstar_median,
        "Tstar_IQR": Tstar_IQR,
        "rdirs": used_rdirs,
        "params": {"m0": m0, "rho0": rho0, "p_cross": p_cross},
    }

    if make_plots:
        # ---- Plot 1: all disorder curves + mean ± SEM ----
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        ax = axes[0]
        for r in range(nR):
            ax.plot(T_sorted, w_per_disorder[r], alpha=0.25, lw=1.0)
        ax.plot(T_sorted, w_mean, lw=2.5, label="mean")
        ax.fill_between(T_sorted, w_mean - w_sem, w_mean + w_sem, alpha=0.25, label="±SEM")
        ax.axhline(p_cross, ls="--", alpha=0.5)
        ax.set_ylabel(r"$w(T)=\langle I\rangle_{t,\mathrm{rep}}$")
        ax.set_title(rf"Retrieval weight: $I=\mathbf{{1}}[m_{{\max}}>{m0} \wedge (m_{{2nd}}/m_{{\max}})<{rho0}]$  (n_disorders={nR})")
        ax.grid(alpha=0.3)
        ax.legend()

        # ---- Plot 2: heatmap w_r(T) ----
        ax = axes[1]
        im = ax.imshow(
            w_per_disorder,
            aspect="auto",
            origin="lower",
            extent=[T_sorted[0], T_sorted[-1], 0, nR],
        )
        ax.set_ylabel("disorder index")
        ax.set_title("Heatmap of $w_r(T)$")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        # ---- Plot 3: histogram of T* ----
        ax = axes[2]
        if Tstar_finite.size:
            ax.hist(Tstar_finite, bins=min(12, max(4, Tstar_finite.size)), density=False)
            ax.axvline(Tstar_median, ls="--", alpha=0.7, label=f"median={Tstar_median:.3g}")
            ax.legend()
            ax.set_title(rf"Crossing temperatures $T_r^*$ where $w_r(T)={p_cross}$ (IQR={Tstar_IQR[0]:.3g}–{Tstar_IQR[1]:.3g})")
        else:
            ax.text(0.5, 0.5, "No crossings found", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Temperature T")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    return out



def plot_overlap_histograms(run_root, rid=None, bins=100, range_q=(-1, 1)):
    """
    Plots vertical histograms of the overlap q (P(q)) for each temperature step.

    Streaming version: does NOT build a giant q_snap array; instead it
    accumulates histogram counts per temperature.
    
    Parameters
    ----------
    run_root : str
        Path to the simulation root directory.
    rid : int or str or None, optional
        If None, aggregates all realizations (disorder average).
        If int (e.g., 0), selects the realization at that index in the sorted list.
        If str (e.g., "r002"), selects that specific folder name.
    bins : int
        Number of histogram bins.
    range_q : tuple
        Plotting range for q.
    """
    # --- 1. Find realizations ---
    rdirs = sorted(glob.glob(os.path.join(run_root, "r*")))
    if not rdirs:
        print(f"No r* dirs found in {run_root}")
        return

    # --- Filtering Logic for Single Realization ---
    if rid is not None:
        if isinstance(rid, int):
            # Select by index
            if 0 <= rid < len(rdirs):
                print(f"Selecting realization index {rid}: {os.path.basename(rdirs[rid])}")
                rdirs = [rdirs[rid]]
                title_suffix = f" (Run {os.path.basename(rdirs[0])})"
            else:
                print(f"Error: rid index {rid} out of range (found {len(rdirs)} runs).")
                return
        else:
            # Select by name
            target = str(rid)
            found = [d for d in rdirs if os.path.basename(d) == target]
            if found:
                print(f"Selecting realization: {target}")
                rdirs = found
                title_suffix = f" (Run {target})"
            else:
                print(f"Error: Realization folder '{target}' not found.")
                return
    else:
        title_suffix = " (Disorder Average)"

    # --- 2. Load system info and temperature axis ---
    sys = np.load(os.path.join(rdirs[0], "sysconfig.npz"))
    if "β" in sys.files:
        betas = sys["β"]
    elif "beta" in sys.files:
        betas = sys["beta"]
    else:
        raise KeyError("Could not find 'beta' or 'β' in sysconfig.npz")

    K = betas.size

    # Calculate Temperatures (handle possible beta=0 gracefully)
    T_axis = np.divide(1.0, betas, out=np.full_like(betas, np.inf), where=betas != 0)

    # --- 3. Prepare histogram accumulators ---
    bin_edges = np.linspace(range_q[0], range_q[1], bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    # hist_counts[k, :] = counts for temperature index k
    hist_counts = np.zeros((K, bins), dtype=np.float64)

    any_q = False

    # --- 4. Stream over q01 data and accumulate histograms ---
    for rdir in rdirs:
        ts_dir = os.path.join(rdir, "timeseries")
        q_files = sorted(glob.glob(os.path.join(ts_dir, "*.q01.npy")))
        if not q_files:
            continue

        for fpath in q_files:
            q_arr = np.load(fpath)

            # Support both (T_chunk, K) and (R, T_chunk, K) just in case
            if q_arr.ndim == 2:
                T_chunk, K_q = q_arr.shape
                if K_q != K:
                    print(f"Warning: Sketchy K dim in {rdir}, file {os.path.basename(fpath)}, skipping.")
                    continue
                q_flat = q_arr  # (T_chunk, K)
            elif q_arr.ndim == 3:
                R_q, T_chunk, K_q = q_arr.shape
                if K_q != K:
                    print(f"Warning: Sketchy K dim in {rdir}, file {os.path.basename(fpath)}, skipping.")
                    continue
                q_flat = q_arr.reshape(-1, K)  # (R_q*T_chunk, K)
            else:
                print(f"Warning: Unexpected q01 shape {q_arr.shape} in {fpath}, skipping.")
                continue

            # Accumulate histogram counts per temperature index
            # q_flat[:, k] is all snapshot values for temperature k in this chunk
            for k in range(K):
                data_k = q_flat[:, k]
                # counts only within range_q (np.histogram's `range` does that)
                counts, _ = np.histogram(data_k, bins=bin_edges, range=range_q)
                hist_counts[k] += counts

            any_q = True
            del q_arr, q_flat

    if not any_q:
        print("No q01 data found.")
        return

    # --- 5. Sort temperatures high -> low ---
    sorted_indices = np.argsort(T_axis)[::-1]  # Descending T
    T_sorted = T_axis[sorted_indices]
    hist_sorted = hist_counts[sorted_indices]

    # --- 6. Plotting ---
    fig_height = max(5, 0.8 * K)
    fig, axes = plt.subplots(K, 1, figsize=(6, fig_height), sharex=True, sharey=False)
    if K == 1:
        axes = [axes]

    print(f"Plotting P(q) for {K} temperatures...")

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for i, ax in enumerate(axes):
        counts = hist_sorted[i]
        total = counts.sum()

        if total > 0:
            # density such that ∫ P(q) dq = 1 over range_q
            density = counts / (total * bin_width)
        else:
            density = counts  # all zeros

        # mimic "stepfilled" style: outline + filled area
        ax.step(bin_centers, density, where="mid", color="navy")
        ax.fill_between(bin_centers, density, step="mid", alpha=0.7, color="navy")

        # Annotation
        label = f"$T={T_sorted[i]:.3f}$ "
        ax.text(
            0.98, 0.9, label,
            transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle="round", facecolor='white', alpha=0.8)
        )

        # Y-axis tweaks
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_ylabel(r"$P(q)$", rotation=0, labelpad=15, fontsize=8)

    axes[-1].set_xlabel(r"Overlap $q$")
    axes[-1].set_xlim(range_q)

    plt.suptitle(r"Parisi Order Parameter $P(q)$" + title_suffix, y=1.005)
    plt.tight_layout()
    plt.show()



# %%
def _select_rdirs(run_root, rid=None):
    rdirs = sorted(glob.glob(os.path.join(run_root, "r*")))
    if not rdirs:
        raise FileNotFoundError(f"No r* dirs found in {run_root}")

    if rid is None:
        return rdirs, " (disorder set)"

    if isinstance(rid, int):
        if rid < 0 or rid >= len(rdirs):
            raise IndexError(f"rid={rid} out of range (found {len(rdirs)} rdirs)")
        return [rdirs[rid]], f" ({os.path.basename(rdirs[rid])})"

    target = str(rid)
    found = [d for d in rdirs if os.path.basename(d) == target]
    if not found:
        raise FileNotFoundError(f"Realization '{target}' not found under {run_root}")
    return found, f" ({target})"


def _load_T_axis(rdir):
    syscfg = np.load(os.path.join(rdir, "sysconfig.npz"))
    beta_key = "β" if "β" in syscfg.files else ("beta" if "beta" in syscfg.files else None)
    if beta_key is None:
        raise KeyError("sysconfig.npz has no 'beta' or 'β'")
    betas = np.asarray(syscfg[beta_key], dtype=np.float64)
    T = np.divide(1.0, betas, out=np.full_like(betas, np.inf), where=betas != 0)
    return T, betas


def analyze_conditional_Pq(
    run_root,
    rid=None,
    m0=0.80,
    rho0=0.40,
    bins=120,
    range_q=(-1.0, 1.0),
    q0_mid=0.30,
    eps=1e-12,
    per_disorder_average=True,
):
    """
    Build conditional overlap distributions P(q | sector) where sector is RR/NN/MX.

    Uses m_sel chunks (R=2 replicas): (2, Tchunk, K, P)
    Uses q01 chunks: ideally (Tchunk, K), but supports (Rq, Tchunk, K) by tiling masks.

    If per_disorder_average=True:
      - normalizes histograms per disorder and temperature (within each sector)
      - then averages densities across disorders equally (nanmean)
    else:
      - pools counts across disorders first (sample-weighted)

    Returns dict with mean densities + weights (and per-disorder if averaged).
    """
    rdirs, title_suffix = _select_rdirs(run_root, rid=rid)
    T_axis, _ = _load_T_axis(rdirs[0])
    K = T_axis.size

    edges = np.linspace(range_q[0], range_q[1], bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bw = edges[1] - edges[0]
    order = np.argsort(T_axis)
    T_sorted = T_axis[order]

    sectors = ["RR", "NN", "MX"]

    # Store per-disorder densities if doing equal-weight disorder average
    dens_per_r = {s: [] for s in sectors}   # each entry -> (K,bins) density or nan where empty
    w_per_r    = {s: [] for s in sectors}   # each entry -> (K,) sector weight among all samples

    # Or pooled accumulators
    pooled_counts = {s: np.zeros((K, bins), dtype=np.float64) for s in sectors}
    pooled_n      = {s: np.zeros(K, dtype=np.float64) for s in sectors}

    any_data = False

    for rdir in rdirs:
        ts_dir = os.path.join(rdir, "timeseries")
        m_files = sorted(glob.glob(os.path.join(ts_dir, "*.m_sel.npy")))
        q_files = sorted(glob.glob(os.path.join(ts_dir, "*.q01.npy")))
        if (not m_files) or (not q_files):
            continue

        # map by common prefix if possible
        def k_m(fp): return os.path.basename(fp).replace(".m_sel.npy", "")
        def k_q(fp): return os.path.basename(fp).replace(".q01.npy", "")

        q_map = {k_q(fp): fp for fp in q_files}
        keys = [k_m(fp) for fp in m_files if k_m(fp) in q_map]

        if len(keys) == 0 and len(m_files) == len(q_files):
            pairs = list(zip(m_files, q_files))
        else:
            pairs = [(os.path.join(ts_dir, k + ".m_sel.npy"), q_map[k]) for k in keys]

        # per-realization accumulators
        counts_r = {s: np.zeros((K, bins), dtype=np.float64) for s in sectors}
        n_r      = {s: np.zeros(K, dtype=np.float64) for s in sectors}

        for mpath, qpath in pairs:
            m = np.load(mpath)  # (2, Tchunk, K, P)
            if m.ndim != 4 or m.shape[0] != 2:
                continue
            R, Tchunk, K0, P = m.shape
            if K0 != K:
                raise ValueError(f"{mpath}: K={K0} but expected K={K}")

            q = np.load(qpath)

            # make q_flat as (nSnap, K) and build mask tiling accordingly
            if q.ndim == 2:
                Tq, Kq = q.shape
                if Kq != K or Tq != Tchunk:
                    continue
                q_flat = q
                tile_factor = 1
            elif q.ndim == 3:
                Rq, Tq, Kq = q.shape
                if Kq != K or Tq != Tchunk:
                    continue
                q_flat = q.reshape(-1, K)
                tile_factor = Rq
            else:
                continue

            # compute retrieval indicator per replica/time/temp from m
            np.abs(m, out=m)
            if P >= 2:
                top2 = np.partition(m, P - 2, axis=-1)[..., -2:]   # (2,Tchunk,K,2)
                mmax = np.max(top2, axis=-1)                       # (2,Tchunk,K)
                m2nd = np.min(top2, axis=-1)
            else:
                mmax = m[..., 0]
                m2nd = np.zeros_like(mmax)

            rho = m2nd / np.maximum(mmax, eps)
            I0 = (mmax[0] > m0) & (rho[0] < rho0)   # (Tchunk,K)
            I1 = (mmax[1] > m0) & (rho[1] < rho0)

            RR = (I0 & I1)
            NN = (~I0 & ~I1)
            MX = (I0 ^ I1)

            # tile masks if q had leading extra dim and got flattened
            if tile_factor != 1:
                RR = np.tile(RR, (tile_factor, 1))
                NN = np.tile(NN, (tile_factor, 1))
                MX = np.tile(MX, (tile_factor, 1))

            # accumulate histograms per temperature
            for k in range(K):
                qk = q_flat[:, k]

                for s, mask in (("RR", RR[:, k]), ("NN", NN[:, k]), ("MX", MX[:, k])):
                    if mask.any():
                        c, _ = np.histogram(qk[mask], bins=edges, range=range_q)
                        counts_r[s][k] += c
                        n_r[s][k] += mask.sum()

            any_data = True
            del m, q, q_flat, top2, mmax, m2nd, rho, I0, I1, RR, NN, MX

        if not any_data:
            continue

        if per_disorder_average:
            # per-disorder normalized densities (per sector, per temperature)
            dens_r = {}
            w_r = {}
            total_per_T = n_r["RR"] + n_r["NN"] + n_r["MX"]

            for s in sectors:
                dens = np.full((K, bins), np.nan, dtype=np.float64)
                for k in range(K):
                    tot = counts_r[s][k].sum()
                    if tot > 0:
                        dens[k] = counts_r[s][k] / (tot * bw)
                dens_r[s] = dens[order]

                w = np.zeros(K, dtype=np.float64)
                good = total_per_T > 0
                w[good] = n_r[s][good] / total_per_T[good]
                w_r[s] = w[order]

                dens_per_r[s].append(dens_r[s])
                w_per_r[s].append(w_r[s])

        else:
            # pool counts directly
            for s in sectors:
                pooled_counts[s] += counts_r[s]
                pooled_n[s] += n_r[s]

    if not any_data:
        print("No usable (m_sel,q01) chunk pairs found.")
        return None

    # build outputs
    out = {
        "T": T_sorted,
        "q_centers": centers,
        "bin_width": bw,
        "sectors": sectors,
        "params": {"m0": m0, "rho0": rho0, "q0_mid": q0_mid},
        "title_suffix": title_suffix,
    }

    if per_disorder_average:
        # convert lists -> arrays (nR, K, bins) and (nR, K)
        dens = {s: np.asarray(dens_per_r[s], dtype=np.float64) for s in sectors}
        wght = {s: np.asarray(w_per_r[s], dtype=np.float64) for s in sectors}

        out["dens_per_disorder"] = dens
        out["w_per_disorder"] = wght

        # mean + SEM across disorders (ignore nan bins where sector empty)
        dens_mean = {}
        dens_sem = {}
        w_mean = {}
        w_sem = {}

        for s in sectors:
            D = dens[s]   # (nR,K,bins)
            W = wght[s]   # (nR,K)
            dens_mean[s] = np.nanmean(D, axis=0)
            # SEM with nan handling: count non-nan per (K,bins)
            nn = np.sum(np.isfinite(D), axis=0)
            sd = np.nanstd(D, axis=0, ddof=1)
            dens_sem[s] = np.where(nn > 1, sd / np.sqrt(nn), 0.0)

            w_mean[s] = np.nanmean(W, axis=0)
            nW = np.sum(np.isfinite(W), axis=0)
            sdW = np.nanstd(W, axis=0, ddof=1)
            w_sem[s] = np.where(nW > 1, sdW / np.sqrt(nW), 0.0)

        out["dens_mean"] = dens_mean
        out["dens_sem"] = dens_sem
        out["w_mean"] = w_mean
        out["w_sem"] = w_sem

    else:
        # pooled densities per sector
        dens_mean = {}
        w_mean = {}
        total_per_T = pooled_n["RR"] + pooled_n["NN"] + pooled_n["MX"]

        for s in sectors:
            dens = np.zeros((K, bins), dtype=np.float64)
            for k in range(K):
                tot = pooled_counts[s][k].sum()
                if tot > 0:
                    dens[k] = pooled_counts[s][k] / (tot * bw)
            dens_mean[s] = dens[order]

            w = np.zeros(K, dtype=np.float64)
            good = total_per_T > 0
            w[good] = pooled_n[s][good] / total_per_T[good]
            w_mean[s] = w[order]

        out["dens_mean"] = dens_mean
        out["w_mean"] = w_mean

    return out


def plot_conditional_Pq(res, k_to_plot=None, max_rows=6):
    """
    Plots:
      (1) sector weights vs T
      (2) mid-weight diagnostics vs T
      (3) conditional histograms at selected temperatures (overlay RR/NN/MX)

    Expects output of analyze_conditional_Pq(..., per_disorder_average=True/False).
    """
    if res is None:
        return

    T = res["T"]
    qc = res["q_centers"]
    bw = res["bin_width"]
    sectors = res["sectors"]
    q0_mid = res["params"]["q0_mid"]
    suffix = res.get("title_suffix", "")

    dens = res["dens_mean"]
    w = res["w_mean"]

    # -------- Plot 1: sector weights vs T --------
    plt.figure(figsize=(7, 4))
    for s in sectors:
        if "w_sem" in res:
            plt.errorbar(T, w[s], yerr=res["w_sem"][s], capsize=2, label=s)
        else:
            plt.plot(T, w[s], label=s)
    plt.xlabel("Temperature T")
    plt.ylabel("sector weight")
    plt.title(rf"Sector weights vs T{suffix}  (RR: both retrieval, NN: both non, MX: mixed)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------- Plot 2: middle weight diagnostics --------
    mid_mask = (np.abs(qc) < q0_mid)
    plt.figure(figsize=(7, 4))
    for s in sectors:
        # conditional prob of being in middle given sector
        p_mid_cond = dens[s][:, mid_mask].sum(axis=1) * bw
        # unconditional contribution to middle = weight(sector)*p_mid_cond
        p_mid_uncond = w[s] * p_mid_cond
        plt.plot(T, p_mid_uncond, label=f"{s}: w·P(|q|<{q0_mid})")
    plt.xlabel("Temperature T")
    plt.ylabel("unconditional middle weight")
    plt.title(rf"Where does the middle support come from?{suffix}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------- Plot 3: conditional histograms at selected temperatures --------
    K = len(T)
    if k_to_plot is None:
        # auto-pick: max mixed weight, plus neighbors, plus one high-T and one low-T
        k_mix = int(np.nanargmax(w["MX"]))
        candidates = [0, max(0, k_mix - 1), k_mix, min(K - 1, k_mix + 1), K - 1]
        # unique while preserving order
        seen = set()
        k_to_plot = [k for k in candidates if (k not in seen and not seen.add(k))]
        k_to_plot = k_to_plot[:max_rows]

    nrows = len(k_to_plot)
    fig, axes = plt.subplots(nrows, 1, figsize=(7, 2.2 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    for ax, k in zip(axes, k_to_plot):
        for s in sectors:
            ax.step(qc, dens[s][k], where="mid", label=s)
        ax.set_title(
            rf"T={T[k]:.3f}   "
            + "  ".join([f"{s}:{w[s][k]:.2f}" for s in sectors])
        )
        ax.set_ylabel("density")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("overlap q")
    axes[0].legend(ncol=3)
    plt.suptitle(rf"Conditional $P(q \mid \mathrm{{sector}})$  with  I=[m_max>{res['params']['m0']} & rho<{res['params']['rho0']}] {suffix}", y=1.01)
    plt.tight_layout()
    plt.show()



# %%
def save_conditionalPq_output(res, out_path="condPq_out.npz", T_window=None):
    """
    Save a compact version of analyze_conditional_Pq output.
    Optionally restrict to a temperature window: T_window=(Tmin,Tmax).
    """
    T = res["T"]
    qc = res["q_centers"]

    ksel = np.arange(T.size)
    if T_window is not None:
        Tmin, Tmax = T_window
        ksel = np.where((T >= Tmin) & (T <= Tmax))[0]

    payload = {
        "T": T[ksel],
        "q_centers": qc,
        "bin_width": res["bin_width"],
        "m0": res["params"]["m0"],
        "rho0": res["params"]["rho0"],
        "q0_mid": res["params"]["q0_mid"],
    }

    for s in ["RR", "NN", "MX"]:
        payload[f"dens_{s}"] = res["dens_mean"][s][ksel]   # (Ksel,bins)
        payload[f"w_{s}"]    = res["w_mean"][s][ksel]      # (Ksel,)

        if "dens_sem" in res:
            payload[f"densSEM_{s}"] = res["dens_sem"][s][ksel]
        if "w_sem" in res:
            payload[f"wSEM_{s}"] = res["w_sem"][s][ksel]

    np.savez_compressed(out_path, **payload)
    print("saved:", out_path, "  (T points:", len(ksel), ", bins:", len(qc), ")")



def _to_list(x):
    """Helper: turn int/None/list/ndarray into a Python list."""
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    else:
        return [x]



# %%
def plot_m_timeseries_grid(
    run_root,
    disorder_idx=0,
    replica=0,
    temperature_idx=0,
    pattern=0,
    abs_val=True,
):
    """
    Plot time series of m for selected replicas, temperatures and patterns,
    streaming over m_sel chunks instead of holding the full (R,T,K,P) array.
    """

    # --- Find realization directory ---
    rdirs = sorted(glob.glob(os.path.join(run_root, "r*")))
    if not rdirs:
        print("No realizations found in", run_root)
        return

    if disorder_idx < 0 or disorder_idx >= len(rdirs):
        raise IndexError(
            f"disorder_idx={disorder_idx} out of range, found {len(rdirs)} realizations."
        )

    rdir = rdirs[disorder_idx]

    # --- Load betas and temps for titles ---
    syscfg = np.load(os.path.join(rdir, "sysconfig.npz"))
    if "beta" in syscfg.files:
        betas = np.array(syscfg["beta"])
    elif "β" in syscfg.files:
        betas = np.array(syscfg["β"])
    else:
        raise KeyError("sysconfig.npz has no 'beta' or 'β'")
    T_vals = 1.0 / betas
    K = betas.size

    # --- List timeseries files ---
    m_files = sorted(glob.glob(os.path.join(rdir, "timeseries", "*.m_sel.npy")))
    if not m_files:
        print(f"No m_sel timeseries found in {rdir}")
        return

    # --- First pass: infer R, K, P and total time length T_total ---
    R = None
    P = None
    T_total = 0
    for fpath in m_files:
        m_chunk = np.load(fpath, mmap_mode="r")
        if m_chunk.ndim != 4:
            raise ValueError(
                f"{fpath} has shape {m_chunk.shape}, expected 4D (R, T_chunk, K, P)"
            )
        R_chunk, T_chunk, K_chunk, P_chunk = m_chunk.shape
        if K_chunk != K:
            raise ValueError(f"Inconsistent K in {fpath}: {K_chunk} vs {K}")
        if R is None:
            R = R_chunk
            P = P_chunk
        else:
            if R_chunk != R or P_chunk != P:
                raise ValueError(
                    f"Inconsistent (R,P) across files: saw (R={R},P={P}) and "
                    f"(R={R_chunk},P={P_chunk})"
                )
        T_total += T_chunk

    # --- Normalize inputs to lists now that we know (R,K,P) ---
    replicas = _to_list(replica)
    betas_idx = _to_list(temperature_idx)
    patterns = _to_list(pattern)

    if replicas is None:
        replicas = list(range(R))
    if betas_idx is None:
        betas_idx = list(range(K))
    if patterns is None:
        patterns = list(range(P))

    for rep in replicas:
        if rep < 0 or rep >= R:
            raise IndexError(f"replica index {rep} out of range [0, {R-1}]")
    for k_idx in betas_idx:
        if k_idx < 0 or k_idx >= K:
            raise IndexError(f"temperature_idx {k_idx} out of range [0, {K-1}]")
    for p_idx in patterns:
        if p_idx < 0 or p_idx >= P:
            raise IndexError(f"pattern index {p_idx} out of range [0, {P-1}]")

    n_rows = len(betas_idx)
    n_cols = len(replicas)

    # --- Allocate storage only for requested time series ---
    # ts[i_row][j_col] has shape (T_total, len(patterns))
    ts = [
        [np.zeros((T_total, len(patterns)), dtype=np.float32) for _ in range(n_cols)]
        for __ in range(n_rows)
    ]

    # --- Second pass: fill time series chunk-by-chunk ---
    t_offset = 0
    for fpath in m_files:
        m_chunk = np.load(fpath)  # (R, T_chunk, K, P)
        if m_chunk.ndim != 4:
            raise ValueError(
                f"{fpath} has shape {m_chunk.shape}, expected 4D (R, T_chunk, K, P)"
            )

        if abs_val:
            m_chunk = np.abs(m_chunk)

        _, T_chunk, _, _ = m_chunk.shape

        for i_row, k_idx in enumerate(betas_idx):
            for j_col, rep in enumerate(replicas):
                # (T_chunk, P)
                sub = m_chunk[rep, :, k_idx, :]
                # select requested patterns into columns
                ts[i_row][j_col][t_offset : t_offset + T_chunk, :] = sub[:, patterns]

        t_offset += T_chunk
        del m_chunk

    # --- Plotting ---
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3 * n_rows),
        sharex=True,
        sharey=True,
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    t = np.arange(T_total)

    for i_row, k_idx in enumerate(betas_idx):
        for j_col, rep in enumerate(replicas):
            ax = axes[i_row, j_col]
            m_ts = ts[i_row][j_col]  # (T_total, len(patterns))

            for j_p, p_idx in enumerate(patterns):
                ax.plot(t, m_ts[:, j_p], label=f"p={p_idx}", alpha=0.8)

            T_val = T_vals[k_idx]
            ax.set_title(f"rep={rep}, T_idx={k_idx}, T={T_val:.3g}")

            if i_row == n_rows - 1:
                ax.set_xlabel("time step")
            if j_col == 0:
                ax.set_ylabel("|m|" if abs_val else "m")

    if n_rows * n_cols == 1:
        axes[0, 0].legend()
    else:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.02, 1.02))

    plt.tight_layout()
    plt.show()



# %%
def find_ever_retrieved_patterns(run_root, disorder_idx=0, threshold=0.9):
    """
    Streaming version: only keeps the per-pattern max |m| across all
    replicas, times, and temperatures.
    """
    # locate realization directory
    rdirs = sorted(glob.glob(os.path.join(run_root, "r*")))
    if not rdirs:
        raise RuntimeError(f"No realizations found in {run_root}")

    if disorder_idx < 0 or disorder_idx >= len(rdirs):
        raise IndexError(
            f"disorder_idx={disorder_idx} out of range; found {len(rdirs)} realizations."
        )

    rdir = rdirs[disorder_idx]

    m_files = sorted(glob.glob(os.path.join(rdir, "timeseries", "*.m_sel.npy")))
    if not m_files:
        raise RuntimeError(f"No m_sel timeseries found in {rdir}")

    max_over_all = None
    P = None

    for fpath in m_files:
        m_chunk = np.load(fpath)  # (R, T_chunk, K, P)
        if m_chunk.ndim != 4:
            raise ValueError(
                f"Expected m_chunk with 4 dims (R,T,K,P), got shape {m_chunk.shape}"
            )

        abs_m = np.abs(m_chunk)
        max_chunk = abs_m.max(axis=(0, 1, 2))  # -> (P,)

        if max_over_all is None:
            max_over_all = max_chunk
            P = max_chunk.shape[0]
        else:
            if max_chunk.shape[0] != P:
                raise ValueError(
                    f"Inconsistent P across chunks: expected {P}, got {max_chunk.shape[0]}"
                )
            max_over_all = np.maximum(max_over_all, max_chunk)

        del m_chunk, abs_m, max_chunk

    mask_ever = max_over_all >= threshold
    ever_indices = np.where(mask_ever)[0]

    return ever_indices, max_over_all


def plot_all_heatmaps_corrected(run_root):
    rdirs = sorted(glob.glob(os.path.join(run_root, "r*")))
    if not rdirs:
        return

    # 1. Load physics and sort T properly
    syscfg = np.load(os.path.join(rdirs[0], "sysconfig.npz"))
    if "beta" in syscfg.files:
        betas = syscfg["beta"]
    elif "β" in syscfg.files:
        betas = syscfg["β"]
    else:
        raise KeyError("sysconfig.npz has no 'beta' or 'β'")
    T_axis = 1.0 / betas
    K = betas.size

    # Sort T ascending (low T -> high T)
    sort_idx = np.argsort(T_axis)
    T_sorted = T_axis[sort_idx]

    n_rows = 5
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 20), sharex=True)
    axes = axes.flatten()

    im = None  # last image for colorbar

    for i, ax in enumerate(axes):
        if i >= len(rdirs):
            ax.axis("off")
            continue

        rdir = rdirs[i]
        m_files = sorted(glob.glob(os.path.join(rdir, "timeseries", "*.m_sel.npy")))
        if not m_files:
            ax.axis("off")
            continue

        try:
            sum_abs = None
            count = 0

            # Streaming average over R,time
            for fpath in m_files:
                m_chunk = np.load(fpath)  # (R, T_chunk, K, P)
                if m_chunk.ndim != 4:
                    raise ValueError(
                        f"{fpath} has unexpected shape {m_chunk.shape}, expected (R,T,K,P)"
                    )
                abs_chunk = np.abs(m_chunk)
                # sum over replicas and time -> (K, P)
                partial_sum = abs_chunk.sum(axis=(0, 1))  # (K, P)
                if sum_abs is None:
                    sum_abs = partial_sum
                else:
                    if partial_sum.shape != sum_abs.shape:
                        raise ValueError(
                            f"Inconsistent (K,P) in {rdir}: got {partial_sum.shape}, "
                            f"expected {sum_abs.shape}"
                        )
                    sum_abs += partial_sum

                count += abs_chunk.shape[0] * abs_chunk.shape[1]
                del m_chunk, abs_chunk, partial_sum

            if sum_abs is None or count == 0:
                ax.axis("off")
                continue

            # <|m|> over replicas and time, shape (K,P)
            m_mean = sum_abs / count

            # Reorder to match sorted T (low->high), then transpose to (P,K)
            if m_mean.shape[0] != K:
                raise ValueError(
                    f"Expected first dim of m_mean to be K={K}, got {m_mean.shape[0]}"
                )
            m_sorted = m_mean[sort_idx, :].T  # (P, K)

            # 3. Plot with pcolormesh
            X, Y = np.meshgrid(T_sorted, np.arange(m_sorted.shape[0]))
            im = ax.pcolormesh(
                X,
                Y,
                m_sorted,
                shading="nearest",
                cmap="viridis",
                vmin=0,
                vmax=1,
            )

            ax.set_title(f"Realization {i} ({os.path.basename(rdir)})")
            ax.set_ylabel("Pattern Index")
            if i >= (n_rows - 1) * n_cols:
                ax.set_xlabel("Temperature ($T$)")

        except Exception as e:
            print(f"Error {rdir}: {e}")
            ax.axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.02, aspect=40)
        cbar.set_label(r"Overlap $|m|$", fontsize=12)

    plt.suptitle(
        "Corrected Phase Diagram (Non-linear T Axis)", fontsize=16, y=1.02
    )
    plt.show()



# %%
def plot_mmax_histograms(
    run_root,
    disorder_idx=0,
    beta_indices=None,
    bins=50,
    value_range=(0.0, 1.0),
    density=True,
):
    """
    For a given realization (disorder_idx), build histograms of m_max(T)
    for each temperature and overlay them in a single plot.

    m_max(T_k, replica, time) = max_mu |m_mu(replica, time, T_k)|

    The histogram at temperature index k is accumulated over all
    replicas and all time steps.

    Parameters
    ----------
    run_root : str
        Root directory containing r*/timeseries/*.m_sel.npy and sysconfig.npz
    disorder_idx : int
        Index of the realization/disorder (which r* dir to use, after sorting).
    beta_indices : None or int or list/array of int
        Temperature indices to show.
        - If None: show all K temperatures.
        - Example: np.arange(1, K) to skip the first one.
    bins : int
        Number of histogram bins.
    value_range : tuple
        (min, max) range for histogram (e.g. (0,1) for |m|).
    density : bool
        If True, normalize histograms to probability density.
    """

    # --- 1. Locate realization directory ---
    rdirs = sorted(glob.glob(os.path.join(run_root, "r*")))
    if not rdirs:
        print(f"No realizations found in {run_root}")
        return

    if disorder_idx < 0 or disorder_idx >= len(rdirs):
        raise IndexError(
            f"disorder_idx={disorder_idx} out of range, found {len(rdirs)} realizations."
        )

    rdir = rdirs[disorder_idx]

    # --- 2. Load betas and temps ---
    syscfg = np.load(os.path.join(rdir, "sysconfig.npz"))
    if "beta" in syscfg.files:
        betas = np.array(syscfg["beta"])
    elif "β" in syscfg.files:
        betas = np.array(syscfg["β"])
    else:
        raise KeyError("sysconfig.npz has no 'beta' or 'β'")
    K = betas.size
    T_vals = 1.0 / betas

    # --- 3. Which beta indices to show? ---
    if beta_indices is None:
        beta_indices = np.arange(K)
    elif isinstance(beta_indices, (int, np.integer)):
        beta_indices = np.array([beta_indices])
    else:
        beta_indices = np.array(list(beta_indices), dtype=int)

    # sanity-check indices
    for k_idx in beta_indices:
        if k_idx < 0 or k_idx >= K:
            raise IndexError(f"beta index {k_idx} out of range [0, {K-1}]")

    # sort them by temperature (optional but usually nicer)
    beta_indices = beta_indices[np.argsort(T_vals[beta_indices])]

    # --- 4. Prepare histogram accumulators ---
    bin_edges = np.linspace(value_range[0], value_range[1], bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # hist_counts[k, :] = counts for temperature index k
    hist_counts = np.zeros((K, bins), dtype=np.float64)

    # --- 5. Stream over m_sel and accumulate m_max histograms ---
    m_files = sorted(glob.glob(os.path.join(rdir, "timeseries", "*.m_sel.npy")))
    if not m_files:
        print(f"No m_sel timeseries found in {rdir}")
        return

    first = np.load(m_files[0], mmap_mode="r")
    if first.ndim != 4:
        raise ValueError(
            f"Expected m_sel with 4 dims (R,T,K,P), got shape {first.shape}"
        )
    R, _, K0, P = first.shape
    if K0 != K:
        raise ValueError(f"Inconsistent K in m_sel: {K0} vs {K}")
    del first

    for fpath in m_files:
        m_chunk = np.load(fpath)  # (R, T_chunk, K, P)
        if m_chunk.ndim != 4:
            raise ValueError(
                f"{fpath} has shape {m_chunk.shape}, expected (R,T,K,P)"
            )

        # m_max_chunk: (R, T_chunk, K) = max_mu |m|
        abs_chunk = np.abs(m_chunk)
        m_max_chunk = abs_chunk.max(axis=-1)

        # flatten (R, T_chunk) into one axis: (N_snap_chunk, K)
        m_max_flat = m_max_chunk.reshape(-1, K)

        # accumulate histogram for each temperature
        for k_idx in range(K):
            data_k = m_max_flat[:, k_idx]
            counts, _ = np.histogram(
                data_k,
                bins=bin_edges,
                range=value_range,
            )
            hist_counts[k_idx] += counts

        del m_chunk, abs_chunk, m_max_chunk, m_max_flat

    # --- 6. Plot all selected temperatures in one big plot ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # color map over the selected temperatures
    cmap = cm.get_cmap("magma")
    colors = cmap(np.linspace(0, 1, len(beta_indices)))

    for c, k_idx in zip(colors, beta_indices):
        counts = hist_counts[k_idx]
        total = counts.sum()
        if total == 0:
            # no data for this temperature; skip
            continue

        if density:
            y = counts / (total * bin_width)
        else:
            y = counts

        T_val = T_vals[k_idx]
        beta_val = betas[k_idx]

        y_step = np.r_[y, y[-1]]  # pad one extra value for the last edge

        ax.step(
            bin_edges,
            y_step,
            where="post",
            color=c,
            alpha=0.9,
            label=fr"$T={T_val:.3g}$",
        )

        ax.fill_between(
            bin_edges,
            y_step,
            step="post",
            color=c,
            alpha=0.3,
        )

    ax.set_xlabel(r"$m_{\max}$")
    ax.set_ylabel("density" if density else "count")
    ax.set_xlim(bin_edges[0], bin_edges[-1])

    ax.set_title(
        f"Histogram of $m_{{\\max}}$ over replicas & time\n"
        f"Realization {disorder_idx} ({os.path.basename(rdir)})"
    )
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()





def plot_centered_average_heatmap(run_root, winner_indices, target_center=None):
    """
    Plots a single averaged heatmap from multiple realizations by aligning their
    retrieved patterns to a common target index.

    Args:
        run_root (str): Path to the experiment folder (e.g., "runs/explore_fina_pool")
        winner_indices (list): A list of integers [r0, r1, ... r9] representing the
                               index of the pattern retrieved at Low T for each realization.
        target_center (int): The index to shift the winner to. 
                             If None, defaults to P // 2 (visual center).
                             User requested 10, so you can pass 10 here.
    """
    rdirs = sorted(glob.glob(os.path.join(run_root, "r*")))
    
    if not rdirs:
        print("No realizations found.")
        return
    if len(rdirs) != len(winner_indices):
        print(f"Error: Found {len(rdirs)} dirs but got {len(winner_indices)} indices.")
        return

    # 1. Load Physics (for T axis and P count)
    try:
        sys = np.load(os.path.join(rdirs[0], "sysconfig.npz"))
        betas = sys["beta"]
        P = int(sys["P"])
        T_axis = 1.0 / betas
        
        # Sort T for plotting
        sort_idx = np.argsort(T_axis)
        T_sorted = T_axis[sort_idx]
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Determine center
    if target_center is None:
        target_center = P // 2
    print(f"Aligning all retrieval peaks to Pattern Index: {target_center}")

    # Accumulator for the average
    m_accum = None
    count = 0

    # 2. Iterate and Align
    for i, rdir in enumerate(rdirs):
        m_files = sorted(glob.glob(os.path.join(rdir, "timeseries", "*.m_sel.npy")))
        if not m_files: continue

        try:
            # Load raw data: (2, Time, K, P)
            m_arr = np.concatenate([np.load(f) for f in m_files], axis=1)
            
            # Time & Replica Average -> (K, P)
            # We take the absolute value before averaging to avoid sign flipping issues
            m_mean = np.mean(np.abs(m_arr), axis=(0, 1))
            
            # Sort by Temperature -> (K_sorted, P)
            m_sorted_T = m_mean[sort_idx, :]
            
            # --- ALIGNMENT STEP ---
            # Get the winner for this realization
            winner_idx = winner_indices[i]
            
            # Calculate shift needed to move winner_idx to target_center
            # We use np.roll (cyclic shift)
            shift = target_center - winner_idx
            
            # Apply shift to the Pattern axis (axis 1)
            m_aligned = np.roll(m_sorted_T, shift, axis=1)
            
            # Accumulate
            if m_accum is None:
                m_accum = np.zeros_like(m_aligned)
            m_accum += m_aligned
            count += 1
            
        except Exception as e:
            print(f"Error processing {rdir}: {e}")

    if count == 0: return

    # 3. Compute Final Average
    m_final = m_accum / count
    
    # Transpose for plotting: X=Temperature (K), Y=Pattern (P)
    # Shape becomes (P, K)
    heatmap_data = m_final.T

    # 4. Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Meshgrid for pcolormesh
    X, Y = np.meshgrid(T_sorted, np.arange(P))
    
    # Use pcolormesh for correct non-linear T axis handling
    im = ax.pcolormesh(X, Y, heatmap_data, shading='nearest', cmap='inferno', vmin=0, vmax=1)
    
    # Visual guide for the centered pattern
    #ax.axhline(target_center, color='cyan', linestyle='--', alpha=0.5, label='Aligned Center')
    
    ax.set_title(f"Averaged Phase Diagram (Aligned to Pattern {target_center})\nAvg over {count} Realizations")
    ax.set_ylabel("Pattern Index (Shifted)")
    ax.set_xlabel("Temperature ($T$)")
    
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r'Average Overlap $\langle |m| \rangle$', fontsize=12)
    
    plt.tight_layout()
    plt.show()


def plot_disorder_magnetization(run_root, disorder_idx, pattern_indices):
    """
    Plots magnetization vs T for specific patterns in a specific disorder realization.
    
    Args:
        run_root (str): Path to the experiment root.
        disorder_idx (int): Integer 0-9 corresponding to folders r000-r009.
        pattern_indices (int or list): Index or list of pattern indices to plot.
    """
    # 1. format directory path (e.g., r005)
    r_dir = os.path.join(run_root, f"r{disorder_idx:03d}")
    
    if not os.path.exists(r_dir):
        print(f"Error: Directory {r_dir} does not exist.")
        return

    # 2. Normalize inputs
    if isinstance(pattern_indices, int):
        pattern_indices = [pattern_indices]

    # 3. Load Temperature (T) from sysconfig
    # We assume sysconfig is consistent across runs, so we load from the target dir
    try:
        sys = np.load(os.path.join(r_dir, "sysconfig.npz"))
        T = 1.0 / sys["beta"]
    except Exception as e:
        print(f"Error loading sysconfig in {r_dir}: {e}")
        return

    # 4. Load Time-series Data
    m_files = sorted(glob.glob(os.path.join(r_dir, "timeseries", "*.m_sel.npy")))
    if not m_files:
        print(f"No .m_sel.npy files found in {r_dir}/timeseries")
        return

    try:
        # Load and concatenate along Time axis (axis=1)
        # Raw Shape: (Replicas=2, Time_Chunks, Temp_Steps, Patterns)
        m_arr = np.concatenate([np.load(f) for f in m_files], axis=1)
        
        # 5. Compute Order Parameters
        # Take ABS first to handle symmetry breaking (+m vs -m in different replicas)
        # Then average over Replicas (axis 0) and Time (axis 1)
        m_means = np.mean(np.abs(m_arr), axis=(0, 1)) 
        # m_means Shape: (Temp_Steps, Patterns)

    except Exception as e:
        print(f"Error processing arrays: {e}")
        return

    # 6. Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for pid in pattern_indices:
        if pid < m_means.shape[1]:
            ax.plot(T, m_means[:, pid], marker='o', label=f'Pattern $\mu={pid}$')
        else:
            print(f"Warning: Pattern index {pid} exceeds available patterns.")

    ax.set_xlabel('Temperature ($T$)')
    ax.set_ylabel(r'Magnetization $\langle |m| \rangle_{t, repl}$')
    ax.set_title(f'Retrieval Stability: Disorder Run {disorder_idx} (r{disorder_idx:03d})')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


# %%
def plot_disorder_magnetization(run_root, disorder_idx, pattern_indices, cmap_name='plasma'):
    # 1. Format directory path (r000, r001...)
    r_dir = os.path.join(run_root, f"r{disorder_idx:03d}")
    
    if not os.path.exists(r_dir):
        print(f"Error: Directory {r_dir} does not exist.")
        return

    # 2. Normalize inputs to list
    if isinstance(pattern_indices, int):
        pattern_indices = [pattern_indices]

    # 3. Load Temperature
    try:
        sys = np.load(os.path.join(r_dir, "sysconfig.npz"))
        T = 1.0 / sys["beta"]
    except Exception as e:
        print(f"Error loading sysconfig: {e}")
        return

    # 4. Load Time-series
    m_files = sorted(glob.glob(os.path.join(r_dir, "timeseries", "*.m_sel.npy")))
    if not m_files:
        print(f"No .m_sel.npy files in {r_dir}")
        return

    try:
        m_arr = np.concatenate([np.load(f) for f in m_files], axis=1)
        # Average |m| over Replicas (0) and Time (1)
        m_means = np.mean(np.abs(m_arr), axis=(0, 1)) 
    except Exception as e:
        print(f"Error processing arrays: {e}")
        return

    # --- PLOTTING CHANGES START HERE ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate N colors from the requested colormap
    # We sample evenly from 0.0 to 0.9 (avoiding 1.0 often helps visibility on white backgrounds for plasma)
    n_patterns = len(pattern_indices)
    colors = plt.get_cmap(cmap_name)(np.linspace(0, 0.85, n_patterns))

    # Iterate with enumerate to access the corresponding color
    for i, pid in enumerate(pattern_indices):
        if pid < m_means.shape[1]:
            ax.plot(T, m_means[:, pid], 
                    marker='o', 
                    color=colors[i], 
                    label=f'Pattern $\mu={pid}$')
        else:
            print(f"Warning: Pattern index {pid} out of bounds.")

    ax.set_xlabel('Temperature ($T$)')
    ax.set_ylabel(r'Magnetization $\langle |m| \rangle$')
    ax.set_title(f'Retrieval (r{disorder_idx:03d}) - Colormap: {cmap_name}')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def analyze_phase_toolkit(run_root):
    rdirs = sorted(glob.glob(os.path.join(run_root, "r*")))
    
    # Store curves for averaging
    pr_curves = []      # Participation Ratio
    m_max_curves = []   # Max Magnetization
    
    for rdir in rdirs:
        m_files = sorted(glob.glob(os.path.join(rdir, "timeseries", "*.m_sel.npy")))
        if not m_files: continue
        
        try:
            # m_arr: (2, T, K, Patterns)
            m_arr = np.concatenate([np.load(f) for f in m_files], axis=1)
            # Average over Replicas (0) and Time (1) -> Shape (K, Patterns)
            m_means = np.mean(np.abs(m_arr), axis=(0, 1))
            
            # --- 1. Max Magnetization ---
            # "How strong is the best match?"
            m_max = np.max(m_means, axis=-1)
            m_max_curves.append(m_max)
            
            # --- 2. Participation Ratio ---
            # "How many patterns contribute?"
            # P = (Sum m^2)^2 / Sum m^4
            numerator = np.sum(m_means**2, axis=-1)**2
            denominator = np.sum(m_means**4, axis=-1)
            # Avoid divide by zero if m=0
            pr = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
            pr_curves.append(pr)
            
        except Exception as e:
            print(f"Skipping {rdir}: {e}")

    # --- Plotting ---
    if not pr_curves: return

    # Load T axis
    sys = np.load(os.path.join(rdirs[0], "sysconfig.npz"))
    T = 1.0 / sys["beta"]
    
    # Averages and Errors
    PR_avg = np.mean(pr_curves, axis=0)
    PR_err = np.std(pr_curves, axis=0) / (2*np.sqrt(len(pr_curves)))
    
    M_avg = np.mean(m_max_curves, axis=0)
    M_err = np.std(m_max_curves, axis=0) / (2*np.sqrt(len(m_max_curves)))
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Max Magnetization (Left Axis)
    color = 'tab:blue'
    ax1.set_xlabel('Temperature ($T$)')
    ax1.set_ylabel(r'Max Magnetization ($m_{max}$)', color=color)
    l1 = ax1.errorbar(T, M_avg, yerr=M_err, fmt='-o', color=color, label='$m_{max}$')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.1)
    ax1.grid(alpha=0.3)

    # Plot Participation Ratio (Right Axis)
    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('Participation Ratio ($P$)', color=color)
    # Reference line at P=1 (Pure State)
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Pure State (P=1)')
    l2 = ax2.errorbar(T, PR_avg, yerr=PR_err, fmt='-s', color=color, label='Participation Ratio')
    ax2.tick_params(axis='y', labelcolor=color)
    # P usually goes from 1 (Pure) to ~1.5/3 (Mixture) to >>1 (Glass)
    # Increased limit to 15.0 to see the paramagnetic noise floor (approx P/3 ~ 13)
    ax2.set_ylim(0.5, 15.0) 

    # Combine legends from both axes
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.title(f"Identifying the Retrieval Phase (N={sys['N']}, P={sys['P']})")
    fig.tight_layout()
    plt.show()

# analyze_phase_toolkit("runs/PT_OFF_3_pool")


# %%
def analyze_phase_toolkit(run_root):
    rdirs = sorted(glob.glob(os.path.join(run_root, "r*")))
    if not rdirs:
        print(f"No realizations found in {run_root}")
        return

    pr_curves = []     # per-realization PR(T) curve
    mmax_curves = []   # per-realization <m_max>(T) curve

    # Load temperature axis from first realization
    sys0 = np.load(os.path.join(rdirs[0], "sysconfig.npz"))
    beta_key = "beta" if "beta" in sys0.files else ("β" if "β" in sys0.files else None)
    if beta_key is None:
        raise KeyError("sysconfig.npz has no 'beta' or 'β'")
    betas = np.array(sys0[beta_key], dtype=np.float64)
    T_axis = 1.0 / betas
    K = betas.size

    for rdir in rdirs:
        m_files = sorted(glob.glob(os.path.join(rdir, "timeseries", "*.m_sel.npy")))
        if not m_files:
            continue

        # Accumulators over (replicas,time) for each temperature index k
        sum_mmax = np.zeros(K, dtype=np.float64)
        sum_pr   = np.zeros(K, dtype=np.float64)
        n_samples = 0  # total number of (replica,time) samples contributed per temperature

        ok = True
        for f in m_files:
            try:
                m_chunk = np.load(f)  # expected shape (R, Tchunk, K, P)
            except Exception as e:
                print(f"Skipping file {f}: {e}")
                ok = False
                break

            if m_chunk.ndim != 4:
                print(f"Skipping {rdir}: {os.path.basename(f)} has shape {m_chunk.shape}, expected (R,T,K,P)")
                ok = False
                break

            R, Tchunk, K0, P = m_chunk.shape
            if K0 != K:
                print(f"Skipping {rdir}: K mismatch in {os.path.basename(f)} ({K0} vs sysconfig K={K})")
                ok = False
                break

            # Work in-place to reduce temporaries (safe since m_chunk is a fresh array)
            np.abs(m_chunk, out=m_chunk)                  # m_chunk := |m|
            # m_max per sample: shape (R, Tchunk, K)
            mmax_inst = m_chunk.max(axis=-1)
            sum_mmax += mmax_inst.sum(axis=(0, 1))

            # PR per sample:
            # s2 = sum_mu |m|^2, s4 = sum_mu |m|^4, PR = (s2^2)/s4
            m2 = m_chunk * m_chunk                        # |m|^2
            s2 = m2.sum(axis=-1)                          # (R, Tchunk, K)
            s4 = (m2 * m2).sum(axis=-1)                   # (R, Tchunk, K)

            pr_inst = np.zeros_like(s2, dtype=np.float64)
            np.divide(s2 * s2, s4, out=pr_inst, where=(s4 != 0.0))
            sum_pr += pr_inst.sum(axis=(0, 1))

            n_samples += R * Tchunk

        if not ok or n_samples == 0:
            continue

        # Per-realization mean curves vs T
        mmax_curve = sum_mmax / n_samples
        pr_curve   = sum_pr   / n_samples

        mmax_curves.append(mmax_curve)
        pr_curves.append(pr_curve)

    if not pr_curves:
        print("No usable realizations found.")
        return

    mmax_curves = np.array(mmax_curves, dtype=np.float64)  # (nR, K)
    pr_curves   = np.array(pr_curves,   dtype=np.float64)  # (nR, K)
    nR = mmax_curves.shape[0]

    # Sort by temperature for prettier plots
    order = np.argsort(T_axis)
    T = T_axis[order]
    mmax_curves = mmax_curves[:, order]
    pr_curves   = pr_curves[:, order]

    # Averages
    M_avg  = mmax_curves.mean(axis=0)
    PR_avg = pr_curves.mean(axis=0)

    # Standard error of the mean across disorder realizations
    if nR > 1:
        M_err  = mmax_curves.std(axis=0, ddof=1) / np.sqrt(nR)
        PR_err = pr_curves.std(axis=0, ddof=1) / np.sqrt(nR)
    else:
        M_err  = np.zeros_like(M_avg)
        PR_err = np.zeros_like(PR_avg)

    # ---- Plotting ----
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel(r"Temperature $T$")
    ax1.set_ylabel(r"$\langle m_{\max}\rangle$ (over replicas & time)")
    ax1.errorbar(T, M_avg, yerr=M_err, fmt='-o', label=r"$\langle m_{\max}\rangle$")

    ax1.set_ylim(0.0, 1.05)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Participation ratio $\langle P\rangle$")
    ax2.axhline(1.0, linestyle='--', alpha=0.5, label=r"Pure state ($P=1$)")
    ax2.errorbar(T, PR_avg, yerr=PR_err, fmt='-s', label=r"$\langle P\rangle$")

    # Keep PR axis reasonable but not clipped
    pr_hi = np.nanmax(PR_avg + PR_err)
    ax2.set_ylim(0.5, max(2.0, min(15.0, 1.1 * pr_hi if np.isfinite(pr_hi) else 15.0)))

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")

    # Optional: show N,P if present
    title_bits = [f"Identifying phases (n_realizations={nR})"]
    if "N" in sys0.files and "P" in sys0.files:
        title_bits.append(f"N={int(sys0['N'])}, P={int(sys0['P'])}")
    plt.title(" — ".join(title_bits))

    fig.tight_layout()
    plt.show()



# %%
def analyze_phase_toolkit(run_root):
    rdirs = sorted(glob.glob(os.path.join(run_root, "r*")))
    if not rdirs:
        print(f"No realizations found in {run_root}")
        return

    # Load temperature axis from first realization
    sys0 = np.load(os.path.join(rdirs[0], "sysconfig.npz"))
    beta_key = "beta" if "beta" in sys0.files else ("β" if "β" in sys0.files else None)
    if beta_key is None:
        raise KeyError("sysconfig.npz has no 'beta' or 'β'")
    betas = np.array(sys0[beta_key], dtype=np.float64)
    T_axis = 1.0 / betas
    K = betas.size

    mmax_curves = []   # per-realization <m_max>(T)
    ipr_curves  = []   # per-realization <IPR>(T)

    for rdir in rdirs:
        m_files = sorted(glob.glob(os.path.join(rdir, "timeseries", "*.m_sel.npy")))
        if not m_files:
            continue

        sum_mmax = np.zeros(K, dtype=np.float64)
        sum_ipr  = np.zeros(K, dtype=np.float64)
        n_samples = 0

        ok = True
        for f in m_files:
            try:
                m = np.load(f)  # (R, Tchunk, K, P)
            except Exception as e:
                print(f"Skipping file {f}: {e}")
                ok = False
                break

            if m.ndim != 4:
                print(f"Skipping {rdir}: {os.path.basename(f)} has shape {m.shape}, expected (R,T,K,P)")
                ok = False
                break

            R, Tchunk, K0, P = m.shape
            if K0 != K:
                print(f"Skipping {rdir}: K mismatch in {os.path.basename(f)} ({K0} vs sysconfig K={K})")
                ok = False
                break

            # |m|
            np.abs(m, out=m)

            # instantaneous m_max per sample: (R, Tchunk, K)
            mmax_inst = m.max(axis=-1)
            sum_mmax += mmax_inst.sum(axis=(0, 1))

            # IPR per sample:
            # s2 = sum |m|^2, s4 = sum |m|^4, IPR = s4 / (s2^2)
            m2 = m * m
            s2 = m2.sum(axis=-1)            # (R, Tchunk, K)
            s4 = (m2 * m2).sum(axis=-1)     # (R, Tchunk, K)

            ipr_inst = np.zeros_like(s2, dtype=np.float64)
            np.divide(s4, s2 * s2, out=ipr_inst, where=(s2 != 0.0))
            sum_ipr += ipr_inst.sum(axis=(0, 1))

            n_samples += R * Tchunk

        if (not ok) or n_samples == 0:
            continue

        mmax_curves.append(sum_mmax / n_samples)
        ipr_curves.append(sum_ipr / n_samples)

    if not mmax_curves:
        print("No usable realizations found.")
        return

    mmax_curves = np.array(mmax_curves, dtype=np.float64)  # (nR, K)
    ipr_curves  = np.array(ipr_curves,  dtype=np.float64)  # (nR, K)
    nR = mmax_curves.shape[0]

    # sort by temperature
    order = np.argsort(T_axis)
    T = T_axis[order]
    mmax_curves = mmax_curves[:, order]
    ipr_curves  = ipr_curves[:, order]

    # means + SEM across realizations
    M_avg  = mmax_curves.mean(axis=0)
    I_avg  = ipr_curves.mean(axis=0)

    if nR > 1:
        M_err = mmax_curves.std(axis=0, ddof=1) / np.sqrt(nR)
        I_err = ipr_curves.std(axis=0, ddof=1) / np.sqrt(nR)
    else:
        M_err = np.zeros_like(M_avg)
        I_err = np.zeros_like(I_avg)

    # ---- plotting ----
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel(r"Temperature $T$")
    ax1.set_ylabel(r"$\langle m_{\max}\rangle$")
    ax1.errorbar(T, M_avg, yerr=M_err, fmt='-o', label=r"$\langle m_{\max}\rangle$")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$\langle \mathrm{IPR}\rangle = \left\langle \frac{\sum_\mu m_\mu^4}{(\sum_\mu m_\mu^2)^2}\right\rangle$")
    ax2.axhline(1.0, linestyle='--', alpha=0.4, label="pure (IPR=1)")
    ax2.errorbar(T, I_avg, yerr=I_err, fmt='-s', label=r"$\langle \mathrm{IPR}\rangle$",color='red')
    ax2.set_ylim(0.0, 1.05)

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")

    title = f"Phase diagnostics (n_realizations={nR})"
    if "N" in sys0.files and "P" in sys0.files:
        title += f" — N={int(sys0['N'])}, P={int(sys0['P'])}"
    plt.title(title)

    fig.tight_layout()
    plt.show()



# %%
analyze_phase_toolkit("runs/hope_pool")


# %% [markdown]
# # Other

# %% [markdown]
# #### Discrete rewiring

# %%
@njit(float64[:, ::1](int64, int8[:, ::1]), nogil=True)
def build_C(N, ξ):
    # cast once to float for BLAS-friendly math
    Xi = ξ.astype(np.float64)
    C = np.ascontiguousarray((Xi.T @ Xi) * (1.0 / N))
    return C

@njit(void(float64[:, ::1], float64[:, ::1], float64, int64), nogil=True)
def update_G_with_C(G, C, ε, k):
    γ = ε / (1.0 + ε * k)
    # two matmuls
    T = G @ C           # P×P
    G[:, :] = (1.0 + γ) * G - γ * (T @ G)
    # (optional) enforce symmetry to kill fp drift
    for μ in range(G.shape[0]):
        for ν in range(μ + 1, G.shape[1]):
            s = 0.5 * (G[μ, ν] + G[ν, μ])
            G[μ, ν] = s
            G[ν, μ] = s


# %%
import numpy as np

def T_erg(alpha, t):
    alpha = np.asarray(alpha)
    sa = np.sqrt(alpha)
    Delta = 1.0 + sa*(1.0 + sa)*t
    beta_c = (1.0/(1.0 + t))*(Delta**2/(1.0 + sa) + t*Delta)
    return 1.0 / beta_c

# Example: plot for a few t's
import matplotlib.pyplot as plt

alphas = np.linspace(0, 1.0, 200)

for t, lbl in [(0.0,"t=0"),(0.1,"t=0.1"),(0.2,"t=0.2"),(0.5,"t=0.5")]:
    plt.plot(alphas, T_erg(alphas,t), label=lbl)

plt.xlabel(r'$\alpha$')
plt.ylabel(r'$T_{\rm erg}(\alpha,t)$')
plt.ylim(0,2)
plt.legend()
plt.show()


# %%
