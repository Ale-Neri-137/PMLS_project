"""
β-ladder tuning utilities (acceptance-driven).

This module is intended to be import-safe (no top-level execution) and light on dependencies.
It assumes you have:
- jitted kernels: do_one_MMC_step, compute_cholG_from_xi_A
- model/init helpers: SysConfig, sample_ξ, build_G_t, build_A_and_Jdiag, init_spins, compute_M_from_σ_ξ,
  compute_E_from_M, _make_seed_matrix
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np
from numba import njit, int8, float64, int64, uint64, void
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context, get_all_start_methods
from scipy.special import erfcinv

# ---- Optional imports (support both "flat files" and a package layout) ----
try:
    # If you later turn this into a package, prefer these imports.
    from .jitted_kernel import do_one_MMC_step, compute_cholG_from_xi_A
    from init_and_checkpoints import (
        SysConfig,
        sample_ξ,
        build_G_t,
        build_A_and_Jdiag,
        init_spins,
        compute_M_from_σ_ξ,
        compute_E_from_M,
        _make_seed_matrix,
    )
except Exception:  # pragma: no cover
    # Fall back to the cleaned single-file modules we created earlier.
    from .jitted_kernel import do_one_MMC_step, compute_cholG_from_xi_A
    from .init_and_checkpoints import (  # type: ignore
        SysConfig,
        sample_ξ,
        build_G_t,
        build_A_and_Jdiag,
        init_spins,
        compute_M_from_σ_ξ,
        compute_E_from_M,
        _make_seed_matrix,
    )

# %% [markdown]
# ## Lighter orchestration

# %% [markdown]
# ###  Configs

# %%
@dataclass(frozen=True)
class TrialConfig:
    equilibration_time: int
    sweeps_per_sample: int
    n_samples: int

@dataclass
class TrialResult:
    rid: int
    beta: np.ndarray        # (K,)
    acc_edge: np.ndarray    # (K-1,) acceptance per edge (both replicas avg)
    I_ts: np.ndarray        #(K,) timeseries of swaps

    """
    mean_E: np.ndarray      # (K,)   avg over replicas
    var_E: np.ndarray       # (K,)   avg over replicas
    """


# %% [markdown]
# ### Runner, worker and executor

# %%
def run_trial_stats(sys: SysConfig, trial: TrialConfig, rid: int) -> TrialResult:
    # disorder/couplings (deterministic per rid)
    ξ  = sample_ξ(sys.N, sys.P, sys.master_seed, rid, sys.c)
    G  = build_G_t(ξ, sys.t)
    A, d = build_A_and_Jdiag(G, ξ)

    # initial microstate (deterministic per rid)
    Σ = init_spins(sys, rid, ξ)                  # your helper from earlier (2, K, N)
    M0 = compute_M_from_σ_ξ(Σ[0], ξ)
    M1 = compute_M_from_σ_ξ(Σ[1], ξ)
    M  = np.stack([M0, M1], axis=0)
    Ξ = np.empty((2, sys.K), np.float64)      # energies per replica, filled by your sweeps
    for b in (0,1):
        Ξ[b] = compute_E_from_M(M[b], G, sys.N)
    Ψ = np.tile(np.arange(sys.K, dtype=np.int64), (2,1))
    seeds = _make_seed_matrix(sys, rid)
    swap_count = np.zeros((2, sys.K-1), dtype=np.int64)

    I_ts = np.zeros((2,trial.n_samples*trial.sweeps_per_sample,sys.K),dtype=np.int8)
    """
    # Welford accumulators
    W_n    = np.zeros((2, sys.K), dtype=np.int64)
    W_mean = np.zeros((2, sys.K), dtype=np.float64)
    W_M2   = np.zeros((2, sys.K), dtype=np.float64)
    """
    # one in-memory stats run
    Simulate_two_replicas_stats(
        sys.N, sys.P, sys.K, 1.0/sys.N,
        Σ, M, Ξ, Ψ,
        A, ξ, d,
        np.ascontiguousarray(sys.β, np.float64),
        seeds,
        trial.equilibration_time, trial.sweeps_per_sample, trial.n_samples,
        swap_count,
        I_ts
    )
    """
    # finalize mean/var per replica → average replicas
    mask = (W_n > 1)
    var  = np.zeros_like(W_mean); var[mask] = W_M2[mask] / (W_n[mask] - 1)
    mean_E = W_mean.mean(axis=0)                 # (K,)
    var_E  = var.mean(axis=0)                    # (K,)
    """
    steps  = trial.n_samples * trial.sweeps_per_sample
    acc_edge = (swap_count.mean(axis=0) / max(1, steps))  # (K-1,)


    

    return TrialResult(rid=rid, beta=sys.β.copy(), acc_edge=acc_edge, I_ts=I_ts)


# %%
def worker_run_stats(rid: int, sys: SysConfig, trial: TrialConfig) -> TrialResult:
    return run_trial_stats(sys, trial, rid)

def pool_orchestrator_stats(sys: SysConfig, trial: TrialConfig,
                            R_workers: int, R_total: int, start_method="fork"):
    mpctx = get_context(start_method)
    out = []
    with ProcessPoolExecutor(max_workers=R_workers, mp_context=mpctx) as ex:
        futs = [ex.submit(worker_run_stats, rid, sys, trial) for rid in range(R_total)]
        for f in as_completed(futs): out.append(f.result())
    # sort by rid
    out.sort(key=lambda r: r.rid)
    return out


# %% [markdown]
# ## Ladder search

# %% [markdown]
# ### save/load and a helper

# %%
def save_optimal_ladder(path: str, sys: SysConfig, acc_edge_mean: Optional[np.ndarray] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    arrays = {
        "N": np.array(sys.N, dtype=np.int64),
        "P": np.array(sys.P, dtype=np.int64),
        "K": np.array(sys.K, dtype=np.int64),
        "t": np.array(sys.t, dtype=np.float64),
        "c": np.array(sys.c, dtype=np.float64),
        "master_seed": np.array(sys.master_seed, dtype=np.int64),
        "beta": np.asarray(sys.β, dtype=np.float64),
        "mu": np.asarray(sys.mu_to_store, dtype=np.int64),
        "spin_init_mode": np.array([sys.spin_init_mode]),
    }
    if acc_edge_mean is not None:
        arrays["acc_edge_mean"] = np.asarray(acc_edge_mean, dtype=np.float64)

    np.savez_compressed(path, **arrays)


def load_optimal_ladder(path: str):
    z = np.load(path, allow_pickle=False)
    acc = z["acc_edge_mean"] if "acc_edge_mean" in z.files else None
    return z["beta"], acc  


# %%
"""
def aggregate_results(results):
    #results: list[TrialResult] with .acc_edge (K-1,), .var_E (K,)
    acc  = np.stack([r.acc_edge for r in results], axis=0).mean(axis=0)   # (K-1,)
    varE = np.stack([r.var_E   for r in results], axis=0).mean(axis=0)    # (K,)
    sigma_mid = np.sqrt(0.5 * (varE[:-1] + varE[1:]))                     # (K-1,)
    return acc, varE, sigma_mid
"""


# %%
def aggregate_results(results, q_lo=0.20):
    """
    Robust aggregation across disorder.
    Returns:
      acc_lo  : low-tail quantile per interface (K-1,)
      acc_med : median per interface (K-1,) (useful for printing/return)
      sigma_mid: sqrt(0.5*(varE_k + varE_{k+1})) (K-1,)
    """
    acc_stack  = np.stack([r.acc_edge for r in results], axis=0)  # (R, K-1)
    varE_stack = np.stack([r.var_E    for r in results], axis=0)  # (R, K)

    acc_lo  = np.quantile(acc_stack, q_lo, axis=0)
    acc_med = np.quantile(acc_stack, 0.50, axis=0)
    """
    # compute sigma_mid per disorder first, then aggregate (median/quantile)
    sigma_mid_stack = np.sqrt(0.5 * (varE_stack[:, :-1] + varE_stack[:, 1:]))  # (R, K-1)
    sigma_mid = np.quantile(sigma_mid_stack, 0.5, axis=0)  # median across disorders
    sigma_mid = np.maximum(sigma_mid, 1e-12)
    """
    #this empirically works best
    varE = np.quantile(varE_stack, 0.50, axis=0)     # (K,) median across disorders
    varE = np.maximum(varE, 0.0)                     # safety
    sigma_mid = np.sqrt(0.5 * (varE[:-1] + varE[1:]))  # (K-1,)
    sigma_mid = np.maximum(sigma_mid, 1e-12)

    return acc_lo, acc_med, sigma_mid


# %% [markdown]
# ### The algorithm

# %%
def _roundtrip_count_from_I(I_bt: np.ndarray) -> int:
    """
    Hot->cold->hot round trips counted from slot->walker series I_bt with shape (T,K).
    Ends are k=0 (hot) and k=K-1 (cold).
    """
    T, K = I_bt.shape
    hot_k, cold_k = 0, K - 1

    w_hot  = I_bt[:, hot_k].astype(np.int64)
    w_cold = I_bt[:, cold_k].astype(np.int64)

    t_hot_start = np.full(K, -1, dtype=np.int64)
    seen_cold   = np.zeros(K, dtype=np.bool_)
    n_rt = 0

    for t in range(T):
        wh = int(w_hot[t])
        wc = int(w_cold[t])

        # cold visit marks "success in between" if a hot-start exists
        if t_hot_start[wc] != -1:
            seen_cold[wc] = True

        # hot visit closes a RT if cold was seen since last hot-start
        if t_hot_start[wh] != -1 and seen_cold[wh]:
            n_rt += 1

        # (re)start at hot
        t_hot_start[wh] = t
        seen_cold[wh] = False

    return n_rt


def _flow_profile_f(I_bt: np.ndarray) -> np.ndarray:
    """
    f(k) = P(last end visited was hot | currently at slot k), estimated from I_bt (T,K).
    Returns f of shape (K,). Uses only labeled walkers (after they've hit an end at least once).
    """
    T, K = I_bt.shape
    hot_k, cold_k = 0, K - 1

    labels = np.zeros(K, dtype=np.int8)  # per walker: 0 unknown, +1 hot, -1 cold
    num = np.zeros(K, dtype=np.float64)
    den = np.zeros(K, dtype=np.float64)

    for t in range(T):
        wh = int(I_bt[t, hot_k])
        wc = int(I_bt[t, cold_k])
        labels[wh] = +1
        labels[wc] = -1

        lab_slots = labels[I_bt[t].astype(np.int64)]  # (K,)
        known = (lab_slots != 0)
        den[known] += 1.0
        num[known] += (lab_slots[known] == +1)

    f = np.full(K, np.nan, dtype=np.float64)
    m = den > 0
    f[m] = num[m] / den[m]
    return f


def _pick_cliff_interface_from_f(f: np.ndarray) -> tuple[int, float]:
    """
    Returns (k_star, slope_max) where k_star maximizes |f[k+1]-f[k]|.
    k_star is an interface index (between slots k_star and k_star+1).
    """
    df = np.diff(f)
    # ignore NaNs by treating them as -inf slope
    a = np.abs(df)
    a = np.where(np.isnan(a), -np.inf, a)
    k_star = int(np.argmax(a))
    slope_max = float(a[k_star])
    return k_star, slope_max


# %%
import numpy as np
from scipy.special import erfcinv

def reshape_betas_from_acceptance(
    betas: np.ndarray,
    acc,                    # (K-1,) or (R,K-1)
    A_star: float = 0.30,
    q_lo: float = 0.20,     # guard level across disorders if acc is stacked
    gamma: float = 0.5,     # damping: 0<gamma<=1 (smaller = more stable)
    clip=(0.75, 1.35),      # clamp multiplicative change per reshape
    eps: float = 1e-6
) -> np.ndarray:
    """
    Fixed-K reshape using acceptance feedback (no energy variance).
    Keeps endpoints fixed and redistributes Δβ to push acceptances toward A_star.
    """
    betas = np.asarray(betas, float)
    K = betas.size
    dB = np.diff(betas)

    A = np.asarray(acc, float)
    if A.ndim == 2:
        # Robustify across disorders in 'difficulty' space:
        # c = 2*erfcinv(A), larger c = harder interface.
        A_clipped = np.clip(A, eps, 1.0 - eps)
        c_stack = 2.0 * erfcinv(A_clipped)              # (R,K-1)
        c_eff = np.quantile(c_stack, 1.0 - q_lo, axis=0)  # guard hard tail
    elif A.ndim == 1:
        c_eff = 2.0 * erfcinv(np.clip(A, eps, 1.0 - eps))
    else:
        raise ValueError("acc must be shape (K-1,) or (R,K-1)")

    c_star = 2.0 * erfcinv(np.clip(A_star, eps, 1.0 - eps))

    # Multiplicative gap update: widen if too easy (c_eff < c_star), tighten if too hard
    fac = (c_star / np.maximum(c_eff, eps)) ** gamma

    fac = np.clip(fac, clip[0], clip[1])

    dB_new = dB * fac

    # Renormalize to keep endpoints fixed
    span = betas[-1] - betas[0]
    dB_new *= span / np.sum(dB_new)

    return betas[0] + np.concatenate(([0.0], np.cumsum(dB_new)))


# %%
def redistribute_betas_fixed_K_mid(beta_min, beta_max, sigma_mid, A_star=0.30):
    """
    Move interior β’s so that Δβ_k * σ_mid,k ≈ const (⟨A⟩ ≈ A_star).
    """
    c = 2.0 * erfcinv(A_star)                     # ~1.813 for 0.30
    w = c / sigma_mid                             # desired (unscaled) spacings per interface
    scale = (beta_max - beta_min) / w.sum()
    dB = scale * w
    return np.concatenate(([beta_min], beta_min + np.cumsum(dB)))

def redistribute_biased_mid(beta_min, beta_max, sigma_mid, low_ifc, A_star=0.30, shrink=0.95):
    """
    Same as above, but shrink Δβ at a specific low-A interface a bit more
    (shrink < 1.0) to give it extra overlap.
    """
    c = 2.0 * erfcinv(A_star)
    w = c / sigma_mid
    w[low_ifc] *= shrink                          # push more overlap to the straggler
    scale = (beta_max - beta_min) / w.sum()
    dB = scale * w
    return np.concatenate(([beta_min], beta_min + np.cumsum(dB)))


# %%
def ladder_search_parallel(
    sys_template,                  # SysConfig without β fixed
    beta_init: np.ndarray,         # 1D array (K,)
    trial,                         # TrialConfig
    R_workers: int, R_total: int,  # pool knobs (disorders)
    A_low=0.20, A_high=0.40, A_star=0.30,
    eps=0.01,                      # hysteresis margin
    q_lo=0.10,
    redistribute_every=2, low_max_for_reshape = 3, n_hot = 2, max_insert = 4,
    gamma_reshape = 0.5, clip_reshape=(0.75, 1.35),
    K_max=64, max_passes=27, verbose=True, start_method = "fork"
):
    betas = np.asarray(beta_init, float)
    passes, inserted_since_reshape, consecutive_reshapes = 0, 0, 0

    while passes < max_passes:
        passes += 1
        betas_prev = betas.copy()
        did_modify = False

        # ---- parallel pilot run on current ladder ----
        sys_now = replace(sys_template, K=betas.size, β=betas)
        results = pool_orchestrator_stats(sys_now, trial, R_workers, R_total, start_method=start_method)

        acc_stack   = np.stack([r.acc_edge for r in results], axis=0)  # (R, K-1)
        I_ts_stack  = np.stack([r.I_ts for r in results], axis=0)      # (R, B, T, K)

        acc_lo      = np.quantile(acc_stack, q_lo, axis=0)              # (K-1)
        acc_trigger = np.quantile(acc_stack, 3*q_lo, axis=0)              # (K-1)

        if verbose:
            print(" ")
            print(" ")
            print(f"pass {passes:2d} | K={betas.size:2d} | ")
            print(" ")
            print("acc_trig=", np.rint(100*acc_trigger).astype(int), sep="")
            print(" ")
            print("acc_low= ", np.rint(100*acc_lo).astype(int), sep="")
            print(" ")

        low  = acc_lo < (A_low  - eps)          # protect against hard disorders
        high = acc_trigger > (A_high + eps)          # cap the easy tail (efficiency)

        n_low = np.count_nonzero(low)
        if betas.size - 1 > n_hot:   high[-n_hot:] = False

        # stop if all within window
        if (not low.any()) and (not high.any()):
            if verbose: print("→ ladder done")
            return betas, acc_stack, I_ts_stack

        if betas.size >= K_max:
            if verbose: print("→ reached K_max")
            return betas, acc_stack, I_ts_stack

        if consecutive_reshapes >2 and betas.size < K_max:

            acc_min_per_ifc = acc_stack.min(axis=0)          # (K-1,)  A_min(k) = min_r A[r,k]
            k_worst = int(np.argmin(acc_min_per_ifc))        # interface index with worst-case acceptance
            if acc_min_per_ifc[k_worst]<A_low - 2* eps:

                betas = np.insert(betas, k_worst + 1, 0.5*(betas[k_worst] + betas[k_worst+1]))

                inserted_since_reshape += 1
                consecutive_reshapes = 0
                did_modify = True
                if verbose: print(f"＋ insert after worst iface: {k_worst}")
                continue

        # ---- reshape if any highs and few lows OR after several inserts ----
        if betas.size >= 3 and (inserted_since_reshape >= redistribute_every or (high.any() and n_low <= low_max_for_reshape)):

            betas = reshape_betas_from_acceptance(betas, acc_stack, A_star,
                    q_lo  = q_lo,              # guard level across disorders if acc is stacked
                    gamma = gamma_reshape,     # damping: 0<gamma<=1 (smaller = more stable)
                    clip  = clip_reshape)      # clamp multiplicative change per reshape

            inserted_since_reshape = 0
            consecutive_reshapes +=1
            did_modify = True
            if verbose: print("↺ reshape (fixed K)")
            continue


        # ---- grow: insert midpoint where definitely too low ----
        new_b = [betas[0]]
        for k, Ak in enumerate(acc_lo):
            # “definitely low” and not too many inserts after a reshape exept for first pass
            if Ak < (A_low - 2*eps) and len(new_b) < K_max and (inserted_since_reshape < max_insert): 
                new_b.append(0.5 * (betas[k] + betas[k+1]))

                inserted_since_reshape += 1
                consecutive_reshapes = 0
                did_modify = True
                if verbose: print(f"＋ insert after iface {k}")
            new_b.append(betas[k+1])

        betas = np.array(new_b, float)

        #STAGNATION GUARD
        if (not did_modify) or (betas.shape == betas_prev.shape and np.allclose(betas, betas_prev)):
            # force one change: insert at worst interface

            acc_min_per_ifc = acc_stack.min(axis=0)          # (K-1,)  A_min(k) = min_r A[r,k]
            k_worst = int(np.argmin(acc_min_per_ifc))

            betas = np.insert(betas, k_worst + 1, 0.5 * (betas[k_worst] + betas[k_worst + 1]))

            inserted_since_reshape += 1
            consecutive_reshapes = 0
            did_modify = True

            if verbose: print(f"⚠ stagnation-guard: insert after iface {k_worst}")
            continue

    # max_passes fallback
    if verbose: print("→ reached max_passes")
    return betas, acc_stack, I_ts_stack

