
import numpy as np
from numba import njit
from numba import int8, int64, float64, uint64, void, boolean
from dataclasses import dataclass

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from .RNG_helpers import xorshift128p_next_float, derive_xorshift_state
from .jitted_kernel import do_one_MMC_step
from .init_and_checkpoints import sample_ξ, build_couplings_for_t_grid, init_spins,_make_seed_matrix
from .init_and_checkpoints import SysConfig
from .init_and_checkpoints import compute_M_from_σ_ξ, compute_E_from_M






@njit(nogil=False, no_cpython_wrapper=False, fastmath=True)
def Simulate_two_replicas_stats_2d(
    N, P, R, invN, k_start,
    Σ, M, Ξ, Ψ,
    A, ξ, d,
    beta_flat,
    seed_matrix,
    equilibration_time, sweeps_per_sample, n_samples,
    swap_count,                    # (2, R-1)
    vertical_swap_count,           # (2, E)
    vertical_swap_attempt,         # (2, E)
    edge_list,                     # (E,2)
    order, mark, token_arr, out_e, # (2,*) work arrays (see runner)
    b_of_r,
    Ψ_ts                           # (2, T, R) int64
):
    # --- equilibrate ---
    for _ in range(equilibration_time):
        do_one_MMC_step(
            N, P, R, invN, k_start,
            Σ[0], M[0], Ξ[0], Ψ[0],
            A, ξ, d,
            beta_flat, seed_matrix[0],
            swap_count[0],
            vertical_swap_count[0], vertical_swap_attempt[0],
            edge_list,
            order[0], mark[0], token_arr[0], out_e[0],
            b_of_r
        )
    for _ in range(equilibration_time):
        do_one_MMC_step(
            N, P, R, invN, k_start,
            Σ[1], M[1], Ξ[1], Ψ[1],
            A, ξ, d,
            beta_flat, seed_matrix[1],
            swap_count[1],
            vertical_swap_count[1], vertical_swap_attempt[1],
            edge_list,
            order[1], mark[1], token_arr[1], out_e[1],
            b_of_r
        )

    # reset stats after equilibration
    for a in range(2):
        for i in range(R - 1):
            swap_count[a, i] = 0
        for e in range(edge_list.shape[0]):
            vertical_swap_count[a, e] = 0
            vertical_swap_attempt[a, e] = 0

    # --- production samples ---
    T = n_samples * sweeps_per_sample
    for n in range(n_samples):
        for t in range(sweeps_per_sample):
            step = n * sweeps_per_sample + t

            do_one_MMC_step(
                N, P, R, invN, k_start,
                Σ[0], M[0], Ξ[0], Ψ[0],
                A, ξ, d,
                beta_flat, seed_matrix[0],
                swap_count[0],
                vertical_swap_count[0], vertical_swap_attempt[0],
                edge_list,
                order[0], mark[0], token_arr[0], out_e[0],
                b_of_r
            )
            Ψ_ts[0, step] = Ψ[0]

            do_one_MMC_step(
                N, P, R, invN, k_start,
                Σ[1], M[1], Ξ[1], Ψ[1],
                A, ξ, d,
                beta_flat, seed_matrix[1],
                swap_count[1],
                vertical_swap_count[1], vertical_swap_attempt[1],
                edge_list,
                order[1], mark[1], token_arr[1], out_e[1],
                b_of_r
            )
            Ψ_ts[1, step] = Ψ[1]


@dataclass(frozen=True)
class TrialConfig:
    equilibration_time: int
    sweeps_per_sample: int
    n_samples: int

@dataclass
class TrialResult:
    rid: int
    beta_by_t: tuple[np.ndarray, ...]   # ragged (B,) of (K_b,)
    acc_h_by_t: tuple[np.ndarray, ...]  # ragged (B,) of (K_b-1,)
    acc_v: np.ndarray                   # (E,) accepted/attempted (avg over chains)
    Ψ_ts: np.ndarray                    # (2, T, R) int64


    """
    mean_E: np.ndarray      # (K,)   avg over replicas
    var_E: np.ndarray       # (K,)   avg over replicas
    """



### Runner, worker and executor


    """
    # Welford accumulators
    W_n    = np.zeros((2, sys.K), dtype=np.int64)
    W_mean = np.zeros((2, sys.K), dtype=np.float64)
    W_M2   = np.zeros((2, sys.K), dtype=np.float64)
    """

    """
    # finalize mean/var per replica → average replicas
    mask = (W_n > 1)
    var  = np.zeros_like(W_mean); var[mask] = W_M2[mask] / (W_n[mask] - 1)
    mean_E = W_mean.mean(axis=0)                 # (K,)
    var_E  = var.mean(axis=0)                    # (K,)
    """


def run_trial_stats(sys: SysConfig, trial: TrialConfig, rid: int,
                    edge_list: np.ndarray | None = None) -> TrialResult:

    # disorder (deterministic per rid)
    ξ = sample_ξ(sys.N, sys.P, sys.master_seed, rid, sys.c)

    # couplings for all t_b
    G_all, A, d = build_couplings_for_t_grid(ξ, sys.t_grid)   # A:(B,N,P) d:(B,N)

    # init state (deterministic per rid)
    Σ = init_spins(sys, rid, ξ)                               # (2,R,N)

    M = np.empty((2, sys.R, sys.P), dtype=np.float64)
    Ξ = np.empty((2, sys.R),       dtype=np.float64)

    for chain in (0, 1):
        for b in range(sys.B):
            r0, r1 = sys.k_start[b], sys.k_start[b + 1]
            M[chain, r0:r1] = compute_M_from_σ_ξ(Σ[chain, r0:r1], ξ)
            Ξ[chain, r0:r1] = compute_E_from_M(M[chain, r0:r1], G_all[b], sys.N)

    Ψ = np.tile(np.arange(sys.R, dtype=np.int64), (2, 1))

    seeds = _make_seed_matrix(sys, rid)                       # (2, R+B+1, 2)

    # edges: allow “no vertical swaps” by default
    if edge_list is None:
        edge_list = np.empty((0, 2), dtype=np.int64)
    else:
        edge_list = np.ascontiguousarray(edge_list, dtype=np.int64)

    E = edge_list.shape[0]

    # counters
    swap_count = np.zeros((2, sys.R - 1), dtype=np.int64)
    vertical_swap_count   = np.zeros((2, E), dtype=np.int64)
    vertical_swap_attempt = np.zeros((2, E), dtype=np.int64)

    # matching scratch (per chain)
    order     = np.empty((2, E), dtype=np.int64)
    out_e     = np.empty((2, E), dtype=np.int64)
    mark      = np.zeros((2, sys.R), dtype=np.int32)
    token_arr = np.ones((2, 1), dtype=np.int32)

    for chain in (0, 1):
        order[chain] = np.arange(E, dtype=np.int64)

    # Ψ time series
    T = trial.n_samples * trial.sweeps_per_sample
    Ψ_ts = np.empty((2, T, sys.R), dtype=np.int64)

    # run
    Simulate_two_replicas_stats_2d(
        sys.N, sys.P, sys.R, 1.0 / sys.N, sys.k_start,
        Σ, M, Ξ, Ψ,
        A, ξ, d,
        np.ascontiguousarray(sys.beta, np.float64),
        seeds,
        trial.equilibration_time, trial.sweeps_per_sample, trial.n_samples,
        swap_count,
        vertical_swap_count, vertical_swap_attempt,
        edge_list,
        order, mark, token_arr, out_e,
        sys.b_of_r,
        Ψ_ts
    )

    steps = max(1, trial.n_samples * trial.sweeps_per_sample)

    # horizontal acceptance, sliced per ladder b
    acc_h_flat = swap_count.mean(axis=0) / steps   # (R-1,)

    acc_h_by_t = []
    beta_by_t  = []
    for b in range(sys.B):
        r0, r1 = sys.k_start[b], sys.k_start[b + 1]
        beta_by_t.append(sys.beta[r0:r1].copy())
        acc_h_by_t.append(acc_h_flat[r0:r1-1].copy())   # length K_b-1

    # vertical acceptance per edge
    if E == 0:
        acc_v = np.empty((0,), dtype=np.float64)
    else:
        att = vertical_swap_attempt.mean(axis=0).astype(np.float64)
        acc = vertical_swap_count.mean(axis=0).astype(np.float64)
        acc_v = np.zeros(E, dtype=np.float64)
        mask = att > 0
        acc_v[mask] = acc[mask] / att[mask]

    return TrialResult(
        rid=rid,
        beta_by_t=tuple(beta_by_t),
        acc_h_by_t=tuple(acc_h_by_t),
        acc_v=acc_v,
        Ψ_ts=Ψ_ts
    )


def worker_run_stats(rid: int, sys: SysConfig, trial: TrialConfig, edge_list=None) -> TrialResult:
    return run_trial_stats(sys, trial, rid, edge_list=edge_list)

def pool_orchestrator_stats(sys: SysConfig, trial: TrialConfig,
                            R_workers: int, R_total: int, start_method="fork",
                            edge_list=None):
    mpctx = get_context(start_method)
    out = []
    with ProcessPoolExecutor(max_workers=R_workers, mp_context=mpctx) as ex:
        futs = [ex.submit(worker_run_stats, rid, sys, trial, edge_list) for rid in range(R_total)]
        for f in as_completed(futs):
            out.append(f.result())
    out.sort(key=lambda r: r.rid)
    return out