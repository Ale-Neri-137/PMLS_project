import numpy as np
from numba import njit
from numba import int8, int64, float64, uint64, void, boolean
from dataclasses import dataclass

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from .RNG_helpers import xorshift128p_next_float, derive_xorshift_state
from .jitted_kernel import do_one_Metropolis_sweep_return_ΔE,swap_probability
from .init_and_checkpoints import build_A_and_Jdiag, build_G_t, sample_ξ

@njit(void( boolean,                                   # parity,
            int64,                                     # K,
            float64[::1], int64[::1],                  # E,I,
            float64[::1],                              # β,
            uint64[:,::1],                             # seed_array
            int64[::1]),                               # swap_count
nogil=True, no_cpython_wrapper=True
)
def attempt_swap_1d(parity, K, E,I, β, seed_array, swap_count):

    for k1 in range(np.int64(parity), K-1, 2):
        k2 = k1 + 1
        p = swap_probability(E[I[k1]], E[I[k2]], β[k1], β[k2])

        if (p >= 1.0) or (xorshift128p_next_float(seed_array[-1]) < p): #Metropolis
                
            I_k1 = I[k1]
            I_k2 = I[k2]
            I[k1], I[k2] = I_k2, I_k1  # Swap indices in I directly

            swap_count[k1] +=1

@njit(
    void(
        int64, int64, int64, float64,          # N, P, K, invN
        int8[:,  ::1],                         # σ
        float64[:,::1],                        # m
        float64[::1],                          # E
        int64[::1],                            # I
        float64[:,::1],                        # A
        int8[:,  ::1],                         # ξ
        float64[::1],                          # d
        float64[::1],                          # β
        uint64[:,::1],                         # seed_array
        int64[::1]                             # swap_count
    ),
    nogil=True, no_cpython_wrapper=True, fastmath=True
)
def do_one_MMC_step_1d(
    N, P, K, invN,
    σ, m, E, I,
    A, ξ, d, 
    β, seed_array, swap_count
):
    """
    One MMC macro-step for ONE PT chain:
      - local update at each slot (Metropolis or HS with prob p_hs)
      - even-edge swap pass
      - local update again
      - odd-edge swap pass
    """
    # -------- first local sweep (then even swaps) --------
    for k in range(K):
        rep_k = I[k]  # actual replica index currently sitting at slot k

            # Metropolis returns ΔE (increment E[rep])
        E[rep_k] += do_one_Metropolis_sweep_return_ΔE(
                N, P, invN,
                σ[rep_k], m[rep_k],
                A, ξ, d,
                β[k],
                seed_array[k]
            )

    attempt_swap_1d(0, K, E, I, β, seed_array, swap_count)

    # -------- second local sweep (then odd swaps) --------
    for k in range(K):
        rep_k = I[k]  # actual replica index currently sitting at slot k

            # Metropolis returns ΔE (increment E[rep])
        E[rep_k] += do_one_Metropolis_sweep_return_ΔE(
                N, P, invN,
                σ[rep_k], m[rep_k],
                A, ξ, d,
                β[k],
                seed_array[k]
            )

    attempt_swap_1d(1, K, E, I, β, seed_array, swap_count)


@njit(void( int64,int64,int64,float64,                                    # N,P,K,invN
            int8[:,:,::1],float64[:,:,::1],float64[:,::1],int64[:,::1],   # Σ,M,Ξ,Ψ
            float64[:,::1],int8[:,::1],float64[::1],                      # A,ξ,d
            float64[::1],                                                 # β
            uint64[:,:,::1],                                              # seed_matrix
            int64,int64,                                                  # eq_time, sweeps_per_sample
            int64,                                                        # n_samples
            int64[:,::1],                                                 # replica_swap_count (state)
            int8[:,:,::1]                                                 # I_ts
          ),
      nogil=False, no_cpython_wrapper=False, fastmath = True)
def Simulate_two_replicas_stats_1d( N,P,K,invN,
                                 Σ,M,Ξ,Ψ,
                                 A,ξ,d,
                                 β,
                                 seed_matrix,
                                 equilibration_time, sweeps_per_sample,
                                 n_samples,
                                 replica_swap_count,
                                 I_ts ):
    


    # --- equilibrate ---
    for _ in range(equilibration_time):
        do_one_MMC_step_1d(N,P,K,invN, Σ[0],M[0],Ξ[0],Ψ[0], A,ξ,d, β, seed_matrix[0], replica_swap_count[0])
    for _ in range(equilibration_time):
        do_one_MMC_step_1d(N,P,K,invN, Σ[1],M[1],Ξ[1],Ψ[1], A,ξ,d, β, seed_matrix[1], replica_swap_count[1])

    replica_swap_count.fill(0)

    # --- production samples ---
    for n in range(n_samples):
        # decorrelate between samples
        for t in range(sweeps_per_sample):
            do_one_MMC_step_1d(N,P,K,invN, Σ[0],M[0],Ξ[0],Ψ[0], A,ξ,d, β, seed_matrix[0], replica_swap_count[0])
            I_ts[0,sweeps_per_sample*n+t] = Ψ[0]

        for t in range(sweeps_per_sample):
            do_one_MMC_step_1d(N,P,K,invN, Σ[1],M[1],Ξ[1],Ψ[1], A,ξ,d, β, seed_matrix[1], replica_swap_count[1])
            I_ts[1,sweeps_per_sample*n+t] = Ψ[1]



        """
        # read energies & Welford update per replica and temp
        for k in range(K):
            # replica 0
            x0 = Ξ[0, Ψ[0,k]]
            n0 = W_n[0,k] + 1
            delta0 = x0 - W_mean[0,k]
            W_mean[0,k] += delta0 / n0
            W_M2[0,k]   += delta0 * (x0 - W_mean[0,k])
            W_n[0,k]     = n0

            # replica 1
            x1 = Ξ[1, Ψ[1,k]]
            n1 = W_n[1,k] + 1
            delta1 = x1 - W_mean[1,k]
            W_mean[1,k] += delta1 / n1
            W_M2[1,k]   += delta1 * (x1 - W_mean[1,k])
            W_n[1,k]     = n1

        """



@dataclass(frozen=True)
class SysConfig_1d:
    N: int; P: int; K: int
    t: float
    c: float 
    β: np.ndarray            # (K,)
    mu_to_store: np.ndarray  # e.g. [0]
    master_seed: int
    spin_init_mode: str = "random"   # this can be changed for different experiments

    def __post_init__(self):
        beta = np.ascontiguousarray(np.asarray(self.β, dtype=np.float64))
        mu   = np.ascontiguousarray(np.asarray(self.mu_to_store, dtype=np.int64))
        if beta.ndim != 1 or beta.size != self.K: raise ValueError("β must be 1D(K)")
        if mu.size == 0: mu = np.array([0], dtype=np.int64)
        if np.any((mu < 0) | (mu >= self.P)): raise ValueError("mu out of range")
        object.__setattr__(self, "β", beta)
        object.__setattr__(self, "mu_to_store", np.unique(mu))

        c = float(self.c)
        if not (0.0 <= c <= 1.0): raise ValueError("c must be in [0,1]")
        object.__setattr__(self, "c", c)


def init_spins_1d(sys: SysConfig_1d, rid: int, xi: np.ndarray) -> np.ndarray:
    """
    Returns Σ with shape (2, K, N), dtype=int8.
    Uses sys.spin_init_mode to choose the strategy.
    Modes now: "random".
    """
    Σ = np.empty((2, sys.K, sys.N), dtype=np.int8)

    if sys.spin_init_mode == "random":
        for b in (0, 1):
            rng = np.random.default_rng(int(derive_xorshift_state(sys.master_seed, (rid, b, "sigma"))[0]))
            Σ[b] = (rng.integers(0, 2, size=(sys.K, sys.N), dtype=np.int8) * 2 - 1)

    
    elif sys.spin_init_mode.startswith("corrupted"):

        target = xi[:, 0] # Target pattern 0
        
        # 2. Parse noise level (e.g., "corrupted_0.10")
        try:
            noise_level = float(sys.spin_init_mode.split("_")[1])
        except:
            noise_level = 0.10 # Default 10%
            
        # 3. Create corrupted replicas
        for b in (0, 1):
            # Deterministic seed for flips
            rng = np.random.default_rng(
                int(derive_xorshift_state(sys.master_seed, (rid, b, "sigma_init"))[0])
            )
            
            # Broadcast target to (K, N)
            base_config = np.tile(target, (sys.K, 1))
            
            # Generate flip mask
            flips = rng.random(size=(sys.K, sys.N)) < noise_level
            
            # Apply flips: sigma = xi * (-1 if flip else 1)
            # -1 * 1 = -1 (flip), 1 * 1 = 1 (keep)
            modifiers = np.where(flips, -1, 1).astype(np.int8)
            Σ[b] = base_config * modifiers

    else:
        raise ValueError(f"Unknown spin_init_mode: {sys.spin_init_mode!r}")

    return Σ


def _make_seed_matrix_1d(sys: SysConfig_1d, rid: int) -> np.ndarray:
    seeds = np.empty((2, sys.K + 1, 2), dtype=np.uint64)
    for b in (0,1):
        for k in range(sys.K):
            seeds[b, k] = derive_xorshift_state(sys.master_seed, (rid, b, k, "spin"))
        seeds[b, -1] = derive_xorshift_state(sys.master_seed, (rid, b, "swap"))
    return seeds


def compute_M_from_σ_ξ_1d(σ: np.ndarray, ξ: np.ndarray) -> np.ndarray:
    """
    σ: (K, N) ±1 int8; ξ: (N, P) ±1 int8
    Returns M: (K, P) float64 where M[k, mu] = (1/N) sum_i ξ[i,mu] * σ[k,i]
    """
    N, P = ξ.shape
    S = σ.astype(np.float64)  # (K, N)
    X = ξ.astype(np.float64)   # (N, P)
    return (S @ X) / float(N)   # (K, P)

def compute_E_from_M_1d(M: np.ndarray, G: np.ndarray, N: int) -> np.ndarray:
    """
    M: (K, P) ; G: (P, P)
    H = -N/2 * sum_μ,ν M_μ G_μ,ν M_ν  per K
    """
    MG = M @ G    # (K, P)
    return -0.5 * N * np.einsum("kp,kp->k", M, MG)

### Lighter orchestration

####  Configs

@dataclass(frozen=True)
class TrialConfig_1d:
    equilibration_time: int
    sweeps_per_sample: int
    n_samples: int

@dataclass
class TrialResult_1d:
    rid: int
    beta: np.ndarray        # (K,)
    acc_edge: np.ndarray    # (K-1,) acceptance per edge (both replicas avg)
    I_ts: np.ndarray        #(K,) timeseries of swaps

    """
    mean_E: np.ndarray      # (K,)   avg over replicas
    var_E: np.ndarray       # (K,)   avg over replicas
    """

#### Runner, worker and executor

def run_trial_stats_1d(sys: SysConfig_1d, trial: TrialConfig_1d, rid: int) -> TrialResult_1d:
    # disorder/couplings (deterministic per rid)
    ξ  = sample_ξ(sys.N, sys.P, sys.master_seed, rid, sys.c)
    G  = build_G_t(ξ, sys.t)
    A, d = build_A_and_Jdiag(G, ξ)

    # initial microstate (deterministic per rid)
    Σ = init_spins_1d(sys, rid, ξ)                  # your helper from earlier (2, K, N)
    M0 = compute_M_from_σ_ξ_1d(Σ[0], ξ)
    M1 = compute_M_from_σ_ξ_1d(Σ[1], ξ)
    M  = np.stack([M0, M1], axis=0)
    Ξ = np.empty((2, sys.K), np.float64)      # energies per replica, filled by your sweeps
    for b in (0,1):
        Ξ[b] = compute_E_from_M_1d(M[b], G, sys.N)
    Ψ = np.tile(np.arange(sys.K, dtype=np.int64), (2,1))
    seeds = _make_seed_matrix_1d(sys, rid)
    swap_count = np.zeros((2, sys.K-1), dtype=np.int64)

    I_ts = np.zeros((2,trial.n_samples*trial.sweeps_per_sample,sys.K),dtype=np.int8)
    """
    # Welford accumulators
    W_n    = np.zeros((2, sys.K), dtype=np.int64)
    W_mean = np.zeros((2, sys.K), dtype=np.float64)
    W_M2   = np.zeros((2, sys.K), dtype=np.float64)
    """
    # one in-memory stats run
    Simulate_two_replicas_stats_1d(
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


    

    return TrialResult_1d(rid=rid, beta=sys.β.copy(), acc_edge=acc_edge, I_ts=I_ts)



def worker_run_stats_1d(rid: int, sys: SysConfig_1d, trial: TrialConfig_1d) -> TrialResult_1d:
    return run_trial_stats_1d(sys, trial, rid)

def pool_orchestrator_stats_1d(sys: SysConfig_1d, trial: TrialConfig_1d,
                            R_workers: int, R_total: int, start_method="fork"):
    mpctx = get_context(start_method)
    out = []
    with ProcessPoolExecutor(max_workers=R_workers, mp_context=mpctx) as ex:
        futs = [ex.submit(worker_run_stats_1d, rid, sys, trial) for rid in range(R_total)]
        for f in as_completed(futs): out.append(f.result())
    # sort by rid
    out.sort(key=lambda r: r.rid)
    return out


