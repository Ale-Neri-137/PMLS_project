
from __future__ import annotations
from dataclasses import dataclass, field
import os
import numpy as np
from numpy.typing import NDArray
import numpy as np


from .RNG_helpers import derive_xorshift_state

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SysConfig:
    N: int
    P: int

    t_grid: NDArray[np.float64]      # (B,)
    K: NDArray[np.int64]             # (B,)
    beta_by_t: tuple[NDArray[np.float64], ...]
    c: float
    mu_to_store: NDArray[np.int64]
    master_seed: int
    spin_init_mode: str = "random"

    # derived
    B: int = field(init=False)
    R: int = field(init=False)
    k_start: NDArray[np.int64] = field(init=False)
    beta: NDArray[np.float64] = field(init=False)
    b_of_r: NDArray[np.int64] = field(init=False)

    def __post_init__(self):
        # Basic scalars
        if self.N <= 0 or self.P <= 0:
            raise ValueError("N, P must be positive")
        c = float(self.c)
        if not (0.0 <= c <= 1.0):
            raise ValueError("c must be in [0,1]")
        object.__setattr__(self, "c", c)

        # t grid
        t_grid = np.ascontiguousarray(np.asarray(self.t_grid, dtype=np.float64))
        if t_grid.ndim != 1 or t_grid.size == 0:
            raise ValueError("t_grid must be 1D and non-empty")
        B = int(t_grid.size)
        object.__setattr__(self, "t_grid", t_grid)
        object.__setattr__(self, "B", B)

        # K
        K = np.ascontiguousarray(np.asarray(self.K, dtype=np.int64))
        if K.ndim != 1 or K.size != B:
            raise ValueError("K must be 1D with same length as t_grid")
        if np.any(K <= 0):
            raise ValueError("All K_b must be positive")
        object.__setattr__(self, "K", K)

        # mu_to_store
        mu = np.ascontiguousarray(np.asarray(self.mu_to_store, dtype=np.int64))
        if mu.size == 0:
            mu = np.array([0], dtype=np.int64)
        if np.any((mu < 0) | (mu >= self.P)):
            raise ValueError("mu_to_store contains indices out of range [0,P)")
        object.__setattr__(self, "mu_to_store", np.unique(mu))

        # k_start and R
        k_start = np.empty(B + 1, dtype=np.int64)
        k_start[0] = 0
        for b in range(B):
            k_start[b + 1] = k_start[b] + K[b]
        R = int(k_start[B])
        object.__setattr__(self, "k_start", np.ascontiguousarray(k_start))
        object.__setattr__(self, "R", R)

        # beta_by_t normalization + flatten
        if len(self.beta_by_t) != B:
            raise ValueError("beta_by_t must have length B")

        beta_flat = np.empty(R, dtype=np.float64)
        beta_by_t_norm = []

        for b in range(B):
            bet = np.ascontiguousarray(np.asarray(self.beta_by_t[b], dtype=np.float64))
            if bet.ndim != 1 or bet.size != K[b]:
                raise ValueError(f"beta_by_t[{b}] must be 1D of length K[{b}]={K[b]}")
            beta_flat[k_start[b]:k_start[b+1]] = bet
            beta_by_t_norm.append(bet)

        object.__setattr__(self, "beta", np.ascontiguousarray(beta_flat))
        object.__setattr__(self, "beta_by_t", tuple(beta_by_t_norm))


        # optional lookup b_of_r (useful everywhere)
        b_of_r = np.empty(R, dtype=np.int64)
        for b in range(B):
            for r in range(k_start[b], k_start[b+1]):
                b_of_r[r] = b
        object.__setattr__(self, "b_of_r", np.ascontiguousarray(b_of_r))

@dataclass(frozen=True)
class RunConfig:
    run_root: str
    equilibration_time: int
    sampling_interval: int
    chunk_size: int
    N_data_target: int
    
    def __post_init__(self):
        if self.equilibration_time < 0 or self.sampling_interval <= 0 or self.chunk_size <= 0:
            raise ValueError("bad run params")
        if self.N_data_target <= 0: raise ValueError("N_data_target must be > 0")



# ---------------------------------------------------------------------------
# Seeding / init
# ---------------------------------------------------------------------------

def sample_ξ(N, P, master_seed, rid, c: float = 0.0):

    if c == 0.0:
        seed0 = derive_xorshift_state(master_seed, (rid, "ξ"))[0]
    else:
        c_tag = np.float64(c).view(np.uint64).item()   # stable, no string/rounding issues
        seed0 = derive_xorshift_state(master_seed, (rid, "ξ", c_tag))[0]
    rng = np.random.default_rng(int(seed0))


    if c == 0.0:
        return (rng.integers(0, 2, size=(N, P), dtype=np.int8) * 2 - 1).astype(np.int8)

    p = 0.5 * (1.0 - np.sqrt(c))

    eta = np.empty(N, dtype=np.int8)
    eta[:N//2] = 1
    eta[N//2:] = -1
    rng.shuffle(eta)

    flip = (rng.random((N, P)) < p)
    s = np.where(flip, -1, 1).astype(np.int8)

    return (eta[:, None] * s).astype(np.int8)

def init_spins(sys: SysConfig, rid: int, xi: np.ndarray) -> np.ndarray:
    """
    Returns Σ with shape (2, R, N), dtype=int8.
    """
    Σ = np.empty((2, sys.R, sys.N), dtype=np.int8)

    if sys.spin_init_mode == "random":
        for chain in (0, 1):
            for r in range(sys.R):
                rng = np.random.default_rng(
                    int(derive_xorshift_state(sys.master_seed, (rid, chain, r, "sigma"))[0])
                )
                Σ[chain, r] = (rng.integers(0, 2, size=sys.N, dtype=np.int8) * 2 - 1).astype(np.int8)

    elif sys.spin_init_mode.startswith("corrupted"):
        target = xi[:, 0]  # pattern 0

        try:
            noise_level = float(sys.spin_init_mode.split("_")[1])
        except Exception:
            noise_level = 0.10

        base_config = np.tile(target, (sys.R, 1)).astype(np.int8)

        for chain in (0, 1):
            rng = np.random.default_rng(
                int(derive_xorshift_state(sys.master_seed, (rid, chain, "sigma_init"))[0])
            )
            flips = rng.random(size=(sys.R, sys.N)) < noise_level
            modifiers = np.where(flips, -1, 1).astype(np.int8)
            Σ[chain] = base_config * modifiers

    else:
        raise ValueError(f"Unknown spin_init_mode: {sys.spin_init_mode!r}")

    return Σ




def _make_seed_matrix(sys: SysConfig, rid: int) -> np.ndarray:
    """
    seeds shape: (2, R + B + 1, 2)

    Indexing convention (matches your jitted code):
      seeds[chain, r]         : local Metropolis RNG for slot r
      seeds[chain, R + b]     : horizontal swaps RNG for ladder b
      seeds[chain, R + B]     : vertical swaps RNG (shared across all vertical edges)
    """
    seeds = np.empty((2, sys.R + sys.B + 1, 2), dtype=np.uint64)

    for chain in (0, 1):
        # local streams
        for r in range(sys.R):
            seeds[chain, r] = derive_xorshift_state(sys.master_seed, (rid, chain, r, "spin"))

        # horizontal swap streams (one per b)
        for b in range(sys.B):
            seeds[chain, sys.R + b] = derive_xorshift_state(sys.master_seed, (rid, chain, b, "hswap"))

        # vertical swap stream
        seeds[chain, sys.R + sys.B] = derive_xorshift_state(sys.master_seed, (rid, chain, "vswap"))

    return seeds

# ---------------------------------------------------------------------------
# dreaming-kernel builders (numpy)
# ---------------------------------------------------------------------------

def build_G_t(ξ: np.ndarray, t: float) -> np.ndarray:
    ξf = np.ascontiguousarray(ξ, dtype=np.float64)
    N, P = ξf.shape
    C = (ξf.T @ ξf) / N
    I = np.eye(P, dtype=np.float64)
    G = np.linalg.solve(I + t * C, (1.0 + t) * I)
    return 0.5 * (G + G.T)

def build_A_and_Jdiag(G: np.ndarray, ξ: np.ndarray):

    Gf  = np.ascontiguousarray(G,  dtype=np.float64)   # (P,P)
    ξf = np.ascontiguousarray(ξ, dtype=np.float64)   # (N,P)
    N   = ξf.shape[0]

    # A[j, μ] = Σ_ν G[μ,ν] * ξ[j,ν]  → A = ξ @ G^T
    A = ξf @ Gf.T                                  # (N,P)
    # Jd[j] = (1/N) Σ_μ A[j,μ]*ξ[j,μ]
    Jd = (A * ξf).sum(axis=1) / float(N)           # (N,)

    # enforce exact dtype/contiguity for Numba callers
    A  = np.ascontiguousarray(A,  dtype=np.float64)
    Jd = np.ascontiguousarray(Jd, dtype=np.float64)
    return A, Jd

def build_couplings_for_t_grid(ξ: np.ndarray, t_grid: np.ndarray):
    """
    Returns:
      G_all : (B, P, P) float64
      A_all : (B, N, P) float64
      d_all : (B, N)    float64   (J_diag per t)
    """
    t_grid = np.ascontiguousarray(np.asarray(t_grid, dtype=np.float64))
    B = t_grid.size
    N, P = ξ.shape

    G_all = np.empty((B, P, P), dtype=np.float64)
    A_all = np.empty((B, N, P), dtype=np.float64)
    d_all = np.empty((B, N),    dtype=np.float64)

    for b in range(B):
        Gb = build_G_t(ξ, float(t_grid[b]))
        Ab, db = build_A_and_Jdiag(Gb, ξ)
        G_all[b] = Gb
        A_all[b] = Ab
        d_all[b] = db

    return (np.ascontiguousarray(G_all),
            np.ascontiguousarray(A_all),
            np.ascontiguousarray(d_all))


def compute_M_from_σ_ξ(σ: np.ndarray, ξ: np.ndarray) -> np.ndarray:
    """
    σ: (L, N) ±1 int8   (L can be K_b or R)
    ξ: (N, P) ±1 int8
    Returns M: (L, P) float64 with M[l,mu] = (1/N) Σ_i σ[l,i] ξ[i,mu]
    """
    N, P = ξ.shape
    S = σ.astype(np.float64, copy=False)   # (L, N)
    X = ξ.astype(np.float64, copy=False)   # (N, P)
    return (S @ X) / float(N)              # (L, P)

def compute_E_from_M(M: np.ndarray, G: np.ndarray, N: int) -> np.ndarray:
    """
    M: (L, P) ; G: (P, P)
    E_l = -N/2 * M_l^T G M_l
    """
    MG = M @ G
    return -0.5 * N * np.einsum("lp,lp->l", M, MG)

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

# Here we decide what constitutes a state. This is used both for chunking and to resume a run.


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def realization_dir(run_root: str, rid: int) -> str:
    return os.path.join(run_root, f"r{rid:03d}")

#The first version (truly atomic) works on native Linux systems, not on WSL
"""
def atomic_save_npz(path: str, **arrays) -> None:
    d = os.path.dirname(path); ensure_dir(d)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        with open(tmp, "rb") as f: os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
"""
def atomic_save_npz(path: str, **arrays) -> None:
    """
    Simpler, non-atomic save that works reliably on WSL2/Windows mounts.
    """
    d = os.path.dirname(path)
    ensure_dir(d)
    # Write directly to the file, skipping the temp-and-swap dance
    np.savez_compressed(path, **arrays)
    # Force OS to flush to disk immediately to prevent 0KB files on crash
    # (Note: we can't fsync a file path directly in Python without opening it, 
    # but closing np.savez usually flushes enough for this context)


def save_disorder(rdir: str, ξ: np.ndarray) -> None:
    atomic_save_npz(os.path.join(rdir, "disorder.npz"),
                    ξ=ξ.astype(np.int8, copy=False))

def load_disorder(rdir: str) -> np.ndarray:
    z = np.load(os.path.join(rdir, "disorder.npz"))
    return z["ξ"].astype(np.int8, copy=False)



def save_sysconfig(rdir: str, sys: SysConfig) -> None:
    atomic_save_npz(
        os.path.join(rdir, "sysconfig.npz"),
        N=np.int64(sys.N),
        P=np.int64(sys.P),
        t_grid=np.ascontiguousarray(sys.t_grid, dtype=np.float64),
        K=np.ascontiguousarray(sys.K, dtype=np.int64),
        c=np.float64(sys.c),
        master_seed=np.int64(sys.master_seed),
        beta=np.ascontiguousarray(sys.beta, dtype=np.float64),  # flat (R,)
        mu=np.ascontiguousarray(sys.mu_to_store, dtype=np.int64),
        spin_init_mode=np.array(sys.spin_init_mode, dtype="U"),
    )

def load_sysconfig(rdir: str) -> SysConfig:
    z = np.load(os.path.join(rdir, "sysconfig.npz"))

    t_grid = z["t_grid"].astype(np.float64, copy=False)
    K      = z["K"].astype(np.int64, copy=False)
    beta   = z["beta"].astype(np.float64, copy=False)  # flat (R,)

    # Rebuild beta_by_t from flat beta + K (works with your ragged ladders)
    k_start = np.insert(np.cumsum(K), 0, 0)
    beta_by_t = tuple(beta[k_start[b]:k_start[b+1]] for b in range(K.shape[0]))
    sim = z["spin_init_mode"]
    spin_init_mode = str(sim.item()) if sim.size == 1 else str(sim)

    return SysConfig(
        N=int(z["N"]),
        P=int(z["P"]),
        t_grid=t_grid,
        K=K,
        beta_by_t=beta_by_t,             # <-- if your SysConfig takes ragged input
        c=float(z["c"]),
        master_seed=int(z["master_seed"]),
        mu_to_store=z["mu"].astype(np.int64, copy=False),
        spin_init_mode=spin_init_mode,
    )


def save_checkpoint(rdir: str, state: dict, n_done: int) -> None:
    atomic_save_npz(
        os.path.join(rdir, "checkpoint.npz"),
        Σ=state["Σ"], M=state["M"], Ξ=state["Ξ"], Ψ=state["Ψ"],
        seeds=state["seeds"],
        swap_count=state["swap_count"],
        vertical_swap_count=state["vertical_swap_count"],
        vertical_swap_attempt=state["vertical_swap_attempt"],
        n_done=np.int64(n_done),
    )

def load_checkpoint(rdir: str):
    z = np.load(os.path.join(rdir, "checkpoint.npz"))
    state = {
        "Σ": z["Σ"], "M": z["M"], "Ξ": z["Ξ"], "Ψ": z["Ψ"],
        "seeds": z["seeds"],
        "swap_count": z["swap_count"],
        "vertical_swap_count": z["vertical_swap_count"],
        "vertical_swap_attempt": z["vertical_swap_attempt"],
    }
    return state, int(z["n_done"])

def save_edges(rdir: str, edge_list: np.ndarray) -> None:
    edge_list = np.ascontiguousarray(edge_list, dtype=np.int64)
    atomic_save_npz(os.path.join(rdir, "edges.npz"), edge_list=edge_list)

def load_edges(rdir: str) -> np.ndarray:
    z = np.load(os.path.join(rdir, "edges.npz"))
    return z["edge_list"].astype(np.int64, copy=False)

# ---------------------------------------------------------------------------
# Start / resume APIs
# ---------------------------------------------------------------------------

def start_fresh(run_root: str, rid: int, sys: SysConfig, run: RunConfig):
    rdir = realization_dir(run_root, rid)
    ensure_dir(rdir)

    save_sysconfig(rdir, sys)

    # disorder
    ξ = sample_ξ(sys.N, sys.P, sys.master_seed, rid, sys.c)
    save_disorder(rdir, ξ)  # (no scalar t anymore)

    # edges (production assumption: you already have them)
    edge_list = load_edges(rdir)  # or: edge_list = sys.edge_list
    edge_list = np.ascontiguousarray(edge_list, dtype=np.int64)
    if edge_list.ndim != 2 or edge_list.shape[1] != 2:
        raise ValueError("edge_list must be (E,2)")
    if edge_list.size:
        if edge_list.min() < 0 or edge_list.max() >= sys.R:
            raise ValueError("edge_list has node outside [0, R)")

    E = edge_list.shape[0]

    # couplings for all t_b
    G_all, A, d = build_couplings_for_t_grid(ξ, sys.t_grid)  # A:(B,N,P), d:(B,N)

    # init state over all slots r
    Σ = init_spins(sys, rid, ξ)  # (2,R,N)

    M = np.empty((2, sys.R, sys.P), dtype=np.float64)
    Ξ = np.empty((2, sys.R), dtype=np.float64)

    for chain in (0, 1):
        for b in range(sys.B):
            r0, r1 = sys.k_start[b], sys.k_start[b+1]
            M[chain, r0:r1] = compute_M_from_σ_ξ(Σ[chain, r0:r1], ξ)
            Ξ[chain, r0:r1] = compute_E_from_M(M[chain, r0:r1], G_all[b], sys.N)

    Ψ = np.tile(np.arange(sys.R, dtype=np.int64), (2, 1))
    seeds = _make_seed_matrix(sys, rid)  # (2, R+B+1, 2)

    # swap stats
    swap_count = np.zeros((2, sys.R - 1), dtype=np.int64)       # horizontal
    vertical_swap_count   = np.zeros((2, E), dtype=np.int64)    # per-edge accepted
    vertical_swap_attempt = np.zeros((2, E), dtype=np.int64)    # per-edge attempted

    atomic_save_npz(os.path.join(rdir, "checkpoint_init.npz"),
        Σ=Σ, M=M, Ξ=Ξ, Ψ=Ψ, seeds=seeds,
        swap_count=swap_count,
        vertical_swap_count=vertical_swap_count,
        vertical_swap_attempt=vertical_swap_attempt,
        n_done=np.int64(0),
    )

    save_checkpoint(rdir,
        dict(Σ=Σ, M=M, Ξ=Ξ, Ψ=Ψ,
             seeds=seeds,
             swap_count=swap_count,
             vertical_swap_count=vertical_swap_count,
             vertical_swap_attempt=vertical_swap_attempt),
        n_done=0,
    )

    return dict(mode="fresh", rdir=rdir, sys=sys,
                ξ=ξ, G_all=G_all, A=A, d=d,
                Σ=Σ, M=M, Ξ=Ξ, Ψ=Ψ,
                seeds=seeds,
                swap_count=swap_count,
                vertical_swap_count=vertical_swap_count,
                vertical_swap_attempt=vertical_swap_attempt,
                edge_list=edge_list,
                n_done=0)


def resume(path_to_realization: str, run: RunConfig):
    rdir = path_to_realization

    sys = load_sysconfig(rdir)
    ξ = load_disorder(rdir)

    edge_list = load_edges(rdir)  # or sys.edge_list
    edge_list = np.ascontiguousarray(edge_list, dtype=np.int64)
    E = edge_list.shape[0]

    G_all, A, d = build_couplings_for_t_grid(ξ, sys.t_grid)

    state, n_done = load_checkpoint(rdir)
    
    if state["vertical_swap_count"].shape[1] != E or state["vertical_swap_attempt"].shape[1] != E:
        raise RuntimeError("edge_list length changed vs checkpoint")

    # optional: ensure horizontal counter length matches current sys.R
    if state["swap_count"].shape[1] != sys.R - 1:
        raise RuntimeError("swap_count length changed vs sys.R")


    # (optional) assert checkpoint shapes match current E
    if state["vertical_swap_count"].shape[1] != E:
        raise RuntimeError("edge_list length changed vs checkpoint")

    return dict(mode="resume", rdir=rdir, sys=sys,
                ξ=ξ, G_all=G_all, A=A, d=d,
                Σ=state["Σ"], M=state["M"], Ξ=state["Ξ"], Ψ=state["Ψ"],
                seeds=state["seeds"],
                swap_count=state["swap_count"],
                vertical_swap_count=state["vertical_swap_count"],
                vertical_swap_attempt=state["vertical_swap_attempt"],
                edge_list=edge_list,
                n_done=n_done)

