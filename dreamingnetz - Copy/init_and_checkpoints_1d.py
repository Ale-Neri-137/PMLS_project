"""
SysConfig + initialization + checkpoint I/O for DreamingNetz.

This file is meant to be *import-safe*: it defines helpers and APIs, but does not
run simulations at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import tempfile
import shutil
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SysConfig:
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

MASK64 = (1 << 64) - 1

def splitmix64_py(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64

def tag_to_u64(tag) -> int:
    h = hashlib.blake2b(digest_size=8)
    if isinstance(tag, (int, np.integer)):
        h.update(b'i'); h.update(int(tag).to_bytes(8, 'little', signed=False))
    else:
        b = str(tag).encode('utf-8')
        h.update(b's'); h.update(len(b).to_bytes(4, 'little')); h.update(b)
    return int.from_bytes(h.digest(), 'little')

def derive_xorshift_state(master_seed: int, tag_tuple) -> np.ndarray:
    x = int(master_seed) & MASK64
    for t in tag_tuple:
        x ^= tag_to_u64(t)
        x = splitmix64_py(x); x = splitmix64_py(x)
    s0 = splitmix64_py(x)
    s1 = splitmix64_py(s0 ^ x)
    if (s0 | s1) == 0:
        s1 = 1
    return np.array([np.uint64(s0), np.uint64(s1)], dtype=np.uint64)




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


def _make_seed_matrix(sys: SysConfig, rid: int) -> np.ndarray:
    seeds = np.empty((2, sys.K + 1, 2), dtype=np.uint64)
    for b in (0,1):
        for k in range(sys.K):
            seeds[b, k] = derive_xorshift_state(sys.master_seed, (rid, b, k, "spin"))
        seeds[b, -1] = derive_xorshift_state(sys.master_seed, (rid, b, "swap"))
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


def compute_M_from_σ_ξ(σ: np.ndarray, ξ: np.ndarray) -> np.ndarray:
    """
    σ: (K, N) ±1 int8; ξ: (N, P) ±1 int8
    Returns M: (K, P) float64 where M[k, mu] = (1/N) sum_i ξ[i,mu] * σ[k,i]
    """
    N, P = ξ.shape
    S = σ.astype(np.float64)  # (K, N)
    X = ξ.astype(np.float64)   # (N, P)
    return (S @ X) / float(N)   # (K, P)

def compute_E_from_M(M: np.ndarray, G: np.ndarray, N: int) -> np.ndarray:
    """
    M: (K, P) ; G: (P, P)
    H = -N/2 * sum_μ,ν M_μ G_μ,ν M_ν  per K
    """
    MG = M @ G    # (K, P)
    return -0.5 * N * np.einsum("kp,kp->k", M, MG)

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

# Here we decide what constitutes a state. This is used both for chunking and to resume a run.


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def realization_dir(run_root: str, rid: int) -> str:
    return os.path.join(run_root, f"r{rid:03d}")

#The first verion (truly atomic) works on native Linux systems, not on WSL
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


def save_disorder(rdir: str, ξ: np.ndarray, t: float) -> None:
    atomic_save_npz(os.path.join(rdir, "disorder.npz"),
                    ξ=ξ.astype(np.int8, copy=False), t=np.float64(t))

def load_disorder(rdir: str) -> tuple[np.ndarray, float]:
    z = np.load(os.path.join(rdir, "disorder.npz"))
    return z["ξ"].astype(np.int8, copy=False), float(z["t"])


def save_sysconfig(rdir: str, sys: SysConfig) -> None:
    atomic_save_npz(
        os.path.join(rdir, "sysconfig.npz"),
        N=np.int64(sys.N), P=np.int64(sys.P), K=np.int64(sys.K),
        t=np.float64(sys.t), c=np.float64(sys.c),
        master_seed=np.int64(sys.master_seed),
        beta=np.ascontiguousarray(sys.β, dtype=np.float64),
        mu=np.ascontiguousarray(sys.mu_to_store, dtype=np.int64),
        spin_init_mode=np.array(sys.spin_init_mode, dtype="U"),
    )

def load_sysconfig(rdir: str) -> SysConfig:
    """
    If sysconfig.npz is missing, you must be in fresh mode (provide SysConfig). Otherwise it’s an error.
    """
    z = np.load(os.path.join(rdir, "sysconfig.npz"))
    return SysConfig(
        N=int(z["N"]), P=int(z["P"]), K=int(z["K"]),
        t=float(z["t"]), c=float(z["c"]),
        master_seed=int(z["master_seed"]),
        β=z["beta"].astype(np.float64, copy=False),
        mu_to_store=z["mu"].astype(np.int64, copy=False),
        spin_init_mode=str(z["spin_init_mode"]),
    )


def save_checkpoint(rdir: str, state: dict, n_done: int, β: np.ndarray) -> None:
    atomic_save_npz(os.path.join(rdir, "checkpoint.npz"),
        Σ=state["Σ"], M=state["M"], Ξ=state["Ξ"], Ψ=state["Ψ"],
        seeds=state["seeds"], n_swaps=state["n_swaps"],
        β=np.ascontiguousarray(β, dtype=np.float64),
        n_done=np.int64(n_done),
    )

def load_checkpoint(rdir: str):
    z = np.load(os.path.join(rdir, "checkpoint.npz"))
    state = {"Σ": z["Σ"], "M": z["M"], "Ξ": z["Ξ"], "Ψ": z["Ψ"],
             "seeds": z["seeds"], "n_swaps": z["n_swaps"], "β": z["β"]}
    return state, int(z["n_done"])

# ---------------------------------------------------------------------------
# Start / resume APIs
# ---------------------------------------------------------------------------

def start_fresh(run_root: str, rid: int, sys: SysConfig, run: RunConfig):
    rdir = realization_dir(run_root, rid); ensure_dir(rdir)

    save_sysconfig(rdir, sys)

    ξ = sample_ξ(sys.N, sys.P, sys.master_seed, rid, sys.c)
    save_disorder(rdir, ξ, sys.t)
    G = build_G_t(ξ, sys.t); A, d = build_A_and_Jdiag(G, ξ)

    Σ = init_spins(sys, rid, ξ)   # ← uses sys.spin_init_mode

    M = np.empty((2, sys.K, sys.P), dtype=np.float64)
    Ξ = np.empty((2, sys.K), dtype=np.float64)
    for b in (0, 1):
        M[b] = compute_M_from_σ_ξ(Σ[b], ξ)
        Ξ[b] = compute_E_from_M(M[b], G, sys.N)

    Ψ = np.tile(np.arange(sys.K, dtype=np.int64), (2, 1))
    seeds = _make_seed_matrix(sys, rid)
    n_swaps = np.zeros((2, sys.K - 1), dtype=np.int64)

    #this will be untouched, initialization sanity check
    atomic_save_npz(os.path.join(rdir, "checkpoint_init.npz"),
        Σ=Σ, M=M, Ξ=Ξ, Ψ=Ψ, seeds=seeds, n_swaps=n_swaps,
        n_done=np.int64(0), β=np.ascontiguousarray(sys.β, dtype=np.float64))
    
    #this will be overwritten after thermalization
    save_checkpoint(rdir,
        {"Σ": Σ, "M": M, "Ξ": Ξ, "Ψ": Ψ, "seeds": seeds, "n_swaps": n_swaps},
        n_done=0, β=sys.β)

    return dict(mode="fresh", rdir=rdir, sys=sys, ξ=ξ, G=G, A=A, d=d,
                Σ=Σ, M=M, Ξ=Ξ, Ψ=Ψ, seeds=seeds, n_swaps=n_swaps, n_done=0)

def resume(path_to_realization: str, run: RunConfig):
    rdir = path_to_realization

    # Load physics from disk 
    sys = load_sysconfig(rdir)

    # Rebuild couplings from stored disorder
    ξ, t = load_disorder(rdir)
    if float(t) != float(sys.t):
        raise RuntimeError("Stored disorder t != sys.t (this should not happen).")
    G = build_G_t(ξ, t); A, d = build_A_and_Jdiag(G, ξ)

    state, n_done = load_checkpoint(rdir)
    Σ, M, Ξ, Ψ = state["Σ"], state["M"], state["Ξ"], state["Ψ"]
    seeds, n_swaps = state["seeds"], state["n_swaps"]

    return dict(mode="resume", rdir=rdir, sys=sys, ξ=ξ, G=G, A=A, d=d,
                Σ=Σ, M=M, Ξ=Ξ, Ψ=Ψ, seeds=seeds, n_swaps=n_swaps, n_done=n_done)
