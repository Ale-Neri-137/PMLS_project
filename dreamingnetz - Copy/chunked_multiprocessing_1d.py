"""Chunked multi-processing over disorder realizations.

This module handles:
- writing per-chunk time series arrays atomically
- resuming progress from chunk files + checkpoints
- running one realization per process via ProcessPoolExecutor

Notes:
- This module *expects* the following symbols to be importable (adjust imports as needed):
    - Simulate_two_replicas  (your Numba kernel)
    - realization_dir, start_fresh, resume, save_checkpoint (I/O + state)
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Any, cast

import numpy as np

# ---- Optional explicit imports (edit these to match your package layout) ----
from .jitted_kernel import Simulate_two_replicas
from .init_and_checkpoints import realization_dir, start_fresh, resume, save_checkpoint


# ----------------------------- Atomic writes -----------------------------

def _fsync_dir(dirpath: str) -> None:
    """Best-effort directory fsync for crash-safety."""
    try:
        dirfd = os.open(dirpath, os.O_DIRECTORY)  # type: ignore[attr-defined]
    except Exception:
        return
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)


def _atomic_save_npy(path: str, arr: np.ndarray) -> None:
    """Atomically write a .npy file (write temp, validate, replace)."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".npy", dir=d or None)
    os.close(fd)
    try:
        np.save(tmp, arr)
        # sanity check: ensure file is readable (catches partial writes)
        np.load(tmp, mmap_mode="r")
        os.replace(tmp, path)
        if d:
            _fsync_dir(d)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass

class ChunkWriter:
    """
    Writes per-chunk files:
        timeseries/
          E.{start:09d}.npy      shape (2, T, K)
          q01.{start:09d}.npy    shape (T, K)
          m_sel.{start:09d}.npy  shape (2, T, K, P_sel)  (optional)
    """
    def __init__(self, rdir: str, K: int, P_sel: int, store_m_sel: bool = True):
        self.base = os.path.join(rdir, "timeseries")
        self.K = int(K)
        self.P_sel = int(P_sel)
        self.store_m_sel = bool(store_m_sel)
        os.makedirs(self.base, exist_ok=True)

    def write(self, start_idx: int,
              E_chunk: np.ndarray,      # (2, T, K)
              q01_chunk: np.ndarray,    # (T, K)
              m_sel_chunk: np.ndarray | None  # (2, T, K, P_sel) or None
              ):
        T = int(E_chunk.shape[1])
        pref = os.path.join(self.base, f"{start_idx:09d}")
        _atomic_save_npy(pref + ".E.npy",   E_chunk)
        _atomic_save_npy(pref + ".q01.npy", q01_chunk)
        if self.store_m_sel:
            if m_sel_chunk is None:
                raise ValueError("store expects m_sel but got None")
            _atomic_save_npy(pref + ".m_sel.npy", m_sel_chunk)


# %% [markdown]
# ## The chunked simulation loop

# %%
def run_chunked(state: dict, run, *, rid: int | None = None, log_every_chunk: bool = True):
    """
    Works for both fresh and resume states:
      state: { rdir, sys, ξ, G, A, d, Σ, M, Ξ, Ψ, seeds, n_swaps, n_done, mode }
      run  : RunConfig
    """

    prefix = f"[rid={rid:03d} pid={os.getpid()}]" if rid is not None else f"[pid={os.getpid()}]"

    sys = state["sys"]
    rdir = state["rdir"]
    Σ, M, Ξ, Ψ = state["Σ"], state["M"], state["Ξ"], state["Ψ"]
    seeds, n_swaps = state["seeds"], state["n_swaps"]
    ξ, A, d = state["ξ"], state["A"], state["d"]

    invN = 1.0 / float(sys.N)
    n_done = int(state["n_done"])
    target = int(run.N_data_target)

    # Prepare writer
    P_sel = int(sys.mu_to_store.size)
    writer = ChunkWriter(rdir, K=sys.K, P_sel=P_sel, store_m_sel=True)

    # One-time equilibration if FRESH and n_done == 0
    if state.get("mode") == "fresh" and n_done == 0 and run.equilibration_time > 0:
        t_eq0 = time.perf_counter()
        # We can simply call the kernel with N_data=0 and eq_time>0
        Simulate_two_replicas(
            sys.N, sys.P, sys.K, invN,
            Σ, M, Ξ, Ψ,
            A, ξ, d,
            np.ascontiguousarray(sys.β, dtype=np.float64),
            seeds,
            run.equilibration_time,  # eq_time
            run.sampling_interval,   # doesn’t matter for N_data=0
            0,                       # N_data = 0 → only equilibrate
            np.empty((2,0,sys.K), dtype=np.float64),
            np.empty((2,0,sys.K,P_sel), dtype=np.float64),
            sys.mu_to_store.astype(np.int64, copy=False),
            np.empty((0,sys.K), dtype=np.float64),
            n_swaps
        )
        # Reset swap counters after equilibration so production stats start clean
        n_swaps.fill(0)
        # Save a checkpoint *post*-equilibration
        save_checkpoint(rdir,
            {"Σ": Σ, "M": M, "Ξ": Ξ, "Ψ": Ψ, "seeds": seeds, "n_swaps": n_swaps},
            n_done=n_done, β=sys.β)
        if log_every_chunk:
            print(f"{prefix} [eq] t_eq={time.perf_counter()-t_eq0:.2f}s | n_done={n_done}", flush=True)

    t0 = time.perf_counter()
    while n_done < target:
        take = min(run.chunk_size, target - n_done)
        start_idx = n_done

        # Allocate chunk buffers
        E_chunk   = np.empty((2, take, sys.K), dtype=np.float64)
        m_sel     = np.empty((2, take, sys.K, P_sel), dtype=np.float64)
        q01_chunk = np.empty((take, sys.K), dtype=np.float64)

        t_chunk = time.perf_counter()
        # Fill via your kernel (does 2x MMC sweeps per sample internally)
        Simulate_two_replicas(
            sys.N, sys.P, sys.K, invN,
            Σ, M, Ξ, Ψ,
            A, ξ, d,
            np.ascontiguousarray(sys.β, dtype=np.float64),
            seeds,
            0,                       # eq_time handled outside
            run.sampling_interval,
            take,
            E_chunk, m_sel,
            sys.mu_to_store.astype(np.int64, copy=False),
            q01_chunk,
            n_swaps
        )

        # Persist this chunk
        writer.write(start_idx, E_chunk, q01_chunk, m_sel)

        # Update progress & checkpoint
        n_done += take
        save_checkpoint(rdir,
            {"Σ": Σ, "M": M, "Ξ": Ξ, "Ψ": Ψ, "seeds": seeds, "n_swaps": n_swaps},
            n_done=n_done, β=sys.β)

        if log_every_chunk:
            elapsed = time.perf_counter() - t_chunk
            total   = time.perf_counter() - t0
            steps   = max(1, n_done * run.sampling_interval)
            acc     = float(n_swaps.mean() / steps)
            print(f"{prefix} [chunk] n_done={n_done}/{target} "
                  f"acc≈{acc:.3f} t_total={total/60:.1f}m", flush=True)

    # return updated state
    state["n_done"] = n_done
    return state


# %% [markdown]
# ## The runner and the orchestrator

# %% [markdown]
# #### Sanity checks when resuming

# %%
def safe_len_from_chunks(rdir: str) -> int:
    """Reconstruct length from chunk files (most trustworthy for resume)."""
    base = os.path.join(rdir, "timeseries")
    if not os.path.isdir(base): return 0
    starts = []
    for name in os.listdir(base):
        if name.endswith(".E.npy"):
            stem = name[:-len(".E.npy")]
            try: starts.append(int(stem))
            except: pass
    if not starts: return 0
    total = 0
    for s in sorted(starts):
        E = np.load(os.path.join(base, f"{s:09d}.E.npy"), mmap_mode="r")
        total = max(total, s + E.shape[1])
    return total

def reconcile_n_done(rdir: str, n_done_ckpt: int) -> int:
    """Prefer the length implied by chunk files; warn if mismatch."""
    n_files = safe_len_from_chunks(rdir)
    if n_files != n_done_ckpt:
        print(f"[reconcile] {os.path.basename(rdir)}: checkpoint n_done={n_done_ckpt} "
              f"vs files={n_files} → using files", flush=True)
    return n_files



# %%
def has_valid_sysconfig(rdir: str) -> bool:
    path = os.path.join(rdir, "sysconfig.npz")
    if not os.path.exists(path):
        return False
    try:
        z = np.load(path)
        # touch required fields; will raise if corrupt/empty
        _ = int(z["N"]); _ = int(z["P"]); _ = int(z["K"])
        _ = float(z["t"]);_ = float(z["c"])
        _ = int(z["master_seed"])
        _ = z["beta"]; _ = z["mu"]
        return True
    except Exception:
        return False


# %% [markdown]
# ### The worker entry ( one process = one realization )

# %%
def worker_run(rid: int, run, sys_if_needed):
    rdir = realization_dir(run.run_root, rid)
    ok = os.path.isdir(rdir) and has_valid_sysconfig(rdir)

    if ok:
        state = resume(rdir, run)
    else:
        # folder missing or has broken/empty sysconfig.npz → start fresh
        if os.path.isdir(rdir):
            shutil.rmtree(rdir, ignore_errors=True)  # wipe partial dir
        if sys_if_needed is None:
            raise RuntimeError(f"r{rid:03d}: need SysConfig for fresh start")
        state = start_fresh(run.run_root, rid, sys_if_needed, run)

    # reconcile and run
    n_done_ckpt = int(cast(int, state["n_done"]))   
    state["n_done"] = reconcile_n_done(rdir, n_done_ckpt)
    t0 = time.perf_counter()
    state = run_chunked(state, run, rid = rid)

    return {
        "rid": rid,
        "rdir": rdir,
        "n_written": state["n_done"],
        "t_minutes": (time.perf_counter() - t0)/60.0,
    }



# %% [markdown]
# ### The pool executor

# %%
def run_pool(run, sys_for_fresh, R_workers: int, R_total: int, start_method="fork"):
    os.makedirs(run.run_root, exist_ok=True)
    mpctx = get_context(start_method)
    results = []
    with ProcessPoolExecutor(max_workers=R_workers, mp_context=mpctx) as ex:
        futs = [ex.submit(worker_run, rid, run, sys_for_fresh) for rid in range(R_total)]
        for f in as_completed(futs):
            results.append(f.result())
            r = results[-1]
            print(f"[pool] rid={r['rid']:03d} done | "
                  f"t≈{r['t_minutes']:.1f}m | {os.path.basename(r['rdir'])}", flush=True)
    return sorted(results, key=lambda x: x["rid"])

