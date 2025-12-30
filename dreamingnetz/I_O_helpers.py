
import numpy as np
import os
import shutil
import tempfile


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

def ensure_edges_file(run_root: str, rdir: str) -> None:
    """
    Production: edges are precomputed once and stored at run_root/edges.npz.
    Each realization dir must have its own copy so start_fresh() can load it.
    """
    src = os.path.join(run_root, "edges.npz")
    dst = os.path.join(rdir, "edges.npz")

    if os.path.exists(dst):
        return
    if not os.path.exists(src):
        raise RuntimeError("Missing edges.npz in run_root (production requires it).")
    os.makedirs(rdir, exist_ok=True)
    shutil.copy2(src, dst)

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


def has_valid_sysconfig(rdir: str) -> bool:
    path = os.path.join(rdir, "sysconfig.npz")
    if not os.path.exists(path):
        return False
    try:
        z = np.load(path)
        _ = int(z["N"]); _ = int(z["P"])
        _ = z["t_grid"]; _ = z["K"]
        _ = float(z["c"])
        _ = int(z["master_seed"])
        _ = z["beta"]; _ = z["mu"]
        _ = z["spin_init_mode"]
        return True
    except Exception:
        return False
