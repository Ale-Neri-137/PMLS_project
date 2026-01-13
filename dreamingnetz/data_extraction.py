from __future__ import annotations

"""
dreamingnetz.data_extraction

Drop-in replacement for the user's original data_extraction.py with the SAME public
calling signature for `f(...)`.

Main improvements:
- Structural validation when loading (fast by default):
    * contiguous chunk starts per kind
    * consistent shapes/dtypes across chunks
    * cross-kind sample-length consistency (strict by default)
    * checkpoint/edges shape checks
- Optional deep validation (finite/value sanity checks on sampled slices).
- More robust `mu` selection for magnetization:
    * If mu is a scalar int (as in pt_tau), returns shape (..,2,T,R) by squeezing the last axis.
    * If mu is list/array, returns shape (..,2,T,R,len(mu)).
    * Supports passing mu labels (values present in mu_to_store) or positional indices.

Configuration (no signature changes):
    VALIDATION_LEVEL = "fast" | "deep"
    VALIDATION_MODE  = "strict" | "salvage"
"""

from dataclasses import dataclass
from pathlib import Path
import re
import warnings
import numpy as np


# ----------------------------
# Validation configuration
# ----------------------------

VALIDATION_LEVEL: str = "deep"   # "fast" or "deep"
VALIDATION_MODE: str = "strict"  # "strict" or "salvage"

_DEEP_HEAD = 64
_DEEP_TAIL = 64


# ----------------------------
# Small containers
# ----------------------------

@dataclass(frozen=True)
class Meta:
    rdir: Path
    N: int
    P: int
    R: int
    B: int
    t_grid: np.ndarray
    K: np.ndarray
    k_start: np.ndarray
    beta: np.ndarray
    mu_to_store: np.ndarray
    edge_list: np.ndarray
    n_samples: int


@dataclass(frozen=True)
class AccData:
    h_accepted: np.ndarray
    h_accepted_per_sample: np.ndarray
    h_mask_valid: np.ndarray
    v_accepted: np.ndarray
    v_attempted: np.ndarray
    v_acc: np.ndarray
    edge_list: np.ndarray


# ----------------------------
# Path helpers
# ----------------------------

def _as_path(p) -> Path:
    return p if isinstance(p, Path) else Path(str(p))


def _is_realization_dir(p: Path) -> bool:
    return (p / "sysconfig.npz").exists() and (p / "timeseries").is_dir()


def _find_realization_dirs(run_root: Path, rids: list[int] | None = None) -> list[Path]:
    run_root = _as_path(run_root)
    if rids is not None:
        out: list[Path] = []
        for rid in rids:
            cand = run_root / f"r{rid:03d}"
            if not _is_realization_dir(cand):
                raise FileNotFoundError(f"Missing realization dir or sysconfig/timeseries: {cand}")
            out.append(cand)
        return out

    out: list[Path] = []
    pat = re.compile(r"^r(\d+)$")
    for d in sorted(run_root.iterdir()):
        if d.is_dir() and pat.match(d.name) and _is_realization_dir(d):
            out.append(d)
    if not out:
        raise FileNotFoundError(f"No realization folders found under: {run_root}")
    return out


# ----------------------------
# Chunk scanning / coverage
# ----------------------------

def _chunk_start(p: Path) -> int:
    try:
        return int(p.name.split(".")[0])
    except Exception as e:
        raise ValueError(f"Cannot parse chunk start from filename: {p.name}") from e


def _scan_chunks(rdir: Path, kind: str, *, R: int, P_sel: int | None) -> tuple[int, np.dtype, int]:
    tsdir = rdir / "timeseries"
    files = sorted(tsdir.glob(f"*.{kind}.npy"), key=_chunk_start)
    if not files:
        return 0, np.dtype(np.float64), (0 if kind != "m_sel" else int(P_sel or 0))

    dtype0: np.dtype | None = None
    inferred_P_sel = int(P_sel or 0)

    expected = None
    total = 0

    for f in files:
        s = _chunk_start(f)
        try:
            arr = np.load(f, mmap_mode="r")
        except Exception as e:
            raise RuntimeError(f"Failed to read chunk (corrupted/truncated?): {f}") from e

        if dtype0 is None:
            dtype0 = arr.dtype
            if kind == "m_sel":
                inferred_P_sel = int(arr.shape[-1])

        if kind == "E":
            if arr.ndim != 3 or arr.shape[0] != 2 or arr.shape[2] != R:
                raise RuntimeError(f"Bad shape for {kind} chunk {f}: got {arr.shape}, expected (2,T,{R})")
            tlen = int(arr.shape[1])
        elif kind == "q01":
            if arr.ndim != 2 or arr.shape[1] != R:
                raise RuntimeError(f"Bad shape for {kind} chunk {f}: got {arr.shape}, expected (T,{R})")
            tlen = int(arr.shape[0])
        else:  # m_sel
            if arr.ndim != 4 or arr.shape[0] != 2 or arr.shape[2] != R:
                raise RuntimeError(f"Bad shape for {kind} chunk {f}: got {arr.shape}, expected (2,T,{R},P_sel)")
            if inferred_P_sel == 0:
                inferred_P_sel = int(arr.shape[-1])
            if int(arr.shape[-1]) != inferred_P_sel:
                raise RuntimeError(
                    f"Inconsistent P_sel across m_sel chunks in {rdir}: {f} has P_sel={arr.shape[-1]}, "
                    f"expected {inferred_P_sel}"
                )
            tlen = int(arr.shape[1])

        if dtype0 is not None and arr.dtype != dtype0:
            raise RuntimeError(f"Inconsistent dtype across {kind} chunks in {rdir}: {f} has {arr.dtype}, expected {dtype0}")

        if expected is None:
            expected = s
        if s != expected:
            raise RuntimeError(f"Non-contiguous {kind} chunk starts in {rdir}: got {s}, expected {expected} (file {f.name})")
        expected = s + tlen
        total = expected

    return int(total), dtype0 or np.dtype(np.float64), inferred_P_sel


def _deep_sanity_check_chunk(arr: np.ndarray, kind: str, f: Path) -> None:
    if arr.size == 0:
        return

    def _check(x, where: str):
        if not np.all(np.isfinite(x)):
            raise RuntimeError(f"Found NaN/Inf in {kind} chunk {f} ({where}).")

    if kind == "E":
        T = arr.shape[1]
        _check(arr[:, :min(_DEEP_HEAD, T), :], "head")
        _check(arr[:, max(0, T - _DEEP_TAIL):, :], "tail")
    elif kind == "q01":
        T = arr.shape[0]
        _check(arr[:min(_DEEP_HEAD, T), :], "head")
        _check(arr[max(0, T - _DEEP_TAIL):, :], "tail")
    else:
        T = arr.shape[1]
        _check(arr[:, :min(_DEEP_HEAD, T), :, :], "head")
        _check(arr[:, max(0, T - _DEEP_TAIL):, :, :], "tail")


def _infer_n_samples_from_kinds(
    rdir: Path, *, R: int, P_sel_hint: int, level: str, mode: str
) -> tuple[int, dict[str, int], int]:
    T_E, _, _ = _scan_chunks(rdir, "E", R=R, P_sel=None)
    T_q, _, _ = _scan_chunks(rdir, "q01", R=R, P_sel=None)
    T_m, _, P_sel = _scan_chunks(rdir, "m_sel", R=R, P_sel=P_sel_hint)

    T_by = {"E": T_E, "q01": T_q, "m_sel": T_m}

    if level == "deep":
        tsdir = rdir / "timeseries"
        for kind in ("E", "q01", "m_sel"):
            files = sorted(tsdir.glob(f"*.{kind}.npy"), key=_chunk_start)
            if not files:
                continue
            sample_files = (files[0], files[-1]) if len(files) > 1 else (files[0],)
            for f in sample_files:
                arr = np.load(f, mmap_mode="r")
                _deep_sanity_check_chunk(arr, kind, f)

    present = [T for T in (T_E, T_q, T_m) if T > 0]
    if not present:
        return 0, T_by, P_sel

    if mode == "strict":
        missing = [k for k, T in T_by.items() if T == 0]
        if missing:
            raise RuntimeError(f"Missing timeseries kinds in {rdir}: {missing}. Found lengths {T_by}.")
        if not (T_E == T_q == T_m):
            raise RuntimeError(f"Timeseries length mismatch in {rdir}: {T_by}.")
        return int(T_E), T_by, P_sel

    T_eff = int(min(T_E, T_q, T_m))
    if not (T_E == T_q == T_m):
        warnings.warn(
            f"[salvage] Timeseries length mismatch in {rdir}: {T_by}. Using T=min={T_eff}.",
            RuntimeWarning, stacklevel=2
        )
    return T_eff, T_by, P_sel


# ----------------------------
# Metadata loading
# ----------------------------

def _load_meta(rdir: Path) -> Meta:
    rdir = _as_path(rdir)

    z = np.load(rdir / "sysconfig.npz")
    try:
        N = int(z["N"])
        P = int(z["P"])
    except Exception as e:
        raise RuntimeError(f"{rdir}/sysconfig.npz is missing required keys N,P") from e

    t_grid = z["t_grid"].astype(np.float64, copy=False)
    K = z["K"].astype(np.int64, copy=False)
    beta = z["beta"].astype(np.float64, copy=False)

    if "mu" in z:
        mu = z["mu"].astype(np.int64, copy=False)
    elif "mu_to_store" in z:
        mu = z["mu_to_store"].astype(np.int64, copy=False)
    else:
        raise RuntimeError(f"{rdir}/sysconfig.npz missing 'mu' (or 'mu_to_store')")

    B = int(K.shape[0])
    if t_grid.shape[0] != B:
        raise RuntimeError(f"{rdir}/sysconfig.npz inconsistent: len(t_grid)={t_grid.shape[0]} != len(K)={B}")

    k_start = np.insert(np.cumsum(K), 0, 0).astype(np.int64)
    R = int(k_start[-1])
    if beta.size != R:
        raise RuntimeError(f"{rdir}/sysconfig.npz inconsistent: beta.size={beta.size} != R=sum(K)={R}")

    edge_list = np.zeros((0, 2), dtype=np.int64)
    edges_path = rdir / "edges.npz"
    if edges_path.exists():
        ez = np.load(edges_path)
        if "edge_list" in ez:
            edge_list = ez["edge_list"].astype(np.int64, copy=False)
            if edge_list.ndim != 2 or edge_list.shape[1] != 2:
                raise RuntimeError(f"{edges_path} has bad edge_list shape: {edge_list.shape}")

    T_eff, _T_by, inferred_P_sel = _infer_n_samples_from_kinds(
        rdir, R=R, P_sel_hint=int(mu.size), level=VALIDATION_LEVEL, mode=VALIDATION_MODE
    )
    if inferred_P_sel and inferred_P_sel != int(mu.size):
        raise RuntimeError(
            f"{rdir}: m_sel chunks have P_sel={inferred_P_sel}, but sysconfig mu_to_store has size {mu.size}."
        )

    return Meta(
        rdir=rdir, N=N, P=P, R=R, B=B,
        t_grid=t_grid, K=K, k_start=k_start,
        beta=beta, mu_to_store=mu,
        edge_list=edge_list,
        n_samples=int(T_eff),
    )


def _assert_meta_compatible(m0: Meta, mi: Meta) -> None:
    for name in ("N", "P", "R", "B"):
        if getattr(m0, name) != getattr(mi, name):
            raise RuntimeError(
                f"Meta mismatch across realizations: {name}: {getattr(m0, name)} vs {getattr(mi, name)}"
            )

    def eq(a, b):
        return a.shape == b.shape and np.array_equal(a, b)

    for name in ("t_grid", "K", "k_start", "beta", "mu_to_store", "edge_list"):
        a = getattr(m0, name)
        b = getattr(mi, name)
        if not eq(a, b):
            raise RuntimeError(f"Meta mismatch across realizations for {name} ({m0.rdir} vs {mi.rdir})")


# ----------------------------
# Consolidate chunked timeseries into one .npy (cached)
# ----------------------------

def _consolidated_path(rdir: Path, kind: str) -> Path:
    outdir = rdir / "timeseries_full"
    outdir.mkdir(exist_ok=True)
    return outdir / f"{kind}.full.npy"


def _maybe_use_cached(out_path: Path, expected_shape: tuple[int, ...], *, mmap: bool) -> np.ndarray | None:
    if not out_path.exists():
        return None
    try:
        arr = np.load(out_path, mmap_mode="r" if mmap else None)
    except Exception:
        return None
    if tuple(arr.shape) != tuple(expected_shape):
        return None
    return arr


def _load_or_build_timeseries(rdir: Path, kind: str, *, mmap: bool = True, rebuild: bool = False) -> np.ndarray:
    rdir = _as_path(rdir)
    meta = _load_meta(rdir)
    T = int(meta.n_samples)

    if kind not in {"E", "q01", "m_sel"}:
        raise ValueError(f"Unknown timeseries kind: {kind}")

    out_path = _consolidated_path(rdir, kind)

    if kind == "E":
        expected_shape = (2, T, meta.R)
    elif kind == "q01":
        expected_shape = (T, meta.R)
    else:
        expected_shape = (2, T, meta.R, int(meta.mu_to_store.size))

    if not rebuild:
        cached = _maybe_use_cached(out_path, expected_shape, mmap=mmap)
        if cached is not None:
            return cached

    tsdir = rdir / "timeseries"
    files = sorted(tsdir.glob(f"*.{kind}.npy"), key=_chunk_start)
    if not files:
        if VALIDATION_MODE == "strict" and T > 0:
            raise RuntimeError(f"Missing timeseries for kind={kind} in {rdir} but meta.n_samples={T}.")
        arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float64, shape=expected_shape)
        del arr
        return np.load(out_path, mmap_mode="r" if mmap else None)

    first = np.load(files[0], mmap_mode="r")
    dtype = first.dtype
    if kind == "m_sel":
        P_sel = int(first.shape[-1])
        if P_sel != int(meta.mu_to_store.size):
            raise RuntimeError(
                f"{rdir}: m_sel chunks have P_sel={P_sel} but meta.mu_to_store.size={meta.mu_to_store.size}"
            )

    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype, shape=expected_shape)

    expected = 0
    for f in files:
        s = _chunk_start(f)
        if s != expected:
            raise RuntimeError(f"Non-contiguous {kind} chunks: got start {s}, expected {expected} in {rdir}")
        chunk = np.load(f, mmap_mode="r")

        if kind == "E":
            if chunk.ndim != 3 or chunk.shape[0] != 2 or chunk.shape[2] != meta.R:
                raise RuntimeError(f"{f} has bad shape {chunk.shape}, expected (2,T,{meta.R})")
            tlen = int(chunk.shape[1])
            if s >= T:
                break
            tcopy = min(tlen, T - s)
            out[:, s:s+tcopy, :] = chunk[:, :tcopy, :]
            expected += tlen

        elif kind == "q01":
            if chunk.ndim != 2 or chunk.shape[1] != meta.R:
                raise RuntimeError(f"{f} has bad shape {chunk.shape}, expected (T,{meta.R})")
            tlen = int(chunk.shape[0])
            if s >= T:
                break
            tcopy = min(tlen, T - s)
            out[s:s+tcopy, :] = chunk[:tcopy, :]
            expected += tlen

        else:  # m_sel
            if chunk.ndim != 4 or chunk.shape[0] != 2 or chunk.shape[2] != meta.R:
                raise RuntimeError(f"{f} has bad shape {chunk.shape}, expected (2,T,{meta.R},P_sel)")
            if int(chunk.shape[-1]) != int(meta.mu_to_store.size):
                raise RuntimeError(f"{f} has P_sel={chunk.shape[-1]}, expected {meta.mu_to_store.size}")
            tlen = int(chunk.shape[1])
            if s >= T:
                break
            tcopy = min(tlen, T - s)
            out[:, s:s+tcopy, :, :] = chunk[:, :tcopy, :, :]
            expected += tlen

    del out
    return np.load(out_path, mmap_mode="r" if mmap else None)


# ----------------------------
# Acceptance extraction
# ----------------------------

def _horizontal_interface_mask_valid(meta: Meta) -> np.ndarray:
    mask = np.ones(meta.R - 1, dtype=bool)
    for b in range(meta.B - 1):
        r_boundary = int(meta.k_start[b + 1] - 1)
        if 0 <= r_boundary < meta.R - 1:
            mask[r_boundary] = False
    return mask


def _load_acc(rdir: Path, *, mmap: bool = True) -> AccData:
    rdir = _as_path(rdir)
    meta = _load_meta(rdir)

    ck_path = rdir / "checkpoint.npz"
    if not ck_path.exists():
        raise FileNotFoundError(f"Missing checkpoint.npz in {rdir}")
    ck = np.load(ck_path)

    if "swap_count" not in ck:
        raise RuntimeError(f"{ck_path} missing 'swap_count'")
    h_accepted = ck["swap_count"].astype(np.int64, copy=False)
    if h_accepted.shape != (2, meta.R - 1):
        raise RuntimeError(f"{ck_path} swap_count has shape {h_accepted.shape}, expected {(2, meta.R-1)}")

    E = int(meta.edge_list.shape[0])
    v_accepted = np.zeros((2, E), dtype=np.int64)
    v_attempted = np.zeros((2, E), dtype=np.int64)
    if E > 0:
        if "vertical_swap_count" in ck:
            v_accepted = ck["vertical_swap_count"].astype(np.int64, copy=False)
        if "vertical_swap_attempt" in ck:
            v_attempted = ck["vertical_swap_attempt"].astype(np.int64, copy=False)
        if v_accepted.shape != (2, E):
            raise RuntimeError(f"{ck_path} vertical_swap_count shape {v_accepted.shape}, expected {(2,E)}")
        if v_attempted.shape != (2, E):
            raise RuntimeError(f"{ck_path} vertical_swap_attempt shape {v_attempted.shape}, expected {(2,E)}")

    with np.errstate(divide="ignore", invalid="ignore"):
        v_acc = v_accepted / v_attempted
        v_acc = v_acc.astype(np.float64, copy=False)
        v_acc[v_attempted == 0] = np.nan

    n = max(1, int(meta.n_samples))
    h_accepted_per_sample = (h_accepted / float(n)).astype(np.float64, copy=False)
    h_mask = _horizontal_interface_mask_valid(meta)

    return AccData(
        h_accepted=h_accepted,
        h_accepted_per_sample=h_accepted_per_sample,
        h_mask_valid=h_mask,
        v_accepted=v_accepted,
        v_attempted=v_attempted,
        v_acc=v_acc,
        edge_list=meta.edge_list,
    )


# ----------------------------
# mu selection helpers
# ----------------------------

def _resolve_mu_selector(mu_to_store: np.ndarray, mu):
    """
    Return a selector to be used as ts[..., selector] on the last axis.

    - scalar mu -> int (axis squeezed)
    - list/array mu -> np.ndarray (axis preserved)

    Heuristic:
      - If all requested values are present in mu_to_store, treat them as labels.
      - Else treat as positional indices.
    """
    mu_to_store = np.asarray(mu_to_store, dtype=np.int64)
    P_sel = int(mu_to_store.size)
    stored_set = set(int(x) for x in mu_to_store.tolist())
    pos = {int(val): i for i, val in enumerate(mu_to_store.tolist())}

    if isinstance(mu, (int, np.integer)):
        v = int(mu)
        if v in stored_set:
            return int(pos[v])
        if v < 0 or v >= P_sel:
            raise IndexError(
                f"Requested mu index {v} out of range for stored P_sel={P_sel}. "
                f"Stored mu_to_store={mu_to_store.tolist()}."
            )
        return int(v)

    req = np.asarray(mu, dtype=np.int64)
    if req.ndim == 0:
        v = int(req)
        if v in stored_set:
            return int(pos[v])
        if v < 0 or v >= P_sel:
            raise IndexError(
                f"Requested mu index {v} out of range for stored P_sel={P_sel}. "
                f"Stored mu_to_store={mu_to_store.tolist()}."
            )
        return int(v)

    req_list = [int(x) for x in req.tolist()]
    if all(v in stored_set for v in req_list):
        return np.asarray([pos[v] for v in req_list], dtype=np.int64)

    if any(v < 0 or v >= P_sel for v in req_list):
        raise IndexError(
            f"Requested mu indices {req_list} out of range for stored P_sel={P_sel}. "
            f"Stored mu_to_store={mu_to_store.tolist()}."
        )
    return np.asarray(req_list, dtype=np.int64)


# ----------------------------
# Public API
# ----------------------------

def f(
    folder,
    what: str,
    *,
    rid: int | None = None,
    rids: list[int] | None = None,
    mu: int | list[int] | np.ndarray | None = None,
    mmap: bool = True,
    rebuild_cache: bool = False,
):
    folder = _as_path(folder)

    if _is_realization_dir(folder):
        rdirs = [folder]
    else:
        if rid is not None:
            rdirs = _find_realization_dirs(folder, [rid])
        else:
            rdirs = _find_realization_dirs(folder, rids)

    key = what.strip().lower()

    if key in {"meta"}:
        metas = [_load_meta(rd) for rd in rdirs]
        if len(metas) > 1:
            m0 = metas[0]
            for mi in metas[1:]:
                _assert_meta_compatible(m0, mi)
        return metas[0] if len(metas) == 1 else metas

    if key in {"acc"}:
        out = [_load_acc(rd, mmap=mmap) for rd in rdirs]
        return out[0] if len(out) == 1 else out

    if key in {"e", "energy"}:
        kind = "E"
    elif key in {"q01", "overlap", "ea"}:
        kind = "q01"
    elif key in {"m", "mattis", "mag", "magnetization"}:
        kind = "m_sel"
    else:
        raise ValueError(f"Unknown `what`: {what}")

    arrs = []
    for rd in rdirs:
        meta = _load_meta(rd)
        ts = _load_or_build_timeseries(rd, kind, mmap=mmap, rebuild=rebuild_cache)

        if kind == "m_sel" and mu is not None:
            sel = _resolve_mu_selector(meta.mu_to_store, mu)
            ts = ts[..., sel]  # int squeezes, vector preserves

        arrs.append(ts)

    if len(arrs) == 1:
        return arrs[0]
    return np.stack(arrs, axis=0)
