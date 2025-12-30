
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np
from numba import njit, int8, float64, int64, uint64, void
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context, get_all_start_methods
from scipy.special import erfcinv





import numpy as np
from numpy.typing import NDArray
from typing import Iterable, Tuple, Set



def _nearest_index_sorted(arr: NDArray[np.float64], x: float) -> int:
    """
    arr must be 1D strictly increasing (or nondecreasing).
    Returns index j minimizing |arr[j] - x| with deterministic tie-break (prefer left).
    """
    j = int(np.searchsorted(arr, x, side="left"))
    if j <= 0:
        return 0
    if j >= arr.size:
        return int(arr.size - 1)

    left = j - 1
    right = j
    dl = abs(arr[left] - x)
    dr = abs(arr[right] - x)
    return left if dl <= dr else right


def build_edge_list_two_sided_nearest_beta(
    beta_by_t: Tuple[NDArray[np.float64], ...],
    k_start: NDArray[np.int64],
    *,
    fully_connect_hottest: bool = True,
) -> NDArray[np.int64]:
    """
    beta_by_t[b]: (K_b,) inverse temperatures for ladder at t_b
    k_start: (B+1,) with k_start[0]=0 and k_start[b+1]=k_start[b]+K_b

    Returns:
      edge_list: (E,2) int64, undirected edges on global r-indices.
    """
    B = len(beta_by_t)
    if k_start.shape[0] != B + 1:
        raise ValueError("k_start must have shape (B+1,) matching beta_by_t")

    # sanity: each ladder must be monotone increasing in beta (hot->cold)
    for b in range(B):
        bet = np.asarray(beta_by_t[b], dtype=np.float64)
        if bet.ndim != 1 or bet.size != (k_start[b+1] - k_start[b]):
            raise ValueError(f"beta_by_t[{b}] has wrong shape")
        if np.any(np.diff(bet) < 0):
            raise ValueError(f"beta_by_t[{b}] must be nondecreasing (hot->cold)")

    edges: Set[Tuple[int, int]] = set()

    # --- vertical edges between adjacent t's: (b) <-> (b+1) ---
    for b in range(B - 1):
        bet_a = np.asarray(beta_by_t[b], dtype=np.float64)
        bet_c = np.asarray(beta_by_t[b+1], dtype=np.float64)

        r0a = int(k_start[b])
        r0c = int(k_start[b+1])

        # a -> c
        for ka, x in enumerate(bet_a):
            kc = _nearest_index_sorted(bet_c, float(x))
            u = r0a + ka
            v = r0c + kc
            if u != v:
                edges.add((u, v) if u < v else (v, u))

        # c -> a  (two-sided)
        for kc, x in enumerate(bet_c):
            ka = _nearest_index_sorted(bet_a, float(x))
            u = r0c + kc
            v = r0a + ka
            if u != v:
                edges.add((u, v) if u < v else (v, u))

    # --- full connect hottest (k=0) across all t's ---
    if fully_connect_hottest and B >= 2:
        hot_nodes = [int(k_start[b]) for b in range(B)]  # k=0 in each ladder
        for i in range(B):
            ui = hot_nodes[i]
            for j in range(i + 1, B):
                vj = hot_nodes[j]
                edges.add((ui, vj) if ui < vj else (vj, ui))

    if not edges:
        return np.empty((0, 2), dtype=np.int64)

    edge_list = np.array(sorted(edges), dtype=np.int64)
    return np.ascontiguousarray(edge_list)
