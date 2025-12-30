import numpy as np
from numba import njit
from numba import int8, float64, uint64
# ─────────────────────────────────────────────────────────────────────────────
#  Xorshift128+ core: 2-word state → used inside jitted functions
# ─────────────────────────────────────────────────────────────────────────────
@njit(uint64(uint64[::1]),  nogil=True, inline='always', no_cpython_wrapper=True)
def xorshift128p_next_uint(state):
    """
    In-place update of a 2-element uint64[::1] (s[0], s[1]).
    Returns next 64-bit unsigned integer.
    """
    s1 = state[0]
    s0 = state[1]
    state[0] = s0
    s1 ^= (s1 << np.uint64(23))
    s1 ^= (s1 >> np.uint64(17))
    s1 ^= s0
    s1 ^= (s0 >> np.uint64(26))
    state[1] = s1
    return s0 + s1


@njit(float64(uint64[::1]), nogil=True, inline='always', no_cpython_wrapper=True)
def xorshift128p_next_float(state):
    u = xorshift128p_next_uint(state) >> np.uint64(11)   # upper 53 bits
    return u * np.float64(1.1102230246251565e-16) #2^-53                      # scale to [0,1)x


# ─────────────────────────────────────────────────────────────────────────────
#  Bounded integer 0 … bound-1  (round-down modulo, power-of-two fast-path)
# ─────────────────────────────────────────────────────────────────────────────
@njit(uint64(uint64[::1], uint64), inline='always', nogil=True, no_cpython_wrapper=True)
def next_uint_bounded(state, bound):
    # power-of-two fast path
    if bound & (bound - 1) == 0:
        return xorshift128p_next_uint(state) & (bound - 1)
    # otherwise simple modulo – bias ≈ bound / 2⁶⁴  < 1.5e-18  for bound<=250
    return xorshift128p_next_uint(state) % bound

@njit(int8(uint64[::1]), nogil=True, inline='always', no_cpython_wrapper=True)
def rand_pm1(state):
    x = xorshift128p_next_uint(state)
    # take the top bit (better quality than low bits) → 0 or 1
    b = np.uint64(x >> np.uint64(63))
    # map {0,1} → {+1,-1} without overflow
    return np.int8(1 - 2 * np.int64(b))



import hashlib
# ─────────────────────────────────────────────────────────────────────────────
#  Hashlib seed generator → used in the outer functions
# ─────────────────────────────────────────────────────────────────────────────
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
