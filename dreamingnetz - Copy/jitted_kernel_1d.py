"""Numba-jitted kernels: RNG, MMC (Metropolis + HS), PT swaps, 2-replica driver.

"""

import math
import numpy as np
from numba import njit
from numba import int8, int64, float64, uint64, void, boolean

__all__ = [
    # RNG
    "xorshift128p_next_uint", "xorshift128p_next_float", "next_uint_bounded",
    "rand_pm1", "randn_std",
    # Metropolis
    "do_one_Metropolis_sweep_return_ΔE",
    # HS helpers
    "_recompute_m", "_recompute_E", "HS_blocked_gibbs_sweep_return_E",
    # PT
    "swap_probability", "attempt_swap",
    # MMC macro-step
    "do_one_MMC_step",
    # LinAlg helpers (jitted)
    "chol_spd", "solve_chol", "compute_cholG_from_xi_A",
    # 2-replica driver
    "Simulate_two_replicas",
]
# ─────────────────────────────────────────────────────────────────────────────
#  Xorshift128+ core: 2-word state → next uint64
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








@njit(float64(  int64, int64, float64,
                int8[::1], float64[::1],
                float64[:,::1], int8[:,::1], float64[::1],
                float64, 
                uint64[::1]),
nogil=True, no_cpython_wrapper=True , fastmath = True
)
def do_one_Metropolis_sweep_return_ΔE(N,P,invN, σ,m, A,ξ,d, β, seed):

    ΔE = 0.0

    for _ in range (N):

        j = int64(next_uint_bounded(seed, np.uint64(N)))

        σ_j = σ[j]
        Aj  = A[j]
        ξj  = ξ[j]

        h_j  = 0.0                                #internal field
        J_jj = d[j]                               #self interaction

        for μ in range(P):  h_j  += Aj[μ] * m[μ]

        ΔH = 2 * (h_j * σ_j - J_jj)  

        if ΔH <= 0.0  or xorshift128p_next_float(seed) < math.exp(-β * ΔH): 

            m -= (2.0 * σ_j * invN) * ξj
            
            σ[j] = -σ_j

            ΔE += ΔH

    return ΔE








@njit(void(int64, int64, float64, int8[::1], int8[:,::1], float64[::1]), 
      nogil=True, inline='always', fastmath=True,no_cpython_wrapper=False)
def _recompute_m(N, P, invN, sigma, xi, m_out):
    """
    Recomputes m_out in-place from sigma and xi.
    Complexity: O(N*P)
    """
    for mu in range(P):
        dot = 0.0
        for i in range(N):
            dot += sigma[i] * xi[i, mu]
        m_out[mu] = dot * invN


@njit(float64(int64, int64, int8[::1], float64[::1], float64[:,::1]), 
      nogil=True, inline='always', fastmath=True ,no_cpython_wrapper=False)
def _recompute_E(N, P, sigma, m, A):
    """
    Recomputes Total Energy from sigma, m, and A.
    Uses H = -0.5 * sum(sigma_i * h_i) where h_i = sum(A_imu * m_mu)
    Complexity: O(N*P)
    """
    total_h_sigma = 0.0
    
    # We loop over N and P to calculate the local field h_i on the fly
    for i in range(N):
        h_i = 0.0
        for mu in range(P):
            h_i += A[i, mu] * m[mu]
        total_h_sigma += h_i * sigma[i]
        
    return -0.5 * total_h_sigma








@njit(float64(uint64[::1]), nogil=True, no_cpython_wrapper=True, fastmath=True)
def randn_std(seed):
    # Box–Muller: one N(0,1)
    u1 = xorshift128p_next_float(seed)
    if u1 < 1e-12:
        u1 = 1e-12
    u2 = xorshift128p_next_float(seed)
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return r * math.cos(theta)








@njit(float64(
        int64, int64, float64,          # N, P, invN
        int8[::1], float64[::1],        # σ (N), m (P)  [in place]
        int8[:, ::1], float64[:, ::1],  # ξ (N,P), L (P,P) with G = L L^T
        float64,                        # β
        uint64[::1],                    # seed
        float64[::1], float64[::1]      # scratch u(P), z(P)
      ),
      nogil=True, no_cpython_wrapper=True, fastmath=True)
def HS_blocked_gibbs_sweep_return_E(N, P, invN, σ, m, ξ, L, β, seed, u, z):
    """
    One HS blocked Gibbs update at fixed β for ONE replica.
    - Updates σ and m in place.
    - Returns E_new = -N/2 * m^T G m, computed as -N/2 * ||L^T m||^2.
    Scratch:
      u: length P  (used for L^T m_old, then eta, then L^T m_new)
      z: length P  (used for mean, then HS field)
    """

    # --- 1) u = L^T m_old (store in u) ---
    for i in range(P):
        acc = 0.0
        for j in range(i, P):          # L[j,i] nonzero for j>=i (L lower-triangular)
            acc += L[j, i] * m[j]
        u[i] = acc

    # --- 2) z = mu = sqrt(β N) * G m = sqrt(β N) * L (L^T m) = sBN * L u ---
    sBN = math.sqrt(β * float64(N))
    for i in range(P):
        acc = 0.0
        for j in range(i + 1):         # lower-triangular multiply
            acc += L[i, j] * u[j]
        z[i] = sBN * acc               # z temporarily holds mu

    # --- 3) sample z = mu + L eta, eta~N(0,I) ---
    # overwrite u with eta
    for i in range(P):
        u[i] = randn_std(seed)

    # add correlated noise: z += L u
    for i in range(P):
        acc = 0.0
        for j in range(i + 1):
            acc += L[i, j] * u[j]
        z[i] += acc

    # --- 4) sample σ | z, accumulate m in-place ---
    # IMPORTANT: compute u_old before zeroing m (done above), now m can be reused as accumulator
    for mu in range(P):
        m[mu] = 0.0

    s = math.sqrt(β * invN)            # sqrt(β/N)
    for i in range(N):
        dot = 0.0
        for mu in range(P):
            dot += float64(ξ[i, mu]) * z[mu]
        h = s * dot

        # p = 1/(1+exp(-2h)) via tanh for stability
        p = 0.5 * (1.0 + math.tanh(h))
        if xorshift128p_next_float(seed) < p:
            σi = int8(1)
        else:
            σi = int8(-1)
        σ[i] = σi

        # accumulate m numerator
        for mu in range(P):
            m[mu] += float64(σi) * float64(ξ[i, mu])

    for mu in range(P):
        m[mu] *= invN

    # --- 5) E_new = -N/2 * ||L^T m_new||^2 ---
    for i in range(P):
        acc = 0.0
        for j in range(i, P):
            acc += L[j, i] * m[j]
        u[i] = acc

    norm2 = 0.0
    for i in range(P):
        norm2 += u[i] * u[i]

    return -0.5 * float64(N) * norm2



# from now on σ is meant to be a K x N array and β is a 1d array of length K.
# E is an array of length K
# Seed_array is an array of seeds of length K+1, one stream for each PT "replica" and the last is for swaps.
#
# I is an array of indices to keep track of the swaps without moving entire configurations








@njit(float64(  float64, float64,
                float64, float64),
nogil=True, no_cpython_wrapper=True, inline='always', fastmath = True
)
def swap_probability(E_a,E_b, β_k1,β_k2):
    
    return math.exp((β_k2 - β_k1)*(E_b - E_a))


@njit(void( boolean,                                   # parity,
            int64,                                     # K,
            float64[::1], int64[::1],                  # E,I,
            float64[::1],                              # β,
            uint64[:,::1],                             # seed_array
            int64[::1]),                               # swap_count
nogil=True, no_cpython_wrapper=True
)
def attempt_swap(parity, K, E,I, β, seed_array, swap_count):

    for k1 in range(np.int64(parity), K-1, 2):
        k2 = k1 + 1
        p = swap_probability(E[I[k1]], E[I[k2]], β[k1], β[k2])

        if (p >= 1.0) or (xorshift128p_next_float(seed_array[-1]) < p): #Metropolis
                
            I_k1 = I[k1]
            I_k2 = I[k2]
            I[k1], I[k2] = I_k2, I_k1  # Swap indices in I directly

            swap_count[k1] +=1








#   σ : (K, N)   int8
#   m : (K, P)   float64
#   E : (K,)     float64
#   I : (K,)     int64      slot -> replica index permutation
#   A : (N, P)   float64    = ξ @ G^T  (used only by Metropolis)
#   ξ : (N, P)   int8
#   d : (N,)     float64    diagonal self-term for Metropolis ΔH
#   L : (P, P)   float64    Cholesky of G: G = L L^T  (used by HS)
#   β : (K,)     float64    inverse temperatures per slot
#   seed_array : (K+1, 2) uint64 seed_array[k] for local RNG stream,seed_array[-1] for swaps
#   swap_count : (K-1,) int64 accepted-swap counts per interface (for this chain)
#   u_scratch  : (K, P) float64 scratch per replica (HS uses u_scratch[rep])
#   z_scratch  : (K, P) float64 scratch per replica (HS uses z_scratch[rep])
#   p_hs       : probability of doing hs, 1-p_hs is the probability of doing Metropolis

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
        float64[:,::1],                        # L
        float64[::1],                          # β
        uint64[:,::1],                         # seed_array
        int64[::1],                            # swap_count
        float64[:,::1],                        # u_scratch
        float64[:,::1],                        # z_scratch
        float64                                # p_hs
    ),
    nogil=True, no_cpython_wrapper=True, fastmath=True
)
def do_one_MMC_step(
    N, P, K, invN,
    σ, m, E, I,
    A, ξ, d, L,
    β, seed_array, swap_count,
    u_scratch, z_scratch,
    p_hs
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
        rep = I[k]  # actual replica index currently sitting at slot k

        # decide kernel (HS with prob p_hs)
        if xorshift128p_next_float(seed_array[k]) < p_hs:
            # HS returns E_new (overwrite E[rep])
            E[rep] = HS_blocked_gibbs_sweep_return_E(
                N, P, invN,
                σ[rep], m[rep],
                ξ, L,
                β[k],                 
                seed_array[k],
                u_scratch[rep], z_scratch[rep]
            )
        else:
            # Metropolis returns ΔE (increment E[rep])
            E[rep] += do_one_Metropolis_sweep_return_ΔE(
                N, P, invN,
                σ[rep], m[rep],
                A, ξ, d,
                β[k],
                seed_array[k]
            )

    attempt_swap(0, K, E, I, β, seed_array, swap_count)

    # -------- second local sweep (then odd swaps) --------
    for k in range(K):
        rep = I[k]

        if xorshift128p_next_float(seed_array[k]) < p_hs:
            E[rep] = HS_blocked_gibbs_sweep_return_E(
                N, P, invN,
                σ[rep], m[rep],
                ξ, L,
                β[k],
                seed_array[k],
                u_scratch[rep], z_scratch[rep]
            )
        else:
            E[rep] += do_one_Metropolis_sweep_return_ΔE(
                N, P, invN,
                σ[rep], m[rep],
                A, ξ, d,
                β[k],
                seed_array[k]
            )

    attempt_swap(1, K, E, I, β, seed_array, swap_count)








@njit(nogil=True, fastmath=True)
def chol_spd(A, L, P):
    # A = L L^T, A SPD, L lower-triangular
    for i in range(P):
        for j in range(P):
            L[i, j] = 0.0

    for i in range(P):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                v = A[i, i] - s
                if v <= 1e-14:
                    v = 1e-14
                L[i, i] = math.sqrt(v)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]


@njit(nogil=True, fastmath=True)
def solve_chol(L, b, x, P):
    # Solve (L L^T) x = b. L lower-triangular.
    # forward: L y = b  (store y into x)
    for i in range(P):
        s = b[i]
        for k in range(i):
            s -= L[i, k] * x[k]
        x[i] = s / L[i, i]

    # backward: L^T x = y  (overwrite x)
    for i in range(P - 1, -1, -1):
        s = x[i]
        for k in range(i + 1, P):
            s -= L[k, i] * x[k]
        x[i] = s / L[i, i]


@njit(
    nogil=True, fastmath=True
)
def compute_cholG_from_xi_A(
    N, P,
    xi, A,
    XtX, XtA, Grec, Lx,
    b, x,
    L_G
):
    """
    Recover G from xi and A=xi@G^T (with G symmetric in your use-case),
    then compute its Cholesky L_G such that G = L_G L_G^T.

    Solves: (xi^T xi) G = (xi^T A)

    All arrays must be preallocated:
      xi   : (N,P) int8
      A    : (N,P) float64
      XtX  : (P,P) float64
      XtA  : (P,P) float64
      Grec : (P,P) float64
      Lx   : (P,P) float64  chol(XtX)
      b,x  : (P,)  float64  scratch
      L_G  : (P,P) float64  output chol(G)
    """

    # form XtX = xi^T xi, XtA = xi^T A
    for mu in range(P):
        for nu in range(P):
            sxx = 0.0
            sxa = 0.0
            for i in range(N):
                xim = float64(xi[i, mu])
                sxx += xim * float64(xi[i, nu])
                sxa += xim * A[i, nu]
            XtX[mu, nu] = sxx
            XtA[mu, nu] = sxa

    # chol(XtX)
    chol_spd(XtX, Lx, P)

    # solve for each column of G: XtX * Grec[:,nu] = XtA[:,nu]
    for nu in range(P):
        for mu in range(P):
            b[mu] = XtA[mu, nu]
            x[mu] = 0.0
        solve_chol(Lx, b, x, P)
        for mu in range(P):
            Grec[mu, nu] = x[mu]

    # symmetrize (numerics)
    for i in range(P):
        for j in range(P):
            Grec[i, j] = 0.5 * (Grec[i, j] + Grec[j, i])

    # chol(G)
    chol_spd(Grec, L_G, P)








@njit(void( int64,int64,int64,float64,                                    # N,P,K,invN       sizes
            int8[:,:,::1],float64[:,:,::1],float64[:,::1],int64[:,::1],   # Σ,M,Ξ,Ψ,         state variables
            float64[:,::1],int8[:,::1],float64[::1],                      # A,ξ,d            structure
            float64[::1],                                                 # β,                
            uint64[:,:,::1],                                              # seed_matrix,
            int64,int64,int64,                                            # eq_time, sam_interv, N_data,
            float64[:,:,:],float64[:,:,:,:],int64[::1],                   # E_ts,m_ts,μ_to_store  
            float64[:, :], # observables are not C-contiguous             # q01_ts
            int64[:,::1]                                                  # replica_swap_count  state var
           ),                          
nogil=False, no_cpython_wrapper=False
)
def Simulate_two_replicas( N,P,K,invN,
                           Σ,M,Ξ,Ψ, 
                           A,ξ,d,
                           β,
                           seed_matrix,
                           equilibration_time, sampling_interval, N_data,
                           E_ts,m_ts,μ_to_store,
                           q01_ts,
                           replica_swap_count ):

    """
    New args:

        Σ,M,Ξ,Ψ : Array of spins, Mattis overlaps, energies, box indeces for PT, extra dimension for 2 replicas.

        seed_matrix : RNG seed matrix for Xorshift128+, extra dimension for the 2 replicas.

        E_ts, m_ts, q_01_ts are empty arrays of shape (2,N_data,K,shape(obs)) to be filled during the run
    """

    p_hs = 0.1
    u_scratch = np.zeros((K,P),dtype = np.float64)
    z_scratch = np.zeros((K,P),dtype = np.float64)
    #stuff for cholesky
    XtX  = np.empty((P,P), np.float64)
    XtA  = np.empty((P,P), np.float64)
    Grec = np.empty((P,P), np.float64)
    Lx   = np.empty((P,P), np.float64)
    L    = np.empty((P,P), np.float64)
    b    = np.empty(P, np.float64)
    x    = np.empty(P, np.float64)

    compute_cholG_from_xi_A(N, P, ξ, A, XtX, XtA, Grec, Lx, b, x, L)

    #THERMALIZE
    for _ in range(equilibration_time):
        do_one_MMC_step(N,P,K,invN, Σ[0],M[0],Ξ[0],Ψ[0], A,ξ,d,L, β, seed_matrix[0],
        replica_swap_count[0],u_scratch,z_scratch,p_hs)

    for _ in range(equilibration_time):
        do_one_MMC_step(N,P,K,invN, Σ[1],M[1],Ξ[1],Ψ[1], A,ξ,d,L, β, seed_matrix[1],
        replica_swap_count[1],u_scratch,z_scratch,p_hs)

    #START SAMPLING
    for n in range(N_data):

        #EVOLVE
        for _ in range(sampling_interval):
            do_one_MMC_step(N,P,K,invN, Σ[0],M[0],Ξ[0],Ψ[0], A,ξ,d,L, β, seed_matrix[0],
            replica_swap_count[0],u_scratch,z_scratch,p_hs)

        
        for _ in range(sampling_interval):
            do_one_MMC_step(N,P,K,invN, Σ[1],M[1],Ξ[1],Ψ[1], A,ξ,d,L, β, seed_matrix[1],
            replica_swap_count[1],u_scratch,z_scratch,p_hs)
        """
        #SNAP BACK (prevent drift)
        for b in range(2):
            for k in range(K):
                _recompute_m(N, P, invN, Σ[b, Ψ[b,k]], ξ, M[b, Ψ[b,k]]) 

                Ξ[b, Ψ[b,k]] = _recompute_E(N, P, Σ[b, Ψ[b,k]], M[b, Ψ[b,k]], A)
        """#the snap back is useless because hs already recomputes from scratch
        #FILL THE TIME SERIES
        for k in range(K):
            E_ts[0,n,k] = Ξ[0, Ψ[0,k]]
            E_ts[1,n,k] = Ξ[1, Ψ[1,k]]

            for p in range (μ_to_store.shape[0]):

                μ = μ_to_store[p]

                m_ts[0,n,k,p] = M[0, Ψ[0,k], μ]
                m_ts[1,n,k,p] = M[1, Ψ[1,k], μ]
                           
            # Gauge-fix to align with retrieval pattern ξ_0
            #sgn0 = 1.0 if m_ts[0,n,k,0] >= 0 else -1.0
            #sgn1 = 1.0 if m_ts[1,n,k,0] >= 0 else -1.0

            q_01_res = 0.                                   # Edward-Anderson overlap
            for i in range(N):
                q_01_res += Σ[0, Ψ[0,k], i] * Σ[1, Ψ[1,k], i]

            q01_ts[n,k] = q_01_res / N