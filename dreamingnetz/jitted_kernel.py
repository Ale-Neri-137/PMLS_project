
import math
import numpy as np
from numba import njit
from numba import int8, int64, float64, uint64, void, boolean, int32, types

from .RNG_helpers import xorshift128p_next_float, next_uint_bounded

"""
this first 3 functions are unaware of any PT scheme we would like to make, they see a single system/configuration/box
"""

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


##______helpers_______##

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

##____________________##





""" Now PT starts
We will call parallel tempering replicas simply boxes, to not get confused with replicas in the spin glass sense

B = number of t values, b ∈ {0, ... , B-1}  t values are indexed with b    ->  t[b] or t_b 

For each t_b there is a ladder of K_b temperatures, K is an array on length B 

R = Σ_b(K_b)  is the total number of boxes, r will be a flattened box index, r[b,k] = k_start[b] + k

From now on σ is meant to be a R x N array and β is an array of length R. E is an array of length R

Seed_array is an array of seeds (seed = 2 word state) of length Σ_b(K_b+1) + 1 = R + B + 1, why?

        # each box has its own RNG stream for local Metropolis evolution,
        # each fixed-b temperature ladder has its own RNG stream for swaps
        # all the swaps between different-b boxes are handled by a single RNG stream

I is an array of indices of length R to keep track of the swaps without moving entire configurations

"""

@njit(float64(  float64, float64,
                float64, float64),
nogil=True, no_cpython_wrapper=True, inline='always', fastmath = False
)
def swap_probability(E_1,E_2, β_k1,β_k2):

    exponent = (β_k2 - β_k1)*(E_2 - E_1)

    if exponent > 0 : return 1.0

    else:             return math.exp(exponent)


@njit(void( boolean,                                   # parity,
            int64,int64[::1],                          # R,k_start = [0,cumsum(K)]
            float64[::1], int64[::1],                  # E,I,
            float64[::1],                              # β,
            uint64[:,::1],                             # seed_array
            int64[::1]),                               # swap_count
nogil=True, no_cpython_wrapper=True
)
def attempt_swaps(parity, R,k_start, E,I, β, seed_array, swap_count):

    for b in range (k_start.shape[0]-1):

        for k1 in range(k_start[b]+np.int64(parity), k_start[b+1]-1, 2):
            k2 = k1 + 1
            p = swap_probability(E[I[k1]], E[I[k2]], β[k1], β[k2])

            if (p >= 1.0) or (xorshift128p_next_float(seed_array[R + b]) < p): #Metropolis
                    
                I_k1 = I[k1]
                I_k2 = I[k2]
                I[k1], I[k2] = I_k2, I_k1  # Swap indices in I directly

                swap_count[k1] +=1


""" enter vertical swaps
We will call swaps between configurations at different t "vertical swaps",
to address those, we need to compute extra 2 energies from stratch

A is an array of matrices, the first dimension indexed by b, it depends on the disorder and on t,
and it is used to build the hamiltonian
"""
@njit(types.UniTuple(float64, 3)(
                int64, int64,                       # number of nodes and patterns
                int8[::1],int8[::1],                # spin configs at the 2 boxes     
                float64[::1],float64[::1],         # magnetizations at the 2 boxes         
                float64, float64,                   # energies of the 2 configs before the swap 
                float64, float64,                   # temperatures of the 2 boxes
                float64[:,::1],float64[:,::1]),      # form of the hamiltonian in the 2 boxes (different t-> different H)


nogil=True, no_cpython_wrapper=True, fastmath = False
)
def vertical_swap_probability_return_proposed_E(N,P, σ_1,σ_2, m_1,m_2, E_1,E_2, β_k1,β_k2, A_b1,A_b2):

    E_1_prop = _recompute_E(N, P, σ_2, m_2, A_b1)
    E_2_prop = _recompute_E(N, P, σ_1, m_1, A_b2)

    exponent = β_k1*E_1 + β_k2*E_2 - (β_k1*E_1_prop + β_k2*E_2_prop)
    
    if exponent > 0 : return                  1.0, E_1_prop, E_2_prop

    else:             return   math.exp(exponent), E_1_prop, E_2_prop



#______________________________ ↓ vertical_swap_proposal_helpers ↓ ______________________________#

@njit(void(int64[::1], uint64[::1]),
      nogil=False, inline='always', no_cpython_wrapper=False)
def shuffle_inplace_int64(order, rng_state):
    for i in range(order.shape[0] - 1, 0, -1):
        j = int64(next_uint_bounded(rng_state, uint64(i + 1)))
        tmp = order[i]
        order[i] = order[j]
        order[j] = tmp


@njit(int64(int64[:, ::1], uint64[::1], int64[::1],
            int32[::1], int32[::1],
            int64[::1]),
      nogil=False, no_cpython_wrapper=False)
def maximal_matching_edge_indices(edge_list, seed, order,
                                  mark, token_arr,
                                  out_e):
    """
    Returns n, with selected edges = out_e[0:n].
    Matching is maximal (greedy), randomized each call.
    """
    E = edge_list.shape[0]

    shuffle_inplace_int64(order, seed)

    # advance token
    token = int32(token_arr[0] + 1)
    if token == 0:
        # overflow wrap: rare; clear marks once
        for v in range(mark.shape[0]):
            mark[v] = int32(0)
        token = int32(1)
    token_arr[0] = token

    n = int64(0)
    for ii in range(E):
        e = order[ii]
        u = edge_list[e, 0]
        v = edge_list[e, 1]
        if mark[u] != token and mark[v] != token:
            mark[u] = token
            mark[v] = token
            out_e[n] = e
            n += 1

    return n

#______________________________ ↑ vertical_swap_proposal_helpers ↑ ______________________________#




@njit(void( int64,int64,                               # N,P
            int8[:,  ::1],                             # σ
            float64[:,::1],                            # m
            float64[::1],                              # E
            int64[::1],                                # I
            float64[:,:,::1],                          # A
            float64[::1],                              # β,
            uint64[:,::1],                             # seed_array
            int64[::1], int64[::1],                    # vertical_swap_count, vertical_swap_attempt
            int64[:, ::1], int64[::1],                 # edges, order
            int32[::1], int32[::1],                    # mark, token_arr
            int64[::1],                                # out_e,
            int64[::1]),                               # b_of_r
nogil=True, no_cpython_wrapper=True
)
def attempt_vertical_swaps(N,P, σ,m,E,I, A, β, seed_array, vertical_swap_count, vertical_swap_attempt,
                          edge_list, order, mark, token_arr, out_e,
                          b_of_r):
    
    n = maximal_matching_edge_indices(edge_list, seed_array[-1], order, mark, token_arr, out_e)

    for i in range (n):
        
        r_1 , r_2 = edge_list[out_e[i],0], edge_list[out_e[i],1]
        b_1 , b_2 = b_of_r[r_1], b_of_r[r_2]

        vertical_swap_attempt[out_e[i]] += 1

        p, E_1_prop, E_2_prop = vertical_swap_probability_return_proposed_E(N,P, 
                                                                            
                            σ[I[r_1]], σ[I[r_2]],
                            m[I[r_1]], m[I[r_2]],
                            E[I[r_1]], E[I[r_2]], 
                            β[  r_1 ], β[  r_2 ], 
                            A[  b_1 ], A[  b_2 ])

        if (p >= 1.0) or (xorshift128p_next_float(seed_array[-1]) < p): #Metropolis
                
                #vertical swap count[that edge index] += 1
                    
                I_r1 = I[r_1]
                I_r2 = I[r_2]
                I[r_1], I[r_2] = I_r2, I_r1  # Swap indices in I directly

                vertical_swap_count[out_e[i]] +=1

                E[I[r_1]] = E_1_prop
                E[I[r_2]] = E_2_prop


#   σ : (R, N)      int8
#   m : (R, P)      float64
#   E : (R,)        float64
#   I : (R,)        int64      box -> index permutation
#   A : (B, N, P)   float64    = ξ @ G^T  (used by Metropolis)
#   ξ : (N, P)      int8
#   d : (B, N,)     float64    diagonal self-term for Metropolis ΔH
#   β : (R,)        float64    inverse temperatures per box

@njit(
    void(
        int64, int64, int64, float64, int64[::1],   # N, P, R, invN, k_start
        int8[:,  ::1],                              # σ
        float64[:,::1],                             # m
        float64[::1],                               # E
        int64[::1],                                 # I
        float64[:,:,::1],                           # A
        int8[:,  ::1],                              # ξ
        float64[:,::1],                             # d
        float64[::1],                               # β
        uint64[:,::1],                              # seed_array
        int64[::1],                                 # swap_count
        int64[::1],int64[::1],                      # vertical_swap_count, vertical_swap_attempt
        int64[:, ::1], int64[::1],                  # edges, order
        int32[::1], int32[::1],                     # mark, token_arr
        int64[::1],                                 # out_e,
        int64[::1]),                                # b_of_r
    nogil=True, no_cpython_wrapper=True, fastmath=True
)
def do_one_MMC_step(
    N, P, R, invN, k_start,
    σ, m, E, I,
    A, ξ, d,
    β, seed_array, swap_count,
    vertical_swap_count, vertical_swap_attempt,
    edge_list, order, mark, token_arr, out_e, 
    b_of_r
):
    """
    One MMC macro-step for ONE PT chain:
      - local update at each slot (Metropolis)
      - even-edge swap pass
      - local update again
      - odd-edge swap pass
      - local update again
      - vertical swaps
    """
    # -------- first local sweep (then even swaps) --------
    for r in range(R):

        E[I[r]] += do_one_Metropolis_sweep_return_ΔE(
                N, P, invN,
                σ[I[r]], m[I[r]],
                A[b_of_r[r]], ξ, d[b_of_r[r]],
                β[r],
                seed_array[r])
        E[I[r]] += do_one_Metropolis_sweep_return_ΔE(
                N, P, invN,
                σ[I[r]], m[I[r]],
                A[b_of_r[r]], ξ, d[b_of_r[r]],
                β[r],
                seed_array[r])

    attempt_swaps(0, R,k_start, E, I, β, seed_array, swap_count)

    # -------- second local sweep (then odd swaps) --------
    for r in range(R):

        E[I[r]] += do_one_Metropolis_sweep_return_ΔE(
                N, P, invN,
                σ[I[r]], m[I[r]],
                A[b_of_r[r]], ξ, d[b_of_r[r]],
                β[r],
                seed_array[r])
        E[I[r]] += do_one_Metropolis_sweep_return_ΔE(
                N, P, invN,
                σ[I[r]], m[I[r]],
                A[b_of_r[r]], ξ, d[b_of_r[r]],
                β[r],
                seed_array[r])

    attempt_swaps(1, R,k_start, E, I, β, seed_array, swap_count)

    # ----- third local sweep (then some vertical swaps) ---
    for r in range(R):

        E[I[r]] += do_one_Metropolis_sweep_return_ΔE(
                N, P, invN,
                σ[I[r]], m[I[r]],
                A[b_of_r[r]], ξ, d[b_of_r[r]],
                β[r],
                seed_array[r])
        E[I[r]] += do_one_Metropolis_sweep_return_ΔE(
                N, P, invN,
                σ[I[r]], m[I[r]],
                A[b_of_r[r]], ξ, d[b_of_r[r]],
                β[r],
                seed_array[r])

    attempt_vertical_swaps(N,P, σ,m,E,I, A, β, seed_array, vertical_swap_count, vertical_swap_attempt,
                          edge_list, order, mark, token_arr, out_e,
                          b_of_r)





@njit(void( int64,int64,int64,float64,                                    # N,P,R,invN       sizes
            int64[::1],                                                   # K 
            int8[:,:,::1],float64[:,:,::1],float64[:,::1],int64[:,::1],   # Σ,M,Ξ,Ψ,         state variables
            float64[:,:,::1],int8[:,::1],float64[:,::1],                  # A,ξ,d            structure
            float64[::1],                                                 # β,                
            uint64[:,:,::1],                                              # seed_matrix,
            int64,int64,int64,                                            # eq_time, sam_interv, N_data,
            float64[:,:,:],float64[:,:,:,:],int64[::1],                   # E_ts,m_ts,μ_to_store  
            float64[:, :], # observables are not C-contiguous             # q01_ts
            int64[:,::1],                                                 # replica_swap_count  state var
            int64[:,::1],int64[:,::1],                                    # replica_vertical_swap_count, replica_vertical_swap_attempt,
            int64[:, ::1]                                                 # edge_list   
           ),                          
nogil=False, no_cpython_wrapper=False
)
def Simulate_two_replicas( N,P,R,invN, K,
                           Σ,M,Ξ,Ψ, 
                           A,ξ,d,
                           β,
                           seed_matrix,
                           equilibration_time, sampling_interval, N_data,
                           E_ts,m_ts,μ_to_store,
                           q01_ts,
                           replica_swap_count, replica_vertical_swap_count, replica_vertical_swap_attempt,
                           edge_list ):

    """
    New args:

        Σ,M,Ξ,Ψ : Array of spins, Mattis overlaps, energies, box indeces for PT, extra dimension for 2 replicas.

        seed_matrix : RNG seed matrix for Xorshift128+, extra dimension for the 2 replicas.

        E_ts, m_ts, q_01_ts are empty arrays of shape (2,N_data,K,shape(obs)) to be filled during the run
    """

    k_start   = np.ascontiguousarray(np.hstack((np.array([0]),np.cumsum(K))))

    b_of_r    = np.empty(R, dtype=np.int64)

    for b in range(K.shape[0]):
        for r in range(k_start[b], k_start[b+1]):  b_of_r[r] = b

    order     = np.arange(len(edge_list), dtype=np.int64)
    mark      = np.zeros(R              , dtype=np.int32)
    token_arr = np.array([1]            , dtype=np.int32)
    out_e     = np.empty(len(edge_list) , dtype=np.int64)

    #THERMALIZE
    for _ in range(equilibration_time):
        do_one_MMC_step(N,P,R,invN,k_start, Σ[0],M[0],Ξ[0],Ψ[0], A,ξ,d, β, seed_matrix[0],
        replica_swap_count[0],replica_vertical_swap_count[0], replica_vertical_swap_attempt[0],
        edge_list,order,mark,token_arr,out_e,b_of_r)

    for _ in range(equilibration_time):
        do_one_MMC_step(N,P,R,invN,k_start, Σ[1],M[1],Ξ[1],Ψ[1], A,ξ,d, β, seed_matrix[1],
        replica_swap_count[1],replica_vertical_swap_count[1], replica_vertical_swap_attempt[1],
        edge_list,order,mark,token_arr,out_e,b_of_r)

    #START SAMPLING
    for n in range(N_data):

        #EVOLVE
        for _ in range(sampling_interval):
            do_one_MMC_step(N,P,R,invN,k_start, Σ[0],M[0],Ξ[0],Ψ[0], A,ξ,d, β, seed_matrix[0],
            replica_swap_count[0],replica_vertical_swap_count[0], replica_vertical_swap_attempt[0],
            edge_list,order,mark,token_arr,out_e,b_of_r)
        
        for _ in range(sampling_interval):
            do_one_MMC_step(N,P,R,invN,k_start, Σ[1],M[1],Ξ[1],Ψ[1], A,ξ,d, β, seed_matrix[1],
            replica_swap_count[1],replica_vertical_swap_count[1], replica_vertical_swap_attempt[1],
            edge_list,order,mark,token_arr,out_e,b_of_r)
        
        #SNAP BACK (prevent drift)
        for α in range(2):
            for r in range(R):
                _recompute_m(N, P, invN, Σ[α, Ψ[α,r]], ξ, M[α, Ψ[α,r]]) 

                Ξ[α, Ψ[α,r]] = _recompute_E(N, P, Σ[α, Ψ[α,r]], M[α, Ψ[α,r]], A[b_of_r[r]])
        
        #FILL THE TIME SERIES
        for r in range(R):
            E_ts[0,n,r] = Ξ[0, Ψ[0,r]]
            E_ts[1,n,r] = Ξ[1, Ψ[1,r]]

            for p in range (μ_to_store.shape[0]):

                μ = μ_to_store[p]

                m_ts[0,n,r,p] = M[0, Ψ[0,r], μ]
                m_ts[1,n,r,p] = M[1, Ψ[1,r], μ]
                           
            # Gauge-fix to align with retrieval pattern ξ_0
            #sgn0 = 1.0 if m_ts[0,n,k,0] >= 0 else -1.0
            #sgn1 = 1.0 if m_ts[1,n,k,0] >= 0 else -1.0

            q_01_res = 0.                                   # Edward-Anderson overlap
            for i in range(N):
                q_01_res += Σ[0, Ψ[0,r], i] * Σ[1, Ψ[1,r], i]

            q01_ts[n,r] = q_01_res / N