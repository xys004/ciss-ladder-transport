import os
import pandas as pd
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
import time
from numba import jit

@jit(nopython=True)
def get_coo_arrays(w, N, eta01, eta02, Gamma1, Gamma2, Gamma3, Gamma4, gamma_per, gamma_per1, gamma01, gamma02, l_R1, l_R2, l_D, eta):
    # Maximum possible non-zero elements
    # Diagonals: 8*N
    # gamma_per: 8*N
    # gamma_per1: 8*N
    # Hopping (next and prev): 8*(N-1) * 2 = 16*(N-1)
    # SOI (next and prev): 8*(N-1) * 2 = 16*(N-1)
    # Total ~ 24*N + 32*(N-1)
    max_elements = 24*N + 32*N
    
    rows = np.zeros(max_elements, dtype=np.int32)
    cols = np.zeros(max_elements, dtype=np.int32)
    data = np.zeros(max_elements, dtype=np.complex128)
    
    idx = 0
    Dphi = (2 * np.pi / (10 - 1))
    
    for n in range(1, N+1):
        l = (n-1) % N + 1
        
        # Diagonals
        rows[idx] = n-1; cols[idx] = l-1; data[idx] = -(w - 0.0 + Gamma1[n-1] + eta*1.0j + 1.0j*eta01[n-1]); idx+=1
        rows[idx] = n-1+N; cols[idx] = l-1+N; data[idx] = -(w - 0.0 + Gamma2[n-1] + eta*1.0j + 1.0j*eta01[n-1]); idx+=1
        rows[idx] = n-1+2*N; cols[idx] = l-1+2*N; data[idx] = -(w - 0.0 + Gamma3[n-1] + eta*1.0j + 1.0j*eta01[n-1]); idx+=1
        rows[idx] = n-1+3*N; cols[idx] = l-1+3*N; data[idx] = -(w - 0.0 + Gamma4[n-1] + eta*1.0j + 1.0j*eta01[n-1]); idx+=1
        
        rows[idx] = n-1+4*N; cols[idx] = l-1+4*N; data[idx] = -(w - 0.0 + Gamma1[n-1] + eta*1.0j + 1.0j*eta02[n-1]); idx+=1
        rows[idx] = n-1+5*N; cols[idx] = l-1+5*N; data[idx] = -(w - 0.0 + Gamma2[n-1] + eta*1.0j + 1.0j*eta02[n-1]); idx+=1
        rows[idx] = n-1+6*N; cols[idx] = l-1+6*N; data[idx] = -(w - 0.0 + Gamma3[n-1] + eta*1.0j + 1.0j*eta02[n-1]); idx+=1
        rows[idx] = n-1+7*N; cols[idx] = l-1+7*N; data[idx] = -(w - 0.0 + Gamma4[n-1] + eta*1.0j + 1.0j*eta02[n-1]); idx+=1

        # gamma_per
        rows[idx] = n-1; cols[idx] = l-1+4*N; data[idx] = gamma_per; idx+=1
        rows[idx] = n-1+N; cols[idx] = l-1+5*N; data[idx] = gamma_per; idx+=1
        rows[idx] = n-1+2*N; cols[idx] = l-1+6*N; data[idx] = gamma_per; idx+=1
        rows[idx] = n-1+3*N; cols[idx] = l-1+7*N; data[idx] = gamma_per; idx+=1
        rows[idx] = n-1+4*N; cols[idx] = l-1; data[idx] = gamma_per; idx+=1
        rows[idx] = n-1+5*N; cols[idx] = l-1+N; data[idx] = gamma_per; idx+=1
        rows[idx] = n-1+6*N; cols[idx] = l-1+2*N; data[idx] = gamma_per; idx+=1
        rows[idx] = n-1+7*N; cols[idx] = l-1+3*N; data[idx] = gamma_per; idx+=1

        # gamma_per1
        rows[idx] = n-1; cols[idx] = l-1+6*N; data[idx] = gamma_per1; idx+=1
        rows[idx] = n-1+N; cols[idx] = l-1+7*N; data[idx] = gamma_per1; idx+=1
        rows[idx] = n-1+2*N; cols[idx] = l-1+4*N; data[idx] = gamma_per1; idx+=1
        rows[idx] = n-1+3*N; cols[idx] = l-1+5*N; data[idx] = gamma_per1; idx+=1
        rows[idx] = n-1+4*N; cols[idx] = l-1+2*N; data[idx] = gamma_per1; idx+=1
        rows[idx] = n-1+5*N; cols[idx] = l-1+3*N; data[idx] = gamma_per1; idx+=1
        rows[idx] = n-1+6*N; cols[idx] = l-1; data[idx] = gamma_per1; idx+=1
        rows[idx] = n-1+7*N; cols[idx] = l-1+N; data[idx] = gamma_per1; idx+=1

        # Next
        l_next = n % N + 1
        pp = (n-1) * Dphi
        if n < N:
            rows[idx] = n-1; cols[idx] = l_next-1; data[idx] = gamma01; idx+=1
            rows[idx] = n-1+4*N; cols[idx] = l_next-1+4*N; data[idx] = gamma02; idx+=1
            rows[idx] = n-1+N; cols[idx] = l_next-1+N; data[idx] = gamma01; idx+=1
            rows[idx] = n-1+5*N; cols[idx] = l_next-1+5*N; data[idx] = gamma02; idx+=1
            rows[idx] = n-1+2*N; cols[idx] = l_next-1+2*N; data[idx] = gamma01; idx+=1
            rows[idx] = n-1+6*N; cols[idx] = l_next-1+6*N; data[idx] = gamma02; idx+=1
            rows[idx] = n-1+3*N; cols[idx] = l_next-1+3*N; data[idx] = gamma01; idx+=1
            rows[idx] = n-1+7*N; cols[idx] = l_next-1+7*N; data[idx] = gamma02; idx+=1

            rows[idx] = n-1; cols[idx] = l_next-1+2*N; data[idx] = (-1j*l_R1*np.exp(-pp*1j) + l_D*np.exp( pp*1j)); idx+=1
            rows[idx] = n-1+N; cols[idx] = l_next-1+3*N; data[idx] = (-1j*l_R1*np.exp( pp*1j) - l_D*np.exp(-pp*1j)); idx+=1
            rows[idx] = n-1+2*N; cols[idx] = l_next-1; data[idx] = (-1j*l_R1*np.exp( pp*1j) - l_D*np.exp(-pp*1j)); idx+=1
            rows[idx] = n-1+3*N; cols[idx] = l_next-1+N; data[idx] = (-1j*l_R1*np.exp(-pp*1j) + l_D*np.exp( pp*1j)); idx+=1

            beta = np.pi
            rows[idx] = n-1+4*N; cols[idx] = l_next-1+6*N; data[idx] = (-1j*l_R2*np.exp(-pp*1j - beta*1j) + l_D*np.exp( pp*1j)); idx+=1
            rows[idx] = n-1+5*N; cols[idx] = l_next-1+7*N; data[idx] = (-1j*l_R2*np.exp( pp*1j + beta*1j) - l_D*np.exp(-pp*1j)); idx+=1
            rows[idx] = n-1+6*N; cols[idx] = l_next-1+4*N; data[idx] = (-1j*l_R2*np.exp( pp*1j + beta*1j) - l_D*np.exp(-pp*1j)); idx+=1
            rows[idx] = n-1+7*N; cols[idx] = l_next-1+5*N; data[idx] = (-1j*l_R2*np.exp(-pp*1j - beta*1j) + l_D*np.exp( pp*1j)); idx+=1

        # Prev
        l_prev = (n-2) % N + 1
        pp_prev = (n-2) * Dphi
        if n > 1:
            rows[idx] = n-1; cols[idx] = l_prev-1; data[idx] = gamma01; idx+=1
            rows[idx] = n-1+4*N; cols[idx] = l_prev-1+4*N; data[idx] = gamma02; idx+=1
            rows[idx] = n-1+N; cols[idx] = l_prev-1+N; data[idx] = gamma01; idx+=1
            rows[idx] = n-1+5*N; cols[idx] = l_prev-1+5*N; data[idx] = gamma02; idx+=1
            rows[idx] = n-1+2*N; cols[idx] = l_prev-1+2*N; data[idx] = gamma01; idx+=1
            rows[idx] = n-1+6*N; cols[idx] = l_prev-1+6*N; data[idx] = gamma02; idx+=1
            rows[idx] = n-1+3*N; cols[idx] = l_prev-1+3*N; data[idx] = gamma01; idx+=1
            rows[idx] = n-1+7*N; cols[idx] = l_prev-1+7*N; data[idx] = gamma02; idx+=1

            rows[idx] = n-1; cols[idx] = l_prev-1+2*N; data[idx] = ( 1j*l_R1*np.exp(-pp_prev*1j) - l_D*np.exp( pp_prev*1j)); idx+=1
            rows[idx] = n-1+N; cols[idx] = l_prev-1+3*N; data[idx] = ( 1j*l_R1*np.exp( pp_prev*1j) + l_D*np.exp(-pp_prev*1j)); idx+=1
            rows[idx] = n-1+2*N; cols[idx] = l_prev-1; data[idx] = ( 1j*l_R1*np.exp( pp_prev*1j) + l_D*np.exp(-pp_prev*1j)); idx+=1
            rows[idx] = n-1+3*N; cols[idx] = l_prev-1+N; data[idx] = ( 1j*l_R1*np.exp(-pp_prev*1j) - l_D*np.exp( pp_prev*1j)); idx+=1

            beta = np.pi
            rows[idx] = n-1+4*N; cols[idx] = l_prev-1+6*N; data[idx] = ( 1j*l_R2*np.exp(-pp_prev*1j - beta*1j) - l_D*np.exp( pp_prev*1j)); idx+=1
            rows[idx] = n-1+5*N; cols[idx] = l_prev-1+7*N; data[idx] = ( 1j*l_R2*np.exp( pp_prev*1j + beta*1j) + l_D*np.exp(-pp_prev*1j)); idx+=1
            rows[idx] = n-1+6*N; cols[idx] = l_prev-1+4*N; data[idx] = ( 1j*l_R2*np.exp( pp_prev*1j + beta*1j) + l_D*np.exp(-pp_prev*1j)); idx+=1
            rows[idx] = n-1+7*N; cols[idx] = l_prev-1+5*N; data[idx] = ( 1j*l_R2*np.exp(-pp_prev*1j - beta*1j) - l_D*np.exp( pp_prev*1j)); idx+=1

    return rows[:idx], cols[:idx], data[:idx]

def get_transmission(w, N, eta01, eta02, Gamma1, Gamma2, Gamma3, Gamma4, gamma_per, gamma_per1, gamma01, gamma02, l_R1, l_R2, l_D, eta, B):
    rows, cols, data = get_coo_arrays(w, N, eta01, eta02, Gamma1, Gamma2, Gamma3, Gamma4, gamma_per, gamma_per1, gamma01, gamma02, l_R1, l_R2, l_D, eta)
    AR1 = sparse.coo_matrix((data, (rows, cols)), shape=(8*N, 8*N)).tocsr()
    
    # We sum_duplicates because some elements like diagonals could theoretically be written multiple times (though shouldn't in this case, but good practice).
    # .tocsr() sums duplicates automatically.
    
    GR1 = np.asarray(spsolve(AR1, B)).ravel()
    
    T = np.zeros(8, dtype=complex)
    T[0] = GR1[N-1]   # 1u
    T[1] = GR1[2*N-1] # 1d
    T[2] = GR1[3*N-1] # 1ud
    T[3] = GR1[4*N-1] # 1du
    T[4] = GR1[5*N-1] # 2u
    T[5] = GR1[6*N-1] # 2d
    T[6] = GR1[7*N-1] # 2ud
    T[7] = GR1[8*N-1] # 2du
    return T

def run_simulation(N, W, M, gamma_out_param, lambda_soc_param, output_path):
    print(f"Running simulation for N={N}, W={W}, M={M}, gamma_out={gamma_out_param}, lambda_soc={lambda_soc_param}")
    start_time = time.time()
    
    gamma01 = 1.0
    gamma02 = 1.0
    l_R1 = lambda_soc_param
    l_R2 = lambda_soc_param
    l_D = 0.0
    eta = 0.00001
    
    p = 0.0
    val = 1.0*(1+p)*1j
    val1 = 1.0*(1-p)*1j
    Gamma1 = np.array([val,  *np.zeros(N-2), val ], dtype=np.complex128)
    Gamma2 = np.array([val,  *np.zeros(N-2), val1], dtype=np.complex128)
    Gamma3 = np.array([val1, *np.zeros(N-2), val ], dtype=np.complex128)
    Gamma4 = np.array([val1, *np.zeros(N-2), val1], dtype=np.complex128)

    B = np.zeros(8*N, dtype=np.complex128)
    B[0] = -1; B[N] = -1; B[4*N] = -1; B[5*N] = -1

    def generate_eta():
        if W == 0.0:
            return np.zeros(N, dtype=np.float64), np.zeros(N, dtype=np.float64)
        eta01 = np.random.rand(N) - 0.5
        eta01 *= W
        eta01[-1] -= np.sum(eta01)
        eta02 = np.random.rand(N) - 0.5
        eta02 *= W
        eta02[-1] -= np.sum(eta02)
        return eta01, eta02

    w_array = np.linspace(-4, 4, 901)
    etas = [generate_eta() for _ in range(M)]
    Gz_res = np.zeros(len(w_array))

    for idx_w, w in enumerate(w_array):
        # Group 1
        gamma_per = 0.0
        gamma_per1 = gamma_out_param
        T_realizations_g1 = np.zeros((M, 8), dtype=complex)
        for i in range(M):
            T_realizations_g1[i] = get_transmission(w, N, etas[i][0], etas[i][1], Gamma1, Gamma2, Gamma3, Gamma4, gamma_per, gamma_per1, gamma01, gamma02, l_R1, l_R2, l_D, eta, B)
        T_mean_g1 = np.mean(T_realizations_g1, axis=0)
        
        # Group 2
        gamma_per = 0.0
        gamma_per1 = -gamma_out_param
        T_realizations_g2 = np.zeros((M, 8), dtype=complex)
        for i in range(M):
            T_realizations_g2[i] = get_transmission(w, N, etas[i][0], etas[i][1], Gamma1, Gamma2, Gamma3, Gamma4, gamma_per, gamma_per1, gamma01, gamma02, l_R1, l_R2, l_D, eta, B)
        T_mean_g2 = np.mean(T_realizations_g2, axis=0)

        Trans1u = T_mean_g1[0]
        Trans1d = T_mean_g1[1]
        Trans1ud = T_mean_g1[2]
        Trans1du = T_mean_g1[3]
        
        Trans2u = T_mean_g2[4]
        Trans2d = T_mean_g2[5]
        Trans2ud = T_mean_g2[6]
        Trans2du = T_mean_g2[7]
        
        Gz = (np.abs(Trans1u)**2 - np.abs(Trans1du)**2 + np.abs(Trans1ud)**2 - np.abs(Trans1d)**2 +
              np.abs(Trans2u)**2 - np.abs(Trans2du)**2 + np.abs(Trans2ud)**2 - np.abs(Trans2d)**2)
              
        Gz_res[idx_w] = Gz

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame({'E': w_array, 'Gz': Gz_res})
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}. Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    # a) Coherent case: W = 0.0, M = 1
    for N in [10, 37, 91]:
        run_simulation(N, 0.0, 1, 1.0, 0.1, f"audit_workspace/trans_vs_N/data_N{N}.csv")

    # b) Dephasing case: W = 0.5, M = 50
    for N in [10, 37, 91]:
        run_simulation(N, 0.5, 50, 1.0, 0.1, f"audit_workspace/trans_vs_N_decoherencia1/w05/data_decoheren_N{N}.csv")
        
    # c) Control 1 (N=10 only, coherent): gamma_out = 0.0, lambda_soc = 0.1
    run_simulation(10, 0.0, 1, 0.0, 0.1, "audit_workspace/controls/data_gamma0_N10.csv")
    
    # d) Control 2 (N=10 only, coherent): gamma_out = 1.0, lambda_soc = 0.0
    run_simulation(10, 0.0, 1, 1.0, 0.0, "audit_workspace/controls/data_lambda0_N10.csv")
