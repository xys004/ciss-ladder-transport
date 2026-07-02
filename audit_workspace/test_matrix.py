import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
from numba import jit

@jit(nopython=True)
def get_matrix_AR(w, N, eta01, eta02, Gamma1, Gamma2, Gamma3, Gamma4, gamma_per, gamma_per1, gamma01, gamma02, l_R1, l_R2, l_D, eta):
    A = np.zeros((8*N, 8*N), dtype=np.complex128)
    Dphi = (2 * np.pi / (10 - 1))
    
    for n in range(1, N+1):
        l = (n-1) % N + 1
        A[n-1,     l-1    ] = -(w - 0.0 + Gamma1[n-1] + eta*1.0j + 1.0j*eta01[n-1])
        A[n-1+N,   l-1+N  ] = -(w - 0.0 + Gamma2[n-1] + eta*1.0j + 1.0j*eta01[n-1])
        A[n-1+2*N, l-1+2*N] = -(w - 0.0 + Gamma3[n-1] + eta*1.0j + 1.0j*eta01[n-1])
        A[n-1+3*N, l-1+3*N] = -(w - 0.0 + Gamma4[n-1] + eta*1.0j + 1.0j*eta01[n-1])
        
        A[n-1+4*N, l-1+4*N] = -(w - 0.0 + Gamma1[n-1] + eta*1.0j + 1.0j*eta02[n-1])
        A[n-1+5*N, l-1+5*N] = -(w - 0.0 + Gamma2[n-1] + eta*1.0j + 1.0j*eta02[n-1])
        A[n-1+6*N, l-1+6*N] = -(w - 0.0 + Gamma3[n-1] + eta*1.0j + 1.0j*eta02[n-1])
        A[n-1+7*N, l-1+7*N] = -(w - 0.0 + Gamma4[n-1] + eta*1.0j + 1.0j*eta02[n-1])

        A[n-1,     l-1+4*N] = gamma_per
        A[n-1+N,   l-1+5*N] = gamma_per
        A[n-1+2*N, l-1+6*N] = gamma_per
        A[n-1+3*N, l-1+7*N] = gamma_per

        A[n-1+4*N, l-1    ] = gamma_per
        A[n-1+5*N, l-1+N  ] = gamma_per
        A[n-1+6*N, l-1+2*N] = gamma_per
        A[n-1+7*N, l-1+3*N] = gamma_per

        A[n-1,     l-1+6*N] = gamma_per1
        A[n-1+N,   l-1+7*N] = gamma_per1
        A[n-1+2*N, l-1+4*N] = gamma_per1
        A[n-1+3*N, l-1+5*N] = gamma_per1

        A[n-1+4*N, l-1+2*N] = gamma_per1
        A[n-1+5*N, l-1+3*N] = gamma_per1
        A[n-1+6*N, l-1    ] = gamma_per1
        A[n-1+7*N, l-1+N  ] = gamma_per1

        l_next = n % N + 1
        pp = (n-1) * Dphi
        tt1 = gamma01 if n < N else 0.0
        tt2 = gamma02 if n < N else 0.0

        A[n-1,     l_next-1    ] += tt1
        A[n-1+4*N, l_next-1+4*N] += tt2
        A[n-1+N,   l_next-1+N  ] += tt1
        A[n-1+5*N, l_next-1+5*N] += tt2
        A[n-1+2*N, l_next-1+2*N] += tt1
        A[n-1+6*N, l_next-1+6*N] += tt2
        A[n-1+3*N, l_next-1+3*N] += tt1
        A[n-1+7*N, l_next-1+7*N] += tt2

        l_R11 = l_R1 if n < N else 0.0
        l_R12 = l_R2 if n < N else 0.0

        A[n-1,     l_next-1+2*N] = (-1j*l_R11*np.exp(-pp*1j) + l_D*np.exp( pp*1j))
        A[n-1+N,   l_next-1+3*N] = (-1j*l_R11*np.exp( pp*1j) - l_D*np.exp(-pp*1j))
        A[n-1+2*N, l_next-1    ] = (-1j*l_R11*np.exp( pp*1j) - l_D*np.exp(-pp*1j))
        A[n-1+3*N, l_next-1+N  ] = (-1j*l_R11*np.exp(-pp*1j) + l_D*np.exp( pp*1j))

        beta = np.pi
        A[n-1+4*N, l_next-1+6*N] = (-1j*l_R12*np.exp(-pp*1j - beta*1j) + l_D*np.exp( pp*1j))
        A[n-1+5*N, l_next-1+7*N] = (-1j*l_R12*np.exp( pp*1j + beta*1j) - l_D*np.exp(-pp*1j))
        A[n-1+6*N, l_next-1+4*N] = (-1j*l_R12*np.exp( pp*1j + beta*1j) - l_D*np.exp(-pp*1j))
        A[n-1+7*N, l_next-1+5*N] = (-1j*l_R12*np.exp(-pp*1j - beta*1j) + l_D*np.exp( pp*1j))

        l_prev = (n-2) % N + 1
        pp_prev = (n-2) * Dphi

        tt1_prev = gamma01 if n > 1 else 0.0
        tt2_prev = gamma02 if n > 1 else 0.0

        A[n-1,     l_prev-1    ] += tt1_prev
        A[n-1+4*N, l_prev-1+4*N] += tt2_prev
        A[n-1+N,   l_prev-1+N  ] += tt1_prev
        A[n-1+5*N, l_prev-1+5*N] += tt2_prev
        A[n-1+2*N, l_prev-1+2*N] += tt1_prev
        A[n-1+6*N, l_prev-1+6*N] += tt2_prev
        A[n-1+3*N, l_prev-1+3*N] += tt1_prev
        A[n-1+7*N, l_prev-1+7*N] += tt2_prev

        l_R111 = l_R1 if n > 1 else 0.0
        l_R122 = l_R2 if n > 1 else 0.0

        A[n-1,     l_prev-1+2*N] += ( 1j*l_R111*np.exp(-pp_prev*1j) - l_D*np.exp( pp_prev*1j))
        A[n-1+N,   l_prev-1+3*N] += ( 1j*l_R111*np.exp( pp_prev*1j) + l_D*np.exp(-pp_prev*1j))
        A[n-1+2*N, l_prev-1    ] += ( 1j*l_R111*np.exp( pp_prev*1j) + l_D*np.exp(-pp_prev*1j))
        A[n-1+3*N, l_prev-1+N  ] += ( 1j*l_R111*np.exp(-pp_prev*1j) - l_D*np.exp( pp_prev*1j))

        A[n-1+4*N, l_prev-1+6*N] += ( 1j*l_R122*np.exp(-pp_prev*1j - beta*1j) - l_D*np.exp( pp_prev*1j))
        A[n-1+5*N, l_prev-1+7*N] += ( 1j*l_R122*np.exp( pp_prev*1j + beta*1j) + l_D*np.exp(-pp_prev*1j))
        A[n-1+6*N, l_prev-1+4*N] += ( 1j*l_R122*np.exp( pp_prev*1j + beta*1j) + l_D*np.exp(-pp_prev*1j))
        A[n-1+7*N, l_prev-1+5*N] += ( 1j*l_R122*np.exp(-pp_prev*1j - beta*1j) - l_D*np.exp( pp_prev*1j))

    return A

def build_base_sparse_matrix(N, eta01, eta02, Gamma1, Gamma2, Gamma3, Gamma4, gamma_per, gamma_per1, gamma01, gamma02, l_R1, l_R2, l_D, eta):
    rows = []
    cols = []
    data = []
    
    n = np.arange(N)
    
    # Diagonals
    for b in range(8):
        rows.append(n + b*N)
        cols.append(n + b*N)
        
    d0 = -(0.0 + Gamma1 + 1j*eta + 1j*eta01)
    d1 = -(0.0 + Gamma2 + 1j*eta + 1j*eta01)
    d2 = -(0.0 + Gamma3 + 1j*eta + 1j*eta01)
    d3 = -(0.0 + Gamma4 + 1j*eta + 1j*eta01)
    
    d4 = -(0.0 + Gamma1 + 1j*eta + 1j*eta02)
    d5 = -(0.0 + Gamma2 + 1j*eta + 1j*eta02)
    d6 = -(0.0 + Gamma3 + 1j*eta + 1j*eta02)
    d7 = -(0.0 + Gamma4 + 1j*eta + 1j*eta02)
    
    data.extend([d0, d1, d2, d3, d4, d5, d6, d7])
    
    # gamma_per
    if gamma_per != 0:
        for b1, b2 in [(0,4), (1,5), (2,6), (3,7), (4,0), (5,1), (6,2), (7,3)]:
            rows.append(n + b1*N)
            cols.append(n + b2*N)
            data.append(np.full(N, gamma_per, dtype=np.complex128))
            
    # gamma_per1
    if gamma_per1 != 0:
        for b1, b2 in [(0,6), (1,7), (2,4), (3,5), (4,2), (5,3), (6,0), (7,1)]:
            rows.append(n + b1*N)
            cols.append(n + b2*N)
            data.append(np.full(N, gamma_per1, dtype=np.complex128))
            
    if N > 1:
        n_off = np.arange(N-1)
        pp = n_off * (2 * np.pi / 9.0)
        
        # Next (n to n+1)
        rows_next = []
        cols_next = []
        data_next = []
        
        for b in range(4):
            rows_next.append(n_off + b*N)
            cols_next.append(n_off + 1 + b*N)
            data_next.append(np.full(N-1, gamma01, dtype=np.complex128))
            
            rows_next.append(n_off + (b+4)*N)
            cols_next.append(n_off + 1 + (b+4)*N)
            data_next.append(np.full(N-1, gamma02, dtype=np.complex128))
            
        rows_next.append(n_off)
        cols_next.append(n_off + 1 + 2*N)
        data_next.append(-1j*l_R1*np.exp(-1j*pp) + l_D*np.exp(1j*pp))
        
        rows_next.append(n_off + N)
        cols_next.append(n_off + 1 + 3*N)
        data_next.append(-1j*l_R1*np.exp(1j*pp) - l_D*np.exp(-1j*pp))
        
        rows_next.append(n_off + 2*N)
        cols_next.append(n_off + 1)
        data_next.append(-1j*l_R1*np.exp(1j*pp) - l_D*np.exp(-1j*pp))
        
        rows_next.append(n_off + 3*N)
        cols_next.append(n_off + 1 + N)
        data_next.append(-1j*l_R1*np.exp(-1j*pp) + l_D*np.exp(1j*pp))
        
        beta = np.pi
        
        rows_next.append(n_off + 4*N)
        cols_next.append(n_off + 1 + 6*N)
        data_next.append(-1j*l_R2*np.exp(-1j*(pp + beta)) + l_D*np.exp(1j*pp))
        
        rows_next.append(n_off + 5*N)
        cols_next.append(n_off + 1 + 7*N)
        data_next.append(-1j*l_R2*np.exp(1j*(pp + beta)) - l_D*np.exp(-1j*pp))
        
        rows_next.append(n_off + 6*N)
        cols_next.append(n_off + 1 + 4*N)
        data_next.append(-1j*l_R2*np.exp(1j*(pp + beta)) - l_D*np.exp(-1j*pp))
        
        rows_next.append(n_off + 7*N)
        cols_next.append(n_off + 1 + 5*N)
        data_next.append(-1j*l_R2*np.exp(-1j*(pp + beta)) + l_D*np.exp(1j*pp))
        
        # Prev (n+1 to n)
        rows_prev = []
        cols_prev = []
        data_prev = []
        
        for b in range(4):
            rows_prev.append(n_off + 1 + b*N)
            cols_prev.append(n_off + b*N)
            data_prev.append(np.full(N-1, gamma01, dtype=np.complex128))
            
            rows_prev.append(n_off + 1 + (b+4)*N)
            cols_prev.append(n_off + (b+4)*N)
            data_prev.append(np.full(N-1, gamma02, dtype=np.complex128))
            
        rows_prev.append(n_off + 1)
        cols_prev.append(n_off + 2*N)
        data_prev.append(1j*l_R1*np.exp(-1j*pp) - l_D*np.exp(1j*pp))
        
        rows_prev.append(n_off + 1 + N)
        cols_prev.append(n_off + 3*N)
        data_prev.append(1j*l_R1*np.exp(1j*pp) + l_D*np.exp(-1j*pp))
        
        rows_prev.append(n_off + 1 + 2*N)
        cols_prev.append(n_off)
        data_prev.append(1j*l_R1*np.exp(1j*pp) + l_D*np.exp(-1j*pp))
        
        rows_prev.append(n_off + 1 + 3*N)
        cols_prev.append(n_off + N)
        data_prev.append(1j*l_R1*np.exp(-1j*pp) - l_D*np.exp(1j*pp))
        
        rows_prev.append(n_off + 1 + 4*N)
        cols_prev.append(n_off + 6*N)
        data_prev.append(1j*l_R2*np.exp(-1j*(pp + beta)) - l_D*np.exp(1j*pp))
        
        rows_prev.append(n_off + 1 + 5*N)
        cols_prev.append(n_off + 7*N)
        data_prev.append(1j*l_R2*np.exp(1j*(pp + beta)) + l_D*np.exp(-1j*pp))
        
        rows_prev.append(n_off + 1 + 6*N)
        cols_prev.append(n_off + 4*N)
        data_prev.append(1j*l_R2*np.exp(1j*(pp + beta)) + l_D*np.exp(-1j*pp))
        
        rows_prev.append(n_off + 1 + 7*N)
        cols_prev.append(n_off + 5*N)
        data_prev.append(1j*l_R2*np.exp(-1j*(pp + beta)) - l_D*np.exp(1j*pp))
        
        rows.extend(rows_next + rows_prev)
        cols.extend(cols_next + cols_prev)
        data.extend(data_next + data_prev)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    
    # We should sum duplicate entries instead of keeping them separate for testing array diff
    sparse_mat = sparse.coo_matrix((data, (rows, cols)), shape=(8*N, 8*N))
    sparse_mat.sum_duplicates()
    return sparse_mat

# Test
N = 10
np.random.seed(42)
eta01 = np.random.rand(N)
eta02 = np.random.rand(N)
Gamma1 = np.random.rand(N) + 1j*np.random.rand(N)
Gamma2 = np.random.rand(N) + 1j*np.random.rand(N)
Gamma3 = np.random.rand(N) + 1j*np.random.rand(N)
Gamma4 = np.random.rand(N) + 1j*np.random.rand(N)
gamma_per = 0.5
gamma_per1 = 0.2
gamma01 = 1.0
gamma02 = 1.0
l_R1 = 0.1
l_R2 = 0.1
l_D = 0.05
eta = 0.001

A_dense = get_matrix_AR(0.0, N, eta01, eta02, Gamma1, Gamma2, Gamma3, Gamma4, gamma_per, gamma_per1, gamma01, gamma02, l_R1, l_R2, l_D, eta)
A_sparse = build_base_sparse_matrix(N, eta01, eta02, Gamma1, Gamma2, Gamma3, Gamma4, gamma_per, gamma_per1, gamma01, gamma02, l_R1, l_R2, l_D, eta).toarray()

diff = np.abs(A_dense - A_sparse)
max_diff = np.max(diff)
print("Max difference:", max_diff)

if max_diff > 1e-10:
    idx = np.where(diff == max_diff)
    print("Mismatched indices:", list(zip(idx[0], idx[1])))
    print("Dense val:", A_dense[idx[0][0], idx[1][0]])
    print("Sparse val:", A_sparse[idx[0][0], idx[1][0]])
