import numpy as np
import pandas as pd
from scipy.integrate import simpson
import warnings

# Suppress RuntimeWarnings from exp overflow to keep output clean
warnings.filterwarnings('ignore', category=RuntimeWarning)

def fermi_old_L(E, i_bias=4.0, muLup=1.0, kbT=1e-9):
    return 1 / (np.exp((E - (muLup/2 + i_bias/2))/kbT) + 1)

def fermi_old_R(E, i_bias=4.0, muRup=-1.0, kbT=1e-9):
    return 1 / (np.exp((E + (muRup/2 + i_bias/2))/kbT) + 1)

def window_old(E):
    return fermi_old_L(E) - fermi_old_R(E)

def window_new(E):
    muL = 2.0
    muR = -2.0
    kbT = 1e-9
    def f(E, mu):
        return 1 / (np.exp(np.clip((E - mu)/kbT, -700, 700)) + 1)
    return f(E, muL) - f(E, muR)

data_rows = []

Ns = [10, 37, 91]
for N in Ns:
    # Coherent
    df_coh = pd.read_csv(f"audit_workspace/trans_vs_N/data_N{N}.csv")
    E_coh = df_coh['E'].values
    Gz_coh = df_coh['Gz'].values
    
    Icoh_old = simpson(Gz_coh * window_old(E_coh), x=E_coh)
    Icoh_new = simpson(Gz_coh * window_new(E_coh), x=E_coh)
    
    # Dephasing (w0.5 means eta_d=0.5)
    df_deph = pd.read_csv(f"audit_workspace/trans_vs_N_decoherencia1/w05/data_decoheren_N{N}.csv")
    E_deph = df_deph['E'].values
    Gz_deph = df_deph['Gz'].values
    
    Ideph_new = simpson(Gz_deph * window_new(E_deph), x=E_deph)
    
    data_rows.append({
        'N': N,
        'Type': 'Coherent_Dephasing',
        'Icoh_old': Icoh_old,
        'Icoh_new': Icoh_new,
        'Ideph_new': Ideph_new,
        'max_Gz': np.max(Gz_coh)
    })

# FASE 4 calculations: controls
for control in ['gamma0', 'lambda0']:
    df_ctrl = pd.read_csv(f"audit_workspace/controls/data_{control}_N10.csv")
    E_ctrl = df_ctrl['E'].values
    Gz_ctrl = df_ctrl['Gz'].values
    
    Iz_new = simpson(Gz_ctrl * window_new(E_ctrl), x=E_ctrl)
    max_Gz = np.max(Gz_ctrl)
    
    data_rows.append({
        'N': 10,
        'Type': f'Control_{control}',
        'Icoh_old': None,
        'Icoh_new': Iz_new,
        'Ideph_new': None,
        'max_Gz': max_Gz
    })

df_res = pd.DataFrame(data_rows)
df_res.to_csv("audit_workspace/summary_baseline_check.csv", index=False)
print(df_res.to_string())
