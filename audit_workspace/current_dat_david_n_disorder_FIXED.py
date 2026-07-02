# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.integrate import simps
import warnings
warnings.filterwarnings("ignore")

kbT = 0.000000001
def f(E, mu):
    return 1 / (np.exp((E - mu) / kbT) + 1)

def window(E):
    return f(E, 2.0) - f(E, -2.0)

list_N = [10, 19, 28, 37, 46, 55, 64, 73, 82, 91]
Iz, Iz1, Iz2, Iz01 = [], [], [], []

for j in list_N:
    try:
        data = pd.read_csv(f'desorden1000/w05/data_disorder_N{j}.csv')
        Iz.append(simps(data['Gz'] * window(data['E']), data['E']))
    except: Iz.append(np.nan)
    try:
        data1 = pd.read_csv(f'desorden1000/w/data_disorder_N{j}.csv')
        Iz1.append(simps(data1['Gz'] * window(data1['E']), data1['E']))
    except: Iz1.append(np.nan)
    try:
        data2 = pd.read_csv(f'desorden1000/w2/data_disorder_N{j}.csv')
        Iz2.append(simps(data2['Gz'] * window(data2['E']), data2['E']))
    except: Iz2.append(np.nan)
    try:
        data01 = pd.read_csv(f'trans_vs_N/data_N{j}.csv')
        Iz01.append(simps(data01['Gz'] * window(data01['E']), data01['E']))
    except: Iz01.append(np.nan)

np.savetxt('chain_cs_N_disorder_w05_FIXED.dat', np.stack((list_N, Iz), axis=1), fmt='%.6e')
np.savetxt('chain_cs_N_disorder_w_FIXED.dat', np.stack((list_N, Iz1), axis=1), fmt='%.6e')
np.savetxt('chain_cs_N_disorder_w2_FIXED.dat', np.stack((list_N, Iz2), axis=1), fmt='%.6e')
