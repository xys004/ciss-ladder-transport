# -*- coding: utf-8 -*-
"""
================================================================================
current_dat_david_N_disorder_promedio.py
================================================================================

DESCRIPCIÓN GENERAL
-------------------
Este script calcula la CORRIENTE DE ESPÍN en la dirección z (Iz) como función
de la longitud de la cadena N, para diferentes intensidades de desorden
estático de Anderson (W), usando las funciones de transmisión Gz(ω) calculadas
previamente con el código 'trans_cadena_bi_W_SO_gammaout0_GZ_disorder'.

Este código es el análogo exacto del script de dephasing ('current_dat_david_
N_decoherence.py'), pero opera sobre datos generados con DESORDEN ESTÁTICO
en las energías de sitio (E_n aleatorio real) en lugar de dephasing imaginario.

La fórmula física es idéntica en ambos casos:
  Iz(N) = ∫ dω  Gz(ω) × [f_L(ω) - f_R(ω)]

================================================================================
POSICIÓN EN LA CADENA DE CÁLCULO
================================================================================

  PASO 1 (código de desorden): Calcular Gz(ω, N) con promedio de M realizaciones
         de desorden de Anderson E_n ~ Uniforme(-W/2, W/2).
         → archivos CSV: data_disorder_N{j}.csv  (por N, W, y número de realizaciones M)

  PASO 2 (ESTE CÓDIGO): Integrar Gz(ω) × [f_L(ω) - f_R(ω)] sobre ω
         → corriente de espín Iz(N) para cada amplitud de desorden W

================================================================================
DIFERENCIAS CLAVE RESPECTO AL CÓDIGO DE DEPHASING
================================================================================

1. FUENTE DE DATOS: archivos de desorden, no de dephasing
   • Dephasing: 'trans_vs_N_decoherencia1/{w05,w,w2}/data_decoheren_N{j}.csv'
   • Desorden  : 'desorden{M}/{w05,w,w2}/data_disorder_N{j}.csv'
   La organización por carpetas es diferente: aquí las subcarpetas son
   'desorden100/', 'desorden1000/', 'desorden10000/', reflejando el número de
   realizaciones M con que se promedió Gz(ω).

2. TRES BATERÍAS DE REALIZACIONES (M = 100, 1000, 10000) POR CADA W:
   • El código de dephasing cargaba solo UN archivo por W y por N.
   • Este código carga TRES archivos por W y por N, correspondientes a
     M = 100, 1000 y 10000 realizaciones del desorden:
         C,  C3, C6  → W = 0.5 con M = 100, 1000, 10000
         C1, C4, C7  → W = 1.0 con M = 100, 1000, 10000
         C2, C5, C8  → W = 2.0 con M = 100, 1000, 10000
   Esto permite:
     (a) Verificar la convergencia del promedio con M
     (b) Calcular un promedio ponderado entre baterías (ver más abajo)

3. PROMEDIO PONDERADO ENTRE BATERÍAS (actualmente desactivado):
   Las líneas comentadas implementan un promedio ponderado que combina los
   resultados de las tres baterías de realizaciones:
       Iz_ponderado = (100/11100)·C + (1000/11100)·C3 + (10000/11100)·C6
   donde los pesos son proporcionales a M (M=100, M=1000, M=10000), y el
   denominador 11100 = 100 + 1000 + 10000 es la normalización. Este esquema
   da más peso a la batería con mayor M (más precisa estadísticamente).
   En la versión activa del script, solo se usa la batería de M=10000 (C6, C7, C8).

4. TEMPERATURA AÚN MÁS PRÓXIMA A CERO:
   • Dephasing: kbT = 1e-9
   • Desorden:  kbT = 1e-13
   Ambos implementan el límite T → 0, pero aquí la temperatura es 4 órdenes
   de magnitud menor. Esto endurecerá aún más la función escalón de Fermi,
   reduciendo el redondeo numérico de f_L - f_R en los bordes del intervalo
   de integración. Prácticamente el resultado es idéntico en ambos casos.

5. INSTALACIÓN EXPLÍCITA DE VERSIONES FIJADAS DE numpy/scipy:
   El script fija numpy==1.23.5 y scipy==1.10.1 al inicio, garantizando
   reproducibilidad en Google Colab (donde las versiones pueden cambiar).
   Esta práctica es recomendable para resultados numéricos estables.

================================================================================
CONFIGURACIÓN ACTIVA vs COMENTADA
================================================================================

ACTIVA (líneas sin comentar):
  Iz  = C6   → W = 0.5, M = 10000 realizaciones
  Iz1 = C7   → W = 1.0, M = 10000 realizaciones
  Iz2 = C8   → W = 2.0, M = 10000 realizaciones
  Iz01= C01  → W = 0   (sistema coherente, referencia)

COMENTADA (promedio ponderado entre las tres baterías):
  Iz  = (100·C  + 1000·C3 + 10000·C6) / 11100
  Iz1 = (100·C1 + 1000·C4 + 10000·C7) / 11100
  Iz2 = (100·C2 + 1000·C5 + 10000·C8) / 11100

================================================================================
PARÁMETROS FÍSICOS
================================================================================
  N_list     : [10, 19, 28, ..., 91]   Longitudes de cadena evaluadas
  W          : {0.5, 1.0, 2.0}         Amplitud del desorden [meV]
  V (bias)   : i = 4.0 meV             Voltaje de polarización
  μ_L        : muLup/2 + i/2 = 2.5 meV Potencial químico electrodo izquierdo
  kbT        : 1e-13 meV               Temperatura térmica (≈ 0 K)
  M activo   : 10000                   Realizaciones usadas en los datos activos

ESTRUCTURA DEL SCRIPT
-----------------------
  1. Instalación de dependencias con versiones fijadas
  2. Importaciones y parámetros globales
  3. Bucle sobre N: carga de 9+1 archivos CSV y cálculo de 9+1 integrales
  4. Selección del resultado activo (M=10000) o comentado (promedio ponderado)
  5. Guardado en archivos .dat y copia a Google Drive
  6. Gráfica Iz vs N para los cuatro casos (W=0, 0.5, 1.0, 2.0)

AUTOR: (agregar nombre del autor)
FECHA: (agregar fecha)
REFERENCIA: Anderson, P.W., Phys. Rev. 109, 1492 (1958) – localización.
            Landauer, R., IBM J. Res. Dev. 1, 223 (1957) – fórmula de corriente.
            Datta, S., "Electronic Transport in Mesoscopic Systems" (1995).
================================================================================
"""

# ── Fijación de versiones para reproducibilidad en Google Colab ───────────────
# Se instalan versiones específicas de numpy y scipy antes de importarlas.
# Esto es crucial en Colab, donde las versiones por defecto pueden cambiar
# entre sesiones y afectar los resultados numéricos de simps() y spsolve().
!pip install numpy==1.23.5 scipy==1.10.1


# ─── Importaciones ────────────────────────────────────────────────────────────
import pandas as pd                       # Lectura de CSV con datos de Gz(ω)
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve  # No usada aquí (heredada de la plantilla)
from scipy import sparse                  # No usada aquí
from scipy.integrate import simps, trapz  # simps: integración por regla de Simpson
import time                               # No usada aquí
import random                             # No usada aquí
from numba import jit                     # No usada aquí
from google.colab import drive
drive.mount('/content/drive', force_remount=True)   # Montar Google Drive para
                                                    # acceder a los CSV de Gz(ω)

import warnings
warnings.filterwarnings("ignore")


# ─── Parámetros físicos globales ──────────────────────────────────────────────

# Temperatura térmica (unidades de energía, meV)
# kbT = 1e-13 implementa el límite T → 0 con mayor precisión que el código
# de dephasing (donde kbT = 1e-9). La diferencia práctica es insignificante:
# ambos valores hacen que f_L - f_R sea esencialmente una función escalón.
kbT = 0.0000000000001   # ~ 0 K (límite de temperatura cero)

# ── Acumuladores de corriente de espín ───────────────────────────────────────
# Cuatro arrays para almacenar Iz(N) en función del desorden W:
Iz   = np.array([], dtype=float)   # Iz(N) para W = 0.5 meV
Iz1  = np.array([], dtype=float)   # Iz(N) para W = 1.0 meV
Iz2  = np.array([], dtype=float)   # Iz(N) para W = 2.0 meV
Iz01 = np.array([], dtype=float)   # Iz(N) para W = 0   (sin desorden, referencia)

# ── Parámetros del voltaje de polarización ────────────────────────────────────
# El voltaje V = i = 4.0 meV se aplica simétricamente:
#   μ_L_eff = muLup/2 + i/2 = 0.5 + 2.0 = 2.5 meV  (electrodo izquierdo)
#   μ_R_eff = muRup/2 + i/2 = -0.5 + 2.0 = 1.5 meV  (electrodo derecho, ver nota)
#
# NOTA (idéntica al código de dephasing):
# La expresión del electrodo derecho usa +E en el exponente:
#   f_R = 1/(exp((E + (muRup/2 + i/2))/kbT) + 1)
# que coloca el potencial efectivo derecho en -1.5 meV (no -2.0 meV).
# Verificar frente a la convención del artículo de referencia.
i       = 4.0    # Voltaje de polarización V = 4.0 meV
                 # NOTA: renombrar a 'V' o 'bias' evitaría la colisión semántica
                 # con el índice del bucle 'for i in range(M)' de otros scripts

muLup   =  1.0   # Potencial químico electrodo izquierdo, espín ↑ [meV]
muRup   = -1.0   # Potencial químico electrodo derecho,   espín ↑ [meV]
muLdown =  0.0   # Potencial químico electrodo izquierdo, espín ↓ (no usado en la integral)
muRdown =  0.0   # Potencial químico electrodo derecho,   espín ↓ (no usado en la integral)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: BUCLE PRINCIPAL – CARGA DE DATOS Y CÁLCULO DE Iz(N)
# ══════════════════════════════════════════════════════════════════════════════

# Lista de longitudes de cadena evaluadas (número de sitios por cadena).
# Misma familia de N que en el código de dephasing: paso de 9 sitios.
# NOTA: 'list' sobreescribe el built-in de Python → renombrar a 'N_list'.
list = [10, 19, 28, 37, 46, 55, 64, 73, 82, 91]

for j in list:

    # ── Carga de los archivos CSV de Gz(ω) ────────────────────────────────────
    # NOVEDAD respecto al código de dephasing: se cargan TRES baterías de
    # realizaciones (M = 100, 1000, 10000) para cada amplitud de desorden W.
    # La organización de carpetas es:
    #   desorden100/   → datos promediados sobre M = 100  realizaciones
    #   desorden1000/  → datos promediados sobre M = 1000 realizaciones
    #   desorden10000/ → datos promediados sobre M = 10000 realizaciones
    # Dentro de cada carpeta, subcarpetas por amplitud de desorden:
    #   w05/ → W = 0.5 meV
    #   w/   → W = 1.0 meV
    #   w2/  → W = 2.0 meV

    # ── W = 0.5 meV, tres baterías de realizaciones ───────────────────────────
    data  = pd.read_csv(f'drive/MyDrive/Rashba/desorden100/w05/data_disorder_N{j}.csv')    # M = 100
    data3 = pd.read_csv(f'drive/MyDrive/Rashba/desorden1000/w05/data_disorder_N{j}.csv')   # M = 1000
    data6 = pd.read_csv(f'drive/MyDrive/Rashba/desorden10000/w05/data_disorder_N{j}.csv')  # M = 10000

    # ── W = 1.0 meV, tres baterías de realizaciones ───────────────────────────
    data1 = pd.read_csv(f'drive/MyDrive/Rashba/desorden100/w/data_disorder_N{j}.csv')      # M = 100
    data4 = pd.read_csv(f'drive/MyDrive/Rashba/desorden1000/w/data_disorder_N{j}.csv')     # M = 1000
    data7 = pd.read_csv(f'drive/MyDrive/Rashba/desorden10000/w/data_disorder_N{j}.csv')    # M = 10000

    # ── W = 2.0 meV, tres baterías de realizaciones ───────────────────────────
    data2 = pd.read_csv(f'drive/MyDrive/Rashba/desorden100/w2/data_disorder_N{j}.csv')     # M = 100
    data5 = pd.read_csv(f'drive/MyDrive/Rashba/desorden1000/w2/data_disorder_N{j}.csv')    # M = 1000
    data8 = pd.read_csv(f'drive/MyDrive/Rashba/desorden10000/w2/data_disorder_N{j}.csv')   # M = 10000

    # ── W = 0 (sin desorden): caso coherente de referencia ───────────────────
    # Mismo archivo que en el código de dephasing: Gz(ω) calculada sin
    # ningún tipo de desorden ni dephasing (sistema completamente coherente).
    data01 = pd.read_csv(f'drive/MyDrive/Rashba/trans_vs_N/data_N{j}.csv')

    # ── Cálculo de la corriente de espín con la fórmula de Landauer-Büttiker ──
    #
    #   Iz = ∫ dω  Gz(ω) × [f_L(ω) - f_R(ω)]
    #
    # donde:
    #   f_L(ω) = 1 / {exp[(ω - μ_L_eff) / kbT] + 1}   con μ_L_eff = muLup/2 + i/2
    #   f_R(ω) = 1 / {exp[(ω + μ_R_eff) / kbT] + 1}   con μ_R_eff = muRup/2 + i/2
    #
    # La integración numérica usa scipy.integrate.simps (regla de Simpson),
    # más precisa que la regla del trapecio para funciones suaves.
    # El eje de integración es data['E'] (malla de 901 puntos en [-4, 4] meV).
    #
    # En el límite kbT → 0 (aquí kbT = 1e-13), f_L - f_R → función escalón
    # en la ventana de energía efectiva de transmisión.

    # ── W = 0.5 meV ───────────────────────────────────────────────────────────
    C  = simps(data['Gz']  * (1/(np.exp((data['E']  - (muLup/2 + i/2))/kbT) + 1)
                             - 1/(np.exp((data['E']  + (muRup/2 + i/2))/kbT) + 1)), data['E'])   # M = 100

    C3 = simps(data3['Gz'] * (1/(np.exp((data3['E'] - (muLup/2 + i/2))/kbT) + 1)
                             - 1/(np.exp((data3['E'] + (muRup/2 + i/2))/kbT) + 1)), data3['E'])  # M = 1000

    C6 = simps(data6['Gz'] * (1/(np.exp((data6['E'] - (muLup/2 + i/2))/kbT) + 1)
                             - 1/(np.exp((data6['E'] + (muRup/2 + i/2))/kbT) + 1)), data6['E'])  # M = 10000

    # ── W = 1.0 meV ───────────────────────────────────────────────────────────
    C1 = simps(data1['Gz'] * (1/(np.exp((data1['E'] - (muLup/2 + i/2))/kbT) + 1)
                             - 1/(np.exp((data1['E'] + (muRup/2 + i/2))/kbT) + 1)), data1['E'])  # M = 100

    C4 = simps(data4['Gz'] * (1/(np.exp((data4['E'] - (muLup/2 + i/2))/kbT) + 1)
                             - 1/(np.exp((data4['E'] + (muRup/2 + i/2))/kbT) + 1)), data4['E'])  # M = 1000

    C7 = simps(data7['Gz'] * (1/(np.exp((data7['E'] - (muLup/2 + i/2))/kbT) + 1)
                             - 1/(np.exp((data7['E'] + (muRup/2 + i/2))/kbT) + 1)), data7['E'])  # M = 10000

    # ── W = 2.0 meV ───────────────────────────────────────────────────────────
    C2 = simps(data2['Gz'] * (1/(np.exp((data2['E'] - (muLup/2 + i/2))/kbT) + 1)
                             - 1/(np.exp((data2['E'] + (muRup/2 + i/2))/kbT) + 1)), data2['E'])  # M = 100

    C5 = simps(data5['Gz'] * (1/(np.exp((data5['E'] - (muLup/2 + i/2))/kbT) + 1)
                             - 1/(np.exp((data5['E'] + (muRup/2 + i/2))/kbT) + 1)), data5['E'])  # M = 1000

    C8 = simps(data8['Gz'] * (1/(np.exp((data8['E'] - (muLup/2 + i/2))/kbT) + 1)
                             - 1/(np.exp((data8['E'] + (muRup/2 + i/2))/kbT) + 1)), data8['E'])  # M = 10000

    # ── W = 0 (sin desorden, referencia coherente) ────────────────────────────
    C01 = simps(data01['Gz'] * (1/(np.exp((data01['E'] - (muLup/2 + i/2))/kbT) + 1)
                              - 1/(np.exp((data01['E'] + (muRup/2 + i/2))/kbT) + 1)), data01['E'])

    # ── Selección del resultado a acumular ────────────────────────────────────
    #
    # OPCIÓN 1 – PROMEDIO PONDERADO (actualmente COMENTADA):
    # Combina las tres baterías usando pesos proporcionales al número de
    # realizaciones M. El denominador 11100 = 100 + 1000 + 10000 normaliza.
    # Esta opción da más peso estadístico a la batería M=10000 (más precisa)
    # sin descartar la información de las baterías M=100 y M=1000.
    # Útil para un estimador de mínima varianza si las baterías son independientes.
    #
    # Iz   = np.append(Iz,   (100/11100)*C  + (1000/11100)*C3 + (10000/11100)*C6)
    # Iz1  = np.append(Iz1,  (100/11100)*C1 + (1000/11100)*C4 + (10000/11100)*C7)
    # Iz2  = np.append(Iz2,  (100/11100)*C2 + (1000/11100)*C5 + (10000/11100)*C8)
    # Iz01 = np.append(Iz01, C01)
    #
    # OPCIÓN 2 – SOLO BATERÍA M=10000 (actualmente ACTIVA):
    # Usa exclusivamente los resultados con mayor número de realizaciones,
    # que tienen la menor varianza estadística. Es la opción más directa
    # cuando M=10000 ya es suficiente para la convergencia del promedio.
    Iz   = np.append(Iz,   C6)    # W = 0.5, M = 10000
    Iz1  = np.append(Iz1,  C7)    # W = 1.0, M = 10000
    Iz2  = np.append(Iz2,  C8)    # W = 2.0, M = 10000
    Iz01 = np.append(Iz01, C01)   # W = 0   (coherente)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: GUARDADO DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

# Re-declarar la lista de N (redundante pero inofensivo)
list = [10, 19, 28, 37, 46, 55, 64, 73, 82, 91]

# Construir matrices de dos columnas (N, Iz) para cada amplitud de desorden W
A  = np.stack((list, Iz),   axis=1)   # W = 0.5 meV
A1 = np.stack((list, Iz1),  axis=1)   # W = 1.0 meV
A2 = np.stack((list, Iz2),  axis=1)   # W = 2.0 meV
A3 = np.stack((list, Iz01), axis=1)   # W = 0   (sin desorden)

# Guardar en archivos .dat: dos columnas separadas por espacios, 6 cifras decimales
nombre  = 'chain_cs_N_disorder05.dat'         # Iz vs N, W = 0.5 meV
np.savetxt(nombre,  A,  fmt='%.6e')

nombre1 = 'chain_cs_N_disorder1.dat'          # Iz vs N, W = 1.0 meV
np.savetxt(nombre1, A1, fmt='%.6e')

nombre2 = 'chain_cs_N_disorder2.dat'          # Iz vs N, W = 2.0 meV
np.savetxt(nombre2, A2, fmt='%.6e')

nombre3 = 'chain_cs_N_withou_disorder.dat'    # Iz vs N, sin desorden
# NOTA: 'withou' es un typo → corregir a 'without' para consistencia
# con el archivo análogo del código de dephasing ('chain_cs_N_without_decoherence.dat')
np.savetxt(nombre3, A3, fmt='%.6e')

# Líneas comentadas para extensiones futuras (otros valores de W o p):
# nombre4 = 'chain_cs_p_0_d05.dat'
# np.savetxt(nombre4, A4, fmt='%.6e')
# nombre5 = 'chain_cs_p_0_d06.dat'
# np.savetxt(nombre5, A5, fmt='%.6e')

# ── Copia a Google Drive ──────────────────────────────────────────────────────
# Copia todos los archivos 'chain_*' al directorio de gráficas de desorden en Drive.
# NOTA: la carpeta destino es 'disorder/' (diferente a 'decoherence/' del otro script).
!cp chain* drive/MyDrive/Rashba/graficas/disorder/.


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: GRÁFICA – CORRIENTE DE ESPÍN Iz vs LONGITUD DE CADENA N
# ══════════════════════════════════════════════════════════════════════════════
#
# Compara Iz(N) para cuatro amplitudes de desorden W = {0, 0.5, 1.0, 2.0} meV.
# Esta figura es el análogo de la del código de dephasing, pero con desorden
# de Anderson en lugar de decoherencia de Büttiker.
#
# Comportamiento esperado según la teoría de localización de Anderson:
#   W = 0:   corriente coherente pura (referencia); puede oscilar con N
#   W = 0.5: desorden débil → régimen difusivo, Iz ~ 1/N (ley de Ohm de espín)
#   W = 1.0: desorden moderado → inicio de la localización
#   W = 2.0: desorden fuerte  → localización de Anderson; Iz decae exponencialmente
# ──────────────────────────────────────────────────────────────────────────────

direc = "chain_cs_N_disorder.png"    # Nombre del archivo de la figura

plt.figure(figsize=(16, 7))

plt.plot(list, Iz,        label="W=0.5")   # Desorden W = 0.5 meV (M=10000 realizaciones)
plt.plot(list, Iz1,       label="W=1.0")   # Desorden W = 1.0 meV
plt.plot(list, Iz2,       label="W=2.0")   # Desorden W = 2.0 meV
plt.plot(list, Iz01.real, label="W=0")     # Sin desorden (coherente, referencia)
# NOTA: .real se aplica a Iz01 aunque es dtype=float (no complejo).
# Esto es inofensivo pero sugiere que en versiones anteriores Iz01 era complejo.
# En este script Iz01 es float por definición, por lo que .real es redundante.

plt.legend()
plt.xlabel("N")       # Longitud de la cadena (número de sitios)
plt.ylabel("$I_z$")   # Corriente de espín en z (unidades: e/h × meV en u.n.)

# Líneas comentadas para ajuste de escala:
# plt.xlim(-2.0, 2.0)
# plt.xticks(np.arange(-3, 3, 0.5))
