# -*- coding: utf-8 -*-
"""
================================================================================
current_dat_david_N_decoherence.py
================================================================================

DESCRIPCIÓN GENERAL
-------------------
Este script calcula la CORRIENTE DE ESPÍN en la dirección z (Iz) como función
de la longitud de la cadena N, para diferentes intensidades de dephasing
(decoherencia de fase), usando las funciones de transmisión Gz(ω) calculadas
previamente con el código 'trans_cadena_bi_W_SO_gammaout0_GZ_decoherence'.

Este código es el PASO FINAL de la cadena de cálculo:

  PASO 1 (código anterior): Calcular Gz(ω, N) con promedio de desorden
          → archivos CSV: data_decoheren_N{j}.csv  (para cada N y nivel de η_d)

  PASO 2 (ESTE CÓDIGO): Integrar Gz(ω) × [f_L(ω) - f_R(ω)] sobre ω
          → corriente de espín Iz(N) para cada nivel de dephasing

================================================================================
FÓRMULA FÍSICA: CORRIENTE DE ESPÍN DE LANDAUER-BÜTTIKER
================================================================================

La corriente de espín Iz se calcula mediante la fórmula de Landauer-Büttiker
generalizada al espín:

  Iz(N) = ∫ dω  Gz(ω) × [f_L(ω, μ_L) - f_R(ω, μ_R)]

donde:
  • Gz(ω)     : conductancia de espín en z (función de la energía),
                obtenida del código de dephasing/desorden previo.
  • f_L(ω)    : distribución de Fermi-Dirac del electrodo izquierdo
                  f_L = 1 / [exp((ω - μ_L) / k_B T) + 1]
  • f_R(ω)    : distribución de Fermi-Dirac del electrodo derecho
                  f_R = 1 / [exp((ω - μ_R) / k_B T) + 1]
  • μ_L, μ_R  : potenciales químicos de los electrodos izquierdo y derecho

La diferencia f_L - f_R actúa como la "ventana de transmisión": solo las
energías entre μ_R y μ_L contribuyen a la corriente neta de espín.

LÍMITE DE TEMPERATURA CERO (kbT → 0):
  En el límite T → 0 (kbT = 10⁻⁹ ≈ 0 aquí), la función de Fermi se convierte
  en una función escalón de Heaviside:
      f(ω, μ) → θ(μ - ω)
  y la integral se reduce a:
      Iz(N) = ∫_{μ_R}^{μ_L} dω  Gz(ω)

POTENCIALES QUÍMICOS Y VOLTAJE APLICADO:
  El voltaje de polarización V se distribuye simétricamente alrededor de ω=0:
      μ_L = +i/2  (electrodo izquierdo)
      μ_R = -i/2  (electrodo derecho)
  con i = 4.0 meV en este script. La corriente de espín se calcula como la
  integral de Gz(ω) en la ventana [-i/2, +i/2] = [-2, +2] meV.

  Los parámetros muLup, muRup, muLdown, muRdown permiten extender la fórmula
  a una polarización dependiente del espín (no usada completamente aquí, ya
  que solo muLup se usa en la expresión de la integral).

================================================================================
NIVELES DE DEPHASING COMPARADOS
================================================================================

El script carga datos de CUATRO configuraciones distintas, identificadas por
la intensidad del dephasing η_d (amplitud de la distribución de los η_n):

  data  (w05): η_d = 0.5  → dephasing moderado-bajo
  data1 (w)  : η_d = 1.0  → dephasing moderado
  data2 (w2) : η_d = 2.0  → dephasing fuerte
  data3 (N)  : η_d = 0    → sin dephasing (sistema coherente puro, referencia)

Para cada valor de N ∈ {10, 19, 28, 37, 46, 55, 64, 73, 82, 91} se calcula
Iz(N) con cada nivel de dephasing, permitiendo estudiar cómo la decoherencia
afecta la corriente de espín en función de la longitud de la cadena.

NOTA sobre las carpetas de Google Drive:
  Los archivos de transmisión precomputados están organizados en subcarpetas
  por nivel de dephasing dentro de 'drive/MyDrive/Rashba/trans_vs_N_decoherencia1/':
    w05/ → η_d = 0.5
    w/   → η_d = 1.0
    w2/  → η_d = 2.0
  El caso sin dephasing está en 'drive/MyDrive/Rashba/trans_vs_N/'.

================================================================================
ESTRUCTURA DEL SCRIPT
================================================================================
  1. Importaciones y parámetros globales
  2. Bucle sobre longitudes de cadena N: carga de datos y cálculo de Iz(N)
  3. Guardado de resultados en archivos .dat por nivel de dephasing
  4. Copia a Google Drive
  5. Gráfica Iz vs N para los cuatro niveles de dephasing

AUTOR: (agregar nombre del autor)
FECHA: (agregar fecha)
REFERENCIA: Landauer, R., IBM J. Res. Dev. 1, 223 (1957);
            Büttiker, M., Phys. Rev. B 33, 3020 (1986);
            Datta, S., "Electronic Transport in Mesoscopic Systems" (1995).
================================================================================
"""

# ─── Importaciones ────────────────────────────────────────────────────────────
import pandas as pd                       # Lectura de archivos CSV con los datos de Gz(ω)
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve  # No usada aquí (heredada de la plantilla)
from scipy import sparse                  # No usada aquí
from scipy.integrate import simps, trapz  # simps: integración numérica por regla de Simpson
import time                               # No usada aquí
import random                             # No usada aquí
from numba import jit                     # No usada aquí (heredada de la plantilla)
from google.colab import drive
drive.mount('/content/drive', force_remount=True)   # Montar Google Drive en Colab
                                                    # Necesario para acceder a los CSV
                                                    # con las funciones de transmisión

import warnings
warnings.filterwarnings("ignore")        # Suprime warnings de scipy/numpy


# ─── Parámetros físicos globales ──────────────────────────────────────────────

# Temperatura térmica en unidades de energía (meV)
# kbT ≈ 0 implementa el límite de temperatura cero:
# la distribución de Fermi-Dirac se convierte en función escalón θ(μ - ω).
# Con kbT = 10⁻⁹ se evita la división por cero numérica en exp((ω-μ)/kbT).
kbT = 0.000000001   # ~ 0 K (límite de temperatura cero)

# ── Acumuladores de corriente de espín para cada nivel de dephasing ──────────
# Cada array Iz_k almacenará un valor Iz(N) por cada longitud N en 'list'.
Iz  = np.array([], dtype=float)   # Iz(N) para η_d = 0.5
Iz1 = np.array([], dtype=float)   # Iz(N) para η_d = 1.0
Iz2 = np.array([], dtype=float)   # Iz(N) para η_d = 2.0
Iz3 = np.array([], dtype=float)   # Iz(N) para η_d = 0  (caso coherente, referencia)

# ── Parámetros del voltaje de polarización y potenciales químicos ─────────────
# El voltaje V se aplica simétricamente entre los electrodos:
#   μ_L = +(i/2)  (electrodo izquierdo: más alto)
#   μ_R = -(i/2)  (electrodo derecho:   más bajo)
# La diferencia de potencial es V = μ_L - μ_R = i = 4.0 meV.
i      = 4.0    # Voltaje de polarización V = 4.0 meV (nota: reusar el nombre 'i'
                # puede confundirse con el índice del bucle interior; renombrar a 'V'
                # o 'bias' mejoraría la legibilidad)

# Potenciales químicos para espín ↑ (up) y espín ↓ (down)
# En este script solo se usa muLup en la expresión de la integral.
# muRup, muLdown, muRdown están definidos pero no modifican el resultado
# (sus valores son cero), lo que corresponde al caso sin campo de intercambio.
muLup   =  1.0   # Potencial químico del electrodo izquierdo, espín ↑ [meV]
muRup   = -1.0   # Potencial químico del electrodo derecho,   espín ↑ [meV]
muLdown =  0.0   # Potencial químico del electrodo izquierdo, espín ↓ [meV] (no usado)
muRdown =  0.0   # Potencial químico del electrodo derecho,   espín ↓ [meV] (no usado)

# NOTA sobre la expresión de la integral:
# Los potenciales que entran en la función de Fermi son:
#   μ_L_eff = (muLup/2) + (i/2) = 0.5 + 2.0 = 2.5 meV
#   μ_R_eff = (muRup/2) + (i/2) = -0.5 + 2.0 = 1.5 meV   ← ERROR POTENCIAL
#
# ADVERTENCIA: En la función de Fermi del electrodo derecho el código usa:
#   -1/(exp((E + (muRup/2 + i/2))/kbT) + 1)
# Lo esperado en Landauer simétrico sería:
#   -1/(exp((E - μ_R)/kbT) + 1)  con μ_R = -i/2 = -2.0
# La expresión actual coloca μ_R_eff = -(muRup/2 + i/2) = -(-0.5+2.0) = -1.5.
# Verificar con la convención exacta del artículo de referencia.


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: BUCLE PRINCIPAL – CARGA DE DATOS Y CÁLCULO DE Iz(N)
# ══════════════════════════════════════════════════════════════════════════════

# Lista de longitudes de cadena a calcular (número de sitios por cadena)
# Los valores están espaciados en 9 sitios: 10, 19, 28, ..., 91
# correspondiendo a una familia de cadenas con el mismo parámetro de red.
list = [10, 19, 28, 37, 46, 55, 64, 73, 82, 91]
# NOTA: el nombre 'list' sobreescribe el built-in de Python.
# Renombrar a 'N_list' o 'chain_lengths' evitaría este conflicto.

for j in list:

    # ── Carga de los archivos de transmisión Gz(ω) ───────────────────────────
    # Cada archivo CSV fue generado por el código de dephasing anterior y contiene
    # dos columnas: 'E' (energía en meV) y 'Gz' (conductancia de espín en z).
    # Los archivos están organizados en subcarpetas por nivel de dephasing η_d.

    # η_d = 0.5: dephasing moderado-bajo (carpeta 'w05/')
    nombre_archivo  = f'drive/MyDrive/Rashba/trans_vs_N_decoherencia1/w05/data_decoheren_N{j}.csv'
    data  = pd.read_csv(nombre_archivo)

    # η_d = 1.0: dephasing moderado (carpeta 'w/')
    nombre_archivo1 = f'drive/MyDrive/Rashba/trans_vs_N_decoherencia1/w/data_decoheren_N{j}.csv'
    data1 = pd.read_csv(nombre_archivo1)

    # η_d = 2.0: dephasing fuerte (carpeta 'w2/')
    nombre_archivo2 = f'drive/MyDrive/Rashba/trans_vs_N_decoherencia1/w2/data_decoheren_N{j}.csv'
    data2 = pd.read_csv(nombre_archivo2)

    # η_d = 0: sin dephasing, sistema coherente puro (carpeta 'trans_vs_N/')
    # Este es el caso de referencia para comparar el efecto de la decoherencia.
    nombre_archivo3 = f'drive/MyDrive/Rashba/trans_vs_N/data_N{j}.csv'
    data3 = pd.read_csv(nombre_archivo3)

    # ── Cálculo de la corriente de espín Iz(N) ────────────────────────────────
    #
    # Se aplica la fórmula de Landauer-Büttiker para corriente de espín:
    #
    #   Iz = ∫ dω  Gz(ω) × [f_L(ω) - f_R(ω)]
    #
    # donde las distribuciones de Fermi-Dirac son:
    #   f_L(ω) = 1 / {exp[(ω - μ_L) / kbT] + 1}  con μ_L = muLup/2 + i/2
    #   f_R(ω) = 1 / {exp[(ω + μ_R) / kbT] + 1}  con μ_R evaluada con signo positivo
    #
    # La integración numérica usa scipy.integrate.simps (regla de Simpson compuesta),
    # que es más precisa que la regla del trapecio para funciones suaves.
    # El eje de integración es data['E'] (malla de 901 puntos en [-4, 4] meV).
    #
    # En el límite kbT → 0, f_L - f_R → θ(ω - μ_R) - θ(ω - μ_L),
    # que es 1 en el intervalo [μ_R, μ_L] y 0 fuera de él.
    # Por lo tanto Iz ≈ ∫_{μ_R}^{μ_L} Gz(ω) dω en la ventana de transmisión.

    # Corriente para η_d = 0.5
    Current  = simps(
        data['Gz'] * (
            1/(np.exp((data['E']  - (muLup/2 + i/2)) / kbT) + 1)   # f_L(ω)
          - 1/(np.exp((data['E']  + (muRup/2 + i/2)) / kbT) + 1)   # f_R(ω)
        ),
        data['E']    # Variable de integración: energía ω
    )

    # Corriente para η_d = 1.0
    Current1 = simps(
        data1['Gz'] * (
            1/(np.exp((data1['E'] - (muLup/2 + i/2)) / kbT) + 1)
          - 1/(np.exp((data1['E'] + (muRup/2 + i/2)) / kbT) + 1)
        ),
        data1['E']
    )

    # Corriente para η_d = 2.0
    Current2 = simps(
        data2['Gz'] * (
            1/(np.exp((data2['E'] - (muLup/2 + i/2)) / kbT) + 1)
          - 1/(np.exp((data2['E'] + (muRup/2 + i/2)) / kbT) + 1)
        ),
        data2['E']
    )

    # Corriente para η_d = 0 (coherente, referencia)
    Current3 = simps(
        data3['Gz'] * (
            1/(np.exp((data3['E'] - (muLup/2 + i/2)) / kbT) + 1)
          - 1/(np.exp((data3['E'] + (muRup/2 + i/2)) / kbT) + 1)
        ),
        data3['E']
    )

    # ── Acumulación del resultado para este N ─────────────────────────────────
    # Cada Iz_k crece en un elemento por iteración del bucle, construyendo
    # los vectores Iz(N) de longitud len(list) = 10
    Iz  = np.append(Iz,  Current)
    Iz1 = np.append(Iz1, Current1)
    Iz2 = np.append(Iz2, Current2)
    Iz3 = np.append(Iz3, Current3)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: GUARDADO DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

# Re-declarar la lista de N (idéntica a la del bucle; redundante pero inofensivo)
list = [10, 19, 28, 37, 46, 55, 64, 73, 82, 91]

# Construir matrices de dos columnas: (N, Iz) para cada nivel de dephasing
# Columna 1: longitud de cadena N
# Columna 2: corriente de espín Iz(N) [en unidades de e/h × meV]
A  = np.stack((list, Iz),  axis=1)   # η_d = 0.5
A1 = np.stack((list, Iz1), axis=1)   # η_d = 1.0
A2 = np.stack((list, Iz2), axis=1)   # η_d = 2.0
A3 = np.stack((list, Iz3), axis=1)   # η_d = 0  (coherente)

# Guardar en archivos .dat: dos columnas separadas por espacios, 6 cifras decimales
nombre  = 'chain_cs_N_decoherence05.dat'   # Corriente de espín vs N, η_d = 0.5
np.savetxt(nombre,  A,  fmt='%.6e')

nombre1 = 'chain_cs_N_decoherence1.dat'    # Corriente de espín vs N, η_d = 1.0
np.savetxt(nombre1, A1, fmt='%.6e')

nombre2 = 'chain_cs_N_decoherence2.dat'    # Corriente de espín vs N, η_d = 2.0
np.savetxt(nombre2, A2, fmt='%.6e')

nombre3 = 'chain_cs_N_without_decoherence.dat'   # Corriente de espín vs N, sin dephasing
np.savetxt(nombre3, A3, fmt='%.6e')

# Líneas comentadas para posibles extensiones futuras (otros valores de η_d):
# nombre4 = 'chain_cs_p_0_d05.dat'
# np.savetxt(nombre4, A4, fmt='%.6e')
# nombre5 = 'chain_cs_p_0_d06.dat'
# np.savetxt(nombre5, A5, fmt='%.6e')

# ── Copia de los archivos generados a Google Drive ───────────────────────────
# El comando shell !cp copia todos los archivos 'chain_*' al directorio de Drive
# donde se almacenan las gráficas de decoherencia.
!cp chain* drive/MyDrive/Rashba/graficas/decoherence/.


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: GRÁFICA – CORRIENTE DE ESPÍN Iz vs LONGITUD DE CADENA N
# ══════════════════════════════════════════════════════════════════════════════
#
# Se grafica Iz(N) para los cuatro niveles de dephasing en la misma figura,
# permitiendo comparar visualmente cómo la decoherencia afecta la corriente
# de espín en función de la longitud de la cadena.
#
# Comportamiento esperado:
#   - Sin dephasing (η_d = 0): la corriente puede oscilar o decaer lentamente
#     con N dependiendo de la longitud de coherencia de espín.
#   - Con dephasing creciente: la corriente tiende a comportarse de forma más
#     difusiva (decaimiento ~ 1/N) o a saturar, dependiendo del régimen.
# ──────────────────────────────────────────────────────────────────────────────

direc = "chain_cs_N_disorder.png"   # Nombre del archivo de imagen (el nombre
                                     # dice 'disorder' pero corresponde a dephasing;
                                     # posible typo → renombrar a 'chain_cs_N_decoherence.png')

plt.figure(figsize=(16, 7))

plt.plot(list, Iz,  label="W=0.5")   # η_d = 0.5 (W denota aquí la amplitud del dephasing)
plt.plot(list, Iz1, label="W=1.0")   # η_d = 1.0
plt.plot(list, Iz2, label="W=2.0")   # η_d = 2.0
plt.plot(list, Iz3, label="W=0")     # Sin dephasing (caso coherente)

# Etiquetas y leyenda
plt.legend()
plt.xlabel("N")          # Longitud de la cadena (número de sitios)
plt.ylabel("$I_z$")      # Corriente de espín en z (unidades: e/h × meV en u.n.)

# Líneas comentadas para posibles ajustes de escala:
# plt.xlim(-2.0, 2.0)
# plt.xticks(np.arange(-3, 3, 0.5))
