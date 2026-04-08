# -*- coding: utf-8 -*-
"""
================================================================================
trans_cadena_bi_W_SO_gammaout0_GZ_disorder_N91.py
================================================================================

DESCRIPCIÓN GENERAL
-------------------
Este script calcula la conductancia de espín en la dirección z (Gz) como
función de la energía para una doble cadena lineal (bilineal) con interacción
espín-órbita tipo Rashba, en presencia de DESORDEN ESTÁTICO en las energías
de sitio (disorder on-site energies).

El desorden se introduce asignando a cada sitio n de cada cadena una energía
de sitio aleatoria E_n, extraída de una distribución uniforme en [-W/2, W/2]
(con W=1 en este script). A diferencia del código de dephasing, donde la
aleatoriedad rompía la coherencia de fase temporalmente (η imaginario),
aquí el desorden es REAL: modifica las energías del Hamiltoniano sin añadir
amortiguamiento extra. Esto corresponde al modelo de Anderson de localización.

================================================================================
DIFERENCIAS CLAVE RESPECTO AL CÓDIGO DE DEPHASING
================================================================================

1. DESORDEN EN ENERGÍAS DE SITIO (E01, E02 aleatorios):
   • En el código de dephasing: η[i] y η1[i] son arrays que se suman a la
     parte IMAGINARIA de la diagonal (±i·η_n), rompiendo la coherencia de fase.
   • En este código: E01[i] y E02[i] son arrays que modifican la parte REAL
     de la diagonal (ε_n = E_n_random), simulando impurezas estáticas.
   El Hamiltoniano efectivo por sitio pasa de:
       ω - 0 + Γ ± i·η_n   (dephasing)
   a:
       ω - E_n + Γ ± i·η_global   (disorder)
   donde E_n es aleatorio y η_global es el pequeño escalar de regularización.

2. ETA VUELVE A SER UN ESCALAR GLOBAL PEQUEÑO:
   En el código de dephasing, η era un array de N valores aleatorios por
   realización. Aquí η recupera su rol de simple regularizador numérico
   (η = 0.00001), garantizando la causalidad de la función de Green sin
   añadir decoherencia. Toda la aleatoriedad está en E01[i] y E02[i].

3. CADENAS 1 Y 2 CON DESORDEN INDEPENDIENTE:
   E01[i] y E02[i] se generan independientemente para cada realización i,
   modelando impurezas no correlacionadas entre las dos cadenas.

4. FUNCIÓN Random() SIN RESTRICCIÓN DE MEDIA CERO:
   • Código de dephasing: eta[-1] += -sum(eta) forzaba media exactamente 0
     (necesario para que η_n no desplace el espectro).
   • Este código: Random() retorna directamente np.random.rand(i)-0.5 sin
     corrección de media. En desorden de Anderson esto es válido porque la
     media del desorden no desplaza el espectro al tomar suficientes
     realizaciones (la media de E_n es cero en distribución).

5. FIRMA DE matrix_AR y matrix_AA (sin argumentos eta, eta1):
   Las matrices ya no reciben η como argumento de realización, sino que
   usan el escalar global η directamente por clausura (closure sobre la
   variable global del módulo).

================================================================================
MODELO FÍSICO: DESORDEN DE ANDERSON
================================================================================

El Hamiltoniano de tight-binding con desorden de Anderson es:

  H = Σ_n  E_n |n><n|  +  Σ_{<n,m>} t_{nm} |n><m|  +  H_SOI

donde E_n son variables aleatorias uniformes en [-W/2, W/2] con W=1 aquí.

En el límite de desorden débil (W << t), el sistema es difusivo: la
conductancia decae como 1/N (ley de Ohm).
En el límite de desorden fuerte (W >> t), ocurre localización de Anderson:
la conductancia decae exponencialmente con N.

La conductancia de espín Gz compite con ambos regímenes, y su comportamiento
con N permite estudiar si la SOI protege la corriente de espín frente a la
localización.

================================================================================
PARÁMETROS DEL SCRIPT
================================================================================
  N          = 91        Número de sitios por cadena
  M          = 10000     Número de realizaciones del desorden
  gamma01/02 = 1.0 meV   Hopping intra-cadena
  l_R1/R2    = 0.1 meV   Fuerza de Rashba (λ_SOC = 0.1 meV)
  gamma_out1 = 1.0 meV   Acoplamiento a electrodos (γ_out = 1 meV)
  W          = 1.0       Amplitud del desorden (energías en [-0.5, 0.5] meV)
  eta        = 0.00001   Regularización numérica (pequeña, no física)
  l_D        = 0.0       Dresselhaus apagado
  beta       = π         Desfase geométrico entre cadenas

ESTRUCTURA DEL SCRIPT
-----------------------
  1. Importaciones y parámetros globales
  2. matrix_AR: función de Green retardada (E_n aleatorio, η escalar global)
  3. matrix_AA: función de Green avanzada
  4. den_espectral*: extracción de amplitudes de transmisión
  5. rho_w*: barrido en energía con doble bucle y promedio de desorden
  6. Random(): generador de energías de sitio aleatorias
  7. Bloque principal: cálculo de Gz, guardado en .dat y conversión a .csv

AUTOR: David Verrilli, Nelson Bolívar
FECHA: 2024
REFERENCIA: Anderson, P.W., Phys. Rev. 109, 1492 (1958) – localización.
            Haug & Jauho, "Quantum Kinetics in Transport and Optics of
            Semiconductors" – funciones de Green en transporte cuántico.
================================================================================
"""

# ─── Importaciones ────────────────────────────────────────────────────────────
import pandas as pd                       # Manipulación de datos y exportación a CSV
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve  # Resolución eficiente de A·x = b con A dispersa
from scipy import sparse                  # Almacenamiento en formato CSR
from scipy.integrate import simps, trapz  # Integración numérica (importadas, no usadas aquí)
import random                             # Generador de números aleatorios (no usado aquí)
import time                               # Medición de tiempos de ejecución
from numba import jit, cuda               # Compilación JIT para acelerar bucles numéricos
from numpy import array

import warnings
warnings.filterwarnings("ignore")        # Suprime warnings de numpy/scipy durante el cálculo


# ─── Parámetros globales ───────────────────────────────────────────────────────
nu    = 1                   # Factor de degeneración (no usado explícitamente)
muAB  = 1                   # Potencial químico relativo (no usado)
eta   = 0.0001              # Regularización numérica inicial (sobreescrita en bloque principal)
e     = 1                   # Carga del electrón (unidades naturales)
c     = 1                   # Velocidad de la luz (unidades naturales)
h     = 1                   # Constante de Planck reducida (ℏ = 1)
phi_0 = c * h / e           # Cuanto de flujo magnético φ₀ = hc/e

stepc  = 901                # Número de puntos en el barrido de energía
step   = 91                 # Longitud de cadena de referencia (no usada directamente)
stepD  = 9001               # Pasos para barridos de desorden (no usado aquí)
beta   = np.pi              # Desfase β = π de la SOI de Rashba en la cadena 2
                            # Este desfase diferencia geométricamente las dos cadenas


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: MATRICES DE LA FUNCIÓN DE GREEN CON DESORDEN ESTÁTICO
# ══════════════════════════════════════════════════════════════════════════════

@jit   # Numba JIT: acelera la construcción de la matriz en el bucle de realizaciones
def matrix_AR(w, E01, E02, N, theta,
              Gamma1, Gamma2, Gamma3, Gamma4,
              gamma01, gamma02, gamma_per, gamma_per1,
              U, M1, M2, l_R1, l_R2, l_D):
    """
    Construye la matriz del sistema lineal para la función de Green RETARDADA
    en presencia de desorden estático en las energías de sitio.

    DIFERENCIA FUNDAMENTAL RESPECTO AL CÓDIGO DE DEPHASING:
    --------------------------------------------------------
    • Aquí E01 y E02 son arrays(N) con energías de sitio ALEATORIAS REALES:
          E01[n] ~ Uniforme(-0.5, 0.5)   para cadena 1
          E02[n] ~ Uniforme(-0.5, 0.5)   para cadena 2
      Esto modifica la parte REAL de la diagonal del operador:
          -(ω - E_n_random + Γ + i·η_global)
      produciendo dispersión elástica (preserva la coherencia de fase pero
      rompe la traslación espacial del sistema).

    • η es el escalar global pequeño (η = 0.00001), no un array aleatorio.
      Su único rol aquí es la regularización numérica, NO el dephasing.

    • La firma NO incluye eta como argumento: la función usa la variable
      global η directamente (clausura), a diferencia del código de dephasing
      donde η era un parámetro explícito de cada realización.

    Parámetros
    ----------
    w         : float      → energía actual en el barrido
    E01       : array(N)   → energías de sitio aleatorias de la cadena 1
                             (realización i del desorden, generada con Random())
    E02       : array(N)   → energías de sitio aleatorias de la cadena 2
                             (realización i, independiente de E01)
    N         : int        → número de sitios por cadena
    theta     : float      → fase Aharonov-Bohm (= 0 aquí, sin flujo magnético)
    Gamma1    : array(N)   → anchos de nivel, canal (xi=+1, ↑) en ambas cadenas
    Gamma2    : array(N)   → anchos de nivel, canal (xi=+1, ↓) en ambas cadenas
    Gamma3    : array(N)   → anchos de nivel, canal (xi=-1, ↑) en ambas cadenas
    Gamma4    : array(N)   → anchos de nivel, canal (xi=-1, ↓) en ambas cadenas
    gamma01   : float      → hopping a lo largo de la cadena 1
    gamma02   : float      → hopping a lo largo de la cadena 2
    gamma_per : float      → acoplamiento perpendicular intra-espín entre cadenas
    gamma_per1: float      → acoplamiento perpendicular inter-espín entre cadenas
    U         : array(N)   → potencial adicional de sitio (= 0 aquí)
    M1, M2    : float      → masas efectivas (no usadas explícitamente)
    l_R1      : float      → intensidad de la SOI de Rashba en la cadena 1
    l_R2      : float      → intensidad de la SOI de Rashba en la cadena 2
    l_D       : float      → intensidad de la SOI de Dresselhaus (= 0 aquí)

    Retorna
    -------
    A : scipy.sparse.csr_matrix (8N × 8N, compleja)
        Operador [ω - H(E_n_random) + iΓ + i·η] en formato CSR.

    Estructura de bloques (8N × 8N):
      [0:N)    cadena 1, xi=+1, ↑    [4N:5N)  cadena 2, xi=+1, ↑
      [N:2N)   cadena 1, xi=+1, ↓    [5N:6N)  cadena 2, xi=+1, ↓
      [2N:3N)  cadena 1, xi=-1, ↑    [6N:7N)  cadena 2, xi=-1, ↑
      [3N:4N)  cadena 1, xi=-1, ↓    [7N:8N)  cadena 2, xi=-1, ↓
    """
    A = np.zeros((8*N, 8*N), dtype=complex)

    # Incremento de ángulo azimutal entre sitios para la fase de Rashba.
    # NOTA: el denominador (10-1) está hardcodeado. En general debería
    # ser (N-1) para que la fase barra [0, 2π] independientemente de N.
    Dphi = (2 * np.pi / (10 - 1))

    for n in range(1, N+1):

        # ── Diagonal principal: energía de sitio + electrodo + regularización ──
        #
        # La forma general de cada elemento diagonal es:
        #     -(ω - E_n + Γ_canal[n] + i·η)
        #
        # CLAVE DE ESTE CÓDIGO: E01[n-1] y E02[n-1] son valores aleatorios
        # REALES específicos de esta realización i del desorden (Anderson).
        # El escalar global η actúa solo como regularizador numérico.
        l = (n-1) % N + 1   # Índice circular (= n para cadena abierta)

        # Cadena 1 – cuatro canales, cada uno con su propio ancho de nivel Gamma_j
        # pero la misma energía de sitio aleatoria E01[n-1]
        A[n-1,     l-1    ] = -(w - E01[n-1] + Gamma1[n-1] + eta*1.0j)  # xi=+1, ↑
        A[n-1+N,   l-1+N  ] = -(w - E01[n-1] + Gamma2[n-1] + eta*1.0j)  # xi=+1, ↓
        A[n-1+2*N, l-1+2*N] = -(w - E01[n-1] + Gamma3[n-1] + eta*1.0j)  # xi=-1, ↑
        A[n-1+3*N, l-1+3*N] = -(w - E01[n-1] + Gamma4[n-1] + eta*1.0j)  # xi=-1, ↓

        # Cadena 2 – mismos Gamma_j pero con energía de sitio aleatoria E02[n-1]
        # Las dos cadenas tienen desorden INDEPENDIENTE entre sí
        A[n-1+4*N, l-1+4*N] = -(w - E02[n-1] + Gamma1[n-1] + eta*1.0j)  # xi=+1, ↑
        A[n-1+5*N, l-1+5*N] = -(w - E02[n-1] + Gamma2[n-1] + eta*1.0j)  # xi=+1, ↓
        A[n-1+6*N, l-1+6*N] = -(w - E02[n-1] + Gamma3[n-1] + eta*1.0j)  # xi=-1, ↑
        A[n-1+7*N, l-1+7*N] = -(w - E02[n-1] + Gamma4[n-1] + eta*1.0j)  # xi=-1, ↓

        # ── Acoplamiento perpendicular entre cadenas (intra-espín, mismo xi) ───
        # gamma_per acopla los mismos canales de espín entre cadena 1 y cadena 2
        A[n-1,     l-1+4*N] = gamma_per   # c1(xi=+1,↑) ↔ c2(xi=+1,↑)
        A[n-1+N,   l-1+5*N] = gamma_per   # c1(xi=+1,↓) ↔ c2(xi=+1,↓)
        A[n-1+2*N, l-1+6*N] = gamma_per   # c1(xi=-1,↑) ↔ c2(xi=-1,↑)
        A[n-1+3*N, l-1+7*N] = gamma_per   # c1(xi=-1,↓) ↔ c2(xi=-1,↓)

        A[n-1+4*N, l-1    ] = gamma_per   # Términos hermíticos (c2 ↔ c1)
        A[n-1+5*N, l-1+N  ] = gamma_per
        A[n-1+6*N, l-1+2*N] = gamma_per
        A[n-1+7*N, l-1+3*N] = gamma_per

        # ── Acoplamiento perpendicular inter-espín con inversión de xi ─────────
        # gamma_per1 acopla cadenas con inversión del índice de propagación xi
        A[n-1,     l-1+6*N] = gamma_per1  # c1(xi=+1,↑) ↔ c2(xi=-1,↑)
        A[n-1+N,   l-1+7*N] = gamma_per1  # c1(xi=+1,↓) ↔ c2(xi=-1,↓)
        A[n-1+2*N, l-1+4*N] = gamma_per1  # c1(xi=-1,↑) ↔ c2(xi=+1,↑)
        A[n-1+3*N, l-1+5*N] = gamma_per1  # c1(xi=-1,↓) ↔ c2(xi=+1,↓)

        A[n-1+4*N, l-1+2*N] = gamma_per1
        A[n-1+5*N, l-1+3*N] = gamma_per1
        A[n-1+6*N, l-1    ] = gamma_per1
        A[n-1+7*N, l-1+N  ] = gamma_per1

        # ══════════════════════════════════════════════════════════════════
        # HOPPING HACIA EL SITIO n+1 (vecino derecho)
        # ══════════════════════════════════════════════════════════════════
        l  = n % N + 1       # Índice del sitio n+1
        pp = (n-1) * Dphi    # Fase angular acumulada hasta el sitio n (para Rashba)

        # Hopping estándar con condición de contorno abierta:
        # el último elemento de tt1/tt2 es 0 → no hay hopping desde el extremo derecho
        tt1 = np.multiply([*np.ones(N-1), 0], gamma01)
        tt2 = np.multiply([*np.ones(N-1), 0], gamma02)

        # Hopping sin SOI – diagonal en espín
        A[n-1,     l-1    ] += tt1[n-1];   A[n-1+4*N, l-1+4*N] += tt2[n-1]
        A[n-1+N,   l-1+N  ] += tt1[n-1];   A[n-1+5*N, l-1+5*N] += tt2[n-1]
        A[n-1+2*N, l-1+2*N] += tt1[n-1];   A[n-1+6*N, l-1+6*N] += tt2[n-1]
        A[n-1+3*N, l-1+3*N] += tt1[n-1];   A[n-1+7*N, l-1+7*N] += tt2[n-1]

        # Máscaras de Rashba con condición de contorno abierta
        l_R11 = np.multiply([*np.ones(N-1), 0], l_R1)
        l_R12 = np.multiply([*np.ones(N-1), 0], l_R2)

        # SOI de Rashba y Dresselhaus hacia n+1 – cadena 1:
        # Mezcla espines opuestos entre sitios vecinos con fase acumulada pp
        A[n-1,     l-1+2*N] = (-1j*l_R11[n-1]*np.exp(-pp*1j) + l_D*np.exp( pp*1j))  # xi=+1,↑ → xi=-1,↑
        A[n-1+N,   l-1+3*N] = (-1j*l_R11[n-1]*np.exp( pp*1j) - l_D*np.exp(-pp*1j))  # xi=+1,↓ → xi=-1,↓
        A[n-1+2*N, l-1    ] = (-1j*l_R11[n-1]*np.exp( pp*1j) - l_D*np.exp(-pp*1j))  # xi=-1,↑ → xi=+1,↑
        A[n-1+3*N, l-1+N  ] = (-1j*l_R11[n-1]*np.exp(-pp*1j) + l_D*np.exp( pp*1j))  # xi=-1,↓ → xi=+1,↓

        # SOI hacia n+1 – cadena 2 con desfase adicional β = π
        A[n-1+4*N, l-1+6*N] = (-1j*l_R12[n-1]*np.exp(-pp*1j - beta*1j) + l_D*np.exp( pp*1j))
        A[n-1+5*N, l-1+7*N] = (-1j*l_R12[n-1]*np.exp( pp*1j + beta*1j) - l_D*np.exp(-pp*1j))
        A[n-1+6*N, l-1+4*N] = (-1j*l_R12[n-1]*np.exp( pp*1j + beta*1j) - l_D*np.exp(-pp*1j))
        A[n-1+7*N, l-1+5*N] = (-1j*l_R12[n-1]*np.exp(-pp*1j - beta*1j) + l_D*np.exp( pp*1j))

        # ══════════════════════════════════════════════════════════════════
        # HOPPING HACIA EL SITIO n-1 (vecino izquierdo)
        # ══════════════════════════════════════════════════════════════════
        l  = (n-2) % N + 1   # Índice del sitio n-1
        pp = (n-2) * Dphi    # Fase angular del sitio n-1 (para la SOI)

        # Hopping con condición de contorno abierta: primer elemento = 0
        tt1 = np.multiply([0, *np.ones(N-1)], gamma01)
        tt2 = np.multiply([0, *np.ones(N-1)], gamma02)

        # Hopping sin SOI hacia n-1
        A[n-1,     l-1    ] += tt1[n-1];   A[n-1+4*N, l-1+4*N] += tt2[n-1]
        A[n-1+N,   l-1+N  ] += tt1[n-1];   A[n-1+5*N, l-1+5*N] += tt2[n-1]
        A[n-1+2*N, l-1+2*N] += tt1[n-1];   A[n-1+6*N, l-1+6*N] += tt2[n-1]
        A[n-1+3*N, l-1+3*N] += tt1[n-1];   A[n-1+7*N, l-1+7*N] += tt2[n-1]

        # Máscaras de Rashba para el hopping hacia n-1
        l_R111 = np.multiply([0, *np.ones(N-1)], l_R1)
        l_R122 = np.multiply([0, *np.ones(N-1)], l_R2)

        # SOI hacia n-1 – cadena 1 (signo opuesto al hopping n→n+1, hermicidad)
        A[n-1,     l-1+2*N] += ( 1j*l_R111[n-1]*np.exp(-pp*1j) - l_D*np.exp( pp*1j))
        A[n-1+N,   l-1+3*N] += ( 1j*l_R111[n-1]*np.exp( pp*1j) + l_D*np.exp(-pp*1j))
        A[n-1+2*N, l-1    ] += ( 1j*l_R111[n-1]*np.exp( pp*1j) + l_D*np.exp(-pp*1j))
        A[n-1+3*N, l-1+N  ] += ( 1j*l_R111[n-1]*np.exp(-pp*1j) - l_D*np.exp( pp*1j))

        # SOI hacia n-1 – cadena 2 con desfase β
        A[n-1+4*N, l-1+6*N] += ( 1j*l_R122[n-1]*np.exp(-pp*1j - beta*1j) - l_D*np.exp( pp*1j))
        A[n-1+5*N, l-1+7*N] += ( 1j*l_R122[n-1]*np.exp( pp*1j + beta*1j) + l_D*np.exp(-pp*1j))
        A[n-1+6*N, l-1+4*N] += ( 1j*l_R122[n-1]*np.exp( pp*1j + beta*1j) + l_D*np.exp(-pp*1j))
        A[n-1+7*N, l-1+5*N] += ( 1j*l_R122[n-1]*np.exp(-pp*1j - beta*1j) - l_D*np.exp( pp*1j))

    A = sparse.csr_matrix(A)
    return A


@jit
def matrix_AA(w, E01, E02, N, theta,
              Gamma1, Gamma2, Gamma3, Gamma4,
              gamma01, gamma02, gamma_per, gamma_per1,
              U, M1, M2, l_R1, l_R2, l_D):
    """
    Construye la matriz del sistema lineal para la función de Green AVANZADA
    con desorden estático en las energías de sitio.

    Idéntica a matrix_AR salvo por el signo del término de regularización:
        matrix_AR:  -(ω - E_n_random + Γ + i·η)   → G^R
        matrix_AA:  -(ω - E_n_random + Γ - i·η)   → G^A = [G^R]†

    El desorden E_n_random afecta AMBAS funciones de Green por igual (está
    en la parte real), mientras que ±i·η es la pequeña parte imaginaria
    que distingue retardada de avanzada.

    Ver matrix_AR para la documentación completa de parámetros.
    """
    A = np.zeros((8*N, 8*N), dtype=complex)

    Dphi  = (2 * np.pi / (10 - 1))
    Dphi1 = (2 * np.pi / (10 - 1))  # Variable separada; igual a Dphi aquí

    for n in range(1, N+1):
        l = (n-1) % N + 1

        # ── Diagonal: IGUAL que matrix_AR pero con -i·η (función avanzada) ─────
        # El desorden E_n_random está en la parte real → mismo signo en G^R y G^A
        A[n-1,     l-1    ] = -(w - E01[n-1] + Gamma1[n-1] - eta*1.0j)
        A[n-1+N,   l-1+N  ] = -(w - E01[n-1] + Gamma2[n-1] - eta*1.0j)
        A[n-1+2*N, l-1+2*N] = -(w - E01[n-1] + Gamma3[n-1] - eta*1.0j)
        A[n-1+3*N, l-1+3*N] = -(w - E01[n-1] + Gamma4[n-1] - eta*1.0j)

        A[n-1+4*N, l-1+4*N] = -(w - E02[n-1] + Gamma1[n-1] - eta*1.0j)
        A[n-1+5*N, l-1+5*N] = -(w - E02[n-1] + Gamma2[n-1] - eta*1.0j)
        A[n-1+6*N, l-1+6*N] = -(w - E02[n-1] + Gamma3[n-1] - eta*1.0j)
        A[n-1+7*N, l-1+7*N] = -(w - E02[n-1] + Gamma4[n-1] - eta*1.0j)

        # ── Acoplamientos perpendiculares: idénticos a matrix_AR ──────────────
        A[n-1,     l-1+4*N] = gamma_per;   A[n-1+4*N, l-1    ] = gamma_per
        A[n-1+N,   l-1+5*N] = gamma_per;   A[n-1+5*N, l-1+N  ] = gamma_per
        A[n-1+2*N, l-1+6*N] = gamma_per;   A[n-1+6*N, l-1+2*N] = gamma_per
        A[n-1+3*N, l-1+7*N] = gamma_per;   A[n-1+7*N, l-1+3*N] = gamma_per

        A[n-1,     l-1+6*N] = gamma_per1;  A[n-1+4*N, l-1+2*N] = gamma_per1
        A[n-1+N,   l-1+7*N] = gamma_per1;  A[n-1+5*N, l-1+3*N] = gamma_per1
        A[n-1+2*N, l-1+4*N] = gamma_per1;  A[n-1+6*N, l-1    ] = gamma_per1
        A[n-1+3*N, l-1+5*N] = gamma_per1;  A[n-1+7*N, l-1+N  ] = gamma_per1

        # ── Hopping hacia n+1 ──────────────────────────────────────────────────
        l  = n % N + 1
        pp = (n-1) * Dphi

        tt1 = np.multiply([*np.ones(N-1), 0], gamma01)
        tt2 = np.multiply([*np.ones(N-1), 0], gamma02)

        A[n-1,     l-1    ] += tt1[n-1];   A[n-1+4*N, l-1+4*N] += tt2[n-1]
        A[n-1+N,   l-1+N  ] += tt1[n-1];   A[n-1+5*N, l-1+5*N] += tt2[n-1]
        A[n-1+2*N, l-1+2*N] += tt1[n-1];   A[n-1+6*N, l-1+6*N] += tt2[n-1]
        A[n-1+3*N, l-1+3*N] += tt1[n-1];   A[n-1+7*N, l-1+7*N] += tt2[n-1]

        l_R11 = np.multiply([*np.ones(N-1), 0], l_R1)
        l_R12 = np.multiply([*np.ones(N-1), 0], l_R2)

        # SOI hacia n+1 – cadena 1
        A[n-1,     l-1+2*N] = (-1j*l_R11[n-1]*np.exp(-pp*1j) + l_D*np.exp( pp*1j))
        A[n-1+N,   l-1+3*N] = (-1j*l_R11[n-1]*np.exp( pp*1j) - l_D*np.exp(-pp*1j))
        A[n-1+2*N, l-1    ] = (-1j*l_R11[n-1]*np.exp( pp*1j) - l_D*np.exp(-pp*1j))
        A[n-1+3*N, l-1+N  ] = (-1j*l_R11[n-1]*np.exp(-pp*1j) + l_D*np.exp( pp*1j))

        pp = (n-1) * Dphi1
        # SOI hacia n+1 – cadena 2 con desfase β
        A[n-1+4*N, l-1+6*N] = (-1j*l_R12[n-1]*np.exp(-pp*1j)*np.exp(-beta*1j) + l_D*np.exp( pp*1j))
        A[n-1+5*N, l-1+7*N] = (-1j*l_R12[n-1]*np.exp( pp*1j)*np.exp( beta*1j) - l_D*np.exp(-pp*1j))
        A[n-1+6*N, l-1+4*N] = (-1j*l_R12[n-1]*np.exp( pp*1j)*np.exp( beta*1j) - l_D*np.exp(-pp*1j))
        A[n-1+7*N, l-1+5*N] = (-1j*l_R12[n-1]*np.exp(-pp*1j)*np.exp(-beta*1j) + l_D*np.exp( pp*1j))

        # ── Hopping hacia n-1 ──────────────────────────────────────────────────
        l  = (n-2) % N + 1
        pp = (n-2) * Dphi

        tt1 = np.multiply([0, *np.ones(N-1)], gamma01)
        tt2 = np.multiply([0, *np.ones(N-1)], gamma02)

        A[n-1,     l-1    ] += tt1[n-1];   A[n-1+4*N, l-1+4*N] += tt2[n-1]
        A[n-1+N,   l-1+N  ] += tt1[n-1];   A[n-1+5*N, l-1+5*N] += tt2[n-1]
        A[n-1+2*N, l-1+2*N] += tt1[n-1];   A[n-1+6*N, l-1+6*N] += tt2[n-1]
        A[n-1+3*N, l-1+3*N] += tt1[n-1];   A[n-1+7*N, l-1+7*N] += tt2[n-1]

        l_R111 = np.multiply([0, *np.ones(N-1)], l_R1)
        l_R122 = np.multiply([0, *np.ones(N-1)], l_R2)

        # SOI hacia n-1 – cadena 1
        A[n-1,     l-1+2*N] += ( 1j*l_R111[n-1]*np.exp(-pp*1j) - l_D*np.exp( pp*1j))
        A[n-1+N,   l-1+3*N] += ( 1j*l_R111[n-1]*np.exp( pp*1j) + l_D*np.exp(-pp*1j))
        A[n-1+2*N, l-1    ] += ( 1j*l_R111[n-1]*np.exp( pp*1j) + l_D*np.exp(-pp*1j))
        A[n-1+3*N, l-1+N  ] += ( 1j*l_R111[n-1]*np.exp(-pp*1j) - l_D*np.exp( pp*1j))

        pp = (n-2) * Dphi1
        # SOI hacia n-1 – cadena 2 con desfase β
        A[n-1+4*N, l-1+6*N] += ( 1j*l_R122[n-1]*np.exp(-pp*1j)*np.exp(-beta*1j) - l_D*np.exp( pp*1j))
        A[n-1+5*N, l-1+7*N] += ( 1j*l_R122[n-1]*np.exp( pp*1j)*np.exp( beta*1j) + l_D*np.exp(-pp*1j))
        A[n-1+6*N, l-1+4*N] += ( 1j*l_R122[n-1]*np.exp( pp*1j)*np.exp( beta*1j) + l_D*np.exp(-pp*1j))
        A[n-1+7*N, l-1+5*N] += ( 1j*l_R122[n-1]*np.exp(-pp*1j)*np.exp(-beta*1j) - l_D*np.exp( pp*1j))

    A = sparse.csr_matrix(A)
    return A


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: EXTRACCIÓN DE AMPLITUDES DE TRANSMISIÓN
# ══════════════════════════════════════════════════════════════════════════════
#
# Estas funciones extraen el elemento G^R(N, 1) de la función de Green,
# que corresponde a la amplitud de transmisión desde el primer sitio (fuente)
# hasta el último sitio (detector) para cada canal de espín y dirección xi.
#
# El vector solución GR1 se obtiene resolviendo A·GR1 = B con B[0]=B[N]=
# B[4N]=B[5N]=-1. La solución GR1[k] es entonces G^R_{k, fuente}.
#
# CONVENCIÓN DE SIGNO: estas funciones retornan +GR1[k] (signo positivo),
# igual que en el código de dephasing y diferente del primer código coherente
# que usaba -GR1[k]. El signo se cancela al calcular |T|² = T·T*.
#
# Índices en el vector GR1:
#   GR1[N-1]   → c1, xi=+1, ↑,  sitio N   (bloque [0:N))
#   GR1[2N-1]  → c1, xi=+1, ↓,  sitio N   (bloque [N:2N))
#   GR1[3N-1]  → c1, xi=-1, ↑,  sitio N   (bloque [2N:3N))
#   GR1[4N-1]  → c1, xi=-1, ↓,  sitio N   (bloque [3N:4N))
#   GR1[5N-1]  → c2, xi=+1, ↑,  sitio N   (bloque [4N:5N))
#   GR1[6N-1]  → c2, xi=+1, ↓,  sitio N   (bloque [5N:6N))
#   GR1[7N-1]  → c2, xi=-1, ↑,  sitio N   (bloque [6N:7N))
#   GR1[8N-1]  → c2, xi=-1, ↓,  sitio N   (bloque [7N:8N))
# ──────────────────────────────────────────────────────────────────────────────

@jit
def den_espectral1u(GR1, GA1, N):
    """Amplitud G^R(N,1): cadena 1, xi=+1, espín ↑. Extrae GR1[N-1]."""
    return GR1[N-1]

@jit
def den_espectral2u(GR1, GA1, N):
    """Amplitud G^R(N,1): cadena 2, xi=+1, espín ↑. Extrae GR1[5N-1]."""
    return GR1[5*N-1]

@jit
def den_espectral1d(GR1, GA1, N):
    """Amplitud G^R(N,1): cadena 1, xi=+1, espín ↓. Extrae GR1[2N-1]."""
    return GR1[2*N-1]

@jit
def den_espectral2d(GR1, GA1, N):
    """Amplitud G^R(N,1): cadena 2, xi=+1, espín ↓. Extrae GR1[6N-1]."""
    return GR1[6*N-1]

@jit
def den_espectral1ud(GR1, GA1, N):
    """
    Amplitud G^R(N,1): cadena 1, xi=-1, espín ↑. Extrae GR1[3N-1].
    El sufijo 'ud' indica inversión de la dirección de propagación xi: +1 → -1,
    manteniendo el espín ↑ (up).
    """
    return GR1[3*N-1]

@jit
def den_espectral2ud(GR1, GA1, N):
    """Amplitud G^R(N,1): cadena 2, xi=-1, espín ↑. Extrae GR1[7N-1]."""
    return GR1[7*N-1]

@jit
def den_espectral1du(GR1, GA1, N):
    """
    Amplitud G^R(N,1): cadena 1, xi=-1, espín ↓. Extrae GR1[4N-1].
    El sufijo 'du' indica inversión de xi: +1 → -1, espín ↓ (down).
    """
    return GR1[4*N-1]

@jit
def den_espectral2du(GR1, GA1, N):
    """Amplitud G^R(N,1): cadena 2, xi=-1, espín ↓. Extrae GR1[8N-1]."""
    return GR1[8*N-1]


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: BARRIDO EN ENERGÍA CON PROMEDIO DE DESORDEN
# ══════════════════════════════════════════════════════════════════════════════
#
# DIFERENCIA RESPECTO AL CÓDIGO DE DEPHASING:
# --------------------------------------------
# • En el código de dephasing, la aleatoriedad se pasaba como argumentos
#   eta[i] y eta1[i] (arrays de dephasing). Aquí se pasa como E01[i] y E02[i]
#   (arrays de energías de sitio aleatorias). Los parámetros eta y eta1
#   desaparecen de la firma de estas funciones.
#
# ESTRUCTURA DEL DOBLE BUCLE (idéntica al código de dephasing):
# -------------------------------------------------------------
#   Para cada energía ω en la malla w1:
#       T_realizaciones = []
#       Para cada realización i = 0..M-1:
#           Construir A^R con E01[i], E02[i] (desorden de esta realización)
#           Resolver A^R · G^R = B
#           Construir A^A con E01[i], E02[i]
#           Resolver A^A · G^A = B
#           Extraer T_i del canal correspondiente
#           Acumular T_i
#       Promediar: <T>(ω) = mean(T_realizaciones)
#
# El promedio sobre M realizaciones estima el promedio de desorden:
#   <G^R(ω)>_E = (1/M) Σ_{i=1}^{M} G^R(ω; {E_n^(i)})
#
# NOTA SOBRE ERGODICIDAD:
#   El promedio aritmético sobre realizaciones equivale al promedio
#   espacial solo si el sistema es ergódico. Para sistemas con localización
#   de Anderson, el promedio aritmético de G puede estar dominado por
#   realizaciones raras (no localizadas). En esos casos, el promedio del
#   logaritmo (promedio geométrico) es más representativo.
#   Este script calcula el promedio aritmético.
# ──────────────────────────────────────────────────────────────────────────────

def rho_w1u(M, w_0, w_f, nw, B, N, E01, E02, theta,
            Gamma1, Gamma2, Gamma3, Gamma4,
            gamma01, gamma02, gamma_per, gamma_per1,
            U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido energético con promedio de desorden: cadena 1, xi=+1, espín ↑.

    Parámetros
    ----------
    M    : int             → número de realizaciones del desorden
    E01  : list de M arrays → cada E01[i] es un array(N) de energías aleatorias de c1
    E02  : list de M arrays → cada E02[i] es un array(N) de energías aleatorias de c2
    Los demás parámetros son los del Hamiltoniano (ver matrix_AR).

    Retorna
    -------
    w1   : array(nw)       → malla de energías en [w_0, w_f]
    Den02: array(nw, cpx)  → <G^R_{N,1}^{(c1,xi=+1,↑)}>(ω) promediado sobre M realizaciones
    """
    w1    = np.linspace(w_0, w_f, nw)
    Den02 = np.array([], dtype=complex)   # Acumulador del promedio sobre ω

    for w0 in w1:
        Den01 = np.array([], dtype=complex)   # Amplitudes de las M realizaciones para este ω

        for i in range(M):   # Bucle sobre realizaciones del desorden de Anderson
            w   = w0
            # G^R con las energías de sitio aleatorias de la realización i
            AR1 = matrix_AR(w, E01[i], E02[i], N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GR1 = np.asarray(spsolve(AR1, B)).ravel()

            # G^A con las mismas energías de sitio aleatorias
            AA1 = matrix_AA(w, E01[i], E02[i], N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GA1 = np.asarray(spsolve(AA1, B)).ravel()

            Den01 = np.append(Den01, den_espectral1u(GR1, GA1, N))

        # Promedio aritmético sobre las M realizaciones
        Den02 = np.append(Den02, np.mean(Den01))

    return w1, Den02


def rho_w2u(M, w_0, w_f, nw, B, N, E01, E02, theta,
            Gamma1, Gamma2, Gamma3, Gamma4,
            gamma01, gamma02, gamma_per, gamma_per1,
            U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido energético con promedio de desorden: cadena 2, xi=+1, espín ↑.
    Extrae GR1[5N-1] en cada realización y promedia sobre M.
    """
    w1    = np.linspace(w_0, w_f, nw)
    Den02 = np.array([], dtype=complex)
    for w0 in w1:
        Den01 = np.array([], dtype=complex)
        for i in range(M):
            w   = w0
            AR1 = matrix_AR(w, E01[i], E02[i], N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GR1 = np.asarray(spsolve(AR1, B)).ravel()
            AA1 = matrix_AA(w, E01[i], E02[i], N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GA1 = np.asarray(spsolve(AA1, B)).ravel()
            Den01 = np.append(Den01, den_espectral2u(GR1, GA1, N))
        Den02 = np.append(Den02, np.mean(Den01))
    return w1, Den02


def rho_w1d(M, w_0, w_f, nw, B, N, E01, E02, theta,
            Gamma1, Gamma2, Gamma3, Gamma4,
            gamma01, gamma02, gamma_per, gamma_per1,
            U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido energético con promedio de desorden: cadena 1, xi=+1, espín ↓.
    Extrae GR1[2N-1] en cada realización y promedia sobre M.
    """
    w1    = np.linspace(w_0, w_f, nw)
    Den02 = np.array([], dtype=complex)
    for w0 in w1:
        Den01 = np.array([], dtype=complex)
        for i in range(M):
            w   = w0
            AR1 = matrix_AR(w, E01[i], E02[i], N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GR1 = np.asarray(spsolve(AR1, B)).ravel()
            AA1 = matrix_AA(w, E01[i], E02[i], N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GA1 = np.asarray(spsolve(AA1, B)).ravel()
            Den01 = np.append(Den01, den_espectral1d(GR1, GA1, N))
        Den02 = np.append(Den02, np.mean(Den01))
    return w1, Den02


def rho_w2d(M, w_0, w_f, nw, B, N, E01, E02, theta,
            Gamma1, Gamma2, Gamma3, Gamma4,
            gamma01, gamma02, gamma_per, gamma_per1,
            U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido energético con promedio de desorden: cadena 2, xi=+1, espín ↓.
    Extrae GR1[6N-1] en cada realización y promedia sobre M.
    """
    w1    = np.linspace(w_0, w_f, nw)
    Den02 = np.array([], dtype=complex)
    for w0 in w1:
        Den01 = np.array([], dtype=complex)
        for i in range(M):
            w   = w0
            AR1 = matrix_AR(w, E01[i], E02[i], N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GR1 = np.asarray(spsolve(AR1, B)).ravel()
            AA1 = matrix_AA(w, E01[i], E02[i], N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GA1 = np.asarray(spsolve(AA1, B)).ravel()
            Den01 = np.append(Den01, den_espectral2d(GR1, GA1, N))
        Den02 = np.append(Den02, np.mean(Den01))
    return w1, Den02


def rho_w1ud(M, w_0, w_f, nw, B, N, E01, E02, theta,
             Gamma1, Gamma2, Gamma3, Gamma4,
             gamma01, gamma02, gamma_per, gamma_per1,
             U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido energético con promedio de desorden: cadena 1, xi=-1, espín ↑.
    Extrae GR1[3N-1] – transmisión con inversión de dirección de propagación.
    """
    w1   = np.linspace(w_0, w_f, nw)
    Den2 = np.array([], dtype=complex)
    for w0 in w1:
        Den1 = np.array([], dtype=complex)
        for i in range(M):
            w   = w0
            AR1 = matrix_AR(w, E01[i], E02[i], N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GR1 = np.asarray(spsolve(AR1, B)).ravel()
            AA1 = matrix_AA(w, E01[i], E02[i], N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GA1 = np.asarray(spsolve(AA1, B)).ravel()
            Den1 = np.append(Den1, den_espectral1ud(GR1, GA1, N))
        Den2 = np.append(Den2, np.mean(Den1))
    return w1, Den2


def rho_w2ud(M, w_0, w_f, nw, B, N, E01, E02, theta,
             Gamma1, Gamma2, Gamma3, Gamma4,
             gamma01, gamma02, gamma_per, gamma_per1,
             U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido energético con promedio de desorden: cadena 2, xi=-1, espín ↑.
    Extrae GR1[7N-1].
    """
    w1   = np.linspace(w_0, w_f, nw)
    Den2 = np.array([], dtype=complex)
    for w0 in w1:
        Den1 = np.array([], dtype=complex)
        for i in range(M):
            w   = w0
            AR1 = matrix_AR(w, E01[i], E02[i], N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GR1 = np.asarray(spsolve(AR1, B)).ravel()
            AA1 = matrix_AA(w, E01[i], E02[i], N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GA1 = np.asarray(spsolve(AA1, B)).ravel()
            Den1 = np.append(Den1, den_espectral2ud(GR1, GA1, N))
        Den2 = np.append(Den2, np.mean(Den1))
    return w1, Den2


def rho_w1du(M, w_0, w_f, nw, B, N, E01, E02, theta,
             Gamma1, Gamma2, Gamma3, Gamma4,
             gamma01, gamma02, gamma_per, gamma_per1,
             U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido energético con promedio de desorden: cadena 1, xi=-1, espín ↓.
    Extrae GR1[4N-1].
    """
    w1   = np.linspace(w_0, w_f, nw)
    Den2 = np.array([], dtype=complex)
    for w0 in w1:
        Den1 = np.array([], dtype=complex)
        for i in range(M):
            w   = w0
            AR1 = matrix_AR(w, E01[i], E02[i], N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GR1 = np.asarray(spsolve(AR1, B)).ravel()
            AA1 = matrix_AA(w, E01[i], E02[i], N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GA1 = np.asarray(spsolve(AA1, B)).ravel()
            Den1 = np.append(Den1, den_espectral1du(GR1, GA1, N))
        Den2 = np.append(Den2, np.mean(Den1))
    return w1, Den2


def rho_w2du(M, w_0, w_f, nw, B, N, E01, E02, theta,
             Gamma1, Gamma2, Gamma3, Gamma4,
             gamma01, gamma02, gamma_per, gamma_per1,
             U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido energético con promedio de desorden: cadena 2, xi=-1, espín ↓.
    Extrae GR1[8N-1].
    """
    w1   = np.linspace(w_0, w_f, nw)
    Den2 = np.array([], dtype=complex)
    for w0 in w1:
        Den1 = np.array([], dtype=complex)
        for i in range(M):
            w   = w0
            AR1 = matrix_AR(w, E01[i], E02[i], N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GR1 = np.asarray(spsolve(AR1, B)).ravel()
            AA1 = matrix_AA(w, E01[i], E02[i], N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GA1 = np.asarray(spsolve(AA1, B)).ravel()
            Den1 = np.append(Den1, den_espectral2du(GR1, GA1, N))
        Den2 = np.append(Den2, np.mean(Den1))
    return w1, Den2


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: GENERADOR DE REALIZACIONES DE DESORDEN DE ANDERSON
# ══════════════════════════════════════════════════════════════════════════════

def Random(i):
    """
    Genera un vector de N = i energías de sitio aleatorias uniformes en [-W/2, W/2]
    con W = 1 (amplitud del desorden de Anderson).

    DIFERENCIA CRÍTICA respecto a la función Random() del código de dephasing:
    --------------------------------------------------------------------------
    • Código de DEPHASING:
          eta001 = np.random.rand(i) - 0.5
          eta001[-1] += -sum(eta001)   ← corrección de media cero
      La corrección era necesaria porque el dephasing imaginario η_n desplaza
      el espectro si tiene media distinta de cero.

    • Código de DESORDEN (este):
          disor = np.random.rand(i) - 0.5   ← sin corrección
      No se aplica corrección de media porque:
      (a) Al promediar sobre M → ∞ realizaciones, la media de E_n tiende a
          cero automáticamente (ley de los grandes números).
      (b) El desorden real E_n no desplaza la posición de los polos de G^R
          de forma sistemática: solo los dispersa aleatoriamente.
      (c) Una pequeña media no nula por realización es la fluctuación estadística
          normal del desorden de Anderson y no debe eliminarse artificialmente.

    El vector resultante corresponde al modelo de Anderson con:
      E_n ~ Uniforme(-W/2, W/2),   W = 1 meV

    Parámetro
    ---------
    i : int → número de sitios (= N)

    Retorna
    -------
    disor : array(i) de floats en [-0.5, 0.5]
            Energías de sitio aleatorias para una realización del desorden.
    """
    disor = np.random.rand(i) - 0.5   # Distribución uniforme en [-0.5, 0.5] meV
    return disor


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: BLOQUE PRINCIPAL – CÁLCULO DE Gz CON DESORDEN DE ANDERSON
# ══════════════════════════════════════════════════════════════════════════════

# ── Parámetros físicos del sistema ────────────────────────────────────────────
gamma01 = 1.0       # Hopping intra-cadena 1 (unidad de energía, e.g. meV)
gamma02 = 1.0       # Hopping intra-cadena 2
l_R1    = 0.1       # Fuerza de Rashba cadena 1: λ_SOC = 0.1 meV
l_R12   = 0.0       # Variable auxiliar (no usada directamente)
l_R2    = 0.1       # Fuerza de Rashba cadena 2
l_EO1   = 0.0       # Zeeman efectivo cadena 1 (no implementado)
l_EO2   = 0.0       # Zeeman efectivo cadena 2 (no implementado)
l_D     = 0.0       # Dresselhaus apagado
l_Z     = 0.0       # Zeeman apagado
eta     = 0.00001   # Regularización numérica (escalar, NO el dephasing del código anterior)
                    # Valor muy pequeño: solo evita la singularidad en G^R = 1/(ω-H+iη)

# ── Parámetros del promedio de desorden ──────────────────────────────────────
M = 10000           # Número de realizaciones del desorden de Anderson
                    # ADVERTENCIA: con N=91 y M=10000 el cálculo es muy costoso.
                    # Para pruebas rápidas usar M=100 o M=1000.

# ── Acoplamiento a electrodos ─────────────────────────────────────────────────
gamma_out  = 0.0    # Acoplamiento perpendicular intra-espín (apagado aquí)
gamma_out1 = 1.0    # Acoplamiento perpendicular inter-espín: γ_out = 1.0 meV

# ── Bucle sobre longitudes de cadena ─────────────────────────────────────────
# list contiene los valores de N a calcular. Solo N=91 en este script.
# Para comparar con múltiples longitudes extender a: [10, 28, 50, 91, 150].
# NOTA: el nombre 'list' sobreescribe el built-in de Python → renombrar
#       preferiblemente a 'N_list' o 'chain_lengths' para evitar conflictos.
list = [91]

p = 0.0   # Asimetría de los electrodos. p=0: configuración simétrica (no magnética)
          # Con p≠0 se pueden modelar electrodos ferromagnéticos (configuración P o AP)

for i in list:

    # ── Generación de M realizaciones del desorden de Anderson ───────────────
    # E01[k] = array(N) de energías aleatorias para la cadena 1, realización k
    # E02[k] = array(N) de energías aleatorias para la cadena 2, realización k
    # Cadenas 1 y 2 tienen desorden INDEPENDIENTE (no correlacionado entre sí)
    E01 = [Random(i) for _ in range(M)]   # M realizaciones para cadena 1
    E02 = [Random(i) for _ in range(M)]   # M realizaciones para cadena 2

    theta = 0       # Sin flujo Aharonov-Bohm
    M1    = 1.0     # Masa efectiva cadena 1
    M2    = 1.0     # Masa efectiva cadena 2
    U     = np.zeros(i)   # Sin desorden adicional (el desorden ya está en E01, E02)

    # ── Vector fuente B ───────────────────────────────────────────────────────
    # Excita los 4 canales del primer sitio de cada cadena:
    #   B[0]  = -1 → c1, xi=+1, ↑, sitio 1
    #   B[i]  = -1 → c1, xi=+1, ↓, sitio 1
    #   B[4i] = -1 → c2, xi=+1, ↑, sitio 1
    #   B[5i] = -1 → c2, xi=+1, ↓, sitio 1
    B       = np.zeros(8*i, dtype=complex)
    B[0]    = -1
    B[i]    = -1
    B[4*i]  = -1
    B[5*i]  = -1

    # ── Anchos de nivel de los electrodos (cuatro canales) ───────────────────
    # val  = i·(1+p) = i·1  → electrodo paralelo (P)
    # val1 = i·(1-p) = i·1  → electrodo antiparalelo (AP)
    # Con p=0: val = val1 = i → todos los electrodos son equivalentes
    val  = 1.0*(1+p)*1j   # Ancho del electrodo P
    val1 = 1.0*(1-p)*1j   # Ancho del electrodo AP

    # Gamma_j es no nulo solo en los extremos (n=1 y n=N): electrodos left/right
    # La asimetría izquierda/derecha configura polarización P o AP:
    #   Gamma1 [val,  0..., val ] → P-P  para (xi=+1, ↑)
    #   Gamma2 [val,  0..., val1] → P-AP para (xi=+1, ↓)
    #   Gamma3 [val1, 0..., val ] → AP-P para (xi=-1, ↑)
    #   Gamma4 [val1, 0..., val1] → AP-AP para (xi=-1, ↓)
    Gamma1 = [val,  *np.zeros(i-2), val ]
    Gamma2 = [val,  *np.zeros(i-2), val1]
    Gamma3 = [val1, *np.zeros(i-2), val ]
    Gamma4 = [val1, *np.zeros(i-2), val1]

    start_time = time.time()

    # ── Cálculo de los 8 canales de transmisión ───────────────────────────────
    # Grupo 1: gamma_per = 0, gamma_per1 = +gamma_out1 = +1.0
    gamma_per  = gamma_out      # = 0 (acoplamiento intra-espín apagado)
    gamma_per1 = gamma_out1     # = 1.0

    w1, Trans1u  = rho_w1u( M, -4.0, 4.0, 901, B, i, E01, E02, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)

    gamma_per = gamma_out;  gamma_per1 = gamma_out1
    w1, Trans1d  = rho_w1d( M, -4.0, 4.0, 901, B, i, E01, E02, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)

    gamma_per = gamma_out;  gamma_per1 = gamma_out1
    w1, Trans1ud = rho_w1ud(M, -4.0, 4.0, 901, B, i, E01, E02, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)

    gamma_per = gamma_out;  gamma_per1 = gamma_out1
    w1, Trans1du = rho_w1du(M, -4.0, 4.0, 901, B, i, E01, E02, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)

    # Grupo 2: gamma_per = 0, gamma_per1 = -gamma_out1 = -1.0
    # Invertir el signo de gamma_per1 es necesario para calcular correctamente
    # las contribuciones a Gz que involucran el acoplamiento inter-espín negado
    gamma_per  = (-1.0)*gamma_out    # = 0
    gamma_per1 = (-1.0)*gamma_out1   # = -1.0

    w1, Trans2u  = rho_w2u( M, -4.0, 4.0, 901, B, i, E01, E02, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)

    gamma_per = (-1.0)*gamma_out;  gamma_per1 = (-1.0)*gamma_out1
    w1, Trans2d  = rho_w2d( M, -4.0, 4.0, 901, B, i, E01, E02, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)

    gamma_per = (-1.0)*gamma_out;  gamma_per1 = (-1.0)*gamma_out1
    w1, Trans2ud = rho_w2ud(M, -4.0, 4.0, 901, B, i, E01, E02, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)

    gamma_per = (-1.0)*gamma_out;  gamma_per1 = (-1.0)*gamma_out1
    w1, Trans2du = rho_w2du(M, -4.0, 4.0, 901, B, i, E01, E02, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)

    end_time  = time.time()
    diff_time = end_time - start_time
    print(f"El tiempo de ejecución fue de {diff_time} segundos.")

    # ── Conductancia de espín Gz promediada sobre el desorden de Anderson ─────
    #
    # Gz(ω) = <|T↑↑|²> - <|T↓↑|²> + <|T↑↓|²> - <|T↓↓|²>  (c1 + c2)
    #
    # Cada Trans* ya es el promedio aritmético sobre M realizaciones:
    #   Trans1u  = <G^R_{N,1}^{(c1,xi=+1,↑)}>(ω)  → canal up conservando espín
    #   Trans1du = <G^R_{N,1}^{(c1,xi=-1,↓)}>(ω)  → canal down con inversión de xi
    #   Trans1ud = <G^R_{N,1}^{(c1,xi=-1,↑)}>(ω)  → canal up con inversión de xi
    #   Trans1d  = <G^R_{N,1}^{(c1,xi=+1,↓)}>(ω)  → canal down conservando espín
    #   (análogos para cadena 2)
    #
    # El módulo cuadrado <T>·<T>* es el cuadrado del promedio, NO el promedio
    # del cuadrado <|T|²>. Para M grande, la diferencia es despreciable si
    # las fluctuaciones son pequeñas (régimen difusivo). En régimen localizado,
    # <T>·<T>* puede subestimar la transmisión real.
    Trans = (Trans1u *np.conjugate(Trans1u)    # +|<T↑↑>|²  cadena 1
           - Trans1du*np.conjugate(Trans1du)   # -|<T↓↑>|²  cadena 1
           + Trans1ud*np.conjugate(Trans1ud)   # +|<T↑↓>|²  cadena 1
           - Trans1d *np.conjugate(Trans1d)    # -|<T↓↓>|²  cadena 1
           + Trans2u *np.conjugate(Trans2u)    # +|<T↑↑>|²  cadena 2
           - Trans2du*np.conjugate(Trans2du)   # -|<T↓↑>|²  cadena 2
           + Trans2ud*np.conjugate(Trans2ud)   # +|<T↑↓>|²  cadena 2
           - Trans2d *np.conjugate(Trans2d))   # -|<T↓↓>|²  cadena 2

    # ── Guardado de resultados ─────────────────────────────────────────────────
    # Paso 1: Guardar en archivo .dat con dos columnas (energía, Gz)
    A      = np.stack((w1, Trans.real), axis=1)
    nombre = f'trans_SO_disorder_Z_N={i}.dat'
    np.savetxt(nombre, A, fmt='%.6e')

    # Paso 2: Releer con pandas y exportar a .csv para facilitar análisis posterior
    # El archivo .dat tiene columnas separadas por espacios; pandas lo interpreta
    # con delimiter='\s+' (uno o más espacios).
    data01 = pd.read_csv(
        f'trans_SO_disorder_Z_N={i}.dat',
        delimiter=r'\s+',       # Separador: espacio(s)
        header=None,
        names=['E', 'Gz']       # Columna 1: energía [meV]; Columna 2: Gz(ω)
    )

    # Paso 3: Guardar como CSV (más portable, compatible con Excel/Google Sheets)
    data01.to_csv(f'data_disorder_N{i}.csv', index=False)
    # Archivo resultante: 'data_disorder_N91.csv' con cabecera 'E,Gz'

    # Líneas para montar en Google Drive (útil en Google Colab):
    # !cp data* drive/MyDrive/Rashba/.
