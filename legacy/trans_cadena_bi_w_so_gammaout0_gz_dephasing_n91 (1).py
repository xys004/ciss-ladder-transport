# -*- coding: utf-8 -*-
"""
================================================================================
trans_cadena_bi_W_SO_gammaout0_GZ_decoherence_N91.py
================================================================================

DESCRIPCIÓN GENERAL
-------------------
Este script calcula la conductancia de espín en la dirección z (Gz) como
función de la energía para una doble cadena lineal (bilineal) con interacción
espín-órbita tipo Rashba, en presencia de DECOHERENCIA (dephasing).

La decoherencia se implementa mediante el modelo de Büttiker de sondas de
desfase (dephasing probes / voltage probes): a cada sitio n de cada cadena se
le asigna un ensanchamiento imaginario aleatorio η_n que rompe la coherencia
cuántica de fase. Este η_n fluctúa entre realizaciones, simulando el efecto
estadístico del acoplamiento del sistema con un baño de dephasing.

DIFERENCIAS CLAVE RESPECTO AL CÓDIGO SIN DECOHERENCIA
------------------------------------------------------
1. ETA COMO ARRAY ALEATORIO (η_n por sitio):
   En el código anterior η era un escalar global pequeño (regularización
   puramente numérica). Aquí η[i] y η1[i] son vectores de N componentes
   independientes para cada realización i, generados aleatoriamente con
   la función Random(). Esto implementa el dephasing espacialmente
   heterogéneo (distinto en cada sitio).

2. CUATRO GAMMAS INDEPENDIENTES (Gamma1, Gamma2, Gamma3, Gamma4):
   En lugar de dos arrays de anchos de nivel (uno por cadena), ahora hay
   cuatro, uno por cada bloque de espín-xi:
     - Gamma1 → cadena 1, xi=+1, ↑  AND  cadena 2, xi=+1, ↑
     - Gamma2 → cadena 1, xi=+1, ↓  AND  cadena 2, xi=+1, ↓
     - Gamma3 → cadena 1, xi=-1, ↑  AND  cadena 2, xi=-1, ↑
     - Gamma4 → cadena 1, xi=-1, ↓  AND  cadena 2, xi=-1, ↓
   La asimetría val / val1 entre extremos permite modelar configuraciones
   PARALELAS (P) y ANTIPARALELAS (AP) de la polarización de los electrodos.

3. DOBLE BUCLE + PROMEDIO DE DESORDEN:
   Las funciones rho_w* tienen ahora un doble bucle:
     - Bucle externo: sobre la malla de energías ω
     - Bucle interno: sobre M realizaciones del desorden η
   El resultado en cada punto energético es el PROMEDIO sobre realizaciones,
   implementando el promedio de desorden <G^R>_disorder.

4. ACELERACIÓN JIT (Numba):
   Las funciones matrix_AR, matrix_AA y den_espectral* están decoradas con
   @jit (Just-In-Time compilation de Numba) para reducir el tiempo de cómputo,
   que es crítico dado el alto número de realizaciones (M = 10000).

MODELO DE DEPHASING: SONDAS DE BÜTTIKER
-----------------------------------------
En el modelo de Büttiker, la decoherencia de fase se introduce acoplando
cada sitio n a un electrodo ficticio "volátil" (voltage probe). El efecto
neto sobre la función de Green retardada es añadir un término imaginario
extra a la energía de cada sitio:

    G^R(ω) ~ 1 / (ω - ε_n + iΓ_electrodos + i·η_n)

donde η_n ~ η_d · r_n con r_n un número aleatorio de media cero.
La magnitud η_d (intensidad del dephasing) controla qué tan fuerte es la
decoherencia: η_d → 0 recupera el régimen coherente, η_d grande → régimen
difusivo/clásico.

CONFIGURACIONES DE ELECTRODOS (Paralela / Antiparalela)
---------------------------------------------------------
Los cuatro Gamma permiten definir la polarización relativa de los electrodos:
  val  = i·(1+p)·1j   → ancho del electrodo "paralelo"
  val1 = i·(1-p)·1j   → ancho del electrodo "antiparalelo"

  Gamma1 = [val,  0, ..., 0,  val ]  → P-P
  Gamma2 = [val,  0, ..., 0,  val1]  → P-AP
  Gamma3 = [val1, 0, ..., 0,  val ]  → AP-P
  Gamma4 = [val1, 0, ..., 0,  val1]  → AP-AP

Con p=0, val = val1 = i·1j, todos los electrodos son equivalentes (caso
no magnético). Variar p permite estudiar la magnetorresistencia de espín.

OBSERVABLE CALCULADO
---------------------
Conductancia de espín en z:

  Gz(ω) = <|T↑↑|²> - <|T↑↓|²> + <|T↓↑|²> - <|T↓↓|²>  (cadena 1 + cadena 2)

donde <·> denota promedio sobre M realizaciones del dephasing.
Cada T es la amplitud de la función de Green retardada en el último sitio.

PARÁMETROS DE ESTE SCRIPT (valores fijos en el código)
--------------------------------------------------------
  N          = 91       Número de sitios por cadena
  M          = 10000    Número de realizaciones de desorden
  gamma01/02 = 1.0 meV  Hopping intra-cadena (unidad de energía)
  l_R1/R2    = 0.1 meV  Fuerza de Rashba (λ_SOC = 0.1 meV)
  gamma_out1 = 1.0 meV  Acoplamiento a electrodos (γ_out = 1 meV)
  l_D        = 0.0      Dresselhaus apagado
  beta       = π        Desfase geométrico entre cadenas

ESTRUCTURA DEL SCRIPT
-----------------------
  1. Importaciones y parámetros globales
  2. matrix_AR: matriz de la función de Green retardada (con η por sitio)
  3. matrix_AA: matriz de la función de Green avanzada
  4. den_espectral*: extracción de amplitudes de transmisión
  5. rho_w*: barrido en energía con promedio sobre realizaciones
  6. Random(): generador de vectores de dephasing con media cero
  7. Bloque principal: cálculo de Gz, guardado y conversión a CSV

AUTOR: David Verrilli, Nelson Bolívar.
FECHA: 2024
REFERENCIA: Büttiker, M., Phys. Rev. B 33, 3020 (1986) – modelo de sondas
            de desfase. Haug & Jauho, "Quantum Kinetics in Transport and
            Optics of Semiconductors" – funciones de Green en transporte.
================================================================================
"""

# ─── Importaciones ────────────────────────────────────────────────────────────
import pandas as pd                       # Manipulación de datos y exportación a CSV
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve  # Resolución de sistemas lineales dispersos A·x=b
from scipy import sparse                  # Almacenamiento eficiente de matrices dispersas
from scipy.integrate import simps, trapz  # Integración numérica (importadas pero no usadas)
import random                             # Números aleatorios (no usado directamente)
import time                               # Medición de tiempos de ejecución
from numba import jit, cuda               # Aceleración JIT con Numba
from numpy import array

import warnings
warnings.filterwarnings("ignore")        # Suprime warnings de numpy/scipy


# ─── Parámetros globales ───────────────────────────────────────────────────────
nu    = 1                   # Factor de degeneración (no usado explícitamente)
muAB  = 1                   # Potencial químico relativo entre cadenas (no usado)
eta   = 0.0001              # Valor por defecto de η (sobrescrito en el bloque principal)
e     = 1                   # Carga del electrón (unidades naturales)
c     = 1                   # Velocidad de la luz (unidades naturales)
h     = 1                   # Constante de Planck (ℏ = 1)
phi_0 = c * h / e           # Cuanto de flujo φ₀ = hc/e

stepc  = 901                # Puntos en el barrido de energía
step   = 91                 # Longitud de cadena referencia (no usada aquí directamente)
stepD  = 9001               # Pasos para barridos de desorden (no usado aquí)
beta   = np.pi              # Desfase β = π de la SOI de Rashba en cadena 2


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: MATRICES DE LA FUNCIÓN DE GREEN CON DEPHASING
# ══════════════════════════════════════════════════════════════════════════════

@jit   # Compilación JIT de Numba: la primera llamada compila, las siguientes son rápidas
def matrix_AR(w, eta, eta1, E01, E02, N, theta,
              Gamma1, Gamma2, Gamma3, Gamma4,
              gamma01, gamma02, gamma_per, gamma_per1,
              U, M1, M2, l_R1, l_R2, l_D):
    """
    Construye la matriz del sistema lineal para la función de Green RETARDADA
    en presencia de dephasing espacialmente heterogéneo.

    La diferencia esencial con el código sin dephasing:
      - eta  : array de N floats → η_n para cada sitio de la cadena 1
      - eta1 : array de N floats → η_n para cada sitio de la cadena 2
      - Gamma1/2/3/4 : cuatro arrays independientes (uno por bloque xi-sigma)

    El término diagonal toma la forma:
        -(ω - E0_n + Γ_j[n] + i·η_n)
    donde η_n es la amplitud de dephasing del sitio n en esta realización.
    Cuando η_n = 0 para todos los sitios, se recupera el sistema coherente.

    Los parámetros Gamma1..4 permiten configuraciones de electrodos diferentes
    para cada canal de transporte (ver Sección 2 del encabezado del módulo).

    Parámetros
    ----------
    w         : float      → energía actual en el barrido
    eta       : array(N)   → dephasing por sitio para la cadena 1 (realización i)
    eta1      : array(N)   → dephasing por sitio para la cadena 2 (realización i)
    E01       : array(N)   → energías de sitio de la cadena 1 (= 0 aquí)
    E02       : array(N)   → energías de sitio de la cadena 2 (= 0 aquí)
    N         : int        → número de sitios por cadena
    theta     : float      → fase Aharonov-Bohm (= 0 aquí)
    Gamma1    : array(N)   → anchos de nivel del canal (xi=+1, ↑) en ambas cadenas
    Gamma2    : array(N)   → anchos de nivel del canal (xi=+1, ↓) en ambas cadenas
    Gamma3    : array(N)   → anchos de nivel del canal (xi=-1, ↑) en ambas cadenas
    Gamma4    : array(N)   → anchos de nivel del canal (xi=-1, ↓) en ambas cadenas
    gamma01   : float      → hopping a lo largo de la cadena 1
    gamma02   : float      → hopping a lo largo de la cadena 2
    gamma_per : float      → acoplamiento perpendicular intra-espín entre cadenas
    gamma_per1: float      → acoplamiento perpendicular inter-espín entre cadenas
    U         : array(N)   → potencial de sitio (desorden estático; = 0 aquí)
    M1, M2    : float      → masas efectivas (no usadas explícitamente)
    l_R1      : float      → fuerza Rashba en cadena 1
    l_R2      : float      → fuerza Rashba en cadena 2
    l_D       : float      → fuerza Dresselhaus (= 0 aquí)

    Retorna
    -------
    A : scipy.sparse.csr_matrix (8N × 8N, compleja)
        Operador [ω - H_eff + iΓ + i·η] para la función de Green retardada.

    Convención de bloques (idéntica al código sin dephasing):
      [0:N)   cadena 1, xi=+1, ↑    [4N:5N)  cadena 2, xi=+1, ↑
      [N:2N)  cadena 1, xi=+1, ↓    [5N:6N)  cadena 2, xi=+1, ↓
      [2N:3N) cadena 1, xi=-1, ↑    [6N:7N)  cadena 2, xi=-1, ↑
      [3N:4N) cadena 1, xi=-1, ↓    [7N:8N)  cadena 2, xi=-1, ↓
    """
    A = np.zeros((8*N, 8*N), dtype=complex)

    # Incremento de ángulo azimutal entre sitios (para la fase de Rashba)
    # NOTA: el denominador (10-1) está hardcodeado; debería ser N-1 en general
    Dphi = (2 * np.pi / (10 - 1))

    for n in range(1, N+1):
        # ── Diagonal principal ───────────────────────────────────────────────
        # Forma: -(ω - E0_n + Γ_canal[n] + i·η_n)
        #
        # NOVEDAD respecto al código coherente:
        #   • Cada bloque xi-sigma usa su propio Gamma (Gamma1, Gamma2, Gamma3, Gamma4)
        #     en lugar de un único Gamma1/Gamma2 por cadena.
        #   • El dephasing η[n-1] (cadena 1) o η1[n-1] (cadena 2) es específico
        #     de esta realización y varía sitio a sitio.
        l = (n-1) % N + 1   # Índice circular (= n aquí)

        # Cadena 1 – cuatro canales con sus respectivos Gamma y el mismo η de c1
        A[n-1,     l-1    ] = -(w - E01[n-1] + Gamma1[n-1] + eta[n-1]*1.0j)   # xi=+1, ↑
        A[n-1+N,   l-1+N  ] = -(w - E01[n-1] + Gamma2[n-1] + eta[n-1]*1.0j)   # xi=+1, ↓
        A[n-1+2*N, l-1+2*N] = -(w - E01[n-1] + Gamma3[n-1] + eta[n-1]*1.0j)   # xi=-1, ↑
        A[n-1+3*N, l-1+3*N] = -(w - E01[n-1] + Gamma4[n-1] + eta[n-1]*1.0j)   # xi=-1, ↓

        # Cadena 2 – mismos Gamma1..4 pero con el dephasing independiente η1 de c2
        A[n-1+4*N, l-1+4*N] = -(w - E02[n-1] + Gamma1[n-1] + eta1[n-1]*1.0j)  # xi=+1, ↑
        A[n-1+5*N, l-1+5*N] = -(w - E02[n-1] + Gamma2[n-1] + eta1[n-1]*1.0j)  # xi=+1, ↓
        A[n-1+6*N, l-1+6*N] = -(w - E02[n-1] + Gamma3[n-1] + eta1[n-1]*1.0j)  # xi=-1, ↑
        A[n-1+7*N, l-1+7*N] = -(w - E02[n-1] + Gamma4[n-1] + eta1[n-1]*1.0j)  # xi=-1, ↓

        # ── Acoplamiento perpendicular entre cadenas (intra-espín, mismo xi) ───
        # gamma_per: conecta los mismos canales entre cadena 1 y cadena 2
        A[n-1,     l-1+4*N] = gamma_per   # c1(xi=+1,↑) ↔ c2(xi=+1,↑)
        A[n-1+N,   l-1+5*N] = gamma_per   # c1(xi=+1,↓) ↔ c2(xi=+1,↓)
        A[n-1+2*N, l-1+6*N] = gamma_per   # c1(xi=-1,↑) ↔ c2(xi=-1,↑)
        A[n-1+3*N, l-1+7*N] = gamma_per   # c1(xi=-1,↓) ↔ c2(xi=-1,↓)

        A[n-1+4*N, l-1    ] = gamma_per   # Término hermítico
        A[n-1+5*N, l-1+N  ] = gamma_per
        A[n-1+6*N, l-1+2*N] = gamma_per
        A[n-1+7*N, l-1+3*N] = gamma_per

        # ── Acoplamiento perpendicular inter-espín con inversión de xi ─────────
        # gamma_per1: conecta cadenas con inversión del índice xi entre ellas
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
        pp = (n-1) * Dphi    # Fase angular acumulada hasta el sitio n

        # Vectores de hopping con condición de contorno abierta:
        # el último elemento es 0 → no hay hopping desde el extremo derecho
        tt1 = [*np.ones(N-1), 0]
        tt1 = np.multiply(tt1, gamma01)

        tt2 = [*np.ones(N-1), 0]
        tt2 = np.multiply(tt2, gamma02)

        # Hopping estándar (sin SOI) – diagonal en espín
        A[n-1,     l-1    ] += tt1[n-1]
        A[n-1+N,   l-1+N  ] += tt1[n-1]
        A[n-1+2*N, l-1+2*N] += tt1[n-1]
        A[n-1+3*N, l-1+3*N] += tt1[n-1]

        A[n-1+4*N, l-1+4*N] += tt2[n-1]
        A[n-1+5*N, l-1+5*N] += tt2[n-1]
        A[n-1+6*N, l-1+6*N] += tt2[n-1]
        A[n-1+7*N, l-1+7*N] += tt2[n-1]

        # Máscara de Rashba con condición de contorno abierta
        l_R11 = np.multiply([*np.ones(N-1), 0], l_R1)
        l_R12 = np.multiply([*np.ones(N-1), 0], l_R2)

        # SOI de Rashba y Dresselhaus hacia n+1 (cadena 1)
        # Mezcla de espín opuesto entre sitios vecinos con fase pp acumulada
        A[n-1,     l-1+2*N] = (-1j*l_R11[n-1]*np.exp(-pp*1j) + l_D*np.exp( pp*1j))  # xi=+1,↑ → xi=-1,↑
        A[n-1+N,   l-1+3*N] = (-1j*l_R11[n-1]*np.exp( pp*1j) - l_D*np.exp(-pp*1j))  # xi=+1,↓ → xi=-1,↓
        A[n-1+2*N, l-1    ] = (-1j*l_R11[n-1]*np.exp( pp*1j) - l_D*np.exp(-pp*1j))  # xi=-1,↑ → xi=+1,↑
        A[n-1+3*N, l-1+N  ] = (-1j*l_R11[n-1]*np.exp(-pp*1j) + l_D*np.exp( pp*1j))  # xi=-1,↓ → xi=+1,↓

        # SOI de Rashba hacia n+1 (cadena 2) con desfase adicional β = π
        A[n-1+4*N, l-1+6*N] = (-1j*l_R12[n-1]*np.exp(-pp*1j - beta*1j) + l_D*np.exp( pp*1j))
        A[n-1+5*N, l-1+7*N] = (-1j*l_R12[n-1]*np.exp( pp*1j + beta*1j) - l_D*np.exp(-pp*1j))
        A[n-1+6*N, l-1+4*N] = (-1j*l_R12[n-1]*np.exp( pp*1j + beta*1j) - l_D*np.exp(-pp*1j))
        A[n-1+7*N, l-1+5*N] = (-1j*l_R12[n-1]*np.exp(-pp*1j - beta*1j) + l_D*np.exp( pp*1j))

        # ══════════════════════════════════════════════════════════════════
        # HOPPING HACIA EL SITIO n-1 (vecino izquierdo)
        # ══════════════════════════════════════════════════════════════════
        l  = (n-2) % N + 1   # Índice del sitio n-1
        pp = (n-2) * Dphi    # Fase angular del sitio n-1 (para la SOI)

        # Condición de contorno abierta: el primer elemento es 0
        tt1 = [0, *np.ones(N-1)]
        tt1 = np.multiply(tt1, gamma01)

        tt2 = [0, *np.ones(N-1)]
        tt2 = np.multiply(tt2, gamma02)

        # Hopping estándar hacia n-1
        A[n-1,     l-1    ] += tt1[n-1]
        A[n-1+N,   l-1+N  ] += tt1[n-1]
        A[n-1+2*N, l-1+2*N] += tt1[n-1]
        A[n-1+3*N, l-1+3*N] += tt1[n-1]

        A[n-1+4*N, l-1+4*N] += tt2[n-1]
        A[n-1+5*N, l-1+5*N] += tt2[n-1]
        A[n-1+6*N, l-1+6*N] += tt2[n-1]
        A[n-1+7*N, l-1+7*N] += tt2[n-1]

        # Máscaras de Rashba para el hopping hacia n-1
        l_R111 = np.multiply([0, *np.ones(N-1)], l_R1)
        l_R122 = np.multiply([0, *np.ones(N-1)], l_R2)

        # SOI hacia n-1 – signo opuesto al hopping n→n+1 (hermicidad del Hamiltoniano)
        # Cadena 1:
        A[n-1,     l-1+2*N] += ( 1j*l_R111[n-1]*np.exp(-pp*1j) - l_D*np.exp( pp*1j))
        A[n-1+N,   l-1+3*N] += ( 1j*l_R111[n-1]*np.exp( pp*1j) + l_D*np.exp(-pp*1j))
        A[n-1+2*N, l-1    ] += ( 1j*l_R111[n-1]*np.exp( pp*1j) + l_D*np.exp(-pp*1j))
        A[n-1+3*N, l-1+N  ] += ( 1j*l_R111[n-1]*np.exp(-pp*1j) - l_D*np.exp( pp*1j))

        # Cadena 2 con desfase β:
        A[n-1+4*N, l-1+6*N] += ( 1j*l_R122[n-1]*np.exp(-pp*1j - beta*1j) - l_D*np.exp( pp*1j))
        A[n-1+5*N, l-1+7*N] += ( 1j*l_R122[n-1]*np.exp( pp*1j + beta*1j) + l_D*np.exp(-pp*1j))
        A[n-1+6*N, l-1+4*N] += ( 1j*l_R122[n-1]*np.exp( pp*1j + beta*1j) + l_D*np.exp(-pp*1j))
        A[n-1+7*N, l-1+5*N] += ( 1j*l_R122[n-1]*np.exp(-pp*1j - beta*1j) - l_D*np.exp( pp*1j))

    A = sparse.csr_matrix(A)
    return A


@jit
def matrix_AA(w, eta, eta1, E01, E02, N, theta,
              Gamma1, Gamma2, Gamma3, Gamma4,
              gamma01, gamma02, gamma_per, gamma_per1,
              U, M1, M2, l_R1, l_R2, l_D):
    """
    Construye la matriz del sistema lineal para la función de Green AVANZADA
    con dephasing espacialmente heterogéneo.

    Idéntica a matrix_AR salvo por el signo del término de dephasing:
        matrix_AR:  -(ω - E0 + Γ + i·η)   → G^R
        matrix_AA:  -(ω - E0 + Γ - i·η)   → G^A = [G^R]†

    El cambio de signo en η convierte la función retardada en la avanzada,
    tal como la regularización +iη → -iη lo hace en el límite coherente.

    Ver matrix_AR para la documentación completa de parámetros.
    """
    A = np.zeros((8*N, 8*N), dtype=complex)

    Dphi  = (2 * np.pi / (10 - 1))
    Dphi1 = (2 * np.pi / (10 - 1))  # Idéntico a Dphi; variable separada para extensibilidad

    for n in range(1, N+1):
        l = (n-1) % N + 1

        # ── Diagonal: MISMO que matrix_AR pero con -i·η (función avanzada) ─────
        A[n-1,     l-1    ] = -(w - E01[n-1] + Gamma1[n-1] - eta[n-1]*1.0j)
        A[n-1+N,   l-1+N  ] = -(w - E01[n-1] + Gamma2[n-1] - eta[n-1]*1.0j)
        A[n-1+2*N, l-1+2*N] = -(w - E01[n-1] + Gamma3[n-1] - eta[n-1]*1.0j)
        A[n-1+3*N, l-1+3*N] = -(w - E01[n-1] + Gamma4[n-1] - eta[n-1]*1.0j)

        A[n-1+4*N, l-1+4*N] = -(w - E02[n-1] + Gamma1[n-1] - eta1[n-1]*1.0j)
        A[n-1+5*N, l-1+5*N] = -(w - E02[n-1] + Gamma2[n-1] - eta1[n-1]*1.0j)
        A[n-1+6*N, l-1+6*N] = -(w - E02[n-1] + Gamma3[n-1] - eta1[n-1]*1.0j)
        A[n-1+7*N, l-1+7*N] = -(w - E02[n-1] + Gamma4[n-1] - eta1[n-1]*1.0j)

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

        pp = (n-1) * Dphi1   # Para cadena 2 (puede diferir si Dphi1 ≠ Dphi)
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

        # SOI hacia n-1 – cadena 1 (signo invertido respecto al hopping n→n+1)
        A[n-1,     l-1+2*N] += ( 1j*l_R111[n-1]*np.exp(-pp*1j) - l_D*np.exp( pp*1j))
        A[n-1+N,   l-1+3*N] += ( 1j*l_R111[n-1]*np.exp( pp*1j) + l_D*np.exp(-pp*1j))
        A[n-1+2*N, l-1    ] += ( 1j*l_R111[n-1]*np.exp( pp*1j) + l_D*np.exp(-pp*1j))
        A[n-1+3*N, l-1+N  ] += ( 1j*l_R111[n-1]*np.exp(-pp*1j) - l_D*np.exp( pp*1j))

        pp = (n-2) * Dphi1   # Fase para cadena 2 hacia n-1
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
# DIFERENCIA RESPECTO AL CÓDIGO SIN DEPHASING:
#   Las funciones aquí retornan +GR1[k] en lugar de -GR1[k].
#   Esto corresponde a una diferente convención de signo en el vector fuente B
#   o en la definición de la amplitud de transmisión. Físicamente ambas
#   convenciones son equivalentes ya que la transmisión usa |G|², pero
#   es importante ser consistente dentro del mismo código.
#
# Convención de índices en el vector solución GR1:
#   GR1[N-1]    → cadena 1, xi=+1, ↑,  último sitio   (bloque [0:N))
#   GR1[2N-1]   → cadena 1, xi=+1, ↓,  último sitio   (bloque [N:2N))
#   GR1[3N-1]   → cadena 1, xi=-1, ↑,  último sitio   (bloque [2N:3N))
#   GR1[4N-1]   → cadena 1, xi=-1, ↓,  último sitio   (bloque [3N:4N))
#   GR1[5N-1]   → cadena 2, xi=+1, ↑,  último sitio   (bloque [4N:5N))
#   GR1[6N-1]   → cadena 2, xi=+1, ↓,  último sitio   (bloque [5N:6N))
#   GR1[7N-1]   → cadena 2, xi=-1, ↑,  último sitio   (bloque [6N:7N))
#   GR1[8N-1]   → cadena 2, xi=-1, ↓,  último sitio   (bloque [7N:8N))
# ──────────────────────────────────────────────────────────────────────────────

@jit
def den_espectral1u(GR1, GA1, N):
    """
    Amplitud de transmisión: cadena 1, xi=+1, espín ↑ (canal 'up').
    Extrae GR1[N-1] → último sitio del bloque [0:N).
    Nota: signo positivo (diferente al código coherente que usaba -GR1).
    """
    return GR1[N-1]

@jit
def den_espectral2u(GR1, GA1, N):
    """
    Amplitud de transmisión: cadena 2, xi=+1, espín ↑.
    Extrae GR1[5N-1] → último sitio del bloque [4N:5N).
    """
    return GR1[5*N-1]

@jit
def den_espectral1d(GR1, GA1, N):
    """
    Amplitud de transmisión: cadena 1, xi=+1, espín ↓ (canal 'down').
    Extrae GR1[2N-1] → último sitio del bloque [N:2N).
    """
    return GR1[2*N-1]

@jit
def den_espectral2d(GR1, GA1, N):
    """
    Amplitud de transmisión: cadena 2, xi=+1, espín ↓.
    Extrae GR1[6N-1] → último sitio del bloque [5N:6N).
    """
    return GR1[6*N-1]

@jit
def den_espectral1ud(GR1, GA1, N):
    """
    Amplitud de transmisión con inversión de xi: cadena 1, xi=-1, espín ↑.
    Extrae GR1[3N-1] → último sitio del bloque [2N:3N).
    El sufijo 'ud' indica xi: +1→-1 ('up-to-down' en la dirección de propagación).
    """
    return GR1[3*N-1]

@jit
def den_espectral2ud(GR1, GA1, N):
    """
    Amplitud de transmisión con inversión de xi: cadena 2, xi=-1, espín ↑.
    Extrae GR1[7N-1] → último sitio del bloque [6N:7N).
    """
    return GR1[7*N-1]

@jit
def den_espectral1du(GR1, GA1, N):
    """
    Amplitud de transmisión con inversión de xi: cadena 1, xi=-1, espín ↓.
    Extrae GR1[4N-1] → último sitio del bloque [3N:4N).
    El sufijo 'du' indica xi: +1→-1, espín ↓.
    """
    return GR1[4*N-1]

@jit
def den_espectral2du(GR1, GA1, N):
    """
    Amplitud de transmisión con inversión de xi: cadena 2, xi=-1, espín ↓.
    Extrae GR1[8N-1] → último sitio del bloque [7N:8N).
    """
    return GR1[8*N-1]


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: BARRIDO EN ENERGÍA CON PROMEDIO DE DESORDEN
# ══════════════════════════════════════════════════════════════════════════════
#
# ESTRUCTURA GENERAL DE CADA FUNCIÓN rho_w*:
# ------------------------------------------
# Todas las funciones siguen el mismo patrón de doble bucle:
#
#   Para cada energía ω en la malla w1:
#       Den_realizacion = []
#       Para cada realización i = 0..M-1:
#           Construir A^R con eta[i] (dephasing de esta realización)
#           Resolver A^R · G^R = B  →  G^R
#           Construir A^A con eta[i]
#           Resolver A^A · G^A = B  →  G^A
#           Extraer amplitud de transmisión T_i del canal correspondiente
#           Acumular T_i en Den_realizacion
#       Promediar: <T>(ω) = mean(Den_realizacion)
#       Acumular <T>(ω) en la salida
#
# El promedio sobre M realizaciones implementa el promedio de desorden:
#   <G^R(ω)>_η = (1/M) Σ_{i=1}^{M} G^R(ω; {η_n^(i)})
#
# NOTA SOBRE EFICIENCIA:
#   El triple bucle implícito (ω × M × construcción de matriz 8N×8N) es
#   computacionalmente intensivo. Para N=91 y M=10000, son ~9 millones de
#   resoluciones de sistemas lineales de tamaño 728×728. El @jit de Numba
#   en las funciones de construcción de matrices ayuda a reducir el tiempo,
#   pero el cálculo puede durar varias horas.
# ──────────────────────────────────────────────────────────────────────────────

def rho_w1u(M, eta, eta1, w_0, w_f, nw, B, N, E01, E02, theta,
            Gamma1, Gamma2, Gamma3, Gamma4,
            gamma01, gamma02, gamma_per, gamma_per1,
            U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido energético con promedio de desorden: cadena 1, xi=+1, espín ↑.

    Parámetros
    ----------
    M    : int              → número de realizaciones del dephasing
    eta  : list de M arrays → cada eta[i] es un array(N) de dephasing para c1
    eta1 : list de M arrays → cada eta1[i] es un array(N) de dephasing para c2
    Los demás parámetros son los del Hamiltoniano (ver matrix_AR).

    Retorna
    -------
    w1   : array(nw) → malla de energías
    Den02: array(nw, complex) → <G^R_{N,1}^{(c1,xi=+1,↑)}>(ω) promediado sobre M realizaciones
    """
    w1    = np.linspace(w_0, w_f, nw)
    Den02 = np.array([], dtype=complex)    # Acumulador del promedio sobre ω

    for w0 in w1:
        Den01 = np.array([], dtype=complex)    # Acumulador de realizaciones para este ω

        for i in range(M):   # Bucle sobre M realizaciones del dephasing
            w   = w0
            # Función de Green retardada con el dephasing de la realización i
            AR1 = matrix_AR(w, eta[i], eta1[i], E01, E02, N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GR1 = np.asarray(spsolve(AR1, B)).ravel()

            # Función de Green avanzada con el mismo dephasing
            AA1 = matrix_AA(w, eta[i], eta1[i], E01, E02, N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GA1 = np.asarray(spsolve(AA1, B)).ravel()

            # Extraer la amplitud T_i para este canal y acumularla
            Den01 = np.append(Den01, den_espectral1u(GR1, GA1, N))

        # Promedio sobre las M realizaciones: <T>(ω) = (1/M) Σ T_i
        Den02 = np.append(Den02, np.mean(Den01))

    return w1, Den02


def rho_w2u(M, eta, eta1, w_0, w_f, nw, B, N, E01, E02, theta,
            Gamma1, Gamma2, Gamma3, Gamma4,
            gamma01, gamma02, gamma_per, gamma_per1,
            U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido energético con promedio de desorden: cadena 2, xi=+1, espín ↑.
    Extrae GR1[5N-1] en cada realización y promedia.
    """
    w1    = np.linspace(w_0, w_f, nw)
    Den02 = np.array([], dtype=complex)

    for w0 in w1:
        Den01 = np.array([], dtype=complex)
        for i in range(M):
            w   = w0
            AR1 = matrix_AR(w, eta[i], eta1[i], E01, E02, N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GR1 = np.asarray(spsolve(AR1, B)).ravel()

            AA1 = matrix_AA(w, eta[i], eta1[i], E01, E02, N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GA1 = np.asarray(spsolve(AA1, B)).ravel()

            Den01 = np.append(Den01, den_espectral2u(GR1, GA1, N))
        Den02 = np.append(Den02, np.mean(Den01))

    return w1, Den02


def rho_w1d(M, eta, eta1, w_0, w_f, nw, B, N, E01, E02, theta,
            Gamma1, Gamma2, Gamma3, Gamma4,
            gamma01, gamma02, gamma_per, gamma_per1,
            U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido energético con promedio de desorden: cadena 1, xi=+1, espín ↓.
    Extrae GR1[2N-1] en cada realización y promedia.
    """
    w1    = np.linspace(w_0, w_f, nw)
    Den02 = np.array([], dtype=complex)

    for w0 in w1:
        Den01 = np.array([], dtype=complex)
        for i in range(M):
            w   = w0
            AR1 = matrix_AR(w, eta[i], eta1[i], E01, E02, N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GR1 = np.asarray(spsolve(AR1, B)).ravel()

            AA1 = matrix_AA(w, eta[i], eta1[i], E01, E02, N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GA1 = np.asarray(spsolve(AA1, B)).ravel()

            Den01 = np.append(Den01, den_espectral1d(GR1, GA1, N))
        Den02 = np.append(Den02, np.mean(Den01))

    return w1, Den02


def rho_w2d(M, eta, eta1, w_0, w_f, nw, B, N, E01, E02, theta,
            Gamma1, Gamma2, Gamma3, Gamma4,
            gamma01, gamma02, gamma_per, gamma_per1,
            U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido energético con promedio de desorden: cadena 2, xi=+1, espín ↓.
    Extrae GR1[6N-1] en cada realización y promedia.
    """
    w1    = np.linspace(w_0, w_f, nw)
    Den02 = np.array([], dtype=complex)

    for w0 in w1:
        Den01 = np.array([], dtype=complex)
        for i in range(M):
            w   = w0
            AR1 = matrix_AR(w, eta[i], eta1[i], E01, E02, N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GR1 = np.asarray(spsolve(AR1, B)).ravel()

            AA1 = matrix_AA(w, eta[i], eta1[i], E01, E02, N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GA1 = np.asarray(spsolve(AA1, B)).ravel()

            Den01 = np.append(Den01, den_espectral2d(GR1, GA1, N))
        Den02 = np.append(Den02, np.mean(Den01))

    return w1, Den02


def rho_w1ud(M, eta, eta1, w_0, w_f, nw, B, N, E01, E02, theta,
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
            AR1 = matrix_AR(w, eta[i], eta1[i], E01, E02, N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GR1 = np.asarray(spsolve(AR1, B)).ravel()

            AA1 = matrix_AA(w, eta[i], eta1[i], E01, E02, N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GA1 = np.asarray(spsolve(AA1, B)).ravel()

            Den1 = np.append(Den1, den_espectral1ud(GR1, GA1, N))
        Den2 = np.append(Den2, np.mean(Den1))

    return w1, Den2


def rho_w2ud(M, eta, eta1, w_0, w_f, nw, B, N, E01, E02, theta,
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
            AR1 = matrix_AR(w, eta[i], eta1[i], E01, E02, N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GR1 = np.asarray(spsolve(AR1, B)).ravel()

            AA1 = matrix_AA(w, eta[i], eta1[i], E01, E02, N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GA1 = np.asarray(spsolve(AA1, B)).ravel()

            Den1 = np.append(Den1, den_espectral2ud(GR1, GA1, N))
        Den2 = np.append(Den2, np.mean(Den1))

    return w1, Den2


def rho_w1du(M, eta, eta1, w_0, w_f, nw, B, N, E01, E02, theta,
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
            AR1 = matrix_AR(w, eta[i], eta1[i], E01, E02, N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GR1 = np.asarray(spsolve(AR1, B)).ravel()

            AA1 = matrix_AA(w, eta[i], eta1[i], E01, E02, N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GA1 = np.asarray(spsolve(AA1, B)).ravel()

            Den1 = np.append(Den1, den_espectral1du(GR1, GA1, N))
        Den2 = np.append(Den2, np.mean(Den1))

    return w1, Den2


def rho_w2du(M, eta, eta1, w_0, w_f, nw, B, N, E01, E02, theta,
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
            AR1 = matrix_AR(w, eta[i], eta1[i], E01, E02, N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GR1 = np.asarray(spsolve(AR1, B)).ravel()

            AA1 = matrix_AA(w, eta[i], eta1[i], E01, E02, N, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)
            GA1 = np.asarray(spsolve(AA1, B)).ravel()

            Den1 = np.append(Den1, den_espectral2du(GR1, GA1, N))
        Den2 = np.append(Den2, np.mean(Den1))

    return w1, Den2


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: GENERADOR DE REALIZACIONES DE DEPHASING
# ══════════════════════════════════════════════════════════════════════════════

def Random(i):
    """
    Genera un vector aleatorio de N = i componentes con MEDIA EXACTAMENTE CERO.

    El vector se usa como η_n (dephasing por sitio) en una realización del
    modelo de Büttiker. La amplitud de cada componente está acotada en [-0.5, 0.5]
    (distribución uniforme), y la restricción de media cero garantiza que el
    dephasing no desplace el espectro de energía del sistema.

    Implementación:
        1. Genera i números uniformes en [0, 1) y les resta 0.5 → rango [-0.5, 0.5]
        2. Corrige el último elemento para que la suma sea exactamente cero:
           η_i[-1] += -Σ(η_i)
           Esto introduce una pequeña correlación en el último sitio, pero para
           i >> 1 el efecto es despreciable.

    Parámetro
    ---------
    i : int → número de sitios (= N)

    Retorna
    -------
    eta001 : array(i) → dephasing aleatorio con media cero, ∈ [-0.5, 0.5]

    NOTA SOBRE LA ESCALA DE DEPHASING:
    La amplitud física del dephasing se controla EXTERNAMENTE multiplicando
    este vector por un factor η_d antes de pasarlo a matrix_AR. En el bloque
    principal de este script, el vector se usa directamente (sin escalar),
    por lo que η_d efectivo ~ 0.5. Para estudiar η_d como parámetro, se
    debe escalar: eta_scaled = eta_d * Random(N).
    """
    eta001 = np.random.rand(i) - 0.5    # Distribución uniforme en [-0.5, 0.5]
    eta001[-1] += -sum(eta001)           # Corrección para media exactamente cero
    return eta001


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: BLOQUE PRINCIPAL – CÁLCULO DE Gz CON DEPHASING
# ══════════════════════════════════════════════════════════════════════════════

# ── Parámetros físicos del sistema ────────────────────────────────────────────
gamma01 = 1.0       # Hopping intra-cadena 1 (unidad de energía, e.g. meV)
gamma02 = 1.0       # Hopping intra-cadena 2
l_R1    = 0.1       # Fuerza de Rashba en cadena 1: λ_SOC = 0.1 meV
l_R12   = 0.0       # Variable auxiliar (no usada directamente)
l_R2    = 0.1       # Fuerza de Rashba en cadena 2
l_EO1   = 0.0       # Zeeman efectivo cadena 1 (no implementado)
l_EO2   = 0.0       # Zeeman efectivo cadena 2 (no implementado)
l_D     = 0.0       # Dresselhaus apagado
l_Z     = 0.0       # Zeeman apagado
eta     = 0.000001  # Valor global de η (reemplazado por arrays aleatorios en el bucle)

# ── Parámetros del promedio de desorden ──────────────────────────────────────
M = 10000           # Número de realizaciones del dephasing
                    # ADVERTENCIA: M=10000 con N=91 es muy costoso computacionalmente.
                    # Para pruebas rápidas reducir a M=100 o M=1000.

# ── Acoplamiento a electrodos ─────────────────────────────────────────────────
gamma_out  = 0.0    # Acoplamiento perpendicular intra-espín (apagado)
gamma_out1 = 1.0    # Acoplamiento perpendicular inter-espín: γ_out = 1.0 meV

# Arrays de realizaciones de dephasing (pre-generados fuera del bucle de N)
# Cada eta[i] y eta1[i] es un array de N floats con media cero.
# Se generan aquí como listas vacías y se rellenan dentro del bucle sobre 'list'.
eta01 = np.array([], dtype=float)   # Variable auxiliar (no usada finalmente)
eta02 = np.array([], dtype=float)   # Variable auxiliar (no usada finalmente)

# ── Bucle sobre longitudes de cadena ─────────────────────────────────────────
# list contiene los valores de N a calcular. Actualmente sólo N=91.
# Para calcular varias longitudes basta con extender la lista: [10, 20, 50, 91].
# NOTA: el nombre 'list' sobreescribe el built-in de Python; mejor renombrar
#       a 'N_list' o 'chain_lengths' para evitar conflictos.
list_N = [91]

p = 0.0   # Asimetría de los electrodos. p=0: configuración simétrica (no magnética).
          # p≠0: electrodo izquierdo más ancho que el derecho (configura polarización).

for i in list_N:
    # ── Parámetros dependientes de N ─────────────────────────────────────────
    N   = i
    E01 = np.zeros(i)   # Energías de sitio uniformes (banda plana) en cadena 1
    E02 = np.zeros(i)   # Energías de sitio uniformes en cadena 2

    # Generación de M realizaciones del dephasing para cadena 1 y cadena 2.
    # eta[k]  = array(N) con η_n^(k) para la k-ésima realización de la cadena 1
    # eta1[k] = array(N) con η_n^(k) para la k-ésima realización de la cadena 2
    # Las cadenas tienen dephasing INDEPENDIENTE entre sí (eta ≠ eta1).
    eta  = [Random(i) for _ in range(M)]
    eta1 = [Random(i) for _ in range(M)]

    theta = 0       # Sin flujo Aharonov-Bohm
    M1    = 1.0     # Masa efectiva cadena 1
    M2    = 1.0     # Masa efectiva cadena 2
    U     = np.zeros(i)   # Sin desorden estático

    # ── Vector fuente B ───────────────────────────────────────────────────────
    # Excita los 4 canales de entrada del primer sitio (n=1):
    #   B[0]  = -1 → cadena 1, xi=+1, ↑,  sitio 1
    #   B[i]  = -1 → cadena 1, xi=+1, ↓,  sitio 1
    #   B[4i] = -1 → cadena 2, xi=+1, ↑,  sitio 1
    #   B[5i] = -1 → cadena 2, xi=+1, ↓,  sitio 1
    B       = np.zeros(8*i, dtype=complex)
    B[0]    = -1
    B[i]    = -1
    B[4*i]  = -1
    B[5*i]  = -1

    # ── Anchos de nivel de los electrodos (cuatro canales independientes) ─────
    # val  = i·(1+p) → electrodo "paralelo" (el hopping al lead izquierdo y derecho es val)
    # val1 = i·(1-p) → electrodo "antiparalelo"
    # Con p=0: val = val1 = i → configuración no magnética (todos los electrodos iguales)
    val  = 1.0*(1+p)*1j   # Ancho del electrodo paralelo
    val1 = 1.0*(1-p)*1j   # Ancho del electrodo antiparalelo

    # Cada Gamma_j es no nulo SÓLO en los extremos de la cadena (n=1 y n=N),
    # modelando el acoplamiento a los electrodos izquierdo y derecho.
    # La asimetría entre extremos permite configurar electrodos P o AP:
    #   Gamma1: [val , 0..0, val ] → P-P   (xi=+1, ↑)
    #   Gamma2: [val , 0..0, val1] → P-AP  (xi=+1, ↓)
    #   Gamma3: [val1, 0..0, val ] → AP-P  (xi=-1, ↑)
    #   Gamma4: [val1, 0..0, val1] → AP-AP (xi=-1, ↓)
    Gamma1 = [val,  *np.zeros(i-2), val ]
    Gamma2 = [val,  *np.zeros(i-2), val1]
    Gamma3 = [val1, *np.zeros(i-2), val ]
    Gamma4 = [val1, *np.zeros(i-2), val1]

    # ── Medición del tiempo de ejecución ─────────────────────────────────────
    start_time = time.time()

    # ── Cálculo de los 8 canales de transmisión ───────────────────────────────
    # Se calculan con gamma_per = 0, gamma_per1 = ±gamma_out1.
    # La inversión de signo de gamma_per1 en el grupo 2 es necesaria para
    # construir la fórmula de Gz con los términos cruzados adecuados.

    # Grupo 1: gamma_per = 0, gamma_per1 = +gamma_out1
    gamma_per  = gamma_out      # = 0
    gamma_per1 = gamma_out1     # = 1.0

    w1, Trans1u  = rho_w1u( M, eta, eta1, -4.0, 4.0, 901, B, i, E01, E02, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)

    gamma_per = gamma_out;  gamma_per1 = gamma_out1
    w1, Trans1d  = rho_w1d( M, eta, eta1, -4.0, 4.0, 901, B, i, E01, E02, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)

    gamma_per = gamma_out;  gamma_per1 = gamma_out1
    w1, Trans1ud = rho_w1ud(M, eta, eta1, -4.0, 4.0, 901, B, i, E01, E02, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)

    gamma_per = gamma_out;  gamma_per1 = gamma_out1
    w1, Trans1du = rho_w1du(M, eta, eta1, -4.0, 4.0, 901, B, i, E01, E02, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)

    # Grupo 2: gamma_per = 0, gamma_per1 = -gamma_out1
    # La negación de gamma_per1 permite capturar la contribución del canal
    # con acoplamiento perpendicular de signo opuesto, necesaria para la
    # completitud de la fórmula de conductancia de espín Gz.
    gamma_per  = (-1.0)*gamma_out    # = 0 (sin cambio efectivo cuando gamma_out=0)
    gamma_per1 = (-1.0)*gamma_out1   # = -1.0

    w1, Trans2u  = rho_w2u( M, eta, eta1, -4.0, 4.0, 901, B, i, E01, E02, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)

    gamma_per = (-1.0)*gamma_out;  gamma_per1 = (-1.0)*gamma_out1
    w1, Trans2d  = rho_w2d( M, eta, eta1, -4.0, 4.0, 901, B, i, E01, E02, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)

    gamma_per = (-1.0)*gamma_out;  gamma_per1 = (-1.0)*gamma_out1
    w1, Trans2ud = rho_w2ud(M, eta, eta1, -4.0, 4.0, 901, B, i, E01, E02, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)

    gamma_per = (-1.0)*gamma_out;  gamma_per1 = (-1.0)*gamma_out1
    w1, Trans2du = rho_w2du(M, eta, eta1, -4.0, 4.0, 901, B, i, E01, E02, theta,
                            Gamma1, Gamma2, Gamma3, Gamma4,
                            gamma01, gamma02, gamma_per, gamma_per1,
                            U, M1, M2, l_R1, l_R2, l_D)

    end_time  = time.time()
    diff_time = end_time - start_time
    print(f"El tiempo de ejecución fue de {diff_time} segundos.")

    # ── Conductancia de espín Gz promediada sobre el desorden ─────────────────
    # Gz(ω) = <|T↑↑|²> - <|T↑↓|²> + <|T↓↑|²> - <|T↓↓|²>  (cadenas 1 y 2)
    #
    # donde cada término <|T|²> = T · T* ya es el promedio sobre realizaciones
    # (porque cada Trans* fue calculado como <T>(ω) = mean sobre M realizaciones).
    #
    # Los canales 'u' (up, ↑) contribuyen con signo + y los 'd' (down, ↓) con −,
    # proyectando la corriente de transmisión sobre el eje z del espín.
    Trans = (Trans1u *np.conjugate(Trans1u)    # <T↑↑>·<T↑↑>* (cadena 1, xi=+1)
           - Trans1du*np.conjugate(Trans1du)   # -<T↓↑>·<T↓↑>* (cadena 1, xi=-1, ↓)
           + Trans1ud*np.conjugate(Trans1ud)   # +<T↑↓>·<T↑↓>* (cadena 1, xi=-1, ↑)
           - Trans1d *np.conjugate(Trans1d)    # -<T↓↓>·<T↓↓>* (cadena 1, xi=+1, ↓)
           + Trans2u *np.conjugate(Trans2u)    # +<T↑↑>·<T↑↑>* (cadena 2, xi=+1)
           - Trans2du*np.conjugate(Trans2du)   # -<T↓↑>·<T↓↑>* (cadena 2, xi=-1, ↓)
           + Trans2ud*np.conjugate(Trans2ud)   # +<T↑↓>·<T↑↓>* (cadena 2, xi=-1, ↑)
           - Trans2d *np.conjugate(Trans2d))   # -<T↓↓>·<T↓↓>* (cadena 2, xi=+1, ↓)

    # ── Guardado de resultados ─────────────────────────────────────────────────
    # Paso 1: Guardar en archivo de texto con dos columnas: (energía, Gz)
    A      = np.stack((w1, Trans.real), axis=1)
    nombre = f'trans_SO_decoheren_Z_N={i}.dat'
    np.savetxt(nombre, A, fmt='%.6e')

    # Paso 2: Releer como DataFrame de pandas para conversión a CSV
    # Esto permite usar las herramientas de análisis de pandas y facilita
    # la compatibilidad con Google Colab y Google Drive.
    data01 = pd.read_csv(
        f'trans_SO_decoheren_Z_N={i}.dat',
        delimiter=r'\s+',       # Separador: uno o más espacios
        header=None,
        names=['E', 'Gz']       # Columnas: energía (E) y conductancia de espín z (Gz)
    )

    # Paso 3: Exportar a formato CSV (más portable y fácil de leer con pandas/Excel)
    data01.to_csv(f'data_decoheren_N{i}.csv', index=False)
    # El archivo resultante tiene cabecera 'E,Gz' y una fila por punto energético.

    # Líneas comentadas para montar en Google Drive (útil en entorno Colab):
    # !cp data* drive/MyDrive/Rashba/.
