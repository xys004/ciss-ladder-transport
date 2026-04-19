================================================================================
trans_cadena_bi_W_SO_gammaout0.py
================================================================================

DESCRIPCIÓN GENERAL
-------------------
Este script calcula la transmisión de carga y de espín a través de una
doble cadena lineal (bilineal) con interacción espín-órbita tipo Rashba y
Dresselhaus. Se usa el formalismo de funciones de Green retardada y avanzada,
construidas mediante el método de la ecuación de movimiento (EOM, Equation
of Motion).

MODELO FÍSICO
-------------
Se consideran dos cadenas de N sitios cada una (cadena 1 y cadena 2),
acopladas entre sí de forma perpendicular al eje de la cadena. Cada sitio
tiene dos grados de libertad de espín (↑ y ↓), y cada espín puede propagarse
en dos direcciones (de derecha a izquierda, índice xi = ±1). El espacio de
Hilbert del sistema tiene por tanto dimensión 8N.

El índice del estado de un sitio n en la matriz sigue la convención:
  - Bloque [0   : N  ) → cadena 1, xi=+1, espín ↑  (sigma=+1)
  - Bloque [N   : 2N ) → cadena 1, xi=+1, espín ↓  (sigma=-1)
  - Bloque [2N  : 3N ) → cadena 1, xi=-1, espín ↑
  - Bloque [3N  : 4N ) → cadena 1, xi=-1, espín ↓
  - Bloque [4N  : 5N ) → cadena 2, xi=+1, espín ↑
  - Bloque [5N  : 6N ) → cadena 2, xi=+1, espín ↓
  - Bloque [6N  : 7N ) → cadena 2, xi=-1, espín ↑
  - Bloque [7N  : 8N ) → cadena 2, xi=-1, espín ↓

INTERACCIÓN ESPÍN-ÓRBITA (SOI)
-------------------------------
Se incluyen dos mecanismos de SOI:
  1. Rashba (l_R1 para cadena 1, l_R2 para cadena 2): acopla estados de distinto
     espín entre sitios vecinos con una fase que depende de la dirección de
     propagación. La fase de Rashba incluye un ángulo acumulado pp = (n-1)*Δφ.
  2. Dresselhaus (l_D): también acopla espines opuestos, con signo opuesto al
     Rashba en la fase exponencial. En esta versión del código, l_D = 0.

Adicionalmente, la cadena 2 lleva un desfase extra β = π en la exponencial
de la SOI de Rashba, lo que permite modelar geometrías físicas distintas
para cada cadena.

ACOPLAMIENTO ENTRE CADENAS (perpendicular)
-------------------------------------------
El acoplamiento perpendicular (entre las dos cadenas) se parametriza mediante:
  - gamma_per  : acoplamiento intra-espín (cadena 1 ↔ cadena 2, mismo espín)
  - gamma_per1 : acoplamiento inter-espín con inversión de xi entre cadenas

ELECTRODOS Y ANCHOS DE NIVEL (Gamma)
-------------------------------------
Los electrodos se modelan de forma efectiva mediante términos de amortiguamiento
imaginarios Gamma1 y Gamma2 (anchos de nivel), que sólo son no nulos en los
sitios extremos (n=1 y n=N) de cada cadena, reproduciendo el acoplamiento
lead-izquierda / lead-derecha del formalismo de Landauer-Büttiker.

TRANSMISIÓN Y CONDUCTANCIA DE ESPÍN
-------------------------------------
La conductancia diferencial de carga (G₀), y las componentes del tensor de
conductancia de espín (Gx, Gy, Gz) se calculan como combinaciones cuadráticas
de elementos de la función de Green retardada evaluada en el último sitio de
la cadena (n=N), siguiendo las fórmulas de Meir-Wingreen / Landauer.

  - G₀ (carga):  suma de |G|² sobre todos los canales de espín
  - Gz (espín z): diferencia |G↑↑|² - |G↓↓|² + términos de cadena 2
  - Gx (espín x): parte real de los términos cruzados G↑↓ + G↓↑
  - Gy (espín y): parte imaginaria de los términos cruzados

ESTRUCTURA DEL SCRIPT
-----------------------
  1. Parámetros globales
  2. Función matrix_AR: construye la matriz de la función de Green retardada
  3. Función matrix_AA: construye la matriz de la función de Green avanzada
  4. Funciones den_espectral*: extraen el elemento de la FG relevante
  5. Funciones rho_w*: barren en energía y calculan la FG
  6. Bloques de ejecución: calculan G₀, Gz, Gx, Gy para distintos parámetros

AUTOR: David Verrilli, Nelson Bolívar.
FECHA: 2024
REFERENCIA: Basado en el formalismo de ecuación de movimiento para funciones
            de Green fuera del equilibrio. Ver, e.g., Haug & Jauho,
            "Quantum Kinetics in Transport and Optics of Semiconductors".
================================================================================
"""

# ─── Importaciones ────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve  # Resuelve sistemas lineales dispersos: A·x = b
from scipy import sparse                  # Almacenamiento eficiente de matrices dispersas
from scipy.integrate import simps, trapz # Integración numérica (no usada directamente aquí)
import random                             # Generación de números aleatorios (no usado aquí)
import time                               # Medición de tiempos de ejecución
from numba import jit, cuda               # Aceleración JIT (importado pero no usado en este archivo)
from numpy import array

import warnings
warnings.filterwarnings("ignore")        # Suprime warnings de numpy/scipy durante la ejecución

# ─── Parámetros globales ───────────────────────────────────────────────────────
# Estos valores son constantes físicas o parámetros de control numérico globales.
# Algunos están comentados porque se definen localmente en cada bloque de cálculo.

nu    = 1                   # Factor de degeneración 
muAB  = 1                   # Potencial químico relativo entre cadenas (no usado directamente)
eta   = 0.0001              # Parámetro de regularización: pequeña parte imaginaria
                            # que garantiza la convergencia de la función de Green
                            # retardada (G^R ~ 1/(ω - H + iη))
e     = 1                   # Carga del electrón (unidades naturales, e=1)
c     = 1                   # Velocidad de la luz (unidades naturales)
h     = 1                   # Constante de Planck reducida (ℏ=1)
phi_0 = c * h / e           # Cuanto de flujo magnético φ₀ = hc/e

stepc  = 901                # Número de puntos en el barrido de energía (variable central)
step   = 91                 # Número de pasos para barridos secundarios (no usado aquí)
stepD  = 901                # Pasos para barridos de desorden (no usado aquí)
beta   = np.pi              # Desfase adicional en la SOI de Rashba de la cadena 2 (β = π)
                            # Este parámetro diferencia la fase de Rashba entre cadena 1 y 2


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: CONSTRUCCIÓN DE LAS MATRICES DE LA FUNCIÓN DE GREEN
# ══════════════════════════════════════════════════════════════════════════════

def matrix_AR(w, E01, E02, N, theta, Gamma1, Gamma2,
              gamma01, gamma02, gamma_per, gamma_per1,
              U, M1, M2, l_R1, l_R2, l_D):
    """
    Construye la matriz del sistema lineal para la función de Green RETARDADA.

    La función de Green retardada satisface:
        [ω - H + iΓ + iη] G^R = I

    que se reescribe como el sistema lineal:
        A^R · G^R = I   →   A^R · (columna de G^R) = (vector fuente B)

    Parámetros
    ----------
    w         : float   → energía (frecuencia) actual en el barrido
    E01       : array(N)→ energías de sitio en la cadena 1 (nivel 0 de cada sitio)
    E02       : array(N)→ energías de sitio en la cadena 2
    N         : int     → número de sitios por cadena
    theta     : float   → fase de Aharonov-Bohm (flujo magnético); actualmente = 0
    Gamma1    : array(N)→ anchos de nivel (imaginarios) de la cadena 1.
                          Sólo son no nulos en los extremos (n=1 y n=N),
                          modelando el acoplamiento a los electrodos izquierdo/derecho.
    Gamma2    : array(N)→ ídem para la cadena 2
    gamma01   : float   → hopping (salto entre vecinos) a lo largo de la cadena 1
    gamma02   : float   → hopping a lo largo de la cadena 2
    gamma_per : float   → acoplamiento intra-espín perpendicular entre cadenas
                          (cadena 1 ↔ cadena 2, mismo espín, mismo xi)
    gamma_per1: float   → acoplamiento inter-espín perpendicular entre cadenas
                          (cadena 1 ↔ cadena 2, diferente xi)
    U         : array(N)→ potencial de sitio (desorden estático); actualmente = 0
    M1, M2    : float   → masas efectivas de las cadenas (no usadas explícitamente aquí)
    l_R1      : float   → fuerza de la interacción Rashba en la cadena 1
    l_R2      : float   → fuerza de la interacción Rashba en la cadena 2
    l_D       : float   → fuerza de la interacción Dresselhaus (actualmente = 0)

    Retorna
    -------
    A : scipy.sparse.csr_matrix de forma (8N × 8N) y tipo complejo
        Representa el operador [ω - H_eff + iη] en el espacio de Hilbert
        del sistema bilineal.

    Estructura de bloques
    ---------------------
    La matriz A tiene dimensión 8N × 8N, dividida en 8 bloques de N×N:
      Filas/columnas [0:N)   → cadena 1, xi=+1, espín ↑
      Filas/columnas [N:2N)  → cadena 1, xi=+1, espín ↓
      Filas/columnas [2N:3N) → cadena 1, xi=-1, espín ↑
      Filas/columnas [3N:4N) → cadena 1, xi=-1, espín ↓
      Filas/columnas [4N:5N) → cadena 2, xi=+1, espín ↑
      Filas/columnas [5N:6N) → cadena 2, xi=+1, espín ↓
      Filas/columnas [6N:7N) → cadena 2, xi=-1, espín ↑
      Filas/columnas [7N:8N) → cadena 2, xi=-1, espín ↓
    """
    # Inicialización de la matriz densa de ceros con tipo complejo
    A = np.zeros((8*N, 8*N), dtype=complex)

    # ── Paso angular acumulado entre sitios (para la fase de Rashba) ──────────
    # Se discretiza el ángulo azimutal de la cadena en N sitios, de 0 a 2π.
    # Dphi es el incremento de ángulo entre sitios consecutivos.
    # Nota: el factor (10-1) en el denominador está hardcodeado y parece
    # corresponder a un parámetro del modelo geométrico (quizás el radio de
    # la hélice o número de vueltas). Debería ser N-1 en general.
    Dphi = (2 * np.pi / (10 - 1))

    for n in range(1, N+1):
        # ── Diagonal principal: término cinético y de amortiguamiento ──────────
        # Cada entrada diagonal corresponde a -(ω - E₀ + Γ + iη)
        # La parte imaginaria +iη es la regularización retardada.
        # Γ son los anchos de nivel de los electrodos (parte imaginaria pura: Γ = iΓ_val).

        l = (n-1) % N + 1    # Índice circular para el sitio n (aquí simplemente = n)

        # Cadena 1: los 4 bloques de espín (xi=±1, sigma=±1) comparten la misma energía E01[n-1]
        A[n-1,       l-1      ] = -(w - E01[n-1] + Gamma1[n-1] + eta*1.0j)  # cadena 1, xi=+1, ↑
        A[n-1+N,     l-1+N    ] = -(w - E01[n-1] + Gamma1[n-1] + eta*1.0j)  # cadena 1, xi=+1, ↓
        A[n-1+2*N,   l-1+2*N  ] = -(w - E01[n-1] + Gamma1[n-1] + eta*1.0j)  # cadena 1, xi=-1, ↑
        A[n-1+3*N,   l-1+3*N  ] = -(w - E01[n-1] + Gamma1[n-1] + eta*1.0j)  # cadena 1, xi=-1, ↓

        # Cadena 2: análogo con energía E02[n-1] y Gamma2[n-1]
        A[n-1+4*N,   l-1+4*N  ] = -(w - E02[n-1] + Gamma2[n-1] + eta*1.0j)  # cadena 2, xi=+1, ↑
        A[n-1+5*N,   l-1+5*N  ] = -(w - E02[n-1] + Gamma2[n-1] + eta*1.0j)  # cadena 2, xi=+1, ↓
        A[n-1+6*N,   l-1+6*N  ] = -(w - E02[n-1] + Gamma2[n-1] + eta*1.0j)  # cadena 2, xi=-1, ↑
        A[n-1+7*N,   l-1+7*N  ] = -(w - E02[n-1] + Gamma2[n-1] + eta*1.0j)  # cadena 2, xi=-1, ↓

        # ── Acoplamiento perpendicular entre cadenas (intra-espín, mismo xi) ───
        # gamma_per acopla los mismos bloques de espín entre cadena 1 y cadena 2,
        # sin mezcla de espín: (cadena1, xi=+1, ↑) ↔ (cadena2, xi=+1, ↑), etc.
        A[n-1,     l-1+4*N] = gamma_per   # c1 xi=+1 ↑  ↔  c2 xi=+1 ↑
        A[n-1+N,   l-1+5*N] = gamma_per   # c1 xi=+1 ↓  ↔  c2 xi=+1 ↓
        A[n-1+2*N, l-1+6*N] = gamma_per   # c1 xi=-1 ↑  ↔  c2 xi=-1 ↑
        A[n-1+3*N, l-1+7*N] = gamma_per   # c1 xi=-1 ↓  ↔  c2 xi=-1 ↓

        A[n-1+4*N, l-1    ] = gamma_per   # c2 xi=+1 ↑  ↔  c1 xi=+1 ↑  (hermítico)
        A[n-1+5*N, l-1+N  ] = gamma_per
        A[n-1+6*N, l-1+2*N] = gamma_per
        A[n-1+7*N, l-1+3*N] = gamma_per

        # ── Acoplamiento perpendicular inter-espín con inversión de xi ─────────
        # gamma_per1 acopla bloques de cadenas distintas con inversión de xi.
        # Esto puede representar, por ejemplo, un acoplamiento espín-órbita
        # perpendicular entre cadenas.
        A[n-1,     l-1+6*N] = gamma_per1  # c1 xi=+1 ↑  ↔  c2 xi=-1 ↑
        A[n-1+N,   l-1+7*N] = gamma_per1  # c1 xi=+1 ↓  ↔  c2 xi=-1 ↓
        A[n-1+2*N, l-1+4*N] = gamma_per1  # c1 xi=-1 ↑  ↔  c2 xi=+1 ↑
        A[n-1+3*N, l-1+5*N] = gamma_per1  # c1 xi=-1 ↓  ↔  c2 xi=+1 ↓

        A[n-1+4*N, l-1+2*N] = gamma_per1
        A[n-1+5*N, l-1+3*N] = gamma_per1
        A[n-1+6*N, l-1    ] = gamma_per1
        A[n-1+7*N, l-1+N  ] = gamma_per1

        # ══════════════════════════════════════════════════════════════════
        # HOPPING HACIA EL SITIO n+1 (vecino derecho)
        # ══════════════════════════════════════════════════════════════════
        l = n % N + 1        # Índice del sitio n+1 (con condición de contorno abierta
                             # ya que tt1[N-1]=0 impide el hopping desde el último sitio)

        # Fase angular acumulada en el sitio n (para la SOI de Rashba)
        pp = (n-1) * Dphi

        # ── Construcción de los vectores de hopping con condición de contorno abierta ──
        # tt1 y tt2 son arrays de N elementos que valen gamma01 (gamma02) para
        # los sitios 1..N-1, y 0 para el sitio N, evitando el hopping periódico.
        tt1 = np.ones(N-1)
        tt1 = [*tt1, 0]              # tt1[N-1] = 0 → no hay hopping desde el extremo derecho
        tt1 = np.multiply(tt1, gamma01)

        tt2 = np.ones(N-1)
        tt2 = [*tt2, 0]
        tt2 = np.multiply(tt2, gamma02)

        # Hopping estándar (sin SOI) hacia n+1: bloque diagonal en espín
        # Cadena 1:
        A[n-1,     l-1    ] += tt1[n-1]   # xi=+1, ↑
        A[n-1+N,   l-1+N  ] += tt1[n-1]   # xi=+1, ↓
        A[n-1+2*N, l-1+2*N] += tt1[n-1]   # xi=-1, ↑
        A[n-1+3*N, l-1+3*N] += tt1[n-1]   # xi=-1, ↓
        # Cadena 2:
        A[n-1+4*N, l-1+4*N] += tt2[n-1]
        A[n-1+5*N, l-1+5*N] += tt2[n-1]
        A[n-1+6*N, l-1+6*N] += tt2[n-1]
        A[n-1+7*N, l-1+7*N] += tt2[n-1]

        # ── SOI de Rashba y Dresselhaus hacia n+1 ──────────────────────────────
        # Los términos de SOI acoplan sitios de DISTINTO espín en sitios vecinos.
        # La forma general del término Rashba del hopping n→n+1 es:
        #   H_R = -i·l_R·e^{-i·pp}  (para ↑→↓) y  +i·l_R·e^{+i·pp} (para ↓→↑)
        # El término Dresselhaus tiene signo opuesto en la exponencial.
        #
        # Para cadena 1 (sin desfase β):
        #   (xi=+1, ↑) → (xi=-1, ↑): H = -i·l_R1·e^{-i·pp} + l_D·e^{+i·pp}
        #   (xi=+1, ↓) → (xi=-1, ↓): H = -i·l_R1·e^{+i·pp} - l_D·e^{-i·pp}
        #   etc.

        # Máscara para el hopping de Rashba (condición de contorno abierta)
        RR1 = [*np.ones(N-1), 0]
        l_R11 = np.multiply(RR1, l_R1)  # Rashba efectivo con condición de contorno

        RR2 = [*np.ones(N-1), 0]
        l_R12 = np.multiply(RR2, l_R2)

        # Cadena 1 – SOI hacia n+1 (mezcla de espín opuesto, mismo sitio → vecino)
        A[n-1,     l-1+2*N] =  (-1j*l_R11[n-1]*np.exp(-pp*1j) + l_D*np.exp( pp*1j))  # xi=+1,↑ → xi=-1,↑
        A[n-1+N,   l-1+3*N] =  (-1j*l_R11[n-1]*np.exp( pp*1j) - l_D*np.exp(-pp*1j))  # xi=+1,↓ → xi=-1,↓
        A[n-1+2*N, l-1    ] =  (-1j*l_R11[n-1]*np.exp( pp*1j) - l_D*np.exp(-pp*1j))  # xi=-1,↑ → xi=+1,↑
        A[n-1+3*N, l-1+N  ] =  (-1j*l_R11[n-1]*np.exp(-pp*1j) + l_D*np.exp( pp*1j))  # xi=-1,↓ → xi=+1,↓

        # Cadena 2 – SOI hacia n+1 con desfase adicional β=π
        A[n-1+4*N, l-1+6*N] =  (-1j*l_R12[n-1]*np.exp(-pp*1j - beta*1j) + l_D*np.exp( pp*1j))
        A[n-1+5*N, l-1+7*N] =  (-1j*l_R12[n-1]*np.exp( pp*1j + beta*1j) - l_D*np.exp(-pp*1j))
        A[n-1+6*N, l-1+4*N] =  (-1j*l_R12[n-1]*np.exp( pp*1j + beta*1j) - l_D*np.exp(-pp*1j))
        A[n-1+7*N, l-1+5*N] =  (-1j*l_R12[n-1]*np.exp(-pp*1j - beta*1j) + l_D*np.exp( pp*1j))

        # ══════════════════════════════════════════════════════════════════
        # HOPPING HACIA EL SITIO n-1 (vecino izquierdo)
        # ══════════════════════════════════════════════════════════════════
        l = (n-2) % N + 1    # Índice del sitio n-1
        pp = (n-2) * Dphi    # Fase angular del sitio n-1

        # Condición de contorno abierta: tt1[0] = 0 impide hopping desde el extremo izquierdo
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

        # SOI hacia n-1: signo opuesto al hopping hacia n+1 (hermicidad del Hamiltoniano)
        # Para n→n-1, la fase de Rashba invierte el signo del exponente complejo.
        RR11 = [0, *np.ones(N-1)]
        l_R111 = np.multiply(RR11, l_R1)

        RR22 = [0, *np.ones(N-1)]
        l_R122 = np.multiply(RR22, l_R2)

        # Cadena 1 – SOI hacia n-1
        A[n-1,     l-1+2*N] += ( 1j*l_R111[n-1]*np.exp(-pp*1j) - l_D*np.exp( pp*1j))
        A[n-1+N,   l-1+3*N] += ( 1j*l_R111[n-1]*np.exp( pp*1j) + l_D*np.exp(-pp*1j))
        A[n-1+2*N, l-1    ] += ( 1j*l_R111[n-1]*np.exp( pp*1j) + l_D*np.exp(-pp*1j))
        A[n-1+3*N, l-1+N  ] += ( 1j*l_R111[n-1]*np.exp(-pp*1j) - l_D*np.exp( pp*1j))

        # Cadena 2 – SOI hacia n-1 con desfase β
        A[n-1+4*N, l-1+6*N] += ( 1j*l_R122[n-1]*np.exp(-pp*1j - beta*1j) - l_D*np.exp( pp*1j))
        A[n-1+5*N, l-1+7*N] += ( 1j*l_R122[n-1]*np.exp( pp*1j + beta*1j) + l_D*np.exp(-pp*1j))
        A[n-1+6*N, l-1+4*N] += ( 1j*l_R122[n-1]*np.exp( pp*1j + beta*1j) + l_D*np.exp(-pp*1j))
        A[n-1+7*N, l-1+5*N] += ( 1j*l_R122[n-1]*np.exp(-pp*1j - beta*1j) - l_D*np.exp( pp*1j))

    # Convertir a formato CSR (Compressed Sparse Row) para eficiencia en el solver
    A = sparse.csr_matrix(A)
    return A


def matrix_AA(w, E01, E02, N, theta, Gamma1, Gamma2,
              gamma01, gamma02, gamma_per, gamma_per1,
              U, M1, M2, l_R1, l_R2, l_D):
    """
    Construye la matriz del sistema lineal para la función de Green AVANZADA.

    La función de Green avanzada es el conjugado hermítico de la retardada:
        G^A(ω) = [G^R(ω)]†

    Su matriz del sistema lineal es idéntica a la retardada salvo por el signo
    del término de regularización: +iη → -iη.

    Todos los parámetros y la estructura de bloques son idénticos a matrix_AR.
    La única diferencia es:
        -(w - E0 + Γ + iη)   [retardada]
    vs. -(w - E0 + Γ - iη)   [avanzada]

    Ver matrix_AR para la documentación completa de parámetros.
    """
    A = np.zeros((8*N, 8*N), dtype=complex)

    Dphi  = (2 * np.pi / (10 - 1))
    Dphi1 = (2 * np.pi / (10 - 1))   # Idéntico a Dphi; se declara por separado
                                      # por si en el futuro se quieren diferenciar
                                      # los pasos angulares de cada cadena

    for n in range(1, N+1):
        l = (n-1) % N + 1

        # ── Diagonal: MISMO que matrix_AR pero con -iη en lugar de +iη ────────
        A[n-1,     l-1    ] = -(w - E01[n-1] + Gamma1[n-1] - eta*1.0j)  # G^A: parte imaginaria negativa
        A[n-1+N,   l-1+N  ] = -(w - E01[n-1] + Gamma1[n-1] - eta*1.0j)
        A[n-1+2*N, l-1+2*N] = -(w - E01[n-1] + Gamma1[n-1] - eta*1.0j)
        A[n-1+3*N, l-1+3*N] = -(w - E01[n-1] + Gamma1[n-1] - eta*1.0j)
        A[n-1+4*N, l-1+4*N] = -(w - E02[n-1] + Gamma2[n-1] - eta*1.0j)
        A[n-1+5*N, l-1+5*N] = -(w - E02[n-1] + Gamma2[n-1] - eta*1.0j)
        A[n-1+6*N, l-1+6*N] = -(w - E02[n-1] + Gamma2[n-1] - eta*1.0j)
        A[n-1+7*N, l-1+7*N] = -(w - E02[n-1] + Gamma2[n-1] - eta*1.0j)

        # ── Acoplamientos perpendiculares: idénticos a matrix_AR ──────────────
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

        # ── Hopping hacia n+1 ──────────────────────────────────────────────────
        l  = n % N + 1
        pp = (n-1) * Dphi

        tt1 = [*np.ones(N-1), 0]
        tt1 = np.multiply(tt1, gamma01)

        tt2 = [*np.ones(N-1), 0]
        tt2 = np.multiply(tt2, gamma02)

        A[n-1,     l-1    ] += tt1[n-1]
        A[n-1+N,   l-1+N  ] += tt1[n-1]
        A[n-1+2*N, l-1+2*N] += tt1[n-1]
        A[n-1+3*N, l-1+3*N] += tt1[n-1]

        A[n-1+4*N, l-1+4*N] += tt2[n-1]
        A[n-1+5*N, l-1+5*N] += tt2[n-1]
        A[n-1+6*N, l-1+6*N] += tt2[n-1]
        A[n-1+7*N, l-1+7*N] += tt2[n-1]

        RR1  = [*np.ones(N-1), 0]
        l_R11 = np.multiply(RR1, l_R1)

        RR2  = [*np.ones(N-1), 0]
        l_R12 = np.multiply(RR2, l_R2)

        # Cadena 1 – SOI hacia n+1 (G^A tiene las mismas entradas off-diagonal que G^R)
        A[n-1,     l-1+2*N] = (-1j*l_R11[n-1]*np.exp(-pp*1j) + l_D*np.exp( pp*1j))
        A[n-1+N,   l-1+3*N] = (-1j*l_R11[n-1]*np.exp( pp*1j) - l_D*np.exp(-pp*1j))
        A[n-1+2*N, l-1    ] = (-1j*l_R11[n-1]*np.exp( pp*1j) - l_D*np.exp(-pp*1j))
        A[n-1+3*N, l-1+N  ] = (-1j*l_R11[n-1]*np.exp(-pp*1j) + l_D*np.exp( pp*1j))

        # Cadena 2 – SOI hacia n+1 con desfase β
        pp = (n-1) * Dphi1   # Puede diferir de Dphi si Dphi1 ≠ Dphi en el futuro
        A[n-1+4*N, l-1+6*N] = (-1j*l_R12[n-1]*np.exp(-pp*1j)*np.exp(-beta*1j) + l_D*np.exp( pp*1j))
        A[n-1+5*N, l-1+7*N] = (-1j*l_R12[n-1]*np.exp( pp*1j)*np.exp( beta*1j) - l_D*np.exp(-pp*1j))
        A[n-1+6*N, l-1+4*N] = (-1j*l_R12[n-1]*np.exp( pp*1j)*np.exp( beta*1j) - l_D*np.exp(-pp*1j))
        A[n-1+7*N, l-1+5*N] = (-1j*l_R12[n-1]*np.exp(-pp*1j)*np.exp(-beta*1j) + l_D*np.exp( pp*1j))

        # ── Hopping hacia n-1 ──────────────────────────────────────────────────
        l  = (n-2) % N + 1
        pp = (n-2) * Dphi

        tt1 = [0, *np.ones(N-1)]
        tt1 = np.multiply(tt1, gamma01)

        tt2 = [0, *np.ones(N-1)]
        tt2 = np.multiply(tt2, gamma02)

        A[n-1,     l-1    ] += tt1[n-1]
        A[n-1+N,   l-1+N  ] += tt1[n-1]
        A[n-1+2*N, l-1+2*N] += tt1[n-1]
        A[n-1+3*N, l-1+3*N] += tt1[n-1]

        A[n-1+4*N, l-1+4*N] += tt2[n-1]
        A[n-1+5*N, l-1+5*N] += tt2[n-1]
        A[n-1+6*N, l-1+6*N] += tt2[n-1]
        A[n-1+7*N, l-1+7*N] += tt2[n-1]

        RR11 = [0, *np.ones(N-1)]
        l_R111 = np.multiply(RR11, l_R1)

        RR22 = [0, *np.ones(N-1)]
        l_R122 = np.multiply(RR22, l_R2)

        pp = (n-2) * Dphi   # Fase del sitio n-1 para cadena 1

        # Cadena 1 – SOI hacia n-1
        A[n-1,     l-1+2*N] += ( 1j*l_R111[n-1]*np.exp(-pp*1j) - l_D*np.exp( pp*1j))
        A[n-1+N,   l-1+3*N] += ( 1j*l_R111[n-1]*np.exp( pp*1j) + l_D*np.exp(-pp*1j))
        A[n-1+2*N, l-1    ] += ( 1j*l_R111[n-1]*np.exp( pp*1j) + l_D*np.exp(-pp*1j))
        A[n-1+3*N, l-1+N  ] += ( 1j*l_R111[n-1]*np.exp(-pp*1j) - l_D*np.exp( pp*1j))

        pp = (n-1) * Dphi1  # Fase para cadena 2 (usa índice n-1, no n-2; revisar si es intencional)

        # Cadena 2 – SOI hacia n-1
        A[n-1+4*N, l-1+6*N] += ( 1j*l_R122[n-1]*np.exp(-pp*1j)*np.exp(-beta*1j) - l_D*np.exp( pp*1j))
        A[n-1+5*N, l-1+7*N] += ( 1j*l_R122[n-1]*np.exp( pp*1j)*np.exp( beta*1j) + l_D*np.exp(-pp*1j))
        A[n-1+6*N, l-1+4*N] += ( 1j*l_R122[n-1]*np.exp( pp*1j)*np.exp( beta*1j) + l_D*np.exp(-pp*1j))
        A[n-1+7*N, l-1+5*N] += ( 1j*l_R122[n-1]*np.exp(-pp*1j)*np.exp(-beta*1j) - l_D*np.exp( pp*1j))

    A = sparse.csr_matrix(A)
    return A


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: EXTRACCIÓN DE ELEMENTOS DE LA FUNCIÓN DE GREEN
# ══════════════════════════════════════════════════════════════════════════════
#
# Estas funciones extraen el elemento G(N, 1) de la función de Green,
# es decir, la amplitud de propagación desde el sitio 1 hasta el sitio N,
# lo que en el formalismo de Landauer corresponde a la amplitud de transmisión.
#
# El vector solución GR1 se obtiene de resolver A·GR1 = B con B = -δ(primer sitio),
# por lo que GR1[k] = G^R_{k, fuente}.
#
# Convención de índices en el vector de solución GR1:
#   GR1[n-1]        → cadena 1, xi=+1, ↑,  sitio n
#   GR1[n-1 + N]    → cadena 1, xi=+1, ↓,  sitio n
#   GR1[n-1 + 2N]   → cadena 1, xi=-1, ↑,  sitio n
#   GR1[n-1 + 3N]   → cadena 1, xi=-1, ↓,  sitio n
#   GR1[n-1 + 4N]   → cadena 2, xi=+1, ↑,  sitio n
#   GR1[n-1 + 5N]   → cadena 2, xi=+1, ↓,  sitio n
#   GR1[n-1 + 6N]   → cadena 2, xi=-1, ↑,  sitio n
#   GR1[n-1 + 7N]   → cadena 2, xi=-1, ↓,  sitio n
#
# El elemento de transmisión spin-up→spin-up desde la cadena 1 hasta el último
# sitio de la cadena 1 corresponde a GR1[N-1].
# ──────────────────────────────────────────────────────────────────────────────

def den_espectral1u(GR1, GA1, N):
    """
    Extrae G^R_{N, 1} para el canal (cadena 1, xi=+1, espín ↑).

    Índice en GR1: N-1   (sitio N, bloque [0:N), espín ↑, xi=+1 de cadena 1)
    Signo negativo: convención G = -A·B con B[0] = -1, por lo que G real es -GR1.
    """
    den_esp = -GR1[N-1]
    return den_esp


def den_espectral2u(GR1, GA1, N):
    """
    Extrae G^R_{N, 1} para el canal (cadena 2, xi=+1, espín ↑).

    Índice en GR1: 5N-1   (sitio N, bloque [4N:5N), espín ↑, xi=+1 de cadena 2)
    """
    den_esp = -GR1[5*N-1]
    return den_esp


def den_espectral1d(GR1, GA1, N):
    """
    Extrae G^R_{N, 1} para el canal (cadena 1, xi=+1, espín ↓).

    Índice en GR1: 2N-1   (sitio N, bloque [N:2N), espín ↓, xi=+1 de cadena 1)
    """
    den_esp = -GR1[2*N-1]
    return den_esp


def den_espectral2d(GR1, GA1, N):
    """
    Extrae G^R_{N, 1} para el canal (cadena 2, xi=+1, espín ↓).

    Índice en GR1: 6N-1   (sitio N, bloque [5N:6N), espín ↓, xi=+1 de cadena 2)
    """
    den_esp = -GR1[6*N-1]
    return den_esp


def den_espectral1ud(GR1, GA1, N):
    """
    Extrae G^R_{N, 1} para el canal mixto (cadena 1, xi=-1, espín ↑).

    Corresponde a la transmisión con inversión de la dirección de propagación xi,
    manteniendo el espín ↑. Índice: 3N-1 (bloque [2N:3N)).
    El sufijo 'ud' indica up→down en xi (xi=+1 → xi=-1), espín up.
    """
    den_esp = -GR1[3*N-1]
    return den_esp


def den_espectral2ud(GR1, GA1, N):
    """
    Extrae G^R_{N, 1} para el canal (cadena 2, xi=-1, espín ↑).

    Índice: 7N-1   (bloque [6N:7N))
    """
    den_esp = -GR1[7*N-1]
    return den_esp


def den_espectral1du(GR1, GA1, N):
    """
    Extrae G^R_{N, 1} para el canal (cadena 1, xi=-1, espín ↓).

    Índice: 4N-1   (bloque [3N:4N))
    El sufijo 'du' indica down→up en xi, espín down.
    """
    den_esp = -GR1[4*N-1]
    return den_esp


def den_espectral2du(GR1, GA1, N):
    """
    Extrae G^R_{N, 1} para el canal (cadena 2, xi=-1, espín ↓).

    Índice: 8N-1   (bloque [7N:8N))
    """
    den_esp = -GR1[8*N-1]
    return den_esp


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: BARRIDO EN ENERGÍA – CÁLCULO DE LA FUNCIÓN DE GREEN
# ══════════════════════════════════════════════════════════════════════════════
#
# Cada función rho_w* calcula un elemento específico de la función de Green
# en función de la energía ω, resolviendo el sistema lineal A·G = B para
# cada valor de ω en una malla uniforme.
#
# Todas las funciones siguen la misma estructura:
#   1. Crear la malla de energías w1 = linspace(w_0, w_f, nw)
#   2. Para cada ω:
#       a. Construir A^R (matrix_AR) y resolver A^R·x = B → G^R
#       b. Construir A^A (matrix_AA) y resolver A^A·x = B → G^A
#       c. Extraer el elemento de Green relevante (den_espectral*)
#   3. Retornar (w1, array de G^R evaluados)
#
# El vector fuente B = [−1, 0, ..., 0, −1, 0, ..., 0, −1, 0, ..., −1, ...]
# con −1 en las posiciones 0, N, 4N y 5N, lo que corresponde a inyectar
# desde el primer sitio de cada uno de los 4 canales de entrada activos:
#   cadena 1 xi=+1 ↑, cadena 1 xi=+1 ↓, cadena 2 xi=+1 ↑, cadena 2 xi=+1 ↓
# ──────────────────────────────────────────────────────────────────────────────

def rho_w1u(w_0, w_f, nw, B, N, E01, E02, theta, Gamma1, Gamma2,
            gamma01, gamma02, gamma_per, gamma_per1,
            U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido en energía para el canal de transmisión: cadena 1, xi=+1, espín ↑.

    Resuelve A^R·G^R = B y A^A·G^A = B para cada ω en [w_0, w_f],
    extrayendo G^R[N-1] (transmisión ↑ al final de la cadena 1).

    Retorna
    -------
    w1    : array de energías
    Den01 : array complejo con G^R(ω) para este canal
    """
    w1    = np.linspace(w_0, w_f, nw)
    Den01 = np.array([], dtype=complex)

    for w0 in w1:
        w   = w0
        # Función de Green retardada: A^R · G^R = B
        AR1 = matrix_AR(w, E01, E02, N, theta, Gamma1, Gamma2,
                        gamma01, gamma02, gamma_per, gamma_per1,
                        U, M1, M2, l_R1, l_R2, l_D)
        GR1 = spsolve(AR1, B)
        GR1 = np.asarray(GR1).ravel()

        # Función de Green avanzada: A^A · G^A = B
        AA1 = matrix_AA(w, E01, E02, N, theta, Gamma1, Gamma2,
                        gamma01, gamma02, gamma_per, gamma_per1,
                        U, M1, M2, l_R1, l_R2, l_D)
        GA1 = spsolve(AA1, B)
        GA1 = np.asarray(GA1).ravel()

        # Extraer y acumular el elemento de transmisión
        Den01 = np.append(Den01, den_espectral1u(GR1, GA1, N))

    return w1, Den01


def rho_w2u(w_0, w_f, nw, B, N, E01, E02, theta, Gamma1, Gamma2,
            gamma01, gamma02, gamma_per, gamma_per1,
            U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido en energía: cadena 2, xi=+1, espín ↑.
    Extrae G^R[5N-1] (transmisión al último sitio de la cadena 2, espín ↑).
    """
    w1    = np.linspace(w_0, w_f, nw)
    Den02 = np.array([], dtype=complex)

    for w0 in w1:
        w   = w0
        AR1 = matrix_AR(w, E01, E02, N, theta, Gamma1, Gamma2,
                        gamma01, gamma02, gamma_per, gamma_per1,
                        U, M1, M2, l_R1, l_R2, l_D)
        GR1 = np.asarray(spsolve(AR1, B)).ravel()

        AA1 = matrix_AA(w, E01, E02, N, theta, Gamma1, Gamma2,
                        gamma01, gamma02, gamma_per, gamma_per1,
                        U, M1, M2, l_R1, l_R2, l_D)
        GA1 = np.asarray(spsolve(AA1, B)).ravel()

        Den02 = np.append(Den02, den_espectral2u(GR1, GA1, N))

    return w1, Den02


def rho_w1d(w_0, w_f, nw, B, N, E01, E02, theta, Gamma1, Gamma2,
            gamma01, gamma02, gamma_per, gamma_per1,
            U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido en energía: cadena 1, xi=+1, espín ↓.
    Extrae G^R[2N-1].
    """
    w1    = np.linspace(w_0, w_f, nw)
    Den01 = np.array([], dtype=complex)

    for w0 in w1:
        w   = w0
        AR1 = matrix_AR(w, E01, E02, N, theta, Gamma1, Gamma2,
                        gamma01, gamma02, gamma_per, gamma_per1,
                        U, M1, M2, l_R1, l_R2, l_D)
        GR1 = np.asarray(spsolve(AR1, B)).ravel()

        AA1 = matrix_AA(w, E01, E02, N, theta, Gamma1, Gamma2,
                        gamma01, gamma02, gamma_per, gamma_per1,
                        U, M1, M2, l_R1, l_R2, l_D)
        GA1 = np.asarray(spsolve(AA1, B)).ravel()

        Den01 = np.append(Den01, den_espectral1d(GR1, GA1, N))

    return w1, Den01


def rho_w2d(w_0, w_f, nw, B, N, E01, E02, theta, Gamma1, Gamma2,
            gamma01, gamma02, gamma_per, gamma_per1,
            U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido en energía: cadena 2, xi=+1, espín ↓.
    Extrae G^R[6N-1].
    """
    w1    = np.linspace(w_0, w_f, nw)
    Den02 = np.array([], dtype=complex)

    for w0 in w1:
        w   = w0
        AR1 = matrix_AR(w, E01, E02, N, theta, Gamma1, Gamma2,
                        gamma01, gamma02, gamma_per, gamma_per1,
                        U, M1, M2, l_R1, l_R2, l_D)
        GR1 = np.asarray(spsolve(AR1, B)).ravel()

        AA1 = matrix_AA(w, E01, E02, N, theta, Gamma1, Gamma2,
                        gamma01, gamma02, gamma_per, gamma_per1,
                        U, M1, M2, l_R1, l_R2, l_D)
        GA1 = np.asarray(spsolve(AA1, B)).ravel()

        Den02 = np.append(Den02, den_espectral2d(GR1, GA1, N))

    return w1, Den02


def rho_w1ud(w_0, w_f, nw, B, N, E01, E02, theta, Gamma1, Gamma2,
             gamma01, gamma02, gamma_per, gamma_per1,
             U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido en energía: cadena 1, canal con inversión de xi, espín ↑.
    Extrae G^R[3N-1] → transmisión con cambio de dirección de propagación.
    """
    w1   = np.linspace(w_0, w_f, nw)
    Den1 = np.array([], dtype=complex)

    for w0 in w1:
        w   = w0
        AR1 = matrix_AR(w, E01, E02, N, theta, Gamma1, Gamma2,
                        gamma01, gamma02, gamma_per, gamma_per1,
                        U, M1, M2, l_R1, l_R2, l_D)
        GR1 = np.asarray(spsolve(AR1, B)).ravel()

        AA1 = matrix_AA(w, E01, E02, N, theta, Gamma1, Gamma2,
                        gamma01, gamma02, gamma_per, gamma_per1,
                        U, M1, M2, l_R1, l_R2, l_D)
        GA1 = np.asarray(spsolve(AA1, B)).ravel()

        Den1 = np.append(Den1, den_espectral1ud(GR1, GA1, N))

    return w1, Den1


def rho_w2ud(w_0, w_f, nw, B, N, E01, E02, theta, Gamma1, Gamma2,
             gamma01, gamma02, gamma_per, gamma_per1,
             U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido en energía: cadena 2, canal con inversión de xi, espín ↑.
    Extrae G^R[7N-1].
    """
    w1   = np.linspace(w_0, w_f, nw)
    Den2 = np.array([], dtype=complex)

    for w0 in w1:
        w   = w0
        AR1 = matrix_AR(w, E01, E02, N, theta, Gamma1, Gamma2,
                        gamma01, gamma02, gamma_per, gamma_per1,
                        U, M1, M2, l_R1, l_R2, l_D)
        GR1 = np.asarray(spsolve(AR1, B)).ravel()

        AA1 = matrix_AA(w, E01, E02, N, theta, Gamma1, Gamma2,
                        gamma01, gamma02, gamma_per, gamma_per1,
                        U, M1, M2, l_R1, l_R2, l_D)
        GA1 = np.asarray(spsolve(AA1, B)).ravel()

        Den2 = np.append(Den2, den_espectral2ud(GR1, GA1, N))

    return w1, Den2


def rho_w1du(w_0, w_f, nw, B, N, E01, E02, theta, Gamma1, Gamma2,
             gamma01, gamma02, gamma_per, gamma_per1,
             U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido en energía: cadena 1, canal con inversión de xi, espín ↓.
    Extrae G^R[4N-1].
    """
    w1   = np.linspace(w_0, w_f, nw)
    Den5 = np.array([], dtype=complex)

    for w0 in w1:
        w   = w0
        AR1 = matrix_AR(w, E01, E02, N, theta, Gamma1, Gamma2,
                        gamma01, gamma02, gamma_per, gamma_per1,
                        U, M1, M2, l_R1, l_R2, l_D)
        GR1 = np.asarray(spsolve(AR1, B)).ravel()

        AA1 = matrix_AA(w, E01, E02, N, theta, Gamma1, Gamma2,
                        gamma01, gamma02, gamma_per, gamma_per1,
                        U, M1, M2, l_R1, l_R2, l_D)
        GA1 = np.asarray(spsolve(AA1, B)).ravel()

        Den5 = np.append(Den5, den_espectral1du(GR1, GA1, N))

    return w1, Den5


def rho_w2du(w_0, w_f, nw, B, N, E01, E02, theta, Gamma1, Gamma2,
             gamma01, gamma02, gamma_per, gamma_per1,
             U, M1, M2, l_R1, l_R2, l_D):
    """
    Barrido en energía: cadena 2, canal con inversión de xi, espín ↓.
    Extrae G^R[8N-1].
    """
    w1   = np.linspace(w_0, w_f, nw)
    Den6 = np.array([], dtype=complex)

    for w0 in w1:
        w   = w0
        AR1 = matrix_AR(w, E01, E02, N, theta, Gamma1, Gamma2,
                        gamma01, gamma02, gamma_per, gamma_per1,
                        U, M1, M2, l_R1, l_R2, l_D)
        GR1 = np.asarray(spsolve(AR1, B)).ravel()

        AA1 = matrix_AA(w, E01, E02, N, theta, Gamma1, Gamma2,
                        gamma01, gamma02, gamma_per, gamma_per1,
                        U, M1, M2, l_R1, l_R2, l_D)
        GA1 = np.asarray(spsolve(AA1, B)).ravel()

        Den6 = np.append(Den6, den_espectral2du(GR1, GA1, N))

    return w1, Den6


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: CÁLCULO DE LA CONDUCTANCIA DE CARGA – G₀
# ══════════════════════════════════════════════════════════════════════════════
#
# La conductancia de carga total (en unidades de e²/h) se obtiene sumando
# los módulos cuadrados de TODOS los elementos de la función de Green de
# transmisión (Landauer-Büttiker):
#
#   G₀ = (4·e²/h) · Σ_{canales} |T_canal(ω)|²
#
# donde T_canal = G^R_{N,1} para cada combinación (cadena, xi, espín).
# El factor 4 surge de las dos cadenas × dos direcciones de xi.
# ──────────────────────────────────────────────────────────────────────────────

# ── Parámetros para la transmisión de carga (bloque 1) ──────────────────────
N       = 10        # Número de sitios por cadena
gamma01 = 1.0       # Hopping en cadena 1 (en unidades de energía, e.g. eV)
gamma02 = 1.0       # Hopping en cadena 2
l_R1    = 0.0       # Rashba cadena 1 apagado (sin SOI para este bloque)
l_R12   = 0.0       # Variable no usada directamente
l_R2    = 0.0       # Rashba cadena 2 apagado
l_EO1   = 0.0       # Zeeman efectivo cadena 1 (no implementado)
l_EO2   = 0.0       # Zeeman efectivo cadena 2 (no implementado)
l_D     = 0.0       # Dresselhaus apagado
l_Z     = 0.0       # Zeeman apagado
eta     = 0.001     # Regularización numérica más gruesa para G₀

E01 = np.zeros(N)   # Energías de sitio uniformes (banda plana)
E02 = np.zeros(N)
theta = 0           # Sin flujo de Aharonov-Bohm
M1  = 1.0           # Masa efectiva cadena 1 (no usada explícitamente)
M2  = 1.0
U   = np.zeros(N)   # Sin desorden

# ── Vector fuente B: excita los 4 canales de entrada del primer sitio ────────
# B[0]   = -1 → cadena 1, xi=+1, ↑,  sitio 1
# B[N]   = -1 → cadena 1, xi=+1, ↓,  sitio 1
# B[4N]  = -1 → cadena 2, xi=+1, ↑,  sitio 1
# B[5N]  = -1 → cadena 2, xi=+1, ↓,  sitio 1
B      = np.zeros(8*N, dtype=complex)
B[0]   = -1
B[N]   = -1
B[4*N] = -1
B[5*N] = -1

# ── Anchos de nivel (acoplamiento a electrodos) ───────────────────────────────
# p controla la asimetría entre el electrodo izquierdo y el derecho.
# val = i·(1+p) → ancho del electrodo izquierdo de cadena 1
# val1 = i·1    → ancho del electrodo derecho / cadena 2 (siempre simétrico)
p     = 0.0
val   = 1.0*(1+p)*1j    # Ancho de nivel complejo puro: Γ = iΓ_val
val1  = 1.0*1j

Gamma1 = np.zeros(N-2)
Gamma1 = [val, *Gamma1, val]    # Sólo los extremos son no nulos

Gamma2 = np.zeros(N-2)
Gamma2 = [val1, *Gamma2, val1]

# ── Acoplamiento perpendicular entre cadenas ─────────────────────────────────
# gamma_out  = 0   → sin acoplamiento intra-espín entre cadenas
# gamma_out1 = 1.0 → acoplamiento inter-espín activo (gamma_per1)
gamma_out  = 0.0
gamma_out1 = 1.0

start_time = time.time()

# ─── Cálculo de los 8 canales de transmisión (4 por cadena × 2 por signo de gamma_per) ─

# Grupo 1: gamma_per = gamma_out, gamma_per1 = gamma_out1
gamma_per  = gamma_out
gamma_per1 = gamma_out1
E01 = np.zeros(N)
E02 = np.zeros(N)
w1, Trans1u  = rho_w1u( -3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)

gamma_per  = gamma_out
gamma_per1 = gamma_out1
E01 = np.zeros(N)
E02 = np.zeros(N)
w1, Trans1d  = rho_w1d( -3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)

gamma_per  = gamma_out
gamma_per1 = gamma_out1
E01 = np.zeros(N)
E02 = np.zeros(N)
w1, Trans1ud = rho_w1ud(-3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)

gamma_per  = gamma_out
gamma_per1 = gamma_out1
E01 = np.zeros(N)
E02 = np.zeros(N)
w1, Trans1du = rho_w1du(-3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)

# Grupo 2: gamma_per = -gamma_out, gamma_per1 = gamma_out1
# La inversión de signo en gamma_per permite calcular la contribución del
# canal con acoplamiento perpendicular de signo opuesto (necesario para
# obtener las componentes del tensor de conductancia de espín).
gamma_per  = (-1.0)*gamma_out
gamma_per1 = (1.0)*gamma_out1
E01 = np.zeros(N)
E02 = np.zeros(N)
w1, Trans2u  = rho_w2u( -3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)

gamma_per  = (-1.0)*gamma_out
gamma_per1 = (1.0)*gamma_out1
E01 = np.zeros(N)
E02 = np.zeros(N)
w1, Trans2d  = rho_w2d( -3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)

gamma_per  = (-1.0)*gamma_out
gamma_per1 = (1.0)*gamma_out1
E01 = np.zeros(N)
E02 = np.zeros(N)
w1, Trans2ud = rho_w2ud(-3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)

gamma_per  = (-1.0)*gamma_out
gamma_per1 = (1.0)*gamma_out1
E01 = np.zeros(N)
E02 = np.zeros(N)
w1, Trans2du = rho_w2du(-3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)

end_time  = time.time()
diff_time = end_time - start_time
print(f"El tiempo de ejecución fue de {diff_time} segundos.")

# ── Conductancia de carga G₀ ──────────────────────────────────────────────────
# G₀ = 4 · Σ |T_i|²  donde la suma recorre los 8 canales de transmisión.
# El factor 4 es la prefactor de cuantización e²/h en unidades naturales.
# Cada par |T|² = T·conj(T) da el módulo cuadrado de la amplitud de transmisión.
Trans = (Trans1u *np.conjugate(Trans1u)
        + Trans1du*np.conjugate(Trans1du)
        + Trans1ud*np.conjugate(Trans1ud)
        + Trans1d *np.conjugate(Trans1d)
        + Trans2u *np.conjugate(Trans2u)
        + Trans2du*np.conjugate(Trans2du)
        + Trans2ud*np.conjugate(Trans2ud)
        + Trans2d *np.conjugate(Trans2d))

# ── Guardar tiempo de ejecución ───────────────────────────────────────────────
ff = open('archivo_tiempo_cadena_bi_W_SO.dat', 'w')
ff.write('El tiempo de ejecucion es: ' + str(diff_time))
ff.flush()

# ── Guardar resultados en archivo de texto ────────────────────────────────────
A      = np.stack((w1, Trans.real), axis=1)
nombre = "Salida_cadena_bi_W_SO.dat"
np.savetxt(nombre, A, fmt='%.6e')   # Columna 1: energía, Columna 2: G₀(ω)

# ── Graficar la transmisión de carga ─────────────────────────────────────────
plt.figure(figsize=(16, 7))
plt.plot(w1, Trans.real)
plt.xlabel("Energy [meV]")
plt.ylabel("$G_0$")   # Conductancia de carga en unidades de e²/h


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: CONDUCTANCIA DE ESPÍN – COMPONENTE Gz
# ══════════════════════════════════════════════════════════════════════════════
#
# La conductancia de espín Gz mide el flujo neto de espín en la dirección z:
#
#   Gz = |T↑↑|² - |T↑↓|² + |T↓↑|² - |T↓↓|²  (+ contribuciones cadena 2)
#
# donde T↑↑ es la amplitud de transmisión de ↑→↑ (conservando espín) y
# T↑↓ es la amplitud de transmisión de ↑→↓ (con inversión de espín).
# Esta combinación con signos alternos es la proyección de σ_z sobre la
# corriente de transmisión en el formalismo de Landauer para el espín.
# ──────────────────────────────────────────────────────────────────────────────

N       = 10
gamma01 = 1.2       # Hopping ligeramente diferente para este bloque (puede ser un typo)
gamma02 = 1.2
l_R1    = 0.1       # Rashba encendido
l_R12   = 0.0
l_R2    = 0.1
l_EO1   = 0.0
l_EO2   = 0.0
l_D     = 0.0
l_Z     = 0.0
eta     = 0.001

E01 = np.zeros(N)
E02 = np.zeros(N)
theta = 0
M1  = 1.0
M2  = 1.0
U   = np.zeros(N)

B      = np.zeros(8*N, dtype=complex)
B[0]   = -1
B[N]   = -1
B[4*N] = -1
B[5*N] = -1

p     = 0.0
val   = 1.0*(1+p)*1j
val1  = 1.0*1j
Gamma1 = [val,  *np.zeros(N-2), val ]
Gamma2 = [val1, *np.zeros(N-2), val1]

gamma_out  = 0.0
gamma_out1 = 1.0

start_time = time.time()

# Grupo 1 de canales para Gz
gamma_per  = gamma_out
gamma_per1 = gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans1u  = rho_w1u( -3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)
gamma_per  = gamma_out;  gamma_per1 = gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans1d  = rho_w1d( -3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)
gamma_per  = gamma_out;  gamma_per1 = gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans1ud = rho_w1ud(-3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)
gamma_per  = gamma_out;  gamma_per1 = gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans1du = rho_w1du(-3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)

# Grupo 2 de canales para Gz (acoplamiento perpendicular negado)
gamma_per  = (-1.0)*gamma_out;  gamma_per1 = (-1.0)*gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans2u  = rho_w2u( -3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)
gamma_per  = (-1.0)*gamma_out;  gamma_per1 = (-1.0)*gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans2d  = rho_w2d( -3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)
gamma_per  = (-1.0)*gamma_out;  gamma_per1 = (-1.0)*gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans2ud = rho_w2ud(-3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)
gamma_per  = (-1.0)*gamma_out;  gamma_per1 = (-1.0)*gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans2du = rho_w2du(-3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)

end_time  = time.time()
diff_time = end_time - start_time
print(f"El tiempo de ejecución fue de {diff_time} segundos.")

# ── Conductancia de espín Gz ───────────────────────────────────────────────────
# Gz = Σ [|T_↑|² - |T_↓|²]  donde ↑/↓ corresponde a los canales de espín up/down.
# Los canales 'u' (up) entran con signo positivo y los 'd' (down) con negativo,
# proyectando así la corriente de espín sobre el eje z.
Trans = (Trans1u *np.conjugate(Trans1u)
        - Trans1du*np.conjugate(Trans1du)
        + Trans1ud*np.conjugate(Trans1ud)
        - Trans1d *np.conjugate(Trans1d)
        + Trans2u *np.conjugate(Trans2u)
        - Trans2du*np.conjugate(Trans2du)
        + Trans2ud*np.conjugate(Trans2ud)
        - Trans2d *np.conjugate(Trans2d))

ff = open('archivo_tiempo_cadena_bi_W_SO_Z.dat', 'w')
ff.write('El tiempo de ejecucion es: ' + str(diff_time))
ff.flush()

A      = np.stack((w1, Trans.real), axis=1)
nombre = "Salida_cadena_bi_W_SO_Z.dat"
np.savetxt(nombre, A, fmt='%.6e')

plt.figure(figsize=(16, 7))
plt.plot(w1, Trans.real)
plt.xlabel("Energy [meV]")
plt.ylabel("$G_z$")   # Conductancia de espín en z (en unidades de ℏ/4π · e²/h)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6: CONDUCTANCIA DE ESPÍN – COMPONENTE Gx
# ══════════════════════════════════════════════════════════════════════════════
#
# La conductancia de espín en la dirección x se obtiene a partir de la
# parte REAL de los términos cruzados entre amplitudes de espín ↑ y ↓:
#
#   Gx ∝ Re[T_du · T_u* + T_u · T_du* + T_d · T_ud* + T_ud · T_d*]
#
# Esta combinación es la proyección de σ_x = (σ+ + σ-) sobre el tensor
# de transmisión, y captura la polarización de espín transversal en x.
# El prefactor 4 mantiene la consistencia con las otras componentes.
# ──────────────────────────────────────────────────────────────────────────────

N       = 28        # Cadena más larga para este cálculo
gamma01 = 1.0
gamma02 = 1.0
l_R1    = 0.1
l_R12   = 0.0
l_R2    = 0.1
l_EO1   = 0.0
l_EO2   = 0.0
l_D     = 0.0
l_Z     = 0.0
eta     = 0.000001  # Regularización muy pequeña para mejor resolución espectral

E01 = np.zeros(N)
E02 = np.zeros(N)
theta = 0
M1  = 1.0
M2  = 1.0
U   = np.zeros(N)

B      = np.zeros(8*N, dtype=complex)
B[0]   = -1
B[N]   = -1
B[4*N] = -1
B[5*N] = -1

p     = 0.0
val   = 1.0*(1+p)*1j
val1  = 1.0*1j
Gamma1 = [val,  *np.zeros(N-2), val ]
Gamma2 = [val1, *np.zeros(N-2), val1]

gamma_out  = 0.0
gamma_out1 = 1.0

start_time = time.time()

# Canales para Gx (gamma_per1 = ±gamma_out1)
gamma_per = gamma_out;  gamma_per1 = gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans1u  = rho_w1u( -3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)
gamma_per = gamma_out;  gamma_per1 = gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans1d  = rho_w1d( -3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)
gamma_per = gamma_out;  gamma_per1 = gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans1ud = rho_w1ud(-3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)
gamma_per = gamma_out;  gamma_per1 = gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans1du = rho_w1du(-3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)

# Para Gx, la segunda serie invierte AMBOS signos de gamma_per y gamma_per1
gamma_per = (-1.0)*gamma_out;  gamma_per1 = (-1.0)*gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans2u  = rho_w2u( -3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)
gamma_per = (-1.0)*gamma_out;  gamma_per1 = (-1.0)*gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans2d  = rho_w2d( -3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)
gamma_per = (-1.0)*gamma_out;  gamma_per1 = (-1.0)*gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans2ud = rho_w2ud(-3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)
gamma_per = (-1.0)*gamma_out;  gamma_per1 = (-1.0)*gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans2du = rho_w2du(-3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)

end_time  = time.time()
diff_time = end_time - start_time
print(f"El tiempo de ejecución fue de {diff_time} segundos.")

# ── Conductancia de espín Gx ───────────────────────────────────────────────────
# Gx = 4·Re[T_du·T_u* + T_u·T_du* + T_d·T_ud* + T_ud·T_d*]
# La suma T·T* + T*·T = 2·Re[T·T*], luego la fórmula es equivalente a
# Gx = 8·Re[T_du·T_u*] + 8·Re[T_d·T_ud*] + contribuciones de cadena 2
Trans = 4.0*(Trans1du*np.conjugate(Trans1u)  + Trans1u*np.conjugate(Trans1du)
           + Trans2du*np.conjugate(Trans2u)  + Trans2u*np.conjugate(Trans2du)
           + Trans1d *np.conjugate(Trans1ud) + Trans1ud*np.conjugate(Trans1d)
           + Trans2d *np.conjugate(Trans2ud) + Trans2ud*np.conjugate(Trans2d))

ff = open('archivo_tiempo_cadena_bi_W_SO_X.dat', 'w')
ff.write('El tiempo de ejecucion es: ' + str(diff_time))
ff.flush()

A      = np.stack((w1, Trans.real), axis=1)
nombre = "Salida_cadena_bi_W_SO_X.dat"
np.savetxt(nombre, A, fmt='%.6e')

plt.figure(figsize=(16, 7))
plt.plot(w1, Trans.real)
plt.xlabel("Energy [meV]")
plt.ylabel("$G_x$")
plt.ylim(-1.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7: CONDUCTANCIA DE ESPÍN – COMPONENTE Gy
# ══════════════════════════════════════════════════════════════════════════════
#
# La conductancia de espín en y se obtiene a partir de la parte IMAGINARIA
# de los términos cruzados entre amplitudes de espín ↑ y ↓:
#
#   Gy ∝ Im[T_du · T_u* - T_u · T_du* + ...]
#       = 4i · (T_du·T_u* - T_u·T_du*) + ...
#
# La proyección de σ_y = -i(σ+ - σ-) da lugar a los términos imaginarios
# cruzados. La fórmula usa el factor (4i) para extraer la parte de espín en y.
# ──────────────────────────────────────────────────────────────────────────────

N       = 28
gamma01 = 1.0
gamma02 = 1.0
l_R1    = 0.1
l_R12   = 0.0
l_R2    = 0.1
l_EO1   = 0.0
l_EO2   = 0.0
l_D     = 0.0
l_Z     = 0.0
eta     = 0.000001

E01 = np.zeros(N)
E02 = np.zeros(N)
theta = 0
M1  = 1.0
M2  = 1.0
U   = np.zeros(N)

B      = np.zeros(8*N, dtype=complex)
B[0]   = -1
B[N]   = -1
B[4*N] = -1
B[5*N] = -1

p     = 0.0
val   = 1.0*(1+p)*1j
val1  = 1.0*1j
Gamma1 = [val,  *np.zeros(N-2), val ]
Gamma2 = [val1, *np.zeros(N-2), val1]

gamma_out  = 0.0
gamma_out1 = 1.0

start_time = time.time()

# Los canales de Gy son los mismos que los de Gx (gamma_per1 = ±gamma_out1)
gamma_per = gamma_out;  gamma_per1 = gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans1u  = rho_w1u( -3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)
gamma_per = gamma_out;  gamma_per1 = gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans1d  = rho_w1d( -3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)
gamma_per = gamma_out;  gamma_per1 = gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans1ud = rho_w1ud(-3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)
gamma_per = gamma_out;  gamma_per1 = gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans1du = rho_w1du(-3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)

gamma_per = (-1.0)*gamma_out;  gamma_per1 = (-1.0)*gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans2u  = rho_w2u( -3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)
gamma_per = (-1.0)*gamma_out;  gamma_per1 = (-1.0)*gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans2d  = rho_w2d( -3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)
gamma_per = (-1.0)*gamma_out;  gamma_per1 = (-1.0)*gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans2ud = rho_w2ud(-3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)
gamma_per = (-1.0)*gamma_out;  gamma_per1 = (-1.0)*gamma_out1
E01 = np.zeros(N); E02 = np.zeros(N)
w1, Trans2du = rho_w2du(-3.0, 3.0, 901, B, N, E01, E02, theta, Gamma1, Gamma2, gamma01, gamma02, gamma_per, gamma_per1, U, M1, M2, l_R1, l_R2, l_D)

end_time  = time.time()
diff_time = end_time - start_time
print(f"El tiempo de ejecución fue de {diff_time} segundos.")

# ── Conductancia de espín Gy ───────────────────────────────────────────────────
# Gy = 4i · (T_du·T_u* - T_u·T_du* + ...) = 4i · 2i·Im[T_du·T_u*] + ...
# El prefactor 4i convierte la diferencia antisimétrica T·T* - T*·T = 2i·Im[T·T*]
# en la contribución al espín en y. El resultado final es real porque
# Im[algo imaginario puro] es real.
Trans = 4.0*(1j)*(Trans1du*np.conjugate(Trans1u)  - Trans1u*np.conjugate(Trans1du)
               + Trans2du*np.conjugate(Trans2u)  - Trans2u*np.conjugate(Trans2du)
               + Trans1d *np.conjugate(Trans1ud) - Trans1ud*np.conjugate(Trans1d)
               + Trans2d *np.conjugate(Trans2ud) - Trans2ud*np.conjugate(Trans2d))

ff = open('archivo_tiempo_cadena_bi_W_SO_Y.dat', 'w')
ff.write('El tiempo de ejecucion es: ' + str(diff_time))
ff.flush()

A      = np.stack((w1, Trans.real), axis=1)
nombre = "Salida_cadena_bi_W_SO_Y.dat"
np.savetxt(nombre, A, fmt='%.6e')

plt.figure(figsize=(16, 7))
plt.plot(w1, Trans.real)
plt.xlabel("Energy [meV]")
plt.ylabel("$G_y$")
plt.ylim(-1.0, 1.0)

# ── Mostrar todas las figuras ─────────────────────────────────────────────────
plt.show()
