**PASO 1 — ANÁLISIS ONTOLÓGICO:**
- **Objetos:** Variedades diferenciables Lorentzianas (espaciotiempo 4D con signatura -,+,+,+). Campos escalares continuos y derivables (lapso $\alpha(x,y,z)$, factor conforme espacial $\Omega(x,y,z)$). Campos vectoriales (vector de desplazamiento $\beta^i$, cuadrivector normal euleriano $n^\mu$). Tensores de segundo orden (tensor métrico 4D $g_{\mu\nu}$, métrica espacial 3D $\gamma_{ij}$, tensor de Einstein $G_{\mu\nu}$, curvatura extrínseca $K_{ij}$, tensor de esfuerzos espaciales $S_{ij}$). Operadores diferenciales geométricos (conexión de Levi-Civita $D_i$, operador Hessiano espacial $D_i D_j$, Laplaciano plano $\Delta_\delta$, Laplaciano espacial $\Delta_\gamma$).
- **Objetivo:** Simplificación algebraica tensorial y derivación analítica. Verificación formal y satisfactibilidad de una hipótesis matemática (comprobar si una desigualdad diferencial específica acoplando el Hessiano del lapso y el factor conforme inyecta suficiente positividad en la densidad de energía euleriana $\rho_E$ para dominar los esfuerzos principales negativos $\sigma_a$, confinando el sistema en el sector de energía Tipo I de Hawking-Ellis).

**PASO 2 — MATRIZ DE DECISIÓN TECNOLÓGICA:**
- **Herramienta seleccionada:** **SageMath** (vía WSL2) emparejado con su motor simbólico subyacente **SymPy**.
- **Justificación:** El problema requiere geometría diferencial rigurosa sobre variedades continuas parametrizadas. El módulo `SageManifolds` de SageMath es el estándar de oro de código abierto para cálculos exactos de tensores de métrica, conexiones de Levi-Civita y tensores de curvatura (Riemann, Ricci, Einstein) sin errores de expansión manual. Dado que trabajamos con funciones simbólicas genéricas ($\alpha, \Omega$), Sage delega la simplificación y el aislamiento analítico de términos (como el Hessiano) a SymPy. Z3 no es útil aquí para cálculo simbólico tensorial. QuTiP y GAP se descartan por completo debido a que el dominio es relatividad general clásica y topología diferencial, no cuántica ni teoría de grupos discretos.

**PASO 3 — PROCEDIMIENTO:**
1. **Inicialización Topológica:** Definir una variedad espaciotemporal 4D $\mathcal{M}$ equipada con una métrica Lorentziana y un sistema de coordenadas cartesianas globales $(t, x, y, z)$.
2. **Definición Simbólica:** Declarar las coordenadas espaciales como variables y definir simbólicamente las funciones de clase $C^2$: lapso $\alpha(x,y,z)$ y factor conforme $\Omega(x,y,z)$. Definir componentes constantes del vector de velocidad $v_x, v_y, v_z$.
3. **Construcción de la Métrica ADM:** 
   - Construir la métrica espacial conformemente plana $\gamma_{ij} = \Omega^4 \delta_{ij}$.
   - Establecer el vector de desplazamiento (shift) $\beta^i = (-v_x, -v_y, -v_z)$ representando la traslación rígida de la burbuja warp.
   - Ensamblar las componentes de la métrica 4D completa $g_{\mu\nu}$ mediante la parametrización de Painlevé-Gullstrand generalizada.
4. **Cálculo de Tensores Geométricos:** Solicitar al motor computacional calcular automáticamente la conexión covariante, la curvatura de Riemann y el tensor de Einstein $G_{\mu\nu}$ a partir de $g_{\mu\nu}$.
5. **Definición del Observador Euleriano y Proyección de Energía:**
   - Construir el cuadrivector unitario temporal ortogonal a las rebanadas espaciales: $n^\mu = \frac{1}{\alpha}(1, \beta^x, \beta^y, \beta^z)$.
   - Calcular la densidad de energía euleriana: $\rho_E = \frac{1}{8\pi} G_{\mu\nu} n^\mu n^\nu$.
   - Definir el proyector espacial $h_{\mu\nu} = g_{\mu\nu} + n_\mu n_\nu$ y computar el tensor de tensiones espaciales $S_{\mu\nu} = \frac{1}{8\pi} G_{\rho\sigma} h^\rho_{\,\,\mu} h^\sigma_{\,\,\nu}$.
6. **Aserción de la Conjetura de Compensación:**
   - Calcular la traza del tensor de tensiones $S$.
   - Formular la desigualdad maestra de la condición de Hawking-Ellis evaluando la expresión analítica de $\rho_E - \frac{1}{3}S$.
   - Verificar simbólicamente que los coeficientes que acompañan al Laplaciano espacial del lapso ($\Delta_\gamma \alpha$) y a la curvatura escalar conforme ($\Delta_\delta \Omega$) en esta expresión permitan compensar activamente la negatividad del término de corte (shear) inducido por el desplazamiento $\beta^i$.

**PASO 4 — PSEUDOCÓDIGO:**

```python
# EJECUTAR EN ENTORNO SAGEMATH (WSL2)
from sage.all import *

# 1. Definición del Espaciotiempo 4D y Coordenadas
M = Manifold(4, 'M', structure='Lorentzian')
X.<t, x, y, z> = M.chart()

# 2. Variables Simbólicas y Funciones C^2
alpha = function('alpha')(x, y, z)
Omega = function('Omega')(x, y, z)
vx, vy, vz = var('v_x v_y v_z', domain='real')

# 3. Construcción de la Métrica ADM 3+1
g = M.metric()

# Componentes espaciales conformemente planos
Omega4 = Omega**4
beta_x, beta_y, beta_z = -vx, -vy, -vz

# Formas covariantes del desplazamiento: beta_i = gamma_ij * beta^j
beta_cov_x = Omega4 * beta_x
beta_cov_y = Omega4 * beta_y
beta_cov_z = Omega4 * beta_z
beta_sq = beta_cov_x*beta_x + beta_cov_y*beta_y + beta_cov_z*beta_z

# Asignación tensorial de la métrica 4D
g[0,0] = -alpha**2 + beta_sq
g[0,1] = beta_cov_x
g[0,2] = beta_cov_y
g[0,3] = beta_cov_z
g[1,1] = Omega4
g[2,2] = Omega4
g[3,3] = Omega4
# SageManifolds maneja implícitamente g[i,j] = g[j,i]

# 4. Cálculo del Tensor de Einstein (G_{\mu\nu})
# Este paso computa la conexión de Levi-Civita y la curvatura automáticamente.
G = g.einstein_tensor()

# 5. Observador Euleriano y Proyecciones de Energía-Impulso
n = M.vector_field('n')
n[0] = 1 / alpha
n[1] = beta_x / alpha
n[2] = beta_y / alpha
n[3] = beta_z / alpha

# Densidad de Energía Euleriana: rho_E = 1/(8*pi) * G(n, n)
rho_E = (1 / (8*pi)) * G(n, n).expr()
rho_E_simp = rho_E.simplify_full()

# Tensor proyector ortogonal espacial h_{\mu\nu} = g_{\mu\nu} + n_\mu n_\nu
n_cov = n.down(g)
h = M.tensor_field(0, 2, 'h', sym=(0,1))
for i in range(4):
    for j in range(4):
        h[i,j] = g[i,j] + n_cov[i]*n_cov[j]

# Traza de las tensiones espaciales S = 1/(8*pi) * G_{ab} h^{ab}
h_up = h.up(g)
S_trace = sum(sum((1 / (8*pi)) * G[i,j].expr() * h_up[i,j].expr() for i in range(4)) for j in range(4))
S_trace_simp = S_trace.simplify_full()

# 6. Evaluación Formal de la Hipótesis
print("=== VERIFICACIÓN FORMAL DE LA CONJETURA TIPO I ===")

# Para confinar el sistema en el sector Tipo I, la energía debe dominar 
# las tensiones principales. Evaluamos el umbral rho_E - S/3.
Compensacion_Umbral = (rho_E_simp - (1/3)*S_trace_simp).simplify_full()

print("Cota diferencial requerida (rho_E - S/3):")
print(Compensacion_Umbral)

# BLOQUE DE EVALUACIÓN LÓGICA (Éxito o Refutación):
# - ÉXITO: Si la expresión matemática `Compensacion_Umbral` muestra algebraicamente
#   que las derivadas de segundo orden (+ C_1 * D_i D^i alpha) y (- C_2 * Delta_delta Omega) 
#   poseen signos geométricos que permiten compensar el término cinemático 
#   negativo dependiente del desplazamiento (v^2).
# - REFUTACIÓN: Si las segundas derivadas no aparecen, se cancelan, 
#   o tienen un signo invertido que exacerba el problema de energía negativa.
```