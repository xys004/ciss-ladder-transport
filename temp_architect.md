PASO 1 — ANÁLISIS ONTOLÓGICO:
Objetos matemáticos involucrados: 
- Variedades Lorentzianas (espaciotiempo 4D, signatura -,+,+,+).
- Campos escalares abstractos $C^2$ (función de lapso $\alpha(x^i)$ y factor conforme $\Omega(x^i)$).
- Campos vectoriales (vector de desplazamiento/shift $\beta^i$ y cuadrivector normal euleriano $n^\mu$).
- Tensores de orden 2 (métrica espacial $\gamma_{ij}$, métrica ADM espaciotemporal $g_{\mu\nu}$, tensor de Einstein $G_{\mu\nu}$, curvatura extrínseca $K_{ij}$, tensor de energía-impulso $T_{\mu\nu}$ y proyecciones espaciales $S_{ij}$).
- Operadores diferenciales geométricos (conexión de Levi-Civita, Laplaciano plano $\Delta_\delta$, Laplaciano espacial $\Delta_\gamma$, operador Hessiano espacial $D_i D_j$).

Objetivo de la operación: 
Derivación analítica tensorial automatizada y simplificación algebraica para formular y verificar formalmente la conjetura del artículo: evaluar si las expresiones diferenciales resultantes de $\rho_E$ (densidad de energía euleriana) y $S_{ij}$ (esfuerzos principales) muestran un acoplamiento donde el Hessiano del lapso y el Laplaciano del factor conforme aportan la positividad escalar necesaria para compensar matemáticamente los términos cinemáticos negativos de corte (shear) y garantizar que la materia requerida satisfaga las condiciones Tipo I de Hawking-Ellis.

PASO 2 — MATRIZ DE DECISIÓN:
Herramienta elegida: SageMath (utilizando su entorno simbólico interno SymPy para simplificación).
Justificación: El análisis tensorial en relatividad general sobre una métrica genérica no especificada explícitamente (donde el lapso y factor conforme quedan como funciones de las coordenadas) hace imposible usar entornos numéricos puros y hace tedioso/propenso a errores usar sólo manipulación algebraica simple. `SageManifolds` en SageMath soporta nativamente variedades diferenciables, campos de tensores métricos, conexiones afines y cálculo de curvatura (Ricci/Einstein) directo a partir de una definición métrica. SymPy servirá de backend para reducir las abstracciones generadas. Z3, QuTiP o GAP no resuelven dominios continuos de geometría seudo-riemanniana simbólica.

PASO 3 — PROCEDIMIENTO:
1. Instanciar la variedad de espaciotiempo 4D $\mathcal{M}$ como variedad Lorentziana y su carta coordenada $(t, x, y, z)$.
2. Declarar paramétricamente las funciones abstractas y simbólicas $\alpha(x, y, z)$ y $\Omega(x, y, z)$, junto con las variables constantes de velocidad $v_x, v_y, v_z$.
3. Configurar la métrica 3+1 (Painlevé-Gullstrand con lapso):
   - Definir la parte espacial como $\gamma_{ij} = \Omega^4 \delta_{ij}$.
   - Establecer el shift (vector de arrastre macroscópico) como $\beta^i = (-v_x, -v_y, -v_z)$.
   - Ensamblar la métrica ADM 4D $g_{\mu\nu}$.
4. Computar las conexiones, curvaturas y derivar el tensor de Einstein $G_{\mu\nu}$ en automático.
5. Efectuar la descomposición del observador euleriano:
   - Declarar $n^\mu = \frac{1}{\alpha}(1, \beta^x, \beta^y, \beta^z)$.
   - Proyectar la densidad de energía euleriana: $\rho_E = \frac{1}{8\pi} G_{\mu\nu}n^\mu n^\nu$.
   - Definir el tensor métrico inducido $h_{\mu\nu} = g_{\mu\nu} + n_\mu n_\nu$ para proyectar los esfuerzos espaciales $S_{\mu\nu}$.
6. Validación formal de la hipótesis:
   - Calcular la traza de los esfuerzos $S$.
   - Extraer la desigualdad fundamental $\rho_E - \frac{1}{3}S$ (para evaluar el margen Tipo I frente a los esfuerzos promedio).
   - Analizar si el resultado incluye en su estructura los términos Laplacianos de $\alpha$ y $\Omega$ con los signos geométricos correctos que admitan una compensación estricta al término cinemático perjudicial (velocidad al cuadrado).

PASO 4 — PSEUDOCÓDIGO:
```python
# EJECUTAR EN ENTORNO SAGEMATH (WSL2)
from sage.all import *

# 1. Definición del Espaciotiempo 4D y Carta de Coordenadas
M = Manifold(4, 'M', structure='Lorentzian')
X.<t, x, y, z> = M.chart()

# 2. Variables Simbólicas de Campos C^2 y Cinemática
alpha = function('alpha')(x, y, z)
Omega = function('Omega')(x, y, z)
vx, vy, vz = var('v_x v_y v_z', domain='real')

# 3. Construcción Paramétrica de la Métrica ADM
g = M.metric()

# Factores de forma espaciales (conformemente planos)
Omega4 = Omega**4
beta_x, beta_y, beta_z = -vx, -vy, -vz

# Producto y proyección del vector shift
beta_cov_x = Omega4 * beta_x
beta_cov_y = Omega4 * beta_y
beta_cov_z = Omega4 * beta_z
beta_sq = beta_cov_x*beta_x + beta_cov_y*beta_y + beta_cov_z*beta_z

# Asignación del Tensor de Métrica
g[0,0] = -alpha**2 + beta_sq
g[0,1] = beta_cov_x
g[0,2] = beta_cov_y
g[0,3] = beta_cov_z
g[1,1] = Omega4
g[2,2] = Omega4
g[3,3] = Omega4
# Sage rellena por simetría las componentes como g[i,0]

# 4. Cálculo Automático de Tensores Geométricos y Curvatura
G = g.einstein_tensor()

# 5. Proyecciones para Observador Euleriano
n = M.vector_field('n')
n[0] = 1 / alpha
n[1] = beta_x / alpha
n[2] = beta_y / alpha
n[3] = beta_z / alpha

# Energía Euleriana Observable
rho_E = (1 / (8*pi)) * G(n, n).expr()
rho_E_simp = rho_E.simplify_full()

# Definición del Proyector Espacial y Extracción de Tensiones
n_cov = n.down(g)
h = M.tensor_field(0, 2, 'h', sym=(0,1))
for i in range(4):
    for j in range(4):
        h[i,j] = g[i,j] + n_cov[i]*n_cov[j]

h_up = h.up(g)
S_trace = sum(sum((1 / (8*pi)) * G[i,j].expr() * h_up[i,j].expr() for i in range(4)) for j in range(4))
S_trace_simp = S_trace.simplify_full()

# 6. Aserción de la Condición Elíptica (Tipo I de Hawking-Ellis)
print("=== VERIFICACIÓN DE COMPENSACIÓN ELÍPTICA (TIPO I) ===")

# Diferencial para determinar el umbral entre energía y tensiones
Condicion_Compensacion = (rho_E_simp - (1/3)*S_trace_simp).simplify_full()

print("Ecuación diferencial resultante (rho_E - S/3):")
print(Condicion_Compensacion)

# CONDICIÓN DE EVALUACIÓN DE ÉXITO O REFUTACIÓN:
# ÉXITO: Si la ecuación `Condicion_Compensacion` arroja explícitamente términos
# dependientes de D_i D_j alpha (derivadas de segundo orden) y de Delta Omega
# cuyos signos intrínsecos indiquen que un lapso fuertemente convexo puede 
# escalar positivamente sobre el término cuadrático de la velocidad (shear).
# REFUTACIÓN: Si en la expansión dichos términos de derivadas segundas espaciales
# se anulan o sus signos actúan en la misma dirección que la cizalla, 
# demostrando que no pueden utilizarse para inyectar positividad energética.
```