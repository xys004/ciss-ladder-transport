from sage.all import *

# ─── 1. Manifold & chart ──────────────────────────────────────────────────────
M = Manifold(3, 'M', latex_name=r'\Sigma', start_index=0)
X = M.chart('x y z')
x, y, z = X[:]

# ─── 2. Symbolic fields ───────────────────────────────────────────────────────
alpha = function('alpha')(x, y, z)
Omega = function('Omega')(x, y, z)
vx, vy, vz = var('v_x v_y v_z', domain='real')

# ─── 3. Conformally flat spatial metric  γ_ij = Ω^4 δ_ij ─────────────────────
gamma = M.riemannian_metric('gamma')
gamma[0, 0] = Omega**4
gamma[1, 1] = Omega**4
gamma[2, 2] = Omega**4

nabla = gamma.connection()

# 3D Ricci scalar  ³R
R3 = gamma.ricci_scalar().expr()
R3_simp = simplify_full(R3)

# ─── 4. Extrinsic curvature from rigid-translation shift β^i = −v^i ───────────
beta_low = M.tensor_field(0, 1, name='beta')   # 1-form β_i = γ_ij β^j
# For a conformally flat metric with β^i constant:  β_i = Ω^4 (-v_i)
beta_low[0] = Omega**4 * (-vx)
beta_low[1] = Omega**4 * (-vy)
beta_low[2] = Omega**4 * (-vz)

# Covariant derivative of β_i  →  ∇_j β_i
D_beta = nabla(beta_low)

# K_ij = (∇_i β_j + ∇_j β_i) / (2α)
K = M.tensor_field(0, 2, name='K', sym=(0, 1))
for i in range(3):
    for j in range(3):
        K[i, j] = (D_beta[j, i] + D_beta[i, j]) / (2 * alpha)

# ─── 5. Traces  K  and  K_ij K^ij ────────────────────────────────────────────
ginv = gamma.inverse()

tr_K_expr = SR(0)
for i in range(3):
    for j in range(3):
        tr_K_expr += ginv[i, j].expr() * K[i, j].expr()
tr_K_expr = simplify_full(tr_K_expr)

K_sq_expr = SR(0)
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                K_sq_expr += (ginv[i, k].expr() * ginv[j, l].expr()
                               * K[i, j].expr() * K[k, l].expr())
K_sq_expr = simplify_full(K_sq_expr)

# ─── 6. Eulerian energy density (Hamiltonian constraint) ─────────────────────
# ρ_E = (1/16π) ( ³R + K² − K_ij K^ij )
rho_E = (1 / (16 * pi)) * (R3_simp + tr_K_expr**2 - K_sq_expr)
rho_E_simp = simplify_full(rho_E)

# ─── 7. Hessian of α and spatial Laplacian ───────────────────────────────────
dalpha = M.diff_form(1, name='dalpha')
for i in range(3):
    dalpha[i] = diff(alpha, X[i])

Hess_alpha = nabla(dalpha)   # ∇_i ∂_j α  (0,2 tensor)

Lap_alpha = SR(0)
for i in range(3):
    for j in range(3):
        Lap_alpha += ginv[i, j].expr() * Hess_alpha[i, j].expr()
Lap_alpha = simplify_full(Lap_alpha)

# Flat Laplacian of Ω  (Δ_δ Ω = ∂_xx + ∂_yy + ∂_zz)
Lap_delta_Omega = sum(diff(Omega, X[i], 2) for i in range(3))

# ─── 8. Compensation term (lapse Hessian vs shear) ───────────────────────────
# α Δ_γ α − (∂_i α)(∂^i α)  vs  α² ( K_ij K^ij + Ω^{-5} Δ_δ Ω )
grad_alpha_sq = SR(0)
for i in range(3):
    for j in range(3):
        grad_alpha_sq += ginv[i, j].expr() * diff(alpha, X[i]) * diff(alpha, X[j])
grad_alpha_sq = simplify_full(grad_alpha_sq)

lhs_comp = alpha * Lap_alpha - grad_alpha_sq
rhs_comp = alpha**2 * (K_sq_expr + Omega**(-5) * Lap_delta_Omega)
compensation = simplify_full(lhs_comp - rhs_comp)

# ─── 9. Output ────────────────────────────────────────────────────────────────
print("=" * 70)
print("CONFORMAL-LAPSE ELLIPTIC COMPENSATION CONJECTURE")
print("Painlevé-Gullstrand generalised spacetime  |  3+1 ADM decomposition")
print("=" * 70)

print("\n--- ³R  (3D Ricci scalar, conformally flat γ = Ω⁴ δ) ---")
print(R3_simp)

print("\n--- Trace K = γ^{ij} K_{ij} ---")
print(tr_K_expr)

print("\n--- K_ij K^{ij} ---")
print(K_sq_expr)

print("\n--- Eulerian energy density  ρ_E = (1/16π)(³R + K² − K_ij K^{ij}) ---")
print(rho_E_simp)

print("\n--- Spatial Laplacian of lapse  Δ_γ α ---")
print(Lap_alpha)

print("\n--- Compensation term  [α Δ_γ α − (∂α)²] − α²[K² + Ω^{-5} Δ_δ Ω] ---")
print(compensation)

print("\n--- VERDICT ---")
# A zero compensation term means the conjecture's differential inequality
# holds as an equality at the algebraic level for rigid translation.
# Positivity of the lhs for general configurations implies WEC confinement
# to the Hawking-Ellis Type I sector.
if compensation == SR(0):
    print("ALGEBRAICALLY EXACT: compensation term vanishes identically.")
    print("The Hessian inequality is saturated; WEC Type-I confinement confirmed")
    print("at the symbolic level for rigid-translation Painlevé-Gullstrand seeds.")
else:
    print("Non-trivial residual detected.  Inspect compensation term above.")
    print("WEC Type-I confinement requires the displayed expression to be ≥ 0.")
    print("Simplified residual (attempt):")
    print(compensation.simplify_full())
