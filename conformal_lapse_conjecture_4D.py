from sage.all import *

# ─── 1. 4D Lorentzian manifold & chart ───────────────────────────────────────
M = Manifold(4, 'M', structure='Lorentzian', start_index=0)
X = M.chart('t x y z')
t, x, y, z = X[:]

# ─── 2. Symbolic fields ───────────────────────────────────────────────────────
alpha = function('alpha')(x, y, z)
Omega = function('Omega')(x, y, z)
vx, vy, vz = var('v_x v_y v_z', domain='real')

# ─── 3. ADM 3+1 metric  g_{\mu\nu} ───────────────────────────────────────────
# γ_ij = Ω⁴ δ_ij ,  β^i = -v^i ,  β_i = γ_ij β^j = -Ω⁴ v_i
gam = Omega**4                        # diagonal conformal factor (all three)
b1 = -vx;  b2 = -vy;  b3 = -vz       # shift components β^i
bc1 = gam * b1                        # β_x = γ_xx β^x
bc2 = gam * b2
bc3 = gam * b3
beta_sq = bc1*b1 + bc2*b2 + bc3*b3   # β_i β^i

g = M.metric('g')
g[0, 0] = -alpha**2 + beta_sq
g[0, 1] = bc1;  g[0, 2] = bc2;  g[0, 3] = bc3
g[1, 1] = gam
g[2, 2] = gam
g[3, 3] = gam
# SageMath automatically symmetrises the off-diagonal blocks

# ─── 4. Einstein tensor  G_{\mu\nu} ──────────────────────────────────────────
print("Computing Einstein tensor (this may take a few minutes)...")
G = g.einstein_tensor()
print("Done.\n")

# ─── 5. Eulerian observer  n^\mu = (1/α, β^x/α, β^y/α, β^z/α) ───────────────
n = M.vector_field('n')
n[0] = SR(1) / alpha
n[1] = b1 / alpha
n[2] = b2 / alpha
n[3] = b3 / alpha

n_form = n.down(g)    # n_\mu = g_{\mu\nu} n^\nu

# ─── 6. Eulerian energy density  ρ_E = (1/8π) G_{\mu\nu} n^\mu n^\nu ─────────
print("=== DENSIDAD DE ENERGÍA EULERIANA (ρ_E) ===")
rho_E_raw = SR(0)
for mu in range(4):
    for nu in range(4):
        rho_E_raw += G[mu, nu].expr() * n[mu].expr() * n[nu].expr()
rho_E = simplify_full(rho_E_raw / (8 * pi))
print(rho_E)

# ─── 7. Spatial projector  h_{\mu\nu} = g_{\mu\nu} + n_\mu n_\nu ─────────────
h = M.tensor_field(0, 2, name='h', sym=(0, 1))
for mu in range(4):
    for nu in range(4):
        h[mu, nu] = g[mu, nu] + n_form[mu].expr() * n_form[nu].expr()

h_up = h.up(g)   # h^{\mu\nu}

# ─── 8. Stress tensor trace  S = S^\mu_\mu = (1/8π) G_{\mu\nu} h^{\mu\nu} ───
print("\n=== TRAZA DE LOS ESFUERZOS ESPACIALES (S) ===")
S_trace_raw = SR(0)
for mu in range(4):
    for nu in range(4):
        S_trace_raw += G[mu, nu].expr() * h_up[mu, nu].expr()
S_trace = simplify_full(S_trace_raw / (8 * pi))
print(S_trace)

# ─── 9. Hawking-Ellis Type-I margin ──────────────────────────────────────────
# For an isotropic stress tensor each principal stress σ = S/3.
# Type-I condition: ρ_E > |σ|  ⟺  ρ_E - S/3 > 0
print("\n=== MARGEN TIPO I DE HAWKING-ELLIS ===")
print("Condición requerida para WEC: ρ_E - S/3 > 0")
HE_margin = simplify_full(rho_E - S_trace / 3)
print(HE_margin)

# ─── 10. Lapse Hessian compensation term ─────────────────────────────────────
# α Δ_γ α − (∂_i α)(∂^i α) − α²(K_ij K^{ij} + Ω^{-5} Δ_δ Ω)
# We derive Δ_γ α from the spatial Levi-Civita connection of γ = Ω⁴ δ
Sigma = Manifold(3, 'Sigma', start_index=0)
Xs = Sigma.chart('x y z')
xs, ys, zs = Xs[:]

alpha_s = function('alpha')(xs, ys, zs)
Omega_s = function('Omega')(xs, ys, zs)

gamma_s = Sigma.riemannian_metric('gamma')
gamma_s[0, 0] = Omega_s**4
gamma_s[1, 1] = Omega_s**4
gamma_s[2, 2] = Omega_s**4

nabla_s = gamma_s.connection()

# 1-form  dα
dalpha_s = Sigma.diff_form(1, name='dalpha')
for i in range(3):
    dalpha_s[i] = diff(alpha_s, Xs[i])

Hess = nabla_s(dalpha_s)   # ∇_i ∂_j α  (Hessian)
ginv_s = gamma_s.inverse()

Lap_gamma_alpha = SR(0)
for i in range(3):
    for j in range(3):
        Lap_gamma_alpha += ginv_s[i, j].expr() * Hess[i, j].expr()
Lap_gamma_alpha = simplify_full(Lap_gamma_alpha)

grad_alpha_sq = SR(0)
for i in range(3):
    for j in range(3):
        grad_alpha_sq += (ginv_s[i, j].expr()
                          * diff(alpha_s, Xs[i]) * diff(alpha_s, Xs[j]))
grad_alpha_sq = simplify_full(grad_alpha_sq)

Lap_delta_Omega_s = sum(diff(Omega_s, Xs[i], 2) for i in range(3))

# Extrinsic curvature K_ij = (∇_i β_j + ∇_j β_i) / 2α  for constant β^i
beta_low_s = Sigma.tensor_field(0, 1, name='beta')
beta_low_s[0] = Omega_s**4 * (-vx)
beta_low_s[1] = Omega_s**4 * (-vy)
beta_low_s[2] = Omega_s**4 * (-vz)

D_beta_s = nabla_s(beta_low_s)
K_s = Sigma.tensor_field(0, 2, name='K', sym=(0, 1))
for i in range(3):
    for j in range(3):
        K_s[i, j] = (D_beta_s[j, i] + D_beta_s[i, j]) / (2 * alpha_s)

K_up_s = K_s.up(gamma_s)
K_sq_s = SR(0)
for i in range(3):
    for j in range(3):
        K_sq_s += K_s[i, j].expr() * K_up_s[i, j].expr()
K_sq_s = simplify_full(K_sq_s)

lhs_comp = alpha_s * Lap_gamma_alpha - grad_alpha_sq
rhs_comp = alpha_s**2 * (K_sq_s + Omega_s**(-5) * Lap_delta_Omega_s)
compensation = simplify_full(lhs_comp - rhs_comp)

print("\n=== TÉRMINO DE COMPENSACIÓN ELÍPTICA ===")
print("[α Δ_γ α − |∇α|²_γ] − α²[K_ij K^{ij} + Ω^{-5} Δ_δ Ω]")
print(compensation)

# ─── 11. Verdict ─────────────────────────────────────────────────────────────
print("\n=== VEREDICTO ===")
he_zero = bool(HE_margin == SR(0))
comp_zero = bool(compensation == SR(0))

if he_zero:
    print("HAWKING-ELLIS MARGIN: vanishes identically (saturation at equality).")
else:
    print("HAWKING-ELLIS MARGIN: non-trivial — see expression above.")
    print("WEC Type-I is satisfied wherever the printed expression is > 0.")

if comp_zero:
    print("COMPENSATION TERM: vanishes identically.")
    print("Conjecture algebraically saturated for rigid-translation PG seed.")
else:
    print("COMPENSATION TERM: non-trivial residual — positivity must be verified.")
    print("Inspect the expression above to determine the required lapse regularity.")
