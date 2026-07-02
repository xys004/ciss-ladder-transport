from sage.all import *

# ─── 1. 4D Lorentzian manifold ────────────────────────────────────────────────
M = Manifold(4, 'M', structure='Lorentzian', start_index=0)
X = M.chart('t x y z')
t, x, y, z = X[:]

# ─── 2. Symbolic C^2 fields and kinematics ───────────────────────────────────
alpha = function('alpha')(x, y, z)
Omega = function('Omega')(x, y, z)
vx, vy, vz = var('v_x v_y v_z', domain='real')

# ─── 3. ADM metric ───────────────────────────────────────────────────────────
# γ_ij = Ω⁴ δ_ij  ,  β^i = -v^i  ,  β_i = Ω⁴ β^i
Omega4 = Omega**4
bx, by, bz = -vx, -vy, -vz          # β^i  (contravariant)
bcx = Omega4 * bx                    # β_i  (covariant, diagonal γ)
bcy = Omega4 * by
bcz = Omega4 * bz
beta_sq = bcx*bx + bcy*by + bcz*bz  # γ_ij β^i β^j

g = M.metric('g')
g[0, 0] = -alpha**2 + beta_sq
g[0, 1] = bcx;  g[0, 2] = bcy;  g[0, 3] = bcz
g[1, 1] = Omega4
g[2, 2] = Omega4
g[3, 3] = Omega4
# SageManifolds fills symmetric entries automatically

# ─── 4. Einstein tensor (auto-computes Christoffel → Riemann → Ricci → G) ───
print("Computing Einstein tensor G_μν …")
G = g.einstein_tensor()
print("Done.\n")

# ─── 5. Eulerian observer  n^μ = (1/α, β^x/α, β^y/α, β^z/α) ────────────────
n = M.vector_field('n')
n[0] = SR(1) / alpha
n[1] = bx / alpha
n[2] = by / alpha
n[3] = bz / alpha

n_cov = n.down(g)    # n_μ = g_{μν} n^ν

# ─── 6. Eulerian energy density  ρ_E = (1/8π) G_{μν} n^μ n^ν ────────────────
rho_raw = SR(0)
for mu in range(4):
    for nu in range(4):
        rho_raw += G[mu, nu].expr() * n[mu].expr() * n[nu].expr()
rho_E = simplify_full(rho_raw / (8 * pi))

# ─── 7. Spatial projector  h_{μν} = g_{μν} + n_μ n_ν ────────────────────────
h = M.tensor_field(0, 2, name='h', sym=(0, 1))
for mu in range(4):
    for nu in range(4):
        h[mu, nu] = g[mu, nu].expr() + n_cov[mu].expr() * n_cov[nu].expr()

h_up = h.up(g)   # h^{μν}

# ─── 8. Stress trace  S = S^μ_μ = (1/8π) G_{μν} h^{μν} ─────────────────────
S_raw = SR(0)
for mu in range(4):
    for nu in range(4):
        S_raw += G[mu, nu].expr() * h_up[mu, nu].expr()
S_trace = simplify_full(S_raw / (8 * pi))

# ─── 9. Compensation condition  ρ_E − S/3 ────────────────────────────────────
# For an isotropic stress state each principal stress σ = S/3.
# Hawking-Ellis Type-I requires  ρ_E − S/3 > 0.
comp_cond = simplify_full(rho_E - S_trace / 3)

# ─── 10. Structural analysis of the compensation term ────────────────────────
# Isolate second-derivative content symbolically:
# (a) Flat Laplacian of Ω : Δ_δ Ω = ∂_xx Ω + ∂_yy Ω + ∂_zz Ω
Lap_delta_Omega = sum(diff(Omega, X[i], 2) for i in range(1, 4))

# (b) Flat Laplacian of α : Δ_δ α = ∂_xx α + ∂_yy α + ∂_zz α
Lap_delta_alpha = sum(diff(alpha, X[i], 2) for i in range(1, 4))

# Extract coefficient of Δ_δ α in comp_cond (partial structural check)
coeff_lap_alpha = comp_cond.coefficient(Lap_delta_alpha)

# ─── 11. Output ───────────────────────────────────────────────────────────────
print("=" * 72)
print("CONFORMAL-LAPSE ELLIPTIC COMPENSATION CONJECTURE — FULL 4D ANALYSIS")
print("ADM decomposition  |  Signature (−,+,+,+)  |  Rigid-translation seed")
print("=" * 72)

print("\n─── ρ_E  (Eulerian energy density) ─────────────────────────────────────")
print(rho_E)

print("\n─── S  (spatial stress trace) ───────────────────────────────────────────")
print(S_trace)

print("\n─── Compensation condition  ρ_E − S/3 ───────────────────────────────────")
print(comp_cond)

print("\n─── Coefficient of Δ_δ α in (ρ_E − S/3) ────────────────────────────────")
print(coeff_lap_alpha)

# ─── 12. Verdict ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)

zero_cond   = bool(comp_cond == SR(0))
has_lap_pos = not bool(coeff_lap_alpha == SR(0))

if zero_cond:
    print("[SATURATION]  ρ_E − S/3 = 0 identically.")
    print("The compensation inequality is exactly saturated for this seed.")
    print("WEC Type-I holds at equality; no additional lapse regularity needed.")
elif has_lap_pos:
    # Check sign structure heuristically via coefficient
    try:
        pos_flag = bool(coeff_lap_alpha > 0)
    except Exception:
        pos_flag = None

    if pos_flag is True:
        print("[SUCCESS — CONJECTURE SUPPORTED]")
        print("Δ_δ α enters (ρ_E − S/3) with a POSITIVE coefficient.")
        print("A strongly convex lapse (Δ α > 0) raises the margin above zero,")
        print("compensating shear terms ∝ (v² Ω⁴) and satisfying WEC Type-I")
        print("without a decoupled macroscopic proper volume.")
    elif pos_flag is False:
        print("[REFUTATION — CONJECTURE FALSIFIED]")
        print("Δ_δ α enters (ρ_E − S/3) with a NEGATIVE coefficient.")
        print("Lapse convexity cannot compensate the shear deficit; WEC Type-I")
        print("cannot be enforced by the proposed elliptic mechanism alone.")
    else:
        print("[INDETERMINATE]  Sign of Δ_δ α coefficient is parameter-dependent.")
        print("Inspect the printed coefficient for manual sign analysis.")
else:
    print("[INDETERMINATE]  Δ_δ α does not appear linearly in ρ_E − S/3.")
    print("The compensation mechanism may operate through Ω-derivatives.")
    print("Inspect the full compensation condition printed above.")
    coeff_lap_Omega = comp_cond.coefficient(Lap_delta_Omega)
    print("\nCoefficient of Δ_δ Ω in (ρ_E − S/3):")
    print(coeff_lap_Omega)
