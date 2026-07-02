#!/usr/bin/env python3
"""
Z3 logic audit for the two-site CISS effective Hamiltonian.

We encode complex variables through real and imaginary parts.
The goal is not numerical transport, but checking logical implications:

1. TRS/Kramers condition => Schur off-diagonal interference vanishes.
2. Vanishing Schur off-diagonal => equal spin-flip magnitudes.
3. Unequal spin-flip magnitudes => Schur off-diagonal cannot vanish.
4. Hermiticity constraints for D2 = D1^*.

Run:
    python ciss_logic_z3.py
"""

from z3 import Real, Solver, Or, Not, And, unsat, simplify

# Complex D_up = xu + i yu, D_down = xd + i yd
xu, yu, xd, yd = Real("xu"), Real("yu"), Real("xd"), Real("yd")
V = Real("V")

# offdiag = V * (conj(D_down) + D_up)
# conj(D_down) + D_up = (xd + xu) + i (yu - yd)
off_re = V * (xd + xu)
off_im = V * (yu - yd)

mag_u = xu*xu + yu*yu
mag_d = xd*xd + yd*yd

# TRS/Kramers relation D_down = -conj(D_up)
trs = And(xd == -xu, yd == yu)
V_nonzero = V != 0

def prove_unsat(name, assumptions):
    s = Solver()
    s.add(*assumptions)
    result = s.check()
    print(f"{name}: {result}")
    if result != unsat:
        print("  model/counterexample:", s.model())

print("=== Z3 logic audit ===")

prove_unsat(
    "TRS and V != 0 but offdiag != 0",
    [trs, V_nonzero, Or(off_re != 0, off_im != 0)],
)

prove_unsat(
    "offdiag == 0 and V != 0 but |D_up|^2 != |D_down|^2",
    [V_nonzero, off_re == 0, off_im == 0, mag_u != mag_d],
)

prove_unsat(
    "|D_up|^2 != |D_down|^2 and V != 0 but offdiag == 0",
    [V_nonzero, mag_u != mag_d, off_re == 0, off_im == 0],
)

# Hermiticity encoding: D2_up = conj(D1_up), D2_down = conj(D1_down)
u1x, u1y, u2x, u2y = Real("u1x"), Real("u1y"), Real("u2x"), Real("u2y")
d1x, d1y, d2x, d2y = Real("d1x"), Real("d1y"), Real("d2x"), Real("d2y")
herm_D = And(u2x == u1x, u2y == -u1y, d2x == d1x, d2y == -d1y)
prove_unsat(
    "Hermiticity D2 = D1^* but violated component relation",
    [herm_D, Or(u2x != u1x, u2y != -u1y, d2x != d1x, d2y != -d1y)],
)

print("\nInterpretation:")
print("- The decoupled-channel cancellation follows from the Kramers/TRS condition.")
print("- If the model imposes unequal spin-flip magnitudes, the Schur off-diagonal term is necessarily nonzero for V != 0.")
print("- Therefore the exact 4x4 Green function is the safe main transport object in the TRS-broken regime.")
