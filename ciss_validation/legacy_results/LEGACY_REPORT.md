# Legacy Mathematica observable audit

Parameters: Lambda=0.002, t=0.005, Gamma=0.001, alpha=0.06283185307179587, Deco1=(3.992106913713086e-06+2.511620781172535e-07j), Deco2=(3.992106913713086e-06-2.511620781172535e-07j)

## Range convergence
```
DP=0: i1[-1,1]=1.57021814, i1[-2,2]=1.57046825, diff=0.00025; i2 diff=0.00025
DP=0.2: i1[-1,1]=2.35929537, i1[-2,2]=2.35954548, diff=0.00025; i2 diff=0.00025
DP=0.5: i1[-1,1]=2.36167925, i1[-2,2]=2.36192936, diff=0.00025; i2 diff=0.00025
```

## Complex transmission diagnostic
```
complex T1 max |Im| = 0.00947164, min Re = 4.74722e-06
complex T2 max |Im| = 0.00947164, min Re = 4.74722e-06
complex T1/T2 are not real transmissions
```

## Scan [-0.5,0.5]
DP range: -0.5 to 0.5 eV, N=101
DP=0: i1_A=1.57021814, i2_A=1.57037449, i2_B=1.57037449, P_A=-4.9782288e-05, P_B=-4.9782288e-05
i1_A min/max: 0.779473161, 2.36167925
i2_A min/max: 0.780116165, 2.36231871
max |P_A|=0.000412290353, max |P_B|=0.000359210063
max |A_A|=0.00082424088, max |A_B|=0.000718162155
max |R_A|=0.999902784, max |R_B|=1.00071868
max relative suppressions: up_A=0.00082424088, down_A=0.000824920813, up_B=0.000718678282, down_B=0.000718162155
max |P_exact_4x4|=1.85517241e-16, max |P_decoupled_physical|=0
near-90 quantities: none
sum near zero count, threshold 1e-3: 0

## Scan [-0.2,0.2]
DP range: -0.2 to 0.2 eV, N=81
DP=0: i1_A=1.57021814, i2_A=1.57037449, i2_B=1.57037449, P_A=-4.9782288e-05, P_B=-4.9782288e-05
i1_A min/max: 0.78185583, 2.35929537
i2_A min/max: 0.782490921, 2.35992692
max |P_A|=0.00040597834, max |P_B|=0.000357338196
max |A_A|=0.000811627176, max |A_B|=0.000714421103
max |R_A|=0.999903334, max |R_B|=1.00071493
max relative suppressions: up_A=0.000811627176, down_A=0.00081228645, up_B=0.000714931865, down_B=0.000714421103
max |P_exact_4x4|=2.01089591e-16, max |P_decoupled_physical|=0
near-90 quantities: none
sum near zero count, threshold 1e-3: 0

## Diagnostic answers
1. The notebook formula does not directly compute Landauer P=(Iup-Idown)/(Iup+Idown); it computes two legacy proxies i1/i2 and one can form ratios from them.
2. i1/i2 are not physical net currents: at DP=0 they are nonzero.
3. Version A is the literal notebook form with GaRs1 also in i2; version B is the symmetric corrected variant.
4. The complex K transmission form is not a real positive transmission.
5. Recommended wording: effective spin-channel imbalance / spin-dependent spectral asymmetry, separated from transport polarization.
6. Recommendation: D) Separar claramente legacy spin preference de transport polarization; for physical transport use C) redo with Landauer/NEGF.