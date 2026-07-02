# CISS validation report

## SymPy
```
C*Cdag =
Matrix([[Du*conjugate(Du) + V**2, V*(Du + conjugate(Dd))], [V*(Dd + conjugate(Du)), Dd*conjugate(Dd) + V**2]])
offdiag(0,1) = V*(Du + conjugate(Dd))
offdiag under Dd=-Du* = 0
decoupled-channel exact only in TRS limit
```

## Z3
```
1 TRS & V!=0 & offdiag!=0: unsat
2 offdiag=0 & V!=0 & |Du|^2!=|Dd|^2: unsat
3 |Du|^2!=|Dd|^2 & V!=0 & offdiag=0: unsat
4a |Du|^2!=|Dd|^2 & TRS: unsat
4b |Du|^2!=|Dd|^2 & broken TRS: sat; model=[yd = 0, xd = 0, xu = -1, yu = 0]
```

## Summary
| case | bias convention | max |P_exact| | max |P_dec| | high-bias P_exact |
|---|---:|---:|---:|---:|
| trs | left0 | 7.02959e-17 | 0 | -0 |
| trs | symmetric | 8.01639e-17 | 0 | 0 |
| noso | left0 | 0 | 0 | -0 |
| moderate | left0 | 0.00435672 | 0.00435872 | -0.000131576 |
| moderate | symmetric | 0.00440934 | 0.00441134 | 0.000252795 |
| moderate_rev_alpha | left0 | 0.00724883 | 0.00725139 | 0.000141897 |
| fig4_g1mev | left0 | 0.0014379 | 0.00143833 | 3.22827e-05 |
| fig4_g0p1mev | left0 | 0.00147493 | 0.00147539 | 3.18435e-05 |
| fig4_g0p06mev | left0 | 0.00147665 | 0.00147711 | 3.18231e-05 |
| fig4_g1mev | symmetric | 0.00144074 | 0.00144117 | 0.000764178 |
| fig4_g0p1mev | symmetric | 0.00147823 | 0.0014787 | 0.000773476 |
| fig4_g0p06mev | symmetric | 0.00147998 | 0.00148045 | 0.000773894 |

## Grid alias test
```
fig4_g1mev_left0: {'quad_P': 3.228267681958046e-05, 'grid_101_P': 0.023139062883352084, 'grid_201_P': 0.013437155041719143, 'grid_501_P': -0.020092637882611234, 'grid_1001_P': 0.0033128819789738713}
fig4_g0p1mev_left0: {'quad_P': 3.1843476450642137e-05, 'grid_101_P': 0.02392837297609701, 'grid_201_P': 0.01366575817644774, 'grid_501_P': -0.02964091133451364, 'grid_1001_P': 0.011912739045912038}
fig4_g0p06mev_left0: {'quad_P': 3.1823099805184696e-05, 'grid_101_P': 0.023933633432697037, 'grid_201_P': 0.013666345874383666, 'grid_501_P': -0.02972540196651321, 'grid_1001_P': 0.01206210065121786}
```

## Conclusions
P~90% exact 4x4 survives? no
P~90% decoupled appears? no
Chirality reversal alpha->-alpha flips sign? yes
Recommendation: C) Eliminar 90% y reformular resultados.