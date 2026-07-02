# Physical polarization search report

Stage 1 used the saved 50,000-point random scan; this finalizer locally perturbed the top 30 and strictly validated the top 55 with quad.
Accepted candidates passing filters: 55 / 55.
Accepted candidates with chirality_score < 0.2: 0 / 55.
Note: the raw CSV retains the top-ranked raw candidates from the 50,000-point stage, not every rejected random draw.

## Best Robust Candidate
largest robust |P_tr| = 0.999988234
P_tr = -0.999988234
I_up = -3.686546832e-11, I_down = -6.266450885e-06, I_tot = -6.266487751e-06
A_ch = 0.999994117, A_peak = 0.181678063
V=0.00115906883, Lambda=0.05, GammaL=1.10648115e-05, GammaR=1e-05
eps1=0.0321598778, eps2=0.0197121685, a_down/a_up=9.8833826
theta_up=1.61055615, theta_down=3.24587473, bias=-0.0400926618
P_rev=-0.999988229, chirality_score=2
P_TRS_control=3.529e-14, P_noSO_control=-0.000e+00
zero_bias Iup/Idown=0.000e+00/0.000e+00
left0 same candidate: P_tr=-0.99998772, I_tot=-5.720256697e-06, passes=True

## Stricter Current Threshold
best accepted with |I_tot|>1e-6: 0.999988234

## Answers
A. Largest robust physical |P_tr| found: 0.999988234.
B. Survives all filters: True ().
C. Regime: strongly asymmetric effective spin-flip amplitudes/phases plus asymmetric lead broadenings/off-resonant onsite detuning.
D. Chirality reversal flips sign? score=2; not cleanly.
E. TRS control gives zero: True.
F. Tiny denominator artifact: no.
G. Stable under integration refinement/wider window: yes for accepted candidates.
H. This broad effective search finds transport polarization at the level reported above; use the |I_tot|>1e-6 line as the conservative robustness bar.
I. If requiring chirality sign flip as a hard filter, no validated candidate in this run qualifies.
J. Recommended wording: large spin-selective transport can occur in the effective Hermitian model, but not as a clean chirality-reversing CISS polarization in this scan; safest wording is that the model identifies a symmetry mechanism unless a chirality-odd regime is found.
