# Ion oxygen K shell

The `hydrogenic-gos` ion path implements the hydrogenic 1s expression of
Heredia-Avalos et al., *Physical Review A* **72**, 052902 (2005), Table I
and Appendix A, Eqs. (A.1)-(A.9), (A.13):
https://doi.org/10.1103/PhysRevA.72.052902.

```
B = 543.4 eV
Zeff = 7.7
H = Zeff^2 R = 806.6815452623143 eV
Qbar = q_au^2 / Zeff^2
Wbar = E_eV / H
kappa^2 = Wbar - 1
```

The physical edge is a gate at `E > B`, not a shift of the reduced energy.
For `B < E < H`, use the logarithmic continuation in Eq. (A.8). For
`E > H`, use Eqs. (A.5)-(A.6), with the continuous `atan2` Coulomb phase.
At `E = H` the common Coulomb factor is `exp(-4/(Qbar+1))`. The 1s-shell
occupancy of two is included once; conversion from reduced energy to eV
divides the GOS by `H`.

The ion kernel preserves the published amplitude: no optical-area multiplier
and no hydrogenic ELF rolloff. The optional `normalize_fsum=True` Python
argument remains an explicitly requested optical-area diagnostic, not the
production ion convention. The legacy electron optical K shell is unchanged.

For ion runs only, the separate excitation, outer-ionization, and K-continuum
parameterizations now share one molecular oscillator-strength budget. At each
q, the four outer-ionization amplitudes retain their fitted ratios and fill

```
F_outer(q) = 10 - F_excitation(q) - F_K,continuum(q).
```

This is the `joint-outer-k-continuum-v1` allocation. It replaces the previous
outer-shell formula, which allocated the full molecular sum before adding the
K continuum. It does not rescale either the final valence ELF or the corrected
K continuum. Calls from the electron generator and the `old-optical`
diagnostic retain the historical allocation.

The construction applies the molecular S1 constraint to the excitation,
outer-ionization, and K terms separated in Dingfelder, *Applied Radiation and
Isotopes* **83**, 142 (2014), Eqs. (13)-(26), while using the unscaled
Heredia-Avalos K strength in that budget.

The GOS-to-ELF conversion is
`ELF_K = (pi/2) Ep_fit^2 / 10 * (df_K/dE) / E`. The ion generator then applies
its existing phase-specific optical-sum normalization to obtain microscopic
cross sections per H2O at the physical material density. The finite-q change
does not alter that density normalization, Born prefactors, projectile
kinematics, or the separate Barkas OOS 8+2 normalization. Barkas does reuse
the corrected optical K-shell shape, so its K-shell-containing tables also
become stale.

## Checks and limits

Independent Appendix integrations give K-continuum strengths 1.7369142153,
1.8901138201, 1.9847230846, and 1.9999909711 at q = 0, 5, 10, and 30
in inverse Bohr radii. The optical fractional moment is therefore about
0.17369142, not a forced 0.179. These are checks of the published approximate
spectrum, not proof of an independent two-electron K-shell sum.

The occupancy remainder `2 - F_K,continuum(q)` is 0.2630858 electron at q=0,
0.1098862 at q=5, 0.0152772 at q=10, and 0.0000090 at q=30. It is audited as
an occupancy-closure diagnostic, not asserted to be a K-bound strength, and it
is not added to production.
Heredia-Avalos gives an inner-shell ionization GOS only; the negative-kappa
branch is its analytic continuation between the experimental edge and the
hydrogenic branch boundary, not a bound-state line list. A physical bound
channel would require independent condensed-water/ice excitation energies and
finite-q strengths. Adding a hydrogenic line remainder to this threshold-gated
continuum would be an unsupported double count.

`audit_finite_q_sum_rule.py` reports excitation, outer-ionization, K-continuum,
and the occupancy-closure diagnostic separately. It also compares energy-
weighted oscillator moments and proton DCS stopping moments at two numerical
resolutions; its 160/320 defaults bracket the production `dE=300` setting. The
exact amplitude budget is 10 electrons. The reconstructed ELF
S1 and S2 sums remain within 0.5% and 0.14%, respectively, of 10 over
q=0--30 because threshold gating, partitioning, and the approximate dielectric
inversion are numerical/model operations rather than exact analytic Drude
integrals.

`tests/test_hydrogenic_kshell.py` checks 80-decimal reference values for both
branches, their common boundary, the physical edge, units and broadcasting,
integrated strengths, the joint 10-electron amplitude budget, component ELF
sums, oscillator and DCS stopping-moment convergence, generator wiring, table
provenance, and DCS/TCS consistency. The existing Barkas and relativistic-
kernel tests remain relevant.

## Regeneration

NPZ caches and DAT comments identify `heredia-avalos-2005-unshifted-v1`,
`published-unscaled`, and `joint-outer-k-continuum-v1`. Caches missing those
identifiers are rejected when the hydrogenic K shell is selected. Legacy or
incompatible DAT files cannot be silently merged with new energy patches.
Regenerate the full desired energy range with `--no-merge-energy-patches`;
subsequent compatible patches can be merged. This code change does not
regenerate any production table or simulation.
