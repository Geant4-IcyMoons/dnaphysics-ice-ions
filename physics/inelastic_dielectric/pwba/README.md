# PWBA

`kernels.py::PWBAKernel` evaluates the nonrelativistic Born integrands.
Its immutable state contains projectile mass (M/me), nuclear charge, K-shell
selection, molecular ELF conversion, and the screening-amplitude callback.
It does not read generator globals. Frozen-projectile screening remains
inside the momentum integral, including for neutral states.

For a bare ion, the prefactor is
`Z^2 * (ELF_scale/N_H2O) / [pi*a0*(T/(M/me))]` multiplying the logarithmic-q
ELF integral. Loss and total incident energies are in eV; q is in inverse
Bohr radii; the result is m2/eV. The existing high-mass transfer cutoff is
preserved. Valence rolloff is unchanged; hydrogenic K-shell rolloff is absent.

Select this baseline with `--relativistic-projectile-dcs=false` (default).
The generator adds polarization only after constructing the Born DCS.

Implementation checks:

```bash
python -m pytest -q tests/test_projectile_form_factors.py tests/test_ion_dcs_export_grid.py
```

Component-by-component target and stopping-moment convergence checks live
in [the K-shell benchmark](../k_shell/benchmarking/README.md). The projectile
benchmark checks both PWBA and RPWBA across all fixed states. These checks
are not stopping-power fits or experimental validation.
