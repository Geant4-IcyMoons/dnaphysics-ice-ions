# K-shell and molecular sum-rule audit

```bash
python -m physics.inelastic_dielectric.k_shell.benchmarking.audit_finite_q_sum_rule
python -m pytest -q tests/test_hydrogenic_kshell.py tests/test_ion_optical_normalization.py
```

The audit reports excitation, outer ionization, unscaled K continuum,
occupancy remainder, and stopping-moment/grid-refinement diagnostics as JSON
on stdout. The occupancy remainder is not assigned a fictitious bound
spectrum. No new plotting campaign is needed for this check. See the
[model documentation](../README.md) for published branches and limitations.
