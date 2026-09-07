# Optical and finite-q checks

The existing optical and finite-q diagnostic routines are available through:

```bash
python -m physics.inelastic_dielectric.finite_q.emfietzoglou_model_optical_limit
python -m physics.inelastic_dielectric.finite_q.emfietzoglou_model_finite_q
```

They write PDF figures beneath `plots/` here and use `../data/ice data.xlsx`.
The optional tabulated model overlays are archived in the same data folder;
they are model curves, not independent measurements. These plotting entry
points retain the historical electron optical K-shell diagnostic. The ion
K-shell and molecular sum-rule audit is separate:

```bash
python -m physics.inelastic_dielectric.k_shell.benchmarking.audit_finite_q_sum_rule
```
