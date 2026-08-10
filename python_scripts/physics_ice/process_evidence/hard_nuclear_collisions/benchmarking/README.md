# Hard nuclear-collision benchmarks

`benchmark_nlh_adaptive_kernels.py` compares adaptive NLH/BCA kernels with
independent dense impact-parameter and higher-order quadrature calculations.
Committed reference summaries are under `reference_results/`.

The scattering algorithm follows Mendenhall and Weller, *Nucl. Instrum.
Methods B* **227**, 420--430 (2005),
<https://doi.org/10.1016/j.nimb.2004.08.014>. Pair-potential provenance and
validity limits are recorded in `../../../nep_mbpol/PROVENANCE.json`.

Run from the repository root:

```bash
python_scripts/physics_ice/nep_mbpol/.venv/bin/python \
  python_scripts/physics_ice/process_evidence/hard_nuclear_collisions/benchmarking/benchmark_nlh_adaptive_kernels.py
```
