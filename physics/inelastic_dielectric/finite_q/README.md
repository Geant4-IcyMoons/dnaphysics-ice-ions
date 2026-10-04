# Ice dielectric response

`emfietzoglou_model_finite_q.py` supplies the production optical parameters,
complex finite-q valence response, and Kyriakou partitioning for amorphous
and hexagonal ice. `emfietzoglou_model_optical_limit.py` retains the optical
diagnostics. Experimental comparison input is in `data/ice data.xlsx`.
Their energies are in eV and momentum transfer is in inverse Bohr radii.

The hydrogenic core implementation lives in `../k_shell/hydrogenic.py` and
is explicitly imported by the finite-q module. Its functions remain visible
through that model API, avoiding divergent core definitions. The historical
electron optical K-shell diagnostic is distinct and remains q-independent.

The phase densities are defined once in `physics/constants.py`: amorphous
0.9343471678603292 g/cm3; hexagonal 0.9335 g/cm3. Fitted plasma energies are
spectral parameters, not additional material densities.

```bash
python -m pytest -q tests/test_ion_optical_normalization.py tests/test_hydrogenic_kshell.py
python -m physics.inelastic_dielectric.k_shell.benchmarking.audit_finite_q_sum_rule
```

`optical_input.py` holds the common Born/oscillator optical normalization and
the partitioned q=0 spectrum used by polarization. The unchanged Born
normalization and valence rolloff have one implementation, not duplicate fits.
The joint molecular allocation and unscaled K continuum are documented in
[validation](../validation/README.md).
Optical diagnostic figures use this component's ignored `benchmarking/plots/` directory.
