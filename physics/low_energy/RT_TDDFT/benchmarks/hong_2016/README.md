# First H+ capture benchmark

## Implemented and outstanding

The [independent calculation](../../docs/BENCHMARK.md) now runs the collision,
projectile-frame translation and subsequent propagation, and capture analysis.
Its reduced-resolution execution test has completed. It does **not** reproduce
the published numerical curves yet. `capture.py` implements the determinant
counting underlying Eqs. 14--16; `capture_curve.py` assembles the Figure 2 ordinate.

The unresolved historical assets in [protocol.json](protocol.json) concern an
exact reconstruction of the authors' setup. They do not block running our
independent setup, whose adopted geometry, potentials and numerical differences
are explicit. The earlier `hplus.py` density pilot remains a separate diagnostic.

The desired P1 is the probability of exactly one electron in the separated
projectile region within the Kohn--Sham determinant approximation. It is
inclusive over the residual target: it does not establish an intact H2O+ ion
or exclude simultaneous excitation/ionization. The mean electron count is a
different quantity. Probabilities above two electrons are retained as a
contamination diagnostic rather than silently reassigned to H0 or H-.

## Counting and complex orbital export

For each spin, form S_ij = integral_A psi_i* psi_j. The coefficients of
`det(I - S + z*S)` are the electron-number probabilities. We evaluate this
polynomial from the Hermitian overlap eigenvalues and convolve the two spin
blocks. This includes off-diagonal overlaps, is invariant under rotations of
the occupied orbitals, and avoids enumeration of electron permutations.
The initially occupied orbitals must be orthonormal. Do not renormalize
CAP-depleted orbitals or substitute density/squared-modulus output.

In the reference workflow, A is the entire projectile-centered simulation box
after the coordinate/momentum translation and subsequent continuum clearing.
The supplied snapshots must come from that stage. The analyzer does not perform
the translation, alter the dynamics, or pretend that a stationary spatial
population proves convergence to physical bound-state probabilities.

Octopus 16.4 exports complete complex amplitudes with `OutputFormat = binary`.
Its cube output contains only the real component and is unsuitable here.
For an already established reference input, request these additional outputs:

```text
UnitsOutput = atomic
%Output
 wfs | "output_format" | binary
 density | "output_format" | mesh_index
%
OutputWfsNumber = "1-4"
```

This is an output fragment, not a complete solver input. Preserve the original
spin-polarized state setup and record native file names for both spin blocks.
Export the native mesh once per unchanged mesh and record three Cartesian
spacings. Do not reorder the mesh or reuse it across a changed mesh. Preserve
restart/orbital identities through the translation. Native .obf files are read
in bounded chunks and mapped from disk; their complex phases are retained.

## Analysis manifest

Supply JSON with the following fields. All file paths are relative to the
manifest; all spatial and temporal quantities below use atomic units.

- `schema_version`: 1; `units`: `atomic`; `projectile`: `H+`;
  `active_electrons`: 8.
- `collision`: `energy_ev_total` (1000), `orientation` (a--f), and
  `impact_parameter_bohr` (nonnegative).
- `mesh`: native `.mesh_index` file; `spacing_bohr`: three Cartesian spacings.
- `region`: `projectile_centered_spherical_box`; `region_radius_bohr`: box radius.
- `minimum_separation_bohr`: minimum accepted distance from any target nucleus.
- `initial_orthogonality_tolerance`: explicit numerical tolerance for the
  initial full-mesh occupied-orbital Gram matrices.
- `stationarity_tolerance`: explicit maximum absolute change in any P(n)
  over the supplied late frames. This is a chosen numerical criterion, not a
  literature value or a substitute for time/box/absorber convergence.
- `initial.orbitals`: `up` and `down`, each listing four native `.obf` files.
- `frames`: at least two entries with increasing `time_au`, measured
  `projectile_position_bohr` (origin after translation), three measured
  `target_positions_bohr`, and `orbitals` in the same format and occupied order.

The implementation rejects nonorthonormal initial states, invalid overlap
spectra, inconsistent native mesh lengths, incomplete files, and target nuclei
inside the analysis box. The whole sphere includes the absorber shell, as in
the reference region; continuum clearing must therefore be assessed in the
post-translation evolution. Checks based on nuclear distances alone cannot
exclude extended target orbitals; separation convergence remains necessary.

```bash
python -m physics.low_energy.RT_TDDFT.capture analysis.json --output capture.json
python -m physics.low_energy.RT_TDDFT.capture_curve case_*/capture.json \
  --output capture_curve.csv
```

The curve collector refuses points that failed the supplied stationarity test.
It exports P1 and `2*pi*b*P1`, with b in bohr. It does not integrate incomplete
impact-parameter tails or average orientations automatically. With
`--reference reference.csv`, it compares supplied values on an exactly matching
grid. Reference CSV columns are `orientation`, `impact_parameter_bohr`,
`two_pi_b_p1_bohr`, and `provenance`. No reference values are bundled or invented.

The code tests use analytical and synthetic orbital data, explicitly not
collision results. They check determinant expansion, complex phase handling,
spin counting, CAP norm loss, endianness, rejection conditions, and the full
export-to-curve analysis path. The independent runner also has a real Octopus integration test. HPC execution
and numerical reproduction remain outstanding.
