# GPAW constrained-first carbon q=1 diagnostic

PBS array `113111[0-3]` ran four independent C+--H2O calculations at 12 A
in the oxygen-back geometry with GPAW 25.7.0. Each task used eight MPI ranks,
approximately 3.76 GB peak PBS memory, and 36--41 s wall time. All tasks exited
normally (`Exit_status=0`).

Each task independently converged isolated C+ and neutral H2O with direct
orbital optimization, embedded their atom-centred dzp orbital blocks into the
full-system basis, and constructed the requested carbon 2p density matrix.
GPAW's cDFT potential was then installed before the prepared coefficients were
restored. Every recorded complex iteration verifies that the external cDFT
potential was active; no unconstrained complex energy iteration occurred.

| Component | Inner iterations | C charge | C spin (electron) | Final selected 2p population | DFT energy (eV) |
|---|---:|---:|---:|---:|---:|
| parallel, pz | 19 | 1.00005945 | 0.99998683 | 0.99999885 | -2.88911510 |
| perpendicular in-plane, px | 23 | 1.00006751 | 0.99997674 | 0.99972264 | -2.88956544 |
| perpendicular normal, py | 23 | 1.00006475 | 0.99998016 | 0.99972337 | -2.88759020 |
| fractional px/py/pz | 21 | 1.00006394 | 0.99998120 | 0.333289/0.333289/0.333334 | -2.64450054 |

The charge and spin residuals are below 0.00007 electron, substantially inside
the 0.01-electron numerical gate. The selected pure-orbital identities remain
stable throughout each trace. The initial source-defined cDFT coefficients
of 0.1 hartree already satisfy the population gate, so the multiplier optimizer
does not need to move them.

This establishes a reproducible numerical construction for all three pure
components of the C+ 2P manifold and for the fractional diagnostic. It does
not establish a unique q=1 potential or physical averaging rule. Basis, grid,
cell, population-definition, separated-fragment, radial-continuation,
counterpoise, force, and independent-method validation remain pending. No
geometry array or production soft-collision table was released.

## Immutable-checkpoint audit

The three final pure-component checkpoints were subsequently analyzed without
any electronic iteration.  Squared overlaps of the final singly occupied
orbital (rows: parallel, perpendicular in-plane, perpendicular normal) with a
single fixed carbon `(p_x, p_y, p_z)` reference frame are

```text
[[0.000000, 0.000000, 0.999995],
 [0.999723, 0.000000, 0.000000],
 [0.000000, 0.999723, 0.000000]]
```

The determinant spin expectations are 0.752503, 0.752510, and 0.752511,
respectively, compared with 0.75 for an uncontaminated doublet.  Pairwise
integrated absolute differences between the final total pseudo-densities are
1.160--1.197 electrons.  Thus the orbital labels are preserved, spin
contamination is small, and the three checkpoints contain distinct densities.
The full matrix, pairwise density metrics, SOMO densities, equal-weight carbon
2p reference densities, checkpoint hashes, and raw density arrays are in
`checkpoint_analysis.json` and `checkpoint_analysis.npz` in this directory.
These diagnostics establish numerical state identity, not physical validity.

## Independent multiplier-start repetitions

Jobs 113139 and 113140 repeated the parallel (`p_z`) component from entirely
fresh fragment and complex orbital preparations, using initial charge and spin
multipliers of 0.05 and 0.15 Ha, respectively.  Both terminated normally and
retained the same SOMO: squared mutual SOMO overlap is 0.999996, and each has
squared overlap 0.999999 with the original 0.10-Ha SOMO.  Nevertheless, the
two repeat densities differ by 0.02274 electron in integrated absolute density,
and their reported energies differ by 0.01624 eV (the repeat--original density
differences are 0.01131 and 0.01144 electron).  They therefore **fail** the
requested multiplier-start reproducibility test.  GPAW accepted both initial
multipliers without optimizing them because the existing 0.01-electron
constraint threshold was already met.  These calculations demonstrate stable
orbital identity, but not a multiplier-independent constrained stationary
state; neither repeat is promoted to physical or production evidence.

Sources:

- GPAW cDFT documentation: <https://gpaw.readthedocs.io/documentation/cdft/cdft.html>
- GPAW MOM documentation: <https://gpaw.readthedocs.io/documentation/mom/mom.html>
- GPAW direct-optimization documentation:
  <https://gpaw.readthedocs.io/documentation/do/do.html>
- A. V. Ivanov, G. Levi, and H. Jonsson, *J. Chem. Theory Comput.* **17**,
  5034--5049 (2021), <https://doi.org/10.1021/acs.jctc.1c00157>.
