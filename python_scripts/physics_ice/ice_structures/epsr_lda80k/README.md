# 80 K low-density amorphous ice from EPSR

This package imports and audits the published 80 K low-density amorphous-ice
(LDA) EPSR model. It replaces the rejected NEP-MB-pol melt--quench candidate
as the evidence-constrained amorphous target; it does not alter or rescale that
candidate.

## Primary provenance

The source is the STFC ISIS Disordered Materials Data archive
[`AmorIce.zip`](https://edata.stfc.ac.uk/items/cdb607e7-7f9d-41ca-a037-6b06dd36a792),
DOI [10.5286/edata/729](https://doi.org/10.5286/edata/729), licensed CC BY 4.0.
The retained `LDAneutron` calculation contains the actual EPSR coordinate
model, three isotope-resolved neutron datasets, the refined partial RDFs, the
calculated diffraction functions, and the final EPSR output. Every consumed
file and the archive itself have pinned SHA-256 digests in
`source_manifest.json`.

The experimental and structural basis is documented by Finney et al.,
*Phys. Rev. Lett.* **88**, 225503 (2002),
[10.1103/PhysRevLett.88.225503](https://doi.org/10.1103/PhysRevLett.88.225503),
and Bowron et al., *J. Chem. Phys.* **125**, 194502 (2006),
[10.1063/1.2378921](https://doi.org/10.1063/1.2378921). The retained archive
was published with Soper, *J. Chem. Phys.* **150**, 234503 (2019),
[10.1063/1.5096460](https://doi.org/10.1063/1.5096460).

The archived model has 3,000 water molecules in a 45.796719 A cubic cell at
80 K. Its density is derived from the box and composition, not imposed by this
code. The expected value is approximately 0.93435 g cm^-3.
The molecular-origin plus relative-site coordinate conversion and harmonic
bond records follow the authoritative
[EPSR v26 User's Guide](https://www.isis.stfc.ac.uk/OtherFiles/Disordered%20Materials/EPSR26%20Manual%202019-11-27.pdf).

## What is implemented

- `fetch.py` downloads or ingests the exact archive, extracts only the eight
  required LDA files, verifies every digest, and writes a receipt.
- `model.py` strictly parses the EPSR v26 molecular `.ato` layout and performs
  a deterministic, coordinate-preserving conversion to periodic extended XYZ.
- `validate.py` checks source identity, composition, temperature, density,
  the archived harmonic water-restraint topology, lossless conversion, and completeness of the three
  archived diffraction fits. In parallel it evaluates the three
  intermolecular partial RDFs and CHILL+ local order. It reports comparisons
  with the archived 1,721-configuration averages without inventing a physical
  accuracy cutoff.
- `pbs/run_epsr_lda80k_validation.pbs` runs the independent observables on the
  idle queue, then attests and registers the structure only if all hard gates
  pass.

This is an exact audit of a published EPSR artifact, not a new refinement.
The archive retains one final coordinate realization. It would be incorrect
to create nominal replicas by rotating, translating, or copying it. New
independent EPSR replicas require the original data, potentials, documented
settings, and an available EPSR executable; that continuation is deliberately
separate from the present accepted single-configuration model.

## Reproduce

From the repository root:

```bash
PYTHON=python_scripts/physics_ice/nep_mbpol/.venv/bin/python
$PYTHON python_scripts/physics_ice/ice_structures/epsr_lda80k/fetch.py
$PYTHON python_scripts/physics_ice/ice_structures/epsr_lda80k/stage_plot_fonts.py
qsub pbs/run_epsr_lda80k_validation.pbs
```

An already downloaded archive can be supplied without network access:

```bash
$PYTHON python_scripts/physics_ice/ice_structures/epsr_lda80k/fetch.py \
  --archive /path/to/AmorIce.zip
```

The compact collision structure, sidecar attestation, registry, JSON decision,
CSV diagnostics, and PNG figure are written to `artifacts/`. Raw third-party
inputs stay under the ignored process-evidence `validation/runs/` directory.
The PBS workflow also reuses the common periodic-ice renderer to produce the
three-panel `epsr_lda80k_structure.png` molecular view; no second plotting
implementation is maintained.

The PBS request is four CPUs and 2 GB. There are four independent structural
tasks (OO, OH, HH, and CHILL+); a larger allocation cannot accelerate this
single archived configuration and would waste idle resources. Each output is
written atomically, and the workflow is idempotent, so scheduler preemption
does not corrupt or invalidate a rerun.

The final completed cluster run used approximately 116 MB at peak. The 2 GB request
therefore retains a wide Python/Scipy margin without reserving the previous
16 GB calibration allocation.

`stage_plot_fonts.py` copies the four locally installed URW Base 35 Nimbus
Mono PS faces into the ignored run directory. This is necessary because the
current compute-node image lacks them. The PBS job points Matplotlib to that
shared runtime copy and fails explicitly if any face is unavailable, rather
than silently producing a non-Courier figure. The font files are not vendored.
