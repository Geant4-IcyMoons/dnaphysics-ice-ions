# Cross sections

This catalogue holds completed, provenance checked products under `physics/`.
The dielectric tables are grouped by projectile charge, ice phase, PWBA or
RPWBA kernel, and polarization setting. Each original DAT file is stored as a
separate `.dat.gz`; `MANIFEST.csv` records its original SHA-256. This keeps the
large DCS tables below GitHub's regular Git file limit. The original headers
and numeric values are restored byte for byte:

```bash
python physics/cross_sections/restore.py /path/to/destination --family hydrogen
```

Check every compressed copy without extracting several gigabytes:

```bash
python physics/cross_sections/restore.py --verify-only
```

To regenerate the 86-entry ZIP of original DAT files in `Downloads`, with
source checksums and ZIP CRCs verified before it replaces an earlier copy:

```bash
python physics/cross_sections/package_hydrogen.py
```

The ZIP itself exceeds GitHub's regular-file limit. The constituent tables,
manifest, provenance, and this packaging script are tracked on `ion_modular`.

`H1` is H⁺ and `H0` is neutral hydrogen. Every combination of the two charge
states, two ice phases, PWBA/RPWBA, and polarization off/on is present: 16 cases,
64 DAT files. The H⁺ polarization-on products are completed 5% continuations:
accepted 0.1% and 1% points remain in the mixed-tolerance tables. The H⁰
polarization-on products are **diagnostic only** and must not be treated as
validated transport tables. Their `DIAGNOSTIC_ONLY.json` and flagged CSV are
stored alongside their DAT files. Historical `barkas` filenames select the full
nonlinear polarization correction, not a pure cubic Barkas term.

`elastic/carbon_nlh_pair/` contains the numerically qualified carbon–H and
carbon–O binary elastic DCS index and its 962 indexed maps. The maps are packed
into `pair_dcs.tar.gz` to avoid adding nearly a thousand individual files to
the work quota. Restore them with `--family elastic`. These are charge state
independent nuclear pair DCS; they are not phase dependent ice transport tables.
The [elastic protocol](../elastic/hard_collisions/README.md) states their
physical and numerical limits.

Run `python physics/cross_sections/build.py` on the production workspace to
regenerate the catalogue from its pinned source campaigns. The builder checks
completion records and table headers. It reads those outputs without modifying
them. Provenance submission records and the corresponding frozen code snapshots
are in `provenance/`; these preserve the generator used for each table family
without replacing the current working-tree code.
