# Nuclear elastic collisions

| Component | Responsibility |
| --- | --- |
| `nlh/` | NLH repulsive potential and published coefficients |
| `zbl/` | Universal-ZBL kernel, full/soft backends and campaign runner |
| `handoff/` | Two-phase boundary-study configuration, preparation and comparison |
| `bca/` | Shared binary scattering, ice transport, statistics and table support |
| `simulate.py` | Explicit-ice trajectory entry point |
| `hard_collisions/` | NLH production controller, configuration and tests |
| `jobs/` | ZBL PBS launcher |

Shared structures live in [models/ice](../../models/ice/README.md), outside
`physics/`. The dielectric, CTMC and elastic processes remain separate.

From the repository root:

```bash
python -m physics.elastic.simulate --help
python -m physics.elastic.zbl.run_campaign prepare --help
```

The relocation preserves the collision formulas and structure payloads.
Legacy local paths remain symlinks for existing callers and provenance records.
The hard controller still snapshots its explicitly pinned historical runtime;
relocation does not change that runtime or qualify a new production campaign.
Changed source fingerprints must not be used to resume or re-sign old ZBL runs.

[The handoff study](handoff/PROTOCOL.md) is executable, but not physically qualified.
The combined diagnostic runner is `physics.elastic.handoff.run`; the older
`simulate.py` entry point still selects the branches separately.

Migration checks: 163 tests passed and one opt-in physical test was skipped.
CLI checks passed. These are software checks, not physical validation.
