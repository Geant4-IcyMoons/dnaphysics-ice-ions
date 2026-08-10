# Structure-workflow cleanup record

On 2026-08-08 the accepted structure products were consolidated here and the
inactive raw preparation products were removed. Before deletion, PBS reported
no active job whose name matched `nep`, `ice`, or `epsr`.

Removed unversioned products:

- three completed rejected NEP-MB-pol amorphous melt--quench run directories;
- the nine-case rejected amorphous fixed-density stress-scan run directory;
- three completed raw 100 K ice-Ih equilibration directories, after their
  final snapshots, checksums, thermodynamic summaries, validation data, and
  acceptance decision had been retained here;
- the completed EPSR validation scheduler log, whose job identifier and
  executable provenance remain in the validation JSON;
- Python bytecode caches; and
- a seed-1000 GenIce2 `initial` XYZ/JSON duplicate that was byte-identical to
  the retained `melt_start` pair (XYZ SHA-256
  `79ed02df5244e384af5047d88cd456052fd2991d2ca182c2e95815c9afb4f1b7`).

The cleanup reduced `nep_mbpol/runs/` from approximately 594 MB to an empty
placeholder and left the complete structure library at approximately 11 MB.
The deleted raw files are not locally recoverable, but they are reproducible
from the retained code, inputs, seeds, model checksum, manifests, and PBS
protocols. The compact rejected-campaign reports remain under
`rejected_candidates/` so the negative scientific result is not lost.
