# Completed execution checks

These two 1 keV prescribed-proton runs used one water molecule extracted from
the accepted amorphous structure. They tested time steps of 0.04 and 0.02 atomic
units on the same coarse 0.4 angstrom grid. This is an execution/time-step
check, not the Hong benchmark or an ice-phase prediction.

Each directory retains exact GS/TD inputs, a result receipt, and compressed
solver logs. Receipts contain the original source hashes, settings, and
execution paths. Those historical paths identify the run; they are not
portable input locations. Source structures are available through the shared
ice registry. The inputs can be replayed sequentially as `inp` with the pinned
solver, using the same working directory to carry the GS restart into TD.

The analyzed last density was at the last output interval, slightly before the
last propagated step; its step and projectile position are recorded explicitly.
Sphere populations and norm loss are diagnostics, not event probabilities.
The two aborted five-water runs are not retained as completed evidence.
Raw wavefunctions, restart directories, caches, and temporary analysis products
are omitted. No new numerical collision was performed when organizing these
records for publication.
