# CTMC charge-exchange validation

- `PROVENANCE.md` records published definitions, parameters, validity limits,
  and the distinction between microscopic H2O cross sections and ice-density
  scaling.
- `RUNBOOK.md` defines production, checkpoint, merge, adaptive-refinement and
  release gates.
- `PARALLEL_EXECUTION.md` records the carbon production and failure history.

Paper-shaped plots are benchmarking products. Release additionally requires
impact-domain convergence, the default 0.5% numerical interpolation gate,
zero unresolved trajectory failures, and a validated Geant4 handover.
