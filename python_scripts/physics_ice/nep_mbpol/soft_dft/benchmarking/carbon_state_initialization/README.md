# Carbon CDFT state-initialization gate

This gate records rejected numerical paths before any soft-collision potential
is accepted.  It is distinct from the official CP2K regression examples.

## Zero-strength C4+ result (2026-08-09)

Job `112047.pbs02` used CP2K 2025.2, density-Hirshfeld CDFT, an all-electron
GAPW carbon `1s2` atomic occupation, `SCF_GUESS ATOMIC`, and initial
`STRENGTH 0`.  The inner electronic SCF converged, but the first CDFT
evaluation gave

- target projectile population: 2 electrons;
- calculated projectile population: 3.999998290621 electrons;
- residual: +2.000 electrons;
- multiplier: 0 hartree.

The job was stopped after this result.  It is direct evidence that the
zero-strength calculation converges to the charge-transferred branch; it is
not a constrained C4+ result and must not seed production.  The concise file
provenance is stored in `zero_strength_q4_rejection.json`.

The replacement workflow follows the CP2K 2025.2 source behavior:

1. evaluate source-defined fixed-multiplier diagnostics with active CDFT,
   explicit `STRENGTH`, and `CDFT/OUTER_SCF MAX_SCF 0`;
2. establish two inner-converged, same-branch residuals of opposite sign;
3. start CP2K BISECT from each endpoint in separate uninterrupted runs, using
   `STEP_SIZE=(lambda_start-lambda_opposite)/r(lambda_start)` so the
   source-defined update `lambda_next=lambda_start-STEP_SIZE*r(lambda_start)`
   visits the opposite side and creates the bracket inside that CP2K process;
4. accept a paired wavefunction and multiplier only if both directions agree.

The BISECT implementation is in CP2K
[`qs_outer_scf.F`](https://github.com/cp2k/cp2k/blob/v2025.2/src/qs_outer_scf.F).
The `&BS` occupation is documented to apply only with
[`SCF_GUESS ATOMIC`](https://manual.cp2k.org/cp2k-2025_2-branch/CP2K_INPUT/FORCE_EVAL/SUBSYS/KIND/BS.html).

## Rejected q=1 fixed-multiplier launch (2026-08-09)

Job `112837.pbs02` exposed two code defects before producing a molecular SCF
result. First, `CDFT/OUTER_SCF MAX_SCF 0` was supplied without an `OPTIMIZER`.
CP2K defaulted to `NONE`, and `qs_scf_output.F` aborted while printing the
CDFT setup because that routine has no `NONE` case. The fixed-multiplier
renderer now names `BISECT`; `MAX_SCF 0` still exits before an optimizer update.

Second, the atomic-kind printout contained 5.50 carbon electrons rather than
the five required for C+. CP2K v2025.2 applies `NEL/2` to each spin-resolved
shell occupation. The earlier q=1 `ALPHA NEL -1` definition therefore encoded
a half-electron shift. The registry now uses exact even shifts derived from
`qs_kind_types.F`; q=1 is `ALPHA NEL 0`, `BETA NEL -2` for 2p. The same
correction was applied and tested across q=0--6.

The compact checksum and accounting record is
[`q1_optimizer_and_bs_rejection.json`](q1_optimizer_and_bs_rejection.json).

The subsequent `ALPHA=0.5` q1 fixed-multiplier probe entered the molecular
inner SCF but oscillated through 31 iterations. Its complete input/output and
best/final residuals are retained in the run-directory `rejection.json`; its
WFN is deliberately absent and cannot seed another calculation. The two fresh
Pulay controls and their agreement thresholds were preregistered before
submission in
[`q1_mixing_control_contract.json`](q1_mixing_control_contract.json). They
differ only in `complex_mixing_alpha` (`0.2` versus `0.1`) and use independent
`SCF_GUESS ATOMIC` starts. A q2 calculation, if required, is a solver control
only and cannot validate q2 or replace q1.
The failed calculation contains no accepted energy, force, population, WFN,
or branch evidence.
