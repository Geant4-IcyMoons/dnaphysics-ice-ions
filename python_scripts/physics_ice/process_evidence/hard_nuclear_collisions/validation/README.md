# Hard nuclear-collision validation

`CARBON_GATE.md` defines the atomistic carbon decision gate. Kernel convergence
is necessary but insufficient: phase, orientation, replica and thickness
dependence must be assessed before density-only scaling can be accepted.

H and He additionally require a non-overlapping handover with HTran, which is
a complete elastic model rather than a soft-only kernel.

If a restart reaches its declared Monte Carlo sampling ceiling before the
absolute fixed-width statistical gate, `--extended-maximum-trajectories` may raise that
liveness ceiling. The extension preserves the signed controller configuration,
deterministic trajectory indices and existing checkpoint batches; it neither
loosens the tolerance nor changes a physical parameter. Every applied ceiling
is recorded in `runtime_trajectory_ceiling_history`.

For production, `--unlimited-trajectories` removes the operational sampling
ceiling: sampling stops only when every statistical gate passes or the batch
scheduler ends the process. The implementation uses the platform integer bound
only to define the finite family of sequential confidence checks. Atomic batch
checkpoints make a subsequent PBS submission continue the same deterministic
trajectory stream.

Scalar convergence follows Glynn and Whitt's simultaneous absolute fixed-width
procedure. Per-case widths are frozen from the independent raw calibration
sample using the JCGM 101:2008 numerical tolerance for two meaningful
significant digits. Two digits are the predeclared project reporting
requirement, not a precision prescribed by JCGM or the fixed-width literature.
The literature supplies the stopping procedure and JCGM supplies the decimal
tolerance definition. This active requirement is frozen before any further v3
sampling; looser-digit sensitivity calculations are diagnostic only and cannot
be used for acceptance. Mean-relative stopping is not used.
