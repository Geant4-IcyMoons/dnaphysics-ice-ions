# Low-energy charge-exchange validation

The initial gate is C+ + H2O -> C + H2O+ for the H2O `1b1` donor channel.
Restart-safe CDFT and MIXED_CDFT runs live under ignored `validation/runs/`.
Acceptance requires converged diabatic curves, a stable crossing and coupling,
charge localisation, curved nuclear trajectories, impact-parameter and
orientation integration, and CTMC overlap validation.

The implementation and CP2K method reference are documented in
`../../../nep_mbpol/low_energy_charge_exchange/README.md`.
