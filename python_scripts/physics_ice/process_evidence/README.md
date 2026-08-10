# Process benchmarking and validation evidence

This directory is the canonical evidence layer for every physical process in
`physics_ice`. Production solvers and table generators remain in their model
packages; their independent comparisons and scientific acceptance decisions
live here and must never be inferred from a successful program exit.

Each process has two explicit areas:

- `benchmarking/`: comparisons with external measurements, published curves,
  independently evaluated formulae, or numerically denser reference solvers;
- `validation/`: convergence studies, applicability limits, uncertainty,
  decision gates, and checksum-linked acceptance records.

Software regression tests remain in `tests/`. They establish implementation
behavior, not physical validity. Generated workflows and large intermediate
products belong under a process's ignored `validation/runs/` directory.

| Process | Production implementation | Evidence |
| --- | --- | --- |
| Electronic excitation/ionisation | `../barkas_dcs.py` and PWBA generators | `electronic_excitation_ionisation/` |
| CTMC charge exchange | `../charge_exchange_ctmc.py` and `../generate_*_charge_exchange_ctmc.py` | `charge_exchange_ctmc/` |
| Ice structures | `../nep_mbpol/` structure preparation | `ice_structures/` |
| Hard nuclear collisions | `../nep_mbpol/nlh/` and `../nep_mbpol/bca/` | `hard_nuclear_collisions/` |
| Soft nuclear collisions | failed charge-resolved work in `../nep_mbpol/soft_dft/`; diagnostic full-ZBL baseline in `../nep_mbpol/zbl_soft/` | `soft_nuclear_collisions/` |
| Low-energy charge exchange | `../nep_mbpol/low_energy_charge_exchange/` | `low_energy_charge_exchange/` |

An accepted artifact must identify its input checksum, software commit,
numerical settings, reference source, measured discrepancies, uncertainty or
validity limits, and pass/fail decision. A numerical tolerance, including the
0.5% interpolation targets used by this repository, is not a statement of
physical accuracy.
