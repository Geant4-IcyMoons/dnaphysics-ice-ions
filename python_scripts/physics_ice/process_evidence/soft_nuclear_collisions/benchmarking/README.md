# Soft nuclear-collision benchmarks

The CP2K scaling calculation measures computational performance only. Physical
benchmarking must compare charge-resolved interaction surfaces or derived
stopping/transport moments with independent electronic-structure calculations
and experimental data where available. No such comparison is accepted yet.

Carbon fixed-charge soft-potential development was closed after failed
physical validation. The concise retained record is
`../../../nep_mbpol/soft_dft/benchmarking/CARBON_SOFT_COLLISION_CLOSURE_2026-08-10.md`;
bulky diagnostic run products are intentionally excluded from the production
evidence tree.

Method provenance is in `../../../nep_mbpol/soft_dft/README.md`, including CP2K
CDFT documentation and Holmberg and Laasonen, *J. Chem. Theory Comput.* **13**,
587--601 (2017), <https://doi.org/10.1021/acs.jctc.6b01085>.

The universal-ZBL runtime is a separate diagnostic baseline. Its software
reference is `../../../nep_mbpol/zbl_soft/kernel.py`; current tests establish
Python/C++ agreement and finite H/He/C/O/S--H/O kernels only. Physical
benchmark products must still compare minimum-transfer convergence, nuclear
stopping and angular-transport moments with independent calculations and the
available HTran proton/alpha model. No ZBL physical acceptance result is
recorded yet.
