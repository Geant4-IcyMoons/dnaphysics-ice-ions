# Electronic excitation and ionisation benchmarks

`benchmark_barkas_sbethe.py` compares the Barkas differential-correction
implementation with the independently distributed SBETHE v2 calculation. It
checks the downloaded archive checksum before compiling or executing it and
keeps formula parity separate from differences in the target optical data.

Primary reference: Salvat and Quesada, SBETHE v2 dataset,
<https://data.mendeley.com/datasets/7zw25f428t/2>.

Run from the repository root:

```bash
python_scripts/physics_ice/nep_mbpol/.venv/bin/python \
  python_scripts/physics_ice/process_evidence/electronic_excitation_ionisation/benchmarking/benchmark_barkas_sbethe.py
```
