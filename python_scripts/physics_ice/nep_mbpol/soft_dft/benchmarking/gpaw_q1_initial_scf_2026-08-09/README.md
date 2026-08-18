# Rejected GPAW q=1 initial-SCF trajectory

PBS job 113096 tested C+ plus H2O at 12 A in the oxygen-back orientation with
GPAW 25.7.0 on eight MPI ranks. The input copied the published GPAW He2 cDFT
example's Davidson(3) eigensolver and Mixer(beta=0.25, nmaxold=3,
weight=100), while retaining GPAW's normal tight SCF acceptance criteria.

This trajectory is rejected as `noncontracting_unconstrained_scf_limit_cycle`.
It never entered cDFT or optimized a multiplier. Across 88 initial-SCF
iterations, the reported energy ranged from below -123 eV to above +42 eV;
the final eigenstate and density log10 changes were +1.47 and -0.34. The fixed
total magnetic moment remained approximately 1 electron. The job was stopped
after 90 s (Exit_status=15); peak PBS memory was 2,873,700 kB. No wavefunction
or electronic state is accepted or reusable.

The first controlled successor, job 113100, used GPAW's documented defaults:
Davidson with two inner iterations, default Pulay mixer and automatic band
count. It repeated the noncontracting behavior and was stopped after 37 SCF
iterations (38 s, peak PBS memory 2,865,076 kB); it also never reached cDFT.

Job 113103 changed only Davidson's inner iteration count from two to five. It
also failed to contract before cDFT began and was stopped; orbital work alone
therefore did not repair the density iteration.

Jobs 113105 and 113106 then used the convergence guide's difficult-SCF mixer
values (`beta=0.04`, `method=difference`, `nmaxold=8`, `weight=100`) with two
and five Davidson updates, respectively. Both ran independently from fresh
atomic guesses on eight MPI ranks. Both reached GPAW's 333-iteration limit and
ended with `KohnShamConvergenceError` (Exit_status 42), before cDFT or any
multiplier optimization began. The final density log10 changes were -1.07 and
-0.88; neither is close to convergence. Peak PBS memory was approximately
2.91 GB per job. No state or wavefunction from any trajectory in this record
is accepted or reusable.

Thus GPAW independently reproduces the initial electronic-state instability;
switching from CP2K's outer-CDFT machinery to GPAW's L-BFGS-B implementation
does not by itself solve the carbon entrance-state problem. These are
numerical controls, not evidence that the q=1 diabatic state is physically
valid.

Sources:

- GPAW convergence guide: <https://gpaw.readthedocs.io/documentation/convergence.html>
- GPAW cDFT documentation and He2 example:
  <https://gpaw.readthedocs.io/documentation/cdft/cdft.html>
