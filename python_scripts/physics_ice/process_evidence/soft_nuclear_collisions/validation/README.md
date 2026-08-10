# Soft nuclear-collision validation

Required gates are basis, grid, cell, SCF/CDFT, functional, electronic-state,
charge-localisation, asymptotic and counterpoise convergence; NLH overlap;
representative amorphous and ice-Ih environments; and independent stopping and
angular-moment validation. Adaptive radial interpolation at 0.5% is a numerical
table criterion, not physical accuracy.

Generated restartable calculations belong under ignored `validation/runs/`.

The universal-ZBL diagnostic baseline has separate release gates:

1. Python/C++ agreement for screening, impact-parameter cross sections and
   exact recoil kinematics for every H/He/C/O/S--H/O pair;
2. minimum-transfer and recoil-threshold convergence;
3. HTran comparisons for proton and alpha without overlapping the models;
4. independent nuclear-stopping and angular-transport comparisons for C, O
   and S; and
5. an explicit non-overlap construction before any ZBL-soft/NLH-hard bundle.

Passing software tests does not satisfy these physical gates. Until they pass,
`DNA_ZBL_ALLOW_VALIDATION_PENDING=1` is required to run the Geant4 process.
