# Liamsuwan and Nikjoo (2013) digitization

`liamsuwan_2013_figures_12_14_digitized.csv` contains manually traced values
from figures 12--14 of T. Liamsuwan and H. Nikjoo, *Physics in Medicine and
Biology* **58**, 641--672 (2013), DOI
`10.1088/0031-9155/58/3/641`.

The source is the publisher PDF named
`Liamsuwan_2013_Phys._Med._Biol._58_641.pdf`, currently available at
`~/work/dnaphysics-ice-ions/literature/`. Its resolved path and SHA-256 are
recorded by the validation report. The source panels are embedded raster images:

- figure 12: PDF page 25, 600 x 563 pixels; plot rectangle `(84, 1)` to
  `(582, 499)`, logarithmic axes 1--1000 keV/u and 1e-24--1e-12 cm2;
- figure 13: PDF page 26, 602 x 579 pixels; plot rectangle `(85, 19)` to
  `(584, 517)`, logarithmic axes 1--10000 keV/u and 1e-19--1e-15 cm2;
- figure 14: PDF page 27, 600 x 592 pixels; plot rectangle `(68, 11)` to
  `(583, 529)`, logarithmic energy 1--10000 keV/u and linear fraction 0--1.

The CSV retains every traced pixel coordinate. `value` is obtained only from
the stated axis transformation; it is not fitted to the present calculation.
The uncertainty column is fractional for figures 12--13 and absolute for
figure 14. It covers finite raster resolution and manual curve-centre
selection; it is not uncertainty in the published CTMC model. Figure-12 values
include the separation factors printed in the source panel, namely
`10^(q-5) sigma_SC`.
