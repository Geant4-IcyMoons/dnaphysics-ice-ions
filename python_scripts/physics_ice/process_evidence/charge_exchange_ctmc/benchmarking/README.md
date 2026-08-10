# CTMC charge-exchange benchmarks

`benchmark_carbon_charge_exchange_ctmc.py` constructs the observables defined
in figures 12--14 and equation 23 of Liamsuwan and Nikjoo, *Phys. Med. Biol.*
**58**, 641--672 (2013), <https://doi.org/10.1088/0031-9155/58/3/641>.

It compares like with like: pure single capture (`SC`), pure single loss
(`SL`), and equilibrium fractions from the adjacent total decrease/increase
rates. It never digitizes or fits unavailable paper data silently. Boundary,
energy-drift, probability-conservation, channel-identity and archive-integrity
checks are emitted with the presentation plot.

The presentation plot converts the calculated energy per nucleon to total
C-12 kinetic energy and displays the common Geant4 model interval from 10 keV
to 100 MeV. Paper/data residuals are calculated only where the computed curve
has support; sparse diagnostic runs never extend their endpoint values into
unsampled energies. A production CTMC grid spans 1--10,000 keV/u
(12 keV--120 MeV total). The figure-12 separation factors are written
compactly as $10^{q-5}\sigma_{\rm SC}$ on the capture ordinate.
