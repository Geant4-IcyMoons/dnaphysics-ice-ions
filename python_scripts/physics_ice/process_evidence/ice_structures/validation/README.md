# Ice-structure validation

`HEXAGONAL_ICE.md` defines the 100 K ice-Ih acceptance protocol. Its executable
implementation remains `../../../nep_mbpol/validate_hexagonal_ice.py`, while
accepted structures and checksum-linked numerical records remain beside each
structure under `../../../ice_structures/<model>/validation/` or its
model-specific `artifacts/` directory.

The protocol uses thermodynamic stationarity, Bernal--Fowler ice rules,
CHILL+, RDFs and ice-Ih Bragg maxima. Passing one observable cannot substitute
for the complete decision gate.

`epsr_lda80k/PROTOCOL.md` separately defines the source-identity, conversion,
and archived-fit gate for the published 80 K low-density amorphous EPSR model.
It does not treat the rejected NEP-MB-pol melt--quench candidate as equivalent
evidence and does not fabricate independent replicas from one archived model.
