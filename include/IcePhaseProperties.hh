#ifndef IcePhaseProperties_h
#define IcePhaseProperties_h 1

#include "globals.hh"

namespace dna_ice {

// Nominal bulk mass densities used throughout the ice data pipeline.
// These values mirror python_scripts/physics_ice/constants.py.
inline constexpr G4double kHexagonalIceDensityGPerCm3 = 0.917;
inline constexpr G4double kAmorphousIceDensityGPerCm3 = 0.94;
inline constexpr G4double kWaterDensityGPerCm3 = 1.0;

}  // namespace dna_ice

#endif
