//
// ********************************************************************
// ICE-SPECIFIC DIFFUSION-CONTROLLED REACTION MODEL FOR EUROPA CHEMISTRY
// ********************************************************************
//
// PURPOSE: Implements position-dependent diffusion-controlled reaction kinetics
//          using Arrhenius diffusion coefficients for accurate ice chemistry
//          simulation in Europa's temperature-stratified subsurface environment
//
// PHYSICS CONTEXT:
// - Diffusion-controlled reactions are the dominant chemical process in ice
// - Reaction rates limited by how fast molecules can diffuse toward each other
// - Classic Smoluchowski equation: k = 4π(D₁ + D₂)R where R is reaction radius
// - In Europa's ice: D varies dramatically with temperature and depth
// - Temperature gradients: ~100K (surface) to ~273K (ocean interface)
// - Diffusion coefficient variations: 10³-10⁴ magnitude across temperature range
//
// KEY DIFFERENCES FROM VANILLA GEANT4:
// VANILLA: Uses stored, temperature-independent diffusion coefficients
//          - Results in uniform reaction rates regardless of local ice temperature
//          - Ignores physics of temperature-dependent molecular mobility
//          - Produces incorrect chemical concentration profiles in stratified ice
//
// ICE VERSION: Uses dynamic Arrhenius diffusion coefficients
//              - Calculates D = D₀ × exp(-Eₐ/kT) for each reactant pair
//              - OH: D₀=9×10⁻⁸ m²/s, Eₐ=0.14 eV (higher barrier, temperature sensitive)
//              - H:  D₀=9×10⁻⁸ m²/s, Eₐ=0.02 eV (lower barrier, less temperature sensitive)
//              - Reaction rates reflect actual molecular mobility in ice
//
// TECHNICAL IMPLEMENTATION:
// - Inherits from G4DiffusionControlledReactionModel to preserve all Geant4 functionality
// - Overrides GetTimeToEncounter() method - the core rate calculation
// - Integrates with GetIceTemperatureAtPosition() for spatial temperature lookup
// - Uses ArrheniusConstants_ice.hh for coefficient calculations
// - Maintains full compatibility with Geant4 reaction framework
//
// COMPUTATIONAL EFFICIENCY:
// - Single method override minimizes performance impact
// - Additional temperature lookup and exponential calculation per encounter time
// - Computational cost negligible compared to physics accuracy improvement
// - Same algorithmic complexity as vanilla implementation
//
// CRITICAL IMPORTANCE:
// - Diffusion-controlled kinetics determine overall chemical evolution timescales
// - Incorrect diffusion coefficients lead to wrong equilibrium concentrations
// - Temperature dependence essential for realistic chemical gradients in ice
// - Key component for quantitative Europa ocean chemistry predictions
// G4DiffusionControlledReactionModel_ice.hh - Ice-specific diffusion controlled reactions
//
// Author: Hoang TRAN (original)
// Modified for ice chemistry with Arrhenius diffusion coefficients
//

#ifndef G4DiffusionControlledReactionModel_ice_hh
#define G4DiffusionControlledReactionModel_ice_hh 1

#include "G4DiffusionControlledReactionModel.hh"

class G4MolecularConfiguration;
class G4Track;

/**
 * G4DiffusionControlledReactionModel_ice: ICE-SPECIFIC REACTION KINETICS MODEL
 * 
 * ROLE IN GEANT4 CHEMISTRY:
 * - Implements Smoluchowski diffusion-controlled reaction theory
 * - Calculates encounter times between reactive molecules
 * - Controls reaction kinetics in systems where diffusion limits reaction rate
 * - Most chemical reactions in ice are diffusion-controlled (not activation-controlled)
 * 
 * SMOLUCHOWSKI THEORY:
 * - Reaction rate k = 4π(D₁ + D₂)R where R is reaction radius
 * - Time to encounter ∝ 1/(D₁ + D₂) → inversely proportional to diffusion sum
 * - Small changes in D cause large changes in reaction rates
 * - In Europa ice: D varies by 10³-10⁴ → reaction rates vary by same magnitude
 * 
 * ICE-SPECIFIC MODIFICATIONS:
 * - VANILLA: Uses stored coefficients → spatially uniform kinetics
 * - ICE: Uses Arrhenius coefficients → realistic spatial variation
 * 
 * PHYSICS IMPLEMENTATION:
 * - OH encounters: Highly temperature dependent (Eₐ=0.14 eV)
 * - H encounters: Moderately temperature dependent (Eₐ=0.02 eV)
 * - Mixed encounters: Average of temperature dependencies
 * - All encounters: Position-dependent via temperature lookup
 * 
 * COMPUTATIONAL EFFICIENCY:
 * - Single method override: GetTimeToEncounter()
 * - Minimal performance impact for major physics improvement
 * - Same algorithmic complexity as vanilla Geant4
 * 
 * CHEMISTRY IMPACT:
 * - Correct spatial distribution of reaction rates
 * - Realistic chemical concentration gradients
 * - Quantitative predictions for Europa ocean chemistry
 */
class G4DiffusionControlledReactionModel_ice : public G4DiffusionControlledReactionModel
{
 public:
  // CONSTRUCTORS: Standard inheritance pattern
  G4DiffusionControlledReactionModel_ice();
  ~G4DiffusionControlledReactionModel_ice() override = default;

  G4DiffusionControlledReactionModel_ice(
    const G4DiffusionControlledReactionModel_ice&) = delete;
  G4DiffusionControlledReactionModel_ice& operator =(
    const G4DiffusionControlledReactionModel_ice&) = delete;

  // CORE METHOD OVERRIDE: Where ice physics is implemented
  
  // Calculates time until molecules encounter and potentially react
  // VANILLA DIFFERENCE: Uses stored diffusion coefficients in Smoluchowski equation
  // ICE VERSION: Uses Arrhenius coefficients → position-dependent encounter times
  // PHYSICS: encounter_time ∝ 1/(D₁ + D₂) where D₁,D₂ are ice-specific
  G4double GetTimeToEncounter(const G4Track& trackA, const G4Track& trackB);

 private:
  // ICE-SPECIFIC HELPER: Core innovation for position-dependent kinetics
  
  // Get Arrhenius diffusion coefficient from track position and molecule type
  // VANILLA EQUIVALENT: None - vanilla uses only stored values
  // USAGE: Called for both reactants to get temperature-dependent coefficients
  G4double GetArrheniusDiffusionCoefficient(const G4Track& track, const G4MolecularConfiguration* molConf);
};

#endif
