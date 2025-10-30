//
// ********************************************************************
// ICE-SPECIFIC DIFFUSION-CONTROLLED REACTION MODEL IMPLEMENTATION
// ********************************************************************
//
// PURPOSE: Provides the computational core for diffusion-controlled reaction
//          kinetics in ice using position-dependent Arrhenius coefficients
//          for quantitative Europa ocean chemistry simulation
//
// SMOLUCHOWSKI THEORY IMPLEMENTATION:
// - Classic diffusion-controlled kinetics: k = 4π(D₁ + D₂)R
// - Encounter time calculation: t ∝ 1/(D₁ + D₂)
// - Reaction probability depends on molecular approach dynamics
// - Ice modification: D₁,D₂ are temperature-dependent via Arrhenius equation
//
// COMPUTATIONAL STRATEGY:
// - Override only GetTimeToEncounter() method for minimal code impact
// - Preserve all vanilla Geant4 numerical algorithms and stability
// - Insert Arrhenius coefficient calculation at critical computation point
// - Maintain identical algorithmic flow to avoid numerical artifacts
//
// PHYSICS PARAMETERS:
// - OH molecules: D₀=9×10⁻⁸ m²/s, Eₐ=0.14 eV (high temperature sensitivity)
// - H molecules:  D₀=9×10⁻⁸ m²/s, Eₐ=0.02 eV (moderate temperature sensitivity)
// - Temperature range: 100K-273K (Europa ice shell thermal profile)
// - Coefficient variations: 10³-10⁴ magnitude across temperature range
//
// NUMERICAL CONSIDERATIONS:
// - Encounter time calculations are numerically sensitive
// - Coefficient accuracy directly impacts reaction rate accuracy
// - Ice-specific coefficients essential for quantitative chemistry
// - Computational overhead minimal compared to physics accuracy gain
// G4DiffusionControlledReactionModel_ice.cc - Ice-specific diffusion controlled reactions
//
// Author: Hoang TRAN (original)
// Modified for ice chemistry with Arrhenius diffusion coefficients
//

#include "G4DiffusionControlledReactionModel_ice.hh"
#include "G4DNAMolecularReactionTable.hh"
#include "G4Molecule.hh"
#include "G4MolecularConfiguration.hh"
// ---------------------------------------------------------------------
//  G4DiffusionControlledReactionModel_ice.cc
//  Customized by G. Yoffe for ice (solid H2O) chemistry
//
//  Provides diffusion-controlled reaction kinetics in ice using
//  Arrhenius coefficients. Differs from the original by using
//  temperature-dependent coefficients for ice, not liquid water.
// ---------------------------------------------------------------------
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "Randomize.hh"
#include "G4Track.hh"

// Ice-specific includes for Arrhenius diffusion coefficients
#include "ArrheniusConstants_ice.hh"
#include "G4OH.hh"
#include "G4Hydrogen.hh"

// Helper function to get ice temperature at position (from G4DNABrownianTransportation_ice.cc)
extern G4double GetIceTemperatureAtPosition(const G4ThreeVector& position);

G4DiffusionControlledReactionModel_ice::G4DiffusionControlledReactionModel_ice()
    : G4DiffusionControlledReactionModel()
{
    // ICE-SPECIFIC CONSTRUCTOR: Initializes with vanilla base functionality
    // PURPOSE: Sets up diffusion-controlled reaction model for ice chemistry
    // TECHNICAL NOTE: All Smoluchowski theory implementation inherited,
    //                 ice modifications applied only at coefficient access points
}

G4double G4DiffusionControlledReactionModel_ice::GetArrheniusDiffusionCoefficient(const G4Track& track, const G4MolecularConfiguration* molConf) {
    // ICE-SPECIFIC COEFFICIENT CALCULATOR: Core helper function
    // 
    // PURPOSE: Provides temperature-dependent diffusion coefficients for kinetics calculations
    // PHYSICS: D = D₀ × exp(-Eₐ/kT) where Eₐ is activation energy for diffusion
    // 
    // MOLECULAR PARAMETERS:
    // - OH: Eₐ=0.14 eV → strong temperature dependence, slow in cold ice
    // - H:  Eₐ=0.02 eV → weak temperature dependence, relatively mobile in cold ice
    // - Other molecules: Use stored coefficients (no ice-specific data available)
    
    G4ThreeVector position = track.GetPosition();
    G4double temperature = GetIceTemperatureAtPosition(position);
    
    if (molConf->GetDefinition() == G4OH::Definition()) {
        return ArrheniusConstants_ice::CalculateArrheniusDiffusion(
            ArrheniusConstants_ice::D0_OH, ArrheniusConstants_ice::Ea_OH, temperature);
    }
    else if (molConf->GetDefinition() == G4Hydrogen::Definition()) {
        return ArrheniusConstants_ice::CalculateArrheniusDiffusion(
            ArrheniusConstants_ice::D0_H, ArrheniusConstants_ice::Ea_H, temperature);
    }
    
    // For other molecules, use the stored coefficient
    // FUTURE ENHANCEMENT: Add ice-specific parameters for more molecular species
    return molConf->GetDiffusionCoefficient();
}

G4double G4DiffusionControlledReactionModel_ice::GetTimeToEncounter(
  const G4Track& trackA, const G4Track& trackB)
{
  // ICE-SPECIFIC ENCOUNTER TIME CALCULATION: Core kinetics method
  // 
  // PURPOSE: Calculates time until two molecules encounter and potentially react
  // PHYSICS: Based on Smoluchowski diffusion-controlled reaction theory
  // EQUATION: t = (r²/6D) × (-ln(random)) where D = D₁ + D₂
  // 
  // VANILLA VS ICE COMPARISON:
  // VANILLA: D₁,D₂ from stored coefficients → spatially uniform kinetics
  // ICE: D₁,D₂ from Arrhenius calculations → position-dependent kinetics
  // 
  // CRITICAL PHYSICS POINT:
  // - Small changes in D cause large changes in encounter time
  // - In Europa ice: D varies by 10³-10⁴ → encounter times vary by same magnitude
  // - Cold ice regions: Very slow encounters, low reaction rates
  // - Warm ice regions: Fast encounters, high reaction rates
  
  // Step 1: Get molecular configurations for both tracks
  auto pMolConfA = GetMolecule(trackA)->GetMolecularConfiguration();
  auto pMolConfB = GetMolecule(trackB)->GetMolecularConfiguration();

  // Step 2: CRITICAL ICE MODIFICATION - Calculate ice-specific coefficients
  // VANILLA: D = pMolConfA->GetDiffusionCoefficient() + pMolConfB->GetDiffusionCoefficient()
  // ICE: D = Arrhenius_A(position) + Arrhenius_B(position)
  G4double D = GetArrheniusDiffusionCoefficient(trackA, pMolConfA) + 
               GetArrheniusDiffusionCoefficient(trackB, pMolConfB);

  // Step 3: Error checking for zero diffusion (standard Geant4 safety)
  if(D == 0)
  {
    G4ExceptionDescription exceptionDescription;
    exceptionDescription << "The total diffusion coefficient for : "
                         << pMolConfA->GetName() << " and "
                         << pMolConfB->GetName() << " is null ";
    G4Exception("G4DiffusionControlledReactionModel_ice"
                "::GetTimeToEncounter()",
                "G4DiffusionControlledReactionModel_ice03", FatalException,
                exceptionDescription);
  }

  // Step 4: Calculate initial separation distance
  G4double r0 = (trackA.GetPosition() - trackB.GetPosition()).mag();
  if(r0 == 0)
  {
    // Handle edge case of coincident molecules
    r0 = 1e-5 * nm;
  }

  // Step 5: Apply Smoluchowski encounter time formula with ice-specific D
  // PHYSICS: Independent Reaction Time (IRT) with exponential probability distribution
  // RESULT: Realistic encounter times reflecting actual molecular mobility in ice
  G4double irt = (r0 * r0 / (6 * D)) * (-::log(G4UniformRand()));
  return irt;
  
  // SUMMARY: This method now provides position-dependent encounter times
  // reflecting the actual temperature-dependent molecular mobility in Europa's ice
}
